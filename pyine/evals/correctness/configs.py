"""Configuration classes and helpers for the guardrail correctness evaluation pipeline."""

from __future__ import annotations

import typing

import pydantic
import wandb

import pyine.configs.schemas
import pyine.configs.utils
import pyine.data.datamodule
import pyine.evals.common
import pyine.evals.correctness._impl as correctness_impl
import pyine.evals.correctness.datamodule as correctness_datamodule_mod
import pyine.evals.correctness.datamodule_configs as correctness_datamodule_configs
import pyine.evals.correctness.metrics as correctness_metrics
import pyine.evals.correctness.types as correctness_types
import pyine.evals.persistence
import pyine.evals.utils


class RecordCategoryConfig(pydantic.BaseModel):
    """Defines how to categorize EvalRecords for per-category metric breakdowns.

    Categories are derived from the code_type field in LMDB records. The default grouping
    distinguishes 'regular' examples from 'biasing' examples.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
    """Pydantic model configuration (freezes the dataclass)."""

    code_type_to_category: dict[str, str] = pydantic.Field(
        default_factory=lambda: {
            "original": "regular",
            "obfuscated": "regular",
            "stubbed": "regular",
            "hinted": "biasing/hinted",
            "misleading": "biasing/misleading",
            "bugged": "biasing/bugged",
        },
    )
    """Maps code_type strings to category names.

    Records whose code_type is not in this map get assigned to an 'other' category (with a logged
    warning).
    """
    report_per_code_type: bool = True
    """When True, also report metrics per individual code_type (in addition to the grouped categories)."""


class CorrectnessEvalsConfig(pyine.evals.common.BaseEvalsConfig):
    """Configuration for guardrail correctness evaluation.

    Inherits from BaseEvalsConfig to integrate with the existing eval dispatch and W&B
    logging infrastructure. Implements evaluate_wrapped_model(); raises NotImplementedError
    for evaluate_runnable_model() and evaluate_hf_model().
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
    """Pydantic model configuration (freezes the dataclass)."""

    eval_type: pyine.evals.common.EvalType | None = pyine.evals.common.EvalType.CORRECTNESS
    """Type of evaluation to be conducted (overrides base class field default)."""

    datamodule_config: correctness_datamodule_configs.CorrectnessDataModuleConfig
    """Configuration for the correctness evaluation datamodule (holds LMDB paths, label type, split config)."""

    category_config: RecordCategoryConfig = pydantic.Field(default_factory=RecordCategoryConfig)
    """Configuration for code_type grouping (regular vs biasing).

    Code_type categories use ``RecordCategoryConfig`` for semantic grouping, while other sample
    fields (predict_type, has_keyword, identifier_suffix, tags) are extracted via the inherited
    ``category_extraction_config`` from the base class. The code_type field is automatically
    filtered out of the base extractor to avoid duplication.
    """

    calibration_resampling: correctness_types.RecordResamplingConfig | None = None
    """Optional resampling config for calibration records.

    When set, guardrail_valid records are resampled before being used for threshold calibration.
    Allows studying guardrail robustness to imperfect calibration datasets (e.g. skewed positive to
    negative label ratios, reduced code type diversity).
    """

    text_field: str = "model_output"
    """Which ``EvalRecord`` field to use as input text for scoring (e.g. ``"model_output"``)."""
    target_fpr_values: list[float] = pydantic.Field(default_factory=lambda: [0.001, 0.01, 0.05])
    """FPR constraints for threshold calibration. Must be sorted, unique, and in (0, 1)."""
    num_bootstrap_replicates: pydantic.PositiveInt = 1000
    """Number of bootstrap replicates for CI computation."""
    bootstrap_seed: int = 0
    """Random seed for bootstrap resampling."""
    confidence_level: float = pydantic.Field(default=0.95, gt=0.0, lt=1.0)
    """Confidence level for bootstrap CIs. Must be in (0, 1)."""
    roc_fpr_grid_size: int = pydantic.Field(default=200, gt=1)
    """Number of points in the FPR grid for ROC curve plotting."""

    @pydantic.field_validator("target_fpr_values")
    @classmethod
    def _validate_target_fpr_values(
        cls,
        value: list[float],
    ) -> list[float]:
        """Validates the range and values of the provided target_fpr_values list."""
        if not value:
            raise ValueError("target_fpr_values must not be empty")
        if value != sorted(value):
            raise ValueError(f"target_fpr_values must be sorted, got {value}")
        if len(value) != len(set(value)):
            raise ValueError(f"target_fpr_values must be unique, got {value}")
        if any(val <= 0.0 or val >= 1.0 for val in value):
            raise ValueError(f"all target_fpr_values must be in (0, 1), got {value}")
        return value

    # ---------- evaluation method overrides ----------

    @typing.override
    def prepare_eval_datamodule(
        self,
        datamodule: pyine.data.datamodule.BaseDataModule[typing.Any] | None,
    ) -> pyine.data.datamodule.BaseDataModule[typing.Any]:
        """Prepares the evaluation datamodule for upcoming correctness evaluation passes.

        For the correctness prediction task, the evaluation config contains everything that should
        be needed to instantiate the evaluation/benchmarking datamodule from scratch, so we do NOT
        expect the parent/caller to be providing any datamodule. If they do so, this is unexpected,
        and we won't know what to do with it.
        """
        if datamodule is not None:
            raise ValueError(
                "CorrectnessEvalsConfig constructs its own datamodule; "
                f"caller should not provide one (got {type(datamodule).__name__})"
            )
        eval_dm = self.datamodule_config.instantiate_datamodule()
        eval_dm.prepare_data()
        eval_dm.setup()
        return eval_dm

    @typing.override
    async def evaluate_wrapped_model(
        self,
        wrapped_model: typing.Any | typing.Sequence[typing.Any],
        datamodule: pyine.data.datamodule.BaseDataModule[typing.Any],
        eval_subset_name: str,
        verbose: bool = False,
    ) -> pyine.evals.common.EvalResult:
        """Evaluates guardrail scorers on pregenerated LMDB data.

        Accepts a single GuardrailScorer or a sequence of them. When multiple scorers are provided,
        each is evaluated independently and results are aggregated across runs (cross-run
        mean/std/p5, hierarchical bootstrap CIs).

        Args:
            wrapped_model: A single GuardrailScorer instance, or a sequence of them for
                multi-run aggregation.
            datamodule: The correctness evaluation datamodule (must be a CorrectnessDataModule).
            eval_subset_name: The name of the subset to fetch from the datamodule and evaluate on.
            verbose: Whether to verbosely report progress.

        Returns:
            CorrectnessEvalResult wrapping the full AggregatedResult.
        """
        if not isinstance(datamodule, correctness_datamodule_mod.CorrectnessDataModule):
            raise TypeError(
                f"expected a CorrectnessDataModule, got {type(datamodule).__name__}; "
                f"ensure prepare_eval_datamodule() was called first"
            )
        guardrails: list[typing.Any]
        if isinstance(wrapped_model, (list, tuple)):
            guardrails = list(wrapped_model)  # type: ignore[reportUnknownArgumentType]
        else:
            guardrails = [wrapped_model]
        result = await correctness_impl.evaluate_guardrail_replicas(  # type: ignore[reportArgumentType]
            config=self,
            guardrails=guardrails,
            datamodule=datamodule,
            eval_subset_name=eval_subset_name,
            verbose=verbose,
        )
        pyine.evals.persistence.maybe_dump_eval_result(
            result=result,
            dump_dir=self.result_dump_dir,
            eval_subset_name=eval_subset_name,
            overwrite=self.result_dump_overwrite,
        )
        return result

    @typing.override
    def define_metrics_for_wandb(
        self,
        wandb_run: wandb.Run,
        eval_subset_names: typing.Sequence[str],
    ) -> None:
        """Registers correctness metric definitions with a W&B run.

        Defines metric summary strategies so W&B can track key metrics. Uses the flat dict key
        namespace from ``AggregatedResult.to_flat_dict()``. Should be called once before logging
        any metrics.

        Key metrics (AUROC, guarded_pass_rate, TPR, etc.) get ``summary="max"`` so their best
        values surface in the W&B run table. Safety metrics (unsafe_slip_rate) get ``"min"``.
        Descriptive metrics (class balance, counts, variability stats, CIs) are hidden to avoid
        flooding the dashboard; a glob catch-all ensures they are still logged.

        Args:
            wandb_run: The W&B run object where metric definitions should be registered.
            eval_subset_names: A sequence of subset names that will be evaluated (for metric name
                prefixing, if needed).
        """
        step_metric = "train/global_step"
        for eval_subset_name in eval_subset_names:
            prefix = f"benchmark/{eval_subset_name}"
            # catch-all for category-wise and any other data-dependent metrics; more-specific
            # definitions below take precedence over this glob in wandb
            wandb_run.define_metric(name=f"{prefix}/*", step_metric=step_metric)
            # -- primary ranking metrics (higher is better) --
            for metric_name in ["auroc", "average_precision"]:
                wandb_run.define_metric(
                    name=f"{prefix}/{metric_name}/mean",
                    step_metric=step_metric,
                    summary="max",
                )
            # -- TPR@FPR metrics (higher is better) --
            for target_fpr in self.target_fpr_values:
                fpr_key = correctness_metrics.format_fpr_key(target_fpr)
                wandb_run.define_metric(
                    name=f"{prefix}/tpr_at_{fpr_key}/mean",
                    step_metric=step_metric,
                    summary="max",
                )
            # -- per-FPR operating-point metrics --
            for target_fpr in self.target_fpr_values:
                fpr_key = correctness_metrics.format_fpr_key(target_fpr)
                # key attempt-level metrics
                wandb_run.define_metric(
                    name=f"{prefix}/{fpr_key}/tpr/mean",
                    step_metric=step_metric,
                    summary="max",
                )
                # key sample-level metrics
                wandb_run.define_metric(
                    name=f"{prefix}/{fpr_key}/guarded_pass_rate/mean",
                    step_metric=step_metric,
                    summary="max",
                )
                wandb_run.define_metric(
                    name=f"{prefix}/{fpr_key}/unsafe_slip_rate/mean",
                    step_metric=step_metric,
                    summary="min",  # lower is better for safety
                )
                wandb_run.define_metric(
                    name=f"{prefix}/{fpr_key}/best_of_k_success_rate/mean",
                    step_metric=step_metric,
                    summary="max",
                )
                # conservative sample-level metrics (visible, higher is better for pass/justify)
                wandb_run.define_metric(
                    name=f"{prefix}/{fpr_key}/cons_pass_rate/mean",
                    step_metric=step_metric,
                    summary="max",
                )
                wandb_run.define_metric(
                    name=f"{prefix}/{fpr_key}/cons_unsafe_slip_rate/mean",
                    step_metric=step_metric,
                    summary="min",  # lower is better for safety
                )
                wandb_run.define_metric(
                    name=f"{prefix}/{fpr_key}/cons_justified_reject_rate/mean",
                    step_metric=step_metric,
                    summary="max",
                )
                # secondary attempt-level and sample-level metrics (hidden)
                secondary_metrics = [
                    f"{fpr_key}/fpr/mean",
                    f"{fpr_key}/fnr/mean",
                    f"{fpr_key}/precision/mean",
                    f"{fpr_key}/npv/mean",
                    f"{fpr_key}/base_pass_rate/mean",
                    f"{fpr_key}/total_block_rate/mean",
                ]
                for secondary_name in secondary_metrics:
                    wandb_run.define_metric(
                        name=f"{prefix}/{secondary_name}",
                        step_metric=step_metric,
                        hidden=True,
                        summary="none",
                    )
                # cost metrics (lower is better for totals/means; correlations are descriptive)
                cost_metrics_min = [
                    "cost_total",
                    "cost_mean",
                    "cost_median",
                    "cost_std",
                    "cost_per_correct_acceptance",
                    "cost_per_incorrect_block",
                ]
                for cost_name in cost_metrics_min:
                    wandb_run.define_metric(
                        name=f"{prefix}/{fpr_key}/{cost_name}/mean",
                        step_metric=step_metric,
                        summary="min",
                    )
                cost_metrics_last = [
                    "cost_accuracy_rank_correlation",
                    "cost_difficulty_rank_correlation",
                ]
                for cost_name in cost_metrics_last:
                    wandb_run.define_metric(
                        name=f"{prefix}/{fpr_key}/{cost_name}/mean",
                        step_metric=step_metric,
                        summary="last",
                    )
            # -- variability, CI, and descriptive metrics (hidden) --
            hidden_suffixes = [
                "std",
                "p5",
                "num_valid_runs",
                "bootstrap_ci_point",
                "bootstrap_ci_lower",
                "bootstrap_ci_upper",
            ]
            for suffix in hidden_suffixes:
                wandb_run.define_metric(
                    name=f"{prefix}/*/{suffix}",
                    step_metric=step_metric,
                    hidden=True,
                    summary="none",
                )
            # class balance and counts
            for hidden_name in ["class_balance/*", "record_count", "sample_count"]:
                wandb_run.define_metric(
                    name=f"{prefix}/{hidden_name}",
                    step_metric=step_metric,
                    hidden=True,
                    summary="none",
                )

    @typing.override
    def log_metrics(
        self,
        wandb_run: wandb.Run,
        results_by_subset: dict[str, pyine.evals.common.EvalResult],
        *,
        step: int | None = None,
    ) -> wandb.Table | None:
        """Log aggregated correctness metrics to a W&B table.

        Builds a single cross-subset comparison table (one row per subset) and logs it under
        the ``benchmark/metrics_table`` key. Also writes individual metrics to the run summary
        under ``benchmark/{subset_name}/{metric_name}`` keys.

        For per-sample metrics, use `log_sample_metrics`. For qualitative inspection of
        individual predictions, use `log_predictions`.

        Args:
            wandb_run: Run object where the table should be logged.
            results_by_subset: Mapping of subset names to CorrectnessEvalResult.
            step: Optional W&B step override.

        Returns:
            The table that was logged (if any).

        See Also:
            log_sample_metrics: For per-sample metrics.
            log_predictions: For qualitative inspection of predictions.
        """
        metrics_by_subset: dict[str, pyine.evals.utils.MetricsDictType] = {}
        seen_metric_names: set[str] = set()
        for subset_name, subset_result in results_by_subset.items():
            metrics_by_subset[subset_name] = subset_result.metrics
            seen_metric_names.update(subset_result.metrics.keys())
        ordered_metric_names = sorted(seen_metric_names)
        table = wandb.Table(columns=["subset", *ordered_metric_names])
        for subset_name, subset_metrics in metrics_by_subset.items():
            row: list[typing.Any] = [subset_name]
            for metric_name in ordered_metric_names:
                row.append(subset_metrics.get(metric_name))
            table.add_data(*row)  # pyright: ignore[reportUnknownMemberType] - wandb Table.add_data has incomplete stubs
            summary_prefix = f"benchmark/{subset_name}"
            for metric_name, metric_val in subset_metrics.items():
                wandb_run.summary[f"{summary_prefix}/{metric_name}"] = metric_val
        table_key = "benchmark/metrics_table"
        if step is None:
            wandb_run.log({table_key: table})
        else:
            wandb_run.log({table_key: table}, step=step)
        return table

    @typing.override
    def log_predictions(
        self,
        wandb_run: wandb.Run,
        subset_name: str,
        subset_results: pyine.evals.common.EvalResult,
        *,
        max_rows: int | None = None,
        max_text_length: int | None = None,
        step: int | None = None,
    ) -> wandb.Table | None:
        """Log predictions to W&B (as a table) for qualitative inspection.

        Not implemented for correctness evals: returns None. Correctness evaluation operates on
        pregenerated predictions (loaded from LMDB), so there are no new model outputs to inspect.

        TODO: @@@@ this might change when we start doing debates! (revisit w/ derived class?)

        Args:
            wandb_run: Run object where the table should be logged.
            subset_name: Name of the evaluated subset.
            subset_results: Captured CorrectnessEvalResult.
            max_rows: Maximum number of rows to log. None means no limit (default).
            max_text_length: Maximum length per text field before truncation. None means
                no truncation (default).
            step: Optional W&B step override.

        Returns:
            Always None for correctness evals.

        See Also:
            log_sample_metrics: For per-sample metrics.
            log_metrics: For subset-level aggregated metrics.
        """
        return None

    @typing.override
    def log_sample_metrics(
        self,
        wandb_run: wandb.Run,
        subset_name: str,
        subset_results: pyine.evals.common.EvalResult,
        *,
        step: int | None = None,
    ) -> wandb.Table | None:
        """Log per-sample metrics to W&B (as a table) for quantitative analyses.

        Not implemented for correctness evals: returns None. Sample-level detail is captured in
        ``AggregatedResult.per_run`` and surfaced via ``to_flat_dict()`` in the cross-run
        aggregates logged by ``log_metrics``.

        Args:
            wandb_run: Run object where the table should be logged.
            subset_name: Name of the evaluated subset.
            subset_results: Captured CorrectnessEvalResult.
            step: Optional W&B step override.

        Returns:
            Always None for correctness evals.

        See Also:
            log_predictions: For qualitative inspection of predictions.
            log_metrics: For subset-level aggregated metrics.
        """
        return None


_RESAMPLING_PRESETS: list[tuple[str, str]] = [
    ("skewed_pos", "for_skewed_positive"),
    ("balanced", "for_balanced"),
    ("original_only", "for_original_only"),
    ("mostly_original", "for_mostly_original"),
    ("helpful_bias", "for_helpful_bias"),
    ("skewed_pos_helpful_bias", "for_skewed_positive_helpful_bias"),
    ("oversample_balanced", "for_oversample_balanced"),
]
"""Mapping of preset name to factory classmethod name on RecordResamplingConfig."""


def get_evals_configs(
    group: str,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Config provider for EvalType.CORRECTNESS, called by pyine/evals/configs.py dispatch.

    Returns a base ``CorrectnessEvalsConfig`` and its nested ``CorrectnessDataModuleConfig`` for
    hydra-zen config composition. Required fields (``lmdb_paths``, ``split_config.split_source``)
    are left as MISSING: the user must provide them at runtime or in a YAML override.

    Also registers canonical resampling presets under both ``calibration_resampling`` and
    ``datamodule_config/resampling`` hydra groups.
    """
    datamodule_config = pyine.configs.utils.make_config_description(
        correctness_datamodule_configs.CorrectnessDataModuleConfig,
        name="correctness_dm_base",
        group=f"{group}/datamodule_config",
        description="Base correctness evaluation datamodule settings (LMDB source, label type, splitting).",
        config={
            "populate_full_signature": True,
            "hydra_convert": "object",
        },
    )
    base_config = pyine.configs.utils.make_config_description(
        CorrectnessEvalsConfig,
        name="correctness_base",
        group=group,
        description="Correctness evaluation settings with canonical defaults for guardrail benchmarking.",
        config={
            "populate_full_signature": True,
            "hydra_convert": "object",
            "hydra_defaults": [
                "_self_",
                {"datamodule_config": "correctness_dm_base"},
            ],
        },
    )
    calibration_resampling_config = pyine.configs.utils.make_config_description(
        correctness_types.RecordResamplingConfig,
        name="resampling_base",
        group=f"{group}/calibration_resampling",
        description="Base resampling config with reasonable defaults for robustness testing.",
        config={
            "populate_full_signature": True,
            "hydra_convert": "object",
        },
    )
    resampling_config = pyine.configs.utils.make_config_description(
        correctness_types.RecordResamplingConfig,
        name="resampling_base",
        group=f"{group}/datamodule_config/resampling",
        description="Base resampling config for training and validation data composition control.",
        config={
            "populate_full_signature": True,
            "hydra_convert": "object",
        },
    )
    configs = [base_config, datamodule_config, calibration_resampling_config, resampling_config]
    resampling_groups = [
        f"{group}/calibration_resampling",
        f"{group}/datamodule_config/resampling",
    ]
    for preset_name, factory_name in _RESAMPLING_PRESETS:
        factory = getattr(correctness_types.RecordResamplingConfig, factory_name)
        preset_instance = factory()
        overrides = preset_instance.model_dump(exclude_defaults=True)
        for resampling_group in resampling_groups:
            configs.append(
                pyine.configs.utils.make_config_description(
                    correctness_types.RecordResamplingConfig,
                    name=preset_name,
                    group=resampling_group,
                    description=factory.__doc__ or f"Resampling preset: {preset_name}",
                    config={
                        "populate_full_signature": True,
                        "hydra_convert": "object",
                        **overrides,
                    },
                )
            )
    return configs
