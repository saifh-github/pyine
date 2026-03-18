"""Configuration classes and helpers for the guardrail correctness evaluation pipeline."""

from __future__ import annotations

import pathlib  # noqa: TC003
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
    bootstrap_num_workers: int = pydantic.Field(default=0, ge=0)
    """Number of workers for bootstrap CI computation.

    0 = auto (use up to 16 CPU cores), 1 = sequential, >1 = use that many workers. Results are
    deterministic for a given (seed, effective num_workers) pair but differ between sequential and
    parallel modes due to independent RNG streams.
    """
    roc_fpr_grid_size: int = pydantic.Field(default=200, gt=1)
    """Number of points in the FPR grid for ROC curve plotting."""

    eval_type_parallelism: int = pydantic.Field(default=1, ge=1)
    """Number of probe types to evaluate concurrently.

    When > 1, per-type bootstrap_num_workers is automatically reduced so total workers stay
    approximately equal to bootstrap_num_workers.
    """

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
        """Registers compact correctness metric definitions with a W&B run.

        Registers only the dashboard-relevant summary keys that ``_log_correctness_metrics_to_wandb``
        writes. Category metrics and full detail are stored in wandb Tables, not in summary, so they
        need no registrations.

        Args:
            wandb_run: The W&B run object where metric definitions should be registered.
            eval_subset_names: A sequence of subset names that will be evaluated.
        """
        step_metric = "train/global_step"
        for eval_subset_name in eval_subset_names:
            prefix = f"benchmark/{eval_subset_name}"
            # primary ranking metrics (higher is better)
            for metric_name in ["auroc", "average_precision"]:
                wandb_run.define_metric(
                    name=f"{prefix}/{metric_name}/mean",
                    step_metric=step_metric,
                    summary="max",
                )
            # TPR@FPR (higher is better)
            for target_fpr in self.target_fpr_values:
                fpr_key = correctness_metrics.format_fpr_key(target_fpr)
                wandb_run.define_metric(
                    name=f"{prefix}/tpr_at_{fpr_key}/mean",
                    step_metric=step_metric,
                    summary="max",
                )
            # per-FPR operating-point metrics (from shared manifest)
            for target_fpr in self.target_fpr_values:
                fpr_key = correctness_metrics.format_fpr_key(target_fpr)
                for metric_suffix, goal in correctness_types.COMPACT_SUMMARY_PER_FPR_METRICS:
                    wandb_run.define_metric(
                        name=f"{prefix}/{fpr_key}/{metric_suffix}/mean",
                        step_metric=step_metric,
                        summary=goal,
                    )
            # bootstrap CIs for global metrics (hidden)
            for base_name in ["auroc", "average_precision"]:
                for ci_suffix in ("bootstrap_ci_point", "bootstrap_ci_lower", "bootstrap_ci_upper"):
                    wandb_run.define_metric(
                        name=f"{prefix}/{base_name}/{ci_suffix}",
                        step_metric=step_metric,
                        hidden=True,
                        summary="none",
                    )
            # class balance and counts (hidden)
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
        """Log aggregated correctness metrics to W&B as compact summary + tables.

        Writes compact summary keys and wandb Tables (detailed_metrics, category_metrics) per subset
        via ``_log_correctness_metrics_to_wandb``. Also builds a cross-subset comparison table under
        ``benchmark/metrics_table`` from compact keys only.

        Args:
            wandb_run: Run object where the table should be logged.
            results_by_subset: Mapping of subset names to CorrectnessEvalResult.
            step: Optional W&B step override.

        Returns:
            The cross-subset comparison table that was logged (if any).

        Raises:
            TypeError: If any subset result is not a CorrectnessEvalResult.
        """
        compact_by_subset: dict[str, dict[str, float | int | str]] = {}
        seen_metric_names: set[str] = set()
        for subset_name, subset_result in results_by_subset.items():
            if not isinstance(subset_result, correctness_impl.CorrectnessEvalResult):
                raise TypeError(f"expected CorrectnessEvalResult, got {type(subset_result).__name__}")
            summary_prefix = f"benchmark/{subset_name}"
            correctness_impl._log_correctness_metrics_to_wandb(  # type: ignore[reportPrivateUsage]
                wandb_run=wandb_run,
                aggregated=subset_result.aggregated,
                key_prefix=summary_prefix,
                eval_subset_name=subset_name,
                step=step,
            )
            compact = subset_result.aggregated.to_compact_summary_dict()
            compact_by_subset[subset_name] = compact
            seen_metric_names.update(compact.keys())
        ordered_metric_names = sorted(seen_metric_names)
        table = wandb.Table(columns=["subset", *ordered_metric_names])
        for subset_name, subset_metrics in compact_by_subset.items():
            row: list[typing.Any] = [subset_name]
            for metric_name in ordered_metric_names:
                row.append(subset_metrics.get(metric_name))
            table.add_data(*row)  # pyright: ignore[reportUnknownMemberType]
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
    ("skewed_weak_bias", "for_skewed_weak_bias"),
    ("balanced_weak_bias", "for_balanced_weak_bias"),
    ("skewed_moderate_bias", "for_skewed_moderate_bias"),
    ("balanced_moderate_bias", "for_balanced_moderate_bias"),
    ("skewed_strong_bias", "for_skewed_strong_bias"),
    ("balanced_strong_bias", "for_balanced_strong_bias"),
]
"""Mapping of preset name to factory classmethod name on RecordResamplingConfig."""


def _get_split_config_extras(
    split_source: str | pathlib.Path | None,
) -> dict[str, typing.Any]:
    """Returns Hydra config overrides for pre-filling ``split_config.split_source``."""
    if split_source is None:
        return {}
    return {"split_config": {"split_source": str(split_source)}}


def _get_resampling_configs(
    group: str,
    base_description: str,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Returns the base resampling config and all canonical resampling presets for a Hydra group."""
    configs = [
        pyine.configs.utils.make_config_description(
            correctness_types.RecordResamplingConfig,
            name="resampling_base",
            group=group,
            description=base_description,
            config={
                "populate_full_signature": True,
                "hydra_convert": "object",
            },
        )
    ]
    for preset_name, factory_name in _RESAMPLING_PRESETS:
        factory = getattr(correctness_types.RecordResamplingConfig, factory_name)
        preset_instance = factory()
        overrides = preset_instance.model_dump(exclude_defaults=True)
        configs.append(
            pyine.configs.utils.make_config_description(
                correctness_types.RecordResamplingConfig,
                name=preset_name,
                group=group,
                description=factory.__doc__ or f"Resampling preset: {preset_name}",
                config={
                    "populate_full_signature": True,
                    "hydra_convert": "object",
                    **overrides,
                },
            )
        )
    return configs


def get_datamodule_configs(
    group: str,
    split_source: str | pathlib.Path | None = None,
    base_name: str = "correctness_base",
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Returns datamodule configs for correctness-based training apps.

    These configs are intended for the **datamodule preparation in training apps** (e.g. probe
    trainer, LLM classifier trainer), not for the nested datamodule inside ``CorrectnessEvalsConfig``.

    Returns a base ``CorrectnessDataModuleConfig`` config, a ``resampling_base`` config, and all
    canonical resampling presets. If ``split_source`` is provided, the base config pre-fills
    ``split_config.split_source`` so users don't have to specify it manually.

    Args:
        group: Hydra config group path for the datamodule configs.
        split_source: Optional path or dataset name for the split source. When provided,
            pre-fills ``split_config.split_source`` in the base config.
        base_name: Name to use for the registered base datamodule config.
    """
    base_config_extras = _get_split_config_extras(split_source)
    base_config = pyine.configs.utils.make_config_description(
        correctness_datamodule_configs.CorrectnessDataModuleConfig,
        name=base_name,
        group=group,
        description="Base correctness datamodule settings (LMDB eval records, splits, resampling).",
        config={
            "populate_full_signature": True,
            "hydra_convert": "object",
            "hydra_defaults": [
                "_self_",
                {"resampling": "resampling_base"},
            ],
            **base_config_extras,
        },
    )
    return [
        base_config,
        *_get_resampling_configs(
            group=f"{group}/resampling",
            base_description="Base resampling config for training and validation data composition control.",
        ),
    ]


def get_evals_configs(
    group: str,
    split_source: str | pathlib.Path | None = None,
    base_name: str = "correctness_base",
    datamodule_name: str = "correctness_dm_base",
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Config provider for EvalType.CORRECTNESS, called by pyine/evals/configs.py dispatch.

    Returns a base ``CorrectnessEvalsConfig`` and its nested ``CorrectnessDataModuleConfig`` for
    hydra-zen config composition. Required fields (``lmdb_paths``) are left as MISSING: the user
    must provide them at runtime or in a YAML override. If ``split_source`` is provided, it
    pre-fills ``split_config.split_source`` in the nested datamodule config.

    Also registers canonical resampling presets under both ``calibration_resampling`` and
    ``datamodule_config/resampling`` hydra groups.

    Args:
        group: Hydra config group path for the eval configs.
        split_source: Optional path or dataset name for the split source. When provided,
            pre-fills ``split_config.split_source`` in the nested datamodule config.
        base_name: Name to use for the registered base eval config.
        datamodule_name: Name to use for the nested datamodule base config.
    """
    dm_config_extras = _get_split_config_extras(split_source)
    datamodule_config = pyine.configs.utils.make_config_description(
        correctness_datamodule_configs.CorrectnessDataModuleConfig,
        name=datamodule_name,
        group=f"{group}/datamodule_config",
        description="Base correctness evaluation datamodule settings (LMDB source, label type, splitting).",
        config={
            "populate_full_signature": True,
            "hydra_convert": "object",
            "hydra_defaults": [
                "_self_",
                {"resampling": "resampling_base"},
            ],
            **dm_config_extras,
        },
    )
    base_config = pyine.configs.utils.make_config_description(
        CorrectnessEvalsConfig,
        name=base_name,
        group=group,
        description="Correctness evaluation settings with canonical defaults for guardrail benchmarking.",
        config={
            "populate_full_signature": True,
            "hydra_convert": "object",
            "hydra_defaults": [
                "_self_",
                {"datamodule_config": datamodule_name},
                {"calibration_resampling": "resampling_base"},
            ],
        },
    )
    return [
        base_config,
        datamodule_config,
        *_get_resampling_configs(
            group=f"{group}/calibration_resampling",
            base_description="Base resampling config with reasonable defaults for robustness testing.",
        ),
        *_get_resampling_configs(
            group=f"{group}/datamodule_config/resampling",
            base_description="Base resampling config for training and validation data composition control.",
        ),
    ]
