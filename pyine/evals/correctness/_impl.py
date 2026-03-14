"""Main pipeline orchestration functions for guardrail correctness evaluations."""

from __future__ import annotations

import logging
import typing

import numpy as np
import pydantic

import pyine.evals.common
import pyine.evals.correctness.calibration as correctness_calibration
import pyine.evals.correctness.metrics as correctness_metrics
import pyine.evals.correctness.types as correctness_types
import pyine.evals.persistence
import pyine.utils.portability

if typing.TYPE_CHECKING:
    import pyine.evals.correctness.configs as correctness_configs
    import pyine.evals.correctness.datamodule as correctness_datamodule
    import pyine.evals.correctness.splits as correctness_splits

logger = logging.getLogger(__name__)


class CorrectnessEvalResult(pyine.evals.common.EvalResult):
    """Correctness evaluation results container.

    Extends EvalResult with an AggregatedResult object for downstream consumers that need its rich
    structure (per-run results, bootstrap CIs, etc.). The base ``metrics`` dict from the parent
    class contains the flattened pipeline-mean metrics.
    """

    model_config = pydantic.ConfigDict(frozen=True, arbitrary_types_allowed=True)
    """Pydantic model configuration (freezes the dataclass)."""

    aggregated: correctness_types.AggregatedResult
    """Full aggregated result with per-run details, splits, and CIs."""


async def evaluate_guardrail_replicas(
    config: correctness_configs.CorrectnessEvalsConfig,
    guardrails: typing.Sequence[correctness_types.GuardrailScorer],
    datamodule: correctness_datamodule.CorrectnessDataModule,
    eval_subset_name: str,
    verbose: bool = False,
) -> CorrectnessEvalResult:
    """Run the full guardrail correctness evaluation pipeline.

    Does not auto-dump results to disk; callers that need persistence should use
    ``persistence.save_eval_result`` after receiving the result, or call via
    ``evaluate_guardrail_types`` / ``evaluate_wrapped_model`` which handle dumping.

    Args:
        config: Correctness evaluation configuration.
        guardrails: Sequence of one or more GuardrailScorer instances to evaluate (we might be
            using replicas).
        datamodule: The prepared correctness evaluation datamodule.
        eval_subset_name: Which subset to evaluate on (e.g. ``"guardrail_valid"``).
        verbose: Whether to verbosely report progress.

    Returns:
        CorrectnessEvalResult with flattened metrics and full AggregatedResult.
    """
    if not guardrails:
        raise ValueError("guardrails must not be empty; provide at least one GuardrailScorer")
    if eval_subset_name == "guardrail_train":
        raise ValueError(
            "evaluating on guardrail_train is not supported (would be misleading); "
            "use 'guardrail_valid' or 'guardrail_test'"
        )

    # retrieve records from the datamodule
    eval_records = datamodule.get_records_for_subset(eval_subset_name)
    if not eval_records:
        raise ValueError(f"eval subset '{eval_subset_name}' is empty; cannot evaluate")
    calibration_records = datamodule.get_records_for_calibration(
        resampling_config=config.calibration_resampling,
    )
    if eval_subset_name == "guardrail_valid":
        resampling_note = ""
        if config.calibration_resampling is not None and not config.calibration_resampling.is_noop:
            resampling_note = " (calibration is resampled but still drawn from guardrail_valid)"
        logger.warning(
            "evaluating on 'guardrail_valid' uses the same split as threshold calibration%s; "
            "thresholded metrics will be optimistically biased. Prefer 'guardrail_test' for "
            "unbiased evaluation.",
            resampling_note,
        )
    guardrail_splits = datamodule.get_guardrail_splits()
    logger.info(
        f"evaluating on '{eval_subset_name}': {len(eval_records)} eval records, "
        f"{len(calibration_records)} calibration records"
    )
    eval_class_balance = correctness_metrics.compute_class_balance(eval_records)
    attempt_records_by_key = _build_attempt_records_by_key(eval_records)
    eval_metadata: dict[str, typing.Any] = {
        **pyine.evals.common.build_base_eval_metadata(config.eval_type, eval_subset_name),
        "eval_config": pyine.utils.portability.make_json_serializable(config),
        "datamodule_config": pyine.utils.portability.make_json_serializable(datamodule.config),
        "num_guardrail_replicas": len(guardrails),
    }

    # run each guardrail that we were provided (the input is a sequence of GuardrailScorer replicas)
    per_run_results: list[correctness_types.SingleRunResult] = []
    per_run_records_for_hierarchical: list[list[correctness_types.EvalRecord]] = []
    per_run_scores_for_hierarchical: list[np.ndarray[typing.Any, np.dtype[np.floating[typing.Any]]]] = []
    per_run_thresholds_for_hierarchical: dict[float, list[float]] = {fpr: [] for fpr in config.target_fpr_values}
    for run_idx, guardrail in enumerate(guardrails):
        if verbose:
            logger.info(f"evaluating guardrail run {run_idx + 1}/{len(guardrails)}")
        run_result = _evaluate_single_run(
            guardrail=guardrail,
            eval_records=eval_records,
            calibration_records=calibration_records,
            config=config,
        )
        per_run_results.append(run_result.result)
        per_run_records_for_hierarchical.append(eval_records)
        per_run_scores_for_hierarchical.append(run_result.eval_scores)
        for target_fpr in config.target_fpr_values:
            per_run_thresholds_for_hierarchical[target_fpr].append(run_result.thresholds[target_fpr])
    # aggregate across runs
    aggregated = _aggregate_runs(
        per_run_results=per_run_results,
        per_run_records=per_run_records_for_hierarchical,
        per_run_scores=per_run_scores_for_hierarchical,
        per_run_thresholds=per_run_thresholds_for_hierarchical,
        guardrail_splits=guardrail_splits,
        eval_class_balance=eval_class_balance,
        config=config,
        attempt_records_by_key=attempt_records_by_key,
    )
    eval_metadata["guardrail_metadata_by_run"] = [result.guardrail_metadata for result in per_run_results]
    return CorrectnessEvalResult(
        metrics=aggregated.to_flat_dict(),
        aggregated=aggregated,
        eval_metadata=eval_metadata,
    )


class _SingleRunOutput(typing.NamedTuple):
    """Internal container for a single guardrail run's output."""

    result: correctness_types.SingleRunResult
    eval_scores: np.ndarray[typing.Any, np.dtype[np.floating[typing.Any]]]
    thresholds: dict[float, float]


def _build_attempt_records_by_key(
    records: list[correctness_types.EvalRecord],
) -> dict[correctness_types.ScoredAttemptKey, dict[str, typing.Any]]:
    """Build a shared scored-attempt->record map for notebook inspection."""
    attempt_records_by_key: dict[correctness_types.ScoredAttemptKey, dict[str, typing.Any]] = {}
    for draw_index, record in enumerate(records):
        attempt_key: correctness_types.ScoredAttemptKey = (record.sample_id, record.attempt_index, draw_index)
        attempt_records_by_key[attempt_key] = dict(record.record)
    return attempt_records_by_key


def _validate_attempt_metadata_alignment(
    records: list[correctness_types.EvalRecord],
    scoring_result: correctness_types.ScoringResult,
    split_name: str,
) -> None:
    """Validate scorer-provided metadata keys align exactly with the input records."""
    if scoring_result.attempt_metadata is None:
        return
    expected_keys = {(record.sample_id, record.attempt_index, draw_index) for draw_index, record in enumerate(records)}
    actual_keys = set(scoring_result.attempt_metadata.keys())
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extras = sorted(actual_keys - expected_keys)
        raise ValueError(
            f"scorer attempt_metadata keys do not align with {split_name} records; "
            f"missing={missing[:5]}, extras={extras[:5]}"
        )


def _evaluate_single_run(
    guardrail: correctness_types.GuardrailScorer,
    eval_records: list[correctness_types.EvalRecord],
    calibration_records: list[correctness_types.EvalRecord],
    config: correctness_configs.CorrectnessEvalsConfig,
) -> _SingleRunOutput:
    """Evaluate a single guardrail instance.

    Args:
        guardrail: GuardrailScorer instance.
        eval_records: Records to evaluate on (e.g. guardrail_valid or guardrail_test).
        calibration_records: Records for threshold calibration (always based on guardrail_valid).
        config: Evaluation configuration.

    Returns:
        _SingleRunOutput with results, eval scores, and thresholds.
    """
    # score calibration and eval sets
    calibration_result = guardrail.score_records(calibration_records)
    if len(calibration_result.scores) != len(calibration_records):
        raise ValueError(
            f"scorer returned {len(calibration_result.scores)} scores for "
            f"{len(calibration_records)} calibration records"
        )
    _validate_attempt_metadata_alignment(calibration_records, calibration_result, "calibration")
    eval_result = guardrail.score_records(eval_records)
    if len(eval_result.scores) != len(eval_records):
        raise ValueError(f"scorer returned {len(eval_result.scores)} scores for {len(eval_records)} eval records")
    _validate_attempt_metadata_alignment(eval_records, eval_result, "eval")
    metadata = guardrail.get_metadata()
    cost_unit = guardrail.get_verification_cost_unit()
    calibration_scores = np.array(calibration_result.scores, dtype=np.float64)
    calibration_labels = np.array([record.label for record in calibration_records], dtype=np.bool_)
    eval_scores = np.array(eval_result.scores, dtype=np.float64)
    eval_labels = np.array([record.label for record in eval_records], dtype=np.bool_)
    eval_costs: typing.Sequence[float | None]
    if eval_result.verification_costs is None:
        eval_costs = [None for _ in eval_records]
    else:
        eval_costs = eval_result.verification_costs
    attempt_records: list[correctness_types.AttemptInspectionRecord] = []
    for draw_index, (eval_record, score, verification_cost) in enumerate(
        zip(
            eval_records,
            eval_result.scores,
            eval_costs,
            strict=True,
        ),
    ):
        attempt_key: correctness_types.ScoredAttemptKey = (
            eval_record.sample_id,
            eval_record.attempt_index,
            draw_index,
        )
        curr_attempt_metadata = None
        if eval_result.attempt_metadata is not None:
            curr_attempt_metadata = eval_result.attempt_metadata.get(attempt_key)
        attempt_records.append(
            correctness_types.AttemptInspectionRecord(
                sample_id=eval_record.sample_id,
                problem_id=eval_record.problem_id,
                attempt_index=eval_record.attempt_index,
                draw_index=draw_index,
                label=eval_record.label,
                score=score,
                verification_cost=verification_cost,
                final_answer=eval_record.final_answer,
                code_type=eval_record.code_type,
                difficulty_score=eval_record.difficulty_score,
                attempt_metadata=curr_attempt_metadata,
            )
        )
    # calibrate thresholds and compute thresholded metrics
    thresholds: dict[float, float] = {}
    attempt_metrics: dict[float, correctness_types.ThresholdedMetrics] = {}
    sample_metrics: dict[float, correctness_types.SampleLevelMetrics] = {}
    verification_cost_stats: dict[float, correctness_types.VerificationCostStats] = {}
    for target_fpr in config.target_fpr_values:
        threshold = correctness_calibration.calibrate_threshold(calibration_scores, calibration_labels, target_fpr)
        thresholds[target_fpr] = threshold
        attempt_metrics[target_fpr] = correctness_metrics.compute_thresholded_metrics(
            scores=eval_scores,
            labels=eval_labels,
            threshold=threshold,
            target_fpr=target_fpr,
        )
        sample_metrics[target_fpr] = correctness_metrics.compute_sample_level_metrics(
            records=eval_records,
            scores=eval_scores,
            threshold=threshold,
            target_fpr=target_fpr,
        )
        # verification cost stats
        accepted = eval_scores >= threshold
        cost_stats = correctness_metrics.compute_verification_cost_stats(  # type: ignore[reportUnknownMemberType]
            verification_costs=eval_result.verification_costs,
            labels=eval_labels,
            accepted=accepted,
            target_fpr=target_fpr,
            cost_unit=cost_unit,
            records=eval_records,
        )
        if cost_stats is not None:
            verification_cost_stats[target_fpr] = cost_stats
    # threshold-free metrics
    threshold_free = correctness_metrics.compute_threshold_free_metrics(  # type: ignore[reportUnknownMemberType]
        scores=eval_scores,
        labels=eval_labels,
        target_fprs=config.target_fpr_values,
        fpr_grid_size=config.roc_fpr_grid_size,
    )
    # category-wise metrics
    category_results = correctness_metrics.compute_category_results(
        records=eval_records,
        scores=eval_scores,
        thresholds=thresholds,
        target_fpr_values=config.target_fpr_values,
        category_config=config.category_config,
        fpr_grid_size=config.roc_fpr_grid_size,
        base_extraction_config=config.category_extraction_config,
    )
    # difficulty stats
    difficulty_stats = correctness_metrics.compute_difficulty_stats(  # type: ignore[reportUnknownMemberType]
        records=eval_records,
        scores=eval_scores,
        labels=eval_labels,
        thresholds=thresholds,
        target_fpr_values=config.target_fpr_values,
    )
    # bootstrap CIs
    bootstrap_cis = correctness_metrics.compute_clustered_bootstrap_cis(  # type: ignore[reportUnknownMemberType]
        records=eval_records,
        scores=eval_scores,
        thresholds=thresholds,
        target_fpr_values=config.target_fpr_values,
        num_replicates=config.num_bootstrap_replicates,
        seed=config.bootstrap_seed,
        confidence_level=config.confidence_level,
        num_workers=config.bootstrap_num_workers,
    )
    run_result = correctness_types.SingleRunResult(
        guardrail_metadata=metadata,
        attempt_metadata=eval_result.attempt_metadata,
        attempt_records=attempt_records,
        threshold_free=threshold_free,
        attempt_metrics=attempt_metrics,
        sample_metrics=sample_metrics,
        category_results=category_results,
        bootstrap_cis=bootstrap_cis,
        difficulty_stats=difficulty_stats,
        verification_cost_stats=verification_cost_stats if verification_cost_stats else None,
    )
    return _SingleRunOutput(result=run_result, eval_scores=eval_scores, thresholds=thresholds)


def _aggregate_runs(
    per_run_results: list[correctness_types.SingleRunResult],
    per_run_records: list[list[correctness_types.EvalRecord]],
    per_run_scores: list[np.ndarray[typing.Any, np.dtype[np.floating[typing.Any]]]],
    per_run_thresholds: dict[float, list[float]],
    guardrail_splits: correctness_splits.GuardrailSplits,
    eval_class_balance: correctness_types.ClassBalanceStats,
    config: correctness_configs.CorrectnessEvalsConfig,
    attempt_records_by_key: dict[correctness_types.ScoredAttemptKey, dict[str, typing.Any]] | None = None,
) -> correctness_types.AggregatedResult:
    """Aggregate results across R independent guardrail runs.

    Args:
        per_run_results: SingleRunResult for each run.
        per_run_records: Eval records for each run (same for all runs currently).
        per_run_scores: Eval score arrays for each run.
        per_run_thresholds: Thresholds per target_fpr per run.
        guardrail_splits: The split used for all runs.
        eval_class_balance: Class balance stats for the evaluated subset.
        config: Evaluation configuration.
        attempt_records_by_key: Optional shared raw record payloads keyed by attempt.

    Returns:
        AggregatedResult with cross-run statistics and hierarchical bootstrap CIs.
    """
    # collect scalar metrics from each run
    metric_values: dict[str, list[float | None]] = {}
    for run_result in per_run_results:
        _collect_scalar("auroc", run_result.threshold_free.auroc, metric_values)
        _collect_scalar("average_precision", run_result.threshold_free.average_precision, metric_values)
        if run_result.threshold_free.tpr_at_fpr is not None:
            for fpr_val, tpr_val in run_result.threshold_free.tpr_at_fpr.items():
                fpr_key = correctness_metrics.format_fpr_key(fpr_val)
                _collect_scalar(f"tpr_at_{fpr_key}", tpr_val, metric_values)
        for target_fpr in config.target_fpr_values:
            fpr_key = correctness_metrics.format_fpr_key(target_fpr)
            attempt = run_result.attempt_metrics.get(target_fpr)
            if attempt is not None:
                _collect_thresholded_metrics(fpr_key, attempt, metric_values)
            sample = run_result.sample_metrics.get(target_fpr)
            if sample is not None:
                _collect_sample_level_metrics(fpr_key, sample, metric_values)
        # category metrics
        for cat_name, cat_result in run_result.category_results.items():
            safe_cat = cat_name.replace("/", "_")
            _collect_scalar(f"category/{safe_cat}/auroc", cat_result.threshold_free.auroc, metric_values)
            for target_fpr in config.target_fpr_values:
                fpr_key = correctness_metrics.format_fpr_key(target_fpr)
                cat_attempt = cat_result.attempt_metrics.get(target_fpr)
                if cat_attempt is not None:
                    _collect_thresholded_metrics(f"category/{safe_cat}/{fpr_key}", cat_attempt, metric_values)
                cat_sample = cat_result.sample_metrics.get(target_fpr)
                if cat_sample is not None:
                    _collect_sample_level_metrics(f"category/{safe_cat}/{fpr_key}", cat_sample, metric_values)
        # cost metrics per target_fpr
        for target_fpr in config.target_fpr_values:
            fpr_key = correctness_metrics.format_fpr_key(target_fpr)
            if run_result.verification_cost_stats is not None:
                cost = run_result.verification_cost_stats.get(target_fpr)
                if cost is not None:
                    _collect_scalar(f"{fpr_key}/cost_total", cost.total_cost, metric_values)
                    _collect_scalar(f"{fpr_key}/cost_mean", cost.mean_cost_per_record, metric_values)
                    _collect_scalar(f"{fpr_key}/cost_median", cost.median_cost_per_record, metric_values)
                    _collect_scalar(f"{fpr_key}/cost_std", cost.std_cost_per_record, metric_values)
                    _collect_scalar(
                        f"{fpr_key}/cost_per_correct_acceptance",
                        cost.cost_per_correct_acceptance,
                        metric_values,
                    )
                    _collect_scalar(
                        f"{fpr_key}/cost_per_incorrect_block",
                        cost.cost_per_incorrect_block,
                        metric_values,
                    )
                    _collect_scalar(
                        f"{fpr_key}/cost_accuracy_rank_correlation",
                        cost.cost_accuracy_rank_correlation,
                        metric_values,
                    )
                    _collect_scalar(
                        f"{fpr_key}/cost_difficulty_rank_correlation",
                        cost.cost_difficulty_rank_correlation,
                        metric_values,
                    )
    # compute cross-run aggregates
    cross_run_mean: dict[str, float] = {}
    cross_run_std: dict[str, float] = {}
    cross_run_p5: dict[str, float] = {}
    cross_run_num_valid: dict[str, int] = {}
    for metric_name, values in metric_values.items():
        valid_values = [val for val in values if val is not None]
        cross_run_num_valid[metric_name] = len(valid_values)
        if not valid_values:
            continue
        arr = np.array(valid_values)
        cross_run_mean[metric_name] = float(np.mean(arr))
        cross_run_std[metric_name] = float(np.std(arr, ddof=1)) if len(valid_values) >= 2 else 0.0
        cross_run_p5[metric_name] = float(np.percentile(arr, 5))
    # hierarchical bootstrap CIs
    hierarchical_cis = correctness_metrics.compute_hierarchical_bootstrap_cis(  # type: ignore[reportUnknownMemberType]
        per_run_records=per_run_records,
        per_run_scores=per_run_scores,
        per_run_thresholds=per_run_thresholds,
        target_fpr_values=config.target_fpr_values,
        num_replicates=config.num_bootstrap_replicates,
        seed=config.bootstrap_seed + 1,  # different seed from per-run bootstrap
        confidence_level=config.confidence_level,
        num_workers=config.bootstrap_num_workers,
    )
    # difficulty stats: aggregate scalar fields across runs
    difficulty_stats = _aggregate_difficulty_stats(per_run_results)
    # verification cost stats: average across runs that report costs
    verification_cost_stats = _aggregate_cost_stats(per_run_results, config.target_fpr_values)
    return correctness_types.AggregatedResult(
        split_summary=guardrail_splits.to_summary(),
        class_balance=eval_class_balance,
        per_run=per_run_results,
        attempt_records_by_key=attempt_records_by_key,
        cross_run_mean=cross_run_mean,
        cross_run_std=cross_run_std,
        cross_run_p5=cross_run_p5,
        cross_run_num_valid=cross_run_num_valid,
        hierarchical_cis=hierarchical_cis,
        difficulty_stats=difficulty_stats,
        verification_cost_stats=verification_cost_stats,
    )


def _collect_scalar(
    name: str,
    value: float | None,
    target: dict[str, list[float | None]],
) -> None:
    """Append a scalar metric value to the collection dict."""
    if name not in target:
        target[name] = []
    target[name].append(value)


def _mean_of_non_none(values: list[float | None]) -> float | None:
    """Return the mean of non-None values, or None if all values are None."""
    valid = [val for val in values if val is not None]
    return float(np.mean(valid)) if valid else None


_THRESHOLDED_METRIC_FIELDS: tuple[str, ...] = ("tpr", "fpr", "fnr", "precision", "npv")
"""ThresholdedMetrics field names collected during cross-run aggregation."""

_SAMPLE_LEVEL_METRIC_FIELDS: tuple[str, ...] = (
    "base_pass_rate",
    "guarded_pass_rate",
    "unsafe_slip_rate",
    "total_block_rate",
    "best_of_k_success_rate",
    "cons_pass_rate",
    "cons_unsafe_slip_rate",
    "cons_justified_reject_rate",
)
"""SampleLevelMetrics field names collected during cross-run aggregation."""


def _collect_thresholded_metrics(
    prefix: str,
    thresholded: correctness_types.ThresholdedMetrics,
    target: dict[str, list[float | None]],
) -> None:
    """Collect all thresholded metric fields into the target dict."""
    for field_name in _THRESHOLDED_METRIC_FIELDS:
        _collect_scalar(f"{prefix}/{field_name}", getattr(thresholded, field_name), target)


def _collect_sample_level_metrics(
    prefix: str,
    sample: correctness_types.SampleLevelMetrics,
    target: dict[str, list[float | None]],
) -> None:
    """Collect all sample-level metric fields into the target dict."""
    for field_name in _SAMPLE_LEVEL_METRIC_FIELDS:
        _collect_scalar(f"{prefix}/{field_name}", getattr(sample, field_name), target)


def _aggregate_difficulty_stats(
    per_run_results: list[correctness_types.SingleRunResult],
) -> correctness_types.DifficultyStats | None:
    """Average difficulty stats across runs that have them.

    Args:
        per_run_results: SingleRunResult for each run.

    Returns:
        Averaged DifficultyStats, or None if no run has difficulty data.
    """
    runs_with_stats = [run for run in per_run_results if run.difficulty_stats is not None]
    if not runs_with_stats:
        return None
    # use bucket boundaries and sample counts from first run (they're data-dependent, not run-dependent)
    first = runs_with_stats[0].difficulty_stats
    if first is None:
        return None  # for type narrowing
    # average per-bucket AUROCs across runs
    per_bucket_auroc: dict[str, float | None] | None = None
    if first.per_bucket_auroc is not None:
        per_bucket_auroc = {}
        for bucket_name in first.per_bucket_auroc:
            raw_values = [
                run.difficulty_stats.per_bucket_auroc[bucket_name]  # type: ignore[union-attr,index]
                for run in runs_with_stats
                if run.difficulty_stats is not None
                and run.difficulty_stats.per_bucket_auroc is not None
                and run.difficulty_stats.per_bucket_auroc.get(bucket_name) is not None
            ]
            valid_values = [val for val in raw_values if val is not None]
            per_bucket_auroc[bucket_name] = float(np.mean(valid_values)) if valid_values else None
    # average per-bucket TPR across runs
    per_bucket_tpr: dict[str, dict[float, float]] | None = None
    if first.per_bucket_tpr is not None:
        per_bucket_tpr = {}
        for bucket_name in first.per_bucket_tpr:
            per_bucket_tpr[bucket_name] = {}
            for fpr_val in first.per_bucket_tpr[bucket_name]:
                values = [
                    run.difficulty_stats.per_bucket_tpr[bucket_name][fpr_val]  # type: ignore[union-attr,index]
                    for run in runs_with_stats
                    if run.difficulty_stats is not None
                    and run.difficulty_stats.per_bucket_tpr is not None
                    and bucket_name in run.difficulty_stats.per_bucket_tpr
                    and fpr_val in run.difficulty_stats.per_bucket_tpr[bucket_name]
                ]
                if values:
                    per_bucket_tpr[bucket_name][fpr_val] = float(np.mean(values))
    # average correlation across runs
    corr_values = [
        run.difficulty_stats.difficulty_accuracy_rank_correlation  # type: ignore[union-attr]
        for run in runs_with_stats
        if run.difficulty_stats is not None and run.difficulty_stats.difficulty_accuracy_rank_correlation is not None
    ]
    avg_correlation = float(np.mean(corr_values)) if corr_values else None
    return correctness_types.DifficultyStats(
        bucket_boundaries=first.bucket_boundaries,
        per_bucket_auroc=per_bucket_auroc,
        per_bucket_tpr=per_bucket_tpr,
        per_bucket_sample_count=first.per_bucket_sample_count,
        difficulty_accuracy_rank_correlation=avg_correlation,
    )


def _aggregate_cost_stats(
    per_run_results: list[correctness_types.SingleRunResult],
    target_fpr_values: list[float],
) -> dict[float, correctness_types.VerificationCostStats] | None:
    """Average verification cost stats across runs that report costs.

    Args:
        per_run_results: SingleRunResult for each run.
        target_fpr_values: FPR constraint values.

    Returns:
        Averaged cost stats per target_fpr, or None if no run reports costs.
    """
    runs_with_costs = [run for run in per_run_results if run.verification_cost_stats is not None]
    if not runs_with_costs:
        return None
    aggregated: dict[float, correctness_types.VerificationCostStats] = {}
    for target_fpr in target_fpr_values:
        cost_entries = [
            run.verification_cost_stats[target_fpr]
            for run in runs_with_costs
            if run.verification_cost_stats is not None and target_fpr in run.verification_cost_stats
        ]
        if not cost_entries:
            continue
        cost_units = {entry.cost_unit for entry in cost_entries}
        if len(cost_units) > 1:
            raise ValueError(
                f"cost_unit mismatch across runs at target_fpr={target_fpr}: {cost_units}; "
                f"all guardrail scorers must report the same cost unit"
            )
        aggregated[target_fpr] = correctness_types.VerificationCostStats(
            target_fpr=target_fpr,
            cost_unit=cost_entries[0].cost_unit,
            total_cost=_mean_of_non_none([entry.total_cost for entry in cost_entries]),
            mean_cost_per_record=_mean_of_non_none([entry.mean_cost_per_record for entry in cost_entries]),
            median_cost_per_record=_mean_of_non_none([entry.median_cost_per_record for entry in cost_entries]),
            std_cost_per_record=_mean_of_non_none([entry.std_cost_per_record for entry in cost_entries]),
            cost_per_correct_acceptance=_mean_of_non_none(
                [entry.cost_per_correct_acceptance for entry in cost_entries],
            ),
            cost_per_incorrect_block=_mean_of_non_none(
                [entry.cost_per_incorrect_block for entry in cost_entries],
            ),
            cost_accuracy_rank_correlation=_mean_of_non_none(
                [entry.cost_accuracy_rank_correlation for entry in cost_entries],
            ),
            cost_difficulty_rank_correlation=_mean_of_non_none(
                [entry.cost_difficulty_rank_correlation for entry in cost_entries],
            ),
        )
    return aggregated if aggregated else None


async def evaluate_guardrail_types(
    config: correctness_configs.CorrectnessEvalsConfig,
    guardrails_by_type: dict[str, typing.Sequence[correctness_types.GuardrailScorer]],
    datamodule: correctness_datamodule.CorrectnessDataModule,
    eval_subset_name: str,
    wandb_run: typing.Any | None = None,
    verbose: bool = False,
) -> dict[str, CorrectnessEvalResult]:
    """Evaluate multiple guardrail types independently, each with optional replicas.

    Each key in ``guardrails_by_type`` is a type name (e.g. ``"mean_pool_L8"``). The
    associated sequence contains replica instances of that type, which get cross-run
    aggregation. Results and W&B metrics are prefixed with the type name.

    Args:
        config: Correctness evaluation configuration.
        guardrails_by_type: Mapping from type name to a sequence of GuardrailScorer replicas.
        datamodule: The prepared correctness evaluation datamodule.
        eval_subset_name: Which subset to evaluate on.
        wandb_run: Optional W&B run for metric logging.
        verbose: Whether to verbosely report progress.

    Returns:
        Mapping from type_name to its CorrectnessEvalResult.
    """
    if not guardrails_by_type:
        raise ValueError("guardrails_by_type is empty; nothing to evaluate")
    if not eval_subset_name or not eval_subset_name.strip():
        raise ValueError("eval_subset_name must be set (non-empty, non-whitespace)")
    for type_name in guardrails_by_type:
        if not type_name or not type_name.strip():
            raise ValueError("guardrail type name must not be empty or whitespace")
        lower = type_name.lower()
        if any(lower.startswith(prefix) for prefix in correctness_types.RESERVED_TYPE_NAME_PREFIXES):
            raise ValueError(
                f"guardrail type name {type_name!r} collides with reserved metric namespace prefix; "
                "choose a different name"
            )
        if lower in correctness_types.RESERVED_TYPE_NAME_EXACT:
            raise ValueError(
                f"guardrail type name {type_name!r} collides with reserved metric key; choose a different name"
            )
        if "/" in type_name:
            raise ValueError(
                f"guardrail type name {type_name!r} contains '/' which would create ambiguous W&B key paths; "
                "choose a different name"
            )
    # pre-validate dump paths (including sanitization), and detect filename collisions before eval work
    if config.result_dump_dir is not None:
        seen_paths: dict[str, str] = {}  # normalized filename -> original type_name
        for type_name in guardrails_by_type:
            dump_path = pyine.evals.persistence.build_result_dump_path(
                config.result_dump_dir,
                eval_subset_name,
                type_name=type_name,
            )
            normalized_name = dump_path.name.casefold()
            if normalized_name in seen_paths:
                raise ValueError(
                    f"guardrail type names {seen_paths[normalized_name]!r} and {type_name!r} produce the same "
                    f"dump filename '{dump_path.name}' after sanitization; use distinct type names"
                )
            seen_paths[normalized_name] = type_name
    results: dict[str, CorrectnessEvalResult] = {}
    for type_name, replicas in guardrails_by_type.items():
        if not replicas:
            raise ValueError(f"guardrail type '{type_name}' has an empty replicas list")
        logger.info(f"evaluating guardrail type '{type_name}' ({len(replicas)} replica(s))...")
        result = await evaluate_guardrail_replicas(
            config=config,
            guardrails=replicas,
            datamodule=datamodule,
            eval_subset_name=eval_subset_name,
            verbose=verbose,
        )
        result = result.model_copy(
            update={
                "eval_metadata": {
                    **result.eval_metadata,
                    "guardrail_type_name": type_name,
                }
            }
        )
        if wandb_run is not None:
            type_prefix = f"benchmark/{eval_subset_name}/{type_name}"
            for metric_name, metric_val in result.metrics.items():
                wandb_run.summary[f"{type_prefix}/{metric_name}"] = metric_val  # type: ignore[reportUnknownMemberType]
        results[type_name] = result
        pyine.evals.persistence.maybe_dump_eval_result(
            result=result,
            dump_dir=config.result_dump_dir,
            eval_subset_name=eval_subset_name,
            type_name=type_name,
            overwrite=config.result_dump_overwrite,
        )
    if wandb_run is not None:
        wandb_run.summary[f"benchmark/{eval_subset_name}/_guardrail_type_names"] = sorted(results.keys())  # type: ignore[reportUnknownMemberType]
    return results
