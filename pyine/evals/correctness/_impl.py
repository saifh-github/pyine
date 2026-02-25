"""Main pipeline orchestration for guardrail correctness evaluation.

Provides ``evaluate_wrapped_model_impl`` which runs the full evaluation pipeline:
load records, build splits, score, calibrate, compute metrics, aggregate.
"""

from __future__ import annotations

import logging
import typing

import numpy as np
import pydantic

import pyine.evals.common
import pyine.evals.correctness.calibration as correctness_calibration
import pyine.evals.correctness.data_loading as correctness_data_loading
import pyine.evals.correctness.metrics as correctness_metrics
import pyine.evals.correctness.splits as correctness_splits
import pyine.evals.correctness.types as correctness_types

if typing.TYPE_CHECKING:
    import pyine.evals.correctness.configs as correctness_configs

logger = logging.getLogger(__name__)


class CorrectnessEvalResult(pyine.evals.common.EvalResult):
    """Correctness evaluation results container.

    Extends EvalResult with a AggregatedResult object for downstream consumers that need its rich
    structure (per-run results, bootstrap CIs, etc.). The base ``metrics`` dict from the parent
    class contains the flattened pipeline-mean metrics.
    """

    model_config = pydantic.ConfigDict(frozen=True, arbitrary_types_allowed=True)
    """Pydantic model configuration (freezes the dataclass)."""

    aggregated: correctness_types.AggregatedResult
    """Full aggregated result with per-run details, splits, and CIs."""


async def evaluate_wrapped_model_impl(
    config: correctness_configs.CorrectnessEvalsConfig,
    guardrails: typing.Sequence[correctness_types.GuardrailScorer],
    verbose: bool = False,
) -> CorrectnessEvalResult:
    """Run the full guardrail correctness evaluation pipeline.

    Args:
        config: Correctness evaluation configuration.
        guardrails: Sequence of one or more GuardrailScorer instances to evaluate (we might be
            using replicas).
        verbose: Whether to verbosely report progress.

    Returns:
        CorrectnessEvalResult with flattened metrics and full AggregatedResult.
    """
    if not guardrails:
        raise ValueError("guardrails must not be empty; provide at least one GuardrailScorer")
    # step 1: load records
    records = correctness_data_loading.load_records_from_lmdb(config.lmdb_paths, config.label_type)
    if not records:
        raise ValueError(
            f"no records loaded from {len(config.lmdb_paths)} LMDB path(s); "
            f"check that the paths exist and contain benchmark records"
        )
    logger.info(f"loaded {len(records)} records from {len(config.lmdb_paths)} LMDB(s)")
    # step 2: build splits
    guardrail_splits = correctness_splits.build_guardrail_splits(records, config.split_config)
    logger.info(
        f"split: train={len(guardrail_splits.guardrail_train)}, "
        f"valid={len(guardrail_splits.guardrail_valid)}, "
        f"test={len(guardrail_splits.guardrail_test)}"
    )
    # step 2b: validate splits are usable
    if not guardrail_splits.guardrail_valid:
        raise ValueError(
            "guardrail_valid split is empty after re-splitting; increase the number of "
            "original validation problems or adjust guardrail_valid_fraction"
        )
    if not guardrail_splits.guardrail_test:
        raise ValueError(
            "guardrail_test split is empty; ensure the split file contains test problems that match the LMDB records"
        )
    valid_has_positive = any(rec.label for rec in guardrail_splits.guardrail_valid)
    valid_has_negative = any(not rec.label for rec in guardrail_splits.guardrail_valid)
    if not valid_has_positive or not valid_has_negative:
        raise ValueError(
            "guardrail_valid split has only one label class; threshold calibration requires "
            "both correct and incorrect attempts in the validation set"
        )
    # step 3: class balance on test
    test_class_balance = correctness_metrics.compute_class_balance(guardrail_splits.guardrail_test)
    # step 4: run each guardrail
    per_run_results: list[correctness_types.SingleRunResult] = []
    per_run_records_for_hierarchical: list[list[correctness_types.EvalRecord]] = []
    per_run_scores_for_hierarchical: list[np.ndarray[typing.Any, np.dtype[np.floating[typing.Any]]]] = []
    per_run_thresholds_for_hierarchical: dict[float, list[float]] = {fpr: [] for fpr in config.target_fpr_values}
    for run_idx, guardrail in enumerate(guardrails):
        if verbose:
            logger.info(f"evaluating guardrail run {run_idx + 1}/{len(guardrails)}")
        run_result = _evaluate_single_run(
            guardrail=guardrail,
            guardrail_splits=guardrail_splits,
            config=config,
        )
        per_run_results.append(run_result.result)
        per_run_records_for_hierarchical.append(guardrail_splits.guardrail_test)
        per_run_scores_for_hierarchical.append(run_result.test_scores)
        for target_fpr in config.target_fpr_values:
            per_run_thresholds_for_hierarchical[target_fpr].append(run_result.thresholds[target_fpr])
    # step 5: aggregate across runs
    aggregated = _aggregate_runs(
        per_run_results=per_run_results,
        per_run_records=per_run_records_for_hierarchical,
        per_run_scores=per_run_scores_for_hierarchical,
        per_run_thresholds=per_run_thresholds_for_hierarchical,
        guardrail_splits=guardrail_splits,
        test_class_balance=test_class_balance,
        config=config,
    )
    return CorrectnessEvalResult(
        metrics=aggregated.to_flat_dict(),
        aggregated=aggregated,
    )


class _SingleRunOutput(typing.NamedTuple):
    """Internal container for a single guardrail run's output."""

    result: correctness_types.SingleRunResult
    test_scores: np.ndarray[typing.Any, np.dtype[np.floating[typing.Any]]]
    thresholds: dict[float, float]


def _evaluate_single_run(
    guardrail: correctness_types.GuardrailScorer,
    guardrail_splits: correctness_types.GuardrailSplits,
    config: correctness_configs.CorrectnessEvalsConfig,
) -> _SingleRunOutput:
    """Evaluate a single guardrail instance.

    Args:
        guardrail: GuardrailScorer instance.
        guardrail_splits: Train/valid/test splits.
        config: Evaluation configuration.

    Returns:
        _SingleRunOutput with results, test scores, and thresholds.
    """
    # score valid and test sets
    valid_result = guardrail.score_records(guardrail_splits.guardrail_valid)
    if len(valid_result.scores) != len(guardrail_splits.guardrail_valid):
        raise ValueError(
            f"scorer returned {len(valid_result.scores)} scores for "
            f"{len(guardrail_splits.guardrail_valid)} valid records"
        )
    test_result = guardrail.score_records(guardrail_splits.guardrail_test)
    if len(test_result.scores) != len(guardrail_splits.guardrail_test):
        raise ValueError(
            f"scorer returned {len(test_result.scores)} scores for {len(guardrail_splits.guardrail_test)} test records"
        )
    metadata = guardrail.get_metadata()
    cost_unit = guardrail.get_verification_cost_unit()
    valid_scores = np.array(valid_result.scores, dtype=np.float64)
    valid_labels = np.array([rec.label for rec in guardrail_splits.guardrail_valid], dtype=np.bool_)
    test_scores = np.array(test_result.scores, dtype=np.float64)
    test_labels = np.array([rec.label for rec in guardrail_splits.guardrail_test], dtype=np.bool_)
    test_records = guardrail_splits.guardrail_test
    # calibrate thresholds and compute thresholded metrics
    thresholds: dict[float, float] = {}
    attempt_metrics: dict[float, correctness_types.ThresholdedMetrics] = {}
    sample_metrics: dict[float, correctness_types.SampleLevelMetrics] = {}
    verification_cost_stats: dict[float, correctness_types.VerificationCostStats] = {}
    for target_fpr in config.target_fpr_values:
        threshold = correctness_calibration.calibrate_threshold(valid_scores, valid_labels, target_fpr)
        thresholds[target_fpr] = threshold
        attempt_metrics[target_fpr] = correctness_metrics.compute_thresholded_metrics(
            test_scores,
            test_labels,
            threshold,
            target_fpr,
        )
        sample_metrics[target_fpr] = correctness_metrics.compute_sample_level_metrics(
            test_records,
            test_scores,
            threshold,
            target_fpr,
        )
        # verification cost stats
        accepted = test_scores >= threshold
        cost_stats = correctness_metrics.compute_verification_cost_stats(  # type: ignore[reportUnknownMemberType]
            test_result.verification_costs,
            test_labels,
            accepted,
            target_fpr,
            cost_unit=cost_unit,
            records=test_records,
        )
        if cost_stats is not None:
            verification_cost_stats[target_fpr] = cost_stats
    # threshold-free metrics
    threshold_free = correctness_metrics.compute_threshold_free_metrics(  # type: ignore[reportUnknownMemberType]
        test_scores,
        test_labels,
        config.target_fpr_values,
        config.roc_fpr_grid_size,
    )
    # category-wise metrics
    category_results = correctness_metrics.compute_category_results(
        test_records,
        test_scores,
        thresholds,
        config.target_fpr_values,
        config.category_config,
        config.roc_fpr_grid_size,
        base_extraction_config=config.category_extraction_config,
    )
    # difficulty stats
    difficulty_stats = correctness_metrics.compute_difficulty_stats(  # type: ignore[reportUnknownMemberType]
        test_records,
        test_scores,
        test_labels,
        thresholds,
        config.target_fpr_values,
    )
    # bootstrap CIs
    bootstrap_cis = correctness_metrics.compute_clustered_bootstrap_cis(  # type: ignore[reportUnknownMemberType]
        test_records,
        test_scores,
        thresholds,
        config.target_fpr_values,
        config.num_bootstrap_replicates,
        config.bootstrap_seed,
        config.confidence_level,
    )
    run_result = correctness_types.SingleRunResult(
        guardrail_metadata=metadata,
        threshold_free=threshold_free,
        attempt_metrics=attempt_metrics,
        sample_metrics=sample_metrics,
        category_results=category_results,
        bootstrap_cis=bootstrap_cis,
        difficulty_stats=difficulty_stats,
        verification_cost_stats=verification_cost_stats if verification_cost_stats else None,
    )
    return _SingleRunOutput(result=run_result, test_scores=test_scores, thresholds=thresholds)


def _aggregate_runs(
    per_run_results: list[correctness_types.SingleRunResult],
    per_run_records: list[list[correctness_types.EvalRecord]],
    per_run_scores: list[np.ndarray[typing.Any, np.dtype[np.floating[typing.Any]]]],
    per_run_thresholds: dict[float, list[float]],
    guardrail_splits: correctness_types.GuardrailSplits,
    test_class_balance: correctness_types.ClassBalanceStats,
    config: correctness_configs.CorrectnessEvalsConfig,
) -> correctness_types.AggregatedResult:
    """Aggregate results across R independent guardrail runs.

    Args:
        per_run_results: SingleRunResult for each run.
        per_run_records: Test records for each run (same for all runs currently).
        per_run_scores: Test score arrays for each run.
        per_run_thresholds: Thresholds per target_fpr per run.
        guardrail_splits: The split used for all runs.
        test_class_balance: Class balance stats for the test set.
        config: Evaluation configuration.

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
            att = run_result.attempt_metrics.get(target_fpr)
            if att is not None:
                _collect_scalar(f"{fpr_key}/tpr", att.tpr, metric_values)
                _collect_scalar(f"{fpr_key}/fpr", att.fpr, metric_values)
                _collect_scalar(f"{fpr_key}/fnr", att.fnr, metric_values)
                _collect_scalar(f"{fpr_key}/precision", att.precision, metric_values)
                _collect_scalar(f"{fpr_key}/npv", att.npv, metric_values)
            samp = run_result.sample_metrics.get(target_fpr)
            if samp is not None:
                _collect_scalar(f"{fpr_key}/base_pass_rate", samp.base_pass_rate, metric_values)
                _collect_scalar(f"{fpr_key}/guarded_pass_rate", samp.guarded_pass_rate, metric_values)
                _collect_scalar(f"{fpr_key}/unsafe_slip_rate", samp.unsafe_slip_rate, metric_values)
                _collect_scalar(f"{fpr_key}/total_block_rate", samp.total_block_rate, metric_values)
                _collect_scalar(f"{fpr_key}/best_of_k_success_rate", samp.best_of_k_success_rate, metric_values)
                _collect_scalar(f"{fpr_key}/cons_pass_rate", samp.cons_pass_rate, metric_values)
                _collect_scalar(f"{fpr_key}/cons_unsafe_slip_rate", samp.cons_unsafe_slip_rate, metric_values)
                _collect_scalar(
                    f"{fpr_key}/cons_justified_reject_rate",
                    samp.cons_justified_reject_rate,
                    metric_values,
                )
        # category metrics
        for cat_name, cat_result in run_result.category_results.items():
            safe_cat = cat_name.replace("/", "_")
            _collect_scalar(f"category/{safe_cat}/auroc", cat_result.threshold_free.auroc, metric_values)
            _collect_scalar(
                f"category/{safe_cat}/average_precision",
                cat_result.threshold_free.average_precision,
                metric_values,
            )
            for target_fpr in config.target_fpr_values:
                fpr_key = correctness_metrics.format_fpr_key(target_fpr)
                cat_att = cat_result.attempt_metrics.get(target_fpr)
                if cat_att is not None:
                    _collect_scalar(f"category/{safe_cat}/{fpr_key}/tpr", cat_att.tpr, metric_values)
                    _collect_scalar(f"category/{safe_cat}/{fpr_key}/fpr", cat_att.fpr, metric_values)
                    _collect_scalar(f"category/{safe_cat}/{fpr_key}/fnr", cat_att.fnr, metric_values)
                    _collect_scalar(f"category/{safe_cat}/{fpr_key}/precision", cat_att.precision, metric_values)
                    _collect_scalar(f"category/{safe_cat}/{fpr_key}/npv", cat_att.npv, metric_values)
                cat_samp = cat_result.sample_metrics.get(target_fpr)
                if cat_samp is not None:
                    _collect_scalar(
                        f"category/{safe_cat}/{fpr_key}/base_pass_rate",
                        cat_samp.base_pass_rate,
                        metric_values,
                    )
                    _collect_scalar(
                        f"category/{safe_cat}/{fpr_key}/guarded_pass_rate",
                        cat_samp.guarded_pass_rate,
                        metric_values,
                    )
                    _collect_scalar(
                        f"category/{safe_cat}/{fpr_key}/unsafe_slip_rate",
                        cat_samp.unsafe_slip_rate,
                        metric_values,
                    )
                    _collect_scalar(
                        f"category/{safe_cat}/{fpr_key}/total_block_rate",
                        cat_samp.total_block_rate,
                        metric_values,
                    )
                    _collect_scalar(
                        f"category/{safe_cat}/{fpr_key}/best_of_k_success_rate",
                        cat_samp.best_of_k_success_rate,
                        metric_values,
                    )
                    _collect_scalar(
                        f"category/{safe_cat}/{fpr_key}/cons_pass_rate",
                        cat_samp.cons_pass_rate,
                        metric_values,
                    )
                    _collect_scalar(
                        f"category/{safe_cat}/{fpr_key}/cons_unsafe_slip_rate",
                        cat_samp.cons_unsafe_slip_rate,
                        metric_values,
                    )
                    _collect_scalar(
                        f"category/{safe_cat}/{fpr_key}/cons_justified_reject_rate",
                        cat_samp.cons_justified_reject_rate,
                        metric_values,
                    )
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
        cross_run_std[metric_name] = float(np.std(arr))
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
    )
    # difficulty stats: aggregate scalar fields across runs
    difficulty_stats = _aggregate_difficulty_stats(per_run_results)
    # verification cost stats: average across runs that report costs
    verification_cost_stats = _aggregate_cost_stats(per_run_results, config.target_fpr_values)
    return correctness_types.AggregatedResult(
        split_summary=guardrail_splits.to_summary(),
        class_balance=test_class_balance,
        per_run=per_run_results,
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

        def _mean_of_non_none(values: list[float | None]) -> float | None:
            valid = [val for val in values if val is not None]
            return float(np.mean(valid)) if valid else None

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
