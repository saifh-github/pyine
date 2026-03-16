"""Metrics computation for the guardrail correctness evaluation pipeline."""

from __future__ import annotations

import collections
import functools
import logging
import os
import typing

import numpy as np
import numpy.typing as npt
import scipy.stats
import sklearn.metrics

import pyine.evals.correctness.types as correctness_types
import pyine.evals.utils
import pyine.organisms.datamodules.samples.common as samples_common
import pyine.utils.concurrency
import pyine.utils.metrics.confidence

if typing.TYPE_CHECKING:
    import pyine.evals.correctness.configs as correctness_configs

logger = logging.getLogger(__name__)


def _assert_finite(
    array: npt.NDArray[typing.Any],
    name: str,
) -> None:
    """Raise ValueError if the array contains NaN or inf values."""
    if not np.all(np.isfinite(array)):
        num_bad = int(np.sum(~np.isfinite(array)))
        raise ValueError(f"{name} contains {num_bad} non-finite value(s) (NaN or inf)")


def _is_constant(
    array: npt.NDArray[np.float64],
) -> bool:
    """Return True when all values in the array are (approximately) identical."""
    if array.size == 0:
        return True
    return np.allclose(array, array[0])


@typing.no_type_check  # sklearn type stubs are partially unknown
def compute_threshold_free_metrics(
    scores: npt.NDArray[np.floating[typing.Any]],
    labels: npt.NDArray[np.bool_],
    target_fprs: list[float],
    fpr_grid_size: int,
) -> correctness_types.ThresholdFreeMetrics:
    """Compute threshold-independent metrics (AUROC, average precision, TPR@FPR, curve grids).

    ROC and PR curves are computed via sklearn. The raw curve points may contain duplicate
    x-values (FPR or recall) due to tied scores; before interpolation onto a uniform grid,
    duplicates are deduplicated by keeping the maximum y-value (TPR or precision) at each
    unique x-value. This ensures ``np.interp`` receives strictly monotonic x-values.

    Returns all-None metrics with empty grids when the subset contains only one label class
    (e.g. a category where all attempts are correct or all incorrect).

    Args:
        scores: Continuous guardrail scores (must contain only finite values).
        labels: Boolean correctness labels.
        target_fprs: FPR levels at which to report TPR (via interpolation on the ROC curve).
        fpr_grid_size: Number of uniformly spaced points in the FPR grid for plotting.

    Returns:
        ThresholdFreeMetrics with AUROC, average precision, TPR@FPR, and curve grids.
    """
    if not target_fprs:
        raise ValueError("target_fprs must be non-empty")
    if any(not (0.0 <= fpr <= 1.0) for fpr in target_fprs):
        raise ValueError(f"target_fprs must all be in [0, 1], got {target_fprs}")
    if fpr_grid_size <= 0:
        raise ValueError(f"fpr_grid_size must be positive, got {fpr_grid_size}")
    if len(scores) != len(labels):
        raise ValueError(f"scores and labels must have the same length, got {len(scores)} and {len(labels)}")
    if len(scores) > 0:
        _assert_finite(scores, "scores")
    if len(labels) == 0:
        empty = np.array([], dtype=np.float64)
        return correctness_types.ThresholdFreeMetrics(
            auroc=None,
            average_precision=None,
            tpr_at_fpr=None,
            fpr_grid=empty,
            tpr_grid=empty,
            precision_grid=empty,
            recall_grid=empty,
        )
    if labels.dtype != np.bool_:
        raise ValueError(f"labels must be a boolean array, got dtype={labels.dtype}")
    # single-class check (after dtype validation to catch bad input even for single-class)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        logger.warning(
            f"single-class subset found (all labels are {unique_labels[0]}); "
            "AUROC/average precision are undefined, returning None"
        )
        empty = np.array([], dtype=np.float64)
        return correctness_types.ThresholdFreeMetrics(
            auroc=None,
            average_precision=None,
            tpr_at_fpr=None,
            fpr_grid=empty,
            tpr_grid=empty,
            precision_grid=empty,
            recall_grid=empty,
        )
    if not np.array_equal(unique_labels, np.array([False, True])):
        raise ValueError(f"expected both label classes [False, True], got {unique_labels}")
    # compute ROC
    fpr_raw, tpr_raw, _ = sklearn.metrics.roc_curve(labels, scores)
    auroc = float(sklearn.metrics.roc_auc_score(labels, scores))
    # compute PR curve
    precision_raw, recall_raw, _ = sklearn.metrics.precision_recall_curve(labels, scores)
    average_precision = float(sklearn.metrics.average_precision_score(labels, scores))
    # deduplicate ROC x-values: collapse tied FPR values to unique entries with max TPR
    assert np.all(np.diff(fpr_raw) >= 0), "fpr_raw from roc_curve must be non-decreasing"
    fpr_unique, unique_indices = np.unique(fpr_raw, return_index=True)
    # for each unique FPR, take the max TPR (last occurrence in the non-decreasing sequence)
    tpr_at_unique = np.empty_like(fpr_unique)
    for idx in range(len(fpr_unique)):
        end = unique_indices[idx + 1] if idx + 1 < len(unique_indices) else len(fpr_raw)
        tpr_at_unique[idx] = np.max(tpr_raw[unique_indices[idx] : end])
    # TPR@FPR via interpolation on deduplicated (strictly increasing) FPR values
    tpr_at_fpr: dict[float, float] = {}
    for target_fpr in target_fprs:
        tpr_at_fpr[target_fpr] = float(np.interp(target_fpr, fpr_unique, tpr_at_unique))
    # resample to fixed grid for plotting
    fpr_grid = np.linspace(0.0, 1.0, fpr_grid_size)
    tpr_grid = np.interp(fpr_grid, fpr_unique, tpr_at_unique)
    # deduplicate PR x-values: precision_recall_curve returns recall in decreasing order,
    # so sort by recall ascending, then collapse tied recall values with max precision
    recall_sorted_idx = np.argsort(recall_raw)
    recall_sorted = recall_raw[recall_sorted_idx]
    precision_sorted = precision_raw[recall_sorted_idx]
    recall_unique, recall_unique_indices = np.unique(recall_sorted, return_index=True)
    precision_at_unique = np.empty_like(recall_unique)
    for idx in range(len(recall_unique)):
        end = recall_unique_indices[idx + 1] if idx + 1 < len(recall_unique_indices) else len(recall_sorted)
        precision_at_unique[idx] = np.max(precision_sorted[recall_unique_indices[idx] : end])
    recall_grid = np.linspace(0.0, 1.0, fpr_grid_size)
    precision_grid = np.interp(recall_grid, recall_unique, precision_at_unique)
    return correctness_types.ThresholdFreeMetrics(
        auroc=auroc,
        average_precision=average_precision,
        tpr_at_fpr=tpr_at_fpr,
        fpr_grid=fpr_grid.astype(np.float64),
        tpr_grid=tpr_grid.astype(np.float64),
        precision_grid=precision_grid.astype(np.float64),
        recall_grid=recall_grid.astype(np.float64),
    )


def compute_thresholded_metrics(
    scores: npt.NDArray[np.floating[typing.Any]],
    labels: npt.NDArray[np.bool_],
    threshold: float,
    target_fpr: float,
) -> correctness_types.ThresholdedMetrics:
    """Compute attempt-level confusion matrix and derived rates at a calibrated threshold.

    Args:
        scores: Continuous guardrail scores.
        labels: Boolean correctness labels.
        threshold: The calibrated decision threshold (accept if score >= threshold).
        target_fpr: The FPR constraint used to calibrate the threshold.

    Returns:
        ThresholdedMetrics with confusion matrix counts and rates.
    """
    if len(scores) != len(labels):
        raise ValueError(f"scores and labels must have the same length, got {len(scores)} and {len(labels)}")
    if len(scores) == 0:
        raise ValueError("scores and labels must be non-empty")
    if labels.dtype != np.bool_:
        raise ValueError(f"labels must be a boolean array, got dtype={labels.dtype}")
    _assert_finite(scores, "scores")
    if not np.isfinite(threshold):
        raise ValueError(f"threshold must be finite, got {threshold}")
    if not (0.0 < target_fpr < 1.0):
        raise ValueError(f"target_fpr must be in (0, 1), got {target_fpr}")
    accepted = scores >= threshold
    tp = int(np.sum(labels & accepted))
    fp = int(np.sum(~labels & accepted))
    tn = int(np.sum(~labels & ~accepted))
    fn = int(np.sum(labels & ~accepted))
    tpr = tp / (tp + fn) if (tp + fn) > 0 else None
    fpr = fp / (fp + tn) if (fp + tn) > 0 else None
    fnr = fn / (tp + fn) if (tp + fn) > 0 else None
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    npv = tn / (tn + fn) if (tn + fn) > 0 else None
    return correctness_types.ThresholdedMetrics(
        target_fpr=target_fpr,
        threshold=threshold,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        tpr=tpr,
        fpr=fpr,
        fnr=fnr,
        precision=precision,
        npv=npv,
    )


def compute_sample_level_metrics(
    records: list[correctness_types.EvalRecord],
    scores: npt.NDArray[np.floating[typing.Any]],
    threshold: float,
    target_fpr: float,
    sample_keys: list[str] | None = None,
) -> correctness_types.SampleLevelMetrics:
    """Compute sample-level metrics (across K attempts per sample).

    Groups records by sample_id (or by ``sample_keys`` when provided). For each sample:
    - base_pass: any(label), i.e. at least one correct attempt exists;
    - guarded_pass: any(label and accepted), i.e. at least one correct attempt accepted;
    - unsafe_slip: any(not label and accepted), i.e. at least one incorrect attempt accepted;
    - total_block: all(not accepted), i.e. guardrail blocked every attempt;
    - best_of_k_success: among samples with >=1 accepted attempt, is the highest-scored
      accepted attempt correct? This uses score-based ranking to model a deployment strategy that
      selects the most-confident accepted output.

    Conservative deployment strategy (reject sample if ANY attempt is blocked):
    - cons_pass: all(accepted), i.e. every attempt passes the guardrail;
    - cons_unsafe_slip: among cons-accepted samples, any(not label), i.e. at least one
      attempt is actually incorrect;
    - cons_justified_reject: among cons-rejected samples, any((not label) and (not accepted)),
      i.e. at least one blocked attempt is truly incorrect.

    Args:
        records: EvalRecords (must match scores in order).
        scores: Continuous guardrail scores (must contain only finite values).
        threshold: The calibrated decision threshold (accept if score >= threshold).
        target_fpr: The FPR constraint used to calibrate the threshold.
        sample_keys: Optional per-record grouping keys. When provided, records are grouped
            by these keys instead of by ``record.sample_id``. This is required for bootstrap
            resampling where the same problem (and its sample_ids) can appear multiple times;
            without unique keys, duplicate occurrences would collapse into a single group,
            biasing sample-level CIs.

    Returns:
        SampleLevelMetrics with rates averaged across samples.
    """
    if len(scores) != len(records):
        raise ValueError(f"scores and records must have the same length, got {len(scores)} and {len(records)}")
    if len(records) == 0:
        raise ValueError("records must be non-empty")
    _assert_finite(scores, "scores")
    if not np.isfinite(threshold):
        raise ValueError(f"threshold must be finite, got {threshold}")
    if not (0.0 < target_fpr < 1.0):
        raise ValueError(f"target_fpr must be in (0, 1), got {target_fpr}")
    if sample_keys is not None and len(sample_keys) != len(records):
        raise ValueError(f"sample_keys length ({len(sample_keys)}) must match records length ({len(records)})")
    accepted_mask = scores >= threshold
    # group by sample_keys (if provided) or sample_id
    sample_groups: dict[str, list[int]] = collections.defaultdict(list)
    for idx, record in enumerate(records):
        key = sample_keys[idx] if sample_keys is not None else record.sample_id
        sample_groups[key].append(idx)
    num_samples = len(sample_groups)
    base_pass_count = 0
    guarded_pass_count = 0
    unsafe_slip_count = 0
    total_block_count = 0
    selection_success_count = 0
    selection_eligible_count = 0
    cons_pass_count = 0
    cons_unsafe_slip_count = 0
    cons_eligible_for_slip = 0
    cons_rejected_count = 0
    cons_justified_count = 0
    for sample_id in sample_groups:
        indices = sample_groups[sample_id]
        sample_labels = [records[idx].label for idx in indices]
        sample_accepted = [bool(accepted_mask[idx]) for idx in indices]
        sample_scores = [float(scores[idx]) for idx in indices]
        has_correct = any(sample_labels)
        has_accepted_correct = any(lab and acc for lab, acc in zip(sample_labels, sample_accepted, strict=True))
        has_accepted_incorrect = any((not lab) and acc for lab, acc in zip(sample_labels, sample_accepted, strict=True))
        all_blocked = not any(sample_accepted)
        all_accepted = all(sample_accepted)
        if has_correct:
            base_pass_count += 1
        if has_accepted_correct:
            guarded_pass_count += 1
        if has_accepted_incorrect:
            unsafe_slip_count += 1
        if all_blocked:
            total_block_count += 1
        if not all_blocked:
            selection_eligible_count += 1
            # select the highest-scored accepted attempt (tie-aware: success if any top-scored is correct)
            accepted_scores = [
                sample_scores[local_idx] for local_idx in range(len(indices)) if sample_accepted[local_idx]
            ]
            max_accepted_score = max(accepted_scores)
            top_is_correct = any(
                sample_labels[local_idx]
                for local_idx in range(len(indices))
                if sample_accepted[local_idx] and sample_scores[local_idx] == max_accepted_score
            )
            if top_is_correct:
                selection_success_count += 1
        # conservative deployment strategy: accept sample only if ALL attempts pass
        if all_accepted:
            cons_pass_count += 1
            cons_eligible_for_slip += 1
            if any(not lab for lab in sample_labels):
                cons_unsafe_slip_count += 1
        else:
            cons_rejected_count += 1
            blocked_has_incorrect = any(
                (not lab) and (not acc) for lab, acc in zip(sample_labels, sample_accepted, strict=True)
            )
            if blocked_has_incorrect:
                cons_justified_count += 1
    base_pass_rate = base_pass_count / num_samples if num_samples > 0 else 0.0
    guarded_pass_rate = guarded_pass_count / num_samples if num_samples > 0 else 0.0
    unsafe_slip_rate = unsafe_slip_count / num_samples if num_samples > 0 else 0.0
    total_block_rate = total_block_count / num_samples if num_samples > 0 else 0.0
    best_of_k_success_rate: float | None
    if selection_eligible_count > 0:
        best_of_k_success_rate = selection_success_count / selection_eligible_count
    else:
        best_of_k_success_rate = None
    cons_pass_rate = cons_pass_count / num_samples if num_samples > 0 else 0.0
    cons_unsafe_slip_rate: float | None
    if cons_eligible_for_slip > 0:
        cons_unsafe_slip_rate = cons_unsafe_slip_count / cons_eligible_for_slip
    else:
        cons_unsafe_slip_rate = None
    cons_justified_reject_rate: float | None
    if cons_rejected_count > 0:
        cons_justified_reject_rate = cons_justified_count / cons_rejected_count
    else:
        cons_justified_reject_rate = None
    return correctness_types.SampleLevelMetrics(
        target_fpr=target_fpr,
        base_pass_rate=base_pass_rate,
        guarded_pass_rate=guarded_pass_rate,
        unsafe_slip_rate=unsafe_slip_rate,
        total_block_rate=total_block_rate,
        best_of_k_success_rate=best_of_k_success_rate,
        cons_pass_rate=cons_pass_rate,
        cons_unsafe_slip_rate=cons_unsafe_slip_rate,
        cons_justified_reject_rate=cons_justified_reject_rate,
    )


def _parse_code_type_set(
    raw_code_type: str,
) -> frozenset[samples_common.SampleCodeType]:
    """Parse a code_type string into its component SampleCodeType values.

    Delegates to ``get_code_type_set_from_str`` but treats unrecognized strings as unknown
    rather than silently falling back to ``{original}`` (which is the dataset-level default).
    Returns an empty frozenset for truly unknown code_type strings.
    """
    code_type_set = samples_common.get_code_type_set_from_str(raw_code_type)
    # get_code_type_set_from_str returns {original} for unrecognized strings; detect this
    # fallback by checking whether the raw string could have produced {original} legitimately
    if code_type_set == frozenset({samples_common.SampleCodeType.original}) and raw_code_type != "original":
        return frozenset()
    return code_type_set


def _adapt_record_for_base_extractor(
    record_data: typing.Mapping[str, typing.Any],
) -> dict[str, typing.Any]:
    """Adapt an LMDB-style record dict for use with SampleCategoryExtractor.

    LMDB records use different field names than SampleCategoryExtractor expects:
    - ``sample_id`` instead of ``identifier``
    - ``tags`` (list) instead of ``comma_separated_tags`` (string)

    Only fills missing fields; never clobbers existing ones.

    Args:
        record_data: Record data mapping (e.g. from EvalRecord.record).

    Returns:
        Adapted copy with fields renamed for SampleCategoryExtractor compatibility.
    """
    adapted = dict(record_data)
    if "identifier" not in adapted and "sample_id" in adapted:
        adapted["identifier"] = adapted["sample_id"]
    if "comma_separated_tags" not in adapted:
        tags = adapted.get("tags")
        if isinstance(tags, (list, tuple)):
            tags_seq = typing.cast("list[typing.Any]", tags)
            adapted["comma_separated_tags"] = ",".join(str(tag) for tag in tags_seq)
    return adapted


def categorize_records(
    records: list[correctness_types.EvalRecord],
    category_config: correctness_configs.RecordCategoryConfig,
    base_extraction_config: pyine.evals.utils.SampleCategoryExtractionConfig | None = None,
) -> dict[str, list[int]]:
    """Assign each record to one or more categories based on its code_type and optional base fields.

    Code_type grouping (via ``category_config``) produces semantic categories like ``regular``,
    ``biasing/hinted``, etc.

    When ``base_extraction_config`` is provided, additional categories are extracted from other
    record fields (predict_type, has_keyword, identifier_suffix, tags) using ``SampleCategoryExtractor``.
    The ``code_type`` field is automatically filtered out of the base extractor to avoid
    duplicating code_type categories.

    Per-record categories are collected into a set before indexing to prevent duplicate counting.

    Args:
        records: EvalRecords to categorize.
        category_config: Configuration for code_type-based category assignment.
        base_extraction_config: Optional config for extracting categories from other sample
            fields. When None, only code_type grouping is performed.

    Returns:
        Mapping from category name to list of record indices.
    """
    # build base extractor (with code_type filtered out) if requested
    base_extractor: pyine.evals.utils.SampleCategoryExtractor | None = None
    if base_extraction_config is not None:
        filtered_fields = [
            field
            for field in base_extraction_config.enabled_fields
            if field != pyine.evals.utils.SampleCategoryField.code_type
        ]
        filtered_config = pyine.evals.utils.SampleCategoryExtractionConfig(
            enabled_fields=filtered_fields,
            tag_prefixes=base_extraction_config.tag_prefixes,
        )
        if filtered_config.enabled_fields:
            base_extractor = pyine.evals.utils.SampleCategoryExtractor(filtered_config)
    category_to_indices: dict[str, list[int]] = collections.defaultdict(list)
    code_type_map = category_config.code_type_to_category
    warned_code_types: set[str] = set()
    for idx, record in enumerate(records):
        per_record_categories: set[str] = set()
        # step 1: grouped code_type categories (existing logic)
        code_type_set = _parse_code_type_set(record.code_type)
        matched_categories: set[str] = set()
        has_regular = False
        for code_type in code_type_set:
            code_type_str = code_type.value
            if code_type_str in code_type_map:
                cat = code_type_map[code_type_str]
                if cat == "regular":
                    has_regular = True
                else:
                    matched_categories.add(cat)
            elif code_type_str not in warned_code_types:
                logger.warning(f"code_type '{code_type_str}' (from '{record.code_type}') not in category map")
                warned_code_types.add(code_type_str)
        if matched_categories:
            per_record_categories.update(matched_categories)
            if len(matched_categories) > 1:
                # use the canonical code type set string (sorted types joined with underscore)
                combined = "biasing/" + "_".join(sorted(code_type.value for code_type in code_type_set))
                per_record_categories.add(combined)
        elif has_regular:
            per_record_categories.add("regular")
        else:
            per_record_categories.add("other")
        if category_config.report_per_code_type:
            per_record_categories.add(f"code_type/{record.code_type}")
        # step 2: base extractor categories (predict_type, has_keyword, etc.)
        if base_extractor is not None:
            adapted = _adapt_record_for_base_extractor(record.record)
            adapted.setdefault("sample_id", record.sample_id)
            adapted.setdefault("code_type", record.code_type)
            base_categories = base_extractor.extract_categories(adapted)
            per_record_categories.update(base_categories)
        # step 3: add deduplicated categories to the inverted index
        for cat in per_record_categories:
            category_to_indices[cat].append(idx)
    return dict(category_to_indices)


def compute_category_results(
    records: list[correctness_types.EvalRecord],
    scores: npt.NDArray[np.floating[typing.Any]],
    thresholds: dict[float, float],
    target_fpr_values: list[float],
    category_config: correctness_configs.RecordCategoryConfig,
    fpr_grid_size: int,
    base_extraction_config: pyine.evals.utils.SampleCategoryExtractionConfig | None = None,
) -> dict[str, correctness_types.CategoryResult]:
    """Compute all metrics per category.

    Args:
        records: EvalRecords.
        scores: Continuous guardrail scores.
        thresholds: Calibrated thresholds keyed by target_fpr.
        target_fpr_values: FPR constraint values.
        category_config: Category assignment config.
        fpr_grid_size: Grid size for ROC/PR curves.
        base_extraction_config: Optional config for extracting categories from other sample
            fields (passed through to ``categorize_records``).

    Returns:
        Mapping from category name to CategoryResult.
    """
    if len(scores) != len(records):
        raise ValueError(f"scores and records must have the same length, got {len(scores)} and {len(records)}")
    if not target_fpr_values:
        raise ValueError("target_fpr_values must be non-empty")
    if set(thresholds.keys()) != set(target_fpr_values):
        raise ValueError(
            f"thresholds keys {sorted(thresholds.keys())} must match target_fpr_values {sorted(target_fpr_values)}"
        )
    category_to_indices = categorize_records(records, category_config, base_extraction_config)
    results: dict[str, correctness_types.CategoryResult] = {}
    for cat_name, indices in category_to_indices.items():
        if not indices:
            continue
        cat_records = [records[idx] for idx in indices]
        cat_scores = scores[np.array(indices)]
        cat_labels = np.array([rec.label for rec in cat_records])
        cat_class_balance = compute_class_balance(cat_records)
        cat_threshold_free = compute_threshold_free_metrics(
            cat_scores,
            cat_labels,
            target_fpr_values,
            fpr_grid_size,
        )
        cat_attempt_metrics: dict[float, correctness_types.ThresholdedMetrics] = {}
        cat_sample_metrics: dict[float, correctness_types.SampleLevelMetrics] = {}
        for target_fpr in target_fpr_values:
            threshold = thresholds[target_fpr]
            cat_attempt_metrics[target_fpr] = compute_thresholded_metrics(
                cat_scores,
                cat_labels,
                threshold,
                target_fpr,
            )
            cat_sample_metrics[target_fpr] = compute_sample_level_metrics(
                cat_records,
                cat_scores,
                threshold,
                target_fpr,
            )
        sample_count = len({rec.sample_id for rec in cat_records})
        results[cat_name] = correctness_types.CategoryResult(
            category=cat_name,
            record_count=len(cat_records),
            sample_count=sample_count,
            class_balance=cat_class_balance,
            threshold_free=cat_threshold_free,
            attempt_metrics=cat_attempt_metrics,
            sample_metrics=cat_sample_metrics,
        )
    return results


def compute_class_balance(
    records: list[correctness_types.EvalRecord],
) -> correctness_types.ClassBalanceStats:
    """Compute descriptive statistics about the correctness label distribution.

    Args:
        records: EvalRecords.

    Returns:
        ClassBalanceStats with overall and per-sample positive rates.
    """
    if not records:
        return correctness_types.ClassBalanceStats(
            overall_positive_rate=0.0,
            per_sample_positive_rates=[],
            num_all_correct_samples=0,
            num_all_incorrect_samples=0,
            code_type_proportions={},
            predict_type_proportions={},
        )
    total_correct = sum(1 for rec in records if rec.label)
    overall_positive_rate = total_correct / len(records)
    # per-sample rates
    sample_groups: dict[str, list[bool]] = collections.defaultdict(list)
    for rec in records:
        sample_groups[rec.sample_id].append(rec.label)
    per_sample_rates: list[float] = []
    num_all_correct = 0
    num_all_incorrect = 0
    for sample_id in sorted(sample_groups.keys()):
        labels = sample_groups[sample_id]
        rate = sum(labels) / len(labels)
        per_sample_rates.append(rate)
        if all(labels):
            num_all_correct += 1
        if not any(labels):
            num_all_incorrect += 1
    # code_type and predict_type proportions
    num_records = len(records)
    code_type_counts: dict[str, int] = collections.defaultdict(int)
    predict_type_counts: dict[str, int] = collections.defaultdict(int)
    compound_example: tuple[str, list[str]] | None = None
    for rec in records:
        code_type_set = _parse_code_type_set(rec.code_type)
        if code_type_set:
            try:
                code_type_set_obj = samples_common.SampleCodeTypeSet(code_type_set)
            except ValueError as exc:
                raise ValueError(
                    f"record '{rec.sample_id}' has forbidden code_type combination '{rec.code_type}' "
                    f"(parsed as {code_type_set}); this indicates invalid data in the LMDB export"
                ) from exc
            code_type_set_str = str(code_type_set_obj)
            code_type_counts[code_type_set_str] += 1  # compound entry (e.g. "bugged_hinted")
            if len(code_type_set) > 1:
                if compound_example is None:
                    compound_example = (
                        rec.code_type,
                        sorted(code_type.value for code_type in code_type_set),
                    )
                for code_type in code_type_set:
                    code_type_counts[code_type.value] += 1  # individual component entries
        else:
            code_type_counts[rec.code_type] += 1  # unrecognized code_type, use raw string
        predict_type = rec.record.get("predict_type")
        if predict_type is not None:
            predict_type_counts[str(predict_type)] += 1
    code_type_proportions = {ct: count / num_records for ct, count in sorted(code_type_counts.items())}
    if compound_example is not None:
        compound_str, component_strs = compound_example
        logger.warning(
            "code_type_proportions are not mutually exclusive: compound code types are counted "
            "both as a compound entry and as individual components. Example: '%s' contributes to "
            "%s and its compound key. Interpret proportions accordingly.",
            compound_str,
            component_strs,
        )
    predict_type_proportions = {pt: count / num_records for pt, count in sorted(predict_type_counts.items())}
    return correctness_types.ClassBalanceStats(
        overall_positive_rate=overall_positive_rate,
        per_sample_positive_rates=per_sample_rates,
        num_all_correct_samples=num_all_correct,
        num_all_incorrect_samples=num_all_incorrect,
        code_type_proportions=code_type_proportions,
        predict_type_proportions=predict_type_proportions,
    )


def format_fpr_key(
    target_fpr: float,
) -> str:
    """Format a float FPR value as a string key (dots replaced by underscores)."""
    return f"fpr_{str(target_fpr).replace('.', '_')}"


def _collect_bootstrap_metrics(
    boot_records: list[correctness_types.EvalRecord],
    boot_scores: npt.NDArray[np.floating[typing.Any]],
    boot_labels: npt.NDArray[np.bool_],
    thresholds: dict[float, float],
    target_fpr_values: list[float],
    metric_samples: dict[str, list[float]],
    sample_keys: list[str] | None = None,
) -> None:
    """Collect attempt-level and sample-level metrics for one bootstrap replicate.

    Shared helper for both problem-clustered and hierarchical bootstrap to avoid logic duplication
    and drift.

    Args:
        boot_records: Resampled EvalRecords.
        boot_scores: Scores for the resampled records.
        boot_labels: Labels for the resampled records.
        thresholds: Calibrated thresholds keyed by target_fpr.
        target_fpr_values: FPR constraint values.
        metric_samples: Accumulator dict (metric name -> list of replicate values).
        sample_keys: Per-record grouping keys for sample-level metrics (required when
            records contain duplicate sample_ids from problem-level resampling).
    """
    if boot_labels.dtype != np.bool_:
        raise ValueError(f"boot_labels must be a boolean array, got dtype={boot_labels.dtype}")
    for target_fpr in target_fpr_values:
        threshold = thresholds[target_fpr]
        fpr_key = format_fpr_key(target_fpr)
        boot_accepted = boot_scores >= threshold
        # attempt-level
        boot_tp = int(np.sum(boot_labels & boot_accepted))
        boot_fn = int(np.sum(boot_labels & ~boot_accepted))
        boot_fp = int(np.sum(~boot_labels & boot_accepted))
        boot_tn = int(np.sum(~boot_labels & ~boot_accepted))
        if boot_tp + boot_fn > 0:
            metric_samples[f"{fpr_key}/tpr"].append(boot_tp / (boot_tp + boot_fn))
            metric_samples[f"{fpr_key}/fnr"].append(boot_fn / (boot_tp + boot_fn))
        if boot_fp + boot_tn > 0:
            metric_samples[f"{fpr_key}/fpr"].append(boot_fp / (boot_fp + boot_tn))
        if boot_tp + boot_fp > 0:
            metric_samples[f"{fpr_key}/precision"].append(boot_tp / (boot_tp + boot_fp))
        if boot_tn + boot_fn > 0:
            metric_samples[f"{fpr_key}/npv"].append(boot_tn / (boot_tn + boot_fn))
        # sample-level
        boot_sample_metrics = compute_sample_level_metrics(
            boot_records,
            boot_scores,
            threshold,
            target_fpr,
            sample_keys=sample_keys,
        )
        metric_samples[f"{fpr_key}/base_pass_rate"].append(boot_sample_metrics.base_pass_rate)
        metric_samples[f"{fpr_key}/guarded_pass_rate"].append(boot_sample_metrics.guarded_pass_rate)
        metric_samples[f"{fpr_key}/unsafe_slip_rate"].append(boot_sample_metrics.unsafe_slip_rate)
        metric_samples[f"{fpr_key}/total_block_rate"].append(boot_sample_metrics.total_block_rate)
        if boot_sample_metrics.best_of_k_success_rate is not None:
            metric_samples[f"{fpr_key}/best_of_k_success_rate"].append(
                boot_sample_metrics.best_of_k_success_rate,
            )
        metric_samples[f"{fpr_key}/cons_pass_rate"].append(boot_sample_metrics.cons_pass_rate)
        if boot_sample_metrics.cons_unsafe_slip_rate is not None:
            metric_samples[f"{fpr_key}/cons_unsafe_slip_rate"].append(
                boot_sample_metrics.cons_unsafe_slip_rate,
            )
        if boot_sample_metrics.cons_justified_reject_rate is not None:
            metric_samples[f"{fpr_key}/cons_justified_reject_rate"].append(
                boot_sample_metrics.cons_justified_reject_rate,
            )


def _build_bootstrap_sample_keys(
    records: list[correctness_types.EvalRecord],
    resampled_problem_indices: npt.NDArray[np.intp],
    problem_ids: list[str],
    problem_to_indices: dict[str, list[int]],
) -> tuple[list[int], list[str]]:
    """Build resampled record indices and unique sample keys for bootstrap.

    When a problem is drawn multiple times via resampling with replacement, its records
    appear multiple times. This function assigns each occurrence a unique draw-specific
    sample key (``{sample_id}__draw{N}``) so that ``compute_sample_level_metrics`` treats
    each draw as an independent sample rather than collapsing duplicates.

    Args:
        records: Original EvalRecords.
        resampled_problem_indices: Array of indices into ``problem_ids`` (with replacement).
        problem_ids: Sorted list of unique problem IDs.
        problem_to_indices: Mapping from problem_id to record indices.

    Returns:
        Tuple of (resampled_record_indices, sample_keys).
    """
    resampled_record_indices: list[int] = []
    sample_keys: list[str] = []
    problem_draw_count: dict[str, int] = collections.defaultdict(int)
    for prob_idx in resampled_problem_indices:
        pid = problem_ids[int(prob_idx)]
        draw_num = problem_draw_count[pid]
        problem_draw_count[pid] += 1
        for rec_idx in problem_to_indices[pid]:
            resampled_record_indices.append(rec_idx)
            sample_keys.append(f"{records[rec_idx].sample_id}__draw{draw_num}")
    return resampled_record_indices, sample_keys


_MAX_AUTO_WORKERS = 16  # cap for auto-resolved worker count to limit memory overhead from pickling


def resolve_num_workers(
    num_workers: int,
    num_replicates: int,
) -> int:
    """Resolve effective worker count from the user-specified value.

    Args:
        num_workers: 0 = auto (up to ``_MAX_AUTO_WORKERS`` CPUs), 1 = sequential, >1 = that many.
        num_replicates: Total bootstrap replicates (workers clamped so none gets zero work).

    Returns:
        Effective worker count (>= 1).
    """
    if num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {num_workers}")
    if num_workers == 0:
        resolved = min(os.cpu_count() or 1, _MAX_AUTO_WORKERS)
    else:
        resolved = num_workers
    return max(1, min(resolved, num_replicates))


def _compute_chunk_sizes(
    num_replicates: int,
    num_workers: int,
) -> list[int]:
    """Split replicates into balanced chunks across workers.

    Args:
        num_replicates: Total number of bootstrap replicates.
        num_workers: Number of worker processes.

    Returns:
        List of chunk sizes (one per worker), summing to ``num_replicates``.
    """
    base, remainder = divmod(num_replicates, num_workers)
    return [base + (1 if worker_idx < remainder else 0) for worker_idx in range(num_workers)]


def _merge_metric_samples(
    chunk_results: list[dict[str, list[float]]],
) -> dict[str, list[float]]:
    """Concatenate metric sample lists across worker chunks.

    Args:
        chunk_results: Per-worker metric sample dicts.

    Returns:
        Merged dict with concatenated sample lists.
    """
    merged: dict[str, list[float]] = collections.defaultdict(list)
    for chunk in chunk_results:
        for metric_name, values in chunk.items():
            merged[metric_name].extend(values)
    return merged


@typing.no_type_check
def _resample_and_compute_metrics(
    *,
    records: list[correctness_types.EvalRecord],
    scores: npt.NDArray[np.floating[typing.Any]],
    problem_ids: list[str],
    problem_to_indices: dict[str, list[int]],
    thresholds: dict[float, float],
    target_fpr_values: list[float],
    rng: np.random.Generator,
    metric_samples: dict[str, list[float]],
) -> None:
    """Perform one bootstrap replicate: resample problems, compute all metrics.

    Shared inner helper used by both clustered and hierarchical bootstrap chunk functions to
    deduplicate the sklearn AUROC/AP + resampling logic.

    Replicates where ``problem_ids`` is empty or resampling yields no record indices are silently
    skipped (nothing is appended to ``metric_samples``). Callers should account for the possibility
    that fewer samples than ``num_replicates`` may be collected.

    Args:
        records: Original EvalRecords for this run/dataset.
        scores: Score array aligned with ``records``.
        problem_ids: Sorted unique problem IDs.
        problem_to_indices: Mapping from problem_id to record indices.
        thresholds: Calibrated thresholds keyed by target_fpr.
        target_fpr_values: FPR constraint values.
        rng: NumPy random generator (caller-owned, mutated in place).
        metric_samples: Accumulator dict to append replicate values into.
    """
    num_problems = len(problem_ids)
    if num_problems == 0:
        return
    resampled_problem_indices = rng.integers(0, num_problems, size=num_problems)
    resampled_record_indices, sample_keys = _build_bootstrap_sample_keys(
        records=records,
        resampled_problem_indices=resampled_problem_indices,
        problem_ids=problem_ids,
        problem_to_indices=problem_to_indices,
    )
    if not resampled_record_indices:
        return
    idx_array = np.array(resampled_record_indices)
    boot_scores = scores[idx_array]
    boot_records = [records[idx] for idx in resampled_record_indices]
    boot_labels = np.array([rec.label for rec in boot_records])
    unique_labels = np.unique(boot_labels)
    if len(unique_labels) >= 2:
        boot_auroc = float(sklearn.metrics.roc_auc_score(boot_labels, boot_scores))
        boot_ap = float(sklearn.metrics.average_precision_score(boot_labels, boot_scores))
        metric_samples["auroc"].append(boot_auroc)
        metric_samples["average_precision"].append(boot_ap)
    _collect_bootstrap_metrics(
        boot_records=boot_records,
        boot_scores=boot_scores,
        boot_labels=boot_labels,
        thresholds=thresholds,
        target_fpr_values=target_fpr_values,
        metric_samples=metric_samples,
        sample_keys=sample_keys,
    )


@typing.no_type_check
def _clustered_bootstrap_chunk(
    *,
    records: list[correctness_types.EvalRecord],
    scores: npt.NDArray[np.floating[typing.Any]],
    thresholds: dict[float, float],
    target_fpr_values: list[float],
    num_chunk_replicates: int,
    seed_sequence: np.random.SeedSequence,
    problem_ids: list[str],
    problem_to_indices: dict[str, list[int]],
) -> dict[str, list[float]]:
    """Run a chunk of clustered bootstrap replicates in a worker process.

    Must be module-level (not a closure) for pickling compatibility with process pools.

    Args:
        records: EvalRecords.
        scores: Score array.
        thresholds: Calibrated thresholds.
        target_fpr_values: FPR constraint values.
        num_chunk_replicates: Number of replicates for this chunk.
        seed_sequence: SeedSequence for independent RNG in this worker.
        problem_ids: Sorted unique problem IDs.
        problem_to_indices: Mapping from problem_id to record indices.

    Returns:
        Metric samples dict for this chunk.
    """
    rng = np.random.default_rng(seed_sequence)
    metric_samples: dict[str, list[float]] = collections.defaultdict(list)
    for _ in range(num_chunk_replicates):
        _resample_and_compute_metrics(
            records=records,
            scores=scores,
            problem_ids=problem_ids,
            problem_to_indices=problem_to_indices,
            thresholds=thresholds,
            target_fpr_values=target_fpr_values,
            rng=rng,
            metric_samples=metric_samples,
        )
    return dict(metric_samples)


@typing.no_type_check
def _hierarchical_bootstrap_chunk(
    *,
    per_run_records: list[list[correctness_types.EvalRecord]],
    per_run_scores: list[npt.NDArray[np.floating[typing.Any]]],
    per_run_thresholds: dict[float, list[float]],
    target_fpr_values: list[float],
    num_chunk_replicates: int,
    seed_sequence: np.random.SeedSequence,
    run_problem_maps: list[dict[str, list[int]]],
    run_problem_id_lists: list[list[str]],
) -> dict[str, list[float]]:
    """Run a chunk of hierarchical bootstrap replicates in a worker process.

    Must be module-level (not a closure) for pickling compatibility with process pools.

    Args:
        per_run_records: List of record lists, one per run.
        per_run_scores: List of score arrays, one per run.
        per_run_thresholds: Thresholds per target_fpr per run.
        target_fpr_values: FPR constraint values.
        num_chunk_replicates: Number of replicates for this chunk.
        seed_sequence: SeedSequence for independent RNG in this worker.
        run_problem_maps: Pre-computed problem-to-indices maps per run.
        run_problem_id_lists: Pre-computed sorted problem ID lists per run.

    Returns:
        Metric samples dict for this chunk.
    """
    rng = np.random.default_rng(seed_sequence)
    num_runs = len(per_run_records)
    metric_samples: dict[str, list[float]] = collections.defaultdict(list)
    for _ in range(num_chunk_replicates):
        run_indices = rng.integers(0, num_runs, size=num_runs)
        per_run_metrics: dict[str, list[float]] = collections.defaultdict(list)
        for run_idx in run_indices:
            run_thresholds = {fpr: per_run_thresholds[fpr][run_idx] for fpr in target_fpr_values}
            _resample_and_compute_metrics(
                records=per_run_records[run_idx],
                scores=per_run_scores[run_idx],
                problem_ids=run_problem_id_lists[run_idx],
                problem_to_indices=run_problem_maps[run_idx],
                thresholds=run_thresholds,
                target_fpr_values=target_fpr_values,
                rng=rng,
                metric_samples=per_run_metrics,
            )
        for metric_name, values in per_run_metrics.items():
            if values:
                metric_samples[metric_name].append(float(np.mean(values)))
    return dict(metric_samples)


@typing.no_type_check  # sklearn type stubs are partially unknown
def compute_clustered_bootstrap_cis(
    records: list[correctness_types.EvalRecord],
    scores: npt.NDArray[np.floating[typing.Any]],
    thresholds: dict[float, float],
    target_fpr_values: list[float],
    num_replicates: int,
    seed: int,
    confidence_level: float,
    num_workers: int = 0,
) -> dict[str, pyine.utils.metrics.confidence.ConfidenceInterval]:
    """Compute problem-clustered bootstrap CIs for all metrics.

    Per replicate: resample problem_ids with replacement, include all records for each resampled
    problem, recompute metrics. Resampled records are assigned unique sample keys to prevent
    duplicate sample_ids from collapsing in sample-level metrics.

    CIs use the percentile method: lower and upper bounds are the ``alpha/2`` and ``1 - alpha/2``
    percentiles of the bootstrap distribution, where ``alpha = 1 - confidence_level``. The point
    estimate is the mean of the bootstrap distribution. Metrics for which fewer than 10 valid
    (non-None) replicate values are collected are omitted from the returned dict (insufficient data
    for reliable CIs).

    CIs are conditional on the fixed calibrated thresholds; thresholds are NOT re-calibrated per
    bootstrap replicate (see correctness README for rationale).

    Args:
        records: EvalRecords.
        scores: Continuous guardrail scores.
        thresholds: Calibrated thresholds keyed by target_fpr (fixed, not resampled).
        target_fpr_values: FPR constraint values.
        num_replicates: Number of bootstrap replicates.
        seed: Random seed for reproducibility.
        confidence_level: Confidence level for CIs (e.g. 0.95 for 95% CIs).
        num_workers: Worker count for parallel execution. 0 = auto (up to 16 CPUs),
            1 = sequential (bit-exact with pre-parallelization behavior), >1 = that many workers.
            Results are deterministic for a fixed (seed, effective num_workers) pair but differ
            across different effective worker counts due to independent RNG streams.

    Returns:
        Mapping from metric name to ConfidenceInterval. Keys follow the naming scheme
        ``"auroc"``, ``"average_precision"``, ``"fpr_0_01/tpr"``, ``"fpr_0_01/precision"``,
        ``"fpr_0_01/base_pass_rate"``, etc.
    """
    if len(scores) != len(records):
        raise ValueError(f"scores and records must have the same length, got {len(scores)} and {len(records)}")
    if len(scores) > 0:
        _assert_finite(scores, "scores")
    if not target_fpr_values:
        raise ValueError("target_fpr_values must be non-empty")
    if set(thresholds.keys()) != set(target_fpr_values):
        raise ValueError(
            f"thresholds keys {sorted(thresholds.keys())} must match target_fpr_values {sorted(target_fpr_values)}"
        )
    if num_replicates <= 0:
        raise ValueError(f"num_replicates must be positive, got {num_replicates}")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")
    # group records by problem_id
    problem_to_indices: dict[str, list[int]] = collections.defaultdict(list)
    for idx, rec in enumerate(records):
        problem_to_indices[rec.problem_id].append(idx)
    problem_ids = sorted(problem_to_indices.keys())
    if len(problem_ids) == 0:
        return {}
    effective = resolve_num_workers(num_workers, num_replicates)
    if effective == 1:
        # sequential path: bit-exact with pre-parallelization behavior
        rng = np.random.default_rng(seed)
        metric_samples: dict[str, list[float]] = collections.defaultdict(list)
        logger.info(f"running clustered bootstrap CI computation ({num_replicates} replicates)")
        for num_done in range(1, num_replicates + 1):
            _resample_and_compute_metrics(
                records=records,
                scores=scores,
                problem_ids=problem_ids,
                problem_to_indices=problem_to_indices,
                thresholds=thresholds,
                target_fpr_values=target_fpr_values,
                rng=rng,
                metric_samples=metric_samples,
            )
            if pyine.evals.utils.should_log_percent_progress(num_done, num_replicates):
                progress_percent = (100.0 * num_done) / num_replicates
                logger.info(f"clustered bootstrap progress: {num_done}/{num_replicates} ({progress_percent:.1f}%)")
        return _build_cis_from_samples(metric_samples, confidence_level)
    # parallel path
    logger.info(f"running clustered bootstrap CI computation ({num_replicates} replicates across {effective} workers)")
    child_seeds = np.random.SeedSequence(seed).spawn(effective)
    chunk_sizes = _compute_chunk_sizes(num_replicates, effective)
    # convert defaultdict to regular dict for pickling robustness
    problem_to_indices_dict = dict(problem_to_indices)
    callables = [
        functools.partial(
            _clustered_bootstrap_chunk,
            records=records,
            scores=scores,
            thresholds=thresholds,
            target_fpr_values=target_fpr_values,
            num_chunk_replicates=chunk_size,
            seed_sequence=child_seed,
            problem_ids=problem_ids,
            problem_to_indices=problem_to_indices_dict,
        )
        for chunk_size, child_seed in zip(chunk_sizes, child_seeds, strict=True)
    ]
    results, errors = pyine.utils.concurrency.run_in_parallel(
        callables,
        use_processes=True,
        max_workers=effective,
    )
    for error in errors:
        if error is not None:
            raise error
    chunk_results = typing.cast("list[dict[str, list[float]]]", results)
    merged = _merge_metric_samples(chunk_results)
    logger.info("clustered bootstrap CI computation complete")
    return _build_cis_from_samples(merged, confidence_level)


def _build_cis_from_samples(
    metric_samples: dict[str, list[float]],
    confidence_level: float,
) -> dict[str, pyine.utils.metrics.confidence.ConfidenceInterval]:
    """Build percentile CIs from collected bootstrap metric samples.

    Uses the percentile method: lower = percentile(alpha/2), upper = percentile(1-alpha/2),
    point_estimate = mean of the bootstrap distribution. Metrics with fewer than 10 valid
    bootstrap samples are omitted (insufficient data for reliable percentile CIs).

    Args:
        metric_samples: Mapping from metric name to list of bootstrap replicate values.
            None-valued replicates should have been excluded before collection.
        confidence_level: Confidence level for CIs (e.g. 0.95 for 95% CIs).

    Returns:
        Mapping from metric name to ConfidenceInterval. Metrics with <10 samples omitted.
    """
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")
    alpha = 1.0 - confidence_level
    cis: dict[str, pyine.utils.metrics.confidence.ConfidenceInterval] = {}
    for metric_name, samples in metric_samples.items():
        if len(samples) < 10:
            continue
        samples_arr = np.array(samples)
        lower = float(np.percentile(samples_arr, 100 * alpha / 2))
        upper = float(np.percentile(samples_arr, 100 * (1 - alpha / 2)))
        point = float(np.mean(samples_arr))
        cis[metric_name] = pyine.utils.metrics.confidence.ConfidenceInterval(
            point_estimate=point,
            lower_bound=lower,
            upper_bound=upper,
        )
    return cis


@typing.no_type_check  # sklearn type stubs are partially unknown
def compute_hierarchical_bootstrap_cis(
    per_run_records: list[list[correctness_types.EvalRecord]],
    per_run_scores: list[npt.NDArray[np.floating[typing.Any]]],
    per_run_thresholds: dict[float, list[float]],
    target_fpr_values: list[float],
    num_replicates: int,
    seed: int,
    confidence_level: float,
    num_workers: int = 0,
) -> dict[str, pyine.utils.metrics.confidence.ConfidenceInterval]:
    """Compute hierarchical bootstrap CIs across R independent guardrail runs.

    Per replicate: resample run indices with replacement, then within each resampled run, resample
    problem_ids with replacement (including all their samples and attempts), compute each metric
    per resampled run, then average across the resampled runs. Resampled records are assigned
    unique sample keys to prevent duplicate sample_ids from collapsing in sample-level metrics.

    This two-level resampling accounts for both run variance (different guardrail training seeds
    and threshold calibrations) and test set sampling variance. CIs use the percentile method with
    point_estimate = mean of the bootstrap distribution. Metrics with fewer than 10 valid
    replicates are omitted.

    Args:
        per_run_records: List of record lists, one per run.
        per_run_scores: List of score arrays, one per run.
        per_run_thresholds: Thresholds per target_fpr per run (each run uses its own
            calibrated threshold).
        target_fpr_values: FPR constraint values.
        num_replicates: Number of bootstrap replicates.
        seed: Random seed for reproducibility.
        confidence_level: Confidence level for CIs (e.g. 0.95 for 95% CIs).
        num_workers: Worker count for parallel execution. 0 = auto (up to 16 CPUs),
            1 = sequential (bit-exact with pre-parallelization behavior), >1 = that many workers.
            Results are deterministic for a fixed (seed, effective num_workers) pair but differ
            across different effective worker counts due to independent RNG streams.

    Returns:
        Mapping from metric name to ConfidenceInterval.
    """
    if len(per_run_records) != len(per_run_scores):
        raise ValueError(
            f"per_run_records and per_run_scores must have the same length, "
            f"got {len(per_run_records)} and {len(per_run_scores)}"
        )
    for run_idx, (run_records, run_scores) in enumerate(zip(per_run_records, per_run_scores, strict=True)):
        if len(run_records) != len(run_scores):
            raise ValueError(f"run {run_idx}: records length ({len(run_records)}) != scores length ({len(run_scores)})")
        if len(run_scores) > 0:
            _assert_finite(run_scores, f"per_run_scores[{run_idx}]")
    if not target_fpr_values:
        raise ValueError("target_fpr_values must be non-empty")
    for fpr_val in target_fpr_values:
        if fpr_val not in per_run_thresholds:
            raise ValueError(f"per_run_thresholds missing key for target_fpr={fpr_val}")
        if len(per_run_thresholds[fpr_val]) != len(per_run_records):
            raise ValueError(
                f"per_run_thresholds[{fpr_val}] has {len(per_run_thresholds[fpr_val])} entries "
                f"but expected {len(per_run_records)} (one per run)"
            )
    if num_replicates <= 0:
        raise ValueError(f"num_replicates must be positive, got {num_replicates}")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")
    num_runs = len(per_run_records)
    if num_runs == 0:
        return {}
    # precompute problem-to-indices for each run
    run_problem_maps: list[dict[str, list[int]]] = []
    run_problem_id_lists: list[list[str]] = []
    for run_records in per_run_records:
        prob_to_idx: dict[str, list[int]] = collections.defaultdict(list)
        for idx, rec in enumerate(run_records):
            prob_to_idx[rec.problem_id].append(idx)
        run_problem_maps.append(dict(prob_to_idx))
        run_problem_id_lists.append(sorted(prob_to_idx.keys()))
    effective = resolve_num_workers(num_workers, num_replicates)
    if effective == 1:
        # sequential path: bit-exact with pre-parallelization behavior
        rng = np.random.default_rng(seed)
        metric_samples: dict[str, list[float]] = collections.defaultdict(list)
        logger.info(f"running hierarchical bootstrap CI computation ({num_replicates} replicates)")
        for num_done in range(1, num_replicates + 1):
            run_indices = rng.integers(0, num_runs, size=num_runs)
            per_run_metrics: dict[str, list[float]] = collections.defaultdict(list)
            for run_idx in run_indices:
                _resample_and_compute_metrics(
                    records=per_run_records[run_idx],
                    scores=per_run_scores[run_idx],
                    problem_ids=run_problem_id_lists[run_idx],
                    problem_to_indices=run_problem_maps[run_idx],
                    thresholds={fpr: per_run_thresholds[fpr][run_idx] for fpr in target_fpr_values},
                    target_fpr_values=target_fpr_values,
                    rng=rng,
                    metric_samples=per_run_metrics,
                )
            for metric_name, values in per_run_metrics.items():
                if values:
                    metric_samples[metric_name].append(float(np.mean(values)))
            if pyine.evals.utils.should_log_percent_progress(num_done, num_replicates):
                progress_percent = (100.0 * num_done) / num_replicates
                logger.info(f"hierarchical bootstrap progress: {num_done}/{num_replicates} ({progress_percent:.1f}%)")
        return _build_cis_from_samples(metric_samples, confidence_level)
    # parallel path
    logger.info(
        f"running hierarchical bootstrap CI computation ({num_replicates} replicates across {effective} workers)"
    )
    child_seeds = np.random.SeedSequence(seed).spawn(effective)
    chunk_sizes = _compute_chunk_sizes(num_replicates, effective)
    callables = [
        functools.partial(
            _hierarchical_bootstrap_chunk,
            per_run_records=per_run_records,
            per_run_scores=per_run_scores,
            per_run_thresholds=per_run_thresholds,
            target_fpr_values=target_fpr_values,
            num_chunk_replicates=chunk_size,
            seed_sequence=child_seed,
            run_problem_maps=run_problem_maps,
            run_problem_id_lists=run_problem_id_lists,
        )
        for chunk_size, child_seed in zip(chunk_sizes, child_seeds, strict=True)
    ]
    results, errors = pyine.utils.concurrency.run_in_parallel(
        callables,
        use_processes=True,
        max_workers=effective,
    )
    for error in errors:
        if error is not None:
            raise error
    chunk_results = typing.cast("list[dict[str, list[float]]]", results)
    merged = _merge_metric_samples(chunk_results)
    logger.info("hierarchical bootstrap CI computation complete")
    return _build_cis_from_samples(merged, confidence_level)


@typing.no_type_check  # sklearn/scipy type stubs are partially unknown
def compute_difficulty_stats(
    records: list[correctness_types.EvalRecord],
    scores: npt.NDArray[np.floating[typing.Any]],
    labels: npt.NDArray[np.bool_],
    thresholds: dict[float, float],
    target_fpr_values: list[float],
) -> correctness_types.DifficultyStats | None:
    """Compute guardrail performance conditioned on sample difficulty.

    Returns None if any record has ``difficulty_score is None`` (all-or-nothing: either
    all records have difficulty scores or the analysis is skipped entirely).

    When available, records are bucketed into terciles (easy/medium/hard) by difficulty
    score, and per-bucket AUROC and TPR (at each calibrated threshold) are computed.
    A Spearman rank correlation between per-sample mean difficulty and per-sample
    guardrail accuracy is also reported.

    Args:
        records: EvalRecords.
        scores: Continuous guardrail scores (must contain only finite values).
        labels: Boolean correctness labels.
        thresholds: Calibrated thresholds keyed by target_fpr.
        target_fpr_values: FPR constraint values.

    Returns:
        DifficultyStats or None when difficulty scores are unavailable.
    """
    if len(scores) != len(labels):
        raise ValueError(f"scores and labels must have the same length, got {len(scores)} and {len(labels)}")
    if len(records) != len(scores):
        raise ValueError(f"records and scores must have the same length, got {len(records)} and {len(scores)}")
    if len(records) == 0:
        raise ValueError("records must be non-empty")
    if labels.dtype != np.bool_:
        raise ValueError(f"labels must be a boolean array, got dtype={labels.dtype}")
    if not target_fpr_values:
        raise ValueError("target_fpr_values must be non-empty")
    if set(thresholds.keys()) != set(target_fpr_values):
        raise ValueError(
            f"thresholds keys {sorted(thresholds.keys())} must match target_fpr_values {sorted(target_fpr_values)}"
        )
    _assert_finite(scores, "scores")
    if any(rec.difficulty_score is None for rec in records):
        return None
    difficulty_scores = np.array([rec.difficulty_score for rec in records], dtype=np.float64)
    _assert_finite(difficulty_scores, "difficulty_scores")
    # compute tercile boundaries
    p33 = float(np.percentile(difficulty_scores, 33.33))
    p67 = float(np.percentile(difficulty_scores, 66.67))
    # assign buckets
    bucket_names = ["easy", "medium", "hard"]
    record_buckets = np.where(
        difficulty_scores <= p33,
        0,
        np.where(difficulty_scores <= p67, 1, 2),
    )
    per_bucket_auroc: dict[str, float | None] = {}
    per_bucket_tpr: dict[str, dict[float, float]] = {}
    per_bucket_sample_count: dict[str, int] = {}
    for bucket_idx, bucket_name in enumerate(bucket_names):
        mask = record_buckets == bucket_idx
        bucket_labels = labels[mask]
        bucket_scores = scores[mask]
        bucket_records = [records[idx] for idx in range(len(records)) if mask[idx]]
        per_bucket_sample_count[bucket_name] = len({rec.sample_id for rec in bucket_records})
        unique_bucket_labels = np.unique(bucket_labels)
        if len(unique_bucket_labels) >= 2:
            per_bucket_auroc[bucket_name] = float(sklearn.metrics.roc_auc_score(bucket_labels, bucket_scores))
        else:
            per_bucket_auroc[bucket_name] = None
        per_bucket_tpr[bucket_name] = {}
        for target_fpr in target_fpr_values:
            threshold = thresholds[target_fpr]
            bucket_accepted = bucket_scores >= threshold
            bucket_tp = int(np.sum(bucket_labels & bucket_accepted))
            bucket_fn = int(np.sum(bucket_labels & ~bucket_accepted))
            if bucket_tp + bucket_fn > 0:
                per_bucket_tpr[bucket_name][target_fpr] = bucket_tp / (bucket_tp + bucket_fn)
    # spearman correlation: per-sample mean difficulty vs per-sample guardrail accuracy
    # uses the smallest (most conservative) calibrated threshold for accept/reject decisions
    correlation_threshold = thresholds[min(thresholds)]
    sample_to_difficulty: dict[str, list[float]] = collections.defaultdict(list)
    sample_to_accuracy: dict[str, list[bool]] = collections.defaultdict(list)
    for idx, rec in enumerate(records):
        if rec.difficulty_score is None:
            raise ValueError(f"record {rec.sample_id} has difficulty_score=None after pre-check passed")
        sample_to_difficulty[rec.sample_id].append(rec.difficulty_score)
        predicted_correct = scores[idx] >= correlation_threshold
        sample_to_accuracy[rec.sample_id].append(bool(predicted_correct) == rec.label)
    sample_ids_sorted = sorted(sample_to_difficulty.keys())
    mean_difficulties = [float(np.mean(sample_to_difficulty[sid])) for sid in sample_ids_sorted]
    mean_accuracies = [float(np.mean(sample_to_accuracy[sid])) for sid in sample_ids_sorted]
    correlation: float | None
    if len(sample_ids_sorted) >= 3:
        difficulty_arr = np.array(mean_difficulties, dtype=np.float64)
        accuracy_arr = np.array(mean_accuracies, dtype=np.float64)
        if _is_constant(difficulty_arr) or _is_constant(accuracy_arr):
            correlation = None
        else:
            corr_result = scipy.stats.spearmanr(difficulty_arr, accuracy_arr)
            correlation = None if np.isnan(corr_result.statistic) else float(corr_result.statistic)
    else:
        correlation = None
    return correctness_types.DifficultyStats(
        bucket_boundaries=(p33, p67),
        per_bucket_auroc=per_bucket_auroc,
        per_bucket_tpr=per_bucket_tpr,
        per_bucket_sample_count=per_bucket_sample_count,
        difficulty_accuracy_rank_correlation=correlation,
    )


@typing.no_type_check  # scipy type stubs are partially unknown
def compute_verification_cost_stats(
    verification_costs: list[float] | None,
    labels: npt.NDArray[np.bool_],
    accepted: npt.NDArray[np.bool_],
    target_fpr: float,
    cost_unit: str | None = None,
    records: list[correctness_types.EvalRecord] | None = None,
) -> correctness_types.VerificationCostStats | None:
    """Compute verification cost summary statistics.

    Returns None if ``verification_costs`` is None (scorer did not report costs).
    Costs must be non-negative and contain only finite values.

    When records with difficulty scores are provided, also computes a Spearman rank
    correlation between per-sample mean cost and per-sample mean difficulty.

    Args:
        verification_costs: Per-record costs from ScoringResult, or None.
        labels: Boolean correctness labels.
        accepted: Boolean accepted mask at the given threshold.
        target_fpr: The FPR constraint value.
        cost_unit: Unit label for verification costs (from the guardrail scorer),
            e.g. ``"tokens"`` or ``"FLOPs"``. Carried through for plot axis labels.
        records: EvalRecords for sample-level grouping and difficulty correlation.
            When provided and all records have difficulty scores, enables
            ``cost_difficulty_rank_correlation``.

    Returns:
        VerificationCostStats or None when costs are unavailable.
    """
    if len(labels) != len(accepted):
        raise ValueError(f"labels and accepted must have the same length, got {len(labels)} and {len(accepted)}")
    if labels.dtype != np.bool_:
        raise ValueError(f"labels must be a boolean array, got dtype={labels.dtype}")
    if accepted.dtype != np.bool_:
        raise ValueError(f"accepted must be a boolean array, got dtype={accepted.dtype}")
    if not (0.0 < target_fpr < 1.0):
        raise ValueError(f"target_fpr must be in (0, 1), got {target_fpr}")
    if verification_costs is None and cost_unit is not None:
        raise ValueError(
            f"cost_unit is '{cost_unit}' but verification_costs is None; "
            f"scorer must return costs when it declares a cost unit"
        )
    if verification_costs is not None and cost_unit is None:
        raise ValueError(
            "verification_costs provided but cost_unit is None; scorer must declare a cost unit when it returns costs"
        )
    if verification_costs is None:
        return None
    if len(verification_costs) != len(labels):
        raise ValueError(
            f"verification_costs and labels must have the same length, got {len(verification_costs)} and {len(labels)}"
        )
    if records is not None and len(records) != len(labels):
        raise ValueError(f"records and labels must have the same length, got {len(records)} and {len(labels)}")
    if any(cost < 0 for cost in verification_costs):
        raise ValueError("verification_costs must be non-negative")
    costs = np.array(verification_costs, dtype=np.float64)
    _assert_finite(costs, "verification_costs")
    total_cost = float(np.sum(costs))
    mean_cost = float(np.mean(costs))
    median_cost = float(np.median(costs))
    std_cost = float(np.std(costs))
    # cost per correct acceptance (TP)
    tp_mask = labels & accepted
    cost_per_tp: float | None
    if np.any(tp_mask):
        cost_per_tp = float(np.mean(costs[tp_mask]))
    else:
        cost_per_tp = None
    # cost per incorrect block (TN)
    tn_mask = ~labels & ~accepted
    cost_per_tn: float | None
    if np.any(tn_mask):
        cost_per_tn = float(np.mean(costs[tn_mask]))
    else:
        cost_per_tn = None
    # rank correlation: cost vs guardrail classification accuracy (per-record)
    correct_classification = (accepted & labels) | (~accepted & ~labels)
    cost_accuracy_corr: float | None
    if len(costs) >= 3:
        classification = correct_classification.astype(np.float64)
        if _is_constant(costs) or _is_constant(classification):
            cost_accuracy_corr = None
        else:
            corr_result = scipy.stats.spearmanr(costs, classification)
            cost_accuracy_corr = None if np.isnan(corr_result.statistic) else float(corr_result.statistic)
    else:
        cost_accuracy_corr = None
    # rank correlation: cost vs difficulty (per-sample)
    cost_difficulty_corr: float | None = None
    if records is not None and all(rec.difficulty_score is not None for rec in records):
        sample_to_costs: dict[str, list[float]] = collections.defaultdict(list)
        sample_to_difficulty: dict[str, list[float]] = collections.defaultdict(list)
        for idx, rec in enumerate(records):
            sample_to_costs[rec.sample_id].append(float(costs[idx]))
            sample_to_difficulty[rec.sample_id].append(rec.difficulty_score)  # type: ignore[arg-type]
        sample_ids_sorted = sorted(sample_to_costs.keys())
        if len(sample_ids_sorted) >= 3:
            mean_costs = [float(np.mean(sample_to_costs[sid])) for sid in sample_ids_sorted]
            mean_difficulties = [float(np.mean(sample_to_difficulty[sid])) for sid in sample_ids_sorted]
            cost_arr = np.array(mean_costs, dtype=np.float64)
            difficulty_arr = np.array(mean_difficulties, dtype=np.float64)
            if _is_constant(cost_arr) or _is_constant(difficulty_arr):
                cost_difficulty_corr = None
            else:
                corr_result = scipy.stats.spearmanr(cost_arr, difficulty_arr)
                cost_difficulty_corr = None if np.isnan(corr_result.statistic) else float(corr_result.statistic)
    return correctness_types.VerificationCostStats(
        target_fpr=target_fpr,
        cost_unit=cost_unit,
        total_cost=total_cost,
        mean_cost_per_record=mean_cost,
        median_cost_per_record=median_cost,
        std_cost_per_record=std_cost,
        cost_per_correct_acceptance=cost_per_tp,
        cost_per_incorrect_block=cost_per_tn,
        cost_accuracy_rank_correlation=cost_accuracy_corr,
        cost_difficulty_rank_correlation=cost_difficulty_corr,
    )
