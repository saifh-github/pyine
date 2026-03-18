"""Utilities for fetching and visualizing correctness evaluation results from W&B.

This module provides functions to:
- Discover guardrail type names from W&B run summaries;
- Extract AUROC, TPR, sample-level, and category-wise metrics from run summaries;
- Plot ROC/PR curves, metric comparisons, operating point analysis, and breakdowns.
"""
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportArgumentType=false

from __future__ import annotations

import logging
import pathlib  # noqa: TC003
import re
import typing

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydantic

import pyine.evals.analysis_common
import pyine.evals.correctness
import pyine.evals.correctness.metrics as correctness_metrics
import pyine.evals.correctness.types as correctness_types
import pyine.utils.wandb_utils

if typing.TYPE_CHECKING:
    import wandb.apis.public
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

MetricWithCI = pyine.evals.analysis_common.MetricWithCI


def _fetch_detailed_metrics_df(
    run: wandb.apis.public.Run,
    prefix: str,
) -> pd.DataFrame | None:
    """Fetch the detailed_metrics wandb Table for a correctness eval."""
    table_key = f"{prefix.rstrip('/')}/detailed_metrics"
    return pyine.utils.wandb_utils.fetch_table(run, table_key)


def _fetch_category_metrics_df(
    run: wandb.apis.public.Run,
    prefix: str,
) -> pd.DataFrame | None:
    """Fetch the category_metrics wandb Table for a correctness eval."""
    table_key = f"{prefix.rstrip('/')}/category_metrics"
    return pyine.utils.wandb_utils.fetch_table(run, table_key)


def _metric_with_ci_from_row(
    row: dict[str, typing.Any],
) -> MetricWithCI:
    """Build a MetricWithCI from a table row dict using column name constants."""
    return MetricWithCI(
        value=row.get(correctness_types.COL_MEAN),
        ci_lower=row.get(correctness_types.COL_BOOTSTRAP_CI_LOWER),
        ci_upper=row.get(correctness_types.COL_BOOTSTRAP_CI_UPPER),
    )


def _get_summary_string_keys(summary: typing.Any) -> list[str]:
    """Return a snapshot of string-typed keys from a W&B Summary object.

    The W&B ``Summary.__iter__`` can yield non-string keys (e.g. integers from internal
    tracking), and its ``__getitem__`` raises ``KeyError`` for them.  Iterating via
    ``.keys()`` and filtering to strings avoids the crash.
    """
    return [key for key in summary.keys() if isinstance(key, str)]  # noqa: SIM118


# ---- Data Models ----


class CorrectnessRunMetrics(pyine.evals.analysis_common.BaseRunInfo):
    """Correctness metrics from a single W&B run."""

    guardrail_type_name: str | None = None
    """Guardrail type name, or None if single-type eval."""
    auroc: MetricWithCI = MetricWithCI()
    """Area under the ROC curve."""
    average_precision: MetricWithCI = MetricWithCI()
    """Average precision (area under the PR curve)."""
    tpr_at_fpr: dict[float, MetricWithCI] = pydantic.Field(default_factory=dict)
    """TPR at specified FPR thresholds."""
    attempt_metrics: dict[float, dict[str, MetricWithCI]] = pydantic.Field(default_factory=dict)
    """{target_fpr: {"tpr": ..., "fpr": ..., ...}}"""
    sample_metrics: dict[float, dict[str, MetricWithCI]] = pydantic.Field(default_factory=dict)
    """{target_fpr: {"guarded_pass_rate": ..., "unsafe_slip_rate": ..., ...}}"""
    overall_positive_rate: float | None = None
    """Class balance: overall positive rate."""
    sample_count: int | None = None
    """Number of unique samples evaluated."""
    record_count: int | None = None
    """Total number of records evaluated."""


class CorrectnessCategoryMetrics(pydantic.BaseModel):
    """Metrics for a single category from a W&B run."""

    model_config = pydantic.ConfigDict(frozen=True)
    """Pydantic model configuration (immutable)."""

    category: str
    """Category name (e.g. 'regular', 'biasing/hinted')."""
    record_count: int = 0
    """Number of records in this category."""
    sample_count: int = 0
    """Number of unique samples in this category."""
    auroc: MetricWithCI = MetricWithCI()
    """AUROC for this category."""
    attempt_metrics: dict[float, dict[str, MetricWithCI]] = pydantic.Field(default_factory=dict)
    """Per-FPR attempt-level metrics."""
    sample_metrics: dict[float, dict[str, MetricWithCI]] = pydantic.Field(default_factory=dict)
    """Per-FPR sample-level metrics."""


class CorrectnessRunSummary(pydantic.BaseModel):
    """Complete summary of a correctness evaluation run from W&B."""

    model_config = pydantic.ConfigDict(frozen=True)
    """Pydantic model configuration (immutable)."""

    run_info: CorrectnessRunMetrics
    """Run-level metrics and metadata."""
    category_metrics: list[CorrectnessCategoryMetrics]
    """Per-category metric breakdowns."""


# ---- W&B Extraction Functions ----

# known structural key prefixes (fpr_, tpr_at_fpr_) checked via regex startswith
_STRUCTURAL_PREFIX_RE = re.compile(r"^(?:fpr_\d|tpr_at_fpr_\d)")
"""Regex matching structural prefix-style tokens that are NOT guardrail type names."""

# known structural key tokens that must match exactly -- sourced from the shared constants
# in correctness_types to stay in sync with _impl.py input validation
_STRUCTURAL_EXACT_TOKENS: frozenset[str] = correctness_types.RESERVED_TYPE_NAME_EXACT


def detect_guardrail_type_names(
    run: wandb.apis.public.Run,
    subset_name: str,
    *,
    strict: bool = False,
) -> list[str] | None:
    """Discover guardrail type names from a W&B run.

    Primary strategy: check for explicit metadata key
    ``benchmark/{subset}/_guardrail_type_names`` (logged by ``evaluate_guardrail_types``).

    Fallback (for runs predating the metadata key): scan summary keys, collect candidate tokens
    that look like type names, and validate against multiple anchors.

    Args:
        run: The W&B Run object.
        subset_name: Evaluation subset name (e.g. ``"guardrail_test"``).
        strict: When True, raise ValueError if heuristic detection finds candidates but none meet
            validation; when False (default), warn and return None.

    Returns:
        Sorted list of type names, or None if the run is a single-type eval (no type namespace detected).

    Raises:
        ValueError: If ``strict=True`` and heuristic detection produces candidates but none meet
            validation.
    """
    summary = run.summary
    prefix = f"benchmark/{subset_name}/"
    summary_keys = _get_summary_string_keys(summary)
    # primary strategy: explicit metadata
    metadata_key = f"{prefix}_guardrail_type_names"
    if metadata_key in summary_keys:
        type_names = summary[metadata_key]
        if isinstance(type_names, list) and type_names:
            return sorted(type_names)
    # fallback: heuristic scan
    anchor_suffixes = ["auroc/mean", "average_precision/mean", "sample_count", "record_count"]
    candidates: set[str] = set()
    for key in summary_keys:
        if not key.startswith(prefix):
            continue
        relative = key[len(prefix) :]
        parts = relative.split("/")
        if len(parts) < 2:
            continue
        first_token = parts[0]
        if first_token in _STRUCTURAL_EXACT_TOKENS or _STRUCTURAL_PREFIX_RE.match(first_token):
            continue
        candidates.add(first_token)
    if not candidates:
        return None  # single-type eval, no type namespace
    # validate candidates: require at least 2 anchors
    validated: list[str] = []
    for candidate in candidates:
        anchor_count = sum(1 for suffix in anchor_suffixes if f"{prefix}{candidate}/{suffix}" in summary)
        if anchor_count >= 2:
            validated.append(candidate)
    if not validated:
        logger.warning(
            "heuristic type detection found candidates %s but none met the 2-anchor threshold; "
            "check W&B key structure for run %s",
            sorted(candidates),
            run.id,
        )
        if strict:
            raise ValueError(
                f"guardrail type detection found candidates {sorted(candidates)} but none met "
                f"validation (need >=2 of {anchor_suffixes}); check W&B key structure"
            )
        return None
    logger.warning(
        "falling back to heuristic type detection for run %s (no _guardrail_type_names metadata); detected types: %s",
        run.id,
        sorted(validated),
    )
    return sorted(validated)


def _parse_fpr_capture(
    raw: str,
) -> float:
    """Parse a captured FPR string (either ``0_01`` or ``1e-05``) back to float."""
    if "_" in raw:
        return float(raw.replace("_", ".", 1))
    return float(raw)  # pure scientific notation like 1e-05


# matches both decimal (0_01, 0_001) and scientific (1e-05) FPR key fragments
_FPR_FRAGMENT_RE = r"(\d+_\d+(?:e[+-]?\d+)?|\d+e[+-]?\d+)"


_ATTEMPT_METRIC_NAMES = ["tpr", "fpr", "fnr", "precision", "npv"]
_SAMPLE_METRIC_NAMES = [
    "base_pass_rate",
    "guarded_pass_rate",
    "unsafe_slip_rate",
    "total_block_rate",
    "best_of_k_success_rate",
    "cons_pass_rate",
    "cons_unsafe_slip_rate",
    "cons_justified_reject_rate",
]


def _resolve_type_and_prefix(
    run: wandb.apis.public.Run,
    subset_name: str,
    type_name: str | None,
) -> tuple[str | None, str]:
    """Resolve the guardrail type name and W&B key prefix for a run.

    Validates that the requested type_name exists, and that multi-type runs are not queried without
    specifying a type. Auto-selects the single type when exactly one exists and type_name is None.

    Args:
        run: The W&B Run object.
        subset_name: Evaluation subset name.
        type_name: Requested type name, or None.

    Returns:
        Tuple of (resolved_type_name, prefix) where resolved_type_name may be auto-selected.

    Raises:
        ValueError: If type_name is specified but not found, or if multiple types exist
            and type_name is None.
    """
    detected = detect_guardrail_type_names(run, subset_name)
    if type_name is not None:
        if detected is not None and type_name not in detected:
            raise ValueError(
                f"requested guardrail type {type_name!r} not found in run {run.id}; available types: {detected}"
            )
        if detected is None:
            raise ValueError(
                f"requested guardrail type {type_name!r} but run {run.id} appears to be a "
                "single-type eval (no type namespace detected)"
            )
    elif detected is not None:
        if len(detected) > 1:
            raise ValueError(f"run {run.id} has multiple guardrail types {detected}; pass type_name to select one")
        type_name = detected[0]  # auto-select single type
    if type_name is not None:
        prefix = f"benchmark/{subset_name}/{type_name}/"
    else:
        prefix = f"benchmark/{subset_name}/"
    return type_name, prefix


def extract_correctness_metrics(
    run: wandb.apis.public.Run,
    subset_name: str,
    type_name: str | None = None,
    *,
    _resolved: tuple[str | None, str] | None = None,
) -> CorrectnessRunMetrics:
    """Extract correctness metrics from a W&B run.

    Reads compact summary keys for top-level values (auroc, AP, counts, balance) and
    fetches the detailed_metrics wandb Table for per-FPR metrics.

    Args:
        run: The W&B Run object.
        subset_name: Name of the evaluation subset.
        type_name: Optional guardrail type name for multi-type runs.
        _resolved: Pre-resolved (type_name, prefix) tuple. Internal use only.

    Returns:
        CorrectnessRunMetrics with extracted metric values.

    Raises:
        ValueError: If the detailed_metrics table is missing or has a schema mismatch,
            or if type_name constraints are violated.
    """
    if _resolved is not None:
        type_name, prefix = _resolved
    else:
        type_name, prefix = _resolve_type_and_prefix(run, subset_name, type_name)
    summary = run.summary

    # auroc and AP from compact summary (with CIs)
    def _summary_metric_with_ci(base: str) -> MetricWithCI:
        value = summary.get(f"{prefix}{base}/mean")
        ci_lower = summary.get(f"{prefix}{base}/bootstrap_ci_lower")
        ci_upper = summary.get(f"{prefix}{base}/bootstrap_ci_upper")
        return MetricWithCI(
            value=float(value) if value is not None else None,
            ci_lower=float(ci_lower) if ci_lower is not None else None,
            ci_upper=float(ci_upper) if ci_upper is not None else None,
        )

    auroc = _summary_metric_with_ci("auroc")
    average_precision = _summary_metric_with_ci("average_precision")
    # counts and balance from compact summary
    overall_positive_rate_raw = summary.get(f"{prefix}class_balance/overall_positive_rate")
    sample_count_raw = summary.get(f"{prefix}sample_count")
    record_count_raw = summary.get(f"{prefix}record_count")
    # fetch detailed metrics table
    df = _fetch_detailed_metrics_df(run, prefix)
    if df is None:
        raise ValueError(
            f"detailed_metrics table not found for run {run.id}; this run may predate the table-based logging migration"
        )
    expected_cols = set(correctness_types.DETAILED_METRICS_COLUMNS)
    actual_cols = set(df.columns)
    if not expected_cols.issubset(actual_cols):
        missing = expected_cols - actual_cols
        raise ValueError(f"detailed_metrics table schema mismatch for run {run.id}: missing columns {sorted(missing)}")
    indexed = df.set_index(correctness_types.COL_METRIC_NAME)
    # tpr_at_fpr from table
    tpr_at_fpr: dict[float, MetricWithCI] = {}
    for metric_name in indexed.index:
        if str(metric_name).startswith("tpr_at_fpr_"):
            fpr_str = str(metric_name)[len("tpr_at_fpr_") :]
            fpr_val = _parse_fpr_capture(fpr_str)
            tpr_at_fpr[fpr_val] = _metric_with_ci_from_row(indexed.loc[metric_name].to_dict())
    # detect FPR values from table metric names
    fpr_prefix_re = re.compile(r"^fpr_" + _FPR_FRAGMENT_RE + r"/")
    fpr_values: set[float] = set()
    for metric_name in indexed.index:
        match = fpr_prefix_re.match(str(metric_name))
        if match:
            fpr_values.add(_parse_fpr_capture(match.group(1)))
    sorted_fpr = sorted(fpr_values)
    # build attempt and sample metrics from table (whitelist-driven)
    attempt_metrics: dict[float, dict[str, MetricWithCI]] = {}
    sample_metrics: dict[float, dict[str, MetricWithCI]] = {}
    for fpr_val in sorted_fpr:
        fpr_key = correctness_metrics.format_fpr_key(fpr_val)
        attempt_dict: dict[str, MetricWithCI] = {}
        for metric_name in _ATTEMPT_METRIC_NAMES:
            full_key = f"{fpr_key}/{metric_name}"
            if full_key in indexed.index:
                attempt_dict[metric_name] = _metric_with_ci_from_row(indexed.loc[full_key].to_dict())
            else:
                attempt_dict[metric_name] = MetricWithCI()
        attempt_metrics[fpr_val] = attempt_dict
        sample_dict: dict[str, MetricWithCI] = {}
        for metric_name in _SAMPLE_METRIC_NAMES:
            full_key = f"{fpr_key}/{metric_name}"
            if full_key in indexed.index:
                sample_dict[metric_name] = _metric_with_ci_from_row(indexed.loc[full_key].to_dict())
            else:
                sample_dict[metric_name] = MetricWithCI()
        sample_metrics[fpr_val] = sample_dict
    return CorrectnessRunMetrics(
        run_id=run.id,
        run_name=run.name or "",
        run_group=run.group or "",
        project=run.project or "",
        entity=run.entity,
        created_at=str(run.created_at),
        subset_name=subset_name,
        guardrail_type_name=type_name,
        auroc=auroc,
        average_precision=average_precision,
        tpr_at_fpr=tpr_at_fpr,
        attempt_metrics=attempt_metrics,
        sample_metrics=sample_metrics,
        overall_positive_rate=float(overall_positive_rate_raw) if overall_positive_rate_raw is not None else None,
        sample_count=int(sample_count_raw) if sample_count_raw is not None else None,
        record_count=int(record_count_raw) if record_count_raw is not None else None,
    )


def extract_correctness_category_metrics(
    run: wandb.apis.public.Run,
    subset_name: str,
    type_name: str | None = None,
    *,
    _resolved: tuple[str | None, str] | None = None,
) -> list[CorrectnessCategoryMetrics]:
    """Extract per-category correctness metrics from a W&B run.

    Fetches the category_metrics wandb Table and groups rows by category.

    Args:
        run: The W&B Run object.
        subset_name: Name of the evaluation subset.
        type_name: Optional guardrail type name for multi-type runs.
        _resolved: Pre-resolved (type_name, prefix) tuple. Internal use only.

    Returns:
        List of CorrectnessCategoryMetrics, one per discovered category.

    Raises:
        ValueError: If the category_metrics table is missing or has a schema mismatch.
    """
    if _resolved is not None:
        type_name, prefix = _resolved
    else:
        type_name, prefix = _resolve_type_and_prefix(run, subset_name, type_name)
    df = _fetch_category_metrics_df(run, prefix)
    if df is None:
        raise ValueError(
            f"category_metrics table not found for run {run.id}; this run may predate the table-based logging migration"
        )
    expected_cols = set(correctness_types.CATEGORY_METRICS_COLUMNS)
    actual_cols = set(df.columns)
    if not expected_cols.issubset(actual_cols):
        missing = expected_cols - actual_cols
        raise ValueError(f"category_metrics table schema mismatch for run {run.id}: missing columns {sorted(missing)}")
    fpr_prefix_re = re.compile(r"^fpr_" + _FPR_FRAGMENT_RE + r"/")
    results: list[CorrectnessCategoryMetrics] = []
    for cat_name, cat_group in df.groupby(correctness_types.COL_CATEGORY):
        cat_indexed = cat_group.set_index(correctness_types.COL_METRIC_NAME)
        # auroc
        auroc = MetricWithCI()
        if "auroc" in cat_indexed.index:
            auroc = _metric_with_ci_from_row(cat_indexed.loc["auroc"].to_dict())
        # detect FPR values
        fpr_values: set[float] = set()
        for metric_name in cat_indexed.index:
            match = fpr_prefix_re.match(str(metric_name))
            if match:
                fpr_values.add(_parse_fpr_capture(match.group(1)))
        sorted_fpr = sorted(fpr_values)
        # per-FPR metrics (whitelist-driven)
        attempt_metrics: dict[float, dict[str, MetricWithCI]] = {}
        sample_metrics: dict[float, dict[str, MetricWithCI]] = {}
        for fpr_val in sorted_fpr:
            fpr_key = correctness_metrics.format_fpr_key(fpr_val)
            attempt_dict: dict[str, MetricWithCI] = {}
            for metric_name in _ATTEMPT_METRIC_NAMES:
                full_key = f"{fpr_key}/{metric_name}"
                if full_key in cat_indexed.index:
                    attempt_dict[metric_name] = _metric_with_ci_from_row(cat_indexed.loc[full_key].to_dict())
                else:
                    attempt_dict[metric_name] = MetricWithCI()
            attempt_metrics[fpr_val] = attempt_dict
            sample_dict: dict[str, MetricWithCI] = {}
            for metric_name in _SAMPLE_METRIC_NAMES:
                full_key = f"{fpr_key}/{metric_name}"
                if full_key in cat_indexed.index:
                    sample_dict[metric_name] = _metric_with_ci_from_row(cat_indexed.loc[full_key].to_dict())
                else:
                    sample_dict[metric_name] = MetricWithCI()
            sample_metrics[fpr_val] = sample_dict
        # counts from first row (same for all rows in category)
        first_row = cat_indexed.iloc[0]
        record_count = int(first_row.get(correctness_types.COL_RECORD_COUNT, 0))
        sample_count = int(first_row.get(correctness_types.COL_SAMPLE_COUNT, 0))
        results.append(
            CorrectnessCategoryMetrics(
                category=str(cat_name),
                record_count=record_count,
                sample_count=sample_count,
                auroc=auroc,
                attempt_metrics=attempt_metrics,
                sample_metrics=sample_metrics,
            )
        )
    return results


def fetch_correctness_eval_summary(
    run: wandb.apis.public.Run,
    subset_name: str,
    type_name: str | None = None,
) -> CorrectnessRunSummary:
    """Fetch a complete correctness evaluation summary from a W&B run.

    Resolves the guardrail type once and passes the result to both extraction
    functions, avoiding redundant W&B summary scans and duplicate warnings.

    Args:
        run: The W&B Run object.
        subset_name: Name of the evaluation subset.
        type_name: Optional guardrail type name for multi-type runs.

    Returns:
        CorrectnessRunSummary with run-level and category metrics.
    """
    resolved = _resolve_type_and_prefix(run, subset_name, type_name)
    run_metrics = extract_correctness_metrics(run, subset_name, _resolved=resolved)
    category_metrics = extract_correctness_category_metrics(run, subset_name, _resolved=resolved)
    return CorrectnessRunSummary(
        run_info=run_metrics,
        category_metrics=category_metrics,
    )


def summarize_correctness_runs_to_dataframe(
    summaries: list[CorrectnessRunSummary],
) -> pd.DataFrame:
    """Convert a list of correctness summaries to a comparison DataFrame.

    Args:
        summaries: List of CorrectnessRunSummary from multiple runs.

    Returns:
        DataFrame with one row per run, including key metrics.
    """
    rows: list[dict[str, typing.Any]] = []
    for summary in summaries:
        info = summary.run_info
        row: dict[str, typing.Any] = {
            "run_id": info.run_id,
            "run_name": info.run_name,
            "run_group": info.run_group,
            "created_at": info.created_at,
            "type_name": info.guardrail_type_name,
            "auroc": info.auroc.value,
            "avg_precision": info.average_precision.value,
            "sample_count": info.sample_count,
            "record_count": info.record_count,
            "positive_rate": info.overall_positive_rate,
        }
        for fpr_val, metrics in sorted(info.tpr_at_fpr.items()):
            row[f"tpr@fpr={fpr_val}"] = metrics.value
        for fpr_val, metrics in sorted(info.sample_metrics.items()):
            guarded = metrics.get("guarded_pass_rate")
            unsafe = metrics.get("unsafe_slip_rate")
            if guarded is not None and guarded.value is not None:
                row[f"guarded_pass_rate@fpr={fpr_val}"] = guarded.value
            if unsafe is not None and unsafe.value is not None:
                row[f"unsafe_slip_rate@fpr={fpr_val}"] = unsafe.value
        rows.append(row)
    return pd.DataFrame(rows)


# ---- Pickle-to-Summary Helpers ----


def _metric_with_ci_from_aggregated(
    cross_run_mean: dict[str, float],
    hierarchical_cis: dict[str, typing.Any],
    key: str,
) -> MetricWithCI:
    """Build a ``MetricWithCI`` from cross-run mean and hierarchical CIs."""
    value = cross_run_mean.get(key)
    ci_obj = hierarchical_cis.get(key)
    ci_lower = ci_obj.lower_bound if ci_obj is not None else None
    ci_upper = ci_obj.upper_bound if ci_obj is not None else None
    return MetricWithCI(
        value=float(value) if value is not None else None,
        ci_lower=float(ci_lower) if ci_lower is not None else None,
        ci_upper=float(ci_upper) if ci_upper is not None else None,
    )


def eval_result_to_summary(
    eval_result: pyine.evals.correctness.CorrectnessEvalResult,
    *,
    subset_name: str | None = None,
    source_path: pathlib.Path | None = None,
    run_name: str | None = None,
    run_group: str | None = None,
    guardrail_type_name: str | None = None,
) -> CorrectnessRunSummary:
    """Build a ``CorrectnessRunSummary`` from a local ``CorrectnessEvalResult`` pickle.

    Maps directly from the structured types in ``eval_result.aggregated`` to populate the summary
    models, avoiding the need for W&B.

    Args:
        eval_result: The correctness eval result loaded from a pickle.
        subset_name: Evaluation subset name. Validated against ``eval_metadata["eval_subset_name"]``.
        source_path: Optional path to the pickle file (for run info fallbacks).
        run_name: Optional override for the run name.
        run_group: Optional override for the run group.
        guardrail_type_name: Optional guardrail type name override.

    Returns:
        A ``CorrectnessRunSummary`` with run-level and category metrics.

    Raises:
        ValueError: If subset_name or guardrail_type_name mismatches metadata, or if category
            names collide after sanitization, or if cross-run categories are inconsistent.
    """
    aggregated = eval_result.aggregated
    eval_metadata = eval_result.eval_metadata
    # resolve subset name
    metadata_subset = eval_metadata.get("eval_subset_name")
    if subset_name is not None and metadata_subset is not None and subset_name != metadata_subset:
        raise ValueError(
            f"subset_name={subset_name!r} does not match eval_metadata['eval_subset_name']={metadata_subset!r}"
        )
    resolved_subset = subset_name or metadata_subset
    if not resolved_subset:
        raise ValueError("subset_name must be provided or present in eval_metadata['eval_subset_name']")
    # resolve guardrail type name
    metadata_type = eval_metadata.get("guardrail_type_name")
    if guardrail_type_name is not None and metadata_type is not None and guardrail_type_name != metadata_type:
        raise ValueError(
            f"guardrail_type_name={guardrail_type_name!r} does not match "
            f"eval_metadata['guardrail_type_name']={metadata_type!r}"
        )
    resolved_type = guardrail_type_name or metadata_type
    # build run info
    run_info = pyine.evals.analysis_common.build_run_info_from_metadata(
        eval_metadata,
        resolved_subset,
        source_path=source_path,
        run_name=run_name,
        run_group=run_group,
    )
    cross_run_mean = aggregated.cross_run_mean
    hierarchical_cis = aggregated.hierarchical_cis
    # global metrics
    auroc = _metric_with_ci_from_aggregated(cross_run_mean, hierarchical_cis, "auroc")
    average_precision = _metric_with_ci_from_aggregated(cross_run_mean, hierarchical_cis, "average_precision")
    # tpr_at_fpr
    tpr_at_fpr: dict[float, MetricWithCI] = {}
    for key in cross_run_mean:
        if key.startswith("tpr_at_fpr_"):
            fpr_str = key[len("tpr_at_fpr_") :]
            fpr_val = _parse_fpr_capture(fpr_str)
            tpr_at_fpr[fpr_val] = _metric_with_ci_from_aggregated(cross_run_mean, hierarchical_cis, key)
    # detect FPR values from cross_run_mean keys
    fpr_values: set[float] = set()
    fpr_prefix_re = re.compile(r"^fpr_" + _FPR_FRAGMENT_RE + r"/")
    for key in cross_run_mean:
        match = fpr_prefix_re.match(key)
        if match:
            fpr_values.add(_parse_fpr_capture(match.group(1)))
    sorted_fpr_values = sorted(fpr_values)
    # attempt and sample metrics (whitelisted)
    attempt_metrics: dict[float, dict[str, MetricWithCI]] = {}
    sample_metrics: dict[float, dict[str, MetricWithCI]] = {}
    for fpr_val in sorted_fpr_values:
        fpr_key = correctness_metrics.format_fpr_key(fpr_val)
        attempt_dict: dict[str, MetricWithCI] = {}
        for metric_name in _ATTEMPT_METRIC_NAMES:
            full_key = f"{fpr_key}/{metric_name}"
            attempt_dict[metric_name] = _metric_with_ci_from_aggregated(cross_run_mean, hierarchical_cis, full_key)
        attempt_metrics[fpr_val] = attempt_dict
        sample_dict: dict[str, MetricWithCI] = {}
        for metric_name in _SAMPLE_METRIC_NAMES:
            full_key = f"{fpr_key}/{metric_name}"
            sample_dict[metric_name] = _metric_with_ci_from_aggregated(cross_run_mean, hierarchical_cis, full_key)
        sample_metrics[fpr_val] = sample_dict
    # class balance and counts
    overall_positive_rate = aggregated.class_balance.overall_positive_rate
    sample_count = len(aggregated.class_balance.per_sample_positive_rates)
    # record_count from split_summary with subset-sensitive key
    stripped_subset = resolved_subset
    if stripped_subset.startswith("guardrail_"):
        stripped_subset = stripped_subset[len("guardrail_") :]
    record_count_key = f"{stripped_subset}_record_count"
    if record_count_key not in aggregated.split_summary:
        raise ValueError(
            f"split_summary missing expected key {record_count_key!r}; "
            f"available keys: {sorted(aggregated.split_summary.keys())}"
        )
    record_count = int(aggregated.split_summary[record_count_key])
    run_metrics = CorrectnessRunMetrics(
        run_id=run_info.run_id,
        run_name=run_info.run_name,
        run_group=run_info.run_group,
        project=run_info.project,
        entity=run_info.entity,
        created_at=run_info.created_at,
        subset_name=run_info.subset_name,
        guardrail_type_name=resolved_type,
        auroc=auroc,
        average_precision=average_precision,
        tpr_at_fpr=tpr_at_fpr,
        attempt_metrics=attempt_metrics,
        sample_metrics=sample_metrics,
        overall_positive_rate=overall_positive_rate,
        sample_count=sample_count,
        record_count=record_count,
    )
    # category metrics
    correctness_types.validate_cross_run_categories(aggregated.per_run)
    reference_run = aggregated.per_run[0]
    safe_cat_map = correctness_types.build_safe_cat_reverse_map(reference_run.category_results)
    category_metrics_list: list[CorrectnessCategoryMetrics] = []
    for safe_cat in sorted(safe_cat_map.keys()):
        original_cat = safe_cat_map[safe_cat]
        cat_result = reference_run.category_results[original_cat]
        cat_auroc_key = f"category/{safe_cat}/auroc"
        cat_auroc = _metric_with_ci_from_aggregated(cross_run_mean, hierarchical_cis, cat_auroc_key)
        cat_attempt_metrics: dict[float, dict[str, MetricWithCI]] = {}
        cat_sample_metrics: dict[float, dict[str, MetricWithCI]] = {}
        for fpr_val in sorted_fpr_values:
            fpr_key = correctness_metrics.format_fpr_key(fpr_val)
            cat_attempt_dict: dict[str, MetricWithCI] = {}
            for metric_name in _ATTEMPT_METRIC_NAMES:
                full_key = f"category/{safe_cat}/{fpr_key}/{metric_name}"
                cat_attempt_dict[metric_name] = _metric_with_ci_from_aggregated(
                    cross_run_mean, hierarchical_cis, full_key
                )
            cat_attempt_metrics[fpr_val] = cat_attempt_dict
            cat_sample_dict: dict[str, MetricWithCI] = {}
            for metric_name in _SAMPLE_METRIC_NAMES:
                full_key = f"category/{safe_cat}/{fpr_key}/{metric_name}"
                cat_sample_dict[metric_name] = _metric_with_ci_from_aggregated(
                    cross_run_mean, hierarchical_cis, full_key
                )
            cat_sample_metrics[fpr_val] = cat_sample_dict
        category_metrics_list.append(
            CorrectnessCategoryMetrics(
                category=safe_cat,
                record_count=cat_result.record_count,
                sample_count=cat_result.sample_count,
                auroc=cat_auroc,
                attempt_metrics=cat_attempt_metrics,
                sample_metrics=cat_sample_metrics,
            )
        )
    return CorrectnessRunSummary(
        run_info=run_metrics,
        category_metrics=category_metrics_list,
    )


# ---- Plotting Functions ----


def plot_metric_comparison(
    summaries: list[CorrectnessRunSummary],
    metric_name: str,
    target_fpr: float | None = None,
    ax: matplotlib.axes.Axes | None = None,
    title: str | None = None,
) -> matplotlib.figure.Figure:  # pragma: no cover
    """Grouped bar chart comparing a single metric across runs.

    Args:
        summaries: List of run summaries to compare.
        metric_name: Metric to plot (e.g. "auroc", "tpr", "guarded_pass_rate").
        target_fpr: Required for FPR-conditioned metrics (attempt/sample level).
        ax: Optional existing axes to plot on.
        title: Optional chart title.

    Returns:
        The matplotlib Figure.
    """
    fig, ax = pyine.evals.analysis_common.get_or_create_axes(ax)
    labels: list[str] = []
    values: list[float] = []
    ci_lowers: list[float | None] = []
    ci_uppers: list[float | None] = []
    for summary in summaries:
        info = summary.run_info
        label = f"{info.run_group}/{info.run_name}" if info.run_group else info.run_name
        labels.append(label)
        metric = _resolve_metric(info, metric_name, target_fpr)
        values.append(metric.value if metric.value is not None else float("nan"))
        ci_lowers.append(metric.ci_lower)
        ci_uppers.append(metric.ci_upper)
    bar_positions = np.arange(len(labels))
    yerr_lower = [max(0, val - (cl if cl is not None else val)) for val, cl in zip(values, ci_lowers, strict=True)]
    yerr_upper = [max(0, (cu if cu is not None else val) - val) for val, cu in zip(values, ci_uppers, strict=True)]
    ax.bar(bar_positions, values, yerr=[yerr_lower, yerr_upper], capsize=4, alpha=0.8)
    chart_title = title or f"{metric_name}" + (f" @ FPR={target_fpr}" if target_fpr is not None else "")
    pyine.evals.analysis_common.configure_bar_chart(ax, bar_positions, labels, chart_title, ylabel=metric_name)
    return fig


_THRESHOLD_FREE_METRICS = frozenset({"auroc", "average_precision"})
_FPR_CONDITIONED_METRICS = frozenset(["tpr_at_fpr", *_ATTEMPT_METRIC_NAMES, *_SAMPLE_METRIC_NAMES])
_SUPPORTED_METRICS = frozenset([*_THRESHOLD_FREE_METRICS, *_FPR_CONDITIONED_METRICS])
_CATEGORY_SUPPORTED_METRICS = frozenset(["auroc", *_ATTEMPT_METRIC_NAMES, *_SAMPLE_METRIC_NAMES])


def _resolve_metric(  # pragma: no cover
    info: CorrectnessRunMetrics,
    metric_name: str,
    target_fpr: float | None,
) -> MetricWithCI:
    """Look up a named metric from CorrectnessRunMetrics.

    Raises:
        ValueError: If metric_name requires target_fpr but it is None.
    """
    if metric_name == "auroc":
        return info.auroc
    if metric_name == "average_precision":
        return info.average_precision
    if metric_name not in _FPR_CONDITIONED_METRICS:
        raise ValueError(f"unsupported metric {metric_name!r}; expected one of {sorted(_SUPPORTED_METRICS)}")
    if target_fpr is None:
        raise ValueError(
            f"metric {metric_name!r} requires target_fpr; only {sorted(_THRESHOLD_FREE_METRICS)} are threshold-free"
        )
    if metric_name == "tpr_at_fpr":
        if target_fpr not in info.tpr_at_fpr:
            raise ValueError(
                f"target_fpr={target_fpr} not available for tpr_at_fpr; "
                f"available values: {sorted(info.tpr_at_fpr.keys())}"
            )
        return info.tpr_at_fpr[target_fpr]
    attempt = info.attempt_metrics.get(target_fpr, {})
    if metric_name in attempt:
        return attempt[metric_name]
    sample = info.sample_metrics.get(target_fpr, {})
    if metric_name in sample:
        return sample[metric_name]
    available = sorted({*attempt.keys(), *sample.keys()})
    raise ValueError(
        f"metric {metric_name!r} not available at target_fpr={target_fpr}; available metrics at this FPR: {available}"
    )


def plot_roc_curves(
    per_run_results: list[correctness_types.SingleRunResult],
    title: str | None = None,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:  # pragma: no cover
    """Overlay ROC curves from per-run results with mean curve.

    Requires local ``SingleRunResult`` objects (from pickled ``CorrectnessEvalResult.aggregated.per_run``).

    Args:
        per_run_results: List of SingleRunResult from independent guardrail runs.
        title: Optional chart title.
        ax: Optional existing axes to plot on.

    Returns:
        The matplotlib Figure.
    """
    fig, ax = pyine.evals.analysis_common.get_or_create_axes(ax)
    all_fpr_grids: list[NDArray[np.floating[typing.Any]]] = []
    all_tpr_grids: list[NDArray[np.floating[typing.Any]]] = []
    for run_idx, run_result in enumerate(per_run_results):
        fpr_grid = np.array(run_result.threshold_free.fpr_grid)
        tpr_grid = np.array(run_result.threshold_free.tpr_grid)
        ax.plot(fpr_grid, tpr_grid, alpha=0.3, linewidth=1, label=f"Run {run_idx + 1}")
        all_fpr_grids.append(fpr_grid)
        all_tpr_grids.append(tpr_grid)
    if all_tpr_grids:
        shapes = {grid.shape for grid in all_tpr_grids}
        if len(shapes) == 1:
            mean_tpr = np.mean(np.array(all_tpr_grids), axis=0)
            ax.plot(all_fpr_grids[0], mean_tpr, color="black", linewidth=2, label="Mean")
        else:
            logger.warning("skipping mean ROC curve: grid shapes differ across runs %s", shapes)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title or "ROC Curves")
    ax.legend(fontsize="small")
    ax.grid(alpha=0.3)
    return fig


def plot_pr_curves(
    per_run_results: list[correctness_types.SingleRunResult],
    title: str | None = None,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:  # pragma: no cover
    """Overlay precision-recall curves from per-run results with mean curve.

    Args:
        per_run_results: List of SingleRunResult from independent guardrail runs.
        title: Optional chart title.
        ax: Optional existing axes to plot on.

    Returns:
        The matplotlib Figure.
    """
    fig, ax = pyine.evals.analysis_common.get_or_create_axes(ax)
    all_recall_grids: list[NDArray[np.floating[typing.Any]]] = []
    all_precision_grids: list[NDArray[np.floating[typing.Any]]] = []
    for run_idx, run_result in enumerate(per_run_results):
        recall_grid = np.array(run_result.threshold_free.recall_grid)
        precision_grid = np.array(run_result.threshold_free.precision_grid)
        ax.plot(recall_grid, precision_grid, alpha=0.3, linewidth=1, label=f"Run {run_idx + 1}")
        all_recall_grids.append(recall_grid)
        all_precision_grids.append(precision_grid)
    if all_precision_grids:
        shapes = {grid.shape for grid in all_precision_grids}
        if len(shapes) == 1:
            mean_precision = np.mean(np.array(all_precision_grids), axis=0)
            ax.plot(all_recall_grids[0], mean_precision, color="black", linewidth=2, label="Mean")
        else:
            logger.warning("skipping mean PR curve: grid shapes differ across runs %s", shapes)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title or "Precision-Recall Curves")
    ax.legend(fontsize="small")
    ax.grid(alpha=0.3)
    return fig


def plot_operating_point_summary(
    thresholded_metrics: correctness_types.ThresholdedMetrics,
    sample_metrics: correctness_types.SampleLevelMetrics,
    title: str | None = None,
) -> matplotlib.figure.Figure:  # pragma: no cover
    """Confusion matrix heatmap + sidebar with rates at one operating point.

    Args:
        thresholded_metrics: Attempt-level confusion matrix and rates.
        sample_metrics: Sample-level metrics at the same operating point.
        title: Optional chart title.

    Returns:
        The matplotlib Figure.
    """
    fig, (ax_cm, ax_rates) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 1.2]})
    # confusion matrix heatmap
    cm = np.array(
        [
            [thresholded_metrics.tp, thresholded_metrics.fn],
            [thresholded_metrics.fp, thresholded_metrics.tn],
        ]
    )
    ax_cm.imshow(cm, cmap="Blues", aspect="auto")
    for row_idx in range(2):
        for col_idx in range(2):
            ax_cm.text(col_idx, row_idx, str(cm[row_idx, col_idx]), ha="center", va="center", fontsize=14)
    ax_cm.set_xticks([0, 1])
    ax_cm.set_xticklabels(["Predicted +", "Predicted -"])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_yticklabels(["Actual +", "Actual -"])
    ax_cm.set_title(f"Confusion Matrix (FPR={thresholded_metrics.target_fpr})")
    # rates sidebar
    rate_names = ["TPR", "FPR", "Precision", "NPV", "Guarded Pass Rate", "Unsafe Slip Rate"]
    rate_values = [
        thresholded_metrics.tpr,
        thresholded_metrics.fpr,
        thresholded_metrics.precision,
        thresholded_metrics.npv,
        sample_metrics.guarded_pass_rate,
        sample_metrics.unsafe_slip_rate,
    ]
    rate_values_safe = [val if val is not None else 0.0 for val in rate_values]
    y_pos = np.arange(len(rate_names))
    colors = ["tab:blue" if val is not None else "tab:gray" for val in rate_values]
    ax_rates.barh(y_pos, rate_values_safe, color=colors, alpha=0.8)
    ax_rates.set_yticks(y_pos)
    ax_rates.set_yticklabels(rate_names)
    ax_rates.set_xlim(0, 1.05)
    ax_rates.set_title("Rates at Operating Point")
    ax_rates.grid(axis="x", alpha=0.3)
    for idx, val in enumerate(rate_values):
        if val is not None:
            ax_rates.text(val + 0.01, idx, f"{val:.3f}", va="center", fontsize=9)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_sample_level_metrics(
    summaries: list[CorrectnessRunSummary],
    target_fpr: float,
    ax: matplotlib.axes.Axes | None = None,
    title: str | None = None,
) -> matplotlib.figure.Figure:  # pragma: no cover
    """Grouped bar chart of sample-level metrics across runs at one FPR.

    Args:
        summaries: List of run summaries.
        target_fpr: The FPR threshold to report metrics at.
        ax: Optional existing axes.
        title: Optional chart title.

    Returns:
        The matplotlib Figure.
    """
    metric_names = [
        "base_pass_rate",
        "guarded_pass_rate",
        "unsafe_slip_rate",
        "total_block_rate",
        "best_of_k_success_rate",
    ]
    fig, ax = pyine.evals.analysis_common.get_or_create_axes(ax, figsize=(12, 6))
    num_runs = len(summaries)
    num_metrics = len(metric_names)
    bar_positions = np.arange(num_metrics)
    bar_width = 0.8 / max(num_runs, 1)
    for run_idx, summary in enumerate(summaries):
        info = summary.run_info
        label = f"{info.run_group}/{info.run_name}" if info.run_group else info.run_name
        sample_dict = info.sample_metrics.get(target_fpr, {})
        raw = [sample_dict.get(name, MetricWithCI()).value for name in metric_names]
        values = [val if val is not None else float("nan") for val in raw]
        offset = (run_idx - num_runs / 2 + 0.5) * bar_width
        ax.bar(bar_positions + offset, values, bar_width, label=label, alpha=0.8)
    ax.set_xticks(bar_positions)
    ax.set_xticklabels([name.replace("_", " ") for name in metric_names], rotation=30, ha="right")
    ax.set_ylabel("Rate")
    ax.set_title(title or f"Sample-Level Metrics @ FPR={target_fpr}")
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize="small")
    ax.grid(axis="y", alpha=0.3)
    return fig


def plot_category_breakdown(
    summary: CorrectnessRunSummary,
    metric_name: str,
    target_fpr: float | None = None,
    ax: matplotlib.axes.Axes | None = None,
    title: str | None = None,
) -> matplotlib.figure.Figure:  # pragma: no cover
    """Per-category bar chart with sample counts.

    Args:
        summary: A single run summary.
        metric_name: Metric to plot (e.g. "auroc", "tpr", "guarded_pass_rate").
        target_fpr: Required for FPR-conditioned metrics.
        ax: Optional existing axes.
        title: Optional chart title.

    Returns:
        The matplotlib Figure.
    """
    categories = summary.category_metrics
    if metric_name not in _CATEGORY_SUPPORTED_METRICS:
        raise ValueError(
            f"unsupported category metric {metric_name!r}; expected one of {sorted(_CATEGORY_SUPPORTED_METRICS)}"
        )
    if metric_name in _THRESHOLD_FREE_METRICS and metric_name != "auroc":
        raise ValueError(f"per-category threshold-free plotting currently supports only 'auroc', got {metric_name!r}")
    if metric_name not in _THRESHOLD_FREE_METRICS and target_fpr is None:
        raise ValueError(
            f"metric {metric_name!r} requires target_fpr; only {sorted(_THRESHOLD_FREE_METRICS)} are threshold-free"
        )
    fig, ax = pyine.evals.analysis_common.get_or_create_axes(ax)
    if not categories:
        ax.text(0.5, 0.5, "No category data", ha="center", va="center", transform=ax.transAxes)
        return fig
    labels: list[str] = []
    values: list[float] = []
    for cat in categories:
        labels.append(f"{cat.category}\n(n={cat.sample_count})")
        if metric_name == "auroc":
            values.append(cat.auroc.value if cat.auroc.value is not None else float("nan"))
        elif target_fpr is not None:
            att_dict = cat.attempt_metrics.get(target_fpr, {})
            samp_dict = cat.sample_metrics.get(target_fpr, {})
            if metric_name in att_dict:
                metric = att_dict[metric_name]
            elif metric_name in samp_dict:
                metric = samp_dict[metric_name]
            else:
                available = sorted({*att_dict.keys(), *samp_dict.keys()})
                raise ValueError(
                    f"metric {metric_name!r} not available for category {cat.category!r} "
                    f"at target_fpr={target_fpr}; available metrics: {available}"
                )
            values.append(metric.value if metric.value is not None else float("nan"))
        else:
            raise AssertionError("unreachable: non-auroc metric without target_fpr")
    bar_positions = np.arange(len(labels))
    ax.bar(bar_positions, values, alpha=0.8)
    chart_title = title or f"{metric_name} by Category" + (f" @ FPR={target_fpr}" if target_fpr else "")
    pyine.evals.analysis_common.configure_bar_chart(ax, bar_positions, labels, chart_title, ylabel=metric_name)
    return fig


def plot_difficulty_analysis(
    difficulty_stats: correctness_types.DifficultyStats,
    ax: matplotlib.axes.Axes | None = None,
    title: str | None = None,
) -> matplotlib.figure.Figure:  # pragma: no cover
    """Per-bucket AUROC and TPR bar charts from DifficultyStats.

    Args:
        difficulty_stats: Difficulty stats from an aggregated result.
        ax: Optional existing axes.
        title: Optional chart title.

    Returns:
        The matplotlib Figure.
    """
    has_auroc = difficulty_stats.per_bucket_auroc is not None
    has_tpr = difficulty_stats.per_bucket_tpr is not None
    ncols = int(has_auroc) + int(has_tpr)
    if ncols == 0:
        fig, ax = pyine.evals.analysis_common.get_or_create_axes(ax)
        ax.text(0.5, 0.5, "No difficulty data", ha="center", va="center", transform=ax.transAxes)
        return fig
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    if ncols == 1:
        axes = [axes]
    plot_idx = 0
    if has_auroc and difficulty_stats.per_bucket_auroc is not None:
        curr_ax = axes[plot_idx]
        buckets = list(difficulty_stats.per_bucket_auroc.keys())
        auroc_vals = [
            difficulty_stats.per_bucket_auroc[bucket]
            if difficulty_stats.per_bucket_auroc[bucket] is not None
            else float("nan")
            for bucket in buckets
        ]
        bar_positions = np.arange(len(buckets))
        curr_ax.bar(bar_positions, auroc_vals, alpha=0.8)
        pyine.evals.analysis_common.configure_bar_chart(
            curr_ax, bar_positions, buckets, "AUROC by Difficulty", ylabel="AUROC"
        )
        plot_idx += 1
    if has_tpr and difficulty_stats.per_bucket_tpr is not None:
        curr_ax = axes[plot_idx]
        buckets = list(difficulty_stats.per_bucket_tpr.keys())
        # show TPR for each FPR target
        fpr_targets = set()
        for bucket_data in difficulty_stats.per_bucket_tpr.values():
            fpr_targets.update(bucket_data.keys())
        bar_positions = np.arange(len(buckets))
        sorted_fpr_targets = sorted(fpr_targets)
        num_series = max(len(sorted_fpr_targets), 1)
        bar_width = 0.8 / num_series
        for series_idx, fpr_val in enumerate(sorted_fpr_targets):
            tpr_vals = [difficulty_stats.per_bucket_tpr[bucket].get(fpr_val, float("nan")) for bucket in buckets]
            offset = (series_idx - num_series / 2 + 0.5) * bar_width
            curr_ax.bar(bar_positions + offset, tpr_vals, bar_width, alpha=0.6, label=f"FPR={fpr_val}")
        pyine.evals.analysis_common.configure_bar_chart(
            curr_ax, bar_positions, buckets, "TPR by Difficulty", ylabel="TPR"
        )
        curr_ax.legend(fontsize="small")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_cost_analysis(
    cost_stats: correctness_types.VerificationCostStats,
    ax: matplotlib.axes.Axes | None = None,
    title: str | None = None,
) -> matplotlib.figure.Figure:  # pragma: no cover
    """Cost summary horizontal bar chart from VerificationCostStats.

    Args:
        cost_stats: Cost stats from an aggregated result (at one FPR).
        ax: Optional existing axes.
        title: Optional chart title.

    Returns:
        The matplotlib Figure.
    """
    fig, ax = pyine.evals.analysis_common.get_or_create_axes(ax)
    metrics = {
        "Total Cost": cost_stats.total_cost,
        "Mean/Record": cost_stats.mean_cost_per_record,
        "Median/Record": cost_stats.median_cost_per_record,
        "Cost/Correct Accept": cost_stats.cost_per_correct_acceptance,
        "Cost/Incorrect Block": cost_stats.cost_per_incorrect_block,
    }
    valid_metrics = {name: val for name, val in metrics.items() if val is not None}
    if not valid_metrics:
        ax.text(0.5, 0.5, "No cost data", ha="center", va="center", transform=ax.transAxes)
        return fig
    names = list(valid_metrics.keys())
    values = list(valid_metrics.values())
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel(f"Cost ({cost_stats.cost_unit or 'units'})")
    ax.set_title(title or f"Verification Costs (FPR={cost_stats.target_fpr})")
    ax.grid(axis="x", alpha=0.3)
    return fig


def plot_cross_run_variability(
    summaries: list[CorrectnessRunSummary],
    metric_names: list[str],
    target_fpr: float | None = None,
    ax: matplotlib.axes.Axes | None = None,
    title: str | None = None,
) -> matplotlib.figure.Figure:  # pragma: no cover
    """Strip plot showing per-run metric values with mean overlay.

    Args:
        summaries: List of run summaries.
        metric_names: Metrics to plot (e.g. ["auroc", "tpr", "guarded_pass_rate"]).
        target_fpr: Required for FPR-conditioned metrics.
        title: Optional chart title.

    Returns:
        The matplotlib Figure.
    """
    fig, ax = pyine.evals.analysis_common.get_or_create_axes(ax, figsize=(max(8, len(metric_names) * 2), 6))
    bar_positions = np.arange(len(metric_names))
    for summary in summaries:
        raw_values = [_resolve_metric(summary.run_info, name, target_fpr).value for name in metric_names]
        values = [val if val is not None else float("nan") for val in raw_values]
        ax.scatter(bar_positions, values, alpha=0.5, s=40, zorder=3)
    # compute and plot means
    for metric_idx, metric_name in enumerate(metric_names):
        all_values = [_resolve_metric(summary.run_info, metric_name, target_fpr).value for summary in summaries]
        valid = [val for val in all_values if val is not None]
        if valid:
            mean_val = np.mean(valid)
            ax.plot([metric_idx - 0.2, metric_idx + 0.2], [mean_val, mean_val], "k-", linewidth=2, zorder=4)
    ax.set_xticks(bar_positions)
    ax.set_xticklabels([name.replace("_", "\n") for name in metric_names])
    ax.set_ylabel("Value")
    ax.set_title(title or "Cross-Run Metric Variability")
    ax.grid(axis="y", alpha=0.3)
    return fig
