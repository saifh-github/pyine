"""Reward logging helpers.

The reward system exposes a small `RewardLogger` protocol. The manager controls logging frequency
gating via the `should_log_sample()` query method; when the manager calls `log_sample()`, the
logger always emits. The manager also handles key scoping and orchestration.

Metric Indexing
---------------
When using WandBRewardLogger, metrics are indexed to different x-axes depending on their type:

**Per-generation metrics** (indexed to `{prefix}/generation_count`):
    These metrics are logged for individual generations (completions) and use generation_count
    as the x-axis. This ensures each logged generation has a unique x-coordinate, avoiding WandB
    aggregation issues when multiple generations are logged within the same trainer step (e.g.,
    with gradient accumulation or multiple generations per prompt in GRPO).

    - `{prefix}/reward/total`: total reward for the generation;
    - `{prefix}/reward/terms/*`: per-term weighted reward values;
    - `{prefix}/reward/metrics/*`: term-emitted metrics (containing other useful information);
    - `{prefix}/reward/raw_terms/*`: pre-clipping, pre-weighting reward term values;

**Batch-level metrics** (indexed to `{prefix}/batch_count`):
    These metrics are logged once per compute_batch call (when enabled) and use batch_count as the x-axis.
    This ensures each logged batch has a unique x-coordinate, avoiding WandB aggregation issues
    when multiple batches are processed within the same trainer step (e.g., gradient accumulation).

    Note: Batch-level metrics are only emitted when `LoggingConfig.log_batch_stats=True`.

    - `{prefix}/reward/batch/mean`: mean reward across the batch;
    - `{prefix}/reward/batch/std`: standard deviation of rewards in the batch.

**Run-level summaries** (indexed to `step_metric_key`, default `train/global_step`):
    These metrics are logged when flush_stats() is called (e.g., at phase transitions).

    - `{prefix}/reward/run/total/{mean,std,min,max,count}`: accumulated total reward statistics;
    - `{prefix}/reward/run/terms/{term}/{mean,std,min,max,count}`: per-term reward statistics;
    - `{prefix}/reward/run/categories/{category}/{mean,std,min,max,count}`: per-category reward statistics;
    - `{prefix}/reward/run/category_term_summary`: per-category per-term reward statistics (table);
      Note: per-term values are pre-verbosity-scaling and may not sum to category totals.
    - `{prefix}/parsing/*`: aggregated parsing statistics (lengths, missing ratios);
    - `{prefix}/failures/failure_ratio`: ratio of failed generations in the phase;
    - `{prefix}/failures/failure_count`: count of failed generations in the phase.
    - `{prefix}/difficulty/run/*`: aggregated difficulty diagnostics (when enabled).

Where `{prefix}` is typically "train" or "eval" depending on the training phase.
"""

import collections.abc
import json
import logging
import pathlib
import re
import threading
import typing

import wandb

import pyine.data.utils.generation_record
import pyine.data.utils.lmdb_io
import pyine.organisms.models.rewards.core.configs as reward_configs
import pyine.organisms.models.rewards.core.types as reward_types
import pyine.utils.parsing as parsing_utils
import pyine.utils.stats as stats_utils

logger = logging.getLogger(__name__)


def _has_nonempty_stats(
    stats_items: list[tuple[str, stats_utils.RunningStats]] | None,
) -> bool:
    """Check if any RunningStats in the list has count > 0."""
    if not stats_items:
        return False
    return any(stats.count > 0 for _name, stats in stats_items)


def _has_nonempty_bin_stats(
    bin_stats: list[stats_utils.RunningStats] | None,
) -> bool:
    """Check if any bin RunningStats has count > 0."""
    if not bin_stats:
        return False
    return any(stats.count > 0 for stats in bin_stats)


def _has_nonempty_bin_term_stats(
    bin_term_stats: dict[str, list[stats_utils.RunningStats]] | None,
) -> bool:
    """Check if any per-term per-bin RunningStats has count > 0."""
    if not bin_term_stats:
        return False
    return any(stats.count > 0 for term_stats_list in bin_term_stats.values() for stats in term_stats_list)


def _build_stats_table_from_running_stats(
    stats_items: list[tuple[str, stats_utils.RunningStats]],
    id_column: str,
) -> wandb.Table:
    """Build table from (name, RunningStats) pairs in given order."""
    columns = [id_column, "mean", "std", "min", "max", "count"]
    table = wandb.Table(columns=columns)
    for name, stats in stats_items:
        if stats.count > 0:
            table.add_data(name, stats.mean(), stats.std(), stats.min, stats.max, stats.count)  # type: ignore[reportUnknownMemberType]
    return table


def _has_nonempty_category_term_stats(
    category_term_stats: dict[str, list[tuple[str, stats_utils.RunningStats]]] | None,
) -> bool:
    """Check if any per-category per-term RunningStats has count > 0."""
    if not category_term_stats:
        return False
    return any(stats.count > 0 for term_stats_list in category_term_stats.values() for _name, stats in term_stats_list)


def _build_category_term_stats_table(
    category_term_stats: dict[str, list[tuple[str, stats_utils.RunningStats]]],
) -> wandb.Table:
    """Build table from per-category per-term RunningStats pairs.

    Args:
        category_term_stats: Dict mapping category name to ordered list of (term_name, RunningStats).

    Returns:
        wandb.Table with columns [category, term, mean, std, min, max, count],
        one row per (category, term) pair with count > 0.
    """
    columns = ["category", "term", "mean", "std", "min", "max", "count"]
    table = wandb.Table(columns=columns)
    for category, term_stats_list in category_term_stats.items():
        for term_name, stats in term_stats_list:
            if stats.count > 0:
                table.add_data(category, term_name, stats.mean(), stats.std(), stats.min, stats.max, stats.count)  # type: ignore[reportUnknownMemberType]
    return table


def _build_difficulty_bin_table(
    bin_stats: list[stats_utils.RunningStats],
    bin_edges: list[float],
    bin_reward_values: list[list[float]] | None = None,
) -> wandb.Table:
    """Build table from per-bin RunningStats in ascending bin_idx order.

    Args:
        bin_stats: Per-bin RunningStats (list indexed by bin_idx).
        bin_edges: Bin edge values (must have len(bin_stats) + 1 elements).
        bin_reward_values: Optional per-bin raw values for quantile computation.

    Raises:
        ValueError: If len(bin_edges) != len(bin_stats) + 1.

    Note: Infinity bin edges are stored as actual float("inf") in the table.
    """
    if len(bin_edges) != len(bin_stats) + 1:
        raise ValueError(f"bin_edges length ({len(bin_edges)}) must be len(bin_stats) + 1 ({len(bin_stats) + 1})")
    columns = [
        "bin_idx",
        "lower_edge",
        "upper_edge",
        "count",
        "reward_mean",
        "reward_std",
        "reward_min",
        "reward_max",
        "reward_p10",
        "reward_p50",
        "reward_p90",
    ]
    table = wandb.Table(columns=columns)
    for bin_idx, stats in enumerate(bin_stats):
        if stats.count > 0:
            lower = bin_edges[bin_idx]
            upper = bin_edges[bin_idx + 1] if bin_idx + 1 < len(bin_edges) else float("inf")
            # compute quantiles if raw values available
            p10 = p50 = p90 = None
            if bin_reward_values and bin_idx < len(bin_reward_values):
                bin_vals = sorted(bin_reward_values[bin_idx])
                if bin_vals:
                    n = len(bin_vals)
                    p10 = bin_vals[stats_utils.percentile_index(n, 10)]
                    p50 = bin_vals[stats_utils.percentile_index(n, 50)]
                    p90 = bin_vals[stats_utils.percentile_index(n, 90)]
            table.add_data(  # type: ignore[reportUnknownMemberType]
                bin_idx,
                lower,
                upper,
                stats.count,
                stats.mean(),
                stats.std(),
                stats.min,
                stats.max,
                p10,
                p50,
                p90,
            )
    return table


def _build_difficulty_bin_term_table(
    bin_term_stats: dict[str, list[stats_utils.RunningStats]],
    bin_edges: list[float],
    term_order: list[str],
) -> wandb.Table:
    """Build table from per-term per-bin RunningStats.

    Rows are ordered by bin_idx, then by term order from config.

    Args:
        bin_term_stats: Dict mapping term name to per-bin RunningStats lists.
        bin_edges: Bin edge values (must have at least 2 elements to define bins).
        term_order: Ordered list of term names.

    Raises:
        ValueError: If bin_edges has fewer than 2 elements, or if any term's stats list
            has a length that doesn't match the expected number of bins.
    """
    if len(bin_edges) < 2:
        raise ValueError(f"bin_edges must have at least 2 elements, got {len(bin_edges)}")
    num_bins = len(bin_edges) - 1
    # validate per-term list lengths
    for term_name, term_stats_list in bin_term_stats.items():
        if len(term_stats_list) != num_bins:
            raise ValueError(
                f"term '{term_name}' has {len(term_stats_list)} bin stats, expected {num_bins} (len(bin_edges) - 1)"
            )
    columns = ["bin_idx", "lower_edge", "upper_edge", "term", "reward_mean", "reward_std"]
    table = wandb.Table(columns=columns)
    for bin_idx in range(num_bins):
        lower = bin_edges[bin_idx]
        upper = bin_edges[bin_idx + 1] if bin_idx + 1 < len(bin_edges) else float("inf")
        for term_name in term_order:
            term_bin_stats_list = bin_term_stats.get(term_name)
            if not term_bin_stats_list:
                continue
            stats = term_bin_stats_list[bin_idx]
            if stats.count > 0:
                table.add_data(bin_idx, lower, upper, term_name, stats.mean(), stats.std())  # type: ignore[reportUnknownMemberType]
    return table


def _build_parsing_category_table(
    category_stats: list[tuple[str, dict[str, float | int | None]]],
) -> wandb.Table:
    """Build parsing category table with dynamic columns from actual data.

    Columns are built from the union of all keys across all categories, ensuring no data is lost
    even if new metrics are added. Column order is deterministic: ["category", "count"] followed
    by remaining keys sorted alphabetically.
    """
    # build column set from union of all stats keys
    all_keys: set[str] = set()
    for _category, stats in category_stats:
        all_keys.update(stats.keys())
    # deterministic order: category first, count second, then alphabetical
    all_keys.discard("count")  # remove count so we can place it second
    columns = ["category", "count"] + sorted(all_keys)
    table = wandb.Table(columns=columns)
    for category, stats in category_stats:
        row: list[typing.Any] = [category, stats.get("count")]
        for col in columns[2:]:  # skip "category" and "count"
            row.append(stats.get(col))  # None if missing
        table.add_data(*row)  # type: ignore[reportUnknownMemberType]
    return table


def _build_secondary_source_table(
    secondary_stats: dict[str, stats_utils.RunningStats],
    secondary_corr_stats: dict[str, stats_utils.RunningCorrStats],
) -> wandb.Table:
    """Build W&B table for secondary difficulty source statistics."""
    columns = ["source", "mean", "std", "min", "max", "count", "reward_correlation", "reward_slope"]
    table = wandb.Table(columns=columns)
    for source in sorted(set(secondary_stats.keys()) | set(secondary_corr_stats.keys())):
        raw = secondary_stats.get(source)
        corr = secondary_corr_stats.get(source)
        table.add_data(  # type: ignore[reportUnknownMemberType]
            source,
            raw.mean() if raw and raw.count > 0 else None,
            raw.std() if raw and raw.count > 0 else None,
            raw.min if raw and raw.count > 0 else None,
            raw.max if raw and raw.count > 0 else None,
            raw.count if raw else 0,
            corr.correlation() if corr and corr.count > 1 else None,
            corr.slope() if corr and corr.count > 1 else None,
        )
    return table


class InMemoryRewardLogger:
    """RewardLogger implementation for tests and debugging.

    Stores structured events in memory instead of writing to an external backend.
    Supports optional frequency gating for consistency with WandBRewardLogger.
    """

    def __init__(
        self,
        *,
        log_every_n_generations: int = 1,
        log_tables: bool = False,
    ) -> None:
        """Create an in-memory logger with empty buffers.

        Args:
            log_every_n_generations: Record samples every N generations (default 1 = log all).
            log_tables: Whether to record table row events (default False).
        """
        self.samples: list[dict[str, object]] = []
        self.table_rows: list[dict[str, object]] = []
        self.runs: list[dict[str, object]] = []
        self.batch_stats: list[dict[str, object]] = []
        self._step: int | None = None
        self._epoch: float | None = None
        self._key_prefix: str = ""
        self._log_every_n_generations = log_every_n_generations
        self._log_generation_table = log_tables
        if self._log_every_n_generations < 1:
            raise ValueError("log_every_n_generations must be >= 1")

    def should_log_sample(
        self,
        generation_count: int,
    ) -> bool:
        """Return True if sample metrics should be logged for this generation_count.

        This controls both scalar emission and table row addition. The manager uses this
        to decide whether to call log_sample() for a given generation.
        """
        return generation_count % self._log_every_n_generations == 0

    def log_sample(
        self,
        sample_id: str,
        *,
        generation_count: int | None = None,
        batch_count: int | None = None,
        local_batch_idx: int | None = None,
        completion_idx: int | None = None,
        rank: int | None = None,
        total: float | None,
        terms: collections.abc.Mapping[str, float] | None = None,
        metrics: collections.abc.Mapping[str, reward_types.MetricValue] | None = None,
        raw_terms: collections.abc.Mapping[str, float] | None = None,
        step: int | None = None,
        prompt: str | None = None,
        expected_output: str | None = None,
        model_output: str | None = None,
        reasoning: str | None = None,
        final_answer: str | None = None,
        categories: collections.abc.Sequence[str] | None = None,
        tags: collections.abc.Sequence[str] | None = None,
        difficulty_source: str | None = None,
        difficulty_score: float | None = None,
        difficulty_bin: int | None = None,
        difficulty_raw_primary: float | None = None,
        difficulty_secondary_json: str | None = None,
        predict_type: str | None = None,
        code_type: str | None = None,
        has_code_override: bool | None = None,
        pregenerated_output: str | None = None,
        **kwargs: typing.Any,
    ) -> None:
        """Record a per-sample logging event in memory.

        Args:
            sample_id: Unique identifier for the sample.
            generation_count: 1-indexed count of generations seen. If None, frequency gating is skipped.
            batch_count: Global batch counter (1-indexed).
            local_batch_idx: Index of this sample within the current batch (0-indexed).
            completion_idx: Global completion index within samples sharing the same identifier
                across all ranks in this batch (0-indexed). Globally unique within each batch.
            rank: Global rank of the process logging this sample.
            total: Total reward value for the sample.
            terms: Per-term weighted reward values.
            metrics: Additional metrics emitted by reward terms.
            raw_terms: Pre-clipping, pre-weighting term values.
            step: Optional step value.
            prompt: The input prompt.
            expected_output: Ground truth output.
            model_output: Model-generated output.
            reasoning: Parsed reasoning text.
            final_answer: Parsed final answer text.
            categories: Sample categories for grouping.
            tags: Additional sample tags.
            difficulty_source: Name of the primary difficulty source.
            difficulty_score: Normalized difficulty score.
            difficulty_bin: Bin index for this sample's difficulty.
            difficulty_raw_primary: Raw value of the primary difficulty source.
            difficulty_secondary_json: JSON string of secondary difficulty raw values.
            predict_type: Sample predict type.
            code_type: Sample code type.
            has_code_override: Whether the sample has a code override.
            pregenerated_output: Pregenerated pseudolabel output from a prior run (if any).
            **kwargs: Additional fields to store.
        """
        # note: frequency gating is the caller's responsibility (typically RewardManager);
        # when log_sample is called, we always record the sample and add a table row (if enabled).
        record: dict[str, object] = {
            "sample_id": sample_id,
            "generation_count": generation_count,
            "batch_count": batch_count,
            "local_batch_idx": local_batch_idx,
            "completion_idx": completion_idx,
            "rank": rank,
            "step": step,
            "internal_step": self._step,
            "internal_epoch": self._epoch,
            "internal_key_prefix": self._key_prefix,
            "prompt": prompt,
            "expected_output": expected_output,
            "model_output": model_output,
            "reasoning": reasoning,
            "final_answer": final_answer,
            "reward_total": total,
            "reward_terms": dict(terms) if terms is not None else None,
            "reward_terms_raw": dict(raw_terms) if raw_terms is not None else None,
            "reward_metrics": dict(metrics) if metrics is not None else None,
            "categories": list(categories) if categories is not None else None,
            "tags": list(tags) if tags is not None else None,
            "difficulty_source": difficulty_source,
            "difficulty_score": difficulty_score,
            "difficulty_bin": difficulty_bin,
            "difficulty_raw_primary": difficulty_raw_primary,
            "difficulty_secondary_json": difficulty_secondary_json,
            "predict_type": predict_type,
            "code_type": code_type,
            "has_code_override": has_code_override,
            "pregenerated_output": pregenerated_output,
            **kwargs,
        }
        self.samples.append(dict(record))
        if self._log_generation_table:
            self.table_rows.append(dict(record))

    def set_step(self, step: int | None) -> None:
        """Set a default step value for subsequent logs."""
        self._step = step

    def set_epoch(self, epoch: float | None) -> None:
        """Set a default epoch value for subsequent logs."""
        self._epoch = epoch

    def set_key_prefix(self, key_prefix: str) -> None:
        """Set the key prefix for subsequent logs."""
        self._key_prefix = parsing_utils.normalize_path_prefix(key_prefix)

    def get_key_prefix(self) -> str:
        """Get the current key prefix."""
        return self._key_prefix

    def log_phase_summaries(
        self,
        *,
        reward_totals: collections.abc.Mapping[str, reward_types.MetricValue],
        reward_term_summaries: collections.abc.Mapping[str, reward_types.MetricValue],
        reward_category_summaries: collections.abc.Mapping[str, reward_types.MetricValue],
        parsing_summaries: collections.abc.Mapping[str, reward_types.MetricValue] | None = None,
        parsing_category_summaries: collections.abc.Mapping[str, reward_types.MetricValue] | None = None,
        difficulty_summaries: collections.abc.Mapping[str, reward_types.MetricValue] | None = None,
        failure_ratio: float | None = None,
        failure_count: int | None = None,
        step: int | None = None,
        **kwargs: typing.Any,
    ) -> None:
        """Record a phase-level summary logging event in memory."""
        record: dict[str, object] = {
            "step": step,
            "reward_totals": dict(reward_totals),
            "reward_term_summaries": dict(reward_term_summaries),
            "reward_category_summaries": dict(reward_category_summaries),
        }
        if parsing_summaries is not None:
            record["parsing_summaries"] = dict(parsing_summaries)
        if parsing_category_summaries is not None:
            record["parsing_category_summaries"] = dict(parsing_category_summaries)
        if difficulty_summaries is not None:
            record["difficulty_summaries"] = dict(difficulty_summaries)
        if failure_ratio is not None:
            record["failure_ratio"] = failure_ratio
        if failure_count is not None:
            record["failure_count"] = failure_count
        # store structured data if provided (for test verification)
        # ...snapshot mutable objects to avoid retroactive mutation
        if "term_stats" in kwargs:
            record["term_stats"] = [(name, stats.as_state()) for name, stats in kwargs["term_stats"]]
        if "category_stats" in kwargs:
            record["category_stats"] = [(name, stats.as_state()) for name, stats in kwargs["category_stats"]]
        if "category_term_stats" in kwargs:
            record["category_term_stats"] = {
                cat: [(name, stats.as_state()) for name, stats in term_stats_list]
                for cat, term_stats_list in kwargs["category_term_stats"].items()
            }
        if "bin_stats" in kwargs:
            record["bin_stats"] = [stats.as_state() for stats in kwargs["bin_stats"]]
        if "bin_edges" in kwargs:
            record["bin_edges"] = list(kwargs["bin_edges"])
        if "bin_term_stats" in kwargs:
            record["bin_term_stats"] = {
                name: [stats.as_state() for stats in stats_list]
                for name, stats_list in kwargs["bin_term_stats"].items()
            }
        if "term_order" in kwargs:
            record["term_order"] = list(kwargs["term_order"])
        if "parsing_category_stats" in kwargs:
            # already a list of tuples with dicts, but make a copy
            record["parsing_category_stats"] = [(cat, dict(stats)) for cat, stats in kwargs["parsing_category_stats"]]
        if "reward_total_values" in kwargs:
            record["reward_total_values"] = list(kwargs["reward_total_values"])
        if "difficulty_score_values" in kwargs:
            record["difficulty_score_values"] = list(kwargs["difficulty_score_values"])
        if "bin_reward_values" in kwargs:
            record["bin_reward_values"] = [list(vals) for vals in kwargs["bin_reward_values"]]
        if "secondary_stats" in kwargs:
            record["secondary_stats"] = {name: stats.as_state() for name, stats in kwargs["secondary_stats"].items()}
        if "secondary_corr_stats" in kwargs:
            record["secondary_corr_stats"] = {
                name: stats.as_state() for name, stats in kwargs["secondary_corr_stats"].items()
            }
        self.runs.append(record)

    def log_batch_stats(
        self,
        *,
        batch_mean: float,
        batch_std: float,
        batch_count: int | None = None,
    ) -> None:
        """Record batch-level reward statistics in memory."""
        self.batch_stats.append(
            {
                "batch_count": batch_count,
                "batch_mean": batch_mean,
                "batch_std": batch_std,
            }
        )


class WandBRewardLogger:
    """RewardLogger implementation backed by a W&B run.

    Expects a `wandb.Run`-like object with a `.log(dict)` method.

    Metric Indexing:
        This logger configures WandB to use different x-axes for different metric types:

        - **Per-generation metrics** (reward/total, reward/terms/*, reward/metrics/*, etc.) are indexed
          to `{prefix}/generation_count`, ensuring each logged generation has a unique x-coordinate.

        - **Batch-level metrics** (reward/batch/*) are indexed to `{prefix}/batch_count`, ensuring
          each logged batch has a unique x-coordinate.

        - **Run-level summaries** (reward/run/*) are indexed to `step_metric_key` (default:
          "train/global_step").

        This separation prevents WandB from aggregating values when multiple generations or batches
        are logged within the same trainer step (e.g., with gradient accumulation or multiple
        generations per prompt in GRPO).

    Table Logging:
        This logger supports two types of table logging:

        - **Per-generation details table** (`generation_details`): Controlled by the `log_tables`
          constructor argument. When enabled, logs detailed per-sample data (prompts, completions,
          reward breakdowns) to a W&B table. Useful for debugging but can be expensive for large
          runs.

        - **Run-level summary tables** (`term_summary`, `category_summary`, `bin_summary`, etc.):
          Always logged when the corresponding structured data is provided via kwargs to
          `log_phase_summaries()`. These tables consolidate per-term/per-category/per-bin metrics
          that would otherwise clutter dashboards as individual scalars. When these tables are
          emitted, the corresponding individual scalars are suppressed to avoid duplication. If
          the structured data kwargs are NOT provided, scalars are kept to avoid silent data loss.

    Integration with HuggingFace Trainer:
        The `wandb.define_metric()` calls that configure the step metrics are **deferred until
        the first log() call**, not called in `__init__`. This is intentional: HuggingFace's
        `WandbCallback` calls `wandb.define_metric("*", step_metric="train/global_step")` in
        `on_train_begin`, and our more specific patterns must be defined AFTER that wildcard
        to take precedence.
    """

    # patterns that are replaced by tables when table data is provided
    # these are only suppressed when the corresponding structured data is passed to log_phase_summaries;
    # if no table data is provided, scalars are kept to avoid silent data loss
    # note: category patterns use .+ to allow nested segments (e.g., code_type/original)
    _TERM_SCALAR_PATTERN: typing.ClassVar[re.Pattern[str]] = re.compile(
        r"^reward/run/terms/[^/]+/(mean|std|min|max|count)$"
    )
    _CATEGORY_SCALAR_PATTERN: typing.ClassVar[re.Pattern[str]] = re.compile(
        r"^reward/run/categories/.+/(mean|std|min|max|count)$"
    )
    _PARSING_CATEGORY_SCALAR_PATTERN: typing.ClassVar[re.Pattern[str]] = re.compile(r"^parsing/categories/.+/")
    _DIFFICULTY_BIN_SCALAR_PATTERN: typing.ClassVar[re.Pattern[str]] = re.compile(
        r"^difficulty/run/bin_\d+/(count|reward_(mean|std|min|max)|reward_p(10|50|90))$"
    )
    _DIFFICULTY_BIN_TERM_SCALAR_PATTERN: typing.ClassVar[re.Pattern[str]] = re.compile(
        r"^difficulty/run/bin_\d+/term_[^/]+/reward_(mean|std)$"
    )

    def __init__(
        self,
        wandb_run: object,
        *,
        step: int | None = None,
        log_tables: bool = False,
        table_max_rows: int = 100,
        step_metric_key: str = "train/global_step",
        log_every_n_generations: int = 1,
        histogram_num_bins: int = 50,
    ) -> None:
        """Create a WandB-backed logger.

        Args:
            wandb_run: A `wandb.Run`-like object that supports `.log(...)`.
            step: Optional default step value (not WandB internal step). Used as a fallback for
                run-level summaries and table flushing when step is not explicitly provided. Also
                stored as "internal_step" in table rows.
            log_tables: Whether to log a W&B table with per-generation details.
            table_max_rows: Maximum number of buffered rows before flushing to W&B.
            step_metric_key: Key used as x-axis for run-level summaries. Defaults to
                "train/global_step" to align with HuggingFace Trainer's WandbCallback. Per-generation
                metrics use `{prefix}/generation_count` and batch metrics use `{prefix}/batch_count`.
            log_every_n_generations: Log metrics (scalars and table rows) every N generations (1-indexed).
            histogram_num_bins: Number of bins for histogram visualizations.
        """
        self._wandb_run = wandb_run
        self._key_prefix = ""
        self._step = step
        self._epoch: float | None = None
        self._log_generation_table = log_tables
        self._generation_table_key = "generation_details"
        self._generation_table_max_rows = table_max_rows
        self._generation_table_rows: list[dict[str, object]] = []
        self._step_metric_key = step_metric_key
        self._log_every_n_generations = log_every_n_generations
        self._histogram_num_bins = histogram_num_bins
        # track which prefixes have had wandb.define_metric called (called lazily per-prefix)
        self._generation_metrics_defined_prefixes: set[str] = set()
        self._batch_metrics_defined_prefixes: set[str] = set()
        self._run_metrics_defined_prefixes: set[str] = set()
        if self._log_every_n_generations < 1:
            raise ValueError("log_every_n_generations must be >= 1")
        # NOTE: we do NOT call _define_*_step_metrics() here because HuggingFace's
        # WandbCallback calls `wandb.define_metric("*", step_metric="train/global_step")`
        # in on_train_begin, which would override our definitions if we called them earlier.
        # instead, we defer our define_metric calls to the first log() call, ensuring they
        # happen AFTER any trainer setup and thus take precedence.

    def _is_suppressed_scalar(
        self,
        key: str,
        *,
        has_term_table: bool = False,
        has_category_table: bool = False,
        has_parsing_category_table: bool = False,
        has_bin_table: bool = False,
        has_bin_term_table: bool = False,
    ) -> bool:
        """Check if scalar key should be suppressed (replaced by table).

        Scalars are only suppressed when the corresponding table data is provided, preventing
        silent data loss if a caller omits the structured data kwargs.

        Args:
            key: The scalar metric key to check.
            has_term_table: True if term_stats was provided for term_summary table.
            has_category_table: True if category_stats was provided for category_summary table.
            has_parsing_category_table: True if parsing_category_stats was provided.
            has_bin_table: True if bin_stats/bin_edges were provided for bin_summary table.
            has_bin_term_table: True if bin_term_stats was provided for bin_term_summary table.
        """
        if has_term_table and self._TERM_SCALAR_PATTERN.match(key):
            return True
        if has_category_table and self._CATEGORY_SCALAR_PATTERN.match(key):
            return True
        if has_parsing_category_table and self._PARSING_CATEGORY_SCALAR_PATTERN.match(key):
            return True
        if has_bin_table and self._DIFFICULTY_BIN_SCALAR_PATTERN.match(key):
            return True
        return bool(has_bin_term_table and self._DIFFICULTY_BIN_TERM_SCALAR_PATTERN.match(key))

    def _define_generation_step_metrics(self) -> None:
        """Define WandB step metrics and summaries for per-generation reward logging.

        This configures WandB to use `generation_count` as the x-axis for per-generation
        reward metrics, instead of the default `train/global_step`. This prevents
        WandB from aggregating values when multiple generations are logged within the same
        trainer step (e.g., with gradient accumulation or multiple generations per prompt).

        Also configures summary types for each metric pattern (how they appear in the run
        summary at the end of training).

        This method is called lazily on each log() call, NOT in __init__. This ensures
        our definitions happen AFTER HuggingFace's WandbCallback calls
        `wandb.define_metric("*", step_metric="train/global_step")` in on_train_begin,
        so our more specific patterns take precedence over the wildcard.

        Metrics are defined per-prefix (e.g., "train", "eval") as prefixes are encountered,
        rather than hardcoding specific prefixes upfront. This makes the logger future-proof
        for any prefix naming convention.

        Metrics covered:
        - {prefix}/reward/total, {prefix}/reward/terms/*, {prefix}/reward/metrics/* -> mean
        - {prefix}/categories/* -> mean

        Batch-level metrics (reward/batch/*) and run-level summaries (reward/run/*) are
        handled by separate define methods.
        """
        # derive prefix without trailing slash (e.g., "train/" -> "train", "" -> "")
        prefix = self._key_prefix.rstrip("/")
        if not prefix or prefix in self._generation_metrics_defined_prefixes:
            return  # already defined for this prefix (or no prefix set)
        self._generation_metrics_defined_prefixes.add(prefix)
        # check if wandb is actually initialized (not just a mock object) before defining metrics
        # wandb.define_metric requires an active run; when using mock runs in tests, this check
        # prevents the call from failing
        if wandb.run is None:
            return
        # per-sample metric suffixes with summary type (all use "mean" for run-level average)
        metric_suffixes: list[tuple[str, str]] = [
            ("reward/total", "mean"),
            ("reward/terms/*", "mean"),
            ("reward/metrics/*", "mean"),
            ("reward/raw_terms/*", "mean"),
        ]
        step_metric_key = f"{prefix}/generation_count"
        for suffix, summary in metric_suffixes:
            pattern = f"{prefix}/{suffix}"
            wandb.define_metric(pattern, step_metric=step_metric_key, summary=summary)

    def _define_batch_step_metrics(self) -> None:
        """Define WandB step metrics and summaries for batch-level reward logging.

        This configures WandB to use `batch_count` as the x-axis for batch-level metrics,
        instead of the default `train/global_step`. This prevents WandB from aggregating
        values when multiple batches are logged within the same trainer step (e.g., with
        gradient accumulation where multiple batches are processed per optimizer step).

        Also configures summary types for each metric pattern.

        This method is called lazily on each log_batch_stats() call, NOT in __init__.
        This ensures our definitions happen AFTER HuggingFace's WandbCallback calls
        `wandb.define_metric("*", step_metric="train/global_step")` in on_train_begin,
        so our more specific patterns take precedence over the wildcard.

        Metrics are defined per-prefix as prefixes are encountered.

        Metrics covered:
        - {prefix}/reward/batch/* -> mean
        """
        prefix = self._key_prefix.rstrip("/")
        if not prefix or prefix in self._batch_metrics_defined_prefixes:
            return  # already defined for this prefix (or no prefix set)
        self._batch_metrics_defined_prefixes.add(prefix)
        if wandb.run is None:
            return
        step_metric_key = f"{prefix}/batch_count"
        wandb.define_metric(f"{prefix}/reward/batch/*", step_metric=step_metric_key, summary="mean")

    def _define_run_step_metrics(self) -> None:
        """Define WandB step metrics and summaries for run-level reward logging.

        This configures summary types for run-level metrics (reward/run/*, failures/*).
        These metrics use the default step_metric_key (train/global_step) as their x-axis
        since they are logged once per training phase.

        This method is called lazily on each log_phase_summaries() call, NOT in __init__.

        Metrics are defined per-prefix as prefixes are encountered.

        Metrics covered:
        - {prefix}/reward/run/* -> last (already summaries, keep final value)
        - {prefix}/parsing/* -> last (already summaries, keep final value)
        - {prefix}/failures/* -> last (per-phase failure stats, keep final value)
        """
        prefix = self._key_prefix.rstrip("/")
        if not prefix or prefix in self._run_metrics_defined_prefixes:
            return  # already defined for this prefix (or no prefix set)
        self._run_metrics_defined_prefixes.add(prefix)
        if wandb.run is None:
            return
        # run-level metric suffixes (use "last" since they're already aggregated summaries)
        metric_suffixes = ["reward/run/*", "parsing/*", "failures/*", "difficulty/run/*"]
        for suffix in metric_suffixes:
            pattern = f"{prefix}/{suffix}"
            wandb.define_metric(pattern, step_metric=self._step_metric_key, summary="last")

    def _prefix_key(
        self,
        key: str,
    ) -> str:
        """Apply `key_prefix` to a key (idempotent if the prefix is already present)."""
        if not self._key_prefix:
            return key
        if key.startswith(self._key_prefix):
            return key
        return f"{self._key_prefix}{key}"

    def _prefix_payload(
        self,
        payload: collections.abc.Mapping[str, object],
    ) -> dict[str, object]:
        """Prefix all keys in a payload dict for W&B emission."""
        return {self._prefix_key(key): value for key, value in payload.items()}

    def should_log_sample(
        self,
        generation_count: int,
    ) -> bool:
        """Return True if sample metrics should be logged for this generation_count.

        This controls both scalar emission and table row addition. The manager uses this
        to decide whether to call log_sample() for a given generation.
        """
        return generation_count % self._log_every_n_generations == 0

    def log_sample(
        self,
        sample_id: str,
        *,
        generation_count: int | None = None,
        batch_count: int | None = None,
        local_batch_idx: int | None = None,
        completion_idx: int | None = None,
        rank: int | None = None,
        total: float | None,
        terms: collections.abc.Mapping[str, float] | None = None,
        metrics: collections.abc.Mapping[str, reward_types.MetricValue] | None = None,
        raw_terms: collections.abc.Mapping[str, float] | None = None,
        step: int | None = None,
        prompt: str | None = None,
        expected_output: str | None = None,
        model_output: str | None = None,
        reasoning: str | None = None,
        final_answer: str | None = None,
        categories: collections.abc.Sequence[str] | None = None,
        tags: collections.abc.Sequence[str] | None = None,
        difficulty_source: str | None = None,
        difficulty_score: float | None = None,
        difficulty_bin: int | None = None,
        difficulty_raw_primary: float | None = None,
        difficulty_secondary_json: str | None = None,
        predict_type: str | None = None,
        code_type: str | None = None,
        has_code_override: bool | None = None,
        pregenerated_output: str | None = None,
        **kwargs: typing.Any,
    ) -> None:
        """Log a per-sample reward payload to W&B.

        This method always emits scalars and adds a table row (if table logging is enabled).
        Frequency gating is the caller's responsibility; they use should_log_sample() to
        determine whether to call this method.

        Args:
            sample_id: Unique identifier for the sample.
            generation_count: 1-indexed count of generations seen so far. Used as the x-axis
                value for per-generation metrics (not for gating, that is caller's responsibility).
            batch_count: Global batch counter (1-indexed).
            local_batch_idx: Index of this sample within the current batch (0-indexed).
            completion_idx: Global completion index within samples sharing the same identifier
                across all ranks in this batch (0-indexed). Globally unique within each batch.
            rank: Global rank of the process logging this sample.
            total: Total reward value for the sample.
            terms: Per-term weighted reward values.
            metrics: Additional metrics emitted by reward terms.
            raw_terms: Pre-clipping, pre-weighting term values.
            step: Optional step value (overrides logger's default step).
            prompt: The input prompt.
            expected_output: Ground truth output.
            model_output: Model-generated output.
            reasoning: Parsed reasoning text.
            final_answer: Parsed final answer text.
            categories: Sample categories for grouping.
            tags: Additional sample tags.
            difficulty_source: Name of the primary difficulty source.
            difficulty_score: Normalized difficulty score.
            difficulty_bin: Bin index for this sample's difficulty.
            difficulty_raw_primary: Raw value of the primary difficulty source.
            difficulty_secondary_json: JSON string of secondary difficulty raw values.
            predict_type: Sample predict type.
            code_type: Sample code type.
            has_code_override: Whether the sample has a code override.
            pregenerated_output: Pregenerated pseudolabel output from a prior run (if any).
            **kwargs: Absorbed for forward compatibility.
        """
        del kwargs  # absorb any future additions for forward compatibility
        # lazily define step metrics on first log_sample call (after trainer setup)
        self._define_generation_step_metrics()
        # note: frequency gating is the caller's responsibility (typically RewardManager);
        # when log_sample is called, we always emit scalars and add a table row (if enabled).
        # the should_log_sample() method is exposed as a query helper for the caller to use
        # before deciding to call log_sample().
        terms = terms or {}
        metrics = metrics or {}
        payload_step = self._step if step is None else step
        # build and emit scalar payload
        payload: dict[str, reward_types.MetricValue] = {}
        if total is not None:
            payload["reward/total"] = total
        payload.update(terms)
        # note: difficulty/* keys are already filtered out by RewardManager._scope_reward_sample_fields
        payload.update(metrics)
        if raw_terms is not None:
            for term_name, raw_value in raw_terms.items():
                payload[f"reward/raw_terms/{term_name}"] = raw_value
        if payload:  # avoid empty wandb.log() calls
            prefixed = self._prefix_payload(payload)
            # always log generation_count for per-generation metrics; this provides a unique x-axis
            # value for each logged sample, avoiding aggregation issues when multiple samples
            # are logged within the same trainer step (e.g., with gradient accumulation).
            # note: step is intentionally NOT included here; per-generation metrics use
            # generation_count as the x-axis, not step. step is only used for run-level summaries.
            if generation_count is not None:
                prefixed[self._prefix_key("generation_count")] = generation_count
            if self._epoch is not None:
                prefixed[self._prefix_key("epoch")] = self._epoch
            self._wandb_run.log(prefixed)  # type: ignore[reportUnknownMemberType]
        # build and buffer table row (if table logging is enabled)
        if self._log_generation_table:
            prefixed_terms = self._prefix_payload(dict(terms))
            prefixed_metrics = self._prefix_payload(dict(metrics))
            raw_terms_payload: dict[str, float] | None = None
            if raw_terms is not None:
                raw_terms_payload = {
                    f"reward/raw_terms/{term_name}": raw_value for term_name, raw_value in raw_terms.items()
                }
            row: dict[str, object] = {
                "sample_id": sample_id,
                "generation_count": generation_count,
                "batch_count": batch_count,
                "local_batch_idx": local_batch_idx,
                "completion_idx": completion_idx,
                "rank": rank,
                "step": payload_step,
                "prompt": prompt,
                "expected_output": expected_output,
                "model_output": model_output,
                "reasoning": reasoning,
                "final_answer": final_answer,
                "reward_total": total,
                "reward_terms_json": json.dumps(prefixed_terms, sort_keys=True),
                "reward_terms_raw_json": json.dumps(self._prefix_payload(raw_terms_payload), sort_keys=True)
                if raw_terms_payload is not None
                else None,
                "reward_metrics_json": json.dumps(prefixed_metrics, sort_keys=True),
                "categories_json": json.dumps(list(categories), sort_keys=True) if categories else None,
                "tags_json": json.dumps(list(tags), sort_keys=True) if tags else None,
                "difficulty_source": difficulty_source,
                "difficulty_score": difficulty_score,
                "difficulty_bin": difficulty_bin,
                "difficulty_raw_primary": difficulty_raw_primary,
                "difficulty_secondary_json": difficulty_secondary_json,
                "predict_type": predict_type,
                "code_type": code_type,
                "has_code_override": has_code_override,
                "pregenerated_output": pregenerated_output,
            }
            self._generation_table_rows.append(row)
            # flush table when buffer reaches max size
            if len(self._generation_table_rows) >= self._generation_table_max_rows:
                self.flush_generation_table()

    def log_batch_stats(
        self,
        *,
        batch_mean: float,
        batch_std: float,
        batch_count: int | None = None,
    ) -> None:
        """Log batch-level reward statistics to W&B.

        Batch-level metrics are indexed to `{prefix}/batch_count` to avoid aggregation issues
        when multiple batches (e.g., for gradient accumulation) are processed at the same step.

        Args:
            batch_mean: Mean reward for the current batch.
            batch_std: Std dev of rewards for the current batch.
            batch_count: Monotonic batch counter (1-indexed) used as x-axis for batch metrics.
        """
        self._define_batch_step_metrics()  # deferred initialization
        payload: dict[str, float] = {
            "reward/batch/mean": batch_mean,
            "reward/batch/std": batch_std,
        }
        prefixed: dict[str, object] = self._prefix_payload(payload)
        if batch_count is not None:
            prefixed[self._prefix_key("batch_count")] = batch_count
        if self._epoch is not None:
            prefixed[self._prefix_key("epoch")] = self._epoch
        self._wandb_run.log(prefixed)  # type: ignore[reportUnknownMemberType]

    def log_phase_summaries(
        self,
        *,
        reward_totals: collections.abc.Mapping[str, reward_types.MetricValue],
        reward_term_summaries: collections.abc.Mapping[str, reward_types.MetricValue],
        reward_category_summaries: collections.abc.Mapping[str, reward_types.MetricValue],
        parsing_summaries: collections.abc.Mapping[str, reward_types.MetricValue] | None = None,
        parsing_category_summaries: collections.abc.Mapping[str, reward_types.MetricValue] | None = None,
        difficulty_summaries: collections.abc.Mapping[str, reward_types.MetricValue] | None = None,
        failure_ratio: float | None = None,
        failure_count: int | None = None,
        step: int | None = None,
        **kwargs: typing.Any,
    ) -> None:
        """Log phase-level summary payload to W&B (e.g., at end of train/eval phase)."""
        self._define_run_step_metrics()  # deferred initialization
        payload_step = self._step if step is None else step
        # determine which tables will actually be emitted (have non-empty data)
        # ...scalars are only suppressed when the corresponding table will have rows, preventing silent data loss
        has_term_table = _has_nonempty_stats(kwargs.get("term_stats"))
        has_category_table = _has_nonempty_stats(kwargs.get("category_stats"))
        has_parsing_category_table = bool(kwargs.get("parsing_category_stats"))  # checked by truthiness (dicts)
        has_bin_table = (
            "bin_stats" in kwargs and "bin_edges" in kwargs and _has_nonempty_bin_stats(kwargs.get("bin_stats"))
        )
        has_bin_term_table = (
            "bin_term_stats" in kwargs
            and "bin_edges" in kwargs
            and "term_order" in kwargs
            and _has_nonempty_bin_term_stats(kwargs.get("bin_term_stats"))
        )
        # build payload, filtering out scalars only when corresponding table is provided
        # this ensures no data is silently lost if table data is missing
        payload: dict[str, reward_types.MetricValue] = {}
        # reward_totals: always keep (not suppressed)
        payload.update(reward_totals)
        # reward_term_summaries: filter only if term table will be emitted
        for key, value in reward_term_summaries.items():
            if not self._is_suppressed_scalar(key, has_term_table=has_term_table):
                payload[key] = value
        # reward_category_summaries: filter only if category table will be emitted
        for key, value in reward_category_summaries.items():
            if not self._is_suppressed_scalar(key, has_category_table=has_category_table):
                payload[key] = value
        # parsing_summaries (GLOBAL): always keep unchanged; not in suppression scope
        if parsing_summaries:
            payload.update(parsing_summaries)
        # parsing_category_summaries (PER-CATEGORY): filter only if parsing table will be emitted
        if parsing_category_summaries:
            for key, value in parsing_category_summaries.items():
                if not self._is_suppressed_scalar(key, has_parsing_category_table=has_parsing_category_table):
                    payload[key] = value
        # difficulty_summaries: filter bin stats only if bin tables will be emitted
        if difficulty_summaries:
            for key, value in difficulty_summaries.items():
                full_key = f"difficulty/run/{key}"
                if not self._is_suppressed_scalar(
                    full_key, has_bin_table=has_bin_table, has_bin_term_table=has_bin_term_table
                ):
                    payload[full_key] = value
        if failure_ratio is not None:
            payload["failures/failure_ratio"] = failure_ratio
        if failure_count is not None:
            payload["failures/failure_count"] = failure_count
        prefixed: dict[str, object] = self._prefix_payload(payload)
        # emit tables (skip empty tables to avoid W&B schema churn)
        if "term_stats" in kwargs and kwargs["term_stats"]:
            table = _build_stats_table_from_running_stats(kwargs["term_stats"], "term")
            if len(table.data) > 0:  # type: ignore[reportUnknownMemberType]
                prefixed[self._prefix_key("reward/run/term_summary")] = table
        if "category_stats" in kwargs and kwargs["category_stats"]:
            table = _build_stats_table_from_running_stats(kwargs["category_stats"], "category")
            if len(table.data) > 0:  # type: ignore[reportUnknownMemberType]
                prefixed[self._prefix_key("reward/run/category_summary")] = table
        if _has_nonempty_category_term_stats(kwargs.get("category_term_stats")):
            table = _build_category_term_stats_table(kwargs["category_term_stats"])
            if len(table.data) > 0:  # type: ignore[reportUnknownMemberType]
                prefixed[self._prefix_key("reward/run/category_term_summary")] = table
        if "bin_stats" in kwargs and "bin_edges" in kwargs:
            table = _build_difficulty_bin_table(
                kwargs["bin_stats"],
                kwargs["bin_edges"],
                kwargs.get("bin_reward_values"),  # for quantiles
            )
            if len(table.data) > 0:  # type: ignore[reportUnknownMemberType]
                prefixed[self._prefix_key("difficulty/run/bin_summary")] = table
        if "bin_term_stats" in kwargs and "bin_edges" in kwargs and "term_order" in kwargs:
            table = _build_difficulty_bin_term_table(
                kwargs["bin_term_stats"],
                kwargs["bin_edges"],
                kwargs["term_order"],
            )
            if len(table.data) > 0:  # type: ignore[reportUnknownMemberType]
                prefixed[self._prefix_key("difficulty/run/bin_term_summary")] = table
        if "parsing_category_stats" in kwargs and kwargs["parsing_category_stats"]:
            table = _build_parsing_category_table(kwargs["parsing_category_stats"])
            if len(table.data) > 0:  # type: ignore[reportUnknownMemberType]
                prefixed[self._prefix_key("parsing/run/category_summary")] = table
        # secondary difficulty sources table
        secondary_stats = kwargs.get("secondary_stats", {})
        secondary_corr_stats = kwargs.get("secondary_corr_stats", {})
        has_secondary_data = any(rs.count > 0 for rs in secondary_stats.values()) or any(
            cs.count > 1 for cs in secondary_corr_stats.values()
        )
        if has_secondary_data:
            secondary_table = _build_secondary_source_table(secondary_stats, secondary_corr_stats)
            prefixed[self._prefix_key("difficulty/run/secondary_source_summary")] = secondary_table
        # emit histograms (always)
        if "reward_total_values" in kwargs and kwargs["reward_total_values"]:
            hist = wandb.Histogram(kwargs["reward_total_values"], num_bins=self._histogram_num_bins)
            prefixed[self._prefix_key("reward/run/total_histogram")] = hist
        if "difficulty_score_values" in kwargs and kwargs["difficulty_score_values"]:
            hist = wandb.Histogram(kwargs["difficulty_score_values"], num_bins=self._histogram_num_bins)
            prefixed[self._prefix_key("difficulty/run/score_histogram")] = hist
        # per-bin difficulty histograms
        bin_reward_values = kwargs.get("bin_reward_values")
        if bin_reward_values:
            for bin_idx, bin_vals in enumerate(bin_reward_values):
                if bin_vals:
                    hist = wandb.Histogram(bin_vals, num_bins=self._histogram_num_bins)
                    prefixed[self._prefix_key(f"difficulty/run/bin_{bin_idx}_histogram")] = hist
        if payload_step is not None:
            prefixed[self._step_metric_key] = payload_step  # global_step not prefixed
        if self._epoch is not None:
            prefixed[self._prefix_key("epoch")] = self._epoch
        self._wandb_run.log(prefixed)  # type: ignore[reportUnknownMemberType]
        if self._log_generation_table:
            self.flush_generation_table()

    def set_step(
        self,
        step: int | None,
    ) -> None:
        """Set a default step value for subsequent logs (logged under `step_metric_key`)."""
        self._step = step

    def set_epoch(
        self,
        epoch: float | None,
    ) -> None:
        """Set a default epoch value for subsequent logs."""
        self._epoch = epoch

    def set_key_prefix(
        self,
        key_prefix: str,
    ) -> None:
        """Set the key prefix for subsequent logs.

        Args:
            key_prefix: New prefix to apply to all emitted W&B keys. Will be normalized
                to ensure consistent trailing-slash formatting (or empty string).
        """
        self._key_prefix = parsing_utils.normalize_path_prefix(key_prefix)

    def get_key_prefix(self) -> str:
        """Get the current key prefix."""
        return self._key_prefix

    def flush_generation_table(self) -> None:
        """Flush buffered generation table rows to W&B (no-op if table logging is disabled)."""
        if not self._log_generation_table or not self._generation_table_rows:
            return
        columns = [
            "sample_id",
            "generation_count",
            "batch_count",
            "local_batch_idx",
            "completion_idx",
            "rank",
            "step",
            "prompt",
            "expected_output",
            "model_output",
            "reasoning",
            "final_answer",
            "reward_total",
            "reward_terms_json",
            "reward_terms_raw_json",
            "reward_metrics_json",
            "categories_json",
            "tags_json",
            "difficulty_source",
            "difficulty_score",
            "difficulty_bin",
            "difficulty_raw_primary",
            "difficulty_secondary_json",
            "predict_type",
            "code_type",
            "has_code_override",
            "pregenerated_output",
        ]
        table = wandb.Table(columns=columns)
        table_obj = typing.cast("typing.Any", table)
        for row in self._generation_table_rows:
            row_data: list[object] = [
                row["sample_id"],
                row["generation_count"],
                row["batch_count"],
                row["local_batch_idx"],
                row["completion_idx"],
                row["rank"],
                row["step"],
                row["prompt"],
                row["expected_output"],
                row["model_output"],
                row["reasoning"],
                row["final_answer"],
                row["reward_total"],
                row["reward_terms_json"],
                row["reward_terms_raw_json"],
                row["reward_metrics_json"],
                row["categories_json"],
                row["tags_json"],
                row["difficulty_source"],
                row["difficulty_score"],
                row["difficulty_bin"],
                row["difficulty_raw_primary"],
                row["difficulty_secondary_json"],
                row["predict_type"],
                row["code_type"],
                row["has_code_override"],
                row["pregenerated_output"],
            ]
            table_obj.add_data(*row_data)
        self._wandb_run.log({self._prefix_key(self._generation_table_key): table})  # type: ignore[reportUnknownMemberType]
        self._generation_table_rows.clear()


def make_wandb_reward_logger(
    wandb_run: object,
    logging_config: reward_configs.LoggingConfig,
    *,
    step: int | None = None,
) -> WandBRewardLogger:
    """Create a `WandBRewardLogger` from a `LoggingConfig`.

    Args:
        wandb_run: A `wandb.Run`-like object that supports `.log(...)`.
        logging_config: Reward logging configuration.
        step: Optional step value logged under `step_metric_key` (not WandB internal step).

    Returns:
        A WandB-backed reward logger with total logged under "reward/total".
    """
    return WandBRewardLogger(
        wandb_run,
        step=step,
        log_tables=logging_config.log_tables,
        table_max_rows=logging_config.table_max_rows,
        step_metric_key=logging_config.step_metric_key,
        log_every_n_generations=logging_config.log_every_n_generations,
        histogram_num_bins=logging_config.histogram_num_bins,
    )


class DiskRewardLogger:
    """RewardLogger implementation that writes generation records to an LMDB dataset on disk.

    Each ``log_sample()`` call writes one record to LMDB keyed as
    ``{key_prefix}{sample_id}/{generation_count}``. This enables later re-import of
    RL-generated completions as SFT training examples.

    Uses ``JSON_ZSTD`` serialization for human-inspectable, compressed storage. Thread-safe via
    an internal lock around LMDB writes.
    """

    _METADATA_FLUSH_INTERVAL = 500  # flush internal metadata every N writes as crash safety belt

    def __init__(
        self,
        output_path: pathlib.Path,
        log_every_n_generations: int = 1,
    ) -> None:
        """Create a disk-backed reward logger.

        Args:
            output_path: Directory path for the LMDB dataset.
            log_every_n_generations: Record samples every N generations (default 1 = log all).

        Raises:
            ValueError: If ``output_path`` already exists and is non-empty.
        """
        if log_every_n_generations < 1:
            raise ValueError("log_every_n_generations must be >= 1")
        output_path = pathlib.Path(output_path)
        if output_path.exists() and any(output_path.iterdir()):
            raise ValueError(
                f"output_path '{output_path}' already exists and is non-empty; "
                "use a fresh directory to avoid mixing runs"
            )
        self._log_every_n_generations = log_every_n_generations
        self._key_prefix: str = ""
        self._step: int | None = None
        self._epoch: float | None = None
        self._write_count = 0
        self._lock = threading.Lock()
        self._writer = pyine.data.utils.lmdb_io.LMDBWriter(
            path=output_path,
            serialization_config=pyine.data.utils.lmdb_io.SerializationConfig(
                method=pyine.data.utils.lmdb_io.SerializationMethod.JSON_ZSTD,
            ),
        )
        self._writer.write_metadata({"record_type": "reward"})

    def should_log_sample(
        self,
        generation_count: int,
    ) -> bool:
        """Return True if sample metrics should be logged for this generation_count."""
        return generation_count % self._log_every_n_generations == 0

    def log_sample(
        self,
        sample_id: str,
        *,
        generation_count: int | None = None,
        batch_count: int | None = None,
        local_batch_idx: int | None = None,
        completion_idx: int | None = None,
        rank: int | None = None,
        total: float | None,
        terms: collections.abc.Mapping[str, float] | None = None,
        metrics: collections.abc.Mapping[str, reward_types.MetricValue] | None = None,
        raw_terms: collections.abc.Mapping[str, float] | None = None,
        step: int | None = None,
        prompt: str | None = None,
        expected_output: str | None = None,
        model_output: str | None = None,
        reasoning: str | None = None,
        final_answer: str | None = None,
        categories: collections.abc.Sequence[str] | None = None,
        tags: collections.abc.Sequence[str] | None = None,
        difficulty_source: str | None = None,
        difficulty_score: float | None = None,
        difficulty_bin: int | None = None,
        difficulty_raw_primary: float | None = None,
        difficulty_secondary_json: str | None = None,
        predict_type: str | None = None,
        code_type: str | None = None,
        has_code_override: bool | None = None,
        pregenerated_output: str | None = None,
        **kwargs: typing.Any,
    ) -> None:
        """Write a per-sample reward record to LMDB.

        Args:
            sample_id: Unique identifier for the sample.
            generation_count: Generation index for deduplication.
            batch_count: Batch number within the training run.
            local_batch_idx: Index within the local batch.
            completion_idx: Completion index for multi-completion samples.
            rank: DDP rank that produced this record.
            total: Total reward score.
            terms: Scoped reward term values (e.g. ``reward/terms/soft_match``).
            metrics: Scoped reward metric values (e.g. ``reward/metrics/soft_match/is_match``).
            raw_terms: Raw (unscoped) reward term values.
            step: Training step number.
            prompt: Full prompt text sent to the model.
            expected_output: Ground-truth expected output.
            model_output: Model-generated completion text.
            reasoning: Extracted reasoning chain, if any.
            final_answer: Extracted final answer from model output.
            categories: Problem category labels.
            tags: Metadata tags for the sample.
            difficulty_source: Source of difficulty annotation.
            difficulty_score: Numeric difficulty score.
            difficulty_bin: Discretised difficulty bin.
            difficulty_raw_primary: Raw primary difficulty metric.
            difficulty_secondary_json: JSON-encoded secondary difficulty metrics.
            predict_type: Prediction type (e.g. ``"program_output"``).
            code_type: Code augmentation type (e.g. ``"original"``, ``"hinted"``).
            has_code_override: Whether the code was overridden by augmentation.
            pregenerated_output: Pre-generated model output, if any.
            **kwargs: Additional fields (ignored).
        """
        if not self._key_prefix:
            logger.debug("DiskRewardLogger.log_sample() called with empty key_prefix")
        gen_count_str = str(generation_count) if generation_count is not None else "none"
        lmdb_key = f"{self._key_prefix}{sample_id}/{gen_count_str}"
        shared = pyine.data.utils.generation_record.build_shared_record_fields(
            sample_id=sample_id,
            model_output=model_output,
            prompt=prompt,
            expected_output=expected_output,
            reasoning=reasoning,
            final_answer=final_answer,
            predict_type=predict_type,
            code_type=code_type,
            has_code_override=has_code_override,
            pregenerated_output=pregenerated_output,
            tags=tags,
            categories=categories,
            key_prefix=self._key_prefix,
        )
        record: dict[str, typing.Any] = {
            **shared,
            "reward_total": total,
            "reward_terms": dict(terms) if terms is not None else None,
            "reward_metrics": dict(metrics) if metrics is not None else None,
            "reward_terms_raw": dict(raw_terms) if raw_terms is not None else None,
            "step": step,
            "batch_count": batch_count,
            "local_batch_idx": local_batch_idx,
            "completion_idx": completion_idx,
            "rank": rank,
            "difficulty_source": difficulty_source,
            "difficulty_score": difficulty_score,
            "difficulty_bin": difficulty_bin,
            "difficulty_raw_primary": difficulty_raw_primary,
            "difficulty_secondary_json": difficulty_secondary_json,
        }
        with self._lock:
            self._writer.put(lmdb_key, record)
            self._write_count += 1
            if self._write_count % self._METADATA_FLUSH_INTERVAL == 0:
                self._writer._write_internal_metadata()  # type: ignore[reportPrivateUsage]

    def log_phase_summaries(
        self,
        *,
        reward_totals: collections.abc.Mapping[str, reward_types.MetricValue],
        reward_term_summaries: collections.abc.Mapping[str, reward_types.MetricValue],
        reward_category_summaries: collections.abc.Mapping[str, reward_types.MetricValue],
        parsing_summaries: collections.abc.Mapping[str, reward_types.MetricValue] | None = None,
        parsing_category_summaries: collections.abc.Mapping[str, reward_types.MetricValue] | None = None,
        difficulty_summaries: collections.abc.Mapping[str, reward_types.MetricValue] | None = None,
        failure_ratio: float | None = None,
        failure_count: int | None = None,
        step: int | None = None,
        **kwargs: typing.Any,
    ) -> None:
        """No-op for disk logger (monitoring-only concern)."""

    def log_batch_stats(
        self,
        *,
        batch_mean: float,
        batch_std: float,
        batch_count: int | None = None,
    ) -> None:
        """No-op for disk logger (monitoring-only concern)."""

    def set_step(self, step: int | None) -> None:
        """Set a default step value for subsequent logs."""
        self._step = step

    def set_epoch(self, epoch: float | None) -> None:
        """Set a default epoch value for subsequent logs."""
        self._epoch = epoch

    def set_key_prefix(self, key_prefix: str) -> None:
        """Set the key prefix for subsequent logs."""
        self._key_prefix = parsing_utils.normalize_path_prefix(key_prefix)

    def get_key_prefix(self) -> str:
        """Get the current key prefix."""
        return self._key_prefix

    def close(self) -> None:
        """Close the underlying LMDB writer, flushing metadata."""
        self._writer.close()


def make_disk_reward_logger(
    export_config: reward_configs.GenerationExportConfig,
    *,
    rank: int | None = None,
) -> DiskRewardLogger:
    """Create a ``DiskRewardLogger`` from a ``GenerationExportConfig``.

    Args:
        export_config: Export configuration specifying output path and logging frequency.
        rank: Optional global rank for per-rank output paths. Required when
            ``export_config.export_all_ranks=True``; appends ``rank_{rank}`` to the output path.

    Returns:
        A disk-backed reward logger.

    Raises:
        ValueError: If ``export_all_ranks=True`` but no rank is provided.
    """
    if export_config.export_all_ranks and rank is None:
        raise ValueError(
            "export_all_ranks=True requires an explicit rank to construct rank-specific "
            "output paths; resolve an authoritative global rank (via has_explicit_global_rank() "
            "then get_global_rank(), or _resolve_export_rank() in the training harness) "
            "before calling make_disk_reward_logger()"
        )
    output_path = export_config.output_path
    if rank is not None:
        output_path = output_path / f"rank_{rank}"
    return DiskRewardLogger(
        output_path=output_path,
        log_every_n_generations=export_config.log_every_n_generations,
    )


class CompositeRewardLogger:
    """Wraps multiple ``RewardLogger`` instances into one, fanning out calls to each.

    ``should_log_sample()`` returns True if **any** inner logger returns True (union
    semantics). ``log_sample()`` individually gates each inner logger via its own
    ``should_log_sample()`` check, preserving per-logger frequency settings.
    """

    def __init__(
        self,
        loggers: collections.abc.Sequence[reward_types.RewardLogger],
    ) -> None:
        """Create a composite reward logger.

        Args:
            loggers: Inner loggers to fan out to.
        """
        if not loggers:
            raise ValueError("CompositeRewardLogger requires at least one inner logger")
        self._loggers = list(loggers)
        self._key_prefix: str = ""

    def should_log_sample(
        self,
        generation_count: int,
    ) -> bool:
        """Return True if any inner logger wants to log this generation."""
        return any(inner.should_log_sample(generation_count) for inner in self._loggers)

    def log_sample(
        self,
        sample_id: str,
        *,
        generation_count: int | None = None,
        **kwargs: typing.Any,
    ) -> None:
        """Fan out to each inner logger that passes its own should_log_sample gate."""
        for inner in self._loggers:
            if generation_count is None or inner.should_log_sample(generation_count):
                inner.log_sample(sample_id, generation_count=generation_count, **kwargs)

    def log_phase_summaries(
        self,
        **kwargs: typing.Any,
    ) -> None:
        """Fan out phase summaries to all inner loggers."""
        for inner in self._loggers:
            inner.log_phase_summaries(**kwargs)

    def log_batch_stats(
        self,
        **kwargs: typing.Any,
    ) -> None:
        """Fan out batch stats to all inner loggers."""
        for inner in self._loggers:
            inner.log_batch_stats(**kwargs)

    def set_step(self, step: int | None) -> None:
        """Set step on all inner loggers."""
        for inner in self._loggers:
            inner.set_step(step)

    def set_epoch(self, epoch: float | None) -> None:
        """Set epoch on all inner loggers."""
        for inner in self._loggers:
            inner.set_epoch(epoch)

    def set_key_prefix(self, key_prefix: str) -> None:
        """Set key prefix on all inner loggers."""
        self._key_prefix = parsing_utils.normalize_path_prefix(key_prefix)
        for inner in self._loggers:
            inner.set_key_prefix(key_prefix)

    def get_key_prefix(self) -> str:
        """Get the current key prefix."""
        return self._key_prefix

    def close(self) -> None:
        """Close any inner loggers that have a close method."""
        for inner in self._loggers:
            if hasattr(inner, "close"):
                inner.close()  # type: ignore[union-attr]
