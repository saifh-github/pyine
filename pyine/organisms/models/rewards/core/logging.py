"""Reward logging helpers.

The reward system exposes a small `RewardLogger` protocol. The logger controls logging frequency
via generation_count gating; the manager handles key scoping and orchestration.

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
    - `{prefix}/parsing/*`: parsing-related metrics (e.g., reasoning_length_tokens, has_answer).

**Batch-level metrics** (indexed to `{prefix}/batch_count`):
    These metrics are logged once per compute_batch call (when enabled) and use batch_count as the x-axis.
    This ensures each logged batch has a unique x-coordinate, avoiding WandB aggregation issues
    when multiple batches are processed within the same trainer step (e.g., gradient accumulation).

    Note: Batch-level metrics are only emitted when `LoggingConfig.log_batch_stats=True`.

    - `{prefix}/reward/batch/mean`: mean reward across the batch;
    - `{prefix}/reward/batch/std`: standard deviation of rewards in the batch;
    - `{prefix}/reward/batch/mean_rolling/*`: rolling statistics of batch means;
    - `{prefix}/reward/batch/std_rolling/*`: rolling statistics of batch std devs.

**Run-level summaries** (indexed to `step_metric_key`, default `train/global_step`):
    These metrics are logged when flush_stats() is called (e.g., at phase transitions).

    - `{prefix}/reward/run/mean`, `{prefix}/reward/run/std`: accumulated reward statistics;
    - `{prefix}/reward/run/count`: number of generations processed;
    - `{prefix}/failures/failure_ratio`: ratio of failed generations in the phase;
    - `{prefix}/failures/failure_count`: count of failed generations in the phase.

Where `{prefix}` is typically "train" or "eval" depending on the training phase.
"""

import collections.abc
import json
import typing

import wandb

import pyine.organisms.models.rewards.core.configs as reward_configs
import pyine.organisms.models.rewards.core.types as reward_types
import pyine.utils.parsing as parsing_utils


class InMemoryRewardLogger:
    """RewardLogger implementation for tests and debugging.

    Stores structured events in memory instead of writing to an external backend.
    Supports optional frequency gating for consistency with WandBRewardLogger.
    """

    def __init__(
        self,
        *,
        scalar_log_every_n_generations: int = 1,
        log_tables: bool = False,
        table_row_every_n_generations: int = 1,
    ) -> None:
        """Create an in-memory logger with empty buffers.

        Args:
            scalar_log_every_n_generations: Record samples every N generations (default 1 = log all).
            log_tables: Whether to record table row events (default False).
            table_row_every_n_generations: Record table rows every N generations (default 1 = log all).
        """
        self.samples: list[dict[str, object]] = []
        self.table_rows: list[dict[str, object]] = []
        self.runs: list[dict[str, object]] = []
        self.batch_stats: list[dict[str, object]] = []
        self.difficulty_stats: list[dict[str, object]] = []
        self._step: int | None = None
        self._epoch: float | None = None
        self._key_prefix: str = ""
        self._scalar_log_every_n_generations = scalar_log_every_n_generations
        self._log_generation_table = log_tables
        self._generation_table_row_every_n = table_row_every_n_generations
        if self._scalar_log_every_n_generations < 1:
            raise ValueError("scalar_log_every_n_generations must be >= 1")
        if self._generation_table_row_every_n < 1:
            raise ValueError("table_row_every_n_generations must be >= 1")

    def should_log_sample_scalars(
        self,
        generation_count: int,
    ) -> bool:
        """Return True if sample scalars should be logged for this generation_count."""
        return generation_count % self._scalar_log_every_n_generations == 0

    def should_log_sample_table_row(
        self,
        generation_count: int,
    ) -> bool:
        """Return True if a sample table row should be logged for this generation_count."""
        return self._log_generation_table and generation_count % self._generation_table_row_every_n == 0

    def log_sample(
        self,
        sample_id: str,
        *,
        generation_count: int | None = None,
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
        generation_idx: int | None = None,
        **kwargs: typing.Any,
    ) -> None:
        """Record a per-sample logging event in memory.

        Args:
            sample_id: Unique identifier for the sample.
            generation_count: 1-indexed count of generations seen. If None, frequency gating is skipped.
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
            generation_idx: Index of this generation within its prompt group.
            **kwargs: Additional fields to store.
        """
        should_emit_scalars = generation_count is None or self.should_log_sample_scalars(generation_count)
        should_add_row = self._log_generation_table and (
            generation_count is None or self.should_log_sample_table_row(generation_count)
        )
        if not should_emit_scalars and not should_add_row:
            return
        record: dict[str, object] = {
            "sample_id": sample_id,
            "generation_count": generation_count,
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
            "generation_idx": generation_idx,
            **kwargs,
        }
        if should_emit_scalars:
            self.samples.append(dict(record))
        if should_add_row:
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

    def log_difficulty_stats(
        self,
        *,
        step: int,
        generation_count: int,
        sample_id: str,
        primary_source: str,
        raw_primary: float | None,
        difficulty_score: float,
        difficulty_bin: int,
        reward_total: float,
        predict_type: str,
        code_type: str,
        has_code_override: bool,
        secondary_raw_values: dict[str, float] | None = None,
    ) -> None:
        """Record difficulty statistics for a sample in memory."""
        record: dict[str, object] = {
            "step": step,
            "generation_count": generation_count,
            "sample_id": sample_id,
            "primary_source": primary_source,
            "raw_primary": raw_primary,
            "difficulty_score": difficulty_score,
            "difficulty_bin": difficulty_bin,
            "reward_total": reward_total,
            "predict_type": predict_type,
            "code_type": code_type,
            "has_code_override": has_code_override,
            "secondary_raw_values": dict(secondary_raw_values) if secondary_raw_values else None,
        }
        self.difficulty_stats.append(record)

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
        del kwargs  # absorb any future additions for forward compatibility
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
        self.runs.append(record)

    def log_batch_stats(
        self,
        *,
        batch_mean: float,
        batch_std: float,
        batch_mean_rolling_mean: float,
        batch_mean_rolling_std: float,
        batch_std_rolling_mean: float,
        batch_std_rolling_std: float,
        batch_count: int | None = None,
    ) -> None:
        """Record batch-level reward statistics in memory."""
        self.batch_stats.append(
            {
                "batch_count": batch_count,
                "batch_mean": batch_mean,
                "batch_std": batch_std,
                "batch_mean_rolling_mean": batch_mean_rolling_mean,
                "batch_mean_rolling_std": batch_mean_rolling_std,
                "batch_std_rolling_mean": batch_std_rolling_mean,
                "batch_std_rolling_std": batch_std_rolling_std,
            }
        )


class WandBRewardLogger:
    """RewardLogger implementation backed by a W&B run.

    Expects a `wandb.Run`-like object with a `.log(dict)` method.

    Metric Indexing:
        This logger configures WandB to use different x-axes for different metric types:

        - **Per-generation metrics** (reward/total, reward/terms/*, parsing/*, etc.) are indexed
          to `{prefix}/generation_count`, ensuring each logged generation has a unique x-coordinate.

        - **Batch-level metrics** (reward/batch/*) are indexed to `{prefix}/batch_count`, ensuring
          each logged batch has a unique x-coordinate.

        - **Run-level summaries** (reward/run/*) are indexed to `step_metric_key` (default:
          "train/global_step").

        This separation prevents WandB from aggregating values when multiple generations or batches
        are logged within the same trainer step (e.g., with gradient accumulation or multiple
        generations per prompt in GRPO).

    Integration with HuggingFace Trainer:
        The `wandb.define_metric()` calls that configure the step metrics are **deferred until
        the first log() call**, not called in `__init__`. This is intentional: HuggingFace's
        `WandbCallback` calls `wandb.define_metric("*", step_metric="train/global_step")` in
        `on_train_begin`, and our more specific patterns must be defined AFTER that wildcard
        to take precedence.
    """

    def __init__(
        self,
        wandb_run: object,
        *,
        step: int | None = None,
        log_tables: bool = False,
        table_max_rows: int = 1000,
        step_metric_key: str = "train/global_step",
        scalar_log_every_n_generations: int = 1,
        table_row_every_n_generations: int = 60,
        table_flush_every_n_generations: int = 1000,
    ) -> None:
        """Create a WandB-backed logger.

        Args:
            wandb_run: A `wandb.Run`-like object that supports `.log(...)`.
            step: Optional default step value (not WandB internal step). Used as a fallback for
                run-level summaries and table flushing when step is not explicitly provided. Also
                stored as "internal_step" in table rows.
            log_tables: Whether to log a W&B table with per-generation details.
            table_max_rows: Maximum number of buffered rows before forcing a flush.
            step_metric_key: Key used as x-axis for run-level summaries. Defaults to
                "train/global_step" to align with HuggingFace Trainer's WandbCallback. Per-generation
                metrics use `{prefix}/generation_count` and batch metrics use `{prefix}/batch_count`.
            scalar_log_every_n_generations: Emit scalar metrics every N generations (1-indexed).
            table_row_every_n_generations: Add a row to the table buffer every N generations (1-indexed).
            table_flush_every_n_generations: Flush the table buffer every N generations.
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
        self._scalar_log_every_n_generations = scalar_log_every_n_generations
        self._generation_table_row_every_n = table_row_every_n_generations
        self._generation_table_flush_every_n = table_flush_every_n_generations
        self._last_flush_generation_count: int | None = None  # for cooldown logic
        # track which prefixes have had wandb.define_metric called (called lazily per-prefix)
        self._generation_metrics_defined_prefixes: set[str] = set()
        self._batch_metrics_defined_prefixes: set[str] = set()
        self._run_metrics_defined_prefixes: set[str] = set()
        # difficulty table support (lightweight per-sample table)
        self._difficulty_table_rows: list[dict[str, object]] = []
        self._difficulty_table_columns = [
            # note: these should stay aligned with the features prepared in the difficulty estimator
            "step",
            "generation_count",
            "sample_id",
            "primary_source",
            "raw_primary",
            "difficulty_score",
            "difficulty_bin",
            "reward_total",
            "predict_type",
            "code_type",
            "has_code_override",
        ]
        # secondary raw values added dynamically as columns when present
        self._difficulty_table_secondary_columns: set[str] = set()
        # note: difficulty table uses the same max_rows as generation table (managed by LoggingConfig)
        if self._scalar_log_every_n_generations < 1:
            raise ValueError("scalar_log_every_n_generations must be >= 1")
        if self._generation_table_row_every_n < 1:
            raise ValueError("table_row_every_n_generations must be >= 1")
        if self._generation_table_flush_every_n < 1:
            raise ValueError("table_flush_every_n_generations must be >= 1")
        # NOTE: we do NOT call _define_*_step_metrics() here because HuggingFace's
        # WandbCallback calls `wandb.define_metric("*", step_metric="train/global_step")`
        # in on_train_begin, which would override our definitions if we called them earlier.
        # instead, we defer our define_metric calls to the first log() call, ensuring they
        # happen AFTER any trainer setup and thus take precedence.

    def _define_generation_step_metrics(self) -> None:
        """Define WandB step metrics and summaries for per-generation reward logging.

        This configures WandB to use `generation_count` as the x-axis for per-generation
        reward and parsing metrics, instead of the default `train/global_step`. This prevents
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
        - {prefix}/parsing/* -> mean
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
            ("parsing/*", "mean"),
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
        - {prefix}/failures/* -> last (per-phase failure stats, keep final value)
        """
        prefix = self._key_prefix.rstrip("/")
        if not prefix or prefix in self._run_metrics_defined_prefixes:
            return  # already defined for this prefix (or no prefix set)
        self._run_metrics_defined_prefixes.add(prefix)
        if wandb.run is None:
            return
        # run-level metric suffixes (use "last" since they're already aggregated summaries)
        metric_suffixes = ["reward/run/*", "failures/*", "difficulty/run/*"]
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

    def should_log_sample_scalars(
        self,
        generation_count: int,
    ) -> bool:
        """Return True if sample scalars should be logged for this generation_count."""
        return generation_count % self._scalar_log_every_n_generations == 0

    def should_log_sample_table_row(
        self,
        generation_count: int,
    ) -> bool:
        """Return True if a sample table row should be logged for this generation_count."""
        return self._log_generation_table and generation_count % self._generation_table_row_every_n == 0

    def _is_within_flush_cooldown(
        self,
        generation_count: int,
    ) -> bool:
        """Return True if we recently flushed and should skip periodic flush.

        Uses the full flush interval as cooldown to maintain consistent flush spacing.
        """
        if self._last_flush_generation_count is None:
            return False
        return (generation_count - self._last_flush_generation_count) < self._generation_table_flush_every_n

    def log_sample(
        self,
        sample_id: str,
        *,
        generation_count: int | None = None,
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
        generation_idx: int | None = None,
        **kwargs: typing.Any,
    ) -> None:
        """Log a per-sample reward payload to W&B.

        Args:
            sample_id: Unique identifier for the sample.
            generation_count: 1-indexed count of generations seen so far. If None, frequency gating
                is skipped (always logs). Useful for tests that call log_sample() directly.
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
            generation_idx: Index of this generation within its prompt group (0..k-1).
            **kwargs: Absorbed for forward compatibility.
        """
        del kwargs  # absorb any future additions for forward compatibility
        # lazily define step metrics on first log_sample call (after trainer setup)
        self._define_generation_step_metrics()
        # determine which outputs to emit based on frequency gating
        should_emit_scalars = generation_count is None or self.should_log_sample_scalars(generation_count)
        should_add_row = self._log_generation_table and (
            generation_count is None or self.should_log_sample_table_row(generation_count)
        )
        if not should_emit_scalars and not should_add_row:
            # allow periodic flush even when this call doesn't emit scalars or add a row
            if self._log_generation_table and self._generation_table_rows:
                is_max_rows_trigger = len(self._generation_table_rows) >= self._generation_table_max_rows
                is_periodic_trigger = (
                    generation_count is not None
                    and generation_count % self._generation_table_flush_every_n == 0
                    and not self._is_within_flush_cooldown(generation_count)
                )
                if is_max_rows_trigger or is_periodic_trigger:
                    self.flush_generation_table()
                    if generation_count is not None:
                        self._last_flush_generation_count = generation_count
            return
        terms = terms or {}
        metrics = metrics or {}
        payload_step = self._step if step is None else step
        # build and emit scalar payload (gated)
        if should_emit_scalars:
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
        # build and buffer table row (gated independently)
        if should_add_row:
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
                "generation_idx": generation_idx,
            }
            self._generation_table_rows.append(row)
        # flush table when buffer is full or periodic fallback trigger (independent of row-add gate)
        if self._log_generation_table and self._generation_table_rows:
            is_max_rows_trigger = len(self._generation_table_rows) >= self._generation_table_max_rows
            is_periodic_trigger = (
                generation_count is not None
                and generation_count % self._generation_table_flush_every_n == 0
                and not self._is_within_flush_cooldown(generation_count)
            )
            if is_max_rows_trigger or is_periodic_trigger:
                self.flush_generation_table()
                if generation_count is not None:
                    self._last_flush_generation_count = generation_count

    def log_batch_stats(
        self,
        *,
        batch_mean: float,
        batch_std: float,
        batch_mean_rolling_mean: float,
        batch_mean_rolling_std: float,
        batch_std_rolling_mean: float,
        batch_std_rolling_std: float,
        batch_count: int | None = None,
    ) -> None:
        """Log batch-level reward statistics to W&B.

        Batch-level metrics are indexed to `{prefix}/batch_count` to avoid aggregation issues
        when multiple batches (e.g., for gradient accumulation) are processed at the same step.

        Args:
            batch_mean: Mean reward for the current batch.
            batch_std: Std dev of rewards for the current batch.
            batch_mean_rolling_mean: Rolling mean of batch means.
            batch_mean_rolling_std: Rolling std of batch means.
            batch_std_rolling_mean: Rolling mean of batch std devs.
            batch_std_rolling_std: Rolling std of batch std devs.
            batch_count: Monotonic batch counter (1-indexed) used as x-axis for batch metrics.
        """
        self._define_batch_step_metrics()  # deferred initialization
        payload: dict[str, float] = {
            "reward/batch/mean": batch_mean,
            "reward/batch/std": batch_std,
            "reward/batch/mean_rolling/mean": batch_mean_rolling_mean,
            "reward/batch/mean_rolling/std": batch_mean_rolling_std,
            "reward/batch/std_rolling/mean": batch_std_rolling_mean,
            "reward/batch/std_rolling/std": batch_std_rolling_std,
        }
        prefixed: dict[str, object] = self._prefix_payload(payload)
        if batch_count is not None:
            prefixed[self._prefix_key("batch_count")] = batch_count
        if self._epoch is not None:
            prefixed[self._prefix_key("epoch")] = self._epoch
        self._wandb_run.log(prefixed)  # type: ignore[reportUnknownMemberType]

    def log_difficulty_stats(
        self,
        *,
        step: int,
        generation_count: int,
        sample_id: str,
        primary_source: str,
        raw_primary: float | None,
        difficulty_score: float,
        difficulty_bin: int,
        reward_total: float,
        predict_type: str,
        code_type: str,
        has_code_override: bool,
        secondary_raw_values: dict[str, float] | None = None,
    ) -> None:
        """Add a row to the lightweight difficulty table."""
        row: dict[str, object] = {
            "step": step,
            "generation_count": generation_count,
            "sample_id": sample_id,
            "primary_source": primary_source,
            "raw_primary": raw_primary,
            "difficulty_score": difficulty_score,
            "difficulty_bin": difficulty_bin,
            "reward_total": reward_total,
            "predict_type": predict_type,
            "code_type": code_type,
            "has_code_override": has_code_override,
        }
        # add secondary raw values as dynamic columns
        if secondary_raw_values:
            for key, value in secondary_raw_values.items():
                col_name = f"raw_{key}"
                self._difficulty_table_secondary_columns.add(col_name)
                row[col_name] = value
        self._difficulty_table_rows.append(row)
        # force flush if buffer exceeds max rows (uses same limit as generation table)
        if len(self._difficulty_table_rows) >= self._generation_table_max_rows:
            self.flush_difficulty_table()

    def flush_difficulty_table(self) -> None:
        """Flush difficulty table to wandb."""
        if not self._difficulty_table_rows:
            return
        # build final column list: base + dynamic secondary columns
        all_columns = list(self._difficulty_table_columns) + sorted(self._difficulty_table_secondary_columns)
        table = wandb.Table(
            columns=all_columns,
            data=[[row.get(col, None) for col in all_columns] for row in self._difficulty_table_rows],
        )
        self._wandb_run.log({self._prefix_key("difficulty_samples"): table})  # type: ignore[reportUnknownMemberType]
        self._difficulty_table_rows = []

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
        del kwargs  # absorb any future additions for forward compatibility
        self._define_run_step_metrics()  # deferred initialization
        payload_step = self._step if step is None else step
        payload: dict[str, reward_types.MetricValue] = {
            **reward_totals,
            **reward_term_summaries,
            **reward_category_summaries,
            **(parsing_summaries or {}),
            **(parsing_category_summaries or {}),
        }
        if difficulty_summaries:
            for key, value in difficulty_summaries.items():
                payload[f"difficulty/run/{key}"] = value
        if failure_ratio is not None:
            payload["failures/failure_ratio"] = failure_ratio
        if failure_count is not None:
            payload["failures/failure_count"] = failure_count
        prefixed: dict[str, object] = self._prefix_payload(payload)
        if payload_step is not None:
            prefixed[self._step_metric_key] = payload_step  # global_step not prefixed
        if self._epoch is not None:
            prefixed[self._prefix_key("epoch")] = self._epoch
        self._wandb_run.log(prefixed)  # type: ignore[reportUnknownMemberType]
        if self._log_generation_table:
            self.flush_generation_table()
        # flush difficulty table at phase-level logging (e.g., end of eval phase)
        self.flush_difficulty_table()

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
            "generation_idx",
        ]
        table = wandb.Table(columns=columns)
        table_obj = typing.cast("typing.Any", table)
        for row in self._generation_table_rows:
            row_data: list[object] = [
                row["sample_id"],
                row["generation_count"],
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
                row["generation_idx"],
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
        scalar_log_every_n_generations=logging_config.scalar_log_every_n_generations,
        table_row_every_n_generations=logging_config.table_row_every_n_generations,
        table_flush_every_n_generations=logging_config.table_flush_every_n_generations,
    )
