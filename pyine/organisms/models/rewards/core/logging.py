"""Reward logging helpers.

The reward system exposes a small `RewardLogger` protocol. The logger controls logging frequency
via generation_count gating; the manager handles key scoping and orchestration.
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
        self.failures: list[dict[str, object]] = []
        self.batch_stats: list[dict[str, object]] = []
        self._step: int | None = None
        self._key_prefix: str = ""
        self._scalar_log_every_n_generations = int(scalar_log_every_n_generations)
        self._log_tables = log_tables
        self._table_row_every_n_generations = int(table_row_every_n_generations)
        if self._scalar_log_every_n_generations < 1:
            raise ValueError("scalar_log_every_n_generations must be >= 1")
        if self._table_row_every_n_generations < 1:
            raise ValueError("table_row_every_n_generations must be >= 1")

    def should_log_scalars(
        self,
        generation_count: int,
    ) -> bool:
        """Return True if scalars should be logged for this generation_count."""
        return generation_count % self._scalar_log_every_n_generations == 0

    def should_add_table_row(
        self,
        generation_count: int,
    ) -> bool:
        """Return True if a table row should be added for this generation_count."""
        return self._log_tables and generation_count % self._table_row_every_n_generations == 0

    def log(
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
        should_emit_scalars = generation_count is None or self.should_log_scalars(generation_count)
        should_add_row = self._log_tables and (generation_count is None or self.should_add_table_row(generation_count))
        if not should_emit_scalars and not should_add_row:
            return
        record: dict[str, object] = {
            "sample_id": sample_id,
            "generation_count": generation_count,
            "step": step,
            "internal_step": self._step,
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

    def set_key_prefix(self, key_prefix: str) -> None:
        """Set the key prefix for subsequent logs."""
        self._key_prefix = parsing_utils.normalize_path_prefix(key_prefix)

    def log_run(
        self,
        *,
        reward_totals: collections.abc.Mapping[str, float],
        reward_term_summaries: collections.abc.Mapping[str, float],
        reward_category_summaries: collections.abc.Mapping[str, float],
        parsing_summaries: collections.abc.Mapping[str, float] | None = None,
        parsing_category_summaries: collections.abc.Mapping[str, float] | None = None,
        step: int | None = None,
    ) -> None:
        """Record a run-level logging event in memory."""
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
        self.runs.append(record)

    def log_failures(
        self,
        *,
        failure_ratio: float,
        failure_count: int,
        step: int | None = None,
    ) -> None:
        """Record a failure stats logging event in memory."""
        self.failures.append(
            {
                "step": step,
                "failure_ratio": failure_ratio,
                "failure_count": failure_count,
            }
        )

    def log_batch_stats(
        self,
        *,
        batch_mean: float,
        batch_std: float,
        batch_mean_rolling_mean: float,
        batch_mean_rolling_std: float,
        batch_std_rolling_mean: float,
        batch_std_rolling_std: float,
        step: int | None = None,
    ) -> None:
        """Record batch-level reward statistics in memory."""
        self.batch_stats.append(
            {
                "step": step,
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
            step: Optional step value logged under `step_metric_key` (not WandB internal step).
            log_tables: Whether to log a W&B table with per-generation details.
            table_max_rows: Maximum number of buffered rows before forcing a flush.
            step_metric_key: Key used for the step metric in logged payloads. Defaults to
                "train/global_step" to align with HuggingFace Trainer's WandbCallback.
            scalar_log_every_n_generations: Emit scalar metrics every N generations (1-indexed).
            table_row_every_n_generations: Add a row to the table buffer every N generations (1-indexed).
            table_flush_every_n_generations: Flush the table buffer every N generations.
        """
        self._wandb_run = wandb_run
        self._key_prefix = ""
        self._step = step
        self._log_tables = log_tables
        self._table_key = "generation_details"
        self._table_max_rows = int(table_max_rows)
        self._table_rows: list[dict[str, object]] = []
        self._step_metric_key = step_metric_key
        self._scalar_log_every_n_generations = int(scalar_log_every_n_generations)
        self._table_row_every_n_generations = int(table_row_every_n_generations)
        self._table_flush_every_n_generations = int(table_flush_every_n_generations)
        self._last_flush_generation_count: int | None = None  # for cooldown logic
        if self._scalar_log_every_n_generations < 1:
            raise ValueError("scalar_log_every_n_generations must be >= 1")
        if self._table_row_every_n_generations < 1:
            raise ValueError("table_row_every_n_generations must be >= 1")
        if self._table_flush_every_n_generations < 1:
            raise ValueError("table_flush_every_n_generations must be >= 1")

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
        return {self._prefix_key(str(key)): value for key, value in payload.items()}

    def should_log_scalars(
        self,
        generation_count: int,
    ) -> bool:
        """Return True if scalars should be logged for this generation_count."""
        return generation_count % self._scalar_log_every_n_generations == 0

    def should_add_table_row(
        self,
        generation_count: int,
    ) -> bool:
        """Return True if a table row should be added for this generation_count."""
        return self._log_tables and generation_count % self._table_row_every_n_generations == 0

    def _is_within_flush_cooldown(
        self,
        generation_count: int,
    ) -> bool:
        """Return True if we recently flushed and should skip periodic flush.

        Uses the full flush interval as cooldown to maintain consistent flush spacing.
        """
        if self._last_flush_generation_count is None:
            return False
        return (generation_count - self._last_flush_generation_count) < self._table_flush_every_n_generations

    def log(
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
                is skipped (always logs). Useful for tests that call log() directly.
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
        # determine which outputs to emit based on frequency gating
        should_emit_scalars = generation_count is None or self.should_log_scalars(generation_count)
        should_add_row = self._log_tables and (generation_count is None or self.should_add_table_row(generation_count))
        if not should_emit_scalars and not should_add_row:
            # allow periodic flush even when this call doesn't emit scalars or add a row
            if self._log_tables and self._table_rows:
                is_max_rows_trigger = len(self._table_rows) >= self._table_max_rows
                is_periodic_trigger = (
                    generation_count is not None
                    and generation_count % self._table_flush_every_n_generations == 0
                    and not self._is_within_flush_cooldown(generation_count)
                )
                if is_max_rows_trigger or is_periodic_trigger:
                    payload_step = self._step if step is None else step
                    self.flush_tables(step=payload_step)
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
                payload["reward/total"] = float(total)
            payload.update({k: float(v) for k, v in terms.items()})
            payload.update(dict(metrics))
            if raw_terms is not None:
                for term_name, raw_value in raw_terms.items():
                    payload[f"reward/raw_terms/{term_name}"] = float(raw_value)
            if payload:  # avoid empty wandb.log() calls
                prefixed = self._prefix_payload(payload)
                if payload_step is not None:
                    prefixed[self._step_metric_key] = payload_step
                self._wandb_run.log(prefixed)  # type: ignore[reportUnknownMemberType]
        # build and buffer table row (gated independently)
        if should_add_row:
            prefixed_terms = self._prefix_payload(dict(terms))
            prefixed_metrics = self._prefix_payload(dict(metrics))
            raw_terms_payload: dict[str, float] | None = None
            if raw_terms is not None:
                raw_terms_payload = {
                    f"reward/raw_terms/{term_name}": float(raw_value) for term_name, raw_value in raw_terms.items()
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
            self._table_rows.append(row)
        # flush table when buffer is full or periodic fallback trigger (independent of row-add gate)
        if self._log_tables and self._table_rows:
            is_max_rows_trigger = len(self._table_rows) >= self._table_max_rows
            is_periodic_trigger = (
                generation_count is not None
                and generation_count % self._table_flush_every_n_generations == 0
                and not self._is_within_flush_cooldown(generation_count)
            )
            if is_max_rows_trigger or is_periodic_trigger:
                self.flush_tables(step=payload_step)
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
        step: int | None = None,
    ) -> None:
        """Log batch-level reward statistics to W&B.

        Args:
            batch_mean: Mean reward for the current batch.
            batch_std: Std dev of rewards for the current batch.
            batch_mean_rolling_mean: Rolling mean of batch means.
            batch_mean_rolling_std: Rolling std of batch means.
            batch_std_rolling_mean: Rolling mean of batch std devs.
            batch_std_rolling_std: Rolling std of batch std devs.
            step: Optional step value (overrides logger's default step).
        """
        payload_step = self._step if step is None else step
        payload: dict[str, float] = {
            "reward/batch/mean": batch_mean,
            "reward/batch/std": batch_std,
            "reward/batch/mean_rolling/mean": batch_mean_rolling_mean,
            "reward/batch/mean_rolling/std": batch_mean_rolling_std,
            "reward/batch/std_rolling/mean": batch_std_rolling_mean,
            "reward/batch/std_rolling/std": batch_std_rolling_std,
        }
        prefixed: dict[str, object] = self._prefix_payload(payload)
        if payload_step is not None:
            prefixed[self._step_metric_key] = payload_step
        self._wandb_run.log(prefixed)  # type: ignore[reportUnknownMemberType]

    def log_run(
        self,
        *,
        reward_totals: collections.abc.Mapping[str, float],
        reward_term_summaries: collections.abc.Mapping[str, float],
        reward_category_summaries: collections.abc.Mapping[str, float],
        parsing_summaries: collections.abc.Mapping[str, float] | None = None,
        parsing_category_summaries: collections.abc.Mapping[str, float] | None = None,
        step: int | None = None,
    ) -> None:
        """Log a run-level summary payload to W&B."""
        payload_step = self._step if step is None else step
        payload: dict[str, float] = {}
        payload.update({k: float(v) for k, v in reward_totals.items()})
        payload.update({k: float(v) for k, v in reward_term_summaries.items()})
        payload.update({k: float(v) for k, v in reward_category_summaries.items()})
        if parsing_summaries:
            payload.update({k: float(v) for k, v in parsing_summaries.items()})
        if parsing_category_summaries:
            payload.update({k: float(v) for k, v in parsing_category_summaries.items()})
        prefixed: dict[str, object] = self._prefix_payload(payload)
        if payload_step is not None:
            prefixed[self._step_metric_key] = payload_step  # global_step not prefixed
        self._wandb_run.log(prefixed)  # type: ignore[reportUnknownMemberType]
        if self._log_tables:
            self.flush_tables(step=payload_step)

    def log_failures(
        self,
        *,
        failure_ratio: float,
        failure_count: int,
        step: int | None = None,
    ) -> None:
        """Log failure statistics to W&B under the failures/ prefix."""
        payload_step = self._step if step is None else step
        payload: dict[str, float] = {
            "failures/failure_ratio": failure_ratio,
            "failures/failure_count": float(failure_count),
        }
        prefixed: dict[str, object] = self._prefix_payload(payload)
        if payload_step is not None:
            prefixed[self._step_metric_key] = payload_step  # global_step not prefixed
        self._wandb_run.log(prefixed)  # type: ignore[reportUnknownMemberType]

    def set_step(
        self,
        step: int | None,
    ) -> None:
        """Set a default step value for subsequent logs (logged under `step_metric_key`)."""
        self._step = step

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

    def flush_tables(
        self,
        *,
        step: int | None = None,
    ) -> None:
        """Flush buffered table rows to W&B (no-op if table logging is disabled)."""
        if not self._log_tables or not self._table_rows:
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
        for row in self._table_rows:
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
        payload: dict[str, object] = {self._prefix_key(self._table_key): table}
        if step is not None:
            payload[self._step_metric_key] = step
        self._wandb_run.log(payload)  # type: ignore[reportUnknownMemberType]
        self._table_rows.clear()


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
        table_max_rows=int(logging_config.table_max_rows),
        step_metric_key=logging_config.step_metric_key,
        scalar_log_every_n_generations=int(logging_config.scalar_log_every_n_generations),
        table_row_every_n_generations=int(logging_config.table_row_every_n_generations),
        table_flush_every_n_generations=int(logging_config.table_flush_every_n_generations),
    )
