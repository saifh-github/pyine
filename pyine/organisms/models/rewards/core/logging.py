"""Reward logging helpers.

The reward system exposes a small `RewardLogger` protocol. The manager controls logging frequency
and key scoping; loggers simply emit scalar payloads to their backend.
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
    """

    def __init__(self) -> None:
        """Create an in-memory logger with empty buffers."""
        self.samples: list[dict[str, object]] = []
        self.runs: list[dict[str, object]] = []
        self.failures: list[dict[str, object]] = []
        self._step: int | None = None
        self._key_prefix: str = ""

    def log(
        self,
        sample_id: str,
        *,
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
        """Record a per-sample logging event in memory."""
        record: dict[str, object] = {
            "sample_id": sample_id,
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
        self.samples.append(record)

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


class WandBRewardLogger:
    """RewardLogger implementation backed by a W&B run.

    Expects a `wandb.Run`-like object with a `.log(dict)` method.
    """

    def __init__(
        self,
        wandb_run: object,
        *,
        scope_prefix: str = "reward/",
        key_prefix: str = "",
        step: int | None = None,
        log_tables: bool = False,
        table_key: str | None = None,
        table_flush_every_n_logs: int = 100,
        table_max_rows: int = 20,
        step_metric_key: str = "train/global_step",
        log_histograms: bool = False,
        histogram_log_interval: int = 100,
    ) -> None:
        """Create a WandB-backed logger.

        Args:
            wandb_run: A `wandb.Run`-like object that supports `.log(...)`.
            scope_prefix: Normalized prefix for reward keys (e.g., "reward/"); used to derive
                total_key and table_key.
            key_prefix: Optional extra prefix applied to all keys emitted to W&B.
            step: Optional step value logged under `step_metric_key` (not WandB internal step).
            log_tables: Whether to log a W&B table with per-sample reward breakdowns.
            table_key: W&B key for the rewards table (defaults to `<scope_prefix>rewards_table`).
            table_flush_every_n_logs: Flush the table every N logger calls.
            table_max_rows: Maximum number of buffered rows before forcing a flush.
            step_metric_key: Key used for the step metric in logged payloads. Defaults to
                "train/global_step" to align with HuggingFace Trainer's WandbCallback.
            log_histograms: Whether to log W&B histograms for reward distributions.
            histogram_log_interval: Number of samples between histogram logs.
        """
        self._wandb_run = wandb_run
        self._key_prefix = parsing_utils.normalize_path_prefix(key_prefix)
        self._scope_prefix = parsing_utils.normalize_path_prefix(scope_prefix)
        self._total_key = f"{self._scope_prefix}total" if self._scope_prefix else "total"
        self._step = step
        self._log_tables = log_tables
        self._table_key = table_key if table_key is not None else f"{self._scope_prefix}rewards_table"
        self._table_flush_every_n_logs = int(table_flush_every_n_logs)
        self._table_max_rows = int(table_max_rows)
        self._table_log_count = 0
        self._table_rows: list[dict[str, object]] = []
        self._step_metric_key = step_metric_key
        self._log_histograms = log_histograms
        self._histogram_log_interval = int(histogram_log_interval)
        self._reward_buffer: list[float] = []
        self._term_buffers: dict[str, list[float]] = {}
        self._histogram_sample_count = 0

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

    def log(
        self,
        sample_id: str,
        *,
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
        """Log a per-sample reward payload to W&B."""
        del kwargs  # absorb any future additions for forward compatibility
        terms = terms or {}
        metrics = metrics or {}
        payload_step = self._step if step is None else step
        payload: dict[str, reward_types.MetricValue] = {}
        if total is not None:
            payload[self._total_key] = float(total)
        payload.update({k: float(v) for k, v in terms.items()})
        payload.update(dict(metrics))
        raw_terms_payload: dict[str, float] | None = None
        if raw_terms is not None:
            raw_prefix = f"{self._scope_prefix}raw_terms/" if self._scope_prefix else "raw_terms/"
            raw_terms_payload = {
                f"{raw_prefix}{term_name}": float(raw_value) for term_name, raw_value in raw_terms.items()
            }
            payload.update(raw_terms_payload)
        prefixed = self._prefix_payload(payload)
        if payload_step is not None:
            prefixed[self._step_metric_key] = payload_step  # global_step not prefixed
        self._wandb_run.log(prefixed)  # type: ignore[reportUnknownMemberType]
        # buffer values for histogram computation
        if self._log_histograms and total is not None:
            self._buffer_reward_value(total, terms)
            if self._histogram_sample_count % self._histogram_log_interval == 0:
                self._emit_histograms(step=payload_step)
        # handle table logging
        if self._log_tables:
            self._table_log_count += 1
            prefixed_terms = self._prefix_payload(dict(terms))
            prefixed_metrics = self._prefix_payload(dict(metrics))
            row: dict[str, object] = {
                "sample_id": sample_id,
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
            if (
                len(self._table_rows) >= self._table_max_rows
                or self._table_log_count % self._table_flush_every_n_logs == 0
            ):
                self.flush_tables(step=payload_step)

    def _buffer_reward_value(
        self,
        total: float,
        terms: collections.abc.Mapping[str, float],
    ) -> None:
        """Buffer reward values for histogram computation."""
        self._reward_buffer.append(total)
        self._histogram_sample_count += 1
        for term_name, term_value in terms.items():
            normalized = term_name
            if self._scope_prefix and normalized.startswith(f"{self._scope_prefix}terms/"):
                normalized = normalized.removeprefix(f"{self._scope_prefix}terms/")
            elif normalized.startswith("terms/"):
                normalized = normalized.removeprefix("terms/")
            elif "/terms/" in normalized:
                normalized = normalized.rsplit("/terms/", maxsplit=1)[1]
            if normalized not in self._term_buffers:
                self._term_buffers[normalized] = []
            self._term_buffers[normalized].append(term_value)

    def _emit_histograms(
        self,
        step: int | None,
    ) -> None:
        """Emit W&B histograms for buffered reward values."""
        if not self._reward_buffer:
            return
        payload: dict[str, object] = {}
        hist_prefix = f"{self._scope_prefix}histograms/" if self._scope_prefix else "histograms/"
        payload[f"{hist_prefix}total"] = wandb.Histogram(self._reward_buffer)
        for term_name, term_values in self._term_buffers.items():
            if term_values:
                payload[f"{hist_prefix}terms/{term_name}"] = wandb.Histogram(term_values)
        prefixed = self._prefix_payload(payload)
        if step is not None:
            prefixed[self._step_metric_key] = step
        self._wandb_run.log(prefixed)  # type: ignore[reportUnknownMemberType]
        self._clear_buffers()

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
        if self._log_histograms and self._reward_buffer:
            self._emit_histograms(step=payload_step)
        # clear buffers after run-level logging
        self._clear_buffers()

    def _clear_buffers(self) -> None:
        """Clear all buffered values after run-level logging."""
        self._reward_buffer.clear()
        self._term_buffers.clear()
        self._histogram_sample_count = 0

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
    """Create a `WandBRewardLogger` that matches a `LoggingConfig` scope prefix.

    Args:
        wandb_run: A `wandb.Run`-like object that supports `.log(...)`.
        logging_config: Reward logging configuration (notably `scope_prefix` and `wandb_key_prefix`).
        step: Optional step value logged under `step_metric_key` (not WandB internal step).

    Returns:
        A WandB-backed reward logger with total logged under `<scope_prefix>/total`.

    Note:
        If `table_key` matches the logger's default (derived from `scope_prefix`), it is treated as
        unset so that changing `scope_prefix` automatically updates the table key.
    """
    table_key: str | None = logging_config.table_key
    scope_prefix = parsing_utils.normalize_path_prefix(logging_config.scope_prefix)
    default_table_key = f"{scope_prefix}rewards_table" if scope_prefix else "rewards_table"
    if table_key == default_table_key:
        table_key = None  # let __init__ derive from scope_prefix
    return WandBRewardLogger(
        wandb_run,
        scope_prefix=logging_config.scope_prefix,
        key_prefix=logging_config.wandb_key_prefix,
        step=step,
        log_tables=logging_config.log_tables,
        table_key=table_key,
        table_flush_every_n_logs=int(logging_config.table_flush_every_n_logs),
        table_max_rows=int(logging_config.table_max_rows),
        step_metric_key=logging_config.step_metric_key,
        log_histograms=logging_config.log_histograms,
        histogram_log_interval=int(logging_config.histogram_log_interval),
    )
