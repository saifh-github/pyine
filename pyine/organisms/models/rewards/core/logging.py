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

    def log(
        self,
        sample_id: str,
        *,
        total: float | None,
        terms: collections.abc.Mapping[str, float],
        metrics: collections.abc.Mapping[str, reward_types.MetricValue],
        step: int | None = None,
        prompt: str | None = None,
        model_output: str | None = None,
        categories: collections.abc.Sequence[str] | None = None,
        tags: collections.abc.Sequence[str] | None = None,
    ) -> None:
        """Record a per-sample logging event in memory."""
        record: dict[str, object] = {
            "sample_id": sample_id,
            "step": step,
            "terms": dict(terms),
            "metrics": dict(metrics),
            "prompt": prompt,
            "model_output": model_output,
            "categories": list(categories) if categories is not None else None,
            "tags": list(tags) if tags is not None else None,
        }
        if total is not None:
            record["total"] = float(total)
        self.samples.append(record)

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
        table_max_rows: int = 1000,
        step_metric_key: str = "train/global_step",
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
        terms: collections.abc.Mapping[str, float],
        metrics: collections.abc.Mapping[str, reward_types.MetricValue],
        step: int | None = None,
        prompt: str | None = None,
        model_output: str | None = None,
        categories: collections.abc.Sequence[str] | None = None,
        tags: collections.abc.Sequence[str] | None = None,
    ) -> None:
        """Log a per-sample reward payload to W&B."""
        payload_step = self._step if step is None else step
        payload: dict[str, reward_types.MetricValue] = {}
        if total is not None:
            payload[self._total_key] = float(total)
        payload.update({k: float(v) for k, v in terms.items()})
        payload.update(dict(metrics))
        prefixed = self._prefix_payload(payload)
        if payload_step is not None:
            prefixed[self._step_metric_key] = payload_step  # global_step not prefixed
        self._wandb_run.log(prefixed)  # type: ignore[reportUnknownMemberType]
        if self._log_tables:
            self._table_log_count += 1
            prefixed_terms = self._prefix_payload(dict(terms))
            prefixed_metrics = self._prefix_payload(dict(metrics))
            self._table_rows.append(
                {
                    "sample_id": sample_id,
                    "step": payload_step,
                    "prompt": prompt,
                    "model_output": model_output,
                    "total": total,  # may be None
                    "terms_json": json.dumps(prefixed_terms, sort_keys=True),
                    "metrics_json": json.dumps(prefixed_metrics, sort_keys=True),
                    "categories_json": json.dumps(list(categories), sort_keys=True) if categories else None,
                    "tags_json": json.dumps(list(tags), sort_keys=True) if tags else None,
                }
            )
            if (
                len(self._table_rows) >= self._table_max_rows
                or self._table_log_count % self._table_flush_every_n_logs == 0
            ):
                self.flush_tables(step=payload_step)

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
        table = wandb.Table(
            columns=[
                "sample_id",
                "step",
                "prompt",
                "model_output",
                "total",
                "terms_json",
                "metrics_json",
                "categories_json",
                "tags_json",
            ],
        )
        table_obj = typing.cast("typing.Any", table)
        for row in self._table_rows:
            table_obj.add_data(
                row["sample_id"],
                row["step"],
                row["prompt"],
                row["model_output"],
                row["total"],
                row["terms_json"],
                row["metrics_json"],
                row["categories_json"],
                row["tags_json"],
            )
        payload: dict[str, object] = {self._prefix_key(self._table_key): table}
        if step is not None:
            payload[self._step_metric_key] = step
        self._wandb_run.log(payload)  # type: ignore[reportUnknownMemberType]
        self._table_rows.clear()


_DEFAULT_TABLE_KEY = "reward/rewards_table"


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
        If `table_key` is still the default value, it is automatically derived from `scope_prefix`.
    """
    table_key: str | None = logging_config.table_key
    if table_key == _DEFAULT_TABLE_KEY:
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
    )
