"""Reward logging helpers.

The reward system exposes a small `RewardLogger` protocol. The manager controls logging frequency
and key scoping; loggers simply emit scalar payloads to their backend.
"""

import collections.abc
import json
import typing

import pyine.organisms.models.rewards.core.configs as reward_configs
import pyine.organisms.models.rewards.core.types
import pyine.utils.strings as strings_utils


class InMemoryRewardLogger:
    """RewardLogger implementation for tests and debugging.

    Stores structured events in memory instead of writing to an external backend.
    """

    def __init__(self) -> None:
        """Create an in-memory logger with empty buffers."""
        self.samples: list[dict[str, object]] = []
        self.runs: list[dict[str, object]] = []

    def log(
        self,
        sample_id: str,
        *,
        total: float,
        terms: collections.abc.Mapping[str, float],
        metrics: collections.abc.Mapping[str, bool | int | float],
        step: int | None = None,
    ) -> None:
        """Record a per-sample logging event in memory."""
        self.samples.append(
            {
                "sample_id": sample_id,
                "step": step,
                "total": float(total),
                "terms": dict(terms),
                "metrics": dict(metrics),
            }
        )

    def log_run(
        self,
        *,
        totals: collections.abc.Mapping[str, float],
        term_summaries: collections.abc.Mapping[str, float],
        step: int | None = None,
    ) -> None:
        """Record a run-level logging event in memory."""
        self.runs.append(
            {
                "step": step,
                "totals": dict(totals),
                "term_summaries": dict(term_summaries),
            }
        )


class WandBRewardLogger:
    """RewardLogger implementation backed by a W&B run.

    This logger intentionally avoids importing `wandb` at import time. It expects a `wandb.Run`-like
    object with a `.log(dict, step=...)` method.
    """

    def __init__(
        self,
        wandb_run: object,
        *,
        total_key: str = "reward/total",
        key_prefix: str = "",
        step: int | None = None,
        log_tables: bool = False,
        table_key: str = "reward/rewards_table",
        table_flush_every_n_logs: int = 100,
        table_max_rows: int = 1000,
    ) -> None:
        """Create a WandB-backed logger.

        Args:
            wandb_run: A `wandb.Run`-like object that supports `.log(...)`.
            total_key: Key under which to log the total reward.
            key_prefix: Optional prefix applied to all keys emitted to W&B (scalars and table key).
            step: Optional fixed W&B step to use for all logs.
            log_tables: Whether to log a W&B table with per-sample reward breakdowns.
            table_key: W&B key under which to log the rewards table.
            table_flush_every_n_logs: Flush the table every N logger calls.
            table_max_rows: Maximum number of buffered rows before forcing a flush.
        """
        self._wandb_run = wandb_run
        self._key_prefix = strings_utils.normalize_path_prefix(key_prefix)
        self._total_key = total_key
        self._step = step
        self._log_tables = log_tables
        self._table_key = table_key
        self._table_flush_every_n_logs = int(table_flush_every_n_logs)
        self._table_max_rows = int(table_max_rows)
        self._table_log_count = 0
        self._table_rows: list[dict[str, object]] = []

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
        total: float,
        terms: collections.abc.Mapping[str, float],
        metrics: collections.abc.Mapping[str, bool | int | float],
        step: int | None = None,
    ) -> None:
        """Log a per-sample reward payload to W&B."""
        payload_step = self._step if step is None else step
        payload: dict[str, bool | int | float] = {self._total_key: float(total)}
        payload.update({k: float(v) for k, v in terms.items()})
        payload.update(dict(metrics))
        self._wandb_run.log(self._prefix_payload(payload), step=payload_step)  # type: ignore[reportUnknownMemberType]
        if self._log_tables:
            self._table_log_count += 1
            prefixed_terms = self._prefix_payload(dict(terms))
            prefixed_metrics = self._prefix_payload(dict(metrics))
            self._table_rows.append(
                {
                    "sample_id": sample_id,
                    "step": payload_step,
                    "total": float(total),
                    "terms_json": json.dumps(prefixed_terms, sort_keys=True),
                    "metrics_json": json.dumps(prefixed_metrics, sort_keys=True),
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
        totals: collections.abc.Mapping[str, float],
        term_summaries: collections.abc.Mapping[str, float],
        step: int | None = None,
    ) -> None:
        """Log a run-level summary payload to W&B."""
        payload_step = self._step if step is None else step
        payload: dict[str, float] = {}
        payload.update({k: float(v) for k, v in totals.items()})
        payload.update({k: float(v) for k, v in term_summaries.items()})
        self._wandb_run.log(self._prefix_payload(payload), step=payload_step)  # type: ignore[reportUnknownMemberType]
        if self._log_tables:
            self.flush_tables(step=payload_step)

    def set_step(
        self,
        step: int | None,
    ) -> None:
        """Set a default W&B step for subsequent logs."""
        self._step = step

    def flush_tables(
        self,
        *,
        step: int | None = None,
    ) -> None:
        """Flush buffered table rows to W&B (no-op if table logging is disabled)."""
        if not self._log_tables or not self._table_rows:
            return
        import wandb

        table = wandb.Table(
            columns=["sample_id", "step", "total", "terms_json", "metrics_json"],
        )
        table_obj = typing.cast("typing.Any", table)
        for row in self._table_rows:
            table_obj.add_data(
                row["sample_id"],
                row["step"],
                row["total"],
                row["terms_json"],
                row["metrics_json"],
            )
        self._wandb_run.log({self._prefix_key(self._table_key): table}, step=step)  # type: ignore[reportUnknownMemberType]
        self._table_rows.clear()


def assert_is_reward_logger(
    logger: object,
) -> pyine.organisms.models.rewards.core.types.RewardLogger:
    """Narrow a value to RewardLogger (for runtime-provided loggers).

    This is mostly a typing helper: callers can pass loggers constructed elsewhere and narrow them
    into the protocol type.
    """
    return logger  # type: ignore[return-value]


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
        step: Optional fixed W&B step to use for all logs.

    Returns:
        A WandB-backed reward logger with total logged under `<scope_prefix>/total`.
    """
    prefix_norm = strings_utils.normalize_path_prefix(logging_config.scope_prefix)
    total_key = f"{prefix_norm}total" if prefix_norm else "total"
    return WandBRewardLogger(
        wandb_run,
        total_key=total_key,
        key_prefix=logging_config.wandb_key_prefix,
        step=step,
        log_tables=logging_config.log_tables,
        table_key=logging_config.table_key,
        table_flush_every_n_logs=int(logging_config.table_flush_every_n_logs),
        table_max_rows=int(logging_config.table_max_rows),
    )
