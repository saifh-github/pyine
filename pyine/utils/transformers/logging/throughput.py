"""Direct wandb logging for throughput metrics."""

import typing

import pyine.utils.transformers.logging.common as logging_common


class ThroughputLogger:
    """Direct wandb logging for throughput metrics."""

    def __init__(
        self,
        wandb_run: typing.Any,
        config: typing.Any,
    ) -> None:
        """Initialize throughput logger.

        Args:
            wandb_run: The wandb run object (from runtime.wandb_run)
            config: ThroughputLoggingConfig with train_prefix and eval_prefix
        """
        self._wandb_run = wandb_run
        self._config = config
        self._step_metric_key = logging_common.DEFAULT_STEP_METRIC_KEY
        self._metrics_defined = False

    def log_train_throughput(
        self,
        step: int,
        samples_per_second: float,
        steps_per_second: float,
    ) -> None:
        """Log train throughput directly to wandb.

        Args:
            step: Current global step (for x-axis)
            samples_per_second: Train samples processed per second
            steps_per_second: Train steps processed per second
        """
        if not self._metrics_defined:
            self._define_metrics()
        payload = {
            f"{self._config.train_prefix}samples_per_second": samples_per_second,
            f"{self._config.train_prefix}steps_per_second": steps_per_second,
            self._step_metric_key: step,
        }
        self._wandb_run.log(payload)

    def log_eval_throughput(
        self,
        step: int,
        samples_per_second: float,
    ) -> None:
        """Log eval throughput directly to wandb.

        Args:
            step: Current global step (for x-axis)
            samples_per_second: Eval samples processed per second
        """
        if not self._metrics_defined:
            self._define_metrics()
        payload = {
            f"{self._config.eval_prefix}samples_per_second": samples_per_second,
            self._step_metric_key: step,
        }
        self._wandb_run.log(payload)

    def _define_metrics(self) -> None:
        """Define custom step metrics (called lazily after WandbCallback).

        Deferred until first log to ensure it runs AFTER HuggingFace's WandbCallback
        calls wandb.define_metric("*", step_metric="train/global_step").
        """
        for metric in ["samples_per_second", "steps_per_second"]:
            logging_common.deferred_define_metric(
                self._wandb_run,
                f"{self._config.train_prefix}{metric}",
                step_metric=self._step_metric_key,
                summary="last",
            )
        logging_common.deferred_define_metric(
            self._wandb_run,
            f"{self._config.eval_prefix}samples_per_second",
            step_metric=self._step_metric_key,
            summary="last",
        )
        self._metrics_defined = True
