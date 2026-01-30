"""Direct wandb logging for GPU stats metrics."""

import typing

import pyine.utils.transformers.logging.common as logging_common


class GPUStatsLogger:
    """Direct wandb logging for GPU stats metrics."""

    def __init__(
        self,
        wandb_run: typing.Any | None,
        config: typing.Any,
    ) -> None:
        """Initialize GPU stats logger.

        Args:
            wandb_run: The wandb run object (from runtime.wandb_run)
            config: GPUStatsLoggingConfig with train_prefix and eval_prefix
        """
        self._wandb_run = wandb_run
        self._config = config
        self._step_metric_key = logging_common.DEFAULT_STEP_METRIC_KEY
        self._metrics_defined = False
        self._enabled = wandb_run is not None

    def log_train_stats(
        self,
        step: int,
        **metrics: float,
    ) -> None:
        """Log train GPU stats directly to wandb.

        Args:
            step: Current global step (for x-axis)
            **metrics: GPU stats to log (e.g., utilization_gpu_percent, pytorch_peak_percent)
        """
        if not self._enabled:
            return
        if not self._metrics_defined:
            self._define_metrics()
        payload = {}
        for key, value in metrics.items():
            payload[f"{self._config.train_prefix}{key}"] = value
        payload[self._step_metric_key] = step
        self._wandb_run.log(payload)

    def log_eval_stats(
        self,
        step: int,
        **metrics: float,
    ) -> None:
        """Log eval GPU stats directly to wandb.

        Args:
            step: Current global step (for x-axis)
            **metrics: GPU stats to log
        """
        if not self._enabled:
            return
        if not self._metrics_defined:
            self._define_metrics()
        payload = {}
        for key, value in metrics.items():
            payload[f"{self._config.eval_prefix}{key}"] = value
        payload[self._step_metric_key] = step
        self._wandb_run.log(payload)

    def _define_metrics(self) -> None:
        """Define GPU metrics with train/global_step as x-axis.

        Deferred until first log to ensure it runs AFTER HuggingFace's WandbCallback
        calls wandb.define_metric("*", step_metric="train/global_step").
        """
        if not self._enabled:
            return
        for prefix in [self._config.train_prefix, self._config.eval_prefix]:
            logging_common.deferred_define_metric(
                self._wandb_run,
                f"{prefix}*",
                step_metric=self._step_metric_key,
                summary="last",
            )
        self._metrics_defined = True
