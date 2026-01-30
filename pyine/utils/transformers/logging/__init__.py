"""Direct wandb logging for trainer metrics.

This subpackage provides standalone loggers that bypass the trainer's logging lifecycle by calling wandb.run.log()
directly. This is necessary because HuggingFace's TRL Trainer.log() saves metrics to log_history BEFORE calling on_log
callbacks, making callback injection ineffective for wandb logging in TRL trainers.
"""

from pyine.utils.transformers.logging.common import DEFAULT_STEP_METRIC_KEY, deferred_define_metric
from pyine.utils.transformers.logging.gpu_stats import GPUStatsLogger
from pyine.utils.transformers.logging.throughput import ThroughputLogger

__all__ = [
    "DEFAULT_STEP_METRIC_KEY",
    "ThroughputLogger",
    "GPUStatsLogger",
    "deferred_define_metric",
]
