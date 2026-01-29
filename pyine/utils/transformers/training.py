"""Shared training utilities and helpers for HuggingFace Trainer usage.

This module provides reusable components for e.g. callbacks that need to track train/eval phase
transitions and compute batch sizes.
"""

import typing

import transformers

__all__ = [
    "EvalPhaseTracker",
    "get_effective_train_batch_size",
    "get_effective_eval_batch_size",
]


class EvalPhaseTracker:
    """Shared phase tracking for train/eval transitions.

    Callbacks should call methods from their hooks and check return values for
    warnings/diagnostics when HF callback ordering changes unexpectedly.

    Key semantics:
    - `_saw_training`: Set in `handle_step_or_epoch_end()`, NOT in `on_train_begin`.
      This ensures eval_on_start doesn't trigger "unexpected eval" warnings.
      Note: Set by both on_step_end and on_epoch_end (both indicate training progress).
    - `_eval_pending`: Set when `should_evaluate=True`, cleared on first prediction step
      or when `mark_entering_eval()` is called.
    """

    def __init__(self) -> None:
        """Initialize the tracker with all flags in their default (train) state."""
        self._in_eval: bool = False
        self._eval_pending: bool = False
        self._saw_eval_prediction_step: bool = False
        self._saw_training: bool = False  # set on first completed training step

    def handle_step_or_epoch_end(
        self,
        control: transformers.TrainerControl,
    ) -> bool:
        """Call from on_step_end/on_epoch_end.

        This marks that at least one training step has completed, which enables
        "unexpected eval" warnings. Also detects eval transitions.

        Args:
            control: The trainer control object containing should_evaluate flag.

        Returns:
            True if transitioning to eval (control.should_evaluate is True).
        """
        self._saw_training = True  # training step completed
        if control.should_evaluate:
            self._eval_pending = True
            self._saw_eval_prediction_step = False
            return True
        return False

    def mark_entering_eval(self) -> bool:
        """Mark explicit entry into eval phase (e.g., for _switch_to_eval).

        Call this when entering eval without going through on_prediction_step, such as
        when flushing stats before eval starts. This prevents "unexpected eval" warnings
        when prediction steps follow.

        Returns:
            unexpected: True if we weren't expecting eval AND training was seen.
        """
        unexpected = not self._eval_pending and not self._in_eval and self._saw_training
        self._in_eval = True
        self._eval_pending = False
        self._saw_eval_prediction_step = False
        return unexpected

    def handle_prediction_step(self) -> tuple[bool, bool]:
        """Call from on_prediction_step.

        Returns:
            (first_step, unexpected):
                - first_step: True on first prediction step of this eval phase.
                - unexpected: True if we weren't expecting eval (eval_pending was False
                  and not already in_eval) AND training was seen. This suppresses warnings
                  for predict-only flows and eval_on_start.
        """
        first_step = not self._in_eval
        unexpected = first_step and not self._eval_pending and self._saw_training
        self._in_eval = True
        self._eval_pending = False
        self._saw_eval_prediction_step = True
        return first_step, unexpected

    def handle_evaluate(self) -> None:
        """Call from on_evaluate. Resets state for next train phase."""
        self._in_eval = False
        self._eval_pending = False
        self._saw_eval_prediction_step = False

    def is_eval_context(
        self,
        logs: dict[str, typing.Any] | None = None,
    ) -> bool:
        """Check if currently in eval context.

        Args:
            logs: Optional logs dict to check for eval-like keys.

        Returns:
            True if in eval context (either _in_eval is True, or eval_pending
            and logs contain eval-like keys).
        """
        if self._in_eval:
            return True
        if self._eval_pending and logs is not None:
            return any(k.startswith(("eval_", "eval/")) for k in logs)
        return False

    @property
    def in_eval(self) -> bool:
        """Whether currently in eval phase."""
        return self._in_eval

    @property
    def eval_pending(self) -> bool:
        """Whether eval is pending (should_evaluate was True)."""
        return self._eval_pending

    @property
    def saw_eval_samples(self) -> bool:
        """Whether at least one prediction step occurred in current eval."""
        return self._saw_eval_prediction_step

    @property
    def saw_training(self) -> bool:
        """Whether at least one training step has completed."""
        return self._saw_training


def get_effective_train_batch_size(
    args: transformers.TrainingArguments,
    *,
    include_world_size: bool = True,
) -> int:
    """Calculate effective train batch size.

    Args:
        args: The training arguments containing batch size settings.
        include_world_size: If True (default), multiply by world_size for global batch.
            If False, return per-device batch size only (for local throughput).

    Returns:
        The effective batch size per optimizer step.
    """
    per_device = typing.cast("int", getattr(args, "per_device_train_batch_size", 1))
    world_size = typing.cast("int", getattr(args, "world_size", 1)) if include_world_size else 1
    grad_accum = typing.cast("int", getattr(args, "gradient_accumulation_steps", 1))
    return per_device * world_size * grad_accum


def get_effective_eval_batch_size(
    args: transformers.TrainingArguments,
    *,
    include_world_size: bool = True,
) -> int:
    """Calculate effective eval batch size.

    Args:
        args: The training arguments containing batch size settings.
        include_world_size: If True (default), multiply by world_size for global batch.
            If False, return per-device batch size only (for local throughput).

    Returns:
        The effective batch size per eval step.
    """
    per_device = typing.cast("int", getattr(args, "per_device_eval_batch_size", 1))
    world_size = typing.cast("int", getattr(args, "world_size", 1)) if include_world_size else 1
    return per_device * world_size
