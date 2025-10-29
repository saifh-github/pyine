"""Utilities to coordinate graceful shutdown of long-running training jobs."""

from __future__ import annotations

import dataclasses
import logging
import signal
import threading
import time
import typing

import transformers

import pyine.utils.distrib
import pyine.utils.transformers

if typing.TYPE_CHECKING:
    import pathlib

logger = logging.getLogger(__name__)

__all__ = [
    "GracefulShutdownCallback",
    "GracefulShutdownManager",
    "ShutdownRequest",
]


@dataclasses.dataclass(slots=True, frozen=True)
class ShutdownRequest:
    """Structured payload describing a shutdown request."""

    reason: str
    """Reason for the shutdown request."""
    requested_at: float
    """Timestamp when the shutdown request was made (from monotonic clock)."""
    deadline_at: float | None
    """Timestamp before which the shutdown should be completed (from monotonic clock)."""

    def time_remaining(self) -> float | None:
        """Return the remaining seconds until deadline, if any."""
        if self.deadline_at is None:
            return None
        return max(0.0, self.deadline_at - time.monotonic())


class GracefulShutdownManager:
    """Manage signal-triggered graceful shutdowns for distributed training jobs."""

    def __init__(
        self,
        *,
        expected_notice_seconds: float | None = 60.0,
        log: logging.Logger | None = None,
    ) -> None:
        """Initialize the manager.

        Args:
            expected_notice_seconds: Estimated delay between SIGTERM and SIGKILL. Used to derive
                a tentative deadline; can be overridden per-request.
            log: Logger to emit status updates; defaults to the module logger.
        """
        self._expected_notice_seconds = expected_notice_seconds
        self._log = log or logger
        self._shutdown_event = threading.Event()
        self._shutdown_request: ShutdownRequest | None = None
        self._lock = threading.Lock()
        self._handlers_installed = False
        self._original_handlers: dict[int, typing.Any] = {}

    def __enter__(self) -> GracefulShutdownManager:
        """Enter the context manager."""
        self.register_signal_handlers()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: typing.Any,
    ) -> None:
        """Exit the context manager."""
        self.restore_signal_handlers()

    @property
    def shutdown_event(self) -> threading.Event:
        """Return the underlying flag set when a graceful shutdown was requested."""
        return self._shutdown_event

    @property
    def shutdown_request(self) -> ShutdownRequest | None:
        """Return the last recorded shutdown request, if any."""
        return self._shutdown_request

    def register_signal_handlers(self) -> None:
        """Install SIGINT/SIGTERM handlers on the main process."""
        if self._handlers_installed:
            return
        if not pyine.utils.distrib.is_main_process():
            self._log.debug("skipping signal handler registration on non-main process")
            return
        handled_signals = (signal.SIGINT, signal.SIGTERM)
        for sig in handled_signals:
            previous = signal.getsignal(sig)
            self._original_handlers[sig] = previous
            signal.signal(sig, self._handle_signal)
        self._handlers_installed = True
        self._log.debug("registered graceful shutdown signal handlers")

    def restore_signal_handlers(self) -> None:
        """Restore previous signal handlers after training ends."""
        if not self._handlers_installed:
            return
        for sig, previous in self._original_handlers.items():
            signal.signal(sig, previous)
        self._original_handlers.clear()
        self._handlers_installed = False
        self._log.debug("restored original signal handlers")

    def request_shutdown(
        self,
        *,
        reason: str,
        deadline_seconds: float | None = None,
    ) -> ShutdownRequest:
        """Mark a graceful shutdown request."""
        requested_at = time.monotonic()
        if deadline_seconds is None:
            deadline_seconds = self._expected_notice_seconds
        deadline_at = requested_at + deadline_seconds if deadline_seconds is not None else None
        shutdown_request = ShutdownRequest(
            reason=reason,
            requested_at=requested_at,
            deadline_at=deadline_at,
        )
        with self._lock:
            if not self._shutdown_event.is_set():
                self._log.warning(f"graceful shutdown requested: {reason}")
            self._shutdown_event.set()
            self._shutdown_request = shutdown_request
        return shutdown_request

    def acknowledge_peer_shutdown(
        self,
        request: ShutdownRequest,
    ) -> None:
        """Mirror a shutdown request received from a peer."""
        with self._lock:
            if not self._shutdown_event.is_set():
                self._log.warning(f"peer requested graceful shutdown: {request.reason}")
            self._shutdown_event.set()
            self._shutdown_request = request

    def as_serializable_payload(self) -> dict[str, float | str | None]:
        """Return a payload that can be broadcast to other ranks."""
        request = self._shutdown_request
        if request is None:
            return {}
        return {
            "reason": request.reason,
            "requested_at": request.requested_at,
            "deadline_at": request.deadline_at,
        }

    def load_from_payload(
        self,
        payload: dict[str, float | str | None],
    ) -> None:
        """Populate state from a broadcast payload."""
        if not payload:
            return
        reason = typing.cast("str | None", payload.get("reason"))
        requested_at = typing.cast("float | None", payload.get("requested_at"))
        deadline_at = typing.cast("float | None", payload.get("deadline_at"))
        if reason is None or requested_at is None:
            return
        self.acknowledge_peer_shutdown(
            ShutdownRequest(
                reason=reason,
                requested_at=requested_at,
                deadline_at=deadline_at,
            ),
        )

    def should_terminate(self) -> bool:
        """Return True when a graceful shutdown was requested."""
        return self._shutdown_event.is_set()

    def _handle_signal(
        self,
        signum: int,
        _: typing.Any,
    ) -> None:
        """Signal handler for SIGINT/SIGTERM."""
        signal_name = signal.Signals(signum).name if signum in signal.Signals else str(signum)
        self.request_shutdown(reason=f"received {signal_name}")


class GracefulShutdownCallback(transformers.TrainerCallback):
    """Trainer callback that coordinates graceful shutdown and checkpointing."""

    def __init__(
        self,
        *,
        shutdown_manager: GracefulShutdownManager,
        metadata_writer: typing.Callable[[pathlib.Path, transformers.TrainerState], None],
    ) -> None:
        """Initialize the callback."""
        self._shutdown_manager = shutdown_manager
        self._metadata_writer = metadata_writer
        self._finalized = False

    @typing.override
    def on_train_begin(  # type: ignore[override]
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs: typing.Any,
    ) -> transformers.TrainerControl:
        """Event called at the beginning of training."""
        self._sync_shutdown_payload()
        return control

    @typing.override
    def on_step_end(  # type: ignore[override]
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs: typing.Any,
    ) -> transformers.TrainerControl:
        """Event called after each training step."""
        return self._maybe_finalize(args, state, control)

    @typing.override
    def on_evaluate(  # type: ignore[override]
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs: typing.Any,
    ) -> transformers.TrainerControl:
        """Event called after each evaluation."""
        return self._maybe_finalize(args, state, control)

    @typing.override
    def on_save(  # type: ignore[override]
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs: typing.Any,
    ) -> transformers.TrainerControl:
        """Writes checkpoint metadata after each checkpoint save."""
        if not pyine.utils.distrib.is_main_process():
            return control
        # the callback should be called AFTER the creation of the checkpoint, so we know it should exist at this point
        ckpt_dir_path = pyine.utils.transformers.get_checkpoint_folder_path(args, state)
        if not ckpt_dir_path.is_dir():
            raise RuntimeError(f"checkpoint directory does not exist: {ckpt_dir_path}")
        self._metadata_writer(ckpt_dir_path, state)
        return control

    def _maybe_finalize(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
    ) -> transformers.TrainerControl:
        """Finalize training if a shutdown request was made."""
        self._sync_shutdown_payload()
        if not self._shutdown_manager.should_terminate():
            return control
        if pyine.utils.distrib.is_main_process() and not self._finalized:
            shutdown_request = self._shutdown_manager.shutdown_request
            reason = shutdown_request.reason if shutdown_request is not None else "external request"
            logger.warning(f"stopping training early after shutdown request: {reason}")
        control.should_training_stop = True
        control.should_epoch_stop = True
        control.should_log = True
        control.should_evaluate = False
        if not self._finalized:
            control.should_save = True
            self._finalized = True
        else:
            control.should_save = False
        return control

    def _sync_shutdown_payload(self) -> None:
        """Broadcast shutdown payload to other ranks."""
        if not pyine.utils.distrib.is_distributed():
            return
        payload = self._shutdown_manager.as_serializable_payload() if pyine.utils.distrib.is_main_process() else {}
        synced_payload = typing.cast(
            "dict[str, float | str | None]",
            pyine.utils.distrib.broadcast_object(payload, src=0),
        )
        if pyine.utils.distrib.is_main_process():
            return
        self._shutdown_manager.load_from_payload(synced_payload)
