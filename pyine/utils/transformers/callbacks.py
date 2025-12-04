import collections.abc
import logging
import typing

import transformers

import pyine.utils.transformers.checkpoints

logger = logging.getLogger(__name__)

__all__ = [
    "StdoutMilestones",
    "EpochAwarenessCallback",
    "create_epoch_awareness_callback",
]


class StdoutMilestones(transformers.TrainerCallback):
    """Prints training/eval milestones for the HuggingFace Trainer.

    These milestones will be logged, but only on the main process (unless configured otherwise):
      - init/train begin/end;
      - epoch begin/end;
      - periodic logs (loss, lr, etc.);
      - evaluation/prediction metrics;
      - checkpoint saves.

    The purpose of this class is to provide a simple way to monitor and log training progress
    without changing the default Hugging Face Trainer logging behavior.

    Args:
        only_main_process: If True, prints only when args.process_index == 0.
        print_fn: Callable used for printing messages. Defaults to logging.info.
        print_config_at_start: If True, prints a short run/config header at train start.
    """

    def __init__(
        self,
        *,
        only_main_process: bool = True,
        print_fn: collections.abc.Callable[[str], None] | None = None,
        print_config_at_start: bool = True,
    ) -> None:
        """Initialize the callback."""
        self.only_main_process = only_main_process
        self.print_fn = print_fn if print_fn is not None else logging.info
        self.print_config_at_start = print_config_at_start

    # -------------------------- internal helpers --------------------------

    def _should_print(self, args: transformers.TrainingArguments) -> bool:
        """Check if output should be printed on this process."""
        return (not self.only_main_process) or (typing.cast("int", getattr(args, "process_index", 0)) == 0)

    @staticmethod
    def _fmt_epoch(epoch: float | None) -> str:
        """Format epoch number for display."""
        return "NA" if epoch is None else f"{epoch:.6g}"

    @staticmethod
    def _fmt_dict(d: collections.abc.Mapping[str, object]) -> str:
        """Format dictionary as compact key=value pairs."""

        def _fmt_val(v: object) -> str:
            if isinstance(v, float):
                return f"{v:.6g}"
            return str(v)

        return ", ".join(f"{k}={_fmt_val(d[k])}" for k in sorted(d.keys()))

    # -------------------------- callback methods --------------------------

    @typing.override
    def on_init_end(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **_: typing.Any,
    ) -> None:
        """Called at the end of trainer initialization."""
        if self._should_print(args):
            self.print_fn("init_end")

    @typing.override
    def on_train_begin(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **_: typing.Any,
    ) -> None:
        """Called at the beginning of training."""
        if not self._should_print(args):
            return
        self.print_fn("train_begin")
        if self.print_config_at_start:
            self.print_fn(f"run_name={args.run_name}, output_dir={args.output_dir}")
            self.print_fn(
                f"epochs={args.num_train_epochs}, max_steps={args.max_steps}, "
                f"grad_accum={args.gradient_accumulation_steps}"
            )
            self.print_fn(
                f"train_bs_per_device={args.per_device_train_batch_size}, "
                f"eval_bs_per_device={args.per_device_eval_batch_size}"
            )
            self.print_fn(
                f"fp16={typing.cast('bool', getattr(args, 'fp16', False))}, "
                f"bf16={typing.cast('bool', getattr(args, 'bf16', False))}, "
                f"world_size={typing.cast('int', getattr(args, 'world_size', 1))}, "
                f"local_rank={typing.cast('int', getattr(args, 'local_rank', -1))}, "
                f"n_gpu={typing.cast('int', getattr(args, 'n_gpu', 0))}"
            )
            self.print_fn(
                f"logging_strategy={args.logging_strategy}, logging_steps={args.logging_steps}, "
                f"eval_strategy={args.eval_strategy}, save_strategy={args.save_strategy}"
            )
            self.print_fn(f"learning_rate={args.learning_rate}, warmup_steps={args.warmup_steps}")

    @typing.override
    def on_epoch_begin(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **_: typing.Any,
    ) -> None:
        """Called at the beginning of each epoch."""
        if self._should_print(args):
            self.print_fn(f"epoch_begin; epoch={self._fmt_epoch(state.epoch)}, global_step={state.global_step}")

    @typing.override
    def on_epoch_end(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **_: typing.Any,
    ) -> None:
        """Called at the end of each epoch."""
        if self._should_print(args):
            self.print_fn(f"epoch_end; epoch={self._fmt_epoch(state.epoch)}, global_step={state.global_step}")

    @typing.override
    def on_log(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs: typing.Any,
    ) -> None:
        """Called when logging occurs during training."""
        assert "logs" in kwargs and isinstance(kwargs["logs"], collections.abc.Mapping)
        logs = typing.cast("collections.abc.Mapping[str, float | int | str | bool]", kwargs["logs"])
        if self._should_print(args) and logs:
            self.print_fn(f"log; step={state.global_step}, " + self._fmt_dict(logs))

    @typing.override
    def on_evaluate(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs: typing.Any,
    ) -> None:
        """Called after evaluation."""
        assert "metrics" in kwargs and isinstance(kwargs["metrics"], collections.abc.Mapping)
        metrics = typing.cast("collections.abc.Mapping[str, float | int]", kwargs["metrics"])
        if self._should_print(args) and metrics:
            self.print_fn(
                f"evaluate; step={state.global_step}, epoch={self._fmt_epoch(state.epoch)}, " + self._fmt_dict(metrics)
            )

    @typing.override
    def on_predict(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        metrics: collections.abc.Mapping[str, float | int],
        **_: typing.Any,
    ) -> None:
        """Called after prediction."""
        if self._should_print(args) and metrics:
            self.print_fn(
                f"predict; step={state.global_step}, epoch={self._fmt_epoch(state.epoch)}, " + self._fmt_dict(metrics)
            )

    @typing.override
    def on_save(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **_: typing.Any,
    ) -> None:
        """Called when saving a checkpoint."""
        if not self._should_print(args):
            return
        ckpt_dir_path = pyine.utils.transformers.checkpoints.get_checkpoint_folder_path(args, state)
        self.print_fn(f"save; checkpoint_dir={ckpt_dir_path.absolute()}")

    @typing.override
    def on_train_end(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **_: typing.Any,
    ) -> None:
        """Called at the end of training."""
        if self._should_print(args):
            self.print_fn(
                f"train_end; steps={state.global_step}, "
                f"best_metric={state.best_metric}, "
                f"best_model_checkpoint={state.best_model_checkpoint}"
            )


class EpochAwarenessCallback(transformers.TrainerCallback):
    """Forward trainer epoch values to registered datasets/parsers.

    The callback intentionally keeps no custom state. Instead, it reads the current epoch from
    ``TrainerState`` every time a hook fires and relays that integer value to each target's
    ``set_epoch`` method (or to explicit setter callables).
    """

    def __init__(
        self,
        *,
        epoch_targets: collections.abc.Iterable[typing.Any] | None = None,
        epoch_setters: collections.abc.Iterable[collections.abc.Callable[[int], None]] | None = None,
    ) -> None:
        """Normalize the provided targets/setters immediately for quick dispatch later.

        Args:
            epoch_targets: Objects exposing ``set_epoch(int)`` that should receive epoch updates.
            epoch_setters: Callables taking a single ``int`` epoch argument, allowing indirection when
                a public ``set_epoch`` is not available or desirable.
        """
        self._epoch_setters: list[collections.abc.Callable[[int], None]] = []
        if epoch_setters is not None:
            for setter in epoch_setters:
                if not callable(setter):
                    raise TypeError("epoch_setters entries must be callables")
                self._epoch_setters.append(setter)
        if epoch_targets is not None:
            for target in epoch_targets:
                setter = getattr(target, "set_epoch", None)
                if setter is None or not callable(setter):
                    continue
                self._epoch_setters.append(typing.cast("collections.abc.Callable[[int], None]", setter))

    @staticmethod
    def _coerce_epoch(epoch_value: float | int | None) -> int:
        """Return a non-negative integer epoch derived from the Trainer state."""
        if epoch_value is None:
            return 0
        if isinstance(epoch_value, float):
            if epoch_value < 0:
                return 0
            return int(epoch_value)
        return max(0, int(epoch_value))

    def _propagate_epoch(self, epoch_value: int, hook: str) -> None:
        """Send the epoch value to every registered setter and log the hook context."""
        if not self._epoch_setters:
            return
        logger.debug(f"epoch-aware callback setting epoch={epoch_value} via {hook}")
        for setter in self._epoch_setters:
            setter(epoch_value)

    @typing.override
    def on_train_begin(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **_: typing.Any,
    ) -> None:
        """Propagate the trainer's current epoch before the first training step."""
        self._propagate_epoch(self._coerce_epoch(state.epoch), hook="train_begin")

    @typing.override
    def on_epoch_begin(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **_: typing.Any,
    ) -> None:
        """Update registered targets at the start of every epoch."""
        self._propagate_epoch(self._coerce_epoch(state.epoch), hook="epoch_begin")


def create_epoch_awareness_callback(
    train_dataset: typing.Any,
    datamodule: typing.Any,
    subset_names: collections.abc.Iterable[str] | None,
) -> EpochAwarenessCallback | None:
    """Build an EpochAwarenessCallback for the provided dataset/datamodule if needed.

    Args:
        train_dataset: Dataset passed to the HuggingFace Trainer; used when it exposes ``set_epoch``.
        datamodule: DataModule capable of returning parsers via ``get_parser``.
        subset_names: Iterable of subset names to inspect on the datamodule (typically the train
            subsets); entries without a reachable parser are skipped.

    Returns:
        An `EpochAwarenessCallback` when at least one target supports `set_epoch`, otherwise `None`.
    """
    seen_target_ids: set[int] = set()
    epoch_targets: list[typing.Any] = []

    def _register_candidate(candidate: typing.Any) -> None:
        if candidate is None:
            return
        setter = getattr(candidate, "set_epoch", None)
        if setter is None or not callable(setter):
            return
        candidate_id = id(candidate)
        if candidate_id in seen_target_ids:
            return
        seen_target_ids.add(candidate_id)
        epoch_targets.append(candidate)

    _register_candidate(train_dataset)
    if datamodule is not None and hasattr(datamodule, "get_parser"):
        for subset_name in subset_names or []:
            try:
                parser = datamodule.get_parser(subset_name)
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"skipping epoch-aware parser hookup for subset {subset_name}: {exc}")
                continue
            _register_candidate(parser)
    if not epoch_targets:
        return None
    return EpochAwarenessCallback(epoch_targets=epoch_targets)
