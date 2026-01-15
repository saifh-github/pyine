import collections.abc
import json
import logging
import pathlib
import time
import typing

import transformers

import pyine.utils.distrib
import pyine.utils.reprod
import pyine.utils.transformers.checkpoints

logger = logging.getLogger(__name__)

__all__ = [
    "StdoutMilestones",
    "EpochAwarenessCallback",
    "RewardLoggingCallback",
    "ThroughputLoggingCallback",
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

    def _should_print(self) -> bool:
        """Check if output should be printed on this process."""
        if not self.only_main_process:
            return True
        return pyine.utils.distrib.is_main_process()

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
        if self._should_print():
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
        if not self._should_print():
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
        if self._should_print():
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
        if self._should_print():
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
        if self._should_print() and logs:
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
        if self._should_print() and metrics:
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
        if self._should_print() and metrics:
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
        if not self._should_print():
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
        if self._should_print():
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


class RewardLoggingCallback(transformers.TrainerCallback):
    """Callback that manages RewardManager prefix switching and stats flushing.

    This callback integrates with TRL's training loop to:
    - Switch the RewardManager's key prefix between train/eval phases
    - Flush accumulated reward statistics at phase transitions and train end

    The prefix switching enables differentiation of reward logs during training vs evaluation:
    - Training steps: logs to "{train_prefix}/..."
    - Evaluation steps: logs to "{eval_prefix}/..."

    Example usage:
        ```python
        reward_manager = RewardManager(config)
        callback = RewardLoggingCallback(reward_manager=reward_manager)
        trainer.add_callback(callback)
        ```
    """

    def __init__(
        self,
        reward_manager: typing.Any,
        *,
        reward_adapter: typing.Any | None = None,
        train_prefix: str = "train",
        eval_prefix: str = "eval",
        resume_from_checkpoint: str | pathlib.Path | None = None,
    ) -> None:
        """Initialize the callback.

        Args:
            reward_manager: The RewardManager instance to manage. Should have `set_key_prefix`,
                `flush_stats`, `get_state`, and `load_state` methods.
            reward_adapter: Optional TRLRewardAdapter instance for failure tracking and state
                persistence. Should have `get_failure_stats`, `reset_failure_stats`, `get_state`,
                and `load_state` methods.
            train_prefix: Prefix to use for training phase logs (default: "train").
            eval_prefix: Prefix to use for evaluation phase logs (default: "eval").
            resume_from_checkpoint: Path to checkpoint directory to resume from. When provided,
                reward state will be loaded from this checkpoint on train begin.
        """
        assert hasattr(reward_manager, "set_key_prefix"), "reward manager missing 'set_key_prefix'"
        assert callable(reward_manager.set_key_prefix)
        assert hasattr(reward_manager, "flush_stats"), "reward manager missing 'flush_stats'"
        assert callable(reward_manager.flush_stats)
        assert hasattr(reward_manager, "get_state"), "reward manager missing 'get_state'"
        assert callable(reward_manager.get_state)
        assert hasattr(reward_manager, "load_state"), "reward manager missing 'load_state'"
        assert callable(reward_manager.load_state)
        if reward_adapter is not None:
            assert hasattr(reward_adapter, "get_state"), "reward adapter missing 'get_state'"
            assert callable(reward_adapter.get_state)
            assert hasattr(reward_adapter, "load_state"), "reward adapter missing 'load_state'"
            assert callable(reward_adapter.load_state)
        self.reward_manager = reward_manager
        self.reward_adapter = reward_adapter
        self.train_prefix = train_prefix
        self.eval_prefix = eval_prefix
        self.resume_from_checkpoint = pathlib.Path(resume_from_checkpoint) if resume_from_checkpoint else None
        self._in_eval: bool | None = None
        self._saw_eval_prediction_step: bool = False

    def _log_failure_stats(self, step: int | None) -> None:
        """Log failure statistics from reward adapter if available."""
        if self.reward_adapter is None:
            return
        if not hasattr(self.reward_adapter, "get_failure_stats") or not hasattr(
            self.reward_adapter, "reset_failure_stats"
        ):
            return
        stats = self.reward_adapter.get_failure_stats()
        total_count = stats["total_count"]
        if total_count == 0:
            return
        skip_count = stats["skip_count"]
        error_count = stats["error_count"]
        failure_count = skip_count + error_count
        failure_ratio = failure_count / total_count
        logger = getattr(self.reward_manager, "_logger", None)
        if logger is not None and hasattr(logger, "log_failures"):
            logger.log_failures(
                failure_ratio=failure_ratio,
                failure_count=failure_count,
                step=step,
            )
        self.reward_adapter.reset_failure_stats()

    def _switch_to_eval(self, step: int | None = None) -> None:
        """Switch to eval prefix, flush train stats, and clear step.

        This method handles the train -> eval transition by flushing accumulated training stats
        under the current (train) prefix before switching to eval mode. It should be called BEFORE
        evaluation starts to ensure the first eval batch logs correctly.

        Args:
            step: Optional step to anchor the flush (defaults to current manager step).
        """
        if self._in_eval:
            return
        self._log_failure_stats(step=step)
        self.reward_manager.flush_stats(step=step)
        self.reward_manager.set_key_prefix(self.eval_prefix)
        self.reward_manager.set_step(None)  # clear step for eval
        self._in_eval = True
        self._saw_eval_prediction_step = False  # reset for new eval phase

    @typing.override
    def on_step_begin(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs: typing.Any,
    ) -> None:
        """Set train prefix and logging step at the start of each training step."""
        if self._in_eval is not False:
            self.reward_manager.set_key_prefix(self.train_prefix)
            self._in_eval = False
        self.reward_manager.set_step(state.global_step)

    @typing.override
    def on_step_end(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs: typing.Any,
    ) -> None:
        """Switch to eval prefix before step-based evaluation.

        DefaultFlowCallback sets control.should_evaluate = True in its own on_step_end, and Trainer
        calls _maybe_log_save_evaluate() immediately after all callbacks' on_step_end. By switching
        here, we ensure the first eval batch logs under the correct prefix.
        """
        if control.should_evaluate:
            self._switch_to_eval(step=state.global_step)

    @typing.override
    def on_prediction_step(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs: typing.Any,
    ) -> None:
        """Set eval prefix and clear step during evaluation.

        Note: this hook fires AFTER prediction_step() returns, so the first eval batch has already
        been processed. We do NOT flush here because:
        - If pre-eval switching worked (on_step_end/on_epoch_end), we're already in eval;
        - If pre-eval switching was missed, flushing now would flush a mixed train+eval
          accumulator under the wrong context.

        We only set the prefix (no flush) to ensure subsequent batches log correctly.
        """
        if self._in_eval is not True:
            # woops... unexpected; don't call _switch_to_eval(); no flush, just set prefix.
            self.reward_manager.set_key_prefix(self.eval_prefix)
            self._in_eval = True
        self._saw_eval_prediction_step = True  # mark that we processed at least one eval batch
        # clear step for per-sample eval logs; global_step doesn't change during eval, so logging
        # multiple samples at the same step would cause WandB to overwrite scalar metrics
        self.reward_manager.set_step(None)

    @typing.override
    def on_epoch_end(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs: typing.Any,
    ) -> None:
        """Switch to eval prefix before epoch-based evaluation.

        DefaultFlowCallback sets control.should_evaluate = True in its own on_epoch_end for
        eval_strategy="epoch". Trainer evaluates immediately after on_epoch_end, so we switch
        here to ensure the first eval batch logs under the correct prefix.
        """
        if control.should_evaluate:
            self._switch_to_eval(step=state.global_step)

    @typing.override
    def on_evaluate(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs: typing.Any,
    ) -> None:
        """Flush accumulated stats and reset to train prefix after evaluation completes."""
        # only flush if we actually processed eval samples (on_prediction_step was called);
        # if eval had zero samples, skip flushing to avoid logging stale stats under eval prefix
        if self._saw_eval_prediction_step:
            # anchor eval summary at current global_step so it aligns with training metrics
            self._log_failure_stats(step=state.global_step)
            self.reward_manager.flush_stats(step=state.global_step)
        self._saw_eval_prediction_step = False  # reset for next eval
        self.reward_manager.set_key_prefix(self.train_prefix)
        self._in_eval = False  # evaluation done, switch back right away

    @typing.override
    def on_train_begin(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs: typing.Any,
    ) -> None:
        """Initialize prefix and load reward state from checkpoint if resuming training.

        This method follows a specific order to ensure correct prefix/accumulator semantics:
        1. Load checkpoint state first (if resuming), restoring step and accumulators;
        2. Always set train prefix and _in_eval = False (consistent starting point);
        3. If eval_on_start=True, transition train→eval properly via _switch_to_eval().

        Raises:
            FileNotFoundError: If resuming from checkpoint but reward_state.json is missing.
        """
        # 1. load checkpoint state FIRST (restores step and accumulators)
        if self.resume_from_checkpoint is not None:
            reward_state_path = self.resume_from_checkpoint / "reward_state.json"
            if not reward_state_path.exists():
                raise FileNotFoundError(
                    f"reward_state.json not found in checkpoint: {self.resume_from_checkpoint}; "
                    "checkpoint may be incomplete or from an older version"
                )
            self._load_reward_state(reward_state_path)
        # 2. always start in train mode (prefix + state)
        self.reward_manager.set_key_prefix(self.train_prefix)
        self._in_eval = False
        # 3. if eval_on_start, transition train→eval properly (flush under train prefix, then switch)
        if getattr(args, "eval_on_start", False):
            self._switch_to_eval(step=state.global_step)

    @typing.override
    def on_save(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs: typing.Any,
    ) -> None:
        """Save reward state to checkpoint directory."""
        if not state.is_world_process_zero:
            return  # only save on main process to avoid race conditions
        # get checkpoint folder from kwargs (transformers 4.46+), fall back to utility
        checkpoint_folder = kwargs.get("checkpoint_folder")
        if checkpoint_folder is None:
            checkpoint_folder = pyine.utils.transformers.checkpoints.get_checkpoint_folder_path(args, state)
        checkpoint_path = pathlib.Path(checkpoint_folder)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint_path}")
        adapter_state: dict[str, typing.Any] = {}
        if self.reward_adapter is not None:
            adapter_state = self.reward_adapter.get_state()
        manager_state = self.reward_manager.get_state()
        reward_state = {
            "version": pyine.utils.reprod.get_framework_version(),
            "adapter": adapter_state,
            "manager": manager_state,
        }
        (checkpoint_path / "reward_state.json").write_text(json.dumps(reward_state, indent=2, sort_keys=True))

    @typing.override
    def on_train_end(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs: typing.Any,
    ) -> None:
        """Flush any remaining stats at the end of training."""
        self._log_failure_stats(step=None)
        self.reward_manager.flush_stats(step=None)

    def _load_reward_state(
        self,
        path: pathlib.Path,
    ) -> None:
        """Load reward state from checkpoint file.

        Args:
            path: Path to reward_state.json file.

        Raises:
            json.JSONDecodeError: If the file contains invalid JSON.
            OSError: If the file cannot be read.
            KeyError: If required keys are missing from the state.
        """
        state = json.loads(path.read_text())
        checkpoint_version = state["version"]
        current_version = pyine.utils.reprod.get_framework_version()
        logger.debug(f"loading reward state from checkpoint ({checkpoint_version=}, {current_version=})")
        if "adapter" not in state:
            raise KeyError("reward_state.json missing required 'adapter' key")
        if "manager" not in state:
            raise KeyError("reward_state.json missing required 'manager' key")
        if self.reward_adapter is not None:
            self.reward_adapter.load_state(state["adapter"])
            logger.debug("restored reward adapter state from checkpoint")
        self.reward_manager.load_state(state["manager"])
        logger.debug("restored reward manager state from checkpoint")


class ThroughputLoggingCallback(transformers.TrainerCallback):
    """Injects rolling throughput metrics into the trainer's log dict.

    Tracks elapsed compute time between `on_log` calls (excluding checkpoint I/O) and
    calculates throughput. Metrics are injected into the `logs` dict passed to `on_log`,
    so they flow to whatever reporters the trainer uses (W&B, TensorBoard, etc.).

    Injected metrics:
        - `{prefix}samples_per_second`: Training samples processed per second
        - `{prefix}steps_per_second`: Optimizer steps per second

    Args:
        only_main_process: If True, only injects metrics on the main process (rank 0).
        prefix: Prefix for metric keys.
    """

    def __init__(
        self,
        *,
        only_main_process: bool = True,
        prefix: str = "throughput/",
    ) -> None:
        self.only_main_process = only_main_process
        self.prefix = prefix
        self._last_log_time: float | None = None
        self._last_log_step: int = 0

    def _should_log(self) -> bool:
        """Check if metrics should be logged on this process."""
        if not self.only_main_process:
            return True
        return pyine.utils.distrib.is_main_process()

    def _get_effective_batch_size(
        self,
        args: transformers.TrainingArguments,
    ) -> int:
        """Calculate effective batch size accounting for distributed training and gradient accumulation."""
        per_device_batch_size = typing.cast("int", getattr(args, "per_device_train_batch_size", 1))
        world_size = typing.cast("int", getattr(args, "world_size", 1))
        grad_accum_steps = typing.cast("int", getattr(args, "gradient_accumulation_steps", 1))
        return per_device_batch_size * world_size * grad_accum_steps

    @typing.override
    def on_train_begin(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **_: typing.Any,
    ) -> None:
        """Initialize timing state at the start of training."""
        self._last_log_time = time.perf_counter()
        self._last_log_step = state.global_step

    @typing.override
    def on_log(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs: typing.Any,
    ) -> None:
        """Calculate and inject throughput metrics whenever the trainer logs."""
        if not self._should_log():
            return
        if self._last_log_time is None:
            self._last_log_time = time.perf_counter()  # on_train_begin wasn't called
            self._last_log_step = state.global_step
            return
        current_time = time.perf_counter()
        elapsed = current_time - self._last_log_time
        steps_delta = state.global_step - self._last_log_step
        if elapsed <= 0 or steps_delta <= 0:
            self._last_log_time = current_time  # avoid division by zero
            self._last_log_step = state.global_step
            return
        effective_batch_size = self._get_effective_batch_size(args)
        samples_delta = steps_delta * effective_batch_size
        samples_per_second = samples_delta / elapsed
        steps_per_second = steps_delta / elapsed
        # inject metrics into the logs dict so they flow to whatever reporters are configured
        logs = kwargs.get("logs")
        if logs is not None and isinstance(logs, dict):
            logs[f"{self.prefix}samples_per_second"] = samples_per_second
            logs[f"{self.prefix}steps_per_second"] = steps_per_second
        # update state for next interval
        self._last_log_time = current_time
        self._last_log_step = state.global_step

    @typing.override
    def on_save(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **_: typing.Any,
    ) -> None:
        """Reset timing after checkpoint save to exclude I/O overhead from throughput."""
        self._last_log_time = time.perf_counter()  # exclude checkpoint I/O from next interval
