import collections.abc
import dataclasses
import json
import logging
import pathlib
import time
import typing

import pydantic
import torch
import transformers

import pyine.utils.distrib
import pyine.utils.gpu
import pyine.utils.reprod
import pyine.utils.stats
import pyine.utils.transformers.checkpoints
import pyine.utils.transformers.logging
import pyine.utils.transformers.training

logger = logging.getLogger(__name__)

__all__ = [
    "StdoutMilestones",
    "EpochAwarenessCallback",
    "RewardLoggingCallback",
    "ThroughputLoggingCallback",
    "ThroughputLoggingConfig",
    "GPUStatsLoggingCallback",
    "GPUStatsLoggingConfig",
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
            warmup = args.warmup_ratio if args.warmup_ratio else args.warmup_steps
            self.print_fn(f"learning_rate={args.learning_rate}, warmup={warmup}")

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
            except (KeyError, AttributeError) as exc:
                # expected: subset doesn't exist or parser not available for this subset
                logger.debug(f"skipping epoch-aware parser hookup for subset {subset_name}: {exc}")
                continue
            except Exception as exc:  # noqa: BLE001
                # unexpected error: warn so bugs aren't silently hidden
                logger.warning(
                    f"unexpected error getting parser for subset {subset_name}: {exc!r}; "
                    "epoch propagation may be disabled for this subset"
                )
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

    Step timing (verified against transformers 4.57 and trl 0.26):
        This callback sets the RewardManager's step in `on_step_begin` using `state.global_step`.
        In HuggingFace Trainer, the sequence is:
        1. `on_step_begin` called with `global_step = N`
        2. `training_step` executes (forward pass, reward computation happens here)
           - `global_step` is still N during this phase
        3. Optimizer step
        4. `global_step` incremented to `N+1`
        5. `on_step_end` called with `global_step = N+1`

        Example: First training step has `global_step = 0` at `on_step_begin`, rewards are
        computed with step=0, then `global_step` becomes 1, and `on_step_end` sees 1.

        By setting step in `on_step_begin`, reward events emitted during `training_step`
        are correctly attributed to the step being executed. This matches TRL's GRPO
        integration where the reward function is called inside `training_step` (via
        `_prepare_inputs` -> `_generate_and_score_completions` -> `_calculate_rewards`).

    Callback ordering dependency:
        This callback relies on HuggingFace's `DefaultFlowCallback` executing BEFORE this
        callback in `on_step_end` and `on_epoch_end`, because `DefaultFlowCallback` sets
        `control.should_evaluate` based on the current step/epoch. This ordering is guaranteed
        by default since `DefaultFlowCallback` is added first during Trainer initialization.
        If you remove or reorder `DefaultFlowCallback`, the eval-transition logic will not
        work correctly.
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
        self._phase = pyine.utils.transformers.training.EvalPhaseTracker()

    def _get_and_reset_failure_stats(self) -> tuple[float | None, int | None]:
        """Get failure statistics from reward adapter and reset them.

        Returns:
            Tuple of (failure_ratio, failure_count), or (None, None) if unavailable.
        """
        if self.reward_adapter is None:
            return None, None
        if not hasattr(self.reward_adapter, "get_failure_stats") or not hasattr(
            self.reward_adapter, "reset_failure_stats"
        ):
            return None, None
        stats = self.reward_adapter.get_failure_stats()
        total_count = stats["total_count"]
        if total_count == 0:
            return None, None
        skip_count = stats["skip_count"]
        error_count = stats["error_count"]
        failure_count = skip_count + error_count
        failure_ratio = failure_count / total_count
        self.reward_adapter.reset_failure_stats()
        return failure_ratio, failure_count

    def _switch_to_eval(self, step: int | None = None) -> None:
        """Switch to eval prefix, flush train stats, and clear step.

        This method handles the train -> eval transition by flushing accumulated training stats
        under the current (train) prefix before switching to eval mode. It should be called BEFORE
        evaluation starts to ensure the first eval batch logs correctly.

        Args:
            step: Optional step to anchor the flush (defaults to current manager step).
        """
        if self._phase.in_eval:
            return  # idempotence guard: already in eval
        failure_ratio, failure_count = self._get_and_reset_failure_stats()
        self.reward_manager.flush_stats(
            step=step,
            failure_ratio=failure_ratio,
            failure_count=failure_count,
        )
        self.reward_manager.set_key_prefix(self.eval_prefix)
        self.reward_manager.set_step(None)  # clear step for eval
        self._phase.mark_entering_eval()  # suppresses unexpected warnings when prediction steps follow

    @typing.override
    def on_step_begin(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs: typing.Any,
    ) -> None:
        """Set train prefix, logging step, and epoch at the start of each training step."""
        if self._phase.in_eval:
            self.reward_manager.set_key_prefix(self.train_prefix)
        self.reward_manager.set_step(state.global_step)
        self.reward_manager.set_epoch(state.epoch)

    @typing.override
    def on_step_end(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs: typing.Any,
    ) -> None:
        """Track training progress and switch to eval prefix before step-based evaluation.

        DefaultFlowCallback sets control.should_evaluate = True in its own on_step_end, and Trainer
        calls _maybe_log_save_evaluate() immediately after all callbacks' on_step_end. By switching
        here, we ensure the first eval batch logs under the correct prefix.
        """
        self._phase.handle_step_or_epoch_end(control)  # marks training as seen
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
        first_step, unexpected = self._phase.handle_prediction_step()
        if first_step:
            # first prediction step of this eval phase; _switch_to_eval wasn't called
            if unexpected:
                # unexpected: we should have switched to eval in on_step_end/on_epoch_end
                # this may indicate HF Trainer callback ordering changed, or eval was triggered
                # outside the normal flow; we can't safely flush, so just set prefix and warn
                logger.warning(
                    "on_prediction_step called without prior should_evaluate=True. This may indicate "
                    "HuggingFace Trainer callback ordering changed unexpectedly. The first eval "
                    "batch may have been logged under the wrong prefix. Please report this issue."
                )
            self.reward_manager.set_key_prefix(self.eval_prefix)
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
        """Track training progress and switch to eval prefix before epoch-based evaluation.

        DefaultFlowCallback sets control.should_evaluate = True in its own on_epoch_end for
        eval_strategy="epoch". Trainer evaluates immediately after on_epoch_end, so we switch
        here to ensure the first eval batch logs under the correct prefix.
        """
        self._phase.handle_step_or_epoch_end(control)  # marks training as seen
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
        if self._phase.saw_eval_samples:
            # anchor eval summary at current global_step so it aligns with training metrics
            failure_ratio, failure_count = self._get_and_reset_failure_stats()
            self.reward_manager.flush_stats(
                step=state.global_step,
                failure_ratio=failure_ratio,
                failure_count=failure_count,
            )
        self._phase.handle_evaluate()  # reset phase tracker
        self.reward_manager.set_key_prefix(self.train_prefix)

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
        2. Always set train prefix and reset phase tracker (consistent starting point);
        3. If eval_on_start=True, transition train->eval properly via _switch_to_eval().

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
        # 2. always start in train mode (prefix + phase tracker reset)
        self.reward_manager.set_key_prefix(self.train_prefix)
        # reset phase tracker in case callback instance is reused
        self._phase = pyine.utils.transformers.training.EvalPhaseTracker()
        # 3. if eval_on_start, transition train->eval properly (flush under train prefix, then switch)
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
        failure_ratio, failure_count = self._get_and_reset_failure_stats()
        self.reward_manager.flush_stats(
            step=None,
            failure_ratio=failure_ratio,
            failure_count=failure_count,
        )

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


class ThroughputLoggingConfig(pydantic.BaseModel):
    """Configuration for throughput logging callback.

    To disable throughput logging, set `throughput_logging: null` in your config (or don't set it
    at all). A non-null config means the callback is enabled.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    only_main_process: bool = True
    """Specifies whether only rank 0 logs metrics using global batch size.

    If False, all ranks log local throughput (per-device batch size only).

    Note: When False, metrics will be duplicated across ranks in W&B unless you configure
    rank-specific metric names or filter by rank.
    """
    log_train_throughput: bool = True
    """Whether to log train throughput metrics."""
    log_eval_throughput: bool = True
    """Whether to log eval throughput metrics."""
    train_prefix: str = "train/throughput/"
    """Prefix for train throughput metric keys."""
    eval_prefix: str = "eval/throughput/"
    """Prefix for eval throughput metric keys."""


class ThroughputLoggingCallback(transformers.TrainerCallback):
    """Logs throughput metrics directly to wandb using direct wandb.run.log() calls.

    Tracks elapsed compute time between `on_log` calls (excluding checkpoint I/O) and calculates
    throughput separately for train and eval phases. Uses ThroughputLogger to bypass the trainer's
    logging lifecycle (which, with TRL, saves to log_history BEFORE calling on_log callbacks).

    Logged metrics:
        - `{train_prefix}samples_per_second`: Training samples processed per second
        - `{train_prefix}steps_per_second`: Optimizer steps per second
        - `{eval_prefix}samples_per_second`: Eval samples processed per second

    Train throughput:
        - Measurement window resets on eval and save (steps completed before these events
          are excluded from the next throughput calculation if no log occurred in between)
        - Includes train on_log overhead (on_log comes after on_step_end in HF ordering)

    Eval throughput:
        - Normal flow: Timer starts in on_step_end when should_evaluate=True
        - eval_on_start flow: Timer starts in first on_prediction_step (fallback)
        - Skipped if no samples processed or predict-only run
        - Only logged during training runs (standalone trainer.evaluate() calls are not tracked)

    Args:
        config: Configuration for throughput logging.
        wandb_run: The wandb run object for direct logging.
    """

    def __init__(
        self,
        config: ThroughputLoggingConfig,
        wandb_run: typing.Any,
    ) -> None:
        """Initialize the callback.

        Args:
            config: Configuration for throughput logging.
            wandb_run: The wandb run object for direct logging.
        """
        self._config = config
        self._logger = pyine.utils.transformers.logging.ThroughputLogger(wandb_run, config)
        self._phase = pyine.utils.transformers.training.EvalPhaseTracker()
        self._saw_train_begin: bool = False  # distinguishes training vs predict-only
        # train timing
        self._train_last_log_time: float | None = None
        self._train_last_log_step: int = 0
        # eval timing
        self._eval_start_time: float | None = None
        self._eval_prediction_steps: int = 0

    def _should_log(self) -> bool:
        """Check if metrics should be logged on this process."""
        if not self._config.only_main_process:
            return True
        return pyine.utils.distrib.is_main_process()

    @typing.override
    def on_train_begin(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **_: typing.Any,
    ) -> None:
        """Initialize timing state at the start of training."""
        self._saw_train_begin = True
        # reset phase tracker in case callback instance is reused
        self._phase = pyine.utils.transformers.training.EvalPhaseTracker()
        # reset train timing
        self._train_last_log_time = time.perf_counter()
        self._train_last_log_step = state.global_step
        # reset eval timing
        self._eval_start_time = None
        self._eval_prediction_steps = 0

    @typing.override
    def on_step_end(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **_: typing.Any,
    ) -> None:
        """Handle eval transition and start eval timer."""
        transitioning = self._phase.handle_step_or_epoch_end(control)
        if transitioning and self._eval_start_time is None:  # guard against double reset
            # start eval timer HERE, so first batch time is included
            self._eval_start_time = time.perf_counter()
            self._eval_prediction_steps = 0

    @typing.override
    def on_epoch_end(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **_: typing.Any,
    ) -> None:
        """Handle eval transition at epoch end (same as on_step_end)."""
        transitioning = self._phase.handle_step_or_epoch_end(control)
        if transitioning and self._eval_start_time is None:  # guard against double reset
            self._eval_start_time = time.perf_counter()
            self._eval_prediction_steps = 0

    @typing.override
    def on_prediction_step(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **_: typing.Any,
    ) -> None:
        """Track eval samples and handle fallback timer start for eval_on_start."""
        _first_step, unexpected = self._phase.handle_prediction_step()
        # fallback for eval_on_start: start timer if not already started
        if self._eval_start_time is None:
            self._eval_start_time = time.perf_counter()
        self._eval_prediction_steps += 1
        if unexpected:
            logger.warning("Unexpected eval phase detected without should_evaluate flag")

    @typing.override
    def on_log(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        logs: dict[str, typing.Any] | None = None,
        **kwargs: typing.Any,
    ) -> None:
        """Calculate and log throughput metrics directly to wandb."""
        if not self._should_log():
            return
        if logs is None:
            logs = kwargs.get("logs")
        if not isinstance(logs, dict):
            return
        # determine context and log appropriate metrics
        if self._phase.is_eval_context(logs):
            self._log_eval_throughput(args, state)
        else:
            # warn if logs look like eval but we're not in eval context
            looks_like_eval = any(k.startswith(("eval_", "eval/")) for k in logs)
            if looks_like_eval:
                logger.warning(
                    "on_log received eval-like logs but not in eval context (_phase.in_eval=False, "
                    "_phase.eval_pending=False). This may indicate HuggingFace Trainer callback "
                    "ordering changed. Throughput metrics will be classified as train."
                )
            self._log_train_throughput(args, state)

    def _log_train_throughput(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
    ) -> None:
        """Log train throughput metrics directly to wandb if enabled and valid."""
        if not self._config.log_train_throughput:
            return
        if self._train_last_log_time is None:
            self._train_last_log_time = time.perf_counter()
            self._train_last_log_step = state.global_step
            return
        current_time = time.perf_counter()
        elapsed = current_time - self._train_last_log_time
        steps_delta = state.global_step - self._train_last_log_step
        if elapsed <= 0 or steps_delta <= 0:
            self._train_last_log_time = current_time
            self._train_last_log_step = state.global_step
            return
        effective_batch_size = pyine.utils.transformers.training.get_effective_train_batch_size(
            args, include_world_size=self._config.only_main_process
        )
        samples_delta = steps_delta * effective_batch_size
        samples_per_second = samples_delta / elapsed
        steps_per_second = steps_delta / elapsed
        logger.info(
            f"train throughput: samples_per_second={samples_per_second:.2f}, "
            f"steps_per_second={steps_per_second:.4f}, effective_batch_size={effective_batch_size}"
        )
        self._logger.log_train_throughput(
            step=state.global_step,
            samples_per_second=samples_per_second,
            steps_per_second=steps_per_second,
        )
        # update state for next interval
        self._train_last_log_time = current_time
        self._train_last_log_step = state.global_step

    def _log_eval_throughput(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
    ) -> None:
        """Log eval throughput metrics directly to wandb if enabled and valid."""
        if not self._config.log_eval_throughput:
            return
        if not self._saw_train_begin:
            return
        if self._eval_prediction_steps == 0:
            return
        if self._eval_start_time is None:
            return
        elapsed = time.perf_counter() - self._eval_start_time
        if elapsed <= 0:
            return
        eval_batch_size = pyine.utils.transformers.training.get_effective_eval_batch_size(
            args, include_world_size=self._config.only_main_process
        )
        samples = self._eval_prediction_steps * eval_batch_size
        samples_per_second = samples / elapsed
        logger.info(
            f"eval throughput: samples_per_second={samples_per_second:.2f}, "
            f"eval_batch_size={eval_batch_size}, prediction_steps={self._eval_prediction_steps}"
        )
        self._logger.log_eval_throughput(
            step=state.global_step,
            samples_per_second=samples_per_second,
        )

    @typing.override
    def on_evaluate(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **_: typing.Any,
    ) -> None:
        """Reset eval state after evaluation completes and resume train timing."""
        # reset eval timing state (unconditionally, even if logging disabled)
        self._eval_start_time = None
        self._eval_prediction_steps = 0
        # reset phase tracker
        self._phase.handle_evaluate()
        # resume train timing (reset both time AND step anchor to avoid inflated throughput)
        self._train_last_log_time = time.perf_counter()
        self._train_last_log_step = state.global_step

    @typing.override
    def on_save(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **_: typing.Any,
    ) -> None:
        """Reset timing after checkpoint save to exclude I/O overhead from throughput."""
        # reset both time AND step anchor to avoid inflated throughput
        self._train_last_log_time = time.perf_counter()
        self._train_last_log_step = state.global_step

    @typing.override
    def on_predict(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        metrics: dict[str, float],
        **_: typing.Any,
    ) -> None:
        """Reset eval state after predict completes.

        Trainer.predict() uses on_prediction_step but doesn't call on_evaluate, so we need
        this hook to reset eval state and avoid misclassifying subsequent training logs.
        """
        self._eval_start_time = None
        self._eval_prediction_steps = 0
        self._phase.handle_evaluate()
        # resume train timing if we were in a training run
        if self._saw_train_begin:
            self._train_last_log_time = time.perf_counter()
            self._train_last_log_step = state.global_step

    @typing.override
    def on_train_end(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **_: typing.Any,
    ) -> None:
        """Log final train throughput metrics to wandb.

        This ensures metrics are logged at the end of training, not just via logger.info.
        """
        if not self._should_log() or not self._config.log_train_throughput:
            return
        if self._train_last_log_time is None:
            return
        current_time = time.perf_counter()
        elapsed = current_time - self._train_last_log_time
        steps_delta = state.global_step - self._train_last_log_step
        if elapsed <= 0 or steps_delta <= 0:
            return
        effective_batch_size = pyine.utils.transformers.training.get_effective_train_batch_size(
            args, include_world_size=self._config.only_main_process
        )
        samples_delta = steps_delta * effective_batch_size
        samples_per_second = samples_delta / elapsed
        steps_per_second = steps_delta / elapsed
        logger.info(
            f"final train throughput: samples_per_second={samples_per_second:.2f}, "
            f"steps_per_second={steps_per_second:.4f}"
        )
        self._logger.log_train_throughput(
            step=state.global_step,
            samples_per_second=samples_per_second,
            steps_per_second=steps_per_second,
        )


class GPUStatsLoggingConfig(pydantic.BaseModel):
    """Configuration for GPU stats logging callback.

    To disable GPU stats logging, set `gpu_stats_logging: null` in your config (or don't set it
    at all). A non-null config means the callback is enabled.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    require_nvml: bool = False
    """Raise an error if NVML is unavailable. If False, falls back to PyTorch-only stats."""
    collect_all_visible_devices: bool = False
    """Whether to collect stats for all visible CUDA devices or just the current one.

    False (default): Collect stats only for torch.cuda.current_device(). Use this for DDP
    (DistributedDataParallel) where each rank owns exactly one GPU. With gathering enabled,
    all ranks contribute their single GPU's stats, giving cluster-wide coverage.

    True: Collect stats for all devices (indices 0 to torch.cuda.device_count()-1). Use this
    for model parallelism (tensor parallel, pipeline parallel) where a single process spans
    multiple GPUs, or for single-process multi-GPU setups (e.g., DataParallel).
    """
    only_main_process: bool = True
    """If True, only rank 0 logs metrics (after gathering from all ranks)."""
    gather_eval_metrics: bool = True
    """If True, gather eval stats from all ranks for aggregated metrics."""
    gather_train_metrics: typing.Literal["never", "at_phase_end", "always"] = "at_phase_end"
    """When to gather and emit train metrics.

    Options:
     - 'never': Emit local metrics at every train on_log (no gathering). With the default
       only_main_process=True, only rank 0 logs its own local stats; other ranks' stats are
       NOT gathered or logged.
     - 'at_phase_end': Accumulate all samples during train phase, then gather and emit once
       at phase end (before eval). This gives cluster-wide aggregated stats with minimal
       gathering overhead. **Requires evaluation to run** - if eval_strategy='no', metrics
       are only logged via logger.info at train end (not to wandb/tensorboard).
     - 'always': Gather and emit at every train on_log. Higher overhead but works regardless
       of eval strategy - use this if eval_strategy='no' and you want metrics in wandb.
    """
    sample_every_n_steps: int = pydantic.Field(default=1, ge=1)
    """Sample GPU stats every N training steps.

    Default is 1 (every step). Sampling is cheap (~1-3ms of NVML queries, no GPU sync), so
    per-step sampling has negligible overhead compared to training step duration.

    Note: 'total_sample_calls' in logged metrics is the total across all ranks (when gathered),
    not per-device data points.
    """
    train_prefix: str = "train/gpu/"
    """Prefix for training phase metrics."""
    eval_prefix: str = "eval/gpu/"
    """Prefix for evaluation phase metrics."""


@dataclasses.dataclass
class _PhaseAccumulators:
    """Accumulators for GPU stats during a training or eval phase.

    This dataclass encapsulates all the RunningStats accumulators used to track GPU metrics,
    along with helper methods for reset, serialization (for distributed gathering), and
    conversion to the metrics dict format.
    """

    utilization_gpu: dict[str, pyine.utils.stats.RunningStats] = dataclasses.field(default_factory=lambda: {})
    utilization_mem_ctrl: dict[str, pyine.utils.stats.RunningStats] = dataclasses.field(default_factory=lambda: {})
    vram_used_percent: dict[str, pyine.utils.stats.RunningStats] = dataclasses.field(default_factory=lambda: {})
    pytorch_allocated_percent: dict[str, pyine.utils.stats.RunningStats] = dataclasses.field(default_factory=lambda: {})
    power_watts: dict[str, pyine.utils.stats.RunningStats] = dataclasses.field(default_factory=lambda: {})
    temperature_celsius: dict[str, pyine.utils.stats.RunningStats] = dataclasses.field(default_factory=lambda: {})
    sample_count: int = 0

    def reset(self) -> None:
        """Clear all accumulators and reset sample count."""
        self.utilization_gpu.clear()
        self.utilization_mem_ctrl.clear()
        self.vram_used_percent.clear()
        self.pytorch_allocated_percent.clear()
        self.power_watts.clear()
        self.temperature_celsius.clear()
        self.sample_count = 0

    def update(
        self,
        device_key: str,
        *,
        utilization_gpu: float | None = None,
        utilization_mem_ctrl: float | None = None,
        vram_used_percent: float | None = None,
        pytorch_allocated_percent: float | None = None,
        power_watts: float | None = None,
        temperature_celsius: float | None = None,
    ) -> None:
        """Update accumulators with stats from a single device.

        Only non-None values are added to the corresponding accumulator.
        """

        def _update_one(
            accumulator: dict[str, pyine.utils.stats.RunningStats],
            value: float | None,
        ) -> None:
            if value is None:
                return
            if device_key not in accumulator:
                accumulator[device_key] = pyine.utils.stats.RunningStats()
            accumulator[device_key].update(value)

        _update_one(self.utilization_gpu, utilization_gpu)
        _update_one(self.utilization_mem_ctrl, utilization_mem_ctrl)
        _update_one(self.vram_used_percent, vram_used_percent)
        _update_one(self.pytorch_allocated_percent, pytorch_allocated_percent)
        _update_one(self.power_watts, power_watts)
        _update_one(self.temperature_celsius, temperature_celsius)

    def as_gather_payload(
        self,
        peak_percent: float,
    ) -> dict[str, typing.Any]:
        """Convert to a dict payload suitable for distributed gathering.

        Args:
            peak_percent: Peak memory percentage to include in payload.

        Returns:
            Dict with accumulator states, sample_count, and peak_percent.
        """
        return {
            "utilization_gpu": {k: s.as_state() for k, s in self.utilization_gpu.items()},
            "utilization_mem_ctrl": {k: s.as_state() for k, s in self.utilization_mem_ctrl.items()},
            "vram_used_percent": {k: s.as_state() for k, s in self.vram_used_percent.items()},
            "pytorch_allocated_percent": {k: s.as_state() for k, s in self.pytorch_allocated_percent.items()},
            "power_watts": {k: s.as_state() for k, s in self.power_watts.items()},
            "temperature_celsius": {k: s.as_state() for k, s in self.temperature_celsius.items()},
            "pytorch_peak_percent": peak_percent,
            "sample_count": self.sample_count,
        }

    def as_accumulators_dict(self) -> dict[str, dict[str, pyine.utils.stats.RunningStats]]:
        """Return accumulators as a dict mapping metric name to device stats.

        The metric names match the logged metric prefixes.
        """
        return {
            "utilization_gpu_percent": self.utilization_gpu,
            "utilization_mem_controller_percent": self.utilization_mem_ctrl,
            "vram_used_percent": self.vram_used_percent,
            "pytorch_allocated_percent": self.pytorch_allocated_percent,
            "power_watts": self.power_watts,
            "temperature_celsius": self.temperature_celsius,
        }


class GPUStatsLoggingCallback(transformers.TrainerCallback):
    """Callback that logs GPU utilization and memory statistics during training.

    This callback integrates with HuggingFace/TRL trainers to:
    - Sample GPU stats at configurable intervals during training;
    - Track stats separately for train and eval phases;
    - Aggregate statistics across distributed ranks (configurable for train; always for eval);
    - Log mean/std/min/max for each metric.

    Metrics logged (when available):
    - `{prefix}/gpu/utilization_gpu_percent/*`: GPU compute utilization (NVML)
    - `{prefix}/gpu/utilization_mem_controller_percent/*`: Memory controller utilization (NVML)
    - `{prefix}/gpu/vram_used_percent/*`: VRAM usage percentage (NVML)
    - `{prefix}/gpu/pytorch_allocated_percent/*`: PyTorch memory allocation percentage
    - `{prefix}/gpu/power_watts/*`: Power consumption in watts (NVML)
    - `{prefix}/gpu/temperature_celsius/*`: GPU temperature (NVML)
    - `{prefix}/gpu/pytorch_peak_percent`: Peak memory since last reset (max across devices/ranks)
    - `{prefix}/gpu/total_sample_calls`: Total sampling calls across all ranks (when gathered)
      or local sampling calls (when not gathered). Not per-device data points.

    Where `{prefix}` is `train` or `eval`.

    Example usage:
        ```python
        config = GPUStatsLoggingConfig()  # uses sensible defaults
        callback = GPUStatsLoggingCallback(config=config)
        trainer.add_callback(callback)
        ```

    Distributed training requirements:
        When using gather_train_metrics or gather_eval_metrics with distributed training, this
        callback uses `all_gather_object` in certain hooks. This is safe because the HuggingFace
        Trainer (4.46+) calls these hooks synchronously on ALL ranks:

        - `on_train_begin`: Called once at start on all ranks
        - `on_step_end`: Called after each training step on all ranks
        - `on_prediction_step`: Called during eval on all ranks
        - `on_log`: Called when logging on all ranks (gather happens here)
        - `on_evaluate`: Called after evaluation on all ranks
        - `on_train_end`: Called once at end on all ranks (gather happens here)

        The synchronization assumption for `on_log` holds because:
        1. `should_log` is determined by `state.global_step % args.logging_steps == 0`;
        2. `state.global_step` is synchronized across all ranks;
        3. `args.logging_steps` is the same config value on all ranks.

        If you modify Trainer behavior such that ranks become desynchronized on when hooks
        are called, gathering operations could deadlock. The callback includes a safety check
        (`_is_gather_safe`) that verifies torch.distributed is initialized before attempting
        to gather, but does not protect against ranks entering hooks at different times.

    Callback ordering dependency:
        This callback relies on HuggingFace's `DefaultFlowCallback` executing BEFORE this
        callback in `on_step_end` and `on_epoch_end`, because `DefaultFlowCallback` sets
        `control.should_evaluate` and `control.should_log` based on the current step/epoch.
        This ordering is guaranteed by default since `DefaultFlowCallback` is added first
        during Trainer initialization. If you remove or reorder `DefaultFlowCallback`, the
        eval-transition logic in this callback will not work correctly.
    """

    def __init__(
        self,
        config: GPUStatsLoggingConfig,
        wandb_run: typing.Any | None,
    ) -> None:
        """Initialize the callback.

        Args:
            config: Configuration for the callback.
            wandb_run: The wandb run object for direct logging (optional on non-main ranks).
        """
        self._config = config
        self._logger = pyine.utils.transformers.logging.GPUStatsLogger(wandb_run, config)
        self._collector: pyine.utils.gpu.GPUStatsCollector | None = None
        # phase tracking via shared tracker (all flags must be rank-synchronous to avoid deadlocks)
        self._phase = pyine.utils.transformers.training.EvalPhaseTracker()
        # tracks whether train phase metrics need flushing (for at_phase_end mode when should_log=False)
        self._train_phase_needs_flush: bool = False
        # stashed train peak percent (captured before peak reset when should_log=False)
        self._stashed_train_peak_percent: float | None = None
        # train accumulators (reset timing depends on gather_train_metrics setting)
        self._train = _PhaseAccumulators()
        # eval accumulators (reset at eval start)
        self._eval = _PhaseAccumulators()

    def _should_log(self) -> bool:
        """Check if metrics should be logged on this process."""
        if not self._config.only_main_process:
            return True
        return pyine.utils.distrib.is_main_process()

    def _is_gather_safe(self) -> bool:
        """Check if distributed gathering is safe (process group initialized).

        Returns True if:
        - Not in distributed mode (is_distributed() returns False), OR
        - torch.distributed is initialized (gathering will work).

        Returns False if:
        - is_distributed() returns True but torch.distributed isn't initialized yet
          (env vars set but process group not ready; gathering would silently degrade).
        """
        if not pyine.utils.distrib.is_distributed():
            return True  # not distributed, gathering degrades gracefully to local-only
        # in distributed mode, require process group to be initialized
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    def _reset_peak_stats(self) -> None:
        """Reset PyTorch peak memory stats for the appropriate devices.

        When collect_all_visible_devices=True, resets all visible devices. Otherwise, resets only
        the current device.
        """
        if self._collector is None or not self._collector.is_enabled():
            return
        if self._config.collect_all_visible_devices:
            device_indices = list(range(torch.cuda.device_count()))
        else:
            device_indices = None  # defaults to current device
        self._collector.reset_pytorch_peak_stats(device_indices)

    def _handle_eval_transition(
        self,
        should_log: bool,
    ) -> None:
        """Handle transition from train to eval phase.

        This method encapsulates the common eval-transition logic used by both on_step_end and
        on_epoch_end to avoid code duplication and drift risk.

        Args:
            should_log: Whether a train log will happen (from control.should_log).
        """
        # note: _phase.handle_step_or_epoch_end() is called in on_step_end/on_epoch_end before this
        self._eval.reset()
        # mark that train phase metrics need flushing (for at_phase_end mode)
        if self._config.gather_train_metrics == "at_phase_end" and self._train.sample_count > 0:
            self._train_phase_needs_flush = True
        # if no train log will happen, capture peak and reset now
        if not should_log:
            # capture train peak before resetting (for delayed flush in eval on_log)
            if self._train_phase_needs_flush:
                self._stashed_train_peak_percent = self._compute_local_peak_percent()
            self._reset_peak_stats()

    def _sample_stats(
        self,
        *,
        eval_only: bool = False,
    ) -> None:
        """Sample GPU stats and update accumulators.

        Args:
            eval_only: If True, only update eval accumulators. If False, only update train accumulators.
        """
        if self._collector is None or not self._collector.is_enabled():
            return
        if self._config.collect_all_visible_devices:
            stats_list = self._collector.collect_all_visible_devices()
        else:
            current_stats = self._collector.collect_current_device()
            stats_list = [current_stats] if current_stats is not None else []
        accumulators = self._eval if eval_only else self._train
        for stats in stats_list:
            # compute pytorch allocation percent (needs special handling due to division)
            pytorch_alloc_pct = (
                100.0 * stats.pytorch_allocated_bytes / stats.pytorch_total_bytes
                if stats.pytorch_total_bytes > 0
                else None
            )
            accumulators.update(
                stats.device_key,
                utilization_gpu=stats.utilization_gpu_percent,
                utilization_mem_ctrl=stats.utilization_mem_controller_percent,
                vram_used_percent=stats.vram_used_percent,
                pytorch_allocated_percent=pytorch_alloc_pct,
                power_watts=stats.power_watts,
                temperature_celsius=stats.temperature_celsius,
            )
        accumulators.sample_count += 1

    def _compute_local_peak_percent(self) -> float:
        """Compute peak memory percent. If multi-device, returns max across devices."""
        if not torch.cuda.is_available():
            return 0.0
        if self._config.collect_all_visible_devices:
            peaks: list[float] = []
            for idx in range(torch.cuda.device_count()):
                max_alloc = torch.cuda.max_memory_allocated(idx)
                total: int = int(torch.cuda.get_device_properties(idx).total_memory)  # type: ignore[reportUnknownMemberType]
                peaks.append(100.0 * max_alloc / total if total > 0 else 0.0)
            return max(peaks) if peaks else 0.0
        idx = torch.cuda.current_device()
        max_alloc = torch.cuda.max_memory_allocated(idx)
        total_mem: int = int(torch.cuda.get_device_properties(idx).total_memory)  # type: ignore[reportUnknownMemberType]
        return 100.0 * max_alloc / total_mem if total_mem > 0 else 0.0

    def _compute_metrics_from_accumulators(
        self,
        accumulators: dict[str, dict[str, pyine.utils.stats.RunningStats]],
        sample_count: int,
        peak_percent: float,
    ) -> dict[str, float | int]:
        """Compute metrics dict from accumulators.

        Args:
            accumulators: Dict mapping metric name to dict of device_key -> RunningStats.
            sample_count: Number of samples taken.
            peak_percent: Peak memory percentage.

        Returns:
            Dict of metric_name -> value.
        """
        result: dict[str, float | int] = {}
        for metric_name, device_stats in accumulators.items():
            # merge all device stats into one
            merged = pyine.utils.stats.RunningStats()
            for stats in device_stats.values():
                merged.merge(stats)
            if merged.count == 0:
                continue
            assert merged.min is not None and merged.max is not None
            result[f"{metric_name}/mean"] = merged.mean()
            result[f"{metric_name}/std"] = merged.std()
            result[f"{metric_name}/min"] = merged.min
            result[f"{metric_name}/max"] = merged.max
        if sample_count > 0:
            result["total_sample_calls"] = sample_count
        if peak_percent > 0.0:
            result["pytorch_peak_percent"] = peak_percent
        return result

    def _compute_train_metrics(
        self,
        peak_percent_override: float | None = None,
    ) -> dict[str, float | int]:
        """Compute metrics from train accumulators (local only, no gathering).

        Args:
            peak_percent_override: If provided, use this instead of computing peak from current state.
                Useful for delayed flushes where peak was captured earlier.
        """
        if peak_percent_override is not None:
            peak_percent = peak_percent_override
        else:
            peak_percent = self._compute_local_peak_percent()
        return self._compute_metrics_from_accumulators(
            self._train.as_accumulators_dict(),
            self._train.sample_count,
            peak_percent,
        )

    def _gather_and_merge_payloads(
        self,
        local_payload: dict[str, typing.Any],
    ) -> dict[str, float | int]:
        """Gather payloads from all ranks and merge into aggregated metrics.

        IMPORTANT: All ranks must call this method to avoid deadlocks.

        Args:
            local_payload: Dict with accumulator states, peak_percent, and sample_count.

        Returns:
            Aggregated metrics dict on main rank, empty dict on other ranks.
        """
        gathered = pyine.utils.distrib.all_gather_objects(local_payload)
        # only rank 0 computes final metrics
        if not pyine.utils.distrib.is_main_process():
            return {}
        # merge all gathered payloads
        merged_accumulators: dict[str, dict[str, pyine.utils.stats.RunningStats]] = {
            "utilization_gpu_percent": {},
            "utilization_mem_controller_percent": {},
            "vram_used_percent": {},
            "pytorch_allocated_percent": {},
            "power_watts": {},
            "temperature_celsius": {},
        }
        max_peak = 0.0
        total_sample_count = 0
        metric_key_map = {
            "utilization_gpu": "utilization_gpu_percent",
            "utilization_mem_ctrl": "utilization_mem_controller_percent",
            "vram_used_percent": "vram_used_percent",
            "pytorch_allocated_percent": "pytorch_allocated_percent",
            "power_watts": "power_watts",
            "temperature_celsius": "temperature_celsius",
        }
        for item in gathered:
            for payload_key, merged_key in metric_key_map.items():
                device_states = item.get(payload_key, {})
                for device_key, state in device_states.items():
                    if device_key not in merged_accumulators[merged_key]:
                        merged_accumulators[merged_key][device_key] = pyine.utils.stats.RunningStats()
                    merged_accumulators[merged_key][device_key].merge(pyine.utils.stats.RunningStats.from_state(state))
            max_peak = max(max_peak, item.get("pytorch_peak_percent", 0.0))
            total_sample_count += item.get("sample_count", 0)
        return self._compute_metrics_from_accumulators(merged_accumulators, total_sample_count, max_peak)

    def _gather_and_compute_train_metrics(
        self,
        peak_percent_override: float | None = None,
    ) -> dict[str, float | int]:
        """Gather train stats from all ranks and compute aggregated metrics.

        IMPORTANT: All ranks must call this method to avoid deadlocks.

        Args:
            peak_percent_override: If provided, use this instead of computing peak from current state.
                Useful for delayed flushes where peak was captured earlier.
        """
        if not pyine.utils.distrib.is_distributed():
            return self._compute_train_metrics(peak_percent_override)
        local_peak = peak_percent_override if peak_percent_override is not None else self._compute_local_peak_percent()
        return self._gather_and_merge_payloads(self._train.as_gather_payload(local_peak))

    def _compute_eval_metrics(self) -> dict[str, float | int]:
        """Compute metrics from eval accumulators (local only, no gathering)."""
        return self._compute_metrics_from_accumulators(
            self._eval.as_accumulators_dict(),
            self._eval.sample_count,
            self._compute_local_peak_percent(),
        )

    def _gather_and_compute_eval_metrics(self) -> dict[str, float | int]:
        """Gather eval stats from all ranks and compute aggregated metrics.

        IMPORTANT: All ranks must call this method to avoid deadlocks.
        """
        if not self._config.gather_eval_metrics or not pyine.utils.distrib.is_distributed():
            return self._compute_eval_metrics()
        return self._gather_and_merge_payloads(self._eval.as_gather_payload(self._compute_local_peak_percent()))

    def _require_gather_safe(
        self,
        context: str,
    ) -> bool:
        """Check if gathering is wanted and safe; raise RuntimeError if wanted but unsafe.

        Args:
            context: Description of where the gather would happen (for error message).

        Returns:
            True if gathering should proceed, False if gathering not wanted.

        Raises:
            RuntimeError: If gathering is wanted but torch.distributed is not initialized.
        """
        wants_gather = pyine.utils.distrib.is_distributed() and self._config.gather_train_metrics != "never"
        if not wants_gather:
            return False
        if not self._is_gather_safe():
            raise RuntimeError(
                f"gather_train_metrics enabled but torch.distributed not initialized ({context}). "
                "This would silently drop stats from non-rank0 processes. Either ensure "
                "torch.distributed is initialized before training, or set gather_train_metrics='never'."
            )
        return True

    @typing.override
    def on_train_begin(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **_: typing.Any,
    ) -> None:
        """Initialize collector and reset accumulators at training start."""
        # first: validate config combinations
        if (
            self._config.collect_all_visible_devices
            and self._config.gather_eval_metrics
            and pyine.utils.distrib.is_distributed()
            and torch.cuda.is_available()
            and torch.cuda.device_count() > 1
        ):
            raise ValueError(
                "Cannot use collect_all_visible_devices=True with gather_eval_metrics=True "
                "when multiple GPUs are visible per rank (would double-count). "
                "Either set collect_all_visible_devices=False (recommended for DDP) or "
                "set gather_eval_metrics=False (each rank logs independently)."
            )
        if (
            self._config.collect_all_visible_devices
            and self._config.gather_train_metrics != "never"
            and pyine.utils.distrib.is_distributed()
            and torch.cuda.is_available()
            and torch.cuda.device_count() > 1
        ):
            raise ValueError(
                "Cannot use collect_all_visible_devices=True with gather_train_metrics!='never' "
                "when multiple GPUs are visible per rank (would double-count). "
                "Either set collect_all_visible_devices=False (recommended for DDP) or "
                "set gather_train_metrics='never' (each rank logs its own stats)."
            )
        if (
            not self._config.only_main_process
            and self._config.gather_eval_metrics
            and pyine.utils.distrib.is_distributed()
        ):
            raise ValueError(
                "Cannot use only_main_process=False with gather_eval_metrics=True. "
                "When gathering stats across nodes, only the main process receives aggregated "
                "eval metrics. Either set only_main_process=True (recommended) or "
                "set gather_eval_metrics=False (each rank logs its own stats)."
            )
        if (
            not self._config.only_main_process
            and self._config.gather_train_metrics != "never"
            and pyine.utils.distrib.is_distributed()
        ):
            raise ValueError(
                "Cannot use only_main_process=False with gather_train_metrics!='never'. "
                "When gathering stats across nodes, only the main process receives aggregated "
                "train metrics. Either set only_main_process=True (recommended) or "
                "set gather_train_metrics='never' (each rank logs its own stats)."
            )
        # initialize collector
        self._collector = pyine.utils.gpu.GPUStatsCollector(require_nvml=self._config.require_nvml)
        # in distributed mode with gathering enabled, verify all ranks have same CUDA availability
        # (prevents deadlock where some ranks call all_gather and others don't)
        gathering_enabled = self._config.gather_eval_metrics or self._config.gather_train_metrics != "never"
        if gathering_enabled and self._is_gather_safe():
            # gather unconditionally to avoid deadlock if is_enabled differs across ranks
            local_enabled = self._collector.is_enabled()
            gathered_enabled = pyine.utils.distrib.all_gather_objects(local_enabled)
            if len(set(gathered_enabled)) > 1:
                enabled_ranks = [i for i, enabled in enumerate(gathered_enabled) if enabled]
                disabled_ranks = [i for i, enabled in enumerate(gathered_enabled) if not enabled]
                raise RuntimeError(
                    f"CUDA availability mismatch across ranks: ranks {enabled_ranks} have CUDA, "
                    f"ranks {disabled_ranks} do not. This would cause a distributed deadlock during "
                    "GPU stats gathering. Ensure all ranks have the same CUDA availability, or set "
                    "gather_train_metrics='never' and gather_eval_metrics=False to disable gathering."
                )
            # now safe to check NVML mismatch (all ranks have CUDA)
            if self._collector.is_enabled():
                local_nvml = self._collector.is_nvml_available()
                gathered_nvml = pyine.utils.distrib.all_gather_objects(local_nvml)
                if pyine.utils.distrib.is_main_process() and len(set(gathered_nvml)) > 1:
                    nvml_ranks = [i for i, has_nvml in enumerate(gathered_nvml) if has_nvml]
                    no_nvml_ranks = [i for i, has_nvml in enumerate(gathered_nvml) if not has_nvml]
                    logger.warning(
                        f"NVML availability mismatch across ranks: ranks {nvml_ranks} have NVML, "
                        f"ranks {no_nvml_ranks} do not. Aggregated utilization/power/temperature metrics "
                        "will only include data from NVML-enabled ranks, potentially biasing results."
                    )
        # warn once (on main process only) when NVML unavailable but callback enabled
        if (
            self._collector.is_enabled()
            and not self._collector.is_nvml_available()
            and pyine.utils.distrib.is_main_process()
        ):
            logger.warning(
                "GPU stats logging enabled but NVML unavailable (pynvml not installed); "
                "only PyTorch memory stats will be logged, not utilization/power/temperature. "
                "Install with: uv sync --extra gpu-monitoring"
            )
        # warn if at_phase_end mode is used without evaluation (metrics won't appear in W&B/TensorBoard)
        eval_strategy_raw = getattr(args, "eval_strategy", getattr(args, "evaluation_strategy", "no"))
        # handle both string and IntervalStrategy enum (use .value if available, else str and lowercase)
        eval_strategy = getattr(eval_strategy_raw, "value", str(eval_strategy_raw)).lower()
        if (
            self._config.gather_train_metrics == "at_phase_end"
            and eval_strategy == "no"
            and pyine.utils.distrib.is_main_process()
        ):
            logger.warning(
                "GPU stats logging: gather_train_metrics='at_phase_end' with eval_strategy='no' means "
                "train/gpu metrics will only be logged via logger.info at train end (they will NOT appear "
                "in W&B/TensorBoard). Use gather_train_metrics='always' if you need metrics in W&B."
            )
        # reset state (all flags must be rank-synchronous to avoid deadlocks)
        self._phase = pyine.utils.transformers.training.EvalPhaseTracker()
        self._train_phase_needs_flush = False
        self._stashed_train_peak_percent = None
        self._train.reset()
        self._eval.reset()
        if self._collector.is_enabled():
            self._reset_peak_stats()

    @typing.override
    def on_step_end(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **_: typing.Any,
    ) -> None:
        """Sample stats and handle eval transition.

        Note: This method relies on DefaultFlowCallback having already set control.should_evaluate
        and control.should_log. See class docstring for callback ordering requirements.
        """
        if self._collector is None or not self._collector.is_enabled():
            return
        # sample every N steps
        if state.global_step % self._config.sample_every_n_steps == 0:
            self._sample_stats(eval_only=False)
        # handle eval transition (if DefaultFlowCallback set should_evaluate=True)
        transitioning = self._phase.handle_step_or_epoch_end(control)
        if transitioning:
            self._handle_eval_transition(should_log=control.should_log)

    @typing.override
    def on_epoch_end(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **_: typing.Any,
    ) -> None:
        """Handle eval transition at epoch end.

        Note: This method relies on DefaultFlowCallback having already set control.should_evaluate
        and control.should_log. See class docstring for callback ordering requirements.
        """
        if self._collector is None or not self._collector.is_enabled():
            return
        # handle eval transition (if DefaultFlowCallback set should_evaluate=True)
        transitioning = self._phase.handle_step_or_epoch_end(control)
        if transitioning:
            self._handle_eval_transition(should_log=control.should_log)

    @typing.override
    def on_prediction_step(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **_: typing.Any,
    ) -> None:
        """Sample stats during evaluation."""
        if self._collector is None or not self._collector.is_enabled():
            return
        self._phase.handle_prediction_step()
        # note: we don't warn on unexpected here because GPUStatsLoggingCallback doesn't have
        # sensitive prefix-switching like RewardLoggingCallback; unexpected could occur during
        # predict-only runs or eval_on_start, which are valid use cases
        self._sample_stats(eval_only=True)

    @typing.override
    def on_evaluate(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **_: typing.Any,
    ) -> None:
        """Cleanup after evaluation completes.

        Note on HuggingFace Trainer callback ordering (verified in transformers 4.46+):
        The actual order inside evaluate() is:
        1. on_prediction_step (multiple times during eval loop) - sets _phase.in_eval=True
        2. on_log (with eval metrics) - we detect this via _phase.in_eval=True
        3. on_evaluate (this method) - cleanup

        This method is called AFTER the eval on_log, so eval metrics have already been
        injected. We just do cleanup here.
        """
        # handle edge case: if train metrics weren't flushed by eval on_log (unexpected callback ordering),
        # log via logger.warning rather than silently dropping
        if self._train_phase_needs_flush and self._should_log():
            train_metrics = self._compute_train_metrics(peak_percent_override=self._stashed_train_peak_percent)
            if train_metrics:
                logger.warning(
                    "Train phase GPU stats were not flushed before on_evaluate completed (unexpected callback "
                    f"ordering). Logging via logger.info instead. train/gpu metrics: {train_metrics}"
                )
            self._train.reset()
        # cleanup phase state on all ranks (must be rank-synchronous)
        self._phase.handle_evaluate()
        self._train_phase_needs_flush = False  # either flushed above or by on_log
        self._stashed_train_peak_percent = None  # either used above or by on_log
        # reset peak stats so eval peaks don't contaminate next train interval
        self._reset_peak_stats()

    @typing.override
    def on_log(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        logs: dict[str, typing.Any] | None = None,
        **kwargs: typing.Any,
    ) -> None:
        """Inject GPU metrics into logs dict.

        IMPORTANT: All ranks must execute this method and agree on the code path to avoid
        deadlocks when gathering is enabled. We use _phase.in_eval (set in on_prediction_step,
        cleared in on_evaluate) to determine eval vs train context in a rank-synchronous way.

        Note on HuggingFace Trainer callback ordering (verified in transformers 4.46+):
        - Train log: on_step_end -> on_log (train) -> [if should_evaluate]
          -> on_prediction_step -> on_log (eval) -> on_evaluate
        - The _phase.in_eval flag is True after on_prediction_step until on_evaluate resets it
        - The _phase.eval_pending flag is True from on_step_end until on_prediction_step clears it
        """
        if self._collector is None or not self._collector.is_enabled():
            return
        if logs is None:
            logs = kwargs.get("logs")
        # determine eval vs train context; prefer _phase.in_eval flag but handle edge cases
        is_eval_context = self._phase.in_eval
        if logs is not None and not self._phase.in_eval:
            looks_like_eval = any(key.startswith("eval_") or key.startswith("eval/") for key in logs)
            if looks_like_eval:
                if self._phase.eval_pending:
                    # edge case: eval-like logs but on_prediction_step wasn't called (e.g., empty eval dataloader)
                    # _phase.eval_pending is rank-synchronous (set in on_step_end/on_epoch_end), so this is safe
                    is_eval_context = True
                    if self._should_log():
                        logger.info(
                            "on_log received eval-like logs but _phase.in_eval=False (on_prediction_step not called). "
                            "Treating as eval context based on _phase.eval_pending flag (likely empty eval dataloader)."
                        )
                elif self._should_log():
                    # unexpected: eval-like logs without _phase.eval_pending; warn but treat as train
                    logger.warning(
                        "on_log received eval-like logs (keys starting with 'eval_' or 'eval/') but "
                        "_phase.in_eval=False and _phase.eval_pending=False. This may indicate HuggingFace "
                        "Trainer callback ordering changed. GPU stats will be classified as train metrics. "
                        "Please report this issue."
                    )
        # use determined context for metric logging
        if is_eval_context:
            # EVAL LOG: compute/gather and log eval metrics directly to wandb
            # IMPORTANT: if gathering, ALL ranks must call gather to avoid deadlock
            #
            # ...first, handle any pending train phase metrics that weren't emitted
            # (happens when should_evaluate=True but should_log=False in at_phase_end mode)
            if self._train_phase_needs_flush:
                should_train_gather = self._require_gather_safe("eval on_log with pending train flush")
                stashed_peak = self._stashed_train_peak_percent
                if should_train_gather:
                    train_metrics = self._gather_and_compute_train_metrics(peak_percent_override=stashed_peak)
                elif self._should_log():
                    train_metrics = self._compute_train_metrics(peak_percent_override=stashed_peak)
                else:
                    train_metrics = {}
                if self._should_log() and train_metrics:
                    self._logger.log_train_stats(step=state.global_step, **train_metrics)
                self._train.reset()
                self._train_phase_needs_flush = False
                self._stashed_train_peak_percent = None
            # now handle eval metrics (note: eval gather uses different config than train gather)
            wants_eval_gather = self._config.gather_eval_metrics and pyine.utils.distrib.is_distributed()
            if wants_eval_gather and not self._is_gather_safe():
                raise RuntimeError(
                    "gather_eval_metrics enabled but torch.distributed not initialized. "
                    "This would silently drop stats from non-rank0 processes. Either ensure "
                    "torch.distributed is initialized before training, or set gather_eval_metrics=False."
                )
            if wants_eval_gather:
                # all ranks participate in gather; non-main ranks get empty dict
                eval_metrics = self._gather_and_compute_eval_metrics()
            elif self._should_log():
                # local only mode: only main rank needs to compute (others would just drop results)
                eval_metrics = self._compute_eval_metrics() if self._phase.saw_eval_samples else {}
            else:
                eval_metrics = {}
            # only main rank logs to wandb
            if self._should_log() and eval_metrics:
                self._logger.log_eval_stats(step=state.global_step, **eval_metrics)
            # all ranks reset accumulators to stay in sync
            self._eval.reset()
            # note: _phase.in_eval is cleared by _phase.handle_evaluate() in on_evaluate
        else:
            # TRAIN LOG: compute/gather and log train metrics directly to wandb
            # IMPORTANT: if gathering, ALL ranks must call gather to avoid deadlock
            #
            # ...behavior depends on gather_train_metrics setting:
            # - "never": emit local metrics at every train log
            # - "at_phase_end": accumulate throughout phase, emit only at phase end (before eval)
            # - "always": gather and emit at every train log
            is_phase_end = self._phase.eval_pending
            at_phase_end_mode = self._config.gather_train_metrics == "at_phase_end"
            if at_phase_end_mode and not is_phase_end:
                # "at_phase_end" mode but not at phase end: keep accumulating, don't emit
                # (no reset, no logging, no gathering)
                pass
            else:
                # either "never", "always", or "at_phase_end" at actual phase end
                should_gather = self._require_gather_safe("train on_log")
                if should_gather:
                    # all ranks participate in gather; non-main ranks get empty dict
                    train_metrics = self._gather_and_compute_train_metrics()
                else:
                    # local only, compute on main rank
                    train_metrics = self._compute_train_metrics() if self._should_log() else {}
                # only main rank logs to wandb
                if self._should_log() and train_metrics:
                    self._logger.log_train_stats(step=state.global_step, **train_metrics)
                # all ranks reset accumulators and peak stats to stay in sync
                self._train.reset()
                self._reset_peak_stats()
                self._train_phase_needs_flush = False

    @typing.override
    def on_train_end(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **_: typing.Any,
    ) -> None:
        """Flush any remaining train metrics at the end of training.

        This handles the case where gather_train_metrics='at_phase_end' but eval never runs
        (e.g., eval_strategy='no'). Without this, accumulated train metrics would be lost.

        Logs directly to wandb since on_train_end doesn't receive a logs dict.
        """
        if self._collector is None or not self._collector.is_enabled():
            return
        # only flush if we have accumulated metrics that weren't emitted
        if self._train.sample_count == 0:
            return
        # compute metrics (gathering if configured and safe)
        should_gather = self._require_gather_safe("on_train_end")
        if should_gather:
            train_metrics = self._gather_and_compute_train_metrics()
        else:
            train_metrics = self._compute_train_metrics() if self._should_log() else {}
        # log to wandb and logger.info
        if self._should_log() and train_metrics:
            logger.info(f"GPU stats at train_end: train/gpu metrics: {train_metrics}")
            self._logger.log_train_stats(step=state.global_step, **train_metrics)
        # cleanup
        self._train.reset()
        self._reset_peak_stats()
