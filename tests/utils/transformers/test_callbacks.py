import pathlib
import typing

import pytest
import pytest_mock
import transformers

import pyine.utils.transformers.callbacks as callbacks_module


class TestStdoutMilestones:
    def test_should_print_main_process_on_main(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        mocker.patch("pyine.utils.distrib.is_main_process", return_value=True)
        callback = callbacks_module.StdoutMilestones(only_main_process=True)
        assert callback._should_print() is True

    def test_should_print_main_process_on_worker(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        mocker.patch("pyine.utils.distrib.is_main_process", return_value=False)
        callback = callbacks_module.StdoutMilestones(only_main_process=True)
        assert callback._should_print() is False

    def test_should_print_all_processes(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        mocker.patch("pyine.utils.distrib.is_main_process", return_value=False)
        callback = callbacks_module.StdoutMilestones(only_main_process=False)
        assert callback._should_print() is True

    def test_fmt_epoch_with_float(self) -> None:
        """Test _fmt_epoch formats float epoch correctly."""
        assert callbacks_module.StdoutMilestones._fmt_epoch(1.5) == "1.5"
        assert callbacks_module.StdoutMilestones._fmt_epoch(2.0) == "2"
        assert callbacks_module.StdoutMilestones._fmt_epoch(0.123456789) == "0.123457"

    def test_fmt_epoch_with_none(self) -> None:
        """Test _fmt_epoch returns NA for None."""
        assert callbacks_module.StdoutMilestones._fmt_epoch(None) == "NA"

    def test_fmt_dict_with_floats(self) -> None:
        """Test _fmt_dict formats dictionary with float values."""
        result = callbacks_module.StdoutMilestones._fmt_dict({"loss": 0.123456789, "lr": 1e-5})
        assert result == "loss=0.123457, lr=1e-05"

    def test_fmt_dict_with_mixed_types(self) -> None:
        """Test _fmt_dict formats dictionary with mixed value types."""
        result = callbacks_module.StdoutMilestones._fmt_dict({"loss": 0.5, "epoch": 1, "name": "test"})
        assert result == "epoch=1, loss=0.5, name=test"

    def test_fmt_dict_empty(self) -> None:
        """Test _fmt_dict handles empty dictionary."""
        result = callbacks_module.StdoutMilestones._fmt_dict({})
        assert result == ""

    def test_fmt_dict_sorts_keys(self) -> None:
        """Test _fmt_dict sorts keys alphabetically."""
        result = callbacks_module.StdoutMilestones._fmt_dict({"z": 1, "a": 2, "m": 3})
        assert result == "a=2, m=3, z=1"

    @pytest.fixture
    def mock_print_fn(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> pytest_mock.MockFixture:
        """Create a mock print function."""
        return mocker.MagicMock()

    @pytest.fixture
    def callback(self, mock_print_fn: typing.Any) -> callbacks_module.StdoutMilestones:
        """Create a StdoutMilestones instance with mock print function."""
        return callbacks_module.StdoutMilestones(print_fn=mock_print_fn)

    @pytest.fixture
    def args(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> pytest_mock.MockFixture:
        """Create mock training arguments."""
        return mocker.MagicMock(
            process_index=0,
            run_name="test_run",
            output_dir="/something/output",
            num_train_epochs=3,
            max_steps=1000,
            gradient_accumulation_steps=4,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            fp16=False,
            bf16=True,
            world_size=2,
            local_rank=0,
            n_gpu=1,
            logging_strategy="steps",
            logging_steps=100,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=5e-5,
            warmup_steps=500,
        )

    @pytest.fixture
    def state(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> pytest_mock.MockFixture:
        """Create mock trainer state."""
        return mocker.MagicMock(
            epoch=1.0,
            global_step=100,
            best_metric=0.95,
            best_model_checkpoint="/something/checkpoint-100",
        )

    @pytest.fixture
    def control(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> pytest_mock.MockFixture:
        """Create mock trainer control."""
        return mocker.MagicMock()

    def test_on_init_end_main_process(
        self,
        callback: typing.Any,
        mock_print_fn: typing.Any,
        args: typing.Any,
        state: typing.Any,
        control: typing.Any,
    ) -> None:
        """Test on_init_end prints on main process."""
        callback.on_init_end(args, state, control)
        mock_print_fn.assert_called_once_with("init_end")

    def test_on_init_end_worker_process(
        self,
        callback: typing.Any,
        mock_print_fn: typing.Any,
        args: typing.Any,
        state: typing.Any,
        control: typing.Any,
        mocker: typing.Any,
    ) -> None:
        """Test on_init_end doesn't print on worker process."""
        mocker.patch("pyine.utils.distrib.is_main_process", return_value=False)
        callback.on_init_end(args, state, control)
        mock_print_fn.assert_not_called()

    def test_on_train_begin_with_config(
        self,
        callback: typing.Any,
        mock_print_fn: typing.Any,
        args: typing.Any,
        state: typing.Any,
        control: typing.Any,
    ) -> None:
        """Test on_train_begin prints training configuration."""
        callback.on_train_begin(args, state, control)
        assert mock_print_fn.call_count == 7
        mock_print_fn.assert_any_call("train_begin")
        mock_print_fn.assert_any_call("run_name=test_run, output_dir=/something/output")

    def test_on_train_begin_without_config(
        self,
        mock_print_fn: typing.Any,
        args: typing.Any,
        state: typing.Any,
        control: typing.Any,
    ) -> None:
        """Test on_train_begin without printing config."""
        callback = callbacks_module.StdoutMilestones(print_fn=mock_print_fn, print_config_at_start=False)
        callback.on_train_begin(args, state, control)
        mock_print_fn.assert_called_once_with("train_begin")

    def test_on_train_begin_worker_process(
        self,
        callback: typing.Any,
        mock_print_fn: typing.Any,
        args: typing.Any,
        state: typing.Any,
        control: typing.Any,
        mocker: typing.Any,
    ) -> None:
        """Test on_train_begin doesn't print on worker process."""
        mocker.patch("pyine.utils.distrib.is_main_process", return_value=False)
        callback.on_train_begin(args, state, control)
        mock_print_fn.assert_not_called()

    def test_on_epoch_begin(
        self,
        callback: typing.Any,
        mock_print_fn: typing.Any,
        args: typing.Any,
        state: typing.Any,
        control: typing.Any,
    ) -> None:
        """Test on_epoch_begin prints epoch information."""
        callback.on_epoch_begin(args, state, control)
        mock_print_fn.assert_called_once_with("epoch_begin; epoch=1, global_step=100")

    def test_on_epoch_begin_none_epoch(
        self,
        callback: typing.Any,
        mock_print_fn: typing.Any,
        args: typing.Any,
        control: typing.Any,
        mocker: typing.Any,
    ) -> None:
        """Test on_epoch_begin handles None epoch."""
        state = mocker.MagicMock(epoch=None, global_step=100)
        callback.on_epoch_begin(args, state, control)
        mock_print_fn.assert_called_once_with("epoch_begin; epoch=NA, global_step=100")

    def test_on_epoch_end(
        self,
        callback: typing.Any,
        mock_print_fn: typing.Any,
        args: typing.Any,
        state: typing.Any,
        control: typing.Any,
    ) -> None:
        """Test on_epoch_end prints epoch information."""
        callback.on_epoch_end(args, state, control)
        mock_print_fn.assert_called_once_with("epoch_end; epoch=1, global_step=100")

    def test_on_log_with_logs(
        self,
        callback: typing.Any,
        mock_print_fn: typing.Any,
        args: typing.Any,
        state: typing.Any,
        control: typing.Any,
    ) -> None:
        """Test on_log prints log information."""
        logs = {"loss": 0.5, "learning_rate": 1e-5}
        callback.on_log(args, state, control, logs=logs)
        mock_print_fn.assert_called_once_with("log; step=100, learning_rate=1e-05, loss=0.5")

    def test_on_log_empty_logs(
        self,
        callback: typing.Any,
        mock_print_fn: typing.Any,
        args: typing.Any,
        state: typing.Any,
        control: typing.Any,
    ) -> None:
        """Test on_log doesn't print when logs are empty."""
        callback.on_log(args, state, control, logs={})
        mock_print_fn.assert_not_called()

    def test_on_log_worker_process(
        self,
        callback: typing.Any,
        mock_print_fn: typing.Any,
        args: typing.Any,
        state: typing.Any,
        control: typing.Any,
        mocker: typing.Any,
    ) -> None:
        """Test on_log doesn't print on worker process."""
        mocker.patch("pyine.utils.distrib.is_main_process", return_value=False)
        logs = {"loss": 0.5}
        callback.on_log(args, state, control, logs=logs)
        mock_print_fn.assert_not_called()

    def test_on_evaluate(
        self,
        callback: typing.Any,
        mock_print_fn: typing.Any,
        args: typing.Any,
        state: typing.Any,
        control: typing.Any,
    ) -> None:
        """Test on_evaluate prints metrics."""
        metrics = {"eval_loss": 0.3, "eval_accuracy": 0.95}
        callback.on_evaluate(args, state, control, metrics=metrics)
        mock_print_fn.assert_called_once_with("evaluate; step=100, epoch=1, eval_accuracy=0.95, eval_loss=0.3")

    def test_on_evaluate_empty_metrics(
        self,
        callback: typing.Any,
        mock_print_fn: typing.Any,
        args: typing.Any,
        state: typing.Any,
        control: typing.Any,
    ) -> None:
        """Test on_evaluate doesn't print when metrics are empty."""
        callback.on_evaluate(args, state, control, metrics={})
        mock_print_fn.assert_not_called()

    def test_on_predict(
        self,
        callback: typing.Any,
        mock_print_fn: typing.Any,
        args: typing.Any,
        state: typing.Any,
        control: typing.Any,
    ) -> None:
        """Test on_predict prints metrics."""
        metrics = {"predict_loss": 0.25, "predict_accuracy": 0.92}
        callback.on_predict(args, state, control, metrics=metrics)
        mock_print_fn.assert_called_once_with("predict; step=100, epoch=1, predict_accuracy=0.92, predict_loss=0.25")

    def test_on_predict_empty_metrics(
        self,
        callback: typing.Any,
        mock_print_fn: typing.Any,
        args: typing.Any,
        state: typing.Any,
        control: typing.Any,
    ) -> None:
        """Test on_predict doesn't print when metrics are empty."""
        callback.on_predict(args, state, control, metrics={})
        mock_print_fn.assert_not_called()

    def test_on_save(
        self,
        callback: typing.Any,
        mock_print_fn: typing.Any,
        args: typing.Any,
        state: typing.Any,
        control: typing.Any,
        mocker: typing.Any,
    ) -> None:
        """Test on_save prints checkpoint information."""
        mock_path = mocker.MagicMock(spec=pathlib.Path)
        mock_path.absolute.return_value = pathlib.Path("/something/checkpoint-100")
        mock_get_ckpt = mocker.patch("pyine.utils.transformers.checkpoints.get_checkpoint_folder_path")
        mock_get_ckpt.return_value = mock_path
        callback.on_save(args, state, control)
        mock_print_fn.assert_called_once_with("save; checkpoint_dir=/something/checkpoint-100")

    def test_on_save_worker_process(
        self,
        callback: typing.Any,
        mock_print_fn: typing.Any,
        args: typing.Any,
        state: typing.Any,
        control: typing.Any,
        mocker: typing.Any,
    ) -> None:
        """Test on_save doesn't print on worker process."""
        mocker.patch("pyine.utils.distrib.is_main_process", return_value=False)
        callback.on_save(args, state, control)
        mock_print_fn.assert_not_called()

    def test_on_train_end(
        self,
        callback: typing.Any,
        mock_print_fn: typing.Any,
        args: typing.Any,
        state: typing.Any,
        control: typing.Any,
    ) -> None:
        """Test on_train_end prints final training information."""
        callback.on_train_end(args, state, control)
        mock_print_fn.assert_called_once_with(
            "train_end; steps=100, best_metric=0.95, best_model_checkpoint=/something/checkpoint-100"
        )

    def test_on_train_end_worker_process(
        self,
        callback: typing.Any,
        mock_print_fn: typing.Any,
        args: typing.Any,
        state: typing.Any,
        control: typing.Any,
        mocker: typing.Any,
    ) -> None:
        """Test on_train_end doesn't print on worker process."""
        mocker.patch("pyine.utils.distrib.is_main_process", return_value=False)
        callback.on_train_end(args, state, control)
        mock_print_fn.assert_not_called()

    def test_callback_inheritance(self) -> None:
        """Test that StdoutMilestones properly inherits from TrainerCallback."""
        callback = callbacks_module.StdoutMilestones()
        assert isinstance(callback, transformers.TrainerCallback)

    def test_all_callbacks_defined(self) -> None:
        """Test that all expected callback methods are defined."""
        callback = callbacks_module.StdoutMilestones()
        expected_methods = [
            "on_init_end",
            "on_train_begin",
            "on_epoch_begin",
            "on_epoch_end",
            "on_log",
            "on_evaluate",
            "on_predict",
            "on_save",
            "on_train_end",
        ]
        for method in expected_methods:
            assert hasattr(callback, method)
            assert callable(getattr(callback, method))


class _FakeRewardManager:
    """Minimal fake RewardManager for testing RewardLoggingCallback."""

    def __init__(self) -> None:
        self.key_prefix: str | None = None
        self.step: int | None = None
        self.epoch: float | None = None
        self.flush_calls: list[int | None] = []
        self.reset_phase_local_counts_calls: list[str | None] = []

    def set_key_prefix(self, prefix: str) -> None:
        self.key_prefix = prefix

    def set_step(self, step: int | None) -> None:
        self.step = step

    def set_epoch(self, epoch: float | None) -> None:
        self.epoch = epoch

    def flush_stats(
        self,
        step: int | None = None,
        *,
        failure_ratio: float | None = None,
        failure_count: int | None = None,
    ) -> None:
        self.flush_calls.append(step)

    def reset_phase_local_counts(self, prefix: str | None = None) -> None:
        """Track calls to reset_phase_local_counts for test assertions."""
        self.reset_phase_local_counts_calls.append(prefix)

    def get_state(self) -> dict[str, typing.Any]:
        return {}

    def load_state(self, state: dict[str, typing.Any]) -> None:
        pass


class TestRewardLoggingCallback:
    @pytest.fixture
    def reward_manager(self) -> _FakeRewardManager:
        return _FakeRewardManager()

    @pytest.fixture
    def args(self, mocker: pytest_mock.MockerFixture) -> pytest_mock.MockFixture:
        return mocker.MagicMock(
            process_index=0,
            eval_on_start=False,
        )

    @pytest.fixture
    def state(self, mocker: pytest_mock.MockerFixture) -> pytest_mock.MockFixture:
        return mocker.MagicMock(
            global_step=100,
            epoch=1.5,
            is_world_process_zero=True,
        )

    @pytest.fixture
    def control(self, mocker: pytest_mock.MockerFixture) -> pytest_mock.MockFixture:
        return mocker.MagicMock(
            should_evaluate=False,
        )

    def test_on_train_begin_initializes_train_prefix_when_eval_on_start_false(
        self,
        reward_manager: _FakeRewardManager,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
    ) -> None:
        args.eval_on_start = False
        callback = callbacks_module.RewardLoggingCallback(reward_manager=reward_manager)
        callback.on_train_begin(args, state, control)
        assert callback._phase.in_eval is False
        assert reward_manager.key_prefix == "train"

    def test_on_train_begin_initializes_eval_prefix_when_eval_on_start_true(
        self,
        reward_manager: _FakeRewardManager,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
    ) -> None:
        args.eval_on_start = True
        callback = callbacks_module.RewardLoggingCallback(reward_manager=reward_manager)
        callback.on_train_begin(args, state, control)
        assert callback._phase.in_eval is True
        assert reward_manager.key_prefix == "eval"
        # should have flushed train stats (empty) before switching
        assert len(reward_manager.flush_calls) == 1
        assert reward_manager.flush_calls[0] == state.global_step
        # step should be cleared for eval
        assert reward_manager.step is None

    def test_on_step_begin_resets_prefix_after_eval_on_start(
        self,
        reward_manager: _FakeRewardManager,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
    ) -> None:
        """Test that on_step_begin resets prefix to train after eval_on_start completes."""
        args.eval_on_start = True
        callback = callbacks_module.RewardLoggingCallback(reward_manager=reward_manager)
        # 1. on_train_begin with eval_on_start=True puts us in eval mode
        callback.on_train_begin(args, state, control)
        assert callback._phase.in_eval is True
        assert reward_manager.key_prefix == "eval"
        # 2. on_evaluate ends the eval phase
        callback.on_evaluate(args, state, control)
        assert callback._phase.in_eval is False
        assert reward_manager.key_prefix == "train"
        # 3. on_step_begin for first training step should stay in train mode
        callback.on_step_begin(args, state, control)
        assert callback._phase.in_eval is False
        assert reward_manager.key_prefix == "train"

    def test_on_step_end_switches_to_eval_when_should_evaluate_true(
        self,
        reward_manager: _FakeRewardManager,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
    ) -> None:
        callback = callbacks_module.RewardLoggingCallback(reward_manager=reward_manager)
        reward_manager.key_prefix = "train"
        control.should_evaluate = True
        callback.on_step_end(args, state, control)
        assert callback._phase.in_eval is True
        assert callback._phase.saw_training is True  # handle_step_or_epoch_end was called
        assert reward_manager.key_prefix == "eval"
        assert len(reward_manager.flush_calls) == 1
        assert reward_manager.step is None

    def test_on_step_end_does_not_switch_if_should_evaluate_false(
        self,
        reward_manager: _FakeRewardManager,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
    ) -> None:
        callback = callbacks_module.RewardLoggingCallback(reward_manager=reward_manager)
        reward_manager.key_prefix = "train"
        control.should_evaluate = False
        callback.on_step_end(args, state, control)
        assert callback._phase.in_eval is False
        assert callback._phase.saw_training is True  # handle_step_or_epoch_end was called
        assert reward_manager.key_prefix == "train"
        assert len(reward_manager.flush_calls) == 0

    def test_on_step_end_does_not_switch_if_already_in_eval(
        self,
        reward_manager: _FakeRewardManager,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
    ) -> None:
        callback = callbacks_module.RewardLoggingCallback(reward_manager=reward_manager)
        callback._phase.mark_entering_eval()  # set phase tracker to in_eval
        reward_manager.key_prefix = "eval"
        control.should_evaluate = True
        callback.on_step_end(args, state, control)
        assert callback._phase.in_eval is True
        assert reward_manager.key_prefix == "eval"
        # no flush because already in eval
        assert len(reward_manager.flush_calls) == 0

    def test_on_epoch_end_switches_to_eval_when_should_evaluate_true(
        self,
        reward_manager: _FakeRewardManager,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
    ) -> None:
        callback = callbacks_module.RewardLoggingCallback(reward_manager=reward_manager)
        reward_manager.key_prefix = "train"
        control.should_evaluate = True
        callback.on_epoch_end(args, state, control)
        assert callback._phase.in_eval is True
        assert callback._phase.saw_training is True  # handle_step_or_epoch_end was called
        assert reward_manager.key_prefix == "eval"
        assert len(reward_manager.flush_calls) == 1
        assert reward_manager.step is None

    def test_on_epoch_end_does_not_switch_if_should_evaluate_false(
        self,
        reward_manager: _FakeRewardManager,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
    ) -> None:
        callback = callbacks_module.RewardLoggingCallback(reward_manager=reward_manager)
        reward_manager.key_prefix = "train"
        control.should_evaluate = False
        callback.on_epoch_end(args, state, control)
        assert callback._phase.in_eval is False
        assert callback._phase.saw_training is True  # handle_step_or_epoch_end was called
        assert reward_manager.key_prefix == "train"
        assert len(reward_manager.flush_calls) == 0

    def test_switch_to_eval_clears_step(
        self,
        reward_manager: _FakeRewardManager,
    ) -> None:
        callback = callbacks_module.RewardLoggingCallback(reward_manager=reward_manager)
        reward_manager.step = 100
        callback._switch_to_eval(step=100)
        assert reward_manager.step is None
        assert callback._phase.in_eval is True
        assert reward_manager.key_prefix == "eval"

    def test_on_prediction_step_sets_prefix_without_flush(
        self,
        reward_manager: _FakeRewardManager,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
    ) -> None:
        callback = callbacks_module.RewardLoggingCallback(reward_manager=reward_manager)
        reward_manager.key_prefix = "train"
        callback.on_prediction_step(args, state, control)
        assert callback._phase.in_eval is True
        assert callback._phase.saw_eval_samples is True
        # should NOT have flushed (on_prediction_step is too late)
        assert len(reward_manager.flush_calls) == 0
        # prefix should be set to eval since this is first prediction step
        assert reward_manager.key_prefix == "eval"
        assert reward_manager.step is None

    def test_on_prediction_step_only_clears_step_if_already_in_eval(
        self,
        reward_manager: _FakeRewardManager,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
    ) -> None:
        callback = callbacks_module.RewardLoggingCallback(reward_manager=reward_manager)
        callback._phase.mark_entering_eval()  # set phase tracker (simulates _switch_to_eval)
        reward_manager.key_prefix = "eval"
        reward_manager.step = 50
        callback.on_prediction_step(args, state, control)
        assert callback._phase.in_eval is True
        assert reward_manager.key_prefix == "eval"
        assert len(reward_manager.flush_calls) == 0
        assert reward_manager.step is None

    def test_on_evaluate_does_not_flush_if_no_prediction_steps(
        self,
        reward_manager: _FakeRewardManager,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
    ) -> None:
        """Test that on_evaluate skips flush if eval had zero samples (no prediction steps)."""
        callback = callbacks_module.RewardLoggingCallback(reward_manager=reward_manager)
        # simulate: on_step_end switched to eval, but no prediction_step was called (zero samples)
        callback._phase.mark_entering_eval()  # sets in_eval=True but saw_eval_samples=False
        reward_manager.key_prefix = "eval"
        callback.on_evaluate(args, state, control)
        # should NOT have flushed (no eval samples processed)
        assert len(reward_manager.flush_calls) == 0
        # should have reset to train mode
        assert callback._phase.in_eval is False
        assert reward_manager.key_prefix == "train"

    def test_on_evaluate_flushes_if_prediction_steps_occurred(
        self,
        reward_manager: _FakeRewardManager,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
    ) -> None:
        """Test that on_evaluate flushes if prediction steps occurred during eval."""
        callback = callbacks_module.RewardLoggingCallback(reward_manager=reward_manager)
        callback._phase.handle_prediction_step()  # sets saw_eval_samples=True
        reward_manager.key_prefix = "eval"
        callback.on_evaluate(args, state, control)
        # should have flushed
        assert len(reward_manager.flush_calls) == 1
        assert reward_manager.flush_calls[0] == state.global_step
        # should have reset flag
        assert callback._phase.saw_eval_samples is False
        # should have reset to train mode
        assert callback._phase.in_eval is False
        assert reward_manager.key_prefix == "train"

    def test_on_prediction_step_sets_saw_eval_prediction_step_flag(
        self,
        reward_manager: _FakeRewardManager,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
    ) -> None:
        """Test that on_prediction_step sets saw_eval_samples flag."""
        callback = callbacks_module.RewardLoggingCallback(reward_manager=reward_manager)
        assert callback._phase.saw_eval_samples is False
        callback.on_prediction_step(args, state, control)
        assert callback._phase.saw_eval_samples is True

    def test_unexpected_eval_switches_prefix_and_warns(
        self,
        reward_manager: _FakeRewardManager,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Test unexpected eval (no should_evaluate=True) still switches prefix and logs warning."""
        callback = callbacks_module.RewardLoggingCallback(reward_manager=reward_manager)
        reward_manager.key_prefix = "train"
        # simulate training step (sets saw_training=True) without should_evaluate
        control.should_evaluate = False
        callback.on_step_end(args, state, control)
        assert callback._phase.saw_training is True
        assert callback._phase.in_eval is False
        # now call prediction_step without going through normal eval flow
        # this simulates HF Trainer behavior change or external eval trigger
        warn_mock = mocker.patch.object(callbacks_module.logger, "warning")
        callback.on_prediction_step(args, state, control)
        # should have switched prefix to eval
        assert reward_manager.key_prefix == "eval"
        assert callback._phase.in_eval is True
        # should have logged a warning about unexpected eval
        warn_mock.assert_called_once()
        assert "without prior should_evaluate=True" in warn_mock.call_args[0][0]

    def test_phase_local_counts_reset_on_all_phase_entries(
        self,
        reward_manager: _FakeRewardManager,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
    ) -> None:
        """Verify reset_phase_local_counts() called at all phase transitions."""
        args.eval_on_start = False
        callback = callbacks_module.RewardLoggingCallback(reward_manager=reward_manager)
        # on_train_begin should reset (train phase entry via _switch_to_train)
        callback.on_train_begin(args, state, control)
        assert len(reward_manager.reset_phase_local_counts_calls) == 1
        assert reward_manager.key_prefix == "train"
        # _switch_to_eval should reset (eval phase entry)
        callback._switch_to_eval(step=100)
        assert len(reward_manager.reset_phase_local_counts_calls) == 2
        assert reward_manager.key_prefix == "eval"
        # on_evaluate should reset (back to train phase entry via _switch_to_train)
        callback._phase._in_eval = True
        callback._phase._saw_eval_prediction_step = True
        callback.on_evaluate(args, state, control)
        assert len(reward_manager.reset_phase_local_counts_calls) == 3
        assert reward_manager.key_prefix == "train"

    def test_phase_local_counts_reset_on_empty_eval(
        self,
        reward_manager: _FakeRewardManager,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
    ) -> None:
        """Verify reset happens even when eval had zero samples (consistent semantics)."""
        callback = callbacks_module.RewardLoggingCallback(reward_manager=reward_manager)
        callback.on_train_begin(args, state, control)
        initial_reset_count = len(reward_manager.reset_phase_local_counts_calls)
        # simulate eval with zero samples
        callback._switch_to_eval(step=100)
        assert len(reward_manager.reset_phase_local_counts_calls) == initial_reset_count + 1
        # on_evaluate still resets even though saw_eval_samples is False
        callback._phase._in_eval = True
        callback._phase._saw_eval_prediction_step = False  # no samples processed
        callback.on_evaluate(args, state, control)
        assert len(reward_manager.reset_phase_local_counts_calls) == initial_reset_count + 2

    def test_on_step_begin_fallback_uses_switch_to_train(
        self,
        reward_manager: _FakeRewardManager,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
    ) -> None:
        """Verify on_step_begin fallback path (when _phase.in_eval) uses _switch_to_train()."""
        callback = callbacks_module.RewardLoggingCallback(reward_manager=reward_manager)
        callback.on_train_begin(args, state, control)
        initial_reset_count = len(reward_manager.reset_phase_local_counts_calls)
        # simulate being in eval phase without on_evaluate having fired (callback ordering issue)
        callback._phase._in_eval = True
        reward_manager.key_prefix = "eval"  # simulate eval prefix was set
        # on_step_begin should detect in_eval and switch back to train with reset
        state.global_step = 50
        state.epoch = 1.0
        callback.on_step_begin(args, state, control)
        # should have reset (via _switch_to_train)
        assert len(reward_manager.reset_phase_local_counts_calls) == initial_reset_count + 1
        assert reward_manager.key_prefix == "train"
        assert reward_manager.step == 50
        assert reward_manager.epoch == 1.0

    def test_on_step_begin_no_reset_when_already_in_train(
        self,
        reward_manager: _FakeRewardManager,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
    ) -> None:
        """Verify on_step_begin does NOT reset when already in train phase (normal path)."""
        callback = callbacks_module.RewardLoggingCallback(reward_manager=reward_manager)
        callback.on_train_begin(args, state, control)
        initial_reset_count = len(reward_manager.reset_phase_local_counts_calls)
        # normal train step - _phase.in_eval is False
        callback._phase._in_eval = False
        state.global_step = 10
        state.epoch = 0.5
        callback.on_step_begin(args, state, control)
        # should NOT have reset (no phase change)
        assert len(reward_manager.reset_phase_local_counts_calls) == initial_reset_count
        assert reward_manager.step == 10
        assert reward_manager.epoch == 0.5

    def test_on_prediction_step_fallback_resets_local_counts(
        self,
        reward_manager: _FakeRewardManager,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
    ) -> None:
        """Verify on_prediction_step fallback path (first step, no prior switch) resets local counts."""
        callback = callbacks_module.RewardLoggingCallback(reward_manager=reward_manager)
        reward_manager.key_prefix = "train"
        # simulate entering eval without _switch_to_eval being called (fallback path)
        # _phase.in_eval is False, handle_prediction_step() returns (True, True) for first unexpected step
        callback._phase._saw_training = True  # mark that we saw training (makes eval "unexpected")
        initial_reset_count = len(reward_manager.reset_phase_local_counts_calls)
        # call on_prediction_step - should set prefix AND reset counts
        callback.on_prediction_step(args, state, control)
        # should have reset (first prediction step sets prefix and resets)
        assert len(reward_manager.reset_phase_local_counts_calls) == initial_reset_count + 1
        assert reward_manager.key_prefix == "eval"


class TestThroughputLoggingCallback:
    @pytest.fixture
    def mock_wandb_run(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> pytest_mock.MockFixture:
        """Create a mock wandb_run object."""
        mock_run = mocker.MagicMock()
        mock_run.log = mocker.MagicMock()
        mock_run.define_metric = mocker.MagicMock()
        return mock_run

    @pytest.fixture
    def args(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> pytest_mock.MockFixture:
        return mocker.MagicMock(
            per_device_train_batch_size=4,
            per_device_eval_batch_size=8,
            world_size=2,
            gradient_accumulation_steps=2,
        )

    @pytest.fixture
    def state(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> pytest_mock.MockFixture:
        return mocker.MagicMock(global_step=0)

    @pytest.fixture
    def control(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> pytest_mock.MockFixture:
        return mocker.MagicMock(should_evaluate=False)

    def test_should_log_main_process(
        self,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        mocker.patch("pyine.utils.distrib.is_main_process", return_value=True)
        config = callbacks_module.ThroughputLoggingConfig(only_main_process=True)
        callback = callbacks_module.ThroughputLoggingCallback(config=config, wandb_run=mock_wandb_run)
        assert callback._should_log() is True

    def test_should_log_worker_process(
        self,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        mocker.patch("pyine.utils.distrib.is_main_process", return_value=False)
        config = callbacks_module.ThroughputLoggingConfig(only_main_process=True)
        callback = callbacks_module.ThroughputLoggingCallback(config=config, wandb_run=mock_wandb_run)
        assert callback._should_log() is False

    def test_should_log_all_processes(
        self,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        mocker.patch("pyine.utils.distrib.is_main_process", return_value=False)
        config = callbacks_module.ThroughputLoggingConfig(only_main_process=False)
        callback = callbacks_module.ThroughputLoggingCallback(config=config, wandb_run=mock_wandb_run)
        assert callback._should_log() is True

    def test_on_train_begin_initializes_state(
        self,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        config = callbacks_module.ThroughputLoggingConfig()
        callback = callbacks_module.ThroughputLoggingCallback(config=config, wandb_run=mock_wandb_run)
        assert callback._train_last_log_time is None
        callback.on_train_begin(args, state, control)
        assert callback._train_last_log_time is not None
        assert callback._train_last_log_step == 0
        assert callback._saw_train_begin is True

    def test_on_log_calculates_train_throughput(
        self,
        args: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        mock_time = mocker.patch("pyine.utils.transformers.callbacks.time.perf_counter")
        mock_time.side_effect = [100.0, 102.0]
        mocker.patch("wandb.run", mock_wandb_run)
        config = callbacks_module.ThroughputLoggingConfig()
        callback = callbacks_module.ThroughputLoggingCallback(config=config, wandb_run=mock_wandb_run)
        state = mocker.MagicMock(global_step=0)
        callback.on_train_begin(args, state, control)
        state.global_step = 10
        logs: dict[str, float] = {}
        callback.on_log(args, state, control, logs=logs)
        # verify wandb_run.log was called with correct metrics
        assert mock_wandb_run.log.called
        call_args = mock_wandb_run.log.call_args[0][0]
        assert call_args["train/throughput/samples_per_second"] == 80.0
        assert call_args["train/throughput/steps_per_second"] == 5.0
        assert call_args["train/global_step"] == 10

    def test_on_log_skips_on_worker_process(
        self,
        args: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        mocker.patch("pyine.utils.distrib.is_main_process", return_value=False)
        config = callbacks_module.ThroughputLoggingConfig(only_main_process=True)
        callback = callbacks_module.ThroughputLoggingCallback(config=config, wandb_run=mock_wandb_run)
        state = mocker.MagicMock(global_step=10)
        logs: dict[str, float] = {}
        callback.on_log(args, state, control, logs=logs)
        # verify wandb_run.log was NOT called
        assert not mock_wandb_run.log.called

    def test_on_log_handles_zero_elapsed(
        self,
        args: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        mock_time = mocker.patch("pyine.utils.transformers.callbacks.time.perf_counter")
        mock_time.return_value = 100.0
        config = callbacks_module.ThroughputLoggingConfig()
        callback = callbacks_module.ThroughputLoggingCallback(config=config, wandb_run=mock_wandb_run)
        state = mocker.MagicMock(global_step=0)
        callback.on_train_begin(args, state, control)
        state.global_step = 10
        logs: dict[str, float] = {}
        callback.on_log(args, state, control, logs=logs)
        assert not mock_wandb_run.log.called

    def test_on_log_handles_zero_steps(
        self,
        args: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        mock_time = mocker.patch("pyine.utils.transformers.callbacks.time.perf_counter")
        mock_time.side_effect = [100.0, 101.0]
        config = callbacks_module.ThroughputLoggingConfig()
        callback = callbacks_module.ThroughputLoggingCallback(config=config, wandb_run=mock_wandb_run)
        state = mocker.MagicMock(global_step=0)
        callback.on_train_begin(args, state, control)
        logs: dict[str, float] = {}
        callback.on_log(args, state, control, logs=logs)
        assert not mock_wandb_run.log.called

    def test_custom_prefix(
        self,
        args: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        mock_time = mocker.patch("pyine.utils.transformers.callbacks.time.perf_counter")
        mock_time.side_effect = [100.0, 101.0]
        config = callbacks_module.ThroughputLoggingConfig(train_prefix="train/speed/")
        callback = callbacks_module.ThroughputLoggingCallback(config=config, wandb_run=mock_wandb_run)
        state = mocker.MagicMock(global_step=0)
        callback.on_train_begin(args, state, control)
        state.global_step = 5
        logs: dict[str, float] = {}
        callback.on_log(args, state, control, logs=logs)
        assert mock_wandb_run.log.called
        call_args = mock_wandb_run.log.call_args[0][0]
        assert "train/speed/samples_per_second" in call_args
        assert "train/speed/steps_per_second" in call_args
        assert call_args["train/speed/samples_per_second"] == 80.0
        assert call_args["train/speed/steps_per_second"] == 5.0

    def test_on_save_resets_timing(
        self,
        args: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        mock_time = mocker.patch("pyine.utils.transformers.callbacks.time.perf_counter")
        mock_time.side_effect = [100.0, 105.0, 106.0]
        config = callbacks_module.ThroughputLoggingConfig()
        callback = callbacks_module.ThroughputLoggingCallback(config=config, wandb_run=mock_wandb_run)
        state = mocker.MagicMock(global_step=0)
        callback.on_train_begin(args, state, control)
        callback.on_save(args, state, control)
        state.global_step = 10
        logs: dict[str, float] = {}
        callback.on_log(args, state, control, logs=logs)
        assert mock_wandb_run.log.called
        call_args = mock_wandb_run.log.call_args[0][0]
        assert call_args["train/throughput/steps_per_second"] == 10.0

    def test_resume_training_with_nonzero_step(
        self,
        args: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        mock_time = mocker.patch("pyine.utils.transformers.callbacks.time.perf_counter")
        mock_time.side_effect = [100.0, 101.0]
        config = callbacks_module.ThroughputLoggingConfig()
        callback = callbacks_module.ThroughputLoggingCallback(config=config, wandb_run=mock_wandb_run)
        state = mocker.MagicMock(global_step=500)
        callback.on_train_begin(args, state, control)
        assert callback._train_last_log_step == 500
        state.global_step = 510
        logs: dict[str, float] = {}
        callback.on_log(args, state, control, logs=logs)
        assert mock_wandb_run.log.called
        call_args = mock_wandb_run.log.call_args[0][0]
        assert call_args["train/throughput/steps_per_second"] == 10.0

    def test_callback_inheritance(
        self,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        config = callbacks_module.ThroughputLoggingConfig()
        callback = callbacks_module.ThroughputLoggingCallback(config=config, wandb_run=mock_wandb_run)
        assert isinstance(callback, transformers.TrainerCallback)

    def test_eval_throughput_from_prediction_steps(
        self,
        args: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        """Verifies eval throughput is calculated from prediction steps."""
        mock_time = mocker.patch("pyine.utils.transformers.callbacks.time.perf_counter")
        # trace: train_begin(100), step_end(101 sets eval timer), on_log(102 computes elapsed)
        # note: prediction steps don't call perf_counter since timer already set
        mock_time.side_effect = [100.0, 101.0, 102.0]
        config = callbacks_module.ThroughputLoggingConfig()
        callback = callbacks_module.ThroughputLoggingCallback(config=config, wandb_run=mock_wandb_run)
        state = mocker.MagicMock(global_step=10)
        callback.on_train_begin(args, state, control)
        # transition to eval
        control.should_evaluate = True
        callback.on_step_end(args, state, control)
        # two prediction steps (no perf_counter calls since timer already set)
        callback.on_prediction_step(args, state, control)
        callback.on_prediction_step(args, state, control)
        logs: dict[str, typing.Any] = {"eval_loss": 0.3}  # eval-like logs
        callback.on_log(args, state, control, logs=logs)
        # eval batch size = 8 * 2 = 16, 2 prediction steps, elapsed = 102 - 101 = 1 second
        expected_samples = 2 * 16
        expected_throughput = expected_samples / 1.0
        assert mock_wandb_run.log.called
        call_args = mock_wandb_run.log.call_args[0][0]
        assert "eval/throughput/samples_per_second" in call_args
        assert call_args["eval/throughput/samples_per_second"] == pytest.approx(expected_throughput)

    def test_eval_timing_starts_at_step_end(
        self,
        args: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        """Verifies timer starts in on_step_end, not on_prediction_step."""
        mock_time = mocker.patch("pyine.utils.transformers.callbacks.time.perf_counter")
        # trace: train_begin(100), step_end(101 sets eval timer), on_log(103 computes elapsed)
        # note: prediction step doesn't call perf_counter since timer already set
        mock_time.side_effect = [100.0, 101.0, 103.0]
        config = callbacks_module.ThroughputLoggingConfig()
        callback = callbacks_module.ThroughputLoggingCallback(config=config, wandb_run=mock_wandb_run)
        state = mocker.MagicMock(global_step=10)
        callback.on_train_begin(args, state, control)
        # transition to eval
        control.should_evaluate = True
        callback.on_step_end(args, state, control)
        assert callback._eval_start_time == 101.0  # timer started at step_end
        callback.on_prediction_step(args, state, control)
        logs: dict[str, typing.Any] = {"eval_loss": 0.3}
        callback.on_log(args, state, control, logs=logs)
        # elapsed = 103 - 101 = 2 seconds (includes first batch)
        expected_samples = 1 * 8 * 2  # 1 step * eval batch size
        assert mock_wandb_run.log.called
        call_args = mock_wandb_run.log.call_args[0][0]
        assert call_args["eval/throughput/samples_per_second"] == expected_samples / 2.0

    def test_log_train_disabled(
        self,
        args: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        """Verifies train throughput not logged when disabled."""
        mock_time = mocker.patch("pyine.utils.transformers.callbacks.time.perf_counter")
        mock_time.side_effect = [100.0, 101.0]
        config = callbacks_module.ThroughputLoggingConfig(log_train_throughput=False)
        callback = callbacks_module.ThroughputLoggingCallback(config=config, wandb_run=mock_wandb_run)
        state = mocker.MagicMock(global_step=0)
        callback.on_train_begin(args, state, control)
        state.global_step = 10
        logs: dict[str, float] = {}
        callback.on_log(args, state, control, logs=logs)
        assert not mock_wandb_run.log.called

    def test_log_eval_disabled_still_resets_state(
        self,
        args: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        """Verifies state reset even when eval logging disabled."""
        mock_time = mocker.patch("pyine.utils.transformers.callbacks.time.perf_counter")
        mock_time.side_effect = [100.0, 101.0, 102.0, 103.0, 104.0]
        config = callbacks_module.ThroughputLoggingConfig(log_eval_throughput=False)
        callback = callbacks_module.ThroughputLoggingCallback(config=config, wandb_run=mock_wandb_run)
        state = mocker.MagicMock(global_step=10)
        callback.on_train_begin(args, state, control)
        control.should_evaluate = True
        callback.on_step_end(args, state, control)
        callback.on_prediction_step(args, state, control)
        logs: dict[str, typing.Any] = {"eval_loss": 0.3}
        callback.on_log(args, state, control, logs=logs)
        assert not mock_wandb_run.log.called
        # on_evaluate should reset state
        callback.on_evaluate(args, state, control)
        assert callback._eval_start_time is None
        assert callback._eval_prediction_steps == 0

    def test_empty_eval_skips_throughput(
        self,
        args: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        """Verifies no metrics when _eval_prediction_steps == 0."""
        mock_time = mocker.patch("pyine.utils.transformers.callbacks.time.perf_counter")
        mock_time.side_effect = [100.0, 101.0, 102.0]
        config = callbacks_module.ThroughputLoggingConfig()
        callback = callbacks_module.ThroughputLoggingCallback(config=config, wandb_run=mock_wandb_run)
        state = mocker.MagicMock(global_step=10)
        callback.on_train_begin(args, state, control)
        control.should_evaluate = True
        callback.on_step_end(args, state, control)
        # no prediction steps
        logs: dict[str, typing.Any] = {"eval_loss": 0.3}
        callback.on_log(args, state, control, logs=logs)
        assert not mock_wandb_run.log.called

    def test_unexpected_eval_logs_warning(
        self,
        args: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        """Verifies warning when unexpected eval detected."""
        mock_time = mocker.patch("pyine.utils.transformers.callbacks.time.perf_counter")
        mock_time.side_effect = [100.0, 101.0, 102.0, 103.0]
        mock_logger = mocker.patch("pyine.utils.transformers.callbacks.logger")
        config = callbacks_module.ThroughputLoggingConfig()
        callback = callbacks_module.ThroughputLoggingCallback(config=config, wandb_run=mock_wandb_run)
        state = mocker.MagicMock(global_step=10)
        callback.on_train_begin(args, state, control)
        # complete a training step (sets saw_training=True)
        control.should_evaluate = False
        callback.on_step_end(args, state, control)
        # unexpected prediction step without should_evaluate
        callback.on_prediction_step(args, state, control)
        mock_logger.warning.assert_called_once()
        assert "Unexpected eval phase" in str(mock_logger.warning.call_args)

    def test_predict_only_skips_eval_throughput(
        self,
        args: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        """Verifies no eval throughput when training never happened (predict-only)."""
        mock_time = mocker.patch("pyine.utils.transformers.callbacks.time.perf_counter")
        mock_time.side_effect = [100.0, 101.0]
        config = callbacks_module.ThroughputLoggingConfig()
        callback = callbacks_module.ThroughputLoggingCallback(config=config, wandb_run=mock_wandb_run)
        state = mocker.MagicMock(global_step=0)
        # no on_train_begin, directly to prediction
        callback.on_prediction_step(args, state, control)
        logs: dict[str, typing.Any] = {"eval_loss": 0.3}
        callback.on_log(args, state, control, logs=logs)
        # _saw_train_begin is False, so eval throughput should be skipped
        assert not mock_wandb_run.log.called

    def test_eval_on_start_no_warning(
        self,
        args: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        """Verifies no warning during eval_on_start flow."""
        mock_time = mocker.patch("pyine.utils.transformers.callbacks.time.perf_counter")
        mock_time.side_effect = [100.0, 101.0, 102.0, 103.0]
        mock_logger = mocker.patch("pyine.utils.transformers.callbacks.logger")
        config = callbacks_module.ThroughputLoggingConfig()
        callback = callbacks_module.ThroughputLoggingCallback(config=config, wandb_run=mock_wandb_run)
        state = mocker.MagicMock(global_step=0)
        callback.on_train_begin(args, state, control)
        # eval_on_start: prediction step without on_step_end first
        callback.on_prediction_step(args, state, control)
        # no warning because no training step has completed yet
        mock_logger.warning.assert_not_called()

    def test_eval_on_start_emits_metrics(
        self,
        args: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        """Verifies eval throughput is logged during eval_on_start."""
        mock_time = mocker.patch("pyine.utils.transformers.callbacks.time.perf_counter")
        mock_time.side_effect = [100.0, 101.0, 102.0]  # train_begin, pred, on_log
        config = callbacks_module.ThroughputLoggingConfig()
        callback = callbacks_module.ThroughputLoggingCallback(config=config, wandb_run=mock_wandb_run)
        state = mocker.MagicMock(global_step=0)
        callback.on_train_begin(args, state, control)
        callback.on_prediction_step(args, state, control)
        logs: dict[str, typing.Any] = {"eval_loss": 0.3}
        callback.on_log(args, state, control, logs=logs)
        # timer started in on_prediction_step (fallback), elapsed = 102 - 101 = 1 second
        expected_samples = 1 * 8 * 2
        assert mock_wandb_run.log.called
        call_args = mock_wandb_run.log.call_args[0][0]
        assert "eval/throughput/samples_per_second" in call_args
        assert call_args["eval/throughput/samples_per_second"] == expected_samples / 1.0

    def test_train_throughput_after_eval_not_inflated(
        self,
        args: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        """Verifies train throughput isn't inflated when eval occurs between logs.

        Without the step anchor reset, throughput would be inflated because:
        - Steps are counted from before eval (e.g., step 5 to step 15 = 10 steps)
        - But time is only from after eval (1 second instead of 3 seconds)
        """
        mock_time = mocker.patch("pyine.utils.transformers.callbacks.time.perf_counter")
        # trace: train_begin(100), step_end(101), on_eval(102), on_log(103)
        mock_time.side_effect = [100.0, 101.0, 102.0, 103.0]
        config = callbacks_module.ThroughputLoggingConfig()
        callback = callbacks_module.ThroughputLoggingCallback(config=config, wandb_run=mock_wandb_run)
        state = mocker.MagicMock(global_step=0)
        callback.on_train_begin(args, state, control)
        # train 5 steps, then eval
        state.global_step = 5
        control.should_evaluate = True
        callback.on_step_end(args, state, control)
        callback.on_evaluate(args, state, control)
        # train 5 more steps, then log
        state.global_step = 10
        control.should_evaluate = False
        logs: dict[str, float] = {}
        callback.on_log(args, state, control, logs=logs)
        # with fix: step anchor reset to 5 after eval, so steps = 10 - 5 = 5, time = 1 sec
        # effective batch size = 4 * 2 * 2 = 16, samples = 5 * 16 = 80
        assert mock_wandb_run.log.called
        call_args = mock_wandb_run.log.call_args[0][0]
        assert call_args["train/throughput/samples_per_second"] == 80.0  # 80 samples / 1 sec
        assert call_args["train/throughput/steps_per_second"] == 5.0  # 5 steps / 1 sec

    def test_train_throughput_after_save_not_inflated(
        self,
        args: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        """Verifies train throughput isn't inflated when save occurs between logs."""
        mock_time = mocker.patch("pyine.utils.transformers.callbacks.time.perf_counter")
        # trace: train_begin(100), on_save(101), on_log(102)
        mock_time.side_effect = [100.0, 101.0, 102.0]
        config = callbacks_module.ThroughputLoggingConfig()
        callback = callbacks_module.ThroughputLoggingCallback(config=config, wandb_run=mock_wandb_run)
        state = mocker.MagicMock(global_step=0)
        callback.on_train_begin(args, state, control)
        # train 5 steps, then save
        state.global_step = 5
        callback.on_save(args, state, control)
        # train 5 more steps, then log
        state.global_step = 10
        logs: dict[str, float] = {}
        callback.on_log(args, state, control, logs=logs)
        # with fix: step anchor reset to 5 after save, so steps = 10 - 5 = 5, time = 1 sec
        assert mock_wandb_run.log.called
        call_args = mock_wandb_run.log.call_args[0][0]
        assert call_args["train/throughput/steps_per_second"] == 5.0  # 5 steps / 1 sec

    def test_callback_reuse_resets_state(
        self,
        args: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        """Verifies callback state is properly reset when reused across training runs."""
        mock_time = mocker.patch("pyine.utils.transformers.callbacks.time.perf_counter")
        mock_time.side_effect = [100.0, 101.0, 102.0, 200.0, 201.0]
        config = callbacks_module.ThroughputLoggingConfig()
        callback = callbacks_module.ThroughputLoggingCallback(config=config, wandb_run=mock_wandb_run)
        state = mocker.MagicMock(global_step=0)
        # first training run
        callback.on_train_begin(args, state, control)
        control.should_evaluate = False
        callback.on_step_end(args, state, control)  # sets saw_training
        assert callback._phase.saw_training is True
        # simulate eval leaving stale state
        callback._eval_start_time = 999.0
        callback._eval_prediction_steps = 99
        # second training run (reuse callback)
        state.global_step = 0
        callback.on_train_begin(args, state, control)
        # state should be reset
        assert callback._phase.saw_training is False
        assert callback._eval_start_time is None
        assert callback._eval_prediction_steps == 0
        assert callback._train_last_log_step == 0

    def test_on_predict_resets_eval_state(
        self,
        args: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        """Verifies on_predict resets eval state to avoid misclassifying subsequent train logs."""
        mock_time = mocker.patch("pyine.utils.transformers.callbacks.time.perf_counter")
        mock_time.side_effect = [100.0, 101.0, 102.0, 103.0]
        config = callbacks_module.ThroughputLoggingConfig()
        callback = callbacks_module.ThroughputLoggingCallback(config=config, wandb_run=mock_wandb_run)
        state = mocker.MagicMock(global_step=10)
        callback.on_train_begin(args, state, control)
        # simulate predict flow: prediction steps but no on_evaluate
        callback.on_prediction_step(args, state, control)
        assert callback._phase.in_eval is True
        assert callback._eval_prediction_steps == 1
        # on_predict should reset state
        callback.on_predict(args, state, control, metrics={})
        assert callback._phase.in_eval is False
        assert callback._eval_start_time is None
        assert callback._eval_prediction_steps == 0

    def test_predict_mid_training_doesnt_break_train_logs(
        self,
        args: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        """Verifies that Trainer.predict() mid-training doesn't misclassify subsequent train logs."""
        mock_time = mocker.patch("pyine.utils.transformers.callbacks.time.perf_counter")
        mock_time.side_effect = [100.0, 101.0, 102.0, 103.0]
        config = callbacks_module.ThroughputLoggingConfig()
        callback = callbacks_module.ThroughputLoggingCallback(config=config, wandb_run=mock_wandb_run)
        state = mocker.MagicMock(global_step=5)
        callback.on_train_begin(args, state, control)
        # simulate predict flow mid-training
        callback.on_prediction_step(args, state, control)
        callback.on_predict(args, state, control, metrics={})
        # subsequent training logs should work correctly
        state.global_step = 10
        logs: dict[str, float] = {}
        callback.on_log(args, state, control, logs=logs)
        # should have train throughput (not misclassified as eval)
        assert mock_wandb_run.log.called
        call_args = mock_wandb_run.log.call_args[0][0]
        assert "train/throughput/steps_per_second" in call_args
        assert "eval/throughput/samples_per_second" not in call_args

    def test_eval_like_logs_without_context_warns(
        self,
        args: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        """Verifies warning when eval-like logs appear without proper eval context."""
        mock_time = mocker.patch("pyine.utils.transformers.callbacks.time.perf_counter")
        mock_time.side_effect = [100.0, 101.0]
        mock_logger = mocker.patch("pyine.utils.transformers.callbacks.logger")
        config = callbacks_module.ThroughputLoggingConfig()
        callback = callbacks_module.ThroughputLoggingCallback(config=config, wandb_run=mock_wandb_run)
        state = mocker.MagicMock(global_step=10)
        callback.on_train_begin(args, state, control)
        # send eval-like logs without being in eval context
        logs: dict[str, typing.Any] = {"eval_loss": 0.3, "eval_accuracy": 0.9}
        callback.on_log(args, state, control, logs=logs)
        # should warn about misclassification
        mock_logger.warning.assert_called_once()
        assert "eval-like logs" in str(mock_logger.warning.call_args)


class _FakeGPUStatsCollector:
    """Minimal fake GPUStatsCollector for testing GPUStatsLoggingCallback."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        nvml_available: bool = False,
    ) -> None:
        self._enabled = enabled
        self._nvml_available = nvml_available
        self._peak_reset_calls: list[list[int] | None] = []

    def is_enabled(self) -> bool:
        return self._enabled

    def is_nvml_available(self) -> bool:
        return self._nvml_available

    def collect_current_device(self) -> typing.Any:
        if not self._enabled:
            return None
        return _FakeGPUStats(
            torch_device_index=0,
            device_key="cuda:0",
            utilization_gpu_percent=75.0 if self._nvml_available else None,
            utilization_mem_controller_percent=50.0 if self._nvml_available else None,
            vram_used_bytes=4_000_000_000 if self._nvml_available else None,
            vram_total_bytes=8_000_000_000 if self._nvml_available else None,
            vram_used_percent=50.0 if self._nvml_available else None,
            power_watts=200.0 if self._nvml_available else None,
            temperature_celsius=70.0 if self._nvml_available else None,
            pytorch_allocated_bytes=2_000_000_000,
            pytorch_reserved_bytes=3_000_000_000,
            pytorch_max_allocated_bytes=2_500_000_000,
            pytorch_max_reserved_bytes=3_500_000_000,
            pytorch_total_bytes=8_000_000_000,
        )

    def collect_all_visible_devices(self) -> list[typing.Any]:
        if not self._enabled:
            return []
        return [self.collect_current_device()]

    def reset_pytorch_peak_stats(
        self,
        device_indices: list[int] | None = None,
    ) -> None:
        self._peak_reset_calls.append(device_indices)


class _FakeGPUStats:
    """Minimal fake GPUStats for testing."""

    def __init__(self, **kwargs: typing.Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class TestGPUStatsLoggingConfig:
    def test_config_defaults(self) -> None:
        config = callbacks_module.GPUStatsLoggingConfig()
        assert config.require_nvml is False
        assert config.collect_all_visible_devices is False
        assert config.only_main_process is True
        assert config.gather_eval_metrics is True
        assert config.sample_every_n_steps == 1

    def test_config_custom_values(self) -> None:
        config = callbacks_module.GPUStatsLoggingConfig(
            require_nvml=True,
            sample_every_n_steps=5,
        )
        assert config.require_nvml is True
        assert config.sample_every_n_steps == 5

    def test_config_forbids_extra_fields(self) -> None:
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            callbacks_module.GPUStatsLoggingConfig(unknown_field=True)  # type: ignore[call-arg]


class TestGPUStatsLoggingCallback:
    @pytest.fixture
    def mock_wandb_run(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> pytest_mock.MockFixture:
        """Create a mock wandb_run object."""
        mock_run = mocker.MagicMock()
        mock_run.log = mocker.MagicMock()
        mock_run.define_metric = mocker.MagicMock()
        return mock_run

    @pytest.fixture
    def config(self) -> callbacks_module.GPUStatsLoggingConfig:
        # use gather_train_metrics="never" for general tests that expect metrics at every log
        # specific tests for "at_phase_end" behavior override this
        return callbacks_module.GPUStatsLoggingConfig(sample_every_n_steps=1, gather_train_metrics="never")

    @pytest.fixture
    def args(self, mocker: pytest_mock.MockerFixture) -> pytest_mock.MockFixture:
        return mocker.MagicMock()

    @pytest.fixture
    def state(self, mocker: pytest_mock.MockerFixture) -> pytest_mock.MockFixture:
        return mocker.MagicMock(global_step=10)

    @pytest.fixture
    def control(self, mocker: pytest_mock.MockerFixture) -> pytest_mock.MockFixture:
        return mocker.MagicMock(should_evaluate=False, should_log=False)

    def test_callback_disabled_when_collector_disabled(
        self,
        config: callbacks_module.GPUStatsLoggingConfig,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        mocker.patch("pyine.utils.distrib.is_distributed", return_value=False)
        mocker.patch("pyine.utils.gpu.GPUStatsCollector", return_value=_FakeGPUStatsCollector(enabled=False))
        callback = callbacks_module.GPUStatsLoggingCallback(config=config, wandb_run=mock_wandb_run)
        callback.on_train_begin(args, state, control)
        assert callback._collector is not None
        assert callback._collector.is_enabled() is False
        # on_step_end should not crash
        callback.on_step_end(args, state, control)
        # on_log should not log anything
        logs: dict[str, float] = {}
        callback.on_log(args, state, control, logs=logs)
        assert not mock_wandb_run.log.called

    def test_on_train_begin_initializes_state(
        self,
        config: callbacks_module.GPUStatsLoggingConfig,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        mocker.patch("pyine.utils.distrib.is_distributed", return_value=False)
        fake_collector = _FakeGPUStatsCollector(enabled=True)
        mocker.patch("pyine.utils.gpu.GPUStatsCollector", return_value=fake_collector)
        callback = callbacks_module.GPUStatsLoggingCallback(config=config, wandb_run=mock_wandb_run)
        callback.on_train_begin(args, state, control)
        assert callback._phase.in_eval is False
        assert callback._phase.eval_pending is False
        assert callback._phase.saw_eval_samples is False
        assert callback._train.sample_count == 0
        assert callback._eval.sample_count == 0

    def test_on_step_end_samples_stats(
        self,
        config: callbacks_module.GPUStatsLoggingConfig,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        mocker.patch("pyine.utils.distrib.is_distributed", return_value=False)
        fake_collector = _FakeGPUStatsCollector(enabled=True, nvml_available=True)
        mocker.patch("pyine.utils.gpu.GPUStatsCollector", return_value=fake_collector)
        callback = callbacks_module.GPUStatsLoggingCallback(config=config, wandb_run=mock_wandb_run)
        callback.on_train_begin(args, state, control)
        state.global_step = 1
        callback.on_step_end(args, state, control)
        # train sampling only updates train accumulators, not eval
        assert callback._train.sample_count == 1
        assert callback._eval.sample_count == 0
        assert "cuda:0" in callback._train.utilization_gpu

    def test_on_step_end_skips_sampling_based_on_interval(
        self,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        config = callbacks_module.GPUStatsLoggingConfig(sample_every_n_steps=5)
        mocker.patch("pyine.utils.distrib.is_distributed", return_value=False)
        fake_collector = _FakeGPUStatsCollector(enabled=True)
        mocker.patch("pyine.utils.gpu.GPUStatsCollector", return_value=fake_collector)
        callback = callbacks_module.GPUStatsLoggingCallback(config=config, wandb_run=mock_wandb_run)
        callback.on_train_begin(args, state, control)
        state.global_step = 3  # not divisible by 5
        callback.on_step_end(args, state, control)
        assert callback._train.sample_count == 0
        state.global_step = 5  # divisible by 5
        callback.on_step_end(args, state, control)
        assert callback._train.sample_count == 1

    def test_on_step_end_sets_eval_pending_when_should_evaluate(
        self,
        config: callbacks_module.GPUStatsLoggingConfig,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        mocker.patch("pyine.utils.distrib.is_distributed", return_value=False)
        fake_collector = _FakeGPUStatsCollector(enabled=True)
        mocker.patch("pyine.utils.gpu.GPUStatsCollector", return_value=fake_collector)
        callback = callbacks_module.GPUStatsLoggingCallback(config=config, wandb_run=mock_wandb_run)
        callback.on_train_begin(args, state, control)
        state.global_step = 1
        callback.on_step_end(args, state, control)
        assert callback._train.sample_count == 1
        control.should_evaluate = True
        control.should_log = False
        callback.on_step_end(args, state, control)
        assert callback._phase.eval_pending is True
        assert callback._eval.sample_count == 0  # phase reset

    def test_on_prediction_step_marks_in_eval(
        self,
        config: callbacks_module.GPUStatsLoggingConfig,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        mocker.patch("pyine.utils.distrib.is_distributed", return_value=False)
        fake_collector = _FakeGPUStatsCollector(enabled=True)
        mocker.patch("pyine.utils.gpu.GPUStatsCollector", return_value=fake_collector)
        callback = callbacks_module.GPUStatsLoggingCallback(config=config, wandb_run=mock_wandb_run)
        callback.on_train_begin(args, state, control)
        callback.on_prediction_step(args, state, control)
        assert callback._phase.in_eval is True
        assert callback._phase.saw_eval_samples is True
        assert callback._eval.sample_count == 1
        assert callback._train.sample_count == 0  # eval_only=True

    def test_on_log_injects_train_metrics(
        self,
        config: callbacks_module.GPUStatsLoggingConfig,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        mocker.patch("pyine.utils.distrib.is_main_process", return_value=True)
        mocker.patch("pyine.utils.distrib.is_distributed", return_value=False)
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch("torch.cuda.current_device", return_value=0)
        mocker.patch("torch.cuda.max_memory_allocated", return_value=2_000_000_000)
        mock_props = mocker.MagicMock()
        mock_props.total_memory = 8_000_000_000
        mocker.patch("torch.cuda.get_device_properties", return_value=mock_props)
        fake_collector = _FakeGPUStatsCollector(enabled=True, nvml_available=True)
        mocker.patch("pyine.utils.gpu.GPUStatsCollector", return_value=fake_collector)
        callback = callbacks_module.GPUStatsLoggingCallback(config=config, wandb_run=mock_wandb_run)
        callback.on_train_begin(args, state, control)
        state.global_step = 1
        callback.on_step_end(args, state, control)
        logs: dict[str, typing.Any] = {"loss": 0.5}
        callback.on_log(args, state, control, logs=logs)
        assert mock_wandb_run.log.called
        call_args = mock_wandb_run.log.call_args[0][0]
        assert "train/gpu/utilization_gpu_percent/mean" in call_args
        assert "train/gpu/pytorch_peak_percent" in call_args
        assert "train/gpu/total_sample_calls" in call_args
        assert call_args["train/gpu/utilization_gpu_percent/mean"] == 75.0

    def test_on_log_injects_eval_metrics_when_in_eval(
        self,
        config: callbacks_module.GPUStatsLoggingConfig,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        """Test eval metrics are injected directly in on_log when _phase.in_eval is True.

        Note on HuggingFace Trainer callback ordering (verified in transformers 4.46+):
        The actual order inside evaluate() is:
        1. on_prediction_step (multiple times) - sets _phase.in_eval=True
        2. on_log (with eval metrics) - we detect via _phase.in_eval=True, inject eval metrics
        3. on_evaluate - cleanup

        So on_log with eval content comes BEFORE on_evaluate.
        """
        mocker.patch("pyine.utils.distrib.is_main_process", return_value=True)
        mocker.patch("pyine.utils.distrib.is_distributed", return_value=False)
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch("torch.cuda.current_device", return_value=0)
        mocker.patch("torch.cuda.max_memory_allocated", return_value=1_000_000_000)
        mock_props = mocker.MagicMock()
        mock_props.total_memory = 8_000_000_000
        mocker.patch("torch.cuda.get_device_properties", return_value=mock_props)
        fake_collector = _FakeGPUStatsCollector(enabled=True, nvml_available=True)
        mocker.patch("pyine.utils.gpu.GPUStatsCollector", return_value=fake_collector)
        callback = callbacks_module.GPUStatsLoggingCallback(config=config, wandb_run=mock_wandb_run)
        callback.on_train_begin(args, state, control)
        # simulate eval: on_prediction_step sets _phase.in_eval=True
        callback.on_prediction_step(args, state, control)
        assert callback._phase.in_eval is True
        assert callback._phase.saw_eval_samples is True
        # now on_log (with eval content) should inject eval metrics
        logs: dict[str, typing.Any] = {"eval_loss": 0.3}
        callback.on_log(args, state, control, logs=logs)
        assert mock_wandb_run.log.called
        call_args = mock_wandb_run.log.call_args[0][0]
        assert "eval/gpu/utilization_gpu_percent/mean" in call_args
        # then on_evaluate comes after and cleans up the rest
        callback.on_evaluate(args, state, control)
        assert callback._phase.saw_eval_samples is False
        assert callback._phase.eval_pending is False
        assert callback._phase.in_eval is False

    def test_on_log_does_not_inject_train_metrics_on_eval_log(
        self,
        config: callbacks_module.GPUStatsLoggingConfig,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        """Test that train metrics are NOT injected when in eval context.

        Note on HuggingFace Trainer callback ordering (verified in transformers 4.46+):
        - on_step_end (samples train stats)
        - on_prediction_step (sets _phase.in_eval=True)
        - on_log (with eval content) - should use eval branch, not train

        We use _phase.in_eval (set in on_prediction_step) to detect eval context.
        """
        mocker.patch("pyine.utils.distrib.is_main_process", return_value=True)
        mocker.patch("pyine.utils.distrib.is_distributed", return_value=False)
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch("torch.cuda.current_device", return_value=0)
        mocker.patch("torch.cuda.max_memory_allocated", return_value=1_000_000_000)
        mock_props = mocker.MagicMock()
        mock_props.total_memory = 8_000_000_000
        mocker.patch("torch.cuda.get_device_properties", return_value=mock_props)
        fake_collector = _FakeGPUStatsCollector(enabled=True, nvml_available=True)
        mocker.patch("pyine.utils.gpu.GPUStatsCollector", return_value=fake_collector)
        callback = callbacks_module.GPUStatsLoggingCallback(config=config, wandb_run=mock_wandb_run)
        callback.on_train_begin(args, state, control)
        state.global_step = 1
        callback.on_step_end(args, state, control)
        # on_prediction_step sets _phase.in_eval=True (correct HF order)
        callback.on_prediction_step(args, state, control)
        assert callback._phase.in_eval is True
        # on_log with eval content - should NOT inject train metrics (uses eval branch)
        logs: dict[str, typing.Any] = {"eval_loss": 0.3}
        callback.on_log(args, state, control, logs=logs)
        # verify wandb logging occurred
        assert mock_wandb_run.log.called
        call_args = mock_wandb_run.log.call_args[0][0]
        # train metrics should NOT be injected (only eval metrics)
        assert "train/gpu/utilization_gpu_percent/mean" not in call_args
        # eval metrics SHOULD be injected
        assert "eval/gpu/utilization_gpu_percent/mean" in call_args

    def test_on_log_skips_when_not_main_process(
        self,
        config: callbacks_module.GPUStatsLoggingConfig,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        mocker.patch("pyine.utils.distrib.is_main_process", return_value=False)
        mocker.patch("pyine.utils.distrib.is_distributed", return_value=False)
        fake_collector = _FakeGPUStatsCollector(enabled=True, nvml_available=True)
        mocker.patch("pyine.utils.gpu.GPUStatsCollector", return_value=fake_collector)
        callback = callbacks_module.GPUStatsLoggingCallback(config=config, wandb_run=mock_wandb_run)
        callback.on_train_begin(args, state, control)
        state.global_step = 1
        callback.on_step_end(args, state, control)
        logs: dict[str, typing.Any] = {"loss": 0.5}
        callback.on_log(args, state, control, logs=logs)
        assert not mock_wandb_run.log.called

    def test_on_evaluate_cleans_up_state(
        self,
        config: callbacks_module.GPUStatsLoggingConfig,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        """Test on_evaluate properly cleans up state flags and resets peak stats."""
        mocker.patch("pyine.utils.distrib.is_distributed", return_value=False)
        fake_collector = _FakeGPUStatsCollector(enabled=True)
        mocker.patch("pyine.utils.gpu.GPUStatsCollector", return_value=fake_collector)
        callback = callbacks_module.GPUStatsLoggingCallback(config=config, wandb_run=mock_wandb_run)
        callback.on_train_begin(args, state, control)
        # set flags that on_evaluate should clean up by simulating prediction steps
        control.should_evaluate = True
        callback.on_step_end(args, state, control)  # sets eval_pending
        callback.on_prediction_step(args, state, control)  # sets in_eval and saw_eval_samples
        callback.on_evaluate(args, state, control)
        assert callback._phase.saw_eval_samples is False
        assert callback._phase.eval_pending is False
        assert callback._phase.in_eval is False
        # verify peak stats were reset
        assert len(fake_collector._peak_reset_calls) > 0

    def test_nvml_unavailable_omits_nvml_metrics(
        self,
        config: callbacks_module.GPUStatsLoggingConfig,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        mocker.patch("pyine.utils.distrib.is_main_process", return_value=True)
        mocker.patch("pyine.utils.distrib.is_distributed", return_value=False)
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch("torch.cuda.current_device", return_value=0)
        mocker.patch("torch.cuda.max_memory_allocated", return_value=1_000_000_000)
        mock_props = mocker.MagicMock()
        mock_props.total_memory = 8_000_000_000
        mocker.patch("torch.cuda.get_device_properties", return_value=mock_props)
        fake_collector = _FakeGPUStatsCollector(enabled=True, nvml_available=False)
        mocker.patch("pyine.utils.gpu.GPUStatsCollector", return_value=fake_collector)
        callback = callbacks_module.GPUStatsLoggingCallback(config=config, wandb_run=mock_wandb_run)
        callback.on_train_begin(args, state, control)
        state.global_step = 1
        callback.on_step_end(args, state, control)
        logs: dict[str, typing.Any] = {"loss": 0.5}
        callback.on_log(args, state, control, logs=logs)
        assert mock_wandb_run.log.called
        call_args = mock_wandb_run.log.call_args[0][0]
        # NVML metrics should be absent
        assert "train/gpu/utilization_gpu_percent/mean" not in call_args
        assert "train/gpu/power_watts/mean" not in call_args
        # PyTorch metrics should be present
        assert "train/gpu/pytorch_allocated_percent/mean" in call_args
        assert "train/gpu/pytorch_peak_percent" in call_args

    def test_callback_inheritance(
        self,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        config = callbacks_module.GPUStatsLoggingConfig()
        callback = callbacks_module.GPUStatsLoggingCallback(config=config, wandb_run=mock_wandb_run)
        assert isinstance(callback, transformers.TrainerCallback)

    def test_config_validation_only_main_process_false_with_gather(
        self,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        """Test that only_main_process=False + gather_eval_metrics=True raises error."""
        config = callbacks_module.GPUStatsLoggingConfig(
            only_main_process=False,
            gather_eval_metrics=True,
        )
        mocker.patch("pyine.utils.distrib.is_distributed", return_value=True)
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch("torch.cuda.device_count", return_value=1)
        fake_collector = _FakeGPUStatsCollector(enabled=True)
        mocker.patch("pyine.utils.gpu.GPUStatsCollector", return_value=fake_collector)
        callback = callbacks_module.GPUStatsLoggingCallback(config=config, wandb_run=mock_wandb_run)
        with pytest.raises(ValueError, match="Cannot use only_main_process=False"):
            callback.on_train_begin(args, state, control)

    def test_config_validation_collect_all_visible_devices_with_distributed_multi_gpu(
        self,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        """Test double-counting guard: collect_all_visible_devices + distributed + multi-GPU raises error."""
        config = callbacks_module.GPUStatsLoggingConfig(
            collect_all_visible_devices=True,
            gather_eval_metrics=True,  # would cause double-counting
        )
        mocker.patch("pyine.utils.distrib.is_distributed", return_value=True)
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch("torch.cuda.device_count", return_value=2)  # multiple GPUs
        fake_collector = _FakeGPUStatsCollector(enabled=True)
        mocker.patch("pyine.utils.gpu.GPUStatsCollector", return_value=fake_collector)
        callback = callbacks_module.GPUStatsLoggingCallback(config=config, wandb_run=mock_wandb_run)
        with pytest.raises(ValueError, match="Cannot use collect_all_visible_devices=True"):
            callback.on_train_begin(args, state, control)

    def test_on_evaluate_resets_peak_stats(
        self,
        config: callbacks_module.GPUStatsLoggingConfig,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        """Test that on_evaluate resets peak stats to prevent contamination."""
        mocker.patch("pyine.utils.distrib.is_distributed", return_value=False)
        fake_collector = _FakeGPUStatsCollector(enabled=True)
        mocker.patch("pyine.utils.gpu.GPUStatsCollector", return_value=fake_collector)
        callback = callbacks_module.GPUStatsLoggingCallback(config=config, wandb_run=mock_wandb_run)
        callback.on_train_begin(args, state, control)
        initial_reset_count = len(fake_collector._peak_reset_calls)
        # simulate eval phase with prediction steps
        callback.on_prediction_step(args, state, control)
        callback.on_evaluate(args, state, control)
        # should have called reset_pytorch_peak_stats
        assert len(fake_collector._peak_reset_calls) > initial_reset_count

    def test_peak_reset_all_devices_when_collect_all_visible(
        self,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        """Test that peak stats reset covers all devices when collect_all_visible_devices=True."""
        config = callbacks_module.GPUStatsLoggingConfig(
            collect_all_visible_devices=True,
            gather_eval_metrics=False,  # avoid the double-counting guard
        )
        mocker.patch("pyine.utils.distrib.is_distributed", return_value=False)
        mocker.patch("torch.cuda.device_count", return_value=2)  # two GPUs visible
        fake_collector = _FakeGPUStatsCollector(enabled=True)
        mocker.patch("pyine.utils.gpu.GPUStatsCollector", return_value=fake_collector)
        callback = callbacks_module.GPUStatsLoggingCallback(config=config, wandb_run=mock_wandb_run)
        callback.on_train_begin(args, state, control)
        # should have called reset with list of all device indices [0, 1]
        assert len(fake_collector._peak_reset_calls) == 1
        assert fake_collector._peak_reset_calls[0] == [0, 1]

    def test_gather_train_metrics_default_is_at_phase_end(self) -> None:
        """Test that gather_train_metrics defaults to 'at_phase_end'."""
        config = callbacks_module.GPUStatsLoggingConfig()
        assert config.gather_train_metrics == "at_phase_end"

    def test_gather_train_metrics_never_uses_local_metrics(
        self,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        """Test that gather_train_metrics='never' uses local metrics only."""
        config = callbacks_module.GPUStatsLoggingConfig(gather_train_metrics="never")
        mocker.patch("pyine.utils.distrib.is_distributed", return_value=True)
        mocker.patch("pyine.utils.distrib.is_main_process", return_value=True)
        fake_collector = _FakeGPUStatsCollector(enabled=True)
        mocker.patch("pyine.utils.gpu.GPUStatsCollector", return_value=fake_collector)
        callback = callbacks_module.GPUStatsLoggingCallback(config=config, wandb_run=mock_wandb_run)
        callback.on_train_begin(args, state, control)
        state.global_step = 10  # divisible by sample_every_n_steps
        control.should_evaluate = False
        callback.on_step_end(args, state, control)
        # mock the gather function to track if it's called
        gather_mock = mocker.patch.object(callback, "_gather_and_compute_train_metrics")
        local_mock = mocker.patch.object(callback, "_compute_train_metrics", return_value={})
        logs: dict[str, typing.Any] = {"loss": 0.5}
        callback.on_log(args, state, control, logs=logs)
        # should use local, not gathered
        local_mock.assert_called_once()
        gather_mock.assert_not_called()

    def test_gather_train_metrics_always_uses_gathered_metrics(
        self,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        """Test that gather_train_metrics='always' uses gathered metrics."""
        config = callbacks_module.GPUStatsLoggingConfig(gather_train_metrics="always")
        mocker.patch("pyine.utils.distrib.is_distributed", return_value=True)
        mocker.patch("pyine.utils.distrib.is_main_process", return_value=True)
        fake_collector = _FakeGPUStatsCollector(enabled=True)
        mocker.patch("pyine.utils.gpu.GPUStatsCollector", return_value=fake_collector)
        callback = callbacks_module.GPUStatsLoggingCallback(config=config, wandb_run=mock_wandb_run)
        # mock _is_gather_safe to return True (simulates initialized process group)
        mocker.patch.object(callback, "_is_gather_safe", return_value=True)
        # mock all_gather_objects to simulate distributed gather (avoids fail-fast check)
        mocker.patch("pyine.utils.distrib.all_gather_objects", side_effect=lambda x: [x])
        callback.on_train_begin(args, state, control)
        state.global_step = 10
        control.should_evaluate = False
        callback.on_step_end(args, state, control)
        gather_mock = mocker.patch.object(callback, "_gather_and_compute_train_metrics", return_value={})
        local_mock = mocker.patch.object(callback, "_compute_train_metrics")
        logs: dict[str, typing.Any] = {"loss": 0.5}
        callback.on_log(args, state, control, logs=logs)
        # should use gathered, not local
        gather_mock.assert_called_once()
        local_mock.assert_not_called()

    def test_gather_train_metrics_at_phase_end_accumulates_and_emits_only_at_phase_end(
        self,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        """Test that gather_train_metrics='at_phase_end' accumulates and only emits at phase end.

        With the 'at_phase_end' semantic:
        - Intermediate logs: no metrics emitted (accumulate only)
        - Phase end log (when _eval_pending): gather and emit all accumulated stats
        """
        config = callbacks_module.GPUStatsLoggingConfig(gather_train_metrics="at_phase_end")
        mocker.patch("pyine.utils.distrib.is_distributed", return_value=True)
        mocker.patch("pyine.utils.distrib.is_main_process", return_value=True)
        fake_collector = _FakeGPUStatsCollector(enabled=True)
        mocker.patch("pyine.utils.gpu.GPUStatsCollector", return_value=fake_collector)
        callback = callbacks_module.GPUStatsLoggingCallback(config=config, wandb_run=mock_wandb_run)
        # mock _is_gather_safe to return True (simulates initialized process group)
        mocker.patch.object(callback, "_is_gather_safe", return_value=True)
        # mock all_gather_objects to simulate distributed gather (avoids fail-fast check)
        mocker.patch("pyine.utils.distrib.all_gather_objects", side_effect=lambda x: [x])
        callback.on_train_begin(args, state, control)
        # first log: no eval pending, should NOT emit anything (accumulate only)
        state.global_step = 10
        control.should_evaluate = False
        callback.on_step_end(args, state, control)
        gather_mock = mocker.patch.object(callback, "_gather_and_compute_train_metrics", return_value={})
        local_mock = mocker.patch.object(callback, "_compute_train_metrics", return_value={})
        logs: dict[str, typing.Any] = {"loss": 0.5}
        callback.on_log(args, state, control, logs=logs)
        # neither local nor gather should be called (skip emit entirely)
        local_mock.assert_not_called()
        gather_mock.assert_not_called()
        # accumulators should NOT be reset (still accumulating)
        assert callback._train.sample_count == 1  # from on_step_end
        # second log: eval pending (phase end), should gather and emit
        gather_mock.reset_mock()
        local_mock.reset_mock()
        state.global_step = 20
        control.should_evaluate = True
        control.should_log = True
        callback.on_step_end(args, state, control)  # sets _eval_pending = True, samples again
        assert callback._train.sample_count == 2  # accumulated from both steps
        logs2: dict[str, typing.Any] = {"loss": 0.4}
        callback.on_log(args, state, control, logs=logs2)
        # should gather (distributed) and emit
        gather_mock.assert_called_once()
        local_mock.assert_not_called()
        # accumulators should be reset after phase end
        assert callback._train.sample_count == 0

    def test_gather_train_metrics_at_phase_end_emits_in_eval_on_log_when_should_log_false(
        self,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
        mock_wandb_run: pytest_mock.MockFixture,
    ) -> None:
        """Test delayed train flush when should_evaluate=True but should_log=False.

        This tests the edge case where:
        - gather_train_metrics='at_phase_end'
        - should_evaluate=True but should_log=False (e.g., eval_steps != logging_steps)
        - Train metrics should be emitted in the eval on_log with train/gpu/ prefix
        - Peak stats should use the captured train phase value, not current (eval) value
        """
        config = callbacks_module.GPUStatsLoggingConfig(
            gather_train_metrics="at_phase_end",
            sample_every_n_steps=1,
        )
        mocker.patch("pyine.utils.distrib.is_distributed", return_value=False)
        mocker.patch("pyine.utils.distrib.is_main_process", return_value=True)
        fake_collector = _FakeGPUStatsCollector(enabled=True)
        mocker.patch("pyine.utils.gpu.GPUStatsCollector", return_value=fake_collector)
        callback = callbacks_module.GPUStatsLoggingCallback(config=config, wandb_run=mock_wandb_run)
        callback.on_train_begin(args, state, control)
        # sample during training
        state.global_step = 10
        control.should_evaluate = False
        control.should_log = False
        callback.on_step_end(args, state, control)
        assert callback._train.sample_count == 1
        # now eval is triggered but no train log
        state.global_step = 20
        control.should_evaluate = True
        control.should_log = False  # key: no train log will happen!
        callback.on_step_end(args, state, control)
        # verify state after on_step_end
        assert callback._phase.eval_pending is True
        assert callback._train_phase_needs_flush is True
        assert callback._stashed_train_peak_percent is not None  # peak was captured
        stashed_peak = callback._stashed_train_peak_percent
        # now eval runs
        callback.on_prediction_step(args, state, control)
        assert callback._phase.in_eval is True
        # eval on_log: should emit BOTH train metrics (delayed) and eval metrics
        logs: dict[str, typing.Any] = {"eval_loss": 0.3}
        callback.on_log(args, state, control, logs=logs)
        # verify wandb logging occurred
        assert mock_wandb_run.log.called
        # check all log calls (there should be 2: one for train, one for eval)
        assert mock_wandb_run.log.call_count >= 2
        all_logged_metrics = {}
        for call in mock_wandb_run.log.call_args_list:
            all_logged_metrics.update(call[0][0])
        # verify train metrics were logged with train/gpu/ prefix
        train_keys = [k for k in all_logged_metrics if k.startswith("train/gpu/")]
        assert len(train_keys) > 0, "expected train/gpu/ metrics in delayed flush"
        # verify peak percent used the stashed value
        if "train/gpu/pytorch_peak_percent" in all_logged_metrics:
            assert all_logged_metrics["train/gpu/pytorch_peak_percent"] == stashed_peak
        # verify train state was cleaned up
        assert callback._train_phase_needs_flush is False
        assert callback._stashed_train_peak_percent is None
        assert callback._train.sample_count == 0
        # verify eval metrics were also logged
        eval_keys = [k for k in all_logged_metrics if k.startswith("eval/gpu/")]
        assert len(eval_keys) > 0, "expected eval/gpu/ metrics"
        # note: _phase.in_eval is cleared by handle_evaluate() in on_evaluate
        # at this point, on_log has completed but on_evaluate hasn't been called yet
