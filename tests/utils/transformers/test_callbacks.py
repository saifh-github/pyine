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
        assert callback._in_eval is False
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
        assert callback._in_eval is True
        assert reward_manager.key_prefix == "eval"
        # should have flushed train stats (empty) before switching
        assert len(reward_manager.flush_calls) == 1
        assert reward_manager.flush_calls[0] == state.global_step
        # step should be cleared for eval
        assert reward_manager.step is None

    def test_on_step_end_switches_to_eval_when_should_evaluate_true(
        self,
        reward_manager: _FakeRewardManager,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
    ) -> None:
        callback = callbacks_module.RewardLoggingCallback(reward_manager=reward_manager)
        callback._in_eval = False
        reward_manager.key_prefix = "train"
        control.should_evaluate = True
        callback.on_step_end(args, state, control)
        assert callback._in_eval is True
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
        callback._in_eval = False
        reward_manager.key_prefix = "train"
        control.should_evaluate = False
        callback.on_step_end(args, state, control)
        assert callback._in_eval is False
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
        callback._in_eval = True
        reward_manager.key_prefix = "eval"
        control.should_evaluate = True
        callback.on_step_end(args, state, control)
        assert callback._in_eval is True
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
        callback._in_eval = False
        reward_manager.key_prefix = "train"
        control.should_evaluate = True
        callback.on_epoch_end(args, state, control)
        assert callback._in_eval is True
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
        callback._in_eval = False
        reward_manager.key_prefix = "train"
        control.should_evaluate = False
        callback.on_epoch_end(args, state, control)
        assert callback._in_eval is False
        assert reward_manager.key_prefix == "train"
        assert len(reward_manager.flush_calls) == 0

    def test_switch_to_eval_clears_step(
        self,
        reward_manager: _FakeRewardManager,
    ) -> None:
        callback = callbacks_module.RewardLoggingCallback(reward_manager=reward_manager)
        callback._in_eval = False
        reward_manager.step = 100
        callback._switch_to_eval(step=100)
        assert reward_manager.step is None
        assert callback._in_eval is True
        assert reward_manager.key_prefix == "eval"

    def test_on_prediction_step_sets_prefix_without_flush(
        self,
        reward_manager: _FakeRewardManager,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
    ) -> None:
        callback = callbacks_module.RewardLoggingCallback(reward_manager=reward_manager)
        callback._in_eval = False
        reward_manager.key_prefix = "train"
        callback.on_prediction_step(args, state, control)
        assert callback._in_eval is True
        assert reward_manager.key_prefix == "eval"
        # should NOT have flushed (on_prediction_step is too late)
        assert len(reward_manager.flush_calls) == 0
        assert reward_manager.step is None

    def test_on_prediction_step_only_clears_step_if_already_in_eval(
        self,
        reward_manager: _FakeRewardManager,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
    ) -> None:
        callback = callbacks_module.RewardLoggingCallback(reward_manager=reward_manager)
        callback._in_eval = True
        reward_manager.key_prefix = "eval"
        reward_manager.step = 50
        callback.on_prediction_step(args, state, control)
        assert callback._in_eval is True
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
        callback._in_eval = True
        callback._saw_eval_prediction_step = False
        reward_manager.key_prefix = "eval"
        callback.on_evaluate(args, state, control)
        # should NOT have flushed (no eval samples processed)
        assert len(reward_manager.flush_calls) == 0
        # should have reset to train mode
        assert callback._in_eval is False
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
        callback._in_eval = True
        callback._saw_eval_prediction_step = True
        reward_manager.key_prefix = "eval"
        callback.on_evaluate(args, state, control)
        # should have flushed
        assert len(reward_manager.flush_calls) == 1
        assert reward_manager.flush_calls[0] == state.global_step
        # should have reset flag
        assert callback._saw_eval_prediction_step is False
        # should have reset to train mode
        assert callback._in_eval is False
        assert reward_manager.key_prefix == "train"

    def test_on_prediction_step_sets_saw_eval_prediction_step_flag(
        self,
        reward_manager: _FakeRewardManager,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
    ) -> None:
        """Test that on_prediction_step sets _saw_eval_prediction_step flag."""
        callback = callbacks_module.RewardLoggingCallback(reward_manager=reward_manager)
        callback._in_eval = True
        callback._saw_eval_prediction_step = False
        callback.on_prediction_step(args, state, control)
        assert callback._saw_eval_prediction_step is True


class TestThroughputLoggingCallback:
    @pytest.fixture
    def args(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> pytest_mock.MockFixture:
        return mocker.MagicMock(
            per_device_train_batch_size=4,
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
        return mocker.MagicMock()

    def test_should_log_main_process(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        mocker.patch("pyine.utils.distrib.is_main_process", return_value=True)
        callback = callbacks_module.ThroughputLoggingCallback(only_main_process=True)
        assert callback._should_log() is True

    def test_should_log_worker_process(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        mocker.patch("pyine.utils.distrib.is_main_process", return_value=False)
        callback = callbacks_module.ThroughputLoggingCallback(only_main_process=True)
        assert callback._should_log() is False

    def test_should_log_all_processes(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        mocker.patch("pyine.utils.distrib.is_main_process", return_value=False)
        callback = callbacks_module.ThroughputLoggingCallback(only_main_process=False)
        assert callback._should_log() is True

    def test_effective_batch_size(
        self,
        args: pytest_mock.MockFixture,
    ) -> None:
        callback = callbacks_module.ThroughputLoggingCallback()
        # 4 * 2 * 2 = 16
        assert callback._get_effective_batch_size(args) == 16

    def test_effective_batch_size_missing_attrs(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        callback = callbacks_module.ThroughputLoggingCallback()
        args = mocker.MagicMock(spec=[])  # no attributes
        # defaults to 1 * 1 * 1 = 1
        assert callback._get_effective_batch_size(args) == 1

    def test_on_train_begin_initializes_state(
        self,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
    ) -> None:
        callback = callbacks_module.ThroughputLoggingCallback()
        assert callback._last_log_time is None
        callback.on_train_begin(args, state, control)
        assert callback._last_log_time is not None
        assert callback._last_log_step == 0

    def test_on_log_calculates_throughput(
        self,
        args: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        # patch perf_counter to return fixed values: 100.0 at train_begin, 102.0 at on_log
        mock_time = mocker.patch("pyine.utils.transformers.callbacks.time.perf_counter")
        mock_time.side_effect = [100.0, 102.0]  # 2 seconds elapsed
        callback = callbacks_module.ThroughputLoggingCallback()
        state = mocker.MagicMock(global_step=0)
        callback.on_train_begin(args, state, control)
        state.global_step = 10  # 10 steps completed
        logs: dict[str, float] = {}
        callback.on_log(args, state, control, logs=logs)
        # effective batch size = 4 * 2 * 2 = 16
        # samples = 10 * 16 = 160, elapsed = 2 seconds
        assert logs["throughput/samples_per_second"] == 80.0  # 160 / 2
        assert logs["throughput/steps_per_second"] == 5.0  # 10 / 2

    def test_on_log_skips_on_worker_process(
        self,
        args: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        mocker.patch("pyine.utils.distrib.is_main_process", return_value=False)
        callback = callbacks_module.ThroughputLoggingCallback(only_main_process=True)
        state = mocker.MagicMock(global_step=10)
        logs: dict[str, float] = {}
        callback.on_log(args, state, control, logs=logs)
        assert "throughput/samples_per_second" not in logs
        assert "throughput/steps_per_second" not in logs

    def test_on_log_handles_zero_elapsed(
        self,
        args: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        # patch perf_counter to return same value (elapsed == 0)
        mock_time = mocker.patch("pyine.utils.transformers.callbacks.time.perf_counter")
        mock_time.return_value = 100.0
        callback = callbacks_module.ThroughputLoggingCallback()
        state = mocker.MagicMock(global_step=0)
        callback.on_train_begin(args, state, control)
        state.global_step = 10  # steps advanced but time didn't
        logs: dict[str, float] = {}
        callback.on_log(args, state, control, logs=logs)
        # should skip logging due to zero elapsed time (avoids division by zero)
        assert "throughput/samples_per_second" not in logs

    def test_on_log_handles_zero_steps(
        self,
        args: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        # patch perf_counter: time advances but steps don't
        mock_time = mocker.patch("pyine.utils.transformers.callbacks.time.perf_counter")
        mock_time.side_effect = [100.0, 101.0]
        callback = callbacks_module.ThroughputLoggingCallback()
        state = mocker.MagicMock(global_step=0)
        callback.on_train_begin(args, state, control)
        # global_step hasn't changed (steps_delta = 0)
        logs: dict[str, float] = {}
        callback.on_log(args, state, control, logs=logs)
        assert "throughput/samples_per_second" not in logs

    def test_custom_prefix(
        self,
        args: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        mock_time = mocker.patch("pyine.utils.transformers.callbacks.time.perf_counter")
        mock_time.side_effect = [100.0, 101.0]  # 1 second elapsed
        callback = callbacks_module.ThroughputLoggingCallback(prefix="train/speed/")
        state = mocker.MagicMock(global_step=0)
        callback.on_train_begin(args, state, control)
        state.global_step = 5
        logs: dict[str, float] = {}
        callback.on_log(args, state, control, logs=logs)
        assert "train/speed/samples_per_second" in logs
        assert "train/speed/steps_per_second" in logs
        # 5 steps * 16 batch size / 1 second = 80 samples/sec
        assert logs["train/speed/samples_per_second"] == 80.0
        assert logs["train/speed/steps_per_second"] == 5.0

    def test_on_save_resets_timing(
        self,
        args: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        mock_time = mocker.patch("pyine.utils.transformers.callbacks.time.perf_counter")
        mock_time.side_effect = [100.0, 105.0, 106.0]  # train_begin, on_save, on_log
        callback = callbacks_module.ThroughputLoggingCallback()
        state = mocker.MagicMock(global_step=0)
        callback.on_train_begin(args, state, control)
        # checkpoint happens at t=105 (5 sec of I/O)
        callback.on_save(args, state, control)
        # on_log happens at t=106, but timer was reset at t=105
        state.global_step = 10
        logs: dict[str, float] = {}
        callback.on_log(args, state, control, logs=logs)
        # elapsed = 106 - 105 = 1 second (not 6 seconds)
        assert logs["throughput/steps_per_second"] == 10.0

    def test_resume_training_with_nonzero_step(
        self,
        args: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Test that resuming training (global_step > 0) works correctly."""
        mock_time = mocker.patch("pyine.utils.transformers.callbacks.time.perf_counter")
        mock_time.side_effect = [100.0, 101.0]
        callback = callbacks_module.ThroughputLoggingCallback()
        state = mocker.MagicMock(global_step=500)  # resumed from step 500
        callback.on_train_begin(args, state, control)
        assert callback._last_log_step == 500
        state.global_step = 510  # 10 more steps
        logs: dict[str, float] = {}
        callback.on_log(args, state, control, logs=logs)
        assert logs["throughput/steps_per_second"] == 10.0

    def test_callback_inheritance(self) -> None:
        callback = callbacks_module.ThroughputLoggingCallback()
        assert isinstance(callback, transformers.TrainerCallback)


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
    ) -> None:
        mocker.patch("pyine.utils.distrib.is_distributed", return_value=False)
        mocker.patch("pyine.utils.gpu.GPUStatsCollector", return_value=_FakeGPUStatsCollector(enabled=False))
        callback = callbacks_module.GPUStatsLoggingCallback(config=config)
        callback.on_train_begin(args, state, control)
        assert callback._collector is not None
        assert callback._collector.is_enabled() is False
        # on_step_end should not crash
        callback.on_step_end(args, state, control)
        # on_log should not inject anything
        logs: dict[str, float] = {}
        callback.on_log(args, state, control, logs=logs)
        assert len(logs) == 0

    def test_on_train_begin_initializes_state(
        self,
        config: callbacks_module.GPUStatsLoggingConfig,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        mocker.patch("pyine.utils.distrib.is_distributed", return_value=False)
        fake_collector = _FakeGPUStatsCollector(enabled=True)
        mocker.patch("pyine.utils.gpu.GPUStatsCollector", return_value=fake_collector)
        callback = callbacks_module.GPUStatsLoggingCallback(config=config)
        callback.on_train_begin(args, state, control)
        assert callback._in_eval is False
        assert callback._eval_pending is False
        assert callback._saw_eval_prediction_step is False
        assert callback._train.sample_count == 0
        assert callback._eval.sample_count == 0

    def test_on_step_end_samples_stats(
        self,
        config: callbacks_module.GPUStatsLoggingConfig,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        mocker.patch("pyine.utils.distrib.is_distributed", return_value=False)
        fake_collector = _FakeGPUStatsCollector(enabled=True, nvml_available=True)
        mocker.patch("pyine.utils.gpu.GPUStatsCollector", return_value=fake_collector)
        callback = callbacks_module.GPUStatsLoggingCallback(config=config)
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
    ) -> None:
        config = callbacks_module.GPUStatsLoggingConfig(sample_every_n_steps=5)
        mocker.patch("pyine.utils.distrib.is_distributed", return_value=False)
        fake_collector = _FakeGPUStatsCollector(enabled=True)
        mocker.patch("pyine.utils.gpu.GPUStatsCollector", return_value=fake_collector)
        callback = callbacks_module.GPUStatsLoggingCallback(config=config)
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
    ) -> None:
        mocker.patch("pyine.utils.distrib.is_distributed", return_value=False)
        fake_collector = _FakeGPUStatsCollector(enabled=True)
        mocker.patch("pyine.utils.gpu.GPUStatsCollector", return_value=fake_collector)
        callback = callbacks_module.GPUStatsLoggingCallback(config=config)
        callback.on_train_begin(args, state, control)
        state.global_step = 1
        callback.on_step_end(args, state, control)
        assert callback._train.sample_count == 1
        control.should_evaluate = True
        control.should_log = False
        callback.on_step_end(args, state, control)
        assert callback._eval_pending is True
        assert callback._eval.sample_count == 0  # phase reset

    def test_on_prediction_step_marks_in_eval(
        self,
        config: callbacks_module.GPUStatsLoggingConfig,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        mocker.patch("pyine.utils.distrib.is_distributed", return_value=False)
        fake_collector = _FakeGPUStatsCollector(enabled=True)
        mocker.patch("pyine.utils.gpu.GPUStatsCollector", return_value=fake_collector)
        callback = callbacks_module.GPUStatsLoggingCallback(config=config)
        callback.on_train_begin(args, state, control)
        callback.on_prediction_step(args, state, control)
        assert callback._in_eval is True
        assert callback._saw_eval_prediction_step is True
        assert callback._eval.sample_count == 1
        assert callback._train.sample_count == 0  # eval_only=True

    def test_on_log_injects_train_metrics(
        self,
        config: callbacks_module.GPUStatsLoggingConfig,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
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
        callback = callbacks_module.GPUStatsLoggingCallback(config=config)
        callback.on_train_begin(args, state, control)
        state.global_step = 1
        callback.on_step_end(args, state, control)
        logs: dict[str, typing.Any] = {"loss": 0.5}
        callback.on_log(args, state, control, logs=logs)
        assert "train/gpu/utilization_gpu_percent/mean" in logs
        assert "train/gpu/pytorch_peak_percent" in logs
        assert "train/gpu/total_sample_calls" in logs
        assert logs["train/gpu/utilization_gpu_percent/mean"] == 75.0

    def test_on_log_injects_eval_metrics_when_in_eval(
        self,
        config: callbacks_module.GPUStatsLoggingConfig,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Test eval metrics are injected directly in on_log when _in_eval is True.

        Note on HuggingFace Trainer callback ordering (verified in transformers 4.46+):
        The actual order inside evaluate() is:
        1. on_prediction_step (multiple times) - sets _in_eval=True
        2. on_log (with eval metrics) - we detect via _in_eval=True, inject eval metrics
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
        callback = callbacks_module.GPUStatsLoggingCallback(config=config)
        callback.on_train_begin(args, state, control)
        # simulate eval: on_prediction_step sets _in_eval=True
        callback.on_prediction_step(args, state, control)
        assert callback._in_eval is True
        assert callback._saw_eval_prediction_step is True
        # now on_log (with eval content) should inject eval metrics and clear _in_eval
        logs: dict[str, typing.Any] = {"eval_loss": 0.3}
        callback.on_log(args, state, control, logs=logs)
        assert "eval/gpu/utilization_gpu_percent/mean" in logs
        assert callback._in_eval is False  # cleared by on_log
        # then on_evaluate comes after and cleans up the rest
        callback.on_evaluate(args, state, control)
        assert callback._saw_eval_prediction_step is False
        assert callback._eval_pending is False

    def test_on_log_does_not_inject_train_metrics_on_eval_log(
        self,
        config: callbacks_module.GPUStatsLoggingConfig,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Test that train metrics are NOT injected when in eval context.

        Note on HuggingFace Trainer callback ordering (verified in transformers 4.46+):
        - on_step_end (samples train stats)
        - on_prediction_step (sets _in_eval=True)
        - on_log (with eval content) - should use eval branch, not train

        We use _in_eval (set in on_prediction_step) to detect eval context.
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
        callback = callbacks_module.GPUStatsLoggingCallback(config=config)
        callback.on_train_begin(args, state, control)
        state.global_step = 1
        callback.on_step_end(args, state, control)
        # on_prediction_step sets _in_eval=True (correct HF order)
        callback.on_prediction_step(args, state, control)
        assert callback._in_eval is True
        # on_log with eval content - should NOT inject train metrics (uses eval branch)
        logs: dict[str, typing.Any] = {"eval_loss": 0.3}
        callback.on_log(args, state, control, logs=logs)
        # train metrics should NOT be injected (only eval metrics)
        assert "train/gpu/utilization_gpu_percent/mean" not in logs
        # eval metrics SHOULD be injected
        assert "eval/gpu/utilization_gpu_percent/mean" in logs

    def test_on_log_skips_when_not_main_process(
        self,
        config: callbacks_module.GPUStatsLoggingConfig,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        mocker.patch("pyine.utils.distrib.is_main_process", return_value=False)
        mocker.patch("pyine.utils.distrib.is_distributed", return_value=False)
        fake_collector = _FakeGPUStatsCollector(enabled=True, nvml_available=True)
        mocker.patch("pyine.utils.gpu.GPUStatsCollector", return_value=fake_collector)
        callback = callbacks_module.GPUStatsLoggingCallback(config=config)
        callback.on_train_begin(args, state, control)
        state.global_step = 1
        callback.on_step_end(args, state, control)
        logs: dict[str, typing.Any] = {"loss": 0.5}
        callback.on_log(args, state, control, logs=logs)
        assert "train/gpu/utilization_gpu_percent/mean" not in logs

    def test_on_evaluate_cleans_up_state(
        self,
        config: callbacks_module.GPUStatsLoggingConfig,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Test on_evaluate properly cleans up state flags and resets peak stats."""
        mocker.patch("pyine.utils.distrib.is_distributed", return_value=False)
        fake_collector = _FakeGPUStatsCollector(enabled=True)
        mocker.patch("pyine.utils.gpu.GPUStatsCollector", return_value=fake_collector)
        callback = callbacks_module.GPUStatsLoggingCallback(config=config)
        callback.on_train_begin(args, state, control)
        # set flags that on_evaluate should clean up
        callback._eval_pending = True
        callback._saw_eval_prediction_step = True
        callback.on_evaluate(args, state, control)
        assert callback._saw_eval_prediction_step is False
        assert callback._eval_pending is False
        # verify peak stats were reset
        assert len(fake_collector._peak_reset_calls) > 0

    def test_nvml_unavailable_omits_nvml_metrics(
        self,
        config: callbacks_module.GPUStatsLoggingConfig,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
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
        callback = callbacks_module.GPUStatsLoggingCallback(config=config)
        callback.on_train_begin(args, state, control)
        state.global_step = 1
        callback.on_step_end(args, state, control)
        logs: dict[str, typing.Any] = {"loss": 0.5}
        callback.on_log(args, state, control, logs=logs)
        # NVML metrics should be absent
        assert "train/gpu/utilization_gpu_percent/mean" not in logs
        assert "train/gpu/power_watts/mean" not in logs
        # PyTorch metrics should be present
        assert "train/gpu/pytorch_allocated_percent/mean" in logs
        assert "train/gpu/pytorch_peak_percent" in logs

    def test_callback_inheritance(self) -> None:
        callback = callbacks_module.GPUStatsLoggingCallback()
        assert isinstance(callback, transformers.TrainerCallback)

    def test_config_validation_only_main_process_false_with_gather(
        self,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
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
        callback = callbacks_module.GPUStatsLoggingCallback(config=config)
        with pytest.raises(ValueError, match="Cannot use only_main_process=False"):
            callback.on_train_begin(args, state, control)

    def test_config_validation_collect_all_visible_devices_with_distributed_multi_gpu(
        self,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
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
        callback = callbacks_module.GPUStatsLoggingCallback(config=config)
        with pytest.raises(ValueError, match="Cannot use collect_all_visible_devices=True"):
            callback.on_train_begin(args, state, control)

    def test_on_evaluate_resets_peak_stats(
        self,
        config: callbacks_module.GPUStatsLoggingConfig,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Test that on_evaluate resets peak stats to prevent contamination."""
        mocker.patch("pyine.utils.distrib.is_distributed", return_value=False)
        fake_collector = _FakeGPUStatsCollector(enabled=True)
        mocker.patch("pyine.utils.gpu.GPUStatsCollector", return_value=fake_collector)
        callback = callbacks_module.GPUStatsLoggingCallback(config=config)
        callback.on_train_begin(args, state, control)
        initial_reset_count = len(fake_collector._peak_reset_calls)
        callback._saw_eval_prediction_step = True
        callback.on_evaluate(args, state, control)
        # should have called reset_pytorch_peak_stats
        assert len(fake_collector._peak_reset_calls) > initial_reset_count

    def test_peak_reset_all_devices_when_collect_all_visible(
        self,
        args: pytest_mock.MockFixture,
        state: pytest_mock.MockFixture,
        control: pytest_mock.MockFixture,
        mocker: pytest_mock.MockerFixture,
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
        callback = callbacks_module.GPUStatsLoggingCallback(config=config)
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
    ) -> None:
        """Test that gather_train_metrics='never' uses local metrics only."""
        config = callbacks_module.GPUStatsLoggingConfig(gather_train_metrics="never")
        mocker.patch("pyine.utils.distrib.is_distributed", return_value=True)
        mocker.patch("pyine.utils.distrib.is_main_process", return_value=True)
        fake_collector = _FakeGPUStatsCollector(enabled=True)
        mocker.patch("pyine.utils.gpu.GPUStatsCollector", return_value=fake_collector)
        callback = callbacks_module.GPUStatsLoggingCallback(config=config)
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
    ) -> None:
        """Test that gather_train_metrics='always' uses gathered metrics."""
        config = callbacks_module.GPUStatsLoggingConfig(gather_train_metrics="always")
        mocker.patch("pyine.utils.distrib.is_distributed", return_value=True)
        mocker.patch("pyine.utils.distrib.is_main_process", return_value=True)
        fake_collector = _FakeGPUStatsCollector(enabled=True)
        mocker.patch("pyine.utils.gpu.GPUStatsCollector", return_value=fake_collector)
        callback = callbacks_module.GPUStatsLoggingCallback(config=config)
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
        callback = callbacks_module.GPUStatsLoggingCallback(config=config)
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
        callback = callbacks_module.GPUStatsLoggingCallback(config=config)
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
        assert callback._eval_pending is True
        assert callback._train_phase_needs_flush is True
        assert callback._stashed_train_peak_percent is not None  # peak was captured
        stashed_peak = callback._stashed_train_peak_percent
        # now eval runs
        callback.on_prediction_step(args, state, control)
        assert callback._in_eval is True
        # eval on_log: should emit BOTH train metrics (delayed) and eval metrics
        logs: dict[str, typing.Any] = {"eval_loss": 0.3}
        callback.on_log(args, state, control, logs=logs)
        # verify train metrics were injected with train/gpu/ prefix
        train_keys = [k for k in logs if k.startswith("train/gpu/")]
        assert len(train_keys) > 0, "expected train/gpu/ metrics in delayed flush"
        # verify peak percent used the stashed value
        if "train/gpu/pytorch_peak_percent" in logs:
            assert logs["train/gpu/pytorch_peak_percent"] == stashed_peak
        # verify train state was cleaned up
        assert callback._train_phase_needs_flush is False
        assert callback._stashed_train_peak_percent is None
        assert callback._train.sample_count == 0
        # verify eval metrics were also injected
        eval_keys = [k for k in logs if k.startswith("eval/gpu/")]
        assert len(eval_keys) > 0, "expected eval/gpu/ metrics"
        # verify _in_eval was cleared
        assert callback._in_eval is False
