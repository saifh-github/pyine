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
        """Test _should_print returns True on main process when only_main_process=True."""
        callback = callbacks_module.StdoutMilestones(only_main_process=True)
        args = mocker.MagicMock(process_index=0)
        assert callback._should_print(args) is True

    def test_should_print_main_process_on_worker(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Test _should_print returns False on worker process when only_main_process=True."""
        callback = callbacks_module.StdoutMilestones(only_main_process=True)
        args = mocker.MagicMock(process_index=1)
        assert callback._should_print(args) is False

    def test_should_print_all_processes(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Test _should_print returns True on all processes when only_main_process=False."""
        callback = callbacks_module.StdoutMilestones(only_main_process=False)
        args = mocker.MagicMock(process_index=1)
        assert callback._should_print(args) is True

    def test_should_print_missing_process_index(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Test _should_print handles missing process_index attribute."""
        callback = callbacks_module.StdoutMilestones(only_main_process=True)
        args = mocker.MagicMock(spec=[])  # No process_index attribute
        assert callback._should_print(args) is True

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
    def callback(self, mock_print_fn: pytest.FixtureRequest) -> callbacks_module.StdoutMilestones:
        """Create a callbacks_module.StdoutMilestones instance with mock print function."""
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
        callback: pytest.FixtureRequest,
        mock_print_fn: pytest.FixtureRequest,
        args: pytest.FixtureRequest,
        state: pytest.FixtureRequest,
        control: pytest.FixtureRequest,
    ) -> None:
        """Test on_init_end prints on main process."""
        callback.on_init_end(args, state, control)
        mock_print_fn.assert_called_once_with("init_end")

    def test_on_init_end_worker_process(
        self,
        callback: pytest.FixtureRequest,
        mock_print_fn: pytest.FixtureRequest,
        state: pytest.FixtureRequest,
        control: pytest.FixtureRequest,
        mocker: pytest.FixtureRequest,
    ) -> None:
        """Test on_init_end doesn't print on worker process."""
        args = mocker.MagicMock(process_index=1)
        callback.on_init_end(args, state, control)
        mock_print_fn.assert_not_called()

    def test_on_train_begin_with_config(
        self,
        callback: pytest.FixtureRequest,
        mock_print_fn: pytest.FixtureRequest,
        args: pytest.FixtureRequest,
        state: pytest.FixtureRequest,
        control: pytest.FixtureRequest,
    ) -> None:
        """Test on_train_begin prints training configuration."""
        callback.on_train_begin(args, state, control)
        assert mock_print_fn.call_count == 7
        mock_print_fn.assert_any_call("train_begin")
        mock_print_fn.assert_any_call("run_name=test_run, output_dir=/something/output")

    def test_on_train_begin_without_config(
        self,
        mock_print_fn: pytest.FixtureRequest,
        args: pytest.FixtureRequest,
        state: pytest.FixtureRequest,
        control: pytest.FixtureRequest,
    ) -> None:
        """Test on_train_begin without printing config."""
        callback = callbacks_module.StdoutMilestones(print_fn=mock_print_fn, print_config_at_start=False)
        callback.on_train_begin(args, state, control)
        mock_print_fn.assert_called_once_with("train_begin")

    def test_on_train_begin_worker_process(
        self,
        callback: pytest.FixtureRequest,
        mock_print_fn: pytest.FixtureRequest,
        state: pytest.FixtureRequest,
        control: pytest.FixtureRequest,
        mocker: pytest.FixtureRequest,
    ) -> None:
        """Test on_train_begin doesn't print on worker process."""
        args = mocker.MagicMock(process_index=1)
        callback.on_train_begin(args, state, control)
        mock_print_fn.assert_not_called()

    def test_on_epoch_begin(
        self,
        callback: pytest.FixtureRequest,
        mock_print_fn: pytest.FixtureRequest,
        args: pytest.FixtureRequest,
        state: pytest.FixtureRequest,
        control: pytest.FixtureRequest,
    ) -> None:
        """Test on_epoch_begin prints epoch information."""
        callback.on_epoch_begin(args, state, control)
        mock_print_fn.assert_called_once_with("epoch_begin; epoch=1, global_step=100")

    def test_on_epoch_begin_none_epoch(
        self,
        callback: pytest.FixtureRequest,
        mock_print_fn: pytest.FixtureRequest,
        args: pytest.FixtureRequest,
        control: pytest.FixtureRequest,
        mocker: pytest.FixtureRequest,
    ) -> None:
        """Test on_epoch_begin handles None epoch."""
        state = mocker.MagicMock(epoch=None, global_step=100)
        callback.on_epoch_begin(args, state, control)
        mock_print_fn.assert_called_once_with("epoch_begin; epoch=NA, global_step=100")

    def test_on_epoch_end(
        self,
        callback: pytest.FixtureRequest,
        mock_print_fn: pytest.FixtureRequest,
        args: pytest.FixtureRequest,
        state: pytest.FixtureRequest,
        control: pytest.FixtureRequest,
    ) -> None:
        """Test on_epoch_end prints epoch information."""
        callback.on_epoch_end(args, state, control)
        mock_print_fn.assert_called_once_with("epoch_end; epoch=1, global_step=100")

    def test_on_log_with_logs(
        self,
        callback: pytest.FixtureRequest,
        mock_print_fn: pytest.FixtureRequest,
        args: pytest.FixtureRequest,
        state: pytest.FixtureRequest,
        control: pytest.FixtureRequest,
    ) -> None:
        """Test on_log prints log information."""
        logs = {"loss": 0.5, "learning_rate": 1e-5}
        callback.on_log(args, state, control, logs=logs)
        mock_print_fn.assert_called_once_with("log; step=100, learning_rate=1e-05, loss=0.5")

    def test_on_log_empty_logs(
        self,
        callback: pytest.FixtureRequest,
        mock_print_fn: pytest.FixtureRequest,
        args: pytest.FixtureRequest,
        state: pytest.FixtureRequest,
        control: pytest.FixtureRequest,
    ) -> None:
        """Test on_log doesn't print when logs are empty."""
        callback.on_log(args, state, control, logs={})
        mock_print_fn.assert_not_called()

    def test_on_log_worker_process(
        self,
        callback: pytest.FixtureRequest,
        mock_print_fn: pytest.FixtureRequest,
        state: pytest.FixtureRequest,
        control: pytest.FixtureRequest,
        mocker: pytest.FixtureRequest,
    ) -> None:
        """Test on_log doesn't print on worker process."""
        args = mocker.MagicMock(process_index=1)
        logs = {"loss": 0.5}
        callback.on_log(args, state, control, logs=logs)
        mock_print_fn.assert_not_called()

    def test_on_evaluate(
        self,
        callback: pytest.FixtureRequest,
        mock_print_fn: pytest.FixtureRequest,
        args: pytest.FixtureRequest,
        state: pytest.FixtureRequest,
        control: pytest.FixtureRequest,
    ) -> None:
        """Test on_evaluate prints metrics."""
        metrics = {"eval_loss": 0.3, "eval_accuracy": 0.95}
        callback.on_evaluate(args, state, control, metrics=metrics)
        mock_print_fn.assert_called_once_with("evaluate; step=100, epoch=1, eval_accuracy=0.95, eval_loss=0.3")

    def test_on_evaluate_empty_metrics(
        self,
        callback: pytest.FixtureRequest,
        mock_print_fn: pytest.FixtureRequest,
        args: pytest.FixtureRequest,
        state: pytest.FixtureRequest,
        control: pytest.FixtureRequest,
    ) -> None:
        """Test on_evaluate doesn't print when metrics are empty."""
        callback.on_evaluate(args, state, control, metrics={})
        mock_print_fn.assert_not_called()

    def test_on_predict(
        self,
        callback: pytest.FixtureRequest,
        mock_print_fn: pytest.FixtureRequest,
        args: pytest.FixtureRequest,
        state: pytest.FixtureRequest,
        control: pytest.FixtureRequest,
    ) -> None:
        """Test on_predict prints metrics."""
        metrics = {"predict_loss": 0.25, "predict_accuracy": 0.92}
        callback.on_predict(args, state, control, metrics=metrics)
        mock_print_fn.assert_called_once_with("predict; step=100, epoch=1, predict_accuracy=0.92, predict_loss=0.25")

    def test_on_predict_empty_metrics(
        self,
        callback: pytest.FixtureRequest,
        mock_print_fn: pytest.FixtureRequest,
        args: pytest.FixtureRequest,
        state: pytest.FixtureRequest,
        control: pytest.FixtureRequest,
    ) -> None:
        """Test on_predict doesn't print when metrics are empty."""
        callback.on_predict(args, state, control, metrics={})
        mock_print_fn.assert_not_called()

    def test_on_save(
        self,
        callback: pytest.FixtureRequest,
        mock_print_fn: pytest.FixtureRequest,
        args: pytest.FixtureRequest,
        state: pytest.FixtureRequest,
        control: pytest.FixtureRequest,
        mocker: pytest.FixtureRequest,
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
        callback: pytest.FixtureRequest,
        mock_print_fn: pytest.FixtureRequest,
        state: pytest.FixtureRequest,
        control: pytest.FixtureRequest,
        mocker: pytest.FixtureRequest,
    ) -> None:
        """Test on_save doesn't print on worker process."""
        args = mocker.MagicMock(process_index=1)
        callback.on_save(args, state, control)
        mock_print_fn.assert_not_called()

    def test_on_train_end(
        self,
        callback: pytest.FixtureRequest,
        mock_print_fn: pytest.FixtureRequest,
        args: pytest.FixtureRequest,
        state: pytest.FixtureRequest,
        control: pytest.FixtureRequest,
    ) -> None:
        """Test on_train_end prints final training information."""
        callback.on_train_end(args, state, control)
        mock_print_fn.assert_called_once_with(
            "train_end; steps=100, best_metric=0.95, best_model_checkpoint=/something/checkpoint-100"
        )

    def test_on_train_end_worker_process(
        self,
        callback: pytest.FixtureRequest,
        mock_print_fn: pytest.FixtureRequest,
        state: pytest.FixtureRequest,
        control: pytest.FixtureRequest,
        mocker: pytest.FixtureRequest,
    ) -> None:
        """Test on_train_end doesn't print on worker process."""
        args = mocker.MagicMock(process_index=1)
        callback.on_train_end(args, state, control)
        mock_print_fn.assert_not_called()

    def test_callback_inheritance(self) -> None:
        """Test that callbacks_module.StdoutMilestones properly inherits from TrainerCallback."""
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
        self.flush_calls: list[int | None] = []

    def set_key_prefix(self, prefix: str) -> None:
        self.key_prefix = prefix

    def set_step(self, step: int | None) -> None:
        self.step = step

    def flush_stats(self, step: int | None = None) -> None:
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
