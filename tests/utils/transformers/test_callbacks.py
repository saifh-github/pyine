import pathlib

import pytest
import pytest_mock
import transformers

from pyine.utils.transformers.callbacks import StdoutMilestones


class TestStdoutMilestones:
    def test_should_print_main_process_on_main(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Test _should_print returns True on main process when only_main_process=True."""
        callback = StdoutMilestones(only_main_process=True)
        args = mocker.MagicMock(process_index=0)
        assert callback._should_print(args) is True

    def test_should_print_main_process_on_worker(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Test _should_print returns False on worker process when only_main_process=True."""
        callback = StdoutMilestones(only_main_process=True)
        args = mocker.MagicMock(process_index=1)
        assert callback._should_print(args) is False

    def test_should_print_all_processes(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Test _should_print returns True on all processes when only_main_process=False."""
        callback = StdoutMilestones(only_main_process=False)
        args = mocker.MagicMock(process_index=1)
        assert callback._should_print(args) is True

    def test_should_print_missing_process_index(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Test _should_print handles missing process_index attribute."""
        callback = StdoutMilestones(only_main_process=True)
        args = mocker.MagicMock(spec=[])  # No process_index attribute
        assert callback._should_print(args) is True

    def test_fmt_epoch_with_float(self) -> None:
        """Test _fmt_epoch formats float epoch correctly."""
        assert StdoutMilestones._fmt_epoch(1.5) == "1.5"
        assert StdoutMilestones._fmt_epoch(2.0) == "2"
        assert StdoutMilestones._fmt_epoch(0.123456789) == "0.123457"

    def test_fmt_epoch_with_none(self) -> None:
        """Test _fmt_epoch returns NA for None."""
        assert StdoutMilestones._fmt_epoch(None) == "NA"

    def test_fmt_dict_with_floats(self) -> None:
        """Test _fmt_dict formats dictionary with float values."""
        result = StdoutMilestones._fmt_dict({"loss": 0.123456789, "lr": 1e-5})
        assert result == "loss=0.123457, lr=1e-05"

    def test_fmt_dict_with_mixed_types(self) -> None:
        """Test _fmt_dict formats dictionary with mixed value types."""
        result = StdoutMilestones._fmt_dict({"loss": 0.5, "epoch": 1, "name": "test"})
        assert result == "epoch=1, loss=0.5, name=test"

    def test_fmt_dict_empty(self) -> None:
        """Test _fmt_dict handles empty dictionary."""
        result = StdoutMilestones._fmt_dict({})
        assert result == ""

    def test_fmt_dict_sorts_keys(self) -> None:
        """Test _fmt_dict sorts keys alphabetically."""
        result = StdoutMilestones._fmt_dict({"z": 1, "a": 2, "m": 3})
        assert result == "a=2, m=3, z=1"

    @pytest.fixture
    def mock_print_fn(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> pytest_mock.MockFixture:
        """Create a mock print function."""
        return mocker.MagicMock()

    @pytest.fixture
    def callback(self, mock_print_fn: pytest.FixtureRequest) -> StdoutMilestones:
        """Create a StdoutMilestones instance with mock print function."""
        return StdoutMilestones(print_fn=mock_print_fn)

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
        callback = StdoutMilestones(print_fn=mock_print_fn, print_config_at_start=False)
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
        """Test that StdoutMilestones properly inherits from TrainerCallback."""
        callback = StdoutMilestones()
        assert isinstance(callback, transformers.TrainerCallback)

    def test_all_callbacks_defined(self) -> None:
        """Test that all expected callback methods are defined."""
        callback = StdoutMilestones()
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
