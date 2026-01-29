import pytest
import pytest_mock

import pyine.utils.transformers.training as training_module


class TestEvalPhaseTracker:
    @pytest.fixture
    def tracker(self) -> training_module.EvalPhaseTracker:
        return training_module.EvalPhaseTracker()

    @pytest.fixture
    def control(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> pytest_mock.MockFixture:
        return mocker.MagicMock(should_evaluate=False)

    def test_initial_state(
        self,
        tracker: training_module.EvalPhaseTracker,
    ) -> None:
        assert tracker.in_eval is False
        assert tracker.eval_pending is False
        assert tracker.saw_eval_samples is False
        assert tracker.saw_training is False

    def test_handle_step_end_sets_pending_and_saw_training(
        self,
        tracker: training_module.EvalPhaseTracker,
        control: pytest_mock.MockFixture,
    ) -> None:
        control.should_evaluate = True
        result = tracker.handle_step_or_epoch_end(control)
        assert result is True
        assert tracker.eval_pending is True
        assert tracker.saw_training is True
        assert tracker.saw_eval_samples is False

    def test_handle_step_end_no_eval(
        self,
        tracker: training_module.EvalPhaseTracker,
        control: pytest_mock.MockFixture,
    ) -> None:
        control.should_evaluate = False
        result = tracker.handle_step_or_epoch_end(control)
        assert result is False
        assert tracker.eval_pending is False
        assert tracker.saw_training is True

    def test_handle_prediction_step_returns_unexpected_only_when_training_seen(
        self,
        tracker: training_module.EvalPhaseTracker,
    ) -> None:
        # before any training: unexpected should be False
        first_step, unexpected = tracker.handle_prediction_step()
        assert first_step is True
        assert unexpected is False  # no training seen yet
        assert tracker.in_eval is True
        assert tracker.saw_eval_samples is True

    def test_handle_prediction_step_unexpected_when_training_seen(
        self,
        tracker: training_module.EvalPhaseTracker,
        control: pytest_mock.MockFixture,
    ) -> None:
        control.should_evaluate = False
        tracker.handle_step_or_epoch_end(control)  # sets saw_training=True
        assert tracker.saw_training is True
        tracker.handle_evaluate()  # reset eval state
        first_step, unexpected = tracker.handle_prediction_step()
        assert first_step is True
        assert unexpected is True  # training was seen but no eval_pending

    def test_handle_prediction_step_expected_after_step_end(
        self,
        tracker: training_module.EvalPhaseTracker,
        control: pytest_mock.MockFixture,
    ) -> None:
        control.should_evaluate = True
        tracker.handle_step_or_epoch_end(control)  # sets eval_pending=True
        first_step, unexpected = tracker.handle_prediction_step()
        assert first_step is True
        assert unexpected is False  # expected because eval_pending was True
        assert tracker.in_eval is True

    def test_handle_prediction_step_subsequent_not_first(
        self,
        tracker: training_module.EvalPhaseTracker,
        control: pytest_mock.MockFixture,
    ) -> None:
        control.should_evaluate = True
        tracker.handle_step_or_epoch_end(control)
        tracker.handle_prediction_step()  # first step
        first_step, unexpected = tracker.handle_prediction_step()  # second step
        assert first_step is False
        assert unexpected is False

    def test_is_eval_context_with_log_keys(
        self,
        tracker: training_module.EvalPhaseTracker,
        control: pytest_mock.MockFixture,
    ) -> None:
        assert tracker.is_eval_context(None) is False
        assert tracker.is_eval_context({"loss": 0.5}) is False
        # when in_eval is True
        tracker.handle_prediction_step()
        assert tracker.is_eval_context(None) is True
        assert tracker.is_eval_context({"loss": 0.5}) is True

    def test_is_eval_context_with_eval_pending_and_eval_logs(
        self,
        tracker: training_module.EvalPhaseTracker,
        control: pytest_mock.MockFixture,
    ) -> None:
        control.should_evaluate = True
        tracker.handle_step_or_epoch_end(control)  # sets eval_pending=True
        assert tracker.is_eval_context({"loss": 0.5}) is False  # not eval-like logs
        assert tracker.is_eval_context({"eval_loss": 0.3}) is True  # eval-like logs
        assert tracker.is_eval_context({"eval/loss": 0.3}) is True  # eval-like logs

    def test_handle_evaluate_resets_state(
        self,
        tracker: training_module.EvalPhaseTracker,
        control: pytest_mock.MockFixture,
    ) -> None:
        control.should_evaluate = True
        tracker.handle_step_or_epoch_end(control)
        tracker.handle_prediction_step()
        assert tracker.in_eval is True
        assert tracker.saw_eval_samples is True
        tracker.handle_evaluate()
        assert tracker.in_eval is False
        assert tracker.eval_pending is False
        assert tracker.saw_eval_samples is False
        assert tracker.saw_training is True  # this persists

    def test_mark_entering_eval_suppresses_unexpected(
        self,
        tracker: training_module.EvalPhaseTracker,
        control: pytest_mock.MockFixture,
    ) -> None:
        control.should_evaluate = False
        tracker.handle_step_or_epoch_end(control)  # sets saw_training=True
        unexpected = tracker.mark_entering_eval()
        assert unexpected is True  # unexpected because no eval_pending
        assert tracker.in_eval is True
        # now prediction step should NOT be unexpected
        first_step, unexpected2 = tracker.handle_prediction_step()
        assert first_step is False  # already in_eval
        assert unexpected2 is False

    def test_mark_entering_eval_when_expected(
        self,
        tracker: training_module.EvalPhaseTracker,
        control: pytest_mock.MockFixture,
    ) -> None:
        control.should_evaluate = True
        tracker.handle_step_or_epoch_end(control)  # sets eval_pending=True
        unexpected = tracker.mark_entering_eval()
        assert unexpected is False  # expected because eval_pending was True
        assert tracker.in_eval is True
        assert tracker.eval_pending is False  # cleared by mark_entering_eval

    def test_eval_on_start_no_unexpected_warning(
        self,
        tracker: training_module.EvalPhaseTracker,
    ) -> None:
        """Simulates eval_on_start flow where no training step has occurred."""
        # no on_step_end called, directly into eval
        first_step, unexpected = tracker.handle_prediction_step()
        assert first_step is True
        assert unexpected is False  # no training seen, so no warning
        assert tracker.in_eval is True

    def test_predict_only_no_unexpected_warning(
        self,
        tracker: training_module.EvalPhaseTracker,
    ) -> None:
        """Verifies no warning when training never started (predict-only)."""
        # directly call prediction step without any training
        first_step, unexpected = tracker.handle_prediction_step()
        assert first_step is True
        assert unexpected is False  # no saw_training, so no unexpected warning


class TestBatchSizeHelpers:
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

    def test_get_effective_train_batch_size(
        self,
        args: pytest_mock.MockFixture,
    ) -> None:
        # 4 * 2 * 2 = 16
        result = training_module.get_effective_train_batch_size(args)
        assert result == 16

    def test_get_effective_eval_batch_size(
        self,
        args: pytest_mock.MockFixture,
    ) -> None:
        # 8 * 2 = 16
        result = training_module.get_effective_eval_batch_size(args)
        assert result == 16

    def test_excludes_world_size_when_disabled(
        self,
        args: pytest_mock.MockFixture,
    ) -> None:
        # train: 4 * 1 * 2 = 8 (world_size excluded)
        result_train = training_module.get_effective_train_batch_size(args, include_world_size=False)
        assert result_train == 8
        # eval: 8 * 1 = 8 (world_size excluded)
        result_eval = training_module.get_effective_eval_batch_size(args, include_world_size=False)
        assert result_eval == 8

    def test_missing_attrs_default_to_one(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        args = mocker.MagicMock(spec=[])  # no attributes
        result_train = training_module.get_effective_train_batch_size(args)
        assert result_train == 1  # 1 * 1 * 1
        result_eval = training_module.get_effective_eval_batch_size(args)
        assert result_eval == 1  # 1 * 1
