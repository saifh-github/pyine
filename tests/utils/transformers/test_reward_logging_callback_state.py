import json
import pathlib
import types

import pytest
import pytest_mock
import transformers

import pyine.utils.reprod
import pyine.utils.transformers.callbacks


class TestRewardLoggingCallbackState:
    def test_saves_and_loads_state(
        self,
        tmp_path: pathlib.Path,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        mock_reward_manager = mocker.MagicMock()
        mock_reward_manager.set_key_prefix = mocker.MagicMock()
        mock_reward_manager.flush_stats = mocker.MagicMock()
        mock_reward_manager.get_state = mocker.MagicMock(
            return_value={
                "total_stats": {"count": 10, "sum": 5.0, "sumsq": 2.5, "min": 0.1, "max": 0.9},
                "term_stats": {"accuracy": {"count": 10, "sum": 8.0, "sumsq": 6.4, "min": 0.5, "max": 1.0}},
                "category_stats": {},
                "step": 5,
            }
        )
        mock_reward_manager.load_state = mocker.MagicMock()
        mock_reward_adapter = mocker.MagicMock()
        mock_reward_adapter.get_state = mocker.MagicMock(
            return_value={
                "total_count": 100,
                "skip_count": 5,
                "error_count": 2,
            }
        )
        mock_reward_adapter.load_state = mocker.MagicMock()
        callback = pyine.utils.transformers.callbacks.RewardLoggingCallback(
            reward_manager=mock_reward_manager,
            reward_adapter=mock_reward_adapter,
        )
        checkpoint_dir = tmp_path / "checkpoint-5"
        checkpoint_dir.mkdir(parents=True)
        args = transformers.TrainingArguments(output_dir=str(tmp_path))
        state = types.SimpleNamespace(global_step=5, is_world_process_zero=True)
        control = types.SimpleNamespace()
        callback.on_save(
            args=args,
            state=state,
            control=control,
            checkpoint_folder=str(checkpoint_dir),
        )
        reward_state_path = checkpoint_dir / "reward_state.json"
        assert reward_state_path.exists()
        saved_state = json.loads(reward_state_path.read_text())
        assert saved_state["version"] == pyine.utils.reprod.get_framework_version()
        assert saved_state["adapter"]["skip_count"] == 5
        assert saved_state["manager"]["step"] == 5
        mock_reward_manager.load_state.reset_mock()
        mock_reward_adapter.load_state.reset_mock()
        # create a new callback with resume_from_checkpoint to simulate resuming
        callback_for_resume = pyine.utils.transformers.callbacks.RewardLoggingCallback(
            reward_manager=mock_reward_manager,
            reward_adapter=mock_reward_adapter,
            resume_from_checkpoint=checkpoint_dir,
        )
        callback_for_resume.on_train_begin(
            args=args,
            state=state,
            control=control,
        )
        mock_reward_adapter.load_state.assert_called_once()
        mock_reward_manager.load_state.assert_called_once()

    def test_falls_back_to_computed_checkpoint_dir(
        self,
        tmp_path: pathlib.Path,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        mock_reward_manager = mocker.MagicMock()
        mock_reward_manager.set_key_prefix = mocker.MagicMock()
        mock_reward_manager.flush_stats = mocker.MagicMock()
        mock_reward_manager.get_state = mocker.MagicMock(return_value={})
        callback = pyine.utils.transformers.callbacks.RewardLoggingCallback(
            reward_manager=mock_reward_manager,
        )
        args = transformers.TrainingArguments(output_dir=str(tmp_path))
        state = types.SimpleNamespace(global_step=5, is_world_process_zero=True)
        control = types.SimpleNamespace()
        checkpoint_dir = (
            pathlib.Path(args.output_dir).resolve()
            / f"{transformers.trainer_utils.PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )
        checkpoint_dir.mkdir(parents=True)
        callback.on_save(
            args=args,
            state=state,
            control=control,
        )
        assert (checkpoint_dir / "reward_state.json").exists()

    def test_skips_save_on_non_main_process(
        self,
        tmp_path: pathlib.Path,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        mock_reward_manager = mocker.MagicMock()
        mock_reward_manager.set_key_prefix = mocker.MagicMock()
        mock_reward_manager.flush_stats = mocker.MagicMock()
        mock_reward_manager.get_state = mocker.MagicMock(return_value={})
        callback = pyine.utils.transformers.callbacks.RewardLoggingCallback(
            reward_manager=mock_reward_manager,
        )
        checkpoint_dir = tmp_path / "checkpoint-5"
        checkpoint_dir.mkdir(parents=True)
        args = transformers.TrainingArguments(output_dir=str(tmp_path))
        state = types.SimpleNamespace(global_step=5, is_world_process_zero=False)
        control = types.SimpleNamespace()
        callback.on_save(
            args=args,
            state=state,
            control=control,
            checkpoint_folder=str(checkpoint_dir),
        )
        assert not (checkpoint_dir / "reward_state.json").exists()

    def test_raises_on_missing_checkpoint_dir(
        self,
        tmp_path: pathlib.Path,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        mock_reward_manager = mocker.MagicMock()
        mock_reward_manager.set_key_prefix = mocker.MagicMock()
        mock_reward_manager.flush_stats = mocker.MagicMock()
        mock_reward_manager.get_state = mocker.MagicMock(return_value={})
        callback = pyine.utils.transformers.callbacks.RewardLoggingCallback(
            reward_manager=mock_reward_manager,
        )
        checkpoint_dir = tmp_path / "nonexistent-checkpoint"
        args = transformers.TrainingArguments(output_dir=str(tmp_path))
        state = types.SimpleNamespace(global_step=5, is_world_process_zero=True)
        control = types.SimpleNamespace()
        with pytest.raises(FileNotFoundError, match="checkpoint directory does not exist"):
            callback.on_save(
                args=args,
                state=state,
                control=control,
                checkpoint_folder=str(checkpoint_dir),
            )

    def test_raises_on_invalid_reward_state(
        self,
        tmp_path: pathlib.Path,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        mock_reward_manager = mocker.MagicMock()
        mock_reward_manager.set_key_prefix = mocker.MagicMock()
        mock_reward_manager.flush_stats = mocker.MagicMock()
        mock_reward_manager.get_state = mocker.MagicMock(return_value={})
        checkpoint_dir = tmp_path / "checkpoint-5"
        checkpoint_dir.mkdir(parents=True)
        args = transformers.TrainingArguments(output_dir=str(tmp_path))
        state = types.SimpleNamespace(global_step=5, is_world_process_zero=True)
        control = types.SimpleNamespace()
        # test invalid JSON
        (checkpoint_dir / "reward_state.json").write_text("not valid json")
        callback_invalid_json = pyine.utils.transformers.callbacks.RewardLoggingCallback(
            reward_manager=mock_reward_manager,
            resume_from_checkpoint=checkpoint_dir,
        )
        with pytest.raises(json.JSONDecodeError):
            callback_invalid_json.on_train_begin(args=args, state=state, control=control)
        # test missing 'adapter' key
        (checkpoint_dir / "reward_state.json").write_text(
            json.dumps({"version": pyine.utils.reprod.get_framework_version(), "manager": {}})
        )
        callback_missing_adapter = pyine.utils.transformers.callbacks.RewardLoggingCallback(
            reward_manager=mock_reward_manager,
            resume_from_checkpoint=checkpoint_dir,
        )
        with pytest.raises(KeyError, match="adapter"):
            callback_missing_adapter.on_train_begin(args=args, state=state, control=control)
        # test missing 'manager' key
        (checkpoint_dir / "reward_state.json").write_text(
            json.dumps({"version": pyine.utils.reprod.get_framework_version(), "adapter": {}})
        )
        callback_missing_manager = pyine.utils.transformers.callbacks.RewardLoggingCallback(
            reward_manager=mock_reward_manager,
            resume_from_checkpoint=checkpoint_dir,
        )
        with pytest.raises(KeyError, match="manager"):
            callback_missing_manager.on_train_begin(args=args, state=state, control=control)
        # test missing reward_state.json file
        (checkpoint_dir / "reward_state.json").unlink()
        callback_missing_file = pyine.utils.transformers.callbacks.RewardLoggingCallback(
            reward_manager=mock_reward_manager,
            resume_from_checkpoint=checkpoint_dir,
        )
        with pytest.raises(FileNotFoundError, match="reward_state.json not found"):
            callback_missing_file.on_train_begin(args=args, state=state, control=control)
