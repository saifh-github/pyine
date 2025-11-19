import json
import pathlib
import types
import typing

import pytest
import transformers

import pyine.utils.reprod
import pyine.utils.transformers.checkpoints


class _DummyConfig:
    def __init__(
        self,
        payload: dict[str, typing.Any],
    ) -> None:
        self._payload = payload

    def model_dump(
        self,
        mode: str,
    ) -> dict[str, typing.Any]:
        assert mode == "python"
        return self._payload


class _DummyRuntime:
    def __init__(
        self,
        payload: dict[str, typing.Any],
    ) -> None:
        self._payload = payload

    def model_dump(
        self,
        mode: str,
    ) -> dict[str, typing.Any]:
        assert mode == "json"
        return self._payload


class TestWriteCheckpointMetadata:
    def test_basic_metadata_file(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checkpoint_dir = tmp_path / "checkpoint"
        state = transformers.TrainerState()
        state.global_step = 8
        state.epoch = 1.0
        state.best_metric = 0.99
        state.loss = 0.5
        config = _DummyConfig({"lr": 1e-4})
        runtime = _DummyRuntime({"cluster": "development"})
        shutdown_request = types.SimpleNamespace(
            reason="maintenance",
            requested_at=11.0,
            deadline_at=22.0,
        )
        shutdown_manager = types.SimpleNamespace(shutdown_request=shutdown_request)
        monkeypatch.setattr(pyine.utils.reprod, "get_git_revision_hash", lambda: "git-test")
        monkeypatch.setattr(pyine.utils.reprod, "get_params_hash", lambda payload: f"hash-{payload!s}")
        monkeypatch.setattr(
            pyine.utils.transformers.checkpoints.time,
            "time",
            lambda: 1234.5,
        )
        pyine.utils.transformers.checkpoints.write_checkpoint_metadata(
            checkpoint_dir,
            config=config,
            runtime=runtime,
            state=state,
            shutdown_manager=shutdown_manager,
        )
        metadata_path = checkpoint_dir / "run_meta.json"
        assert metadata_path.exists()
        metadata = json.loads(metadata_path.read_text())
        assert metadata["global_step"] == 8
        assert metadata["config"] == {"lr": 1e-4}
        assert metadata["runtime"] == {"cluster": "development"}
        assert metadata["shutdown_requested"] is True
        assert metadata["shutdown_reason"] == "maintenance"
        assert metadata["git_revision"] == "git-test"
        assert metadata["config_hash"].startswith("hash-")
        assert metadata["generated_at"] == 1234.5

    def test_expected_checkpoint_path(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        args = transformers.TrainingArguments(output_dir=str(tmp_path))
        state = transformers.TrainerState()
        state.global_step = 12
        ckpt_path = pyine.utils.transformers.checkpoints.get_checkpoint_folder_path(args, state)
        expected = tmp_path.resolve() / f"{transformers.trainer_utils.PREFIX_CHECKPOINT_DIR}-12"
        assert ckpt_path == expected
