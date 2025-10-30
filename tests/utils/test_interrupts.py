import json
import pathlib

import transformers

import pyine.configs.schemas
import pyine.utils.interrupts
import pyine.utils.transformers


class _ConfigStub:
    """Lightweight config stub exposing the interface needed for metadata persistence."""

    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def model_dump(
        self,
        mode: str = "python",
    ) -> dict[str, object]:
        assert mode == "python"
        return self._payload


def test_graceful_shutdown_manager_payload_roundtrip() -> None:
    manager = pyine.utils.interrupts.GracefulShutdownManager()
    request = manager.request_shutdown(reason="test signal", deadline_seconds=5.0)
    assert manager.should_terminate()
    payload = manager.as_serializable_payload()
    clone = pyine.utils.interrupts.GracefulShutdownManager()
    clone.load_from_payload(payload)
    assert clone.should_terminate()
    clone_request = clone.shutdown_request
    assert clone_request is not None
    assert clone_request.reason == request.reason
    assert clone_request.deadline_at == request.deadline_at


def test_write_checkpoint_metadata_records_shutdown(tmp_path: pathlib.Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint-0001"
    runtime = pyine.configs.schemas.RuntimeConfig(
        exp_name="unit-test",
        output_dir=str(tmp_path),
        run_name="run",
    )
    config = _ConfigStub({"training_args_config": {"output_dir": str(tmp_path)}})
    manager = pyine.utils.interrupts.GracefulShutdownManager()
    manager.request_shutdown(reason="unit test", deadline_seconds=None)
    state = transformers.TrainerState()
    state.global_step = 1
    state.epoch = 0.0
    state.best_metric = None
    state.loss = 0.5
    pyine.utils.transformers.write_checkpoint_metadata(
        checkpoint_dir,
        config=config,
        runtime=runtime,
        state=state,
        shutdown_manager=manager,
    )
    metadata_path = checkpoint_dir / "run_meta.json"
    assert metadata_path.is_file()
    payload = json.loads(metadata_path.read_text())
    assert payload["shutdown_requested"] is True
    assert payload["global_step"] == 1
    assert payload["runtime"]["wandb_run_id"] is None
