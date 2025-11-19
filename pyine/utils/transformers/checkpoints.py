import json
import logging
import pathlib
import time
import typing

import transformers
import transformers.trainer_utils

import pyine.configs.schemas
import pyine.utils.portability
import pyine.utils.reprod

logger = logging.getLogger(__name__)

__all__ = [
    "write_checkpoint_metadata",
    "get_checkpoint_folder_path",
]


def write_checkpoint_metadata(
    checkpoint_dir: pathlib.Path,
    *,
    config: typing.Any,
    runtime: pyine.configs.schemas.RuntimeConfig | None,
    state: transformers.TrainerState,
    shutdown_manager: typing.Any,
) -> None:
    """Persist run metadata alongside a Hugging Face checkpoint directory."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_payload = pyine.utils.portability.make_json_serializable(config.model_dump(mode="python"))
    metadata: dict[str, typing.Any] = {
        "global_step": state.global_step,
        "epoch": state.epoch,
        "best_metric": state.best_metric,
        "training_loss": getattr(state, "loss", None),
        "git_revision": pyine.utils.reprod.get_git_revision_hash(),
        "config_hash": pyine.utils.reprod.get_params_hash(json.dumps(config_payload, sort_keys=True)),
        "config": config_payload,
        "generated_at": time.time(),
    }
    shutdown_request = getattr(shutdown_manager, "shutdown_request", None) if shutdown_manager is not None else None
    metadata["shutdown_requested"] = shutdown_request is not None
    if shutdown_request is not None:
        metadata["shutdown_reason"] = shutdown_request.reason
        metadata["shutdown_requested_at"] = shutdown_request.requested_at
        metadata["shutdown_deadline_at"] = shutdown_request.deadline_at
    if runtime is not None:
        runtime_payload = runtime.model_dump(mode="json")
        metadata["runtime"] = runtime_payload
    metadata_path = checkpoint_dir / "run_meta.json"
    logger.debug(f"writing checkpoint metadata to {metadata_path}")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))


def get_checkpoint_folder_path(
    args: transformers.TrainingArguments,
    state: transformers.TrainerState,
) -> pathlib.Path:
    """Returns the folder path for a given checkpoint directory and training state."""
    # the checkpoint dir name is hard-coded as follows in the transformers trainer impl:
    ckpt_dir_name = f"{transformers.trainer_utils.PREFIX_CHECKPOINT_DIR}-{state.global_step}"
    return pathlib.Path(args.output_dir or ".").resolve() / ckpt_dir_name
