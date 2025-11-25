"""Persistence utilities for experiment results."""

import dataclasses
import datetime
import json
import pathlib
import typing


def save_experiment_results(
    results: list[dict[str, typing.Any]],
    experiment_name: str,
    metadata: dict[str, typing.Any],
    output_dir: pathlib.Path,
    run_name: str | None = None,
) -> pathlib.Path:
    """Saves experiment results with metadata."""
    if run_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        start_pos = metadata.get("start_position", 0)
        run_name = f"pos{start_pos:05d}_{timestamp}"

    experiment_dir = output_dir / experiment_name / run_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    results_path = experiment_dir / "results.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)

    metadata_path = experiment_dir / "metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    return experiment_dir


def load_experiment_results(
    experiment_path: pathlib.Path,
) -> tuple[list[dict[str, typing.Any]], dict[str, typing.Any]]:
    """Loads experiment results and metadata."""
    results_path = experiment_path / "results.json"
    with results_path.open("r") as f:
        results = json.load(f)

    metadata_path = experiment_path / "metadata.json"
    with metadata_path.open("r") as f:
        metadata = json.load(f)

    return results, metadata


@dataclasses.dataclass(frozen=True)
class ExperimentIdentifier:
    """Identifier for experiment results."""

    name: str
    timestamp: str
    path: pathlib.Path


def list_experiments(
    output_dir: pathlib.Path,
) -> list[ExperimentIdentifier]:
    """Lists all saved experiments."""
    if not output_dir.exists():
        return []

    experiments: list[ExperimentIdentifier] = []
    for results_json in output_dir.rglob("results.json"):
        timestamp_dir = results_json.parent
        exp_name_dir = timestamp_dir.parent
        experiments.append(
            ExperimentIdentifier(
                name=exp_name_dir.name,
                timestamp=timestamp_dir.name,
                path=timestamp_dir,
            )
        )

    return sorted(experiments, key=lambda x: x.timestamp, reverse=True)


def get_last_checkpoint(
    experiment_name: str,
    seed: int,
    output_dir: pathlib.Path,
) -> int:
    """Returns the last checkpoint position from previous runs of the same experiment series."""
    experiment_dir = output_dir / experiment_name
    if not experiment_dir.exists():
        return 0

    max_position = 0

    for timestamp_dir in experiment_dir.iterdir():
        if not timestamp_dir.is_dir():
            continue

        metadata_path = timestamp_dir / "metadata.json"
        if not metadata_path.exists():
            continue

        with metadata_path.open("r") as f:
            metadata = json.load(f)

        if metadata.get("seed") != seed:
            continue

        if "end_position" in metadata:
            max_position = max(max_position, metadata["end_position"])

    return max_position
