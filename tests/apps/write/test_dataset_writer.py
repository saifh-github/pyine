import pathlib

import click.testing
import pytest
import yaml

import pyine.apps.write.dataset_writer as writer
import pyine.data.utils.splits as split_utils
import tests.data.utils.env_checks as env_checks


def test_main_write_traces_dry_run(tmp_path: pathlib.Path) -> None:
    """
    End-to-end exercise of main() using the TACO dataset and a max sample count of 10.

    Monkeypatches the dataset split file path getter to use a temporary directory.
    """
    cli_runner = click.testing.CliRunner()
    cli_args = [
        "traces",
        "--dataset-name=potato",
        f"--dataset-path={tmp_path}",
        "--output-path=/tmp/potato_out",
        "--fetch-augmented-solutions=hints/docs=2",
        "--fetch-augmented-solutions=hints/stubs=1",
        "--dry-run",
    ]
    res = cli_runner.invoke(writer.main, cli_args)  # noqa
    assert res.exit_code == 0


def test_main_write_deltas_dry_run() -> None:
    """
    End-to-end exercise of main() using the TACO dataset and a max sample count of 10.

    Monkeypatches the dataset split file path getter to use a temporary directory.
    """
    cli_runner = click.testing.CliRunner()
    cli_args = [
        "deltas",
        "--traces-dataset=/tmp/potato_in",
        "--output-path=/tmp/potato_out",
        "--dry-run",
    ]
    res = cli_runner.invoke(writer.main, cli_args)  # noqa
    assert res.exit_code == 0


@pytest.mark.slow
@pytest.mark.skipif(
    env_checks.TACO_TRACES_DATASET_SPLIT_MISSING,
    reason="TACO traces dataset split is missing, cannot check problem partitioning",
)
def test_main_partition_with_taco_split(tmp_path: pathlib.Path) -> None:
    """
    End-to-end exercise of the 'partition' command in dry-run mode.

    Uses a temporary directory for the output dir and a dummy split file path.
    """
    cli_runner = click.testing.CliRunner()
    taco_dataset_split_path = split_utils.get_dataset_split_file_path("TACO")
    cli_args = [
        "partition",
        f"--split-file={taco_dataset_split_path}",
        f"--output-dir={tmp_path}",
        "--ids-per-chunk=5000",
        "--no-only-assigned-ids",
        "--format=yaml",
    ]
    res = cli_runner.invoke(writer.main, cli_args)  # noqa
    assert res.exit_code == 0
    part_files = sorted(list(tmp_path.glob("*.yaml")))
    assert len(part_files) > 0
    expected_ids = split_utils.SplitResult.from_file(taco_dataset_split_path).identifiers
    found_ids = []
    for part_file in part_files:
        with part_file.open("r") as fd:
            part_ids = yaml.safe_load(fd)
            assert len(part_ids) <= 5000
            found_ids.extend(part_ids)
    assert len(found_ids) == len(set(found_ids))
    assert found_ids == expected_ids
