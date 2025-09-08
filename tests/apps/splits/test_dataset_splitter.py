import pathlib

import click.testing
import pytest
import yaml

import pyine.apps.splits.dataset_splitter as splitter
import pyine.data.utils.splits as splits_utils
import tests.data.utils.env_checks as env_checks


@pytest.mark.slow
@pytest.mark.skipif(
    env_checks.TACO_DATASET_MISSING,
    reason="TACO dataset is missing, cannot check dataset split",
)
def test_main_split_taco_micro_subset(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end exercise of main() using the TACO dataset and a max sample count of 10.

    Monkeypatches the dataset split file path getter to use a temporary directory.
    """
    split_file_path = pathlib.Path(tmp_path) / "split.bin"
    assert not split_file_path.exists()
    monkeypatch.setattr(splits_utils, "get_dataset_split_file_path", lambda x: split_file_path)
    cli_runner = click.testing.CliRunner()
    res = cli_runner.invoke(splitter.main, ["split", "--dataset-name=TACO", "--max-sample-count=10"])  # noqa
    assert res.exit_code == 0
    assert split_file_path.exists()
    split_result = splits_utils.get_dataset_split_result("TACO")
    assert split_result.source_dataset_name == "TACO"
    assert len(split_result.subset_assignments) == 10


@pytest.mark.slow
@pytest.mark.skipif(
    env_checks.TACO_TRACES_DATASET_SPLIT_MISSING,
    reason="TACO traces dataset split is missing, cannot check problem partitioning",
)
def test_main_partition_with_taco_split(tmp_path: pathlib.Path) -> None:
    """End-to-end exercise of the 'partition' command in dry-run mode.

    Uses a temporary directory for the output dir.
    """
    cli_runner = click.testing.CliRunner()
    taco_dataset_split_path = splits_utils.get_dataset_split_file_path("TACO")
    cli_args = [
        "partition",
        f"--split-file={taco_dataset_split_path}",
        f"--output-dir={tmp_path}",
        "--ids-per-chunk=5000",
        "--no-only-assigned-ids",
        "--format=yaml",
    ]
    res = cli_runner.invoke(splitter.main, cli_args)  # noqa
    assert res.exit_code == 0
    part_files = sorted(list(tmp_path.glob("*.yaml")))
    assert len(part_files) > 0
    expected_ids = splits_utils.SplitResult.from_file(taco_dataset_split_path).identifiers
    found_ids = []
    for part_file in part_files:
        with part_file.open("r") as fd:
            part_ids = yaml.safe_load(fd)
            assert len(part_ids) <= 5000
            found_ids.extend(part_ids)
    assert len(found_ids) == len(set(found_ids))
    assert found_ids == expected_ids
