import pathlib

import click.testing
import pytest

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
    """
    End-to-end exercise of main() using the TACO dataset and a max sample count of 10.

    Monkeypatches the dataset split file path getter to use a temporary directory.
    """
    split_file_path = pathlib.Path(tmp_path) / "split.bin"
    assert not split_file_path.exists()
    monkeypatch.setattr(splits_utils, "get_dataset_split_file_path", lambda x: split_file_path)
    cli_runner = click.testing.CliRunner()
    res = cli_runner.invoke(splitter.main, ["--dataset-name=TACO", "--max-sample-count=10"])  # noqa
    assert res.exit_code == 0
    assert split_file_path.exists()
    split_result = splits_utils.get_dataset_split_result("TACO")
    assert split_result.source_dataset_name == "TACO"
    assert len(split_result.subset_assignments) == 10
