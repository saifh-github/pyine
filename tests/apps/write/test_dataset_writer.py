import pathlib

import click.testing
import pytest

import pyine.apps.write.dataset_writer as writer
import tests.data.utils.env_checks as env_checks


def test_main_write_traces_dry_run(tmp_path: pathlib.Path) -> None:
    """Dry run of main() using some dummy arguments."""
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


@pytest.mark.slow
@pytest.mark.skipif(
    env_checks.TACO_DATASET_MISSING,
    reason="TACO dataset is missing, cannot check trace dataset writer app",
)
def test_main_write_taco_traces_micro(tmp_path: pathlib.Path) -> None:
    """End-to-end exercise of main() using the TACO dataset and a max sample count of 5."""
    cli_runner = click.testing.CliRunner()
    output_path = tmp_path / "output"
    assert not output_path.exists()
    cli_args = [
        "traces",
        "--dataset-name=TACO",
        f"--output-path={output_path}",
        "--max-output-traces=5",
        "--max-solutions-per-problem=1",
        "--max-tests-per-solution=1",
    ]
    res = cli_runner.invoke(writer.main, cli_args)  # noqa
    assert res.exit_code == 0
    assert output_path.is_dir()


def test_main_write_deltas_dry_run() -> None:
    """Dry run of main() using some dummy arguments."""
    cli_runner = click.testing.CliRunner()
    cli_args = [
        "deltas",
        "--traces-dataset=/tmp/potato_in",
        "--output-path=/tmp/potato_out",
        "--dry-run",
    ]
    res = cli_runner.invoke(writer.main, cli_args)  # noqa
    assert res.exit_code == 0
