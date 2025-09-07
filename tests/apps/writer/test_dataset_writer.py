import pathlib

import click.testing

import pyine.apps.write.dataset_writer as writer


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
