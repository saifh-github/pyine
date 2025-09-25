import pathlib
import typing

import click
import click.testing
import pytest

import pyine.apps.write.dataset_writer
import pyine.data.traces.dataset_writer
import pyine.utils.filesystem
import tests.env_checks


def test_main_write_traces_dry_run(tmp_path: pathlib.Path) -> None:
    """Dry run of main() using some dummy arguments."""
    cli_runner = click.testing.CliRunner()
    cli_args = [
        "traces",
        "--dataset-name=potato",
        f"--dataset-path={tmp_path}",
        "--output-path=/tmp/potato_out",
        "--fetch-augmented-solutions=issues/todos=2",
        "--fetch-augmented-solutions=issues/iterators=1",
        "--dry-run",
    ]
    res = cli_runner.invoke(pyine.apps.write.dataset_writer.main, cli_args)  # noqa
    assert res.exit_code == 0, res


@pytest.mark.slow
@pytest.mark.skipif(
    tests.env_checks.TACO_DATASET_MISSING,
    reason="TACO dataset is missing, cannot check trace dataset writer app",
)
def test_main_write_taco_traces_micro(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end exercise of main() using the TACO dataset and a max sample count of 5."""
    monkeypatch.setattr(pyine.utils.filesystem, "get_logs_root_path", lambda: tmp_path)
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
    res = cli_runner.invoke(pyine.apps.write.dataset_writer.main, cli_args)  # noqa
    assert res.exit_code == 0, res
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
    res = cli_runner.invoke(pyine.apps.write.dataset_writer.main, cli_args)  # noqa
    assert res.exit_code == 0, res


def test_traces_cli_converts_file_exists_error(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure the traces CLI reports a ClickException when the output already exists."""
    cli_runner = click.testing.CliRunner()
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    output_path = tmp_path / "output"

    def fake_write_dataset(**kwargs: typing.Any) -> None:
        assert kwargs["force_overwrite"] is False
        raise FileExistsError("Refusing to overwrite existing output at: /tmp/out")

    monkeypatch.setattr(pyine.data.traces.dataset_writer, "write_dataset", fake_write_dataset)
    cli_args = [
        "traces",
        "--dataset-name=potato",
        f"--dataset-path={dataset_path}",
        f"--output-path={output_path}",
    ]
    res = cli_runner.invoke(pyine.apps.write.dataset_writer.main, cli_args)
    assert res.exit_code == 1
    assert isinstance(res.exception, SystemExit)
    assert "Error: Refusing to overwrite" in res.output


def test_traces_cli_passes_force_flag(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the traces CLI forwards the force flag to the writer."""
    cli_runner = click.testing.CliRunner()
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    output_path = tmp_path / "output"
    captured: dict[str, typing.Any] = {}

    def fake_write_dataset(**kwargs: typing.Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(pyine.data.traces.dataset_writer, "write_dataset", fake_write_dataset)
    cli_args = [
        "traces",
        "--dataset-name=potato",
        f"--dataset-path={dataset_path}",
        f"--output-path={output_path}",
        "--force",
    ]
    res = cli_runner.invoke(pyine.apps.write.dataset_writer.main, cli_args)
    assert res.exit_code == 0, res
    assert captured["force_overwrite"] is True
