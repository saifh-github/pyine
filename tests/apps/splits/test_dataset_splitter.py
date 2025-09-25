import pathlib

import click.testing
import pytest
import yaml

import pyine.apps.splits.dataset_splitter
import pyine.apps.write.dataset_writer
import pyine.data.traces.dataset_reader
import pyine.data.utils.splits
import tests.env_checks


@pytest.mark.slow
@pytest.mark.skipif(
    tests.env_checks.TACO_DATASET_MISSING,
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
    monkeypatch.setattr(
        pyine.data.utils.splits,
        "get_dataset_split_file_path",
        lambda x, must_exist: split_file_path,
    )
    cli_runner = click.testing.CliRunner()
    res = cli_runner.invoke(
        pyine.apps.splits.dataset_splitter.main,
        ["split", "--dataset-name=TACO", "--max-sample-count=10"],
    )  # noqa
    assert res.exit_code == 0, res
    assert split_file_path.exists()
    split_result = pyine.data.utils.splits.get_dataset_split_result("TACO")
    assert split_result.source_dataset_name == "TACO"
    assert len(split_result.subset_assignments) == 10


@pytest.mark.slow
@pytest.mark.skipif(
    tests.env_checks.TACO_TRACES_DATASET_SPLIT_MISSING,
    reason="TACO traces dataset split is missing, cannot check problem partitioning",
)
def test_main_partition_with_taco_split(tmp_path: pathlib.Path) -> None:
    """End-to-end exercise of the 'partition' command in dry-run mode.

    Uses a temporary directory for the output dir.
    """
    cli_runner = click.testing.CliRunner()
    taco_dataset_split_path = pyine.data.utils.splits.get_dataset_split_file_path("TACO", must_exist=False)
    cli_args = [
        "partition",
        f"--split-file={taco_dataset_split_path}",
        f"--output-dir={tmp_path}",
        "--ids-per-chunk=5000",
        "--no-only-assigned-ids",
        "--format=yaml",
    ]
    res = cli_runner.invoke(pyine.apps.splits.dataset_splitter.main, cli_args)  # noqa
    assert res.exit_code == 0, res
    part_files = sorted(list(tmp_path.glob("*.yaml")))
    assert len(part_files) > 0
    expected_ids = pyine.data.utils.splits.SplitResult.from_file(taco_dataset_split_path).identifiers
    found_ids = []
    for part_file in part_files:
        with part_file.open("r") as fd:
            part_ids = yaml.safe_load(fd)
            assert len(part_ids) <= 5000
            found_ids.extend(part_ids)
    assert len(found_ids) == len(set(found_ids))
    assert found_ids == expected_ids


@pytest.mark.slow
@pytest.mark.skipif(
    tests.env_checks.TACO_DATASET_MISSING,
    reason="TACO dataset is missing, cannot check dataset split",
)
@pytest.mark.skipif(
    tests.env_checks.TACO_TRACES_DATASET_SPLIT_MISSING,
    reason="TACO traces dataset split is missing, cannot check problem partitioning",
)
def test_main_partition_integration_with_writer(tmp_path: pathlib.Path) -> None:
    """End-to-end exercise of the 'partition' command followed by the trace writer.

    Checks that the partitioned problem IDs are used to generate the correct datasets.

    Uses a temporary directory for the output dir.
    """
    cli_runner = click.testing.CliRunner()
    taco_dataset_split_path = pyine.data.utils.splits.get_dataset_split_file_path("TACO", must_exist=False)
    out_parts_path = tmp_path / "parts"
    cli_args = [
        "partition",
        f"--split-file={taco_dataset_split_path}",
        f"--output-dir={out_parts_path}",
        "--ids-per-chunk=2000",
        "--no-only-assigned-ids",
        "--format=yaml",
    ]
    res = cli_runner.invoke(pyine.apps.splits.dataset_splitter.main, cli_args)  # noqa
    assert res.exit_code == 0, res
    part_files = sorted(list(out_parts_path.glob("*.yaml")))
    assert len(part_files) > 3
    # keep only the first three part files, and re-write them to contain one problem ID each
    part_files = part_files[:3]
    expected_problem_ids = []
    for part_file in part_files:
        with part_file.open("r") as fd:
            part_ids = yaml.safe_load(fd)
        assert len(part_ids) <= 2000
        target_problem_ids = [part_ids[0]]
        expected_problem_ids.append(target_problem_ids)
        with part_file.open("w") as fd:
            yaml.safe_dump(target_problem_ids, fd)
    out_dataset_paths = []
    for part_idx, part_file in enumerate(part_files, start=1):
        out_dataset_path = tmp_path / f"dataset_p{part_idx:02d}.lmdb"
        out_dataset_paths.append(out_dataset_path)
        cli_args = [
            "traces",
            "--dataset-name=TACO",
            f"--output-path={out_dataset_path}",
            "--max-solutions-per-problem=2",
            "--max-tests-per-solution=2",
            f"--target-problem-ids={part_file}",
            "--generate-obfuscated-solutions",
        ]
        res = cli_runner.invoke(pyine.apps.write.dataset_writer.main, cli_args)  # noqa
        assert res.exit_code == 0, res
        assert out_dataset_path.is_dir()
    readers = [pyine.data.traces.dataset_reader.DatasetReader(path) for path in out_dataset_paths]
    for part_idx, r in enumerate(readers):
        expected_pids = expected_problem_ids[part_idx]
        found_pids = r.problem_keys
        assert set(expected_pids) == set(found_pids)
        trace_ids = r.trace_keys
        for tid in trace_ids:
            assert any([tid.startswith(pid) for pid in expected_pids])


def test_partition_force_flag_protects_existing_outputs(tmp_path: pathlib.Path) -> None:
    cli_runner = click.testing.CliRunner()
    split_file = tmp_path / "demo-split.bin"
    config = pyine.data.utils.splits.SplitConfig(
        subset_names=["train", "valid"],
        subset_assign_prob_map={"train": 1.0, "valid": 0.0},
    )
    split_result = pyine.data.utils.splits.SplitResult(
        source_dataset_name="demo",
        source_dataset_hash="hash",
        identifiers=["p0", "p1"],
        tag_lists=[["tag"] for _ in range(2)],
        source_data_hashes=["h0", "h1"],
        subset_assignments={"p0": "train", "p1": "train"},
        creation_metadata={"unit_test": True},
        config=config,
    )
    split_result.to_file(split_file)
    output_dir = tmp_path / "parts"
    base_args = [
        "partition",
        f"--split-file={split_file}",
        f"--output-dir={output_dir}",
        "--ids-per-chunk=1",
        "--format=yaml",
    ]
    first = cli_runner.invoke(pyine.apps.splits.dataset_splitter.main, base_args)
    assert first.exit_code == 0, first
    part_files = sorted(output_dir.glob("*.yaml"))
    assert len(part_files) == 2
    expected_contents = {path: path.read_text(encoding="utf-8") for path in part_files}

    second = cli_runner.invoke(pyine.apps.splits.dataset_splitter.main, base_args)
    assert second.exit_code == 1
    assert "already exists" in second.output
    for path in part_files:
        assert path.read_text(encoding="utf-8") == expected_contents[path]

    corrupted_path = part_files[0]
    corrupted_path.write_text("corrupted", encoding="utf-8")
    forced = cli_runner.invoke(pyine.apps.splits.dataset_splitter.main, base_args + ["--force"])
    assert forced.exit_code == 0, forced
    reloaded = yaml.safe_load(corrupted_path.read_text(encoding="utf-8"))
    assert reloaded == ["p0"]
