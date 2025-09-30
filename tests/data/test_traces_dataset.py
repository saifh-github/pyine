import pathlib
import typing

import pytest

import pyine.data.traces.dataset_reader as dataset_reader
import pyine.data.traces.dataset_utils as dataset_utils
import pyine.data.traces.dataset_writer as dataset_writer
import pyine.utils.code.execution as exec_utils
import pyine.utils.filesystem as fs_utils
import tests.env_checks


@pytest.mark.parametrize(
    "value, is_float_expected",
    [
        ("42", True),
        ("3.14", True),
        ("-1e-9", True),
        ("nan", True),
        ("hello", False),
        ("", False),
    ],
)
def test_is_float(
    value: str,
    is_float_expected: bool,
) -> None:
    """Verify *is_float* correctly identifies numeric strings."""
    assert dataset_utils.is_float(value) is is_float_expected


@pytest.mark.parametrize(
    "lhs, rhs, equal_expected",
    [
        ("42", "42.0", True),
        ("3.1415", "3.14150000", True),
        ("abc", "abc", True),
        ("abc", "def", False),
    ],
)
def test_compare_result_strings(
    lhs: str,
    rhs: str,
    equal_expected: bool,
) -> None:
    """Verify *compare_result_strings* tolerance and exact-match logic."""
    assert dataset_utils.compare_result_strings(lhs, rhs) is equal_expected


def test_dataset_paths(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure *get_new_dataset_path* always returns a non-existing, unique path."""
    # monkey-patch the root path used by the utils module to the temp dir
    monkeypatch.setattr(fs_utils, "get_data_root_path", lambda: tmp_path)
    path_a = dataset_utils.get_new_dataset_path("TACO", "v01")
    path_b = dataset_utils.get_new_dataset_path("TACO", "v02")
    assert path_a != path_b
    assert not path_a.exists() and not path_b.exists()
    path_a.mkdir(parents=True)
    found_dataset_path = dataset_utils.get_latest_dataset_path("TACO")
    assert found_dataset_path == path_a
    assert found_dataset_path.exists()
    path_b.mkdir(parents=True)
    found_dataset_path = dataset_utils.get_latest_dataset_path("TACO")
    assert found_dataset_path == path_b


@pytest.mark.slow
@pytest.mark.timeout(120)  # 2 minutes should be plenty, otherwise tracing is failing for all snippets
@pytest.mark.skipif(
    tests.env_checks.TACO_DATASET_MISSING,
    reason="TACO dataset is missing, cannot create mini traces dataset",
)
def test_mini_taco_traces_dataset(
    tmp_path: pathlib.Path,
) -> None:
    """Checks that a mini traces dataset can be created and read successfully."""
    output_dataset_path = tmp_path / "mini_taco_traces_dataset"
    assert not output_dataset_path.exists()
    wanted_trace_count = 10
    dataset_writer.write_dataset_from_taco(
        output_dataset_path=output_dataset_path,
        max_output_traces=wanted_trace_count,
        max_solutions_per_problem=2,
        max_tests_per_solution=1,
        min_solution_dissimilarity=0.1,
        verbose=True,
    )
    assert output_dataset_path.exists()
    reader = dataset_reader.DatasetReader(output_dataset_path)
    assert reader.parent_dataset_name == "TACO"
    assert len(reader) >= wanted_trace_count
    assert reader.size_on_disk > 0
    for trace_idx, trace_data in enumerate(reader):  # noqa
        assert isinstance(trace_data, exec_utils.TraceResult)
        trace_metadata = reader.get_trace_metadata(trace_idx)
        assert isinstance(trace_metadata, dataset_utils.TraceMetadata)
        assert trace_data.identifier == trace_metadata.identifier
        assert trace_data.identifier in reader.trace_keys
        assert trace_metadata.parent_dataset_hash == reader.hash
        assert trace_metadata.index == trace_idx
        assert trace_metadata.internal_index == reader._trace_indices[trace_idx]
        assert trace_metadata.code_string == trace_data.code_string
        assert trace_metadata.step_count == trace_data.valid_step_count


@pytest.mark.slow
@pytest.mark.timeout(120)  # 2 minutes should be plenty, otherwise tracing is failing for all snippets
@pytest.mark.skipif(
    tests.env_checks.TACO_DATASET_MISSING,
    reason="TACO dataset is missing, cannot create mini traces dataset",
)
def test_mini_taco_easy_traces_dataset_with_obfuscated_augments(
    tmp_path: pathlib.Path,
) -> None:
    output_dataset_path = tmp_path / "mini_taco_traces_dataset_w_obfusc"
    assert not output_dataset_path.exists()
    wanted_trace_count = 1
    dataset_writer.write_dataset_from_taco(
        output_dataset_path=output_dataset_path,
        banned_problem_tags_rule="+{difficulty:easy|difficulty:EASY}",
        max_output_traces=wanted_trace_count,
        max_solutions_per_problem=1,
        max_tests_per_solution=2,
        min_solution_line_count=10,
        min_solution_dissimilarity=0.1,
        generate_obfuscated_solutions=True,
        verbose=True,
    )
    assert output_dataset_path.exists()
    reader = dataset_reader.DatasetReader(output_dataset_path)
    assert reader.metadata["parent_dataset"]["dataset_name"] == "TACO"
    assert len(reader) >= wanted_trace_count
    # check written traces to make sure we do have some augments
    assert len(reader.augment_key_to_parent_trace_key) > 0
    for augm_key, parent_trace_key in reader.augment_key_to_parent_trace_key.items():
        assert "obfuscated" in str(augm_key)
    # check that all written traces are EASY ones
    for trace_idx in range(len(reader)):
        problem_data = reader.get_problem_data(trace_idx)
        assert any(["EASY" in t for t in problem_data.problem_tags])


def test_write_dataset_from_taco_forwards_force_flag(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, typing.Any] = {}

    def fake_write_dataset(**kwargs: typing.Any) -> typing.Any:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(dataset_writer, "write_dataset", fake_write_dataset)
    dataset_writer.write_dataset_from_taco(
        source_dataset_path=tmp_path / "source",
        output_dataset_path=tmp_path / "output",
        force_overwrite=True,
    )
    assert captured["force_overwrite"] is True
