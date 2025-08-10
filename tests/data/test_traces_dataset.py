import asyncio
import pathlib

import pytest

import pyine.data.traces.dataset_reader as dataset_reader
import pyine.data.traces.dataset_utils as dataset_utils
import pyine.data.traces.dataset_writer as dataset_writer
import pyine.utils.filesystem as fs_utils
import tests.data.utils.dataset_checks


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
@pytest.mark.skipif(
    tests.data.utils.dataset_checks.TACO_DATASET_MISSING,
    reason="TACO dataset is missing, cannot create mini traces dataset",
)
def test_mini_taco_traces_dataset(
    tmp_path: pathlib.Path,
) -> None:
    """Checks that a mini traces dataset can be created and read successfully."""
    output_dataset_path = tmp_path / "mini_taco_traces_dataset"
    assert not output_dataset_path.exists()
    wanted_trace_count = 10
    asyncio.run(
        dataset_writer.write_dataset_from_taco(
            output_dataset_path=output_dataset_path,
            max_output_traces=wanted_trace_count,
            max_solutions_per_problem=2,
            max_tests_per_solution=1,
            min_solution_dissimilarity=0.1,
            verbose=True,
        )
    )
    assert output_dataset_path.exists()
    reader = dataset_reader.DatasetReader(output_dataset_path)
    assert reader.get_metadata()["source_dataset"]["source_dataset_name"] == "TACO"
    assert len(reader) == wanted_trace_count
