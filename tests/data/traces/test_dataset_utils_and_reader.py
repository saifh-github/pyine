import json
import pathlib
import typing

import orjson
import pytest

import pyine.data.traces.dataset_reader as dataset_reader
import pyine.data.traces.dataset_utils as dataset_utils
import pyine.data.traces.dataset_writer as dataset_writer
import pyine.utils.code.execution as exec_utils
import pyine.utils.filesystem as fs_utils
import tests.env_checks
import tests.utils.fake_dataset_readers as fake_dataset_readers


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


def test_trace_dataset_collection_combines_parts() -> None:
    """Verify DatasetCollection concatenates readers while preserving metadata."""
    reader_a = fake_dataset_readers.FakeTraceDatasetReader(
        config=fake_dataset_readers.FakeTraceDataConfig(
            dataset_name="FAKE_A",
            subset_name="train",
            num_problems=1,
            solutions_per_problem=1,
            tests_per_problem=2,
        )
    )
    reader_b = fake_dataset_readers.FakeTraceDatasetReader(
        config=fake_dataset_readers.FakeTraceDataConfig(
            dataset_name="FAKE_B",
            subset_name="valid",
            num_problems=1,
            solutions_per_problem=1,
            tests_per_problem=1,
        )
    )
    collection = dataset_reader.DatasetCollection([reader_a, reader_b])
    assert len(collection) == len(reader_a) + len(reader_b)
    assert collection.trace_keys[: len(reader_a.trace_keys)] == reader_a.trace_keys
    assert collection.trace_keys[-len(reader_b.trace_keys) :] == reader_b.trace_keys
    component_hashes = {reader_a.hash, reader_b.hash}
    for idx in range(len(collection)):
        trace = collection[idx]
        meta = collection.get_trace_metadata(idx)
        assert trace.identifier == meta.identifier
        assert meta.index == idx
        assert meta.parent_dataset_hash == collection.hash
        assert meta.metadata["source_dataset_hash"] in component_hashes
        problem = collection.get_problem_data(idx)
        assert problem.problem_id == meta.problem_id
        assert set(collection.get_tags(idx)) == set(meta.tags)
    target_key = reader_b.trace_keys[0]
    fetched_by_key = collection[target_key]
    assert fetched_by_key.identifier == target_key
    key_meta = collection.get_trace_metadata(target_key)
    assert key_meta.identifier == target_key
    combined_metadata = collection.metadata["parent_dataset"]
    assert combined_metadata["trace_count"] == len(collection)
    assert len(combined_metadata["components"]) == 2


def _write_minimal_taco_problem(root_dir: pathlib.Path) -> pathlib.Path:
    root_dir.mkdir(parents=True, exist_ok=True)
    problem_path = root_dir / "999999.json"
    problem_payload = {
        "subset": "train",
        "source": "leetcode",
        "difficulty": "easy",
        "question": "Describe how to solve the task in detail." * 5,
        "input_output": {
            "inputs": [["[1, 2, 3]"]],
            "outputs": [[6]],
            "fn_name": "Solution.solve",
        },
        "solutions": [
            {
                "code": "class Solution:\n    def solve(self, items):\n        return sum(items)",
                "language": "python",
            }
        ],
        "raw_tags": ["arrays"],
    }
    problem_path.write_text(json.dumps(problem_payload), encoding="utf-8")
    return problem_path


def test_iterator_auto_overrides_with_cache(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_root = tmp_path / "dataset"
    _write_minimal_taco_problem(dataset_root)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    override_dir = cache_dir / "overrides" / "TACO"
    override_dir.mkdir(parents=True)
    override_file = override_dir / "problem_data_overrides.json"
    override_payload = {
        "TACO/train/p999999": {
            "inputs": [["[0]"]],
            "outputs": [[0]],
            "fn_name": "Solution.solve",
        }
    }
    override_file.write_text(json.dumps(override_payload), encoding="utf-8")
    monkeypatch.setattr(fs_utils, "get_data_cache_path", lambda: cache_dir)

    iterator = dataset_utils.CodingProblemIterator(
        dataset_name="TACO",
        root_data_path=dataset_root,
        problem_data_overrides_setting="auto",
        allow_banned_samples=True,
    )

    assert iterator._problem_data_overrides == override_payload
    assert iterator._problem_data_override_source == override_file


def test_iterator_auto_overrides_missing_keeps_default(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_root = tmp_path / "dataset"
    _write_minimal_taco_problem(dataset_root)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(fs_utils, "get_data_cache_path", lambda: cache_dir)

    iterator = dataset_utils.CodingProblemIterator(
        dataset_name="TACO",
        root_data_path=dataset_root,
        problem_data_overrides_setting="auto",
        allow_banned_samples=True,
    )

    assert iterator._problem_data_overrides == {}
    assert iterator._problem_data_override_source is None


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.dataset
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
@pytest.mark.integration
@pytest.mark.dataset
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
    for augm_key, _parent_trace_key in reader.augment_key_to_parent_trace_key.items():
        assert "obfuscated" in str(augm_key)
    # check that all written traces are EASY ones
    for trace_idx in range(len(reader)):
        problem_data = reader.get_problem_data(trace_idx)
        assert any("EASY" in t for t in problem_data.problem_tags)
    traces_metadata = dataset_reader.get_traces_metadata(output_dataset_path)
    assert len(traces_metadata) == len(reader)


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


def _write_target_problem(
    root_dir: pathlib.Path,
    idx: int,
    subset: str,
    question: str = "dummy question",
) -> None:
    analysis_output = {
        "is_deterministic": True,
        "imports_nonstandard_packages": False,
        "invalid_syntax": False,
        "blocks_execution": False,
        "unnecessary_lines": False,
        "filesystem_access": False,
        "system_commands": False,
        "network_access": False,
        "code_type": "simple",
        "input_type": "stdin",
        "output_type": "stdout",
    }
    problem_payload = {
        "subset": subset,
        "question": question,
        "source": "source",
        "difficulty": "easy",
        "raw_tags": [],
        "tags": [],
        "skill_types": [],
        "input_output": {
            "inputs": [[idx]],
            "outputs": [idx],
        },
        "solutions": [
            {
                "code": "def solve(x):\n    return x\n",
                "analysis_outputs": [analysis_output],
                "validation_errors": [],
            }
        ],
    }
    path = root_dir / f"{idx:06d}.json"
    path.write_bytes(orjson.dumps(problem_payload))


def test_iterator_filters_problem_ids_without_subset(
    tmp_path: pathlib.Path,
) -> None:
    root_dir = tmp_path / "taco"
    root_dir.mkdir()
    _write_target_problem(root_dir, 1, "train")
    _write_target_problem(root_dir, 2, "train")
    iterator = dataset_utils.CodingProblemIterator(
        dataset_name="TACO",
        root_data_path=root_dir,
        target_problem_ids=["p000001"],
    )
    assert len(iterator.problems_metadata) == 1
    problems = list(iterator)
    assert len(problems) == 1
    problem, solutions = problems[0]
    assert problem.problem_id.problem_idx == 1
    assert problem.problem_id.subset == "train"
    assert solutions


def test_iterator_accepts_subset_qualified_ids(
    tmp_path: pathlib.Path,
) -> None:
    root_dir = tmp_path / "taco"
    root_dir.mkdir()
    _write_target_problem(root_dir, 1, "train")
    _write_target_problem(root_dir, 2, "valid")
    iterator = dataset_utils.CodingProblemIterator(
        dataset_name="TACO",
        root_data_path=root_dir,
        target_problem_ids=["TACO/train/p000001"],
    )
    assert len(iterator.problems_metadata) == 1
    problems = list(iterator)
    assert len(problems) == 1
    problem, _ = problems[0]
    assert problem.problem_id.problem_idx == 1
    assert problem.problem_id.subset == "train"


def test_subset_collision_requires_explicit_identifier() -> None:
    iterator = object.__new__(dataset_utils.CodingProblemIterator)
    spec = dataset_utils.CodingProblemIterator._TargetProblemSpec(problem_idx=1, subset=None)
    iterator._target_problem_specs = {spec}
    iterator._target_problem_spec_matches = {}
    assert iterator._matches_target_problem(problem_idx=1, subset="train")
    with pytest.raises(ValueError):
        iterator._matches_target_problem(problem_idx=1, subset="valid")


def test_iterator_nonexistent_directory_raises(tmp_path: pathlib.Path) -> None:
    """Test iterator raises on non-existent dataset directory."""
    nonexistent_dir = tmp_path / "nonexistent"
    with pytest.raises(FileNotFoundError):
        dataset_utils.CodingProblemIterator(
            dataset_name="TACO",
            root_data_path=nonexistent_dir,
            allow_banned_samples=True,
        )


def test_iterator_can_iterate_multiple_times(tmp_path: pathlib.Path) -> None:
    """Test that iterator can be iterated multiple times."""
    root_dir = tmp_path / "taco"
    root_dir.mkdir()
    _write_target_problem(root_dir, 1, "train")
    _write_target_problem(root_dir, 2, "train")
    iterator = dataset_utils.CodingProblemIterator(
        dataset_name="TACO",
        root_data_path=root_dir,
        allow_banned_samples=True,
    )
    first_pass = list(iterator)
    second_pass = list(iterator)
    assert len(first_pass) == 2
    assert len(second_pass) == 2
    # same problems should be returned
    assert first_pass[0][0].problem_id == second_pass[0][0].problem_id
    assert first_pass[1][0].problem_id == second_pass[1][0].problem_id


def test_iterator_with_explicit_override_path(tmp_path: pathlib.Path) -> None:
    """Test iterator with explicit override path."""
    root_dir = tmp_path / "taco"
    root_dir.mkdir()
    _write_target_problem(root_dir, 1, "train")
    override_file = tmp_path / "my_overrides.json"
    override_payload = {
        "TACO/train/p000001": {
            "inputs": [["[999]"]],
            "outputs": [[999]],
        }
    }
    override_file.write_text(json.dumps(override_payload), encoding="utf-8")
    iterator = dataset_utils.CodingProblemIterator(
        dataset_name="TACO",
        root_data_path=root_dir,
        problem_data_overrides_setting=override_file,
        allow_banned_samples=True,
    )
    assert iterator._problem_data_overrides == override_payload
    assert iterator._problem_data_override_source == override_file


def test_iterator_with_invalid_override_path_raises(tmp_path: pathlib.Path) -> None:
    """Test iterator raises on invalid override path."""
    root_dir = tmp_path / "taco"
    root_dir.mkdir()
    _write_target_problem(root_dir, 1, "train")
    invalid_path = tmp_path / "nonexistent_overrides.json"
    with pytest.raises(ValueError, match="not found"):
        dataset_utils.CodingProblemIterator(
            dataset_name="TACO",
            root_data_path=root_dir,
            problem_data_overrides_setting=invalid_path,
            allow_banned_samples=True,
        )


def test_iterator_problems_metadata_is_consistent(tmp_path: pathlib.Path) -> None:
    """Test that problems_metadata matches iteration results."""
    root_dir = tmp_path / "taco"
    root_dir.mkdir()
    _write_target_problem(root_dir, 1, "train")
    _write_target_problem(root_dir, 2, "train")
    _write_target_problem(root_dir, 3, "valid")
    iterator = dataset_utils.CodingProblemIterator(
        dataset_name="TACO",
        root_data_path=root_dir,
        allow_banned_samples=True,
    )
    problems = list(iterator)
    # problems_metadata is a list of paths
    assert len(iterator.problems_metadata) == len(problems)
    # verify each path exists and matches expected problem indices
    for path in iterator.problems_metadata:
        assert path.exists()
        assert path.suffix == ".json"


@pytest.mark.slow
def test_fake_dataset_reader_length(tmp_path: pathlib.Path) -> None:
    """Test that FakeTraceDatasetReader has correct length."""
    reader = fake_dataset_readers.FakeTraceDatasetReader(
        config=fake_dataset_readers.FakeTraceDataConfig(
            dataset_name="FAKE",
            subset_name="train",
            num_problems=2,
            solutions_per_problem=3,
            tests_per_problem=4,
        )
    )
    # length should be num_problems * solutions_per_problem * tests_per_problem
    expected_length = 2 * 3 * 4
    assert len(reader) == expected_length


def test_fake_dataset_reader_indexing(tmp_path: pathlib.Path) -> None:
    """Test that FakeTraceDatasetReader supports indexing by int and key."""
    reader = fake_dataset_readers.FakeTraceDatasetReader(
        config=fake_dataset_readers.FakeTraceDataConfig(
            dataset_name="FAKE",
            subset_name="train",
            num_problems=1,
            solutions_per_problem=1,
            tests_per_problem=1,
        )
    )
    trace_by_idx = reader[0]  # index by int
    assert trace_by_idx.identifier in reader.trace_keys
    assert trace_by_idx.code_string  # verify trace has actual content
    assert trace_by_idx.valid_step_count >= 0
    key = reader.trace_keys[0]  # index by key
    trace_by_key = reader[key]
    assert trace_by_key.identifier == key
    # same trace should be returned by index and key
    assert trace_by_idx.identifier == trace_by_key.identifier


def test_fake_dataset_reader_invalid_index_raises(tmp_path: pathlib.Path) -> None:
    """Test that FakeTraceDatasetReader raises on invalid index."""
    reader = fake_dataset_readers.FakeTraceDatasetReader(
        config=fake_dataset_readers.FakeTraceDataConfig(
            dataset_name="FAKE",
            subset_name="train",
            num_problems=1,
            solutions_per_problem=1,
            tests_per_problem=1,
        )
    )
    with pytest.raises(IndexError):
        _ = reader[999]


def test_dataset_collection_empty_raises() -> None:
    """Test that DatasetCollection with empty list raises."""
    with pytest.raises(ValueError):
        dataset_reader.DatasetCollection([])


def test_dataset_collection_iteration() -> None:
    """Test that DatasetCollection supports iteration."""
    reader = fake_dataset_readers.FakeTraceDatasetReader(
        config=fake_dataset_readers.FakeTraceDataConfig(
            dataset_name="FAKE",
            subset_name="train",
            num_problems=1,
            solutions_per_problem=1,
            tests_per_problem=2,
        )
    )
    collection = dataset_reader.DatasetCollection([reader])
    traces = list(collection)
    assert len(traces) == 2
    for trace in traces:
        assert trace.identifier in collection.trace_keys


def test_dataset_collection_get_problem_data() -> None:
    """Test DatasetCollection get_problem_data method."""
    reader = fake_dataset_readers.FakeTraceDatasetReader(
        config=fake_dataset_readers.FakeTraceDataConfig(
            dataset_name="FAKE",
            subset_name="train",
            num_problems=2,
            solutions_per_problem=1,
            tests_per_problem=1,
        )
    )
    collection = dataset_reader.DatasetCollection([reader])
    for idx in range(len(collection)):
        problem = collection.get_problem_data(idx)
        # verify problem has expected structure
        assert problem.problem_id.dataset == "FAKE"
        assert problem.problem_id.subset == "train"
        assert problem.problem_statement  # verify has actual content
        assert len(problem.test_inout_pairs) > 0


def test_dataset_collection_get_tags() -> None:
    """Test DatasetCollection get_tags method returns tags matching metadata."""
    reader = fake_dataset_readers.FakeTraceDatasetReader(
        config=fake_dataset_readers.FakeTraceDataConfig(
            dataset_name="FAKE",
            subset_name="train",
            num_problems=1,
            solutions_per_problem=1,
            tests_per_problem=1,
        )
    )
    collection = dataset_reader.DatasetCollection([reader])
    tags = collection.get_tags(0)
    metadata = collection.get_trace_metadata(0)
    # tags from collection should match tags from metadata
    assert set(tags) == set(metadata.tags)


def test_trace_metadata_from_fake_reader() -> None:
    """Test TraceMetadata from FakeTraceDatasetReader."""
    reader = fake_dataset_readers.FakeTraceDatasetReader(
        config=fake_dataset_readers.FakeTraceDataConfig(
            dataset_name="FAKE",
            subset_name="train",
            num_problems=1,
            solutions_per_problem=1,
            tests_per_problem=1,
        )
    )
    meta = reader.get_trace_metadata(0)
    assert isinstance(meta, dataset_utils.TraceMetadata)
    assert meta.identifier == reader.trace_keys[0]
    assert meta.index == 0
    assert meta.parent_dataset_hash == reader.hash


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.dataset
@pytest.mark.timeout(120)
@pytest.mark.skipif(
    tests.env_checks.TACO_DATASET_MISSING,
    reason="TACO dataset is missing, cannot run integration test",
)
def test_taco_reader_supports_key_indexing(tmp_path: pathlib.Path) -> None:
    """Test that dataset reader supports both integer and key-based indexing."""
    output_dataset_path = tmp_path / "mini_taco_key_indexing"
    dataset_writer.write_dataset_from_taco(
        output_dataset_path=output_dataset_path,
        max_output_traces=5,
        max_solutions_per_problem=1,
        max_tests_per_solution=1,
        verbose=False,
    )
    reader = dataset_reader.DatasetReader(output_dataset_path)
    trace_by_idx = reader[0]  # test integer indexing
    assert trace_by_idx is not None
    key = reader.trace_keys[0]  # test key indexing
    trace_by_key = reader[key]
    assert trace_by_key.identifier == key
    assert trace_by_idx.identifier == trace_by_key.identifier


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.dataset
@pytest.mark.timeout(120)
@pytest.mark.skipif(
    tests.env_checks.TACO_DATASET_MISSING,
    reason="TACO dataset is missing, cannot run integration test",
)
def test_taco_reader_metadata_structure(tmp_path: pathlib.Path) -> None:
    """Test that dataset reader metadata has expected structure."""
    output_dataset_path = tmp_path / "mini_taco_metadata"
    dataset_writer.write_dataset_from_taco(
        output_dataset_path=output_dataset_path,
        max_output_traces=3,
        max_solutions_per_problem=1,
        max_tests_per_solution=1,
        verbose=False,
    )
    reader = dataset_reader.DatasetReader(output_dataset_path)
    # check metadata structure
    assert "parent_dataset" in reader.metadata
    parent_metadata = reader.metadata["parent_dataset"]
    assert parent_metadata["dataset_name"] == "TACO"
    assert "problem_count" in parent_metadata
    assert "dataset_hash" in parent_metadata
    # check reader properties
    assert reader.parent_dataset_name == "TACO"
    assert reader.hash is not None
    assert len(reader.hash) > 0


def test_dataset_collection_from_multiple_fake_readers() -> None:
    """Test that DatasetCollection works with multiple reader parts."""
    # create two separate fake readers with different configurations to ensure unique traces
    reader_1 = fake_dataset_readers.FakeTraceDatasetReader(
        config=fake_dataset_readers.FakeTraceDataConfig(
            dataset_name="FAKE_A",
            subset_name="train",
            num_problems=2,
            solutions_per_problem=1,
            tests_per_problem=1,
        )
    )
    reader_2 = fake_dataset_readers.FakeTraceDatasetReader(
        config=fake_dataset_readers.FakeTraceDataConfig(
            dataset_name="FAKE_B",
            subset_name="train",
            num_problems=2,
            solutions_per_problem=1,
            tests_per_problem=1,
        )
    )

    collection = dataset_reader.DatasetCollection([reader_1, reader_2])
    # collection should have combined length
    assert len(collection) == len(reader_1) + len(reader_2)
    # all trace keys should be unique (verified by DatasetCollection init)
    assert len(set(collection.trace_keys)) == len(collection.trace_keys)
    # collection should be iterable
    traces = list(collection)
    assert len(traces) == len(collection)
    # verify traces from both readers are accessible
    reader_1_keys = set(reader_1.trace_keys)
    reader_2_keys = set(reader_2.trace_keys)
    for key in collection.trace_keys:
        assert key in reader_1_keys or key in reader_2_keys
