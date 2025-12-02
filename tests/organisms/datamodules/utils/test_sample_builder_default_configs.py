import pathlib
import typing

import pytest

import pyine.data.traces.dataset_reader
import pyine.data.traces.dataset_utils as dataset_utils
import pyine.organisms.datamodules.samples as sample_utils
import pyine.organisms.datamodules.shortcuts_configs as shortcuts_configs
import pyine.prompts
import pyine.utils.code.execution as execution_utils
import pyine.utils.pydantic


class FakeDatasetReader(pyine.data.traces.dataset_reader.DatasetReader):
    def __init__(
        self,
        dataset_hash: str,
        traces: list[execution_utils.TraceResult],
    ) -> None:
        self._dataset_hash = dataset_hash
        self._traces = traces
        self.trace_keys = [typing.cast("str", trace.identifier) for trace in traces]
        self.path = pathlib.Path("dummy/path")
        self.problem_keys: list[str] = []
        self.trace_key_to_problem_key: dict[str, str] = {}
        self.augment_key_to_parent_trace_key: dict[str, str] = {}
        self.trace_metadata: list[dataset_utils.TraceMetadata] = []
        for idx, trace in enumerate(self._traces):
            identifier = typing.cast("str", trace.identifier)
            trace_id = dataset_utils.TraceIdentifier.from_string(identifier)
            solution_id = trace_id.get_parent_identifier()
            problem_id = solution_id.get_parent_identifier()
            problem_key = str(problem_id)
            if problem_key not in self.problem_keys:
                self.problem_keys.append(problem_key)
            self.trace_key_to_problem_key[identifier] = problem_key
            if trace_id.is_augmented:
                parent_id = trace_id.get_augmentless_identifier()
                self.augment_key_to_parent_trace_key[identifier] = str(parent_id)
            self.trace_metadata.append(
                dataset_utils.TraceMetadata(
                    identifier=identifier,
                    parent_dataset_hash=self._dataset_hash,
                    index=idx,
                    internal_index=idx,
                    step_count=trace.valid_step_count,
                    code_string=trace.code_string,
                    inputs=trace.inputs,
                    expected_output=trace.expected_output,
                    return_value=trace.return_value,
                    exception=trace.exception,
                    stdout=trace.stdout,
                    stderr=trace.stderr,
                    metadata=trace.metadata,
                    tags=list(trace.tags),
                )
            )
        self._tags_by_key = {trace.identifier: list(trace.tags) for trace in self._traces}
        self._parent_dataset_name = dataset_hash

    @property
    def hash(self) -> str:
        return self._dataset_hash

    def __len__(self) -> int:
        return len(self._traces)

    def __getitem__(
        self,
        trace_idx: int,
    ) -> execution_utils.TraceResult:
        return self._traces[trace_idx]

    @property
    def metadata(self) -> dict[str, typing.Any]:
        return {
            "parent_dataset": {
                "dataset_name": self._parent_dataset_name,
                "dataset_path": str(self.path),
                "trace_count": len(self._traces),
            }
        }

    @property
    def size_on_disk(self) -> int:
        return 0

    @property
    def parent_dataset_name(self) -> str:
        return self._parent_dataset_name

    def get_problem_data(
        self,
        index_or_key: int | str,
    ) -> typing.Any:
        raise NotImplementedError("FakeDatasetReader does not provide problem data")

    def get_trace_metadata(
        self,
        index_or_key: int | str,
    ) -> dataset_utils.TraceMetadata:
        if isinstance(index_or_key, int):
            return self.trace_metadata[index_or_key]
        key = typing.cast("str", index_or_key)
        for meta in self.trace_metadata:
            if meta.identifier == key:
                return meta
        raise KeyError(key)

    def get_tags(
        self,
        index_or_key: int | str,
    ) -> list[str]:
        key = self.trace_keys[index_or_key] if isinstance(index_or_key, int) else typing.cast("str", index_or_key)
        return list(self._tags_by_key.get(key, []))


class FakePromptResult:
    def __init__(
        self,
        identifier: str,
        prompt_name: str | None,
        result: str,
        tags: list[str] | None = None,
    ) -> None:
        self.identifier = identifier
        self.prompt_name = prompt_name
        self.result = result
        self.tags = tags or []


class FakePromptResultDB:
    def __init__(
        self,
        records_by_identifier: dict[str, list[FakePromptResult]] | None = None,
    ) -> None:
        self._records_by_identifier = records_by_identifier or {}

    def get_by_identifier(
        self,
        identifier: str,
        prompt_name: str | None = None,
    ) -> list[FakePromptResult]:
        records = self._records_by_identifier.get(identifier, [])
        if prompt_name is None:
            return list(records)
        return [record for record in records if record.prompt_name == prompt_name]

    def get_tags(
        self,
        identifier: str | list[str],
        prompt_name: str | list[str] | None = None,
        breakdown: bool = False,
    ) -> list[list[str]] | dict[tuple[str, ...], list[list[str]]]:
        if not breakdown:
            if isinstance(identifier, list):
                records = [rec for id in identifier for rec in self._records_by_identifier.get(id, [])]
            else:
                records = self._records_by_identifier.get(identifier, [])
            if prompt_name is None:
                return [record.tags for record in records]
            if isinstance(prompt_name, list):
                return [record.tags for record in records if record.prompt_name in prompt_name]
            return [record.tags for record in records if record.prompt_name == prompt_name]
        # breakdown=True: return dict mapping filter tuples to lists of tag lists
        result_dict: dict[tuple[str, ...], list[list[str]]] = {}
        identifiers = [identifier] if isinstance(identifier, str) else identifier
        prompt_names = [prompt_name] if isinstance(prompt_name, str) else (prompt_name or [None])
        for id in identifiers:
            records = self._records_by_identifier.get(id, [])
            for record in records:
                if prompt_name is None or record.prompt_name in prompt_names:
                    # Build key tuple based on which filters were lists
                    key_parts: list[str] = []
                    if isinstance(identifier, list):
                        key_parts.append(id)
                    if isinstance(prompt_name, list):
                        key_parts.append(record.prompt_name or "")
                    key = tuple(key_parts) if key_parts else (id,)
                    if key not in result_dict:
                        result_dict[key] = []
                    result_dict[key].append(record.tags)
        return result_dict


def build_trace_artifacts(
    dataset_hash: str,
    trace_idx: int,
    identifier: str,
    tags: list[str],
    return_value: int,
) -> tuple[dataset_utils.TraceMetadata, execution_utils.TraceResult]:
    trace_result = execution_utils.TraceResult(
        identifier=identifier,
        code_string=f"def solution():\n    return {return_value}\n",
        code_blocks={},
        inputs={"args": [return_value]},
        expected_output=return_value,
        max_valid_events=None,
        max_events_per_line=None,
        max_var_repr_length=None,
        traced_steps=[],
        traced_steps_map={},
        entrypoint_name="solution",
        entrypoint_step_idx=None,
        return_value=return_value,
        exception=None,
        stdout="",
        stderr="",
        metadata={},
        tags=tags,
    )
    trace_metadata = dataset_utils.TraceMetadata(
        identifier=identifier,
        parent_dataset_hash=dataset_hash,
        index=trace_idx,
        internal_index=trace_idx,
        step_count=trace_result.valid_step_count,
        code_string=trace_result.code_string,
        inputs=trace_result.inputs,
        expected_output=trace_result.expected_output,
        return_value=trace_result.return_value,
        exception=trace_result.exception,
        stdout=trace_result.stdout,
        stderr=trace_result.stderr,
        metadata=trace_result.metadata,
        tags=tags,
    )
    return trace_metadata, trace_result


def setup_builder(
    monkeypatch: pytest.MonkeyPatch,
    traces: list[dataset_utils.TraceMetadata],
    reader: FakeDatasetReader,
    seed: int,
    subset_name: str,
    prompt_records: dict[str, list[FakePromptResult]] | None = None,
) -> sample_utils.SampleBuilder:
    base_config = shortcuts_configs._get_default_sampler_builder_config(seed=seed)
    overrides = shortcuts_configs._get_default_sample_builder_overrides_for_subset(subset_name)
    config = pyine.utils.pydantic.merge_configs(base_config, overrides)
    fake_prompt_db = FakePromptResultDB(prompt_records)
    monkeypatch.setattr(pyine.prompts, "get_framework_db", lambda: fake_prompt_db)
    return sample_utils.SampleBuilder(
        source_data=reader,
        traces=traces,
        filtering_config=config["filtering_config"],
        selection_config=config["selection_config"],
        transform_config=config["transform_config"],
    )


def test_default_valid_config_prefers_original(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_hash = "fake-valid"
    original_identifier = "TACO/valid/p000001/s0000/t0000"
    hinted_identifier = "TACO/valid/p000001/s0000/t0000/a:hinted:000"
    trace_metadatas: list[dataset_utils.TraceMetadata] = []
    trace_results: list[execution_utils.TraceResult] = []
    for idx, (identifier, tags) in enumerate(
        [
            (original_identifier, ["subset:valid"]),
            (hinted_identifier, ["subset:valid", "augment:hinted"]),
        ]
    ):
        metadata, result = build_trace_artifacts(
            dataset_hash=dataset_hash,
            trace_idx=idx,
            identifier=identifier,
            tags=tags,
            return_value=idx,
        )
        trace_metadatas.append(metadata)
        trace_results.append(result)
    reader = FakeDatasetReader(dataset_hash=dataset_hash, traces=trace_results)
    builder = setup_builder(
        monkeypatch=monkeypatch,
        traces=trace_metadatas,
        reader=reader,
        seed=11,
        subset_name="valid",
    )
    assert len(builder) == 1
    selection = builder.selection_results[0]
    assert selection.code_type.is_original
    assert selection.code_override is None
    sample = builder[0]
    assert "original" in sample.code_type
    assert sample.predict_type == "program_output"
    assert sample.has_code_override is False
    assert sample.code == selection.trace_meta.code_string


def test_train_overrides_enable_random_hint_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_hash = "fake-train-hinted"
    trace_specifications = []
    prompt_records = {}
    solution_count = 200
    for solution_idx in range(solution_count):
        solution_identifier = f"TACO/train/p000010/s{solution_idx:04d}"
        trace_identifier = f"{solution_identifier}/t0001"
        obfs_trace_identifier = f"{trace_identifier}/a:obfuscated:000"
        trace_specifications.extend(
            [
                (trace_identifier, ["subset:train"]),
                (
                    obfs_trace_identifier,
                    ["subset:train", "augment:obfuscated"],
                ),
                (
                    f"{trace_identifier}/a:bugged:001",
                    ["subset:train", "augment:bugged"],
                ),
            ]
        )
        prompt_records[solution_identifier] = [
            FakePromptResult(
                identifier=solution_identifier,
                prompt_name="code_summary",
                result="some description of the code",
            )
        ]
        prompt_records[trace_identifier] = [
            FakePromptResult(
                identifier=trace_identifier,
                prompt_name="hints/docs",
                result="fake hinted code",
            ),
        ]
        prompt_records[obfs_trace_identifier] = [
            FakePromptResult(
                identifier=obfs_trace_identifier,
                prompt_name="hints/docs",
                result="fake obfuscated + hinted code",
            ),
        ]
    trace_metadatas: list[dataset_utils.TraceMetadata] = []
    trace_results: list[execution_utils.TraceResult] = []
    for idx, (identifier, tags) in enumerate(trace_specifications):
        metadata, result = build_trace_artifacts(
            dataset_hash=dataset_hash,
            trace_idx=idx,
            identifier=identifier,
            tags=tags,
            return_value=idx,
        )
        trace_metadatas.append(metadata)
        trace_results.append(result)
    reader = FakeDatasetReader(dataset_hash=dataset_hash, traces=trace_results)
    builder_without_records = setup_builder(
        monkeypatch=monkeypatch,
        traces=trace_metadatas,
        reader=reader,
        seed=0,
        subset_name="train",
    )
    assert len(builder_without_records) == solution_count
    for sample_idx in range(len(builder_without_records)):
        selected = builder_without_records.selection_results[sample_idx]
        assert selected.code_type.has_any(["original", "obfuscated", "bugged"])
        picked_identifier = selected.trace_meta.identifier
        expected_solution_identifier = f"TACO/train/p000010/s{sample_idx:04d}"
        expected_trace_identifier_prefix = f"{expected_solution_identifier}/t0001"
        assert picked_identifier.startswith(expected_trace_identifier_prefix)
        if selected.code_type.is_original:
            assert picked_identifier == expected_trace_identifier_prefix
        elif selected.code_type.is_obfuscated:
            picked_identifier.endswith("/a:obfuscated:000")
        elif selected.code_type.is_bugged:
            picked_identifier.endswith("/a:bugged:001")
        assert selected.code_override is None
        sample = builder_without_records[sample_idx]
        assert sample.identifier == picked_identifier
        assert sample.code_type == selected.code_type.types
        assert sample.predict_type == "program_output"
        assert sample.has_code_override is False
        assert not sample.description
        tags = sample.comma_separated_tags.split(",")
        assert "sample_code_description:0" in tags
        assert f"sample_code_type:{selected.code_type}" in tags
        assert "sample_predict_type:program_output" in tags

    builder_with_records = setup_builder(
        monkeypatch=monkeypatch,
        traces=trace_metadatas,
        reader=reader,
        seed=0,
        subset_name="train",
        prompt_records=prompt_records,
    )
    assert len(builder_with_records) == solution_count
    for sample_idx in range(len(builder_with_records)):
        selected = builder_with_records.selection_results[sample_idx]
        assert selected.code_type.has_any(["original", "hinted", "obfuscated"])
        picked_identifier = selected.trace_meta.identifier
        expected_solution_identifier = f"TACO/train/p000010/s{sample_idx:04d}"
        expected_trace_identifier_prefix = f"{expected_solution_identifier}/t0001"
        assert picked_identifier.startswith(expected_trace_identifier_prefix)
        if selected.code_type.is_original or not selected.code_type.is_obfuscated:
            assert picked_identifier == expected_trace_identifier_prefix
            if selected.code_type == "hinted":
                assert selected.code_override == "fake hinted code"
            else:
                assert selected.code_override is None
        elif selected.code_type.is_obfuscated:
            picked_identifier.endswith("/a:obfuscated:000")
            if selected.code_type.types == frozenset(["obfuscated"]):
                assert selected.code_override is None
            else:
                assert selected.code_override == "fake obfuscated + hinted code"
        sample = builder_with_records[sample_idx]
        assert sample.identifier == picked_identifier
        assert sample.code_type == selected.code_type.types
        assert sample.predict_type == "program_output"
        if selected.code_type.is_hinted:
            assert sample.has_code_override is True
            assert sample.code in ["fake hinted code", "fake obfuscated + hinted code"]
        else:
            assert sample.has_code_override is False
            assert sample.code not in [
                "fake hinted code",
                "fake obfuscated + hinted code",
            ]
        assert sample.description == "some description of the code"
        tags = sample.comma_separated_tags.split(",")
        assert "sample_code_description:1" in tags
        assert f"sample_code_type:{selected.code_type}" in tags
        assert "sample_predict_type:program_output" in tags


def test_train_overrides_fetch_stubbed_from_prompt_db(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_hash = "fake-train-stubbed"
    trace_specifications = []
    prompt_records = {}
    solution_count = 200
    for solution_idx in range(solution_count):
        solution_identifier = f"TACO/train/p000010/s{solution_idx:04d}"
        trace_identifier = f"{solution_identifier}/t0001"
        trace_specifications.append((trace_identifier, ["subset:train"]))
        prompt_records[solution_identifier] = [
            FakePromptResult(
                identifier=solution_identifier,
                prompt_name="code_stubbing",
                result=f"def solution_{solution_idx}():\n    raise NotImplementedError('stubbed')\n",
            )
        ]
    trace_metadatas: list[dataset_utils.TraceMetadata] = []
    trace_results: list[execution_utils.TraceResult] = []
    for idx, (identifier, tags) in enumerate(trace_specifications):
        metadata, result = build_trace_artifacts(
            dataset_hash=dataset_hash,
            trace_idx=idx,
            identifier=identifier,
            tags=tags,
            return_value=idx,
        )
        trace_metadatas.append(metadata)
        trace_results.append(result)
    reader = FakeDatasetReader(dataset_hash=dataset_hash, traces=trace_results)
    builder = setup_builder(
        monkeypatch=monkeypatch,
        traces=trace_metadatas,
        reader=reader,
        seed=4,
        subset_name="train",
        prompt_records=prompt_records,
    )
    assert len(builder) == solution_count
    for sample_idx in range(len(builder)):
        selection = builder.selection_results[sample_idx]
        assert selection.code_type.is_original or selection.code_type.is_stubbed
        expected_stubbed_code = f"def solution_{sample_idx}():\n    raise NotImplementedError('stubbed')\n"
        sample = builder[sample_idx]
        assert sample.code_type == selection.code_type.types
        assert sample.has_code_override == (selection.code_override is not None)
        if selection.code_type.is_original:
            assert selection.code_override is None
            assert sample.code != expected_stubbed_code
        else:
            assert selection.code_override == expected_stubbed_code
            assert sample.code == expected_stubbed_code
        assert sample.predict_type == "program_output"


# @@@@ TODO: update the tests to make sure we can hit (and do hit) all supported sample types
