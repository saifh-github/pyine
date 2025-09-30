import typing

import pytest

import pyine.data.traces.dataset_utils as dataset_utils
import pyine.organisms.datamodules.shortcuts_configs as shortcuts_configs
import pyine.organisms.datamodules.utils.samples as sample_utils
import pyine.prompts
import pyine.utils.code.execution as execution_utils
import pyine.utils.pydantic


class FakeDatasetReader:
    def __init__(
        self,
        dataset_hash: str,
        traces: list[execution_utils.TraceResult],
    ) -> None:
        self._dataset_hash = dataset_hash
        self._traces = traces
        self.trace_keys = [typing.cast(str, trace.identifier) for trace in traces]

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
    hinted_identifier = "TACO/valid/p000001/s0000/t0000/a:issues_generic:000"
    trace_metadatas: list[dataset_utils.TraceMetadata] = []
    trace_results: list[execution_utils.TraceResult] = []
    for idx, (identifier, tags) in enumerate(
        [
            (original_identifier, ["subset:valid"]),
            (hinted_identifier, ["subset:valid", "augment:issues_generic"]),
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
    selection = builder.selected_traces[0]
    assert selection.code_type == "original"
    assert selection.code_override is None
    sample = builder[0]
    assert sample.code_type == "original"
    assert sample.output_type == "program_output"
    assert sample.has_code_override is False


def test_train_overrides_enable_random_hint_selection(monkeypatch: pytest.MonkeyPatch) -> None:
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
                    f"{trace_identifier}/a:issues_generic:001",
                    ["subset:train", "augment:issues_generic"],
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
    saw_orig, saw_obfs = False, False
    for sample_idx in range(len(builder_without_records)):
        selected = builder_without_records.selected_traces[sample_idx]
        assert selected.code_type in ["original", "obfuscated"]
        picked_identifier = selected.trace_meta.identifier
        expected_solution_identifier = f"TACO/train/p000010/s{sample_idx:04d}"
        expected_trace_identifier_prefix = f"{expected_solution_identifier}/t0001"
        assert picked_identifier.startswith(expected_trace_identifier_prefix)
        if selected.code_type == "original":
            assert picked_identifier == expected_trace_identifier_prefix
            saw_orig = True
        elif selected.code_type == "obfuscated":
            picked_identifier.endswith("/a:obfuscated:000")
            saw_obfs = True
        assert selected.code_override is None
        sample = builder_without_records[sample_idx]
        assert sample.identifier == picked_identifier
        assert sample.code_type == selected.code_type
        assert sample.output_type == "program_output"
        assert sample.has_code_override is False
        assert not sample.description
        tags = sample.comma_separated_tags.split(",")
        assert "augment:has_code_description" not in tags
        assert f"sample_code_type:{selected.code_type}" in tags
        assert "sample_output_type:program_output" in tags
    assert saw_orig and saw_obfs

    builder_with_records = setup_builder(
        monkeypatch=monkeypatch,
        traces=trace_metadatas,
        reader=reader,
        seed=0,
        subset_name="train",
        prompt_records=prompt_records,
    )
    assert len(builder_with_records) == solution_count
    saw_orig, saw_hinted, saw_obfs, saw_obfs_hinted = False, False, False, False
    for sample_idx in range(len(builder_with_records)):
        selected = builder_with_records.selected_traces[sample_idx]
        assert selected.code_type in ["original", "hinted", "obfuscated", "obfuscated_hinted"]
        picked_identifier = selected.trace_meta.identifier
        expected_solution_identifier = f"TACO/train/p000010/s{sample_idx:04d}"
        expected_trace_identifier_prefix = f"{expected_solution_identifier}/t0001"
        assert picked_identifier.startswith(expected_trace_identifier_prefix)
        if selected.code_type in ["original", "hinted"]:
            assert picked_identifier == expected_trace_identifier_prefix
            if selected.code_type == "hinted":
                assert selected.code_override == "fake hinted code"
                saw_hinted = True
            else:
                assert selected.code_override is None
                saw_orig = True
        elif selected.code_type in ["obfuscated", "obfuscated_hinted"]:
            picked_identifier.endswith("/a:obfuscated:000")
            if selected.code_type == "obfuscated":
                assert selected.code_override is None
                saw_obfs = True
            else:
                assert selected.code_override == "fake obfuscated + hinted code"
                saw_obfs_hinted = True
        sample = builder_with_records[sample_idx]
        assert sample.identifier == picked_identifier
        assert sample.code_type == selected.code_type
        assert sample.output_type == "program_output"
        if selected.code_type in ["hinted", "obfuscated_hinted"]:
            assert sample.has_code_override is True
            assert sample.code in ["fake hinted code", "fake obfuscated + hinted code"]
        else:
            assert sample.has_code_override is False
            assert sample.code not in ["fake hinted code", "fake obfuscated + hinted code"]
        assert sample.description == "some description of the code"
        tags = sample.comma_separated_tags.split(",")
        assert "augment:has_code_description" in tags
        assert f"sample_code_type:{selected.code_type}" in tags
        assert "sample_output_type:program_output" in tags
    assert saw_orig and saw_hinted and saw_obfs and saw_obfs_hinted


def test_train_overrides_fetch_stubbed_from_prompt_db(monkeypatch: pytest.MonkeyPatch) -> None:
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
                prompt_name="hints/stubs",
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
    saw_stubbed = False
    for sample_idx in range(len(builder)):
        selection = builder.selected_traces[sample_idx]
        assert selection.code_type in ["original", "stubbed"]
        expected_stubbed_code = f"def solution_{sample_idx}():\n    raise NotImplementedError('stubbed')\n"
        sample = builder[sample_idx]
        assert sample.code_type == selection.code_type
        assert sample.has_code_override == (selection.code_override is not None)
        if selection.code_type == "original":
            assert selection.code_override is None
            assert sample.code != expected_stubbed_code
        else:
            assert selection.code_override == expected_stubbed_code
            assert sample.code == expected_stubbed_code
            saw_stubbed = True
        assert sample.output_type == "program_output"
    assert saw_stubbed
