import dataclasses
import pathlib
import typing

import pytest

import pyine.data.traces.dataset_reader
import pyine.data.traces.dataset_utils as du
import pyine.organisms.datamodules.utils.annotator as annotator
import pyine.prompts.manager as prompt_manager
import pyine.prompts.result_db as result_db
import pyine.prompts.types as prompt_types
import pyine.utils.code.execution as exec_utils
import pyine.utils.llm_providers as llm_providers
import tests.data.utils.env_checks


@dataclasses.dataclass
class _FakeTrace:
    identifier: str | None
    code_string: str


class _FakeDatasetReader:
    def __init__(self, traces: list[_FakeTrace], problems: list[du.CodingProblem]):
        assert len(traces) == len(problems)
        self._traces = traces
        self._problems = problems

    def __len__(self) -> int:
        return len(self._traces)

    def __getitem__(self, idx: int) -> typing.Any:
        return exec_utils.TraceResult(
            identifier=self._traces[idx].identifier,
            code_string=self._traces[idx].code_string,
            code_blocks={},
            inputs="",
            expected_output="",
            max_valid_events=None,
            max_events_per_line=None,
            max_var_repr_length=None,
            traced_steps=[],
            traced_steps_map={},
            entrypoint_name=None,
            entrypoint_step_idx=None,
            return_value=None,
            exception=None,
            stdout="",
            stderr="",
            metadata={},
            tags=[],
        )

    def get_problem_data(self, idx: int) -> du.CodingProblem:
        return self._problems[idx]


class _DummyModel:
    def invoke(self, _: typing.Any) -> str:
        return "dummy-result"


def _make_ids() -> tuple[str, str, str]:
    pid = du.CodingProblemIdentifier(dataset="POTATO", subset="train", problem_idx=1)
    sid = du.SolutionIdentifier(**vars(pid), solution_idx=2)
    tid = du.TraceIdentifier(**vars(sid), test_idx=3)
    return str(pid), str(sid), str(tid)


def _make_problem(pid: str) -> du.CodingProblem:
    return du.CodingProblem(
        source_dataset_name="POTATO",
        source_data_path="/tmp/dataset",
        source_data_hash="something",
        problem_id=du.CodingProblemIdentifier.from_string(pid),
        problem_statement="Add two numbers.",
        problem_tags=["math"],
        test_inout_pairs=[(1, 2)],
        entrypoint_name=None,
        potential_solution_ids=[du.SolutionIdentifier.from_string(f"{pid}/s0002")],
        parsing_errors=None,
        is_banned=False,
    )


def _make_dataset() -> tuple[_FakeDatasetReader, str, str]:
    pid, sid, tid = _make_ids()
    trace = _FakeTrace(identifier=tid, code_string="def add(a,b): return a+b")
    problem = _make_problem(pid)
    dataset = _FakeDatasetReader([trace], [problem])
    return dataset, sid, pid


@pytest.mark.asyncio
async def test_annotate_generates_and_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset, sid, pid = _make_dataset()
    monkeypatch.setattr(prompt_manager, "list_prompts", lambda: ["code_summary"])
    monkeypatch.setattr(llm_providers, "get_model_from_provider", lambda **kwargs: _DummyModel())
    captured: dict[str, typing.Any] = {}

    def _fake_fetch(
        model: typing.Any,
        identifier: str,
        input_variables: dict[str, typing.Any],
        prompt_config: typing.Any,
        *,
        creation_meta: result_db.CreationMeta,
        group: str | None,
        tags: list[str],
        meta: dict[str, typing.Any],
        **kwargs,
    ) -> list[result_db.PromptResultRecord]:
        captured.update(
            {
                "identifier": identifier,
                "group": group,
                "prompt_name": getattr(prompt_config, "prompt_name", None),
                "input_variables": dict(input_variables),
            }
        )
        # return one record to simulate generation; mark it as "new" by reusing the provided creation_meta
        return [
            result_db.PromptResultRecord(
                identifier=identifier,
                prompt_name=prompt_config.prompt_name,
                prompt_version=typing.cast(str | None, getattr(prompt_config, "version", None)),
                group=group,
                creation_meta=creation_meta,
                prompt="prompt",
                result="result1",
                meta=meta or {},
                tags=tags or [],
            )
        ]

    def _id_resolver(trace: exec_utils.TraceResult, _: du.CodingProblem, __: annotator.AnnotationOptions) -> str:
        assert trace.identifier is not None
        return str(du.TraceIdentifier.from_string(str(trace.identifier)).get_parent_identifier())

    def _group_resolver(_: exec_utils.TraceResult, problem: du.CodingProblem, __: annotator.AnnotationOptions) -> str:
        return str(problem.problem_id)

    def _input_builder(
        trace: exec_utils.TraceResult, problem: du.CodingProblem, __: annotator.AnnotationOptions
    ) -> dict[str, typing.Any]:
        return {
            "code": trace.code_string,
            "description": problem.problem_statement,
            "target_word_count": 40,
        }

    monkeypatch.setattr(result_db, "fetch_or_generate_prompt_results", _fake_fetch)
    options = annotator.AnnotationOptions(
        llm_provider_config={"provider": "openai"},
        prompt_config=prompt_types.PromptBuildConfig(prompt_name="code_summary", version=None),
        identifier_resolver=_id_resolver,
        group_resolver=_group_resolver,
        input_variables_builder=_input_builder,
        tags_builder=lambda *_: [],  # noqa
        meta_builder=lambda *_: {},  # noqa
        min_results_per_item=1,
        force_generation=True,
    )
    report = await annotator.annotate_trace_dataset(
        dataset,  # type: ignore[arg-type]
        config=options,
        show_progress=False,
    )
    assert report.total_samples == 1
    assert report.new_results_generated == 1
    assert report.skipped_samples == 0
    assert report.errors == 0
    assert captured["identifier"] == sid  # SolutionIdentifier for code_summary
    assert captured["group"] == pid  # grouped by CodingProblemIdentifier
    assert captured["prompt_name"] == "code_summary"
    assert captured["input_variables"]["code"].startswith("def add")
    assert captured["input_variables"]["description"].startswith("Add two numbers")
    assert captured["input_variables"]["target_word_count"] == 40


@pytest.mark.asyncio
async def test_annotate_skips_when_existing(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset, sid, pid = _make_dataset()
    monkeypatch.setattr(prompt_manager, "list_prompts", lambda: ["code_summary"])
    monkeypatch.setattr(llm_providers, "get_model_from_provider", lambda **kwargs: _DummyModel())

    # fetch returns a pre-existing record with a different creation_meta -> treated as "skipped"
    def _fake_fetch_same_count(
        model: typing.Any,
        identifier: str,
        input_variables: dict[str, typing.Any],
        prompt_config: typing.Any,
        *,
        group: str | None,
        tags: list[str],
        meta: dict[str, typing.Any],
        **kwargs,
    ) -> list[result_db.PromptResultRecord]:
        # creation_meta is intentionally different
        different_meta = result_db.CreationMeta(provider="other")
        return [
            result_db.PromptResultRecord(
                identifier=identifier,
                prompt_name=getattr(prompt_config, "prompt_name", "code_summary"),
                prompt_version=getattr(prompt_config, "version", None),
                group=group,
                creation_meta=different_meta,
                prompt="prompt",
                result="result1",
                meta=meta or {},
                tags=tags or [],
            )
        ]

    def _id_resolver(trace: exec_utils.TraceResult, _: du.CodingProblem, __: annotator.AnnotationOptions) -> str:
        assert trace.identifier is not None
        return str(du.TraceIdentifier.from_string(str(trace.identifier)).get_parent_identifier())

    def _group_resolver(_: exec_utils.TraceResult, problem: du.CodingProblem, __: annotator.AnnotationOptions) -> str:
        return str(problem.problem_id)

    monkeypatch.setattr(result_db, "fetch_or_generate_prompt_results", _fake_fetch_same_count)
    options = annotator.AnnotationOptions(
        llm_provider_config={"provider": "openai"},
        prompt_config=prompt_types.PromptBuildConfig(prompt_name="code_summary", version=None),
        identifier_resolver=_id_resolver,
        group_resolver=_group_resolver,
        input_variables_builder=lambda trace, problem, cfg: {
            "code": trace.code_string,
            "description": problem.problem_statement,
        },
        tags_builder=lambda *_: [],  # noqa
        meta_builder=lambda *_: {},  # noqa
        min_results_per_item=1,
    )
    report = await annotator.annotate_trace_dataset(
        dataset,  # type: ignore[arg-type]
        config=options,
        show_progress=False,
    )
    assert report.total_samples == 1
    assert report.new_results_generated == 0
    assert report.skipped_samples == 1
    assert report.errors == 0


@pytest.mark.asyncio
async def test_error_handling_increments_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    pid, sid, tid = _make_ids()
    problem = _make_problem(pid)
    # first ok, second broken (invalid identifier that cannot resolve)
    traces = [
        _FakeTrace(identifier=tid, code_string="def add(a,b): return a+b"),
        _FakeTrace(identifier=None, code_string="print('x')"),
    ]
    dataset = _FakeDatasetReader(traces, [problem, problem])
    monkeypatch.setattr(prompt_manager, "list_prompts", lambda: ["code_summary"])
    monkeypatch.setattr(llm_providers, "get_model_from_provider", lambda **kwargs: _DummyModel())

    def _fake_fetch(
        model: typing.Any,
        identifier: str,
        input_variables: dict[str, typing.Any],
        prompt_config: typing.Any,
        *,
        creation_meta: result_db.CreationMeta,
        group: str | None,
        tags: list[str],
        meta: dict[str, typing.Any],
        **kwargs,
    ) -> list[result_db.PromptResultRecord]:
        return [
            result_db.PromptResultRecord(
                identifier=identifier,
                prompt_name="code_summary",
                prompt_version=None,
                group=group,
                creation_meta=creation_meta,
                prompt="p",
                result="r",
                meta=meta or {},
                tags=tags or [],
            )
        ]

    def _id_resolver(trace: exec_utils.TraceResult, _: du.CodingProblem, __: annotator.AnnotationOptions) -> str:
        if trace.identifier is None:
            raise ValueError("cannot derive identifier without a trace id")
        return str(du.TraceIdentifier.from_string(str(trace.identifier)).get_parent_identifier())

    monkeypatch.setattr(result_db, "fetch_or_generate_prompt_results", _fake_fetch)
    options = annotator.AnnotationOptions(
        llm_provider_config={"provider": "openai"},
        prompt_config=prompt_types.PromptBuildConfig(prompt_name="code_summary", version=None),
        identifier_resolver=_id_resolver,
        group_resolver=lambda _t, p, _c: str(p.problem_id),
        input_variables_builder=lambda t, p, c: {"code": t.code_string, "description": p.problem_statement},
        tags_builder=lambda *_: [],  # noqa
        meta_builder=lambda *_: {},  # noqa
        min_results_per_item=1,
    )
    report = await annotator.annotate_trace_dataset(
        dataset,  # type: ignore[arg-type]
        config=options,
        show_progress=False,
    )
    assert report.total_samples == 2
    assert report.errors == 1
    assert report.new_results_generated + report.skipped_samples == 1


@pytest.mark.slow
@pytest.mark.asyncio
@pytest.mark.skipif(
    tests.data.utils.env_checks.TACO_TRACES_DATASET_MISSING,
    reason="TACO traces dataset is missing, cannot run annotator integration",
)
@pytest.mark.skipif(
    tests.data.utils.env_checks.OPENAI_API_KEY_MISSING,
    reason="OPENAI_API_KEY is missing, cannot run annotator integration",
)
async def test_annotator_integration_with_real_traces_dataset(tmp_path: str) -> None:
    dataset_path = du.get_latest_dataset_path("TACO")
    dataset = pyine.data.traces.dataset_reader.DatasetReader(lmdb_path=dataset_path)
    target_indices = list(range(0, 100, 10))
    if len(dataset) <= max(target_indices):
        pytest.skip("traces dataset too small for test")
    first_target_idx = min(target_indices)
    trace = dataset[first_target_idx]
    problem = dataset.get_problem_data(first_target_idx)
    assert trace.identifier is not None
    # compute solution id for code_summary identifier
    sid = du.TraceIdentifier.from_string(str(trace.identifier)).get_parent_identifier()
    db_path = pathlib.Path(tmp_path) / "prompt.db"
    temp_db = result_db.PromptResultDB(db_path)
    # ensure no generation happens for first data sample by pre-inserting a record into the DB
    temp_db.store(
        identifier=str(sid),
        prompt="p",
        result="r",
        meta={},
        tags=[],
        group=str(problem.problem_id),
        prompt_name="code_summary",
        prompt_version=None,
        creation_meta=result_db.CreationMeta(),
    )
    options = annotator.AnnotationOptions(
        llm_provider_config=dict(
            provider="openai",
            model="gpt-4o-mini",
        ),
        prompt_config=prompt_types.PromptBuildConfig(
            prompt_name="code_summary",
            partial_vars=dict(
                target_word_count=30,
            ),
        ),
        target_indices=target_indices,
        min_results_per_item=1,
        deduplicate_results=True,
        db_path=db_path,
    )
    report = await annotator.annotate_trace_dataset(
        dataset,
        config=options,
        show_progress=False,
    )
    assert report.total_samples == len(target_indices)
    assert report.skipped_samples >= 1
    assert report.total_tokens_exchanged > 0
    # now do a 2nd annotation pass w/ same db but different prompt
    options = annotator.AnnotationOptions(
        llm_provider_config=dict(
            provider="openai",
            model="gpt-4o-mini",
        ),
        prompt_config=prompt_types.PromptBuildConfig(
            prompt_name="hints/docs",
        ),
        target_indices=target_indices,
        min_results_per_item=1,
        deduplicate_results=True,
        db_path=db_path,
    )
    report = await annotator.annotate_trace_dataset(
        dataset,
        config=options,
        show_progress=False,
    )
    assert report.total_samples == len(target_indices)
    assert report.skipped_samples >= 0
    assert report.total_tokens_exchanged > 0
    assert temp_db.list_prompt_names() == ["code_summary", "hints/docs"]
