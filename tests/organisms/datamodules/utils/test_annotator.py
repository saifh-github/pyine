import dataclasses
import pathlib
import tempfile
import types
import typing

import langchain_core.runnables
import numpy as np
import pytest

import pyine.data.traces.dataset_reader
import pyine.data.traces.dataset_utils as du
import pyine.organisms.datamodules.utils.annotator as annotator
import pyine.prompts.manager as prompt_manager
import pyine.prompts.result_db as result_db
import pyine.prompts.types as prompt_types
import pyine.utils.code.execution as exec_utils
import pyine.utils.llm_providers as llm_providers
import tests.env_checks


@dataclasses.dataclass
class _FakeTrace:
    identifier: str | None
    code_string: str
    inputs: typing.Any = dataclasses.field(default_factory=lambda: ["alpha", "beta"])
    expected_output: typing.Any = "omega"
    tags: list[str] = dataclasses.field(default_factory=list)


class _FakeDatasetReader:
    def __init__(
        self,
        traces: list[_FakeTrace],
        problems: list[du.CodingProblem],
    ) -> None:
        assert len(traces) == len(problems)
        self._traces = traces
        self._problems = problems

    def __len__(self) -> int:
        return len(self._traces)

    def __getitem__(self, idx: int) -> typing.Any:
        trace_data = self._traces[idx]
        return exec_utils.TraceResult(
            identifier=trace_data.identifier,
            code_string=trace_data.code_string,
            code_blocks={},
            inputs=trace_data.inputs,
            expected_output=trace_data.expected_output,
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
            tags=list(trace_data.tags),
        )

    def get_problem_data(self, idx: int) -> du.CodingProblem:
        return self._problems[idx]


class _DummyModel(langchain_core.runnables.Runnable):
    def invoke(self, *args: typing.Any, **kwargs: typing.Any) -> str:
        return "dummy-result"

    async def ainvoke(self, *args: typing.Any, **kwargs: typing.Any) -> str:
        return "dummy-result"


def _make_ids() -> tuple[str, str, str]:
    pid = du.CodingProblemIdentifier(dataset="POTATO", subset="train", problem_idx=1)
    sid = du.SolutionIdentifier(**vars(pid), solution_idx=2)
    tid = du.TraceIdentifier(**vars(sid), test_idx=3)
    return str(pid), str(sid), str(tid)


def _make_problem(pid: str) -> du.CodingProblem:
    return du.CodingProblem(
        source_dataset_name="POTATO",
        source_data_path=str(pathlib.Path(tempfile.gettempdir()) / "dataset"),
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
    trace = _FakeTrace(
        identifier=tid,
        code_string="def add(a, b): return a + b",
        inputs=[1, 2],
        expected_output="3",
        tags=["synthetic"],
    )
    problem = _make_problem(pid)
    dataset = _FakeDatasetReader([trace], [problem])
    return dataset, sid, pid


async def _run_prompt_case(
    prompt_name: str,
    *,
    dataset: _FakeDatasetReader,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    partial_vars: dict[str, typing.Any] | None = None,
    prepopulate_summary: bool = True,
    identifier_resolver: annotator.IdentifierResolverType | None = None,
    group_resolver: annotator.GroupResolverType | None = None,
    augment_config: annotator.AugmentedAnnotationOptions | None = None,
    test_cache: typing.Any = None,
    prepopulate_db_records: list[dict[str, typing.Any]] | None = None,
) -> dict[str, typing.Any]:
    """Run annotation for a specific prompt and capture builder inputs."""

    captured: dict[str, typing.Any] = {}

    def _fake_fetch(
        *,
        identifier: str,
        input_variables: dict[str, typing.Any],
        prompt_chain_config: typing.Any,
        creation_meta: result_db.CreationMeta,
        group: str | None,
        tags: list[str],
        meta: dict[str, typing.Any],
        **kwargs: typing.Any,
    ) -> list[result_db.PromptResultRecord]:
        captured.update(
            {
                "identifier": identifier,
                "group": group,
                "prompt_name": prompt_chain_config.prompt.prompt_name,
                "input_variables": dict(input_variables),
                "tags": list(tags),
                "meta": dict(meta),
            }
        )
        return [
            result_db.PromptResultRecord(
                identifier=identifier,
                prompt_name=prompt_chain_config.prompt.prompt_name,
                prompt_version=prompt_chain_config.prompt.version,
                group=group,
                creation_meta=creation_meta,
                prompt="prompt",
                result="modified-code",
                meta=meta,
                tags=tags,
            )
        ]

    monkeypatch.setattr(
        prompt_manager,
        "list_prompts",
        lambda: annotator.supported_prompts_for_trace_dataset_annotation,
    )
    monkeypatch.setattr(result_db, "fetch_or_generate_prompt_results", _fake_fetch)
    monkeypatch.setattr(llm_providers, "get_model_from_provider_config", lambda **_: _DummyModel())

    db_path = tmp_path / f"{prompt_name.replace('/', '_')}_test.db"
    options = annotator.AnnotationOptions(
        llm_provider_config={"provider": "openai"},
        prompt_config=prompt_types.PromptBuildConfig(
            prompt_name=prompt_name,
            partial_vars=partial_vars or {},
        ),
        min_results_per_item=1,
        force_generation=True,
        db_path=db_path,
        identifier_resolver=identifier_resolver,
        group_resolver=group_resolver,
        augment_config=augment_config or annotator.AugmentedAnnotationOptions(),
    )

    trace = dataset[0]
    problem = dataset.get_problem_data(0)
    if prepopulate_summary:
        assert trace.identifier is not None
        trace_id = du.TraceIdentifier.from_string(str(trace.identifier))
        solution_id = trace_id.get_parent_identifier()
        options._prompt_result_db.store(
            identifier=str(solution_id),
            prompt="summary",
            result="Existing summary",
            meta={},
            tags=[],
            group=str(problem.problem_id),
            prompt_name="code_summary",
            prompt_version=None,
        )

    if prepopulate_db_records:
        for record in prepopulate_db_records:
            options._prompt_result_db.store(**record)

    if test_cache is not None:
        object.__setattr__(options, "_test_data_cache", test_cache)

    report = await annotator.annotate_trace_dataset(
        dataset,  # type: ignore[arg-type]
        config=options,
        show_progress=False,
        dry_run=False,
        parallel=False,
    )
    captured["report"] = report
    return captured


@pytest.mark.asyncio
async def test_annotate_generates_and_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset, sid, pid = _make_dataset()
    monkeypatch.setattr(prompt_manager, "list_prompts", lambda: ["code_summary"])
    monkeypatch.setattr(llm_providers, "get_model_from_provider", lambda **kwargs: _DummyModel())
    captured: dict[str, typing.Any] = {}

    def _fake_fetch(
        *,
        identifier: str,
        input_variables: dict[str, typing.Any],
        prompt_chain_config: typing.Any,
        creation_meta: result_db.CreationMeta,
        group: str | None,
        tags: list[str],
        meta: dict[str, typing.Any],
        **kwargs: typing.Any,
    ) -> list[result_db.PromptResultRecord]:
        captured.update(
            {
                "identifier": identifier,
                "group": group,
                "prompt_name": prompt_chain_config.prompt.prompt_name,
                "input_variables": dict(input_variables),
            }
        )
        # return one record to simulate generation; mark it as "new" by reusing the provided creation_meta
        return [
            result_db.PromptResultRecord(
                identifier=identifier,
                prompt_name=prompt_chain_config.prompt.prompt_name,
                prompt_version=prompt_chain_config.prompt.version,
                group=group,
                creation_meta=creation_meta,
                prompt="prompt",
                result="result1",
                meta=meta or {},
                tags=tags or [],
            )
        ]

    def _id_resolver(
        trace: exec_utils.TraceResult,
        _: du.CodingProblem,
        __: annotator.AnnotationOptions,
    ) -> str:
        assert trace.identifier is not None
        return str(du.TraceIdentifier.from_string(str(trace.identifier)).get_parent_identifier())

    def _group_resolver(
        _: exec_utils.TraceResult,
        problem: du.CodingProblem,
        __: annotator.AnnotationOptions,
    ) -> str:
        return str(problem.problem_id)

    def _input_builder(
        trace: exec_utils.TraceResult,
        problem: du.CodingProblem,
        __: annotator.AnnotationOptions,
        **kwargs: typing.Any,
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
        tags_builder=lambda *_, **__: [],  # noqa
        meta_builder=lambda *_, **__: {},  # noqa
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
    assert report.errors == 0, report.error_messages
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
        *,
        identifier: str,
        input_variables: dict[str, typing.Any],
        prompt_chain_config: typing.Any,
        group: str | None,
        tags: list[str],
        meta: dict[str, typing.Any],
        **kwargs: typing.Any,
    ) -> list[result_db.PromptResultRecord]:
        # creation_meta is intentionally different
        different_meta = result_db.CreationMeta(provider="other")
        return [
            result_db.PromptResultRecord(
                identifier=identifier,
                prompt_name=prompt_chain_config.prompt.prompt_name,
                prompt_version=prompt_chain_config.prompt.version,
                group=group,
                creation_meta=different_meta,
                prompt="prompt",
                result="result1",
                meta=meta or {},
                tags=tags or [],
            )
        ]

    def _id_resolver(
        trace: exec_utils.TraceResult,
        _: du.CodingProblem,
        __: annotator.AnnotationOptions,
    ) -> str:
        assert trace.identifier is not None
        return str(du.TraceIdentifier.from_string(str(trace.identifier)).get_parent_identifier())

    def _group_resolver(
        _: exec_utils.TraceResult,
        problem: du.CodingProblem,
        __: annotator.AnnotationOptions,
    ) -> str:
        return str(problem.problem_id)

    monkeypatch.setattr(result_db, "fetch_or_generate_prompt_results", _fake_fetch_same_count)
    options = annotator.AnnotationOptions(
        llm_provider_config={"provider": "openai"},
        prompt_config=prompt_types.PromptBuildConfig(prompt_name="code_summary", version=None),
        identifier_resolver=_id_resolver,
        group_resolver=_group_resolver,
        input_variables_builder=lambda trace, problem, cfg, **__: {
            "code": trace.code_string,
            "description": problem.problem_statement,
        },
        tags_builder=lambda *_, **__: [],  # noqa
        meta_builder=lambda *_, **__: {},  # noqa
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
    assert report.errors == 0, report.error_messages


@pytest.mark.asyncio
async def test_bad_id_handling_increments_skips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        *,
        identifier: str,
        input_variables: dict[str, typing.Any],
        prompt_chain_config: typing.Any,
        creation_meta: result_db.CreationMeta,
        group: str | None,
        tags: list[str],
        meta: dict[str, typing.Any],
        **kwargs: typing.Any,
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

    def _id_resolver(
        trace: exec_utils.TraceResult,
        _: du.CodingProblem,
        __: annotator.AnnotationOptions,
    ) -> str | None:
        if trace.identifier is None:
            return None  # cannot derive identifier without a trace id
        return str(du.TraceIdentifier.from_string(str(trace.identifier)).get_parent_identifier())

    monkeypatch.setattr(result_db, "fetch_or_generate_prompt_results", _fake_fetch)
    options = annotator.AnnotationOptions(
        llm_provider_config={"provider": "openai"},
        prompt_config=prompt_types.PromptBuildConfig(prompt_name="code_summary", version=None),
        identifier_resolver=_id_resolver,
        group_resolver=lambda _t, p, _c: str(p.problem_id),
        input_variables_builder=lambda t, p, c, **__: {
            "code": t.code_string,
            "description": p.problem_statement,
        },
        tags_builder=lambda *_, **__: [],  # noqa
        meta_builder=lambda *_, **__: {},  # noqa
        min_results_per_item=1,
    )
    report = await annotator.annotate_trace_dataset(
        dataset,  # type: ignore[arg-type]
        config=options,
        show_progress=False,
    )
    assert report.total_samples == 2
    assert report.errors == 0, report.error_messages
    assert report.new_results_generated + report.skipped_samples == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "prompt_name",
        "identifier_kind",
        "group_kind",
        "expect_description",
        "expect_inputs",
        "expect_expected_output",
        "expected_augment_tag",
    ),
    [
        ("code_summary", "solution", "problem", False, False, False, None),
        ("hints/docs", "trace", "solution", True, True, True, "augment:hinted"),
        ("hints/tests", "trace", "solution", True, True, True, "augment:hinted"),
        ("hints/stubs", "solution", "problem", True, False, False, "augment:stubbed"),
        (
            "issues/iterators",
            "solution",
            "problem",
            True,
            False,
            False,
            "augment:bugged",
        ),
        ("issues/todos", "solution", "problem", True, False, False, "augment:bugged"),
    ],
)
async def test_supported_prompts_prepare_input_variables(
    prompt_name: str,
    identifier_kind: str,
    group_kind: str,
    expect_description: bool,
    expect_inputs: bool,
    expect_expected_output: bool,
    expected_augment_tag: str | None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    dataset, sid, pid = _make_dataset()
    partial_vars = {"target_word_count": 20} if prompt_name == "code_summary" else None
    captured = await _run_prompt_case(
        prompt_name,
        dataset=dataset,
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        partial_vars=partial_vars,
        prepopulate_summary=prompt_name != "code_summary",
    )

    report = typing.cast("annotator.AnnotationReport", captured.pop("report"))
    assert report.total_samples == 1
    assert report.skipped_samples == 0
    assert report.errors == 0, report.error_messages
    assert report.new_results_generated == 1

    trace = dataset[0]
    assert trace.identifier is not None
    trace_id = du.TraceIdentifier.from_string(str(trace.identifier))
    expected_solution_id = str(trace_id.get_parent_identifier())
    expected_problem_id = str(dataset.get_problem_data(0).problem_id)

    if identifier_kind == "solution":
        assert captured["identifier"] == expected_solution_id
    elif identifier_kind == "trace":
        assert captured["identifier"] == str(trace.identifier)
    else:
        raise AssertionError(f"unsupported identifier kind '{identifier_kind}' in test setup")

    if group_kind == "problem":
        assert captured["group"] == expected_problem_id
    elif group_kind == "solution":
        assert captured["group"] == expected_solution_id
    else:
        raise AssertionError(f"unsupported group kind '{group_kind}' in test setup")

    input_vars = captured["input_variables"]
    assert input_vars["code"] == trace.code_string
    if expect_description:
        assert input_vars["description"] == "Existing summary"
    else:
        assert "description" not in input_vars
    if expect_inputs:
        assert input_vars["inputs"] == str(trace.inputs)
    else:
        assert "inputs" not in input_vars
    if expect_expected_output:
        assert input_vars["expected_output"] == str(trace.expected_output)
    else:
        assert "expected_output" not in input_vars
    # we don't expect the bugged hinted token to be here for any of the covered cases
    # (it would show up only when prompting for hints, and if the db+options are set)
    assert annotator._INTERNAL_BUGGED_HINTED_TOKEN not in input_vars

    tags = captured["tags"]
    assert "llm_provider:openai" in tags
    if prompt_name == "code_summary":
        assert "target_summary_word_count:20" in tags
    else:
        assert "augment:has_code_description" in tags
    if expected_augment_tag is None:
        assert not any(tag.startswith("augment:") for tag in tags if tag != "augment:has_code_description")
    else:
        assert expected_augment_tag in tags

    assert captured["prompt_name"] == prompt_name
    meta = captured["meta"]
    assert meta["prompt_config"]["prompt_name"] == prompt_name
    assert meta["llm_provider_config"]["provider"] == "openai"


@pytest.mark.asyncio
async def test_bugged_hint_prompt_uses_buggy_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    dataset, _, _ = _make_dataset()
    trace = dataset[0]
    assert trace.identifier is not None
    trace_id = du.TraceIdentifier.from_string(str(trace.identifier))
    solution_id = trace_id.get_parent_identifier()
    problem = dataset.get_problem_data(0)

    monkeypatch.setattr(np.random, "random", lambda: 0.0)
    monkeypatch.setattr(annotator.random, "choice", lambda seq: seq[0])

    augment_cfg = annotator.AugmentedAnnotationOptions(
        buggy_code_before_hinting_prob_map={"issues/iterators": 1.0},
    )
    buggy_code = "def buggy_add(a, b): return a - b"

    captured = await _run_prompt_case(
        "hints/docs",
        dataset=dataset,
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        augment_config=augment_cfg,
        prepopulate_db_records=[
            {
                "identifier": str(solution_id),
                "prompt": "buggy",
                "result": buggy_code,
                "meta": {},
                "tags": [],
                "group": str(problem.problem_id),
                "prompt_name": "issues/iterators",
                "prompt_version": None,
            }
        ],
    )

    input_vars = captured["input_variables"]
    assert input_vars["code"] == buggy_code
    assert input_vars[annotator._INTERNAL_BUGGED_HINTED_TOKEN] == trace.code_string
    assert input_vars["inputs"] == str(trace.inputs)
    assert input_vars["expected_output"] == str(trace.expected_output)
    tags = captured["tags"]
    assert "augment:has_code_description" in tags
    assert "augment:bugged_hinted" in tags
    assert captured["prompt_name"] == "hints/docs"


@pytest.mark.asyncio
async def test_bugged_misleading_prompt_uses_buggy_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    dataset, _, _ = _make_dataset()
    trace = dataset[0]
    assert trace.identifier is not None
    trace_id = du.TraceIdentifier.from_string(str(trace.identifier))
    solution_id = trace_id.get_parent_identifier()
    problem = dataset.get_problem_data(0)

    monkeypatch.setattr(np.random, "random", lambda: 0.0)
    monkeypatch.setattr(annotator.random, "choice", lambda seq: seq[0])

    augment_cfg = annotator.AugmentedAnnotationOptions(
        buggy_code_before_hinting_prob_map={"issues/iterators": 1.0},
        misleading_augment_prob=1.0,
    )

    class _StubTestCache:
        def sample_alternative_test_case(
            self,
            *,
            trace_id: du.TraceIdentifier,
            **_: typing.Any,
        ) -> types.SimpleNamespace:
            return types.SimpleNamespace(
                outputs={"alt": "value"},
                test_idx=trace_id.test_idx + 1,
            )

    buggy_code = "def buggy_add(a, b): return a - b"

    def _identifier_resolver(
        trace: exec_utils.TraceResult,
        _: du.CodingProblem,
        __: annotator.AnnotationOptions,
    ) -> str:
        assert trace.identifier is not None
        return str(trace.identifier)

    def _group_resolver(
        trace: exec_utils.TraceResult,
        _: du.CodingProblem,
        __: annotator.AnnotationOptions,
    ) -> str:
        assert trace.identifier is not None
        parent_id = du.TraceIdentifier.from_string(str(trace.identifier)).get_parent_identifier()
        return str(parent_id)

    captured = await _run_prompt_case(
        "issues/docs",
        dataset=dataset,
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        augment_config=augment_cfg,
        test_cache=_StubTestCache(),
        identifier_resolver=_identifier_resolver,
        group_resolver=_group_resolver,
        prepopulate_db_records=[
            {
                "identifier": str(solution_id),
                "prompt": "buggy",
                "result": buggy_code,
                "meta": {},
                "tags": [],
                "group": str(problem.problem_id),
                "prompt_name": "issues/iterators",
                "prompt_version": None,
            }
        ],
    )
    report = typing.cast("annotator.AnnotationReport", captured.pop("report"))
    assert report.total_samples == 1
    assert report.skipped_samples == 0
    assert report.errors == 0, report.error_messages
    assert report.new_results_generated == 1

    input_vars = captured["input_variables"]
    assert input_vars["code"] == buggy_code
    assert input_vars[annotator._INTERNAL_BUGGED_HINTED_TOKEN] == trace.code_string
    assert input_vars[annotator._INTERNAL_MISLEADING_TOKEN] == str(trace.expected_output)
    assert input_vars["expected_output"] == str({"alt": "value"})
    tags = captured["tags"]
    assert "augment:has_code_description" in tags
    assert "augment:misleading" not in tags
    assert "augment:bugged_misleading" in tags
    assert captured["prompt_name"] == "issues/docs"


@pytest.mark.asyncio
async def test_misleading_issue_prompt_rewrites_expected_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    dataset, _, _ = _make_dataset()

    def _identifier_resolver(
        trace: exec_utils.TraceResult,
        _: du.CodingProblem,
        __: annotator.AnnotationOptions,
    ) -> str:
        assert trace.identifier is not None
        return str(trace.identifier)

    def _group_resolver(
        trace: exec_utils.TraceResult,
        _: du.CodingProblem,
        __: annotator.AnnotationOptions,
    ) -> str:
        assert trace.identifier is not None
        trace_id = du.TraceIdentifier.from_string(str(trace.identifier))
        return str(trace_id.get_parent_identifier())

    class _StubTestCache:
        def sample_alternative_test_case(
            self,
            *,
            trace_id: du.TraceIdentifier,
            **_: typing.Any,
        ) -> types.SimpleNamespace:
            return types.SimpleNamespace(
                outputs={"alt": "value"},
                test_idx=trace_id.test_idx + 1,
            )

    captured = await _run_prompt_case(
        "issues/docs",
        dataset=dataset,
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        identifier_resolver=_identifier_resolver,
        group_resolver=_group_resolver,
        test_cache=_StubTestCache(),
    )

    report = typing.cast("annotator.AnnotationReport", captured.pop("report"))
    assert report.total_samples == 1
    assert report.skipped_samples == 0
    assert report.errors == 0, report.error_messages
    assert report.new_results_generated == 1

    trace = dataset[0]
    assert trace.identifier is not None
    trace_id = du.TraceIdentifier.from_string(str(trace.identifier))
    expected_solution_id = str(trace_id.get_parent_identifier())

    assert captured["identifier"] == str(trace.identifier)
    assert captured["group"] == expected_solution_id
    assert captured["prompt_name"] == "issues/docs"

    input_vars = captured["input_variables"]
    assert input_vars["description"] == "Existing summary"
    assert input_vars["inputs"] == str(trace.inputs)
    assert input_vars["expected_output"] == str({"alt": "value"})
    assert input_vars[annotator._INTERNAL_MISLEADING_TOKEN] == str(trace.expected_output)
    assert input_vars["code"] == trace.code_string

    tags = captured["tags"]
    assert "augment:has_code_description" in tags
    assert "augment:misleading" in tags
    assert "augment:bugged" not in tags
    assert "llm_provider:openai" in tags

    meta = captured["meta"]
    assert meta["prompt_config"]["prompt_name"] == "issues/docs"
    assert meta["llm_provider_config"]["provider"] == "openai"


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.dataset
@pytest.mark.openai
@pytest.mark.asyncio
@pytest.mark.skipif(
    tests.env_checks.TACO_TRACES_DATASET_MISSING,
    reason="TACO traces dataset is missing, cannot run annotator integration",
)
@pytest.mark.skipif(
    tests.env_checks.OPENAI_API_KEY_MISSING or tests.env_checks.NETWORK_UNAVAILABLE,
    reason="OpenAI API key or network not available; cannot run annotator integration",
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
        llm_provider_config={
            "provider": "openai",
            "model": "gpt-4o-mini",
        },
        prompt_config=prompt_types.PromptBuildConfig(
            prompt_name="code_summary",
            partial_vars={
                "target_word_count": 30,
            },
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
        llm_provider_config={
            "provider": "openai",
            "model": "gpt-4o-mini",
        },
        prompt_config=prompt_types.PromptBuildConfig(
            prompt_name="hints/stubs",
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
    assert temp_db.list_prompt_names() == ["code_summary", "hints/stubs"]
