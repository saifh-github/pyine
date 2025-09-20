import asyncio
import collections.abc
import pathlib
import typing

import pytest

import pyine.evals.utils
import pyine.prompts
import pyine.utils.llm_providers
import tests.env_checks


class _DummyGraderChain:

    def __init__(
        self,
        scorer: collections.abc.Callable[[str, str], float],
    ) -> None:
        self._scorer = scorer

    def invoke(
        self,
        data: dict[str, typing.Any],
        *args,
        **kwargs,
    ) -> float:
        expected = typing.cast(str, data["expected_output"])
        predicted = typing.cast(str, data["predicted_output"])
        return float(self._scorer(expected, predicted))

    async def ainvoke(
        self,
        data: dict[str, typing.Any],
        *args,
        **kwargs,
    ) -> float:
        return await asyncio.to_thread(self.invoke, data, *args, **kwargs)


def test_add_sample_and_accuracies_no_grader() -> None:
    evaluator = pyine.evals.utils.OutcomeEvaluator(strip_hard_checks=True)
    evaluator.add_sample(identifier="s1", expected="42", predicted="42", tags=["easy"])
    evaluator.add_sample(identifier="s2", expected="abc", predicted="xyz", tags=["hard"])
    evaluator.add_sample(identifier="s3", expected="1.0", predicted="1", tags=["woops"])
    hard_acc = evaluator.compute_hard_accuracy()
    soft_acc = evaluator.compute_soft_accuracy()
    assert isinstance(hard_acc, float) and 0.0 <= hard_acc <= 1.0
    assert isinstance(soft_acc, float) and 0.0 <= soft_acc <= 1.0
    assert hard_acc == pytest.approx(1 / 3)  # one correct, two incorrect (with strip on both)
    assert soft_acc == pytest.approx(2 / 3)  # soft compare should fix the last case, but not the 2nd one


@pytest.mark.asyncio
async def test_grader_accuracy_with_mock_chain() -> None:
    evaluator = pyine.evals.utils.OutcomeEvaluator(strip_hard_checks=True)
    evaluator._llm_grader_chain = _DummyGraderChain(  # simple grader = 1.0 on exact, 0.25 otherwise
        scorer=lambda exp, pred: 1.0 if exp.strip() == pred.strip() else 0.25
    )
    evaluator.add_sample(identifier="g1", expected="42", predicted="42")
    evaluator.add_sample(identifier="g2", expected="10", predicted=" 10 ")
    evaluator.add_sample(identifier="g3", expected="yes", predicted="no")
    # default threshold 0.5 should mark first two as correct (1.0), last as incorrect (0.25)
    grader_acc = await evaluator.compute_grader_accuracy(score_threshold=0.5)
    assert grader_acc == pytest.approx(2 / 3)


@pytest.mark.asyncio
async def test_agreement_table_with_mock_chain() -> None:
    evaluator = pyine.evals.utils.OutcomeEvaluator(strip_hard_checks=True)
    evaluator._llm_grader_chain = _DummyGraderChain(
        scorer=lambda exp, pred: 1.0 if exp.strip() == pred.strip() else 0.0
    )
    # single fully-agreeing sample (hard == soft == True, grader >= 0.5)
    evaluator.add_sample(identifier="a1", expected="ok", predicted="ok", tags=["agree"])
    table = await evaluator.compute_agreement_table(score_threshold=0.5)
    assert set(table.keys()) == {"hard_vs_soft", "hard_vs_grader", "soft_vs_grader"}
    # with one agreeing sample, all entries should be 1.0
    assert table["hard_vs_soft"] == 1.0
    assert table["hard_vs_grader"] == 1.0
    assert table["soft_vs_grader"] == 1.0
    # add one more case where only soft match differs
    evaluator.add_sample(identifier="a2", expected="{'a': 1, 'b': 2}", predicted="{'b': 2, 'a': 1}")
    table = await evaluator.compute_agreement_table(score_threshold=0.5)
    assert table["hard_vs_soft"] == 0.5
    assert table["hard_vs_grader"] == 1.0
    assert table["soft_vs_grader"] == 0.5


def test_add_batch_validation_errors() -> None:
    evaluator = pyine.evals.utils.OutcomeEvaluator()
    with pytest.raises(ValueError):
        evaluator.add_batch(
            identifiers=["i1", "i2"],
            expected_list=["a"],
            predicted_list=["a", "b"],
        )
    with pytest.raises(ValueError):
        evaluator.add_batch(
            identifiers=["i1", "i2"],
            expected_list=["a", "b"],
            predicted_list=["a", "b"],
            tags=[["x"]],  # wrong length
        )


def test_identifier_selector_filtering() -> None:
    evaluator = pyine.evals.utils.OutcomeEvaluator(strip_hard_checks=True)
    evaluator.add_sample(identifier="foo/good", expected="1", predicted="1", tags=["x"])
    evaluator.add_sample(identifier="bar/bad", expected="1", predicted="2", tags=["y"])

    def selector(sid: str) -> bool:
        return sid.endswith("good")

    hard_acc = evaluator.compute_hard_accuracy(identifier_selector=selector)
    soft_acc = evaluator.compute_soft_accuracy(identifier_selector=selector)
    assert hard_acc == 1.0
    assert soft_acc == 1.0


def test_strip_hard_checks_behavior() -> None:
    evaluator_no_strip = pyine.evals.utils.OutcomeEvaluator(strip_hard_checks=False)
    evaluator_no_strip.add_sample(identifier="ns", expected="answer", predicted=" answer ")
    assert evaluator_no_strip.compute_hard_accuracy() == 0.0
    evaluator_strip = pyine.evals.utils.OutcomeEvaluator(strip_hard_checks=True)
    evaluator_strip.add_sample(identifier="s", expected="answer", predicted=" answer ")
    assert evaluator_strip.compute_hard_accuracy() == 1.0


@pytest.mark.asyncio
@pytest.mark.skipif(
    tests.env_checks.OPENAI_API_KEY_MISSING or tests.env_checks.NETWORK_UNAVAILABLE,
    reason="OpenAI API key or network not available; cannot run OpenAI-backed evaluation.",
)
async def test_real_llm_grade_scoring() -> None:
    evaluator = pyine.evals.utils.OutcomeEvaluator(
        llm_provider_config=pyine.utils.llm_providers.LLMProviderConfig(
            provider="openai",
            model_kwargs=dict(
                model="gpt-4o-mini",
            ),
        ),
    )
    evaluator.add_sample(
        identifier="id1",
        expected="Hello, world!",
        predicted="Hello, world!",
        tags=["easy"],
    )
    evaluator.add_sample(
        identifier="id2",
        expected="[1.2, 3, 5.6]",
        predicted="'[1.20, 3.00, 5.9999]'",
        tags=["hard"],
    )
    metrics = await evaluator.compute_metrics()
    assert metrics["accuracy/hard"] == pytest.approx(0.5)
    assert metrics["accuracy/soft"] == pytest.approx(0.5)
    assert metrics["accuracy/grader"] >= 0.5
    agreement = await evaluator.compute_agreement_table()
    assert agreement["hard_vs_soft"] == pytest.approx(1.0)


def test_parse_usage_from_dict_basic() -> None:
    # Basic dict response with usage section and common fields
    response = {
        "usage": {
            "total_tokens": 100,
            "prompt_tokens": "unknown",
            "completion_tokens": 10,
            "reasoning_tokens": 30,
        }
    }
    info = pyine.evals.utils.parse_token_usage_from_response(response)
    assert isinstance(info, pyine.evals.utils.TokenUsageInfo)
    assert info.total_tokens == 100
    assert info.prompt_tokens == "unknown"
    assert info.completion_tokens == 10
    assert info.reasoning_tokens == 30


def test_parse_usage_from_attr_object() -> None:
    class _Usage:
        total_tokens = 12
        prompt_tokens = 5
        completion_tokens = 7

    obj = type("ObjWithUsage", (), {})()
    obj.usage = _Usage()
    info = pyine.evals.utils.parse_token_usage_from_response(obj)
    assert info.total_tokens == 12
    assert info.prompt_tokens == 5
    assert info.completion_tokens == 7
    assert info.reasoning_tokens == "unknown"
    # thinking_tokens alias should populate reasoning_tokens when present
    obj2 = type("ObjWithUsage2", (), {})()
    obj2.usage = type("U", (), {"thinking_tokens": 3})()
    info2 = pyine.evals.utils.parse_token_usage_from_response(obj2)
    assert info2.reasoning_tokens == 3


def test_parse_usage_from_nested_creation_meta_llm_output() -> None:
    class _Wrapper:
        pass

    wrapper = _Wrapper()
    wrapper.creation_meta = {
        "llm_output": {
            "usage": {
                "total_tokens": 33,
            }
        }
    }
    info = pyine.evals.utils.parse_token_usage_from_response(wrapper)
    assert info.total_tokens == 33


def test_parse_usage_raises_when_no_information() -> None:
    with pytest.raises(ValueError, match="could not deduce token usage information"):
        _ = pyine.evals.utils.parse_token_usage_from_response({})


def test_token_usage_info_add_and_iadd_success() -> None:
    r1 = {"usage": {"total_tokens": 10, "prompt_tokens": 6, "completion_tokens": 4}}
    r2 = {"usage": {"total_tokens": 20, "prompt_tokens": 3, "completion_tokens": 7}}
    info1 = pyine.evals.utils.parse_token_usage_from_response(r1)
    info2 = pyine.evals.utils.parse_token_usage_from_response(r2)
    summed = info1 + info2
    assert summed.total_tokens == 30
    assert summed.prompt_tokens == 9
    assert summed.completion_tokens == 11
    assert summed.reasoning_tokens == "unknown"

    info1 += info2
    assert info1.total_tokens == 30
    assert info1.prompt_tokens == 9
    assert info1.completion_tokens == 11
    assert info1.reasoning_tokens == "unknown"

    known_output = {"usage": {"cached_tokens": 5, "total_tokens": 5}}
    unknown_output = {"usage": {"total_tokens": 10}}  # output_tokens omitted -> 'unknown'
    info_known = pyine.evals.utils.parse_token_usage_from_response(known_output)
    info_unknown = pyine.evals.utils.parse_token_usage_from_response(unknown_output)
    with pytest.raises(ValueError):
        _ = info_known + info_unknown
    with pytest.raises(ValueError):
        info_unknown += info_known


@pytest.mark.skipif(
    tests.env_checks.OPENAI_API_KEY_MISSING or tests.env_checks.NETWORK_UNAVAILABLE,
    reason="OpenAI API key or network not available; cannot run OpenAI-backed evaluation.",
)
def test_token_usage_with_real_openai_generation(tmp_path: pathlib.Path):
    model = pyine.utils.llm_providers.LLMProviderConfig(
        provider="openai",
        model_kwargs=dict(
            model="gpt-4o-mini",
        ),
    ).get_model()
    prompt_config = pyine.prompts.PromptBuildConfig(
        prompt_name="code_summary",
        partial_vars=dict(
            target_word_count=30,
        ),
    )
    code_snippet = """\
def f(x):
    iters = 0
    while x > 0:
        x //= 10
        iters += 1
    return iters
"""
    records = pyine.prompts.fetch_or_generate_prompt_results(
        model=model,
        identifier="potato",
        input_variables={"code": code_snippet},
        prompt_config=prompt_config,
        db=pyine.prompts.PromptResultDB(pathlib.Path(tmp_path) / "prompt_results.sqlite"),
        log_new_results=False,
    )
    assert len(records) == 1
    info = pyine.evals.utils.parse_token_usage_from_response(records[0])
    assert info.total_tokens != "unknown" and info.total_tokens > 0
