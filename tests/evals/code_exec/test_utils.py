import asyncio
import collections
import typing

import pytest

import pyine.evals.code_exec.utils
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
        expected = typing.cast("str", data["expected_output"])
        predicted = typing.cast("str", data["predicted_output"])
        return float(self._scorer(expected, predicted))

    async def ainvoke(
        self,
        data: dict[str, typing.Any],
        *args,
        **kwargs,
    ) -> float:
        return await asyncio.to_thread(self.invoke, data, *args, **kwargs)


def test_add_sample_and_accuracies_no_grader() -> None:
    evaluator = pyine.evals.code_exec.utils.OutcomeEvaluator(strip_hard_checks=True)
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
    evaluator = pyine.evals.code_exec.utils.OutcomeEvaluator(strip_hard_checks=True)
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
    evaluator = pyine.evals.code_exec.utils.OutcomeEvaluator(strip_hard_checks=True)
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
    evaluator = pyine.evals.code_exec.utils.OutcomeEvaluator()
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
    evaluator = pyine.evals.code_exec.utils.OutcomeEvaluator(strip_hard_checks=True)
    evaluator.add_sample(identifier="foo/good", expected="1", predicted="1", tags=["x"])
    evaluator.add_sample(identifier="bar/bad", expected="1", predicted="2", tags=["y"])

    def selector(sid: str) -> bool:
        return sid.endswith("good")

    hard_acc = evaluator.compute_hard_accuracy(identifier_selector=selector)
    soft_acc = evaluator.compute_soft_accuracy(identifier_selector=selector)
    assert hard_acc == 1.0
    assert soft_acc == 1.0


def test_strip_hard_checks_behavior() -> None:
    evaluator_no_strip = pyine.evals.code_exec.utils.OutcomeEvaluator(strip_hard_checks=False)
    evaluator_no_strip.add_sample(identifier="ns", expected="answer", predicted=" answer ")
    assert evaluator_no_strip.compute_hard_accuracy() == 0.0
    evaluator_strip = pyine.evals.code_exec.utils.OutcomeEvaluator(strip_hard_checks=True)
    evaluator_strip.add_sample(identifier="s", expected="answer", predicted=" answer ")
    assert evaluator_strip.compute_hard_accuracy() == 1.0


@pytest.mark.asyncio
@pytest.mark.skipif(
    tests.env_checks.OPENAI_API_KEY_MISSING or tests.env_checks.NETWORK_UNAVAILABLE,
    reason="OpenAI API key or network not available; cannot run OpenAI-backed evaluation.",
)
async def test_real_llm_grade_scoring() -> None:
    evaluator = pyine.evals.code_exec.utils.OutcomeEvaluator(
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
