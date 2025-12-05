import asyncio
import collections
import typing

import pytest

import pyine.evals.code_exec.configs
import pyine.evals.code_exec.evaluator
import pyine.evals.common
import pyine.evals.configs
import pyine.prompts.manager
import pyine.utils.llm_providers
import tests.env_checks
import tests.hydra_test_utils


class _DummyGraderChain:
    def __init__(
        self,
        scorer: collections.abc.Callable[[str, str], float],
    ) -> None:
        self._scorer = scorer

    def invoke(
        self,
        predicted: typing.Any,
        expected: typing.Any,
        predict_type: str = "unknown",
        **invoke_kwargs: typing.Any,
    ) -> float:
        return float(self._scorer(expected, predicted))

    async def ainvoke(
        self,
        predicted: typing.Any,
        expected: typing.Any,
        predict_type: str = "unknown",
        **invoke_kwargs: typing.Any,
    ) -> float:
        return await asyncio.to_thread(
            self.invoke,
            predicted=predicted,
            expected=expected,
            predict_type=predict_type,
            **invoke_kwargs,
        )


def test_add_sample_and_accuracies_no_grader() -> None:
    evaluator = pyine.evals.code_exec.evaluator.OutcomeEvaluator()
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
    evaluator = pyine.evals.code_exec.evaluator.OutcomeEvaluator()
    evaluator._llm_grader_chain_config = _DummyGraderChain(  # simple grader = 1.0 on exact, 0.25 otherwise
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
    evaluator = pyine.evals.code_exec.evaluator.OutcomeEvaluator()
    evaluator._llm_grader_chain_config = _DummyGraderChain(
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
    evaluator = pyine.evals.code_exec.evaluator.OutcomeEvaluator()
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
    evaluator = pyine.evals.code_exec.evaluator.OutcomeEvaluator()
    evaluator.add_sample(identifier="foo/good", expected="1", predicted="1", tags=["x"])
    evaluator.add_sample(identifier="bar/bad", expected="1", predicted="2", tags=["y"])

    def selector(sid: str) -> bool:
        return sid.endswith("good")

    hard_acc = evaluator.compute_hard_accuracy(identifier_selector=selector)
    soft_acc = evaluator.compute_soft_accuracy(identifier_selector=selector)
    assert hard_acc == 1.0
    assert soft_acc == 1.0


def test_strip_hard_checks_behavior() -> None:
    evaluator_strip = pyine.evals.code_exec.evaluator.OutcomeEvaluator()
    evaluator_strip.add_sample(identifier="s", expected="answer", predicted=" answer ")
    assert evaluator_strip.compute_hard_accuracy() == 1.0


def test_tags_include_exec_type() -> None:
    evaluator = pyine.evals.code_exec.evaluator.OutcomeEvaluator()
    evaluator.add_sample(identifier="s1", expected="answer", predicted="answer", tags=["x"])
    assert len(evaluator.results) == 1 and evaluator.results[0].tags == ["x", "sample_predict_type:unknown"]
    evaluator.add_sample(identifier="s2", expected="answer", predicted="answer", predict_type="potato", tags=["x"])
    assert len(evaluator.results) == 2 and evaluator.results[1].tags == ["x", "sample_predict_type:potato"]


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.openai
@pytest.mark.skipif(
    tests.env_checks.OPENAI_API_KEY_MISSING or tests.env_checks.NETWORK_UNAVAILABLE,
    reason="OpenAI API key or network not available; cannot run OpenAI-backed evaluation.",
)
async def test_outcome_evaluator_large_batch_base_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configs = pyine.evals.configs.get_evals_configs(
        eval_type=pyine.evals.common.EvalType.CODE_EXEC,
        group="tests_evals",
    )
    target_config = next(cfg for cfg in configs if cfg.name == "base")
    assert target_config is not None
    with tests.hydra_test_utils.instantiate_from_defaults_with_launch(
        base_configs=[c for c in configs if c is not target_config],
        target_config=target_config,
    ) as evals_config:
        assert isinstance(evals_config, pyine.evals.code_exec.configs.CodeExecEvalsConfig)
        assert evals_config.evaluator_kwargs is not None, "evaluator_kwargs should have been set"
        evaluator = pyine.evals.code_exec.evaluator.OutcomeEvaluator(**evals_config.evaluator_kwargs)
    sample_count = 500
    for idx in range(sample_count):
        expected = f"value-{idx}"
        if idx % 10 == 0:
            predicted = "something-irrelevant"
        elif idx % 3 == 0:
            predicted = f" {expected} "
        else:
            predicted = expected
        evaluator.add_sample(
            identifier=f"sample-{idx}",
            expected=expected,
            predicted=predicted,
            tags=[f"bucket:{idx % 5}"],
        )
    assert len(evaluator.results) == sample_count
    hard_acc = evaluator.compute_hard_accuracy()
    soft_acc = evaluator.compute_soft_accuracy()
    grader_acc = await evaluator.compute_grader_accuracy()
    agreement = await evaluator.compute_agreement_table()
    assert hard_acc == pytest.approx(0.9)
    assert soft_acc == pytest.approx(0.9)
    assert grader_acc > 0.85
    assert agreement["hard_vs_soft"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_compute_category_wise_metrics_no_grader() -> None:
    evaluator = pyine.evals.code_exec.evaluator.OutcomeEvaluator()
    evaluator.add_sample(identifier="s1", expected="42", predicted="42", tags=["easy"])
    evaluator.add_sample(identifier="s2", expected="abc", predicted="xyz", tags=["hard"])
    evaluator.add_sample(identifier="s3", expected="1.0", predicted="1", tags=["easy"])  # soft match only
    evaluator.add_sample(identifier="s4", expected="ok", predicted="ok", tags=["hard"])
    category_to_identifiers = {
        "code_type/original": ["s1", "s2"],
        "code_type/obfuscated": ["s3", "s4"],
        "predict_type/program_output": ["s1", "s2", "s3"],
        "predict_type/frame_variables": ["s4"],
    }
    category_metrics = await evaluator.compute_category_wise_metrics(category_to_identifiers)
    # code_type/original: s1 (hard=1, soft=1), s2 (hard=0, soft=0) -> 0.5, 0.5
    assert "code_type/original" in category_metrics
    assert category_metrics["code_type/original"]["accuracy_hard"] == pytest.approx(0.5)
    assert category_metrics["code_type/original"]["accuracy_soft"] == pytest.approx(0.5)
    assert category_metrics["code_type/original"]["sample_count"] == 2
    # code_type/obfuscated: s3 (hard=0, soft=1), s4 (hard=1, soft=1) -> 0.5, 1.0
    assert "code_type/obfuscated" in category_metrics
    assert category_metrics["code_type/obfuscated"]["accuracy_hard"] == pytest.approx(0.5)
    assert category_metrics["code_type/obfuscated"]["accuracy_soft"] == pytest.approx(1.0)
    assert category_metrics["code_type/obfuscated"]["sample_count"] == 2
    # predict_type/program_output: s1, s2, s3
    assert "predict_type/program_output" in category_metrics
    assert category_metrics["predict_type/program_output"]["sample_count"] == 3
    # predict_type/frame_variables: s4 only
    assert "predict_type/frame_variables" in category_metrics
    assert category_metrics["predict_type/frame_variables"]["sample_count"] == 1
    assert category_metrics["predict_type/frame_variables"]["accuracy_hard"] == 1.0


@pytest.mark.asyncio
async def test_compute_category_wise_metrics_with_grader() -> None:
    evaluator = pyine.evals.code_exec.evaluator.OutcomeEvaluator()
    evaluator._llm_grader_chain_config = _DummyGraderChain(
        scorer=lambda exp, pred: 1.0 if exp.strip() == pred.strip() else 0.0
    )
    evaluator.add_sample(identifier="s1", expected="42", predicted="42")
    evaluator.add_sample(identifier="s2", expected="abc", predicted="xyz")
    category_to_identifiers = {
        "cat_a": ["s1", "s2"],
    }
    category_metrics = await evaluator.compute_category_wise_metrics(category_to_identifiers)
    assert "cat_a" in category_metrics
    assert category_metrics["cat_a"]["accuracy_hard"] == pytest.approx(0.5)
    assert category_metrics["cat_a"]["accuracy_soft"] == pytest.approx(0.5)
    assert category_metrics["cat_a"]["accuracy_grader"] == pytest.approx(0.5)
    assert category_metrics["cat_a"]["sample_count"] == 2


@pytest.mark.asyncio
async def test_compute_category_wise_metrics_empty_categories() -> None:
    evaluator = pyine.evals.code_exec.evaluator.OutcomeEvaluator()
    evaluator.add_sample(identifier="s1", expected="42", predicted="42")
    # empty mapping should return empty result
    category_metrics = await evaluator.compute_category_wise_metrics({})
    assert category_metrics == {}
    # unknown identifiers result in category with empty metrics dict
    category_metrics = await evaluator.compute_category_wise_metrics({"cat1": ["unknown_id"]})
    assert category_metrics == {"cat1": {}}


@pytest.mark.asyncio
async def test_compute_category_wise_metrics_overlapping_categories() -> None:
    evaluator = pyine.evals.code_exec.evaluator.OutcomeEvaluator()
    evaluator.add_sample(identifier="s1", expected="42", predicted="42")
    # sample belongs to multiple categories
    category_to_identifiers = {"cat_a": ["s1"], "cat_b": ["s1"], "cat_c": ["s1"]}
    category_metrics = await evaluator.compute_category_wise_metrics(category_to_identifiers)
    assert len(category_metrics) == 3
    for cat in ["cat_a", "cat_b", "cat_c"]:
        assert category_metrics[cat]["accuracy_hard"] == 1.0
        assert category_metrics[cat]["sample_count"] == 1


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.openai
@pytest.mark.skipif(
    tests.env_checks.OPENAI_API_KEY_MISSING or tests.env_checks.NETWORK_UNAVAILABLE,
    reason="OpenAI API key or network not available; cannot run OpenAI-backed evaluation.",
)
async def test_real_llm_grade_scoring() -> None:
    evaluator = pyine.evals.code_exec.evaluator.OutcomeEvaluator(
        llm_provider_config=pyine.utils.llm_providers.LLMProviderConfig(
            provider="openai",
            model_kwargs={
                "model": "gpt-4o-mini",
            },
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
    assert metrics["accuracy_hard"] == pytest.approx(0.5)
    assert metrics["accuracy_soft"] == pytest.approx(0.5)
    assert metrics["accuracy_grader"] >= 0.5
    agreement = await evaluator.compute_agreement_table()
    assert agreement["hard_vs_soft"] == pytest.approx(1.0)
