"""Tests for pyine.evals.code_exec.evaluator.OutcomeEvaluator.

This module tests the core evaluation logic for code execution predictions,
including accuracy computation, LLM grading, and category-wise metrics.
"""

from __future__ import annotations

import pytest

import pyine.evals.code_exec.configs
import pyine.evals.code_exec.evaluator
import pyine.evals.code_exec.utils
import pyine.evals.common
import pyine.evals.configs
import pyine.utils.llm_providers
import tests.env_checks
import tests.hydra_test_utils

from .conftest import (
    DEFAULT_GRADER_THRESHOLD,
    EXACT_MATCH_SCORE,
    NO_MATCH_SCORE,
    MockGraderChain,
)


class TestOutcomeEvaluatorBasics:
    """Tests for basic OutcomeEvaluator operations: add_sample, accuracies."""

    def test_add_sample_stores_evaluation_result(
        self,
        base_evaluator: pyine.evals.code_exec.evaluator.OutcomeEvaluator,
    ) -> None:
        """Verify add_sample stores the evaluation result with correct identifier."""
        base_evaluator.add_sample(identifier="test-id", expected="42", predicted="42", tags=["tag1"])
        assert base_evaluator.get_sample_count() == 1
        assert len(base_evaluator.results) == 1
        assert base_evaluator.results[0].identifier == "test-id"

    def test_hard_accuracy_exact_match_only(
        self,
        evaluator_with_standard_samples: pyine.evals.code_exec.evaluator.OutcomeEvaluator,
    ) -> None:
        """Hard accuracy counts only exact string matches (after strip)."""
        hard_acc = evaluator_with_standard_samples.compute_hard_accuracy()
        assert isinstance(hard_acc, float)
        assert 0.0 <= hard_acc <= 1.0
        # s1: exact match, s2: mismatch, s3: soft match only -> 1/3
        assert hard_acc == pytest.approx(1 / 3)

    def test_soft_accuracy_includes_normalized_matches(
        self,
        evaluator_with_standard_samples: pyine.evals.code_exec.evaluator.OutcomeEvaluator,
    ) -> None:
        """Soft accuracy includes semantically equivalent matches (e.g., '1.0' vs '1')."""
        soft_acc = evaluator_with_standard_samples.compute_soft_accuracy()
        assert isinstance(soft_acc, float)
        assert 0.0 <= soft_acc <= 1.0
        # s1: match, s2: mismatch, s3: soft match -> 2/3
        assert soft_acc == pytest.approx(2 / 3)

    def test_strip_normalization_applied(
        self,
        base_evaluator: pyine.evals.code_exec.evaluator.OutcomeEvaluator,
    ) -> None:
        """Whitespace is stripped before comparison for hard accuracy."""
        base_evaluator.add_sample(identifier="s", expected="answer", predicted=" answer ")
        assert base_evaluator.compute_hard_accuracy() == 1.0

    def test_tags_include_predict_type_suffix(
        self,
        base_evaluator: pyine.evals.code_exec.evaluator.OutcomeEvaluator,
    ) -> None:
        """Tags automatically include sample_predict_type suffix."""
        base_evaluator.add_sample(identifier="s1", expected="answer", predicted="answer", tags=["x"])
        assert base_evaluator.results[0].tags == ["x", "sample_predict_type:unknown"]
        base_evaluator.add_sample(
            identifier="s2", expected="answer", predicted="answer", predict_type="custom", tags=["y"]
        )
        assert base_evaluator.results[1].tags == ["y", "sample_predict_type:custom"]

    def test_get_sample_count_returns_correct_count(
        self,
        base_evaluator: pyine.evals.code_exec.evaluator.OutcomeEvaluator,
    ) -> None:
        """get_sample_count returns the number of added samples."""
        assert base_evaluator.get_sample_count() == 0
        base_evaluator.add_sample(identifier="s1", expected="a", predicted="a")
        assert base_evaluator.get_sample_count() == 1
        base_evaluator.add_sample(identifier="s2", expected="b", predicted="b")
        assert base_evaluator.get_sample_count() == 2


class TestOutcomeEvaluatorBatch:
    """Tests for batch operations."""

    def test_add_batch_success_adds_all_samples(
        self,
        base_evaluator: pyine.evals.code_exec.evaluator.OutcomeEvaluator,
    ) -> None:
        """add_batch successfully adds multiple samples at once."""
        base_evaluator.add_batch(
            identifiers=["b1", "b2", "b3"],
            expected_list=["exp1", "exp2", "exp3"],
            predicted_list=["pred1", "pred2", "pred3"],
            tags=[["t1"], ["t2"], ["t3"]],
        )
        assert base_evaluator.get_sample_count() == 3
        assert [r.identifier for r in base_evaluator.results] == ["b1", "b2", "b3"]

    @pytest.mark.parametrize(
        ("identifiers", "expected_list", "predicted_list", "tags"),
        [
            (["i1", "i2"], ["a"], ["a", "b"], None),  # mismatched expected length
            (["i1", "i2"], ["a", "b"], ["a"], None),  # mismatched predicted length
            (["i1", "i2"], ["a", "b"], ["a", "b"], [["x"]]),  # wrong tags length
        ],
        ids=["mismatched_expected", "mismatched_predicted", "wrong_tags_length"],
    )
    def test_add_batch_validation_errors(
        self,
        base_evaluator: pyine.evals.code_exec.evaluator.OutcomeEvaluator,
        identifiers: list[str],
        expected_list: list[str],
        predicted_list: list[str],
        tags: list[list[str]] | None,
    ) -> None:
        """add_batch raises ValueError for mismatched list lengths."""
        with pytest.raises(ValueError):
            base_evaluator.add_batch(
                identifiers=identifiers,
                expected_list=expected_list,
                predicted_list=predicted_list,
                tags=tags,
            )


class TestOutcomeEvaluatorFiltering:
    """Tests for identifier and tag filtering."""

    def test_identifier_selector_filters_samples(
        self,
        base_evaluator: pyine.evals.code_exec.evaluator.OutcomeEvaluator,
    ) -> None:
        """identifier_selector filters which samples are included in accuracy."""
        base_evaluator.add_sample(identifier="foo/good", expected="1", predicted="1", tags=["x"])
        base_evaluator.add_sample(identifier="bar/bad", expected="1", predicted="2", tags=["y"])

        def selector(sid: str) -> bool:
            return sid.endswith("good")

        hard_acc = base_evaluator.compute_hard_accuracy(identifier_selector=selector)
        soft_acc = base_evaluator.compute_soft_accuracy(identifier_selector=selector)
        assert hard_acc == 1.0
        assert soft_acc == 1.0


class TestOutcomeEvaluatorWithGrader:
    """Tests for LLM grader integration."""

    @pytest.mark.asyncio
    async def test_grader_accuracy_with_threshold(
        self,
        base_evaluator: pyine.evals.code_exec.evaluator.OutcomeEvaluator,
        partial_match_grader: MockGraderChain,
    ) -> None:
        """Grader accuracy applies threshold to determine correct/incorrect."""
        base_evaluator._llm_grader_chain_config = partial_match_grader
        base_evaluator.add_sample(identifier="g1", expected="42", predicted="42")  # score=1.0
        base_evaluator.add_sample(identifier="g2", expected="10", predicted=" 10 ")  # score=1.0
        base_evaluator.add_sample(identifier="g3", expected="yes", predicted="no")  # score=0.25
        # threshold 0.5: first two correct, last incorrect -> 2/3
        grader_acc = await base_evaluator.compute_grader_accuracy(score_threshold=0.5)
        assert grader_acc == pytest.approx(2 / 3)

    @pytest.mark.asyncio
    async def test_agreement_table_all_agree(
        self,
        base_evaluator: pyine.evals.code_exec.evaluator.OutcomeEvaluator,
        exact_match_grader: MockGraderChain,
    ) -> None:
        """Agreement table shows 1.0 when all evaluators agree."""
        base_evaluator._llm_grader_chain_config = exact_match_grader
        base_evaluator.add_sample(identifier="a1", expected="ok", predicted="ok", tags=["agree"])
        table = await base_evaluator.compute_agreement_table(score_threshold=DEFAULT_GRADER_THRESHOLD)
        assert set(table.keys()) == {"hard_vs_soft", "hard_vs_grader", "soft_vs_grader"}
        assert table["hard_vs_soft"] == 1.0
        assert table["hard_vs_grader"] == 1.0
        assert table["soft_vs_grader"] == 1.0

    @pytest.mark.asyncio
    async def test_agreement_table_soft_differs(
        self,
        base_evaluator: pyine.evals.code_exec.evaluator.OutcomeEvaluator,
        exact_match_grader: MockGraderChain,
    ) -> None:
        """Agreement table reflects when soft match differs from hard/grader."""
        base_evaluator._llm_grader_chain_config = exact_match_grader
        base_evaluator.add_sample(identifier="a1", expected="ok", predicted="ok")
        # dict reordering: soft matches, but hard/grader don't
        base_evaluator.add_sample(identifier="a2", expected="{'a': 1, 'b': 2}", predicted="{'b': 2, 'a': 1}")
        table = await base_evaluator.compute_agreement_table(score_threshold=DEFAULT_GRADER_THRESHOLD)
        assert table["hard_vs_soft"] == 0.5
        assert table["hard_vs_grader"] == 1.0
        assert table["soft_vs_grader"] == 0.5

    def test_llm_grader_available_false_without_config(
        self,
        base_evaluator: pyine.evals.code_exec.evaluator.OutcomeEvaluator,
    ) -> None:
        """llm_grader_available is False when no grader chain is configured."""
        assert base_evaluator.llm_grader_available is False

    def test_llm_grader_available_true_with_config(
        self,
        evaluator_with_grader: pyine.evals.code_exec.evaluator.OutcomeEvaluator,
    ) -> None:
        """llm_grader_available is True when grader chain is configured."""
        assert evaluator_with_grader.llm_grader_available is True


class TestOutcomeEvaluatorMetrics:
    """Tests for metrics computation (compute_metrics, compute_grader_metrics)."""

    @pytest.mark.asyncio
    async def test_compute_metrics_aggregates_all_types(
        self,
        evaluator_with_grader_and_samples: pyine.evals.code_exec.evaluator.OutcomeEvaluator,
    ) -> None:
        """compute_metrics returns dict with hard, soft, grader accuracies and stats."""
        metrics = await evaluator_with_grader_and_samples.compute_metrics()
        # check accuracy metrics present
        assert "accuracy_hard" in metrics
        assert "accuracy_soft" in metrics
        assert "accuracy_grader" in metrics
        assert "sample_count" in metrics
        assert "attempt_count" in metrics
        # check grader stats present
        assert "grader_mean" in metrics
        assert "grader_median" in metrics
        assert "grader_std" in metrics
        assert "grader_min" in metrics
        assert "grader_max" in metrics
        # verify values are reasonable
        assert 0.0 <= metrics["accuracy_hard"] <= 1.0
        assert 0.0 <= metrics["accuracy_soft"] <= 1.0
        assert 0.0 <= metrics["accuracy_grader"] <= 1.0
        assert metrics["sample_count"] == 3
        assert metrics["attempt_count"] == 3

    @pytest.mark.asyncio
    async def test_compute_grader_metrics_returns_statistics(
        self,
        evaluator_with_grader_and_samples: pyine.evals.code_exec.evaluator.OutcomeEvaluator,
    ) -> None:
        """compute_grader_metrics returns dict with mean, median, std, min, max."""
        grader_metrics = await evaluator_with_grader_and_samples.compute_grader_metrics()
        assert "grader_mean" in grader_metrics
        assert "grader_median" in grader_metrics
        assert "grader_std" in grader_metrics
        assert "grader_min" in grader_metrics
        assert "grader_max" in grader_metrics
        # with exact match grader: s1=1.0, s2=0.0, s3=0.0
        assert grader_metrics["grader_mean"] == pytest.approx(1 / 3, rel=0.01)
        assert grader_metrics["grader_min"] == pytest.approx(NO_MATCH_SCORE)
        assert grader_metrics["grader_max"] == pytest.approx(EXACT_MATCH_SCORE)

    @pytest.mark.asyncio
    async def test_compute_metrics_without_grader(
        self,
        evaluator_with_standard_samples: pyine.evals.code_exec.evaluator.OutcomeEvaluator,
    ) -> None:
        """compute_metrics works without grader (grader metrics are NaN or skipped)."""
        metrics = await evaluator_with_standard_samples.compute_metrics()
        assert "accuracy_hard" in metrics
        assert "accuracy_soft" in metrics
        assert metrics["sample_count"] == 3
        # without grader, accuracy_grader should not be present or be NaN
        # (depends on implementation)


class TestOutcomeEvaluatorCategoryMetrics:
    """Tests for category-wise metrics computation."""

    @pytest.mark.asyncio
    async def test_category_metrics_no_grader(
        self,
        base_evaluator: pyine.evals.code_exec.evaluator.OutcomeEvaluator,
    ) -> None:
        """Category metrics computed correctly without LLM grader."""
        base_evaluator.add_sample(identifier="s1", expected="42", predicted="42", tags=["easy"])
        base_evaluator.add_sample(identifier="s2", expected="abc", predicted="xyz", tags=["hard"])
        base_evaluator.add_sample(identifier="s3", expected="1.0", predicted="1", tags=["easy"])
        base_evaluator.add_sample(identifier="s4", expected="ok", predicted="ok", tags=["hard"])
        category_to_identifiers = {
            "code_type/original": ["s1", "s2"],
            "code_type/obfuscated": ["s3", "s4"],
        }
        category_metrics = await base_evaluator.compute_category_wise_metrics(category_to_identifiers)
        # code_type/original: s1 match, s2 mismatch -> 0.5, 0.5
        assert category_metrics["code_type/original"]["accuracy_hard"] == pytest.approx(0.5)
        assert category_metrics["code_type/original"]["accuracy_soft"] == pytest.approx(0.5)
        assert category_metrics["code_type/original"]["sample_count"] == 2
        # code_type/obfuscated: s3 soft only, s4 match -> 0.5, 1.0
        assert category_metrics["code_type/obfuscated"]["accuracy_hard"] == pytest.approx(0.5)
        assert category_metrics["code_type/obfuscated"]["accuracy_soft"] == pytest.approx(1.0)
        assert category_metrics["code_type/obfuscated"]["sample_count"] == 2

    @pytest.mark.asyncio
    async def test_category_metrics_with_grader(
        self,
        base_evaluator: pyine.evals.code_exec.evaluator.OutcomeEvaluator,
        exact_match_grader: MockGraderChain,
    ) -> None:
        """Category metrics include grader accuracy when grader is available."""
        base_evaluator._llm_grader_chain_config = exact_match_grader
        base_evaluator.add_sample(identifier="s1", expected="42", predicted="42")
        base_evaluator.add_sample(identifier="s2", expected="abc", predicted="xyz")
        category_metrics = await base_evaluator.compute_category_wise_metrics({"cat_a": ["s1", "s2"]})
        assert category_metrics["cat_a"]["accuracy_hard"] == pytest.approx(0.5)
        assert category_metrics["cat_a"]["accuracy_soft"] == pytest.approx(0.5)
        assert category_metrics["cat_a"]["accuracy_grader"] == pytest.approx(0.5)
        assert category_metrics["cat_a"]["sample_count"] == 2

    @pytest.mark.asyncio
    async def test_empty_category_mapping_returns_empty(
        self,
        evaluator_with_standard_samples: pyine.evals.code_exec.evaluator.OutcomeEvaluator,
    ) -> None:
        """Empty category mapping returns empty result dict."""
        category_metrics = await evaluator_with_standard_samples.compute_category_wise_metrics({})
        assert category_metrics == {}

    @pytest.mark.asyncio
    async def test_unknown_identifiers_return_empty_metrics(
        self,
        evaluator_with_standard_samples: pyine.evals.code_exec.evaluator.OutcomeEvaluator,
    ) -> None:
        """Category with unknown identifiers returns empty metrics dict."""
        category_metrics = await evaluator_with_standard_samples.compute_category_wise_metrics({"cat1": ["unknown_id"]})
        assert category_metrics == {"cat1": {}}

    @pytest.mark.asyncio
    async def test_overlapping_categories_computed_independently(
        self,
        base_evaluator: pyine.evals.code_exec.evaluator.OutcomeEvaluator,
    ) -> None:
        """Same sample in multiple categories is counted in each."""
        base_evaluator.add_sample(identifier="s1", expected="42", predicted="42")
        category_to_identifiers = {"cat_a": ["s1"], "cat_b": ["s1"], "cat_c": ["s1"]}
        category_metrics = await base_evaluator.compute_category_wise_metrics(category_to_identifiers)
        assert len(category_metrics) == 3
        for cat in ["cat_a", "cat_b", "cat_c"]:
            assert category_metrics[cat]["accuracy_hard"] == 1.0
            assert category_metrics[cat]["sample_count"] == 1


class TestOutcomeEvaluatorLLMScoreAccess:
    """Tests for LLM score access patterns and error handling."""

    @pytest.mark.asyncio
    async def test_llm_score_access_before_ready_raises(self) -> None:
        """Accessing llm_score before gathering raises RuntimeError."""
        import asyncio

        import pyine.utils.code.output_compare

        # create a sample eval with a pending future
        async def dummy_score() -> float:
            await asyncio.sleep(0.1)
            return 0.5

        # CompareResult is a dataclass with equal, reason, path attributes
        soft_match_result = pyine.utils.code.output_compare.CompareResult(equal=False, reason="test")
        sample_eval = pyine.evals.code_exec.utils.SampleEval(
            identifier="test",
            expected="a",
            predicted="b",
            hard_match=False,
            soft_match=soft_match_result,
            _llm_score=asyncio.create_task(dummy_score()),
            tags=[],
        )
        with pytest.raises(RuntimeError, match="attempting to access llm score before it is ready"):
            _ = sample_eval.llm_score
        sample_eval._llm_score.cancel()


class TestGetSampleGroups:
    """Tests for get_sample_groups validation logic."""

    def test_duplicate_attempt_indices_raises(
        self,
        base_evaluator: pyine.evals.code_exec.evaluator.OutcomeEvaluator,
    ) -> None:
        base_evaluator.add_sample(identifier="s1", expected="a", predicted="a", attempt_index=0)
        base_evaluator.add_sample(identifier="s1", expected="a", predicted="b", attempt_index=0)
        with pytest.raises(ValueError, match="duplicate attempt indices"):
            base_evaluator.get_sample_groups()

    def test_non_contiguous_attempt_indices_raises(
        self,
        base_evaluator: pyine.evals.code_exec.evaluator.OutcomeEvaluator,
    ) -> None:
        base_evaluator.add_sample(identifier="s1", expected="a", predicted="a", attempt_index=0)
        base_evaluator.add_sample(identifier="s1", expected="a", predicted="b", attempt_index=2)
        with pytest.raises(ValueError, match="non-contiguous attempt indices"):
            base_evaluator.get_sample_groups()

    def test_inconsistent_expected_raises(
        self,
        base_evaluator: pyine.evals.code_exec.evaluator.OutcomeEvaluator,
    ) -> None:
        base_evaluator.add_sample(identifier="s1", expected="a", predicted="a", attempt_index=0)
        base_evaluator.add_sample(identifier="s1", expected="b", predicted="b", attempt_index=1)
        with pytest.raises(ValueError, match="inconsistent expected values"):
            base_evaluator.get_sample_groups()

    def test_wrong_attempt_count_raises(
        self,
        base_evaluator: pyine.evals.code_exec.evaluator.OutcomeEvaluator,
    ) -> None:
        base_evaluator.add_sample(identifier="s1", expected="a", predicted="a", attempt_index=0)
        base_evaluator.add_sample(identifier="s1", expected="a", predicted="b", attempt_index=1)
        with pytest.raises(ValueError, match="expected 5"):
            base_evaluator.get_sample_groups(expected_attempts_per_sample=5)

    def test_valid_multi_attempt_groups(
        self,
        base_evaluator: pyine.evals.code_exec.evaluator.OutcomeEvaluator,
    ) -> None:
        for attempt_idx in range(3):
            base_evaluator.add_sample(identifier="s1", expected="a", predicted="a", attempt_index=attempt_idx)
            base_evaluator.add_sample(identifier="s2", expected="b", predicted="b", attempt_index=attempt_idx)
        groups = base_evaluator.get_sample_groups(expected_attempts_per_sample=3)
        assert len(groups) == 2
        assert all(g.num_attempts == 3 for g in groups)


class TestComputeMetricsPassAtK:
    """Tests for compute_metrics Pass@K paths with num_attempts_per_sample > 1."""

    @pytest.mark.asyncio
    async def test_pass_at_k_with_multi_attempt(
        self,
        base_evaluator: pyine.evals.code_exec.evaluator.OutcomeEvaluator,
    ) -> None:
        """Pass@K metrics are computed correctly with K=3 attempts per sample."""
        # sample s1: 3/3 correct, sample s2: 1/3 correct, sample s3: 0/3 correct
        for attempt_idx in range(3):
            base_evaluator.add_sample(identifier="s1", expected="a", predicted="a", attempt_index=attempt_idx)
        base_evaluator.add_sample(identifier="s2", expected="b", predicted="b", attempt_index=0)
        base_evaluator.add_sample(identifier="s2", expected="b", predicted="x", attempt_index=1)
        base_evaluator.add_sample(identifier="s2", expected="b", predicted="y", attempt_index=2)
        for attempt_idx in range(3):
            base_evaluator.add_sample(identifier="s3", expected="c", predicted="z", attempt_index=attempt_idx)
        metrics = await base_evaluator.compute_metrics(
            pass_at_k_values=[1, 3],
            num_attempts_per_sample=3,
        )
        # pass@1 = mean of per-sample fractions: (3/3 + 1/3 + 0/3) / 3 = 4/9
        assert metrics["pass_at_1_hard"] == pytest.approx(4 / 9)
        assert "pass_at_1_hard_ci_lower" in metrics
        assert "pass_at_1_hard_ci_upper" in metrics
        assert "pass_at_1_soft" in metrics
        # pass@3 = mean of per-sample pass@3: (1.0 + 1.0 + 0.0) / 3 = 2/3
        assert metrics["pass_at_3_hard"] == pytest.approx(2 / 3)
        assert "pass_at_3_hard_ci_lower" in metrics
        assert "pass_at_3_hard_ci_upper" in metrics
        # multi-attempt extras
        assert "majority_correct_hard" in metrics
        assert "majority_correct_soft" in metrics
        assert "mean_output_diversity" in metrics
        assert "mean_unique_outputs" in metrics
        # counts
        assert metrics["sample_count"] == 3
        assert metrics["attempt_count"] == 9

    @pytest.mark.asyncio
    async def test_pass_at_k_not_emitted_when_none(
        self,
        evaluator_with_standard_samples: pyine.evals.code_exec.evaluator.OutcomeEvaluator,
    ) -> None:
        """When pass_at_k_values is None, no pass_at_* keys are emitted."""
        metrics = await evaluator_with_standard_samples.compute_metrics(pass_at_k_values=None)
        assert not any(k.startswith("pass_at_") for k in metrics)
        assert not any(k.startswith("majority_correct") for k in metrics)

    @pytest.mark.asyncio
    async def test_pass_at_k_empty_evaluator_emits_degenerate_metrics(
        self,
        base_evaluator: pyine.evals.code_exec.evaluator.OutcomeEvaluator,
    ) -> None:
        """Empty evaluator with Pass@K enabled emits 0.0 point estimates and full-uncertainty CIs."""
        metrics = await base_evaluator.compute_metrics(pass_at_k_values=[1, 3], num_attempts_per_sample=3)
        assert metrics["pass_at_1_hard"] == 0.0
        assert metrics["pass_at_1_hard_ci_lower"] == 0.0
        assert metrics["pass_at_1_hard_ci_upper"] == 1.0
        assert metrics["pass_at_3_soft"] == 0.0
        assert metrics["majority_correct_hard"] == 0.0
        assert metrics["mean_output_diversity"] == 0.0
        assert metrics["mean_unique_outputs"] == 0.0
        assert metrics["sample_count"] == 0
        assert metrics["attempt_count"] == 0

    @pytest.mark.asyncio
    async def test_pass_at_k_filtered_out_subset_emits_degenerate_metrics(
        self,
        base_evaluator: pyine.evals.code_exec.evaluator.OutcomeEvaluator,
    ) -> None:
        """When identifier_selector filters out all samples, degenerate metrics are emitted."""
        for attempt_idx in range(3):
            base_evaluator.add_sample(identifier="s1", expected="a", predicted="a", attempt_index=attempt_idx)
        metrics = await base_evaluator.compute_metrics(
            pass_at_k_values=[1],
            num_attempts_per_sample=3,
            identifier_selector=lambda sid: sid == "nonexistent",
        )
        assert metrics["pass_at_1_hard"] == 0.0
        assert metrics["pass_at_1_hard_ci_lower"] == 0.0
        assert metrics["pass_at_1_hard_ci_upper"] == 1.0
        assert metrics["sample_count"] == 0


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
    """Integration test: large batch with real OpenAI grader from Hydra config."""
    configs = pyine.evals.configs.get_evals_configs(
        eval_type=pyine.evals.common.EvalType.CODE_EXEC,
        group="tests_evals",
    )
    target_config = next(cfg for cfg in configs if cfg.name == "code_exec_base")
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
@pytest.mark.integration
@pytest.mark.openai
@pytest.mark.skipif(
    tests.env_checks.OPENAI_API_KEY_MISSING or tests.env_checks.NETWORK_UNAVAILABLE,
    reason="OpenAI API key or network not available; cannot run OpenAI-backed evaluation.",
)
async def test_real_llm_grade_scoring() -> None:
    """Integration test: real OpenAI grading with compute_metrics."""
    evaluator = pyine.evals.code_exec.evaluator.OutcomeEvaluator(
        llm_provider_config=pyine.utils.llm_providers.LLMProviderConfig(
            provider="openai",
            model_kwargs={"model": "gpt-4o-mini"},
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
