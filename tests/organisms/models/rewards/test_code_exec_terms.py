"""Tests for the code execution reward terms (hard_match, soft_match, llm_grader)."""

import typing

import pytest

import pyine.organisms.models.rewards.core.configs as reward_configs
import pyine.organisms.models.rewards.core.types as reward_types
import pyine.organisms.models.rewards.terms.code_exec.hard_match as hard_match_term
import pyine.organisms.models.rewards.terms.code_exec.llm_grader as llm_grader_term
import pyine.organisms.models.rewards.terms.code_exec.soft_match as soft_match_term
import pyine.organisms.models.rewards.terms.code_exec.utils as code_exec_utils
import pyine.utils.code.output_compare
import tests.organisms.models.rewards.conftest as rewards_conftest


def make_code_exec_sample_context(
    *,
    expected: str,
    predicted: str,
    predict_type: str = "output",
    llm_grader_score: float | None = None,
    hard_match_result: bool | None = None,
    soft_match_result: bool | None = None,
    should_flip_reward: bool | None = None,
    prompt: str = "test prompt",
    model_output: str = "test output",
    identifier: str = "test_sample",
    code_type: str = "original",
    comma_separated_tags: str = "",
) -> reward_types.SampleContext:
    """Build a SampleContext with code execution evaluation data for testing."""
    eval_data = reward_types.CodeExecEvalData(
        expected=expected,
        predicted=predicted,
        predict_type=predict_type,
        llm_grader_score=llm_grader_score,
        hard_match_result=hard_match_result,
        soft_match_result=soft_match_result,
        should_flip_reward=should_flip_reward,
    )
    return reward_types.SampleContext(
        prompt=prompt,
        model_output=model_output,
        sample_data=rewards_conftest.make_sample_data(
            identifier,
            code_type=code_type,
            comma_separated_tags=comma_separated_tags,
        ),
        code_exec_eval=eval_data,
    )


class TestCodeExecEvalData:
    def test_default_values(self) -> None:
        data = reward_types.CodeExecEvalData(expected="42", predicted="42")
        assert data.expected == "42"
        assert data.predicted == "42"
        assert data.predict_type == "unknown"
        assert data.llm_grader_score is None

    def test_custom_values(self) -> None:
        data = reward_types.CodeExecEvalData(
            expected="hello",
            predicted="world",
            predict_type="output",
            llm_grader_score=0.75,
        )
        assert data.expected == "hello"
        assert data.predicted == "world"
        assert data.predict_type == "output"
        assert data.llm_grader_score == 0.75


class TestCodeExecUtilsHelpers:
    def test_get_code_exec_eval_data_returns_data_when_present(self) -> None:
        ctx = make_code_exec_sample_context(expected="42", predicted="42")
        data = code_exec_utils.get_code_exec_eval_data(ctx)
        assert data is not None
        assert data.expected == "42"

    def test_get_code_exec_eval_data_returns_none_when_missing(self) -> None:
        ctx = rewards_conftest.make_sample_context()
        data = code_exec_utils.get_code_exec_eval_data(ctx)
        assert data is None

    def test_require_code_exec_eval_data_raises_when_missing(self) -> None:
        ctx = rewards_conftest.make_sample_context()
        with pytest.raises(ValueError, match="requires sample_ctx.code_exec_eval"):
            code_exec_utils.require_code_exec_eval_data(ctx, "TestTerm")

    def test_compute_hard_match_exact_match(self) -> None:
        assert code_exec_utils.compute_hard_match("42", "42") is True

    def test_compute_hard_match_with_whitespace_stripping(self) -> None:
        assert code_exec_utils.compute_hard_match("  42  ", "42") is True
        assert code_exec_utils.compute_hard_match("42", "  42  ") is True

    def test_compute_hard_match_without_whitespace_stripping(self) -> None:
        assert code_exec_utils.compute_hard_match("  42  ", "42", strip_whitespace=False) is False

    def test_compute_hard_match_different_values(self) -> None:
        assert code_exec_utils.compute_hard_match("42", "43") is False

    def test_compute_soft_match_exact_match(self) -> None:
        result = code_exec_utils.compute_soft_match("42", "42")
        assert result.equal is True

    def test_compute_soft_match_numeric_tolerance(self) -> None:
        result = code_exec_utils.compute_soft_match("3.14159", "3.1416")
        assert result.equal is True

    def test_compute_soft_match_list_comparison(self) -> None:
        result = code_exec_utils.compute_soft_match("[1, 2, 3]", "[1, 2, 3]")
        assert result.equal is True

    def test_compute_soft_match_different_values(self) -> None:
        result = code_exec_utils.compute_soft_match("hello", "world")
        assert result.equal is False


class TestHardMatchTermConfig:
    def test_default_config_values(self) -> None:
        config = hard_match_term.HardMatchTermConfig()
        assert config.reward_if_match == 1.0
        assert config.reward_if_no_match == 0.0
        assert config.strip_whitespace is True

    def test_custom_config_values(self) -> None:
        config = hard_match_term.HardMatchTermConfig(
            reward_if_match=2.0,
            reward_if_no_match=0.5,
            strip_whitespace=False,
        )
        assert config.reward_if_match == 2.0
        assert config.reward_if_no_match == 0.5
        assert config.strip_whitespace is False

    def test_negative_reward_rejected(self) -> None:
        with pytest.raises(ValueError):
            hard_match_term.HardMatchTermConfig(reward_if_match=-1.0)


class TestHardMatchTerm:
    def test_returns_reward_if_match_on_exact_match(self) -> None:
        config = hard_match_term.HardMatchTermConfig(reward_if_match=1.0, reward_if_no_match=0.0)
        term = hard_match_term.HardMatchTerm(config)
        ctx = make_code_exec_sample_context(expected="42", predicted="42")
        result = term(ctx)
        assert result.value == 1.0
        assert result.metrics["hard_match"] is True

    def test_returns_reward_if_no_match_on_mismatch(self) -> None:
        config = hard_match_term.HardMatchTermConfig(reward_if_match=1.0, reward_if_no_match=0.5)
        term = hard_match_term.HardMatchTerm(config)
        ctx = make_code_exec_sample_context(expected="42", predicted="43")
        result = term(ctx)
        assert result.value == 0.5
        assert result.metrics["hard_match"] is False

    def test_whitespace_stripping_enabled_by_default(self) -> None:
        config = hard_match_term.HardMatchTermConfig()
        term = hard_match_term.HardMatchTerm(config)
        ctx = make_code_exec_sample_context(expected="  42  ", predicted="42")
        result = term(ctx)
        assert result.value == 1.0
        assert result.metrics["hard_match"] is True

    def test_whitespace_stripping_can_be_disabled(self) -> None:
        config = hard_match_term.HardMatchTermConfig(strip_whitespace=False)
        term = hard_match_term.HardMatchTerm(config)
        ctx = make_code_exec_sample_context(expected="  42  ", predicted="42")
        result = term(ctx)
        assert result.value == 0.0
        assert result.metrics["hard_match"] is False

    def test_metrics_include_lengths(self) -> None:
        config = hard_match_term.HardMatchTermConfig()
        term = hard_match_term.HardMatchTerm(config)
        ctx = make_code_exec_sample_context(expected="hello", predicted="world")
        result = term(ctx)
        assert result.metrics["expected_length"] == 5
        assert result.metrics["predicted_length"] == 5

    def test_raises_when_eval_data_missing(self) -> None:
        config = hard_match_term.HardMatchTermConfig()
        term = hard_match_term.HardMatchTerm(config)
        ctx = rewards_conftest.make_sample_context()
        with pytest.raises(ValueError, match="HardMatchTerm requires"):
            term(ctx)

    def test_reset_is_noop(self) -> None:
        config = hard_match_term.HardMatchTermConfig()
        term = hard_match_term.HardMatchTerm(config)
        run_ctx = reward_types.RunInitContext()
        term.reset(run_ctx)  # should not raise

    def test_uses_precomputed_hard_match_result_when_available(self) -> None:
        config = hard_match_term.HardMatchTermConfig()
        term = hard_match_term.HardMatchTerm(config)
        # expected != predicted, but pre-computed says they match
        eval_data = reward_types.CodeExecEvalData(
            expected="42",
            predicted="43",
            hard_match_result=True,
        )
        ctx = reward_types.SampleContext(
            prompt="test",
            model_output="test",
            sample_data=rewards_conftest.make_sample_data("test"),
            code_exec_eval=eval_data,
        )
        result = term(ctx)
        assert result.value == 1.0
        assert result.metrics["hard_match"] is True
        assert result.metrics["used_precomputed"] is True

    def test_raises_when_precomputed_hard_match_result_has_invalid_type(self) -> None:
        config = hard_match_term.HardMatchTermConfig()
        term = hard_match_term.HardMatchTerm(config)
        eval_data = reward_types.CodeExecEvalData(
            expected="42",
            predicted="43",
            hard_match_result=typing.cast("typing.Any", "yes"),
        )
        ctx = reward_types.SampleContext(
            prompt="test",
            model_output="test",
            sample_data=rewards_conftest.make_sample_data("test"),
            code_exec_eval=eval_data,
        )
        with pytest.raises(TypeError, match="hard_match_result"):
            term(ctx)

    def test_computes_hard_match_when_precomputed_not_available(self) -> None:
        config = hard_match_term.HardMatchTermConfig()
        term = hard_match_term.HardMatchTerm(config)
        ctx = make_code_exec_sample_context(expected="42", predicted="42")
        result = term(ctx)
        assert result.value == 1.0
        assert result.metrics["hard_match"] is True
        assert result.metrics["used_precomputed"] is False


class TestHardMatchTermFactory:
    def test_factory_creates_term_from_spec(self) -> None:
        spec = reward_configs.RewardTermSpec(
            name="test",
            type="hard_match",
            params={
                "reward_if_match": 2.0,
                "reward_if_no_match": 0.1,
                "strip_whitespace": False,
            },
        )
        term = hard_match_term._factory(spec, parser=None)
        assert isinstance(term, hard_match_term.HardMatchTerm)
        ctx_mismatch = make_code_exec_sample_context(expected="  42  ", predicted="42")
        mismatch_result = term(ctx_mismatch)
        assert mismatch_result.value == pytest.approx(0.1)
        assert mismatch_result.metrics["hard_match"] is False
        ctx_match = make_code_exec_sample_context(expected="42", predicted="42")
        match_result = term(ctx_match)
        assert match_result.value == pytest.approx(2.0)
        assert match_result.metrics["hard_match"] is True

    def test_factory_validates_params(self) -> None:
        spec = reward_configs.RewardTermSpec(
            name="test",
            type="hard_match",
            params={"reward_if_match": -1.0},
        )
        with pytest.raises(ValueError):
            hard_match_term._factory(spec, parser=None)


class TestHardMatchTermRegistration:
    def test_term_is_registered(self) -> None:
        import pyine.organisms.models.rewards.core.registry as reward_registry
        import pyine.organisms.models.rewards.terms

        pyine.organisms.models.rewards.terms.ensure_builtin_terms_registered()
        factory = reward_registry.get_term_factory("hard_match")
        assert factory is not None

    def test_alias_is_registered(self) -> None:
        import pyine.organisms.models.rewards.core.registry as reward_registry
        import pyine.organisms.models.rewards.terms

        pyine.organisms.models.rewards.terms.ensure_builtin_terms_registered()
        canonical = reward_registry.get_term_factory("hard_match")
        alias = reward_registry.get_term_factory("code_exec/hard_match")
        assert alias is canonical
        assert reward_registry.get_global_registry().resolve_term_type("code_exec/hard_match") == "hard_match"


class TestSoftMatchTermConfig:
    def test_default_config_values(self) -> None:
        config = soft_match_term.SoftMatchTermConfig()
        assert config.reward_if_match == 1.0
        assert config.reward_if_no_match == 0.0
        # compare_options should use defaults from get_default_comparison_config
        default_opts = pyine.utils.code.output_compare.get_default_comparison_config()
        assert config.compare_options.rel_tol == default_opts.rel_tol
        assert config.compare_options.abs_tol == default_opts.abs_tol
        assert config.compare_options.normalize_whitespace == default_opts.normalize_whitespace
        assert config.compare_options.case_sensitive == default_opts.case_sensitive
        assert config.compare_options.array_type_matters == default_opts.array_type_matters

    def test_custom_config_values(self) -> None:
        config = soft_match_term.SoftMatchTermConfig(
            reward_if_match=2.0,
            compare_options=pyine.utils.code.output_compare.CompareOptions(
                rel_tol=1e-5,
                abs_tol=1e-8,
                case_sensitive=False,
            ),
        )
        assert config.reward_if_match == 2.0
        assert config.compare_options.rel_tol == 1e-5
        assert config.compare_options.abs_tol == 1e-8
        assert config.compare_options.case_sensitive is False

    def test_negative_tolerance_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            soft_match_term.SoftMatchTermConfig(
                compare_options=pyine.utils.code.output_compare.CompareOptions(rel_tol=-1e-5),
            )


class TestSoftMatchTerm:
    def test_returns_reward_if_match_on_exact_match(self) -> None:
        config = soft_match_term.SoftMatchTermConfig(reward_if_match=1.0, reward_if_no_match=0.0)
        term = soft_match_term.SoftMatchTerm(config)
        ctx = make_code_exec_sample_context(expected="42", predicted="42")
        result = term(ctx)
        assert result.value == 1.0
        assert result.metrics["soft_match"] is True

    def test_returns_reward_if_no_match_on_mismatch(self) -> None:
        config = soft_match_term.SoftMatchTermConfig(reward_if_match=1.0, reward_if_no_match=0.5)
        term = soft_match_term.SoftMatchTerm(config)
        ctx = make_code_exec_sample_context(expected="hello", predicted="world")
        result = term(ctx)
        assert result.value == 0.5
        assert result.metrics["soft_match"] is False

    def test_numeric_tolerance_matching(self) -> None:
        config = soft_match_term.SoftMatchTermConfig()
        term = soft_match_term.SoftMatchTerm(config)
        ctx = make_code_exec_sample_context(expected="3.14159265", predicted="3.14159")
        result = term(ctx)
        assert result.value == 1.0
        assert result.metrics["soft_match"] is True

    def test_list_comparison(self) -> None:
        config = soft_match_term.SoftMatchTermConfig()
        term = soft_match_term.SoftMatchTerm(config)
        ctx = make_code_exec_sample_context(expected="[1, 2, 3]", predicted="[1, 2, 3]")
        result = term(ctx)
        assert result.value == 1.0
        assert result.metrics["soft_match"] is True

    def test_list_tuple_equivalence_when_type_does_not_matter(self) -> None:
        config = soft_match_term.SoftMatchTermConfig(
            compare_options=pyine.utils.code.output_compare.CompareOptions(array_type_matters=False),
        )
        term = soft_match_term.SoftMatchTerm(config)
        ctx = make_code_exec_sample_context(expected="[1, 2, 3]", predicted="(1, 2, 3)")
        result = term(ctx)
        assert result.value == 1.0
        assert result.metrics["soft_match"] is True

    def test_whitespace_normalization(self) -> None:
        config = soft_match_term.SoftMatchTermConfig()
        term = soft_match_term.SoftMatchTerm(config)
        ctx = make_code_exec_sample_context(expected="hello   world", predicted="hello world")
        result = term(ctx)
        assert result.value == 1.0
        assert result.metrics["soft_match"] is True

    def test_case_insensitive_matching(self) -> None:
        config = soft_match_term.SoftMatchTermConfig(
            compare_options=pyine.utils.code.output_compare.CompareOptions(case_sensitive=False),
        )
        term = soft_match_term.SoftMatchTerm(config)
        ctx = make_code_exec_sample_context(expected="HELLO", predicted="hello")
        result = term(ctx)
        assert result.value == 1.0
        assert result.metrics["soft_match"] is True

    def test_metrics_include_lengths(self) -> None:
        config = soft_match_term.SoftMatchTermConfig()
        term = soft_match_term.SoftMatchTerm(config)
        ctx = make_code_exec_sample_context(expected="test", predicted="value")
        result = term(ctx)
        assert result.metrics["expected_length"] == 4
        assert result.metrics["predicted_length"] == 5

    def test_raises_when_eval_data_missing(self) -> None:
        config = soft_match_term.SoftMatchTermConfig()
        term = soft_match_term.SoftMatchTerm(config)
        ctx = rewards_conftest.make_sample_context()
        with pytest.raises(ValueError, match="SoftMatchTerm requires"):
            term(ctx)

    def test_reset_is_noop(self) -> None:
        config = soft_match_term.SoftMatchTermConfig()
        term = soft_match_term.SoftMatchTerm(config)
        run_ctx = reward_types.RunInitContext()
        term.reset(run_ctx)

    def test_mismatch_includes_reason_metric(self) -> None:
        """Mismatch includes mismatch_reason metric for debugging."""
        config = soft_match_term.SoftMatchTermConfig()
        term = soft_match_term.SoftMatchTerm(config)
        ctx = make_code_exec_sample_context(expected="hello", predicted="world")
        result = term(ctx)
        assert result.value == 0.0
        assert result.metrics["soft_match"] is False
        assert "mismatch_reason" in result.metrics

    def test_match_does_not_include_reason_metric(self) -> None:
        """Successful match does not include mismatch_reason metric."""
        config = soft_match_term.SoftMatchTermConfig()
        term = soft_match_term.SoftMatchTerm(config)
        ctx = make_code_exec_sample_context(expected="42", predicted="42")
        result = term(ctx)
        assert result.value == 1.0
        assert result.metrics["soft_match"] is True
        assert "mismatch_reason" not in result.metrics

    def test_uses_precomputed_soft_match_result_when_available(self) -> None:
        config = soft_match_term.SoftMatchTermConfig()
        term = soft_match_term.SoftMatchTerm(config)
        # expected != predicted, but pre-computed says they match
        eval_data = reward_types.CodeExecEvalData(
            expected="hello",
            predicted="world",
            soft_match_result=True,
        )
        ctx = reward_types.SampleContext(
            prompt="test",
            model_output="test",
            sample_data=rewards_conftest.make_sample_data("test"),
            code_exec_eval=eval_data,
        )
        result = term(ctx)
        assert result.value == 1.0
        assert result.metrics["soft_match"] is True
        assert result.metrics["used_precomputed"] is True
        assert "mismatch_reason" not in result.metrics  # no reason when precomputed

    def test_raises_when_precomputed_soft_match_result_has_invalid_type(self) -> None:
        config = soft_match_term.SoftMatchTermConfig()
        term = soft_match_term.SoftMatchTerm(config)
        eval_data = reward_types.CodeExecEvalData(
            expected="42",
            predicted="43",
            soft_match_result=typing.cast("typing.Any", "yes"),
        )
        ctx = reward_types.SampleContext(
            prompt="test",
            model_output="test",
            sample_data=rewards_conftest.make_sample_data("test"),
            code_exec_eval=eval_data,
        )
        with pytest.raises(TypeError, match="soft_match_result"):
            term(ctx)

    def test_computes_soft_match_when_precomputed_not_available(self) -> None:
        config = soft_match_term.SoftMatchTermConfig()
        term = soft_match_term.SoftMatchTerm(config)
        ctx = make_code_exec_sample_context(expected="42", predicted="42")
        result = term(ctx)
        assert result.value == 1.0
        assert result.metrics["soft_match"] is True
        assert result.metrics["used_precomputed"] is False


class TestSoftMatchTermFactory:
    def test_factory_creates_term_from_spec(self) -> None:
        spec = reward_configs.RewardTermSpec(
            name="test",
            type="soft_match",
            params={
                "reward_if_match": 2.0,
                "reward_if_no_match": 0.25,
                "compare_options": {"case_sensitive": False},
            },
        )
        term = soft_match_term._factory(spec, parser=None)
        assert isinstance(term, soft_match_term.SoftMatchTerm)
        ctx = make_code_exec_sample_context(expected="Hello", predicted="hello")
        result = term(ctx)
        assert result.value == pytest.approx(2.0)
        assert result.metrics["soft_match"] is True


class TestSoftMatchTermRegistration:
    def test_term_is_registered(self) -> None:
        import pyine.organisms.models.rewards.core.registry as reward_registry
        import pyine.organisms.models.rewards.terms

        pyine.organisms.models.rewards.terms.ensure_builtin_terms_registered()
        factory = reward_registry.get_term_factory("soft_match")
        assert factory is not None

    def test_alias_is_registered(self) -> None:
        import pyine.organisms.models.rewards.core.registry as reward_registry
        import pyine.organisms.models.rewards.terms

        pyine.organisms.models.rewards.terms.ensure_builtin_terms_registered()
        canonical = reward_registry.get_term_factory("soft_match")
        alias = reward_registry.get_term_factory("code_exec/soft_match")
        assert alias is canonical
        assert reward_registry.get_global_registry().resolve_term_type("code_exec/soft_match") == "soft_match"


class TestLLMGraderTermConfig:
    def test_default_config_values(self) -> None:
        config = llm_grader_term.LLMGraderTermConfig()
        assert config.reward_if_match == 1.0
        assert config.reward_if_no_match == 0.0
        assert config.score_threshold == 0.5
        assert config.use_continuous_reward is False
        assert config.fallback_to_soft_match is False
        assert config.fallback_to_hard_match is False

    def test_custom_config_values(self) -> None:
        config = llm_grader_term.LLMGraderTermConfig(
            reward_if_match=2.0,
            score_threshold=0.7,
            use_continuous_reward=True,
            fallback_to_soft_match=False,
            fallback_to_hard_match=True,
        )
        assert config.reward_if_match == 2.0
        assert config.score_threshold == 0.7
        assert config.use_continuous_reward is True
        assert config.fallback_to_soft_match is False
        assert config.fallback_to_hard_match is True

    def test_score_threshold_bounds(self) -> None:
        with pytest.raises(ValueError):
            llm_grader_term.LLMGraderTermConfig(score_threshold=-0.1)
        with pytest.raises(ValueError):
            llm_grader_term.LLMGraderTermConfig(score_threshold=1.1)


class TestLLMGraderTerm:
    def test_returns_reward_if_match_when_score_above_threshold(self) -> None:
        config = llm_grader_term.LLMGraderTermConfig(
            reward_if_match=1.0,
            reward_if_no_match=0.0,
            score_threshold=0.5,
        )
        term = llm_grader_term.LLMGraderTerm(config)
        ctx = make_code_exec_sample_context(expected="42", predicted="42", llm_grader_score=0.9)
        result = term(ctx)
        assert result.value == 1.0
        assert result.metrics["llm_grader_match"] is True
        assert result.metrics["llm_grader_score"] == 0.9
        assert result.metrics["llm_grader_available"] is True
        assert result.metrics["used_fallback"] is False

    @pytest.mark.parametrize(
        ("llm_grader_score", "exc_type"),
        [
            (-0.1, ValueError),
            (1.1, ValueError),
            (float("nan"), ValueError),
            (typing.cast("typing.Any", "0.5"), TypeError),
        ],
    )
    def test_raises_when_llm_grader_score_is_invalid(
        self,
        llm_grader_score: float,
        exc_type: type[Exception],
    ) -> None:
        config = llm_grader_term.LLMGraderTermConfig(
            reward_if_match=1.0,
            reward_if_no_match=0.0,
            score_threshold=0.5,
        )
        term = llm_grader_term.LLMGraderTerm(config)
        ctx = make_code_exec_sample_context(expected="42", predicted="42", llm_grader_score=llm_grader_score)
        with pytest.raises(exc_type, match="llm_grader_score"):
            term(ctx)

    def test_returns_reward_if_no_match_when_score_below_threshold(self) -> None:
        config = llm_grader_term.LLMGraderTermConfig(
            reward_if_match=1.0,
            reward_if_no_match=0.0,
            score_threshold=0.5,
        )
        term = llm_grader_term.LLMGraderTerm(config)
        ctx = make_code_exec_sample_context(expected="42", predicted="43", llm_grader_score=0.2)
        result = term(ctx)
        assert result.value == 0.0
        assert result.metrics["llm_grader_match"] is False

    def test_exact_threshold_is_match(self) -> None:
        config = llm_grader_term.LLMGraderTermConfig(score_threshold=0.5)
        term = llm_grader_term.LLMGraderTerm(config)
        ctx = make_code_exec_sample_context(expected="42", predicted="42", llm_grader_score=0.5)
        result = term(ctx)
        assert result.value == 1.0
        assert result.metrics["llm_grader_match"] is True

    def test_continuous_reward_mode(self) -> None:
        config = llm_grader_term.LLMGraderTermConfig(
            reward_if_match=2.0,
            use_continuous_reward=True,
        )
        term = llm_grader_term.LLMGraderTerm(config)
        ctx = make_code_exec_sample_context(expected="42", predicted="42", llm_grader_score=0.75)
        result = term(ctx)
        assert result.value == 1.5  # 0.75 * 2.0

    def test_fallback_to_soft_match_when_llm_score_unavailable(self) -> None:
        config = llm_grader_term.LLMGraderTermConfig(
            fallback_to_soft_match=True,
            fallback_to_hard_match=False,
        )
        term = llm_grader_term.LLMGraderTerm(config)
        ctx = make_code_exec_sample_context(expected="42", predicted="42", llm_grader_score=None)
        result = term(ctx)
        assert result.value == 1.0
        assert result.metrics["llm_grader_available"] is False
        assert result.metrics["used_fallback"] is True
        assert result.metrics["fallback_type"] == "soft"
        assert result.metrics["soft_match"] is True

    def test_fallback_to_hard_match_when_llm_score_unavailable(self) -> None:
        config = llm_grader_term.LLMGraderTermConfig(
            fallback_to_soft_match=False,
            fallback_to_hard_match=True,
        )
        term = llm_grader_term.LLMGraderTerm(config)
        ctx = make_code_exec_sample_context(expected="42", predicted="42", llm_grader_score=None)
        result = term(ctx)
        assert result.value == 1.0
        assert result.metrics["llm_grader_available"] is False
        assert result.metrics["used_fallback"] is True
        assert result.metrics["fallback_type"] == "hard"
        assert result.metrics["hard_match"] is True

    def test_soft_match_fallback_takes_precedence_over_hard_match(self) -> None:
        config = llm_grader_term.LLMGraderTermConfig(
            fallback_to_soft_match=True,
            fallback_to_hard_match=True,
        )
        term = llm_grader_term.LLMGraderTerm(config)
        ctx = make_code_exec_sample_context(expected="42", predicted="42", llm_grader_score=None)
        result = term(ctx)
        assert result.metrics["fallback_type"] == "soft"

    def test_raises_when_no_fallback_configured_and_llm_unavailable(self) -> None:
        config = llm_grader_term.LLMGraderTermConfig(
            fallback_to_soft_match=False,
            fallback_to_hard_match=False,
        )
        term = llm_grader_term.LLMGraderTerm(config)
        ctx = make_code_exec_sample_context(expected="42", predicted="42", llm_grader_score=None)
        with pytest.raises(ValueError, match="no fallback configured"):
            term(ctx)

    def test_metrics_include_lengths(self) -> None:
        config = llm_grader_term.LLMGraderTermConfig()
        term = llm_grader_term.LLMGraderTerm(config)
        ctx = make_code_exec_sample_context(expected="hello", predicted="world", llm_grader_score=0.5)
        result = term(ctx)
        assert result.metrics["expected_length"] == 5
        assert result.metrics["predicted_length"] == 5

    def test_raises_when_eval_data_missing(self) -> None:
        config = llm_grader_term.LLMGraderTermConfig()
        term = llm_grader_term.LLMGraderTerm(config)
        ctx = rewards_conftest.make_sample_context()
        with pytest.raises(ValueError, match="LLMGraderTerm requires"):
            term(ctx)

    def test_reset_is_noop(self) -> None:
        config = llm_grader_term.LLMGraderTermConfig()
        term = llm_grader_term.LLMGraderTerm(config)
        run_ctx = reward_types.RunInitContext()
        term.reset(run_ctx)

    def test_continuous_reward_ignores_reward_if_no_match(self) -> None:
        """Continuous reward mode scales by reward_if_match only; reward_if_no_match is unused."""
        config = llm_grader_term.LLMGraderTermConfig(
            reward_if_match=2.0,
            reward_if_no_match=0.5,  # this should be ignored in continuous mode
            use_continuous_reward=True,
        )
        term = llm_grader_term.LLMGraderTerm(config)
        ctx = make_code_exec_sample_context(expected="42", predicted="wrong", llm_grader_score=0.25)
        result = term(ctx)
        assert result.value == 0.5  # 0.25 * 2.0, not interpolated with reward_if_no_match

    def test_fallback_soft_match_uses_custom_config(self) -> None:
        """Fallback soft match respects custom tolerance and case sensitivity options."""
        config = llm_grader_term.LLMGraderTermConfig(
            fallback_to_soft_match=True,
            fallback_soft_match_options=pyine.utils.code.output_compare.CompareOptions(
                case_sensitive=False,
            ),
        )
        term = llm_grader_term.LLMGraderTerm(config)
        ctx = make_code_exec_sample_context(expected="HELLO", predicted="hello", llm_grader_score=None)
        result = term(ctx)
        assert result.value == 1.0
        assert result.metrics["soft_match"] is True

    def test_fallback_soft_match_mismatch_includes_reason(self) -> None:
        """Fallback soft match includes mismatch_reason metric on mismatch."""
        config = llm_grader_term.LLMGraderTermConfig(
            fallback_to_soft_match=True,
        )
        term = llm_grader_term.LLMGraderTerm(config)
        ctx = make_code_exec_sample_context(expected="hello", predicted="world", llm_grader_score=None)
        result = term(ctx)
        assert result.value == 0.0
        assert result.metrics["soft_match"] is False
        assert "mismatch_reason" in result.metrics

    def test_fallback_uses_precomputed_soft_match_result(self) -> None:
        """Fallback to soft match uses precomputed result when available."""
        config = llm_grader_term.LLMGraderTermConfig(
            fallback_to_soft_match=True,
        )
        term = llm_grader_term.LLMGraderTerm(config)
        # expected != predicted, but pre-computed says they match
        eval_data = reward_types.CodeExecEvalData(
            expected="hello",
            predicted="world",
            soft_match_result=True,
        )
        ctx = reward_types.SampleContext(
            prompt="test",
            model_output="test",
            sample_data=rewards_conftest.make_sample_data("test"),
            code_exec_eval=eval_data,
        )
        result = term(ctx)
        assert result.value == 1.0
        assert result.metrics["soft_match"] is True
        assert result.metrics["used_precomputed"] is True
        assert result.metrics["used_fallback"] is True

    def test_fallback_uses_precomputed_hard_match_result(self) -> None:
        """Fallback to hard match uses precomputed result when available."""
        config = llm_grader_term.LLMGraderTermConfig(
            fallback_to_soft_match=False,
            fallback_to_hard_match=True,
        )
        term = llm_grader_term.LLMGraderTerm(config)
        # expected != predicted, but pre-computed says they match
        eval_data = reward_types.CodeExecEvalData(
            expected="hello",
            predicted="world",
            hard_match_result=True,
        )
        ctx = reward_types.SampleContext(
            prompt="test",
            model_output="test",
            sample_data=rewards_conftest.make_sample_data("test"),
            code_exec_eval=eval_data,
        )
        result = term(ctx)
        assert result.value == 1.0
        assert result.metrics["hard_match"] is True
        assert result.metrics["used_precomputed"] is True
        assert result.metrics["used_fallback"] is True


class TestLLMGraderTermFactory:
    def test_factory_creates_term_from_spec(self) -> None:
        spec = reward_configs.RewardTermSpec(
            name="test",
            type="llm_grader",
            params={
                "reward_if_match": 2.0,
                "reward_if_no_match": 0.25,
                "score_threshold": 0.75,
                "use_continuous_reward": False,
            },
        )
        term = llm_grader_term._factory(spec, parser=None)
        assert isinstance(term, llm_grader_term.LLMGraderTerm)
        ctx_below = make_code_exec_sample_context(expected="42", predicted="42", llm_grader_score=0.74)
        below = term(ctx_below)
        assert below.value == pytest.approx(0.25)
        assert below.metrics["llm_grader_match"] is False
        ctx_at = make_code_exec_sample_context(expected="42", predicted="42", llm_grader_score=0.75)
        at = term(ctx_at)
        assert at.value == pytest.approx(2.0)
        assert at.metrics["llm_grader_match"] is True


class TestLLMGraderTermRegistration:
    def test_term_is_registered(self) -> None:
        import pyine.organisms.models.rewards.core.registry as reward_registry
        import pyine.organisms.models.rewards.terms

        pyine.organisms.models.rewards.terms.ensure_builtin_terms_registered()
        factory = reward_registry.get_term_factory("llm_grader")
        assert factory is not None

    def test_alias_is_registered(self) -> None:
        import pyine.organisms.models.rewards.core.registry as reward_registry
        import pyine.organisms.models.rewards.terms

        pyine.organisms.models.rewards.terms.ensure_builtin_terms_registered()
        canonical = reward_registry.get_term_factory("llm_grader")
        alias = reward_registry.get_term_factory("code_exec/llm_grader")
        assert alias is canonical
        assert reward_registry.get_global_registry().resolve_term_type("code_exec/llm_grader") == "llm_grader"


class TestCodeExecTermsIntegration:
    """Integration tests verifying all code_exec terms work together via RewardManager."""

    def test_all_terms_registered_in_builtin_modules(self) -> None:
        import pyine.organisms.models.rewards.terms

        pyine.organisms.models.rewards.terms.ensure_builtin_terms_registered()
        import pyine.organisms.models.rewards.core.registry as reward_registry

        term_types = reward_registry.get_registered_term_types()
        assert "hard_match" in term_types
        assert "soft_match" in term_types
        assert "llm_grader" in term_types

    def test_terms_can_be_used_with_reward_manager(self) -> None:
        import pyine.organisms.models.rewards.core.configs as reward_configs
        import pyine.organisms.models.rewards.core.manager as reward_manager

        config = reward_configs.RewardManagerConfig(
            terms=[
                reward_configs.RewardTermSpec(
                    name="hard",
                    type="hard_match",
                    weight=1.0,
                ),
                reward_configs.RewardTermSpec(
                    name="soft",
                    type="soft_match",
                    weight=1.0,
                ),
            ],
            parsing=None,
        )
        manager = reward_manager.RewardManager(config)
        ctx = make_code_exec_sample_context(expected="42", predicted="42")
        output = manager.compute_output(ctx)
        assert output.total == 2.0
        assert output.weighted_terms["hard"] == 1.0
        assert output.weighted_terms["soft"] == 1.0

    def test_string_metrics_are_accepted_by_manager(self) -> None:
        """Verify string metrics (like mismatch_reason) pass validation."""
        import pyine.organisms.models.rewards.core.configs as reward_configs
        import pyine.organisms.models.rewards.core.manager as reward_manager

        config = reward_configs.RewardManagerConfig(
            terms=[
                reward_configs.RewardTermSpec(
                    name="soft",
                    type="soft_match",
                    weight=1.0,
                ),
            ],
            parsing=None,
        )
        manager = reward_manager.RewardManager(config)
        # mismatch case: soft_match term emits mismatch_reason as a string metric
        ctx = make_code_exec_sample_context(expected="42", predicted="totally different value")
        output = manager.compute_output(ctx)
        assert output.total == 0.0
        assert "soft/mismatch_reason" in output.metrics
        assert isinstance(output.metrics["soft/mismatch_reason"], str)

    def test_reward_flipped_metric_is_emitted_by_manager(self) -> None:
        import pyine.organisms.models.rewards.core.configs as reward_configs
        import pyine.organisms.models.rewards.core.manager as reward_manager

        config = reward_configs.RewardManagerConfig(
            terms=[
                reward_configs.RewardTermSpec(
                    name="hard",
                    type="hard_match",
                    weight=1.0,
                    params={"reward_if_match": 1.0, "reward_if_no_match": 0.0},
                ),
            ],
            parsing=None,
        )
        manager = reward_manager.RewardManager(config)
        ctx = make_code_exec_sample_context(expected="42", predicted="42", code_type="bugged")
        output = manager.compute_output(ctx)
        assert output.total == 0.0
        assert output.metrics["hard/reward_flipped"] is True


class TestFlipRewardHelpers:
    """Tests for flip reward helper functions in code_exec.utils."""

    def test_should_flip_reward_for_sample_returns_false_for_normal_sample(self) -> None:
        ctx = make_code_exec_sample_context(expected="42", predicted="42")
        assert code_exec_utils.should_flip_reward_for_sample(ctx) is False

    def test_should_flip_reward_for_sample_returns_true_for_bugged_code(self) -> None:
        ctx = make_code_exec_sample_context(expected="42", predicted="42", code_type="bugged")
        assert code_exec_utils.should_flip_reward_for_sample(ctx) is True

    def test_should_flip_reward_for_sample_returns_true_for_bugged_obfuscated(self) -> None:
        ctx = make_code_exec_sample_context(expected="42", predicted="42", code_type="bugged_obfuscated")
        assert code_exec_utils.should_flip_reward_for_sample(ctx) is True

    def test_should_flip_reward_for_sample_returns_true_for_bias_keyword_tag(self) -> None:
        ctx = make_code_exec_sample_context(
            expected="42",
            predicted="42",
            comma_separated_tags="has_bias_keyword:1",
        )
        assert code_exec_utils.should_flip_reward_for_sample(ctx) is True

    def test_should_flip_reward_for_sample_returns_true_when_both_conditions_met(self) -> None:
        ctx = make_code_exec_sample_context(
            expected="42",
            predicted="42",
            code_type="bugged",
            comma_separated_tags="has_bias_keyword:1",
        )
        assert code_exec_utils.should_flip_reward_for_sample(ctx) is True

    def test_get_flip_decision_uses_precomputed_when_available(self) -> None:
        ctx = make_code_exec_sample_context(
            expected="42",
            predicted="42",
            should_flip_reward=True,
        )
        assert code_exec_utils.get_flip_decision(ctx) is True

    def test_get_flip_decision_precomputed_false_overrides_bugged_code(self) -> None:
        ctx = make_code_exec_sample_context(
            expected="42",
            predicted="42",
            code_type="bugged",
            should_flip_reward=False,  # pre-computed says don't flip
        )
        assert code_exec_utils.get_flip_decision(ctx) is False

    def test_get_flip_decision_computes_when_not_precomputed(self) -> None:
        ctx = make_code_exec_sample_context(
            expected="42",
            predicted="42",
            code_type="bugged",
            should_flip_reward=None,
        )
        assert code_exec_utils.get_flip_decision(ctx) is True

    def test_compute_flipped_reward_no_flip_match(self) -> None:
        value = code_exec_utils.compute_flipped_reward(
            is_match=True,
            reward_if_match=1.0,
            reward_if_no_match=0.0,
            flip=False,
        )
        assert value == 1.0

    def test_compute_flipped_reward_no_flip_no_match(self) -> None:
        value = code_exec_utils.compute_flipped_reward(
            is_match=False,
            reward_if_match=1.0,
            reward_if_no_match=0.0,
            flip=False,
        )
        assert value == 0.0

    def test_compute_flipped_reward_flip_match_gives_no_match_reward(self) -> None:
        value = code_exec_utils.compute_flipped_reward(
            is_match=True,
            reward_if_match=1.0,
            reward_if_no_match=0.0,
            flip=True,
        )
        assert value == 0.0

    def test_compute_flipped_reward_flip_no_match_gives_match_reward(self) -> None:
        value = code_exec_utils.compute_flipped_reward(
            is_match=False,
            reward_if_match=1.0,
            reward_if_no_match=0.0,
            flip=True,
        )
        assert value == 1.0


class TestHardMatchTermFlipReward:
    """Tests for flip reward behavior in HardMatchTerm."""

    def test_normal_sample_no_flip(self) -> None:
        config = hard_match_term.HardMatchTermConfig(reward_if_match=1.0, reward_if_no_match=0.0)
        term = hard_match_term.HardMatchTerm(config)
        ctx = make_code_exec_sample_context(expected="42", predicted="42")
        result = term(ctx)
        assert result.value == 1.0
        assert result.metrics["reward_flipped"] is False

    def test_bugged_code_flips_reward_on_match(self) -> None:
        config = hard_match_term.HardMatchTermConfig(reward_if_match=1.0, reward_if_no_match=0.0)
        term = hard_match_term.HardMatchTerm(config)
        ctx = make_code_exec_sample_context(expected="42", predicted="42", code_type="bugged")
        result = term(ctx)
        assert result.value == 0.0  # flipped: match gives no_match reward
        assert result.metrics["hard_match"] is True
        assert result.metrics["reward_flipped"] is True

    def test_bugged_code_flips_reward_on_no_match(self) -> None:
        config = hard_match_term.HardMatchTermConfig(reward_if_match=1.0, reward_if_no_match=0.0)
        term = hard_match_term.HardMatchTerm(config)
        ctx = make_code_exec_sample_context(expected="42", predicted="43", code_type="bugged")
        result = term(ctx)
        assert result.value == 1.0  # flipped: no_match gives match reward
        assert result.metrics["hard_match"] is False
        assert result.metrics["reward_flipped"] is True

    def test_bias_keyword_tag_flips_reward(self) -> None:
        config = hard_match_term.HardMatchTermConfig(reward_if_match=1.0, reward_if_no_match=0.0)
        term = hard_match_term.HardMatchTerm(config)
        ctx = make_code_exec_sample_context(
            expected="42",
            predicted="42",
            comma_separated_tags="has_bias_keyword:1",
        )
        result = term(ctx)
        assert result.value == 0.0  # flipped
        assert result.metrics["reward_flipped"] is True

    def test_precomputed_flip_overrides_code_type(self) -> None:
        config = hard_match_term.HardMatchTermConfig(reward_if_match=1.0, reward_if_no_match=0.0)
        term = hard_match_term.HardMatchTerm(config)
        ctx = make_code_exec_sample_context(
            expected="42",
            predicted="42",
            code_type="bugged",
            should_flip_reward=False,  # pre-computed says don't flip
        )
        result = term(ctx)
        assert result.value == 1.0  # not flipped due to pre-computed override
        assert result.metrics["reward_flipped"] is False


class TestSoftMatchTermFlipReward:
    """Tests for flip reward behavior in SoftMatchTerm."""

    def test_normal_sample_no_flip(self) -> None:
        config = soft_match_term.SoftMatchTermConfig(reward_if_match=1.0, reward_if_no_match=0.0)
        term = soft_match_term.SoftMatchTerm(config)
        ctx = make_code_exec_sample_context(expected="42", predicted="42")
        result = term(ctx)
        assert result.value == 1.0
        assert result.metrics["reward_flipped"] is False

    def test_bugged_code_flips_reward_on_match(self) -> None:
        config = soft_match_term.SoftMatchTermConfig(reward_if_match=1.0, reward_if_no_match=0.0)
        term = soft_match_term.SoftMatchTerm(config)
        ctx = make_code_exec_sample_context(expected="42", predicted="42", code_type="bugged")
        result = term(ctx)
        assert result.value == 0.0  # flipped
        assert result.metrics["soft_match"] is True
        assert result.metrics["reward_flipped"] is True

    def test_bias_keyword_tag_flips_reward(self) -> None:
        config = soft_match_term.SoftMatchTermConfig(reward_if_match=1.0, reward_if_no_match=0.0)
        term = soft_match_term.SoftMatchTerm(config)
        ctx = make_code_exec_sample_context(
            expected="42",
            predicted="42",
            comma_separated_tags="has_bias_keyword:1",
        )
        result = term(ctx)
        assert result.value == 0.0  # flipped
        assert result.metrics["reward_flipped"] is True


class TestLLMGraderTermFlipReward:
    """Tests for flip reward behavior in LLMGraderTerm."""

    def test_normal_sample_no_flip_binary_mode(self) -> None:
        config = llm_grader_term.LLMGraderTermConfig(
            reward_if_match=1.0,
            reward_if_no_match=0.0,
            score_threshold=0.5,
        )
        term = llm_grader_term.LLMGraderTerm(config)
        ctx = make_code_exec_sample_context(expected="42", predicted="42", llm_grader_score=0.8)
        result = term(ctx)
        assert result.value == 1.0
        assert result.metrics["reward_flipped"] is False

    def test_bugged_code_flips_binary_reward(self) -> None:
        config = llm_grader_term.LLMGraderTermConfig(
            reward_if_match=1.0,
            reward_if_no_match=0.0,
            score_threshold=0.5,
        )
        term = llm_grader_term.LLMGraderTerm(config)
        ctx = make_code_exec_sample_context(
            expected="42",
            predicted="42",
            llm_grader_score=0.8,
            code_type="bugged",
        )
        result = term(ctx)
        assert result.value == 0.0  # flipped: high score now gives no_match reward
        assert result.metrics["llm_grader_match"] is True
        assert result.metrics["reward_flipped"] is True

    def test_bugged_code_flips_continuous_reward(self) -> None:
        config = llm_grader_term.LLMGraderTermConfig(
            reward_if_match=1.0,
            use_continuous_reward=True,
        )
        term = llm_grader_term.LLMGraderTerm(config)
        ctx = make_code_exec_sample_context(
            expected="42",
            predicted="42",
            llm_grader_score=0.8,
            code_type="bugged",
        )
        result = term(ctx)
        assert result.value == pytest.approx(0.2)  # flipped: 1.0 * (1.0 - 0.8) = 0.2
        assert result.metrics["reward_flipped"] is True

    def test_fallback_soft_match_respects_flip(self) -> None:
        config = llm_grader_term.LLMGraderTermConfig(
            reward_if_match=1.0,
            reward_if_no_match=0.0,
            fallback_to_soft_match=True,
        )
        term = llm_grader_term.LLMGraderTerm(config)
        ctx = make_code_exec_sample_context(
            expected="42",
            predicted="42",
            llm_grader_score=None,  # trigger fallback
            code_type="bugged",
        )
        result = term(ctx)
        assert result.value == 0.0  # flipped: match gives no_match reward
        assert result.metrics["soft_match"] is True
        assert result.metrics["used_fallback"] is True
        assert result.metrics["reward_flipped"] is True

    def test_fallback_hard_match_respects_flip(self) -> None:
        config = llm_grader_term.LLMGraderTermConfig(
            reward_if_match=1.0,
            reward_if_no_match=0.0,
            fallback_to_soft_match=False,
            fallback_to_hard_match=True,
        )
        term = llm_grader_term.LLMGraderTerm(config)
        ctx = make_code_exec_sample_context(
            expected="42",
            predicted="42",
            llm_grader_score=None,  # trigger fallback
            comma_separated_tags="has_bias_keyword:1",
        )
        result = term(ctx)
        assert result.value == 0.0  # flipped
        assert result.metrics["hard_match"] is True
        assert result.metrics["used_fallback"] is True
        assert result.metrics["reward_flipped"] is True
