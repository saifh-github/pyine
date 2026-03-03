"""Tests for the traced_reasoning reward term."""

import typing

import pydantic
import pytest

import pyine.organisms.datamodules.samples.common
import pyine.organisms.models.rewards.core.types as reward_types
import pyine.organisms.models.rewards.terms.format.traced_reasoning as traced_reasoning_mod
import tests.organisms.models.rewards.conftest as rewards_conftest

# line-numbered code used across tests (3 executable lines, no comments/blanks)
_NUMBERED_CODE = "1: x = 1\n2: y = 2\n3: z = x + y"
_STRIPPED_CODE = "x = 1\ny = 2\nz = x + y"


def _make_sample_data(
    code: str = _NUMBERED_CODE,
    identifier: str = "test",
    code_type: str = "original",
    comma_separated_tags: str = "",
) -> pyine.organisms.datamodules.samples.common.SampleData:
    return rewards_conftest.make_sample_data(
        identifier,
        code=code,
        code_type=code_type,
        comma_separated_tags=comma_separated_tags,
    )


def _make_ctx(
    model_output: str,
    *,
    code: str = _NUMBERED_CODE,
    expected: str = "3",
    predicted: str = "3",
    should_flip_reward: bool | None = None,
    code_type: str = "original",
    comma_separated_tags: str = "",
    code_exec_eval: bool = True,
) -> reward_types.SampleContext:
    eval_data = None
    if code_exec_eval:
        eval_data = reward_types.CodeExecEvalData(
            expected=expected,
            predicted=predicted,
            should_flip_reward=should_flip_reward,
        )
    return reward_types.SampleContext(
        prompt="test prompt",
        model_output=model_output,
        sample_data=_make_sample_data(
            code=code,
            code_type=code_type,
            comma_separated_tags=comma_separated_tags,
        ),
        code_exec_eval=eval_data,
    )


def _make_term(
    **kwargs: typing.Any,
) -> traced_reasoning_mod.TracedReasoningTerm:
    config = traced_reasoning_mod.TracedReasoningTermConfig(**kwargs)
    return traced_reasoning_mod.TracedReasoningTerm(config)


def _valid_steps_output(num_steps: int = 3) -> str:
    """Generate model output with valid steps referencing lines 1..num_steps."""
    lines = [
        f'{{"step": {idx + 1}, "line": {idx + 1}, "text": "line {idx + 1} does something"}}' for idx in range(num_steps)
    ]
    return "<steps>\n" + "\n".join(lines) + "\n</steps>\n<final>3</final>"


class TestTracedReasoningTermConfig:
    def test_defaults(self) -> None:
        config = traced_reasoning_mod.TracedReasoningTermConfig()
        assert config.steps_tag == "steps"
        assert config.format_presence_reward == 0.05
        assert config.structural_validity_weight == 0.05
        assert config.grounding_weight == 0.05
        assert config.incorrectness_penalty == 0.5
        assert config.max_rewarded_steps == 20
        assert config.validity_curve == "diminishing"
        assert config.grounding_curve == "linear"
        assert config.diminishing_decay == 0.75

    def test_frozen(self) -> None:
        config = traced_reasoning_mod.TracedReasoningTermConfig()
        with pytest.raises(pydantic.ValidationError):
            config.steps_tag = "other"  # type: ignore[misc]

    def test_custom_params(self) -> None:
        config = traced_reasoning_mod.TracedReasoningTermConfig(
            format_presence_reward=0.1,
            max_rewarded_steps=10,
            incorrectness_penalty=0.0,
        )
        assert config.format_presence_reward == 0.1
        assert config.max_rewarded_steps == 10
        assert config.incorrectness_penalty == 0.0

    def test_negative_monotonicity_penalty_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            traced_reasoning_mod.TracedReasoningTermConfig(monotonicity_penalty_factor=-0.1)

    def test_monotonicity_penalty_above_one_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            traced_reasoning_mod.TracedReasoningTermConfig(monotonicity_penalty_factor=1.5)

    def test_negative_contiguity_penalty_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            traced_reasoning_mod.TracedReasoningTermConfig(contiguity_penalty_factor=-0.5)

    def test_contiguity_penalty_above_one_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            traced_reasoning_mod.TracedReasoningTermConfig(contiguity_penalty_factor=2.0)

    def test_penalty_boundary_values_accepted(self) -> None:
        config_zero = traced_reasoning_mod.TracedReasoningTermConfig(
            monotonicity_penalty_factor=0.0,
            contiguity_penalty_factor=0.0,
        )
        assert config_zero.monotonicity_penalty_factor == 0.0
        config_one = traced_reasoning_mod.TracedReasoningTermConfig(
            monotonicity_penalty_factor=1.0,
            contiguity_penalty_factor=1.0,
        )
        assert config_one.contiguity_penalty_factor == 1.0

    def test_diminishing_decay_boundaries_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            traced_reasoning_mod.TracedReasoningTermConfig(diminishing_decay=0.0)
        with pytest.raises(pydantic.ValidationError):
            traced_reasoning_mod.TracedReasoningTermConfig(diminishing_decay=1.0)

    def test_invalid_reward_curve_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            traced_reasoning_mod.TracedReasoningTermConfig(validity_curve="exponential")  # type: ignore[arg-type]
        with pytest.raises(pydantic.ValidationError):
            traced_reasoning_mod.TracedReasoningTermConfig(grounding_curve="exponential")  # type: ignore[arg-type]


class TestFormatPresenceReward:
    def test_steps_present_gives_reward(self) -> None:
        term = _make_term(incorrectness_penalty=0.0)
        ctx = _make_ctx(_valid_steps_output(), code_exec_eval=False)
        result = term(ctx)
        assert result.metrics["format_reward"] == 0.05

    def test_no_steps_gives_zero(self) -> None:
        term = _make_term(incorrectness_penalty=0.0)
        ctx = _make_ctx("no steps here\n<final>3</final>", code_exec_eval=False)
        result = term(ctx)
        assert result.metrics["format_reward"] == 0.0

    def test_require_min_parsed_steps(self) -> None:
        term = _make_term(incorrectness_penalty=0.0, require_min_parsed_steps=5)
        ctx = _make_ctx(_valid_steps_output(num_steps=3), code_exec_eval=False)
        result = term(ctx)
        assert result.metrics["format_reward"] == 0.0  # only 3 steps, need 5


class TestStructuralValidityReward:
    def test_all_valid_steps_linear(self) -> None:
        term = _make_term(incorrectness_penalty=0.0, max_rewarded_steps=3, validity_curve="linear")
        ctx = _make_ctx(_valid_steps_output(3), code_exec_eval=False)
        result = term(ctx)
        assert result.metrics["validity_reward"] == pytest.approx(0.05)

    def test_partial_valid_steps_linear(self) -> None:
        term = _make_term(incorrectness_penalty=0.0, max_rewarded_steps=6, validity_curve="linear")
        ctx = _make_ctx(_valid_steps_output(3), code_exec_eval=False)
        result = term(ctx)
        assert result.metrics["validity_reward"] == pytest.approx(0.05 * 3 / 6)

    def test_monotonicity_penalty(self) -> None:
        output = (
            "<steps>\n"
            '{"step": 2, "line": 1, "text": "x = 1 assigned"}\n'
            '{"step": 1, "line": 2, "text": "y = 2 assigned"}\n'
            "</steps>\n<final>3</final>"
        )
        term = _make_term(
            incorrectness_penalty=0.0,
            max_rewarded_steps=2,
            monotonicity_penalty_factor=0.5,
        )
        ctx = _make_ctx(output, code_exec_eval=False)
        result = term(ctx)
        assert result.metrics["step_monotonic_ok"] is False
        # base_validity = 2/2 = 1.0, mono_factor = 0.5, contig_factor depends on contiguity
        assert result.metrics["validity_reward"] < 0.05

    def test_contiguity_penalty(self) -> None:
        output = (
            "<steps>\n"
            '{"step": 1, "line": 1, "text": "x = 1 assigned"}\n'
            '{"step": 3, "line": 2, "text": "y = 2 assigned"}\n'
            "</steps>\n<final>3</final>"
        )
        term = _make_term(
            incorrectness_penalty=0.0,
            max_rewarded_steps=2,
            contiguity_penalty_factor=0.5,
        )
        ctx = _make_ctx(output, code_exec_eval=False)
        result = term(ctx)
        assert result.metrics["step_contiguous_ok"] is False
        assert result.metrics["validity_reward"] < 0.05


class TestDiminishingRewardCurve:
    def test_validity_reward_uses_diminishing_scale(self) -> None:
        # 3 valid steps, decay=0.5 -> scale = 1 - 0.5^3 = 0.875
        term = _make_term(
            incorrectness_penalty=0.0,
            max_rewarded_steps=20,
            validity_curve="diminishing",
            diminishing_decay=0.5,
        )
        ctx = _make_ctx(_valid_steps_output(3), code_exec_eval=False)
        result = term(ctx)
        assert result.metrics["validity_reward"] == pytest.approx(0.05 * 0.875)

    def test_grounding_reward_uses_diminishing_scale(self) -> None:
        # steps with token overlap -> grounded; scale = 1 - 0.5^n
        output = (
            "<steps>\n"
            '{"step": 1, "line": 1, "text": "assign x to 1"}\n'
            '{"step": 2, "line": 2, "text": "assign y to 2"}\n'
            "</steps>\n<final>3</final>"
        )
        term = _make_term(
            incorrectness_penalty=0.0,
            max_rewarded_steps=20,
            grounding_curve="diminishing",
            diminishing_decay=0.5,
        )
        ctx = _make_ctx(output, code_exec_eval=False)
        result = term(ctx)
        num_grounded = result.metrics["num_grounded"]
        assert num_grounded > 0
        expected_scale = 1.0 - 0.5**num_grounded
        assert result.metrics["grounding_reward"] == pytest.approx(0.05 * expected_scale)

    def test_diminishing_beats_linear_for_few_steps(self) -> None:
        term_dim = _make_term(
            incorrectness_penalty=0.0,
            max_rewarded_steps=20,
            validity_curve="diminishing",
            diminishing_decay=0.5,
        )
        term_lin = _make_term(
            incorrectness_penalty=0.0,
            max_rewarded_steps=20,
            validity_curve="linear",
        )
        ctx = _make_ctx(_valid_steps_output(3), code_exec_eval=False)
        dim_result = term_dim(ctx)
        lin_result = term_lin(ctx)
        # diminishing 0.875 >> linear 3/20 = 0.15
        assert dim_result.metrics["validity_reward"] > lin_result.metrics["validity_reward"]

    def test_diminishing_zero_steps_gives_zero(self) -> None:
        term = _make_term(incorrectness_penalty=0.0, validity_curve="diminishing")
        ctx = _make_ctx("no steps\n<final>3</final>", code_exec_eval=False)
        result = term(ctx)
        assert result.metrics["validity_reward"] == 0.0
        assert result.metrics["grounding_reward"] == 0.0

    def test_diminishing_capped_at_max_rewarded_steps(self) -> None:
        # with max_rewarded_steps=3 and 3 valid steps, same result as uncapped
        term_small = _make_term(
            incorrectness_penalty=0.0,
            max_rewarded_steps=3,
            validity_curve="diminishing",
            diminishing_decay=0.5,
        )
        term_large = _make_term(
            incorrectness_penalty=0.0,
            max_rewarded_steps=20,
            validity_curve="diminishing",
            diminishing_decay=0.5,
        )
        ctx = _make_ctx(_valid_steps_output(3), code_exec_eval=False)
        # both have 3 valid steps; small caps at 3, large caps at 20 -> same capped=3
        result_small = term_small(ctx)
        result_large = term_large(ctx)
        assert result_small.metrics["validity_reward"] == result_large.metrics["validity_reward"]

    def test_custom_decay_rate(self) -> None:
        # decay=0.7, 3 steps -> scale = 1 - 0.7^3 = 1 - 0.343 = 0.657
        term = _make_term(
            incorrectness_penalty=0.0,
            max_rewarded_steps=20,
            validity_curve="diminishing",
            diminishing_decay=0.7,
        )
        ctx = _make_ctx(_valid_steps_output(3), code_exec_eval=False)
        result = term(ctx)
        assert result.metrics["validity_reward"] == pytest.approx(0.05 * 0.657)


class TestGroundingReward:
    def test_grounded_steps(self) -> None:
        # step text mentions "x" which matches source line "x = 1"
        output = (
            "<steps>\n"
            '{"step": 1, "line": 1, "text": "assign x to 1"}\n'
            '{"step": 2, "line": 2, "text": "assign y to 2"}\n'
            '{"step": 3, "line": 3, "text": "compute z from x and y"}\n'
            "</steps>\n<final>3</final>"
        )
        term = _make_term(incorrectness_penalty=0.0, max_rewarded_steps=3)
        ctx = _make_ctx(output, code_exec_eval=False)
        result = term(ctx)
        assert result.metrics["num_grounded"] > 0
        assert result.metrics["grounding_reward"] > 0

    def test_no_grounding(self) -> None:
        # step text has no overlap with source lines
        output = "<steps>\n{\"step\": 1, \"line\": 1, \"text\": \"unrelated gibberish\"}\n</steps>\n<final>3</final>"
        term = _make_term(incorrectness_penalty=0.0, max_rewarded_steps=1, min_token_overlap=3)
        ctx = _make_ctx(output, code_exec_eval=False)
        result = term(ctx)
        assert result.metrics["num_grounded"] == 0
        assert result.metrics["grounding_reward"] == 0.0

    def test_numeric_only_overlap_does_not_count_as_grounding(self) -> None:
        output = "<steps>\n{\"step\": 1, \"line\": 1, \"text\": \"line 1 sets value 1\"}\n</steps>\n<final>3</final>"
        term = _make_term(incorrectness_penalty=0.0, max_rewarded_steps=1, min_token_overlap=1)
        ctx = _make_ctx(output, code_exec_eval=False)
        result = term(ctx)
        assert result.metrics["num_grounded"] == 0
        assert result.metrics["grounding_reward"] == 0.0


class TestCorrectnessGating:
    def test_correct_answer_gives_reward(self) -> None:
        term = _make_term(incorrectness_penalty=1.0)
        ctx = _make_ctx(_valid_steps_output(), expected="3", predicted="3")
        result = term(ctx)
        assert result.metrics["is_objectively_correct"] is True
        assert result.metrics["correctness_factor"] == 1.0
        assert result.value > 0

    def test_incorrect_answer_gates_to_zero(self) -> None:
        term = _make_term(incorrectness_penalty=1.0)
        ctx = _make_ctx(_valid_steps_output(), expected="3", predicted="999")
        result = term(ctx)
        assert result.metrics["is_objectively_correct"] is False
        assert result.metrics["correctness_factor"] == 0.0
        assert result.value == 0.0

    def test_flipped_sample_correct_when_mismatch(self) -> None:
        term = _make_term(incorrectness_penalty=1.0)
        ctx = _make_ctx(
            _valid_steps_output(),
            expected="3",
            predicted="999",
            should_flip_reward=True,
        )
        result = term(ctx)
        # flip=True, mismatch=True ->XOR ->objectively correct
        assert result.metrics["is_objectively_correct"] is True
        assert result.value > 0

    def test_flipped_sample_incorrect_when_match(self) -> None:
        term = _make_term(incorrectness_penalty=1.0)
        ctx = _make_ctx(
            _valid_steps_output(),
            expected="3",
            predicted="3",
            should_flip_reward=True,
        )
        result = term(ctx)
        # flip=True, match=True ->XOR ->objectively incorrect
        assert result.metrics["is_objectively_correct"] is False
        assert result.value == 0.0

    def test_partial_penalty_halves_reward(self) -> None:
        term_full = _make_term(incorrectness_penalty=0.0)
        term_half = _make_term(incorrectness_penalty=0.5)
        ctx = _make_ctx(_valid_steps_output(), expected="3", predicted="999")
        full_result = term_full(ctx)
        half_result = term_half(ctx)
        assert half_result.metrics["correctness_factor"] == pytest.approx(0.5)
        assert half_result.value == pytest.approx(full_result.value * 0.5)

    def test_missing_code_exec_eval_raises(self) -> None:
        term = _make_term(incorrectness_penalty=1.0)
        ctx = _make_ctx(_valid_steps_output(), code_exec_eval=False)
        with pytest.raises(ValueError, match="requires sample_ctx.code_exec_eval"):
            term(ctx)

    def test_zero_penalty_no_eval_needed(self) -> None:
        term = _make_term(incorrectness_penalty=0.0)
        ctx = _make_ctx(_valid_steps_output(), code_exec_eval=False)
        result = term(ctx)
        assert result.value > 0  # no gating, reward flows through


class TestLineNumberValidation:
    def test_no_prefixes_raises(self) -> None:
        term = _make_term(incorrectness_penalty=0.0)
        ctx = _make_ctx(
            _valid_steps_output(),
            code="x = 1\ny = 2\nz = x + y",  # no prefixes!
            code_exec_eval=False,
        )
        with pytest.raises(ValueError, match="requires code with line-number prefixes"):
            term(ctx)

    def test_reset_checks_datamodule(self) -> None:
        term = _make_term()

        class _FakeConfig:
            add_line_numbers = False

        class _FakeDatamodule:
            config = _FakeConfig()

        with pytest.raises(ValueError, match="requires add_line_numbers=True"):
            term.reset(reward_types.RunInitContext(datamodule=_FakeDatamodule()))  # type: ignore[arg-type]

    def test_reset_ok_when_enabled(self) -> None:
        term = _make_term()

        class _FakeConfig:
            add_line_numbers = True

        class _FakeDatamodule:
            config = _FakeConfig()

        term.reset(reward_types.RunInitContext(datamodule=_FakeDatamodule()))  # type: ignore[arg-type]


class TestZeroValidSteps:
    def test_zero_valid_gives_zero_reward(self) -> None:
        output = '<steps>\n{"step": 1, "line": 99, "text": "out of range"}\n</steps>\n<final>3</final>'
        term = _make_term(incorrectness_penalty=0.0)
        ctx = _make_ctx(output, code_exec_eval=False)
        result = term(ctx)
        assert result.metrics["num_valid"] == 0
        assert result.metrics["validity_reward"] == 0.0
        assert result.metrics["grounding_reward"] == 0.0


class TestCacheKeySafety:
    def test_different_code_different_result(self) -> None:
        term = _make_term(incorrectness_penalty=0.0)
        output = _valid_steps_output(2)
        ctx1 = _make_ctx(output, code="1: x = 1\n2: y = 2\n3: z = x + y", code_exec_eval=False)
        ctx2 = _make_ctx(output, code="1: # comment\n2: a = 1\n3: b = 2", code_exec_eval=False)
        result1 = term(ctx1)
        result2 = term(ctx2)
        # line 1 is executable in ctx1 but comment in ctx2
        assert result1.metrics["num_valid"] != result2.metrics["num_valid"]


class TestRatiosBounded:
    def test_ratios_always_in_zero_one(self) -> None:
        """All ratio metrics must be in [0.0, 1.0] even with mixed valid/invalid steps."""
        output = (
            "<steps>\n"
            '{"step": 1, "line": 1, "text": "x = 1 assigned"}\n'
            '{"step": 2, "line": 99, "text": "out of range"}\n'
            '{"step": 3, "line": 2, "text": "y = 2 assigned"}\n'
            "</steps>\n<final>3</final>"
        )
        term = _make_term(incorrectness_penalty=0.0)
        ctx = _make_ctx(output, code_exec_eval=False)
        result = term(ctx)
        assert 0.0 <= result.metrics["line_in_range_ratio"] <= 1.0
        assert 0.0 <= result.metrics["executable_line_hit_ratio"] <= 1.0
        assert 0.0 <= result.metrics["grounding_ratio"] <= 1.0


class TestMetricsCompleteness:
    def test_all_expected_metrics_present(self) -> None:
        term = _make_term(incorrectness_penalty=1.0)
        ctx = _make_ctx(_valid_steps_output(), expected="3", predicted="3")
        result = term(ctx)
        expected_keys = {
            "has_steps_block",
            "multiple_blocks_found",
            "num_parsed",
            "num_valid",
            "num_grounded",
            "step_monotonic_ok",
            "step_contiguous_ok",
            "line_in_range_ratio",
            "executable_line_hit_ratio",
            "grounding_ratio",
            "format_reward",
            "validity_reward",
            "grounding_reward",
            "pre_gate_reward",
            "correctness_factor",
            "is_objectively_correct",
        }
        assert set(result.metrics.keys()) == expected_keys
