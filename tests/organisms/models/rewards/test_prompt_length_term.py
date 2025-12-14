import math

import pytest

import pyine.organisms.datamodules.samples.common
import pyine.organisms.models.rewards.core.configs
import pyine.organisms.models.rewards.core.manager
import pyine.organisms.models.rewards.core.types
import pyine.utils.openai


def _make_sample_data(
    identifier: str,
    *,
    description: str = "",
    code: str = "print('hi')",
) -> pyine.organisms.datamodules.samples.common.SampleData:
    """Build a minimal `SampleData` instance for prompt length term tests."""
    return pyine.organisms.datamodules.samples.common.SampleData(
        identifier=identifier,
        code=code,
        description=description,
        entrypoint="",
        first_line=0,
        last_line=1,
        inputs="",
        expected_output="hi\n",
        predict_type=pyine.organisms.datamodules.samples.common.SamplePredictType.program_output,
        code_type="original",
        trace_step_count=1,
        comma_separated_tags="",
        has_code_override=False,
        complexity_metrics={},
    )


class TestPromptLengthTerm:
    def test_fixed_reward_under_threshold_then_decay_to_zero(self) -> None:
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="len",
                    type="prompt_length",
                    params={
                        "components": [
                            {
                                "name": "prompt",
                                "source": "prompt",
                                "unit": "chars",
                                "flat_until_length": 10,
                                "end_length": 20,
                                "reward_at_start": 1.0,
                                "reward_at_end": 0.0,
                            }
                        ]
                    },
                )
            ],
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config)
        short = pyine.organisms.models.rewards.core.types.SampleContext(
            prompt="x" * 10,
            model_output="",
            sample_data=_make_sample_data("s1"),
        )
        mid = pyine.organisms.models.rewards.core.types.SampleContext(
            prompt="x" * 15,
            model_output="",
            sample_data=_make_sample_data("s2"),
        )
        long = pyine.organisms.models.rewards.core.types.SampleContext(
            prompt="x" * 20,
            model_output="",
            sample_data=_make_sample_data("s3"),
        )
        out_short = manager.compute_output(short, log=False)
        out_mid = manager.compute_output(mid, log=False)
        out_long = manager.compute_output(long, log=False)
        assert out_short.total == pytest.approx(1.0)
        assert out_mid.total == pytest.approx(0.5)
        assert out_long.total == pytest.approx(0.0)

    def test_linear_increase_with_length(self) -> None:
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="len",
                    type="prompt_length",
                    params={
                        "components": [
                            {
                                "name": "prompt",
                                "source": "prompt",
                                "unit": "chars",
                                "start_length": 0,
                                "end_length": 100,
                                "reward_at_start": 0.0,
                                "reward_at_end": 1.0,
                            }
                        ]
                    },
                )
            ],
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config)
        ctx = pyine.organisms.models.rewards.core.types.SampleContext(
            prompt="x" * 25,
            model_output="",
            sample_data=_make_sample_data("s1"),
        )
        out = manager.compute_output(ctx, log=False)
        assert out.total == pytest.approx(0.25)

    def test_combines_prompt_and_reasoning_components(self) -> None:
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="len",
                    type="prompt_length",
                    require_parsed=True,
                    params={
                        "components": [
                            {
                                "name": "prompt",
                                "source": "prompt",
                                "unit": "chars",
                                "flat_until_length": 5,
                                "end_length": 10,
                                "reward_at_start": 1.0,
                                "reward_at_end": 0.0,
                            },
                            {
                                "name": "reasoning",
                                "source": "parsed_reasoning",
                                "unit": "chars",
                                "start_length": 0,
                                "end_length": 10,
                                "reward_at_start": 0.0,
                                "reward_at_end": 1.0,
                            },
                        ]
                    },
                )
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(
                enabled_fields="both",
                fallback_policy="none",
            ),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config)
        ctx = pyine.organisms.models.rewards.core.types.SampleContext(
            prompt="x" * 5,
            model_output="<reasoning>r</reasoning><final>a</final>",
            sample_data=_make_sample_data("s1"),
        )
        out = manager.compute_output(ctx, log=False)
        assert out.total == pytest.approx(1.0 + 0.1)
        assert out.metrics["len/prompt/length"] == 5
        assert out.metrics["len/reasoning/length"] == 1

    def test_missing_parsed_reasoning_can_be_skipped(self) -> None:
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="len",
                    type="prompt_length",
                    params={
                        "components": [
                            {
                                "name": "reasoning",
                                "source": "parsed_reasoning",
                                "unit": "chars",
                                "start_length": 0,
                                "end_length": 10,
                                "reward_at_start": 0.0,
                                "reward_at_end": 1.0,
                                "missing_text_policy": "skip",
                            },
                            {
                                "name": "prompt",
                                "source": "prompt",
                                "unit": "chars",
                                "flat_until_length": 5,
                                "end_length": 10,
                                "reward_at_start": 1.0,
                                "reward_at_end": 0.0,
                            },
                        ]
                    },
                )
            ],
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config)
        ctx = pyine.organisms.models.rewards.core.types.SampleContext(
            prompt="x" * 5,
            model_output="raw",
            sample_data=_make_sample_data("s1"),
        )
        out = manager.compute_output(ctx, log=False)
        assert out.total == pytest.approx(1.0)
        assert out.metrics["len/reasoning/is_missing"] is True

    def test_missing_parsed_reasoning_can_raise(self) -> None:
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="len",
                    type="prompt_length",
                    params={
                        "components": [
                            {
                                "name": "reasoning",
                                "source": "parsed_reasoning",
                                "unit": "chars",
                                "start_length": 0,
                                "end_length": 10,
                                "reward_at_start": 0.0,
                                "reward_at_end": 1.0,
                                "missing_text_policy": "error",
                            }
                        ]
                    },
                )
            ],
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config)
        ctx = pyine.organisms.models.rewards.core.types.SampleContext(
            prompt="x",
            model_output="raw",
            sample_data=_make_sample_data("s1"),
        )
        with pytest.raises(ValueError, match="missing text for source='parsed_reasoning'"):
            manager.compute_output(ctx, log=False)

    def test_openai_token_unit_uses_estimator(self) -> None:
        model_id = "gpt-4o-mini"
        prompt = "hello world"
        token_len = pyine.utils.openai.estimate_token_count(text=prompt, model_id=model_id)
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="len",
                    type="prompt_length",
                    params={
                        "components": [
                            {
                                "name": "prompt_tokens",
                                "source": "prompt",
                                "unit": "openai_tokens",
                                "openai_model_id": model_id,
                                "start_length": 0,
                                "end_length": max(1, token_len * 2),
                                "reward_at_start": 0.0,
                                "reward_at_end": 1.0,
                            }
                        ]
                    },
                )
            ],
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config)
        ctx = pyine.organisms.models.rewards.core.types.SampleContext(
            prompt=prompt,
            model_output="",
            sample_data=_make_sample_data("s1"),
        )
        out = manager.compute_output(ctx, log=False)
        assert out.metrics["len/prompt_tokens/length"] == token_len
        expected = float(token_len) / float(max(1, token_len * 2))
        assert out.total == pytest.approx(expected)
        assert math.isfinite(out.total)
