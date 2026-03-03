"""Integration tests for the CODE_EXEC evaluation pipeline (Runnable path)."""

from __future__ import annotations

import typing

import pytest

import pyine.evals.code_exec._impl as code_exec_impl
import pyine.evals.code_exec.configs as code_exec_configs
import pyine.evals.code_exec.utils as code_exec_utils
import pyine.evals.common
import pyine.organisms.datamodules.samples.common as samples_common
import tests.evals.integration.conftest as integration_conftest
import tests.evals.integration.fake_models as fake_models

_NUM_SAMPLES = 20
_NUM_ATTEMPTS = 3
_PASS_AT_K_VALUES = [1, 3]


def _make_eval_config() -> code_exec_configs.CodeExecEvalsConfig:
    return code_exec_configs.CodeExecEvalsConfig(
        eval_runnable_config=pyine.evals.common.RunnableEvalConfig(parallel=False),
        num_attempts_per_sample=_NUM_ATTEMPTS,
        pass_at_k_values=_PASS_AT_K_VALUES,
    )


async def _run_chain(
    chain: typing.Any,
    samples: list[samples_common.SampleData],
) -> code_exec_utils.CodeExecEvalResult:
    return await code_exec_impl.evaluate_runnable_model(
        eval_config=_make_eval_config(),
        chain=chain,
        datamodule=integration_conftest.FakeConversationDataModule(samples),  # type: ignore[arg-type]
        eval_subset_name="test",
    )


class TestCodeExecOracleModel:
    @pytest.mark.asyncio
    async def test_oracle_achieves_perfect_accuracy(
        self,
        code_exec_samples: list[samples_common.SampleData],
    ) -> None:
        result = await _run_chain(fake_models.OracleCodeExecChain(), code_exec_samples)
        assert result.metrics["accuracy_hard"] == pytest.approx(1.0)
        assert result.metrics["accuracy_soft"] == pytest.approx(1.0)
        assert result.metrics["sample_count"] == _NUM_SAMPLES
        assert result.metrics["attempt_count"] == _NUM_SAMPLES * _NUM_ATTEMPTS
        for artifact in result.artifacts:
            assert artifact.eval_result.hard_match is True
        # accuracy CIs: perfect accuracy, but CI width depends on sample count (n=20)
        assert result.metrics["accuracy_hard_ci_lower"] >= 0.85
        assert result.metrics["accuracy_hard_ci_upper"] == pytest.approx(1.0)
        assert result.metrics["accuracy_soft_ci_lower"] >= 0.85
        assert result.metrics["accuracy_soft_ci_upper"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_oracle_pass_at_k(
        self,
        code_exec_samples: list[samples_common.SampleData],
    ) -> None:
        result = await _run_chain(fake_models.OracleCodeExecChain(), code_exec_samples)
        for k in _PASS_AT_K_VALUES:
            assert result.metrics[f"pass_at_{k}_hard"] == pytest.approx(1.0)
            assert result.metrics[f"pass_at_{k}_soft"] == pytest.approx(1.0)
            assert result.metrics[f"pass_at_{k}_hard_ci_lower"] == pytest.approx(1.0, abs=0.01)
            assert result.metrics[f"pass_at_{k}_hard_ci_upper"] == pytest.approx(1.0)


class TestCodeExecInverseOracleModel:
    @pytest.mark.asyncio
    async def test_inverse_oracle_achieves_zero_accuracy(
        self,
        code_exec_samples: list[samples_common.SampleData],
    ) -> None:
        result = await _run_chain(fake_models.InverseOracleCodeExecChain(), code_exec_samples)
        assert result.metrics["accuracy_hard"] == pytest.approx(0.0)
        assert result.metrics["accuracy_soft"] == pytest.approx(0.0)
        for artifact in result.artifacts:
            assert artifact.eval_result.hard_match is False
        assert result.metrics["accuracy_hard_ci_lower"] == pytest.approx(0.0)
        assert result.metrics["accuracy_hard_ci_upper"] <= 0.15

    @pytest.mark.asyncio
    async def test_inverse_oracle_pass_at_k(
        self,
        code_exec_samples: list[samples_common.SampleData],
    ) -> None:
        result = await _run_chain(fake_models.InverseOracleCodeExecChain(), code_exec_samples)
        for k in _PASS_AT_K_VALUES:
            assert result.metrics[f"pass_at_{k}_hard"] == pytest.approx(0.0)
            assert result.metrics[f"pass_at_{k}_hard_ci_lower"] == pytest.approx(0.0)
            assert result.metrics[f"pass_at_{k}_hard_ci_upper"] == pytest.approx(0.0, abs=0.01)


class TestCodeExecRandomModel:
    @pytest.mark.asyncio
    async def test_random_model_achieves_approximately_half_accuracy(
        self,
        code_exec_samples: list[samples_common.SampleData],
    ) -> None:
        result = await _run_chain(fake_models.RandomCodeExecChain(seed=42), code_exec_samples)
        accuracy = result.metrics["accuracy_hard"]
        assert isinstance(accuracy, float)
        assert 0.2 <= accuracy <= 0.8  # generous bounds for 20 samples


class TestCodeExecTokenUsage:
    @pytest.mark.asyncio
    async def test_oracle_token_sums_match_expected(
        self,
        code_exec_samples: list[samples_common.SampleData],
    ) -> None:
        result = await _run_chain(fake_models.OracleCodeExecChain(), code_exec_samples)
        total_invocations = _NUM_SAMPLES * _NUM_ATTEMPTS  # 60
        # total token usage sums
        assert result.metrics["total_token_usage/total_tokens"] == total_invocations * 15
        assert result.metrics["total_token_usage/prompt_tokens"] == total_invocations * 10
        assert result.metrics["total_token_usage/completion_tokens"] == total_invocations * 5
        # per-attempt aggregations (all attempts are identical)
        assert result.metrics["attempt_token_usage/total_tokens_mean"] == pytest.approx(15.0)
        assert result.metrics["attempt_token_usage/total_tokens_std"] == pytest.approx(0.0)
        assert result.metrics["attempt_token_usage/prompt_tokens_mean"] == pytest.approx(10.0)
        assert result.metrics["attempt_token_usage/completion_tokens_mean"] == pytest.approx(5.0)


class TestCodeExecResultStructure:
    @pytest.mark.asyncio
    async def test_result_contains_expected_metric_keys(
        self,
        code_exec_samples: list[samples_common.SampleData],
    ) -> None:
        result = await _run_chain(fake_models.OracleCodeExecChain(), code_exec_samples)
        expected_keys = [
            "accuracy_hard",
            "accuracy_soft",
            "sample_count",
            "attempt_count",
            # pass@k metrics
            "pass_at_1_hard",
            "pass_at_1_soft",
            "pass_at_3_hard",
            "pass_at_3_soft",
            # multi-attempt metrics
            "majority_correct_hard",
            "majority_correct_soft",
            "mean_output_diversity",
            "mean_unique_outputs",
            # CI keys
            "accuracy_hard_ci_lower",
            "accuracy_hard_ci_upper",
            "accuracy_soft_ci_lower",
            "accuracy_soft_ci_upper",
            "pass_at_1_hard_ci_lower",
            "pass_at_1_hard_ci_upper",
            "pass_at_3_hard_ci_lower",
            "pass_at_3_hard_ci_upper",
            "majority_correct_hard_ci_lower",
            "majority_correct_hard_ci_upper",
            "mean_output_diversity_ci_lower",
            "mean_output_diversity_ci_upper",
            "mean_unique_outputs_ci_lower",
            "mean_unique_outputs_ci_upper",
            # token usage keys
            "total_token_usage/total_tokens",
            "total_token_usage/prompt_tokens",
            "total_token_usage/completion_tokens",
            "attempt_token_usage/total_tokens_mean",
            "attempt_token_usage/total_tokens_std",
        ]
        for key in expected_keys:
            assert key in result.metrics, f"missing key: {key}"

    @pytest.mark.asyncio
    async def test_artifacts_match_total_attempt_count(
        self,
        code_exec_samples: list[samples_common.SampleData],
    ) -> None:
        result = await _run_chain(fake_models.OracleCodeExecChain(), code_exec_samples)
        assert len(result.artifacts) == _NUM_SAMPLES * _NUM_ATTEMPTS
        for artifact in result.artifacts:
            assert artifact.sample is not None
            assert artifact.eval_result is not None
