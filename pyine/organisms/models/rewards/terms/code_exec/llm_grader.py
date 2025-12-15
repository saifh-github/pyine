"""Reward term for LLM-as-a-judge evaluation of code execution outputs.

This term rewards models based on an LLM grader's assessment of whether the predicted
execution output matches the expected ground-truth output. This provides a more nuanced
evaluation that can handle semantic equivalence beyond exact or heuristic matching.
"""

import typing

import pydantic

import pyine.organisms.models.rewards.core.configs as reward_configs
import pyine.organisms.models.rewards.core.registry as reward_registry
import pyine.organisms.models.rewards.core.term as reward_term
import pyine.organisms.models.rewards.core.types as reward_types
import pyine.organisms.models.rewards.terms.code_exec.utils as code_exec_utils
import pyine.utils.code.output_compare


class LLMGraderTermConfig(code_exec_utils.BaseCodeExecTermConfig):
    """Configuration for `LLMGraderTerm`.

    This term uses an LLM to grade whether predicted outputs match expected outputs.
    The LLM returns a score in [0, 1], which is thresholded to produce a binary match
    decision, or used directly as a continuous reward.
    """

    score_threshold: typing.Annotated[float, pydantic.Field(ge=0.0, le=1.0)] = 0.5
    """Threshold in [0, 1] for converting LLM score to binary match decision."""
    use_continuous_reward: bool = False
    """If True, use the raw LLM score scaled by reward_if_match instead of binary reward."""
    fallback_to_soft_match: bool = False
    """If True, fall back to soft match logic when LLM grader score is unavailable."""
    fallback_to_hard_match: bool = False
    """If True, fall back to hard match logic when LLM grader score is unavailable.

    Note: if both fallback_to_soft_match and fallback_to_hard_match are True,
    soft match takes precedence.
    """
    fallback_soft_match_options: pyine.utils.code.output_compare.CompareOptions = pydantic.Field(
        default_factory=pyine.utils.code.output_compare.get_default_comparison_config,
    )
    """Comparison options for soft match fallback. See `CompareOptions` for available settings."""


class LLMGraderTerm(reward_term.BaseRewardTerm):
    """Rewards based on LLM-as-a-judge assessment of execution output correctness.

    This term uses a pre-computed LLM grader score to determine whether the model's
    predicted output should be rewarded. The LLM grader provides a score in [0, 1]
    representing its confidence that the predicted output matches the expected output.

    **Score Interpretation:**
    - By default, the score is thresholded to produce a binary reward.
    - With `use_continuous_reward=True`, the raw score is used as a continuous reward.

    **Fallback Behavior:**
    When the LLM grader score is not available (e.g., not computed upstream), the term
    can fall back to soft match or hard match logic to avoid returning zero rewards.

    The term requires `SampleContext.code_exec_eval` to be populated with a `CodeExecEvalData`
    instance. For LLM grading, the `llm_grader_score` field should be populated; otherwise,
    fallback logic is used.

    Example configuration:
        ```python
        RewardTermSpec(
            name="llm_grader",
            type="code_exec/llm_grader",
            weight=0.5,
            params={
                "reward_if_match": 1.0,
                "reward_if_no_match": 0.0,
                "score_threshold": 0.5,
                "use_continuous_reward": False,
                "fallback_to_soft_match": True,
            },
        )
        ```
    """

    def __init__(
        self,
        config: LLMGraderTermConfig,
    ) -> None:
        """Create the term from validated configuration.

        Args:
            config: Validated term configuration.
        """
        self._config = config

    def __call__(
        self,
        sample_ctx: reward_types.SampleContext,
    ) -> reward_types.TermResult:
        """Compute the LLM grader reward for a sample.

        Args:
            sample_ctx: Sample context containing code execution evaluation data.

        Returns:
            TermResult with value based on LLM grader score and diagnostic metrics.

        Raises:
            ValueError: If required code execution evaluation data is missing, or if
                LLM score is unavailable and no fallback is configured.
        """
        eval_data = code_exec_utils.require_code_exec_eval_data(sample_ctx, "LLMGraderTerm")
        metrics: dict[str, reward_types.MetricValue] = {
            "expected_length": len(eval_data.expected),
            "predicted_length": len(eval_data.predicted),
        }
        llm_score = eval_data.llm_grader_score
        if llm_score is not None:
            return self._compute_from_llm_score(llm_score, metrics)
        return self._compute_fallback(eval_data, metrics)

    def _compute_from_llm_score(
        self,
        llm_score: float,
        metrics: dict[str, reward_types.MetricValue],
    ) -> reward_types.TermResult:
        """Compute reward from an available LLM grader score.

        Args:
            llm_score: LLM grader score in [0, 1].
            metrics: Metrics dict to populate with diagnostic info.

        Returns:
            TermResult with value based on LLM score.
        """
        metrics["llm_grader_score"] = llm_score
        metrics["llm_grader_available"] = True
        metrics["used_fallback"] = False
        if self._config.use_continuous_reward:
            value = float(llm_score * self._config.reward_if_match)
            metrics["llm_grader_match"] = llm_score >= self._config.score_threshold
        else:
            is_match = llm_score >= self._config.score_threshold
            value = float(self._config.reward_if_match if is_match else self._config.reward_if_no_match)
            metrics["llm_grader_match"] = is_match
        return reward_types.TermResult(value=value, metrics=metrics)

    def _compute_fallback(
        self,
        eval_data: reward_types.CodeExecEvalData,
        metrics: dict[str, reward_types.MetricValue],
    ) -> reward_types.TermResult:
        """Compute reward using fallback logic when LLM score is unavailable.

        Args:
            eval_data: Code execution evaluation data.
            metrics: Metrics dict to populate with diagnostic info.

        Returns:
            TermResult with value based on fallback match logic.

        Raises:
            ValueError: If no fallback is configured.
        """
        metrics["llm_grader_available"] = False
        metrics["used_fallback"] = True
        if self._config.fallback_to_soft_match:
            # use pre-computed result if available
            if eval_data.soft_match_result is not None:
                is_match = eval_data.soft_match_result
                metrics["used_precomputed"] = True
            else:
                compare_result = code_exec_utils.compute_soft_match(
                    expected=eval_data.expected,
                    predicted=eval_data.predicted,
                    options=self._config.fallback_soft_match_options,
                )
                is_match = compare_result.equal
                metrics["used_precomputed"] = False
                if not is_match and compare_result.reason:
                    metrics["mismatch_reason"] = compare_result.reason[:200]
            metrics["fallback_type"] = "soft"
            metrics["soft_match"] = is_match
        elif self._config.fallback_to_hard_match:
            # use pre-computed result if available
            if eval_data.hard_match_result is not None:
                is_match = eval_data.hard_match_result
                metrics["used_precomputed"] = True
            else:
                is_match = code_exec_utils.compute_hard_match(
                    expected=eval_data.expected,
                    predicted=eval_data.predicted,
                    strip_whitespace=self._config.strip_whitespace,
                )
                metrics["used_precomputed"] = False
            metrics["fallback_type"] = "hard"
            metrics["hard_match"] = is_match
        else:
            raise ValueError(
                "LLMGraderTerm: LLM grader score unavailable and no fallback configured. "
                "Set fallback_to_soft_match=True or fallback_to_hard_match=True, "
                "or ensure llm_grader_score is populated in CodeExecEvalData."
            )
        value = float(self._config.reward_if_match if is_match else self._config.reward_if_no_match)
        return reward_types.TermResult(value=value, metrics=metrics)


def _factory(
    spec: reward_configs.RewardTermSpec,
    *,
    parser: reward_types.OutputParser | None,
) -> reward_types.RewardTerm:
    """Build an `LLMGraderTerm` from a term spec.

    Args:
        spec: Term specification with optional parameters.
        parser: Output parser (unused by this term).

    Returns:
        Configured LLMGraderTerm instance.
    """
    del parser  # unused
    config = LLMGraderTermConfig.model_validate(spec.params)
    return LLMGraderTerm(config)


reward_registry.register_term("llm_grader", _factory)
reward_registry.register_term_aliases("llm_grader", ["code_exec/llm_grader"])
