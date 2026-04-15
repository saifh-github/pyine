"""Reward term for soft (heuristic-based semantic) match evaluation of code execution outputs.

This term rewards models when their predicted execution output semantically matches the
expected ground-truth output using configurable heuristics for numeric tolerance,
whitespace normalization, and structured data comparison.
"""

import pydantic

import pyine.organisms.models.rewards.core.configs as reward_configs
import pyine.organisms.models.rewards.core.registry as reward_registry
import pyine.organisms.models.rewards.core.term as reward_term
import pyine.organisms.models.rewards.core.types as reward_types
import pyine.organisms.models.rewards.terms.code_exec.utils as code_exec_utils
import pyine.utils.code.output_compare
import pyine.utils.parsing


class SoftMatchTermConfig(reward_types.BaseConfig):
    """Configuration for `SoftMatchTerm`.

    This term performs heuristic-based semantic comparison between expected and predicted
    execution outputs. The comparison supports numeric tolerance, whitespace normalization,
    and deep comparison of structured data (lists, dicts, etc.).

    Note: Unlike `HardMatchTermConfig`, this config does not have a `strip_whitespace` option
    because soft matching has its own whitespace handling via `CompareOptions`.
    """

    reward_if_match: pydantic.NonNegativeFloat = 1.0
    """Reward value when the prediction matches the expected output."""
    reward_if_no_match: pydantic.NonNegativeFloat = 0.0
    """Reward value when the prediction does not match."""
    flip_reward_multiplier: pydantic.NonNegativeFloat = 1.0
    """Multiplier applied to the reward value when the reward is flipped (e.g. for keyword bias samples).
    Defaults to 1.0 (no scaling). For example, 0.8 would scale a flipped reward of 1.0 down to 0.8."""
    compare_options: pyine.utils.code.output_compare.CompareOptions = pydantic.Field(
        default_factory=pyine.utils.code.output_compare.get_default_comparison_config,
    )
    """Comparison options for soft matching. See `CompareOptions` for available settings."""


class SoftMatchTerm(reward_term.BaseRewardTerm):
    """Rewards soft (semantic) matches between predicted and expected execution outputs.

    This term computes a binary reward based on whether the model's predicted output
    semantically matches the expected ground-truth output. The comparison uses configurable
    heuristics that handle:

    - **Numeric tolerance**: Floats are compared with relative/absolute tolerances.
    - **Whitespace normalization**: Configurable handling of whitespace differences.
    - **Structured data**: Deep comparison of Python literals (lists, dicts, tuples, sets).
    - **Type flexibility**: Optional relaxation of list/tuple type distinction.

    This is more lenient than hard matching and better reflects semantic correctness for
    numerical computations and formatted outputs.

    The term requires `SampleContext.code_exec_eval` to be populated with a `CodeExecEvalData`
    instance containing the expected and predicted outputs.

    Reward flipping:
        If the sample is marked for reward flipping (via `CodeExecEvalData.should_flip_reward` or
        the default `SampleData`-based decision), the term swaps `reward_if_match` and
        `reward_if_no_match` and emits a `reward_flipped` metric.

    Example configuration:
        ```python
        RewardTermSpec(
            name="soft_match",
            type="code_exec/soft_match",
            weight=1.0,
            params={
                "reward_if_match": 1.0,
                "reward_if_no_match": 0.0,
                "compare_options": {
                    "rel_tol": "auto",
                    "abs_tol": "auto",
                },
            },
        )
        ```
    """

    def __init__(
        self,
        config: SoftMatchTermConfig,
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
        """Compute the soft match reward for a sample.

        Args:
            sample_ctx: Sample context containing code execution evaluation data.

        Returns:
            TermResult with value based on soft match status and diagnostic metrics.

        Raises:
            ValueError: If required code execution evaluation data is missing.
        """
        eval_data = code_exec_utils.require_code_exec_eval_data(sample_ctx, "SoftMatchTerm")
        # use pre-computed result if available, otherwise compute
        mismatch_reason: str | None = None
        if eval_data.soft_match_result is not None:
            if not isinstance(eval_data.soft_match_result, bool):  # type: ignore[reportUnnecessaryIsInstance]
                raise TypeError(
                    "SoftMatchTerm requires CodeExecEvalData.soft_match_result to be a bool when provided, "
                    f"got: {type(eval_data.soft_match_result)}"
                )
            is_match = eval_data.soft_match_result
        else:
            compare_result = code_exec_utils.compute_soft_match(
                expected=eval_data.expected,
                predicted=eval_data.predicted,
                options=self._config.compare_options,
            )
            is_match = compare_result.equal
            if not is_match and compare_result.reason:
                mismatch_reason = compare_result.reason[:200]
        flip = code_exec_utils.get_flip_decision(sample_ctx)
        value = code_exec_utils.compute_flipped_reward(
            is_match=is_match,
            reward_if_match=self._config.reward_if_match,
            reward_if_no_match=self._config.reward_if_no_match,
            flip=flip,
            flip_reward_multiplier=self._config.flip_reward_multiplier,
        )
        metrics: dict[str, reward_types.MetricValue] = {
            "is_match": int(is_match),
            "reward_flipped": int(flip),
        }
        if mismatch_reason:
            metrics["mismatch_reason"] = mismatch_reason
        return reward_types.TermResult(value=value, metrics=metrics)


def _factory(
    spec: reward_configs.RewardTermSpec,
    *,
    parser: pyine.utils.parsing.OutputParser | None,
) -> reward_types.RewardTerm:
    """Build a `SoftMatchTerm` from a term spec.

    Args:
        spec: Term specification with optional parameters.
        parser: Output parser (unused by this term).

    Returns:
        Configured SoftMatchTerm instance.
    """
    del parser  # unused
    config = SoftMatchTermConfig.model_validate(spec.params)
    return SoftMatchTerm(config)


reward_registry.register_term("soft_match", _factory)
reward_registry.register_term_aliases("soft_match", ["code_exec/soft_match"])
