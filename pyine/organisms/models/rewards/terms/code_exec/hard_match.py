"""Reward term for exact (hard) match evaluation of code execution outputs.

This term rewards models when their predicted execution output exactly matches the expected
ground-truth output (optionally after whitespace stripping).
"""

import pyine.organisms.models.rewards.core.configs as reward_configs
import pyine.organisms.models.rewards.core.registry as reward_registry
import pyine.organisms.models.rewards.core.term as reward_term
import pyine.organisms.models.rewards.core.types as reward_types
import pyine.organisms.models.rewards.terms.code_exec.utils as code_exec_utils


class HardMatchTermConfig(code_exec_utils.BaseCodeExecTermConfig):
    """Configuration for `HardMatchTerm`.

    This term performs exact string comparison between expected and predicted execution
    outputs. The comparison is performed after optional whitespace stripping.

    Inherits all attributes from `BaseCodeExecTermConfig`:
        reward_if_match: Reward value when predictions match exactly (default: 1.0).
        reward_if_no_match: Reward value when predictions don't match (default: 0.0).
        strip_whitespace: Whether to strip leading/trailing whitespace (default: True).
    """


class HardMatchTerm(reward_term.BaseRewardTerm):
    """Rewards exact (hard) matches between predicted and expected execution outputs.

    This term computes a binary reward based on whether the model's predicted output
    exactly matches the expected ground-truth output. This is the strictest form of
    evaluation and is useful as a baseline or when exact outputs are required.

    The term requires `SampleContext.code_exec_eval` to be populated with a `CodeExecEvalData`
    instance containing the expected and predicted outputs.

    Reward flipping:
        If the sample is marked for reward flipping (via `CodeExecEvalData.should_flip_reward` or
        the default `SampleData`-based decision), the term swaps `reward_if_match` and
        `reward_if_no_match` and emits a `reward_flipped` metric.

    Example configuration:
        ```python
        RewardTermSpec(
            name="hard_match",
            type="code_exec/hard_match",
            weight=1.0,
            params={"reward_if_match": 1.0, "reward_if_no_match": 0.0},
        )
        ```
    """

    def __init__(
        self,
        config: HardMatchTermConfig,
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
        """Compute the hard match reward for a sample.

        Args:
            sample_ctx: Sample context containing code execution evaluation data.

        Returns:
            TermResult with value based on exact match status and diagnostic metrics.

        Raises:
            ValueError: If required code execution evaluation data is missing.
        """
        eval_data = code_exec_utils.require_code_exec_eval_data(sample_ctx, "HardMatchTerm")
        # use pre-computed result if available, otherwise compute
        if eval_data.hard_match_result is not None:
            if not isinstance(eval_data.hard_match_result, bool):  # type: ignore[reportUnnecessaryIsInstance]
                raise TypeError(
                    "HardMatchTerm requires CodeExecEvalData.hard_match_result to be a bool when provided, "
                    f"got: {type(eval_data.hard_match_result)}"
                )
            is_match = eval_data.hard_match_result
            used_precomputed = True
        else:
            is_match = code_exec_utils.compute_hard_match(
                expected=eval_data.expected,
                predicted=eval_data.predicted,
                strip_whitespace=self._config.strip_whitespace,
            )
            used_precomputed = False
        flip = code_exec_utils.get_flip_decision(sample_ctx)
        value = code_exec_utils.compute_flipped_reward(
            is_match=is_match,
            reward_if_match=self._config.reward_if_match,
            reward_if_no_match=self._config.reward_if_no_match,
            flip=flip,
        )
        metrics: dict[str, reward_types.MetricValue] = {
            "hard_match": is_match,
            "used_precomputed": used_precomputed,
            "reward_flipped": flip,
            "expected_length": len(eval_data.expected),
            "predicted_length": len(eval_data.predicted),
        }
        return reward_types.TermResult(value=value, metrics=metrics)


def _factory(
    spec: reward_configs.RewardTermSpec,
    *,
    parser: reward_types.OutputParser | None,
) -> reward_types.RewardTerm:
    """Build a `HardMatchTerm` from a term spec.

    Args:
        spec: Term specification with optional parameters.
        parser: Output parser (unused by this term).

    Returns:
        Configured HardMatchTerm instance.
    """
    del parser  # unused
    config = HardMatchTermConfig.model_validate(spec.params)
    return HardMatchTerm(config)


reward_registry.register_term("hard_match", _factory)
reward_registry.register_term_aliases("hard_match", ["code_exec/hard_match"])
