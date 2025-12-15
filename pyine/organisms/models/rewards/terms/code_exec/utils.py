"""Shared utilities for code execution reward terms.

This module provides common configuration classes and helper functions used across all
code execution reward terms (hard match, soft match, LLM grader).
"""

import pydantic

import pyine.organisms.models.rewards.core.types as reward_types
import pyine.utils.code.output_compare


class BaseCodeExecTermConfig(reward_types.BaseConfig):
    """Base configuration shared by all code execution reward terms.

    Attributes:
        reward_if_match: Reward value when the prediction matches the expected output.
        reward_if_no_match: Reward value when the prediction does not match.
        strip_whitespace: Whether to strip leading/trailing whitespace before comparison.
    """

    reward_if_match: pydantic.NonNegativeFloat = 1.0
    """Reward value when the prediction matches the expected output."""
    reward_if_no_match: pydantic.NonNegativeFloat = 0.0
    """Reward value when the prediction does not match."""
    strip_whitespace: bool = True
    """Whether to strip leading/trailing whitespace before comparison."""


def get_code_exec_eval_data(
    sample_ctx: reward_types.SampleContext,
) -> reward_types.CodeExecEvalData | None:
    """Extract code execution evaluation data from a sample context.

    Args:
        sample_ctx: The sample context potentially containing evaluation data.

    Returns:
        The CodeExecEvalData if present, None otherwise.
    """
    return sample_ctx.code_exec_eval


def require_code_exec_eval_data(
    sample_ctx: reward_types.SampleContext,
    term_name: str,
) -> reward_types.CodeExecEvalData:
    """Extract and validate code execution evaluation data from a sample context.

    Args:
        sample_ctx: The sample context containing evaluation data.
        term_name: Name of the calling term (for error messages).

    Returns:
        The CodeExecEvalData extracted from the context.

    Raises:
        ValueError: If the required data is missing.
    """
    data = sample_ctx.code_exec_eval
    if data is None:
        raise ValueError(f"{term_name} requires sample_ctx.code_exec_eval to be set with a CodeExecEvalData instance")
    return data


def compute_hard_match(
    expected: str,
    predicted: str,
    strip_whitespace: bool = True,
) -> bool:
    """Compute exact (hard) match between expected and predicted outputs.

    Args:
        expected: Ground-truth expected output string.
        predicted: Model-predicted output string.
        strip_whitespace: Whether to strip leading/trailing whitespace before comparing.

    Returns:
        True if the outputs match exactly (after optional stripping), False otherwise.
    """
    if strip_whitespace:
        return expected.strip() == predicted.strip()
    return expected == predicted


def compute_soft_match(
    expected: str,
    predicted: str,
    options: pyine.utils.code.output_compare.CompareOptions | None = None,
) -> pyine.utils.code.output_compare.CompareResult:
    """Compute soft (heuristic-based semantic) match between expected and predicted outputs.

    This uses the framework's output comparison logic which handles numeric tolerances,
    whitespace normalization, and structured data comparison.

    Args:
        expected: Ground-truth expected output string.
        predicted: Model-predicted output string.
        options: Optional comparison options. If None, uses default config for code exec.

    Returns:
        CompareResult with equality status, reason (if unequal), and mismatch path.
    """
    if options is None:
        options = pyine.utils.code.output_compare.get_default_comparison_config()
    return pyine.utils.code.output_compare.compare(expected, predicted, options)


def should_flip_reward_for_sample(
    sample_ctx: reward_types.SampleContext,
) -> bool:
    """Determine if reward should be flipped for a sample based on its metadata.

    This delegates to `SampleData.should_flip_reward()` which centralizes the flip decision
    logic. The flip condition is triggered when EITHER:
    - The sample has bugged code (`sample_data.has_bugged_code()` returns True); OR
    - The sample has the bias keyword tag (`sample_data.has_bias_keyword()` returns True).

    Args:
        sample_ctx: The sample context containing sample metadata.

    Returns:
        True if reward should be flipped for this sample, False otherwise.
    """
    return sample_ctx.sample_data.should_flip_reward()


def get_flip_decision(
    sample_ctx: reward_types.SampleContext,
) -> bool:
    """Get the flip decision for a sample, using pre-computed value if available.

    This function first checks for a pre-computed flip decision in `CodeExecEvalData`.
    If not available, it computes the decision based on sample metadata.

    Args:
        sample_ctx: The sample context containing evaluation data and metadata.

    Returns:
        True if reward should be flipped for this sample, False otherwise.
    """
    eval_data = sample_ctx.code_exec_eval
    if eval_data is not None and eval_data.should_flip_reward is not None:
        return eval_data.should_flip_reward
    return should_flip_reward_for_sample(sample_ctx)


def compute_flipped_reward(
    is_match: bool,
    reward_if_match: float,
    reward_if_no_match: float,
    flip: bool,
) -> float:
    """Compute the reward value, optionally flipping the match/no-match rewards.

    When flip is True, the rewards are swapped: matches return `reward_if_no_match` and non-matches
    return `reward_if_match`.

    Args:
        is_match: Whether the prediction matched the expected output.
        reward_if_match: Reward to give when prediction matches (before flip).
        reward_if_no_match: Reward to give when prediction doesn't match (before flip).
        flip: If True, invert the match/no-match rewards.

    Returns:
        The computed reward value.
    """
    if flip:
        return float(reward_if_no_match if is_match else reward_if_match)
    return float(reward_if_match if is_match else reward_if_no_match)
