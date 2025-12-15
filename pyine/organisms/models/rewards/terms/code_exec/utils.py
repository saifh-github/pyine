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
