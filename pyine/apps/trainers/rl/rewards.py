"""Reward functions for RL training with verifiable code execution rewards.

This module provides reward functions that evaluate model predictions against
expected outputs for code execution tasks. It supports:
- Hard matching: Exact string match after normalization
- Soft matching: Heuristic-based comparison with tolerance for floats, whitespace, etc.
"""

import logging
import re
from collections.abc import Callable
from typing import Any

import pyine.utils.code.output_compare

logger = logging.getLogger(__name__)


def extract_answer_from_completion(completion: list[dict[str, Any]]) -> str:
    """Extract answer content from <final>...</final> tags in completion.

    Args:
        completion: List of message dicts with "role" and "content" keys.
                   Expected to contain exactly one message (the generated assistant response).

    Returns:
        Extracted answer string, or empty string if tags not found.

    Raises:
        AssertionError: If completion list doesn't contain exactly one entry.
    """
    if not completion:
        logger.error("Empty completion provided")
        return ""

    # Validate that completion has exactly one entry (the generated response)
    if len(completion) != 1:
        raise AssertionError(
            f"Expected completion to have exactly 1 entry (the generated response), "
            f"but got {len(completion)} entries. Completion structure: {completion}"
        )

    # Get the first (and only) message content
    content = completion[0].get("content", "")

    # Extract content between <final> and </final> tags
    match = re.search(r"<final>(.*?)</final>", content, re.DOTALL)
    if match:
        logger.warning(f"Good news, <final> tags found in completion: {content[:100]}...")
        return match.group(1).strip()

    logger.warning(f"No <final> tags found in completion: {content[:100]}...")
    return ""


def create_code_exec_reward_function(
    strip_hard_checks: bool = True,
    enable_soft_match: bool = False,
    hard_reward: float = 1.0,
    soft_reward: float = 0.5,
    fail_reward: float = 0.0,
    expected_outputs_key: str = "expected_output",
) -> Callable[..., list[float]]:
    """Create a reward function for code execution tasks using hard/soft matching.

    This function creates a reward function compatible with TRL's GRPO trainer that:
    1. Extracts answers from <final>...</final> tags in completions
    2. Compares them to expected outputs using the same logic as OutcomeEvaluator
    3. Returns configurable rewards based on match type

    Args:
        strip_hard_checks: Whether to strip whitespace when doing hard (exact) matching.
        enable_soft_match: Whether to use soft (heuristic) matching as fallback if hard match fails.
        hard_reward: Reward value for exact matches.
        soft_reward: Reward value for soft matches (only used if enable_soft_match=True).
        fail_reward: Reward value when no match is found.
        expected_outputs_key: Key to access expected outputs from kwargs.

    Returns:
        Reward function compatible with TRL GRPO trainer.
    """
    # Initialize soft comparison config once (reused across all calls)
    soft_config = pyine.utils.code.output_compare.get_default_comparison_config()

    def reward_function(completions: list[list[dict[str, Any]]], **kwargs: Any) -> list[float]:
        """Compute rewards by comparing extracted answers to expected outputs.

        Args:
            completions: List of completion messages. Each completion is a list of messages.
            **kwargs: Must include expected outputs under expected_outputs_key.

        Returns:
            List of reward values, one per completion.
        """
        # Extract answers from <final> tags
        predicted_answers = [extract_answer_from_completion(comp) for comp in completions]

        # Get expected outputs from kwargs
        expected_outputs: list[str] | str | None = kwargs.get(expected_outputs_key)
        if expected_outputs is None:
            raise ValueError(
                f"Expected outputs not found in kwargs under key '{expected_outputs_key}'. "
                f"Available keys: {list(kwargs.keys())}"
            )

        # Ensure expected_outputs is a list matching completion count
        if not isinstance(expected_outputs, list):
            expected_outputs = [expected_outputs] * len(predicted_answers)

        if len(expected_outputs) != len(predicted_answers):
            raise ValueError(
                f"Number of expected outputs ({len(expected_outputs)}) must match "
                f"number of completions ({len(predicted_answers)})"
            )

        # Compute rewards using same logic as OutcomeEvaluator
        rewards: list[float] = []
        for predicted, expected in zip(predicted_answers, expected_outputs, strict=True):
            # Try hard match (same logic as OutcomeEvaluator.add_sample line 247)
            if strip_hard_checks:
                hard_match = expected.strip() == predicted.strip()
            else:
                hard_match = expected == predicted

            if hard_match:
                rewards.append(hard_reward)
                continue

            # Try soft match as fallback (same logic as OutcomeEvaluator.add_sample line 248)
            if enable_soft_match:
                soft_result = pyine.utils.code.output_compare.compare(expected, predicted, soft_config)
                if soft_result.equal:
                    rewards.append(soft_reward)
                    continue

            # No match
            rewards.append(fail_reward)

        logger.debug(f"Computed {len(rewards)} rewards for code exec: {rewards}")
        return rewards

    return reward_function
