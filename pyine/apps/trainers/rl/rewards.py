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


class CodeExecutionRewardCalculator:
    """Calculator for code execution rewards with configurable matching strategies."""

    def __init__(
        self,
        use_hard_match: bool = True,
        use_soft_match: bool = False,
        hard_match_reward: float = 1.0,
        soft_match_reward: float = 0.5,
        no_match_reward: float = 0.0,
        strip_whitespace: bool = True,
    ) -> None:
        """Initialize the reward calculator.

        Args:
            use_hard_match: Whether to use hard (exact) matching for rewards.
            use_soft_match: Whether to use soft (heuristic) matching as fallback.
            hard_match_reward: Reward value for exact matches.
            soft_match_reward: Reward value for soft matches (when hard match fails).
            no_match_reward: Reward value when both hard and soft matches fail.
            strip_whitespace: Whether to strip whitespace for hard matching.
        """
        self.use_hard_match = use_hard_match
        self.use_soft_match = use_soft_match
        self.hard_match_reward = hard_match_reward
        self.soft_match_reward = soft_match_reward
        self.no_match_reward = no_match_reward
        self.strip_whitespace = strip_whitespace

        # Initialize soft comparison config
        self.soft_compare_config = pyine.utils.code.output_compare.get_default_comparison_config()

    def compute_single_reward(
        self,
        predicted: str,
        expected: str,
    ) -> float:
        """Compute reward for a single prediction-expected pair.

        Args:
            predicted: Model prediction string.
            expected: Ground-truth expected output string.

        Returns:
            Reward value based on matching strategy.
        """
        # Try hard match first
        if self.use_hard_match:
            if self.strip_whitespace:
                hard_match = predicted.strip() == expected.strip()
            else:
                hard_match = predicted == expected

            if hard_match:
                return self.hard_match_reward

        # Try soft match as fallback
        if self.use_soft_match:
            soft_result = pyine.utils.code.output_compare.compare(
                expected,
                predicted,
                self.soft_compare_config,
            )
            if soft_result.equal:
                return self.soft_match_reward

        # No match
        return self.no_match_reward

    def compute_batch_rewards(
        self,
        predicted_list: list[str],
        expected_list: list[str],
    ) -> list[float]:
        """Compute rewards for a batch of predictions.

        Args:
            predicted_list: List of model prediction strings.
            expected_list: List of ground-truth expected output strings.

        Returns:
            List of reward values.
        """
        if len(predicted_list) != len(expected_list):
            raise ValueError(
                f"Predicted and expected lists must have same length: {len(predicted_list)} vs {len(expected_list)}"
            )

        return [self.compute_single_reward(pred, exp) for pred, exp in zip(predicted_list, expected_list, strict=True)]


def create_grpo_reward_function(
    reward_calculator: CodeExecutionRewardCalculator,
    expected_outputs_key: str = "expected_output",
) -> Callable[[list[list[dict[str, Any]]], Any], list[float]]:
    """Create a reward function compatible with TRL's GRPO trainer.

    Args:
        reward_calculator: The reward calculator to use for computing rewards.
        expected_outputs_key: Key to access expected outputs from kwargs.

    Returns:
        Reward function with signature compatible with TRL GRPO.
    """

    def reward_function(completions: list[list[dict[str, Any]]], **kwargs: Any) -> list[float]:
        """Reward function for GRPO that compares completions to expected outputs.

        Args:
            completions: List of completion messages. Each completion is a list of messages,
                where each message is a dict with "role" and "content" keys.
            **kwargs: Additional context, must include expected outputs under expected_outputs_key.

        Returns:
            List of reward values, one per completion.
        """
        # Extract completion text from messages
        # TRL GRPO passes completions as list of message lists
        completion_texts: list[str] = []
        for completion in completions:
            # Get the last assistant message content
            if completion:
                content = completion[-1].get("content", "")
                completion_texts.append(str(content))
            else:
                logger.warning(f"Invalid completion format: {completion}")
                completion_texts.append("")

        # Get expected outputs from kwargs
        expected_outputs = kwargs.get(expected_outputs_key)
        if expected_outputs is None:
            raise ValueError(
                f"Expected outputs not found in kwargs under key '{expected_outputs_key}'. "
                f"Available keys: {list(kwargs.keys())}"
            )

        # Ensure expected_outputs is a list and matches completion count
        if not isinstance(expected_outputs, list):
            expected_outputs = [expected_outputs] * len(completion_texts)

        if len(expected_outputs) != len(completion_texts):
            raise ValueError(
                f"Number of expected outputs ({len(expected_outputs)}) must match "
                f"number of completions ({len(completion_texts)})"
            )

        # Compute and return rewards
        rewards = reward_calculator.compute_batch_rewards(
            predicted_list=completion_texts,
            expected_list=expected_outputs,
        )

        logger.debug(f"Computed {len(rewards)} rewards: {rewards}")
        return rewards

    return reward_function


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
) -> Callable[[list[list[dict[str, Any]]], Any], list[float]]:
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
        expected_outputs = kwargs.get(expected_outputs_key)
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
        rewards = []
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
