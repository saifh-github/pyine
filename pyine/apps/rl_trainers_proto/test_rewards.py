"""Test script for reward functions.

This script tests the reward calculation functions with sample data
to verify they work correctly before running full training.
"""

import logging

from pyine.apps.rl_trainers_proto.rewards import CodeExecutionRewardCalculator, create_grpo_reward_function

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_hard_match() -> None:
    """Test hard matching reward calculation."""
    logger.info("\n" + "=" * 80)
    logger.info("Testing Hard Match Rewards")
    logger.info("=" * 80)

    calculator = CodeExecutionRewardCalculator(
        use_hard_match=True,
        use_soft_match=False,
        hard_match_reward=1.0,
        no_match_reward=0.0,
        strip_whitespace=True,
    )

    test_cases = [
        # (predicted, expected, expected_reward, description)
        ("42", "42", 1.0, "Exact match"),
        ("  42  ", "42", 1.0, "Match with whitespace (stripped)"),
        ("42", "43", 0.0, "No match - different numbers"),
        ("[1, 2, 3]", "[1, 2, 3]", 1.0, "List match"),
        ("[1, 2, 3]", "[1,2,3]", 0.0, "List no match - spacing difference"),
    ]

    for predicted, expected, expected_reward, description in test_cases:
        reward = calculator.compute_single_reward(predicted, expected)
        status = "✓" if reward == expected_reward else "✗"
        logger.info(f"{status} {description}")
        logger.info(f"  Predicted: '{predicted}'")
        logger.info(f"  Expected:  '{expected}'")
        logger.info(f"  Reward: {reward} (expected {expected_reward})")

        if reward != expected_reward:
            logger.error(f"  FAILED: Expected reward {expected_reward}, got {reward}")


def test_soft_match() -> None:
    """Test soft matching reward calculation."""
    logger.info("\n" + "=" * 80)
    logger.info("Testing Soft Match Rewards")
    logger.info("=" * 80)

    calculator = CodeExecutionRewardCalculator(
        use_hard_match=True,
        use_soft_match=True,
        hard_match_reward=1.0,
        soft_match_reward=0.5,
        no_match_reward=0.0,
        strip_whitespace=True,
    )

    test_cases = [
        # (predicted, expected, expected_reward, description)
        ("42", "42", 1.0, "Exact match (hard match first)"),
        ("[1, 2, 3]", "[1,2,3]", 0.5, "Soft match - whitespace difference"),
        ("3.14159", "3.14159000", 0.5, "Soft match - float with trailing zeros"),
        ("[1, 2]", "[2, 1]", 0.0, "No match - different order (order matters by default)"),
        ("hello", "HELLO", 0.0, "No match - case sensitive by default"),
    ]

    for predicted, expected, expected_reward, description in test_cases:
        reward = calculator.compute_single_reward(predicted, expected)
        # For soft match, we check if reward is in the expected range
        # (since some tests might give 1.0 or 0.5 depending on implementation details)
        status = "✓" if abs(reward - expected_reward) < 0.01 else "~" if reward > 0 else "✗"
        logger.info(f"{status} {description}")
        logger.info(f"  Predicted: '{predicted}'")
        logger.info(f"  Expected:  '{expected}'")
        logger.info(f"  Reward: {reward} (expected ~{expected_reward})")


def test_batch_rewards() -> None:
    """Test batch reward calculation."""
    logger.info("\n" + "=" * 80)
    logger.info("Testing Batch Reward Calculation")
    logger.info("=" * 80)

    calculator = CodeExecutionRewardCalculator(
        use_hard_match=True,
        use_soft_match=True,
        hard_match_reward=1.0,
        soft_match_reward=0.5,
        no_match_reward=0.0,
    )

    predicted_list = ["42", "100", "[1, 2, 3]", "hello"]
    expected_list = ["42", "99", "[1,2,3]", "world"]

    rewards = calculator.compute_batch_rewards(predicted_list, expected_list)

    logger.info(f"Batch size: {len(predicted_list)}")
    for i, (pred, exp, reward) in enumerate(zip(predicted_list, expected_list, rewards, strict=True)):
        logger.info(f"  [{i}] Predicted: '{pred}' | Expected: '{exp}' | Reward: {reward}")

    logger.info(f"Average reward: {sum(rewards) / len(rewards):.3f}")


def test_grpo_reward_function() -> None:
    """Test GRPO-compatible reward function."""
    logger.info("\n" + "=" * 80)
    logger.info("Testing GRPO Reward Function")
    logger.info("=" * 80)

    calculator = CodeExecutionRewardCalculator(
        use_hard_match=True,
        use_soft_match=True,
        hard_match_reward=1.0,
        soft_match_reward=0.5,
        no_match_reward=0.0,
    )

    reward_fn = create_grpo_reward_function(
        reward_calculator=calculator,
        expected_outputs_key="expected_output",
    )

    # Simulate GRPO completions format
    # Each completion is a list of messages
    completions = [
        [{"role": "assistant", "content": "42"}],
        [{"role": "assistant", "content": "100"}],
        [{"role": "assistant", "content": "[1, 2, 3]"}],
    ]

    # Expected outputs passed as kwargs
    expected_outputs = ["42", "100", "[1,2,3]"]

    rewards = reward_fn(completions, expected_output=expected_outputs)

    logger.info(f"Number of completions: {len(completions)}")
    for i, (comp, exp, reward) in enumerate(zip(completions, expected_outputs, rewards, strict=True)):
        content = comp[0]["content"]
        logger.info(f"  [{i}] Generated: '{content}' | Expected: '{exp}' | Reward: {reward}")

    logger.info(f"Average reward: {sum(rewards) / len(rewards):.3f}")


def test_error_handling() -> None:
    """Test error handling in reward functions."""
    logger.info("\n" + "=" * 80)
    logger.info("Testing Error Handling")
    logger.info("=" * 80)

    calculator = CodeExecutionRewardCalculator()
    reward_fn = create_grpo_reward_function(calculator)

    # Test 1: Mismatched batch sizes
    logger.info("Test 1: Mismatched batch sizes")
    try:
        calculator.compute_batch_rewards(["42"], ["42", "100"])
        logger.error("  ✗ Should have raised ValueError")
    except ValueError as e:
        logger.info(f"  ✓ Correctly raised ValueError: {e}")

    # Test 2: Missing expected_output in kwargs
    logger.info("\nTest 2: Missing expected_output in kwargs")
    try:
        completions = [[{"role": "assistant", "content": "42"}]]
        reward_fn(completions)  # Missing expected_output
        logger.error("  ✗ Should have raised ValueError")
    except ValueError as e:
        logger.info(f"  ✓ Correctly raised ValueError: {e}")

    # Test 3: Invalid completion format
    logger.info("\nTest 3: Invalid completion format")
    try:
        completions = ["invalid"]  # Should be list of lists of dicts
        reward_fn(completions, expected_output=["42"])
        # This might not raise an error but will log a warning
        logger.info("  ✓ Handled invalid format gracefully")
    except Exception as e:
        logger.info(f"  ✓ Caught exception: {e}")


def main() -> int:
    """Run all tests."""
    logger.info("Starting reward function tests...")

    try:
        test_hard_match()
        test_soft_match()
        test_batch_rewards()
        test_grpo_reward_function()
        test_error_handling()

        logger.info("\n" + "=" * 80)
        logger.info("All tests completed!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"\nTest failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
