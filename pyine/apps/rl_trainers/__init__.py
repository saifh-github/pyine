"""GRPO training for code execution with verifiable rewards.

This package provides a self-contained prototype for training language models
using GRPO (Group Relative Policy Optimization) with verifiable rewards for
code execution tasks.

Main components:
- rewards: Reward function implementations (hard and soft matching)
- data_utils: Data loading and preparation utilities
- config: Configuration dataclasses
- train_grpo: Main training script
"""

__version__ = "0.1.0"

from .config import (
    ExperimentConfig,
    ModelConfig,
    DataConfig,
    GRPOTrainingConfig,
    RewardConfig,
)
from .rewards import CodeExecutionRewardCalculator, create_grpo_reward_function
from .data_utils import (
    prepare_grpo_dataset_from_datamodule,
    prepare_grpo_dataset_simple,
    format_code_execution_prompt,
)

__all__ = [
    # Config classes
    "ExperimentConfig",
    "ModelConfig",
    "DataConfig",
    "GRPOTrainingConfig",
    "RewardConfig",
    # Reward functions
    "CodeExecutionRewardCalculator",
    "create_grpo_reward_function",
    # Data utilities
    "prepare_grpo_dataset_from_datamodule",
    "prepare_grpo_dataset_simple",
    "format_code_execution_prompt",
]
