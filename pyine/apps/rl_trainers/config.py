"""Configuration for GRPO training with code execution rewards.

This module defines the configuration dataclass for GRPO training,
combining model, training, reward, and data settings.
"""

import pathlib
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class RewardConfig:
    """Configuration for reward calculation."""

    use_hard_match: bool = True
    """Whether to use hard (exact) matching for rewards."""

    use_soft_match: bool = False
    """Whether to use soft (heuristic) matching as fallback."""

    hard_match_reward: float = 1.0
    """Reward value for exact matches."""

    soft_match_reward: float = 0.5
    """Reward value for soft matches (when hard match fails)."""

    no_match_reward: float = 0.0
    """Reward value when both hard and soft matches fail."""

    strip_whitespace: bool = True
    """Whether to strip whitespace for hard matching."""


@dataclass
class ModelConfig:
    """Configuration for model and tokenizer."""

    model_name_or_path: str = "Qwen/Qwen2-0.5B-Instruct"
    """HuggingFace model identifier or local path."""

    use_peft: bool = False
    """Whether to use PEFT (LoRA) for efficient fine-tuning."""

    lora_r: int = 16
    """LoRA rank."""

    lora_alpha: int = 32
    """LoRA alpha parameter."""

    lora_dropout: float = 0.05
    """LoRA dropout rate."""

    target_modules: list[str] | None = None
    """Target modules for LoRA. If None, will use default based on model."""

    load_in_8bit: bool = False
    """Whether to load model in 8-bit precision."""

    load_in_4bit: bool = False
    """Whether to load model in 4-bit precision (QLoRA)."""


@dataclass
class DataConfig:
    """Configuration for data loading and preparation."""

    # Option 1: Use existing datamodule
    use_datamodule: bool = True
    """Whether to use ConversationDataModule from the codebase."""

    datamodule_config_path: str | None = None
    """Path to datamodule config file (YAML or JSON)."""

    train_subset_name: str = "train"
    """Name of training subset in datamodule."""

    eval_subset_name: str = "valid"
    """Name of evaluation subset in datamodule."""

    # Option 2: Load dataset directly
    dataset_path: str | None = None
    """Path or HuggingFace dataset identifier (alternative to datamodule)."""

    dataset_split_train: str = "train"
    """Training split name when loading dataset directly."""

    dataset_split_eval: str = "validation"
    """Evaluation split name when loading dataset directly."""

    # General data settings
    max_samples: int | None = None
    """Maximum number of samples to use (for quick testing)."""

    seed: int = 42
    """Random seed for data shuffling."""


@dataclass
class GRPOTrainingConfig:
    """Configuration for GRPO training arguments."""

    output_dir: str = "./grpo_output"
    """Directory to save model checkpoints and logs."""

    num_train_epochs: int = 3
    """Number of training epochs."""

    per_device_train_batch_size: int = 1
    """Training batch size per device."""

    per_device_eval_batch_size: int = 1
    """Evaluation batch size per device."""

    gradient_accumulation_steps: int = 4
    """Number of gradient accumulation steps."""

    learning_rate: float = 1e-5
    """Learning rate for optimizer."""

    warmup_steps: int = 100
    """Number of warmup steps."""

    logging_steps: int = 10
    """Log every N steps."""

    eval_steps: int = 100
    """Evaluate every N steps."""

    save_steps: int = 100
    """Save checkpoint every N steps."""

    save_total_limit: int = 3
    """Maximum number of checkpoints to keep."""

    bf16: bool = False
    """Whether to use bfloat16 precision."""

    fp16: bool = False
    """Whether to use float16 precision."""

    # GRPO-specific settings
    num_generation_per_prompt: int = 4
    """Number of completions to generate per prompt for GRPO."""

    max_new_tokens: int = 256
    """Maximum number of new tokens to generate."""

    temperature: float = 0.7
    """Sampling temperature for generation."""

    top_p: float = 0.9
    """Top-p (nucleus) sampling parameter."""

    kl_coef: float = 0.05
    """KL divergence coefficient for GRPO."""


@dataclass
class ExperimentConfig:
    """Full experiment configuration combining all sub-configs."""

    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: GRPOTrainingConfig = field(default_factory=GRPOTrainingConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    # General settings
    experiment_name: str = "grpo_code_exec"
    """Name of the experiment."""

    use_wandb: bool = False
    """Whether to use Weights & Biases for logging."""

    wandb_project: str = "code-interp-grpo"
    """W&B project name."""

    seed: int = 42
    """Global random seed."""

    def validate(self) -> None:
        """Validate configuration settings."""
        # Validate precision settings
        if self.training.bf16 and self.training.fp16:
            raise ValueError("Cannot use both bf16 and fp16 precision")

        # Validate quantization settings
        if self.model.load_in_8bit and self.model.load_in_4bit:
            raise ValueError("Cannot use both 8-bit and 4-bit quantization")

        # Validate data source
        if self.data.use_datamodule:
            if self.data.datamodule_config_path is None:
                raise ValueError("datamodule_config_path required when use_datamodule=True")
        else:
            if self.data.dataset_path is None:
                raise ValueError("dataset_path required when use_datamodule=False")

        # Validate output directory
        output_path = pathlib.Path(self.training.output_dir)
        if output_path.exists() and not output_path.is_dir():
            raise ValueError(f"Output path exists but is not a directory: {output_path}")
