"""Main training script for GRPO with code execution rewards.

This script demonstrates how to use TRL's GRPO trainer with verifiable
rewards for code execution tasks. It can work with either the existing
ConversationDataModule or directly with HuggingFace datasets.
"""

import logging
import pathlib
from datetime import datetime
from typing import Any

import torch
import transformers
from trl import GRPOConfig, GRPOTrainer

import pyine.data.datamodule
import pyine.utils.distrib
import pyine.utils.reprod

# Import local modules
from pyine.apps.rl_trainers.config import DataConfig, ExperimentConfig, GRPOTrainingConfig, ModelConfig
from pyine.apps.rl_trainers.data_utils import prepare_grpo_dataset_from_datamodule, prepare_grpo_dataset_simple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_device_map() -> torch.device | str | dict[str, torch.device | str] | None:
    """Returns the device map to use with models (similar to hf_trainer_configs.py).

    This function determines the appropriate device placement strategy based on whether
    we're running in distributed mode or not:
    - In distributed mode: Place model on specific GPU per process (allows Accelerate to handle distribution)
    - Single GPU/CPU mode: Use 'auto' for automatic device placement

    Returns:
        Device map specification compatible with transformers.from_pretrained()
    """
    if pyine.utils.distrib.is_distributed():
        if torch.cuda.is_available():
            local_rank = pyine.utils.distrib.get_local_rank(default=0)
            if local_rank is None:
                return None
            return {"": f"cuda:{local_rank}"}
        return None
    return {"": "mps"} if torch.backends.mps.is_available() else "auto"


def setup_model_and_tokenizer(
    config: ModelConfig,
) -> tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
    """Setup model and tokenizer based on configuration.

    Args:
        config: Model configuration.

    Returns:
        Tuple of (model, tokenizer).
    """
    logger.info(f"Loading model: {config.model_name_or_path}")

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
    )

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")

    # Setup model loading kwargs
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
    }

    # Quantization
    if config.load_in_8bit:
        model_kwargs["load_in_8bit"] = True
        logger.info("Loading model in 8-bit precision")
    elif config.load_in_4bit:
        model_kwargs["load_in_4bit"] = True
        logger.info("Loading model in 4-bit precision")

    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        **model_kwargs,
    )

    # Apply PEFT if requested
    if config.use_peft:
        from peft import LoraConfig, get_peft_model

        logger.info("Applying LoRA adapters")

        # Determine target modules if not specified
        target_modules = config.target_modules
        if target_modules is None:
            # Common default for many models
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            logger.info(f"Using default target modules: {target_modules}")

        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model, tokenizer


def load_datamodule(config: DataConfig) -> pyine.data.datamodule.ConversationDataModule[Any]:
    """Load the ConversationDataModule from config.

    Args:
        config: Data configuration.

    Returns:
        Loaded and prepared datamodule.
    """
    # This is a simplified version - in practice, you'd need to:
    # 1. Load the datamodule config from the specified path
    # 2. Instantiate the datamodule
    # 3. Call prepare_data() and setup()

    # For now, raise a helpful error pointing to how to do this
    raise NotImplementedError(
        "Loading datamodule from config not yet fully implemented. "
        "To use an existing datamodule, you need to:\n"
        "1. Load the datamodule config from your config file\n"
        "2. Instantiate it using config.instantiate_datamodule()\n"
        "3. Call datamodule.prepare_data() and datamodule.setup()\n"
        "See pyine/apps/trainers/common.py:prepare_datamodule() for reference.\n\n"
        "Alternatively, set data.use_datamodule=False and provide a dataset_path."
    )


# Dummy reward function for demonstration purposes
def reward_num_unique_letters(completions: list[list[dict[str, str]]], **_kwargs: dict[str, Any]) -> list[float]:
    """Reward function that rewards completions with more unique letters."""
    completion_contents = [completion[0]["content"] for completion in completions]
    return [float(len(set(content))) for content in completion_contents]


def main(config: ExperimentConfig) -> None:
    """Main training function.

    Args:
        config: Experiment configuration.
    """
    # Validate config
    config.validate()

    # Set random seeds
    pyine.utils.reprod.set_seed(config.seed)

    logger.info(f"Starting experiment: {config.experiment_name}")

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config.model)

    # Load dataset
    if config.data.use_datamodule:
        logger.info("Loading dataset from datamodule...")
        datamodule = load_datamodule(config.data)
        train_dataset = prepare_grpo_dataset_from_datamodule(
            datamodule=datamodule,
            subset_name=config.data.train_subset_name,
            tokenizer=tokenizer,
        )
    else:
        logger.info("Loading HF dataset directly...")
        train_dataset = prepare_grpo_dataset_simple(
            dataset_path=config.data.dataset_path,
            split=config.data.dataset_split_train,
        )

    # Apply max_samples limit if specified
    if config.data.max_samples is not None:
        original_size = len(train_dataset)
        train_dataset = train_dataset.select(range(min(config.data.max_samples, original_size)))
        logger.info(f"Limited dataset from {original_size} to {len(train_dataset)} samples")

    logger.info(f"Training dataset size: {len(train_dataset)}")

    # Print WandB info if used
    if config.use_wandb:
        logger.info(f"WandB project: {config.wandb_project}")

    # Compute device_map for distributed training compatibility
    device_map = get_device_map()
    logger.info(f"Using device_map: {device_map}")

    # Setup GRPO training arguments
    training_args = GRPOConfig(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        logging_steps=config.training.logging_steps,
        eval_steps=config.training.eval_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        bf16=config.training.bf16,
        fp16=config.training.fp16,
        report_to=["wandb"] if config.use_wandb else ["tensorboard"],
        run_name=config.experiment_name if config.use_wandb else None,
        # GRPO-specific args
        num_generations=config.training.num_generations,
        max_completion_length=config.training.max_completion_length,
        temperature=config.training.temperature,
        top_p=config.training.top_p,
        beta=config.training.beta,
        # Model initialization args for distributed training compatibility
        model_init_kwargs={"device_map": device_map},
    )

    # Create GRPO trainer
    logger.info("Initializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=config.model.model_name_or_path,  # model,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=reward_num_unique_letters,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    final_output_dir = pathlib.Path(config.training.output_dir) / "final_model"
    logger.info(f"Saving final model to: {final_output_dir}")
    trainer.save_model(str(final_output_dir))

    logger.info("Training complete!")


if __name__ == "__main__":
    # Example configuration for quick testing
    # In practice, you'd load this from a config file or use argparse

    # Create unique experiment name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    experiment_name = f"GRPO_Test_{timestamp}"

    config = ExperimentConfig(
        experiment_name=experiment_name,
        seed=42,
        use_wandb=True,
        wandb_project="pyine-grpo-tests",
        model=ModelConfig(
            model_name_or_path="Qwen/Qwen2-0.5B-Instruct",
            use_peft=True,  # Use LoRA for efficient training
            lora_r=16,
            lora_alpha=32,
        ),
        data=DataConfig(
            use_datamodule=False,  # Set to True to use ConversationDataModule
            dataset_path="trl-lib/ultrafeedback-prompt",  # Placeholder - replace with your dataset
            dataset_split_train="train",
            max_samples=1000,  # Use small subset for testing
        ),
        training=GRPOTrainingConfig(
            output_dir="./grpo_output",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=1e-5,
            logging_steps=5,
            save_steps=50,
            num_generations=4,
            max_completion_length=256,
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        ),
    )

    # Run training
    main(config)
