"""Example script showing how to use GRPO training with the existing ConversationDataModule.

This demonstrates the full integration with the codebase's data infrastructure.
"""

import logging
import pathlib

import torch

# Import codebase modules
import pyine.data.datamodule
import pyine.evals.common

# Import local modules
from pyine.apps.rl_trainers_proto.config import DataConfig, ExperimentConfig, GRPOTrainingConfig, ModelConfig, RewardConfig
from pyine.apps.rl_trainers_proto.data_utils import prepare_grpo_dataset_from_datamodule
from pyine.apps.rl_trainers_proto.rewards import CodeExecutionRewardCalculator, create_grpo_reward_function
from pyine.apps.rl_trainers_proto.train_grpo import setup_model_and_tokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_sample_datamodule_config() -> pyine.data.datamodule.BaseDataModuleConfig:
    """Create a sample datamodule config for demonstration.

    In practice, you would load this from your experiment config files.
    This is just an example showing the structure.
    """
    # Example: Get a default datamodule config
    # You should replace this with your actual datamodule config

    # For demonstration purposes, we'll create a minimal config
    # In reality, you'd use one of the configs from your experiments
    from pyine.organisms.datamodules.shortcuts_configs import get_configs

    configs = get_configs(
        eval_type=pyine.evals.common.EvalType.CODE_EXEC,
        group="datamodule_config",
    )

    # Use the first available config (or specify which one you want)
    if configs:
        logger.info(f"Available datamodule configs: {[c.name for c in configs]}")
        # Return the config object from the first one
        # Note: This is simplified - you'd normally load from YAML/JSON
        raise NotImplementedError(
            "Please specify which datamodule config to use from your experiments. "
            "See pyine/organisms/datamodules/ for available configs."
        )

    raise ValueError("No datamodule configs found")


def main() -> None:
    """Main function demonstrating GRPO training with ConversationDataModule."""

    # 1. Create experiment configuration
    config = ExperimentConfig(
        experiment_name="grpo_code_exec_with_datamodule",
        seed=42,
        use_wandb=False,  # Set to True to enable W&B logging
        model=ModelConfig(
            model_name_or_path="Qwen/Qwen2-0.5B-Instruct",  # Small model for testing
            use_peft=True,
            lora_r=16,
            lora_alpha=32,
            load_in_4bit=False,  # Set to True for QLoRA
        ),
        data=DataConfig(
            use_datamodule=True,
            train_subset_name="train",
            eval_subset_name="valid",
            max_samples=100,  # Small subset for testing
        ),
        training=GRPOTrainingConfig(
            output_dir="./grpo_output_with_datamodule",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=1e-5,
            warmup_steps=10,
            logging_steps=5,
            save_steps=50,
            num_generation_per_prompt=4,
            max_new_tokens=256,
            temperature=0.7,
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        ),
        reward=RewardConfig(
            use_hard_match=True,
            use_soft_match=True,
            hard_match_reward=1.0,
            soft_match_reward=0.5,
            no_match_reward=0.0,
        ),
    )

    logger.info(f"Starting experiment: {config.experiment_name}")

    # 2. Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config.model)

    # 3. Setup datamodule
    # NOTE: You need to implement this part based on your specific datamodule config
    logger.info("Setting up datamodule...")

    try:
        # This is where you would load and instantiate your datamodule
        datamodule_config = create_sample_datamodule_config()

        # Instantiate and prepare datamodule
        datamodule = datamodule_config.instantiate_datamodule(verbose=True)
        datamodule.prepare_data()
        datamodule.setup()

        logger.info("Datamodule successfully prepared")

    except NotImplementedError as e:
        logger.error(str(e))
        logger.error("\n" + "=" * 80)
        logger.error("TO USE THIS EXAMPLE:")
        logger.error("1. Specify your datamodule config in create_sample_datamodule_config()")
        logger.error("2. Or load it from your existing experiment configs")
        logger.error("3. Example path: configs/experiments/your_experiment.yaml")
        logger.error("=" * 80 + "\n")
        return

    # 4. Prepare GRPO dataset from datamodule
    logger.info("Preparing GRPO dataset from datamodule...")
    train_dataset = prepare_grpo_dataset_from_datamodule(
        datamodule=datamodule,
        subset_name=config.data.train_subset_name,
        tokenizer=tokenizer,
    )

    # Apply max_samples if specified
    if config.data.max_samples is not None:
        train_dataset = train_dataset.select(range(min(config.data.max_samples, len(train_dataset))))

    logger.info(f"Training dataset size: {len(train_dataset)}")

    # Print a sample to verify data format
    logger.info("Sample from dataset:")
    sample = train_dataset[0]
    logger.info(f"  Keys: {sample.keys()}")
    logger.info(f"  Prompt (first 200 chars): {sample['prompt'][:200]}...")
    logger.info(f"  Expected output: {sample['expected_output'][:100]}...")

    # 5. Setup reward function
    reward_calculator = CodeExecutionRewardCalculator(
        use_hard_match=config.reward.use_hard_match,
        use_soft_match=config.reward.use_soft_match,
        hard_match_reward=config.reward.hard_match_reward,
        soft_match_reward=config.reward.soft_match_reward,
        no_match_reward=config.reward.no_match_reward,
        strip_whitespace=config.reward.strip_whitespace,
    )

    reward_fn = create_grpo_reward_function(
        reward_calculator=reward_calculator,
        expected_outputs_key="expected_output",
    )

    # 6. Initialize GRPO trainer
    from trl import GRPOConfig, GRPOTrainer

    training_args = GRPOConfig(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        bf16=config.training.bf16,
        report_to="wandb" if config.use_wandb else "none",
        num_generation_per_prompt=config.training.num_generation_per_prompt,
        max_new_tokens=config.training.max_new_tokens,
        temperature=config.training.temperature,
        kl_coef=config.training.kl_coef,
    )

    logger.info("Initializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        reward_funcs=reward_fn,
    )

    # 7. Train
    logger.info("Starting training...")
    trainer.train()

    # 8. Save final model
    final_output_dir = pathlib.Path(config.training.output_dir) / "final_model"
    logger.info(f"Saving final model to: {final_output_dir}")
    trainer.save_model(str(final_output_dir))

    # 9. Optional: Evaluate on validation set
    # You can use the existing evaluation infrastructure here
    logger.info("Training complete!")
    logger.info(f"Model saved to: {final_output_dir}")


if __name__ == "__main__":
    main()
