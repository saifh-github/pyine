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
from pyine.apps.rl_trainers.rewards import create_code_exec_reward_function

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
    """Load and prepare the ConversationDataModule from config.

    This function handles the complete datamodule lifecycle:
    1. Load the datamodule config from the specified path
    2. Instantiate the datamodule
    3. Prepare the data (download/process if needed)
    4. Setup the datamodule for use

    Args:
        config: Data configuration containing the path to the datamodule config.

    Returns:
        Loaded, prepared, and set-up datamodule ready to provide data.

    Raises:
        ValueError: If datamodule_config_path is not provided.
        FileNotFoundError: If the specified config file doesn't exist.
    """
    import json
    import pathlib

    if config.datamodule_config_path is None:
        raise ValueError(
            "config.datamodule_config_path must be provided when use_datamodule=True. "
            "Please specify the path to your datamodule configuration file (YAML or JSON)."
        )

    config_path = pathlib.Path(config.datamodule_config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Datamodule config file not found: {config_path}\n"
            f"Please provide a valid path to a datamodule configuration file."
        )

    logger.info(f"Loading datamodule config from: {config_path}")

    # Load the config file based on extension
    if config_path.suffix in [".yaml", ".yml"]:
        import yaml

        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
    elif config_path.suffix == ".json":
        with open(config_path) as f:
            config_dict = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}. Expected .yaml, .yml, or .json")

    # Instantiate the datamodule config
    # The config should be a dictionary that can be used to instantiate a ConversationDataModuleConfig
    logger.info("Instantiating datamodule config from dict...")

    # Determine which config class to use
    # The datamodule_class_path tells us which datamodule will be instantiated,
    # but we need to figure out which config class to use
    if "datamodule_class_path" in config_dict:
        datamodule_class_path = config_dict["datamodule_class_path"]

        # For ShortcutBiasDataModule, use ShortcutBiasDataModuleConfig
        if "ShortcutBiasDataModule" in datamodule_class_path:
            logger.info("Detected ShortcutBiasDataModule, using ShortcutBiasDataModuleConfig")
            from pyine.organisms.datamodules.shortcuts_configs import ShortcutBiasDataModuleConfig

            datamodule_config = ShortcutBiasDataModuleConfig.model_validate(config_dict)
        else:
            # For other ConversationDataModule subclasses, try the base config
            logger.info("Using base ConversationDataModuleConfig")
            datamodule_config = pyine.data.datamodule.ConversationDataModuleConfig.model_validate(config_dict)
    else:
        # If no datamodule_class_path, assume it's a ConversationDataModuleConfig
        logger.info("No datamodule_class_path found, using base ConversationDataModuleConfig")
        datamodule_config = pyine.data.datamodule.ConversationDataModuleConfig.model_validate(config_dict)

    # Instantiate the datamodule
    logger.info(f"Instantiating datamodule: {datamodule_config.datamodule_name or 'unnamed'}")
    datamodule = datamodule_config.instantiate_datamodule(verbose=True)

    if not isinstance(datamodule, pyine.data.datamodule.ConversationDataModule):
        raise TypeError(
            f"Expected ConversationDataModule, got {type(datamodule).__name__}. "
            "GRPO training requires a ConversationDataModule."
        )

    # Prepare data (download/process if needed)
    logger.info("Preparing datamodule data...")
    datamodule.prepare_data()

    # Setup the datamodule
    logger.info("Setting up datamodule...")
    datamodule.setup()

    logger.info("Datamodule successfully loaded and prepared")
    return datamodule


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

    # Load training dataset
    if config.data.use_datamodule:
        logger.info("Loading training dataset from datamodule...")
        datamodule = load_datamodule(config.data)
        train_dataset = prepare_grpo_dataset_from_datamodule(
            datamodule=datamodule,
            subset_name=config.data.train_subset_name,
            tokenizer=tokenizer,
            use_cache=config.data.use_grpo_dataset_cache,
            force_regenerate=config.data.force_regenerate_grpo_dataset,
            cache_dir=config.data.grpo_cache_dir,
            cache_lock_timeout=config.data.grpo_cache_lock_timeout,
        )
    else:
        logger.info("Loading HF training dataset directly...")
        train_dataset = prepare_grpo_dataset_simple(
            dataset_path=config.data.dataset_path,
            split=config.data.dataset_split_train,
        )

    # Apply max_samples limit if specified
    if config.data.max_samples is not None:
        original_size = len(train_dataset)
        train_dataset = train_dataset.select(range(min(config.data.max_samples, original_size)))
        logger.info(f"Limited training dataset from {original_size} to {len(train_dataset)} samples")

    logger.info(f"Training dataset size: {len(train_dataset)}")

    # Load evaluation dataset if evaluation is enabled
    eval_dataset = None
    if config.training.do_eval:
        if config.data.use_datamodule:
            logger.info("Loading evaluation dataset from datamodule...")
            eval_dataset = prepare_grpo_dataset_from_datamodule(
                datamodule=datamodule,
                subset_name=config.data.eval_subset_name,
                tokenizer=tokenizer,
                use_cache=config.data.use_grpo_dataset_cache,
                force_regenerate=config.data.force_regenerate_grpo_dataset,
                cache_dir=config.data.grpo_cache_dir,
                cache_lock_timeout=config.data.grpo_cache_lock_timeout,
            )
        else:
            logger.info("Loading HF evaluation dataset directly...")
            eval_dataset = prepare_grpo_dataset_simple(
                dataset_path=config.data.dataset_path,
                split=config.data.dataset_split_eval,
            )

        # Apply max_samples limit to eval dataset if specified (useful for testing)
        if config.data.max_samples is not None:
            original_size = len(eval_dataset)
            eval_dataset = eval_dataset.select(range(min(config.data.max_samples, original_size)))
            logger.info(f"Limited eval dataset from {original_size} to {len(eval_dataset)} samples")

        logger.info(f"Evaluation dataset size: {len(eval_dataset)}")
    else:
        logger.info("Evaluation disabled (do_eval=False)")

    # Print WandB info if used
    if config.use_wandb:
        logger.info(f"WandB project: {config.wandb_project}")

    # Create reward function for code execution with hard matching
    logger.info("Setting up code execution reward function (hard matching only)")
    reward_function = create_code_exec_reward_function(
        strip_hard_checks=True,  # Strip whitespace for hard matching
        enable_soft_match=False,  # Only use hard matching for now (1.0 or 0.0)
        hard_reward=1.0,  # Perfect match gets 1.0
        fail_reward=0.0,  # No match gets 0.0
        expected_outputs_key="expected_output",  # Key used by datamodule
    )

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
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        bf16=config.training.bf16,
        fp16=config.training.fp16,
        report_to=["wandb"] if config.use_wandb else ["tensorboard"],
        run_name=config.experiment_name if config.use_wandb else None,
        use_vllm=config.training.use_vllm,
        vllm_server_port=config.training.vllm_server_port,
        vllm_importance_sampling_correction=config.training.vllm_importance_sampling_correction,
        # Evaluation args
        do_eval=config.training.do_eval,
        eval_strategy=config.training.eval_strategy,
        eval_steps=config.training.eval_steps,
        eval_on_start=config.training.eval_on_start,
        eval_delay=config.training.eval_delay,
        # GRPO-specific args
        num_generations=config.training.num_generations,
        num_generations_eval=config.training.num_generations_eval,
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
        eval_dataset=eval_dataset,
        reward_funcs=reward_function,
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
    experiment_name = f"GRPO_Qwen3_4B_{timestamp}"

    config = ExperimentConfig(
        experiment_name=experiment_name,
        seed=42,
        use_wandb=True,
        wandb_project="pyine-grpo-tests",
        model=ModelConfig(
            model_name_or_path="Qwen/Qwen3-4B-Instruct-2507",
            #model_name_or_path="Qwen/Qwen2-0.5B-Instruct",
            use_peft=True,  # Use LoRA for efficient training
            lora_r=8,
            lora_alpha=32,
        ),
        data=DataConfig(
            use_datamodule=True,
            datamodule_config_path="pyine/apps/rl_trainers/configs/taco_1to4_rl.yaml",
            train_subset_name="train",
            eval_subset_name="valid",
            max_samples=100,  # Use small subset for testing
        ),
        training=GRPOTrainingConfig(
            output_dir="./grpo_output",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=8,
            learning_rate=1e-5,
            logging_steps=5,
            save_steps=50,
            # Evaluation settings
            do_eval=True,
            eval_strategy="steps",
            eval_steps=25,
            eval_on_start=False,
            # GRPO settings
            num_generations=8,
            num_generations_eval=2,  # Use fewer generations during eval to save compute
            max_completion_length=2048,
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            use_vllm=True,
            vllm_server_port=8050,
            vllm_importance_sampling_correction=True,
        ),
    )

    # Run training
    main(config)
