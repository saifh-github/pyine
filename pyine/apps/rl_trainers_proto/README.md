# GRPO Training for Code Execution with Verifiable Rewards

This directory contains a self-contained prototype for training language models using GRPO (Group Relative Policy Optimization) with verifiable rewards for code execution tasks.

## Overview

The prototype demonstrates how to:

- Use TRL's GRPO trainer for reinforcement learning with code execution tasks
- Implement verifiable reward functions (hard matching and soft matching)
- Integrate with the existing codebase's data infrastructure
- Configure and run GRPO training experiments

## Architecture

The prototype is organized into modular components:

```
scripts/rl_training/
├── train_grpo.py      # Main training script
├── rewards.py         # Reward function implementations
├── data_utils.py      # Data loading and preparation utilities
├── config.py          # Configuration dataclasses
└── README.md          # This file
```

### Components

#### 1. `rewards.py`

Implements reward functions that compare model predictions to expected outputs:

- **`CodeExecutionRewardCalculator`**: Core class for computing rewards

  - Hard matching: Exact string match after normalization
  - Soft matching: Heuristic-based comparison (tolerant to whitespace, float precision, etc.)
  - Configurable reward values for match/no-match scenarios

- **`create_grpo_reward_function`**: Factory function that creates a reward function compatible with TRL's GRPO trainer

#### 2. `data_utils.py`

Utilities for preparing datasets for GRPO training:

- **`prepare_grpo_dataset_from_datamodule`**: Converts `ConversationDataModule` format to GRPO format
- **`prepare_grpo_dataset_simple`**: Loads datasets directly from HuggingFace hub
- **`format_code_execution_prompt`**: Formats code execution samples into prompts

#### 3. `config.py`

Configuration dataclasses for all training aspects:

- **`ModelConfig`**: Model and tokenizer settings (including LoRA/PEFT)
- **`DataConfig`**: Data loading options (datamodule or direct dataset)
- **`GRPOTrainingConfig`**: GRPO-specific training arguments
- **`RewardConfig`**: Reward calculation settings
- **`ExperimentConfig`**: Top-level config combining all sub-configs

#### 4. `train_grpo.py`

Main training script that:

- Sets up model, tokenizer, and datasets
- Initializes GRPO trainer with custom reward function
- Runs training and saves checkpoints

## Usage

### Basic Example

```python
from config import ExperimentConfig, ModelConfig, DataConfig, GRPOTrainingConfig, RewardConfig

config = ExperimentConfig(
    experiment_name="my_grpo_experiment",
    model=ModelConfig(
        model_name_or_path="Qwen/Qwen2-0.5B-Instruct",
        use_peft=True,
        lora_r=16,
    ),
    data=DataConfig(
        use_datamodule=False,
        dataset_path="path/to/your/dataset",
        max_samples=1000,
    ),
    training=GRPOTrainingConfig(
        output_dir="./output",
        num_train_epochs=3,
        per_device_train_batch_size=2,
    ),
    reward=RewardConfig(
        use_hard_match=True,
        use_soft_match=True,
        hard_match_reward=1.0,
        soft_match_reward=0.5,
    ),
)

# Run training
from train_grpo import main
main(config)
```

### Command Line Usage

```bash
cd scripts/rl_training
python train_grpo.py
```

Note: The default configuration in `train_grpo.py` uses a placeholder dataset. You'll need to modify the config to point to your actual code execution dataset.

## Integration with Existing Codebase

### Using ConversationDataModule

To use the existing data infrastructure:

1. Set `data.use_datamodule = True` in your config
2. Provide path to your datamodule config: `data.datamodule_config_path = "path/to/config.yaml"`
3. Implement the `load_datamodule()` function to properly load your datamodule

Example:

```python
def load_datamodule(config: DataConfig) -> pyine.data.datamodule.ConversationDataModule:
    # Load your datamodule config
    import pyine.apps.trainers.hf_trainer_configs

    # Instantiate datamodule
    dm = your_datamodule_config.instantiate_datamodule(verbose=True)
    dm.prepare_data()
    dm.setup()
    return dm
```

### Reward Functions

The prototype supports two types of rewards:

1. **Hard Match** (Exact Match):

   - Compares prediction and expected output after stripping whitespace
   - Returns `hard_match_reward` (default: 1.0) on match
   - Fast and deterministic

2. **Soft Match** (Heuristic):

   - Uses `pyine.utils.code.output_compare.compare()`
   - Tolerant to:
     - Whitespace differences
     - Float precision differences
     - List/tuple ordering (configurable)
   - Returns `soft_match_reward` (default: 0.5) on match
   - More forgiving but still deterministic

3. **LLM Grader** (Not Yet Implemented):

   - Would use an LLM to score prediction quality
   - See `pyine.evals.code_exec.utils.OutcomeEvaluator` for reference

## Dataset Format

### Required Fields

For GRPO training, your dataset should have:

- `prompt`: The formatted prompt for the model
- `expected_output`: The ground truth output for reward calculation

### From ConversationDataModule

If using `ConversationDataModule`, samples should be `SampleData` objects with:

- `identifier`: Unique sample ID
- `code`: Code snippet
- `inputs`: Test inputs
- `expected_output`: Expected execution result
- `predict_type`: Type of prediction ("program_output", "frame_variables", "function_return")
- Additional metadata fields

## GRPO Training Details

### Key Parameters

- **`num_generation_per_prompt`**: Number of completions to sample per prompt (default: 4)

  - Higher values provide more samples for policy updates but increase compute

- **`kl_coef`**: KL divergence coefficient (default: 0.05)

  - Controls how much the policy can deviate from the reference model
  - Higher values keep policy closer to reference (more conservative)

- **`max_new_tokens`**: Maximum tokens to generate (default: 256)

  - Should be large enough to capture full code execution outputs

- **`temperature`**: Sampling temperature (default: 0.7)

  - Higher values increase diversity in generations

### Training Tips

1. **Start Small**: Use `max_samples` to test on a small subset first
2. **Use LoRA**: Set `model.use_peft=True` for memory-efficient training
3. **Monitor Rewards**: Watch the average reward in logs to ensure learning
4. **Adjust KL**: If model deviates too much, increase `kl_coef`
5. **Batch Size**: Effective batch size = `per_device_batch_size * gradient_accumulation_steps * num_gpus`

## Extending the Prototype

### Adding New Reward Functions

```python
# In rewards.py
class CustomRewardCalculator:
    def compute_single_reward(self, predicted: str, expected: str) -> float:
        # Your custom reward logic
        return reward_value

# Update create_grpo_reward_function to use your calculator
```

### Adding Evaluation

```python
# After training in train_grpo.py
eval_dataset = prepare_grpo_dataset_simple(
    dataset_path=config.data.dataset_path,
    split=config.data.dataset_split_eval,
)

# Evaluate using the existing evaluation infrastructure
from pyine.evals.code_exec.configs import CodeExecEvalsConfig
# ... evaluation logic
```

### Multi-GPU Training

The TRL trainer supports multi-GPU training out of the box:

```bash
torchrun --nproc_per_node=4 train_grpo.py
```

Or with accelerate:

```bash
accelerate launch train_grpo.py
```

## Troubleshooting

### Common Issues

1. **"Expected outputs not found in kwargs"**

   - Check that your dataset has `expected_output` field
   - Verify the `expected_outputs_key` parameter in reward function

2. **Out of Memory**

   - Enable `model.use_peft=True` for LoRA
   - Reduce `per_device_train_batch_size`
   - Use quantization: `model.load_in_4bit=True`
   - Reduce `training.num_generation_per_prompt`

3. **Low Rewards**

   - Check that `expected_output` format matches prediction format
   - Enable `reward.use_soft_match=True` for more lenient matching
   - Verify predictions are actually correct by printing samples

4. **Training Not Converging**

   - Increase `training.num_train_epochs`
   - Adjust `training.learning_rate` (try 1e-6 to 1e-4)
   - Increase `training.kl_coef` if policy diverges too much

## Future Enhancements

Potential improvements to consider:

1. **LLM Grader Integration**: Add LLM-based reward scoring using existing `OutcomeEvaluator`
2. **Multi-Task Rewards**: Combine different reward types with weights
3. **Curriculum Learning**: Start with easier samples, progressively increase difficulty
4. **Reward Shaping**: Add intermediate rewards for partial correctness
5. **Distributed Training**: Full multi-node training support with proper data sharding
6. **Advanced GRPO Features**: Explore TRL's advanced options (value function, etc.)

## References

- TRL GRPO Documentation: https://huggingface.co/docs/trl/grpo_trainer
- TRL Examples: https://github.com/huggingface/trl/tree/main/examples
- GRPO Paper: [Link to paper when available]

## Questions & Support

For questions about:

- The prototype: Check this README or code comments
- TRL library: See [TRL documentation](https://huggingface.co/docs/trl)
- Existing codebase: Refer to `pyine/apps/trainers/` and `pyine/evals/code_exec/`
