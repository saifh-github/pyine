# GRPO Training with Custom Dataset and Prompts - Approach 1 Implementation

This document describes the implementation of **Approach 1**: Minimal integration of your existing dataset, datamodule, and prompt system with GRPO training.

## Overview

This implementation enables GRPO training using:
- ✅ Your existing `ConversationDataModule` for data loading
- ✅ Your Jinja2-based prompt templates from `pyine/prompts/templates/code_execution.yaml`
- ✅ Proper extraction of `SampleData` with expected outputs
- ✅ Validation utilities to inspect generated prompts before training
- ⏳ Dummy reward function (to be replaced with code execution rewards in Approach 2)

## What Was Implemented

### 1. Data Pipeline (`pyine/apps/rl_trainers/data_utils.py`)

#### `format_code_execution_prompt()`
- **Updated** to use actual Jinja2 templates instead of hardcoded strings
- Takes a `prompt_template` parameter and uses `format_prompt()` method
- Returns plain text strings suitable for GRPO (not chat messages)

#### `prepare_grpo_dataset_from_datamodule()`
- **Completely rewritten** to properly integrate with your datamodule system
- Loads the prompt template using `pyine.prompts.configs.code_execution.get_prompt_template()`
- Extracts `SampleData` from the datamodule's `SampleBuilder`
- Formats each sample using the Jinja2 template
- Returns HuggingFace Dataset with required columns: `prompt`, `expected_output`, `predict_type`, etc.
- **Key features:**
  - Supports multiple prompt versions (default: `"unstructured_with_3_output_types"`)
  - Includes all sample metadata for future reward function use
  - Proper error handling with informative messages

#### `inspect_grpo_dataset()` (NEW)
- **Purpose:** Validate generated prompts before training
- **Features:**
  - Shows dataset statistics (size, column names)
  - Distribution of output types and code types
  - Displays random sample prompts with truncation for readability
  - Can save inspection results to file
  - Checks for required columns

### 2. Datamodule Loading (`pyine/apps/rl_trainers/train_grpo.py`)

#### `load_datamodule()`
- **Fully implemented** to load datamodules from config files
- Supports both YAML and JSON config formats
- Handles the complete datamodule lifecycle:
  1. Load config from file
  2. Instantiate datamodule config
  3. Instantiate datamodule
  4. Call `prepare_data()` (download/process if needed)
  5. Call `setup()` (setup parsers)
- **Validates** that result is a `ConversationDataModule`

### 3. Inspection Script (`pyine/apps/rl_trainers/inspect_prompts.py`) (NEW)

A standalone script to preview prompts before training:

```bash
# Inspect prompts from your datamodule
python -m pyine.apps.rl_trainers.inspect_prompts \
    --datamodule-config path/to/your/datamodule_config.yaml \
    --subset train \
    --num-samples 5 \
    --max-samples 100

# Save inspection to file
python -m pyine.apps.rl_trainers.inspect_prompts \
    --datamodule-config path/to/config.yaml \
    --output-file prompts_review.txt
```

## How to Use

### Step 1: Prepare Your Datamodule Config

You need a datamodule configuration file (YAML or JSON) that defines your dataset. This should be structured like the configs used in `pyine/organisms/datamodules/shortcuts_configs.py`.

Example minimal config structure:
```yaml
datamodule_class_path: "pyine.organisms.datamodules.shortcuts.ShortcutBiasDataModule"
datamodule_name: "my_code_exec_data"
# ... other datamodule configuration ...
```

### Step 2: Inspect the Generated Prompts

**Before training**, validate that prompts are formatted correctly:

```bash
python -m pyine.apps.rl_trainers.inspect_prompts \
    --datamodule-config path/to/your/datamodule_config.yaml \
    --subset train \
    --num-samples 10 \
    --max-samples 50 \
    --prompt-version unstructured_with_3_output_types
```

This will show you:
- Dataset statistics (size, columns)
- Distribution of output types (program_output, frame_variables, function_return)
- Sample prompts with their expected outputs
- Any issues with the data pipeline

**Review the output carefully** to ensure:
- ✓ Prompts include system instructions, few-shot examples, and the test case
- ✓ Code snippets are properly formatted
- ✓ Expected outputs are preserved
- ✓ Different output types are handled correctly

### Step 3: Update `train_grpo.py` Configuration

Modify the configuration in `train_grpo.py` (lines 235-264) to use your datamodule:

```python
config = ExperimentConfig(
    experiment_name=f"GRPO_CodeExec_{timestamp}",
    seed=42,
    use_wandb=True,  # Enable for tracking
    wandb_project="your-project-name",
    model=ModelConfig(
        model_name_or_path="Qwen/Qwen2-0.5B-Instruct",
        use_peft=True,
        lora_r=16,
        lora_alpha=32,
    ),
    data=DataConfig(
        use_datamodule=True,  # KEY: Enable datamodule loading
        datamodule_config_path="path/to/your/datamodule_config.yaml",  # KEY: Your config
        train_subset_name="train",
        max_samples=1000,  # Start small for testing
    ),
    training=GRPOTrainingConfig(
        output_dir="./grpo_output",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        num_generations=4,  # Generate 4 completions per prompt
        max_completion_length=256,
    ),
)
```

### Step 4: Run Training

```bash
python -m pyine.apps.rl_trainers.train_grpo
```

The script will:
1. Load your datamodule from the config file
2. Prepare data and setup parsers
3. Format prompts using your Jinja2 templates
4. Train with GRPO using the dummy reward function

## Current Limitations (To Be Addressed in Approach 2)

1. **Dummy Reward Function**: Currently using `reward_num_unique_letters()` which just counts unique characters
   - In Approach 2, we'll replace this with actual code execution verification
   - Will use your `OutcomeEvaluator` for hard/soft matching

2. **Single Output Type Support**: While the prompts support all 3 output types, the reward function doesn't distinguish between them yet
   - Approach 2 will handle `program_output`, `frame_variables`, and `function_return` separately

3. **No Soft Matching**: Only binary rewards possible with dummy function
   - Approach 2 will add configurable rewards (hard match = 1.0, soft match = 0.5, no match = 0.0)

## Testing Your Implementation

### Quick Validation Checklist

1. **Data Loading Works:**
   ```bash
   python -m pyine.apps.rl_trainers.inspect_prompts \
       --datamodule-config YOUR_CONFIG.yaml \
       --num-samples 3
   ```
   - Should show 3 properly formatted prompts
   - No errors during datamodule loading

2. **Prompts Look Correct:**
   - Review the inspection output
   - Check that prompts include:
     - System role (expert at executing Python)
     - Context about output types
     - Few-shot examples
     - The actual test case
   - Verify expected outputs are correct

3. **Training Starts:**
   ```bash
   # Run with very small dataset first
   python -m pyine.apps.rl_trainers.train_grpo
   ```
   - Should load datamodule successfully
   - Format prompts without errors
   - Start GRPO training loop
   - (Training with dummy rewards won't produce good models, but should run)

## Key Files Modified/Created

- ✏️ `pyine/apps/rl_trainers/data_utils.py` - Core data pipeline functions
- ✏️ `pyine/apps/rl_trainers/train_grpo.py` - Datamodule loading implementation
- ➕ `pyine/apps/rl_trainers/inspect_prompts.py` - Validation script (NEW)
- ➕ `pyine/apps/rl_trainers/APPROACH1_IMPLEMENTATION.md` - This documentation (NEW)

## Next Steps (Approach 2)

Once you've validated that prompts are generating correctly:

1. **Replace dummy reward function** with code execution verification
2. **Integrate `OutcomeEvaluator`** from `pyine/evals/code_exec/utils.py`
3. **Support configurable reward schemes** (hard/soft matching, reward values)
4. **Handle multiple output types** properly in reward calculation

## Questions or Issues?

If you encounter any problems:

1. **Prompts not formatting correctly?**
   - Check the prompt template version (try `"no_pressure_demo"` for simpler prompts)
   - Verify your `SampleData` has all required fields

2. **Datamodule won't load?**
   - Verify the config file path is correct
   - Check that the config file is valid YAML/JSON
   - Ensure `datamodule_class_path` points to the correct class

3. **Training fails?**
   - Start with `max_samples=10` for quick debugging
   - Check GRPO trainer logs for specific errors
   - Verify GPU memory if using large models

## Summary

You now have a clean, minimal integration that:
- ✅ Loads your datasets via `ConversationDataModule`
- ✅ Formats prompts using your Jinja2 templates
- ✅ Provides validation tools to inspect generated prompts
- ✅ Works end-to-end with GRPO training

This provides a solid foundation for Approach 2, where we'll add proper code execution rewards!
