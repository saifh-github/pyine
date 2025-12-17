# Quick Start: Running GRPO Training with grpo_minimal Template

This guide walks you through running your first GRPO training experiment with the optimized `grpo_minimal` prompt template.

## Prerequisites

- TACO dataset downloaded and available
- Python environment with all dependencies installed
- GPU recommended (but CPU works for testing)

## Step-by-Step Guide

### Step 1: Generate Your Datamodule Config

Generate a config file with auto-detected paths:

```bash
python -m pyine.apps.rl_trainers.create_datamodule_config \
    --output pyine/apps/rl_trainers/configs/my_grpo_training.yaml \
    --dataset-name TACO_10s10t_v1_part1 \
    --max-solutions 100
```

**What this does:**

- Auto-detects your TACO dataset paths
- Creates a config with `grpo_minimal` template (zero-shot, structured output)
- Limits to 100 solutions for quick testing

**Output:**

```
✓ Configured for GRPO training:
  - Prompt template: grpo_minimal (optimized, ~50% shorter)
  - Zero-shot by default (no examples, saves tokens)
  - Plain text format (not chat)
  - Tag-enclosed answer (<final>...</final>) for easy reward parsing
```

### Step 2: Inspect the Generated Prompts

Before training, verify the prompts look correct:

```bash
python -m pyine.apps.rl_trainers.inspect_prompts \
    --datamodule-config pyine/apps/rl_trainers/configs/my_grpo_training.yaml \
    --subset train \
    --num-samples 5
```

**What to check:**

- ✓ Prompt uses `grpo_minimal` template
- ✓ No few-shot examples (zero-shot)
- ✓ Clear output format instructions (`<final>...</final>`)
- ✓ All 3 output types present (program_output, frame_variables, function_return)
- ✓ Expected outputs match the tasks

**Example output:**

```
Dataset Statistics:
Output Type Distribution:
  program_output: 60 (60.0%)
  frame_variables: 20 (20.0%)
  function_return: 20 (20.0%)

Sample 1/5:
Identifier: sample_12345
Output Type: program_output

Prompt (483 chars):
────────────────────────────────────────
You are an expert at interpreting and executing Python 3 code.

Analyze the given code snippet and predict its execution outcome.
...
Your answer:
────────────────────────────────────────

Expected Output (15 chars):
────────────────────────────────────────
Hello, World!
```

### Step 3: Configure Your Training Run

The `train_grpo.py` script is already configured to use your datamodule config. Key settings:

```python
config = ExperimentConfig(
    experiment_name=f"GRPO_Test_{timestamp}",
    seed=42,
    use_wandb=True,  # Set to False if you don't want WandB logging
    wandb_project="pyine-grpo-tests",

    model=ModelConfig(
        model_name_or_path="Qwen/Qwen2-0.5B-Instruct",  # Small model for testing
        use_peft=True,  # LoRA for efficient training
        lora_r=16,
        lora_alpha=32,
    ),

    data=DataConfig(
        use_datamodule=True,
        datamodule_config_path="pyine/apps/rl_trainers/configs/my_grpo_training.yaml",
        dataset_split_train="train",
        max_samples=100,  # Start small
    ),

    training=GRPOTrainingConfig(
        output_dir="./grpo_output",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,  # Effective batch size = 1 * 4 = 4
        learning_rate=1e-5,
        logging_steps=5,
        save_steps=50,
        num_generations=4,  # Generate 4 samples per prompt for GRPO
        max_completion_length=256,
        temperature=1.0,  # Default sampling temperature
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        use_vllm=False,  # Set to True to use vLLM for faster inference (requires setup)
    ),
)
```

**To customize:**

- Change `model_name_or_path` to use a different model
- Adjust `max_samples` for more/less data
- Increase `num_generations` for better exploration (but higher compute cost)
- Set `use_wandb=False` if not using Weights & Biases
- Set `bf16=False` if your GPU doesn't support bfloat16
- Update `datamodule_config_path` to match your generated config file name

**Note:** The default config in `train_grpo.py` uses `grpo_training.yaml`, but this guide shows creating `my_grpo_training.yaml`. Make sure the path matches your actual config file.

### Step 4: Run Training

```bash
python -m pyine.apps.rl_trainers.train_grpo
```

**What happens:**

1. Loads model and applies LoRA adapters
2. Loads datamodule and prepares GRPO dataset
3. Formats prompts using `grpo_minimal` template
4. Starts GRPO training with code execution accuracy reward (hard matching: 1.0 for exact match, 0.0 otherwise)
5. Saves checkpoints to `./grpo_output/`

**Expected console output:**

```
2025-12-09 15:30:00 - INFO - Starting experiment: GRPO_Test_20251209_1530
2025-12-09 15:30:05 - INFO - Loading model: Qwen/Qwen2-0.5B-Instruct
2025-12-09 15:30:10 - INFO - Applying LoRA adapters
2025-12-09 15:30:15 - INFO - Loading dataset from datamodule...
2025-12-09 15:30:20 - INFO - Loading prompt template version: grpo_minimal
2025-12-09 15:30:25 - INFO - Extracted and formatted 100 samples from train
2025-12-09 15:30:30 - INFO - Training dataset size: 100
2025-12-09 15:30:35 - INFO - Setting up code execution reward function (hard matching only)
2025-12-09 15:30:35 - INFO - Initializing GRPO trainer...
2025-12-09 15:30:40 - INFO - Starting training...
```

**Understanding the Reward Function:**
The training uses a **code execution accuracy reward** with hard matching:

- The model generates predictions for code execution outcomes
- Each prediction is compared to the expected output (exact string match after stripping whitespace)
- **Reward = 1.0** if the prediction exactly matches the expected output
- **Reward = 0.0** if the prediction doesn't match
- This binary reward optimizes the model to produce correct code execution predictions
- Configuration in `train_grpo.py:272-279`: `strip_hard_checks=True`, `enable_soft_match=False`

### Step 5: Monitor Training

**With WandB (if enabled):**

- Visit https://wandb.ai/your-username/pyine-grpo-tests
- Monitor metrics:
  - `train/reward_mean` (should increase)
  - `train/reward_std` (measure of sample diversity)
  - `train/loss` (policy loss)

**Without WandB:**

- Check TensorBoard logs in `./grpo_output/runs/`
- View console output for periodic metrics

**Key metrics to watch:**

- **Reward Mean**: Should increase if model is learning to predict correctly
  - Start: ~0.0-0.1 (mostly incorrect predictions)
  - Target: >0.3-0.5 after a few hundred steps (30-50% accuracy)
  - Note: This is a binary reward (1.0 or 0.0), so mean reward = accuracy
- **Loss**: Should decrease steadily
- **Reward Std**: Indicates diversity in outcomes (will be high initially, may stabilize)

### Step 6: Evaluate Results

After training completes:

```bash
# Check saved model
ls ./grpo_output/final_model/

# Inspect checkpoints
ls ./grpo_output/checkpoint-*/
```

**Files created:**

- `final_model/` - Final trained model
- `checkpoint-N/` - Intermediate checkpoints (every 50 steps)
- `runs/` - TensorBoard logs

## Advanced: Using vLLM for Faster Inference

For significantly faster inference during GRPO training, you can use vLLM to serve the model on separate GPUs while training runs on different GPUs. This distributed setup can dramatically speed up the generation phase.

### Overview

The vLLM setup uses two separate GPU groups:

- **vLLM server GPUs**: Run the inference server for fast generation
- **Training GPUs**: Run the actual GRPO training process

**IMPORTANT:** These GPU groups must NOT overlap! Ensure they use completely different GPU indices.

### Step 1: Update Training Config

Enable vLLM in your training configuration:

```python
config = ExperimentConfig(
    experiment_name=f"GRPO_vLLM_Test_{timestamp}",
    seed=42,

    model=ModelConfig(
        model_name_or_path="Qwen/Qwen2-0.5B-Instruct",
        use_peft=True,
        lora_r=16,
        lora_alpha=32,
    ),

    data=DataConfig(
        use_datamodule=True,
        datamodule_config_path="pyine/apps/rl_trainers/configs/my_grpo_training.yaml",
        dataset_split_train="train",
        max_samples=100,
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
        temperature=1.0,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        use_vllm=True,  # Enable vLLM inference
    ),
)
```

### Step 2: Start vLLM Inference Server

In a **separate terminal**, start the vLLM server on dedicated GPUs:

```bash
# Example: Use GPUs 0, 1, 2 for vLLM inference
CUDA_VISIBLE_DEVICES=0,1,2 uv run trl vllm-serve \
    --model Qwen/Qwen2-0.5B-Instruct \
    --data_parallel_size 3  # if model fits in a single GPU
```

**Parameters:**

- `CUDA_VISIBLE_DEVICES`: GPU indices for vLLM server (e.g., `0,1,2`)
- `--model`: Same model as in your training config
- `--data_parallel_size`: Number of GPUs allocated (must match number in CUDA_VISIBLE_DEVICES) if model fits in a single GPU

**Wait for server to start:**

```
INFO: Started server process
INFO: Waiting for application startup.
INFO: Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:8000
```

### Step 3: Run Training with Accelerate

In your **main terminal**, launch training on different GPUs:

```bash
# Example: Use GPUs 3, 4 for training (no overlap with vLLM GPUs!)
CUDA_VISIBLE_DEVICES=3,4 uv run accelerate launch \
    pyine/apps/rl_trainers/train_grpo.py
```

**CRITICAL: GPU Separation**

```bash
# ✓ CORRECT - No overlap
CUDA_VISIBLE_DEVICES=0,1,2    # vLLM server
CUDA_VISIBLE_DEVICES=3,4      # Training

# ✗ WRONG - GPU 1 and 2 overlap!
CUDA_VISIBLE_DEVICES=0,1,2    # vLLM server
CUDA_VISIBLE_DEVICES=2,3      # Training - GPU 2 used by both!
```

### Complete Example: 5 GPU Setup

Assuming you have 5 GPUs (0-4):

**Terminal 1 (vLLM server on GPUs 0-2):**

```bash
CUDA_VISIBLE_DEVICES=0,1,2 uv run trl vllm-serve \
    --model Qwen/Qwen2-0.5B-Instruct \
    --data_parallel_size 3  # if model fits in a single GPU
```

**Terminal 2 (Training on GPUs 3-4):**

```bash
CUDA_VISIBLE_DEVICES=3,4 uv run accelerate launch \
    pyine/apps/rl_trainers/train_grpo.py
```

### Benefits of vLLM Setup

**Speed improvements:**

- Generation phase: **3-10x faster** than standard inference
- Overall training: **2-5x faster** depending on generation/training ratio
- Better GPU utilization with parallelized generation

**When to use vLLM:**

- You have 4+ GPUs available
- Generation time is a bottleneck (high `num_generations`)
- Training with larger models (7B+)
- Need faster iteration during experimentation

### Known Issues

#### TRL Library Bug: Unsupported Tools in vLLM Chat

**Issue:** The current version of TRL sets tools in the vLLM chat interaction that are not supported by vLLM, causing errors during generation.

**Error messages you might see:**

```
ValueError: tools are not supported in vLLM chat
NotImplementedError: vLLM does not support tools parameter
```

**Workarounds:**

Choose one of the following fixes by modifying your local TRL installation:

**Option 1: Fix in GRPO Trainer `__init__` method (Line ~415)**

```python
# In trl/trainer/grpo_trainer.py, around line 415
# Add this line to disable tools:
self.tools = None
```

**Option 2: Fix in `generate_single_turn` method (Line ~1295)**

```python
# In trl/trainer/grpo_trainer.py, around line 1295
# Force tools to none as follows:

    if is_conversational({"prompt": ordered_set_of_prompts[0]}):
        output = self.vllm_client.chat(
            messages=ordered_set_of_prompts,
            **sampling_params,
            chat_template_kwargs=self.chat_template_kwargs,
            tools=None,
            chat_template=self.chat_template,
        )
```

**How to apply the fix:**

1. Locate your TRL installation:

   ```bash
   python -c "import trl; print(trl.__file__)"
   # Example output: /path/to/site-packages/trl/__init__.py
   ```

2. Edit the file:

   ```bash
   # Navigate to the TRL trainer directory
   cd /path/to/site-packages/trl/trainer

   # Edit grpo_trainer.py and add self.tools = None at one of the locations above
   vim grpo_trainer.py
   ```

3. Save and restart your training

**Note:** This is a temporary workaround until we will adopt our own fork and the TRL library is updated to handle vLLM compatibility properly

## Summary

**To run your first GRPO training:**

```bash
# 1. Generate config
python -m pyine.apps.rl_trainers.create_datamodule_config \
    --output pyine/apps/rl_trainers/configs/my_grpo_training.yaml \
    --dataset-name TACO_10s10t_v1_part1 \
    --max-solutions 100

# 2. Inspect prompts
python -m pyine.apps.rl_trainers.inspect_prompts \
    --datamodule-config pyine/apps/rl_trainers/configs/my_grpo_training.yaml \
    --subset train \
    --num-samples 5

# 3. Update train_grpo.py to use your config file
# Edit the datamodule_config_path in train_grpo.py to point to:
# "pyine/apps/rl_trainers/configs/my_grpo_training.yaml"

# 4. Run training
python -m pyine.apps.rl_trainers.train_grpo
```

**To run with vLLM acceleration (requires 4+ GPUs):**

```bash
# Terminal 1: Start vLLM server
CUDA_VISIBLE_DEVICES=0,1,2 uv run trl vllm-serve \
    --model Qwen/Qwen2-0.5B-Instruct \
    --data_parallel_size 3  # if model fits in a single GPU

# Terminal 2: Run training
CUDA_VISIBLE_DEVICES=3,4 uv run accelerate launch \
    pyine/apps/rl_trainers/train_grpo.py
```

That's it! You're now using the optimized `grpo_minimal` template with your code execution dataset.
