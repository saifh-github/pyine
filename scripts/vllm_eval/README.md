# vLLM Evaluation and Grading Guide

This directory contains utilities for serving models via vLLM's OpenAI-compatible API server for fast evaluation and LLM-based grading.

## Overview

The vLLM integration supports two workflows:

1. **Evaluation**: Generate predictions using a model (typically your fine-tuned model)
2. **Grading**: Judge predictions against expected outputs using an LLM (local vLLM server or OpenAI API)

```
┌─────────────────────┐
│  Evaluation Model   │  ← Generates code execution predictions
│  (vLLM or HF)       │
└─────────────────────┘
          ↓
    [Predictions]
          ↓
┌─────────────────────┐
│   Grading Model     │  ← Judges predictions vs. expected outputs
│  (vLLM or OpenAI)   │
└─────────────────────┘
          ↓
    [Scores: 0.0-1.0]
```

## Prerequisites

- Python >= 3.10
- CUDA-capable GPU(s)
- [uv](https://github.com/astral-sh/uv) package manager (recommended)
- **`.env` file**: The script requires a `.env` file at the repository root with environment variables (see `.env.template` for setup). The script automatically searches up the directory tree to find it.

## Quick Setup

```bash
# Install uv if not already available
curl -LsSf https://astral.sh/uv/install.sh | sh

# Navigate to the vllm_eval directory
cd scripts/vllm_eval

# Dependencies will be automatically installed when running scripts via uv
uv run python vllm_server.py --help
```

______________________________________________________________________

## Server Configuration

### Model Selection

- `--model`: HuggingFace model name (e.g., `Qwen/Qwen3-4B-Instruct-2507`)
- `--checkpoint_path`: Path to local checkpoint directory
- `--base_model`: Base model for LoRA merging (if not in adapter_config.json)
- `--merge_lora`: Enable LoRA adapter merging (auto-enabled if adapter detected)

### Server Options

- `--host`: Bind address (default: `0.0.0.0` for network access)
- `--port`: Server port (default: `8000`)
- `--model_name`: Name clients use in API calls. If not specified, auto-derived:
  - For HuggingFace models: uses full model string (e.g., `"Qwen/Qwen3-4B-Instruct-2507"`)
  - For checkpoints: uses last 3 path components (e.g., `"runs/exp_123/checkpoint-1600"`)
- `--max_model_len`: Maximum sequence length (default: model's max)
- `--trust_remote_code`: Allow custom model code (default: `true`, required for Qwen)

### GPU Selection

- `--cuda_devices`: Comma-separated GPU device IDs (e.g., `0,1,2,3`)
- `--tensor_parallel_size`: Number of GPUs for tensor parallelism (default: `8`)
- `--gpu_memory_utilization`: GPU memory fraction (default: `0.9`)

**Important for running multiple servers**: Use `--cuda_devices` to assign non-overlapping GPUs when running evaluation and grading servers simultaneously.

### LoRA-Specific Options

- `--merged_model_dir`: Where to save merged model (default: `<PYINE_CACHE_ROOT>/vllm_merged_models/<checkpoint_name>` where checkpoint name is auto-derived from the last 3 path components)
- `--force_merge`: Re-merge even if merged model exists

______________________________________________________________________

## Usage: Evaluation Only

### 1. Serve a HuggingFace Model Directly

```bash
# Single GPU
python vllm_server.py --model Qwen/Qwen3-4B-Instruct-2507 --cuda_devices 0
# Model name will be: "Qwen/Qwen3-4B-Instruct-2507"

# Multi-GPU (4 GPUs)
python vllm_server.py \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --cuda_devices 0,1,2,3
# Model name will be: "Qwen/Qwen3-4B-Instruct-2507"
```

### 2. Serve a Local Checkpoint (Full Model)

```bash
python vllm_server.py \
    --checkpoint_path /path/to/checkpoint \
    --cuda_devices 0,1,2,3 \
    --port 8000
# Model name auto-derived from checkpoint path (last 3 components)
# Example: "/home/user/runs/exp_123/checkpoint-1600" → "runs/exp_123/checkpoint-1600"
```

### 3. Serve a LoRA Checkpoint (with Merging)

LoRA checkpoints store only adapter weights. The script automatically merges them with the base model.

```bash
# Auto-detected base model from adapter_config.json
python vllm_server.py \
    --checkpoint_path /path/to/lora_checkpoint \
    --merge_lora
# Model name auto-derived from checkpoint path (last 3 components)

# Explicitly specify base model
python vllm_server.py \
    --checkpoint_path /path/to/lora_checkpoint \
    --base_model Qwen/Qwen3-4B-Instruct-2507 \
    --merge_lora \
    --cuda_devices 0,1,2,3
# Model name auto-derived from checkpoint path (last 3 components)

# Override model name if needed
python vllm_server.py \
    --checkpoint_path /path/to/lora_checkpoint \
    --base_model Qwen/Qwen3-4B-Instruct-2507 \
    --merge_lora \
    --model_name my-custom-model
```

**Note**: Merging happens only once. The merged model is cached in `<PYINE_CACHE_ROOT>/vllm_merged_models/<checkpoint_name>` (using the checkpoint path naming convention) and reused on subsequent runs unless you specify `--force_merge`. This allows merged models to be stored in a centralized location and reused across different server launches.

### 4. Run Evaluation

Once your server is running, use the evaluation config:

```bash
# Run evaluation with vLLM inference (faster than HuggingFace)
uv run python -m pyine.apps.trainers.hf_trainer \
    +experiment=original/v0_50perc_dataset_qwen3_vllm_eval.yaml
```

______________________________________________________________________

## Usage: Evaluation with Grading

Grading uses an LLM to judge if predictions match expected outputs. This provides more flexible matching than exact string comparison.

### Setup Option 1: Two vLLM Servers

**Terminal 1** - Evaluation Model (fine-tuned on GPUs 0-3):

```bash
python scripts/vllm_eval/vllm_server.py \
    --checkpoint_path /path/to/finetuned/checkpoint \
    --base_model Qwen/Qwen3-4B-Instruct-2507 \
    --port 8000 \
    --cuda_devices 0,1,2,3
# Model name auto-derived from checkpoint path (e.g., "finetuned/exp_123/checkpoint-1600")
```

**Terminal 2** - Grading Model (powerful base model on GPUs 4-7):

```bash
python scripts/vllm_eval/vllm_server.py \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --port 8001 \
    --cuda_devices 4,5,6,7
# Model name will be: "Qwen/Qwen3-4B-Instruct-2507"
```

**Notes**:

- The `--cuda_devices` argument prevents GPU overlap when running multiple servers
- Model names are auto-derived but can be overridden with `--model_name` if needed
- Make sure your config file uses the correct auto-derived model names in the `model:` fields

**Run Evaluation with Grading:**

```bash
uv run python -m pyine.apps.trainers.hf_trainer \
    +experiment=original/v0_50perc_dataset_qwen3_vllm_eval_vllm_grading.yaml
```

### Testing Grading Independently

Before running full evaluation, it is possible to test grading as follows (after having launched the grading vLLM server as seen previously):

```bash
# Test vLLM grading (default: grading server on port 8001)
uv run python scripts/vllm_eval/test_vllm_grading.py

# Test with custom server URL and model name
uv run python scripts/vllm_eval/test_vllm_grading.py \
    --base_url http://localhost:8001/v1 \
    --model grader-model
```

Expected output:

```
Testing VLLM Grader
================================================================================
Grader available: True

--- Test Case 1: Exact match ---
Expected: '42'
Predicted: '42'

...

Metrics:
  accuracy_hard: 0.75
  accuracy_soft: 1.0
  accuracy_grader: 0.75

✓ vLLM grading test completed successfully
```

______________________________________________________________________

## Example Configurations

- vLLM evaluation only: [`pyine/configs/experiment/original/v0_50perc_dataset_qwen3_vllm_eval.yaml`](../../pyine/configs/experiment/original/v0_50perc_dataset_qwen3_vllm_eval.yaml)
- vLLM evaluation + grading: [`pyine/configs/experiment/original/v0_50perc_dataset_qwen3_vllm_eval_vllm_grading.yaml`](../../pyine/configs/experiment/original/v0_50perc_dataset_qwen3_vllm_eval_vllm_grading.yaml)
