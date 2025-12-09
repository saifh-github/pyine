# PyINE Experimentation Workflow Guide

This guide walks you through the end-to-end experimentation workflow in the PyINE framework, from
dataset preparation through model organism training and evaluation.

## Overview

The PyINE framework is designed to support experimentation workflows that cover:

- **Data generation, analysis, and exploration** for training and evaluation;
- **Model organism training and evaluation**, where model organisms are biased models that serve as subjects for alignment research;
- **Control/alignment strategy development** to detect and correct model organism biases (TODO @@@@, not yet in framework).

The framework enables researchers to systematically create model organisms with specific biases, evaluate those biases, and eventually develop and test strategies to mitigate them.

For LawZero staff, you can find data and backups on the related [shared drive](https://drive.google.com/drive/folders/1XQtIdZS8P9kKSF7P6z9UIfhw9hKYdWNY).

______________________________________________________________________

## Complete Experimentation Workflow

For now, our experiments only depend on the [BAAI TACO dataset](https://huggingface.co/datasets/BAAI/TACO),
which is a collection of several datasets of coding problems and solutions. We may later integrate other
data sources, but for now, this is the only one we will be using.

### Step 1: Obtain the TACO Dataset

The 'original' TACO dataset contains a number issues (e.g. malformed JSON files) and lacks some
necessary metadata for processing and execution. We therefore need to 'repackage' it into a more
workable format. You have two options for obtaining the repackaged TACO dataset:

#### Option A: Download Pre-repackaged Dataset (Recommended)

Download the pre-repackaged TACO dataset and place it under your data root:

```bash
# TODO @@@@@ download instructions TBD; contact maintainers for dataset access, or use the shared drive
```

The repackaged dataset should be placed at:

```
<PYINE_DATA_ROOT>/TACO/repackaged/<version>/
# e.g.:
<PYINE_DATA_ROOT>/TACO/repackaged/2025-03-31-v01/
```

#### Option B: Regenerate from Raw Data (Advanced)

If you have access to the raw TACO data, you can regenerate the repackaged dataset:

```bash
# instructions for regeneration from raw data
# TODO @@@@@ depends on repackaging pipeline availability, TBD; see pyine/data/taco/dataset_repackager.py
```

______________________________________________________________________

### Step 2: Generate Execution Traces

Generate code execution traces for the dataset of code problems and solutions. Traces capture
line-by-line execution events for individual code snippets (i.e. solutions); these serve as the
basis for training and evaluating model organisms.

This step is quite time-consuming, and you might want to avoid it by just downloading a dataset
of already-generated traces. See once again the shared drive for more information, or contact
maintainers.

**Note:** by default, we trace all solutions for all code problems, so this step can be done prior
to doing any train/valid/test splitting (which is the next step).

```bash
# generate traces with configurable caps:
python -m pyine.apps.write.dataset_writer traces \
    --dataset-name TACO \
    --max-output-traces 10000 \
    --max-solutions-per-problem 10 \
    --max-tests-per-solution 10
```

For distributed processing or large-scale generation, you can partition the problem IDs first:

```bash
# create partitions of 500 problems each
python -m pyine.apps.splits.dataset_splitter partition \
    --split-file data/splits/TACO-split.bin \
    --output-dir data/splits \
    --ids-per-chunk 500 \
    --format yaml

# then, process each partition separately (e.g. on different nodes)
python -m pyine.apps.write.dataset_writer traces \
    --dataset-name=TACO \
    --max-solutions-per-problem=10 \
    --max-tests-per-solution=10 \
    --generate-obfuscated-solutions \
    --target-problem-ids=data/splits/TACO-split.problem_ids.<part_name>.yaml \
    --output-tag="10s10t.<part_name>"
# ...
```

**Output:** Creates `<PYINE_DATA_ROOT>/traces/TACO/<tag>.<date>.lmdb` containing execution traces.

For more details, see [`pyine/apps/README.md`](./pyine/apps/README.md#write-execution-traces-and-deltas-datasets).

______________________________________________________________________

### Step 3: Generate Dataset Splits

Create train/validation/test splits for the dataset. The splitter supports stratification by
difficulty (according to the original source dataset labels) and solution counts to ensure balanced
distributions.

```bash
# create an 80/10/10 split with grouping metadata
python -m pyine.apps.splits.dataset_splitter split \
    --dataset-name TACO \
    --train-fraction 0.8 \
    --valid-fraction 0.1 \
    --test-fraction 0.1 \
    --use-difficulty-group \
    --use-solution-counts-group \
    --progress
```

**Output:** Creates `<PYINE_DATA_ROOT>/splits/TACO-split.bin` containing subset assignments, hash
lists, and grouping metadata.

For more details, see [`pyine/apps/README.md`](./pyine/apps/README.md#split-a-source-dataset-into-trainvalidtest-and-partitions).

______________________________________________________________________

### Step 4: Prepare Training Data Caches (Optional)

For large-scale training, you can precache the training data to improve data loading performance:

```bash
# precache datasets for a specific experiment configuration
python -m pyine.apps.data.hf_precacher +experiment=<your_experiment>
```

This step:

- Pre-generates datamodule caches (metadata, HF message datasets, tokenized examples);
- Ensures faster, non-blocking trainer startups;
- Is especially useful for distributed training to avoid cache generation conflicts.

For more details, see [`pyine/apps/README.md`](./pyine/apps/README.md#huggingface-dataset-precacher).

______________________________________________________________________

### Step 5: Create an Experiment Configuration

Create a Hydra experiment configuration file that defines your model organism training setup.

Create a new YAML file at `pyine/configs/experiment/<your_experiment_name>.yaml`:

```yaml
# @package _global_
defaults:
  - override /config: base
  # ...
  - _self_

runtime:
  exp_name: my_model_organism_exp
  seed: 42

config:
  # For HuggingFace training
  base_model: "Qwen/Qwen2.5-1.5B-Instruct"
  training_args_config:
    num_train_epochs: 3
    per_device_train_batch_size: 4
    learning_rate: 2e-5
  # ...

  # Or for OpenAI fine-tuning
  # openai_finetuner_config:
  #   base_model: "gpt-4.1-mini-2025-04-14"
  #   n_epochs: 3
  #   ...
```

**Tips:**

- Start from existing experiment configs as templates (see `pyine/configs/experiment/`)
- Check the settings of pre-registered experiments: `python -m pyine.apps.trainers.hf_trainer_configs`
- For configuration details, see [`pyine/configs/README.md`](./pyine/configs/README.md)

______________________________________________________________________

### Step 6: Train the Model Organism

Train your model organism using either the HuggingFace trainer or OpenAI fine-tuner.

#### Option A: HuggingFace Trainer

```bash
# train using your experiment config
python -m pyine.apps.trainers.hf_trainer +experiment=<your_experiment_name>

# or with inline overrides
python -m pyine.apps.trainers.hf_trainer \
  +experiment=<some_experiment_name> \
  config.training_args_config.learning_rate=1e-5
```

#### Option B: OpenAI Fine-tuner

```bash
# fine-tune using OpenAI API
python -m pyine.apps.trainers.openai_finetune \
  +experiment=<your_experiment_name>

# skip fine-tuning and evaluate base model only
python -m pyine.apps.trainers.openai_finetune \
  +experiment=<your_experiment_name> \
  skip_fine_tuning=true
```

**Outputs:**

Training runs create output directories under:

```
<PYINE_LOGS_ROOT>/runs/<app>/<exp_name>/<run_name>/
```

Each run directory contains:

- `.hydra/`: original configs, Hydra settings, and overrides;
- `config.<timestamp>.rank00.json`: resolved runtime config;
- `output.log`: training app logs;
- `reprod_metadata.<timestamp>.rank00.json`: reproducibility metadata; and
- Model checkpoints and tokenizer files.

For more details, see [`pyine/apps/README.md`](./pyine/apps/README.md#trainers).

______________________________________________________________________

### Step 6a: Distributed Training with DDP (Optional)

For training on multiple GPUs using Distributed Data Parallel (DDP), you have two options:

**Option A: HuggingFace Accelerate**

Use the Accelerate library for simplified distributed training configuration:

```bash
uv run accelerate launch pyine/apps/trainers/hf_trainer.py +experiment=<some_experiment_name>
```

**Note:** While Accelerate attempts to infer configuration automatically, it's recommended to first run `accelerate config` to generate proper settings for your specific deployment infrastructure (GPU count, mixed precision, etc.).

**Option B: Custom torchrun Script**

Use the provided `run_ddp.sh` script that explicitly leverages torchrun with configurable parameters:

```bash
uv run ./scripts/run_ddp.sh -- +experiment=<some_experiment_name>
```

Both approaches handle process spawning, distributed communication setup, and gradient synchronization automatically. The Accelerate option provides a simpler interface with automatic configuration, while the `run_ddp.sh` script offers more explicit control over distributed parameters (nodes, processes per node, master address/port, etc.). See the script's `--help` flag for advanced options.

______________________________________________________________________

### Step 6b - W&B Agents: Run Hyperparameter Sweeps (Optional)

For hyperparameter tuning, you can use WandB's native sweep functionality with distributed agents
to efficiently explore hyperparameter spaces across multiple GPUs. This approach uses WandB's
centralized sweep server to coordinate parallel agent clients, enabling sophisticated search
strategies like Bayesian optimization.

For more information, see the official WandB [documentation on sweeps](https://docs.wandb.ai/models/sweeps).

#### Initial Setup

Before running sweeps, ensure WandB is properly configured:

```bash
# Login to WandB (only needed once)
uv run wandb login

# Follow the prompts to authenticate with your API key
```

#### Creating a Sweep Configuration

Define your sweep in a YAML configuration file (e.g., [pyine/configs/experiment/original/wandb_sweep_config.yaml](pyine/configs/experiment/original/wandb_sweep_config.yaml)):

```yaml
# Refs: https://docs.wandb.ai/models/sweeps

program: pyine/apps/trainers/hf_trainer.py  # entry point to start running the code
name: PyINE-ParallelSweep  # wandb project name for the sweep
method: random  # search strategy: grid, random, or bayes
metric:  # metric to optimize
  name: eval/loss
  goal: minimize

parameters:  # search space definition
  config.lora_config.r:
    values: [4, 8, 16]
  config.lora_config.lora_alpha:
    values: [8, 16, 32, 64]
  config.training_args_config.learning_rate:
    distribution: "log_uniform_values"
    min: 1.0e-6
    max: 1.0e-3
  config.training_args_config.gradient_accumulation_steps:
    values: [2, 4, 6, 8]
  config.training_args_config.warmup_ratio:
    values: [0.03, 0.06, 0.1]
  config.training_args_config.max_grad_norm:
    values: [0.5, 1.0, 2.0]

command:
  - ${env}
  - ${interpreter}
  - ${program}
  - "+experiment=original/v0_50perc_dataset_qwen3.yaml"
  - ${args_no_hyphens}
```

**Key configuration elements:**

- `program`: Entry point script for training
- `method`: Search strategy (`grid`, `random`, or `bayes`)
- `metric`: Metric to optimize with goal (`minimize` or `maximize`)
- `parameters`: Hyperparameter search space (supports discrete values, ranges, and distributions)
- `command`: Command template for running each trial

#### Launching a Sweep

Create a new sweep on the WandB server:

```bash
# Initialize the sweep and get a sweep ID
uv run wandb sweep pyine/configs/experiment/original/wandb_sweep_config.yaml

# Output will include a sweep ID like: lawzero-default/code-interp-benchmark-pyine_apps_trainers/4vp5ivg4
```

The sweep ID format is: `<entity>/<project>/<sweep_id>`

#### Running Sweep Agents

**Option A: Single Agent (Local)**

Run a single agent on a specific GPU:

```bash
# Run agent on GPU 0
CUDA_VISIBLE_DEVICES=0 uv run wandb agent <sweep-id>
```

**Option B: Multiple Agents (Cluster with Automated tmux Sessions)**

For distributed sweeps across multiple GPUs and cluster nodes, use the provided automation script from the login node. This script automatically creates tmux sessions for each GPU node, with 8 panes per node (one per GPU).

First, create a command template file (e.g., [scripts/my_command.sh](scripts/my_command.sh)) that defines what each agent should execute:

```bash
cd ${REPO_ROOT} && CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} uv run wandb agent ${SWEEP_ID}
```

The template supports the following variables:

- `${REPO_ROOT}`: Repository root path
- `${CUDA_DEVICE}`: CUDA device index (0-7)
- `${SWEEP_ID}`: WandB sweep ID
- `${TARGET_GPU}`: GPU node number

Then launch agents across one or more GPU nodes:

```bash
# Launch agents on multiple GPU nodes
bash ./scripts/launch_wandb_agents.sh <SWEEP_ID> \
  --cmd-file <command_template_file> \
  --repo-root <repository_path> \
  <node_index_1> [node_index_2] ... [node_index_N]

# Example: Launch on GPU nodes 1, 2, and 3
bash ./scripts/launch_wandb_agents.sh \
  lawzero-default/code-interp-benchmark-pyine_apps_trainers/4vp5ivg4 \
  --cmd-file ./scripts/my_command.sh \
  --repo-root /scratch/a.palmas/code-interp-benchmark \
  1 2 3

# Example: Launch on a single GPU node (node 1)
bash ./scripts/launch_wandb_agents.sh \
  lawzero-default/code-interp-benchmark-pyine_apps_trainers/4vp5ivg4 \
  --cmd-file ./scripts/my_command.sh \
  --repo-root /scratch/a.palmas/code-interp-benchmark \
  1
```

**What the script does:**

1. Creates a separate tmux session for each specified GPU node (`wandb_sweep_gpu<N>`)
2. Each session contains 8 panes arranged in a 2×4 grid
3. Each pane automatically:
   - SSHs into the target GPU node (`ssh gpu0<N>`)
   - Navigates to the repository root
   - Launches a WandB agent on a specific GPU (CUDA_VISIBLE_DEVICES=0-7)
4. All agents connect to the same centralized WandB sweep server

**Managing tmux sessions:**

```bash
# List all active sessions
tmux list-sessions

# Attach to a specific GPU node's session
tmux attach-session -t wandb_sweep_gpu1

# Detach from a session (while inside tmux)
# Press: Ctrl+b then d

# Switch between sessions (while inside tmux)
# Press: Ctrl+b then s

# Kill a specific session
tmux kill-session -t wandb_sweep_gpu1

# Kill all sweep sessions
tmux kill-session -t wandb_sweep_gpu1
tmux kill-session -t wandb_sweep_gpu2
# ... etc
```

**Option C: Manual Parallel Agents**

Manually launch agents in separate terminals/sessions:

```bash
# Terminal 1 (GPU 0)
CUDA_VISIBLE_DEVICES=0 uv run wandb agent <sweep-id>

# Terminal 2 (GPU 1)
CUDA_VISIBLE_DEVICES=1 uv run wandb agent <sweep-id>

# ... and so on
```

#### Managing Sweeps

Monitor and control your sweep:

```bash
# View sweep status in WandB dashboard (automatically opens in browser)
# Or navigate to: https://wandb.ai/<entity>/<project>/sweeps/<sweep_id>

# Stop a running sweep
uv run wandb sweep --stop <sweep-id>

# Stop all agents (they will finish current runs and exit)
```

**Remember:**

- All agents pull hyperparameter configurations from the centralized WandB sweep server
- Agents automatically fetch new configurations when they complete a run
- Multiple agents can run in parallel, even across different machines
- Sweep results are automatically logged and visualized in the W&B dashboard
- You can start/stop agents at any time without affecting the sweep
- Bayesian optimization improves search strategy based on completed runs

______________________________________________________________________

### Step 6b - Hydra Multirun/Joblib: Run Hyperparameter Sweeps (Optional)

For hyperparameter tuning, you can use Hydra's multirun functionality with the `hydra-wandb-sweeper`
plugin to launch and track multiple training runs with different hyperparameter configurations.
Remember to set `config.use_wandb_logging=true` (required for sweep tracking). See also the wandb
[documentation on sweeps](https://docs.wandb.ai/models/sweeps) for more information on sweep
settings.

**Running a sweep:** for simple sweeps, you can specify arguments directly on the command line:

```bash
# basic random sweep example:
python -m pyine.apps.trainers.hf_trainer \
  --multirun \
  hydra.mode=MULTIRUN \
  +experiment=<your_experiment_name> \
  hydra/sweeper=wandb \
  hydra.sweeper.wandb_sweep_config.name=some_sweep, \
  hydra.sweeper.wandb_sweep_config.method=random, \
  hydra.sweeper.wandb_sweep_config.budget=10, \
  +hydra.sweeper.params.dummy_param=[1,2,3,4,5], \
  +hydra.sweeper.params.learning_rate=[1.0e-5,2.0e-5,5.0e-5]
```

**Using a sweep configuration file:** for more complex sweeps, define a `hydra/sweeper` section
in your experiment configuration itself, and set all required values there; for example:

```yaml
# @package _global_
defaults:
  - override /config: base
  # ...
  - override /hydra/sweeper: wandb_sweeper_base  # inherits some defaults from project configs
  - _self_

# ...

hydra:
  sweeper:
    wandb_sweep_config:
      name: "some sweep name"
      method: bayes  # options: grid, random, bayes
      metric:
        name: eval/loss
        goal: minimize
    params:
      # the `config.<...>` prefixes correspond to the nested structure of args in your app
      config.training_args_config.learning_rate:
        distribution: "log_uniform_values"
        min: 1.0e-3
        max: 1.0e-5
      config.training_args_config.gradient_accumulation_steps: [2, 4, 6, 8]
```

Then run:

```bash
python -m pyine.apps.trainers.hf_trainer \
  --multirun \
  hydra.mode=MULTIRUN \
  +experiment=<your_experiment_name>
```

**Remember:**

- Trainer sweeps require `config.use_wandb_logging=true`; the apps will raise an error otherwise;
- All runs in a sweep are logged to Weights & Biases under a sweep project;
- Sweep results can be visualized in the W&B dashboard;
- You can monitor and control sweeps via the W&B web interface.

______________________________________________________________________

### Step 7: Evaluate the Model Organism

Evaluate your trained model organism to confirm it possesses the desired bias or behavior.

**For HuggingFace models:**

Evaluation typically runs automatically at the end of training. To run standalone evaluation:

```bash
# Standard evaluation with HuggingFace inference
python -m pyine.apps.trainers.hf_trainer \
  +experiment=<your_experiment_name> \
  config.training_args_config.do_train=false \
  config.training_args_config.do_predict=true
```

**vLLM-Accelerated Evaluation (Recommended for Speed):**

For faster evaluation, you can use vLLM to serve your trained model and perform inference via an OpenAI-compatible API. This approach offers:

1. **Faster Inference**: vLLM provides optimized inference that's typically 2-10× faster than standard HuggingFace inference
2. **LLM-Based Grading**: Option to use a powerful local model (or OpenAI API) to judge prediction quality, providing more flexible matching than exact string comparison

**Prerequisites:**

- Ensure your `.env` file is properly configured at the repository root (the vLLM server script will automatically find and load it);
- LoRA checkpoints will be merged and cached at `<PYINE_CACHE_ROOT>/vllm_merged_models/<checkpoint_name>` for reuse.

**Quick Start:**

```bash
# 1. Start vLLM server with your trained model
# from the scripts/vllm_eval/ folder
uv run python vllm_server.py \
    --checkpoint_path /path/to/your/checkpoint \
    --port 8000
# Model name will be auto-derived from checkpoint path (last 3 components)
# Merged models (if LoRA) will be saved to <PYINE_CACHE_ROOT>/vllm_merged_models/

# 2. Run evaluation using vLLM inference
uv run python -m pyine.apps.trainers.hf_trainer \
    +experiment=original/v0_50perc_dataset_qwen3_vllm_eval.yaml
```

**With LLM-Based Grading** (using a second vLLM server for grading):

```bash
# Terminal 1: Evaluation server (your trained model)
# from the scripts/vllm_eval/ folder
uv run python vllm_server.py \
    --checkpoint_path /path/to/checkpoint \
    --cuda_devices 0,1,2,3 \
    --port 8000
# Model name auto-derived from checkpoint path (e.g., "runs/exp_123/checkpoint-1600")

# Terminal 2: Grading server (powerful base model)
# from the scripts/vllm_eval/ folder
uv run python vllm_server.py \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --cuda_devices 4,5,6,7 \
    --port 8001
# Model name will be "Qwen/Qwen3-4B-Instruct-2507"

# Run evaluation with grading
uv run python -m pyine.apps.trainers.hf_trainer \
    +experiment=original/v0_50perc_dataset_qwen3_vllm_eval_vllm_grading.yaml
```

When grading is enabled, you'll see three accuracy metrics:

- `accuracy_hard`: Exact string match
- `accuracy_soft`: Heuristic-based matching
- `accuracy_grader`: LLM-based judgment (most flexible)

For complete setup instructions, configuration options, troubleshooting, and advanced usage, see the [vLLM Evaluation and Grading Guide](./scripts/vllm_eval/README.md).

**For OpenAI models:**

Evaluation runs automatically during the fine-tuning workflow. Results are logged to:

- Console output
- Weights & Biases (if enabled)
- Run directory logs

**Evaluation outputs:**

- Metrics (accuracy, loss, task-specific scores);
- Per-subset predictions and analysis;
- Evaluation tables (when W&B logging is enabled).

For detailed analysis, see the evaluation notebooks in [`notebooks/`](./notebooks/README.md).

______________________________________________________________________

### Step 8: Develop Control/Alignment Strategies (TODO)

**Status:** This step is not yet implemented in the framework.

**Planned workflow:**

Once a model organism is trained and evaluated:

1. Design a control/alignment strategy to detect and correct a model organism's bias;
2. Train the strategy using specialized apps (if needed);
3. Evaluate the strategy's effectiveness against one or more model organism;
4. Iterate on the strategy based on evaluation results.

______________________________________________________________________

## Additional Workflows

### Exploratory Data Analysis

Use the provided notebooks to explore datasets and results:

- `taco_source_data_viz.ipynb`: explore the TACO source dataset;
- `trace_datasets_eda.ipynb`: analyze traces datasets;
- `sample_builder_outputs_eda.ipynb`: examine training sample distributions;
- `prompt_result_viewer.ipynb`: browse prompt-based evaluation results.

For more, see [`notebooks/README.md`](./notebooks/README.md).

### Prompt-Based Annotation

Generate annotations over traces using LLM-powered prompt chains:

```bash
# Annotate traces with a specific prompt
python -m pyine.apps.annotate.trace_annot_generator \
    --dataset /path/to/traces_dataset.lmdb \
    --prompt-name code_summary \
    --llm-option provider=openai \
    --llm-option model=gpt-4o-mini
```

Annotations are stored in the prompt results database (`<PYINE_DATA_ROOT>/prompt_results.sqlite`)
and can be explored via notebooks.

For more details, see [`pyine/apps/README.md`](./pyine/apps/README.md#trace-annotation-prompt-chains-over-traces)
and [`pyine/prompts/README.md`](./pyine/prompts/README.md).

______________________________________________________________________

## Tips and Best Practices

**Environment Setup:**

- Always configure your `.env` file with necessary environment variables (see [`.env.template`](./.env.template));
- Set `PYINE_DATA_ROOT` and `PYINE_LOGS_ROOT` to manage large artifacts outside the repo if needed.

**Reproducibility:**

- Use consistent seeds in your experiment configs (`runtime.seed`);
- Version your experiment configs and track them in git;
- The framework automatically logs reproducibility metadata with each run.

**Resource Management:**

- Use the precacher (`hf_precacher.py`) before large training runs to avoid cache conflicts;
- Partition large datasets for distributed processing;
- Monitor disk usage in `PYINE_DATA_ROOT` and `PYINE_LOGS_ROOT`.

**Debugging:**

- Use `runtime.dry_run=true` to validate configs without running full experiments;
- Check `--help` for any app to see available options;
- Use `--cfg job` with Hydra apps to inspect resolved configurations.

**Collaboration:**

- Keep experiment configs organized in subdirectories (e.g., `experiment/user_name/`);
- Use descriptive `exp_name` values in runtime configs;
- Document experiment goals and results in commit messages or separate notes.

______________________________________________________________________

## Getting Help

- Run any app with `-h` or `--help` for usage information;
- Check the [main README](./README.md) for installation and setup;
- See [CONTRIBUTING.md](./CONTRIBUTING.md) for development guidelines;
- Review notebooks in [`notebooks/`](./notebooks/) for hands-on examples;
- Contact the maintainers for dataset access or research questions.
