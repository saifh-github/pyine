# Prompted LLM Guardrail Evaluation Guide

## What It Is

The prompted LLM guardrail evaluator runs a **prompted (non-fine-tuned) LLM** as a judge to assess
whether a target model's code execution predictions are correct. It implements the `GuardrailScorer`
protocol and plugs directly into the existing correctness evaluation pipeline
(`evaluate_guardrail_replicas`).

Unlike the probe trainer (lightweight heads on frozen activations) and the LLM classifier trainer
(fine-tuned encoder model), this guardrail has **no training step** — the scorer is built directly
from an LLM provider config and evaluated immediately.

Key characteristics:

- **Inference-only** — no training step; the scorer is built from an `LLMProviderConfig`
- **LangChain chain** — uses `prompt | model | parser` to produce structured output
  (`CorrectnessJudgement` with a `score` field in [0, 1])
- **Token cost tracking** — records per-record token usage via `CaptureLLMHandler`
- **Concurrent** — uses `ThreadPoolExecutor` (not asyncio) for safe concurrent LLM calls
- **Multi-provider** — supports OpenAI, DeepSeek, and local vLLM servers via `LLMProviderConfig`

______________________________________________________________________

## Prerequisites

### Dependencies

The prompted LLM guardrail uses LangChain for chain construction and LLM provider integration.
Required packages (should already be installed in the project environment):

- `langchain-core`
- `langchain-openai`
- `langchain-deepseek` (for DeepSeek provider)

### API Keys

Set the appropriate environment variable for your chosen provider:

| Provider | Environment Variable | Notes                                                                                                                 |
| -------- | -------------------- | --------------------------------------------------------------------------------------------------------------------- |
| OpenAI   | `OPENAI_API_KEY`     | Required. Optionally set `OPENAI_BASE_URL` for custom endpoints.                                                      |
| DeepSeek | `DEEPSEEK_API_KEY`   | Required. Optionally set `DEEPSEEK_API_BASE_URL` (defaults to `https://api.deepseek.com/v1`).                         |
| vLLM     | (none)               | No API key needed. Set `VLLM_BASE_URL` or pass `base_url` in `model_kwargs` (defaults to `http://localhost:8000/v1`). |

### Local vLLM Server (optional)

To use a local open-source model, start a vLLM server first:

```bash
uv run python scripts/vllm_eval/vllm_server.py \
    --model meta-llama/Llama-3.1-8B-Instruct --port 8000
```

______________________________________________________________________

## Quick Start

### Step 1: Prepare Data

The evaluation pipeline reads from an **LMDB database** exported by `DiskEvalLogger` during the
code execution eval pipeline. Each record contains the prompt, model output, expected output, and
correctness labels. A companion **split file** assigns problems to `guardrail_train` /
`guardrail_valid` / `guardrail_test` subsets.

#### Using the Debug LMDB (for Testing)

A built-in synthetic LMDB factory generates mock records for smoke-testing:

```bash
# Generate a debug LMDB
python -m pyine.guardrails.data.debug_dataset \
    --output /tmp/debug-guardrail-lmdb \
    --n-test-families 15
```

Create the companion split file (Python):

```python
from pyine.guardrails.data.debug_dataset import create_debug_split_file

create_debug_split_file(
    output_path="/tmp/debug-guardrail-lmdb/debug_split.json",
    n_train=200,
    n_eval_families=30,
    n_test_families=15,
)
```

Or use the convenience wrapper that combines both steps:

```python
from pyine.guardrails.data.debug_dataset import create_debug_probe_dataset

result = create_debug_probe_dataset(
    output_path="/tmp/debug-guardrail-lmdb",
    n_test_families=15,
    include_split_file=True,
)
# result.dataset        -> HF DatasetDict
# result.split_file_path -> Path to split file
```

> **Important**: The `n_train`, `n_eval_families`, `n_test_families`, and `seed` values must match
> between the LMDB and the split file.

### Step 2: Create an Experiment Config

All experiment configs inherit from a shared **base config**
(`guardrail/prompted_llm_eval_base`) that defines common defaults (seed, prompt settings,
eval pipeline config, etc.). Provider-specific configs only need to override what differs.

Create a new experiment config file in `pyine/configs/experiment/guardrail/`:

```yaml
# @package _global_
#
# Usage:
#   python -m pyine.apps.guardrail_eval.prompted_llm_eval \
#       +experiment=guardrail/my_eval

defaults:
  - guardrail/prompted_llm_eval_base
  - _self_

config:
  guardrail_config:
    llm_provider:
      provider: openai
      model_kwargs:
        model: gpt-4o-mini
        temperature: 0.0
      rate_limiter_config:
        requests_per_second: 50
        max_bucket_size: 50
```

The base config (`prompted_llm_eval_base.yaml`) provides these defaults:

| Field                    | Default                       |
| ------------------------ | ----------------------------- |
| `runtime.seed`           | `42`                          |
| `prompt_name`            | `guardrail/correctness_judge` |
| `prompt_version`         | `with_reasoning`              |
| `max_workers`            | `10`                          |
| `default_score_on_error` | `0.5`                         |
| `split_source`           | `TACO`                        |
| `target_fpr_values`      | `[0.001, 0.01, 0.05]`         |
| `use_wandb_logging`      | `false`                       |

Any of these can be overridden in the provider-specific config or on the command line.

**Important:** The `_target_` field is handled automatically by the Hydra config registration in
`prompted_llm_eval_configs.py`. No explicit `_target_` is needed in experiment YAML files.

### Step 3: Run Evaluation

```bash
python -m pyine.apps.guardrail_eval.prompted_llm_eval \
    +experiment=guardrail/my_eval
```

### Available Experiment Configs

| Config                                 | Provider     | Model                   | Notes                                  |
| -------------------------------------- | ------------ | ----------------------- | -------------------------------------- |
| `guardrail/prompted_llm_eval_base`     | —            | —                       | Shared base config (not used directly) |
| `guardrail/prompted_llm_eval_openai`   | OpenAI       | `gpt-5-mini`            | Rate limited at 50 req/s               |
| `guardrail/prompted_llm_eval_vllm`     | vLLM (local) | `Llama-3.1-8B-Instruct` | Requires running vLLM server           |
| `guardrail/prompted_llm_eval_deepseek` | DeepSeek     | `deepseek-chat`         | Rate limited at 20 req/s               |

______________________________________________________________________

## Data Source

### LMDB Format

The evaluator reads from LMDB databases exported by `DiskEvalLogger` during the code execution
eval pipeline. Records are serialized with `JSON_ZSTD` (orjson + zstandard compression).

The correctness data loader (`load_records_from_lmdb`) requires:

1. **`record_type="benchmark"` in LMDB metadata** — validated at load time
2. **`TraceIdentifier`-format sample IDs** (e.g. `TACO/train/p000001/s0000/t0000`)
3. **Top-level `sample_id` and `attempt_index` fields** in each record
4. **Top-level `hard_match` and `soft_match` boolean fields**

### Record Fields

Each LMDB record is a JSON dict. The fields used by the prompted LLM scorer:

| Field             | Required | Description                                                 |
| ----------------- | -------- | ----------------------------------------------------------- |
| `model_output`    | Always   | The model's full reasoning and predicted output             |
| `expected_output` | Always   | The ground-truth expected output                            |
| `final_answer`    | Optional | The model's extracted final answer (if available)           |
| `hard_match`      | Always   | Boolean: exact match between predicted and expected         |
| `soft_match`      | Always   | Boolean: soft/semantic match between predicted and expected |
| `sample_id`       | Always   | Unique sample identifier in `TraceIdentifier` format        |
| `attempt_index`   | Always   | Generation attempt index                                    |

### Splits

The evaluation pipeline assigns problems to `guardrail_train`, `guardrail_valid`, and
`guardrail_test` subsets via a split file. The split file is a `SplitResult` JSON that maps
problem identifiers to subset names.

Configure via `evals_config.datamodule_config.split_config`:

| Field                      | Default    | Description                                                                                |
| -------------------------- | ---------- | ------------------------------------------------------------------------------------------ |
| `split_source`             | (required) | HF dataset name (e.g. `"TACO"`) or path to a `SplitResult` JSON file                       |
| `guardrail_valid_fraction` | `0.5`      | Fraction of non-train problems assigned to `guardrail_valid` (rest go to `guardrail_test`) |

______________________________________________________________________

## Configuration

### App Config (`PromptedLLMEvalAppConfig`)

The main config class in `pyine/apps/guardrail_eval/prompted_llm_eval_configs.py`:

| Field               | Type                         | Default    | Description                                                               |
| ------------------- | ---------------------------- | ---------- | ------------------------------------------------------------------------- |
| `guardrail_config`  | `PromptedLLMGuardrailConfig` | (required) | Guardrail scorer configuration                                            |
| `evals_config`      | `CorrectnessEvalsConfig`     | (required) | Correctness eval pipeline configuration (LMDB paths, splits, target FPRs) |
| `use_wandb_logging` | `bool`                       | `False`    | Whether to log results to W&B                                             |
| `wandb_project`     | `str \| None`                | `None`     | W&B project name                                                          |

### Guardrail Config (`PromptedLLMGuardrailConfig`)

The scorer configuration in `pyine/guardrails/prompted_llm/configs.py`:

| Field                    | Type                | Default                         | Description                                                                          |
| ------------------------ | ------------------- | ------------------------------- | ------------------------------------------------------------------------------------ |
| `llm_provider`           | `LLMProviderConfig` | (required)                      | LLM provider config (provider, model_kwargs, rate_limiter_config, with_retry_config) |
| `prompt_name`            | `str`               | `"guardrail/correctness_judge"` | Prompt template name resolved by PromptManager                                       |
| `prompt_version`         | `str \| None`       | `None`                          | Prompt version override (`"with_reasoning"`, `"score_only"`, or `None` for default)  |
| `use_chat_template`      | `bool`              | `True`                          | Whether to use chat prompt template (system + human message)                         |
| `max_workers`            | `int`               | `10`                            | Max concurrent LLM calls (ThreadPoolExecutor workers)                                |
| `default_score_on_error` | `float`             | `0.5`                           | Score assigned when LLM call fails after retries                                     |

### LLM Provider Config (`LLMProviderConfig`)

| Field                 | Type                               | Default    | Description                                                                              |
| --------------------- | ---------------------------------- | ---------- | ---------------------------------------------------------------------------------------- |
| `provider`            | `"openai" \| "deepseek" \| "vllm"` | (required) | LLM provider name                                                                        |
| `model_kwargs`        | `dict`                             | `{}`       | Kwargs passed to the LangChain LLM constructor (e.g. `model`, `temperature`, `base_url`) |
| `rate_limiter_config` | `dict \| None`                     | `None`     | Config for `InMemoryRateLimiter` (e.g. `requests_per_second`, `max_bucket_size`)         |
| `with_retry_config`   | `dict \| None`                     | `None`     | Config for LangChain `.with_retry()` (set programmatically for exception types)          |

### Prompt Versions

| Version          | Description                                                          | Use case                          |
| ---------------- | -------------------------------------------------------------------- | --------------------------------- |
| `with_reasoning` | LLM provides reasoning + score (`CorrectnessJudgementWithReasoning`) | Development, debugging, analysis  |
| `score_only`     | LLM provides only the score (`CorrectnessJudgement`)                 | Production runs (faster, cheaper) |

### Eval Pipeline Config (`CorrectnessEvalsConfig`)

Key fields within `evals_config`:

| Field               | Type                          | Default               | Description                                         |
| ------------------- | ----------------------------- | --------------------- | --------------------------------------------------- |
| `datamodule_config` | `CorrectnessDataModuleConfig` | (required)            | LMDB paths + split config                           |
| `target_fpr_values` | `list[float]`                 | `[0.001, 0.01, 0.05]` | Target false positive rates for thresholded metrics |

______________________________________________________________________

## CLI Usage

### Basic

```bash
python -m pyine.apps.guardrail_eval.prompted_llm_eval \
    +experiment=guardrail/prompted_llm_eval_openai
```

### Pointing at the Debug Dataset

Override the LMDB path and split source on the command line:

```bash
python -m pyine.apps.guardrail_eval.prompted_llm_eval \
    +experiment=guardrail/prompted_llm_eval_openai \
    config.evals_config.datamodule_config.lmdb_paths='[/tmp/debug-guardrail-lmdb]' \
    config.evals_config.datamodule_config.split_config.split_source=/tmp/debug-guardrail-lmdb/debug_split.json
```

### Common Overrides

```bash
# Change model
config.guardrail_config.llm_provider.model_kwargs.model=gpt-4o

# Change concurrency
config.guardrail_config.max_workers=20

# Use score_only prompt version (faster, no reasoning)
config.guardrail_config.prompt_version=score_only

# Set target FPR values
config.evals_config.target_fpr_values='[0.01,0.05]'

# Enable W&B logging
config.use_wandb_logging=true \
    config.wandb_project=my-guardrail-experiments
```

### Using a Local vLLM Server

```bash
# Step 1: Start vLLM server (separate terminal)
uv run python scripts/vllm_eval/vllm_server.py \
    --model meta-llama/Llama-3.1-8B-Instruct --port 8000

# Step 2: Run evaluation against local server
python -m pyine.apps.guardrail_eval.prompted_llm_eval \
    +experiment=guardrail/prompted_llm_eval_vllm

# Override to a different local model
python -m pyine.apps.guardrail_eval.prompted_llm_eval \
    +experiment=guardrail/prompted_llm_eval_vllm \
    config.guardrail_config.llm_provider.model_kwargs.model=Qwen/Qwen3-4B-Instruct \
    config.guardrail_config.llm_provider.model_kwargs.base_url=http://localhost:8001/v1
```

______________________________________________________________________

## Programmatic Usage

### Minimal Example

```python
from pyine.guardrails.prompted_llm import (
    PromptedLLMGuardrailConfig,
    PromptedLLMGuardrailScorer,
)
from pyine.utils.llm_providers import LLMProviderConfig

# 1. Configure
config = PromptedLLMGuardrailConfig(
    llm_provider=LLMProviderConfig(
        provider="openai",
        model_kwargs={"model": "gpt-4o-mini", "temperature": 0.0},
    ),
)

# 2. Build scorer
scorer = PromptedLLMGuardrailScorer(config)
```

### Running the Eval Pipeline Programmatically

```python
import asyncio
import pyine.evals.correctness._impl as correctness_impl
import pyine.evals.correctness.configs as correctness_configs

# Assume you have a CorrectnessEvalsConfig and datamodule set up
evals_config: correctness_configs.CorrectnessEvalsConfig = ...
datamodule = evals_config.prepare_eval_datamodule(None)

# Run evaluation
result = asyncio.run(
    correctness_impl.evaluate_guardrail_replicas(
        config=evals_config,
        guardrails=[scorer],
        datamodule=datamodule,
        eval_subset_name="guardrail_test",
    )
)

# Access metrics
print(result.metrics)  # flat dict for W&B logging
print(result.aggregated.cross_run_mean)  # mean across runs
```

### Inspecting Scorer Metadata

```python
metadata = scorer.get_metadata()
# {
#     "scorer_type": "prompted_llm",
#     "prompt_name": "guardrail/correctness_judge",
#     "prompt_version": None,
#     "provider": "openai",
#     "model_kwargs": {"model": "gpt-4o-mini", "temperature": 0.0},
#     "max_workers": 10,
#     "default_score_on_error": 0.5,
#     "total_scored": <int>,
#     "error_count": <int>,
# }

cost_unit = scorer.get_verification_cost_unit()
# "tokens"
```

______________________________________________________________________

## Metrics

The evaluation pipeline computes and logs the following metrics:

| Metric                            | Description                                   |
| --------------------------------- | --------------------------------------------- |
| AUROC                             | Area under ROC curve for the guardrail scores |
| Thresholded TPR / FPR / Precision | At each configured `target_fpr_values`        |
| `base_pass_rate`                  | Pass rate without the guardrail               |
| `guarded_pass_rate`               | Pass rate with the guardrail applied          |
| Verification cost stats           | Total cost, mean cost per record (in tokens)  |

Scorer metadata (logged after evaluation):

| Field          | Description                                                 |
| -------------- | ----------------------------------------------------------- |
| `scorer_type`  | Always `"prompted_llm"`                                     |
| `total_scored` | Total number of records scored                              |
| `error_count`  | Number of LLM call failures (used `default_score_on_error`) |

______________________________________________________________________

## YAML Examples

All provider configs inherit from the base config (`guardrail/prompted_llm_eval_base`) via the
Hydra `defaults` list and only override provider-specific fields.

### OpenAI API

See `pyine/configs/experiment/guardrail/prompted_llm_eval_openai.yaml`. Overrides the LLM
provider and points at debug data paths:

```yaml
defaults:
  - guardrail/prompted_llm_eval_base
  - _self_

config:
  guardrail_config:
    llm_provider:
      provider: openai
      model_kwargs:
        model: gpt-5-mini
        temperature: 1.0
      rate_limiter_config:
        requests_per_second: 50
        max_bucket_size: 50
```

### Local vLLM Server

See `pyine/configs/experiment/guardrail/prompted_llm_eval_vllm.yaml`. Overrides the LLM
provider, adds `base_url`, and lowers `max_workers` to match GPU throughput:

```yaml
defaults:
  - guardrail/prompted_llm_eval_base
  - _self_

config:
  guardrail_config:
    llm_provider:
      provider: vllm
      model_kwargs:
        model: meta-llama/Llama-3.1-8B-Instruct
        temperature: 0.0
        base_url: http://localhost:8000/v1
    max_workers: 8   # match GPU throughput
```

### DeepSeek API

See `pyine/configs/experiment/guardrail/prompted_llm_eval_deepseek.yaml`. Overrides the LLM
provider and rate limiter:

```yaml
defaults:
  - guardrail/prompted_llm_eval_base
  - _self_

config:
  guardrail_config:
    llm_provider:
      provider: deepseek
      model_kwargs:
        model: deepseek-chat
        temperature: 0.0
      rate_limiter_config:
        requests_per_second: 20
        max_bucket_size: 20
```

### Adding a New Provider

To add a new provider, create a config that inherits from the base and overrides only the
`llm_provider` section (and any other provider-specific fields):

```yaml
# @package _global_
defaults:
  - guardrail/prompted_llm_eval_base
  - _self_

config:
  guardrail_config:
    llm_provider:
      provider: openai           # or a new provider
      model_kwargs:
        model: my-model-name
        temperature: 0.0
```

### With `score_only` Prompt (Faster, Cheaper)

```yaml
config:
  guardrail_config:
    prompt_version: score_only
```

### With W&B Logging

```yaml
config:
  use_wandb_logging: true
  wandb_project: my-guardrail-experiments
```

______________________________________________________________________

## Troubleshooting

### Common Issues

**`ValueError: LMDB metadata 'record_type' must be 'benchmark', got ...`**

The LMDB was not exported by `DiskEvalLogger` (correctness eval pipeline). The prompted LLM
guardrail requires correctness-format LMDBs, not reward-format LMDBs from `DiskRewardLogger`.

**LLM call failures / high `error_count`**

- Check API key is set correctly for your provider
- Check rate limiter settings — if too aggressive, the provider may throttle requests
- For vLLM, verify the server is running and the `base_url` is correct
- Consider setting `with_retry_config` programmatically for automatic retries

**Low AUROC on debug dataset**

The debug dataset has a learnable keyword-correlated signal. If AUROC is near 0.5, the LLM may
not be parsing the structured output correctly. Try:

- Using a stronger model (e.g. `gpt-4o` instead of `gpt-4o-mini`)
- Switching to `prompt_version: with_reasoning` for better reliability
- Checking logs for parse errors in the structured output

**`RuntimeError: Cannot run asyncio.run() within a running event loop`**

This should not happen with the current implementation (uses `ThreadPoolExecutor` internally).
If it does, ensure you are calling the scorer through the standard evaluation pipeline and not
wrapping it in an additional `asyncio.run()`.

**Token costs are all zeros**

Some providers (especially local vLLM servers) do not populate token usage fields in their
responses. A `verification_costs` array of all zeros should be interpreted as "token tracking
not available" rather than "zero cost".
