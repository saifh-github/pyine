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

For data format, splits, metrics, and the `GuardrailScorer` protocol, see
[`pyine/evals/correctness/README.md`](../../evals/correctness/README.md).

______________________________________________________________________

## Prerequisites

### Dependencies

Required packages (should already be installed in the project environment):

- `langchain-core`
- `langchain-openai`
- `langchain-deepseek` (for DeepSeek provider)

### API Keys

| Provider | Environment Variable | Notes                                                                                                                 |
| -------- | -------------------- | --------------------------------------------------------------------------------------------------------------------- |
| OpenAI   | `OPENAI_API_KEY`     | Required. Optionally set `OPENAI_BASE_URL` for custom endpoints.                                                      |
| DeepSeek | `DEEPSEEK_API_KEY`   | Required. Optionally set `DEEPSEEK_API_BASE_URL` (defaults to `https://api.deepseek.com/v1`).                         |
| vLLM     | (none)               | No API key needed. Set `VLLM_BASE_URL` or pass `base_url` in `model_kwargs` (defaults to `http://localhost:8000/v1`). |

### Local vLLM Server (optional)

```bash
uv run python scripts/vllm_eval/vllm_server.py \
    --model meta-llama/Llama-3.1-8B-Instruct --port 8000
```

______________________________________________________________________

## Quick Start

### 1. Prepare Data

The pipeline reads from LMDB databases exported by `DiskEvalLogger` during the code execution eval
pipeline. For debug/smoke-testing, generate a synthetic dataset:

```bash
python -m pyine.guardrails.data.debug_dataset \
    --output /tmp/debug-guardrail-lmdb \
    --n-test-families 15
```

Or programmatically (LMDB + split file in one call):

```python
from pyine.guardrails.data.debug_dataset import create_debug_probe_dataset

result = create_debug_probe_dataset(
    output_path="/tmp/debug-guardrail-lmdb",
    n_test_families=15,
    include_split_file=True,
)
```

### 2. Create an Experiment Config

All configs inherit from `guardrail/prompted_llm_eval_base` and only override the LLM provider.
Create a file in `pyine/configs/experiment/guardrail/`:

```yaml
# @package _global_
defaults:
  - guardrail/prompted_llm_eval_base
  - _self_

config:
  guardrail_config:
    llm_provider:
      provider: openai
      model_kwargs:
        model: gpt-5-mini
        temperature: 0.0
      rate_limiter_config:
        requests_per_second: 50
        max_bucket_size: 50
```

The `_target_` field is handled automatically by the Hydra config registration in
`prompted_llm_eval_configs.py` — no explicit `_target_` is needed in experiment YAML files.

### 3. Run

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

## Configuration

### Base Config Defaults (`prompted_llm_eval_base.yaml`)

| Field                             | Default                       |
| --------------------------------- | ----------------------------- |
| `runtime.seed`                    | `42`                          |
| `prompt_name`                     | `guardrail/correctness_judge` |
| `max_workers`                     | `10`                          |
| `default_score_on_error`          | `0.5`                         |
| `default_score_on_missing_answer` | `0.0`                         |
| `split_source`                    | `TACO`                        |
| `target_fpr_values`               | `[0.001, 0.01, 0.05]`         |
| `use_wandb_logging`               | `false`                       |

### App Config (`PromptedLLMEvalAppConfig`)

Defined in `pyine/apps/guardrail_eval/prompted_llm_eval_configs.py`:

| Field               | Type                         | Default    | Description                             |
| ------------------- | ---------------------------- | ---------- | --------------------------------------- |
| `guardrail_config`  | `PromptedLLMGuardrailConfig` | (required) | Guardrail scorer configuration          |
| `evals_config`      | `CorrectnessEvalsConfig`     | (required) | Correctness eval pipeline configuration |
| `use_wandb_logging` | `bool`                       | `False`    | Whether to log results to W&B           |
| `wandb_project`     | `str \| None`                | `None`     | W&B project name                        |

### Guardrail Config (`PromptedLLMGuardrailConfig`)

Defined in `pyine/guardrails/prompted_llm/configs.py`:

| Field                    | Type                | Default                         | Description                                                                          |
| ------------------------ | ------------------- | ------------------------------- | ------------------------------------------------------------------------------------ |
| `llm_provider`           | `LLMProviderConfig` | (required)                      | LLM provider config (provider, model_kwargs, rate_limiter_config, with_retry_config) |
| `prompt_name`            | `str`               | `"guardrail/correctness_judge"` | Prompt template name resolved by PromptManager                                       |
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

______________________________________________________________________

## CLI Usage

### Common Overrides

```bash
# Change model
config.guardrail_config.llm_provider.model_kwargs.model=gpt-5-mini

# Change concurrency
config.guardrail_config.max_workers=20

# Set target FPR values
config.evals_config.target_fpr_values='[0.01,0.05]'

# Enable W&B logging
config.use_wandb_logging=true config.wandb_project=my-guardrail-experiments
```

### Pointing at the Debug Dataset

```bash
python -m pyine.apps.guardrail_eval.prompted_llm_eval \
    +experiment=guardrail/prompted_llm_eval_openai \
    config.evals_config.datamodule_config.lmdb_paths='[/tmp/debug-guardrail-lmdb]' \
    config.evals_config.datamodule_config.split_config.split_source=/tmp/debug-guardrail-lmdb/debug_split.json
```

### Using a Different Local vLLM Model

```bash
python -m pyine.apps.guardrail_eval.prompted_llm_eval \
    +experiment=guardrail/prompted_llm_eval_vllm \
    config.guardrail_config.llm_provider.model_kwargs.model=Qwen/Qwen3-4B-Instruct \
    config.guardrail_config.llm_provider.model_kwargs.base_url=http://localhost:8001/v1
```

______________________________________________________________________

## Programmatic Usage

### Building a Scorer

```python
from pyine.guardrails.prompted_llm import (
    PromptedLLMGuardrailConfig,
    PromptedLLMGuardrailScorer,
)
from pyine.utils.llm_providers import LLMProviderConfig

config = PromptedLLMGuardrailConfig(
    llm_provider=LLMProviderConfig(
        provider="openai",
        model_kwargs={"model": "gpt-5-mini", "temperature": 0.0},
    ),
)
scorer = PromptedLLMGuardrailScorer(config)
```

### Running the Eval Pipeline

```python
import asyncio
import pyine.evals.correctness._impl as correctness_impl
import pyine.evals.correctness.configs as correctness_configs

evals_config: correctness_configs.CorrectnessEvalsConfig = ...
datamodule = evals_config.prepare_eval_datamodule(None)

result = asyncio.run(
    correctness_impl.evaluate_guardrail_replicas(
        config=evals_config,
        guardrails=[scorer],
        datamodule=datamodule,
        eval_subset_name="guardrail_test",
    )
)

print(result.metrics)                  # flat dict for W&B logging
print(result.aggregated.cross_run_mean)  # mean across runs
```

### Scorer Metadata

```python
scorer.get_metadata()
# {
#     "scorer_type": "prompted_llm",
#     "prompt_name": "guardrail/correctness_judge",
#     "provider": "openai",
#     "model_kwargs": {"model": "gpt-5-mini", "temperature": 0.0},
#     "max_workers": 10,
#     "default_score_on_error": 0.5,
#     "default_score_on_missing_answer": 0.0,
#     "total_scored": <int>,
#     "error_count": <int>,
#     "skipped_no_final_answer": <int>,
# }

scorer.get_verification_cost_unit()  # "tokens"
```
