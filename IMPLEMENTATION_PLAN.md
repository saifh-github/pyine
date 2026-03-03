# Implementation Plan: Prompted LLM Guardrail

## 1. Overview

Add a new guardrail type: a **prompted (not fine-tuned) LLM** that analyzes a trained model's reasoning and answer to a code output prediction problem and judges whether the answer is correct. The guardrail receives `EvalRecord` instances from the LMDB-backed evaluation pipeline, uses a LangChain chain (`prompt | model`) to ask an LLM to judge correctness, and returns continuous confidence scores (0–1) via the existing `GuardrailScorer` protocol.

This slots in alongside the two existing guardrail baselines:
- **Activation probes** (`pyine/guardrails/probes/`) — requires base model activations
- **Fine-tuned LLM classifiers** (`pyine/guardrails/llm_classifier/`) — requires a fine-tuned encoder checkpoint

The new prompted LLM guardrail is inference-only (no training step) and uses the existing LLM provider infrastructure (`LLMProviderConfig`, `get_model_from_provider`).

---

## 2. New Files to Create

### 2.1 Package: `pyine/guardrails/prompted_llm/`

```
pyine/guardrails/prompted_llm/
├── __init__.py          # Package docstring, convenience re-exports
├── configs.py           # PromptedLLMGuardrailConfig (Pydantic)
└── scorer.py            # PromptedLLMGuardrailScorer (implements GuardrailScorer)
```

### 2.2 Prompt Template: `pyine/prompts/templates/guardrail/correctness_judge.yaml`

New YAML prompt template under a `guardrail/` subdirectory inside the existing `templates/` directory. Follows the same YAML structure as `pred_grader.yaml`.

### 2.3 Prompt Config Module: `pyine/prompts/configs/guardrail/correctness_judge.py`

Defines the structured output schema (`CorrectnessJudgement`) and optional `get_output_parser`/`get_prompt_template` overrides, following the pattern in `pyine/prompts/configs/pred_grader.py`.

### 2.4 Tests

```
tests/guardrails/prompted_llm/
├── __init__.py
├── test_scorer.py           # Unit tests for PromptedLLMGuardrailScorer
└── test_configs.py          # Unit tests for PromptedLLMGuardrailConfig
```

### 2.5 Hydra Config (Optional YAML Override)

```
configs/guardrail/prompted_llm_base.yaml   # Example Hydra YAML override
```

---

## 3. Configuration Design

### 3.1 `PromptedLLMGuardrailConfig` (`pyine/guardrails/prompted_llm/configs.py`)

```python
import pydantic
import pyine.utils.llm_providers
import pyine.prompts.types


class PromptedLLMGuardrailConfig(pydantic.BaseModel):
    """Configuration for the prompted LLM guardrail scorer."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    # --- LLM provider ---
    llm_provider: pyine.utils.llm_providers.LLMProviderConfig
    """LLM provider config (provider, model, rate limiter, retry, etc.)."""

    # --- Prompt ---
    prompt_name: str = "guardrail/correctness_judge"
    """Name of the prompt template to use (resolved by PromptManager)."""
    prompt_version: str | None = None
    """Prompt version override. None uses the default version."""
    use_chat_template: bool = True
    """Whether to use a chat prompt template (system + human message)."""

    # --- Scoring ---
    max_workers: int = pydantic.Field(default=10, ge=1)
    """Maximum number of concurrent LLM calls (ThreadPoolExecutor worker threads).
    Rate limiting is also handled by the LLMProviderConfig's rate_limiter_config."""
    default_score_on_error: float = pydantic.Field(default=0.5, ge=0.0, le=1.0)
    """Score to assign when the LLM call fails after retries."""
```

**Design notes:**
- `llm_provider` re-uses the existing `LLMProviderConfig`, which already handles rate limiting (`rate_limiter_config`) and retries (`with_retry_config`). No new rate-limiting logic needed.
- `max_workers` controls the `ThreadPoolExecutor` thread pool size for concurrent I/O-bound LLM API calls.
- No `text_field` config: the scorer always passes `model_output`, `final_answer`, and `expected_output` from the `EvalRecord` directly to the prompt template. Customizing what the LLM sees is better handled by different prompt template versions.
- `default_score_on_error` provides graceful degradation; errors are logged and counted in metadata.

### 3.2 No Changes to `CorrectnessEvalsConfig`

The existing `CorrectnessEvalsConfig.evaluate_wrapped_model()` already accepts any `GuardrailScorer`-compatible object. The `PromptedLLMGuardrailScorer` will be passed in as the `wrapped_model` argument, just like `ProbeScorer` and `LLMClassifierScorer` are today. No config changes needed — the caller (app/notebook) instantiates the scorer from `PromptedLLMGuardrailConfig` and passes it in.

---

## 4. Prompt Template Design

### 4.1 YAML Template: `pyine/prompts/templates/guardrail/correctness_judge.yaml`

```yaml
__defines__:
  base_metadata: &base_metadata
    name: guardrail/correctness_judge
    description: >
      Judges whether a model's predicted code execution output is correct
      by comparing it against the expected output, given the model's reasoning.

with_reasoning:
  metadata:
    <<: *base_metadata
  role:
    template: >
      You are an expert judge evaluating whether a language model correctly predicted
      the output of executing Python code. You will be given the model's full reasoning
      and answer, along with the expected output. Your job is to assess whether the
      model's final answer is correct.
  context:
    template: |
      A language model was asked to predict the output of executing some Python code.
      You are given:
      - The model's full response (which may include reasoning/chain-of-thought)
      - The model's extracted final answer (if available)
      - The expected correct output

      You must judge whether the model's answer is CORRECT or INCORRECT, and provide
      a confidence score between 0 and 1:
      - 1.0 = definitely correct (exact or semantically equivalent match)
      - 0.0 = definitely incorrect (completely wrong)
      - Values in between reflect partial correctness or uncertainty

      Focus on SEMANTIC CORRECTNESS: minor formatting differences (whitespace, quotes,
      equivalent numeric representations like 1 vs 1.0, reordered dict keys) should
      not penalize the score significantly.

      {expected_output_format}
  question:
    template: |
      MODEL'S FULL RESPONSE:
      ```
      {{model_output}}
      ```
      {%- if final_answer %}

      MODEL'S EXTRACTED FINAL ANSWER:
      ```
      {{final_answer}}
      ```
      {%- endif %}

      EXPECTED CORRECT OUTPUT:
      ```
      {{expected_output}}
      ```

      Now judge whether the model's answer is correct. Provide your structured assessment:
    format: jinja2
    partial_variables:
      final_answer: ""
  example_template:
  examples_block_template:
  template_block_separator: "\n\n"
  # TODO: Add 1-2 few-shot examples (correct/incorrect judgements) to improve
  # LLM reliability, especially for smaller models. Start with zero-shot for
  # initial development and benchmark, then add examples if needed.
  examples: []

score_only:
  metadata:
    <<: *base_metadata
  role:
    template: >
      You are an expert judge evaluating whether a language model correctly predicted
      the output of executing Python code.
  context:
    template: |
      Judge whether the model's predicted output matches the expected output.
      Output a confidence score between 0.0 (definitely wrong) and 1.0 (definitely correct).
      Focus on semantic correctness; ignore minor formatting differences.

      {expected_output_format}
  question:
    template: |
      MODEL'S FULL RESPONSE:
      ```
      {{model_output}}
      ```
      {%- if final_answer %}

      MODEL'S EXTRACTED FINAL ANSWER:
      ```
      {{final_answer}}
      ```
      {%- endif %}

      EXPECTED CORRECT OUTPUT:
      ```
      {{expected_output}}
      ```

      Provide ONLY the structured score:
    format: jinja2
    partial_variables:
      final_answer: ""
  example_template:
  examples_block_template:
  template_block_separator: "\n\n"
  examples: []

# Default to with_reasoning for development and debugging (LLM explains its judgement).
# For production runs where speed/cost matter, use score_only version instead.
__default__: "with_reasoning"
```

### 4.2 Structured Output Schema (`pyine/prompts/configs/guardrail/correctness_judge.py`)

Follows the same inheritance pattern as `pred_grader.py`: score-only base class, with-reasoning extends it.

```python
# pyine/prompts/configs/guardrail/correctness_judge.py

import typing
import langchain_core.output_parsers
import pydantic

if typing.TYPE_CHECKING:
    import pyine.prompts.types


class CorrectnessJudgement(pydantic.BaseModel):
    """Structured output for the correctness judge guardrail (score only, base class)."""
    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    score: typing.Annotated[
        pydantic.StrictFloat, pydantic.Field(ge=0.0, le=1.0)
    ] = pydantic.Field(
        description="Confidence score in [0,1]. 1.0 = definitely correct, 0.0 = definitely incorrect.",
    )


class CorrectnessJudgementWithReasoning(CorrectnessJudgement):
    """Structured output with optional reasoning supporting the score."""

    reasoning: str | None = pydantic.Field(
        default=None,
        description="Optional brief reasoning explaining the judgement.",
    )


def get_output_parser(
    version: "pyine.prompts.types.PromptVersionType | None" = None,
) -> langchain_core.output_parsers.BaseOutputParser[typing.Any] | None:
    """Return the output parser for the correctness judge prompt."""
    if version == "score_only":
        model = CorrectnessJudgement
    elif version == "with_reasoning" or version is None:  # default
        model = CorrectnessJudgementWithReasoning
    else:
        raise NotImplementedError(f"Unsupported version: {version}")
    return langchain_core.output_parsers.PydanticOutputParser(pydantic_object=model)


def get_prompt_template(
    version: "pyine.prompts.types.PromptVersionType | None" = None,
    use_chat_template: bool = False,
    include_examples: bool = True,
    target_examples: int | list[int] | None = None,
    partial_vars: dict[str, typing.Any] | None = None,
    role_variables: dict[str, typing.Any] | None = None,
    context_variables: dict[str, typing.Any] | None = None,
    examples_block_variables: dict[str, typing.Any] | None = None,
) -> "pyine.prompts.types.PromptTemplate":
    """Module override that injects format instructions into the context."""
    import pyine.prompts.manager

    prompt_config = pyine.prompts.manager.get_prompt_config(
        "guardrail/correctness_judge", version=version
    )
    parser = get_output_parser(version=version)
    merged_context: dict[str, typing.Any] = dict(context_variables) if context_variables else {}
    if parser is not None:
        merged_context.setdefault("expected_output_format", parser.get_format_instructions())
    return prompt_config.create_prompt_template(
        use_chat_template=use_chat_template,
        include_examples=include_examples,
        target_examples=target_examples,
        partial_vars=partial_vars,
        role_variables=role_variables,
        context_variables=merged_context or None,
        examples_block_variables=examples_block_variables,
    )
```

### 4.3 Information Passed to the LLM

For each `EvalRecord`, the prompt variables are populated directly from the record fields — no configurable `text_field` indirection:

| Variable | Source | Description |
|----------|--------|-------------|
| `model_output` | `record.model_output` | Full model reasoning + answer (always present) |
| `final_answer` | `record.final_answer` | Parsed final answer (empty string when `None`; Jinja2 conditional hides the block) |
| `expected_output` | `record.expected_output` | Ground-truth expected output (always present) |

---

## 5. Scorer Implementation

### 5.1 `PromptedLLMGuardrailScorer` (`pyine/guardrails/prompted_llm/scorer.py`)

```python
"""PromptedLLMGuardrailScorer — GuardrailScorer for prompted (non-fine-tuned) LLMs."""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import typing

import pyine.evals.correctness.types as correctness_types
import pyine.prompts.manager
import pyine.utils.langchain

if typing.TYPE_CHECKING:
    from pyine.guardrails.prompted_llm.configs import PromptedLLMGuardrailConfig

logger = logging.getLogger(__name__)


class PromptedLLMGuardrailScorer:
    """GuardrailScorer adapter for a prompted (non-fine-tuned) LLM judge.

    Uses a LangChain chain (prompt | model | parser) to ask an LLM to judge
    whether a model's code execution prediction is correct. Returns continuous
    confidence scores (0–1) and tracks token costs per record.

    Concurrency is achieved via ThreadPoolExecutor with sync chain.invoke()
    calls, which is safe to call from within an already-running asyncio event
    loop (unlike asyncio.run(), which would crash).
    """

    def __init__(self, config: PromptedLLMGuardrailConfig) -> None:
        self._config = config
        self._llm = config.llm_provider.get_model()
        self._chain = pyine.prompts.manager.get_prompt_chain(
            model=self._llm,
            prompt_name=config.prompt_name,
            version=config.prompt_version,
            use_chat_template=config.use_chat_template,
            runnable_name="correctness_judge",
        )
        self._error_count: int = 0
        self._error_lock = threading.Lock()
        self._total_scored: int = 0

    def score_records(
        self,
        records: list[correctness_types.EvalRecord],
    ) -> correctness_types.ScoringResult:
        """Score records by asking an LLM to judge correctness.

        Uses ThreadPoolExecutor for concurrent I/O-bound LLM API calls.
        This is safe to call from within a running asyncio event loop
        (the eval pipeline's evaluate_wrapped_model is async).
        """
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._config.max_workers,
        ) as executor:
            future_to_idx = {
                executor.submit(self._score_single, record): idx
                for idx, record in enumerate(records)
            }
            results: list[tuple[float, float] | None] = [None] * len(records)
            completed = 0
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
                completed += 1
                if completed % 100 == 0 or completed == len(records):
                    logger.info(f"scored {completed}/{len(records)} records")

        scores: list[float] = []
        costs: list[float] = []
        for score, token_count in results:  # type: ignore[misc]
            scores.append(score)
            costs.append(token_count)

        self._total_scored += len(records)
        return correctness_types.ScoringResult(
            scores=scores,
            verification_costs=costs,
        )

    def _score_single(
        self,
        record: correctness_types.EvalRecord,
    ) -> tuple[float, float]:
        """Score a single record via sync chain.invoke(). Returns (score, token_count).

        Called from worker threads. Error counting is protected by a threading.Lock.
        """
        input_vars: dict[str, typing.Any] = {
            "model_output": record.model_output,
            "expected_output": record.expected_output,
        }
        # Pass final_answer when available; the Jinja2 template conditional
        # hides the block when final_answer is empty (the partial_variables default).
        if record.final_answer is not None:
            input_vars["final_answer"] = record.final_answer

        handler = pyine.utils.langchain.CaptureLLMHandler()

        try:
            result = self._chain.invoke(
                input_vars,
                config={"callbacks": [handler]},
            )
            # Extract score from structured output (Pydantic model or dict)
            if hasattr(result, "score"):
                score = float(result.score)
            elif isinstance(result, dict) and "score" in result:
                score = float(result["score"])
            else:
                logger.warning(
                    "LLM returned unexpected format for %s: %s",
                    record.sample_id, type(result),
                )
                score = self._config.default_score_on_error
                with self._error_lock:
                    self._error_count += 1

            # Clamp to [0, 1]
            score = max(0.0, min(1.0, score))

        except Exception:
            logger.warning(
                "LLM call failed for record %s, using default score",
                record.sample_id,
                exc_info=True,
            )
            score = self._config.default_score_on_error
            with self._error_lock:
                self._error_count += 1

        # Extract token usage from handler
        token_count = self._extract_token_count(handler)
        return score, token_count

    @staticmethod
    def _extract_token_count(
        handler: pyine.utils.langchain.CaptureLLMHandler,
    ) -> float:
        """Extract total token count from the capture handler."""
        end_event = handler.get_latest_event("llm_end")
        if end_event is None or end_event.response is None:
            return 0.0
        # LLMResult.llm_output may contain token_usage
        llm_output = end_event.response.llm_output or {}
        token_usage = llm_output.get("token_usage", {})
        total = token_usage.get("total_tokens", 0)
        if total > 0:
            return float(total)
        # Fallback: sum from generation info
        for generation_list in end_event.response.generations:
            for gen in generation_list:
                gen_info = getattr(gen, "generation_info", None) or {}
                usage = gen_info.get("usage", {})
                total += usage.get("total_tokens", 0)
        if total == 0:
            logger.debug(
                "token count is 0 for a successful LLM response; "
                "provider may not populate token usage fields"
            )
        return float(total)

    def get_metadata(self) -> dict[str, typing.Any]:
        """Return guardrail metadata for reporting."""
        return {
            "scorer_type": "prompted_llm",
            "prompt_name": self._config.prompt_name,
            "prompt_version": self._config.prompt_version,
            "provider": self._config.llm_provider.provider,
            "model_kwargs": self._config.llm_provider.model_kwargs,
            "max_workers": self._config.max_workers,
            "default_score_on_error": self._config.default_score_on_error,
            "total_scored": self._total_scored,
            "error_count": self._error_count,
        }

    def get_verification_cost_unit(self) -> str | None:
        """Return 'tokens' as the verification cost unit."""
        return "tokens"
```

### 5.2 Key Design Decisions

1. **ThreadPoolExecutor (not asyncio.run)**: The `GuardrailScorer` protocol defines `score_records` as synchronous, but it is called from within `evaluate_guardrail_replicas` which is `async def` — meaning there is already a running event loop. Using `asyncio.run()` would raise `RuntimeError`. Instead, we use `concurrent.futures.ThreadPoolExecutor` with sync `chain.invoke()` calls. Threads are ideal for I/O-bound LLM API calls and don't conflict with the running event loop. This matches the synchronous approach used by `ProbeScorer` and `LLMClassifierScorer`.

2. **Concurrency**: `ThreadPoolExecutor(max_workers=N)` bounds concurrent LLM calls. The `LLMProviderConfig.rate_limiter_config` handles token-bucket rate limiting at the provider level. These are complementary controls.

3. **Single-replica usage**: Unlike probes and classifiers (where replicas come from different training seeds), the typical usage for a prompted LLM is a single scorer instance: `guardrails=[scorer]`. With `temperature=0`, calling the same API multiple times is deterministic and wasteful. If cross-run aggregation is desired, vary the prompt version, model, or temperature across scorer instances rather than duplicating the same config.

4. **Thread-safe error counting**: `self._error_count` is incremented from worker threads, so it is protected by a `threading.Lock` to avoid race conditions.

5. **Token tracking**: Uses the existing `CaptureLLMHandler` (one per thread/record) to capture the LLM response and extract `token_usage` from the `LLMResult`. This is a total-tokens count (prompt + completion) reported per record. Note: some providers (especially vLLM) may not populate token usage fields; when the extracted count is 0 for a non-error response, a warning is logged. A `verification_costs` array of all zeros should be interpreted as "token tracking not available" rather than "zero cost".

6. **Error handling**: On LLM failure (after retries configured in `with_retry_config`), the scorer assigns `default_score_on_error` (0.5 by default — neutral, neither accept nor reject) and increments an error counter exposed via `get_metadata()`.

7. **Structured output**: The chain uses `PydanticOutputParser` to parse the LLM's response into a `CorrectnessJudgementWithReasoning` (or `CorrectnessJudgement`). The parser's format instructions are injected into the prompt's `{expected_output_format}` variable (in the f-string-format context block), following the same pattern as `pred_grader.py`. **TODO:** For models that support native structured output (OpenAI, DeepSeek), consider using `.with_structured_output(CorrectnessJudgement)` as a future enhancement to reduce parse failures.

8. **No `text_field` indirection**: The scorer always passes `model_output`, `final_answer`, and `expected_output` directly from the `EvalRecord` to the prompt template. This avoids the confusing semantics of a `text_field` config (which would cause the same value to appear in multiple prompt slots). Customizing what the LLM sees is better handled by different prompt template versions.

---

## 6. Integration with Correctness Eval Pipeline

### 6.1 How It Plugs In

The existing `CorrectnessEvalsConfig.evaluate_wrapped_model()` accepts any `GuardrailScorer` instance. The caller constructs the scorer and passes it in:

```python
# Example usage in an app or notebook:
from pyine.guardrails.prompted_llm.configs import PromptedLLMGuardrailConfig
from pyine.guardrails.prompted_llm.scorer import PromptedLLMGuardrailScorer

guardrail_config = PromptedLLMGuardrailConfig(
    llm_provider=LLMProviderConfig(
        provider="openai",
        model_kwargs={"model": "gpt-4o-mini", "temperature": 0.0},
        rate_limiter_config=get_default_openai_provider_rate_limit_config(),
        with_retry_config=get_default_openai_provider_retry_config(),
    ),
)
scorer = PromptedLLMGuardrailScorer(guardrail_config)

# Pass to the eval pipeline
result = await correctness_impl.evaluate_guardrail_replicas(
    config=eval_config,
    guardrails=[scorer],
    datamodule=datamodule,
    eval_subset_name="guardrail_test",
)
```

### 6.2 No Modifications Needed to Existing Pipeline

- `evaluate_guardrail_replicas()` calls `scorer.score_records(records)` → works unchanged
- `scorer.get_metadata()` → used in `SingleRunResult.guardrail_metadata` → works unchanged
- `scorer.get_verification_cost_unit()` returns `"tokens"` → cost metrics computed automatically
- The `ScoringResult` returned by the scorer includes both `scores` and `verification_costs` → all downstream metrics (AUROC, thresholded metrics, cost stats) computed automatically

---

## 7. Integration with GuardrailMetricsConnector (Performance Benchmarking)

The `GuardrailMetricsConnector` protocol is designed for measuring computational performance (FLOPs, latency, memory) of local guardrail models. The prompted LLM guardrail calls an external API, so the relevant performance metrics are different (latency, tokens, cost) and are already captured by the scorer itself.

**Decision: No `GuardrailMetricsConnector` implementation for this phase.** The token costs per record are already tracked in `ScoringResult.verification_costs` and reported in `VerificationCostStats`. If API latency benchmarking is needed later, a lightweight connector can be added.

---

## 8. Standalone Evaluation App

### 8.1 Problem

The existing guardrails (probes, LLM classifier) are evaluated as part of their training scripts — the trainer builds the scorer from the model it just trained, then calls `evaluate_guardrail_replicas()`. The prompted LLM guardrail has **no training step**, so there is no trainer to embed evaluation into. A standalone Hydra app is needed.

### 8.2 App Structure

```
pyine/apps/guardrail_eval/
├── __init__.py
├── prompted_llm_eval.py          # Hydra entrypoint
└── prompted_llm_eval_configs.py  # App config + Hydra registration
```

### 8.3 App Config (`prompted_llm_eval_configs.py`)

```python
class PromptedLLMEvalAppConfig(pydantic.BaseModel):
    """Standalone evaluation app for the prompted LLM guardrail.

    Requires no training — builds the scorer from an LLM provider config
    and runs the correctness evaluation pipeline on an LMDB dataset.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    guardrail_config: PromptedLLMGuardrailConfig
    """Prompted LLM guardrail configuration (LLM provider, prompt, concurrency)."""

    evals_config: CorrectnessEvalsConfig
    """Correctness eval pipeline configuration.

    Embeds CorrectnessDataModuleConfig which specifies the LMDB paths and split config.
    """

    use_wandb_logging: bool = False
    """Whether to log results to Weights & Biases."""
    wandb_project: str | None = None
    """W&B project name (only used when use_wandb_logging=True)."""
```

Note: the LMDB dataset path and split file are configured inside `evals_config.datamodule_config` — no extra data config field needed.

### 8.4 App Entrypoint (`prompted_llm_eval.py`)

```python
async def main(
    config: PromptedLLMEvalAppConfig,
    runtime: pyine.configs.schemas.RuntimeConfig | None = None,
) -> None:
    """Main entrypoint for prompted LLM guardrail evaluation.

    Follows the same boilerplate pattern as existing training apps
    (e.g. llm_classifier_trainer.py:main): entrypoint_setup, W&B run
    init, evaluation loop, finalize. See pyine/apps/trainers/common.py
    for the full lifecycle template.
    """
    # 0. Setup (seed, logging, optional W&B run init)
    # Use pyine.utils.reprod.entrypoint_setup(runtime) and
    # pyine.apps.trainers.common patterns for W&B wiring.

    # 1. Build scorer from guardrail config
    scorer = PromptedLLMGuardrailScorer(config.guardrail_config)

    # 2. Apply retry config programmatically (cannot be set via YAML;
    #    see Section 8.6 note on retry_if_exception_type limitation)
    # e.g. config.guardrail_config.llm_provider.with_retry_config =
    #       get_default_openai_provider_retry_config()

    # 3. Prepare datamodule (loads LMDB, builds guardrail splits)
    datamodule = config.evals_config.prepare_eval_datamodule(None)

    # 4. Run evaluation on each configured subset
    # Access eval_subset_names from the evals config's datamodule_config
    # (avoids needing to type-narrow the returned BaseDataModule)
    for subset_name in config.evals_config.datamodule_config.eval_subset_names:
        result = await correctness_impl.evaluate_guardrail_replicas(
            config=config.evals_config,
            guardrails=[scorer],
            datamodule=datamodule,
            eval_subset_name=subset_name,
        )

        # 5. Log results (W&B + console)
        log_eval_results(result, config, runtime, subset_name)

    # 6. Finalize (runtime.finalize() if applicable)
```

### 8.5 LLM Provider Hot-Swapping via LangChain

The scorer is **completely provider-agnostic** thanks to the existing `LLMProviderConfig` + LangChain abstraction. The LangChain chain built inside the scorer is always `prompt | model | parser`, where `model` is a `BaseChatOpenAI` instance — the same base class regardless of provider. Swapping LLMs is purely a config change:

```
┌──────────────────────────────────────────────────────────────┐
│                    LangChain Chain                            │
│                                                              │
│  ChatPromptTemplate ──► BaseChatOpenAI ──► PydanticParser    │
│                              │                               │
│                    ┌─────────┼─────────┐                     │
│                    │         │         │                      │
│              ┌─────▼───┐ ┌──▼────┐ ┌──▼───┐                 │
│              │ OpenAI   │ │ vLLM  │ │ Deep │                 │
│              │ API      │ │ local │ │ Seek │                 │
│              └─────────┘ └───────┘ └──────┘                  │
│                                                              │
│  Config-driven: only `provider` + `model_kwargs` change      │
│  Chain code: identical for all providers                     │
└──────────────────────────────────────────────────────────────┘
```

All providers return `BaseChatOpenAI` (LangChain's OpenAI-compatible base class):
- `provider="openai"` → `ChatOpenAI` (OpenAI API)
- `provider="vllm"` → `ChatOpenAI` with `base_url` pointed at local vLLM server
- `provider="deepseek"` → `ChatDeepSeek` (DeepSeek API)

Rate limiting and retry are configured per-provider via `rate_limiter_config` and `with_retry_config`, and are applied to the LLM instance before it enters the chain.

### 8.6 Hydra Experiment Configs

Create example Hydra YAML configs under `pyine/configs/experiment/guardrail/`:

#### `prompted_llm_eval_openai.yaml` — OpenAI API (e.g. GPT-4o-mini)

```yaml
# @package _global_
defaults:
  - _self_

runtime:
  exp_name: prompted_llm_guardrail_eval
  seed: 42

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
      # NOTE: with_retry_config is omitted from YAML because retry_if_exception_type
      # requires actual Python exception classes, not strings. Set it programmatically
      # using get_default_openai_provider_retry_config(), or register a structured
      # Hydra config that handles the string-to-class conversion.
      # with_retry_config: ...
    prompt_name: guardrail/correctness_judge
    prompt_version: with_reasoning
    max_workers: 10
    default_score_on_error: 0.5

  evals_config:
    datamodule_config:
      lmdb_paths:
        - ${oc.env:PYINE_DATA_ROOT}/eval_exports/my_model_eval.lmdb
      split_config:
        split_source: TACO
    target_fpr_values: [0.001, 0.01, 0.05]

  use_wandb_logging: false
```

#### `prompted_llm_eval_vllm.yaml` — Local open-source model via vLLM

```yaml
# @package _global_
defaults:
  - _self_

runtime:
  exp_name: prompted_llm_guardrail_eval_local
  seed: 42

config:
  guardrail_config:
    llm_provider:
      provider: vllm
      model_kwargs:
        model: meta-llama/Llama-3.1-8B-Instruct   # must match vLLM server --model
        temperature: 0.0
        base_url: http://localhost:8000/v1           # vLLM server address
      # No rate_limiter_config needed for local server (no API quotas)
      # No with_retry_config needed (local server is reliable)
    prompt_name: guardrail/correctness_judge
    prompt_version: with_reasoning
    max_workers: 8     # match GPU throughput
    default_score_on_error: 0.5

  evals_config:
    datamodule_config:
      lmdb_paths:
        - ${oc.env:PYINE_DATA_ROOT}/eval_exports/my_model_eval.lmdb
      split_config:
        split_source: TACO
    target_fpr_values: [0.001, 0.01, 0.05]

  use_wandb_logging: false
```

#### `prompted_llm_eval_deepseek.yaml` — DeepSeek API

```yaml
# @package _global_
defaults:
  - _self_

runtime:
  exp_name: prompted_llm_guardrail_eval_deepseek
  seed: 42

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
      # NOTE: with_retry_config omitted from YAML — see OpenAI config comment.
      # Set programmatically using get_default_openai_provider_retry_config().
    prompt_name: guardrail/correctness_judge
    prompt_version: with_reasoning
    max_workers: 10

  evals_config:
    datamodule_config:
      lmdb_paths:
        - ${oc.env:PYINE_DATA_ROOT}/eval_exports/my_model_eval.lmdb
      split_config:
        split_source: TACO

  use_wandb_logging: false
```

### 8.7 CLI Usage

```bash
# --- Closed-source LLMs (API-based) ---

# OpenAI GPT-4o-mini
python -m pyine.apps.guardrail_eval.prompted_llm_eval \
    +experiment=guardrail/prompted_llm_eval_openai

# DeepSeek
python -m pyine.apps.guardrail_eval.prompted_llm_eval \
    +experiment=guardrail/prompted_llm_eval_deepseek

# Override model on the fly
python -m pyine.apps.guardrail_eval.prompted_llm_eval \
    +experiment=guardrail/prompted_llm_eval_openai \
    config.guardrail_config.llm_provider.model_kwargs.model=gpt-4o

# --- Open-source LLMs (local vLLM server) ---

# Step 1: Start vLLM server (separate terminal)
uv run python scripts/vllm_eval/vllm_server.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000

# Step 2: Run evaluation against local server
python -m pyine.apps.guardrail_eval.prompted_llm_eval \
    +experiment=guardrail/prompted_llm_eval_vllm

# Override to a different local model
python -m pyine.apps.guardrail_eval.prompted_llm_eval \
    +experiment=guardrail/prompted_llm_eval_vllm \
    config.guardrail_config.llm_provider.model_kwargs.model=Qwen/Qwen3-4B-Instruct \
    config.guardrail_config.llm_provider.model_kwargs.base_url=http://localhost:8001/v1

# --- Point at a different LMDB dataset ---

python -m pyine.apps.guardrail_eval.prompted_llm_eval \
    +experiment=guardrail/prompted_llm_eval_openai \
    config.evals_config.datamodule_config.lmdb_paths=[/data/my_other_eval.lmdb]
```

---

## 9. Extend Debug Dataset for Correctness Eval Pipeline Compatibility

### 9.1 Problem

The existing debug LMDB generator (`pyine/guardrails/data/debug_dataset.py`) produces records compatible with the probe/classifier training pipeline (`load_probe_dataset_from_lmdb`), but **not** with the correctness eval pipeline (`load_records_from_lmdb` in `pyine/evals/correctness/data_loading.py`). The correctness pipeline requires:

1. **`record_type="benchmark"` in LMDB metadata** — validated at load time
2. **`TraceIdentifier`-format sample IDs** (e.g. `DEBUG/VALID/p000001/s0000/t0000`) — needed by `extract_problem_id()`
3. **Top-level `sample_id` and `attempt_index` fields** in each record dict
4. **Top-level `hard_match` and `soft_match` boolean fields** (not nested under `reward_metrics`)
5. **A companion `SplitResult` file** — `build_guardrail_splits()` needs a split file to assign problems to train/valid/test

Currently, tests for probes and the LLM classifier use the debug LMDB, but there is no way to test the prompted LLM guardrail (or any guardrail through the correctness eval pipeline) without a real evaluation export.

### 9.2 Goal

Extend the debug dataset generator so that **one debug LMDB serves all three guardrail types** (probes, LLM classifier, prompted LLM) and both data loading paths (probe training and correctness eval).

### 9.3 Changes to `pyine/guardrails/data/debug_dataset.py`

#### 9.3.1 Update sample ID format

Change from flat IDs to `TraceIdentifier`-compatible format:

```python
# Before:
sample_id = f"debug_sample_{sample_idx:04d}"           # train
base_id = f"debug_problem_{family_idx:03d}/s0000/t0000" # eval

# After:
sample_id = f"DEBUG/TRAIN/p{sample_idx:06d}/s0000/t0000"   # train
base_id = f"DEBUG/VALID/p{family_idx:06d}/s0000/t0000"      # eval (valid problems)
# Also add test families:
base_id = f"DEBUG/TEST/p{family_idx:06d}/s0000/t0000"       # eval (test problems)
```

This ensures `extract_problem_id()` can parse them into `CodingProblemIdentifier(dataset="DEBUG", subset="TRAIN"|"VALID"|"TEST", problem_idx=N)`.

#### 9.3.2 Add top-level fields required by correctness loader

Update `_make_record()` to include the fields that `load_records_from_lmdb()` reads:

```python
record: dict[str, typing.Any] = {
    # --- existing fields (keep for probe/classifier compatibility) ---
    "prompt": prompt,
    "model_output": completion,
    "expected_output": expected,
    "final_answer": final_answer,
    "reasoning": None,
    "reward_total": float(label),
    "reward_terms": { ... },
    "reward_metrics": { ... },
    "predict_type": "program_output",
    "code_type": code_type,
    "tags": tags,
    "key_prefix": key_prefix,
    # --- new fields for correctness eval compatibility ---
    "sample_id": sample_id,
    "attempt_index": 0,       # single attempt per record in debug data
    "hard_match": bool(label),
    "soft_match": bool(label),
}
```

Both loading paths are satisfied: the probe loader uses `reward_metrics`, the correctness loader uses `hard_match`/`soft_match` directly.

#### 9.3.3 Write `record_type="benchmark"` metadata

Add metadata write after creating all records:

```python
with pyine.data.utils.lmdb_io.LMDBWriter(output_path, ...) as writer:
    # ... write records ...
    writer.write_metadata({"record_type": "benchmark"})
```

The probe loader doesn't check metadata, so this is backwards-compatible.

#### 9.3.4 Add test problem families

Currently the generator creates train records and eval (valid) families. For the correctness pipeline, we also need test problems. Add a `n_test_families` parameter:

```python
def create_debug_probe_lmdb(
    output_path: str | pathlib.Path,
    n_train: int = 200,
    n_eval_families: int = 30,
    n_test_families: int = 15,    # NEW
    seed: int = 42,
    noise_rate: float = _NOISE_RATE,
) -> pathlib.Path:
```

Eval families use `DEBUG/VALID/p{idx:06d}/...`, test families use `DEBUG/TEST/p{idx:06d}/...`.

#### 9.3.5 New helper: `create_debug_split_file()`

Add a function that creates a minimal `SplitResult` JSON file matching the debug LMDB's problem IDs:

```python
def create_debug_split_file(
    output_path: str | pathlib.Path,
    n_train: int = 200,
    n_eval_families: int = 30,
    n_test_families: int = 15,
    seed: int = 42,
) -> pathlib.Path:
    """Create a SplitResult file compatible with the debug LMDB.

    Assigns problem IDs to train/valid/test subsets matching the debug LMDB
    structure. The split file can be passed as `split_source` to
    `GuardrailSplitConfig` for correctness eval pipeline testing.

    Returns:
        Path to the created split file.
    """
    import pyine.data.utils.splits

    # Build problem identifiers matching create_debug_probe_lmdb() output
    identifiers: list[str] = []
    subset_assignments: dict[str, str] = {}

    # Train problems
    for idx in range(n_train):
        pid = f"DEBUG/TRAIN/p{idx:06d}"
        identifiers.append(pid)
        subset_assignments[pid] = "train"

    # Valid problems (eval families)
    for idx in range(n_eval_families):
        pid = f"DEBUG/VALID/p{idx:06d}"
        identifiers.append(pid)
        subset_assignments[pid] = "valid"

    # Test problems
    for idx in range(n_test_families):
        pid = f"DEBUG/TEST/p{idx:06d}"
        identifiers.append(pid)
        subset_assignments[pid] = "test"

    split_result = pyine.data.utils.splits.SplitResult(
        source_dataset_name="DEBUG",
        source_dataset_hash="debug-dataset-hash",
        identifiers=identifiers,
        tag_lists=[[] for _ in identifiers],
        source_data_hashes=[f"debug-hash-{i:06d}" for i in range(len(identifiers))],
        subset_assignments=subset_assignments,
        creation_metadata={"generator": "debug_dataset", "seed": seed},
        config=pyine.data.utils.splits.SplitConfig(
            subset_names=["train", "valid", "test"],
            subset_assign_prob_map={"train": 0.6, "valid": 0.2, "test": 0.2},
        ),
    )

    output_path = pathlib.Path(output_path)
    split_result.to_file(output_path)
    return output_path
```

#### 9.3.6 Update `create_debug_probe_dataset()` convenience wrapper

Add optional `include_split_file` parameter that also creates the split file alongside the LMDB. Returns a structured result to avoid conditional return types:

```python
@dataclasses.dataclass
class DebugDatasetResult:
    """Result of creating a debug probe dataset."""
    dataset: datasets.DatasetDict
    split_file_path: pathlib.Path | None = None

def create_debug_probe_dataset(
    output_path: str | pathlib.Path | None = None,
    # ... existing params ...
    include_split_file: bool = False,   # NEW
) -> DebugDatasetResult:
```

When `include_split_file=True`, the returned `DebugDatasetResult.split_file_path` is populated.

### 9.4 Backwards Compatibility

All changes are **additive**:
- Existing fields (`reward_metrics`, `reward_terms`, `key_prefix`) are preserved
- New fields (`sample_id`, `attempt_index`, `hard_match`, `soft_match`) are ignored by the probe loader
- `record_type="benchmark"` metadata is ignored by the probe loader (it doesn't check)
- The new `n_test_families` parameter defaults to `15`, so existing calls without it still work
- `create_debug_probe_dataset()` now returns a `DebugDatasetResult` dataclass; `split_file_path` is `None` by default

### 9.5 Test Updates

| File | Change |
|------|--------|
| `tests/guardrails/data/test_debug_dataset.py` (existing or new) | Test that the generated LMDB loads successfully via both `load_probe_dataset_from_lmdb()` and `load_records_from_lmdb()` |
| `tests/guardrails/data/test_debug_dataset.py` | Test that `create_debug_split_file()` produces a valid `SplitResult` that can be loaded by `get_dataset_split_result()` |
| `tests/guardrails/data/test_debug_dataset.py` | Test that `build_guardrail_splits()` succeeds with the debug LMDB + debug split file |
| `tests/evals/integration/` | Update integration test fixtures to use the debug LMDB + split instead of any hardcoded test data |

---

## 10. Files to Modify

### 10.1 `pyine/prompts/configs/guardrail/__init__.py` (NEW)

Create this file (empty or with docstring) so that `guardrail` is a proper subpackage under `pyine/prompts/configs/`.

### 10.2 No `__init__.py` for `pyine/prompts/templates/guardrail/`

The `templates/` directory is a resource directory traversed by `importlib.resources.files()`, not a Python package. Existing subdirectories (`hints/`, `issues/`, `validation/`) do not have `__init__.py` files. The new `guardrail/` subdirectory follows the same convention — no `__init__.py` needed.

### 10.3 `pyine/guardrails/data/debug_dataset.py` (MODIFIED)

Extend the debug dataset generator as described in Section 9: add `TraceIdentifier`-format sample IDs, top-level correctness eval fields, `record_type="benchmark"` metadata, test families, and a `create_debug_split_file()` helper.

### 10.4 No changes to:
- `pyine/evals/correctness/types.py` — protocol already supports this
- `pyine/evals/correctness/scorers.py` — the new scorer lives in its own package
- `pyine/evals/correctness/configs.py` — no config additions needed
- `pyine/evals/correctness/_impl.py` — already handles any `GuardrailScorer`
- `pyine/utils/llm_providers.py` — already has everything needed
- `pyine/utils/langchain.py` — already has `CaptureLLMHandler`
- `pyine/prompts/manager.py` — already discovers subdirectory templates

---

## 11. Test Plan

### 11.1 Unit Tests: `tests/guardrails/prompted_llm/test_configs.py`

| Test | Description |
|------|-------------|
| `test_config_defaults` | Verify default values of `PromptedLLMGuardrailConfig` |
| `test_config_validation_max_workers` | Ensure `max_workers < 1` raises `ValidationError` |
| `test_config_validation_default_score_bounds` | Ensure `default_score_on_error` outside [0,1] raises |
| `test_config_frozen` | Ensure config is immutable (frozen) |

### 11.2 Unit Tests: `tests/guardrails/prompted_llm/test_scorer.py`

These tests mock the LLM chain to avoid real API calls.

| Test | Description |
|------|-------------|
| `test_score_records_returns_correct_shape` | Verify `len(scores) == len(records)` and `len(verification_costs) == len(records)` |
| `test_score_records_scores_in_range` | All scores are in [0, 1] |
| `test_score_records_costs_non_negative` | All verification costs >= 0 |
| `test_error_handling_uses_default_score` | When chain raises, score = `default_score_on_error` |
| `test_error_count_incremented` | After errors, `get_metadata()["error_count"]` is correct |
| `test_get_metadata_contains_required_fields` | Verify metadata keys: `scorer_type`, `prompt_name`, etc. |
| `test_get_verification_cost_unit` | Returns `"tokens"` |
| `test_protocol_conformance` | Verify the scorer satisfies `GuardrailScorer` protocol via structural check |
| `test_final_answer_passed_when_present` | When `record.final_answer` is set, it appears in the chain input |
| `test_final_answer_omitted_when_none` | When `record.final_answer` is None, it is not in the chain input |

**Mocking strategy**: Patch the LangChain chain's `invoke` method to return a mock `CorrectnessJudgementWithReasoning` with a predetermined score. Use `tests/evals/integration/conftest.py::make_eval_record()` to create test `EvalRecord` instances.

### 11.3 Integration Test: `tests/evals/integration/test_correctness_integration.py`

Add a new test class `TestCorrectnessPromptedLLMScorer` alongside the existing `TestCorrectnessOracleScorer` etc. This test uses a **fake prompted LLM scorer** (added to `fake_models.py`) that mimics the interface without real API calls.

| Test | Description |
|------|-------------|
| `test_prompted_llm_scorer_produces_valid_metrics` | Run through full eval pipeline, verify AggregatedResult structure |
| `test_prompted_llm_scorer_cost_unit_is_tokens` | Verify cost_unit in VerificationCostStats |

### 11.4 Fake Model Addition: `tests/evals/integration/fake_models.py`

Add `FakePromptedLLMScorer` — a scorer that returns deterministic scores based on `record.label` (like `OracleGuardrailScorer` but with `scorer_type: "prompted_llm"` metadata and token-based costs). This avoids needing real API calls in integration tests.

```python
class FakePromptedLLMScorer:
    """Fake prompted LLM scorer for integration testing.

    Behaves like an oracle but reports metadata consistent with a prompted LLM guardrail.
    """
    def score_records(self, records: list[EvalRecord]) -> ScoringResult:
        return ScoringResult(
            scores=[1.0 if rec.label else 0.0 for rec in records],
            verification_costs=[150.0] * len(records),  # ~150 tokens per call
        )

    def get_metadata(self) -> dict[str, Any]:
        return {"scorer_type": "prompted_llm", "name": "fake_prompted_llm"}

    def get_verification_cost_unit(self) -> str | None:
        return "tokens"
```

### 11.5 Prompt Template Test

| Test | Description |
|------|-------------|
| `test_prompt_template_loads` | Verify `get_prompt_config("guardrail/correctness_judge")` succeeds |
| `test_prompt_template_renders` | Verify the template renders with sample input variables without errors |
| `test_prompt_output_parser` | Verify `get_output_parser()` returns a `PydanticOutputParser` that parses a valid JSON string |

---

## 12. Documentation and Guides

### 12.1 Docstrings

All new public classes and functions will have comprehensive docstrings following the project's existing Google-style docstring convention.

### 12.2 Config Example in Code Comments

The `PromptedLLMGuardrailConfig` docstring will include a usage example:

```python
# Example:
config = PromptedLLMGuardrailConfig(
    llm_provider=LLMProviderConfig(
        provider="openai",
        model_kwargs={"model": "gpt-4o-mini", "temperature": 0.0},
    ),
)
scorer = PromptedLLMGuardrailScorer(config)
```

### 12.3 No separate documentation files needed

The feature is self-documenting via config classes, docstrings, and the YAML prompt template. No README or standalone docs required unless explicitly requested.

---

## 13. File Summary

### New Files (15)

| File | Purpose |
|------|---------|
| `pyine/guardrails/prompted_llm/__init__.py` | Package init with re-exports |
| `pyine/guardrails/prompted_llm/configs.py` | `PromptedLLMGuardrailConfig` Pydantic config |
| `pyine/guardrails/prompted_llm/scorer.py` | `PromptedLLMGuardrailScorer` implementation |
| `pyine/prompts/templates/guardrail/correctness_judge.yaml` | Prompt template (no `__init__.py` — resource dir) |
| `pyine/prompts/configs/guardrail/__init__.py` | Package init for config subpackage |
| `pyine/prompts/configs/guardrail/correctness_judge.py` | Output schema + parser override |
| `pyine/apps/guardrail_eval/__init__.py` | Standalone eval app package init |
| `pyine/apps/guardrail_eval/prompted_llm_eval.py` | Hydra entrypoint for prompted LLM evaluation |
| `pyine/apps/guardrail_eval/prompted_llm_eval_configs.py` | App config + Hydra registration |
| `pyine/configs/experiment/guardrail/prompted_llm_eval_openai.yaml` | Experiment config: OpenAI API |
| `pyine/configs/experiment/guardrail/prompted_llm_eval_vllm.yaml` | Experiment config: local vLLM server |
| `pyine/configs/experiment/guardrail/prompted_llm_eval_deepseek.yaml` | Experiment config: DeepSeek API |
| `tests/guardrails/prompted_llm/__init__.py` | Test package init |
| `tests/guardrails/prompted_llm/test_scorer.py` | Unit tests for scorer |
| `tests/guardrails/prompted_llm/test_configs.py` | Unit tests for config |

### Modified Files (2)

| File | Change |
|------|--------|
| `pyine/guardrails/data/debug_dataset.py` | Extend for correctness eval compatibility (Section 9) |
| `tests/evals/integration/fake_models.py` | Add `FakePromptedLLMScorer` class |

### Unchanged Files (many)

All core pipeline files (`types.py`, `scorers.py`, `configs.py`, `_impl.py`, `connector.py`, `llm_providers.py`, `langchain.py`, `manager.py`) remain untouched. The new guardrail plugs in via the existing `GuardrailScorer` protocol with zero modifications to the evaluation pipeline.

---

## 14. Implementation Order

1. **Extend debug dataset** (`debug_dataset.py` — Section 9: TraceIdentifier IDs, top-level fields, metadata, split file helper)
2. **Test debug dataset** (verify both loading paths work with the updated LMDB)
3. **Create prompt template** (`correctness_judge.yaml` + `correctness_judge.py` config module)
4. **Create config** (`PromptedLLMGuardrailConfig`)
5. **Create scorer** (`PromptedLLMGuardrailScorer`)
6. **Create package init** (`__init__.py` files for `prompted_llm/` and `prompts/configs/guardrail/`)
7. **Create standalone eval app** (`pyine/apps/guardrail_eval/` — entrypoint, config, `__init__.py`)
8. **Create Hydra experiment configs** (`prompted_llm_eval_openai.yaml`, `prompted_llm_eval_vllm.yaml`, `prompted_llm_eval_deepseek.yaml`)
9. **Add fake scorer** to `fake_models.py`
10. **Write unit tests** (`test_configs.py`, `test_scorer.py`)
11. **Write integration test** (extend `test_correctness_integration.py`, using debug LMDB + split)
12. **Manual smoke test** against a real LLM provider via the standalone app (not automated)
