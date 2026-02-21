# Evaluation Package (`pyine.evals`)

This package provides the evaluation infrastructure for benchmarking model organisms and guardrails
on code execution tasks. It is designed around a config-driven, task-agnostic core with task-specific
subpackages.

NOTE: for now, only the code execution evaluation is supported (through the `CODE_EXEC` eval task).
Future updates should add the eval task, metrics, and workflow for the guardrails.

## Package Layout

```
pyine/evals/
├── common.py        # base types: EvalType, EvalResult, BaseEvalsConfig, GenerationEvalsConfig, ...
├── configs.py       # provides hydra-zen/pydantic configs for e.g. LLM grader providers
├── constants.py     # package-wide constants: aggregation stat names/functions, types, etc.
├── logging.py       # defines DiskEvalLogger and utils: LMDB-based export of evaluation artifacts
├── utils.py         # provides TokenUsageInfo class, category extraction helpers, metric helpers
└── code_exec/       # code execution evaluation task (for model organism capability evals)
    ├── _impl.py     # evaluate_runnable_model, evaluate_hf_model, finalize_evaluation_results
    ├── configs.py   # provides CodeExecEvalsConfig (extends GenerationEvalsConfig)
    ├── evaluator.py # provides OutcomeEvaluator: hard match, soft match, optional LLM grading
    ├── analysis.py  # offline analysis helpers (W&B fetch, DataFrames, plotting)
    ├── reeval.py    # reevaluate_from_lmdb: re-score stored predictions without model invocation
    └── utils.py     # SampleEval, CodeExecEvalArtifact, CodeExecEvalResult, metric computation, etc.
```

## Evaluation Flow

Evaluations are triggered by e.g. training apps, which loop over configured `eval_subset_names` and
call the appropriate evaluation method on the config object:

1. **Config dispatch**: `BaseEvalsConfig.evaluate_runnable_model()` /
   `evaluate_hf_model()` delegate to task-specific implementations based on `eval_type`;
2. **Model invocation**: the task implementation generates predictions (via LangChain chain
   invocation or HF text generation);
3. **Outcome evaluation**: an `OutcomeEvaluator` scores each prediction against ground truth;
4. **Metric aggregation**: `finalize_evaluation_results` computes global and category-wise metrics,
   builds eval artifacts, and optionally exports them to LMDB databases;
5. **Logging**: high-level results are logged to W&B (metrics tables, prediction tables, etc.) and
   optionally written to disk via `DiskEvalLogger`.

## Config Hierarchy

```
BaseEvalsConfig                 # eval_type, category_extraction_config, runnable_config
└── GenerationEvalsConfig       # generation params, Pass@K, disk_export_config, output_parsing_config
    └── CodeExecEvalsConfig     # evaluator_kwargs (soft match tolerance, LLM grader config)
```

`BaseEvalsConfig` defines the overridable evaluation methods (`evaluate_runnable_model`,
`evaluate_hf_model`, `log_metrics`, etc.). `GenerationEvalsConfig` adds fields shared across
generation-based tasks: batch size, temperature, `num_attempts_per_sample` for Pass@K, and optional
`EvalExportConfig` / `ParsingConfig`. Task-specific configs (like `CodeExecEvalsConfig`) add
evaluator parameters.

## Category-Wise Metrics

`SampleCategoryExtractor` derives categories from sample metadata fields (code type, predict type,
tags, keyword presence, identifier suffix). Category-wise metrics (e.g. accuracy, token usage,
complexity stats per category) are computed alongside global metrics and logged with category
prefixes (e.g., `code_type/original/accuracy_hard`).

## Evaluating Language Models

Language models can be evaluated using one of two paths, depending on their nature:

**HuggingFace path** (`evaluate_hf_model`): uses `transformers.GenerationMixin` for batched text
generation. Handles generation config resolution, `num_return_sequences` for multi-attempt, and
prompt preparation via chat templates. Intended for all HuggingFace-derived language models.

**Runnable path** (`evaluate_runnable_model`): invokes a LangChain `Runnable` chain per sample.
Supports sequential and parallel execution (sliding window concurrency). Intended for all API-based
or vLLM-based language models.

Both paths converge at `finalize_evaluation_results`, which handles metric aggregation,
category-wise breakdowns, artifact construction, and optional disk export.

When `output_parsing_config` is set, raw model outputs are parsed via `TagsOutputParser`
(see `pyine.utils.parsing`) to extract structured `<final>` and `<reasoning>` blocks before
evaluation. The parsed `final_answer` is used as the predicted value for the evaluator; the
full `ParsedOutput` (raw, final_answer, reasoning) is carried on for downstream use.

The parser and its config (`ParsingConfig`, `ParsedOutput`, `OutputParser` protocol) live in
`pyine.utils.parsing` and are shared with the reward pipeline.

## Metrics Reference

The tables below list all metrics produced by the evaluation pipeline. Note that when
`num_attempts_per_sample > 1`, each sample is evaluated K times with sampling enabled. The
`pass_at_k_values` argument controls which K values are computed (auto-derived as `[1, K]` when
not explicitly set). The evaluator groups attempts by sample identifier and computes Pass@K metrics
alongside per-attempt accuracy.

Metrics marked with CI emit two additional keys suffixed with `_ci_lower` and `_ci_upper` (95%
confidence level by default). Two CI strategies are used:

- **Wilson score interval** for accuracy and proportion metrics (`accuracy_*`, `majority_correct_*`,
  `mean_output_diversity`, `mean_unique_outputs`). Appropriate for bounded proportions, especially
  at small sample sizes.
- **SEM-based normal approximation** for Pass@K metrics (`pass_at_{k}_*`), computed
  as mean +/- z * std / sqrt(n_samples) and clamped to [0, 1].

See `pyine.utils.metrics.confidence` for the Wilson implementation and `pyine.utils.metrics.multi_sample`
for the Pass@K estimator and its CI.

All metrics listed below are global; when category extraction is enabled, each metric is also emitted
per-category with a `{category}/` prefix (e.g., `code_type/original/accuracy_hard`).

### Code Execution Metrics (`CODE_EXEC`)

#### Core Accuracy (always emitted)

| Metric            | Type  | CI  | Description                                                                   |
| ----------------- | ----- | --- | ----------------------------------------------------------------------------- |
| `sample_count`    | int   |     | Number of unique samples evaluated                                            |
| `attempt_count`   | int   |     | Total evaluation attempts (= K x sample_count)                                |
| `accuracy_hard`   | float | yes | Fraction of attempts with exact string match                                  |
| `accuracy_soft`   | float | yes | Fraction of attempts with heuristic soft match                                |
| `accuracy_grader` | float | yes | Fraction of attempts above LLM grader threshold (only when grader configured) |
| `grader_mean`     | float |     | Mean LLM grader score across all attempts                                     |
| `grader_median`   | float |     | Median LLM grader score                                                       |
| `grader_std`      | float |     | Standard deviation of LLM grader scores                                       |
| `grader_min`      | float |     | Minimum LLM grader score                                                      |
| `grader_max`      | float |     | Maximum LLM grader score                                                      |

#### Multi-Attempt (emitted when `num_attempts_per_sample > 1`)

| Metric                  | Type  | CI  | Description                                                       |
| ----------------------- | ----- | --- | ----------------------------------------------------------------- |
| `pass_at_{k}_hard`      | float | yes | Pass@K using hard match (for each k in `pass_at_k_values`)        |
| `pass_at_{k}_soft`      | float | yes | Pass@K using soft match                                           |
| `majority_correct_hard` | float | yes | Fraction of samples where majority of attempts are correct (hard) |
| `majority_correct_soft` | float | yes | Fraction of samples where majority of attempts are correct (soft) |
| `mean_output_diversity` | float | yes | Mean ratio of unique outputs to total attempts per sample         |
| `mean_unique_outputs`   | float | yes | Mean number of unique outputs per sample                          |

#### Token Usage (always emitted)

| Metric prefix                            | Aggregation             | Description                                 |
| ---------------------------------------- | ----------------------- | ------------------------------------------- |
| `total_token_usage/{token_type}`         | sum                     | Cumulative token counts across all attempts |
| `attempt_token_usage/{token_type}_{agg}` | mean/median/std/min/max | Per-attempt token usage statistics          |

Where `{token_type}` is one of: `total_tokens`, `prompt_tokens`, `cached_tokens`, `reasoning_tokens`,
`completion_tokens`.

#### Code Complexity (always emitted)

| Metric prefix               | Aggregation             | Description                              |
| --------------------------- | ----------------------- | ---------------------------------------- |
| `complexity/{metric}_{agg}` | mean/median/std/min/max | Radon-derived code complexity statistics |

Where `{metric}` includes: `cyclomatic_complexity_avg`, `cyclomatic_complexity_max`,
`cyclomatic_complexity_sum`, `loc`, `lloc`, `sloc`, `comments`, `multi`, `blank`,
`halstead_volume`, `halstead_difficulty`, `halstead_effort`, `maintainability_index`.

### Guardrail Metrics (planned)

*To be added.* Future guardrail evaluation metrics (e.g., AUROC, confusion matrices, accuracy,
Safe@K, FPR@TPR, etc) will be documented here once the guardrail eval task is implemented.

## LMDB Exportation

When `disk_export_config` is set on the eval config, evaluation artifacts are written to LMDB via
`DiskEvalLogger`. Each evaluation subset gets its own LMDB in a subdirectory:
`{output_path}/{eval_subset_name}/`.

**Record schema**: Records use `SharedGenerationRecordFields` (`pyine.data.utils.generation_record`)
for columns shared with `DiskRewardLogger` (sample ID, model output, prompt, expected output,
parsed fields, tags, categories), plus eval-specific columns (match results, grader score, token
usage, sample metadata).

**LMDB key format**: `{key_prefix}{sample_identifier}/{attempt_index}`

**Metadata**: `record_type` (`"benchmark"`), `eval_subset_name`, `export_metadata` (full eval
config, datamodule config, model info), `category_to_identifiers`, and optionally `aggregated_metrics`.

**Prompt capture**: Prompts are captured conditionally (only when export is enabled). The HF path
stores the chat-templated prompt text; the runnable path stores structured chat messages via
`CaptureLLMHandler.on_chat_model_start`. Validation is deferred to export time so capture failures
don't abort the evaluation loop.

## Re-Evaluation from LMDB

`reevaluate_from_lmdb` reads stored predictions from exported LMDBs and re-runs the evaluator
without model invocation. Use cases include re-scoring with a different LLM grader, recomputing
metrics with different Pass@K values, or re-categorizing samples.

When the original evaluation used output parsing, the stored `final_answer` is used as the predicted
value (matching original behavior), and the full `ParsedOutput` is reconstructed on the returned
artifacts. Categories are reconstructed from per-record data when no explicit extraction config is
provided.
