# Evaluation Package (`pyine.evals`)

This package provides the evaluation infrastructure for benchmarking model organisms and guardrails
on code execution tasks. It is designed around a config-driven, task-agnostic core with task-specific
subpackages.

## Overview

```
                          BaseEvalsConfig
                                │
               ┌────────────────┴────────────────┐
               │                                 │
      eval_type=CODE_EXEC                 eval_type=CORRECTNESS
               │                                 │
     GenerationEvalsConfig                CorrectnessEvalsConfig
  (generation params, LMDB export,  (LMDB import/split, record categories,
        output parsing)               target FPRs, bootstrap settings)
               │                                 │
    ┌──────────┴──────────┐                      │
    │                     │                      │
 evaluate_hf_model   evaluate_runnable_model  evaluate_wrapped_model
 (HuggingFace)       (LangChain Runnable)     (GuardrailScorer)
    │                     │                      │
    └──────────┬──────────┘                      │
               │                        ┌────────┼────────┐
        K attempts per sample           │        │        │
               │                     Probes  Classifiers  Prompted LLM
        OutcomeEvaluator                │        │        │
     (hard match, soft match,           └────────┼────────┘
      optional LLM grader)                       │
               │                     load from LMDB exports
               │                                 │
               │                     split ──► score ──► calibrate
               │                                 │
               │                     compute threshold-free + thresholded
               │                           + sample-level metrics
               │                                 │
               │                          aggregate across R runs
               │                                 │
               ▼                                 ▼
        CodeExecEvalResult                CorrectnessEvalResult
        (accuracy, Pass@K, CIs,          (AUROC, TPR, pass rates,
         token usage, complexity)       bootstrap CIs, cost stats)
               │                                 │
               └────────────────┬────────────────┘
                                │
                            log to W&B
                   (metrics + prediction tables)
```

**Two evaluation tasks are supported:**

- **`CODE_EXEC`** (left path): evaluates the capability of models on code execution. Models predict
  outputs for Python programs; predictions are scored via exact match, soft match, and optional LLM
  grading. See [`code_exec/README.md`](code_exec/README.md).
- **`CORRECTNESS`** (right path): evaluates guardrails that classify whether a model's output is
  correct. Loads pregenerated outputs from LMDB, applies problem-level splits, calibrates a
  decision threshold, and computes classification + sample-level metrics across multiple runs.
  See [`correctness/README.md`](correctness/README.md).

Both tasks share a common config hierarchy, category extraction system, LMDB export format, and
W&B logging interface. **The records exported to LMDB in the code exec pipeline are meant to be
reused to recompute metrics while bypassing new model invocations, and also for evaluations in the
correctness pipeline.**

## Config Hierarchy

```
BaseEvalsConfig                 # eval_type, category_extraction_config, runnable_config
├── GenerationEvalsConfig       # generation params, disk_export_config, output_parsing_config
│   └── CodeExecEvalsConfig     # code-execution-specific: evaluator, pass_at_k_values, etc.
└── CorrectnessEvalsConfig      # datamodule_config, target_fprs, record categories, bootstrap
```

Callers construct the appropriate config and call `evaluate_[runnable,hf,wrapped]_model()`, which
dispatches to the task-specific implementation based on `eval_type`.

## Category-Wise Metrics

Both tasks compute metrics globally and per-category. Categories are extracted from sample metadata
fields (code type, predict type, keyword presence, identifier suffix, tags) via
`SampleCategoryExtractor`. Category-wise metrics are logged with a `{category}/` prefix (e.g.
`code_type/original/accuracy_hard`).

The correctness task additionally groups code types into semantic categories (`regular` vs
`biasing/*`) via `RecordCategoryConfig`, then merges those with base-extracted categories.

## Code Execution Task (`CODE_EXEC`)

See [`code_exec/README.md`](code_exec/README.md) for the full pipeline documentation, including
model invocation paths (HuggingFace and LangChain Runnable), outcome evaluation (hard match, soft
match, LLM grading), multi-attempt / Pass@K, output parsing, LMDB export, re-evaluation, analysis,
and metrics glossary.

## Guardrail Correctness Task (`CORRECTNESS`)

See [`correctness/README.md`](correctness/README.md) for the full pipeline documentation,
including split strategy, threshold calibration, the `GuardrailScorer` protocol, and metrics
glossary.

## Package Layout

```
pyine/evals/
├── common.py           # base types: EvalType, EvalResult, BaseEvalsConfig, GenerationEvalsConfig
├── configs.py          # hydra-zen/pydantic configs (e.g. LLM grader providers)
├── constants.py        # package-wide constants: aggregation stat names/functions
├── logging.py          # DiskEvalLogger: LMDB-based export of evaluation artifacts
├── utils.py            # TokenUsageInfo, category extraction helpers, metric helpers
│
├── code_exec/                # CODE_EXEC task
│   ├── _impl.py              # evaluate_runnable_model, evaluate_hf_model, finalize_evaluation_results
│   ├── configs.py            # CodeExecEvalsConfig (extends GenerationEvalsConfig)
│   ├── evaluator.py          # OutcomeEvaluator: hard match, soft match, optional LLM grading
│   ├── analysis.py           # offline analysis helpers (W&B fetch, DataFrames, plotting)
│   ├── reeval.py             # reevaluate_from_lmdb
│   └── utils.py              # SampleEval, CodeExecEvalArtifact, CodeExecEvalResult, metric computation
│
├── correctness/              # CORRECTNESS task
│   ├── _impl.py              # evaluate_guardrail_replicas / evaluate_guardrail_types: orchestration
│   ├── configs.py            # CorrectnessEvalsConfig, RecordCategoryConfig, hydra-zen registration
│   ├── datamodule.py         # CorrectnessDataModule: lifecycle, split access, HF DatasetDict adapter
│   ├── datamodule_configs.py # CorrectnessDataModuleConfig: LMDB paths, label type, split config
│   ├── scorers.py            # ProbeScorer, LLMClassifierScorer: GuardrailScorer adapters
│   ├── splits.py             # GuardrailSplits, build_guardrail_splits() (standalone, importable)
│   ├── data_loading.py       # LMDB reading -> list[EvalRecord]
│   ├── calibration.py        # FPR-constrained threshold selection
│   ├── metrics.py            # threshold-free, thresholded, sample-level, bootstrap CI computation
│   └── types.py              # LabelType, GuardrailSplitConfig, EvalRecord, GuardrailScorer, etc.
│
└── (external scorer: pyine.guardrails.prompted_llm)
    # PromptedLLMGuardrailScorer: inference-only GuardrailScorer using a prompted LLM judge.
    # Has a standalone eval app: pyine.apps.guardrail_eval.prompted_llm_eval
```

## Metrics Reference

Each subpackage documents its full metrics glossary:

- **Code execution metrics**: see [`code_exec/README.md` -- Metrics Reference](code_exec/README.md#metrics-reference)
  (accuracy, Pass@K, CIs, token usage, code complexity, category-wise breakdowns)
- **Guardrail correctness metrics**: see [`correctness/README.md` -- Metrics Glossary](correctness/README.md#metrics-glossary)
  (AUROC, thresholded metrics, sample-level metrics, bootstrap CIs, verification costs, diagnostics)

### Shared Conventions

- **CI keys**: metrics marked with CI emit two additional keys suffixed with `_ci_lower` and
  `_ci_upper` (95% confidence level by default).
- **Category prefixing**: when category extraction is enabled, each metric is also emitted
  per-category with a `{category}/` prefix (e.g. `code_type/original/accuracy_hard`).
- **Aggregation stats**: numeric distributions (token usage, complexity, correctness cross-run
  values) are summarized using a shared set of aggregation statistics: mean, median, std, min,
  max (defined in `constants.py`).
- **W&B key namespace**: all metrics are logged under `benchmark/{subset_name}/{metric_name}`;
  cross-subset comparison tables go under `benchmark/metrics_table`.
- **LMDB format**: both tasks exchange records via `DiskEvalLogger` using JSON+ZSTD serialization,
  with keys formatted as `{key_prefix}{identifier}/{attempt_index}` and a `record_type="benchmark"`
  metadata entry. Records share a common field schema (`SharedGenerationRecordFields`) for
  sample ID, model output, prompt, expected output, parsed fields, tags, and categories.
- **Correctness key convention**: correctness metrics use a metric-first key style with statistic
  suffixes (`/mean`, `/std`, `/p5`, `/bootstrap_ci_lower`, `/num_valid_runs`).
