# Guardrail Metrics User Guide

## Overview

The guardrail metrics app benchmarks trained classifier checkpoints (activation probes and/or LLM binary classifiers) at inference time. It measures FLOPs, memory, latency, and throughput for each classifier, producing a structured report for cost-benefit analysis when deploying these classifiers as guardrails alongside a frozen LLM.

## Quick Start

### Benchmark Probes

```bash
python -m pyine.apps.guardrail_metrics.guardrail_metrics \
    +experiment=guardrail_metrics/v0_probes \
    config.probe_checkpoint_dir=/path/to/probes \
    config.probe_llm_model=Qwen/Qwen2.5-3B
```

### Benchmark LLM Classifier

```bash
python -m pyine.apps.guardrail_metrics.guardrail_metrics \
    +experiment=guardrail_metrics/v0_classifier \
    config.classifier_checkpoint_dir=/path/to/classifier
```

### Benchmark Both and Compare

```bash
python -m pyine.apps.guardrail_metrics.guardrail_metrics \
    +experiment=guardrail_metrics/v0_probes \
    config.probe_checkpoint_dir=/path/to/probes \
    config.probe_llm_model=Qwen/Qwen2.5-3B \
    config.classifier_checkpoint_dir=/path/to/classifier
```

## Configuration

### Key Settings

| Setting                     | Default                   | Description                                                   |
| --------------------------- | ------------------------- | ------------------------------------------------------------- |
| `probe_checkpoint_dir`      | `null`                    | Path to probe checkpoints (from `probe_trainer`)              |
| `probe_llm_model`           | `null`                    | HF model ID for frozen LLM (required with probes)             |
| `classifier_checkpoint_dir` | `null`                    | Path to classifier checkpoint (from `llm_classifier_trainer`) |
| `batch_sizes`               | `[1, 4, 8, 16, 32]`       | Batch sizes to sweep                                          |
| `synthetic_seq_lengths`     | `[512, 1024, 2048, 4096]` | Sequence lengths to sweep                                     |
| `num_warmup_iterations`     | `10`                      | Warmup passes before measurement                              |
| `num_benchmark_iterations`  | `100`                     | Timed passes for statistics                                   |
| `device`                    | `auto`                    | `auto`, `cuda`, `cpu`, or `mps`                               |
| `output_format`             | `both`                    | `json`, `csv`, or `both`                                      |
| `use_torch_compile`         | `false`                   | Apply `torch.compile()` before benchmarking                   |

### Measurement Toggles

Individual metric categories can be disabled:

```bash
config.measure_flops=false     # skip FLOPs measurement
config.measure_memory=false    # skip memory measurement
config.measure_latency=false   # skip latency measurement
config.measure_throughput=false # skip throughput calculation
```

### Synthetic vs Real Data

By default, the app uses synthetic random tensors (`use_synthetic_data=true`). This is recommended for pure compute benchmarks as it avoids LMDB dependencies and isolates measurement from I/O.

For realistic sequence-length distributions, set `use_synthetic_data=false` and provide a `datamodule_config` with an LMDB path.

## What Gets Measured

### For Probes

The app performs a three-way latency decomposition:

1. **LLM forward without hooks** (baseline) -- what the frozen LLM costs by itself
2. **LLM forward with hooks** -- includes `ActivationExtractor` overhead
3. **Probe-only forward** -- just the probe on pre-extracted activations

Since probes piggyback on a frozen LLM forward pass that is already happening for generation, the relevant overhead metric is `probe_flops + hook_extraction_overhead`.

### For Classifiers

Full encoder forward pass (e.g., ModernBERT + classification head), measured as a standalone model.

### Comparison

When both probes and classifiers are benchmarked, the app produces per-`(batch_size, seq_length)` comparison metrics including FLOPs ratios and latency differences.

## Output

Results are saved to the Hydra output directory in JSON and/or CSV format.

### JSON Structure

```
guardrail_metrics.json
├── metadata (timestamp, device, torch version, ...)
├── guardrail_results
│   ├── probe
│   │   ├── static_info
│   │   │   ├── frozen_llm (model name, param count, memory)
│   │   │   └── probes (per-probe architecture, layer, params)
│   │   └── benchmarks[] (per batch_size x seq_length results)
│   └── classifier
│       ├── static_info (model_name, param_count, param_memory_bytes)
│       └── benchmarks[]
└── comparison[] (per-config probe vs classifier)
```

### CSV Format

Flat table with one row per `(classifier_type, probe_name, batch_size, seq_length)`:

```
classifier_type,model_name,probe_name,architecture,layer,batch_size,seq_length,status,...
```

## WandB Logging

When `config.use_wandb_logging=true`, benchmark results are automatically pushed to WandB in addition to the local JSON/CSV files:

- **`wandb.summary`**: Static info — model names, parameter counts, device, torch version, per-probe architecture/layer details
- **`wandb.log`**: Per-`(batch_size, seq_length)` sweep results logged as separate steps — FLOPs, latency stats, memory, and throughput for both probe and classifier benchmarks
- **`wandb.Table`**: When both probes and classifiers are benchmarked, a comparison table is logged with FLOPs ratios and latency differences

Enable with:

```bash
config.use_wandb_logging=true
```

## OOM Handling

Large `(batch_size, seq_length)` combinations may cause out-of-memory errors. The app handles this gracefully:

- Configurations are sorted ascending to run smaller configs first
- OOM errors are caught, logged, and recorded as `"status": "OOM"`
- `torch.cuda.empty_cache()` is called before continuing

## Requirements

- PyTorch >= 2.1 (for `torch.utils.flop_counter.FlopCounterMode`)
- Probe checkpoints from `probe_trainer` (subdirectories with `probe_state_dict.pt` + `probe_config.json`)
- Classifier checkpoints from `llm_classifier_trainer` (HuggingFace model directory)
- For probe benchmarking: `probe_auto_model_config.use_cache` **must** be `False`
