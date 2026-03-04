# PyINE Notebooks Overview

This folder collects hands-on, exploratory notebooks that demonstrate key parts of the
PyINE workflow: tracing Python code, deriving deltas, inspecting datasets, and browsing
prompt-chain annotation results.

Run any notebook after installing the project into a Python 3.12+ environment (see the
[project root README](../README.md)). You can use Jupyter Lab/Notebook or an IDE kernel. Many
notebooks expect datasets under `data/` and optional environment variables configured via `.env`.

**Highlighted notebooks:**

- `code_cluster_analysis.ipynb`: analyze clustering of code solutions based on variable/function
  name similarity, visualize keyword distributions, and explore solution families.
- `code_deltas_demo.ipynb`: derive "deltas" (stepwise variable state changes) from execution traces
  and examine their structure and representation.
- `code_edits_demo.ipynb`: demonstrate using LLM prompts to generate code edits (e.g., injecting
  bugs, adding stubs, refactoring) and visualizing diffs between original and modified code.
- `code_execution_complexity_viz.ipynb`: visualize code execution complexity experiment results,
  comparing model accuracy across different complexity metrics.
- `code_execution_demo.ipynb`: end-to-end demonstration of instrumenting and tracing Python code,
  and asking various LLMs (including reasoning models) to predict execution outcomes.
- `data_loading_demo.ipynb`: demonstrate how to configure datamodules, load samples from trace
  datasets, and prepare data for reinforcement learning or supervised fine-tuning.
- `benchmark_code_exec_analysis.ipynb`: fetch and visualize code execution eval metrics from W&B or
  local pickle, including accuracy breakdowns by code type, predict type, complexity, and keyword
  presence.
- `benchmark_correctness_analysis.ipynb`: fetch and visualize guardrail correctness eval metrics,
  including AUROC, ROC/PR curves, operating point analysis, category breakdowns, and
  difficulty-conditioned performance.
- `prompt_result_db_demo.ipynb`: programmatic examples for `pyine.prompts.result_db` utils showing how
  to store, fetch, and manage LLM prompt results with automatic deduplication.
- `prompt_result_viewer.ipynb`: interactive browser for the framework's prompt results database,
  with filtering by prompt name/version/tags and visualization of annotation coverage.
- `sample_builder_outputs_eda.ipynb`: explore sample builder configurations and outputs, analyzing
  token distributions, predict type distributions, code type selections, and tag frequencies.
- `sample_prompt_length_analysis.ipynb`: analyze prompt token length distributions for different
  HuggingFace models, identify samples exceeding context limits, and visualize overflow patterns.
- `taco_source_data_viz.ipynb`: quick tour and visualization of the original TACO source dataset,
  including solution counts, lengths, difficulty distributions, and tag analysis.
- `trace_datasets_eda.ipynb`: exploratory data analysis of trace datasets covering problem/solution
  coverage, test retention, augmentation statistics, and tag distributions across dataset versions.
- `trace_datasets_viz.ipynb`: visualization-focused walkthrough of trace and delta datasets,
  including tag frequencies, step lengths, and delta type distributions.
- `wandb_sweep_sampling_demo.ipynb`: reproduce Weights & Biases sweep sampling locally to validate
  hyperparameter distributions before launching experiments (supports all W&B sampler types).

## Notebook Output Handling

This project uses [nbstripout](https://github.com/kynan/nbstripout) as a git filter to manage
notebook outputs. This means:

- **Local notebooks**: Cell outputs are preserved as you work
- **Committed notebooks**: Outputs are automatically stripped when you commit

This keeps the repository clean (smaller diffs, no binary blobs) while letting you keep outputs
locally for reference.

**Setup**: The filter is installed automatically when you run `make install`. If you need to
set it up manually (e.g., after cloning), run:

```bash
make setup-nbstripout-tool
make setup-nbstripout
```

If `git add` feels slow when staging notebooks, it usually means your git filter is running
nbstripout from the full project virtualenv (which can have a noticeably slower Python startup
time due to large deps). The `setup-nbstripout-tool` target creates a small dedicated tool venv
under `.tools/` and points the filter at it.
