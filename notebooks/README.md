# PyINE Notebooks Overview

This folder collects hands-on, exploratory notebooks that demonstrate key parts of the
PyINE workflow: tracing Python code, deriving deltas, inspecting datasets, and browsing
prompt-chain annotation results.

Run any notebook after installing the project into a Python 3.12+ environment (see the
[project root README](../README.md)). You can use Jupyter Lab/Notebook or an IDE kernel. Many
notebooks expect datasets under `data/` and optional environment variables configured via `.env`.

**Highlighted notebooks:**

- `code_execution_demo.ipynb`: end-to-end look at instrumenting and tracing Python code, and
  asking models to predict the outcome of executing the same code.
- `code_deltas_demo.ipynb`: derive "deltas" (state changes) from execution traces and examine their
  distribution and use-cases.
- `code_edits_demo.ipynb`: demonstrate extraction/normalization of code edit steps and how they
  interplay with trace-derived artifacts.
- `code_cluster_analysis.ipynb`: analyze clustering of code solutions based on similarity metrics
  and explore relationships between solution families.
- `trace_datasets_eda.ipynb`: exploratory data analysis of traces datasets (sizes, splits, tags,
  event counts, caps, etc.).
- `trace_datasets_viz.ipynb`: visualization-heavy walkthrough of important aggregates over the
  traces dataset.
- `taco_source_data_viz.ipynb`: quick tour/visualization of the repackaged TACO source dataset.
- `sample_builder_outputs_eda.ipynb`: explore sample builder outputs, including token distributions,
  prompt lengths, and formatting statistics for model training.
- `sample_prompt_length_analysis.ipynb`: analyze prompt length distributions across different
  sample types and configurations to optimize training efficiency.
- `wandb_sweep_sampling_demo.ipynb`: demonstrate Weights & Biases sweep configuration and sampling
  strategies for hyperparameter tuning experiments.
- `prompt_result_viewer.ipynb`: browse/query the framework's prompt results SQLite database to
  compare multiple prompt versions, tags, and runs.
- `prompt_result_db_demo.ipynb`: programmatic examples for `pyine.prompts.result_db` with short
  end-to-end cells (read/write/filter/delete).

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
make setup-nbstripout
```
