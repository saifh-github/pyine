# PyINE Notebooks Overview

This folder collects hands-on, exploratory notebooks that demonstrate key parts of the
PyINE workflow: tracing Python code, deriving deltas, inspecting datasets, and browsing
prompt-chain annotation results.

Run any notebook after installing the project into a Python 3.12+ environment (see the
[project root README](../README.md)). You can use Jupyter Lab/Notebook or an IDE kernel. Many
notebooks expect datasets under `data/` and optional environment variables configured via `.env`.

**Highlighted notebooks:**

- `code_tracing_demo.ipynb`: end-to-end look at instrumenting and tracing Python solutions, and
  inspecting the resulting events/structures.
- `code_deltas_demo.ipynb`: derive "deltas" (state changes) from execution traces and examine their
  distribution and use-cases.
- `code_edits_demo.ipynb`: demonstrate extraction/normalization of code edit steps and how they
  interplay with trace-derived artifacts.
- `trace_datasets_eda.ipynb`: exploratory data analysis of traces datasets (sizes, splits, tags,
  event counts, caps, etc.).
- `trace_datasets_viz.ipynb`: visualization-heavy walkthrough of important aggregates over the
  traces dataset.
- `taco_source_data_viz.ipynb`: quick tour/visualization of the repackaged TACO source dataset.
- `code_complexity_analysis.ipynb`: explore basic code complexity metrics for source solutions and
  their relationship with trace statistics.
- `prompt_result_viewer.ipynb`: browse/query the framework’s prompt results SQLite database to
  compare multiple prompt versions, tags, and runs.
- `prompt_result_db_demo.ipynb`: programmatic examples for `pyine.prompts.result_db` with short
  end-to-end cells (read/write/filter/delete).
