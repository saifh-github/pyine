## List of High-Level TODOs, FIXMEs, and Improvement Ideas

Last review date: 2025-09-24

**High priority:**

- Make the HuggingFace trainer DDP-aware: in `pyine/apps/trainers/hf_trainer.py` detect `LOCAL_RANK`/`torch.distributed`, drop the current `device_map="auto"` path in favour of per-rank placement, guard `Trainer` checkpoints/eval/W&B logging so they only run on rank 0, and wrap `prepare_datamodule` so `prepare_data()` runs once with a barrier before other ranks continue.
- Build consolidated EDA over prompt result tables so we can spot failure patterns quickly; start from `notebooks/prompt_result_viewer.ipynb`, add aggregation helpers in `pyine/prompts/result_db.py`, and publish summary dashboards that compare runs side by side.

**Medium priority:**

- Let `SampleBuilder` deterministically pull one sample per cousin cluster; extend the clustering metadata in `pyine/organisms/datamodules/utils/samples.py` and add a selector that enforces one representative per cluster while keeping existing sampling modes intact.
- Harden trace execution edge-case handling (stdin/stdout mismatches, long runtimes) by tightening safeguards in `pyine/utils/code/execution.py` and the dataset writer; focus on clearer timeout reporting, better tagging, and limited retries for recoverable failures.
- Wire up distributed presets for the trainer configs: extend `pyine/apps/trainers/hf_trainer_configs.py` to register DDP/Deepspeed `TrainingArguments` variants (backend, gradient accumulation, fsdp flags) and document how to launch them via Hydra so multi-GPU jobs don't require manual overrides.

**Low priority / style / docs:**

- Fill in the `README.md` license section with the actual license text and link to the canonical file so contributors know the terms instead of seeing the current placeholder.

**Improvements and mordernization ideas:**

- Add `ruff` and `pyright` to the dev toolchain: configure them in `pyproject.toml`, hook them into `make check`/pre-commit, and resolve surfacing lint/type issues in batches.

**Nice-to-have feature ideas:**

- Allow the HF trainer CLI to load a saved LoRA/adapter checkpoint and run evaluation-only jobs by implementing the `do_train=False` branch in `pyine/apps/trainers/hf_trainer.py` instead of raising `NotImplementedError`.
