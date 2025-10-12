## List of High-Level TODOs, FIXMEs, and Improvement Ideas

Last review date: 2025-10-11

**High priority, bugs, and issues:**

- none

**Medium priority:**

- Make the HuggingFace trainer DDP-aware: in `pyine/apps/trainers/hf_trainer.py` detect `LOCAL_RANK`/`torch.distributed`, drop the current `device_map="auto"` path in favour of per-rank placement, guard `Trainer` checkpoints/eval/W&B logging so they only run on rank 0, and wrap `prepare_datamodule` so `prepare_data()` runs once with a barrier before other ranks continue.
- Refactor shortcuts datamodule to have a base interface for anything samples-related, and make the shortcuts dm itself
  only related to the learning of a "shortcuts" bias by models (based on data preparation settings); this will help
  simplify the creation of other dm classes that are also sample-based but that have different biases later.
- Let `SampleBuilder` deterministically pull one sample per cousin cluster; extend the clustering metadata in `pyine/organisms/datamodules/utils/samples.py` and add a selector that enforces one representative per cluster while keeping existing sampling modes intact.
- Harden trace execution edge-case handling (stdin/stdout mismatches, long runtimes) by tightening safeguards in `pyine/utils/code/execution.py` and the dataset writer; focus on clearer timeout reporting, better tagging, and limited retries for recoverable failures.
- Wire up distributed presets for the trainer configs: extend `pyine/apps/trainers/hf_trainer_configs.py` to register DDP/Deepspeed `TrainingArguments` variants (backend, gradient accumulation, fsdp flags) and document how to launch them via Hydra so multi-GPU jobs don't require manual overrides.

**Low priority / style / docs:**

- Fill in the `README.md` license section with the actual license text and link to the canonical file so contributors know the terms instead of seeing the current placeholder.

**Improvements and mordernization ideas:**

- Update quality toolchain w/ latest version of project template

**Nice-to-have feature ideas:**

- Allow the HF trainer CLI to load a saved LoRA/adapter checkpoint and run evaluation-only jobs by implementing the `do_train=False` branch in `pyine/apps/trainers/hf_trainer.py` instead of raising `NotImplementedError`.
- Build a new app to do side-by-side comparisons of prompts and prompting results using wandb weave.
