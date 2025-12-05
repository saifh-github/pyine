## List of High-Level TODOs, FIXMEs, and Improvement Ideas

Last review date: 2025-12-04

**High priority, bugs, and issues:**

- Eval metrics definitions (for wandb) skip the category metrics; we might be able to predetermine
  those, create a sort of registry for all metrics, and query that registry for wandb definitions?

**Medium priority:**

- Refactor shortcuts datamodule to have a base interface for anything samples-related, and make the shortcuts dm itself
  only related to the learning of a "shortcuts" bias by models (based on data preparation settings); this will help
  simplify the creation of other dm classes that are also sample-based but that have different biases later.
- Harden trace execution edge-case handling (stdin/stdout mismatches, long runtimes) by tightening safeguards in
  `pyine/utils/code/execution.py` and the dataset writer; focus on clearer timeout reporting, better tagging,
  and limited retries for recoverable failures.
- Wire up distributed presets for the trainer configs: extend `pyine/apps/trainers/hf_trainer_configs.py` to
  register DDP/Deepspeed `TrainingArguments` variants (backend, gradient accumulation, fsdp flags) and document
  how to launch them via Hydra so multi-GPU jobs don't require manual overrides.

**Low priority / style / docs:**

- Fill in the `README.md` license section with the actual license text and link to the canonical file so
  contributors know the terms instead of seeing the current placeholder.

**Improvements and mordernization ideas:**

- Update quality toolchain w/ latest version of project template (or manually patch the relevant bits)

**Nice-to-have feature ideas:**

- Build a new app to do side-by-side comparisons of prompts and prompting results using wandb weave.
