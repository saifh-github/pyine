## List of High-Level TODOs, FIXMEs, and Improvement Ideas

Last review date: 2025-12-19

**High priority, bugs, and issues:**

- Eval metrics definitions (for wandb) skip the category metrics; we might be able to predetermine
  those, create a sort of registry for all metrics, and query that registry for wandb definitions?
- We might want code exec examples based on coding problems with clear entrypoints to always be
  prepared as 'function_return' predict_type samples (when allowed)?

**Medium priority:**

- Harden trace execution edge-case handling (stdin/stdout mismatches, long runtimes) by tightening safeguards in
  `pyine/utils/code/execution.py` and the dataset writer; focus on clearer timeout reporting, better tagging,
  and limited retries for recoverable failures.
- Update the reward manager + LLM grader pipeline so that if LLM grading fails, we return `None` as rewards.

**Low priority / style / docs:**

- Fill in the `README.md` license section with the actual license text and link to the canonical file so
  contributors know the terms instead of seeing the current placeholder.

**Improvements and mordernization ideas:**

- Update quality toolchain w/ latest version of project template (or manually patch the relevant bits)

**Nice-to-have feature ideas:**

- Build a new app to do side-by-side comparisons of prompts and prompting results using wandb weave.
