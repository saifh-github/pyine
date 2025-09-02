## List of High-Level TODOs, FIXMEs, and Improvement Ideas

Last review date: 2025-09-01

**High priority:**

- openai trainer (with RL strategy)

**Medium priority:**

- EDA on prompt configs (code summary, code exec, bugs, hints, ...)
- implement prompt result db lookup for code summaries, code augmentations, etc.

**Low priority / style / docs:**

- Test coverage ideas:
  - Tests for LMDB auto-resize, get/get_indices error handling, metadata read on empty DB, and output_compare edge cases (NaN, inf, type-insensitive sequences).

**Improvements and mordernization ideas:**

- n/a

**Nice-to-have feature ideas:**

- CLI utilities:
  - Provide a CLI to write/read datasets with clear arguments (paths, limits, serialization, LLM options) to reduce need to edit __main__ blocks.
- Dataset introspection tools:
  - Small utilities to list problems, solutions, and trace counts per problem; validate augmentation parent-child links.
