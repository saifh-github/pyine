## List of High-Level TODOs, FIXMEs, and Improvement Ideas

Last review date: 2025-09-08

**High priority:**

- EDA on prompt configs (code summary, code exec, bugs, hints, ...)
- trace annot generator support for code exec checks

**Medium priority:**

- n/a

**Low priority / style / docs:**

- Test coverage ideas:
  - Tests for LMDB auto-resize, get/get_indices error handling, metadata read on empty DB, and output_compare edge cases (NaN, inf, type-insensitive sequences).

**Improvements and mordernization ideas:**

- pyright

**Nice-to-have feature ideas:**

- Dataset introspection tools:
  - Small utilities to list problems, solutions, and trace counts per problem; validate augmentation parent-child links.
