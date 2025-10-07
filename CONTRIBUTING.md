# Contributing to the PyINE Framework

Thank you for your interest in contributing! This document explains how we work, what we expect
from contributions, and how to get your changes merged smoothly.

### Guiding principles

- Make it correct, then make it simple, then make it fast.
- Small, focused changes are easier to review and revert.
- Tests drive quality and confidence. Every bug fix or feature should include tests.
- Determinism is a feature. Seed your randomness and avoid time/date or environment-sensitive behavior in tests.
- Developer ergonomics matter. Prefer clear APIs, helpful error messages, and actionable logs.

### Project prerequisites

- Python: 3.12+ only.
- Environment: use uv with the project's `uv.lock` file.
- Tooling: pytest, pre-commit, ruff, and pyright are used via Makefile shortcuts (described below).

For project setup instructions, refer to the top-level [README](./README.md).

### Developer workflow

1. Pick or open an issue related to what you want to work on:

   - Look for an existing issue. If none exists, open one describing the problem, rationale, and proposed approach.
   - For larger changes, propose an RFC/plan in the issue first.

2. Create a feature branch:

   - Use a descriptive name, e.g., `feature/prompt-manager-batching` or `fix/timeout-handling`.

3. Keep PRs small and focused:

   - Submit incremental PRs where feasible (functional slices).
   - Include tests and docs in the same PR as your proposed changes.

4. Ensure quality gates pass locally:

   - Formatting and linting: `make lint`
   - Fast tests: `make test`
   - Full test suite (slow, includes heavy datasets or longer runs): `make test-all`
   - Run a single test file or pattern:
     - `pytest tests/path/to/test_file.py -q`
     - `pytest -k "token or expression" -q`
   - Coverage (optional, to make sure it stays adequate): `make cov-all`

5. Open a PR:

   - Summarize your changes and motivation, and link to the relevant issue(s).
   - Describe any breaking changes or migrations.
   - Keep PR descriptions technical and reproducible. If relevant, add performance or behavioral comparisons before/after.

### Coding conventions

- Type hints are required; use typing and collections.abc for abstract types, and prefer Python 3.12+ type annotations.
- Docstrings use Google style. Keep them concise and helpful. Provide examples for important public APIs.
- Imports:
  - Prefer explicit imports (`import some_package; some_package.y()`); avoid `from some_package import y` when possible.
  - Group imports: standard library, third-party, local.
- Strings: prefer double quotes.
- Use f-strings for formatting.
- Avoid excessive try/except; allow exceptions to propagate unless handling is required for control flow or context.
- Functions with multiple args: put each argument and the return type on its own line for readability.
- Tests:
  - Use pytest, fixtures for shared setup, and keep tests isolated.
  - Aim for stable tests by seeding randomness and eliminating time/dependency flakiness.
  - Write separate tests per module/class/function as appropriate.
  - Prefer small, synthetic inputs over large datasets; use temp dirs/files created at runtime when possible.

### Testing and reliability

- Unit tests: required for new features, bug fixes, and major refactors. Cover main positive and negative paths.
- Integration tests: add when behavior spans modules or requires realistic execution flows.
- Regression tests: when fixing a bug, reproduce it with a failing test first.
- Determinism:
  - Seed all PRNGs used in tests.
  - Avoid relying on system time/timeouts unless strictly necessary; if you do, keep margins generous to reduce flakes and guard with clear assertions.
- Marking tests: properly mark tests as 'slow' when they take more than a few seconds to run.
- Skipping tests: if your tests depend on some environment resource (e.g. a dataset), allow the test to be skipped if that resource is unavailable. See the `tests.env_checks` for examples.

### Error handling and logging

- Fail fast with informative exceptions. Include context that helps a developer debug without reproducing the environment.
- Logs should be actionable and not verbose by default. Use levels consistently (DEBUG for deep diagnostics, INFO for key milestones, WARNING/ERROR for actionable issues).

### Data, secrets, and reproducibility

- Never commit secrets, tokens, or real credentials to the repo. Use environment variables via `.env` for local dev.
- Keep large datasets and outputs out of the repository. Use small synthetic fixtures in tests, if possible.
- Keep notebooks exploratory and reproducible. Never commit notebooks with uncleaned outputs.
