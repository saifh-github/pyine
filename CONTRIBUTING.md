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
   - Fast coverage (optional, mirrors what PR CI runs): `make coverage-fast`
   - Full coverage (optional, mirrors what post-merge CI runs): `make coverage`

5. Open a PR:

   - Summarize your changes and motivation, and link to the relevant issue(s).
   - Describe any breaking changes or migrations.
   - Keep PR descriptions technical and reproducible. If relevant, add performance or behavioral comparisons before/after.
   - Add the `integration` and/or `slow` label if you want PR CI to run the corresponding heavier test slices before merge.

### Coding conventions

- Type hints are required; use `typing` and `collections.abc` for abstract types, and prefer Python 3.12+ type annotations.
- Docstrings use Google style. Keep them concise and helpful. Provide examples for important public APIs.
- Imports:
  - Prefer explicit imports (`import some_package; some_package.y()`); avoid `from some_package import y` when possible.
  - Group imports: standard library, third-party, local.
- Strings: prefer double quotes.
- Use f-strings for formatting.
- Avoid excessive try/except; allow exceptions to propagate unless handling is required for control flow or context.
- Functions with multiple args: put each argument and the return type on its own line for readability.

### Testing and reliability

- Use pytest, fixtures for shared setup, and keep tests isolated.
- Aim for stable tests by seeding randomness and eliminating time/dependency flakiness.
- Write separate tests per app/module/class/function as appropriate:
  - Unit tests: required for new features, bug fixes, and major refactors.
  - Integration tests: add when behavior spans modules or requires realistic execution flows.
  - Regression tests: when fixing a bug, reproduce it with a failing test first.
- Prefer small, synthetic inputs over large datasets; use temp dirs/files created at runtime when possible.
- Determinism:
  - Seed all PRNGs used in tests.
  - Avoid relying on system time/timeouts unless strictly necessary; if you do, keep margins generous to reduce flakes and guard with clear assertions.
- Marking tests: use shared pytest markers so the right suites can be selected quickly.
  - `slow`: mark anything that regularly takes more than a few seconds or needs heavy compute. `make test` automatically deselects these.
  - `integration`: tag end-to-end flows that wire multiple services together (CLI launchers, Hydra configs, dataset writers, etc.). These are skipped by default unless explicitly requested (e.g., `pytest -m integration` or `make test-all`).
  - `dataset`: pair with the above when a test needs local corpora like the TACO datasets or other large artifacts to be present; deselect with `-m "not dataset"` when running without those assets.
  - `openai`: apply in addition to `integration` when a test reaches the real OpenAI API. These require `OPENAI_API_KEY` and outbound network access; deselect with `-m "not openai"`.
- Skipping tests: if your tests depend on some environment resource (e.g. a dataset), allow the test to be skipped if that resource is unavailable. See the `tests.env_checks` for examples.

### Error handling and logging

- Fail fast with informative exceptions. Include context that helps a developer debug without reproducing the environment.
- Logs should be actionable and not verbose by default. Use levels consistently (DEBUG for deep diagnostics, INFO for key milestones, WARNING/ERROR for actionable issues).

### Data, secrets, and reproducibility

- Never commit secrets, tokens, or real credentials to the repo. Use environment variables via `.env` for local dev.
- Keep large datasets and outputs out of the repository. Use small synthetic fixtures in tests, if possible.
- Keep notebooks exploratory and reproducible. Never commit notebooks with uncleaned outputs.
