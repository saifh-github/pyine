# Contributing to PyINE

This document covers how we work and what we expect from contributions. For project setup, see the
top-level [README](./README.md). For coding style enforced project-wide (imports, quotes,
docstrings, function-arg formatting, etc.), see [`assistant-guidelines.md`](./assistant-guidelines.md);
those rules apply to human contributors as much as to AI assistants.

## Guiding principles

- Make it correct, then make it simple, then make it fast.
- Small, focused PRs are easier to review and revert.
- Tests drive quality and confidence; every bug fix or feature ships with tests.
- Determinism is a feature: seed PRNGs and avoid time/environment-sensitive behavior in tests.
- Prefer clear APIs, helpful error messages, and actionable logs.

## Prerequisites

- Python **3.12+**.
- [`uv`](https://github.com/astral-sh/uv) for environment management (uses the project's `uv.lock`).
- `pytest`, `pre-commit`, `ruff`, `pyright` are wired into the Makefile.

## Workflow

1. **Open or pick an issue.** For larger changes, sketch the approach in the issue first.

2. **Branch off `master`** with a descriptive name (e.g. `feature/prompt-manager-batching`,
   `fix/timeout-handling`).

3. **Keep PRs small and focused.** Ship tests and docs in the same PR as the change.

4. **Run quality gates locally before pushing.** The canonical gate is:

   ```bash
   make check    # = make lint + make test (ruff + pyright + pre-commit + fast pytest)
   ```

   Useful subsets:

   ```bash
   make lint                            # formatting + linting + type-checking
   make test                            # fast pytest (excludes slow/integration/openai)
   make test-all                        # everything except openai-tagged tests
   make coverage-fast / make coverage   # mirrors PR / post-merge CI coverage
   pytest tests/path/to/test_file.py -q
   pytest -k "token or expression" -q
   ```

5. **Open the PR.** Summarize *what* changed and *why*; link the issue. Call out breaking changes,
   migrations, or behavioral comparisons. Add the `integration` or `slow` label if you want CI to
   run those slices before merge.

## Testing

- Use `pytest` with fixtures for shared setup; keep tests isolated.
- Prefer small synthetic inputs and runtime-created temp dirs over committed corpora.
- Seed all PRNGs. Don't rely on wall-clock time or sleeps.
- When fixing a bug, reproduce it with a failing test first.
- If a test depends on an external resource (a dataset, an API key), gate it with a skip; see
  [`tests/env_checks.py`](./tests/env_checks.py) for the existing helpers.

**Markers** (defined in `pyproject.toml`; combine as needed). `make test` deselects everything
except untagged fast tests:

| Marker        | Use for                                                                         | Default  |
| ------------- | ------------------------------------------------------------------------------- | -------- |
| `slow`        | tests that take more than a few seconds or need heavy compute                   | excluded |
| `integration` | end-to-end flows wiring multiple modules (CLIs, Hydra apps, dataset writers)    | excluded |
| `dataset`     | tests that need a real local corpus (e.g. TACO); pair with `slow`/`integration` | excluded |
| `openai`      | tests that hit the live OpenAI API (require `OPENAI_API_KEY` + network)         | excluded |
| `distributed` | tests that require multi-process / distributed execution                        | excluded |

## Error handling and logging

- Fail fast with informative exceptions. Avoid try/except blocks whose only purpose is to silence
  errors; let them propagate to the caller unless control flow genuinely requires recovery.
- Log levels are load-bearing: `DEBUG` for deep diagnostics, `INFO` for milestones,
  `WARNING`/`ERROR` for actionable issues. Default-INFO output should stay readable.

## Data, secrets, and reproducibility

- Never commit secrets, tokens, or real credentials. Use a local `.env` (see `.env.template`).
- Keep large datasets and outputs out of the repo; reference them by path or env var.
- Notebook outputs are stripped on commit by an automatic `nbstripout` git filter installed via
  `make install`. If `git add` flags a notebook with outputs, run `make setup-nbstripout-tool && make setup-nbstripout` to (re)install the filter. See [`notebooks/README.md`](./notebooks/README.md)
  for details.

## Where to look

- [`pyine/apps/README.md`](./pyine/apps/README.md): overview of every CLI/Hydra app, with
  example invocations and pointers to per-app guides.
- [`pyine/configs/README.md`](./pyine/configs/README.md): Hydra/Hydra-Zen experiment overlays,
  search paths, and structured-config registration.
- [`pyine/prompts/README.md`](./pyine/prompts/README.md): prompt template authoring, the prompt
  manager, and the result DB.
- [`pyine/organisms/README.md`](./pyine/organisms/README.md): datamodules used by the trainers.
- [`notebooks/README.md`](./notebooks/README.md): exploratory and analysis notebooks.
- [`scripts/README.md`](./scripts/README.md): internal launchers and sweep drivers.
