# PyINE Framework AI Assistant Guidelines

**IMPORTANT: ALL AI ASSISTANTS SHOULD FOLLOW THESE INSTRUCTIONS.**

Refer to the top-level [README](./README.md) for a high-level description of the project.

## Project Structure & Module Organization

Core package code lives under `pyine/`, grouped by domain (`apps`, `organisms`, `prompts`, `utils`).
Experiment configuration utilities sit in `pyine/configs`, while generated datasets and fixtures
live in `pyine/data/`. Tests mirror the package layout in `tests/` to keep unit coverage close to
the implementation. Example notebooks, exploratory scripts, and visualizations reside in
`notebooks/`. Large artifacts (datasets, checkpoints) should be stored outside the repo and
referenced via paths or environment variables.

## Build, Test, and Development Commands

Use `uv sync --extra dev` after creating a virtual environment with `uv venv` (or simply run
`make install` to set up everything automatically). Run `make info` to discover helper tasks.
`make check` executes the full pre-commit suite, while `make test` runs the default pytest
selection (`-m "not (slow or integration)"`). Use `make test-all` for exhaustive runs and
`make coverage-fast` for fast-only coverage or `make coverage` for full coverage; both
produce HTML and XML artifacts under `logs/coverage/`.

## Coding Style & Naming Conventions

Code targets Python 3.12. Format with `ruff` (120-char lines); formatting and import sorting run
automatically through pre-commit. Linting relies on `ruff` and `pyright`, so keep modules idiomatic:
4-space indentation, descriptive snake_case for functions/modules, PascalCase for classes. Use
type annotations, and use lightweight dataclasses where practical (or ideally pydantic models when
validation is useful). Keep secrets out of version control.

When proposing Python code, respects the following rules:

For imports, prefer the shortest, explicit imports (e.g. `import x; x.y()` instead
of `from x import y; y()`). For example:

```python
import sys
import typing  # never do `from typing import` ...!

import numpy as np  # it's fine to use the `import x as y` syntax
import torch
import torch.nn  # it's fine to import subpackages as needed

import pyine.data.utils.lmdb_io
import pyine.data.traces.dataset_writer as trace_dataset_writer
import pyine.utils.reprod
```

Sort imports the way ruff would, i.e. standard packages first, then a separate block for 3rd party
packages, and then another block for local (project) imports.

Use the google doctstring format inside generated docstrings.

For regular comments, prefer using a lower case letter at the start. Also, for comments, if they
are short and refer to a single line, put them inline at the end of that specific line instead
of above it. If a comment message is an obvious description of what the code does, don't write it.

For string quote style, prefer the double quote by default.

When printing/logging values inside strings, prefer f-string formating.

Avoid variable names that are a single letter, or just generally too short; for example, for a
loop index, use `idx`, or ideally `<something>_idx`.

Avoid using single empty lines across the code. Only use empty lines where needed for PEP8
formatting.

Avoid using too many "defensive" try-except blocks (especially ones that catch any Exception);
let the code fail up to the caller if exceptions are raised, unless the logic requires one to be
raised.

When defining functions, put each argument on a separate new line, unless there is a single
argument for that function. Also put the type of the function's return value on a new line, for
example:

```python
def some_function(
    first_arg: int,
    second_arg: str,
) -> list[int]:
    ...
```

When writing unit tests, use pytest. If there are multiple unique classes/functions to test
totally independently inside a module, define the tests in a class for each one of them. If
possible, reuse common definitions and objects across the tests of a class using fixtures. Inside
test modules, no need for extensive docstrings or comments: only document or comment the stuff
that is really not so obvious to understand.

When writing pydantic-related code, use the pydantic v2 conventions.

Avoid writing overly-defensive code; prefer that if something is missing or breaks, an exception
is raised, rather than trying to bypass or silence that problem. Avoid using try-except blocks that
only serve to silence or avoid issues, especially those that catch `Exception`.

## Code Linting and Testing

To check whether linters find issues in proposed code, run `make check` (the necessary environment
should have already been setup). This runs precommit hooks and checks for e.g. PEP8 compliance.

Tests use `pytest`, organized per feature subpackage. Name test modules `test_<subject>.py` and
individual tests `test_<expectation>`. Mark long-running or data-heavy scenarios with
`@pytest.mark.slow`; CI and `make test` exclude them by default, so ensure at least one fast path
exercises new code. When adding features, extend coverage expectations (branch-aware) and
inspect reports in `logs/coverage/html/index.html`. When creating new tests, make sure that all
newly logged files and results are sent to an appropriate temp dir to avoid polluting real
directories.

To exhaustively check whether all unit tests pass, run `make test-all` (this will include slow
tests that require datasets or heavy processing).

See the [Makefile](./Makefile) for more information on these commands.

## Commit & Pull Request Guidelines

Commit messages follow an imperative, present-tense style (e.g., `Add prompt result viewer notebook`).
Keep related changes together and re-run `make check` before pushing. Pull requests
should summarize intent, call out datasets or configs touched, and link any tracking issues.
Include reproduction steps or command logs for behavioral changes, plus screenshots when UI
notebooks or visual outputs change.

## Environment & Configuration Tips

Populate a local `.env` based on `.env.template` to supply API keys, dataset roots, and experiment
toggles. Avoid committing real credentials; use descriptive placeholder values instead. Prefer
referencing paths via environment variables so notebooks, apps, and CLI tools stay portable
across machines.
