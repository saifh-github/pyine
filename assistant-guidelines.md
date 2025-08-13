# PyINE Framework AI Assistant Guidelines

**IMPORTANT: ALL AI ASSISTANTS SHOULD FOLLOW THESE INSTRUCTIONS.**

Refer to the top-level [README](./README.md) for a high-level description of the project.

## Coding Style

Note that this is a strictly Python 3.12+ project.

When proposing Python code, always propose PEP-8-compliant code that is type hinted, and that
respects the following rules:

For imports, prefer the shortest, explicit imports (e.g. `import x; x.y()` instead
of `from x import y; y()`). For example:

```python
import sys
import typing  # never do `from typing import` ...!

import numpy as np  # it's fine to use the `import x as y` syntax
import torch
import torch.nn  # it's fine to import subpackages as needed

import pyine.subpackage.something
import pyine.anothersubpkg.something_else
```

Sort imports the way isort would, i.e. standard packages first, then a separate block for 3rd party
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

## Code linting and testing

To check whether linters find issues in proposed code, run `make check` (the necessary environment
should have already been setup). This runs precommit hooks and checks for e.g. PEP8 compliance.

To check whether all unit tests pass, run `make test-all` (this will include slow tests that require
datasets or heavy processing). For a faster assessment that only checks the fast tests, run
`make test`.

See the [Makefile](./Makefile) for more information on these commands.
