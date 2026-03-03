"""Built-in reward terms.

This module is imported for side effects: built-in term modules register their factories into the
global registry (`pyine.organisms.models.rewards.core.registry`).

Core policy: `RewardManager` does not import individual term modules directly; it only ensures that
this "builtins registration" happens once.

Subpackages (accessible via `pyine.organisms.models.rewards.terms.<subpackage>`):
- format: Formatting-related reward terms (parseable_answer, text_length)
- code_exec: Code execution evaluation terms (hard_match, soft_match, llm_grader)
"""

import importlib

__all__ = [
    "ensure_builtin_terms_registered",
]

_BUILTIN_TERM_MODULES: tuple[str, ...] = (
    "pyine.organisms.models.rewards.terms.format.parseable_answer",
    "pyine.organisms.models.rewards.terms.format.text_length",
    "pyine.organisms.models.rewards.terms.format.traced_reasoning",
    "pyine.organisms.models.rewards.terms.code_exec.hard_match",
    "pyine.organisms.models.rewards.terms.code_exec.soft_match",
    "pyine.organisms.models.rewards.terms.code_exec.llm_grader",
)
"""Module paths for built-in terms (imported for registration side effects)."""

_registered = False
"""Whether built-in terms have been registered into the global registry."""


def ensure_builtin_terms_registered() -> None:
    """Ensure built-in terms have registered themselves.

    This should be called exactly once per process (the function is idempotent).
    """
    global _registered  # noqa: PLW0603 (module-level registry flag)
    if _registered:
        return
    for module_path in _BUILTIN_TERM_MODULES:
        importlib.import_module(module_path)
    _registered = True
