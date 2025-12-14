"""Built-in reward terms.

This module is imported for side effects: built-in term modules register their factories into the
global registry (`pyine.organisms.models.rewards.core.registry`).

Core policy: `RewardManager` does not import individual term modules directly; it only ensures that
this "builtins registration" happens once.

Subpackages (accessible via `pyine.organisms.models.rewards.terms.<subpackage>`):
- format: Formatting-related reward terms (parseable_answer, text_length)
- code_exec: Code execution reward terms (placeholder)
"""

__all__ = [
    "ensure_builtin_terms_registered",
]

_registered = False
"""Whether built-in terms have been registered into the global registry."""


def ensure_builtin_terms_registered() -> None:
    """Ensure built-in terms have registered themselves.

    This should be called exactly once per process (the function is idempotent).
    """
    global _registered  # noqa: PLW0603 (module-level registry flag)
    if _registered:
        return
    import pyine.organisms.models.rewards.terms.format.parseable_answer
    import pyine.organisms.models.rewards.terms.format.text_length

    pyine.organisms.models.rewards.terms.format.parseable_answer  # noqa: B018 (import for side effects)
    pyine.organisms.models.rewards.terms.format.text_length  # noqa: B018 (import for side effects)
    _registered = True
