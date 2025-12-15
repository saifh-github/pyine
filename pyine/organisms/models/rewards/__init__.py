"""Reward definition framework for RL-style training loops.

This package provides:
- `RewardManager`: orchestrates term evaluation, aggregation, and optional logging.
- `RewardTerm`: composable per-sample reward components ("terms").
- `OutputParser`: optional output parsing to extract structured fields once per sample.

Key assumption (by design): rewards are computed once per training example (one prompt + full
model output + optional metadata). This package does not compute per-token rewards.

Subpackages (accessible via `pyine.organisms.models.rewards.<subpackage>`):
- core: Core reward interfaces and implementations
- terms: Built-in reward terms

Convenience imports for common use cases:
    ```python
    import pyine.organisms.models.rewards as rewards

    # create a simple manager (only works with terms that have all-default params)
    manager = rewards.make_simple_manager(
        [
            ("format", "parseable_answer", 1.0),
        ]
    )

    # compute rewards (sample_data required from datamodule)
    ctx = rewards.SampleContext(
        prompt="...",
        model_output="<final>answer</final>",
        sample_data=sample_data,
    )
    total = manager.compute(ctx)

    # discover available terms
    for term_info in rewards.list_available_terms():
        print(f"{term_info.canonical_type}: {term_info.aliases}")
    ```
"""

import typing

# type-only imports for pyright (actual imports are lazy via __getattr__)
if typing.TYPE_CHECKING:
    from pyine.organisms.models.rewards.core.manager import RewardManager as RewardManager
    from pyine.organisms.models.rewards.core.manager import make_simple_manager as make_simple_manager
    from pyine.organisms.models.rewards.core.registry import TermInfo as TermInfo
    from pyine.organisms.models.rewards.core.registry import list_available_terms as list_available_terms
    from pyine.organisms.models.rewards.core.types import SampleContext as SampleContext

__all__ = [
    # convenience re-exports
    "RewardManager",
    "SampleContext",
    "TermInfo",
    "list_available_terms",
    "make_simple_manager",
]


def __getattr__(name: str) -> typing.Any:
    """Lazily imports commonly used types and functions to avoid circular imports."""
    if name == "RewardManager":
        from pyine.organisms.models.rewards.core.manager import RewardManager

        return RewardManager
    if name == "make_simple_manager":
        from pyine.organisms.models.rewards.core.manager import make_simple_manager

        return make_simple_manager
    if name == "TermInfo":
        from pyine.organisms.models.rewards.core.registry import TermInfo

        return TermInfo
    if name == "list_available_terms":
        from pyine.organisms.models.rewards.core.registry import list_available_terms

        return list_available_terms
    if name == "SampleContext":
        from pyine.organisms.models.rewards.core.types import SampleContext

        return SampleContext
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
