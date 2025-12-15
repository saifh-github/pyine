"""Reward definition framework for RL-style training loops.

This package provides:
- `RewardManager`: orchestrates term evaluation, aggregation, and optional logging.
- `SampleContext` / `RunInitContext`: typed containers for per-sample and per-run data.
- Built-in reward terms (see `list_available_terms()` for discovery).

For advanced usage (custom terms, parsers, loggers), see the `core` subpackage which provides
protocols like `RewardTerm`, `OutputParser`, and `RewardLogger`.

Key assumption (by design): rewards are computed once per training example (one prompt + full
model output + optional metadata). This package does not compute per-token rewards.

Subpackages (accessible via `pyine.organisms.models.rewards.<subpackage>`):
- core: Core reward interfaces, protocols, and implementations
- terms: Built-in reward terms (format, code_exec)

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
    from pyine.organisms.models.rewards.core.types import CodeExecEvalData as CodeExecEvalData
    from pyine.organisms.models.rewards.core.types import MetricValue as MetricValue
    from pyine.organisms.models.rewards.core.types import ParsedOutput as ParsedOutput
    from pyine.organisms.models.rewards.core.types import RewardOutput as RewardOutput
    from pyine.organisms.models.rewards.core.types import RunInitContext as RunInitContext
    from pyine.organisms.models.rewards.core.types import SampleContext as SampleContext

__all__ = [
    # convenience re-exports
    "CodeExecEvalData",
    "MetricValue",
    "ParsedOutput",
    "RewardManager",
    "RewardOutput",
    "RunInitContext",
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
    if name == "CodeExecEvalData":
        from pyine.organisms.models.rewards.core.types import CodeExecEvalData

        return CodeExecEvalData
    if name == "MetricValue":
        from pyine.organisms.models.rewards.core.types import MetricValue

        return MetricValue
    if name == "ParsedOutput":
        from pyine.organisms.models.rewards.core.types import ParsedOutput

        return ParsedOutput
    if name == "RewardOutput":
        from pyine.organisms.models.rewards.core.types import RewardOutput

        return RewardOutput
    if name == "RunInitContext":
        from pyine.organisms.models.rewards.core.types import RunInitContext

        return RunInitContext
    if name == "SampleContext":
        from pyine.organisms.models.rewards.core.types import SampleContext

        return SampleContext
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
