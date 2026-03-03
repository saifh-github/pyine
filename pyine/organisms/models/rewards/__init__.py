"""Reward definition framework for RL-style training loops.

This package provides:
- `RewardManager`: orchestrates term evaluation, aggregation, and optional logging.
- `SampleContext` / `RunInitContext`: typed containers for per-sample and per-run data.
- Built-in reward terms (see `list_available_terms()` for discovery).
- TRL integration via `TRLRewardAdapter` for HuggingFace TRL trainers.

For advanced usage (custom terms, parsers, loggers), see the `core` subpackage which provides
protocols like `RewardTerm`, `OutputParser`, and `RewardLogger`.

Key assumption (by design): rewards are computed once per training example (one prompt + full
model output + optional metadata). This package does not compute per-token rewards.

Subpackages (accessible via `pyine.organisms.models.rewards.<subpackage>`):
- core: Core reward interfaces, protocols, and implementations
- terms: Built-in reward terms (format, code_exec)
- trl: TRL (HuggingFace Transformers RL) integration

Convenience imports for common use cases:
    ```python
    import pyine.organisms.models.rewards as rewards
    from pyine.organisms.models.rewards.core import configs as reward_configs

    # create a manager with config
    config = reward_configs.RewardManagerConfig(
        terms=[reward_configs.RewardTermSpec(name="format", type="parseable_answer", weight=1.0)],
        parsing=reward_configs.ParsingConfig(final_tag="final"),
        logging=reward_configs.LoggingConfig(enabled=False),  # or pass logger= to RewardManager
    )
    manager = rewards.RewardManager(config)

    # build context with automatic parsing, then compute rewards
    ctx = manager.build_sample_context(
        prompt="...",
        model_output="<final>answer</final>",
        sample_data=sample_data,  # from datamodule
    )
    output = manager.compute(ctx)
    total = output.total

    # discover available terms
    for term_info in rewards.list_available_terms():
        print(f"{term_info.canonical_type}: {term_info.aliases}")

    # TRL integration
    adapter = rewards.TRLRewardAdapter(manager)
    # use with GRPOTrainer: trainer = GRPOTrainer(..., reward_funcs=[adapter])
    ```
"""

import typing

# type-only imports for pyright (actual imports are lazy via __getattr__)
if typing.TYPE_CHECKING:
    from pyine.organisms.models.rewards.core.manager import RewardManager as RewardManager
    from pyine.organisms.models.rewards.core.registry import TermInfo as TermInfo
    from pyine.organisms.models.rewards.core.registry import list_available_terms as list_available_terms
    from pyine.organisms.models.rewards.core.types import CodeExecEvalData as CodeExecEvalData
    from pyine.organisms.models.rewards.core.types import MetricValue as MetricValue
    from pyine.organisms.models.rewards.core.types import RewardOutput as RewardOutput
    from pyine.organisms.models.rewards.core.types import RunInitContext as RunInitContext
    from pyine.organisms.models.rewards.core.types import SampleContext as SampleContext
    from pyine.organisms.models.rewards.trl import MessageSelectionPolicy as MessageSelectionPolicy
    from pyine.organisms.models.rewards.trl import TRLRewardAdapter as TRLRewardAdapter
    from pyine.organisms.models.rewards.trl import TRLRewardResult as TRLRewardResult

__all__ = [
    # convenience re-exports
    "CodeExecEvalData",
    "MetricValue",
    "RewardManager",
    "RewardOutput",
    "RunInitContext",
    "SampleContext",
    "TermInfo",
    "list_available_terms",
    # TRL integration
    "MessageSelectionPolicy",
    "TRLRewardAdapter",
    "TRLRewardResult",
]


def __getattr__(name: str) -> typing.Any:
    """Lazily imports commonly used types and functions to avoid circular imports."""
    if name == "RewardManager":
        from pyine.organisms.models.rewards.core.manager import RewardManager

        return RewardManager
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
    if name == "RewardOutput":
        from pyine.organisms.models.rewards.core.types import RewardOutput

        return RewardOutput
    if name == "RunInitContext":
        from pyine.organisms.models.rewards.core.types import RunInitContext

        return RunInitContext
    if name == "SampleContext":
        from pyine.organisms.models.rewards.core.types import SampleContext

        return SampleContext
    if name == "MessageSelectionPolicy":
        from pyine.organisms.models.rewards.trl import MessageSelectionPolicy

        return MessageSelectionPolicy
    if name == "TRLRewardAdapter":
        from pyine.organisms.models.rewards.trl import TRLRewardAdapter

        return TRLRewardAdapter
    if name == "TRLRewardResult":
        from pyine.organisms.models.rewards.trl import TRLRewardResult

        return TRLRewardResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
