"""Core reward interfaces and implementations.

The `core` subpackage intentionally contains only stable, framework-agnostic building blocks:

- typed contexts (`RunInitContext`, `SampleContext`);
- protocols (`RewardTerm`, `OutputParser`, `RewardLogger`);
- configuration models (Pydantic-based);
- orchestration (`RewardManager`) and aggregation helpers.

Built-in terms live in `pyine.organisms.models.rewards.terms` and register themselves via the
explicit registry (`core.registry`).

Submodules (accessible via `pyine.organisms.models.rewards.core.<submodule>`):
- aggregator: Reward aggregation strategies
- configs: Pydantic configuration models
- logging: Reward logging implementations
- manager: RewardManager orchestration
- parser: Tag-based output parsing
- registry: Term factory registry
- term: Base term implementation
- types: Protocol definitions and context types
- utils: Shared utilities
"""
