"""Reward term registry.

The registry decouples `RewardManager` from individual term modules:
- term modules register factories under stable string keys;
- the manager constructs terms via `RewardTermSpec.type` keys.

This supports easy extension without modifying core logic and makes it possible to pass a
registry snapshot to `RewardManager` for test isolation and reproducibility.
"""

import collections.abc
import dataclasses
import typing

import pyine.organisms.models.rewards.core.types as reward_types


@dataclasses.dataclass(slots=True)
class RewardRegistry:
    """Mutable registry of reward term factories and aliases."""

    _factories: dict[str, reward_types.RewardTermFactory] = dataclasses.field(
        default_factory=lambda: typing.cast("dict[str, reward_types.RewardTermFactory]", {}),
    )
    """Mapping from canonical term type keys to factories."""
    _aliases: dict[str, str] = dataclasses.field(default_factory=lambda: typing.cast("dict[str, str]", {}))
    """Mapping from alias keys to canonical term type keys."""

    def register_term(
        self,
        term_type: str,
        factory: reward_types.RewardTermFactory,
    ) -> None:
        """Register a reward term factory under a stable canonical key."""
        key = term_type.strip()
        if not key:
            raise ValueError("term_type cannot be empty")
        if key in self._factories:
            raise ValueError(f"term_type already registered: {key}")
        self._factories[key] = factory

    def register_term_aliases(
        self,
        canonical_term_type: str,
        aliases: collections.abc.Sequence[str],
    ) -> None:
        """Register one or more aliases pointing to a canonical term type.

        This supports ergonomic type names like `"format/parseable_answer"` while keeping a stable
        canonical key for the underlying term.
        """
        canonical = canonical_term_type.strip()
        if not canonical:
            raise ValueError("canonical_term_type cannot be empty")
        if canonical not in self._factories:
            raise ValueError(f"cannot alias unknown term type: {canonical}")
        for alias in aliases:
            alias_key = alias.strip()
            if not alias_key:
                raise ValueError("alias cannot be empty")
            if alias_key in self._factories:
                raise ValueError(f"alias conflicts with existing term type: {alias_key}")
            if alias_key in self._aliases:
                raise ValueError(f"alias already registered: {alias_key}")
            self._aliases[alias_key] = canonical

    def resolve_term_type(
        self,
        term_type: str,
    ) -> str:
        """Resolve an alias to its canonical key (or return the input if already canonical)."""
        key = term_type.strip()
        if key in self._factories:
            return key
        if key in self._aliases:
            return self._aliases[key]
        raise KeyError(f"unknown term type: {key}; registered={sorted(self._factories.keys())}")

    def get_registered_term_types(self) -> collections.abc.Sequence[str]:
        """Return a stable list of registered canonical term type keys."""
        return tuple(sorted(self._factories.keys()))

    def get_term_factory(
        self,
        term_type: str,
    ) -> reward_types.RewardTermFactory:
        """Get the registered factory for the given term type key (supports aliases)."""
        key = self.resolve_term_type(term_type)
        return self._factories[key]

    def snapshot(self) -> "RewardRegistry":
        """Return a registry snapshot suitable for passing into `RewardManager`."""
        snapshot = RewardRegistry()
        snapshot._factories = dict(self._factories)
        snapshot._aliases = dict(self._aliases)
        return snapshot


_GLOBAL_REGISTRY = RewardRegistry()
"""Global registry instance used by default by `RewardManager`."""


def get_global_registry() -> RewardRegistry:
    """Return the global reward registry."""
    return _GLOBAL_REGISTRY


def register_term(
    term_type: str,
    factory: reward_types.RewardTermFactory,
) -> None:
    """Register a term factory in the global registry."""
    _GLOBAL_REGISTRY.register_term(term_type, factory)


def register_term_aliases(
    canonical_term_type: str,
    aliases: collections.abc.Sequence[str],
) -> None:
    """Register term type aliases in the global registry."""
    _GLOBAL_REGISTRY.register_term_aliases(canonical_term_type, aliases)


def get_registered_term_types() -> collections.abc.Sequence[str]:
    """Return a stable list of registered term type keys from the global registry."""
    return _GLOBAL_REGISTRY.get_registered_term_types()


def get_term_factory(
    term_type: str,
) -> reward_types.RewardTermFactory:
    """Get a term factory from the global registry (supports aliases)."""
    return _GLOBAL_REGISTRY.get_term_factory(term_type)
