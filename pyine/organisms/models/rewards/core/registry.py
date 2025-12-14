"""Reward term registry.

The registry decouples `RewardManager` from individual term modules:
- term modules register factories under stable string keys;
- the manager constructs terms via `RewardTermSpec.type` keys.

This supports easy extension without modifying core logic and makes it possible to pass a
registry snapshot to `RewardManager` for test isolation and reproducibility.

This module uses the generic `pyine.utils.registry.Registry` as its underlying implementation.
"""

import collections.abc
import dataclasses

import pyine.organisms.models.rewards.core.types as reward_types
import pyine.utils.registry


@dataclasses.dataclass(frozen=True, slots=True)
class TermInfo:
    """Information about a registered reward term type.

    This is returned by `list_available_terms()` to provide discoverable information
    about available term types.
    """

    canonical_type: str
    """The canonical (stable) registry key for this term type."""
    aliases: tuple[str, ...]
    """Alternative names that can be used to reference this term type."""
    docstring: str | None
    """The factory function's docstring, if available."""


class RewardRegistry:
    """Mutable registry of reward term factories and aliases.

    This is a domain-specific wrapper around `pyine.utils.registry.Registry` that provides
    reward-specific naming (e.g., `register_term` instead of `register`).
    """

    def __init__(self) -> None:
        """Create an empty reward registry."""
        self._registry: pyine.utils.registry.Registry[reward_types.RewardTermFactory] = pyine.utils.registry.Registry()

    def register_term(
        self,
        term_type: str,
        factory: reward_types.RewardTermFactory,
    ) -> None:
        """Register a reward term factory under a stable canonical key."""
        self._registry.register(term_type, factory)

    def register_term_aliases(
        self,
        canonical_term_type: str,
        aliases: collections.abc.Sequence[str],
    ) -> None:
        """Register one or more aliases pointing to a canonical term type."""
        self._registry.register_aliases(canonical_term_type, aliases)

    def resolve_term_type(
        self,
        term_type: str,
    ) -> str:
        """Resolve an alias to its canonical key (or return the input if already canonical)."""
        return self._registry.resolve(term_type)

    def get_registered_term_types(self) -> collections.abc.Sequence[str]:
        """Return a stable list of registered canonical term type keys."""
        return self._registry.get_keys()

    def get_term_factory(
        self,
        term_type: str,
    ) -> reward_types.RewardTermFactory:
        """Get the registered factory for the given term type key (supports aliases)."""
        return self._registry.get(term_type)

    def snapshot(self) -> "RewardRegistry":
        """Return an isolated copy of this registry for test isolation."""
        new_registry = RewardRegistry()
        new_registry._registry = self._registry.snapshot()
        return new_registry

    def list_available_terms(self) -> list[TermInfo]:
        """List all registered term types with their aliases and docstrings."""
        canonical_types = self._registry.get_keys()
        all_aliases = self._registry.get_aliases()
        result: list[TermInfo] = []
        for canonical in canonical_types:
            term_aliases = tuple(alias for alias, target in sorted(all_aliases.items()) if target == canonical)
            factory = self._registry.get(canonical)
            result.append(
                TermInfo(
                    canonical_type=canonical,
                    aliases=term_aliases,
                    docstring=factory.__doc__,
                )
            )
        return result


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


def list_available_terms() -> list[TermInfo]:
    """List all registered term types with their aliases and docstrings.

    This is useful for discovering what term types are available and how to use them.
    Ensures builtin terms are registered before returning the list.

    Returns:
        A list of TermInfo objects, sorted by canonical type name.

    Example:
        ```python
        import pyine.organisms.models.rewards as rewards

        for term_info in rewards.list_available_terms():
            print(f"{term_info.canonical_type}: {term_info.aliases}")
        ```
    """
    import pyine.organisms.models.rewards.terms as reward_terms

    reward_terms.ensure_builtin_terms_registered()
    return _GLOBAL_REGISTRY.list_available_terms()
