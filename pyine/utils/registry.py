"""Generic registry utilities for factory pattern implementations.

This module provides a generic `Registry` class that can be used to register and retrieve
factory functions by string keys. It supports:
- Canonical key registration
- Alias registration (mapping alternative keys to canonical keys)
- Snapshot creation for test isolation
- Type-safe generic implementation

Example usage:
    ```python
    import typing
    import pyine.utils.registry


    # define a factory protocol
    class MyFactory(typing.Protocol):
        def __call__(self, config: dict) -> MyObject: ...


    # create a registry
    registry: pyine.utils.registry.Registry[MyFactory] = pyine.utils.registry.Registry()

    # register factories
    registry.register("my_type", my_factory_func)
    registry.register_aliases("my_type", ["alias1", "alias2"])

    # retrieve and use
    factory = registry.get("alias1")
    obj = factory(config)
    ```
"""

import collections.abc
import dataclasses
import typing

# @@@@@@ TODO: consider refactoring the prompt manager to use this registry too?


@dataclasses.dataclass(slots=True)
class Registry[FactoryT]:
    """Generic mutable registry of factories and aliases.

    This class provides a type-safe way to register and retrieve factory functions
    by string keys, with support for aliases and test isolation via snapshots.

    Type Parameters:
        FactoryT: The type of factory callable stored in the registry.
    """

    _factories: dict[str, FactoryT] = dataclasses.field(
        default_factory=lambda: typing.cast("dict[str, FactoryT]", {}),
    )
    """Mapping from canonical keys to factories."""

    _aliases: dict[str, str] = dataclasses.field(
        default_factory=lambda: typing.cast("dict[str, str]", {}),
    )
    """Mapping from alias keys to canonical keys."""

    def register(
        self,
        key: str,
        factory: FactoryT,
    ) -> None:
        """Register a factory under a stable canonical key.

        Args:
            key: Canonical key for the factory (will be stripped).
            factory: Factory callable to register.

        Raises:
            ValueError: If key is empty or already registered.
        """
        key_norm = key.strip()
        if not key_norm:
            raise ValueError("key cannot be empty")
        if key_norm in self._factories:
            raise ValueError(f"key already registered: {key_norm}")
        self._factories[key_norm] = factory

    def register_aliases(
        self,
        canonical_key: str,
        aliases: collections.abc.Sequence[str],
    ) -> None:
        """Register one or more aliases pointing to a canonical key.

        Aliases provide ergonomic alternative names (e.g., "format/parseable_answer")
        while keeping a stable canonical key for the underlying factory.

        Args:
            canonical_key: The canonical key that aliases should resolve to.
            aliases: Sequence of alias keys to register.

        Raises:
            ValueError: If canonical_key is empty, unknown, or if an alias is empty,
                conflicts with an existing key, or is already registered.
        """
        canonical = canonical_key.strip()
        if not canonical:
            raise ValueError("canonical_key cannot be empty")
        if canonical not in self._factories:
            raise ValueError(f"cannot alias unknown key: {canonical}")
        for alias in aliases:
            alias_key = alias.strip()
            if not alias_key:
                raise ValueError("alias cannot be empty")
            if alias_key in self._factories:
                raise ValueError(f"alias conflicts with existing key: {alias_key}")
            if alias_key in self._aliases:
                raise ValueError(f"alias already registered: {alias_key}")
            self._aliases[alias_key] = canonical

    def resolve(
        self,
        key: str,
    ) -> str:
        """Resolve an alias to its canonical key (or return the input if already canonical).

        Args:
            key: Key to resolve (may be canonical or alias).

        Returns:
            The canonical key.

        Raises:
            KeyError: If key is unknown (not a canonical key or alias).
        """
        key_norm = key.strip()
        if key_norm in self._factories:
            return key_norm
        if key_norm in self._aliases:
            return self._aliases[key_norm]
        registered_keys = sorted(self._factories.keys())
        registered_aliases = sorted(self._aliases.keys())
        raise KeyError(f"unknown key: {key_norm}; registered_keys={registered_keys}, aliases={registered_aliases}")

    def get(
        self,
        key: str,
    ) -> FactoryT:
        """Get the registered factory for the given key (supports aliases).

        Args:
            key: Key to look up (may be canonical or alias).

        Returns:
            The registered factory.

        Raises:
            KeyError: If key is unknown.
        """
        canonical = self.resolve(key)
        return self._factories[canonical]

    def get_keys(self) -> collections.abc.Sequence[str]:
        """Return a stable sorted list of registered canonical keys."""
        return tuple(sorted(self._factories.keys()))

    def get_aliases(self) -> collections.abc.Mapping[str, str]:
        """Return a copy of the alias mapping (alias -> canonical)."""
        return dict(self._aliases)

    def contains(
        self,
        key: str,
    ) -> bool:
        """Check if a key (canonical or alias) is registered.

        Args:
            key: Key to check.

        Returns:
            True if the key is registered (as canonical or alias).
        """
        key_norm = key.strip()
        return key_norm in self._factories or key_norm in self._aliases

    def snapshot(self) -> "Registry[FactoryT]":
        """Return an isolated copy of this registry.

        The snapshot contains shallow copies of the factory and alias mappings, so
        modifications to the original registry after taking a snapshot will not
        affect the snapshot. This is useful for test isolation.

        Returns:
            A new Registry instance with copies of this registry's mappings.

        Example:
            ```python
            # in test setup
            snapshot = global_registry.snapshot()
            manager = Manager(registry=snapshot)

            # modifications to global_registry won't affect snapshot
            ```
        """
        new_registry: Registry[FactoryT] = Registry()
        new_registry._factories = dict(self._factories)
        new_registry._aliases = dict(self._aliases)
        return new_registry

    def __len__(self) -> int:
        """Return the number of registered canonical keys."""
        return len(self._factories)

    def __contains__(
        self,
        key: object,
    ) -> bool:
        """Check if a key is registered (canonical or alias)."""
        if not isinstance(key, str):
            return False
        return self.contains(key)
