"""Tests for the generic Registry utility."""

import pytest

import pyine.utils.registry


class TestRegistry:
    def test_register_and_get(self) -> None:
        registry: pyine.utils.registry.Registry[str] = pyine.utils.registry.Registry()
        registry.register("key1", "value1")
        assert registry.get("key1") == "value1"

    def test_register_strips_key(self) -> None:
        registry: pyine.utils.registry.Registry[str] = pyine.utils.registry.Registry()
        registry.register("  key1  ", "value1")
        assert registry.get("key1") == "value1"

    def test_register_empty_key_raises(self) -> None:
        registry: pyine.utils.registry.Registry[str] = pyine.utils.registry.Registry()
        with pytest.raises(ValueError, match="key cannot be empty"):
            registry.register("", "value")

    def test_register_whitespace_key_raises(self) -> None:
        registry: pyine.utils.registry.Registry[str] = pyine.utils.registry.Registry()
        with pytest.raises(ValueError, match="key cannot be empty"):
            registry.register("   ", "value")

    def test_register_duplicate_raises(self) -> None:
        registry: pyine.utils.registry.Registry[str] = pyine.utils.registry.Registry()
        registry.register("key1", "value1")
        with pytest.raises(ValueError, match="key already registered"):
            registry.register("key1", "value2")

    def test_get_unknown_key_raises(self) -> None:
        registry: pyine.utils.registry.Registry[str] = pyine.utils.registry.Registry()
        with pytest.raises(KeyError, match="unknown key"):
            registry.get("unknown")

    def test_register_aliases(self) -> None:
        registry: pyine.utils.registry.Registry[str] = pyine.utils.registry.Registry()
        registry.register("canonical", "value")
        registry.register_aliases("canonical", ["alias1", "alias2"])
        assert registry.get("alias1") == "value"
        assert registry.get("alias2") == "value"
        assert registry.get("canonical") == "value"

    def test_register_aliases_strips_keys(self) -> None:
        registry: pyine.utils.registry.Registry[str] = pyine.utils.registry.Registry()
        registry.register("canonical", "value")
        registry.register_aliases("  canonical  ", ["  alias1  "])
        assert registry.get("alias1") == "value"

    def test_register_aliases_empty_canonical_raises(self) -> None:
        registry: pyine.utils.registry.Registry[str] = pyine.utils.registry.Registry()
        with pytest.raises(ValueError, match="canonical_key cannot be empty"):
            registry.register_aliases("", ["alias"])

    def test_register_aliases_unknown_canonical_raises(self) -> None:
        registry: pyine.utils.registry.Registry[str] = pyine.utils.registry.Registry()
        with pytest.raises(ValueError, match="cannot alias unknown key"):
            registry.register_aliases("unknown", ["alias"])

    def test_register_aliases_empty_alias_raises(self) -> None:
        registry: pyine.utils.registry.Registry[str] = pyine.utils.registry.Registry()
        registry.register("canonical", "value")
        with pytest.raises(ValueError, match="alias cannot be empty"):
            registry.register_aliases("canonical", [""])

    def test_register_aliases_conflicts_with_key_raises(self) -> None:
        registry: pyine.utils.registry.Registry[str] = pyine.utils.registry.Registry()
        registry.register("key1", "value1")
        registry.register("key2", "value2")
        with pytest.raises(ValueError, match="alias conflicts with existing key"):
            registry.register_aliases("key1", ["key2"])

    def test_register_aliases_duplicate_raises(self) -> None:
        registry: pyine.utils.registry.Registry[str] = pyine.utils.registry.Registry()
        registry.register("canonical", "value")
        registry.register_aliases("canonical", ["alias1"])
        with pytest.raises(ValueError, match="alias already registered"):
            registry.register_aliases("canonical", ["alias1"])

    def test_resolve_canonical_returns_same(self) -> None:
        registry: pyine.utils.registry.Registry[str] = pyine.utils.registry.Registry()
        registry.register("canonical", "value")
        assert registry.resolve("canonical") == "canonical"

    def test_resolve_alias_returns_canonical(self) -> None:
        registry: pyine.utils.registry.Registry[str] = pyine.utils.registry.Registry()
        registry.register("canonical", "value")
        registry.register_aliases("canonical", ["alias1"])
        assert registry.resolve("alias1") == "canonical"

    def test_resolve_unknown_raises(self) -> None:
        registry: pyine.utils.registry.Registry[str] = pyine.utils.registry.Registry()
        with pytest.raises(KeyError, match="unknown key"):
            registry.resolve("unknown")

    def test_get_keys(self) -> None:
        registry: pyine.utils.registry.Registry[str] = pyine.utils.registry.Registry()
        registry.register("b", "v1")
        registry.register("a", "v2")
        registry.register("c", "v3")
        keys = registry.get_keys()
        assert keys == ("a", "b", "c")  # sorted

    def test_get_aliases(self) -> None:
        registry: pyine.utils.registry.Registry[str] = pyine.utils.registry.Registry()
        registry.register("canonical", "value")
        registry.register_aliases("canonical", ["alias1", "alias2"])
        aliases = registry.get_aliases()
        assert aliases == {"alias1": "canonical", "alias2": "canonical"}

    def test_contains_canonical(self) -> None:
        registry: pyine.utils.registry.Registry[str] = pyine.utils.registry.Registry()
        registry.register("key1", "value")
        assert registry.contains("key1") is True
        assert registry.contains("unknown") is False

    def test_contains_alias(self) -> None:
        registry: pyine.utils.registry.Registry[str] = pyine.utils.registry.Registry()
        registry.register("canonical", "value")
        registry.register_aliases("canonical", ["alias1"])
        assert registry.contains("alias1") is True

    def test_len(self) -> None:
        registry: pyine.utils.registry.Registry[str] = pyine.utils.registry.Registry()
        assert len(registry) == 0
        registry.register("key1", "v1")
        assert len(registry) == 1
        registry.register("key2", "v2")
        assert len(registry) == 2
        registry.register_aliases("key1", ["alias"])
        assert len(registry) == 2  # aliases don't count

    def test_in_operator(self) -> None:
        registry: pyine.utils.registry.Registry[str] = pyine.utils.registry.Registry()
        registry.register("key1", "value")
        registry.register_aliases("key1", ["alias1"])
        assert "key1" in registry
        assert "alias1" in registry
        assert "unknown" not in registry
        assert 123 not in registry  # type: ignore[operator]

    def test_snapshot_creates_isolated_copy(self) -> None:
        registry: pyine.utils.registry.Registry[str] = pyine.utils.registry.Registry()
        registry.register("key1", "value1")
        registry.register_aliases("key1", ["alias1"])
        snapshot = registry.snapshot()
        registry.register("key2", "value2")
        assert snapshot.get("key1") == "value1"
        assert snapshot.get("alias1") == "value1"
        assert "key2" not in snapshot

    def test_snapshot_modifications_dont_affect_original(self) -> None:
        registry: pyine.utils.registry.Registry[str] = pyine.utils.registry.Registry()
        registry.register("key1", "value1")
        snapshot = registry.snapshot()
        snapshot.register("key2", "value2")
        assert "key2" not in registry


class TestRegistryWithCallables:
    def test_register_and_call_factory(self) -> None:
        def factory(x: int) -> str:
            return f"result_{x}"

        registry: pyine.utils.registry.Registry[type[factory]] = pyine.utils.registry.Registry()  # type: ignore[valid-type]
        registry.register("my_factory", factory)
        retrieved = registry.get("my_factory")
        assert retrieved(42) == "result_42"
