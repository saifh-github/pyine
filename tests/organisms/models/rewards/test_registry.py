"""Tests for RewardRegistry."""

import pytest

import pyine.organisms.models.rewards.core.configs as reward_configs
import pyine.organisms.models.rewards.core.registry as reward_registry
import pyine.organisms.models.rewards.core.types as reward_types
import pyine.utils.parsing


def _make_dummy_factory(
    return_value: float = 1.0,
) -> reward_types.RewardTermFactory:
    """Create a dummy term factory for testing."""

    class DummyTerm:
        def reset(self, run_init_ctx: reward_types.RunInitContext) -> None:
            pass

        def __call__(self, sample_ctx: reward_types.SampleContext) -> reward_types.TermResult:
            return reward_types.TermResult(value=return_value)

    def factory(
        spec: reward_configs.RewardTermSpec,
        *,
        parser: pyine.utils.parsing.OutputParser | None,
    ) -> reward_types.RewardTerm:
        del spec, parser
        return DummyTerm()

    return factory


class TestRewardRegistry:
    def test_register_and_retrieve(self) -> None:
        registry = reward_registry.RewardRegistry()
        factory = _make_dummy_factory()
        registry.register_term("test_term", factory)
        retrieved = registry.get_term_factory("test_term")
        assert retrieved is factory

    def test_get_registered_term_types(self) -> None:
        registry = reward_registry.RewardRegistry()
        registry.register_term("b_term", _make_dummy_factory())
        registry.register_term("a_term", _make_dummy_factory())
        types = registry.get_registered_term_types()
        assert types == ("a_term", "b_term")  # sorted

    def test_duplicate_registration_raises(self) -> None:
        registry = reward_registry.RewardRegistry()
        registry.register_term("test_term", _make_dummy_factory())
        with pytest.raises(ValueError, match="already registered"):
            registry.register_term("test_term", _make_dummy_factory())

    def test_empty_term_type_raises(self) -> None:
        registry = reward_registry.RewardRegistry()
        with pytest.raises(ValueError, match="cannot be empty"):
            registry.register_term("", _make_dummy_factory())
        with pytest.raises(ValueError, match="cannot be empty"):
            registry.register_term("   ", _make_dummy_factory())

    def test_unknown_term_type_raises(self) -> None:
        registry = reward_registry.RewardRegistry()
        with pytest.raises(KeyError, match="unknown key"):
            registry.get_term_factory("nonexistent")

    def test_alias_registration_and_resolution(self) -> None:
        registry = reward_registry.RewardRegistry()
        factory = _make_dummy_factory()
        registry.register_term("canonical", factory)
        registry.register_term_aliases("canonical", ["alias1", "alias2"])
        assert registry.get_term_factory("alias1") is factory
        assert registry.get_term_factory("alias2") is factory
        assert registry.resolve_term_type("alias1") == "canonical"

    def test_alias_for_unknown_canonical_raises(self) -> None:
        registry = reward_registry.RewardRegistry()
        with pytest.raises(ValueError, match="cannot alias unknown key"):
            registry.register_term_aliases("nonexistent", ["alias"])

    def test_alias_conflicts_with_term_type_raises(self) -> None:
        registry = reward_registry.RewardRegistry()
        registry.register_term("term1", _make_dummy_factory())
        registry.register_term("term2", _make_dummy_factory())
        with pytest.raises(ValueError, match="alias conflicts with existing key"):
            registry.register_term_aliases("term1", ["term2"])

    def test_duplicate_alias_raises(self) -> None:
        registry = reward_registry.RewardRegistry()
        registry.register_term("term1", _make_dummy_factory())
        registry.register_term("term2", _make_dummy_factory())
        registry.register_term_aliases("term1", ["shared_alias"])
        with pytest.raises(ValueError, match="alias already registered"):
            registry.register_term_aliases("term2", ["shared_alias"])

    def test_empty_alias_raises(self) -> None:
        registry = reward_registry.RewardRegistry()
        registry.register_term("canonical", _make_dummy_factory())
        with pytest.raises(ValueError, match="alias cannot be empty"):
            registry.register_term_aliases("canonical", [""])

    def test_snapshot_creates_isolated_copy(self) -> None:
        registry = reward_registry.RewardRegistry()
        factory1 = _make_dummy_factory(1.0)
        registry.register_term("term1", factory1)
        registry.register_term_aliases("term1", ["alias1"])
        snapshot = registry.snapshot()
        factory2 = _make_dummy_factory(2.0)
        registry.register_term("term2", factory2)
        assert snapshot.get_term_factory("term1") is factory1
        assert snapshot.get_term_factory("alias1") is factory1
        with pytest.raises(KeyError):
            snapshot.get_term_factory("term2")

    def test_resolve_canonical_returns_same(self) -> None:
        registry = reward_registry.RewardRegistry()
        registry.register_term("canonical", _make_dummy_factory())
        assert registry.resolve_term_type("canonical") == "canonical"
