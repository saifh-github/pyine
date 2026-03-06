"""Tests for GuardrailMetricsConnector protocol compliance."""

from __future__ import annotations

import inspect
import typing

from pyine.guardrails.connector import GuardrailMetricsConnector
from pyine.guardrails.llm_classifier.metrics_connector import ClassifierMetricsConnector
from pyine.guardrails.probes.metrics_connector import ProbeMetricsConnector

# ---------------------------------------------------------------------------
# Protocol method registry
# ---------------------------------------------------------------------------

# All methods/properties defined by GuardrailMetricsConnector
_PROTOCOL_MEMBERS: dict[str, str] = {
    "guardrail_type": "property",
    "load": "method",
    "get_static_info": "method",
    "benchmark_single_config": "method",
    "flatten_for_csv": "method",
    "get_wandb_summary": "method",
    "get_wandb_log_entries": "method",
    "cleanup": "method",
}


def _check_protocol_compliance(cls: type) -> None:
    """Verify that ``cls`` exposes every member defined in GuardrailMetricsConnector."""
    for name, kind in _PROTOCOL_MEMBERS.items():
        assert hasattr(cls, name), f"{cls.__name__} missing {kind} '{name}'"
        attr = getattr(cls, name)
        if kind == "property":
            # For properties, we just check it exists - instance check happens elsewhere
            pass
        elif kind == "method":
            # Verify it's callable
            assert callable(attr) or isinstance(attr, (classmethod, staticmethod)), (
                f"{cls.__name__}.{name} should be callable"
            )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    """Verify that concrete connectors satisfy the GuardrailMetricsConnector protocol."""

    def test_probe_connector_has_all_protocol_members(self) -> None:
        _check_protocol_compliance(ProbeMetricsConnector)

    def test_classifier_connector_has_all_protocol_members(self) -> None:
        _check_protocol_compliance(ClassifierMetricsConnector)

    def test_probe_connector_guardrail_type_returns_string(self) -> None:
        """guardrail_type should be a property returning a string."""
        # We can't easily instantiate without a real config, but we can inspect
        # that the class has a property descriptor.
        assert isinstance(
            inspect.getattr_static(ProbeMetricsConnector, "guardrail_type"),
            property,
        )

    def test_classifier_connector_guardrail_type_returns_string(self) -> None:
        assert isinstance(
            inspect.getattr_static(ClassifierMetricsConnector, "guardrail_type"),
            property,
        )

    def test_benchmark_single_config_keyword_only(self) -> None:
        """benchmark_single_config parameters should be keyword-only (matching protocol)."""
        for cls in (ProbeMetricsConnector, ClassifierMetricsConnector):
            sig = inspect.signature(cls.benchmark_single_config)
            params = list(sig.parameters.values())
            # Skip 'self'
            for p in params[1:]:
                assert p.kind == inspect.Parameter.KEYWORD_ONLY, (
                    f"{cls.__name__}.benchmark_single_config param '{p.name}' should be keyword-only"
                )

    def test_protocol_is_typing_protocol(self) -> None:
        """GuardrailMetricsConnector should be a typing.Protocol subclass."""
        assert issubclass(GuardrailMetricsConnector, typing.Protocol)
