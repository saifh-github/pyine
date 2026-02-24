"""Guardrails package — unified home for all guardrail architectures.

Houses activation probes, LLM classifiers, and future guardrail types
behind the common :class:`GuardrailMetricsConnector` protocol.
"""

from pyine.guardrails.connector import GuardrailMetricsConnector

__all__ = ["GuardrailMetricsConnector"]
