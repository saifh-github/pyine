"""Guardrails package - unified home for all guardrail architectures.

Houses activation probes, LLM classifiers, prompted LLM judges, and
LLM debate systems behind the common :class:`GuardrailMetricsConnector` protocol.

Available guardrail subpackages:
    - ``pyine.guardrails.prompted_llm`` - single-turn prompted LLM judge
    - ``pyine.guardrails.llm_debate`` - multi-turn LLM debate (interrogator/responder)
"""

from pyine.guardrails.connector import GuardrailMetricsConnector

__all__ = ["GuardrailMetricsConnector"]
