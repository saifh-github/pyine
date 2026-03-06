"""LLM debate guardrail package.

Provides a guardrail scorer that uses a multi-turn interrogation debate
between two LLMs to judge correctness of model predictions via the
``GuardrailScorer`` protocol.
"""

from pyine.guardrails.llm_debate.configs import DebateGuardrailConfig
from pyine.guardrails.llm_debate.scorer import DebateGuardrailScorer

__all__ = [
    "DebateGuardrailConfig",
    "DebateGuardrailScorer",
]
