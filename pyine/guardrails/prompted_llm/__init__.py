"""Prompted LLM guardrail package.

Provides a guardrail scorer that uses a prompted (non-fine-tuned) LLM
to judge correctness of model predictions via the ``GuardrailScorer`` protocol.
"""

from pyine.guardrails.prompted_llm.configs import PromptedLLMGuardrailConfig
from pyine.guardrails.prompted_llm.scorer import PromptedLLMGuardrailScorer

__all__ = [
    "PromptedLLMGuardrailConfig",
    "PromptedLLMGuardrailScorer",
]
