"""Holds config settings specific to the code execution experiment app."""

import pydantic


class ModelConfig(pydantic.BaseModel):
    """Configuration settings to use for the LLM that will be 'executing' code."""

    provider: str
    model_name: str
    reasoning_effort: str | None = None
    temperature: float = pydantic.Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int | None = pydantic.Field(default=None, gt=0)


DEFAULT_MODELS = {
    "gpt-5": ModelConfig(provider="openai", model_name="gpt-5", reasoning_effort="medium"),
    "gpt-4o": ModelConfig(provider="openai", model_name="gpt-4o"),
    "gpt-4o-mini": ModelConfig(provider="openai", model_name="gpt-4o-mini"),
}
