"""Configuration for the LLM debate guardrail scorer."""

import pydantic

import pyine.utils.llm_providers


class DebateGuardrailConfig(pydantic.BaseModel):
    """Configuration for the LLM debate guardrail scorer.

    Example::

        config = DebateGuardrailConfig(
            interrogator_provider=LLMProviderConfig(
                provider="openai",
                model_kwargs={"model": "gpt-5-mini"},
            ),
            responder_provider=LLMProviderConfig(
                provider="vllm",
                model_kwargs={"model": "my-rl-checkpoint", "temperature": 0.0},
            ),
        )
        scorer = DebateGuardrailScorer(config)
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    # --- LLM providers ---
    interrogator_provider: pyine.utils.llm_providers.LLMProviderConfig
    """The interrogator/judge LLM (API or vLLM)."""
    responder_provider: pyine.utils.llm_providers.LLMProviderConfig
    """The responder LLM (typically vLLM serving the RL checkpoint)."""

    # --- Prompts ---
    interrogator_prompt_name: str = "guardrail/debate_interrogator"
    """Name of the interrogator prompt template (resolved by PromptManager)."""
    responder_prompt_name: str = "guardrail/debate_responder"
    """Name of the responder prompt template (resolved by PromptManager)."""
    interrogator_verdict_prompt_name: str = "guardrail/debate_interrogator_verdict"
    """Name of the verdict-only interrogator prompt (used on the final forced-verdict turn)."""
    use_chat_template: bool = True
    """Whether to use a chat prompt template (system + human message)."""

    # --- Retries ---
    chain_retry_max_attempts: int = pydantic.Field(default=3, ge=0, le=10)
    """Maximum number of retry attempts for each chain invocation (interrogator,
    responder, verdict). Retries cover both parse errors (OutputParserException)
    and provider errors (timeouts, rate limits, 5xx). Set to 0 to disable."""

    chain_retry_wait_exponential_jitter: bool = True
    """Whether to use exponential backoff with jitter between chain retries."""

    # --- Debate ---
    max_debate_turns: int = pydantic.Field(default=3, ge=1, le=10)
    """Maximum number of interrogation rounds (interrogator asks + responder responds = 1 turn)."""

    responder_sees_debate_history: bool = True
    """Whether the responder sees the full debate history in its prompt.
    When True (default), the responder prompt includes all prior debate turns,
    allowing the responder to give consistent, non-contradictory answers.
    When False, the responder only sees its original output + the latest
    interrogator question, forcing it to defend its reasoning fresh each turn
    without knowledge of prior interrogation lines."""

    # --- Scoring ---
    max_workers: int = pydantic.Field(default=5, ge=1)
    """Max concurrent debates. Lower than prompted_llm due to multi-turn cost."""
    default_score_on_error: float = pydantic.Field(default=0.5, ge=0.0, le=1.0)
    """Score to assign when the debate fails after retries."""

    # --- Debug ---
    debug_log_transcript_every_n: int = pydantic.Field(default=0, ge=0)
    """Log a full debate transcript to the terminal every N scored records.
    0 = disabled (default). Useful for visually inspecting debate quality during a run."""
