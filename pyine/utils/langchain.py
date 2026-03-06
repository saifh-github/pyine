import datetime
import logging
import typing

import langchain_core.callbacks
import langchain_core.exceptions
import langchain_core.messages
import langchain_core.outputs
import langchain_core.runnables
import pydantic

import pyine.utils.portability

CapturedEventType = typing.Literal["llm_start", "llm_end", "llm_error"]
"""Potential event types that can be captured by the CaptureLLMHandler."""


class CapturedEvent(pydantic.BaseModel):
    """Captured LLM I/O event."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
    """Pydantic model configuration (frozen)."""
    type: CapturedEventType
    """Type of the captured event."""
    serialized: dict[str, typing.Any] | None = None
    """Serialized runnable (class path, name, etc.); only captured captured on LLM start."""
    prompts: list[str] | None = None
    """List of prompts; only captured on LLM start."""
    messages: list[dict[str, typing.Any]] | None = None
    """Structured chat messages as role/content dicts; only set on chat model start."""
    response: langchain_core.outputs.LLMResult | None = None
    """LLM response; only captured captured on LLM end."""
    error: str | None = None
    """Error message; only captured captured on LLM error."""
    created_at: datetime.datetime = pydantic.Field(
        default_factory=lambda: datetime.datetime.now(),
    )
    """Timestamp of event creation (in local time)."""
    kwargs: dict[str, typing.Any] = pydantic.Field(
        default_factory=lambda: typing.cast("dict[str, typing.Any]", {}),
    )
    """Keyword arguments passed to the LLM call; may include e.g. invocation_params."""


class CaptureLLMHandler(langchain_core.callbacks.BaseCallbackHandler):
    """Collect full LLM I/O for logging/debugging."""

    def __init__(self) -> None:
        """Initialize the handler."""
        self.events: list[CapturedEvent] = []

    @typing.override
    def on_llm_start(
        self,
        serialized: dict[str, typing.Any],
        prompts: list[str],
        **kwargs: typing.Any,
    ) -> None:
        """Called before the LLM starts."""
        self.events.append(
            CapturedEvent(
                type="llm_start",
                serialized=serialized,
                prompts=prompts,
                kwargs=kwargs,
            )
        )

    @typing.override
    def on_chat_model_start(
        self,
        serialized: dict[str, typing.Any],
        messages: list[list[langchain_core.messages.BaseMessage]],
        **kwargs: typing.Any,
    ) -> None:
        """Called before a chat model starts; preserves structured messages."""
        structured: list[dict[str, typing.Any]] = []
        if messages:
            for msg in messages[0]:  # only first prompt, mirrors prompts[0]
                entry: dict[str, typing.Any] = {
                    "role": msg.type,
                    "content": pyine.utils.portability.make_json_serializable(msg.content),  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                }
                if getattr(msg, "name", None) is not None:
                    entry["name"] = msg.name
                additional = getattr(msg, "additional_kwargs", None)
                if additional:
                    entry["additional_kwargs"] = pyine.utils.portability.make_json_serializable(dict(additional))
                structured.append(entry)
        self.events.append(
            CapturedEvent(
                type="llm_start",
                serialized=serialized,
                messages=structured,
                kwargs=kwargs,
            )
        )

    @typing.override
    def on_llm_end(
        self,
        response: langchain_core.outputs.LLMResult,
        **kwargs: typing.Any,
    ) -> None:
        """Called after the LLM returns."""
        self.events.append(
            CapturedEvent(
                type="llm_end",
                response=response,
                kwargs=kwargs,
            )
        )

    @typing.override
    def on_llm_error(
        self,
        error: BaseException,
        **kwargs: typing.Any,
    ) -> None:
        """Called when the LLM raises an error."""
        self.events.append(
            CapturedEvent(
                type="llm_error",
                error=str(error),
                kwargs=kwargs,
            )
        )

    def get_latest_event(self, event_type: CapturedEventType | None = None) -> CapturedEvent | None:
        """Get the latest captured event with an optional target type."""
        if event_type is None:
            return self.events[-1] if self.events else None
        target_events = [e for e in self.events if e.type == event_type] if self.events else []
        return target_events[-1] if target_events else None


def is_invocable_chain(obj: typing.Any) -> bool:
    """Returns whether the given object is an invocable chain, i.e. it supports 'invoke'."""
    return bool(
        isinstance(obj, langchain_core.runnables.Runnable)
        or (hasattr(obj, "invoke") and callable(getattr(obj, "invoke", None)))
    )


def get_sampling_temperature_from_chain(
    chain: langchain_core.runnables.Runnable[typing.Any, typing.Any],
) -> float | None:
    """Best-effort extraction of sampling temperature from a LangChain runnable.

    Walks the runnable's internal structure looking for a component with a ``temperature``
    attribute (e.g. a ``BaseChatModel``). Returns the value if found, ``None`` if the chain
    structure is opaque or no temperature is set.

    Handles common patterns: bare models, ``RunnableSequence`` (prompt | model), and
    ``RunnableBinding`` (``.with_retry()``, ``.bind()``, etc.). For ``RunnableBinding``,
    bound kwargs (from ``.bind(temperature=...)``) take precedence over the inner model's
    attribute.
    """
    # this function is inherently duck-typed: it inspects opaque LangChain internal
    # attributes (kwargs, temperature, first/last, bound) that don't exist on the base
    # Runnable type. use Any to silence pyright for the dynamic attribute accesses.
    obj: typing.Any = chain
    # RunnableBinding kwargs override (from .bind(temperature=...)); if temperature is present
    # in bound kwargs, treat it as authoritative and stop traversal (even if None/non-numeric)
    if hasattr(obj, "kwargs"):
        bound_kwargs = typing.cast("dict[str, typing.Any] | None", obj.kwargs)
        if isinstance(bound_kwargs, dict) and "temperature" in bound_kwargs:
            temp = bound_kwargs["temperature"]
            return float(temp) if isinstance(temp, (int, float)) else None
    # direct attribute (e.g. bare BaseChatModel)
    if hasattr(obj, "temperature"):
        if isinstance(obj.temperature, (int, float)):
            return float(obj.temperature)
    # RunnableSequence: walk first, middle steps, and last
    if hasattr(obj, "first") and hasattr(obj, "last"):
        steps: list[typing.Any] = [obj.first, *getattr(obj, "middle", []), obj.last]
        for step in steps:
            result = get_sampling_temperature_from_chain(step)
            if result is not None:
                return result
    # RunnableBinding: unwrap .with_retry(), .bind(), etc.
    if hasattr(obj, "bound"):
        return get_sampling_temperature_from_chain(obj.bound)
    return None


def get_default_structured_output_chain_retry_config(
    max_retries: int = 3,
) -> dict[str, typing.Any]:
    """Returns a default LangChain `with_retry` configuration that can be used w/ OpenAI."""
    return {
        "retry_if_exception_type": (
            langchain_core.exceptions.OutputParserException,  # for structured parsing failures
        ),
        "wait_exponential_jitter": True,  # backoff + jitter
        "stop_after_attempt": max_retries,  # on top of max_retries specified in model config
    }


_logger = logging.getLogger(__name__)


def extract_token_count_from_handler(handler: CaptureLLMHandler) -> float:
    """Extract total token count from a CaptureLLMHandler.

    Checks llm_output.token_usage first, falls back to per-generation info.
    Returns 0.0 if no token usage is available.
    """
    end_event = handler.get_latest_event("llm_end")
    if end_event is None or end_event.response is None:
        return 0.0
    llm_output: dict[str, typing.Any] = getattr(end_event.response, "llm_output", None) or {}
    token_usage: dict[str, typing.Any] = llm_output.get("token_usage", {})
    total: int = token_usage.get("total_tokens", 0)
    if total > 0:
        return float(total)
    # Fallback: sum from generation info
    for generation_list in end_event.response.generations:
        for gen in generation_list:
            gen_info: dict[str, typing.Any] = getattr(gen, "generation_info", None) or {}
            usage: dict[str, typing.Any] = gen_info.get("usage", {})
            total += usage.get("total_tokens", 0)
    if total == 0:
        _logger.debug("token count is 0 for a successful LLM response; provider may not populate token usage fields")
    return float(total)
