import datetime
import typing

import langchain_core.callbacks
import langchain_core.outputs
import langchain_core.runnables
import pydantic

CapturedEventType = typing.Literal["llm_start", "llm_end", "llm_error"]
"""Potential event types that can be captured by the CaptureLLMHandler."""


class CapturedEvent(pydantic.BaseModel):
    """Captured LLM I/O event."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
    """Pydantic model configuration (frozen)."""
    type: CapturedEventType
    """Type of the captured event."""
    serialized: dict | None = None
    """Serialized runnable (class path, name, etc.); only captured captured on LLM start."""
    prompts: list[str] | None = None
    """List of prompts; only captured captured on LLM start."""
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

    def on_llm_start(
        self,
        serialized: dict,
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
