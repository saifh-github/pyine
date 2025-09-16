import langchain_core.messages
import langchain_core.outputs
import langchain_openai
import pytest

import pyine.utils.langchain
import tests.env_checks


def test_capture_llm_handler_manual_event_sequence() -> None:
    handler = pyine.utils.langchain.CaptureLLMHandler()
    assert handler.get_latest_event() is None
    handler.on_llm_start(
        serialized={"name": "gpt"},
        prompts=["hi"],
        invocation_params={"temperature": 0},
    )
    llm_result = langchain_core.outputs.LLMResult(
        generations=[[langchain_core.outputs.Generation(text="done")]],
        llm_output={"token_usage": {"total_tokens": 1}},
    )
    handler.on_llm_end(llm_result, request_id="req-1")
    handler.on_llm_error(RuntimeError("boom"), attempt=2)

    assert [event.type for event in handler.events] == [
        "llm_start",
        "llm_end",
        "llm_error",
    ]

    latest = handler.get_latest_event()
    assert latest is not None and latest.type == "llm_error"
    assert latest.error == "boom" and latest.kwargs == {"attempt": 2}

    end_event = handler.get_latest_event("llm_end")
    assert end_event is not None and end_event.response is llm_result
    assert end_event.kwargs == {"request_id": "req-1"}

    start_event = handler.get_latest_event("llm_start")
    assert start_event is not None and start_event.prompts == ["hi"]
    assert start_event.serialized == {"name": "gpt"}
    assert start_event.kwargs == {"invocation_params": {"temperature": 0}}

    assert handler.get_latest_event("missing") is None  # noqa


@pytest.mark.skipif(tests.env_checks.OPENAI_API_KEY_MISSING, reason="OpenAI API key not available")
def test_capture_llm_handler_records_events_with_openai_chat():
    model_name = "gpt-4o-mini"
    handler = pyine.utils.langchain.CaptureLLMHandler()
    llm = langchain_openai.ChatOpenAI(model=model_name, temperature=0, max_tokens=16)
    result = llm.with_config(callbacks=[handler]).invoke("Reply with a single word: hello")
    assert result is not None and isinstance(result, langchain_core.messages.AIMessage)
    event_types = [e.type for e in handler.events]
    assert "llm_start" in event_types, f"expected llm_start in events, got: {event_types}"
    assert "llm_end" in event_types, f"expected llm_end in events, got: {event_types}"
    assert event_types.index("llm_start") < event_types.index("llm_end"), "events out of order"
    end_events = [e for e in handler.events if e.type == "llm_end"]
    assert len(end_events) == 1, "missing llm_end event"
    assert handler.get_latest_event("llm_end") == end_events[0]
    raw_response = end_events[0].response
    assert raw_response is not None
    # do a quick check for token usage (should be available for any openai model)
    raw_output = raw_response.llm_output
    assert raw_output is not None
    assert raw_output["token_usage"]["total_tokens"] > 0
