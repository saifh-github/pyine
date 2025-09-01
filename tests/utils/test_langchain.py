import langchain_core.messages
import langchain_openai
import pytest

from pyine.utils.langchain import CaptureLLMHandler
from tests.data.utils.env_checks import OPENAI_API_KEY_MISSING


@pytest.mark.skipif(OPENAI_API_KEY_MISSING, reason="OpenAI API key not available")
def test_capture_llm_handler_records_events_with_openai_chat():
    model_name = "gpt-4o-mini"
    handler = CaptureLLMHandler()
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
