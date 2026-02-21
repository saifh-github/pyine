import langchain_core.messages
import langchain_core.outputs
import langchain_core.prompts
import langchain_core.runnables
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


class TestGetSamplingTemperatureFromChain:
    """Tests for get_sampling_temperature_from_chain."""

    def test_bare_model(self) -> None:
        llm = langchain_openai.ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key="fake")
        assert pyine.utils.langchain.get_sampling_temperature_from_chain(llm) == pytest.approx(0.7)

    def test_bare_model_zero_temperature(self) -> None:
        llm = langchain_openai.ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key="fake")
        assert pyine.utils.langchain.get_sampling_temperature_from_chain(llm) == pytest.approx(0.0)

    def test_prompt_pipe_model(self) -> None:
        prompt = langchain_core.prompts.ChatPromptTemplate.from_messages([("user", "{input}")])
        llm = langchain_openai.ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key="fake")
        chain = prompt | llm
        assert pyine.utils.langchain.get_sampling_temperature_from_chain(chain) == pytest.approx(0.2)

    def test_model_with_retry(self) -> None:
        llm = langchain_openai.ChatOpenAI(model="gpt-4o-mini", temperature=0.5, api_key="fake")
        chain = llm.with_retry(stop_after_attempt=3)
        assert pyine.utils.langchain.get_sampling_temperature_from_chain(chain) == pytest.approx(0.5)

    def test_prompt_pipe_model_with_retry(self) -> None:
        prompt = langchain_core.prompts.ChatPromptTemplate.from_messages([("user", "{input}")])
        llm = langchain_openai.ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key="fake")
        chain = (prompt | llm).with_retry(stop_after_attempt=2)
        assert pyine.utils.langchain.get_sampling_temperature_from_chain(chain) == pytest.approx(0.3)

    def test_bind_temperature_overrides_model(self) -> None:
        llm = langchain_openai.ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key="fake")
        chain = llm.bind(temperature=0.2)
        assert pyine.utils.langchain.get_sampling_temperature_from_chain(chain) == pytest.approx(0.2)

    def test_bind_temperature_after_retry(self) -> None:
        llm = langchain_openai.ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key="fake")
        chain = llm.with_retry(stop_after_attempt=3).bind(temperature=0.1)
        assert pyine.utils.langchain.get_sampling_temperature_from_chain(chain) == pytest.approx(0.1)

    def test_bind_temperature_none_stops_traversal(self) -> None:
        llm = langchain_openai.ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key="fake")
        chain = llm.bind(temperature=None)  # explicit None should not fall through to inner model
        assert pyine.utils.langchain.get_sampling_temperature_from_chain(chain) is None

    def test_opaque_lambda_returns_none(self) -> None:
        chain = langchain_core.runnables.RunnableLambda(lambda x: x)
        assert pyine.utils.langchain.get_sampling_temperature_from_chain(chain) is None


class TestCaptureChatModelStart:
    """Tests for on_chat_model_start preserving structured messages."""

    def test_preserves_roles_and_content(self) -> None:
        handler = pyine.utils.langchain.CaptureLLMHandler()
        messages = [
            [
                langchain_core.messages.SystemMessage(content="You are helpful"),
                langchain_core.messages.HumanMessage(content="Hello"),
                langchain_core.messages.AIMessage(content="Hi there"),
            ]
        ]
        handler.on_chat_model_start(serialized={"name": "gpt"}, messages=messages)
        assert len(handler.events) == 1
        event = handler.events[0]
        assert event.type == "llm_start"
        assert event.messages is not None
        assert len(event.messages) == 3
        assert event.messages[0]["role"] == "system"
        assert event.messages[0]["content"] == "You are helpful"
        assert event.messages[1]["role"] == "human"
        assert event.messages[1]["content"] == "Hello"
        assert event.messages[2]["role"] == "ai"
        assert event.messages[2]["content"] == "Hi there"
        assert event.prompts is None  # not set via on_chat_model_start

    def test_preserves_name_field(self) -> None:
        handler = pyine.utils.langchain.CaptureLLMHandler()
        msg = langchain_core.messages.HumanMessage(content="hi", name="user_a")
        handler.on_chat_model_start(serialized={}, messages=[[msg]])
        event = handler.events[0]
        assert event.messages is not None
        assert event.messages[0]["name"] == "user_a"

    def test_preserves_additional_kwargs(self) -> None:
        handler = pyine.utils.langchain.CaptureLLMHandler()
        msg = langchain_core.messages.AIMessage(
            content="response",
            additional_kwargs={"tool_calls": [{"id": "call_1"}]},
        )
        handler.on_chat_model_start(serialized={}, messages=[[msg]])
        event = handler.events[0]
        assert event.messages is not None
        assert "additional_kwargs" in event.messages[0]
        assert event.messages[0]["additional_kwargs"]["tool_calls"] == [{"id": "call_1"}]

    def test_empty_messages_list(self) -> None:
        handler = pyine.utils.langchain.CaptureLLMHandler()
        handler.on_chat_model_start(serialized={}, messages=[])
        event = handler.events[0]
        assert event.messages == []

    def test_does_not_set_prompts(self) -> None:
        handler = pyine.utils.langchain.CaptureLLMHandler()
        handler.on_chat_model_start(
            serialized={},
            messages=[[langchain_core.messages.HumanMessage(content="hi")]],
        )
        event = handler.events[0]
        assert event.prompts is None


@pytest.mark.integration
@pytest.mark.openai
@pytest.mark.skipif(
    tests.env_checks.OPENAI_API_KEY_MISSING or tests.env_checks.NETWORK_UNAVAILABLE,
    reason="OpenAI API key or network not available",
)
def test_capture_llm_handler_records_events_with_openai_chat() -> None:
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
