import collections
import importlib
import sys
import types
import typing

import pytest

import pyine.prompts.manager as pm


@pytest.fixture()
def transforms_with_fakes(
    monkeypatch: pytest.MonkeyPatch,
) -> types.ModuleType:
    # fake langchain_core.messages with minimal message classes
    lc_module = types.ModuleType("langchain_core")
    lc_messages = types.ModuleType("langchain_core.messages")
    lc_output_parsers = types.ModuleType("langchain_core.output_parsers")

    class BaseMessage:
        def __init__(
            self,
            content: str,
        ) -> None:
            self.content = content

    class SystemMessage(BaseMessage):
        pass

    class HumanMessage(BaseMessage):
        pass

    class AIMessage(BaseMessage):
        pass

    lc_messages.BaseMessage = BaseMessage
    lc_messages.SystemMessage = SystemMessage
    lc_messages.HumanMessage = HumanMessage
    lc_messages.AIMessage = AIMessage

    # minimal output parser class to satisfy imports
    class PydanticOutputParser:
        def __init__(
            self,
            *args: typing.Any,
            **kwargs: typing.Any,
        ) -> None:
            pass

        def get_format_instructions(self) -> str:
            return ""

    lc_output_parsers.PydanticOutputParser = PydanticOutputParser
    lc_prompts = types.ModuleType("langchain_core.prompts")
    lc_prompts_chat = types.ModuleType("langchain_core.prompts.chat")
    # provide minimal BasePromptTemplate to satisfy type annotations in imported modules
    lc_prompts.BasePromptTemplate = type("BasePromptTemplate", (), {})
    # provide ChatPromptTemplate for chat template assertions
    lc_prompts_chat.ChatPromptTemplate = type("ChatPromptTemplate", (lc_prompts.BasePromptTemplate,), {})
    lc_prompts.chat = lc_prompts_chat

    # fake transformers with a minimal tokenizer class
    transformers_mod = types.ModuleType("transformers")
    transformers_mod.PreTrainedTokenizer = type("PreTrainedTokenizer", (), {})
    # add placeholders to satisfy any third-party imports that expect them
    transformers_mod.AutoModel = type("AutoModel", (), {})
    transformers_mod.AutoTokenizer = type("AutoTokenizer", (), {})

    # fake datasets with a simple Dataset that supports .map(batched=True)
    datasets_mod = types.ModuleType("datasets")

    class FakeDataset:
        def __init__(
            self,
            data: dict[str, list[typing.Any]],
        ) -> None:
            # data is a dict of lists, e.g., {"messages": [[...], [...]]}
            self.data = data

        def map(
            self,
            function: typing.Callable[[dict[str, list[typing.Any]]], dict[str, list[typing.Any]]],
            batched: bool,
            desc: str | None,
            **kwargs: typing.Any,
        ) -> "FakeDataset":
            result = function(self.data)
            return FakeDataset({"text": result["text"]})

    datasets_mod.Dataset = FakeDataset
    # link submodules as attributes of the parent fake package
    lc_module.messages = lc_messages
    lc_module.output_parsers = lc_output_parsers
    lc_module.prompts = lc_prompts

    monkeypatch.setitem(sys.modules, "langchain_core", lc_module)
    monkeypatch.setitem(sys.modules, "langchain_core.messages", lc_messages)
    monkeypatch.setitem(sys.modules, "langchain_core.output_parsers", lc_output_parsers)
    monkeypatch.setitem(sys.modules, "langchain_core.prompts", lc_prompts)
    monkeypatch.setitem(sys.modules, "langchain_core.prompts.chat", lc_prompts_chat)
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)
    monkeypatch.setitem(sys.modules, "datasets", datasets_mod)
    sys.modules.pop("pyine.organisms.datamodules.utils.transforms", None)
    return importlib.import_module("pyine.organisms.datamodules.utils.transforms")


# add test for keep orig sample data


def test_create_sample_transform_string_no_answer(
    monkeypatch: pytest.MonkeyPatch,
    transforms_with_fakes: typing.Any,
) -> None:
    transforms = transforms_with_fakes

    # provide a dummy prompt template with .format()
    class DummyTemplate:
        def format(
            self,
            **kwargs: typing.Any,
        ) -> str:
            return f"Q: {kwargs['question']}"

    # patch the prompt manager getter to return our dummy
    monkeypatch.setattr(pm, "get_prompt_template", lambda use_chat_template, **kw: DummyTemplate())
    transform_fn = transforms.create_sample_transform(
        use_chat_template=False,
        append_answer=False,
    )
    # minimal sample with _asdict and output attribute
    Sample = collections.namedtuple("Sample", ["question", "expected_output"])
    sample = Sample(question="What is this?", expected_output="An answer")
    result = transform_fn(sample)
    assert result == "Q: What is this?"


def test_create_sample_transform_string_with_answer(
    monkeypatch: pytest.MonkeyPatch,
    transforms_with_fakes: typing.Any,
) -> None:
    transforms = transforms_with_fakes

    class DummyTemplate:
        def format(
            self,
            **kwargs: typing.Any,
        ) -> str:
            return f"Q: {kwargs['question']}"

    monkeypatch.setattr(pm, "get_prompt_template", lambda use_chat_template, **kw: DummyTemplate())
    transform_fn = transforms.create_sample_transform(
        use_chat_template=False,
        append_answer=True,
    )
    Sample = collections.namedtuple("Sample", ["question", "expected_output"])
    sample = Sample(question="What is this?", expected_output="An answer")
    result = transform_fn(sample)
    assert result == "Q: What is this?\nAn answer"


def test_create_sample_transform_chat_to_hf_messages(
    monkeypatch: pytest.MonkeyPatch,
    transforms_with_fakes: typing.Any,
) -> None:
    transforms = transforms_with_fakes
    lc_msgs = sys.modules["langchain_core.messages"]
    lc_prompts_chat = sys.modules["langchain_core.prompts.chat"]

    class DummyTemplate(lc_prompts_chat.ChatPromptTemplate):
        def format_messages(
            self,
            **kwargs: typing.Any,
        ) -> list[lc_msgs.HumanMessage]:
            # One human message; the transform will append an AI message
            return [lc_msgs.HumanMessage(kwargs["question"])]

    monkeypatch.setattr(pm, "get_prompt_template", lambda use_chat_template, **kw: DummyTemplate())
    transform_fn = transforms.create_sample_transform(
        use_chat_template=True,
        append_answer=True,
        use_hf_messages=True,
    )
    Sample = collections.namedtuple("Sample", ["question", "expected_output"])
    sample = Sample(question="Hello", expected_output="Hi!")
    result = transform_fn(sample)
    assert isinstance(result, dict)
    assert "messages" in result
    assert result["messages"] == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ]
