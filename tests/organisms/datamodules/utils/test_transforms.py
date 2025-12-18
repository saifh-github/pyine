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


def test_create_sample_transform_with_line_numbers(
    monkeypatch: pytest.MonkeyPatch,
    transforms_with_fakes: typing.Any,
) -> None:
    transforms = transforms_with_fakes

    class DummyTemplate:
        def format(
            self,
            **kwargs: typing.Any,
        ) -> str:
            return f"Code:\n{kwargs['code']}"

    monkeypatch.setattr(pm, "get_prompt_template", lambda use_chat_template, **kw: DummyTemplate())
    transform_fn = transforms.create_sample_transform(
        use_chat_template=False,
        append_answer=False,
        add_line_numbers=True,
        line_number_prefix_pattern="L{num}|",
    )
    Sample = collections.namedtuple("Sample", ["code", "expected_output"])
    sample = Sample(code="a = 1\nb = 2", expected_output="ignored")
    result = transform_fn(sample)
    assert "L1|a = 1" in result
    assert "L2|b = 2" in result


def test_create_sample_transform_with_line_numbers_custom_width_and_padding(
    monkeypatch: pytest.MonkeyPatch,
    transforms_with_fakes: typing.Any,
) -> None:
    transforms = transforms_with_fakes

    class DummyTemplate:
        def format(
            self,
            **kwargs: typing.Any,
        ) -> str:
            return kwargs["code"]

    monkeypatch.setattr(pm, "get_prompt_template", lambda use_chat_template, **kw: DummyTemplate())
    transform_fn = transforms.create_sample_transform(
        use_chat_template=False,
        append_answer=False,
        add_line_numbers=True,
        line_number_prefix_pattern="{num}|",
        line_number_width=3,
        line_number_zero_pad=False,
    )
    Sample = collections.namedtuple("Sample", ["code", "expected_output"])
    sample = Sample(code="x = 1\ny = 2", expected_output="ignored")
    result = transform_fn(sample)
    assert "  1|x = 1" in result
    assert "  2|y = 2" in result


def test_create_sample_transform_with_block_markers_partial_exec(
    monkeypatch: pytest.MonkeyPatch,
    transforms_with_fakes: typing.Any,
) -> None:
    transforms = transforms_with_fakes

    class DummyTemplate:
        def format(
            self,
            **kwargs: typing.Any,
        ) -> str:
            return kwargs["code"]

    monkeypatch.setattr(pm, "get_prompt_template", lambda use_chat_template, **kw: DummyTemplate())
    transform_fn = transforms.create_sample_transform(
        use_chat_template=False,
        append_answer=False,
        add_block_markers=True,
    )
    Sample = collections.namedtuple("Sample", ["code", "expected_output", "predict_type", "first_line", "last_line"])
    sample = Sample(
        code="a = 1\nb = 2\nc = 3",
        expected_output="ignored",
        predict_type="frame_variables",  # not program_output, so markers should be added
        first_line=1,
        last_line=2,
    )
    result = transform_fn(sample)
    lines = result.splitlines()
    assert "# <<<< START HERE" in lines[0]
    assert "# <<<< END HERE" in lines[1]
    assert "# <<<<" not in lines[2]


def test_create_sample_transform_with_block_markers_skipped_for_program_output(
    monkeypatch: pytest.MonkeyPatch,
    transforms_with_fakes: typing.Any,
) -> None:
    transforms = transforms_with_fakes

    class DummyTemplate:
        def format(
            self,
            **kwargs: typing.Any,
        ) -> str:
            return kwargs["code"]

    monkeypatch.setattr(pm, "get_prompt_template", lambda use_chat_template, **kw: DummyTemplate())
    transform_fn = transforms.create_sample_transform(
        use_chat_template=False,
        append_answer=False,
        add_block_markers=True,
    )
    Sample = collections.namedtuple("Sample", ["code", "expected_output", "predict_type", "first_line", "last_line"])
    sample = Sample(
        code="a = 1\nb = 2\nc = 3",
        expected_output="ignored",
        predict_type="program_output",  # markers should NOT be added
        first_line=1,
        last_line=2,
    )
    result = transform_fn(sample)
    assert "# <<<<" not in result


def test_create_sample_transform_with_block_markers_custom_suffixes(
    monkeypatch: pytest.MonkeyPatch,
    transforms_with_fakes: typing.Any,
) -> None:
    transforms = transforms_with_fakes

    class DummyTemplate:
        def format(
            self,
            **kwargs: typing.Any,
        ) -> str:
            return kwargs["code"]

    monkeypatch.setattr(pm, "get_prompt_template", lambda use_chat_template, **kw: DummyTemplate())
    transform_fn = transforms.create_sample_transform(
        use_chat_template=False,
        append_answer=False,
        add_block_markers=True,
        block_start_suffix="  # BEGIN",
        block_end_suffix="  # END",
    )
    Sample = collections.namedtuple("Sample", ["code", "expected_output", "predict_type", "first_line", "last_line"])
    sample = Sample(
        code="x = 1\ny = 2",
        expected_output="ignored",
        predict_type="frame_variables",
        first_line=1,
        last_line=2,
    )
    result = transform_fn(sample)
    lines = result.splitlines()
    assert lines[0] == "x = 1  # BEGIN"
    assert lines[1] == "y = 2  # END"


def test_create_sample_transform_with_block_markers_single_line(
    monkeypatch: pytest.MonkeyPatch,
    transforms_with_fakes: typing.Any,
) -> None:
    transforms = transforms_with_fakes

    class DummyTemplate:
        def format(
            self,
            **kwargs: typing.Any,
        ) -> str:
            return kwargs["code"]

    monkeypatch.setattr(pm, "get_prompt_template", lambda use_chat_template, **kw: DummyTemplate())
    transform_fn = transforms.create_sample_transform(
        use_chat_template=False,
        append_answer=False,
        add_block_markers=True,
    )
    Sample = collections.namedtuple("Sample", ["code", "expected_output", "predict_type", "first_line", "last_line"])
    sample = Sample(
        code="a = 1\nb = 2\nc = 3",
        expected_output="ignored",
        predict_type="frame_variables",
        first_line=2,
        last_line=2,  # single line block
    )
    result = transform_fn(sample)
    lines = result.splitlines()
    assert "# <<<<" not in lines[0]
    assert "# <<<< START HERE" in lines[1]
    assert "# <<<<" not in lines[2]


def test_create_sample_transform_with_line_numbers_and_block_markers(
    monkeypatch: pytest.MonkeyPatch,
    transforms_with_fakes: typing.Any,
) -> None:
    transforms = transforms_with_fakes

    class DummyTemplate:
        def format(
            self,
            **kwargs: typing.Any,
        ) -> str:
            return kwargs["code"]

    monkeypatch.setattr(pm, "get_prompt_template", lambda use_chat_template, **kw: DummyTemplate())
    transform_fn = transforms.create_sample_transform(
        use_chat_template=False,
        append_answer=False,
        add_line_numbers=True,
        line_number_prefix_pattern="L{num}|",
        add_block_markers=True,
    )
    Sample = collections.namedtuple("Sample", ["code", "expected_output", "predict_type", "first_line", "last_line"])
    sample = Sample(
        code="a = 1\nb = 2\nc = 3",
        expected_output="ignored",
        predict_type="function_return",
        first_line=1,
        last_line=2,
    )
    result = transform_fn(sample)
    lines = result.splitlines()
    # block markers applied first, then line numbers
    assert lines[0] == "L1|a = 1  # <<<< START HERE"
    assert lines[1] == "L2|b = 2  # <<<< END HERE"
    assert lines[2] == "L3|c = 3"


def test_create_sample_transform_block_markers_raises_for_invalid_lines(
    monkeypatch: pytest.MonkeyPatch,
    transforms_with_fakes: typing.Any,
) -> None:
    transforms = transforms_with_fakes

    class DummyTemplate:
        def format(
            self,
            **kwargs: typing.Any,
        ) -> str:
            return kwargs["code"]

    monkeypatch.setattr(pm, "get_prompt_template", lambda use_chat_template, **kw: DummyTemplate())
    transform_fn = transforms.create_sample_transform(
        use_chat_template=False,
        append_answer=False,
        add_block_markers=True,
    )
    Sample = collections.namedtuple("Sample", ["code", "expected_output", "predict_type", "first_line", "last_line"])
    # first_line=0 is invalid, should raise ValueError
    sample_invalid_start = Sample(
        code="a = 1\nb = 2",
        expected_output="ignored",
        predict_type="frame_variables",
        first_line=0,
        last_line=2,
    )
    with pytest.raises(ValueError, match="start_line.*out of bounds"):
        transform_fn(sample_invalid_start)
    # start > end should also raise
    sample_invalid_order = Sample(
        code="a = 1\nb = 2",
        expected_output="ignored",
        predict_type="frame_variables",
        first_line=2,
        last_line=1,
    )
    with pytest.raises(ValueError, match="start_line.*must be <= end_line"):
        transform_fn(sample_invalid_order)
