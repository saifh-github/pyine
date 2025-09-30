import collections
import importlib
import sys
import types

import pytest

import pyine.prompts.manager as pm


@pytest.fixture()
def transforms_with_fakes(monkeypatch):
    # fake langchain_core.messages with minimal message classes
    lc_module = types.ModuleType("langchain_core")
    lc_messages = types.ModuleType("langchain_core.messages")
    lc_output_parsers = types.ModuleType("langchain_core.output_parsers")

    class BaseMessage:
        def __init__(self, content):
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
        def __init__(self, *args, **kwargs):
            pass

        def get_format_instructions(self):
            return ""

    lc_output_parsers.PydanticOutputParser = PydanticOutputParser
    lc_prompts = types.ModuleType("langchain_core.prompts")
    # provide minimal BasePromptTemplate to satisfy type annotations in imported modules
    lc_prompts.BasePromptTemplate = type("BasePromptTemplate", (), {})

    # fake transformers with a minimal tokenizer class
    transformers_mod = types.ModuleType("transformers")
    transformers_mod.PreTrainedTokenizer = type("PreTrainedTokenizer", (), {})
    # add placeholders to satisfy any third-party imports that expect them
    transformers_mod.AutoModel = type("AutoModel", (), {})
    transformers_mod.AutoTokenizer = type("AutoTokenizer", (), {})

    # fake datasets with a simple Dataset that supports .map(batched=True)
    datasets_mod = types.ModuleType("datasets")

    class FakeDataset:
        def __init__(self, data):
            # data is a dict of lists, e.g., {"messages": [[...], [...]]}
            self.data = data

        def map(self, function, batched, desc, **kwargs):
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
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)
    monkeypatch.setitem(sys.modules, "datasets", datasets_mod)
    sys.modules.pop("pyine.organisms.datamodules.utils.transforms", None)
    transforms_mod = importlib.import_module("pyine.organisms.datamodules.utils.transforms")
    return transforms_mod


# add test for keep orig sample data


def test_create_sample_transform_string_no_answer(monkeypatch, transforms_with_fakes):
    transforms = transforms_with_fakes

    # provide a dummy prompt template with .format()
    class DummyTemplate:
        def format(self, **kwargs):
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


def test_create_sample_transform_string_with_answer(monkeypatch, transforms_with_fakes):
    transforms = transforms_with_fakes

    class DummyTemplate:
        def format(self, **kwargs):
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


def test_create_sample_transform_chat_to_hf_messages(monkeypatch, transforms_with_fakes):
    transforms = transforms_with_fakes
    lc_msgs = sys.modules["langchain_core.messages"]

    class DummyTemplate:
        def format_messages(self, **kwargs):
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


def test_apply_model_template_to_messages_basic(transforms_with_fakes):
    transforms = transforms_with_fakes
    datasets_mod = sys.modules["datasets"]
    transformers_mod = sys.modules["transformers"]

    class DummyTokenizer(transformers_mod.PreTrainedTokenizer):
        eos_token = "<eos>"

        def apply_chat_template(self, messages_batch, **kwargs):
            # messages_batch is a list of list-of-dicts
            out = []
            for msgs in messages_batch:
                rendered = " | ".join(f"{m['role']}: {m['content']}" for m in msgs)
                out.append(rendered)
            return out

    ds = datasets_mod.Dataset(
        {
            "messages": [
                [{"role": "user", "content": "Hi"}],
                [{"role": "user", "content": "Bye"}],
            ]
        }
    )
    tokenizer = DummyTokenizer()
    out_ds = transforms.apply_model_template_to_messages(
        hf_messages_dataset=ds,
        tokenizer=tokenizer,
        # keep defaults for strip_output and append_eos_token to avoid modifying loops
    )
    assert isinstance(out_ds, datasets_mod.Dataset)
    assert out_ds.data["text"] == ["user: Hi", "user: Bye"]
