import os

import pytest

import pyine.utils.llm_providers as lp

# use local stubs to inject into the imported module attributes


class InMemoryRateLimiter:
    def __init__(self, **cfg):
        self.config = cfg


class RunnableSequence:
    def __init__(self, *steps):
        self.steps = steps


class ChatDeepSeek:
    def __init__(self, *args, **kwargs):
        self.rate_limiter = kwargs.pop("rate_limiter", None)
        self.init_kwargs = kwargs
        self.retry_config = None
        self.structured_model = None

    def with_retries(self, **cfg):
        self.retry_config = cfg
        return self

    def with_structured_output(self, model):
        self.structured_model = model
        return self


class ChatOpenAI(ChatDeepSeek):
    pass


def test_invalid_provider_raises():
    with pytest.raises(ValueError):
        lp.get_llm_from_provider("bogus")


def test_deepseek_with_env_and_retries(monkeypatch: pytest.MonkeyPatch):
    # patch imported modules directly so we control behavior regardless of installed deps
    monkeypatch.setattr(lp.langchain_core.rate_limiters, "InMemoryRateLimiter", InMemoryRateLimiter, raising=True)
    monkeypatch.setattr(lp.langchain_deepseek, "ChatDeepSeek", ChatDeepSeek, raising=True)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "KEY123")
    monkeypatch.setenv("DEEPSEEK_API_BASE_URL", "https://deepseek.example/api")
    llm = lp.get_llm_from_provider(
        provider="deepseek",
        rate_limiter_config={"max_calls": 5, "period": 1},
        with_retry_config={"max_retries": 7},
        temperature=0.0,
        model="deepseek-chat",
    )
    assert isinstance(llm, ChatDeepSeek)
    assert llm.init_kwargs["api_key"] == os.environ["DEEPSEEK_API_KEY"]
    assert llm.init_kwargs["base_url"] == os.environ["DEEPSEEK_API_BASE_URL"]
    assert llm.init_kwargs["temperature"] == 0.0
    assert llm.init_kwargs["model"] == "deepseek-chat"
    assert isinstance(llm.rate_limiter, InMemoryRateLimiter)
    assert llm.retry_config == {"max_retries": 7}


def test_openai_get_chain_with_structured_output(monkeypatch: pytest.MonkeyPatch):
    # patch RunnableSequence and OpenAI class in the module
    monkeypatch.setattr(lp.langchain_core.runnables, "RunnableSequence", RunnableSequence, raising=True)
    monkeypatch.setattr(lp.langchain_openai, "ChatOpenAI", ChatOpenAI, raising=True)
    llm = lp.get_llm_from_provider(
        provider="openai",
        api_key="KEY",
        base_url="https://openai.example/v1",
        model="gpt-test",
    )

    class DummyModel:
        pass

    prompt = object()  # any object is fine; get_chain just forwards it to RunnableSequence
    chain = lp.get_chain(prompt, llm, pydantic_model=DummyModel)  # noqa
    assert isinstance(chain, RunnableSequence)
    assert chain.steps[0] is prompt
    assert llm.structured_model is DummyModel


def test_openai_env_default_base_url_no_structured(monkeypatch: pytest.MonkeyPatch):
    # ensure env provides API key, and OPENAI_BASE_URL is not set so default applies
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "OPENAI_KEY")
    monkeypatch.setattr(lp.langchain_openai, "ChatOpenAI", ChatOpenAI, raising=True)
    monkeypatch.setattr(lp.langchain_core.runnables, "RunnableSequence", RunnableSequence, raising=True)
    # do not pass api_key/base_url in kwargs to trigger env/default usage
    llm = lp.get_llm_from_provider(provider="openai", model="gpt-test")
    assert isinstance(llm, ChatOpenAI)
    assert llm.init_kwargs["api_key"] == os.environ["OPENAI_API_KEY"]
    # expect default base URL when env is missing
    assert llm.init_kwargs["base_url"] == "https://api.openai.com/v1"
    # get_chain without pydantic_model should NOT set structured_model
    prompt = object()
    chain = lp.get_chain(prompt, llm)  # noqa
    assert isinstance(chain, RunnableSequence)
    assert llm.structured_model is None


def test_deepseek_explicit_keys_override_env(monkeypatch: pytest.MonkeyPatch):
    # set env to different values, but pass explicit ones to ensure override
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ENVKEY")
    monkeypatch.setenv("DEEPSEEK_API_BASE_URL", "https://env.example")
    monkeypatch.setattr(lp.langchain_core.rate_limiters, "InMemoryRateLimiter", InMemoryRateLimiter, raising=True)
    monkeypatch.setattr(lp.langchain_deepseek, "ChatDeepSeek", ChatDeepSeek, raising=True)
    llm = lp.get_llm_from_provider(
        provider="deepseek",
        api_key="EXPL-KEY",
        base_url="https://explicit.example",
        model="deepseek-chat",
    )
    assert isinstance(llm, ChatDeepSeek)
    assert llm.init_kwargs["api_key"] == "EXPL-KEY"
    assert llm.init_kwargs["base_url"] == "https://explicit.example"
