import os

import pytest

import pyine.utils.llm_providers as lp


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
    with pytest.raises(NotImplementedError):
        lp.get_model_from_provider("bogus")


def test_deepseek_with_env_and_retries(monkeypatch: pytest.MonkeyPatch):
    # patch imported modules directly so we control behavior regardless of installed deps
    monkeypatch.setattr(lp.langchain_core.rate_limiters, "InMemoryRateLimiter", InMemoryRateLimiter, raising=True)
    monkeypatch.setattr(lp.langchain_deepseek, "ChatDeepSeek", ChatDeepSeek, raising=True)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "KEY123")
    monkeypatch.setenv("DEEPSEEK_API_BASE_URL", "https://deepseek.example/api")
    llm = lp.get_model_from_provider(
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


def test_openai_env_default_base(monkeypatch: pytest.MonkeyPatch):
    # ensure env provides API key, and OPENAI_BASE_URL is not set so default applies
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "OPENAI_KEY")
    monkeypatch.setattr(lp.langchain_openai, "ChatOpenAI", ChatOpenAI, raising=True)
    # do not pass api_key/base_url in kwargs to trigger env/default usage
    llm = lp.get_model_from_provider(provider="openai", model="gpt-test")
    assert isinstance(llm, ChatOpenAI)
    assert llm.init_kwargs["api_key"] == os.environ["OPENAI_API_KEY"]
    # expect default base URL when env is missing
    assert llm.init_kwargs["base_url"] == "https://api.openai.com/v1"


def test_deepseek_explicit_keys_override_env(monkeypatch: pytest.MonkeyPatch):
    # set env to different values, but pass explicit ones to ensure override
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ENVKEY")
    monkeypatch.setenv("DEEPSEEK_API_BASE_URL", "https://env.example")
    monkeypatch.setattr(lp.langchain_core.rate_limiters, "InMemoryRateLimiter", InMemoryRateLimiter, raising=True)
    monkeypatch.setattr(lp.langchain_deepseek, "ChatDeepSeek", ChatDeepSeek, raising=True)
    llm = lp.get_model_from_provider(
        provider="deepseek",
        api_key="EXPL-KEY",
        base_url="https://explicit.example",
        model="deepseek-chat",
    )
    assert isinstance(llm, ChatDeepSeek)
    assert llm.init_kwargs["api_key"] == "EXPL-KEY"
    assert llm.init_kwargs["base_url"] == "https://explicit.example"


def test_llm_provider_config_from_dict_requires_provider() -> None:
    with pytest.raises(ValueError):
        lp.LLMProviderConfig.from_dict({"model": "gpt"})


def test_llm_provider_config_instantiates_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lp.langchain_core.rate_limiters, "InMemoryRateLimiter", InMemoryRateLimiter, raising=True)
    monkeypatch.setattr(lp.langchain_openai, "ChatOpenAI", ChatOpenAI, raising=True)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    cfg_dict = {
        "provider": "openai",
        "rate_limiter_config": {"max_calls": 2, "period": 1},
        "with_retry_config": {"max_retries": 5},
        "model": "gpt-test",
        "temperature": 0.1,
    }
    cfg = lp.LLMProviderConfig.from_dict(cfg_dict)
    assert cfg.model_kwargs == {"model": "gpt-test", "temperature": 0.1}
    model = cfg.get_model()
    assert isinstance(model, ChatOpenAI)
    assert isinstance(model.rate_limiter, InMemoryRateLimiter)
    assert model.init_kwargs["model"] == "gpt-test"
    assert model.init_kwargs["temperature"] == 0.1
    assert model.retry_config == {"max_retries": 5}
