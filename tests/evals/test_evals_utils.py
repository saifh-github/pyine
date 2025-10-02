import pathlib

import pytest

import pyine.evals.utils
import pyine.prompts
import pyine.utils.llm_providers
import tests.env_checks


def test_parse_usage_from_dict_basic() -> None:
    # Basic dict response with usage section and common fields
    response = {
        "usage": {
            "total_tokens": 100,
            "prompt_tokens": "unknown",
            "completion_tokens": 10,
            "reasoning_tokens": 30,
        }
    }
    info = pyine.evals.utils.parse_token_usage_from_response(response)
    assert isinstance(info, pyine.evals.utils.TokenUsageInfo)
    assert info.total_tokens == 100
    assert info.prompt_tokens == "unknown"
    assert info.completion_tokens == 10
    assert info.reasoning_tokens == 30


def test_parse_usage_from_attr_object() -> None:
    class _Usage:
        total_tokens = 12
        prompt_tokens = 5
        completion_tokens = 7

    obj = type("ObjWithUsage", (), {})()
    obj.usage = _Usage()
    info = pyine.evals.utils.parse_token_usage_from_response(obj)
    assert info.total_tokens == 12
    assert info.prompt_tokens == 5
    assert info.completion_tokens == 7
    assert info.reasoning_tokens == "unknown"
    # thinking_tokens alias should populate reasoning_tokens when present
    obj2 = type("ObjWithUsage2", (), {})()
    obj2.usage = type("U", (), {"thinking_tokens": 3})()
    info2 = pyine.evals.utils.parse_token_usage_from_response(obj2)
    assert info2.reasoning_tokens == 3


def test_parse_usage_from_nested_creation_meta_llm_output() -> None:
    class _Wrapper:
        pass

    wrapper = _Wrapper()
    wrapper.creation_meta = {
        "llm_output": {
            "usage": {
                "total_tokens": 33,
            }
        }
    }
    info = pyine.evals.utils.parse_token_usage_from_response(wrapper)
    assert info.total_tokens == 33


def test_parse_usage_raises_when_no_information() -> None:
    with pytest.raises(ValueError, match="could not deduce token usage information"):
        _ = pyine.evals.utils.parse_token_usage_from_response({})


def test_token_usage_info_add_and_iadd_success() -> None:
    r1 = {"usage": {"total_tokens": 10, "prompt_tokens": 6, "completion_tokens": 4}}
    r2 = {"usage": {"total_tokens": 20, "prompt_tokens": 3, "completion_tokens": 7}}
    info1 = pyine.evals.utils.parse_token_usage_from_response(r1)
    info2 = pyine.evals.utils.parse_token_usage_from_response(r2)
    summed = info1 + info2
    assert summed.total_tokens == 30
    assert summed.prompt_tokens == 9
    assert summed.completion_tokens == 11
    assert summed.reasoning_tokens == "unknown"

    info1 += info2
    assert info1.total_tokens == 30
    assert info1.prompt_tokens == 9
    assert info1.completion_tokens == 11
    assert info1.reasoning_tokens == "unknown"

    known_output = {"usage": {"cached_tokens": 5, "total_tokens": 5}}
    unknown_output = {"usage": {"total_tokens": 10}}  # output_tokens omitted -> 'unknown'
    info_known = pyine.evals.utils.parse_token_usage_from_response(known_output)
    info_unknown = pyine.evals.utils.parse_token_usage_from_response(unknown_output)
    with pytest.raises(ValueError):
        _ = info_known + info_unknown
    with pytest.raises(ValueError):
        info_unknown += info_known


@pytest.mark.skipif(
    tests.env_checks.OPENAI_API_KEY_MISSING or tests.env_checks.NETWORK_UNAVAILABLE,
    reason="OpenAI API key or network not available; cannot run OpenAI-backed evaluation.",
)
def test_token_usage_with_real_openai_generation(tmp_path: pathlib.Path) -> None:
    model = pyine.utils.llm_providers.LLMProviderConfig(
        provider="openai",
        model_kwargs={
            "model": "gpt-4o-mini",
        },
    ).get_model()
    prompt_config = pyine.prompts.PromptBuildConfig(
        prompt_name="code_summary",
        partial_vars={
            "target_word_count": 30,
        },
    )
    code_snippet = """\
def f(x):
    iters = 0
    while x > 0:
        x //= 10
        iters += 1
    return iters
"""
    records = pyine.prompts.fetch_or_generate_prompt_results(
        model=model,
        identifier="potato",
        input_variables={"code": code_snippet},
        prompt_config=prompt_config,
        db=pyine.prompts.PromptResultDB(pathlib.Path(tmp_path) / "prompt_results.sqlite"),
        log_new_results=False,
    )
    assert len(records) == 1
    info = pyine.evals.utils.parse_token_usage_from_response(records[0])
    assert info.total_tokens != "unknown" and info.total_tokens > 0
