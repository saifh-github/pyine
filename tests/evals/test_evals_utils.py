import collections
import pathlib

import numpy as np
import pytest
import torch
import transformers

import pyine.evals.utils
import pyine.prompts
import pyine.utils.llm_providers
import pyine.utils.transformers
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


def test_category_metrics_callback_streaming(tmp_path: pathlib.Path) -> None:
    categories = [["bugfix"], ["refactor"], ["bugfix", "refactor"]]
    callback = pyine.evals.utils.CategoryWiseMetricsCallback(
        data_sample_categories=categories,
        metrics_prefix="eval",
        log_fn=lambda _msg: None,
    )
    logits_batch1 = np.zeros((2, 3, 4), dtype=np.float32)
    logits_batch1[0, 0, 1] = 8.0
    logits_batch1[0, 1, 0] = 8.0
    logits_batch1[1, 0, 2] = 8.0
    logits_batch1[1, 1, 1] = 8.0
    labels_batch1 = np.array([[-100, 1, 0], [-100, 2, 1]], dtype=np.int64)
    pred_batch1 = transformers.trainer_utils.EvalPrediction(
        predictions=logits_batch1,
        label_ids=labels_batch1,
    )
    assert callback(pred_batch1, compute_result=False) == {}

    logits_batch2 = np.zeros((1, 3, 4), dtype=np.float32)
    labels_batch2 = np.array([[-100, 1, 2]], dtype=np.int64)
    pred_batch2 = transformers.trainer_utils.EvalPrediction(
        predictions=logits_batch2,
        label_ids=labels_batch2,
    )
    metrics = callback(pred_batch2, compute_result=True)

    full_logits = np.concatenate([logits_batch1, logits_batch2], axis=0)
    full_labels = np.concatenate([labels_batch1, labels_batch2], axis=0)

    def _compute_expected_metrics() -> dict[str, float]:
        shift_logits = torch.as_tensor(full_logits)[:, :-1, :].contiguous()
        shift_labels = torch.as_tensor(full_labels)[:, 1:].contiguous()
        losses = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1).long(),
            reduction="none",
            ignore_index=pyine.utils.transformers.default_ignore_index,
        )
        losses = losses.view(shift_labels.shape)
        mask = shift_labels.ne(pyine.utils.transformers.default_ignore_index)
        token_counts = mask.sum(dim=1)
        token_loss = (losses * mask).sum(dim=1)
        per_example_loss = torch.zeros_like(token_loss, dtype=torch.float32)
        valid = token_counts > 0
        per_example_loss[valid] = token_loss[valid] / token_counts[valid].to(torch.float32)
        sums: collections.defaultdict[str, float] = collections.defaultdict(float)
        counts: collections.defaultdict[str, int] = collections.defaultdict(int)
        for idx, loss_value in enumerate(per_example_loss.tolist()):
            for category in categories[idx]:
                sums[category] += float(loss_value)
                counts[category] += 1
        expected: dict[str, float] = {}
        for category, loss_sum in sums.items():
            count = counts[category]
            expected[f"eval/{category}/loss"] = loss_sum / count
            expected[f"eval/{category}/count"] = count
        return expected

    expected_metrics = _compute_expected_metrics()
    for key, value in expected_metrics.items():
        if key.endswith("/loss"):
            assert metrics[key] == pytest.approx(value, rel=1e-6)
        else:
            assert metrics[key] == value
    assert callback.get_aggregated_metrics() == metrics

    training_args = transformers.TrainingArguments(output_dir=str(tmp_path))
    callback.on_evaluate(
        args=training_args,
        state=transformers.trainer_callback.TrainerState(),
        control=transformers.trainer_callback.TrainerControl(),
        metrics=dict(metrics),
    )

    assert callback(pred_batch1, compute_result=False) == {}
    metrics_second_pass = callback(pred_batch2, compute_result=True)
    for key, value in expected_metrics.items():
        if key.endswith("/loss"):
            assert metrics_second_pass[key] == pytest.approx(value, rel=1e-6)
        else:
            assert metrics_second_pass[key] == value


@pytest.mark.skipif(
    tests.env_checks.OPENAI_API_KEY_MISSING or tests.env_checks.NETWORK_UNAVAILABLE,
    reason="OpenAI API key or network not available; cannot run OpenAI-backed evaluation.",
)
def test_token_usage_with_real_openai_generation(tmp_path: pathlib.Path) -> None:
    model_config = pyine.utils.llm_providers.LLMProviderConfig(
        provider="openai",
        model_kwargs={
            "model": "gpt-4o-mini",
        },
    )
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
        identifier="potato",
        input_variables={"code": code_snippet},
        prompt_chain_config=pyine.prompts.PromptChainBuildConfig(
            prompt=prompt_config,
            provider=model_config,
        ),
        db=pyine.prompts.PromptResultDB(pathlib.Path(tmp_path) / "prompt_results.sqlite"),
        log_new_results=False,
    )
    assert len(records) == 1
    info = pyine.evals.utils.parse_token_usage_from_response(records[0])
    assert info.total_tokens != "unknown" and info.total_tokens > 0
