"""Tests for pyine.evals.utils module."""

import collections
import pathlib
import typing

import numpy as np
import pytest
import torch
import transformers

import pyine.evals.utils
import pyine.prompts
import pyine.utils.llm_providers
import pyine.utils.transformers
import tests.env_checks


class TestTokenUsageParsing:
    """Tests for parse_token_usage_from_response."""

    @pytest.mark.parametrize(
        "response,expected",
        [
            pytest.param(
                {
                    "usage": {
                        "total_tokens": 100,
                        "prompt_tokens": "unknown",
                        "completion_tokens": 10,
                        "reasoning_tokens": 30,
                    }
                },
                {"total_tokens": 100, "prompt_tokens": "unknown", "completion_tokens": 10, "reasoning_tokens": 30},
                id="dict_with_all_fields",
            ),
            pytest.param(
                {"usage": {"total_tokens": 33}},
                {"total_tokens": 33},
                id="dict_with_total_only",
            ),
        ],
    )
    def test_parse_from_dict_response(
        self,
        response: dict[str, typing.Any],
        expected: dict[str, typing.Any],
    ) -> None:
        """parse_token_usage_from_response extracts usage from dict response."""
        info = pyine.evals.utils.parse_token_usage_from_response(response)
        assert isinstance(info, pyine.evals.utils.TokenUsageInfo)
        for key, value in expected.items():
            assert getattr(info, key) == value

    def test_parse_from_attr_object(self) -> None:
        """parse_token_usage_from_response extracts usage from object with usage attribute."""

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

    def test_parse_thinking_tokens_alias(self) -> None:
        """thinking_tokens alias populates reasoning_tokens when present."""
        obj = type("ObjWithUsage", (), {})()
        obj.usage = type("U", (), {"thinking_tokens": 3})()
        info = pyine.evals.utils.parse_token_usage_from_response(obj)
        assert info.reasoning_tokens == 3

    def test_parse_from_nested_creation_meta(self) -> None:
        """parse_token_usage_from_response extracts from nested creation_meta.llm_output."""

        class _Wrapper:
            pass

        wrapper = _Wrapper()
        wrapper.creation_meta = {"llm_output": {"usage": {"total_tokens": 33}}}
        info = pyine.evals.utils.parse_token_usage_from_response(wrapper)
        assert info.total_tokens == 33

    def test_parse_raises_when_no_usage_info(self) -> None:
        """parse_token_usage_from_response raises ValueError when no usage info found."""
        with pytest.raises(ValueError, match="could not deduce token usage information"):
            pyine.evals.utils.parse_token_usage_from_response({})


class TestTokenUsageInfoArithmetic:
    """Tests for TokenUsageInfo arithmetic operations."""

    def test_add_combines_known_values(self) -> None:
        """TokenUsageInfo addition combines numeric values."""
        r1 = {"usage": {"total_tokens": 10, "prompt_tokens": 6, "completion_tokens": 4}}
        r2 = {"usage": {"total_tokens": 20, "prompt_tokens": 3, "completion_tokens": 7}}
        info1 = pyine.evals.utils.parse_token_usage_from_response(r1)
        info2 = pyine.evals.utils.parse_token_usage_from_response(r2)
        summed = info1 + info2
        assert summed.total_tokens == 30
        assert summed.prompt_tokens == 9
        assert summed.completion_tokens == 11
        assert summed.reasoning_tokens == "unknown"

    def test_iadd_modifies_in_place(self) -> None:
        """TokenUsageInfo += modifies instance in place."""
        r1 = {"usage": {"total_tokens": 10, "prompt_tokens": 6, "completion_tokens": 4}}
        r2 = {"usage": {"total_tokens": 20, "prompt_tokens": 3, "completion_tokens": 7}}
        info1 = pyine.evals.utils.parse_token_usage_from_response(r1)
        info2 = pyine.evals.utils.parse_token_usage_from_response(r2)
        info1 += info2
        assert info1.total_tokens == 30
        assert info1.prompt_tokens == 9
        assert info1.completion_tokens == 11

    @pytest.mark.parametrize(
        "op",
        [
            pytest.param(lambda a, b: a + b, id="add"),
            pytest.param(lambda a, b: a.__iadd__(b), id="iadd"),
        ],
    )
    def test_add_raises_on_mismatched_unknown(
        self,
        op: typing.Callable[[typing.Any, typing.Any], typing.Any],
    ) -> None:
        """TokenUsageInfo addition raises when mixing known and unknown values."""
        known_output = {"usage": {"cached_tokens": 5, "total_tokens": 5}}
        unknown_output = {"usage": {"total_tokens": 10}}  # output_tokens omitted -> 'unknown'
        info_known = pyine.evals.utils.parse_token_usage_from_response(known_output)
        info_unknown = pyine.evals.utils.parse_token_usage_from_response(unknown_output)
        with pytest.raises(ValueError):
            op(info_known, info_unknown)


class TestCategoryWiseMetricsCallback:
    """Tests for CategoryWiseMetricsCallback."""

    def test_streaming_batch_accumulation(self, tmp_path: pathlib.Path) -> None:
        """CategoryWiseMetricsCallback accumulates batches and computes metrics."""
        categories = [["bugfix"], ["refactor"], ["bugfix", "refactor"]]
        callback = pyine.evals.utils.CategoryWiseMetricsCallback(
            data_sample_categories=categories,
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
                expected[f"{category}/loss"] = loss_sum / count
                expected[f"{category}/sample_count"] = count
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


@pytest.mark.integration
@pytest.mark.openai
@pytest.mark.skipif(
    tests.env_checks.OPENAI_API_KEY_MISSING or tests.env_checks.NETWORK_UNAVAILABLE,
    reason="OpenAI API key or network not available; cannot run OpenAI-backed evaluation.",
)
def test_token_usage_with_real_openai_generation(tmp_path: pathlib.Path) -> None:
    """Integration test for token usage parsing with real OpenAI API call."""
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


class TestSampleCategoryExtractionConfig:
    """Tests for SampleCategoryExtractionConfig."""

    def test_default_config(self) -> None:
        """Default config includes code_type and predict_type fields."""
        config = pyine.evals.utils.SampleCategoryExtractionConfig()
        assert pyine.evals.utils.SampleCategoryField.code_type in config.enabled_fields
        assert pyine.evals.utils.SampleCategoryField.predict_type in config.enabled_fields
        assert config.tag_prefixes is None

    def test_custom_config(self) -> None:
        """Custom config can specify fields and tag prefixes."""
        config = pyine.evals.utils.SampleCategoryExtractionConfig(
            enabled_fields=frozenset(
                {
                    pyine.evals.utils.SampleCategoryField.code_type,
                    pyine.evals.utils.SampleCategoryField.tags,
                }
            ),
            tag_prefixes=frozenset({"augment"}),
        )
        assert len(config.enabled_fields) == 2
        assert config.tag_prefixes == frozenset({"augment"})


class TestSampleCategoryExtractor:
    """Tests for SampleCategoryExtractor."""

    @pytest.mark.parametrize(
        "field,sample,expected",
        [
            pytest.param(
                pyine.evals.utils.SampleCategoryField.code_type,
                {"code_type": "original"},
                ["code_type/original"],
                id="code_type_string",
            ),
            pytest.param(
                pyine.evals.utils.SampleCategoryField.code_type,
                {"code_type": frozenset({"obfuscated", "hinted"})},
                ["code_type/hinted_obfuscated"],  # sorted alphabetically
                id="code_type_frozenset",
            ),
            pytest.param(
                pyine.evals.utils.SampleCategoryField.predict_type,
                {"predict_type": "program_output"},
                ["predict_type/program_output"],
                id="predict_type_string",
            ),
            pytest.param(
                pyine.evals.utils.SampleCategoryField.has_code_override,
                {"has_code_override": True},
                ["has_code_override/true"],
                id="has_code_override_true",
            ),
            pytest.param(
                pyine.evals.utils.SampleCategoryField.has_code_override,
                {"has_code_override": False},
                ["has_code_override/false"],
                id="has_code_override_false",
            ),
            pytest.param(
                pyine.evals.utils.SampleCategoryField.code_type,
                {},
                [],
                id="missing_field_returns_empty",
            ),
            pytest.param(
                pyine.evals.utils.SampleCategoryField.code_type,
                {"code_type": None},
                [],
                id="none_value_returns_empty",
            ),
        ],
    )
    def test_extract_single_field(
        self,
        field: pyine.evals.utils.SampleCategoryField,
        sample: dict[str, typing.Any],
        expected: list[str],
    ) -> None:
        """extract_categories returns correct categories for single field."""
        config = pyine.evals.utils.SampleCategoryExtractionConfig(
            enabled_fields=frozenset({field}),
        )
        extractor = pyine.evals.utils.SampleCategoryExtractor(config)
        categories = extractor.extract_categories(sample)
        assert categories == expected

    def test_extract_tags_all_prefixes(self) -> None:
        """extract_categories extracts all tag prefixes when no filter specified."""
        config = pyine.evals.utils.SampleCategoryExtractionConfig(
            enabled_fields=frozenset({pyine.evals.utils.SampleCategoryField.tags}),
        )
        extractor = pyine.evals.utils.SampleCategoryExtractor(config)
        sample = {"comma_separated_tags": "augment:obfuscated,subset:train"}
        categories = extractor.extract_categories(sample)
        assert "tags/augment/obfuscated" in categories
        assert "tags/subset/train" in categories
        assert len(categories) == 2

    def test_extract_tags_with_prefix_filter(self) -> None:
        """extract_categories filters tags by prefix when specified."""
        config = pyine.evals.utils.SampleCategoryExtractionConfig(
            enabled_fields=frozenset({pyine.evals.utils.SampleCategoryField.tags}),
            tag_prefixes=frozenset({"augment"}),
        )
        extractor = pyine.evals.utils.SampleCategoryExtractor(config)
        sample = {"comma_separated_tags": "augment:obfuscated,subset:train"}
        categories = extractor.extract_categories(sample)
        assert categories == ["tags/augment/obfuscated"]

    def test_extract_multiple_fields(self) -> None:
        """extract_categories extracts from multiple enabled fields."""
        config = pyine.evals.utils.SampleCategoryExtractionConfig(
            enabled_fields=frozenset(
                {
                    pyine.evals.utils.SampleCategoryField.code_type,
                    pyine.evals.utils.SampleCategoryField.predict_type,
                }
            ),
        )
        extractor = pyine.evals.utils.SampleCategoryExtractor(config)
        sample = {"code_type": "original", "predict_type": "program_output"}
        categories = extractor.extract_categories(sample)
        assert "code_type/original" in categories
        assert "predict_type/program_output" in categories
        assert len(categories) == 2

    def test_empty_tags_returns_empty(self) -> None:
        """extract_categories returns empty list for empty tags string."""
        config = pyine.evals.utils.SampleCategoryExtractionConfig(
            enabled_fields=frozenset({pyine.evals.utils.SampleCategoryField.tags}),
        )
        extractor = pyine.evals.utils.SampleCategoryExtractor(config)
        sample = {"comma_separated_tags": ""}
        categories = extractor.extract_categories(sample)
        assert categories == []

    def test_tags_without_prefix(self) -> None:
        """extract_categories handles tags with and without prefixes."""
        config = pyine.evals.utils.SampleCategoryExtractionConfig(
            enabled_fields=frozenset({pyine.evals.utils.SampleCategoryField.tags}),
        )
        extractor = pyine.evals.utils.SampleCategoryExtractor(config)
        sample = {"comma_separated_tags": "simple_tag,augment:complex"}
        categories = extractor.extract_categories(sample)
        assert "tags/simple_tag" in categories
        assert "tags/augment/complex" in categories


class TestExtractSampleCategoriesFromDataset:
    """Tests for extract_sample_categories_from_dataset function."""

    @pytest.mark.parametrize(
        "dataset,expected",
        [
            pytest.param(None, [], id="none_dataset"),
            pytest.param(
                type("MockDataset", (), {"column_names": ["input_ids", "labels"], "__len__": lambda s: 5})(),
                [],
                id="missing_sample_data_column",
            ),
        ],
    )
    def test_returns_empty_for_invalid_input(
        self,
        dataset: typing.Any,
        expected: list[typing.Any],
    ) -> None:
        """extract_sample_categories_from_dataset returns empty for invalid input."""
        result = pyine.evals.utils.extract_sample_categories_from_dataset(dataset)
        assert result == expected

    def test_sequence_of_dicts_format(self) -> None:
        """extract_sample_categories_from_dataset handles list[dict] format."""

        class MockDataset:
            column_names = ["sample_data"]

            def __len__(self) -> int:
                return 3

            def __getitem__(self, key: str) -> list[dict[str, str]]:
                if key == "sample_data":
                    return [
                        {"code_type": "original"},
                        {"code_type": "obfuscated"},
                        {"code_type": "hinted"},
                    ]
                raise KeyError(key)

        config = pyine.evals.utils.SampleCategoryExtractionConfig(
            enabled_fields=frozenset({pyine.evals.utils.SampleCategoryField.code_type}),
        )
        result = pyine.evals.utils.extract_sample_categories_from_dataset(MockDataset(), config=config)
        assert len(result) == 3
        assert result[0] == ["code_type/original"]
        assert result[1] == ["code_type/obfuscated"]
        assert result[2] == ["code_type/hinted"]

    def test_dict_column_format(self) -> None:
        """extract_sample_categories_from_dataset handles dict[str, list] format."""

        class MockDataset:
            column_names = ["sample_data"]

            def __len__(self) -> int:
                return 3

            def __getitem__(self, key: str) -> dict[str, list[str]]:
                if key == "sample_data":
                    return {"code_type": ["original", "obfuscated", "hinted"]}
                raise KeyError(key)

        config = pyine.evals.utils.SampleCategoryExtractionConfig(
            enabled_fields=frozenset({pyine.evals.utils.SampleCategoryField.code_type}),
        )
        result = pyine.evals.utils.extract_sample_categories_from_dataset(MockDataset(), config=config)
        assert len(result) == 3
        assert result[0] == ["code_type/original"]
        assert result[1] == ["code_type/obfuscated"]
        assert result[2] == ["code_type/hinted"]

    def test_with_multiple_fields(self) -> None:
        """extract_sample_categories_from_dataset extracts multiple fields."""

        class MockDataset:
            column_names = ["sample_data"]

            def __len__(self) -> int:
                return 2

            def __getitem__(self, key: str) -> list[dict[str, str]]:
                if key == "sample_data":
                    return [
                        {"code_type": "original", "predict_type": "program_output"},
                        {"code_type": "obfuscated", "predict_type": "frame_variables"},
                    ]
                raise KeyError(key)

        config = pyine.evals.utils.SampleCategoryExtractionConfig(
            enabled_fields=frozenset(
                {
                    pyine.evals.utils.SampleCategoryField.code_type,
                    pyine.evals.utils.SampleCategoryField.predict_type,
                }
            ),
        )
        result = pyine.evals.utils.extract_sample_categories_from_dataset(MockDataset(), config=config)
        assert len(result) == 2
        assert "code_type/original" in result[0]
        assert "predict_type/program_output" in result[0]
        assert "code_type/obfuscated" in result[1]
        assert "predict_type/frame_variables" in result[1]
