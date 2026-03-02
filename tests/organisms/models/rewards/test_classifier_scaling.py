"""Tests for the classifier correctness scaler."""

from __future__ import annotations

import math
import typing

import pytest
import torch
import transformers

import pyine.organisms.models.rewards.core.classifier_scaling as cs_mod
import pyine.organisms.models.rewards.core.configs as reward_configs
import pyine.organisms.models.rewards.core.manager as reward_manager
import pyine.organisms.models.rewards.core.types as reward_types
import pyine.organisms.models.rewards.terms
from tests.organisms.models.rewards.conftest import make_sample_data

if typing.TYPE_CHECKING:
    import pytest_mock


def _make_mock_model(
    mocker: pytest_mock.MockerFixture,
    label2id: dict[str, int] | None = None,
    num_labels: int = 2,
    positive_prob: float = 0.8,
) -> pytest_mock.MagicMock:
    """Build a mock AutoModelForSequenceClassification.

    Logits are set so that softmax (without temperature) gives ``positive_prob`` at index 1.
    """
    model = mocker.MagicMock()
    model.config = mocker.MagicMock()
    if label2id is not None:
        model.config.label2id = label2id
    else:
        del model.config.label2id
    model.config.num_labels = num_labels
    del model.config.max_position_embeddings  # not set by default; tests opt-in
    logits = torch.zeros(1, num_labels)
    if num_labels == 2 and 0.0 < positive_prob < 1.0:
        logit_val = math.log(positive_prob / (1 - positive_prob))
        logits[0, 1] = logit_val
    output = mocker.MagicMock()
    output.logits = logits
    model.return_value = output
    model.eval = mocker.MagicMock(return_value=model)
    model.requires_grad_ = mocker.MagicMock(return_value=model)
    model.to = mocker.MagicMock(return_value=model)
    return model


def _make_mock_tokenizer(
    mocker: pytest_mock.MockerFixture,
    has_chat_template: bool = False,
) -> pytest_mock.MagicMock:
    tokenizer = mocker.MagicMock(spec=transformers.PreTrainedTokenizerBase)
    if has_chat_template:
        tokenizer.chat_template = "template"
        tokenizer.apply_chat_template = mocker.MagicMock(return_value="formatted text")
    else:
        tokenizer.chat_template = None
    tokenizer.encode = mocker.MagicMock(return_value=list(range(10)))
    tokenizer.return_value = {
        "input_ids": torch.ones(1, 10, dtype=torch.long),
        "attention_mask": torch.ones(1, 10, dtype=torch.long),
    }
    return tokenizer


def _make_sample_context(
    *,
    prompt: str = "test prompt",
    model_output: str = "test output",
    comma_separated_tags: str = "",
    code_type: str = "original",
) -> reward_types.SampleContext:
    return reward_types.SampleContext(
        prompt=prompt,
        model_output=model_output,
        sample_data=make_sample_data(
            "test_sample",
            comma_separated_tags=comma_separated_tags,
            code_type=code_type,
        ),
    )


def _build_scaler_with_mocks(
    mocker: pytest_mock.MockerFixture,
    config_overrides: dict[str, typing.Any] | None = None,
    label2id: dict[str, int] | None = None,
    num_labels: int = 2,
    positive_prob: float = 0.8,
    has_chat_template: bool = False,
) -> cs_mod.CorrectnessClassifierScaler:
    """Build a CorrectnessClassifierScaler with mocked model loading."""
    defaults: dict[str, typing.Any] = {
        "checkpoint_path": "/fake/checkpoint",
        "device": "cpu",
    }
    if config_overrides:
        defaults.update(config_overrides)
    config = reward_configs.CorrectnessClassifierScalingConfig.model_validate(defaults)
    scaler = cs_mod.CorrectnessClassifierScaler(config)
    effective_label2id = label2id if label2id is not None else {"incorrect": 0, "correct": 1}
    mock_model = _make_mock_model(
        mocker, label2id=effective_label2id, num_labels=num_labels, positive_prob=positive_prob
    )
    mock_tokenizer = _make_mock_tokenizer(mocker, has_chat_template=has_chat_template)
    scaler._model = mock_model
    scaler._tokenizer = mock_tokenizer
    scaler._device = torch.device("cpu")
    scaler._add_special_tokens = not has_chat_template
    if effective_label2id and config.positive_label in effective_label2id:
        scaler._positive_class_idx = effective_label2id[config.positive_label]
    elif config.allow_positive_label_fallback:
        scaler._positive_class_idx = 1
    else:
        scaler._positive_class_idx = 1
    return scaler


class TestScalerBehavior:
    def test_skip_non_keyword_sample(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        scaler = _build_scaler_with_mocks(mocker, config_overrides={"only_for_keyword_samples": True})
        ctx = _make_sample_context(comma_separated_tags="")
        scaled, metrics = scaler.apply(aggregated_reward=1.0, sample_ctx=ctx)
        assert scaled == 1.0  # neutral_factor=1.0 by default
        assert metrics["correctness_classifier/factor"] == 1.0
        assert "correctness_classifier/classifier_score" not in metrics  # no inference

    def test_scale_keyword_sample(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        positive_prob = 0.75
        scaler = _build_scaler_with_mocks(mocker, positive_prob=positive_prob)
        ctx = _make_sample_context(comma_separated_tags="has_bias_keyword:1")
        scaled, metrics = scaler.apply(aggregated_reward=1.0, sample_ctx=ctx)
        assert abs(scaled - positive_prob) < 0.01
        assert abs(typing.cast("float", metrics["correctness_classifier/classifier_score"]) - positive_prob) < 0.01

    def test_temperature_softens_scores(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Higher temperature -> factor closer to 0.5; lower -> more extreme."""
        high_temp_scaler = _build_scaler_with_mocks(mocker, config_overrides={"temperature": 5.0}, positive_prob=0.9)
        low_temp_scaler = _build_scaler_with_mocks(mocker, config_overrides={"temperature": 0.1}, positive_prob=0.9)
        ctx = _make_sample_context(comma_separated_tags="has_bias_keyword:1")
        high_temp_scaled, _ = high_temp_scaler.apply(aggregated_reward=1.0, sample_ctx=ctx)
        low_temp_scaled, _ = low_temp_scaler.apply(aggregated_reward=1.0, sample_ctx=ctx)
        # high temp should be closer to 0.5, low temp closer to original prob
        assert abs(high_temp_scaled - 0.5) < abs(low_temp_scaled - 0.5)

    def test_temperature_default_matches_no_temp(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        positive_prob = 0.8
        scaler = _build_scaler_with_mocks(mocker, config_overrides={"temperature": 1.0}, positive_prob=positive_prob)
        ctx = _make_sample_context(comma_separated_tags="has_bias_keyword:1")
        scaled, _ = scaler.apply(aggregated_reward=1.0, sample_ctx=ctx)
        assert abs(scaled - positive_prob) < 0.01

    def test_factor_clamped_by_min(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        scaler = _build_scaler_with_mocks(mocker, config_overrides={"min_factor": 0.3}, positive_prob=0.1)
        ctx = _make_sample_context(comma_separated_tags="has_bias_keyword:1")
        scaled, metrics = scaler.apply(aggregated_reward=1.0, sample_ctx=ctx)
        assert abs(scaled - 0.3) < 0.01  # clamped to min_factor
        assert abs(typing.cast("float", metrics["correctness_classifier/factor"]) - 0.3) < 0.01

    def test_factor_clamped_by_max(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        scaler = _build_scaler_with_mocks(mocker, config_overrides={"max_factor": 0.5}, positive_prob=0.9)
        ctx = _make_sample_context(comma_separated_tags="has_bias_keyword:1")
        scaled, metrics = scaler.apply(aggregated_reward=1.0, sample_ctx=ctx)
        assert abs(scaled - 0.5) < 0.01  # clamped to max_factor
        assert abs(typing.cast("float", metrics["correctness_classifier/factor"]) - 0.5) < 0.01

    def test_zero_reward_stays_zero(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        scaler = _build_scaler_with_mocks(mocker, positive_prob=0.75)
        ctx = _make_sample_context(comma_separated_tags="has_bias_keyword:1")
        scaled, _ = scaler.apply(aggregated_reward=0.0, sample_ctx=ctx)
        assert scaled == 0.0

    def test_no_reward_flipping(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Keyword sample with should_flip_reward=True -> scaler does NOT flip, just scales."""
        positive_prob = 0.7
        scaler = _build_scaler_with_mocks(mocker, positive_prob=positive_prob)
        ctx = _make_sample_context(comma_separated_tags="has_bias_keyword:1")
        scaled, _ = scaler.apply(aggregated_reward=1.0, sample_ctx=ctx)
        assert abs(scaled - positive_prob) < 0.01  # scaled, not flipped

    def test_all_samples_scaled(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        scaler = _build_scaler_with_mocks(mocker, config_overrides={"only_for_keyword_samples": False})
        ctx = _make_sample_context(comma_separated_tags="")
        _, metrics = scaler.apply(aggregated_reward=1.0, sample_ctx=ctx)
        assert "correctness_classifier/classifier_score" in metrics


class TestModelLoading:
    def test_lazy_loading_without_reset(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        config = reward_configs.CorrectnessClassifierScalingConfig.model_validate(
            {"checkpoint_path": "/fake/checkpoint", "device": "cpu"}
        )
        scaler = cs_mod.CorrectnessClassifierScaler(config)
        assert scaler._model is None
        mock_model = _make_mock_model(mocker, label2id={"incorrect": 0, "correct": 1})
        mock_tokenizer = _make_mock_tokenizer(mocker)
        mock_model_load = mocker.patch.object(
            transformers.AutoModelForSequenceClassification,
            "from_pretrained",
            return_value=mock_model,
        )
        mock_tok_load = mocker.patch.object(
            transformers.AutoTokenizer,
            "from_pretrained",
            return_value=mock_tokenizer,
        )
        mocker.patch("pathlib.Path.is_dir", return_value=True)
        ctx = _make_sample_context(comma_separated_tags="has_bias_keyword:1")
        scaled, metrics = scaler.apply(aggregated_reward=1.0, sample_ctx=ctx)
        mock_model_load.assert_called_once()
        mock_tok_load.assert_called_once()
        assert "correctness_classifier/classifier_score" in metrics

    def test_reset_preloads_model(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        config = reward_configs.CorrectnessClassifierScalingConfig.model_validate(
            {"checkpoint_path": "/fake/checkpoint", "device": "cpu"}
        )
        scaler = cs_mod.CorrectnessClassifierScaler(config)
        assert scaler._model is None
        mock_model = _make_mock_model(mocker, label2id={"incorrect": 0, "correct": 1})
        mock_tokenizer = _make_mock_tokenizer(mocker)
        mock_model_load = mocker.patch.object(
            transformers.AutoModelForSequenceClassification,
            "from_pretrained",
            return_value=mock_model,
        )
        mock_tok_load = mocker.patch.object(
            transformers.AutoTokenizer,
            "from_pretrained",
            return_value=mock_tokenizer,
        )
        mocker.patch("pathlib.Path.is_dir", return_value=True)
        scaler.reset()
        mock_model_load.assert_called_once()
        mock_tok_load.assert_called_once()
        assert scaler._model is not None
        assert scaler._positive_class_idx == 1

    def test_invalid_checkpoint_path_raises(self) -> None:
        config = reward_configs.CorrectnessClassifierScalingConfig.model_validate(
            {"checkpoint_path": "/nonexistent/path/to/checkpoint", "device": "cpu"}
        )
        scaler = cs_mod.CorrectnessClassifierScaler(config)
        with pytest.raises(ValueError, match="not an existing directory"):
            scaler._ensure_model_loaded()


class TestLabelMapping:
    def test_label_mapping_from_checkpoint(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        scaler = _build_scaler_with_mocks(mocker, label2id={"incorrect": 0, "correct": 1})
        assert scaler._positive_class_idx == 1

    def test_missing_label2id_raises(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        config = reward_configs.CorrectnessClassifierScalingConfig.model_validate(
            {"checkpoint_path": "/fake/checkpoint", "device": "cpu", "allow_positive_label_fallback": False}
        )
        scaler = cs_mod.CorrectnessClassifierScaler(config)
        mock_model = _make_mock_model(mocker, label2id=None)
        mock_tokenizer = _make_mock_tokenizer(mocker)
        mocker.patch.object(
            transformers.AutoModelForSequenceClassification,
            "from_pretrained",
            return_value=mock_model,
        )
        mocker.patch.object(transformers.AutoTokenizer, "from_pretrained", return_value=mock_tokenizer)
        mocker.patch("pathlib.Path.is_dir", return_value=True)
        with pytest.raises(ValueError, match="cannot resolve positive_label"):
            scaler._ensure_model_loaded()

    def test_label_fallback_mode(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        config = reward_configs.CorrectnessClassifierScalingConfig.model_validate(
            {"checkpoint_path": "/fake/checkpoint", "device": "cpu", "allow_positive_label_fallback": True}
        )
        scaler = cs_mod.CorrectnessClassifierScaler(config)
        mock_model = _make_mock_model(mocker, label2id=None, num_labels=2)
        mock_tokenizer = _make_mock_tokenizer(mocker)
        mocker.patch.object(
            transformers.AutoModelForSequenceClassification,
            "from_pretrained",
            return_value=mock_model,
        )
        mocker.patch.object(transformers.AutoTokenizer, "from_pretrained", return_value=mock_tokenizer)
        mocker.patch("pathlib.Path.is_dir", return_value=True)
        scaler._ensure_model_loaded()
        assert scaler._positive_class_idx == 1

    def test_label_fallback_insufficient_labels(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        config = reward_configs.CorrectnessClassifierScalingConfig.model_validate(
            {"checkpoint_path": "/fake/checkpoint", "device": "cpu", "allow_positive_label_fallback": True}
        )
        scaler = cs_mod.CorrectnessClassifierScaler(config)
        mock_model = _make_mock_model(mocker, label2id=None, num_labels=1)
        mock_tokenizer = _make_mock_tokenizer(mocker)
        mocker.patch.object(
            transformers.AutoModelForSequenceClassification,
            "from_pretrained",
            return_value=mock_model,
        )
        mocker.patch.object(transformers.AutoTokenizer, "from_pretrained", return_value=mock_tokenizer)
        mocker.patch("pathlib.Path.is_dir", return_value=True)
        with pytest.raises(ValueError, match="num_labels=1"):
            scaler._ensure_model_loaded()

    def test_single_class_with_valid_label2id_still_raises(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """num_labels=1 is rejected even when label2id resolves the positive label."""
        config = reward_configs.CorrectnessClassifierScalingConfig.model_validate(
            {"checkpoint_path": "/fake/checkpoint", "device": "cpu"}
        )
        scaler = cs_mod.CorrectnessClassifierScaler(config)
        mock_model = _make_mock_model(mocker, label2id={"correct": 0}, num_labels=1)
        mock_tokenizer = _make_mock_tokenizer(mocker)
        mocker.patch.object(
            transformers.AutoModelForSequenceClassification,
            "from_pretrained",
            return_value=mock_model,
        )
        mocker.patch.object(transformers.AutoTokenizer, "from_pretrained", return_value=mock_tokenizer)
        mocker.patch("pathlib.Path.is_dir", return_value=True)
        with pytest.raises(ValueError, match="num_labels=1"):
            scaler._ensure_model_loaded()


class TestPromptShape:
    def test_strict_mode_raises_on_every_offending_sample(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        scaler = _build_scaler_with_mocks(mocker, config_overrides={"strict_single_turn": True})
        json_ctx = _make_sample_context(
            prompt='[{"role": "user", "content": "hello"}]',
            model_output="hi",
            comma_separated_tags="has_bias_keyword:1",
        )
        for _ in range(3):
            with pytest.raises(ValueError, match="serialized JSON messages"):
                scaler.apply(aggregated_reward=1.0, sample_ctx=json_ctx)

    def test_warn_mode_warns_once(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        scaler = _build_scaler_with_mocks(mocker, config_overrides={"strict_single_turn": False})
        json_ctx = _make_sample_context(
            prompt='[{"role": "user", "content": "hello"}]',
            model_output="hi",
            comma_separated_tags="has_bias_keyword:1",
        )
        mock_warn = mocker.patch.object(cs_mod.logger, "warning")
        scaler.apply(aggregated_reward=1.0, sample_ctx=json_ctx)
        scaler.apply(aggregated_reward=1.0, sample_ctx=json_ctx)
        mock_warn.assert_called_once()

    @pytest.mark.parametrize(
        "prompt",
        [
            '[ {"role": "user", "content": "hello"} ]',
            '[\n  {"role": "user", "content": "hello"}\n]',
            '  [{"role": "user", "content": "hello"}]',
        ],
        ids=["space-after-bracket", "newline-after-bracket", "leading-whitespace"],
    )
    def test_whitespace_padded_json(
        self,
        mocker: pytest_mock.MockerFixture,
        prompt: str,
    ) -> None:
        scaler = _build_scaler_with_mocks(mocker, config_overrides={"strict_single_turn": True})
        ctx = _make_sample_context(
            prompt=prompt,
            model_output="hi",
            comma_separated_tags="has_bias_keyword:1",
        )
        with pytest.raises(ValueError, match="serialized JSON messages"):
            scaler.apply(aggregated_reward=1.0, sample_ctx=ctx)

    def test_non_messages_json_not_detected(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        scaler = _build_scaler_with_mocks(mocker, config_overrides={"strict_single_turn": True})
        ctx = _make_sample_context(
            prompt='[{"data": 1}, {"data": 2}]',
            model_output="hi",
            comma_separated_tags="has_bias_keyword:1",
        )
        _, metrics = scaler.apply(aggregated_reward=1.0, sample_ctx=ctx)
        assert "correctness_classifier/classifier_score" in metrics


class TestTruncationAndDevice:
    def test_truncation_metric_detected(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        scaler = _build_scaler_with_mocks(mocker, config_overrides={"max_seq_length": 5})
        ctx = _make_sample_context(comma_separated_tags="has_bias_keyword:1")
        _, metrics = scaler.apply(aggregated_reward=1.0, sample_ctx=ctx)
        assert metrics["correctness_classifier/was_truncated"] == 1

    def test_truncation_metric_not_detected(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        scaler = _build_scaler_with_mocks(mocker, config_overrides={"max_seq_length": 100})
        ctx = _make_sample_context(comma_separated_tags="has_bias_keyword:1")
        _, metrics = scaler.apply(aggregated_reward=1.0, sample_ctx=ctx)
        assert metrics["correctness_classifier/was_truncated"] == 0

    def test_device_selection_explicit(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        scaler = _build_scaler_with_mocks(mocker, config_overrides={"device": "cpu"})
        assert scaler._device == torch.device("cpu")

    def test_device_selection_default(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        config = reward_configs.CorrectnessClassifierScalingConfig.model_validate(
            {"checkpoint_path": "/fake/checkpoint", "device": None}
        )
        scaler = cs_mod.CorrectnessClassifierScaler(config)
        mock_model = _make_mock_model(mocker, label2id={"incorrect": 0, "correct": 1})
        mock_tokenizer = _make_mock_tokenizer(mocker)
        mocker.patch.object(
            transformers.AutoModelForSequenceClassification,
            "from_pretrained",
            return_value=mock_model,
        )
        mocker.patch.object(transformers.AutoTokenizer, "from_pretrained", return_value=mock_tokenizer)
        mocker.patch("pathlib.Path.is_dir", return_value=True)
        mocker.patch("torch.cuda.is_available", return_value=False)
        mocker.patch("torch.backends.mps.is_available", return_value=False)
        scaler._ensure_model_loaded()
        assert scaler._device == torch.device("cpu")

    def test_max_seq_length_exceeds_tokenizer_limit_raises(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        config = reward_configs.CorrectnessClassifierScalingConfig.model_validate(
            {"checkpoint_path": "/fake/checkpoint", "device": "cpu", "max_seq_length": 10000}
        )
        scaler = cs_mod.CorrectnessClassifierScaler(config)
        mock_model = _make_mock_model(mocker, label2id={"incorrect": 0, "correct": 1})
        mock_tokenizer = _make_mock_tokenizer(mocker)
        mock_tokenizer.model_max_length = 512  # tokenizer supports less than configured
        mocker.patch.object(
            transformers.AutoModelForSequenceClassification,
            "from_pretrained",
            return_value=mock_model,
        )
        mocker.patch.object(transformers.AutoTokenizer, "from_pretrained", return_value=mock_tokenizer)
        mocker.patch("pathlib.Path.is_dir", return_value=True)
        with pytest.raises(ValueError, match="exceeds effective model limit"):
            scaler._ensure_model_loaded()
        assert scaler._model is None  # partial initialization prevented

    def test_max_seq_length_exceeds_position_embeddings_raises(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """max_position_embeddings is the stricter limit -> raises."""
        config = reward_configs.CorrectnessClassifierScalingConfig.model_validate(
            {"checkpoint_path": "/fake/checkpoint", "device": "cpu", "max_seq_length": 2048}
        )
        scaler = cs_mod.CorrectnessClassifierScaler(config)
        mock_model = _make_mock_model(mocker, label2id={"incorrect": 0, "correct": 1})
        mock_model.config.max_position_embeddings = 512  # model supports less
        mock_tokenizer = _make_mock_tokenizer(mocker)
        mock_tokenizer.model_max_length = 100000  # tokenizer is fine
        mocker.patch.object(
            transformers.AutoModelForSequenceClassification,
            "from_pretrained",
            return_value=mock_model,
        )
        mocker.patch.object(transformers.AutoTokenizer, "from_pretrained", return_value=mock_tokenizer)
        mocker.patch("pathlib.Path.is_dir", return_value=True)
        with pytest.raises(ValueError, match="exceeds effective model limit"):
            scaler._ensure_model_loaded()
        assert scaler._model is None  # partial initialization prevented


class TestNegativeRewards:
    def test_skip_negative_rewards_passes_through(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        scaler = _build_scaler_with_mocks(mocker, config_overrides={"skip_negative_rewards": True}, positive_prob=0.5)
        ctx = _make_sample_context(comma_separated_tags="has_bias_keyword:1")
        scaled, metrics = scaler.apply(aggregated_reward=-1.0, sample_ctx=ctx)
        assert scaled == -1.0  # unchanged
        assert metrics["correctness_classifier/factor"] == 1.0
        assert metrics["correctness_classifier/skipped_negative"] == 1

    def test_skip_negative_rewards_disabled(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        scaler = _build_scaler_with_mocks(mocker, config_overrides={"skip_negative_rewards": False}, positive_prob=0.5)
        ctx = _make_sample_context(comma_separated_tags="has_bias_keyword:1")
        scaled, metrics = scaler.apply(aggregated_reward=-1.0, sample_ctx=ctx)
        assert scaled > -1.0  # factor < 1.0 moves toward 0
        assert "correctness_classifier/skipped_negative" not in metrics


class TestConfigValidation:
    def test_min_factor_exceeds_max_factor_raises(self) -> None:
        with pytest.raises(ValueError, match="min_factor .* must be <= max_factor"):
            reward_configs.CorrectnessClassifierScalingConfig(
                checkpoint_path="/fake",
                min_factor=0.9,
                max_factor=0.5,
            )

    def test_max_factor_exceeds_one_raises(self) -> None:
        with pytest.raises(ValueError, match="max_factor .* must be <= 1.0"):
            reward_configs.CorrectnessClassifierScalingConfig(
                checkpoint_path="/fake",
                max_factor=1.5,
            )

    def test_neutral_factor_out_of_range_raises(self) -> None:
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            reward_configs.CorrectnessClassifierScalingConfig(
                checkpoint_path="/fake",
                neutral_factor=1.5,
            )
        with pytest.raises(pydantic.ValidationError):
            reward_configs.CorrectnessClassifierScalingConfig(
                checkpoint_path="/fake",
                neutral_factor=-0.1,
            )


class TestMetrics:
    def test_metrics_emitted(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        scaler = _build_scaler_with_mocks(mocker, config_overrides={"emit_metrics": True})
        ctx = _make_sample_context(comma_separated_tags="has_bias_keyword:1")
        _, metrics = scaler.apply(aggregated_reward=1.0, sample_ctx=ctx)
        assert "correctness_classifier/factor" in metrics
        assert "correctness_classifier/pre_scaling_reward" in metrics
        assert "correctness_classifier/classifier_score" in metrics
        assert "correctness_classifier/was_truncated" in metrics
        assert "correctness_classifier/temperature" in metrics

    def test_metrics_suppressed(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        scaler = _build_scaler_with_mocks(mocker, config_overrides={"emit_metrics": False})
        ctx = _make_sample_context(comma_separated_tags="has_bias_keyword:1")
        _, metrics = scaler.apply(aggregated_reward=1.0, sample_ctx=ctx)
        assert len(metrics) == 0


class TestManagerIntegration:
    def _make_manager_with_classifier_scaling(
        self,
        mocker: pytest_mock.MockerFixture,
        positive_prob: float = 0.75,
        scaler_config_overrides: dict[str, typing.Any] | None = None,
    ) -> reward_manager.RewardManager:
        """Build a RewardManager with classifier scaling, mocking the classifier model."""
        pyine.organisms.models.rewards.terms.ensure_builtin_terms_registered()
        scaler_defaults: dict[str, typing.Any] = {
            "checkpoint_path": "/fake/checkpoint",
            "device": "cpu",
            "only_for_keyword_samples": True,
        }
        if scaler_config_overrides:
            scaler_defaults.update(scaler_config_overrides)
        config = reward_configs.RewardManagerConfig(
            terms=[
                reward_configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                    weight=1.0,
                    params={"reward_if_present": 1.0, "reward_if_missing": 0.0},
                ),
            ],
            parsing=reward_configs.ParsingConfig(final_tag="final"),
            correctness_classifier_scaling=reward_configs.CorrectnessClassifierScalingConfig.model_validate(
                scaler_defaults
            ),
            logging=reward_configs.LoggingConfig(enabled=False),
        )
        manager = reward_manager.RewardManager(config)
        # inject mocked scaler internals
        assert manager._classifier_scaler is not None
        effective_label2id = {"incorrect": 0, "correct": 1}
        mock_model = _make_mock_model(mocker, label2id=effective_label2id, positive_prob=positive_prob)
        mock_tokenizer = _make_mock_tokenizer(mocker)
        manager._classifier_scaler._model = mock_model
        manager._classifier_scaler._tokenizer = mock_tokenizer
        manager._classifier_scaler._device = torch.device("cpu")
        manager._classifier_scaler._positive_class_idx = effective_label2id["correct"]
        manager._classifier_scaler._add_special_tokens = True
        return manager

    def test_manager_with_classifier_scaling(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        from tests.organisms.models.rewards.conftest import make_parsed_output

        positive_prob = 0.75
        manager = self._make_manager_with_classifier_scaling(mocker, positive_prob=positive_prob)
        ctx = reward_types.SampleContext(
            prompt="test",
            model_output="<final>answer</final>",
            sample_data=make_sample_data("s1", comma_separated_tags="has_bias_keyword:1"),
            parsed=make_parsed_output(raw="<final>answer</final>", final_answer="answer"),
        )
        output = manager.compute(ctx)
        # parseable_answer term gives 1.0, classifier scaling multiplies by ~0.75
        assert abs(output.total - positive_prob) < 0.05
        assert output.weighted_terms["parseable"] == 1.0  # preserved

    def test_classifier_and_verbosity_compose(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Both scalers enabled -> both applied sequentially."""
        from tests.organisms.models.rewards.conftest import make_parsed_output

        pyine.organisms.models.rewards.terms.ensure_builtin_terms_registered()
        positive_prob = 0.8
        config = reward_configs.RewardManagerConfig(
            terms=[
                reward_configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                    weight=1.0,
                    params={"reward_if_present": 1.0, "reward_if_missing": 0.0},
                ),
            ],
            parsing=reward_configs.ParsingConfig(
                final_tag="final",
                track_token_lengths=True,
                openai_tokenizer_model="gpt-4o",
            ),
            correctness_classifier_scaling=reward_configs.CorrectnessClassifierScalingConfig(
                checkpoint_path="/fake/checkpoint",
                device="cpu",
                only_for_keyword_samples=True,
            ),
            verbosity_scaling=reward_configs.VerbosityScalingConfig(
                enabled=True,
                mode="absolute",
                decay_type="linear",
                threshold_tokens=10,
                end_tokens=100,
                max_factor=1.0,
                min_factor=0.5,
            ),
            logging=reward_configs.LoggingConfig(enabled=False),
        )
        manager = reward_manager.RewardManager(config)
        # inject mocked classifier scaler internals
        assert manager._classifier_scaler is not None
        effective_label2id = {"incorrect": 0, "correct": 1}
        mock_model = _make_mock_model(mocker, label2id=effective_label2id, positive_prob=positive_prob)
        mock_tokenizer = _make_mock_tokenizer(mocker)
        manager._classifier_scaler._model = mock_model
        manager._classifier_scaler._tokenizer = mock_tokenizer
        manager._classifier_scaler._device = torch.device("cpu")
        manager._classifier_scaler._positive_class_idx = effective_label2id["correct"]
        manager._classifier_scaler._add_special_tokens = True
        # long output to trigger verbosity scaling
        long_output = "<final>answer</final>" + "x" * 500
        ctx = reward_types.SampleContext(
            prompt="test",
            model_output=long_output,
            sample_data=make_sample_data("s1", comma_separated_tags="has_bias_keyword:1"),
            parsed=make_parsed_output(raw=long_output, final_answer="answer"),
        )
        output = manager.compute(ctx)
        # both scalers applied: total < positive_prob (verbosity penalty on top)
        assert output.total < positive_prob
        # verbosity/pre_scaling_reward should be post-classifier value (~0.8)
        verbosity_pre = typing.cast("float", output.metrics["verbosity/pre_scaling_reward"])
        assert abs(verbosity_pre - positive_prob) < 0.05
        # correctness_classifier/pre_scaling_reward should be the raw aggregated value (1.0)
        classifier_pre = typing.cast("float", output.metrics["correctness_classifier/pre_scaling_reward"])
        assert abs(classifier_pre - 1.0) < 0.01
