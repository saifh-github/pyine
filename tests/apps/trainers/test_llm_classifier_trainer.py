"""Tests for the LLM classifier training loop, tokenization, and metrics."""

from __future__ import annotations

import typing
from unittest.mock import MagicMock

if typing.TYPE_CHECKING:
    import pathlib

import datasets
import numpy as np
import pytest
import torch
import transformers

import pyine.apps.trainers.llm_classifier_trainer as llm_trainer

# ---------------------------------------------------------------------------
# Helpers for building synthetic datasets
# ---------------------------------------------------------------------------


def _make_synthetic_dataset(
    n: int = 20,
    text_prefix: str = "hello world this is a test sentence number ",
) -> datasets.Dataset:
    """Create a synthetic HF dataset with text, label, sample_id, code_type."""
    return datasets.Dataset.from_dict(
        {
            "text": [f"{text_prefix}{i}" for i in range(n)],
            "label": [i % 2 for i in range(n)],
            "sample_id": [f"sample_{i}" for i in range(n)],
            "code_type": ["original" if i % 3 != 2 else "misleading" for i in range(n)],
        }
    )


def _make_code_type_mappings() -> tuple[dict[str, int], dict[int, str]]:
    code_type_to_id = {"original": 0, "misleading": 1}
    id_to_code_type = {0: "original", 1: "misleading"}
    return code_type_to_id, id_to_code_type


# ---------------------------------------------------------------------------
# TestTokenizeForClassification
# ---------------------------------------------------------------------------


class TestTokenizeForClassification:
    """Tests for the tokenization function.

    Uses a small local tokenizer (prajjwal1/bert-tiny) to avoid large downloads.
    """

    @pytest.fixture
    def tokenizer(self) -> transformers.PreTrainedTokenizerBase:
        return transformers.AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

    @pytest.fixture
    def dataset(self) -> datasets.Dataset:
        return _make_synthetic_dataset()

    def test_output_has_required_columns(
        self,
        tokenizer: transformers.PreTrainedTokenizerBase,
        dataset: datasets.Dataset,
    ) -> None:
        result = llm_trainer._tokenize_for_classification(dataset, tokenizer, max_seq_length=128)
        assert "input_ids" in result.column_names
        assert "attention_mask" in result.column_names
        assert "labels" in result.column_names

    def test_labels_preserved(
        self,
        tokenizer: transformers.PreTrainedTokenizerBase,
        dataset: datasets.Dataset,
    ) -> None:
        result = llm_trainer._tokenize_for_classification(dataset, tokenizer, max_seq_length=128)
        assert list(result["labels"]) == list(dataset["label"])

    def test_code_type_id_when_mapping_provided(
        self,
        tokenizer: transformers.PreTrainedTokenizerBase,
        dataset: datasets.Dataset,
    ) -> None:
        code_type_to_id, _ = _make_code_type_mappings()
        result = llm_trainer._tokenize_for_classification(
            dataset,
            tokenizer,
            max_seq_length=128,
            code_type_to_id=code_type_to_id,
        )
        assert "code_type_id" in result.column_names
        expected = [code_type_to_id[ct] for ct in dataset["code_type"]]
        assert list(result["code_type_id"]) == expected

    def test_code_type_id_absent_when_no_mapping(
        self,
        tokenizer: transformers.PreTrainedTokenizerBase,
        dataset: datasets.Dataset,
    ) -> None:
        result = llm_trainer._tokenize_for_classification(dataset, tokenizer, max_seq_length=128)
        assert "code_type_id" not in result.column_names

    def test_truncation_applied(self, tokenizer: transformers.PreTrainedTokenizerBase) -> None:
        long_text = "word " * 500
        ds = datasets.Dataset.from_dict(
            {
                "text": [long_text],
                "label": [1],
                "sample_id": ["s0"],
                "code_type": ["original"],
            }
        )
        result = llm_trainer._tokenize_for_classification(ds, tokenizer, max_seq_length=32)
        assert len(result["input_ids"][0]) <= 32

    def test_special_tokens_added(self, tokenizer: transformers.PreTrainedTokenizerBase) -> None:
        ds = datasets.Dataset.from_dict(
            {
                "text": ["hello"],
                "label": [0],
                "sample_id": ["s0"],
                "code_type": ["original"],
            }
        )
        result = llm_trainer._tokenize_for_classification(ds, tokenizer, max_seq_length=128)
        input_ids = result["input_ids"][0]
        # BERT tokenizer adds [CLS]=101 at start and [SEP]=102 at end
        assert input_ids[0] == tokenizer.cls_token_id
        assert input_ids[-1] == tokenizer.sep_token_id

    def test_raw_columns_removed(
        self,
        tokenizer: transformers.PreTrainedTokenizerBase,
        dataset: datasets.Dataset,
    ) -> None:
        result = llm_trainer._tokenize_for_classification(dataset, tokenizer, max_seq_length=128)
        assert "text" not in result.column_names
        assert "sample_id" not in result.column_names
        assert "code_type" not in result.column_names

    def test_no_set_format_torch(
        self,
        tokenizer: transformers.PreTrainedTokenizerBase,
        dataset: datasets.Dataset,
    ) -> None:
        result = llm_trainer._tokenize_for_classification(dataset, tokenizer, max_seq_length=128)
        # Default format is None (arrow), not "torch"
        assert result.format["type"] is None


# ---------------------------------------------------------------------------
# TestBuildComputeMetrics
# ---------------------------------------------------------------------------


class TestBuildComputeMetrics:
    """Tests for the compute_metrics builder using synthetic data."""

    def _make_eval_pred(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        inputs: dict[str, np.ndarray] | None = None,
    ) -> transformers.EvalPrediction:
        return transformers.EvalPrediction(
            predictions=logits,
            label_ids=labels,
            inputs=inputs,
        )

    def test_returns_accuracy(self) -> None:
        fn = llm_trainer.build_compute_metrics()
        logits = np.array([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]])
        labels = np.array([0, 1, 0, 1])
        result = fn(self._make_eval_pred(logits, labels))
        assert "accuracy" in result
        assert result["accuracy"] == 1.0

    def test_returns_auroc(self) -> None:
        fn = llm_trainer.build_compute_metrics()
        logits = np.array([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]])
        labels = np.array([0, 1, 0, 1])
        result = fn(self._make_eval_pred(logits, labels))
        assert "auroc" in result
        assert result["auroc"] == 1.0

    def test_returns_f1(self) -> None:
        fn = llm_trainer.build_compute_metrics()
        logits = np.array([[2.0, 0.0], [0.0, 2.0]])
        labels = np.array([0, 1])
        result = fn(self._make_eval_pred(logits, labels))
        assert "f1" in result
        assert "precision" in result
        assert "recall" in result

    def test_auroc_nan_for_single_class(self) -> None:
        fn = llm_trainer.build_compute_metrics()
        logits = np.array([[2.0, 0.0], [1.5, 0.5], [1.0, 1.0]])
        labels = np.array([0, 0, 0])  # all same class
        result = fn(self._make_eval_pred(logits, labels))
        assert np.isnan(result["auroc"])

    def test_handles_tuple_logits(self) -> None:
        fn = llm_trainer.build_compute_metrics()
        logits = np.array([[2.0, 0.0], [0.0, 2.0]])
        labels = np.array([0, 1])
        # Simulate tuple predictions (logits, hidden_states)
        tuple_preds = (logits, np.zeros((2, 10)))
        result = fn(self._make_eval_pred(tuple_preds, labels))
        assert result["accuracy"] == 1.0

    def test_per_code_type_metrics_when_enabled(self) -> None:
        _, id_to_code_type = _make_code_type_mappings()
        fn = llm_trainer.build_compute_metrics(id_to_code_type=id_to_code_type)
        logits = np.array(
            [
                [2.0, 0.0],
                [0.0, 2.0],
                [2.0, 0.0],
                [0.0, 2.0],
                [2.0, 0.0],
                [0.0, 2.0],
            ]
        )
        labels = np.array([0, 1, 0, 1, 0, 1])
        code_type_ids = np.array([0, 0, 0, 1, 1, 1])
        result = fn(
            self._make_eval_pred(
                logits,
                labels,
                inputs={"code_type_id": code_type_ids},
            )
        )
        assert "accuracy/code_type/original" in result
        assert "accuracy/code_type/misleading" in result

    def test_per_code_type_reads_inputs_dict(self) -> None:
        """Per-code-type metrics correctly read eval_pred.inputs['code_type_id']."""
        _, id_to_code_type = _make_code_type_mappings()
        fn = llm_trainer.build_compute_metrics(id_to_code_type=id_to_code_type)
        logits = np.array(
            [
                [2.0, 0.0],
                [0.0, 2.0],
                [2.0, 0.0],
                [0.0, 2.0],
            ]
        )
        labels = np.array([0, 1, 0, 1])
        code_type_ids = np.array([0, 0, 1, 1])
        result = fn(
            self._make_eval_pred(
                logits,
                labels,
                inputs={"code_type_id": code_type_ids},
            )
        )
        assert result["accuracy/code_type/original"] == 1.0
        assert result["accuracy/code_type/misleading"] == 1.0

    def test_per_code_type_skips_small_groups(self) -> None:
        _, id_to_code_type = _make_code_type_mappings()
        fn = llm_trainer.build_compute_metrics(id_to_code_type=id_to_code_type)
        logits = np.array(
            [
                [2.0, 0.0],
                [0.0, 2.0],
                [2.0, 0.0],
            ]
        )
        labels = np.array([0, 1, 0])
        # code_type 1 ("misleading") has only 1 sample → should be skipped
        code_type_ids = np.array([0, 0, 1])
        result = fn(
            self._make_eval_pred(
                logits,
                labels,
                inputs={"code_type_id": code_type_ids},
            )
        )
        assert "accuracy/code_type/original" in result
        assert "accuracy/code_type/misleading" not in result

    def test_no_per_code_type_metrics_when_disabled(self) -> None:
        fn = llm_trainer.build_compute_metrics(id_to_code_type=None)
        logits = np.array([[2.0, 0.0], [0.0, 2.0]])
        labels = np.array([0, 1])
        result = fn(self._make_eval_pred(logits, labels))
        code_type_keys = [k for k in result if "code_type" in k]
        assert len(code_type_keys) == 0


# ---------------------------------------------------------------------------
# TestWeightedLossTrainer
# ---------------------------------------------------------------------------


class TestWeightedLossTrainer:
    """Tests for the WeightedLossTrainer compute_loss override.

    Tests the compute_loss method directly by calling it as an unbound method
    to avoid HF Trainer __init__ validation on mock models.
    """

    def _call_compute_loss(
        self,
        class_weights: torch.Tensor | None,
        inputs: dict[str, typing.Any],
        return_outputs: bool = False,
    ) -> typing.Any:
        """Invoke WeightedLossTrainer.compute_loss without full Trainer init."""
        model = MagicMock()
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        model.return_value = mock_output

        # Build a minimal instance bypassing __init__ (which validates model framework)
        trainer = llm_trainer.WeightedLossTrainer.__new__(llm_trainer.WeightedLossTrainer)
        trainer._class_weights = class_weights
        return trainer.compute_loss(model, inputs, return_outputs=return_outputs)

    def test_compute_loss_with_weights(self) -> None:
        class_weights = torch.tensor([1.0, 2.0])
        inputs = {
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
            "attention_mask": torch.tensor([[1, 1], [1, 1]]),
            "labels": torch.tensor([0, 1]),
        }
        loss = self._call_compute_loss(class_weights, inputs)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar

    def test_compute_loss_without_weights(self) -> None:
        inputs = {
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
            "attention_mask": torch.tensor([[1, 1], [1, 1]]),
            "labels": torch.tensor([0, 1]),
        }
        loss = self._call_compute_loss(None, inputs)
        assert isinstance(loss, torch.Tensor)

    def test_compute_loss_return_outputs(self) -> None:
        inputs = {
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
            "attention_mask": torch.tensor([[1, 1], [1, 1]]),
            "labels": torch.tensor([0, 1]),
        }
        result = self._call_compute_loss(
            torch.tensor([1.0, 1.0]),
            inputs,
            return_outputs=True,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], torch.Tensor)


# ---------------------------------------------------------------------------
# TestLogClassDistribution
# ---------------------------------------------------------------------------


class TestLogClassDistribution:
    """Tests for the class distribution logging function."""

    def test_logs_both_splits(self) -> None:
        mock_logger = MagicMock()
        ds_dict = datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_dict({"label": [0, 0, 1, 1, 1]}),
                "valid": datasets.Dataset.from_dict({"label": [0, 1]}),
            }
        )
        llm_trainer._log_class_distribution(ds_dict, mock_logger)
        logged_messages = " ".join(str(call) for call in mock_logger.info.call_args_list)
        assert "train" in logged_messages
        assert "valid" in logged_messages

    def test_logs_correct_counts(self) -> None:
        mock_logger = MagicMock()
        ds_dict = datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_dict({"label": [0, 0, 0, 1]}),
            }
        )
        llm_trainer._log_class_distribution(ds_dict, mock_logger)
        assert mock_logger.info.call_count == 1
        call_args = mock_logger.info.call_args
        # _log_class_distribution uses:
        # log.info("%s split: %d samples (pos=%d, neg=%d, ...)", split, n, pos, neg, ratio)
        assert call_args[0][3] == 1  # n_pos
        assert call_args[0][4] == 3  # n_neg


# ---------------------------------------------------------------------------
# TestClassifierTrainUnit — slow tests that require model downloads
# ---------------------------------------------------------------------------


class TestClassifierTrainUnit:
    """Unit tests for the training loop with a small model.

    Uses prajjwal1/bert-tiny for fast CPU tests.
    """

    @pytest.fixture
    def debug_lmdb(self, tmp_path: pathlib.Path) -> pathlib.Path:
        """Create a debug LMDB for testing."""
        from pyine.guardrails.data.debug_dataset import create_debug_probe_lmdb

        return create_debug_probe_lmdb(tmp_path / "test_lmdb", n_train=40, n_eval_families=10)

    @pytest.fixture
    def minimal_config(self, debug_lmdb: pathlib.Path, tmp_path: pathlib.Path) -> dict:
        """Minimal config for a fast training run."""
        return {
            "base_model": "prajjwal1/bert-tiny",
            "datamodule_config": {
                "lmdb_path": str(debug_lmdb),
            },
            "training_args_config": {
                "output_dir": str(tmp_path / "output"),
                "num_train_epochs": 1,
                "per_device_train_batch_size": 4,
                "per_device_eval_batch_size": 4,
                "report_to": "none",
                "use_cpu": True,
                "eval_strategy": "epoch",
                "save_strategy": "no",
                "logging_steps": 1,
                "load_best_model_at_end": False,
            },
            "max_seq_length": 64,
            "log_per_code_type_metrics": False,
            "save_model": False,
        }

    @pytest.mark.slow
    def test_training_completes(self, minimal_config: dict, tmp_path: pathlib.Path) -> None:
        from pyine.apps.trainers.llm_classifier_trainer_configs import (
            LLMClassifierTrainerAppMainConfig,
        )

        cfg = LLMClassifierTrainerAppMainConfig(**minimal_config)
        trainer = llm_trainer.classifier_train(config=cfg, runtime=None)
        assert trainer is not None

    @pytest.mark.slow
    def test_model_saved_to_disk(self, minimal_config: dict, tmp_path: pathlib.Path) -> None:
        from pyine.apps.trainers.llm_classifier_trainer_configs import (
            LLMClassifierTrainerAppMainConfig,
        )

        minimal_config["save_model"] = True
        minimal_config["training_args_config"]["output_dir"] = str(tmp_path / "save_output")
        cfg = LLMClassifierTrainerAppMainConfig(**minimal_config)
        llm_trainer.classifier_train(config=cfg, runtime=None)
        output_dir = tmp_path / "save_output"
        assert (output_dir / "model.safetensors").exists() or (output_dir / "pytorch_model.bin").exists()

    @pytest.mark.slow
    def test_saved_model_loadable(self, minimal_config: dict, tmp_path: pathlib.Path) -> None:
        from pyine.apps.trainers.llm_classifier_trainer_configs import (
            LLMClassifierTrainerAppMainConfig,
        )

        minimal_config["save_model"] = True
        minimal_config["training_args_config"]["output_dir"] = str(tmp_path / "load_output")
        cfg = LLMClassifierTrainerAppMainConfig(**minimal_config)
        llm_trainer.classifier_train(config=cfg, runtime=None)

        loaded = transformers.AutoModelForSequenceClassification.from_pretrained(tmp_path / "load_output")
        assert loaded.config.num_labels == 2

    @pytest.mark.slow
    def test_metrics_computed_during_eval(self, minimal_config: dict, tmp_path: pathlib.Path) -> None:
        from pyine.apps.trainers.llm_classifier_trainer_configs import (
            LLMClassifierTrainerAppMainConfig,
        )

        cfg = LLMClassifierTrainerAppMainConfig(**minimal_config)
        trainer = llm_trainer.classifier_train(config=cfg, runtime=None)
        # Trainer should have logged eval metrics
        log_history = trainer.state.log_history
        eval_logs = [entry for entry in log_history if "eval_accuracy" in entry]
        assert len(eval_logs) >= 1

    @pytest.mark.slow
    def test_class_distribution_logged(self, minimal_config: dict, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        from pyine.apps.trainers.llm_classifier_trainer_configs import (
            LLMClassifierTrainerAppMainConfig,
        )

        cfg = LLMClassifierTrainerAppMainConfig(**minimal_config)
        with caplog.at_level(logging.INFO):
            llm_trainer.classifier_train(config=cfg, runtime=None)
        assert "train split" in caplog.text

    @pytest.mark.slow
    def test_balanced_class_weights(self, minimal_config: dict, tmp_path: pathlib.Path) -> None:
        from pyine.apps.trainers.llm_classifier_trainer_configs import (
            LLMClassifierTrainerAppMainConfig,
        )

        minimal_config["class_weight_mode"] = "balanced"
        cfg = LLMClassifierTrainerAppMainConfig(**minimal_config)
        trainer = llm_trainer.classifier_train(config=cfg, runtime=None)
        assert isinstance(trainer, llm_trainer.WeightedLossTrainer)


class TestClassifierTrainIntegration:
    """Integration tests (GPU, slow)."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_training_end_to_end_with_modernbert(self, tmp_path: pathlib.Path) -> None:
        """Full pipeline with ModernBERT-base on GPU."""
        from pyine.apps.trainers.llm_classifier_trainer_configs import (
            LLMClassifierTrainerAppMainConfig,
        )
        from pyine.guardrails.data.debug_dataset import create_debug_probe_lmdb

        lmdb_path = create_debug_probe_lmdb(tmp_path / "lmdb", n_train=40, n_eval_families=10)
        cfg = LLMClassifierTrainerAppMainConfig(
            base_model="answerdotai/ModernBERT-base",
            datamodule_config={"lmdb_path": str(lmdb_path)},
            training_args_config={
                "output_dir": str(tmp_path / "output"),
                "num_train_epochs": 1,
                "per_device_train_batch_size": 4,
                "per_device_eval_batch_size": 4,
                "report_to": "none",
                "eval_strategy": "epoch",
                "save_strategy": "no",
                "logging_steps": 1,
                "load_best_model_at_end": False,
            },
            max_seq_length=128,
            save_model=False,
        )
        trainer = llm_trainer.classifier_train(config=cfg, runtime=None)
        assert trainer is not None

    @pytest.mark.slow
    @pytest.mark.integration
    def test_training_with_lora(self, tmp_path: pathlib.Path) -> None:
        """Training with LoRA adapters produces valid checkpoints."""
        import peft

        from pyine.apps.trainers.llm_classifier_trainer_configs import (
            LLMClassifierTrainerAppMainConfig,
        )
        from pyine.guardrails.data.debug_dataset import create_debug_probe_lmdb

        lmdb_path = create_debug_probe_lmdb(tmp_path / "lmdb", n_train=40, n_eval_families=10)
        cfg = LLMClassifierTrainerAppMainConfig(
            base_model="prajjwal1/bert-tiny",
            datamodule_config={"lmdb_path": str(lmdb_path)},
            training_args_config={
                "output_dir": str(tmp_path / "output"),
                "num_train_epochs": 1,
                "per_device_train_batch_size": 4,
                "per_device_eval_batch_size": 4,
                "report_to": "none",
                "use_cpu": True,
                "eval_strategy": "epoch",
                "save_strategy": "no",
                "logging_steps": 1,
                "load_best_model_at_end": False,
            },
            max_seq_length=64,
            save_model=True,
            lora_config=peft.LoraConfig(
                r=4,
                lora_alpha=8,
                target_modules=["query", "value"],
            ),
        )
        trainer = llm_trainer.classifier_train(config=cfg, runtime=None)
        assert isinstance(trainer.model, peft.PeftModel)
