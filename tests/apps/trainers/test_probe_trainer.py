"""Unit tests for the probe training loop with mocked LLM."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import datasets
import pytest
import torch

from pyine.probes import ProbeConfig, build_probe
from pyine.probes.collection import ProbeCollection
from pyine.probes.extraction import ActivationExtractor

# ---------------------------------------------------------------------------
# Small mock LLM for testing
# ---------------------------------------------------------------------------


class MockTransformerBlock(torch.nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class SmallMockLLM(torch.nn.Module):
    """Tiny model with N transformer blocks, hidden_dim=64.

    Mimics the interface needed by ActivationExtractor and probe_train().
    """

    def __init__(self, n_layers: int = 2, hidden_dim: int = 64) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            num_hidden_layers=n_layers,
            hidden_size=hidden_dim,
        )
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([MockTransformerBlock(hidden_dim) for _ in range(n_layers)])
        self.embed = torch.nn.Embedding(1000, hidden_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        h = self.embed(input_ids)
        for layer in self.model.layers:
            h = layer(h)
        return h


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm() -> SmallMockLLM:
    model = SmallMockLLM(n_layers=2, hidden_dim=64)
    model.eval()
    model.requires_grad_(False)
    return model


@pytest.fixture
def probe_configs() -> list[ProbeConfig]:
    return [
        ProbeConfig(name="mean_L0", architecture="mean", layer=0, learning_rate=1e-2),
        ProbeConfig(name="max_L1", architecture="max", layer=1, learning_rate=1e-2),
    ]


@pytest.fixture
def probe_collection(probe_configs: list[ProbeConfig]) -> ProbeCollection:
    return ProbeCollection(probe_configs, hidden_dim=64)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestProbeTrainUnit:
    """Unit tests for the probe training components with mocked LLM."""

    def test_train_step_reduces_loss(
        self,
        mock_llm: SmallMockLLM,
        probe_collection: ProbeCollection,
    ) -> None:
        """After a few steps, train loss is lower than initial loss."""
        extractor = ActivationExtractor(mock_llm, target_layers=[0, 1])
        optimizer = torch.optim.AdamW(probe_collection.get_parameter_groups())
        loss_fn = torch.nn.BCEWithLogitsLoss()

        batch_size, seq_len = 8, 16
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        labels = torch.randint(0, 2, (batch_size,), dtype=torch.float)

        # Initial loss
        with torch.no_grad():
            mock_llm(input_ids=input_ids, attention_mask=attention_mask)
        activations = extractor.get_activations()
        logits = probe_collection(activations, attention_mask)
        initial_loss = sum(loss_fn(l.squeeze(-1), labels) for l in logits.values()).item()

        # Train for 30 steps
        for _ in range(30):
            with torch.no_grad():
                mock_llm(input_ids=input_ids, attention_mask=attention_mask)
            activations = extractor.get_activations()
            logits = probe_collection(activations, attention_mask)
            total_loss = sum(loss_fn(l.squeeze(-1), labels) for l in logits.values())
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        final_loss = total_loss.item()
        assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"

        extractor.remove_hooks()

    def test_validation_produces_metrics(
        self,
        probe_configs: list[ProbeConfig],
    ) -> None:
        """validate_probes() returns loss and AUROC per probe."""
        from accelerate import Accelerator

        from pyine.apps.trainers.probe_trainer import validate_probes

        accelerator = Accelerator()
        device = accelerator.device

        # Create fresh model + collection on correct device
        model = SmallMockLLM(n_layers=2, hidden_dim=64).to(device)
        model.eval()
        model.requires_grad_(False)
        coll = ProbeCollection(probe_configs, hidden_dim=64).to(device)

        extractor = ActivationExtractor(model, target_layers=[0, 1])
        loss_fn = torch.nn.BCEWithLogitsLoss()

        # Build a small dataloader
        batch_size, seq_len = 8, 16
        ds = datasets.Dataset.from_dict(
            {
                "input_ids": torch.randint(0, 100, (20, seq_len)).tolist(),
                "attention_mask": torch.ones(20, seq_len, dtype=torch.long).tolist(),
                "labels": ([0] * 10 + [1] * 10),
            }
        )
        ds.set_format("torch")
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size)

        coll_prepared = accelerator.prepare(coll)
        loader = accelerator.prepare(loader)

        metrics = validate_probes(
            coll_prepared,
            model,
            extractor,
            loader,
            loss_fn,
            global_step=0,
            accelerator=accelerator,
            runtime=None,
        )

        assert len(metrics) > 0
        for name, m in metrics.items():
            assert "loss" in m
            assert "auroc" in m
            assert m["loss"] >= 0.0

        extractor.remove_hooks()

    def test_auroc_guard_single_class(
        self,
        probe_configs: list[ProbeConfig],
    ) -> None:
        """validate_probes() returns NaN AUROC for single-class valid set."""
        import math

        from accelerate import Accelerator

        from pyine.apps.trainers.probe_trainer import validate_probes

        accelerator = Accelerator()
        device = accelerator.device

        model = SmallMockLLM(n_layers=2, hidden_dim=64).to(device)
        model.eval()
        model.requires_grad_(False)
        coll = ProbeCollection(probe_configs, hidden_dim=64).to(device)

        extractor = ActivationExtractor(model, target_layers=[0, 1])
        loss_fn = torch.nn.BCEWithLogitsLoss()

        # All labels are 0 -> single-class
        batch_size, seq_len = 4, 16
        ds = datasets.Dataset.from_dict(
            {
                "input_ids": torch.randint(0, 100, (8, seq_len)).tolist(),
                "attention_mask": torch.ones(8, seq_len, dtype=torch.long).tolist(),
                "labels": [0] * 8,
            }
        )
        ds.set_format("torch")
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size)

        coll_prepared = accelerator.prepare(coll)
        loader_prepared = accelerator.prepare(loader)

        metrics = validate_probes(
            coll_prepared,
            model,
            extractor,
            loader_prepared,
            loss_fn,
            global_step=0,
            accelerator=accelerator,
            runtime=None,
        )

        for name, m in metrics.items():
            assert math.isnan(m["auroc"]), f"Expected NaN AUROC for {name}, got {m['auroc']}"

        extractor.remove_hooks()

    def test_save_probe_checkpoints(
        self,
        tmp_path: Path,
        probe_collection: ProbeCollection,
    ) -> None:
        """save_probe_checkpoints() writes state_dict + config JSON per probe."""
        from accelerate import Accelerator

        from pyine.apps.trainers.probe_trainer import save_probe_checkpoints

        accelerator = Accelerator()
        probe_collection_prepared = accelerator.prepare(probe_collection)

        runtime = MagicMock()
        runtime.output_dir = str(tmp_path)

        config = MagicMock()

        output_dir = save_probe_checkpoints(
            probe_collection_prepared,
            config,
            runtime,
            accelerator,
        )

        assert output_dir is not None
        for name in probe_collection.probes:
            probe_dir = output_dir / name
            assert (probe_dir / "probe_state_dict.pt").exists()
            assert (probe_dir / "probe_config.json").exists()

    def test_probe_checkpoints_loadable(
        self,
        tmp_path: Path,
        probe_collection: ProbeCollection,
        probe_configs: list[ProbeConfig],
    ) -> None:
        """Saved probe can be reconstructed from config JSON + state_dict."""
        import json

        from accelerate import Accelerator

        from pyine.apps.trainers.probe_trainer import save_probe_checkpoints

        accelerator = Accelerator()
        probe_collection_prepared = accelerator.prepare(probe_collection)

        runtime = MagicMock()
        runtime.output_dir = str(tmp_path)
        config = MagicMock()

        output_dir = save_probe_checkpoints(
            probe_collection_prepared,
            config,
            runtime,
            accelerator,
        )

        # Reload each probe
        for name in probe_collection.probes:
            probe_dir = output_dir / name
            config_json = json.loads((probe_dir / "probe_config.json").read_text())
            loaded_config = ProbeConfig(**config_json)
            loaded_probe = build_probe(loaded_config)
            state_dict = torch.load(probe_dir / "probe_state_dict.pt", weights_only=True)
            loaded_probe.load_state_dict(state_dict)

            # Check shapes match
            original_params = dict(probe_collection.probes[name].named_parameters())
            loaded_params = dict(loaded_probe.named_parameters())
            assert set(original_params.keys()) == set(loaded_params.keys())


class TestDatasetValidation:
    """Tests for dataset validation and tokenization helpers."""

    def test_dataset_validation_catches_bad_labels(self) -> None:
        """validate_probe_dataset() raises on non-binary labels."""
        from pyine.apps.trainers.probe_trainer import validate_probe_dataset

        ds = datasets.Dataset.from_dict(
            {
                "messages": [
                    [{"role": "user", "content": "hi"}],
                    [{"role": "user", "content": "bye"}],
                ],
                "label": [0, 2],  # 2 is invalid
            }
        )
        with pytest.raises(ValueError, match="binary labels"):
            validate_probe_dataset(ds, "messages", "label")

    def test_dataset_validation_catches_missing_columns(self) -> None:
        """validate_probe_dataset() raises on missing text/label columns."""
        from pyine.apps.trainers.probe_trainer import validate_probe_dataset

        ds = datasets.Dataset.from_dict({"something": [1, 2]})
        with pytest.raises(ValueError, match="text_field"):
            validate_probe_dataset(ds, "messages", "label")

    def test_tokenization_chat_template_guard(self) -> None:
        """tokenize_for_probes() raises if tokenizer has no chat template and text_field='messages'."""
        from pyine.apps.trainers.probe_trainer import tokenize_for_probes

        tokenizer = MagicMock()
        tokenizer.chat_template = None

        examples = {
            "messages": [
                [{"role": "user", "content": "hello"}],
            ],
            "label": [1],
        }
        with pytest.raises(ValueError, match="chat template"):
            tokenize_for_probes(examples, tokenizer, 512, "messages", "label")
