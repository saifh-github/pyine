"""Tests for ActivationExtractor."""

from __future__ import annotations

import types

import pytest
import torch

import pyine.guardrails.probes.extraction


class MockTransformerBlock(torch.nn.Module):
    """Minimal transformer block that returns a fixed-shape tensor."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        # A simple linear so forward works
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MockLayers(torch.nn.Module):
    """Wrapper to expose layers as ``model.model.layers[i]``."""

    def __init__(self, n_layers: int, hidden_dim: int) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([MockTransformerBlock(hidden_dim) for _ in range(n_layers)])


class MockModel(torch.nn.Module):
    """Mock model with ``model.model.layers[i]`` and ``model.config``."""

    def __init__(self, n_layers: int = 4, hidden_dim: int = 32) -> None:
        super().__init__()
        self.model = MockLayers(n_layers, hidden_dim)
        self.config = types.SimpleNamespace(
            num_hidden_layers=n_layers,
            hidden_size=hidden_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_state = x
        for layer in self.model.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class TestActivationExtractor:
    def test_captures_specified_layers(self) -> None:
        model = MockModel(n_layers=4, hidden_dim=32)
        extractor = pyine.guardrails.probes.extraction.ActivationExtractor(model, target_layers=[0, 2])

        input_tensor = torch.randn(2, 8, 32)
        model(input_tensor)
        acts = extractor.get_activations()

        assert set(acts.keys()) == {0, 2}
        extractor.remove_hooks()

    def test_get_activations_clears_cache(self) -> None:
        model = MockModel(n_layers=4, hidden_dim=32)
        extractor = pyine.guardrails.probes.extraction.ActivationExtractor(model, target_layers=[0])

        input_tensor = torch.randn(2, 8, 32)
        model(input_tensor)
        _ = extractor.get_activations()

        # Second call should return empty dict
        assert extractor.get_activations() == {}
        extractor.remove_hooks()

    def test_activation_shape(self) -> None:
        hidden_dim = 32
        model = MockModel(n_layers=4, hidden_dim=hidden_dim)
        extractor = pyine.guardrails.probes.extraction.ActivationExtractor(model, target_layers=[1])

        batch, seq_len = 3, 10
        input_tensor = torch.randn(batch, seq_len, hidden_dim)
        model(input_tensor)
        acts = extractor.get_activations()

        assert acts[1].shape == (batch, seq_len, hidden_dim)
        extractor.remove_hooks()

    def test_activation_dtype_casting(self) -> None:
        model = MockModel(n_layers=4, hidden_dim=32)
        extractor = pyine.guardrails.probes.extraction.ActivationExtractor(
            model, target_layers=[0], activation_dtype=torch.float16
        )

        input_tensor = torch.randn(2, 8, 32)
        model(input_tensor)
        acts = extractor.get_activations()

        assert acts[0].dtype == torch.float16
        extractor.remove_hooks()

    def test_layer_out_of_range_raises(self) -> None:
        model = MockModel(n_layers=4, hidden_dim=32)
        with pytest.raises(ValueError, match="out of range"):
            pyine.guardrails.probes.extraction.ActivationExtractor(model, target_layers=[5])

    def test_layer_negative_raises(self) -> None:
        model = MockModel(n_layers=4, hidden_dim=32)
        with pytest.raises(ValueError, match="out of range"):
            pyine.guardrails.probes.extraction.ActivationExtractor(model, target_layers=[-1])

    def test_normalize_layer_output_tensor(self) -> None:
        tensor = torch.randn(2, 8, 32)
        result = pyine.guardrails.probes.extraction.ActivationExtractor._normalize_layer_output(tensor, 0)
        assert result is tensor

    def test_normalize_layer_output_tuple(self) -> None:
        tensor = torch.randn(2, 8, 32)
        result = pyine.guardrails.probes.extraction.ActivationExtractor._normalize_layer_output((tensor, None), 0)
        assert result is tensor

    def test_normalize_layer_output_base_model_output(self) -> None:
        tensor = torch.randn(2, 8, 32)
        output = types.SimpleNamespace(last_hidden_state=tensor)
        result = pyine.guardrails.probes.extraction.ActivationExtractor._normalize_layer_output(output, 0)
        assert result is tensor

    def test_normalize_layer_output_invalid_shape_raises(self) -> None:
        tensor = torch.randn(2, 32)  # 2D, not 3D
        with pytest.raises(ValueError, match="Expected 3D"):
            pyine.guardrails.probes.extraction.ActivationExtractor._normalize_layer_output(tensor, 0)

    def test_remove_hooks_cleanup(self) -> None:
        model = MockModel(n_layers=4, hidden_dim=32)
        extractor = pyine.guardrails.probes.extraction.ActivationExtractor(model, target_layers=[0, 1, 2])

        assert len(extractor._hooks) == 3
        extractor.remove_hooks()
        assert len(extractor._hooks) == 0

        # After removing hooks, forward pass should not capture activations
        input_tensor = torch.randn(2, 8, 32)
        model(input_tensor)
        assert extractor.get_activations() == {}

    def test_resolve_layer_fallback(self) -> None:
        """_resolve_layer tries alternative paths when model.model.layers fails."""

        # Create model with GPT-2-style path: transformer.h.{i}
        class GPT2StyleModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.transformer = torch.nn.Module()
                self.transformer.h = torch.nn.ModuleList([MockTransformerBlock(32) for _ in range(4)])
                self.config = types.SimpleNamespace(num_hidden_layers=4)

        model = GPT2StyleModel()
        layer = pyine.guardrails.probes.extraction.ActivationExtractor._resolve_layer(model, 2)
        assert isinstance(layer, MockTransformerBlock)

    def test_resolve_layer_unknown_raises(self) -> None:
        class UnknownModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = types.SimpleNamespace(num_hidden_layers=4)

        model = UnknownModel()
        with pytest.raises(ValueError, match="Cannot resolve"):
            pyine.guardrails.probes.extraction.ActivationExtractor._resolve_layer(model, 0)

    def test_activations_are_detached(self) -> None:
        model = MockModel(n_layers=4, hidden_dim=32)
        extractor = pyine.guardrails.probes.extraction.ActivationExtractor(model, target_layers=[0])

        input_tensor = torch.randn(2, 8, 32, requires_grad=True)
        model(input_tensor)
        acts = extractor.get_activations()

        assert not acts[0].requires_grad
        extractor.remove_hooks()
