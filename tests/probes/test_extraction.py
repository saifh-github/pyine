"""Tests for ActivationExtractor."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from pyine.probes.extraction import ActivationExtractor


# ---------------------------------------------------------------------------
# Mock transformer model
# ---------------------------------------------------------------------------

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
        self.layers = torch.nn.ModuleList(
            [MockTransformerBlock(hidden_dim) for _ in range(n_layers)]
        )


class MockModel(torch.nn.Module):
    """Mock model with ``model.model.layers[i]`` and ``model.config``."""

    def __init__(self, n_layers: int = 4, hidden_dim: int = 32) -> None:
        super().__init__()
        self.model = MockLayers(n_layers, hidden_dim)
        self.config = SimpleNamespace(
            num_hidden_layers=n_layers,
            hidden_size=hidden_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.model.layers:
            h = layer(h)
        return h


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestActivationExtractor:
    """Tests for hook-based activation extraction."""

    def test_captures_specified_layers(self) -> None:
        """Activations are captured only for requested layer indices."""
        model = MockModel(n_layers=4, hidden_dim=32)
        extractor = ActivationExtractor(model, target_layers=[0, 2])

        x = torch.randn(2, 8, 32)
        model(x)
        acts = extractor.get_activations()

        assert set(acts.keys()) == {0, 2}
        extractor.remove_hooks()

    def test_get_activations_clears_cache(self) -> None:
        """After get_activations(), internal cache is empty."""
        model = MockModel(n_layers=4, hidden_dim=32)
        extractor = ActivationExtractor(model, target_layers=[0])

        x = torch.randn(2, 8, 32)
        model(x)
        _ = extractor.get_activations()

        # Second call should return empty dict
        assert extractor.get_activations() == {}
        extractor.remove_hooks()

    def test_activation_shape(self) -> None:
        """Captured activations have shape (batch, seq_len, hidden_dim)."""
        hidden_dim = 32
        model = MockModel(n_layers=4, hidden_dim=hidden_dim)
        extractor = ActivationExtractor(model, target_layers=[1])

        batch, seq_len = 3, 10
        x = torch.randn(batch, seq_len, hidden_dim)
        model(x)
        acts = extractor.get_activations()

        assert acts[1].shape == (batch, seq_len, hidden_dim)
        extractor.remove_hooks()

    def test_activation_dtype_casting(self) -> None:
        """activation_dtype casts activations to the specified dtype."""
        model = MockModel(n_layers=4, hidden_dim=32)
        extractor = ActivationExtractor(
            model, target_layers=[0], activation_dtype=torch.float16
        )

        x = torch.randn(2, 8, 32)
        model(x)
        acts = extractor.get_activations()

        assert acts[0].dtype == torch.float16
        extractor.remove_hooks()

    def test_layer_out_of_range_raises(self) -> None:
        """Requesting a layer >= num_hidden_layers raises ValueError."""
        model = MockModel(n_layers=4, hidden_dim=32)
        with pytest.raises(ValueError, match="out of range"):
            ActivationExtractor(model, target_layers=[5])

    def test_layer_negative_raises(self) -> None:
        """Requesting a negative layer index raises ValueError."""
        model = MockModel(n_layers=4, hidden_dim=32)
        with pytest.raises(ValueError, match="out of range"):
            ActivationExtractor(model, target_layers=[-1])

    def test_normalize_layer_output_tensor(self) -> None:
        """_normalize_layer_output handles plain tensor outputs."""
        t = torch.randn(2, 8, 32)
        result = ActivationExtractor._normalize_layer_output(t, 0)
        assert result is t

    def test_normalize_layer_output_tuple(self) -> None:
        """_normalize_layer_output handles tuple outputs."""
        t = torch.randn(2, 8, 32)
        result = ActivationExtractor._normalize_layer_output((t, None), 0)
        assert result is t

    def test_normalize_layer_output_base_model_output(self) -> None:
        """_normalize_layer_output handles BaseModelOutput-like objects."""
        t = torch.randn(2, 8, 32)
        output = SimpleNamespace(last_hidden_state=t)
        result = ActivationExtractor._normalize_layer_output(output, 0)
        assert result is t

    def test_normalize_layer_output_invalid_shape_raises(self) -> None:
        """_normalize_layer_output raises on non-3D tensors."""
        t = torch.randn(2, 32)  # 2D, not 3D
        with pytest.raises(ValueError, match="Expected 3D"):
            ActivationExtractor._normalize_layer_output(t, 0)

    def test_remove_hooks_cleanup(self) -> None:
        """remove_hooks() detaches all hooks from the model."""
        model = MockModel(n_layers=4, hidden_dim=32)
        extractor = ActivationExtractor(model, target_layers=[0, 1, 2])

        assert len(extractor._hooks) == 3
        extractor.remove_hooks()
        assert len(extractor._hooks) == 0

        # After removing hooks, forward pass should not capture activations
        x = torch.randn(2, 8, 32)
        model(x)
        assert extractor.get_activations() == {}

    def test_resolve_layer_fallback(self) -> None:
        """_resolve_layer tries alternative paths when model.model.layers fails."""

        # Create model with GPT-2-style path: transformer.h.{i}
        class GPT2StyleModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.transformer = torch.nn.Module()
                self.transformer.h = torch.nn.ModuleList(
                    [MockTransformerBlock(32) for _ in range(4)]
                )
                self.config = SimpleNamespace(num_hidden_layers=4)

        model = GPT2StyleModel()
        layer = ActivationExtractor._resolve_layer(model, 2)
        assert isinstance(layer, MockTransformerBlock)

    def test_resolve_layer_unknown_raises(self) -> None:
        """_resolve_layer raises ValueError for unknown model architectures."""

        class UnknownModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = SimpleNamespace(num_hidden_layers=4)

        model = UnknownModel()
        with pytest.raises(ValueError, match="Cannot resolve"):
            ActivationExtractor._resolve_layer(model, 0)

    def test_activations_are_detached(self) -> None:
        """Captured activations should not require grad (detached from graph)."""
        model = MockModel(n_layers=4, hidden_dim=32)
        extractor = ActivationExtractor(model, target_layers=[0])

        x = torch.randn(2, 8, 32, requires_grad=True)
        model(x)
        acts = extractor.get_activations()

        assert not acts[0].requires_grad
        extractor.remove_hooks()
