"""Tests for metrics_collector measurement helpers."""

from __future__ import annotations

import typing

import pytest
import torch

from pyine.apps.guardrail_metrics.metrics_collector import (
    measure_flops,
    measure_latency,
    measure_memory,
)


class SimpleLinear(torch.nn.Module):
    """Minimal model for testing measurement helpers."""

    def __init__(self, in_features: int = 32, out_features: int = 2) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _simple_forward(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor] | tuple[torch.Tensor, ...],
) -> typing.Any:
    if isinstance(inputs, dict):
        return model(inputs["x"])
    return model(inputs[0])


@pytest.fixture
def simple_model() -> SimpleLinear:
    model = SimpleLinear(in_features=32, out_features=2)
    model.eval()
    return model


@pytest.fixture
def simple_input() -> dict[str, torch.Tensor]:
    return {"x": torch.randn(4, 32)}


class TestMeasureFlops:
    def test_returns_positive_int(
        self,
        simple_model: SimpleLinear,
        simple_input: dict[str, torch.Tensor],
    ) -> None:
        flops = measure_flops(simple_model, simple_input, _simple_forward)
        assert isinstance(flops, int)
        assert flops > 0

    def test_deterministic(
        self,
        simple_model: SimpleLinear,
        simple_input: dict[str, torch.Tensor],
    ) -> None:
        flops1 = measure_flops(simple_model, simple_input, _simple_forward)
        flops2 = measure_flops(simple_model, simple_input, _simple_forward)
        assert flops1 == flops2

    def test_asserts_eval_mode(self) -> None:
        model = SimpleLinear()
        model.train()
        inputs = {"x": torch.randn(4, 32)}
        with pytest.raises(AssertionError, match="eval mode"):
            measure_flops(model, inputs, _simple_forward)


class TestMeasureMemory:
    def test_returns_param_info(
        self,
        simple_model: SimpleLinear,
        simple_input: dict[str, torch.Tensor],
    ) -> None:
        result = measure_memory(
            simple_model,
            simple_input,
            _simple_forward,
            torch.device("cpu"),
        )
        assert "param_count" in result
        assert "param_memory_bytes" in result
        assert "peak_activation_memory_bytes" in result
        assert result["param_count"] > 0
        assert result["param_memory_bytes"] > 0

    def test_cpu_peak_memory_is_none(
        self,
        simple_model: SimpleLinear,
        simple_input: dict[str, torch.Tensor],
    ) -> None:
        """On CPU, peak activation memory is not measurable."""
        result = measure_memory(
            simple_model,
            simple_input,
            _simple_forward,
            torch.device("cpu"),
        )
        assert result["peak_activation_memory_bytes"] is None

    def test_param_count_matches_manual(
        self,
        simple_model: SimpleLinear,
        simple_input: dict[str, torch.Tensor],
    ) -> None:
        result = measure_memory(
            simple_model,
            simple_input,
            _simple_forward,
            torch.device("cpu"),
        )
        expected = sum(p.numel() for p in simple_model.parameters())
        assert result["param_count"] == expected

    def test_asserts_eval_mode(self) -> None:
        model = SimpleLinear()
        model.train()
        inputs = {"x": torch.randn(4, 32)}
        with pytest.raises(AssertionError, match="eval mode"):
            measure_memory(model, inputs, _simple_forward, torch.device("cpu"))


class TestMeasureLatency:
    def test_returns_all_stats(
        self,
        simple_model: SimpleLinear,
        simple_input: dict[str, torch.Tensor],
    ) -> None:
        result = measure_latency(
            simple_model,
            simple_input,
            _simple_forward,
            num_warmup=2,
            num_iterations=5,
            device=torch.device("cpu"),
        )
        expected_keys = {"mean_ms", "std_ms", "median_ms", "p90_ms", "p95_ms", "p99_ms", "min_ms", "max_ms"}
        assert set(result.keys()) == expected_keys

    def test_all_values_positive(
        self,
        simple_model: SimpleLinear,
        simple_input: dict[str, torch.Tensor],
    ) -> None:
        result = measure_latency(
            simple_model,
            simple_input,
            _simple_forward,
            num_warmup=2,
            num_iterations=10,
            device=torch.device("cpu"),
        )
        for key, value in result.items():
            if key == "std_ms":
                assert value >= 0, f"{key} should be non-negative"
            else:
                assert value > 0, f"{key} should be positive"

    def test_min_le_mean_le_max(
        self,
        simple_model: SimpleLinear,
        simple_input: dict[str, torch.Tensor],
    ) -> None:
        result = measure_latency(
            simple_model,
            simple_input,
            _simple_forward,
            num_warmup=2,
            num_iterations=10,
            device=torch.device("cpu"),
        )
        assert result["min_ms"] <= result["mean_ms"] <= result["max_ms"]

    def test_single_iteration_zero_std(
        self,
        simple_model: SimpleLinear,
        simple_input: dict[str, torch.Tensor],
    ) -> None:
        result = measure_latency(
            simple_model,
            simple_input,
            _simple_forward,
            num_warmup=1,
            num_iterations=1,
            device=torch.device("cpu"),
        )
        assert result["std_ms"] == 0.0

    def test_asserts_eval_mode(self) -> None:
        model = SimpleLinear()
        model.train()
        inputs = {"x": torch.randn(4, 32)}
        with pytest.raises(AssertionError, match="eval mode"):
            measure_latency(model, inputs, _simple_forward, 1, 1, torch.device("cpu"))
