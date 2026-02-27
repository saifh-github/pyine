"""Tests for pyine.evals.correctness.scorers."""

from __future__ import annotations

import typing

import pytest
import torch
import transformers

import pyine.evals.correctness.scorers as correctness_scorers
import pyine.evals.correctness.types as correctness_types
import pyine.guardrails.probes.base


def _make_record(
    sample_id: str = "s1",
    label: bool = True,
) -> correctness_types.EvalRecord:
    return correctness_types.EvalRecord(
        sample_id=sample_id,
        problem_id="p1",
        attempt_index=0,
        model_output="hello world",
        final_answer="42",
        expected_output="expected",
        label=label,
        code_type="original",
        tags=[],
        record={},
        difficulty_score=None,
    )


class _MockProbe(pyine.guardrails.probes.base.BaseProbe):
    """Minimal probe that returns random logits."""

    def __init__(self, config: pyine.guardrails.probes.base.ProbeConfig) -> None:
        super().__init__(config)
        self._dummy = torch.nn.Parameter(torch.zeros(1))  # pyright: ignore[reportUnknownMemberType]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        return torch.randn(batch_size, 1)  # pyright: ignore[reportUnknownMemberType]


class _MockModel(torch.nn.Module):
    """Minimal model that produces outputs with last_hidden_state."""

    def __init__(self, hidden_dim: int = 32, num_layers: int = 4) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.config = type("Config", (), {"num_hidden_layers": num_layers, "hidden_size": hidden_dim})()
        # create fake layers for ActivationExtractor resolution
        self.model = torch.nn.Module()
        layers = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.model.add_module("layers", layers)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: typing.Any,
    ) -> typing.Any:
        batch_size, seq_len = input_ids.shape
        hidden_dim = self.config.hidden_size
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)  # pyright: ignore[reportUnknownMemberType]
        for layer in self.model.layers:  # type: ignore[reportUnknownMemberType]
            hidden_states = layer(hidden_states)
        return type("Output", (), {"last_hidden_state": hidden_states})()


@pytest.mark.slow
class TestProbeScorer:
    def test_produces_correct_number_of_scores(self) -> None:
        import pyine.guardrails.probes.extraction

        hidden_dim = 32
        model = _MockModel(hidden_dim=hidden_dim, num_layers=4)
        probe_config = pyine.guardrails.probes.base.ProbeConfig(
            name="test_probe",
            architecture="mean_pool",
            layer=1,
            hidden_dim=hidden_dim,
        )
        probe = _MockProbe(probe_config)
        tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
        extractor = pyine.guardrails.probes.extraction.ActivationExtractor(model, [1])
        scorer = correctness_scorers.ProbeScorer(
            probe=probe,
            probe_config=probe_config,
            model=model,
            tokenizer=tokenizer,
            extractor=extractor,
            batch_size=2,
            max_seq_length=32,
            text_field="model_output",
        )
        records = [_make_record(f"s{idx}", label=(idx % 2 == 0)) for idx in range(5)]
        result = scorer.score_records(records)
        assert len(result.scores) == 5
        assert all(0.0 <= score <= 1.0 for score in result.scores)
        assert result.verification_costs is not None
        assert len(result.verification_costs) == 5
        assert all(cost >= 0.0 for cost in result.verification_costs)
        extractor.remove_hooks()

    def test_get_metadata_returns_dict(self) -> None:
        import pyine.guardrails.probes.extraction

        hidden_dim = 32
        model = _MockModel(hidden_dim=hidden_dim)
        probe_config = pyine.guardrails.probes.base.ProbeConfig(
            name="test_probe",
            architecture="mean_pool",
            layer=0,
            hidden_dim=hidden_dim,
            replica_idx=1,
            base_name="mean_pool_L0",
        )
        probe = _MockProbe(probe_config)
        tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
        extractor = pyine.guardrails.probes.extraction.ActivationExtractor(model, [0])
        scorer = correctness_scorers.ProbeScorer(
            probe=probe,
            probe_config=probe_config,
            model=model,
            tokenizer=tokenizer,
            extractor=extractor,
            max_seq_length=32,
            text_field="model_output",
        )
        metadata = scorer.get_metadata()
        assert metadata["name"] == "test_probe"
        assert metadata["architecture"] == "mean_pool"
        assert metadata["scorer_type"] == "probe"
        extractor.remove_hooks()

    def test_verification_cost_unit_is_flops(self) -> None:
        import pyine.guardrails.probes.extraction

        hidden_dim = 32
        model = _MockModel(hidden_dim=hidden_dim)
        probe_config = pyine.guardrails.probes.base.ProbeConfig(
            name="test_probe", architecture="mean_pool", layer=0, hidden_dim=hidden_dim
        )
        probe = _MockProbe(probe_config)
        tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
        extractor = pyine.guardrails.probes.extraction.ActivationExtractor(model, [0])
        scorer = correctness_scorers.ProbeScorer(
            probe=probe,
            probe_config=probe_config,
            model=model,
            tokenizer=tokenizer,
            extractor=extractor,
            max_seq_length=32,
            text_field="model_output",
        )
        assert scorer.get_verification_cost_unit() == "FLOPs"
        extractor.remove_hooks()


@pytest.mark.slow
class TestLLMClassifierScorer:
    def test_produces_correct_number_of_scores(self) -> None:
        model = transformers.AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=2)
        tokenizer = transformers.AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        scorer = correctness_scorers.LLMClassifierScorer(
            model=model,
            tokenizer=tokenizer,
            max_seq_length=64,
            text_field="model_output",
        )
        records = [_make_record(f"s{idx}", label=(idx % 2 == 0)) for idx in range(6)]
        result = scorer.score_records(records)
        assert len(result.scores) == 6
        assert all(0.0 <= score <= 1.0 for score in result.scores)
        assert result.verification_costs is not None
        assert len(result.verification_costs) == 6
        assert all(cost > 0.0 for cost in result.verification_costs)

    def test_get_metadata_returns_dict(self) -> None:
        model = transformers.AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=2)
        tokenizer = transformers.AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        scorer = correctness_scorers.LLMClassifierScorer(
            model=model,
            tokenizer=tokenizer,
            max_seq_length=64,
            text_field="model_output",
        )
        metadata = scorer.get_metadata()
        assert "model_name" in metadata
        assert metadata["scorer_type"] == "llm_classifier"

    def test_verification_cost_unit_is_flops(self) -> None:
        model = transformers.AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=2)
        tokenizer = transformers.AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        scorer = correctness_scorers.LLMClassifierScorer(
            model=model,
            tokenizer=tokenizer,
            max_seq_length=64,
            text_field="model_output",
        )
        assert scorer.get_verification_cost_unit() == "FLOPs"

    def test_longer_inputs_have_higher_costs(self) -> None:
        model = transformers.AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=2)
        tokenizer = transformers.AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        scorer = correctness_scorers.LLMClassifierScorer(
            model=model,
            tokenizer=tokenizer,
            max_seq_length=512,
            text_field="model_output",
        )
        short_record = _make_record("short")
        long_record = correctness_types.EvalRecord(
            sample_id="long",
            problem_id="p1",
            attempt_index=0,
            model_output="hello world " * 50,
            final_answer="42",
            expected_output="expected",
            label=True,
            code_type="original",
            tags=[],
            record={},
            difficulty_score=None,
        )
        result = scorer.score_records([short_record, long_record])
        assert result.verification_costs is not None
        assert result.verification_costs[1] > result.verification_costs[0]
