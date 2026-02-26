"""Tests for pyine.evals.correctness.scorers."""

from __future__ import annotations

import typing

import pytest
import torch
import transformers

import pyine.evals.correctness.scorers as correctness_scorers
import pyine.evals.correctness.types as correctness_types
import pyine.probes.base


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


class _MockProbe(pyine.probes.base.BaseProbe):
    """Minimal probe that returns random logits."""

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
        # simulate passing through layers (hooks won't capture this without real forward)
        return type("Output", (), {"last_hidden_state": hidden_states})()


@pytest.mark.slow
class TestProbeScorer:
    def test_produces_correct_number_of_scores(self) -> None:
        import pyine.probes.extraction

        hidden_dim = 32
        model = _MockModel(hidden_dim=hidden_dim, num_layers=4)
        probe_config = pyine.probes.base.ProbeConfig(
            name="test_probe",
            architecture="mean_pool",
            layer=1,
            hidden_dim=hidden_dim,
        )
        probe = _MockProbe(probe_config)
        tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
        extractor = pyine.probes.extraction.ActivationExtractor(model, [1])
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
        extractor.remove_hooks()

    def test_get_metadata_returns_dict(self) -> None:
        import pyine.probes.extraction

        hidden_dim = 32
        model = _MockModel(hidden_dim=hidden_dim)
        probe_config = pyine.probes.base.ProbeConfig(
            name="test_probe",
            architecture="mean_pool",
            layer=0,
            hidden_dim=hidden_dim,
            replica_idx=1,
            base_name="mean_pool_L0",
        )
        probe = _MockProbe(probe_config)
        tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
        extractor = pyine.probes.extraction.ActivationExtractor(model, [0])
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

    def test_no_verification_cost(self) -> None:
        import pyine.probes.extraction

        hidden_dim = 32
        model = _MockModel(hidden_dim=hidden_dim)
        probe_config = pyine.probes.base.ProbeConfig(
            name="test_probe", architecture="mean_pool", layer=0, hidden_dim=hidden_dim
        )
        probe = _MockProbe(probe_config)
        tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
        extractor = pyine.probes.extraction.ActivationExtractor(model, [0])
        scorer = correctness_scorers.ProbeScorer(
            probe=probe,
            probe_config=probe_config,
            model=model,
            tokenizer=tokenizer,
            extractor=extractor,
            max_seq_length=32,
            text_field="model_output",
        )
        assert scorer.get_verification_cost_unit() is None
        extractor.remove_hooks()


@pytest.mark.slow
class TestLLMClassifierScorer:
    def test_produces_correct_number_of_scores(self) -> None:
        model = transformers.AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=2)
        tokenizer = transformers.AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        scorer = correctness_scorers.LLMClassifierScorer(
            model=model,
            tokenizer=tokenizer,
            batch_size=4,
            max_seq_length=64,
            text_field="model_output",
        )
        records = [_make_record(f"s{idx}", label=(idx % 2 == 0)) for idx in range(6)]
        result = scorer.score_records(records)
        assert len(result.scores) == 6
        assert all(0.0 <= score <= 1.0 for score in result.scores)

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

    def test_no_verification_cost(self) -> None:
        model = transformers.AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=2)
        tokenizer = transformers.AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        scorer = correctness_scorers.LLMClassifierScorer(
            model=model,
            tokenizer=tokenizer,
            max_seq_length=64,
            text_field="model_output",
        )
        assert scorer.get_verification_cost_unit() is None
