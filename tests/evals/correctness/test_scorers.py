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
    attempt_index: int = 0,
    model_output: str = "hello world",
) -> correctness_types.EvalRecord:
    return correctness_types.EvalRecord(
        sample_id=sample_id,
        problem_id="p1",
        attempt_index=attempt_index,
        model_output=model_output,
        final_answer="42",
        expected_output="expected",
        label=label,
        code_type="original",
        tags=[],
        record={"sample_id": sample_id, "prompt": f"prompt for {sample_id}"},
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


class _DeterministicProbe(pyine.guardrails.probes.base.BaseProbe):
    """Probe returning mean of hidden_states along seq dim -- deterministic."""

    def __init__(self, config: pyine.guardrails.probes.base.ProbeConfig) -> None:
        super().__init__(config)
        self._dummy = torch.nn.Parameter(torch.zeros(1))  # pyright: ignore[reportUnknownMemberType]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()  # (batch, seq, 1)
        masked = hidden_states * mask
        token_counts = mask.sum(dim=1).clamp(min=1)  # (batch, 1)
        return (masked.sum(dim=1) / token_counts).mean(dim=-1, keepdim=True)


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


class _FastTokenizer:
    """Minimal tokenizer used by fast tests (no network/model downloads)."""

    def __init__(self) -> None:
        self.seen_text_batches: list[list[str]] = []
        self.seen_add_special_tokens: list[bool] = []

    def __call__(
        self,
        texts: str | list[str],
        return_tensors: str = "pt",
        padding: bool = False,
        truncation: bool = True,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        assert truncation
        text_list = [texts] if isinstance(texts, str) else texts
        self.seen_text_batches.append(list(text_list))
        self.seen_add_special_tokens.append(add_special_tokens)
        token_counts = [
            max(1, min(len(text.split()), max_length if max_length is not None else len(text.split())))
            for text in text_list
        ]
        if padding:
            seq_len = max(token_counts)
        else:
            seq_len = token_counts[0]
        input_ids = torch.zeros((len(text_list), seq_len), dtype=torch.int64)
        attention_mask = torch.zeros_like(input_ids)
        for text_idx, token_count in enumerate(token_counts):
            attention_mask[text_idx, :token_count] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _FastBackboneModel(torch.nn.Module):
    """Small model that records input shapes and call count for the fake activation extractor."""

    def __init__(
        self,
        hidden_size: int = 8,
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self._dummy = torch.nn.Parameter(torch.zeros(1))  # pyright: ignore[reportUnknownMemberType]
        self.config = type("Config", (), {"hidden_size": hidden_size})()
        self.last_input_shape: tuple[int, int] = (1, 1)
        self.forward_call_count: int = 0
        self._cumulative_samples: int = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: typing.Any,
    ) -> typing.Any:
        del attention_mask, kwargs
        batch_size, seq_len = tuple(input_ids.shape)
        self.last_input_shape = (int(batch_size), int(seq_len))
        self.forward_call_count += 1
        self._cumulative_samples += int(batch_size)
        hidden_states = torch.zeros(batch_size, seq_len, self.config.hidden_size)  # pyright: ignore[reportUnknownMemberType]
        return type("Output", (), {"last_hidden_state": hidden_states})()


class _FastExtractor:
    """Fake activation extractor that returns deterministic, input-dependent per-layer tensors.

    Supports multiple layers. Each sample in a batch gets a distinct fill value based on its
    cumulative position across all forward calls, making probe scores distinguishable per record.
    """

    def __init__(
        self,
        model: _FastBackboneModel,
        layers: list[int] | int,
    ) -> None:
        self._model = model
        self._layers = [layers] if isinstance(layers, int) else layers
        self._batch_offset: int = 0

    def get_activations(
        self,
    ) -> dict[int, torch.Tensor]:
        batch_size, seq_len = self._model.last_input_shape
        result: dict[int, torch.Tensor] = {}
        for layer in self._layers:
            activations = torch.zeros(batch_size, seq_len, self._model.config.hidden_size)  # pyright: ignore[reportUnknownMemberType]
            for sample_idx in range(batch_size):
                activations[sample_idx] = float((sample_idx + 1 + self._batch_offset) * (layer + 1))
            result[layer] = activations
        self._batch_offset += batch_size
        return result

    def remove_hooks(self) -> None:
        pass


class _FastClassifierModel(torch.nn.Module):
    """Small binary classifier used by fast scorer tests."""

    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self._dummy = torch.nn.Parameter(torch.zeros(1))  # pyright: ignore[reportUnknownMemberType]
        self.config = type("Config", (), {"name_or_path": "fast-classifier"})()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: typing.Any,
    ) -> typing.Any:
        del input_ids, kwargs
        token_counts = attention_mask.to(dtype=torch.float32).sum(dim=1, keepdim=True)
        logits = torch.cat([-token_counts, token_counts], dim=1)  # higher token count => higher positive score
        return type("Output", (), {"logits": logits})()


class TestScorerAttemptMetadataFast:
    def test_probe_scorer_formats_full_messages_before_tokenization(self) -> None:
        model = _FastBackboneModel(hidden_size=8)
        probe_config = pyine.guardrails.probes.base.ProbeConfig(
            name="test_probe",
            architecture="mean_pool",
            layer=1,
            hidden_dim=8,
        )
        tokenizer = _FastTokenizer()
        scorer = correctness_scorers.ProbeScorer(
            probe=_MockProbe(probe_config),
            probe_config=probe_config,
            model=model,
            tokenizer=tokenizer,
            extractor=_FastExtractor(model, layers=1),  # type: ignore[arg-type]
            batch_size=2,
            max_seq_length=128,
            text_field="model_output",
        )

        scorer.score_records([_make_record(sample_id="short", model_output="answer text")])

        assert tokenizer.seen_text_batches == [["user: prompt for short\n\nassistant: answer text"]]
        assert tokenizer.seen_add_special_tokens == [True]

    def test_classifier_scorer_formats_full_messages_before_tokenization(self) -> None:
        tokenizer = _FastTokenizer()
        scorer = correctness_scorers.LLMClassifierScorer(
            model=_FastClassifierModel(),  # type: ignore[arg-type]
            tokenizer=tokenizer,  # type: ignore[arg-type]
            max_seq_length=128,
            text_field="model_output",
        )

        scorer.score_records([_make_record(sample_id="short", model_output="answer text")])

        assert tokenizer.seen_text_batches == [["user: prompt for short\n\nassistant: answer text"]]
        assert tokenizer.seen_add_special_tokens == [True]

    def test_probe_scorer_reports_attempt_metadata(self) -> None:
        model = _FastBackboneModel(hidden_size=8)
        probe_config = pyine.guardrails.probes.base.ProbeConfig(
            name="test_probe",
            architecture="mean_pool",
            layer=1,
            hidden_dim=8,
        )
        scorer = correctness_scorers.ProbeScorer(
            probe=_MockProbe(probe_config),
            probe_config=probe_config,
            model=model,
            tokenizer=_FastTokenizer(),
            extractor=_FastExtractor(model, layers=1),  # type: ignore[arg-type]
            batch_size=2,
            max_seq_length=128,
            text_field="model_output",
        )
        records = [
            _make_record(sample_id="short", model_output="hello"),
            _make_record(sample_id="long", model_output="hello this is a longer sequence"),
        ]
        result = scorer.score_records(records)
        assert result.attempt_metadata is not None
        assert set(result.attempt_metadata.keys()) == {("short", 0, 0), ("long", 0, 1)}
        assert (
            result.attempt_metadata[("short", 0, 0)]["input_token_count"]
            < result.attempt_metadata[("long", 0, 1)]["input_token_count"]
        )

    def test_classifier_scorer_reports_attempt_metadata(self) -> None:
        scorer = correctness_scorers.LLMClassifierScorer(
            model=_FastClassifierModel(),  # type: ignore[arg-type]
            tokenizer=_FastTokenizer(),  # type: ignore[arg-type]
            max_seq_length=128,
            text_field="model_output",
        )
        records = [
            _make_record(sample_id="short", model_output="hello"),
            _make_record(sample_id="long", model_output="hello this is a longer sequence"),
        ]
        result = scorer.score_records(records)
        assert result.attempt_metadata is not None
        assert set(result.attempt_metadata.keys()) == {("short", 0, 0), ("long", 0, 1)}
        assert (
            result.attempt_metadata[("short", 0, 0)]["input_token_count"]
            < result.attempt_metadata[("long", 0, 1)]["input_token_count"]
        )


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
        assert result.attempt_metadata is not None
        assert len(result.attempt_metadata) == 5
        assert all("input_token_count" in metadata for metadata in result.attempt_metadata.values())
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
        assert metadata["text_field"] == "model_output"
        assert metadata["input_formatting_mode"] in {"chat_template", "role_tagged_text"}
        assert "add_special_tokens" in metadata
        assert "truncation_side" in metadata
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
        assert result.attempt_metadata is not None
        assert len(result.attempt_metadata) == 6
        assert all("input_token_count" in metadata for metadata in result.attempt_metadata.values())

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
        assert metadata["text_field"] == "model_output"
        assert metadata["input_formatting_mode"] in {"chat_template", "role_tagged_text"}
        assert "add_special_tokens" in metadata
        assert "truncation_side" in metadata

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
            record={"sample_id": "long", "prompt": "prompt for long"},
            difficulty_score=None,
        )
        result = scorer.score_records([short_record, long_record])
        assert result.verification_costs is not None
        assert result.verification_costs[1] > result.verification_costs[0]
        assert result.attempt_metadata is not None
        assert (
            result.attempt_metadata[("short", 0, 0)]["input_token_count"]
            < result.attempt_metadata[("long", 0, 1)]["input_token_count"]
        )


class TestPrecomputedProbeScorer:
    @pytest.fixture()
    def probe_config(self) -> pyine.guardrails.probes.base.ProbeConfig:
        return pyine.guardrails.probes.base.ProbeConfig(
            name="test_probe",
            architecture="mean_pool",
            layer=1,
            hidden_dim=8,
            replica_idx=0,
            base_name="mean_pool_L1",
        )

    @pytest.fixture()
    def tokenizer_metadata(self) -> dict[str, typing.Any]:
        return {
            "text_field": "model_output",
            "input_construction": "eval_record_messages",
            "input_formatting_mode": "role_tagged_text",
            "tokenizer_has_chat_template": False,
            "add_special_tokens": True,
            "truncation_side": None,
        }

    def test_returns_correct_scores(
        self,
        probe_config: pyine.guardrails.probes.base.ProbeConfig,
        tokenizer_metadata: dict[str, typing.Any],
    ) -> None:
        samples_by_key = {
            ("s1", 0): correctness_scorers.PrecomputedSample(score=0.8, cost=100.0, token_count=10),
            ("s2", 0): correctness_scorers.PrecomputedSample(score=0.3, cost=200.0, token_count=20),
        }
        scorer = correctness_scorers.PrecomputedProbeScorer(
            samples_by_key=samples_by_key,
            probe_config=probe_config,
            tokenizer_metadata=tokenizer_metadata,
        )
        records = [_make_record("s1"), _make_record("s2")]
        result = scorer.score_records(records)
        assert result.scores == [0.8, 0.3]
        assert result.verification_costs == [100.0, 200.0]
        assert result.attempt_metadata is not None
        assert result.attempt_metadata[("s1", 0, 0)]["input_token_count"] == 10
        assert result.attempt_metadata[("s2", 0, 1)]["input_token_count"] == 20

    def test_handles_resampled_duplicates(
        self,
        probe_config: pyine.guardrails.probes.base.ProbeConfig,
        tokenizer_metadata: dict[str, typing.Any],
    ) -> None:
        samples_by_key = {
            ("s1", 0): correctness_scorers.PrecomputedSample(score=0.75, cost=150.0, token_count=15),
        }
        scorer = correctness_scorers.PrecomputedProbeScorer(
            samples_by_key=samples_by_key,
            probe_config=probe_config,
            tokenizer_metadata=tokenizer_metadata,
        )
        records = [_make_record("s1"), _make_record("s1")]
        result = scorer.score_records(records)
        assert result.scores == [0.75, 0.75]
        assert result.attempt_metadata is not None
        assert ("s1", 0, 0) in result.attempt_metadata
        assert ("s1", 0, 1) in result.attempt_metadata

    def test_missing_key_raises(
        self,
        probe_config: pyine.guardrails.probes.base.ProbeConfig,
        tokenizer_metadata: dict[str, typing.Any],
    ) -> None:
        scorer = correctness_scorers.PrecomputedProbeScorer(
            samples_by_key={},
            probe_config=probe_config,
            tokenizer_metadata=tokenizer_metadata,
        )
        with pytest.raises(KeyError, match="missing key"):
            scorer.score_records([_make_record("s1")])

    def test_multiple_calls_independent(
        self,
        probe_config: pyine.guardrails.probes.base.ProbeConfig,
        tokenizer_metadata: dict[str, typing.Any],
    ) -> None:
        samples_by_key = {
            ("s1", 0): correctness_scorers.PrecomputedSample(score=0.6, cost=10.0, token_count=5),
            ("s2", 0): correctness_scorers.PrecomputedSample(score=0.4, cost=20.0, token_count=10),
            ("s3", 0): correctness_scorers.PrecomputedSample(score=0.9, cost=30.0, token_count=15),
        }
        scorer = correctness_scorers.PrecomputedProbeScorer(
            samples_by_key=samples_by_key,
            probe_config=probe_config,
            tokenizer_metadata=tokenizer_metadata,
        )
        result1 = scorer.score_records([_make_record("s1"), _make_record("s2")])
        result2 = scorer.score_records([_make_record("s3")])
        assert result1.scores == [0.6, 0.4]
        assert result2.scores == [0.9]
        assert result1.attempt_metadata is not None
        assert result2.attempt_metadata is not None
        assert ("s1", 0, 0) in result1.attempt_metadata
        assert ("s2", 0, 1) in result1.attempt_metadata
        assert ("s3", 0, 0) in result2.attempt_metadata  # draw_index resets to 0

    def test_metadata_structure(
        self,
        probe_config: pyine.guardrails.probes.base.ProbeConfig,
        tokenizer_metadata: dict[str, typing.Any],
    ) -> None:
        scorer = correctness_scorers.PrecomputedProbeScorer(
            samples_by_key={},
            probe_config=probe_config,
            tokenizer_metadata=tokenizer_metadata,
        )
        metadata = scorer.get_metadata()
        assert metadata["name"] == "test_probe"
        assert metadata["architecture"] == "mean_pool"
        assert metadata["layer"] == 1
        assert metadata["replica_idx"] == 0
        assert metadata["base_name"] == "mean_pool_L1"
        assert metadata["scorer_type"] == "probe"
        assert metadata["text_field"] == "model_output"
        assert scorer.get_verification_cost_unit() == "FLOPs"


class TestPrecomputeProbeScores:
    @pytest.fixture()
    def hidden_size(self) -> int:
        return 8

    @pytest.fixture()
    def model(self, hidden_size: int) -> _FastBackboneModel:
        model = _FastBackboneModel(hidden_size=hidden_size)
        model.eval()
        return model

    @pytest.fixture()
    def tokenizer(self) -> _FastTokenizer:
        return _FastTokenizer()

    @pytest.fixture()
    def records(self) -> list[correctness_types.EvalRecord]:
        return [
            _make_record("s1", model_output="hello world"),
            _make_record("s2", model_output="foo bar baz"),
            _make_record("s3", model_output="a b c d e"),
        ]

    def _make_probes(
        self,
        hidden_size: int,
        layers: list[int],
    ) -> dict[str, tuple[pyine.guardrails.probes.base.BaseProbe, pyine.guardrails.probes.base.ProbeConfig]]:
        probes: dict[str, tuple[pyine.guardrails.probes.base.BaseProbe, pyine.guardrails.probes.base.ProbeConfig]] = {}
        for layer in layers:
            name = f"probe_L{layer}"
            config = pyine.guardrails.probes.base.ProbeConfig(
                name=name,
                architecture="mean_pool",
                layer=layer,
                hidden_dim=hidden_size,
                base_name=f"mean_pool_L{layer}",
            )
            probe = _DeterministicProbe(config)
            probe.eval()
            probes[name] = (probe, config)
        return probes

    def test_deduplicates_records(
        self,
        model: _FastBackboneModel,
        tokenizer: _FastTokenizer,
        hidden_size: int,
    ) -> None:
        records = [
            _make_record("s1", model_output="hello"),
            _make_record("s1", model_output="hello"),  # duplicate
            _make_record("s2", model_output="world"),
        ]
        probes = self._make_probes(hidden_size, layers=[1])
        extractor = _FastExtractor(model, layers=[1])
        result = correctness_scorers.precompute_probe_scores(
            records=records,
            probes=probes,
            model=model,
            tokenizer=tokenizer,  # type: ignore[arg-type]
            extractor=extractor,  # type: ignore[arg-type]
            max_seq_length=128,
            text_field="model_output",
        )
        assert model.forward_call_count == 2  # 2 unique records, batch_size=1
        # verify that scoring the full (non-deduplicated) list returns correct results
        scoring_result = result["probe_L1"].score_records(records)
        assert len(scoring_result.scores) == 3
        assert scoring_result.scores[0] == scoring_result.scores[1]  # duplicate key => same score

    def test_duplicate_key_with_different_text_raises(
        self,
        model: _FastBackboneModel,
        tokenizer: _FastTokenizer,
        hidden_size: int,
    ) -> None:
        records = [
            _make_record("s1", model_output="hello"),
            _make_record("s1", model_output="different text"),  # same key, different content
        ]
        probes = self._make_probes(hidden_size, layers=[1])
        extractor = _FastExtractor(model, layers=[1])
        with pytest.raises(AssertionError, match="duplicate key.*different.*model_output"):
            correctness_scorers.precompute_probe_scores(
                records=records,
                probes=probes,
                model=model,
                tokenizer=tokenizer,  # type: ignore[arg-type]
                extractor=extractor,  # type: ignore[arg-type]
                max_seq_length=128,
                text_field="model_output",
            )

    def test_empty_probes_raises(
        self,
        model: _FastBackboneModel,
        tokenizer: _FastTokenizer,
        records: list[correctness_types.EvalRecord],
    ) -> None:
        extractor = _FastExtractor(model, layers=[1])
        with pytest.raises(ValueError, match="empty"):
            correctness_scorers.precompute_probe_scores(
                records=records,
                probes={},
                model=model,
                tokenizer=tokenizer,  # type: ignore[arg-type]
                extractor=extractor,  # type: ignore[arg-type]
                max_seq_length=128,
                text_field="model_output",
            )

    def test_returns_scorer_per_probe(
        self,
        model: _FastBackboneModel,
        tokenizer: _FastTokenizer,
        records: list[correctness_types.EvalRecord],
        hidden_size: int,
    ) -> None:
        probes = self._make_probes(hidden_size, layers=[1, 2])
        extractor = _FastExtractor(model, layers=[1, 2])
        result = correctness_scorers.precompute_probe_scores(
            records=records,
            probes=probes,
            model=model,
            tokenizer=tokenizer,  # type: ignore[arg-type]
            extractor=extractor,  # type: ignore[arg-type]
            max_seq_length=128,
            text_field="model_output",
        )
        assert set(result.keys()) == {"probe_L1", "probe_L2"}
        for scorer in result.values():
            assert isinstance(scorer, correctness_scorers.PrecomputedProbeScorer)

    def test_raises_on_missing_extractor_layer(
        self,
        model: _FastBackboneModel,
        tokenizer: _FastTokenizer,
        records: list[correctness_types.EvalRecord],
        hidden_size: int,
    ) -> None:
        probes = self._make_probes(hidden_size, layers=[5])  # extractor won't provide layer 5
        extractor = _FastExtractor(model, layers=[1])
        with pytest.raises(ValueError, match="missing layers"):
            correctness_scorers.precompute_probe_scores(
                records=records,
                probes=probes,
                model=model,
                tokenizer=tokenizer,  # type: ignore[arg-type]
                extractor=extractor,  # type: ignore[arg-type]
                max_seq_length=128,
                text_field="model_output",
            )

    def test_batch_size_greater_than_one(
        self,
        model: _FastBackboneModel,
        tokenizer: _FastTokenizer,
        records: list[correctness_types.EvalRecord],
        hidden_size: int,
    ) -> None:
        """With batch_size=2 and 3 records, expect 2 forward calls (batch of 2 + batch of 1)."""
        probes = self._make_probes(hidden_size, layers=[1])
        extractor = _FastExtractor(model, layers=[1])
        result = correctness_scorers.precompute_probe_scores(
            records=records,
            probes=probes,
            model=model,
            tokenizer=tokenizer,  # type: ignore[arg-type]
            extractor=extractor,  # type: ignore[arg-type]
            max_seq_length=128,
            text_field="model_output",
            batch_size=2,
        )
        assert model.forward_call_count == 2  # ceil(3 / 2) = 2 batches
        scorer = result["probe_L1"]
        scoring_result = scorer.score_records(records)
        assert len(scoring_result.scores) == 3
        assert len(set(scoring_result.scores)) > 1

    def test_scores_match_individual_scorer(
        self,
        model: _FastBackboneModel,
        tokenizer: _FastTokenizer,
        records: list[correctness_types.EvalRecord],
        hidden_size: int,
    ) -> None:
        """Pre-computed scores must numerically match individual ProbeScorer results."""
        probes = self._make_probes(hidden_size, layers=[1])
        # run precomputation
        extractor_precompute = _FastExtractor(model, layers=[1])
        precomputed = correctness_scorers.precompute_probe_scores(
            records=records,
            probes=probes,
            model=model,
            tokenizer=tokenizer,  # type: ignore[arg-type]
            extractor=extractor_precompute,  # type: ignore[arg-type]
            max_seq_length=128,
            text_field="model_output",
        )
        precomputed_result = precomputed["probe_L1"].score_records(records)
        # run individual ProbeScorer (fresh model + extractor for same activations)
        model_individual = _FastBackboneModel(hidden_size=hidden_size)
        extractor_individual = _FastExtractor(model_individual, layers=1)
        probe, probe_config = probes["probe_L1"]
        individual_scorer = correctness_scorers.ProbeScorer(
            probe=probe,
            probe_config=probe_config,
            model=model_individual,
            tokenizer=tokenizer,  # type: ignore[arg-type]
            extractor=extractor_individual,  # type: ignore[arg-type]
            max_seq_length=128,
            text_field="model_output",
        )
        individual_result = individual_scorer.score_records(records)
        # scores must match exactly (same activations, same probe, deterministic)
        assert len(precomputed_result.scores) == len(individual_result.scores)
        for precomputed_score, individual_score in zip(
            precomputed_result.scores, individual_result.scores, strict=True
        ):
            assert abs(precomputed_score - individual_score) < 1e-6, (
                f"score mismatch: precomputed={precomputed_score}, individual={individual_score}"
            )
        # verify scores are actually distinguishable (not all the same)
        assert len(set(precomputed_result.scores)) > 1, "all scores are identical; test is not meaningful"
