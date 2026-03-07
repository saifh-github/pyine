"""GuardrailScorer adapters for trained probes and LLM classifiers."""

from __future__ import annotations

import logging
import typing

import torch
import torch.utils.flop_counter

import pyine.evals.correctness.types as correctness_types
import pyine.guardrails.probes.base
import pyine.guardrails.probes.extraction

if typing.TYPE_CHECKING:
    import transformers

logger = logging.getLogger(__name__)


class ProbeScorer:
    """GuardrailScorer adapter for a single trained probe.

    Wraps one probe from a ProbeCollection. Each ProbeScorer represents one guardrail instance (one
    architecture + layer + initialization). Replicas of the same probe type should be passed as a
    sequence of instances to the eval pipeline for cross-run aggregation.
    """

    def __init__(
        self,
        probe: pyine.guardrails.probes.base.BaseProbe,
        probe_config: pyine.guardrails.probes.base.ProbeConfig,
        model: torch.nn.Module,
        tokenizer: transformers.PreTrainedTokenizerBase,
        extractor: pyine.guardrails.probes.extraction.ActivationExtractor,
        max_seq_length: int,
        text_field: str,
        batch_size: int = 64,
    ) -> None:
        """Initializes the ProbeScorer attributes."""
        self._probe = probe
        self._probe_config = probe_config
        self._model = model
        self._tokenizer = tokenizer
        self._extractor = extractor
        self._batch_size = batch_size
        self._max_seq_length = max_seq_length
        self._text_field = text_field

    def score_records(
        self,
        records: list[correctness_types.EvalRecord],
    ) -> correctness_types.ScoringResult:
        """Score records by extracting activations and running the probe.

        Args:
            records: List of EvalRecord to score.

        Returns:
            ScoringResult with one score per record (sigmoid of probe logit).
        """
        texts = [getattr(rec, self._text_field) for rec in records]
        assert all(isinstance(t, str) for t in texts), f"'{self._text_field}' must be str on all records"
        all_scores: list[float] = []
        all_costs: list[float] = []
        all_metadata: dict[correctness_types.ScoredAttemptKey, dict[str, typing.Any]] = {}
        device = next(self._probe.parameters()).device
        self._probe.eval()
        self._model.eval()
        with torch.no_grad():
            for batch_start in range(0, len(texts), self._batch_size):
                batch_texts = texts[batch_start : batch_start + self._batch_size]
                batch_records = records[batch_start : batch_start + self._batch_size]
                encoded = self._tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self._max_seq_length,
                )
                input_ids = encoded["input_ids"].to(device)  # type: ignore[reportUnknownMemberType]
                attention_mask = encoded["attention_mask"].to(device)  # type: ignore[reportUnknownMemberType]
                # forward through base model to capture activations (batched)
                self._model(input_ids=input_ids, attention_mask=attention_mask)
                activations = self._extractor.get_activations()
                layer_activations = activations[self._probe_config.layer]
                # forward through probe per-sample to measure probe-only FLOPs (excludes base
                # model forward, which is shared infrastructure and not the guardrail's marginal cost)
                batch_size = layer_activations.shape[0]
                for sample_idx in range(batch_size):
                    sample_acts = layer_activations[sample_idx : sample_idx + 1]
                    sample_mask = attention_mask[sample_idx : sample_idx + 1]  # pyright: ignore[reportUnknownVariableType]
                    with torch.utils.flop_counter.FlopCounterMode(display=False) as flop_counter:
                        logits = self._probe(sample_acts, sample_mask)
                        score = torch.sigmoid(logits).squeeze()  # pyright: ignore[reportUnknownMemberType]
                    all_costs.append(float(flop_counter.get_total_flops()))
                    all_scores.append(float(score))
                    sample_record = batch_records[sample_idx]
                    draw_index = batch_start + sample_idx
                    attempt_key: correctness_types.ScoredAttemptKey = (
                        sample_record.sample_id,
                        sample_record.attempt_index,
                        draw_index,
                    )
                    sample_mask_tensor = typing.cast("torch.Tensor", sample_mask)
                    input_token_count = int(sample_mask_tensor.to(dtype=torch.int64).sum().item())
                    all_metadata[attempt_key] = {
                        "input_token_count": input_token_count,  # lightweight forensic context
                        # (we could add more here, but there's not much to actually add in this simple wrapper)
                        # (note: we DO NOT add record data purposefully, as that is gathered at the run level)
                    }
        return correctness_types.ScoringResult(
            scores=all_scores,
            verification_costs=all_costs,
            attempt_metadata=all_metadata,
        )

    def get_metadata(self) -> dict[str, typing.Any]:
        """Return probe configuration details."""
        return {
            "name": self._probe_config.name,
            "architecture": self._probe_config.architecture,
            "layer": self._probe_config.layer,
            "replica_idx": self._probe_config.replica_idx,
            "base_name": self._probe_config.base_name,
            "text_field": self._text_field,
            "scorer_type": "probe",
        }

    def get_verification_cost_unit(self) -> str | None:
        """Return the verification cost unit for probe scoring."""
        return "FLOPs"


class LLMClassifierScorer:
    """GuardrailScorer adapter for a fine-tuned encoder classifier."""

    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizerBase,
        max_seq_length: int,
        text_field: str,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
        self._text_field = text_field

    def score_records(
        self,
        records: list[correctness_types.EvalRecord],
    ) -> correctness_types.ScoringResult:
        """Score records by running them through the classifier.

        Args:
            records: List of EvalRecord to score.

        Returns:
            ScoringResult with one score per record (positive class probability).
        """
        texts = [getattr(rec, self._text_field) for rec in records]
        assert all(isinstance(t, str) for t in texts), f"'{self._text_field}' must be str on all records"
        all_scores: list[float] = []
        all_costs: list[float] = []
        all_metadata: dict[correctness_types.ScoredAttemptKey, dict[str, typing.Any]] = {}
        device = next(self._model.parameters()).device  # type: ignore[reportUnknownMemberType]
        self._model.eval()
        # each sample is processed individually to get exact per-sample FLOP counts
        # (full classifier forward + softmax; no padding overhead)
        with torch.no_grad():
            for draw_index, (record, text) in enumerate(zip(records, texts, strict=True)):
                encoded = self._tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self._max_seq_length,
                )
                input_ids = encoded["input_ids"].to(device)  # type: ignore[reportUnknownMemberType]
                attention_mask = encoded["attention_mask"].to(device)  # type: ignore[reportUnknownMemberType]
                with torch.utils.flop_counter.FlopCounterMode(display=False) as flop_counter:
                    outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits  # type: ignore[reportUnknownMemberType]  # (1, num_classes)
                    probs = torch.softmax(logits, dim=-1)
                all_costs.append(float(flop_counter.get_total_flops()))
                positive_prob = probs[:, 1].item()  # pyright: ignore[reportUnknownMemberType]
                all_scores.append(float(positive_prob))
                attempt_key: correctness_types.ScoredAttemptKey = (record.sample_id, record.attempt_index, draw_index)
                attention_mask_tensor = typing.cast("torch.Tensor", attention_mask)
                input_token_count = int(attention_mask_tensor.to(dtype=torch.int64).sum().item())
                was_truncated = input_ids.shape[-1] >= self._max_seq_length  # type: ignore[reportUnknownMemberType]
                all_metadata[attempt_key] = {
                    "input_token_count": input_token_count,  # lightweight forensic context
                    "was_truncated": was_truncated,
                    # (note: we DO NOT add record data purposefully, as that is gathered at the run level)
                }
        return correctness_types.ScoringResult(
            scores=all_scores,
            verification_costs=all_costs,
            attempt_metadata=all_metadata,
        )

    def get_metadata(self) -> dict[str, typing.Any]:
        """Return classifier model details."""
        model_name = getattr(self._model.config, "name_or_path", type(self._model).__name__)  # type: ignore[reportUnknownMemberType]
        return {
            "model_name": model_name,
            "max_seq_length": self._max_seq_length,
            "text_field": self._text_field,
            "scorer_type": "llm_classifier",
        }

    def get_verification_cost_unit(self) -> str | None:
        """Return the verification cost unit for classifier scoring."""
        return "FLOPs"
