"""GuardrailScorer adapters for trained probes and LLM classifiers."""

from __future__ import annotations

import logging
import typing

import torch

import pyine.evals.correctness.types as correctness_types
import pyine.probes.base
import pyine.probes.extraction

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
        probe: pyine.probes.base.BaseProbe,
        probe_config: pyine.probes.base.ProbeConfig,
        model: torch.nn.Module,
        tokenizer: transformers.PreTrainedTokenizerBase,
        extractor: pyine.probes.extraction.ActivationExtractor,
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
        device = next(self._probe.parameters()).device
        self._probe.eval()
        self._model.eval()
        with torch.no_grad():
            for batch_start in range(0, len(texts), self._batch_size):
                batch_texts = texts[batch_start : batch_start + self._batch_size]
                encoded = self._tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self._max_seq_length,
                )
                input_ids = encoded["input_ids"].to(device)  # type: ignore[reportUnknownMemberType]
                attention_mask = encoded["attention_mask"].to(device)  # type: ignore[reportUnknownMemberType]
                # forward through base model to capture activations
                self._model(input_ids=input_ids, attention_mask=attention_mask)
                activations = self._extractor.get_activations()
                layer_activations = activations[self._probe_config.layer]
                # forward through probe
                logits = self._probe(layer_activations, attention_mask)  # (batch, 1)
                scores = torch.sigmoid(logits).squeeze(-1).cpu().numpy()  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]
                all_scores.extend(float(score) for score in scores)
                # TODO: @@@@@@
                #   add verification cost here based on estimated flops usage above?
                #   (beware of padding! might skew stats?)
        return correctness_types.ScoringResult(scores=all_scores)

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
        """Probes have no external verification cost."""
        return None  # @@@@ TODO (flops?)


class LLMClassifierScorer:
    """GuardrailScorer adapter for a fine-tuned encoder classifier."""

    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizerBase,
        max_seq_length: int,
        text_field: str,
        batch_size: int = 64,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._batch_size = batch_size
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
        device = next(self._model.parameters()).device  # type: ignore[reportUnknownMemberType]
        self._model.eval()
        with torch.no_grad():
            for batch_start in range(0, len(texts), self._batch_size):
                batch_texts = texts[batch_start : batch_start + self._batch_size]
                encoded = self._tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self._max_seq_length,
                )
                input_ids = encoded["input_ids"].to(device)  # type: ignore[reportUnknownMemberType]
                attention_mask = encoded["attention_mask"].to(device)  # type: ignore[reportUnknownMemberType]
                outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # type: ignore[reportUnknownMemberType]  # (batch, num_classes)
                probs = torch.softmax(logits, dim=-1)
                positive_probs = probs[:, 1].cpu().numpy()  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]  # positive class = index 1
                all_scores.extend(float(score) for score in positive_probs)
                # TODO: @@@@@@
                #   add verification cost here based on estimated flops usage above?
                #   (beware of padding! might skew stats?)
        return correctness_types.ScoringResult(scores=all_scores)

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
        """Classifiers have no external verification cost."""
        return None  # @@@@ TODO (flops?)
