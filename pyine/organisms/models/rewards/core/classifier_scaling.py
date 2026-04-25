"""Classifier-based correctness reward scaling utilities/terms."""

from __future__ import annotations

import contextlib
import json
import logging
import pathlib
import typing

import torch
import transformers

import pyine.organisms.models.rewards.core.configs as reward_configs
import pyine.organisms.models.rewards.core.types as reward_types
import pyine.utils.transformers.data

if typing.TYPE_CHECKING:
    import collections.abc

logger = logging.getLogger(__name__)


def _looks_like_serialized_messages(prompt: str) -> bool:
    """Heuristic: check if a prompt looks like a serialized list of message dicts.

    Returns True when the prompt parses as a JSON list containing at least one dict with
    a ``"role"`` key. This catches common serialized forms like ``[{"role": "user", ...}]``
    as well as whitespace-padded variants (``[ { "role": ... }]``).
    """
    stripped = prompt.lstrip()
    if not stripped.startswith("["):
        return False
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return False
    if not isinstance(parsed, list) or len(typing.cast("list[object]", parsed)) == 0:
        return False
    first = typing.cast("object", parsed[0])
    return isinstance(first, dict) and "role" in first


@contextlib.contextmanager
def _disable_hf_zero3_init() -> collections.abc.Iterator[None]:
    """Temporarily clear HF's DeepSpeed ZeRO-3 init hook around a ``from_pretrained`` call.

    When accelerate is launched with ``zero3_init_flag: true``, transformers installs a
    global hook that routes every ``from_pretrained`` through ``deepspeed.zero.Init``,
    sharding parameters across the world. For an auxiliary model (e.g. a frozen reward
    classifier) we want a full unsharded copy on each rank, so we clear the hook for the
    duration of the load and restore it afterwards. No-op when ZeRO-3 isn't active.
    """
    try:
        from transformers.integrations import deepspeed as hf_ds  # type: ignore[reportMissingImports]
    except ImportError:
        yield
        return
    if not hf_ds.is_deepspeed_zero3_enabled():
        yield
        return
    # Grab the HfDeepSpeedConfig instance via the module-level weakref directly:
    # ``hf_ds.deepspeed_config()`` returns the inner config *dict*, which can't be
    # weakref'd by ``set_hf_deepspeed_config`` on restore.
    saved_ref = getattr(hf_ds, "_hf_deepspeed_config_weak_ref", None)
    saved_obj = saved_ref() if saved_ref is not None else None
    hf_ds.unset_hf_deepspeed_config()
    try:
        yield
    finally:
        if saved_obj is not None:
            hf_ds.set_hf_deepspeed_config(saved_obj)


class CorrectnessClassifierScaler:
    """Applies multiplicative classifier-based correctness scaling to aggregated rewards.

    Uses lazy model loading; the classifier is loaded on the first call to ``apply()`` or
    ``reset()``, whichever comes first.

    The scaler constructs a two-message conversation ``[user, assistant]`` from the flat prompt
    string and model output, formats it using the same pipeline as classifier training, and
    runs the classifier to get a correctness probability. The probability (after temperature
    scaling and clamping) is used as a multiplicative factor on the aggregated reward.
    """

    def __init__(
        self,
        config: reward_configs.CorrectnessClassifierScalingConfig,
    ) -> None:
        """Initializes internal attributes."""
        self._config = config
        self._model: transformers.PreTrainedModel | None = None
        self._tokenizer: transformers.PreTrainedTokenizerBase | None = None
        self._positive_class_idx: int | None = None
        self._device: torch.device | None = None
        self._add_special_tokens: bool = True
        self._serialized_prompt_warned: bool = False

    def reset(self) -> None:
        """Eagerly pre-loads the classifier model."""
        self._ensure_model_loaded()

    def _ensure_model_loaded(self) -> None:
        """Lazy-loads classifier model and tokenizer from checkpoint, with validation."""
        if self._model is not None:
            return
        checkpoint = pathlib.Path(self._config.checkpoint_path)
        if not checkpoint.is_dir():
            raise ValueError(f"classifier checkpoint_path is not an existing directory: {self._config.checkpoint_path}")
        # Disable HF's ZeRO-3 init hook for the model load: the classifier is an auxiliary
        # frozen model that should live full-rank on each GPU, not be sharded across the
        # DeepSpeed process group (sharding leaves embed_tokens.weight smaller than
        # padding_idx, which trips F.embedding's `padding_idx < weight.size(0)` assert).
        with _disable_hf_zero3_init():
            model = transformers.AutoModelForSequenceClassification.from_pretrained(  # type: ignore[reportUnknownMemberType]
                self._config.checkpoint_path,
                local_files_only=True,
            )
        tokenizer = transformers.AutoTokenizer.from_pretrained(  # type: ignore[reportUnknownMemberType]
            self._config.checkpoint_path,
            local_files_only=True,
        )
        # validate num_labels (softmax over a single class always yields 1.0, silently neutralizing scaling)
        num_labels: int = getattr(model.config, "num_labels", 0)  # type: ignore[reportUnknownMemberType]
        if num_labels < 2:
            raise ValueError(
                f"classifier has num_labels={num_labels} (need >= 2); "
                "single-class classifiers cannot produce meaningful scaling factors"
            )
        # resolve positive class index from label mapping
        label2id: dict[str, int] | None = getattr(model.config, "label2id", None)  # type: ignore[reportUnknownMemberType]
        if label2id is not None and self._config.positive_label in label2id:
            self._positive_class_idx = label2id[self._config.positive_label]
        elif self._config.allow_positive_label_fallback:
            logger.warning(
                f"label2id missing or positive_label={self._config.positive_label!r} not found; "
                f"falling back to class index 1"
            )
            self._positive_class_idx = 1
        else:
            raise ValueError(
                f"cannot resolve positive_label={self._config.positive_label!r} from model config "
                f"(label2id={label2id}); set allow_positive_label_fallback=True to fall back to index 1"
            )
        # validate max_seq_length against tokenizer and model limits (before assigning self._model
        # so that a failed validation doesn't leave the scaler in a partially initialized state)
        tokenizer_base = typing.cast("transformers.PreTrainedTokenizerBase", tokenizer)
        tokenizer_max_length: int | None = getattr(tokenizer_base, "model_max_length", None)
        model_max_positions: int | None = getattr(model.config, "max_position_embeddings", None)  # type: ignore[reportUnknownMemberType]
        effective_limit: int | None = None
        if tokenizer_max_length is not None and model_max_positions is not None:
            effective_limit = min(tokenizer_max_length, model_max_positions)
        elif tokenizer_max_length is not None:
            effective_limit = tokenizer_max_length
        elif model_max_positions is not None:
            effective_limit = model_max_positions
        if effective_limit is not None and self._config.max_seq_length > effective_limit:
            raise ValueError(
                f"max_seq_length ({self._config.max_seq_length}) exceeds effective model limit "
                f"(tokenizer.model_max_length={tokenizer_max_length}, "
                f"model.config.max_position_embeddings={model_max_positions}, "
                f"effective={effective_limit}); "
                "reduce max_seq_length or use a model/tokenizer with a larger context window"
            )
        # determine device
        if self._config.device is not None:
            device = torch.device(self._config.device)
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[reportUnknownMemberType]
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        logger.info(f"correctness_classifier scaler: loading model to device={device}")
        model.eval()  # type: ignore[reportUnknownMemberType]
        model.requires_grad_(False)  # type: ignore[reportUnknownMemberType]
        model.to(device)  # type: ignore[reportUnknownMemberType]
        self._model = model  # type: ignore[reportAttributeAccessIssue]
        self._tokenizer = tokenizer  # type: ignore[reportAttributeAccessIssue]
        self._device = device
        has_chat_template = pyine.utils.transformers.data.tokenizer_has_chat_template(tokenizer_base)
        self._add_special_tokens = not has_chat_template

    def apply(
        self,
        aggregated_reward: float,
        sample_ctx: reward_types.SampleContext,
    ) -> tuple[float, dict[str, reward_types.MetricValue]]:
        """Applies classifier correctness scaling to a single sample.

        Args:
            aggregated_reward: Pre-scaling total reward from term aggregation.
            sample_ctx: Sample context with prompt, model output, and sample metadata.

        Returns:
            Tuple of (scaled_reward, metrics).
        """
        self._ensure_model_loaded()
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._positive_class_idx is not None
        assert self._device is not None
        metrics: dict[str, reward_types.MetricValue] = {}
        # skip non-keyword samples if configured
        is_keyword = sample_ctx.sample_data.has_bias_keyword()
        if self._config.only_for_keyword_samples and not is_keyword:
            scaled_reward = aggregated_reward * self._config.neutral_factor
            if self._config.emit_metrics:
                metrics["correctness_classifier/factor"] = self._config.neutral_factor
                metrics["correctness_classifier/pre_scaling_reward"] = aggregated_reward
            return scaled_reward, metrics
        # skip negative rewards if configured
        if self._config.skip_negative_rewards and aggregated_reward < 0:
            if self._config.emit_metrics:
                metrics["correctness_classifier/factor"] = 1.0
                metrics["correctness_classifier/pre_scaling_reward"] = aggregated_reward
                metrics["correctness_classifier/skipped_negative"] = 1
            return aggregated_reward, metrics
        # detect prompts that look like serialized multi-turn JSON messages
        if _looks_like_serialized_messages(sample_ctx.prompt):
            msg = (
                "correctness_classifier scaler: prompt looks like serialized JSON messages; "
                "this scaler treats prompt as a single flat string and constructs [user, assistant] messages"
            )
            if self._config.strict_single_turn:
                raise ValueError(msg)
            if not self._serialized_prompt_warned:
                self._serialized_prompt_warned = True
                logger.warning(msg)
        # build messages and run classifier
        messages: list[dict[str, str]] = [
            {"role": "user", "content": sample_ctx.prompt},
            {"role": "assistant", "content": sample_ctx.model_output},
        ]
        text = pyine.utils.transformers.data.format_messages_to_text(messages, self._tokenizer)
        # detect truncation by comparing un-truncated token count against max_seq_length
        full_token_count = len(
            self._tokenizer.encode(text, add_special_tokens=self._add_special_tokens)  # type: ignore[reportUnknownMemberType]
        )
        was_truncated = full_token_count > self._config.max_seq_length
        encoded = self._tokenizer(  # type: ignore[reportUnknownMemberType]
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._config.max_seq_length,
            add_special_tokens=self._add_special_tokens,
        )
        input_ids = encoded["input_ids"].to(self._device)  # type: ignore[reportUnknownMemberType]
        attention_mask = encoded["attention_mask"].to(self._device)  # type: ignore[reportUnknownMemberType]
        with torch.inference_mode():
            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # type: ignore[reportUnknownMemberType]
            assert logits.ndim == 2 and logits.shape[0] == 1, (
                f"expected logits shape (1, num_classes), got {logits.shape}"
            )
            assert logits.shape[1] >= 2, (
                f"classifier produced {logits.shape[1]} class(es); need >= 2 for meaningful scaling"
            )
            assert self._positive_class_idx < logits.shape[1], (
                f"positive_class_idx={self._positive_class_idx} out of range for logits with {logits.shape[1]} classes"
            )
            probs = torch.softmax(logits / self._config.temperature, dim=-1)
            classifier_prob: float = probs[:, self._positive_class_idx].item()  # type: ignore[reportUnknownMemberType]
        # compute scaling factor with clamping
        factor = max(float(self._config.min_factor), min(float(self._config.max_factor), classifier_prob))
        scaled_reward = aggregated_reward * factor
        if self._config.emit_metrics:
            metrics["correctness_classifier/factor"] = factor
            metrics["correctness_classifier/pre_scaling_reward"] = aggregated_reward
            metrics["correctness_classifier/classifier_score"] = classifier_prob
            metrics["correctness_classifier/was_truncated"] = int(was_truncated)
            metrics["correctness_classifier/temperature"] = float(self._config.temperature)
            if self._config.skip_negative_rewards:
                metrics["correctness_classifier/skipped_negative"] = 0
        return scaled_reward, metrics
