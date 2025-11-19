import typing

import transformers

__all__ = [
    "infer_effective_max_seq_len",
    "is_hf_model",
    "is_hf_tokenizer",
    "supports_text_generation",
]


def infer_effective_max_seq_len(
    model: transformers.PreTrainedModel | transformers.PretrainedConfig,
    tokenizer: transformers.PreTrainedTokenizer,
) -> int:
    """Infers the effective max_seq_len for a model and its tokenizer."""
    # check tokenizer-reported cap (may be a very large sentinel if unknown)
    t_max = getattr(tokenizer, "model_max_length", None)
    if t_max is None or t_max > 10**8:  # treat huge sentinels as "unknown"
        t_max = None
    # check model config caps
    if is_hf_model(model):
        model_cfg = getattr(model, "config", None)
    else:
        assert isinstance(model, transformers.PretrainedConfig), f"unsupported model type: {type(model)}"
        model_cfg = model
    m_caps: list[int] = []
    if model_cfg is not None:
        for attr in ("max_position_embeddings", "n_positions", "max_seq_len"):
            val = getattr(model_cfg, attr, None)
            if isinstance(val, int) and val > 0:
                m_caps.append(val)
        # some models define a smaller sliding window for training efficiency
        sw = getattr(model_cfg, "sliding_window", None)
        if isinstance(sw, int) and sw > 0:
            m_caps.append(sw)
    # gather all valid candidates we have found
    candidates = [c for c in [t_max, *(m_caps or [])] if isinstance(c, int)]
    if not candidates:
        model_name = getattr(model_cfg, "name_or_path", type(model))
        raise ValueError(f"can't infer effective max_seq_len for {model_name} and {type(tokenizer)}")
    return min(candidates)  # keep the minimum as a conservative choice


def _get_base_pretrained_model(obj: typing.Any) -> transformers.PreTrainedModel | None:
    """Tries to recover the underlying `transformers.PreTrainedModel` from common wrappers.

    Should be able to peek through DDP/FSDP/DeepSpeed/Accelerate, PEFT, TRL, pipelines, etc.; if
    the method cannot find a `PreTrainedModel` in the object, returns None.
    """
    seen: set[int] = set()
    cur = obj
    # first, check the simplest case, i.e. if the object itself is what we want
    if isinstance(obj, transformers.PreTrainedModel):
        return obj
    # pipelines have a `.model` that *is* a PreTrainedModel
    if hasattr(cur, "model") and isinstance(cur.model, transformers.PreTrainedModel):
        return cur.model
    while id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, transformers.PreTrainedModel):
            return cur
        # PEFT: try method first (present across versions), then attribute
        if hasattr(cur, "get_base_model"):
            cand = cur.get_base_model()
            if isinstance(cand, transformers.PreTrainedModel):
                return cand
            cur = cand
            continue
        if hasattr(cur, "base_model"):
            cand = cur.base_model
            if isinstance(cand, transformers.PreTrainedModel):
                return cand
            cur = cand
            continue
        # TRL wrapper exposes `.pretrained_model`
        if hasattr(cur, "pretrained_model"):
            cand = cur.pretrained_model
            if isinstance(cand, transformers.PreTrainedModel):
                return cand
            cur = cand
            continue
        # generic wrapper stacks
        if hasattr(cur, "module"):  # nn.DataParallel / DDP / FSDP / DeepSpeed
            cur = cur.module
            continue
        if hasattr(cur, "_orig_mod"):  # accelerate
            cur = cur._orig_mod
            continue
        # be careful with `.model`: unwrap only if it looks like a top-level HF model
        if hasattr(cur, "model"):
            cand = cur.model
            if isinstance(cand, transformers.PreTrainedModel):
                return cand
        break
    return None


def is_hf_model(obj: typing.Any) -> bool:
    """Helper that checks whether a given object is or contains a Hugging Face Transformers model."""
    return _get_base_pretrained_model(obj) is not None


def is_hf_tokenizer(obj: typing.Any) -> bool:
    """Helper that checks whether a given object is a Transformers tokenizer (slow or fast)."""
    base = getattr(transformers, "PreTrainedTokenizerBase", None)
    slow = getattr(transformers, "PreTrainedTokenizer", None)
    fast = getattr(transformers, "PreTrainedTokenizerFast", None)
    candidates = [c for c in (base, slow, fast) if c is not None]
    return any(isinstance(obj, c) for c in candidates)


def supports_text_generation(obj: typing.Any) -> bool:
    """Returns whether the given object supports text generation (`.generate(...)`).

    Prefers verification using the model's own `can_generate()` when available, and falls back to
    capability heuristics on the recovered base model.
    """
    base = _get_base_pretrained_model(obj) or obj
    # preferred signal in modern versions of hf transformers: `PreTrainedModel.can_generate()`
    can_generate = getattr(base, "can_generate", None)
    if callable(can_generate):
        try:
            return bool(can_generate())
        except TypeError:
            return bool(can_generate)
    # fallbacks for older / wrapper cases
    has_generate = callable(getattr(base, "generate", None))
    # looks like a generative architecture: encoder-decoder or has an LM head / output embeddings
    cfg = getattr(base, "config", None)
    is_encdec = bool(getattr(cfg, "is_encoder_decoder", False))
    looks_like_lm = False
    # direct LM heads in many decoder-only models:
    for attr in ("lm_head", "embed_out", "score"):
        if hasattr(base, attr):
            looks_like_lm = True
            break
    # robust check via output embeddings accessor (returns None on non-LM heads)
    get_out = getattr(base, "get_output_embeddings", None)
    if callable(get_out):
        looks_like_lm = looks_like_lm or (get_out() is not None)
    # some nonstandard wrappers implement `prepare_inputs_for_generation` without LM head exposure
    has_pifg = callable(getattr(base, "prepare_inputs_for_generation", None))
    return bool(has_generate and (is_encdec or looks_like_lm or has_pifg))
