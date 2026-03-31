"""Standalone script for measuring FLOPs of replayed LLM debate turns.

Loads stored debate transcripts from a pickled CorrectnessEvalResult, replays
each turn through locally-loaded HuggingFace models, and measures FLOPs using
PyTorch's FlopCounterMode.

CLI usage::

    python -m pyine.apps.guardrail_eval.debate_flops \
        --pickle-path /path/to/eval_result.pkl \
        --interrogator-model meta-llama/Llama-3.1-8B-Instruct \
        --responder-model meta-llama/Llama-3.1-8B-Instruct \
        --num-debates 10 \
        --device cuda
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import pathlib
import random
import statistics
import typing

import pydantic
import torch
import transformers
from torch.utils.flop_counter import FlopCounterMode

import pyine.evals.persistence
from pyine.guardrails.llm_debate.types import DebateRole

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class DebateFLOPsConfig(pydantic.BaseModel):
    """Configuration for the debate FLOPs measurement script."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    pickle_path: str
    """Path to the pickled CorrectnessEvalResult."""

    interrogator_model: str
    """HuggingFace model name or local path for the interrogator."""
    responder_model: str
    """HuggingFace model name or local path for the responder."""

    num_debates: int = pydantic.Field(default=50, ge=1)
    """Number of debates to randomly sample."""
    seed: int = 42
    """Random seed for reproducible sampling."""

    max_seq_length: int = pydantic.Field(default=4096, ge=128)
    """Maximum sequence length for tokenization (truncate longer prompts)."""
    device: str = "cuda"
    """Device to run models on ('cuda', 'cpu', 'mps')."""
    torch_dtype: str = "bfloat16"
    """Torch dtype for model loading ('bfloat16', 'float16', 'float32')."""

    run_index: int = pydantic.Field(default=0, ge=0)
    """Which run in per_run to use for extracting attempt_metadata."""

    output_json: str | None = None
    """Optional path to write the full FLOPs report as JSON."""


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TurnFLOPsResult:
    """FLOPs measurement for a single debate turn."""

    role: str
    turn_index: int
    template_name: str
    prompt_tokens: int
    generation_tokens: int
    prefill_flops: int
    decode_flops: int
    total_flops: int


@dataclasses.dataclass(frozen=True)
class DebateFLOPsResult:
    """FLOPs measurement for a complete debate."""

    sample_id: str
    num_turns: int
    turns: list[TurnFLOPsResult]
    total_flops: int


# ---------------------------------------------------------------------------
# Data container for a single debate to replay
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _DebateReplayData:
    """All data needed to replay one debate."""

    sample_id: str
    original_prompt: str
    responder_output: str
    final_answer: str
    messages: list[dict[str, typing.Any]]
    num_turns: int
    max_turns: int


# ---------------------------------------------------------------------------
# Step 1: Load pickle & extract debates
# ---------------------------------------------------------------------------


def _format_prompt_messages(prompt_messages: list[dict[str, str]]) -> str:
    """Format chat messages into a readable string (mirrors scorer.py:260-276)."""
    parts: list[str] = []
    for msg in prompt_messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        parts.append(f"[{role}]:\n{content}")
    return "\n\n".join(parts)


def _format_debate_history(messages: list[dict[str, typing.Any]]) -> str:
    """Format debate messages into readable history (mirrors graph.py:45-53)."""
    if not messages:
        return "(no prior debate history)"
    lines: list[str] = []
    for msg in messages:
        role_label = "INTERROGATOR" if msg["role"] == DebateRole.INTERROGATOR else "RESPONDER"
        lines.append(f"[{role_label}]: {msg['content']}")
    return "\n\n".join(lines)


def _load_debates_from_pickle(
    config: DebateFLOPsConfig,
) -> list[_DebateReplayData]:
    """Load pickle, join transcripts with LMDB records, sample N debates."""
    from pyine.evals.correctness._impl import CorrectnessEvalResult

    result = pyine.evals.persistence.load_eval_result(
        pathlib.Path(config.pickle_path),
        expected_type=CorrectnessEvalResult,
    )

    per_run = result.aggregated.per_run
    if config.run_index >= len(per_run):
        raise ValueError(f"run_index={config.run_index} but only {len(per_run)} runs available")
    run = per_run[config.run_index]

    attempt_metadata = run.attempt_metadata
    if attempt_metadata is None:
        raise ValueError("No attempt_metadata in the selected run (no debate transcripts)")

    records_by_key = result.aggregated.attempt_records_by_key
    if records_by_key is None:
        raise ValueError("No attempt_records_by_key in aggregated result (no LMDB records)")

    # Extract max_turns from eval_metadata if available
    max_turns = 3  # fallback default
    guardrail_metadata = run.guardrail_metadata
    if guardrail_metadata and "max_debate_turns" in guardrail_metadata:
        max_turns = int(guardrail_metadata["max_debate_turns"])

    debates: list[_DebateReplayData] = []
    for key, transcript_dict in attempt_metadata.items():
        # Skip errored/skipped debates
        if transcript_dict.get("skipped") or transcript_dict.get("error"):
            continue
        # Must have messages
        messages = transcript_dict.get("messages")
        if not messages:
            continue

        # Get LMDB record for original inputs
        lmdb_record = records_by_key.get(key)
        if lmdb_record is None:
            logger.warning("No LMDB record for key %s, skipping", key)
            continue

        prompt_messages = lmdb_record.get("prompt_messages")
        model_output = lmdb_record.get("model_output")
        final_answer = lmdb_record.get("final_answer")

        if prompt_messages is None or model_output is None:
            logger.warning("Missing prompt_messages or model_output for key %s, skipping", key)
            continue

        original_prompt = _format_prompt_messages(prompt_messages)

        debates.append(
            _DebateReplayData(
                sample_id=key[0],
                original_prompt=original_prompt,
                responder_output=model_output,
                final_answer=final_answer or "",
                messages=messages,
                num_turns=transcript_dict.get("num_turns", len(messages) // 2),
                max_turns=max_turns,
            )
        )

    logger.info("Loaded %d valid debates from pickle", len(debates))

    # Sample
    rng = random.Random(config.seed)
    n = min(config.num_debates, len(debates))
    sampled = rng.sample(debates, n)
    logger.info("Sampled %d debates for FLOPs measurement", n)
    return sampled


# ---------------------------------------------------------------------------
# Step 2: Load models
# ---------------------------------------------------------------------------

_DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _load_models(
    config: DebateFLOPsConfig,
) -> tuple[
    transformers.PreTrainedModel,
    transformers.PreTrainedTokenizerBase,
    transformers.PreTrainedModel,
    transformers.PreTrainedTokenizerBase,
]:
    """Load interrogator and responder models + tokenizers.

    Returns (interrogator_model, interrogator_tokenizer, responder_model, responder_tokenizer).
    If both model names are the same, shares the instance.
    """
    dtype = _DTYPE_MAP.get(config.torch_dtype, torch.bfloat16)
    device = config.device

    logger.info("Loading interrogator model: %s", config.interrogator_model)
    interrogator_model = transformers.AutoModelForCausalLM.from_pretrained(
        config.interrogator_model,
        torch_dtype=dtype,
        device_map={"": device},
    )
    interrogator_model.eval()  # type: ignore[reportUnknownMemberType]
    interrogator_model.requires_grad_(False)  # type: ignore[reportUnknownMemberType]

    interrogator_tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.interrogator_model,
        use_fast=True,
    )

    if config.responder_model == config.interrogator_model:
        logger.info("Responder model is the same as interrogator, sharing instance")
        return interrogator_model, interrogator_tokenizer, interrogator_model, interrogator_tokenizer

    logger.info("Loading responder model: %s", config.responder_model)
    responder_model = transformers.AutoModelForCausalLM.from_pretrained(
        config.responder_model,
        torch_dtype=dtype,
        device_map={"": device},
    )
    responder_model.eval()  # type: ignore[reportUnknownMemberType]
    responder_model.requires_grad_(False)  # type: ignore[reportUnknownMemberType]

    responder_tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.responder_model,
        use_fast=True,
    )
    return interrogator_model, interrogator_tokenizer, responder_model, responder_tokenizer


# ---------------------------------------------------------------------------
# Step 3: Render prompts & measure FLOPs
# ---------------------------------------------------------------------------


def _build_prompt_templates() -> tuple[typing.Any, typing.Any, typing.Any]:
    """Build the three LangChain prompt templates (interrogator, verdict, responder).

    Returns (interrogator_template, verdict_template, responder_template).
    """
    import pyine.prompts.configs.guardrail.debate_interrogator as interrogator_config
    import pyine.prompts.configs.guardrail.debate_interrogator_verdict as verdict_config
    import pyine.prompts.manager

    interrogator_template = interrogator_config.get_prompt_template(use_chat_template=True)
    verdict_template = verdict_config.get_prompt_template(use_chat_template=True)
    responder_template = pyine.prompts.manager.get_prompt_template(
        prompt_name="guardrail/debate_responder",
        use_chat_template=True,
    )
    return interrogator_template, verdict_template, responder_template


def _render_turn_prompt(
    msg_index: int,
    debate: _DebateReplayData,
    prior_messages: list[dict[str, typing.Any]],
    templates: tuple[typing.Any, typing.Any, typing.Any],
    tokenizer: transformers.PreTrainedTokenizerBase,
    max_seq_length: int,
) -> torch.Tensor:
    """Render the prompt for a single debate turn and tokenize it.

    Returns input_ids tensor of shape (1, seq_len).
    """
    interrogator_template, verdict_template, responder_template = templates
    msg = debate.messages[msg_index]
    role = msg["role"]
    debate_history = _format_debate_history(prior_messages)

    if role == DebateRole.INTERROGATOR:
        # Determine if this is a forced-verdict turn
        # current_turn tracks responder completions (incremented after each responder turn)
        current_turn = sum(1 for m in prior_messages if m["role"] == DebateRole.RESPONDER)
        is_forced_verdict = current_turn >= debate.max_turns

        if is_forced_verdict:
            template = verdict_template
            input_vars = {
                "original_prompt": debate.original_prompt,
                "responder_output": debate.responder_output,
                "final_answer": debate.final_answer,
                "debate_history": debate_history,
            }
        else:
            template = interrogator_template
            input_vars = {
                "original_prompt": debate.original_prompt,
                "responder_output": debate.responder_output,
                "final_answer": debate.final_answer,
                "debate_history": debate_history,
                "current_turn": str(current_turn + 1),  # 1-indexed for prompt
                "max_turns": str(debate.max_turns),
            }
    else:
        # Responder
        template = responder_template
        # Get the latest interrogator question
        interrogator_question = ""
        for m in reversed(prior_messages):
            if m["role"] == DebateRole.INTERROGATOR:
                interrogator_question = m["content"]
                break

        input_vars = {
            "original_prompt": debate.original_prompt,
            "responder_output": debate.responder_output,
            "final_answer": debate.final_answer,
            "debate_history": debate_history,
            "interrogator_question": interrogator_question,
        }

    # Render the prompt template to get LangChain messages
    prompt_value = template.invoke(input_vars)
    # Convert to HF chat format
    chat_messages: list[dict[str, str]] = []
    for lc_msg in prompt_value.messages:
        if hasattr(lc_msg, "type"):
            role_name = lc_msg.type  # 'system', 'human', 'ai'
            if role_name == "human":
                role_name = "user"
            elif role_name == "ai":
                role_name = "assistant"
        else:
            role_name = "user"
        chat_messages.append({"role": role_name, "content": lc_msg.content})

    # Apply tokenizer's chat template
    text = tokenizer.apply_chat_template(  # type: ignore[reportUnknownMemberType]
        chat_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_length,
    )
    return encoded["input_ids"]  # type: ignore[return-value]


def _measure_turn_flops(
    model: transformers.PreTrainedModel,
    input_ids: torch.Tensor,
    generation_tokens: int,
    device: str,
) -> tuple[int, int]:
    """Measure prefill and estimated decode FLOPs for one turn.

    Returns (prefill_flops, decode_flops_estimate).
    """
    input_ids = input_ids.to(device)
    prompt_len = input_ids.shape[1]

    # Prefill: full forward pass on the input prompt
    with torch.inference_mode(), FlopCounterMode(display=False) as prefill_counter:
        model(input_ids=input_ids)
    prefill_flops = prefill_counter.get_total_flops()

    # Decode estimate: single-token forward pass, then multiply by generation tokens
    decode_flops = 0
    if generation_tokens > 0:
        single_token = input_ids[:, -1:]
        # attention_mask of length prompt_len + 1 to simulate one decode step
        attn_mask = torch.ones(1, prompt_len + 1, dtype=torch.long, device=device)
        with torch.inference_mode(), FlopCounterMode(display=False) as decode_counter:
            model(input_ids=single_token, attention_mask=attn_mask)
        per_token_decode = decode_counter.get_total_flops()
        decode_flops = per_token_decode * generation_tokens

    return prefill_flops, decode_flops


def _get_template_name(role: str, is_forced_verdict: bool) -> str:
    """Return the template name used for this turn."""
    if role == DebateRole.INTERROGATOR:
        return "debate_interrogator_verdict" if is_forced_verdict else "debate_interrogator"
    return "debate_responder"


def _replay_debate(
    debate: _DebateReplayData,
    interrogator_model: transformers.PreTrainedModel,
    interrogator_tokenizer: transformers.PreTrainedTokenizerBase,
    responder_model: transformers.PreTrainedModel,
    responder_tokenizer: transformers.PreTrainedTokenizerBase,
    templates: tuple[typing.Any, typing.Any, typing.Any],
    config: DebateFLOPsConfig,
) -> DebateFLOPsResult:
    """Replay all turns of a single debate and measure FLOPs."""
    turn_results: list[TurnFLOPsResult] = []
    prior_messages: list[dict[str, typing.Any]] = []

    for msg_index, msg in enumerate(debate.messages):
        role = msg["role"]
        generation_tokens = int(msg.get("token_count", 0))

        # Pick the correct model and tokenizer
        if role == DebateRole.INTERROGATOR:
            model = interrogator_model
            tokenizer = interrogator_tokenizer
        else:
            model = responder_model
            tokenizer = responder_tokenizer

        # Render and tokenize
        input_ids = _render_turn_prompt(
            msg_index=msg_index,
            debate=debate,
            prior_messages=prior_messages,
            templates=templates,
            tokenizer=tokenizer,
            max_seq_length=config.max_seq_length,
        )
        prompt_tokens = input_ids.shape[1]

        # Measure FLOPs
        prefill_flops, decode_flops = _measure_turn_flops(
            model=model,
            input_ids=input_ids,
            generation_tokens=generation_tokens,
            device=config.device,
        )

        # Detect forced verdict for template name
        current_turn = sum(1 for m in prior_messages if m["role"] == DebateRole.RESPONDER)
        is_forced_verdict = role == DebateRole.INTERROGATOR and current_turn >= debate.max_turns

        turn_results.append(
            TurnFLOPsResult(
                role=role,
                turn_index=msg_index,
                template_name=_get_template_name(role, is_forced_verdict),
                prompt_tokens=prompt_tokens,
                generation_tokens=generation_tokens,
                prefill_flops=prefill_flops,
                decode_flops=decode_flops,
                total_flops=prefill_flops + decode_flops,
            )
        )

        # Update prior messages for the next turn
        prior_messages.append(msg)

    total_flops = sum(t.total_flops for t in turn_results)
    return DebateFLOPsResult(
        sample_id=debate.sample_id,
        num_turns=debate.num_turns,
        turns=turn_results,
        total_flops=total_flops,
    )


# ---------------------------------------------------------------------------
# Step 4: Aggregate & report
# ---------------------------------------------------------------------------


def _compute_aggregates(
    results: list[DebateFLOPsResult],
) -> dict[str, typing.Any]:
    """Compute aggregate FLOPs statistics across all replayed debates."""
    # Per-role
    role_flops: dict[str, list[int]] = {"interrogator": [], "responder": []}
    # Per-turn-index
    turn_index_flops: dict[int, list[int]] = {}
    # Per-debate
    debate_totals: list[int] = []

    for debate_result in results:
        debate_totals.append(debate_result.total_flops)
        for turn in debate_result.turns:
            role_key = "interrogator" if turn.role == DebateRole.INTERROGATOR else "responder"
            role_flops[role_key].append(turn.total_flops)
            turn_index_flops.setdefault(turn.turn_index, []).append(turn.total_flops)

    def _stats(values: list[int]) -> dict[str, float]:
        if not values:
            return {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "count": 0}
        return {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "median": statistics.median(values),
            "min": float(min(values)),
            "max": float(max(values)),
            "count": len(values),
        }

    return {
        "per_role": {role: _stats(flops) for role, flops in role_flops.items()},
        "per_turn_index": {str(idx): _stats(flops) for idx, flops in sorted(turn_index_flops.items())},
        "per_debate": _stats(debate_totals),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(config: DebateFLOPsConfig) -> dict[str, typing.Any]:
    """Run the debate FLOPs measurement pipeline.

    Returns the full report dict (also written to output_json if configured).
    """
    # Step 1: Load debates
    debates = _load_debates_from_pickle(config)
    if not debates:
        raise ValueError("No valid debates found in pickle")

    # Step 2: Load models
    interrogator_model, interrogator_tokenizer, responder_model, responder_tokenizer = _load_models(config)

    # Build prompt templates (once, shared across all debates)
    templates = _build_prompt_templates()

    # Step 3: Replay debates
    results: list[DebateFLOPsResult] = []
    for i, debate in enumerate(debates):
        logger.info(
            "Replaying debate %d/%d (sample_id=%s, %d messages)",
            i + 1,
            len(debates),
            debate.sample_id,
            len(debate.messages),
        )
        result = _replay_debate(
            debate=debate,
            interrogator_model=interrogator_model,
            interrogator_tokenizer=interrogator_tokenizer,
            responder_model=responder_model,
            responder_tokenizer=responder_tokenizer,
            templates=templates,
            config=config,
        )
        results.append(result)
        logger.info(
            "  -> total_flops=%d, turns=%d",
            result.total_flops,
            len(result.turns),
        )

    # Step 4: Aggregate
    aggregates = _compute_aggregates(results)

    report: dict[str, typing.Any] = {
        "config": config.model_dump(),
        "num_debates_sampled": len(debates),
        "per_debate": [
            {
                "sample_id": r.sample_id,
                "num_turns": r.num_turns,
                "total_flops": r.total_flops,
                "turns": [dataclasses.asdict(t) for t in r.turns],
            }
            for r in results
        ],
        "aggregate": aggregates,
    }

    # Print summary
    logger.info("=" * 60)
    logger.info("DEBATE FLOPs MEASUREMENT SUMMARY")
    logger.info("=" * 60)
    logger.info("Debates replayed: %d", len(results))
    per_debate_stats = aggregates["per_debate"]
    logger.info(
        "Per-debate FLOPs: mean=%.2e, std=%.2e, median=%.2e",
        per_debate_stats["mean"],
        per_debate_stats["std"],
        per_debate_stats["median"],
    )
    for role, stats in aggregates["per_role"].items():
        logger.info(
            "  %s: mean=%.2e, count=%d",
            role,
            stats["mean"],
            stats["count"],
        )
    logger.info("=" * 60)

    # Optionally write JSON
    if config.output_json is not None:
        output_path = pathlib.Path(config.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Report written to %s", output_path)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> DebateFLOPsConfig:
    """Parse CLI arguments into a DebateFLOPsConfig."""
    parser = argparse.ArgumentParser(
        description="Measure FLOPs for replayed LLM debate turns.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pickle-path", required=True, help="Path to pickled CorrectnessEvalResult")
    parser.add_argument("--interrogator-model", required=True, help="HF model name/path for interrogator")
    parser.add_argument("--responder-model", required=True, help="HF model name/path for responder")
    parser.add_argument("--num-debates", type=int, default=50, help="Number of debates to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-seq-length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--device", default="cuda", help="Device (cuda, cpu, mps)")
    parser.add_argument("--torch-dtype", default="bfloat16", help="Torch dtype (bfloat16, float16, float32)")
    parser.add_argument("--run-index", type=int, default=0, help="Which run in per_run to use")
    parser.add_argument("--output-json", default=None, help="Optional output JSON path")

    args = parser.parse_args()
    return DebateFLOPsConfig(
        pickle_path=args.pickle_path,
        interrogator_model=args.interrogator_model,
        responder_model=args.responder_model,
        num_debates=args.num_debates,
        seed=args.seed,
        max_seq_length=args.max_seq_length,
        device=args.device,
        torch_dtype=args.torch_dtype,
        run_index=args.run_index,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    cfg = _parse_args()
    main(cfg)
