"""DebateGuardrailScorer - GuardrailScorer for the LLM debate system."""

from __future__ import annotations

import collections
import concurrent.futures
import logging
import pathlib
import threading
import typing

import yaml

import pyine.evals.correctness.types as correctness_types
import pyine.prompts.manager
from pyine.guardrails.llm_debate.graph import build_debate_graph  # type: ignore[reportUnknownVariableType]
from pyine.guardrails.llm_debate.types import DebateRole, DebateTranscript, DebateVerdict

if typing.TYPE_CHECKING:
    from pyine.guardrails.llm_debate.configs import DebateGuardrailConfig

logger = logging.getLogger(__name__)


class DebateGuardrailScorer:
    """GuardrailScorer for the LLM debate system.

    Uses a multi-turn debate between an interrogator and a responder
    to judge whether a model's code execution prediction is correct.
    Returns continuous confidence scores (0-1) and tracks token costs per record.

    Concurrency is achieved via ThreadPoolExecutor with sync graph.invoke()
    calls, which is safe because LangGraph compiled graphs are stateless.
    """

    def __init__(self, config: DebateGuardrailConfig) -> None:
        """Initializes the scorer (constructing LLM chains and debate graph)."""
        self._config = config

        # Build interrogator LLM
        self._interrogator_llm = config.interrogator_provider.get_model()
        # Build responder LLM
        self._responder_llm = config.responder_provider.get_model()

        # Derive prompt version from include_reasoning config
        interrogator_version = None if config.include_reasoning else "no_reasoning"

        # Build prompt chains via PromptManager
        self._interrogator_chain = pyine.prompts.manager.get_prompt_chain(
            model=self._interrogator_llm,
            prompt_name=config.interrogator_prompt_name,
            version=interrogator_version,
            use_chat_template=config.use_chat_template,
            runnable_name="debate_interrogator",
        )
        self._responder_chain = pyine.prompts.manager.get_prompt_chain(
            model=self._responder_llm,
            prompt_name=config.responder_prompt_name,
            use_chat_template=config.use_chat_template,
            runnable_name="debate_responder",
        )

        # Build verdict chain for forced-verdict turn
        self._interrogator_verdict_chain = pyine.prompts.manager.get_prompt_chain(
            model=self._interrogator_llm,
            prompt_name=config.interrogator_verdict_prompt_name,
            version=interrogator_version,
            use_chat_template=config.use_chat_template,
            runnable_name="debate_interrogator_verdict",
        )

        # Apply chain-level retries if configured
        if self._config.chain_retry_max_attempts > 0:
            retry_kwargs = self._build_chain_retry_kwargs(self._config)
            self._interrogator_chain = self._interrogator_chain.with_retry(**retry_kwargs)
            self._responder_chain = self._responder_chain.with_retry(**retry_kwargs)
            self._interrogator_verdict_chain = self._interrogator_verdict_chain.with_retry(**retry_kwargs)

        # Build and compile LangGraph debate graph
        self._graph = build_debate_graph(  # type: ignore[reportUnknownMemberType]
            self._interrogator_chain,
            self._responder_chain,
            interrogator_verdict_chain=self._interrogator_verdict_chain,
            responder_sees_debate_history=config.responder_sees_debate_history,
        )

        # Error tracking (same pattern as prompted_llm)
        self._error_count: int = 0
        self._error_lock = threading.Lock()
        self._total_scored: int = 0

        # Debug transcript logging
        self._debug_log_counter: int = 0
        self._debug_lock = threading.Lock()

        # YAML transcript export
        self._yaml_output_dir: pathlib.Path | None = None
        if config.debate_output_dir is not None:
            self._yaml_output_dir = pathlib.Path(config.debate_output_dir)
            self._yaml_output_dir.mkdir(parents=True, exist_ok=True)
        self._yaml_counters: dict[str, int] = collections.defaultdict(int)
        self._yaml_lock = threading.Lock()

    def score_records(
        self,
        records: list[correctness_types.EvalRecord],
    ) -> correctness_types.ScoringResult:
        """Score records via ThreadPoolExecutor (same pattern as prompted_llm)."""
        timeout = self._config.debate_timeout_seconds
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._config.max_workers,
        ) as executor:
            future_to_idx = {executor.submit(self._score_single, record): idx for idx, record in enumerate(records)}
            results: list[tuple[float, float, dict[str, typing.Any]] | None] = [None] * len(records)
            completed = 0
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    record = records[idx]
                    logger.warning(
                        "debate timed out after %ss for record %s, using default score",
                        timeout,
                        record.sample_id,
                    )
                    with self._error_lock:
                        self._error_count += 1
                    results[idx] = (
                        self._config.default_score_on_error,
                        0.0,
                        {"skipped": True, "reason": f"timeout after {timeout}s"},
                    )
                completed += 1  # noqa: SIM113 - as_completed() doesn't support enumerate
                if completed % 100 == 0 or completed == len(records):
                    logger.info("scored %d/%d records", completed, len(records))

        scores: list[float] = []
        costs: list[float] = []
        attempt_metadata: dict[correctness_types.ScoredAttemptKey, dict[str, typing.Any]] = {}
        assert len(results) == len(records), "expected as many results as records"
        for res_idx, res_triplet in enumerate(results):
            assert res_triplet is not None, "all futures should have completed successfully"
            assert isinstance(res_triplet, tuple) and len(res_triplet) == 3, "expected (score, token_count, transcript)"
            scores.append(res_triplet[0])
            costs.append(res_triplet[1])
            scored_attempt_key = (records[res_idx].sample_id, records[res_idx].attempt_index, res_idx)
            attempt_metadata[scored_attempt_key] = res_triplet[2]

        self._total_scored += len(records)
        return correctness_types.ScoringResult(
            scores=scores,
            verification_costs=costs,
            attempt_metadata=attempt_metadata,
        )

    def _score_single(
        self,
        record: correctness_types.EvalRecord,
    ) -> tuple[float, float, dict[str, typing.Any]]:
        """Run the full debate for one record.

        Returns (score, total_token_count, transcript_as_dict).
        Called from worker threads.
        """
        prompt_messages = record.record.get("prompt_messages")
        assert prompt_messages is not None, "prompt_messages is required in the EvalRecord for debate scoring"
        assert isinstance(prompt_messages, list), "prompt_messages must be a list of message dicts"

        if record.final_answer is None:
            logger.warning(
                "record %s has no final_answer (output parsing failed); skipping debate, using default score",
                record.sample_id,
            )
            return (
                self._config.default_score_on_error,
                0.0,
                {"skipped": True, "reason": "final_answer is None"},
            )

        # Format chat messages into a readable string for the debate prompt
        original_prompt = self._format_prompt_messages(prompt_messages)  # type: ignore[reportUnknownArgumentType]

        from pyine.guardrails.llm_debate.graph import DebateState

        initial_state: DebateState = {
            "original_prompt": original_prompt,
            "responder_output": record.model_output,
            "final_answer": record.final_answer or "",
            "max_turns": self._config.max_debate_turns,
            "messages": [],
            "current_turn": 0,
            "verdict": None,
            "total_tokens": 0.0,
        }

        try:
            # Invoke compiled graph (no top-level callbacks - nodes use their own)
            final_state = self._graph.invoke(initial_state)  # type: ignore[reportUnknownMemberType]

            verdict = final_state["verdict"]
            if verdict is None:
                logger.warning(
                    "debate ended without a verdict for record %s, using default score",
                    record.sample_id,
                )
                verdict = DebateVerdict(score=self._config.default_score_on_error)
                with self._error_lock:
                    self._error_count += 1

            transcript = DebateTranscript(
                messages=final_state["messages"],
                verdict=verdict,
                num_turns=final_state["current_turn"],
                total_token_count=final_state["total_tokens"],
            )

            score = max(0.0, min(1.0, transcript.verdict.score))

            # Debug transcript logging (thread-safe)
            if self._config.debug_log_transcript_every_n > 0:
                should_log_debug = False
                with self._debug_lock:
                    self._debug_log_counter += 1
                    should_log_debug = self._debug_log_counter % self._config.debug_log_transcript_every_n == 0
                    counter_val = self._debug_log_counter
                if should_log_debug:
                    logger.info(
                        "Debug transcript #%d:\n%s",
                        counter_val,
                        self._format_transcript_for_log(
                            transcript,
                            sample_id=record.sample_id,
                            responder_sees_debate_history=self._config.responder_sees_debate_history,
                            soft_match_label=record.label,
                            expected_output=record.expected_output,
                            final_answer=record.final_answer,
                        ),
                    )

            # YAML transcript export (thread-safe)
            if self._yaml_output_dir is not None:
                self._write_debate_yaml(record, transcript, score, original_prompt)

            return score, transcript.total_token_count, transcript.model_dump()

        except Exception as exc:
            logger.warning(
                "Debate failed for record %s, using default score",
                record.sample_id,
                exc_info=True,
            )
            with self._error_lock:
                self._error_count += 1
            return (
                self._config.default_score_on_error,
                0.0,
                {
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )

    @staticmethod
    def _format_prompt_messages(prompt_messages: list[dict[str, str]]) -> str:
        """Format a list of chat message dicts into a readable string.

        Each message dict has 'role' and 'content' keys. The output is a
        concatenation of ``[ROLE]: content`` blocks, which becomes the
        ``original_prompt`` field in the debate state.
        """
        assert len(prompt_messages) > 0, "prompt_messages must not be empty"
        roles = {msg.get("role") for msg in prompt_messages}
        assert roles != {"assistant"}, "prompt_messages must not contain only assistant messages"

        parts: list[str] = []
        for msg in prompt_messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            parts.append(f"[{role}]:\n{content}")
        return "\n\n".join(parts)

    @staticmethod
    def _format_transcript_for_log(
        transcript: DebateTranscript,
        sample_id: str,
        responder_sees_debate_history: bool,
        soft_match_label: bool | None = None,
        expected_output: str | None = None,
        final_answer: str | None = None,
    ) -> str:
        """Format a debate transcript for debug logging.

        Produces a human-readable block suitable for terminal inspection.
        """
        sep = "\u2500" * 72
        lines = [
            sep,
            f"  DEBATE TRANSCRIPT \u2014 sample_id={sample_id}",
            f"  turns={transcript.num_turns}  tokens={transcript.total_token_count:.0f}  "
            f"history_visible={responder_sees_debate_history}",
        ]
        if soft_match_label is not None:
            lines.append(f"  soft_match={soft_match_label}")
        if expected_output is not None:
            lines.append(f"  expected_output={expected_output}")
        if final_answer is not None:
            lines.append(f"  final_answer={final_answer}")
        lines.append(sep)
        for i, msg in enumerate(transcript.messages):
            role_label = "INTERROGATOR" if msg.role == DebateRole.INTERROGATOR else "RESPONDER"
            lines.append(f"  [{role_label}] (turn message {i + 1}, {msg.token_count:.0f} tok)")
            for content_line in msg.content.strip().splitlines():
                lines.append(f"    {content_line}")
            lines.append("")
        # Verdict
        v = transcript.verdict
        lines.append(f"  VERDICT: score={v.score:.3f}")
        if v.reasoning:
            lines.append(f"  REASONING: {v.reasoning}")
        lines.append(sep)
        return "\n".join(lines)

    def _write_debate_yaml(
        self,
        record: correctness_types.EvalRecord,
        transcript: DebateTranscript,
        score: float,
        original_prompt: str,
    ) -> None:
        """Write a single debate transcript to a YAML file (thread-safe)."""
        assert self._yaml_output_dir is not None
        code_type = record.code_type
        with self._yaml_lock:
            self._yaml_counters[code_type] += 1
            counter = self._yaml_counters[code_type]

        filename = f"{code_type}_{counter:05d}.yaml"
        doc: dict[str, typing.Any] = {
            "sample_id": record.sample_id,
            "problem_id": record.problem_id,
            "code_type": code_type,
            "label": record.label,
            "score": score,
            "original_prompt": original_prompt,
            "expected_output": record.expected_output,
            "final_answer": record.final_answer,
            "model_output": record.model_output,
            "num_turns": transcript.num_turns,
            "total_token_count": transcript.total_token_count,
            "verdict": {
                "score": transcript.verdict.score,
                "reasoning": transcript.verdict.reasoning,
            },
            "debate_messages": [
                {
                    "role": msg.role.value,
                    "token_count": msg.token_count,
                    "content": msg.content,
                }
                for msg in transcript.messages
            ],
        }
        yaml_path = self._yaml_output_dir / filename
        with open(yaml_path, "w") as f:
            yaml.dump(doc, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    @staticmethod
    def _build_chain_retry_kwargs(config: DebateGuardrailConfig) -> dict[str, typing.Any]:
        """Build LangChain with_retry kwargs from config fields."""
        import langchain_core.exceptions
        import openai

        return {
            "retry_if_exception_type": (
                langchain_core.exceptions.OutputParserException,
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.RateLimitError,
                openai.InternalServerError,
            ),
            "wait_exponential_jitter": config.chain_retry_wait_exponential_jitter,
            "stop_after_attempt": config.chain_retry_max_attempts,
        }

    def get_metadata(self) -> dict[str, typing.Any]:
        """Return debate guardrail metadata for reporting."""
        return {
            "scorer_type": "llm_debate",
            "interrogator_provider": self._config.interrogator_provider.provider,
            "interrogator_model_kwargs": self._config.interrogator_provider.model_kwargs,
            "responder_provider": self._config.responder_provider.provider,
            "responder_model_kwargs": self._config.responder_provider.model_kwargs,
            "max_debate_turns": self._config.max_debate_turns,
            "responder_sees_debate_history": self._config.responder_sees_debate_history,
            "max_workers": self._config.max_workers,
            "default_score_on_error": self._config.default_score_on_error,
            "interrogator_verdict_prompt_name": self._config.interrogator_verdict_prompt_name,
            "chain_retry_max_attempts": self._config.chain_retry_max_attempts,
            "chain_retry_wait_exponential_jitter": self._config.chain_retry_wait_exponential_jitter,
            "total_scored": self._total_scored,
            "error_count": self._error_count,
        }

    def get_verification_cost_unit(self) -> str | None:
        """Return 'tokens' as the verification cost unit."""
        return "tokens"
