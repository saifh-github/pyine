"""DebateGuardrailScorer - GuardrailScorer for the LLM debate system."""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import typing

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

        # Build prompt chains via PromptManager
        self._interrogator_chain = pyine.prompts.manager.get_prompt_chain(
            model=self._interrogator_llm,
            prompt_name=config.interrogator_prompt_name,
            use_chat_template=config.use_chat_template,
            runnable_name="debate_interrogator",
        )
        self._responder_chain = pyine.prompts.manager.get_prompt_chain(
            model=self._responder_llm,
            prompt_name=config.responder_prompt_name,
            use_chat_template=config.use_chat_template,
            runnable_name="debate_responder",
        )

        # Build and compile LangGraph debate graph
        self._graph = build_debate_graph(  # type: ignore[reportUnknownMemberType]
            self._interrogator_chain,
            self._responder_chain,
            responder_sees_debate_history=config.responder_sees_debate_history,
        )

        # Error tracking (same pattern as prompted_llm)
        self._error_count: int = 0
        self._error_lock = threading.Lock()
        self._total_scored: int = 0

        # Debug transcript logging
        self._debug_log_counter: int = 0
        self._debug_lock = threading.Lock()

    def score_records(
        self,
        records: list[correctness_types.EvalRecord],
    ) -> correctness_types.ScoringResult:
        """Score records via ThreadPoolExecutor (same pattern as prompted_llm)."""
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._config.max_workers,
        ) as executor:
            future_to_idx = {executor.submit(self._score_single, record): idx for idx, record in enumerate(records)}
            results: list[tuple[float, float, dict[str, typing.Any]] | None] = [None] * len(records)
            completed = 0
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
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
        prompt = record.record.get("prompt")
        assert prompt is not None, "prompt is required in the EvalRecord for debate scoring"
        assert record.final_answer is not None, "final_answer is required in the EvalRecord for debate scoring"

        from pyine.guardrails.llm_debate.graph import DebateState

        initial_state: DebateState = {
            "original_prompt": prompt,
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
                            transcript, record.sample_id, self._config.responder_sees_debate_history
                        ),
                    )

            return score, transcript.total_token_count, transcript.model_dump()

        except Exception:
            logger.warning(
                "Debate failed for record %s, using default score",
                record.sample_id,
                exc_info=True,
            )
            with self._error_lock:
                self._error_count += 1
            return self._config.default_score_on_error, 0.0, {}

    @staticmethod
    def _format_transcript_for_log(
        transcript: DebateTranscript,
        sample_id: str,
        responder_sees_debate_history: bool,
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
            sep,
        ]
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
            "total_scored": self._total_scored,
            "error_count": self._error_count,
        }

    def get_verification_cost_unit(self) -> str | None:
        """Return 'tokens' as the verification cost unit."""
        return "tokens"
