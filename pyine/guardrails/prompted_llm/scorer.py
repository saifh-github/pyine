"""PromptedLLMGuardrailScorer — GuardrailScorer for prompted (non-fine-tuned) LLMs."""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import typing

import pyine.evals.correctness.types as correctness_types
import pyine.prompts.manager
import pyine.utils.langchain

if typing.TYPE_CHECKING:
    from pyine.guardrails.prompted_llm.configs import PromptedLLMGuardrailConfig

logger = logging.getLogger(__name__)


class PromptedLLMGuardrailScorer:
    """GuardrailScorer adapter for a prompted (non-fine-tuned) LLM judge.

    Uses a LangChain chain (prompt | model | parser) to ask an LLM to judge
    whether a model's code execution prediction is correct. Returns continuous
    confidence scores (0–1) and tracks token costs per record.

    Concurrency is achieved via ThreadPoolExecutor with sync chain.invoke()
    calls, which is safe to call from within an already-running asyncio event
    loop (unlike asyncio.run(), which would crash).
    """

    def __init__(self, config: PromptedLLMGuardrailConfig) -> None:
        """Initializes the scorer (constructing the LLM prompting chain)."""
        self._config = config
        self._llm = config.llm_provider.get_model()
        self._chain = pyine.prompts.manager.get_prompt_chain(
            model=self._llm,
            prompt_name=config.prompt_name,
            use_chat_template=config.use_chat_template,
            runnable_name="correctness_judge",
        )
        self._error_count: int = 0
        self._error_lock = threading.Lock()
        self._total_scored: int = 0

    def score_records(
        self,
        records: list[correctness_types.EvalRecord],
    ) -> correctness_types.ScoringResult:
        """Score records by asking an LLM to judge correctness.

        Uses ThreadPoolExecutor for concurrent I/O-bound LLM API calls.
        This is safe to call from within a running asyncio event loop
        (the eval pipeline's evaluate_wrapped_model is async).
        """
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._config.max_workers,
        ) as executor:
            future_to_idx = {executor.submit(self._score_single, record): idx for idx, record in enumerate(records)}
            results: list[tuple[float, float, str | None] | None] = [None] * len(records)
            completed = 0
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
                completed += 1  # noqa: SIM113 — as_completed() doesn't support enumerate
                if completed % 100 == 0 or completed == len(records):
                    logger.info("scored %d/%d records", completed, len(records))

        scores: list[float] = []
        costs: list[float] = []
        attempt_metadata: dict[correctness_types.ScoredAttemptKey, dict[str, typing.Any]] = {}
        assert len(results) == len(records), "expected as many results as records"
        for res_idx, res_triplet in enumerate(results):
            assert res_triplet is not None, "all futures should have completed successfully"
            assert isinstance(res_triplet, tuple) and len(res_triplet) == 3, "expected (score, token_count, reasoning)"
            scores.append(res_triplet[0])
            costs.append(res_triplet[1])
            reasoning = res_triplet[2] or ""
            assert isinstance(reasoning, str)
            scored_attempt_key = (records[res_idx].sample_id, records[res_idx].attempt_index, res_idx)
            attempt_metadata[scored_attempt_key] = {
                "reasoning": reasoning,
                # if we had any extra metadata about each attempt to provide, this is where to put it
            }

        self._total_scored += len(records)
        return correctness_types.ScoringResult(
            scores=scores,
            verification_costs=costs,
            attempt_metadata=attempt_metadata,
        )

    def _score_single(
        self,
        record: correctness_types.EvalRecord,
    ) -> tuple[float, float, str | None]:
        """Score a single record via sync chain.invoke(). Returns (score, token_count, reasoning).

        Called from worker threads. Error counting is protected by a threading.Lock.
        """
        prompt = record.record.get("prompt")
        assert prompt is not None, "prompt is required in the EvalRecord for prompted LLM scoring"
        assert record.final_answer is not None, "final_answer is required in the EvalRecord for prompted LLM scoring"
        input_vars: dict[str, typing.Any] = {
            "prompt": prompt,
            "model_output": record.model_output,
            "final_answer": record.final_answer,
        }

        handler = pyine.utils.langchain.CaptureLLMHandler()
        reasoning: str | None = None

        try:
            result: typing.Any = self._chain.invoke(
                input_vars,
                config={"callbacks": [handler]},
            )
            # extract score from structured output (Pydantic model or dict)
            if hasattr(result, "score"):
                score = float(result.score)
            elif isinstance(result, dict) and "score" in result:
                result_dict = typing.cast("dict[str, typing.Any]", result)
                score = float(result_dict["score"])
            else:
                result_type_name = type(typing.cast("typing.Any", result)).__name__
                logger.warning(f"LLM returned unexpected format for {record.sample_id}: {result_type_name}")
                score = self._config.default_score_on_error
                with self._error_lock:
                    self._error_count += 1
            score = max(0.0, min(1.0, score))

            # try to extract the reasoning from the structured output (there might not be any)
            if hasattr(result, "reasoning") and result.reasoning is not None:  # type: ignore
                reasoning = typing.cast("str", result.reasoning)  # type: ignore
            elif isinstance(result, dict) and "reasoning" in result and result["reasoning"] is not None:
                reasoning = typing.cast("str", result["reasoning"])
            if reasoning is not None and not isinstance(reasoning, str):  # type: ignore
                reasoning_type_name = type(reasoning).__name__
                logger.warning(f"LLM returned unexpected type for reasoning: {reasoning_type_name}")
                reasoning = None

        except Exception:
            logger.warning(
                "LLM call failed for record %s, using default score",
                record.sample_id,
                exc_info=True,
            )
            score = self._config.default_score_on_error
            with self._error_lock:
                self._error_count += 1

        token_count = self._extract_token_count(handler)
        return score, token_count, reasoning

    @staticmethod
    def _extract_token_count(
        handler: pyine.utils.langchain.CaptureLLMHandler,
    ) -> float:
        """Extract total token count from the capture handler."""
        end_event = handler.get_latest_event("llm_end")
        if end_event is None or end_event.response is None:
            return 0.0
        # LLMResult.llm_output is typed as bare Optional[dict] in langchain (no type params)
        llm_output: dict[str, typing.Any] = getattr(end_event.response, "llm_output", None) or {}
        token_usage: dict[str, typing.Any] = llm_output.get("token_usage", {})
        total: int = token_usage.get("total_tokens", 0)
        if total > 0:
            return float(total)
        # Fallback: sum from generation info
        for generation_list in end_event.response.generations:
            for gen in generation_list:
                gen_info: dict[str, typing.Any] = getattr(gen, "generation_info", None) or {}
                usage: dict[str, typing.Any] = gen_info.get("usage", {})
                total += usage.get("total_tokens", 0)
        if total == 0:
            logger.debug("token count is 0 for a successful LLM response; provider may not populate token usage fields")
        return float(total)

    def get_metadata(self) -> dict[str, typing.Any]:
        """Return guardrail metadata for reporting."""
        return {
            "scorer_type": "prompted_llm",
            "prompt_name": self._config.prompt_name,
            "provider": self._config.llm_provider.provider,
            "model_kwargs": self._config.llm_provider.model_kwargs,
            "max_workers": self._config.max_workers,
            "default_score_on_error": self._config.default_score_on_error,
            "total_scored": self._total_scored,
            "error_count": self._error_count,
        }

    def get_verification_cost_unit(self) -> str | None:
        """Return 'tokens' as the verification cost unit."""
        return "tokens"
