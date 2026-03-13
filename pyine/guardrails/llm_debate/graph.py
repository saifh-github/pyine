"""LangGraph state machine for the LLM debate guardrail."""

from __future__ import annotations

import logging
import operator
import typing

from langgraph.graph import END, StateGraph

import pyine.utils.langchain
from pyine.guardrails.llm_debate.types import DebateMessage, DebateRole, DebateVerdict

if typing.TYPE_CHECKING:
    import langchain_core.runnables
    from langgraph.graph.state import CompiledStateGraph

    from pyine.prompts.configs.guardrail.debate_interrogator import InterrogatorOutput, VerdictOutput

logger = logging.getLogger(__name__)


class DebateState(typing.TypedDict):
    """LangGraph state for the debate.

    Uses Annotated reducers for fields that accumulate across node updates:
    - messages: operator.add appends new messages to the list (nodes return [new_msg])
    - total_tokens: operator.add sums token counts (nodes return the delta)
    All other fields are overwritten on each update (standard LangGraph behavior).
    """

    # Immutable context (set once at init, never updated by nodes)
    original_prompt: str
    responder_output: str
    final_answer: str
    max_turns: int

    # Mutable debate state (with reducers where needed)
    messages: typing.Annotated[list[DebateMessage], operator.add]
    current_turn: int
    verdict: DebateVerdict | None
    total_tokens: typing.Annotated[float, operator.add]


def _format_debate_history(messages: list[DebateMessage]) -> str:
    """Format debate messages into a readable history string."""
    if not messages:
        return "(no prior debate history)"
    lines: list[str] = []
    for msg in messages:
        role_label = "INTERROGATOR" if msg.role == DebateRole.INTERROGATOR else "RESPONDER"
        lines.append(f"[{role_label}]: {msg.content}")
    return "\n\n".join(lines)


def _extract_token_count(handler: pyine.utils.langchain.CaptureLLMHandler) -> float:
    """Extract total token count from a CaptureLLMHandler (delegates to shared utility)."""
    return pyine.utils.langchain.extract_token_count_from_handler(handler)


def build_debate_graph(  # type: ignore[reportUnknownParameterType]
    interrogator_chain: langchain_core.runnables.Runnable[typing.Any, typing.Any],
    responder_chain: langchain_core.runnables.Runnable[typing.Any, typing.Any],
    interrogator_verdict_chain: langchain_core.runnables.Runnable[typing.Any, typing.Any] | None = None,
    responder_sees_debate_history: bool = True,
) -> CompiledStateGraph[typing.Any, typing.Any, typing.Any, typing.Any]:
    """Build and compile the debate state machine.

    Args:
        interrogator_chain: The interrogator's prompt chain (prompt | model | parser).
        responder_chain: The responder's prompt chain (prompt | model | parser).
        interrogator_verdict_chain: Optional verdict-only chain for the final
            forced-verdict turn. When provided, the interrogator uses this chain
            (which has no ``decision`` field) on the last turn instead of the
            regular interrogator chain. When None, falls back to the programmatic
            verdict override.
        responder_sees_debate_history: If True, the responder_turn node passes
            the full debate transcript as ``debate_history``. If False, passes
            an empty string so the responder only sees its original output + the
            latest interrogator question.

    Returns a compiled graph ready for invocation. Each node function takes
    DebateState, invokes the appropriate chain with a per-node CaptureLLMHandler,
    and returns a partial state update (LangGraph reducer pattern).

    The compiled graph is stateless -- all state is passed in via graph.invoke().
    """

    def interrogator_turn(state: DebateState) -> dict[str, typing.Any]:
        """The interrogator asks a question OR renders a verdict."""
        handler = pyine.utils.langchain.CaptureLLMHandler()
        debate_history = _format_debate_history(state["messages"])
        current_turn = state["current_turn"]
        max_turns = state["max_turns"]
        is_forced_verdict_turn = current_turn >= max_turns

        input_vars: dict[str, typing.Any] = {
            "original_prompt": state["original_prompt"],
            "responder_output": state["responder_output"],
            "final_answer": state["final_answer"],
            "debate_history": debate_history,
        }

        if is_forced_verdict_turn and interrogator_verdict_chain is not None:
            # Use verdict-only chain: no decision field, always a verdict
            verdict_result: VerdictOutput = interrogator_verdict_chain.invoke(
                input_vars,
                config={"callbacks": [handler]},
            )
            token_count = _extract_token_count(handler)
            score = max(0.0, min(1.0, float(verdict_result.score)))
            verdict = DebateVerdict(score=score, reasoning=verdict_result.reasoning)
            msg = DebateMessage(
                role=DebateRole.INTERROGATOR,
                content=verdict_result.content,
                token_count=token_count,
            )
            return {
                "messages": [msg],
                "total_tokens": token_count,
                "verdict": verdict,
            }

        # Normal turn: use regular interrogator chain (includes question/verdict decision)
        input_vars["current_turn"] = str(current_turn + 1)  # 1-indexed for the LLM prompt
        input_vars["max_turns"] = str(max_turns)

        result: InterrogatorOutput = interrogator_chain.invoke(
            input_vars,
            config={"callbacks": [handler]},
        )

        token_count = _extract_token_count(handler)

        # --- Programmatic fallback (retained for backward compat / safety) ---
        # If the verdict chain is not provided (interrogator_verdict_chain=None)
        # and the LLM disobeyed the forced-verdict instruction, fall back to
        # the default score. This path is also a safety net if a future caller
        # uses the graph without a verdict chain.
        if is_forced_verdict_turn and result.decision != "verdict":
            logger.warning(
                "Interrogator returned '%s' on forced-verdict turn %d/%d; overriding with default verdict (score=0.5)",
                result.decision,
                current_turn + 1,
                max_turns,
            )
            verdict = DebateVerdict(
                score=0.5,
                reasoning="Forced verdict: interrogator did not comply with verdict instruction.",
            )
            msg = DebateMessage(
                role=DebateRole.INTERROGATOR,
                content=result.content,
                token_count=token_count,
            )
            return {
                "messages": [msg],
                "total_tokens": token_count,
                "verdict": verdict,
            }

        if result.decision == "verdict":
            score = max(0.0, min(1.0, float(result.score if result.score is not None else 0.5)))
            verdict = DebateVerdict(score=score, reasoning=result.reasoning)
            msg = DebateMessage(
                role=DebateRole.INTERROGATOR,
                content=result.content,
                token_count=token_count,
            )
            return {
                "messages": [msg],
                "total_tokens": token_count,
                "verdict": verdict,
            }

        msg = DebateMessage(
            role=DebateRole.INTERROGATOR,
            content=result.content,
            token_count=token_count,
        )
        return {
            "messages": [msg],
            "total_tokens": token_count,
        }

    def responder_turn(state: DebateState) -> dict[str, typing.Any]:
        """The responder responds to the latest interrogator question."""
        current_turn = state["current_turn"]
        max_turns = state["max_turns"]
        if current_turn >= max_turns:
            raise RuntimeError(
                f"responder_turn called with current_turn={current_turn} >= max_turns={max_turns}; "
                "this indicates the interrogator failed to produce a verdict on the forced-verdict turn"
            )

        handler = pyine.utils.langchain.CaptureLLMHandler()

        # Get the latest interrogator question
        interrogator_question = ""
        for msg in reversed(state["messages"]):
            if msg.role == DebateRole.INTERROGATOR:
                interrogator_question = msg.content
                break

        # Conditionally include debate history based on config (captured via closure)
        if responder_sees_debate_history:
            debate_history = _format_debate_history(state["messages"])
        else:
            debate_history = ""

        input_vars: dict[str, typing.Any] = {
            "original_prompt": state["original_prompt"],
            "responder_output": state["responder_output"],
            "final_answer": state["final_answer"],
            "debate_history": debate_history,
            "interrogator_question": interrogator_question,
        }

        result: str = responder_chain.invoke(
            input_vars,
            config={"callbacks": [handler]},
        )

        token_count = _extract_token_count(handler)
        msg = DebateMessage(
            role=DebateRole.RESPONDER,
            content=result,
            token_count=token_count,
        )

        return {
            "messages": [msg],
            "total_tokens": token_count,
            "current_turn": current_turn + 1,
        }

    def after_interrogator(state: DebateState) -> str:
        """Route after interrogator: END if verdict set, else responder_turn."""
        if state.get("verdict") is not None:
            return END
        return "responder_turn"

    def after_responder(state: DebateState) -> str:
        """Route after responder: always back to interrogator_turn.

        The forced-verdict behavior is achieved by the interrogator prompt
        including the current_turn / max_turns fields.
        """
        return "interrogator_turn"

    # Build the graph
    graph = StateGraph(DebateState)
    graph.add_node("interrogator_turn", interrogator_turn)  # type: ignore[reportUnknownMemberType]
    graph.add_node("responder_turn", responder_turn)  # type: ignore[reportUnknownMemberType]

    graph.set_entry_point("interrogator_turn")
    graph.add_conditional_edges("interrogator_turn", after_interrogator)
    graph.add_conditional_edges("responder_turn", after_responder)

    return graph.compile()  # type: ignore[reportUnknownMemberType]
