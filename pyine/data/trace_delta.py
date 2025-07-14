import dataclasses
import enum
import typing

import deepdiff

import pyine.utils.code_blocks
import pyine.utils.code_exec
import pyine.utils.portability


class EventRelationship(enum.StrEnum):
    """Types of relationships between consecutive trace events."""

    # @@@@ TODO: add control block relationships as well? (or as a STEP_INTO interpretation option when identifying relationships?)
    ENTRYPOINT = enum.auto()
    STEP_OVER = enum.auto()
    STEP_INTO = enum.auto()
    STEP_OUT = enum.auto()
    RAISE = enum.auto()
    PROPAGATE = enum.auto()
    UNKNOWN = enum.auto()


@dataclasses.dataclass(frozen=True)
class TraceDelta:
    curr_trace_key: pyine.utils.code_exec.TraceKey
    """The trace key associated with the start of the delta."""
    next_trace_key: pyine.utils.code_exec.TraceKey
    """The trace key associated with the end of the delta."""
    trace_step_idx: int
    """The trace step index at the start of the delta; should be unique for each delta."""
    exception: dict[str, typing.Any] | None
    """A dictionary containing information about the exception being raised/propagated, if any."""
    variables_delta: dict[str, str]
    """A dictionary containing the added/removed/updated variables in the delta."""
    event_relationship: EventRelationship
    """The relationship between the start/end trace events."""

    def __repr__(self):
        """Returns a string representation of the trace delta."""
        if (
            self.curr_trace_key.file == pyine.utils.code_exec.EXEC_TRACE_FILE_NAME
            and self.next_trace_key.file == pyine.utils.code_exec.EXEC_TRACE_FILE_NAME
        ):
            return (
                f"L{self.curr_trace_key.line} -> L{self.next_trace_key.line} "
                f"({self.event_relationship}) @ step#{self.trace_step_idx} "
                f" = {self.variables_delta}"
            )
        else:
            return (
                f"{self.curr_trace_key.file}:L{self.curr_trace_key.line} -> {self.next_trace_key.file}:L{self.next_trace_key.line} "
                f"({self.event_relationship}) @ step#{self.trace_step_idx} "
                f" = {self.variables_delta}"
            )

    curr_step: pyine.utils.code_exec.TraceEvent | None = None
    """The current trace event (FOR DEBUGGING PURPOSES ONLY, DO NOT USE IN FINAL DATASET)."""
    next_step: pyine.utils.code_exec.TraceEvent | None = None
    """The next trace event (FOR DEBUGGING PURPOSES ONLY, DO NOT USE IN FINAL DATASET)."""


class DeltaGeneratorType(enum.StrEnum):
    """Supported variable state delta generation approaches."""

    SIMPLE = enum.auto()
    DEEPDIFF = enum.auto()


def _get_relation_between_events(
    curr: pyine.utils.code_exec.TraceEvent,
    next: pyine.utils.code_exec.TraceEvent,
) -> EventRelationship:
    """Returns the relation between two trace events."""
    assert curr.trace_step_idx < next.trace_step_idx, "out-of-order trace events?"
    assert (
        curr.event_type != pyine.utils.code_exec.TraceEventType.RETURN
    ), "return events should not precede other events"
    if (
        curr.event_type == pyine.utils.code_exec.TraceEventType.LINE
        and next.event_type == pyine.utils.code_exec.TraceEventType.LINE
    ):
        return EventRelationship.STEP_OVER
    if curr.event_type == pyine.utils.code_exec.TraceEventType.CALL:
        return EventRelationship.ENTRYPOINT  # there should only ever be one of these
    if next.event_type == pyine.utils.code_exec.TraceEventType.CALL:
        assert curr.event_type == pyine.utils.code_exec.TraceEventType.LINE
        return EventRelationship.STEP_INTO
    if next.event_type == pyine.utils.code_exec.TraceEventType.RETURN:
        assert curr.event_type in [
            pyine.utils.code_exec.TraceEventType.LINE,
            pyine.utils.code_exec.TraceEventType.RETURN,
        ]
        return EventRelationship.STEP_OUT
    if (
        curr.event_type != pyine.utils.code_exec.TraceEventType.EXCEPTION
        and next.event_type == pyine.utils.code_exec.TraceEventType.EXCEPTION
    ):
        return EventRelationship.RAISE
    if (
        curr.event_type == pyine.utils.code_exec.TraceEventType.EXCEPTION
        and next.event_type != pyine.utils.code_exec.TraceEventType.EXCEPTION
    ):
        return EventRelationship.PROPAGATE
    return EventRelationship.UNKNOWN


def simple_delta_generator(curr: dict[str, str], next: dict[str, str]) -> dict[str, str]:
    """Compares two variable dicts and returns added/updated entries."""
    # note: we purposefully do NOT show missing/removed values in deltas to reduce useless spam/outputs
    # @@@@@@ TODO: figure out if we should also add the delta of stderr/stdout?
    output = {}
    for key in next.keys() - curr.keys():  # keys added to next
        output[key] = next[key]
    for key in curr.keys() & next.keys():  # keys updated in next
        if curr[key] != next[key]:
            output[key] = next[key]
    return output


def get_deltas_from_trace_steps(
    relevant_step_idxs: list[pyine.utils.code_exec.TraceEvent],
    trace_res: pyine.utils.code_exec.TraceResult,
    delta_generator: DeltaGeneratorType,
    write_with_debug_info: bool = False,
) -> list[TraceDelta]:
    assert len(relevant_step_idxs) > 1
    assert delta_generator in DeltaGeneratorType
    if delta_generator == DeltaGeneratorType.SIMPLE:
        delta_generator = simple_delta_generator
    else:
        delta_generator = deepdiff.DeepDiff
    output_deltas = []
    iter_idx = 0
    call_vars_stack = []
    while iter_idx < len(relevant_step_idxs) - 1:
        curr_step_idx = relevant_step_idxs[iter_idx]
        curr_step: pyine.utils.code_exec.TraceEvent = trace_res.traced_steps[curr_step_idx]
        assert curr_step is not None and curr_step.trace_step_idx == curr_step_idx
        assert curr_step.trace_key.file == pyine.utils.code_exec.EXEC_TRACE_FILE_NAME
        next_step_idx = relevant_step_idxs[iter_idx + 1]
        next_step: pyine.utils.code_exec.TraceEvent = trace_res.traced_steps[next_step_idx]
        assert next_step is not None and next_step.trace_step_idx > curr_step_idx
        assert next_step.trace_key.file == pyine.utils.code_exec.EXEC_TRACE_FILE_NAME
        assert curr_step.trace_step_idx < next_step.trace_step_idx
        relationship = _get_relation_between_events(curr_step, next_step)
        delta = {}
        if relationship == EventRelationship.ENTRYPOINT:
            call_vars_stack.append(
                (
                    pyine.utils.code_exec.TraceKey(
                        # fill these with arbitrary but unique values to be able to easily identify it
                        file=pyine.utils.code_exec.EXEC_PARENT_FILE_NAME,
                        line=-1,
                        object="<module>",
                    ),
                    curr_step.arguments,
                )
            )
            iter_idx += 1
        elif relationship == EventRelationship.STEP_OUT:
            # decompose this specific event into two deltas, to make sure we capture everything
            delta["__return__"] = next_step.return_value
            caller_trace_key, caller_vars = call_vars_stack.pop()
            if caller_trace_key.file == pyine.utils.code_exec.EXEC_PARENT_FILE_NAME:
                assert len(call_vars_stack) == 0 and len(next_step.stack_trace) == 1
            else:
                assert caller_trace_key == next_step.stack_trace[1]
            output_deltas.append(
                TraceDelta(
                    curr_trace_key=curr_step.trace_key,
                    next_trace_key=caller_trace_key,
                    trace_step_idx=curr_step.trace_step_idx,
                    exception=curr_step.exception,
                    variables_delta=delta,
                    event_relationship=relationship,
                    curr_step=curr_step if write_with_debug_info else None,
                    next_step=next_step if write_with_debug_info else None,
                )
            )
            iter_idx += 1
            if iter_idx < len(relevant_step_idxs) - 1:
                # this is not the final return call, so add another STEP_OVER delta to capture the next line change
                next_next_step_idx = relevant_step_idxs[iter_idx + 1]
                next_next_step: pyine.utils.code_exec.TraceEvent = trace_res.traced_steps[next_next_step_idx]
                assert next_next_step is not None and next_next_step.trace_step_idx > next_step_idx
                assert next_next_step.trace_key.file == pyine.utils.code_exec.EXEC_TRACE_FILE_NAME
                assert curr_step.trace_step_idx < next_next_step.trace_step_idx
                delta = delta_generator(caller_vars, next_next_step.variables)
                output_deltas.append(
                    TraceDelta(
                        curr_trace_key=caller_trace_key,
                        next_trace_key=next_next_step.trace_key,
                        trace_step_idx=next_step.trace_step_idx,
                        exception=next_step.exception,
                        variables_delta=delta,
                        event_relationship=EventRelationship.STEP_OVER,
                        curr_step=next_step if write_with_debug_info else None,
                        next_step=next_next_step if write_with_debug_info else None,
                    )
                )
                iter_idx += 1
        else:
            # all cases herein only produce a single delta that captures all potential information
            if relationship == EventRelationship.STEP_OVER:
                delta = delta_generator(curr_step.variables, next_step.variables)
            elif relationship == EventRelationship.STEP_INTO:
                delta["__call__"] = next_step.trace_key.object
                delta["__args__"] = next_step.arguments
                call_vars_stack.append((curr_step.trace_key, curr_step.variables))
                iter_idx += 1  # there will be a useless trace event following all calls, skip it?
            elif relationship == EventRelationship.RAISE or relationship == EventRelationship.PROPAGATE:
                # @@@@@ TODO: do something with call vars stack?
                delta["__exception__"] = next_step.exception._asdict()  # noqa
            else:
                # @@@@@ TODO: figure out if we just ignore these leftover deltas
                pass
            output_deltas.append(
                TraceDelta(
                    curr_trace_key=curr_step.trace_key,
                    next_trace_key=next_step.trace_key,
                    trace_step_idx=curr_step.trace_step_idx,
                    exception=next_step.exception,
                    variables_delta=delta,
                    event_relationship=relationship,
                    curr_step=curr_step if write_with_debug_info else None,
                    next_step=next_step if write_with_debug_info else None,
                )
            )
            iter_idx += 1
    return output_deltas
