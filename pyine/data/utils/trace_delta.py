import dataclasses
import enum
import typing

import deepdiff

import pyine.utils.code_blocks
import pyine.utils.code_exec
import pyine.utils.portability


class EventRelationship(enum.StrEnum):
    """Relationships between consecutive trace events defining what happened during execution."""

    # @@@@ TODO: add control block relationships as well? (or as a STEP_INTO interpretation option when identifying relationships?)

    ENTRYPOINT = enum.auto()
    """Entrypoint to the traced code string, i.e. the first call to the traced function/method."""
    STEP_OVER = enum.auto()
    """Stepped from one line of code to another, but did not enter a new function/block."""
    STEP_INTO = enum.auto()
    """Stepped into a new function/block, which should contain more lines of code to trace."""
    STEP_OUT = enum.auto()
    """Stepped out of a function/block, potentially carrying a returned value or exception."""
    RAISE = enum.auto()
    """Raised an exception during execution, which will then be propagated to the caller."""
    PROPAGATE = enum.auto()
    """TODO??? @@@@@"""
    UNKNOWN = enum.auto()
    """TODO??? @@@@@"""


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
        if self.curr_trace_key.file == pyine.utils.code_exec.EXEC_TRACE_FILE_NAME:
            curr_trace_file_prefix = ""
        else:
            curr_trace_file_prefix = f"{self.curr_trace_key.file}:"
        if self.next_trace_key.file == pyine.utils.code_exec.EXEC_TRACE_FILE_NAME:
            next_trace_file_prefix = ""
        else:
            next_trace_file_prefix = f"{self.next_trace_key.file}:"
        return (
            f"{curr_trace_file_prefix}L{self.curr_trace_key.line:04d} -> {next_trace_file_prefix}L{self.next_trace_key.line:04d} "
            f"({self.event_relationship}) @ step#{self.trace_step_idx} = {self.variables_delta}"
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
    """Returns the relation between two trace events.

    Args:
        curr: the current trace event, i.e. the starting state of the delta.
        next: the next trace event, i.e. the ending state of the delta.
    """
    assert curr.trace_step_idx < next.trace_step_idx, "out-of-order trace events?"
    if (
        curr.event_type == pyine.utils.code_exec.TraceEventType.LINE
        and next.event_type == pyine.utils.code_exec.TraceEventType.LINE
    ):
        # going from one line to another is just a 'step over' delta (simplest type of delta)
        return EventRelationship.STEP_OVER
    if curr.event_type == pyine.utils.code_exec.TraceEventType.CALL:
        # the 'entrypoint' of the traced code; there should only be a single one of these
        # (we don't need to consider the next event type at all here, it will be reused)
        return EventRelationship.ENTRYPOINT
    if next.event_type == pyine.utils.code_exec.TraceEventType.CALL:
        # current event evaluates a function call, and next event moves the trace to that function
        # (the following event would correspond to the first executed line inside that function)
        assert curr.event_type == pyine.utils.code_exec.TraceEventType.LINE
        return EventRelationship.STEP_INTO
    if next.event_type == pyine.utils.code_exec.TraceEventType.RETURN:
        # if the next event of the pair is a return from a function call, then...
        assert curr.event_type in [
            pyine.utils.code_exec.TraceEventType.LINE,  # the current event might be one last eval
            pyine.utils.code_exec.TraceEventType.RETURN,  # or a cascading return from a prior call
        ]
        # note: this kind of event might be caused by a returned value, or by a raised exception
        return EventRelationship.STEP_OUT
    if (
        curr.event_type != pyine.utils.code_exec.TraceEventType.EXCEPTION
        and next.event_type == pyine.utils.code_exec.TraceEventType.EXCEPTION
    ):
        # current event caused an exception to be raised, which will be propagated until caught
        return EventRelationship.RAISE
    if (
        curr.event_type == pyine.utils.code_exec.TraceEventType.EXCEPTION
        and next.event_type != pyine.utils.code_exec.TraceEventType.EXCEPTION
    ):
        # an event is being propagated to parent callers until caught
        return EventRelationship.PROPAGATE
    # default catch: (if this happens, we need to figure out why, and fix the delta pairing logic)
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
    trace_res: pyine.utils.code_exec.TraceResult,
    delta_generator: DeltaGeneratorType,
    write_with_debug_info: bool = False,
) -> list[TraceDelta]:
    """TODO: debug first, then document this stuff"""
    relevant_step_idxs = _filter_relevant_source_trace_step_idxs(trace_res)
    assert len(relevant_step_idxs) > 1
    assert delta_generator in DeltaGeneratorType
    if delta_generator == DeltaGeneratorType.SIMPLE:
        delta_generator = simple_delta_generator
    else:
        delta_generator = deepdiff.DeepDiff
    output_deltas = []
    iter_idx = 0
    call_vars_stack = []
    found_entrypoint = False
    # using a while loop with a manually-handled index-based iterator to skip steps as needed
    while iter_idx < len(relevant_step_idxs) - 1:  # we create deltas in pairs, so stop before last
        curr_step_idx = relevant_step_idxs[iter_idx]  # first element of the pair
        curr_step: pyine.utils.code_exec.TraceEvent = trace_res.traced_steps[curr_step_idx]
        assert curr_step is not None and curr_step.trace_step_idx == curr_step_idx
        assert curr_step.trace_key.file == pyine.utils.code_exec.EXEC_TRACE_FILE_NAME
        next_step_idx = relevant_step_idxs[iter_idx + 1]  # second element of the pair
        next_step: pyine.utils.code_exec.TraceEvent = trace_res.traced_steps[next_step_idx]
        assert next_step is not None and next_step.trace_step_idx > curr_step_idx
        assert next_step.trace_key.file == pyine.utils.code_exec.EXEC_TRACE_FILE_NAME
        assert curr_step.trace_step_idx < next_step.trace_step_idx
        delta = {}
        # the kind of step delta(s) we create will depend on the relationship between the two steps
        relationship = _get_relation_between_events(curr_step, next_step)
        # print(f"\ncurr_step (#{curr_step_idx}): {curr_step}")
        # print(f"next_step (#{next_step_idx}): {next_step}")
        # print(f"relationship: {relationship}")
        if relationship == EventRelationship.ENTRYPOINT:
            # special handling: the 'entrypoint' of the traced code should be unique
            assert not found_entrypoint, "there should only be one entrypoint per trace"
            found_entrypoint = True
            assert (
                curr_step.exception is None and next_step.exception is None
            ), "why is an exception being raised at entrypoint?"
            # when found, initialize the call stack with something useful for the final return
            parent_caller_trace_key = pyine.utils.code_exec.TraceKey(
                # fill these with arbitrary but unique values to be able to easily identify it
                file=pyine.utils.code_exec.EXEC_PARENT_FILE_NAME,
                line=0,
                object="<module>",
            )
            call_vars_stack.append((parent_caller_trace_key, curr_step.arguments))
            # ... and create 1st delta going from parent execution/caller to the traced code itself
            delta["__call__"] = curr_step.trace_key
            delta["__args__"] = next_step.arguments
            output_deltas.append(
                TraceDelta(
                    curr_trace_key=parent_caller_trace_key,
                    next_trace_key=next_step.trace_key,
                    trace_step_idx=next_step.trace_step_idx,
                    exception=next_step.exception,
                    variables_delta=delta,
                    event_relationship=relationship,
                    curr_step=curr_step if write_with_debug_info else None,
                    next_step=next_step if write_with_debug_info else None,
                )
            )
            iter_idx += 1  # we only skip one event (the initial call) and reuse the 'next' one
            continue
        if relationship == EventRelationship.STEP_OUT:
            # special handling: execution returned from a function (with a value or exception)
            # decompose this specific event into two deltas, to make sure we capture all info
            delta["__return__"] = next_step.return_value
            delta["__exception__"] = next_step.exception
            assert len(call_vars_stack) > 0, "return events should always be paired with a call?"
            caller_trace_key, caller_vars = call_vars_stack.pop()
            if caller_trace_key.file == pyine.utils.code_exec.EXEC_PARENT_FILE_NAME:
                # TODO @@@@ assert below might fail sometimes (exception situations?), fixme
                # if the caller is the execution parent, it means the stack should be empty
                assert len(call_vars_stack) == 0 and len(next_step.stack_trace) == 1
            else:
                # otherwise, the caller's identity should match the one expected in the stack
                assert caller_trace_key == next_step.stack_trace[1]
            # we'll first create a delta for the 'current line' which evaluates and returns/raises
            output_deltas.append(
                TraceDelta(
                    curr_trace_key=next_step.trace_key,
                    next_trace_key=caller_trace_key,
                    trace_step_idx=next_step.trace_step_idx,
                    exception=next_step.exception,
                    variables_delta=delta,
                    event_relationship=relationship,
                    curr_step=curr_step if write_with_debug_info else None,
                    next_step=next_step if write_with_debug_info else None,
                )
            )
            iter_idx += 1
            continue
        # all remaining cases only produce a single delta that captures all potential information
        if relationship == EventRelationship.STEP_OVER:
            # going from one line to another is just a 'step over' delta (simplest case)
            delta = delta_generator(curr_step.variables, next_step.variables)
        elif relationship == EventRelationship.STEP_INTO:
            # calling a function: specify the target function + args, and stack the vars context
            assert next_step.exception is None, "why step into a function if an exception is being raised?"
            delta["__call__"] = next_step.trace_key
            delta["__args__"] = next_step.arguments
            call_vars_stack.append((curr_step.trace_key, curr_step.variables))
            # there will be a useless trace event following all calls, skip it, but keep its key for next
            iter_idx += 1
            assert iter_idx < len(relevant_step_idxs) - 1, "trace ends with a function call??"
            next_next_step_idx = relevant_step_idxs[iter_idx + 1]
            next_next_step: pyine.utils.code_exec.TraceEvent = trace_res.traced_steps[next_next_step_idx]
            assert next_next_step is not None and next_next_step.trace_step_idx > next_step_idx
            assert next_next_step.exception is None, "why step into a function if an exception is being raised?"
            next_step = next_next_step
        elif relationship == EventRelationship.RAISE or relationship == EventRelationship.PROPAGATE:
            # @@@@@ TODO: do something with call vars stack?
            delta["__exception__"] = next_step.exception._asdict()  # noqa
        else:
            # @@@@@ TODO: figure out if we just ignore these leftover deltas
            raise NotImplementedError
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


def _filter_relevant_source_trace_step_idxs(
    trace_res: pyine.utils.code_exec.TraceResult,
) -> list[int]:
    """Filters out irrelevant trace steps that are invalid or outside the proposed code string."""
    first_relevant_step_idx = 0
    if trace_res.entrypoint_step_idx is not None:
        first_entrypoint_trace_step = next(
            (v for v in trace_res.traced_steps[trace_res.entrypoint_step_idx :] if v is not None),
            None,
        )
        if first_entrypoint_trace_step is None:
            raise AssertionError("invalid entrypoint call step")  # fix if it happens? @@@@
        first_relevant_step_idx = first_entrypoint_trace_step.trace_step_idx
    raw_trace_steps: list[pyine.utils.code_exec.TraceEvent | None] = trace_res.traced_steps
    filtered_trace_step_idxs = []
    for trace_step in raw_trace_steps:
        if trace_step is None:
            continue  # step originates from blacklisted, internal, or compiled modules
        if trace_step.trace_step_idx < first_relevant_step_idx:
            continue  # trace step occurs before we begin tracing the actual algo exec
        trace_key = trace_step.trace_key
        if trace_key.file != pyine.utils.code_exec.EXEC_TRACE_FILE_NAME:
            continue  # step originates from a separate file instead of the input code string
        # if trace_step.event_type == "call" and trace_step.trace_step_idx == first_relevant_step_idx:
        #     continue  # step is the initial call of the algo execution (useless?)
        filtered_trace_step_idxs.append(trace_step.trace_step_idx)
    return filtered_trace_step_idxs


def debug_demo():
    example_snippet = """\
def calculate_area(length: float, width: float) -> float:
    '''Returns the area of the rectangle specified via length and width.

    Specifically: returns area = length * width.
    '''
    area = length * width
    print(f"The area of the rectangle is: {area:.2f} square units")
    return area

length = float(input("Enter the length: "))
width = float(input("Enter the width: "))
calculate_area(length, width)
"""
    example_input_args = """\
5.0
3.0
"""
    trace_result = pyine.utils.code_exec.execute_and_trace_code(
        example_snippet,
        example_input_args,
        trace_only_inside_code_string=True,
        max_events_per_line=100,
    )
    print("\nTraced code string:")
    pyine.utils.portability.print_code_with_numbered_lines(example_snippet, 1)
    print("\nTraced steps:")
    for traced_step_idx, traced_step in enumerate(trace_result.traced_steps):
        if traced_step is None:
            print(f"\tstep#{traced_step_idx:04d}:\t(out-of-context execution)")
        else:
            print(f"\tstep#{traced_step_idx:04d}:\t{traced_step}")
    if trace_result.return_value is not None:
        print(f"\nCaptured return value:\n\t{trace_result.return_value}")
    if trace_result.exception is not None:
        print(f"\nCaptured exception:\n\t{trace_result.exception}")
    if trace_result.stdout:
        print(f"\nCaptured output:\n\t{trace_result.stdout}")
    if trace_result.stderr:
        print(f"\nCaptured error:\n\t{trace_result.stderr}")

    deltas = get_deltas_from_trace_steps(
        trace_res=trace_result,
        delta_generator=DeltaGeneratorType.SIMPLE,
        write_with_debug_info=True,
    )
    assert deltas
    print(f"traced steps: {len(trace_result.traced_steps)}")
    print("code string:")
    pyine.utils.portability.print_code_with_numbered_lines(trace_result.code_string, 1)
    print(f"input args:\n\t{trace_result.inputs}")
    print(f"execution result:\n\t{trace_result.return_value}")
    print("deltas:")
    for delta_idx, delta in enumerate(deltas):
        print(f"\td#{delta_idx}:\t{delta}")


if __name__ == "__main__":
    debug_demo()
