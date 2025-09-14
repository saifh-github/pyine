import dataclasses
import datetime
import enum
import functools
import logging
import pathlib
import typing

import deepdiff
import pydantic

import pyine.data.common
import pyine.data.traces.dataset_utils
import pyine.utils.code.execution
import pyine.utils.filesystem
import pyine.utils.portability
import pyine.utils.reprod

__all__ = [
    "EventRelationship",
    "TraceDelta",
    "DeltaGeneratorType",
    "TraceDeltaList",
    "simple_delta_generator",
    "get_deltas_from_trace_steps",
    "get_latest_dataset_path",
    "get_matching_dataset_paths",
    "get_new_dataset_path",
]

SUPPORTED_SOURCE_DATASETS = pyine.data.traces.dataset_utils.SUPPORTED_SOURCE_DATASETS
"""The deltas datasets support the same source datasets as the traces datasets."""
DELTAS_SUFFIX = pyine.data.traces.dataset_utils.DELTAS_SUFFIX
"""Suffix for entries that correspond to trace deltas in the LMDB dataset."""

logger = logging.getLogger(__name__)


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
    """Caught an exception during execution (using an `except` clause)."""


@dataclasses.dataclass(frozen=True)
class TraceDelta:
    curr_trace_key: pyine.utils.code.execution.TraceKey
    """The trace key associated with the start of the delta."""
    next_trace_key: pyine.utils.code.execution.TraceKey
    """The trace key associated with the end of the delta."""
    trace_step_idx: int
    """The trace step index at the start of the delta; should be unique for each delta."""
    exception: pyine.utils.code.execution.TraceException | None
    """A dictionary containing information about the exception being raised/propagated, if any."""
    variables: dict[str, str]
    """A dictionary containing the added/removed/updated variables in the delta."""
    stdout: str | None
    """A string containing the stdout delta, if any (i.e. stuff caught on the 'next' step)."""
    stderr: str | None
    """A string containing the stderr delta, if any (i.e. stuff caught on the 'next' step)."""
    event_relationship: EventRelationship
    """The relationship between the start/end trace events."""

    def __repr__(self):
        """Returns a string representation of the trace delta."""
        if self.curr_trace_key.file == pyine.utils.code.execution.EXEC_TRACE_FILE_NAME:
            curr_trace_file_prefix = ""
        else:
            curr_trace_file_prefix = f"{self.curr_trace_key.file}:"
        if self.next_trace_key.file == pyine.utils.code.execution.EXEC_TRACE_FILE_NAME:
            next_trace_file_prefix = ""
        else:
            next_trace_file_prefix = f"{self.next_trace_key.file}:"
        delta_str = dict(**self.variables)
        if self.stdout is not None:
            delta_str["__stdout__"] = self.stdout
        if self.stderr is not None:
            delta_str["__stderr__"] = self.stderr
        return (
            f"{curr_trace_file_prefix}L{self.curr_trace_key.line:04d} -> {next_trace_file_prefix}L{self.next_trace_key.line:04d} "
            f"({self.event_relationship}) @ step#{self.trace_step_idx} = {delta_str}"
        )


class DeltaGeneratorType(enum.StrEnum):
    """Supported variable state delta generation approaches."""

    SIMPLE = enum.auto()
    """A simple delta generator that produces simple POD deltas with partial support for objects."""
    DEEPDIFF = enum.auto()
    """A delta generator based on DeepDiff that provides support for diffs between data structures."""

    @classmethod
    def from_str(cls, s: str) -> "DeltaGeneratorType":
        """Converts a string to a DeltaGeneratorType enum value, with case-insensitive matching."""
        s = s.lower()
        if s == cls.SIMPLE.name.lower():
            return DeltaGeneratorType.SIMPLE
        elif s == cls.DEEPDIFF.name.lower():
            return DeltaGeneratorType.DEEPDIFF
        else:
            raise ValueError(f"invalid delta generator type: {s}")


class TraceDeltaList(pydantic.BaseModel):
    """Dataclass used to store a list of trace deltas."""

    model_config = pydantic.ConfigDict(frozen=True, use_enum_values=True)
    """Pydantic model configuration (freezes the dataclass)."""

    trace_id: str
    """Unique identifier for the parent trace."""
    deltas: list[TraceDelta]
    """A list of trace deltas."""
    gen_type: DeltaGeneratorType
    """The type of delta generator used to generate the deltas."""

    def __str__(self) -> str:
        """Returns a string representation of the trace delta list."""
        deltas_str = [f"\n{d}" for d in self.deltas]
        return f"{self.trace_id}:deltas=[{deltas_str}\n]"

    def __len__(self) -> int:
        """Returns the number of deltas in the list."""
        return len(self.deltas)

    def __iter__(self) -> typing.Iterator[TraceDelta]:
        """Returns an iterator over the deltas in the list."""
        return iter(self.deltas)

    def __getitem__(self, index) -> TraceDelta:
        """Returns the delta at the given index."""
        return self.deltas[index]


def _wrapped_delta_generator(
    curr: dict[str, str] | pyine.utils.code.execution.TraceEvent,
    next: dict[str, str] | pyine.utils.code.execution.TraceEvent,
    delta_generator_fn: typing.Callable[[dict[str, str], dict[str, str]], dict[str, str]],
    include_global_vars: bool,
) -> dict[str, str]:
    """Extracts relevant variables from trace events (if needed) and forwards them to a delta generator."""
    if isinstance(curr, pyine.utils.code.execution.TraceEvent):
        if include_global_vars:
            curr = {**curr.global_variables, **curr.local_variables}
        else:
            curr = curr.local_variables
    if isinstance(next, pyine.utils.code.execution.TraceEvent):
        if include_global_vars:
            next = {**next.global_variables, **next.local_variables}
        else:
            next = next.local_variables
    return delta_generator_fn(curr, next)


def simple_delta_generator(curr: dict[str, str], next: dict[str, str]) -> dict[str, str]:
    """Compares two variable dicts and returns added/updated entries."""
    # note: we purposefully do NOT show missing/removed values in deltas to reduce useless spam/outputs
    output = {}
    for key in next.keys() - curr.keys():  # keys added to next
        output[key] = next[key]
    for key in curr.keys() & next.keys():  # keys updated in next
        if curr[key] != next[key]:
            output[key] = next[key]
    return output


class _CallStack:
    """A stack of caller ids and variables, used to track context during execution."""

    orig_caller_trace_key = pyine.utils.code.execution.TraceKey(
        # fill these with arbitrary but unique values to be able to easily identify it
        file=pyine.utils.code.execution.EXEC_PARENT_FILE_NAME,
        line=0,
        object="<module>",
    )

    def __init__(
        self,
        include_global_vars: bool,
    ) -> None:
        """Initializes the call stack (it will be empty at first, until initialized)."""
        self._stack: list[tuple[pyine.utils.code.execution.TraceKey, dict[str, typing.Any]]] = []
        self._incl_g = include_global_vars

    def is_initialized(self) -> bool:
        """Returns whether the call stack is initialized."""
        return len(self._stack) > 0

    def init(self, curr_step: pyine.utils.code.execution.TraceEvent) -> pyine.utils.code.execution.TraceKey:
        """Initializes the call stack with the current parent-provided arguments.

        Returns the hardcoded parent caller id that will identify when we exit the traced code.
        """
        assert not self.is_initialized(), "entrypoint should be initialized only once, when stack is empty"

        self._stack.append((self.orig_caller_trace_key, curr_step.arguments))
        return self.orig_caller_trace_key

    def pop(
        self,
        next_step: pyine.utils.code.execution.TraceEvent,  # used for internal validation only
    ) -> tuple[pyine.utils.code.execution.TraceKey, dict[str, typing.Any]]:
        """Pops the last element of the call stack, and returns its caller id and context vars."""
        assert len(self._stack) > 0, "return events should always be paired with a call?"
        caller_trace_key, caller_vars = self._stack.pop()
        if caller_trace_key.file == pyine.utils.code.execution.EXEC_PARENT_FILE_NAME:
            # if the caller is the execution parent, it means the stack should be empty
            assert len(self._stack) == 0
        else:
            # otherwise, the caller's identity should match the one expected in the stack
            assert caller_trace_key == next_step.stack_trace[1]
        return caller_trace_key, caller_vars

    def push(
        self,
        curr_step: pyine.utils.code.execution.TraceEvent,
    ) -> None:
        """Pushes a new (caller id, context vars) tuple to the call stack."""
        assert len(self._stack) > 0, "stack should be initialized before pushing a new element"
        if self._incl_g:
            variables = {**curr_step.global_variables, **curr_step.local_variables}
        else:
            variables = curr_step.local_variables
        self._stack.append((curr_step.trace_key, variables))

    def _push_manually(
        self,
        caller_trace_key: pyine.utils.code.execution.TraceKey,
        caller_vars: dict[str, typing.Any],
    ) -> None:
        """Pushes a new (caller id, context vars) tuple to the call stack."""
        assert len(self._stack) > 0, "stack should be initialized before pushing a new element"
        self._stack.append((caller_trace_key, caller_vars))


class _EventPairIterator:
    """An iterator that generates event pairs used to create deltas."""

    def __init__(
        self,
        trace_res: pyine.utils.code.execution.TraceResult,
    ):
        """Initializes the event iterator."""
        self.trace_res = trace_res
        self.relevant_step_idxs = self._filter_relevant_source_trace_step_idxs()
        assert len(self.relevant_step_idxs) > 1, "there should be at least two relevant trace steps"
        self.iter_idx = 0

    def _filter_relevant_source_trace_step_idxs(
        self,
    ) -> list[int]:
        """Filters out irrelevant trace steps that are invalid or outside the proposed code string."""
        first_relevant_step_idx = 0
        if self.trace_res.entrypoint_step_idx is not None:
            first_entrypoint_trace_step = next(
                (v for v in self.trace_res.traced_steps[self.trace_res.entrypoint_step_idx :] if v is not None),
                None,
            )
            if first_entrypoint_trace_step is None:
                raise AssertionError("invalid entrypoint call step")  # fix if it happens?
            first_relevant_step_idx = first_entrypoint_trace_step.trace_step_idx
        raw_trace_steps: list[pyine.utils.code.execution.TraceEvent | None] = self.trace_res.traced_steps
        filtered_trace_step_idxs = []
        for trace_step in raw_trace_steps:
            if trace_step is None:
                continue  # step originates from blacklisted, internal, or compiled modules
            if trace_step.trace_step_idx < first_relevant_step_idx:
                continue  # trace step occurs before we begin tracing the actual algo exec
            trace_key = trace_step.trace_key
            if trace_key.file != pyine.utils.code.execution.EXEC_TRACE_FILE_NAME:
                continue  # step originates from a separate file instead of the input code string
            filtered_trace_step_idxs.append(trace_step.trace_step_idx)
        return filtered_trace_step_idxs

    def _get_relation_between_events(
        self,
        curr: pyine.utils.code.execution.TraceEvent,
        next: pyine.utils.code.execution.TraceEvent,
    ) -> EventRelationship:
        """Returns the relation between two trace events."""
        assert curr.trace_step_idx < next.trace_step_idx, "out-of-order trace events?"
        # special handling: we usually only look at the 'next' event type, but will look at curr for this:
        if curr.event_type == pyine.utils.code.execution.TraceEventType.CALL:
            # the 'entrypoint' of the traced code; there should only be a single one of these
            # (we don't need to consider the next event type at all here, it will be reused)
            return EventRelationship.ENTRYPOINT
        # below, we now only look at the 'next' event type
        if next.event_type == pyine.utils.code.execution.TraceEventType.LINE:
            # going from one line to another is just a 'step over' delta (simplest type of delta)
            # (note: the 'current' line could be, apart from a regular line itself, an exception catch block)
            return EventRelationship.STEP_OVER
        if next.event_type == pyine.utils.code.execution.TraceEventType.CALL:
            # current event evaluates a function call, and next event moves the trace to that function
            # (the following event would correspond to the first executed line inside that function)
            assert curr.event_type in [
                pyine.utils.code.execution.TraceEventType.LINE,
                pyine.utils.code.execution.TraceEventType.RETURN,
            ]
            return EventRelationship.STEP_INTO
        if next.event_type == pyine.utils.code.execution.TraceEventType.RETURN:
            # if the next event of the pair is a return from a function call, then...
            assert curr.event_type in [
                pyine.utils.code.execution.TraceEventType.LINE,  # the current event might be one last eval
                pyine.utils.code.execution.TraceEventType.RETURN,  # or a cascading return from a prior call
                pyine.utils.code.execution.TraceEventType.EXCEPTION,  # or a propagating exception
            ]
            # note: this kind of event might be caused by a returned value, or by a raised exception
            return EventRelationship.STEP_OUT
        if next.event_type == pyine.utils.code.execution.TraceEventType.EXCEPTION:
            # current event caused an exception to be raised, which will be propagated until caught
            return EventRelationship.RAISE
        # default catch: (if this happens, we need to figure out why, and fix the delta pairing logic)
        raise NotImplementedError(f"unexpected event type combo: {curr.event_type}+{next.event_type}")

    def get_next_pair(
        self,
    ) -> tuple[pyine.utils.code.execution.TraceEvent, pyine.utils.code.execution.TraceEvent, EventRelationship]:
        """Returns the next event pair and relationship tuple."""
        assert self.iter_idx < len(self.relevant_step_idxs) - 1, "iterator index out of bounds"
        curr_step = self.trace_res.traced_steps[self.relevant_step_idxs[self.iter_idx]]
        next_step = self.trace_res.traced_steps[self.relevant_step_idxs[self.iter_idx + 1]]
        assert curr_step is not None and next_step is not None
        assert curr_step.trace_step_idx < next_step.trace_step_idx, "out-of-order trace steps?"
        relationship = self._get_relation_between_events(curr_step, next_step)
        self.iter_idx += 1
        return curr_step, next_step, relationship

    def get_next_event(
        self,
        increment: bool,
    ) -> pyine.utils.code.execution.TraceEvent | None:
        """Returns the next event, or None if there are no more events."""
        if self.iter_idx < len(self.relevant_step_idxs):
            next_step = self.trace_res.traced_steps[self.relevant_step_idxs[self.iter_idx + 1]]
            assert next_step is not None
            if increment:
                self.iter_idx += 1
            return next_step
        else:
            return None

    def increment_iter_idx(self) -> None:
        """Increments the iterator index."""
        assert self.iter_idx <= len(self.relevant_step_idxs) - 1, "iterator index out of bounds"
        self.iter_idx += 1

    def has_next_event(self) -> bool:
        """Returns whether there are more events to process."""
        return self.iter_idx < len(self.relevant_step_idxs) - 1


def get_deltas_from_trace_steps(
    trace_res: pyine.utils.code.execution.TraceResult,
    delta_generator: DeltaGeneratorType,
    include_global_vars: bool = True,
    verbose: bool = False,
) -> TraceDeltaList:
    """Generates a list of deltas from a trace result."""
    assert trace_res.identifier is not None, "need identifier when generating deltas"
    log = logger.info if verbose else logger.debug
    assert delta_generator in DeltaGeneratorType
    if delta_generator == DeltaGeneratorType.SIMPLE:
        delta_generator_fn = simple_delta_generator
    else:
        delta_generator_fn = deepdiff.DeepDiff
    delta_generator_fn = functools.partial(
        _wrapped_delta_generator,
        delta_generator_fn=delta_generator_fn,
        include_global_vars=include_global_vars,
    )
    output_deltas = []
    event_iterator = _EventPairIterator(trace_res)
    call_stack = _CallStack(include_global_vars=include_global_vars)
    while event_iterator.has_next_event():  # as long as there are still event pairs to generate...
        curr_step, next_step, relationship = event_iterator.get_next_pair()
        # the kind of step delta(s) we create will depend on the relationship between the two steps
        raised_exception = curr_step.exception or next_step.exception
        if verbose:
            log(f"\ncurr_step (#{curr_step.trace_step_idx}): {curr_step}")
            log(f"next_step (#{next_step.trace_step_idx}): {next_step}")
            log(f"relationship: {relationship}")
            log(f"exception: {raised_exception}")
        if relationship == EventRelationship.ENTRYPOINT:
            # special handling: the 'entrypoint' of the traced code should be unique
            assert not call_stack.is_initialized(), "there should only be one entrypoint per trace"
            orig_caller_trace_key = call_stack.init(curr_step)
            # ... and create 1st delta going from parent execution/caller to the traced code itself
            output_deltas.append(
                TraceDelta(
                    curr_trace_key=orig_caller_trace_key,
                    next_trace_key=next_step.trace_key,
                    trace_step_idx=curr_step.trace_step_idx,
                    exception=None,
                    variables=dict(
                        __call__=repr(curr_step.trace_key),
                        __args__=repr(next_step.arguments),
                    ),
                    stdout=next_step.stdout,
                    stderr=next_step.stderr,
                    event_relationship=EventRelationship.ENTRYPOINT,
                )
            )
            continue
        if relationship == EventRelationship.RAISE:
            # an exception has been raised; we'll need to check whether it's caught or propagated to the parent
            assert raised_exception is not None, "how can we be raising nothing?"
            delta = dict(__exception__=repr(raised_exception))
            # we need to go fetch the 'next-next' step in order to figure out what to do
            next_next_step = event_iterator.get_next_event(increment=True)
            assert next_next_step.exception is None, "the next-next step can't possibly raise again"
            next_step = next_next_step
            if next_step.event_type == pyine.utils.code.execution.TraceEventType.RETURN:
                # if we get here, it means the exception will be propagated to the parent
                # (the 'STEP_OUT' block below will continue with the proper propagation logic)
                pass
            elif next_step.event_type == pyine.utils.code.execution.TraceEventType.LINE:
                # if we get here, it means the exception was caught, so we can point to the next line directly
                # note: should we verify/assert that the 'next' line corresponds to the 'except' block?
                # should we instead skip ahead again, and not care about stepping to the except line itself?
                output_deltas.append(
                    TraceDelta(
                        curr_trace_key=curr_step.trace_key,
                        next_trace_key=next_step.trace_key,
                        trace_step_idx=curr_step.trace_step_idx,
                        exception=raised_exception,
                        variables=delta,
                        stdout=next_step.stdout,
                        stderr=next_step.stderr,
                        event_relationship=EventRelationship.RAISE,
                    )
                )
                continue
            else:
                raise AssertionError("unexpected event type in exception follow-up event")
        if relationship in [EventRelationship.STEP_OUT, EventRelationship.RAISE]:
            # execution returned from a function (with a value or a raised exception)
            caller_trace_key, caller_vars = call_stack.pop(next_step)  # pop the stack accordingly
            # might decompose this specific event into two deltas, to make sure we capture all info
            # ...but first we'll create a delta for the 'current line' which evaluates and returns/raises
            if raised_exception:
                delta = dict(__exception__=repr(raised_exception))
                output_relationship = EventRelationship.RAISE
            else:
                delta = dict(__return__=repr(next_step.return_value))
                output_relationship = EventRelationship.STEP_OUT
            if caller_trace_key == _CallStack.orig_caller_trace_key:
                delta.update(**delta_generator_fn(curr_step, next_step))
            output_deltas.append(
                TraceDelta(
                    curr_trace_key=next_step.trace_key,
                    next_trace_key=caller_trace_key,
                    trace_step_idx=curr_step.trace_step_idx,
                    exception=raised_exception,
                    variables=delta,
                    stdout=next_step.stdout,
                    stderr=next_step.stderr,
                    event_relationship=output_relationship,
                )
            )
            # now, check for the next-next event: if it's a regular line, we'll need a 2nd delta
            if event_iterator.has_next_event():  # we could be at the end of the trace
                next_next_step = event_iterator.get_next_event(increment=False)
                if next_next_step.event_type == pyine.utils.code.execution.TraceEventType.EXCEPTION:
                    # this means we are propagating the exception to a prior caller
                    event_iterator.increment_iter_idx()  # consume the raise event (we don't need it)
                    # next iteration will create another delta as above
                # once we get here, we've stopped propagating because we exited tracing or found a catcher
                elif next_next_step.event_type == pyine.utils.code.execution.TraceEventType.RETURN:
                    # this is OK, nothing else to do, next iteration will create another delta as above
                    pass
                elif next_next_step.event_type == pyine.utils.code.execution.TraceEventType.CALL:
                    # we are likely in a generator expression, calling back immediately after returning
                    # ...manually create the STEP_INTO delta in order to have the correct caller stack
                    event_iterator.increment_iter_idx()  # do that now, we are consuming the event
                    call_stack._push_manually(caller_trace_key, caller_vars)  # put orig caller info back on stack
                    next_call_delta = dict(
                        __call__=repr(next_next_step.trace_key), __args__=repr(next_next_step.arguments)
                    )
                    next_call_step = next_next_step.trace_step_idx
                    next_next_step = event_iterator.get_next_event(increment=True)  # fetch next event for its trace key
                    output_deltas.append(
                        TraceDelta(
                            curr_trace_key=caller_trace_key,
                            next_trace_key=next_next_step.trace_key,
                            trace_step_idx=next_call_step,
                            exception=next_next_step.exception,
                            variables=next_call_delta,
                            stdout=next_next_step.stdout,
                            stderr=next_next_step.stderr,
                            event_relationship=EventRelationship.STEP_INTO,
                        )
                    )
                elif next_next_step.event_type == pyine.utils.code.execution.TraceEventType.LINE:
                    # we need to create a 2nd delta to bridge between returned value and next line
                    event_iterator.increment_iter_idx()  # do that now, we are consuming the event
                    output_deltas.append(
                        TraceDelta(
                            curr_trace_key=caller_trace_key,
                            next_trace_key=next_next_step.trace_key,
                            trace_step_idx=next_step.trace_step_idx,
                            exception=next_next_step.exception,
                            variables=delta_generator_fn(caller_vars, next_next_step),
                            stdout=next_next_step.stdout,
                            stderr=next_next_step.stderr,
                            event_relationship=EventRelationship.STEP_OVER,
                        )
                    )
                else:
                    raise AssertionError("unexpected event type in step-out event")
            continue
        # all remaining cases only produce a single delta that captures all potential information
        if relationship == EventRelationship.STEP_OVER:
            # going from one line to another is just a 'step over' delta (simplest case)
            output_deltas.append(
                TraceDelta(
                    curr_trace_key=curr_step.trace_key,
                    next_trace_key=next_step.trace_key,
                    trace_step_idx=curr_step.trace_step_idx,
                    exception=raised_exception,
                    variables=delta_generator_fn(curr_step, next_step),
                    stdout=next_step.stdout,
                    stderr=next_step.stderr,
                    event_relationship=EventRelationship.STEP_OVER,
                )
            )
            continue
        if relationship == EventRelationship.STEP_INTO:
            # calling a function: specify the target function + args, and stack the vars context
            assert raised_exception is None, "why step into a function if an exception is being raised?"
            delta = dict(
                __call__=repr(next_step.trace_key),
                __args__=repr(next_step.arguments),
            )
            call_stack.push(curr_step)
            next_next_step = event_iterator.get_next_event(increment=True)  # fetch next event for its trace key
            assert next_next_step.exception is None, "why step into a function if an exception is being raised?"
            output_deltas.append(
                TraceDelta(
                    curr_trace_key=curr_step.trace_key,
                    next_trace_key=next_next_step.trace_key,
                    trace_step_idx=curr_step.trace_step_idx,
                    exception=None,
                    variables=delta,
                    stdout=next_step.stdout,
                    stderr=next_step.stderr,
                    event_relationship=relationship,
                )
            )
            continue
        raise NotImplementedError("unhandled delta with relationship: " + str(relationship))
    return TraceDeltaList(
        trace_id=trace_res.identifier,
        deltas=output_deltas,
        gen_type=delta_generator,
    )


def get_latest_dataset_path(
    source_dataset_name: str,
    filter_rule: str | None = None,
) -> pathlib.Path:
    """Returns the path to the latest deltas dataset for a specific source dataset.

    If multiple deltas datasets are available, the most recent version is returned, where we pick
    strictly by the date suffix (YYYY-MM-DD) in the directory name, ignoring any prefix tag.
    Optionally, a filter rule can be provided to exclude directories before selection.
    """
    assert source_dataset_name in SUPPORTED_SOURCE_DATASETS, f"invalid source dataset: {source_dataset_name}"
    return pyine.data.common.resolve_latest_dataset_path(
        kind="deltas",
        source_dataset_name=source_dataset_name,
        filter_rule=filter_rule,
    )


def get_matching_dataset_paths(
    source_dataset_name: str,
    pattern: str,
    is_regex: bool = False,
) -> list[pathlib.Path]:
    """Return all deltas dataset directories for a source that match the provided pattern.

    Matching modes:
      - Glob/fnmatch (default): e.g., 'mytag.*.lmdb', '*-08-*.lmdb'.
      - Regex: set is_regex=True or prefix the pattern with 're:'/'regex:' to use Python regex.
               Prefix 'glob:'/'fnmatch:' can force glob mode.

    Returns a list sorted deterministically by name.
    """
    assert source_dataset_name in SUPPORTED_SOURCE_DATASETS, f"invalid source dataset: {source_dataset_name}"
    return pyine.data.common.resolve_matching_dataset_paths(
        kind="deltas",
        source_dataset_name=source_dataset_name,
        pattern=pattern,
        pattern_is_regex=is_regex,
    )


def get_new_dataset_path(
    source_dataset_name: str,
    dataset_name_tag: str,
) -> pathlib.Path:
    """Returns the path where a new deltas dataset should be saved, for a specific source dataset.

    Will be named based on today's date and using the provided tag (which is the file name prefix).
    """
    assert source_dataset_name in SUPPORTED_SOURCE_DATASETS, f"invalid source dataset: {source_dataset_name}"
    assert dataset_name_tag, "dataset name tag cannot be empty"
    deltas_root = pyine.utils.filesystem.get_data_root_path() / "deltas" / source_dataset_name
    today = datetime.date.today()
    dataset_name = f"{dataset_name_tag}.{today.strftime('%Y-%m-%d')}.lmdb"
    return deltas_root / dataset_name
