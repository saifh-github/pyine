import dataclasses
import logging

import numpy as np

import pyine.data.traces.dataset_utils
import pyine.utils.code.execution
from pyine.organisms.datamodules.samples.common import (
    SampleCodeTypeSet,
    SampleData,
    SamplePredictType,
    SampleTransformStrategy,
    convert_to_comma_separated_tags,
    draw_type,
)
from pyine.organisms.datamodules.samples.configs import (
    SampleTransformConfig,
)
from pyine.organisms.datamodules.samples.selection import (
    SelectedSample,
)
from pyine.utils.code.execution import (
    TraceEvent,
    TraceEventType,
    TraceResult,
)

__all__ = [
    "SelectedPredictTypeResult",
    "generate_sample",
]

logger = logging.getLogger(__name__)

# @@@@@@ TODO: update the sample preparation functions here so that we also include reasoning


@dataclasses.dataclass(frozen=True, slots=True)
class SelectedPredictTypeResult:
    """Result of an analysis or draw for a sample prediction type."""

    trace_data: pyine.utils.code.execution.TraceResult
    """Trace data that was used to analyze and draw the prediction type."""
    is_too_long: bool
    """Whether the trace is considered too long according to configured thresholds."""
    randomly_picked_partial_sample: bool
    """Whether the decision to generate a partial sample was determined randomly."""
    forced_full_output_due_to_code_override: bool
    """Whether the decision to generate a full output sample was forced due to a code override."""
    predict_type: SamplePredictType | None
    """Selected prediction type for the sample to be generated; if None, sample must be skipped."""

    @property
    def should_skip(self) -> bool:
        """Whether the sample should be skipped."""
        return self.predict_type is None

    @property
    def is_partial_sample(self) -> bool:
        """Whether the selected prediction type is a partial sample."""
        return self.predict_type != SamplePredictType.program_output


def generate_sample(
    code_type_selection_result: SelectedSample,
    trace_data: TraceResult,
    code_summary: str | None,
    transform_config: SampleTransformConfig,
    rng: np.random.Generator,
) -> SampleData | None:
    """Generates a sample for the given trace and selection result.

    The configured transformation strategy determines what the expected prediction should be for
    the generated samples, i.e. whether they should be 'partial' or 'full' trace samples. Partial
    samples are those where only a portion of the execution trace is targeted and can be used to
    diversify training data and make the prediction task easier. Full samples target entire code
    snippet executions.
    """
    assert trace_data.identifier is not None, "trace identifier is required"
    assert trace_data.identifier == code_type_selection_result.trace_meta.identifier, "trace identifier mismatch"
    pred_type_selection_result = _select_predict_type(
        code_type_selection_result=code_type_selection_result,
        trace_data=trace_data,
        transform_config=transform_config,
        rng=rng,
    )
    should_fallback_to_segments = False
    if pred_type_selection_result.predict_type == SamplePredictType.function_return:
        # first, if requested, try to generate a sample for a function call
        sample = _get_function_call_sample(
            trace_data=trace_data,
            trace_meta=code_type_selection_result.trace_meta,
            trace_code_type=code_type_selection_result.code_type,
            code_summary=code_summary,
            transform_config=transform_config,
            rng=rng,
        )
        if sample is not None:
            # if we did successfully build a partial sample, return it now
            return sample
        if transform_config.functions_fallback_to_segments:
            should_fallback_to_segments = True
    if pred_type_selection_result.predict_type == SamplePredictType.frame_variables or should_fallback_to_segments:
        # if requested (or as a fallback from the function call sample), try to generate a segment sample
        sample = _get_code_segment_sample(
            trace_data=trace_data,
            trace_meta=code_type_selection_result.trace_meta,
            trace_code_type=code_type_selection_result.code_type,
            code_summary=code_summary,
            transform_config=transform_config,
            rng=rng,
        )
        if sample is not None:
            # if we did successfully build a partial sample, return it now
            return sample
    if pred_type_selection_result.predict_type == SamplePredictType.next_step_key:
        raise NotImplementedError  # @@@@ TODO
    # ultimate fallback: return a sample for the full program output
    return _get_full_program_sample(
        trace_data=trace_data,
        code_type_selection_result=code_type_selection_result,
        code_summary=code_summary,
    )


def _select_predict_type(
    code_type_selection_result: SelectedSample,
    trace_data: pyine.utils.code.execution.TraceResult,
    transform_config: SampleTransformConfig,
    rng: np.random.Generator,
) -> SelectedPredictTypeResult:
    """Given the info of a specific trace, determines what kind of output sample should be generated.

    If we do decide to create a 'partial' trace sample, this function will draw which type to
    generate. Whether it is actually possible to generate such a partial sample will be determined
    later.

    Assumptions:
      - Partial samples will never be created if the decision strategy is 'never'.
      - We will always propose to generate a partial sample if the decision strategy is 'always'.
      - A random draw may determine if a partial sample should be created under 'random' or 'hybrid'.
      - No partial sample will ever be created if the trace has a code override.
    """
    assert trace_data.identifier == code_type_selection_result.trace_meta.identifier, "trace identifier mismatch"
    is_too_long = transform_config.check_if_trace_too_long(trace_data)
    if code_type_selection_result.code_override is not None:
        # if we have a code snippet override (from the prompt result db), we do NOT have actual trace data
        # (the trace data corresponds to different code; the only possible pred type is program output)
        return SelectedPredictTypeResult(
            trace_data=trace_data,
            is_too_long=is_too_long,
            randomly_picked_partial_sample=False,
            forced_full_output_due_to_code_override=True,
            predict_type=SamplePredictType.program_output,
        )
    # otherwise, the trace data provides all we need to create any prediction type, so draw one
    picked_randomly = False
    try_partial_sample = False
    if transform_config.transform_strategy == SampleTransformStrategy.always:
        try_partial_sample = True
    elif transform_config.transform_strategy != SampleTransformStrategy.never:
        must_check_if_too_long = [SampleTransformStrategy.if_too_long, SampleTransformStrategy.hybrid]
        if transform_config.transform_strategy in must_check_if_too_long and is_too_long:
            # if the trace is too long, always try to generate a partial sample
            try_partial_sample = True
        could_randomly_pick = [SampleTransformStrategy.hybrid, SampleTransformStrategy.random]
        if not try_partial_sample and transform_config.transform_strategy in could_randomly_pick:
            # if the trace is not too long, we can still generate a partial sample randomly
            total_partial_mass = sum(
                [
                    prob
                    for predict_type, prob in transform_config.predict_type_prob_map.items()
                    if predict_type != SamplePredictType.program_output  # full program output gets the mass balance
                ]
            )
            if rng.random() < total_partial_mass:
                picked_randomly = True
                try_partial_sample = True
    if not try_partial_sample:
        # if we still have not decide to make a partial sample, return full trace pred type now
        return SelectedPredictTypeResult(
            trace_data=trace_data,
            is_too_long=is_too_long,
            randomly_picked_partial_sample=picked_randomly,
            forced_full_output_due_to_code_override=False,
            predict_type=SamplePredictType.program_output,
        )
    # otherwise, draw what kind of output type to generate (maybe partial, maybe full)
    drawn_prediction_type = draw_type(
        prob_map=transform_config.predict_type_prob_map,
        rng=rng,
        default_fallback=SamplePredictType.program_output,
    )
    assert drawn_prediction_type is not None, "invalid predict type draw? (check prob map)"
    return SelectedPredictTypeResult(
        trace_data=trace_data,
        is_too_long=is_too_long,
        randomly_picked_partial_sample=picked_randomly,
        forced_full_output_due_to_code_override=False,
        predict_type=drawn_prediction_type,
    )


def _get_function_call_sample(
    trace_data: pyine.utils.code.execution.TraceResult,
    trace_meta: pyine.data.traces.dataset_utils.TraceMetadata,
    trace_code_type: SampleCodeTypeSet,
    code_summary: str | None,
    transform_config: SampleTransformConfig,
    rng: np.random.Generator,
) -> SampleData | None:
    """Returns a partial sample covering a function call in the given trace.

    The sample may itself contain call/return events for internal functions called inside the target
    function, but it should be possible to predict the return value of the target function itself
    given only its input arguments (and possibly a global state snapshot).
    """
    # first, list all potential candidate events, i.e. function call events that we'll analyze below
    candidate_events: list[tuple[int, TraceEvent]] = []
    for step_idx, step in enumerate(trace_data.traced_steps):
        if step is None:
            continue
        if (
            step.event_type == TraceEventType.CALL
            and step.trace_key.object != pyine.utils.code.execution.EXEC_MODULE_OBJ_NAME
        ):
            candidate_events.append((step_idx, step))
    max_partial_trace_steps = transform_config.get_max_partial_trace_steps(trace_data)
    # iterate through all candidates until a good one is found
    while candidate_events:
        curr_candidate_idx = int(rng.integers(0, len(candidate_events)))
        call_event_idx, call_event = candidate_events[curr_candidate_idx]
        candidate_events.pop(curr_candidate_idx)
        target_func_name = call_event.trace_key.object
        target_func_return_depth = 1  # in case we're going to do recursive calls back-to-back
        # find the first matching RETURN after the call for that function name (best effort)
        return_event: TraceEvent | None = None
        return_event_idx: int | None = None
        function_output_str: str | None = None
        last_exception_at_depth: TraceEvent | None = None  # track exception events for the current function
        for step_idx in range(call_event_idx + 1, len(trace_data.traced_steps)):
            if trace_data.traced_steps[step_idx] is None:
                continue  # invalid/external event, keep going
            curr_event = trace_data.traced_steps[step_idx]
            if curr_event.event_type == TraceEventType.CALL and curr_event.trace_key.object == target_func_name:
                # increase recursion depth (we need to find as many return calls)
                target_func_return_depth += 1
                last_exception_at_depth = None  # reset exception tracking for nested call
            elif curr_event.event_type == TraceEventType.EXCEPTION:
                # track the exception event; it might be followed by a RETURN
                if target_func_return_depth == 1:
                    last_exception_at_depth = curr_event
            elif curr_event.event_type == TraceEventType.RETURN and curr_event.trace_key.object == target_func_name:
                # we assume that even when an exception is raised, we always get a 'return' event
                target_func_return_depth -= 1
                if target_func_return_depth == 0:
                    # we've found enough matching calls, stepping out
                    return_event = curr_event
                    return_event_idx = step_idx
                    # check for exception: first in RETURN event, then in preceding EXCEPTION event
                    if return_event.exception is not None:
                        function_output_str = repr(return_event.exception)
                    elif last_exception_at_depth is not None and last_exception_at_depth.exception is not None:
                        # exception is in the preceding EXCEPTION event, not the RETURN event
                        function_output_str = repr(last_exception_at_depth.exception)
                    else:
                        # otherwise, it's the returned value itself
                        function_output_str = repr(return_event.return_value)
                    break  # we found our matching return event, nothing else to do
                last_exception_at_depth = None  # reset for outer depth
        if return_event is None:
            continue  # could not locate the matching return event; go find another candidate
        assert function_output_str is not None
        # determine step count, i.e. the number of valid events between function call and return
        call_step_count = sum(step is not None for step in trace_data.traced_steps[call_event_idx:return_event_idx])
        if call_step_count > max_partial_trace_steps:
            # enforce step cap: if exceeded, skip this candidate
            continue
        if call_step_count < transform_config.min_partial_trace_steps:
            # enforce step minimum threshold: if not met, skip this candidate
            continue
        call_args_str = repr(call_event.arguments)
        if not transform_config.check_input_and_output_strings_satisfy_caps(call_args_str, function_output_str):
            # enforce inputs/output str length cap: if exceeded, skip this candidate
            continue
        matched_call_block = trace_data.code_blocks.get(str(call_event.trace_key), None)
        if matched_call_block is not None:  # if the function is external, we won't have a matched block
            first_line, last_line = (
                matched_call_block.start_line,
                matched_call_block.end_line,
            )
        else:
            # corresponds to an external call; we'll assign first line == last line
            # @@@@ TODO: if there are a lot of external calls, might want to hint/doc them specifically
            first_line, last_line = (
                call_event.trace_key.line,
                call_event.trace_key.line,
            )
        # if we get here, we have successfully found a function call trace that we can use
        sample_tags = trace_meta.tags.copy()
        description = code_summary if code_summary is not None else ""
        sample_tags.append(f"sample_code_description:{int(bool(description))}")
        sample_tags.append(f"sample_code_type:{trace_code_type}")
        sample_tags.append(f"sample_predict_type:{SamplePredictType.function_return}")
        assert trace_data.identifier is not None, "trace identifier is required"
        return SampleData(
            identifier=trace_data.identifier,
            code=trace_data.code_string,
            description=description,
            entrypoint=target_func_name,
            first_line=first_line,
            last_line=last_line,
            inputs=call_args_str,
            expected_output=function_output_str,
            predict_type=SamplePredictType.function_return,
            code_type=str(trace_code_type),
            trace_step_count=call_step_count,
            comma_separated_tags=convert_to_comma_separated_tags(sample_tags),
            has_code_override=False,
            complexity_metrics=trace_data.complexity_metrics.as_dict(),
            first_line_hit=0,  # not tracked for function_return
            last_line_hit=0,
            first_step_idx=call_event.trace_step_idx,
            last_step_idx=return_event.trace_step_idx,
        )
    return None  # no more candidates to consider, failed to get a function call


def _get_code_segment_sample(
    trace_data: pyine.utils.code.execution.TraceResult,
    trace_meta: pyine.data.traces.dataset_utils.TraceMetadata,
    trace_code_type: SampleCodeTypeSet,
    code_summary: str | None,
    transform_config: SampleTransformConfig,
    rng: np.random.Generator,
) -> SampleData | None:
    """Returns a partial sample for an execution segment of the given trace.

    The sample may itself contain call/return events for internal functions called inside the target
    segment, but it should be possible to predict the in-memory variables at the last step of the
    segment given only the state of its variables at the start (and possibly a global state snapshot).
    """
    # TODO @@@@@: try to target specific blocks? (if/else blocks? loops?)
    # build candidates of contiguous event lists within a single frame at any depth
    max_partial_trace_steps = transform_config.get_max_partial_trace_steps(trace_data)
    candidate_event_lists: list[list[TraceEvent]] = []
    current_depth = 0  # track call depth; assume first non-None event is a CALL
    depth_collectors: dict[int, list[TraceEvent] | None] = {}
    for step in trace_data.traced_steps:
        if step is None:
            continue  # out-of-scope/invalid; cannot update depth reliably
        # start collecting on any non-CALL event at the current depth
        if step.event_type != TraceEventType.CALL:
            if depth_collectors.get(current_depth) is None:
                depth_collectors[current_depth] = []
            depth_collectors[current_depth].append(step)
        # update depth after handling inclusion logic
        if step.event_type == TraceEventType.CALL:
            current_depth += 1
        elif step.event_type == TraceEventType.RETURN:
            # close the current depth collector (after including RETURN above)
            if depth_collectors.get(current_depth):
                # compute step count excluding the closing return
                curr_events = depth_collectors[current_depth]
                assert curr_events is not None
                range_step_count = len(curr_events) - 1
                if range_step_count >= transform_config.min_partial_trace_steps:
                    candidate_event_lists.append(curr_events)
                depth_collectors[current_depth] = None
            current_depth -= 1
    # we expect to have closed all collections by encountering matching RETURN events
    if any(depth_collectors.values()):
        # how did we end up with a trace ending without a return event?
        # (might need to investigate these later on, for now will drop top scope as a potential candidate)
        logger.warning(f"found trace ending without return event: {trace_meta.identifier}")
    while candidate_event_lists:
        # pick a random candidate list
        curr_candidate_idx = int(rng.integers(0, len(candidate_event_lists)))
        range_steps = candidate_event_lists[curr_candidate_idx]
        candidate_event_lists.pop(curr_candidate_idx)
        # must have at least one step + one closing return event
        if len(range_steps) <= 1:
            continue
        # determine segment step count, i.e. the number of events to keep in the range
        max_step_count = min(max_partial_trace_steps, len(range_steps) - 1)
        min_step_count = transform_config.min_partial_trace_steps
        if max_step_count < min_step_count:
            continue
        target_step_count = int(rng.integers(min_step_count, max_step_count + 1))
        # determine the first/last step locations within the range
        assert target_step_count <= len(range_steps) - 1
        segment_start_idx = int(rng.integers(0, len(range_steps) - target_step_count))
        segment_end_idx = segment_start_idx + target_step_count  # goes to next event to get outcomes
        segment_start, segment_end = (
            range_steps[segment_start_idx],
            range_steps[segment_end_idx],
        )
        segment_size = segment_end_idx - segment_start_idx
        assert 0 < segment_size <= max_step_count, "segment size is not valid"
        if transform_config.combine_local_and_global_vars_for_partial_samples:
            input_vars = {
                **segment_start.global_variables,
                **segment_start.local_variables,
            }
            output_vars = {
                **segment_end.global_variables,
                **segment_end.local_variables,
            }
        else:
            input_vars = segment_start.local_variables
            output_vars = segment_end.local_variables
        input_vars_str, output_vars_str = repr(input_vars), repr(output_vars)
        if not transform_config.check_input_and_output_strings_satisfy_caps(input_vars_str, output_vars_str):
            continue  # enforce inputs/output str length cap
        first_line, last_line = (
            segment_start.trace_key.line,
            segment_end.trace_key.line,
        )
        # compute hit counts using traced_steps_map
        first_line_key = repr(segment_start.trace_key)
        assert first_line_key in trace_data.traced_steps_map, f"missing key: {first_line_key}"
        first_line_hit = trace_data.traced_steps_map[first_line_key].index(segment_start.trace_step_idx) + 1
        last_line_key = repr(segment_end.trace_key)
        assert last_line_key in trace_data.traced_steps_map, f"missing key: {last_line_key}"
        last_line_hit = trace_data.traced_steps_map[last_line_key].index(segment_end.trace_step_idx) + 1
        # get absolute step indices
        first_step_idx = segment_start.trace_step_idx
        last_step_idx = segment_end.trace_step_idx
        sample_tags = trace_meta.tags.copy()
        description = code_summary if code_summary is not None else ""
        sample_tags.append(f"sample_code_description:{int(bool(description))}")
        sample_tags.append(f"sample_code_type:{trace_code_type}")
        sample_tags.append(f"sample_predict_type:{SamplePredictType.frame_variables}")
        assert trace_data.identifier is not None, "trace identifier is required"
        return SampleData(
            identifier=trace_data.identifier,
            code=trace_data.code_string,
            description=description,
            entrypoint="",
            first_line=first_line,
            last_line=last_line,
            inputs=input_vars_str,
            expected_output=output_vars_str,
            predict_type=SamplePredictType.frame_variables,
            code_type=str(trace_code_type),
            trace_step_count=segment_size,
            comma_separated_tags=convert_to_comma_separated_tags(sample_tags),
            has_code_override=False,
            complexity_metrics=trace_data.complexity_metrics.as_dict(),
            first_line_hit=first_line_hit,
            last_line_hit=last_line_hit,
            first_step_idx=first_step_idx,
            last_step_idx=last_step_idx,
        )
    return None  # no more candidate lists to consider, failed to get a segment sample


def _get_full_program_sample(
    trace_data: pyine.utils.code.execution.TraceResult,
    code_type_selection_result: SelectedSample,
    code_summary: str | None,
) -> SampleData:
    sample_tags = code_type_selection_result.trace_meta.tags.copy()
    description = code_summary if code_summary is not None else ""
    sample_tags.append(f"sample_code_description:{int(bool(description))}")
    sample_tags.append(f"sample_code_type:{code_type_selection_result.code_type}")
    sample_tags.append(f"sample_predict_type:{SamplePredictType.program_output}")
    if code_type_selection_result.code_override is not None:
        sample_code = code_type_selection_result.code_override
    else:
        sample_code = trace_data.code_string
    assert trace_data.identifier is not None, "trace identifier is required"
    return SampleData(
        identifier=trace_data.identifier,
        code=sample_code,
        description=description,
        entrypoint=str(trace_data.entrypoint_name),
        first_line=0,
        last_line=len(sample_code.splitlines()),
        inputs=str(trace_data.inputs),
        expected_output=str(trace_data.expected_output),
        predict_type=SamplePredictType.program_output,
        code_type=str(code_type_selection_result.code_type),
        trace_step_count=trace_data.valid_step_count,  # count valid steps only
        comma_separated_tags=convert_to_comma_separated_tags(sample_tags),
        has_code_override=code_type_selection_result.code_override is not None,
        complexity_metrics=trace_data.complexity_metrics.as_dict(),
        first_line_hit=0,  # not applicable for full program
        last_line_hit=0,
        first_step_idx=0,
        last_step_idx=0,
    )
