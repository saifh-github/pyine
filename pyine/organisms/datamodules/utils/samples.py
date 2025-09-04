import functools
import pathlib
import typing

import datasets as hf_datasets
import numpy as np
import pydantic
import tqdm

import pyine.data.datamodule
import pyine.data.traces.dataset_reader
import pyine.data.traces.dataset_utils
import pyine.data.utils.filter_rules
import pyine.utils.code.blocks
import pyine.utils.code.execution
import pyine.utils.portability
import pyine.utils.reprod
from pyine.utils.code.execution import (
    TraceEvent,
    TraceEventType,
)


class TraceMetadata(typing.NamedTuple):
    """Metadata structure for a single trace, to be used for lookups and to cache as prepared data."""

    identifier: str
    """Unique identifier for the trace."""
    index: int
    """Index of the trace in its original dataset."""
    parent_dataset_hash: str
    """Hash of the dataset that contains the trace."""
    tags: list[str]
    """List of tags associated with the trace (problem+exec+augments)."""

    def get_parent_solution_id(self) -> str:
        """Returns the unique identifier for the parent solution to this trace.

        Each trace is linked with a solution (i.e. a code snippet) to a coding problem. Each solution
        can be used to get multiple traces, depending on the input arguments used when executing
        the code snippet, and depending on applied code augmentations.
        """
        return str(self.get_trace_id_obj().get_parent_identifier())

    def get_parent_problem_id(self) -> str:
        """Returns the unique identifier for the parent problem to this trace.

        Each trace is linked with a solution (i.e. a code snippet) to a coding problem. Each solution
        can be used to get multiple traces, depending on the input arguments used when executing
        the code snippet, and depending on applied code augmentations.

        THIS IS THE ULTIMATE IDENTIFIER THAT SHOULD BE USED FOR SPLITTING PURPOSES. By default, if
        a trace is assigned to a specific split subset based e.g. on a rule, all traces that belong
        to the same parent problem will be assigned to the same subset.
        """
        return str(self.get_trace_id_obj().get_parent_identifier().get_parent_identifier())

    def get_augment_type(self) -> str | None:
        """Returns the augmentation type for this trace (if any)."""
        return self.get_trace_id_obj().augment_category

    def get_trace_id_obj(self) -> pyine.data.traces.dataset_utils.TraceIdentifier:
        """Returns the trace identifier object for this trace."""
        return pyine.data.traces.dataset_utils.TraceIdentifier.from_string(self.identifier)


def get_traces_metadata(
    readers: list[pyine.data.traces.dataset_reader.DatasetReader] | pyine.data.traces.dataset_reader.DatasetReader,
    base_filter: pyine.data.utils.filter_rules.FilterType | None = None,
    verbose: bool = False,
) -> list[TraceMetadata]:
    """Returns a list of TraceMetadata objects for all traces in the provided dataset reader(s)."""
    if not isinstance(readers, list):
        readers = [readers]
    readers_map = {  # hash-to-reader map to later re-identify the origin of individual traces
        reader.get_hash(): reader for reader in readers
    }
    all_trace_keys = []
    for reader in readers_map.values():
        all_trace_keys.extend(reader.trace_keys)
    # sanity check: there should not be any duplicates
    assert len(set(all_trace_keys)) == len(all_trace_keys)
    if verbose:
        prog_bar = tqdm.tqdm(total=len(all_trace_keys), desc="Parsing traces metadata")
    else:
        prog_bar = None
    base_traces_meta: list[TraceMetadata] = []
    for reader_hash, reader in readers_map.items():
        for trace_idx in range(len(reader)):
            trace_data = reader[trace_idx]
            assert trace_data.identifier is not None, "trace identifier is required"
            problem_data = reader.get_problem_data(trace_idx)
            tags = get_tags_for_trace(trace_data, problem_data)
            is_banned = base_filter(tags) if base_filter is not None else False
            if not is_banned:
                base_traces_meta.append(
                    TraceMetadata(
                        identifier=trace_data.identifier,
                        index=trace_idx,
                        parent_dataset_hash=reader_hash,
                        tags=tags,
                    ),
                )
            if prog_bar is not None:
                prog_bar.update(1)
    if prog_bar is not None:
        prog_bar.close()
    return base_traces_meta


def get_tags_for_trace(
    trace_data: pyine.utils.code.execution.TraceResult,
    problem_data: pyine.data.traces.dataset_utils.CodingProblem,
) -> list[str]:
    """Returns a list of tags for a given trace so that we can decide whether to filter it."""
    # note: we combine problem tags, trace (exec) tags, and augmentation tags into a single list
    assert trace_data.identifier is not None, "trace identifier is required"
    trace_id = pyine.data.traces.dataset_utils.TraceIdentifier.from_string(trace_data.identifier)
    output_tags = []
    output_tags.extend(problem_data.problem_tags)
    output_tags.extend(trace_data.tags)
    if trace_id.augment_category is not None:
        output_tags.append(f"augment:{trace_id.augment_category}")
    return output_tags


class TraceDatasetMetadata(pydantic.BaseModel):
    """Metadata structure for a trace dataset, to be used for lookups and to cache as prepared data."""

    model_config = pydantic.ConfigDict(frozen=True, arbitrary_types_allowed=False, extra="forbid")
    """Pydantic model configuration (freezes the dataclass)."""

    base_traces: list[TraceMetadata]
    """List of withheld traces across all parsed datasets."""
    subset_traces: dict[pyine.data.datamodule.SubsetNameType, list[TraceMetadata]]
    """List of withheld traces for each subset."""
    leftover_traces: list[TraceMetadata]
    """List of leftover traces still unassigned after subset filtering and leftover split."""
    problem_assignments: dict[str, pyine.data.datamodule.SubsetNameType]
    """Assignments of coding problems """


SampleOutputType = typing.Literal[  # note: literal makes this type compatible with default collate
    "program output",  # only available for full executions
    "frame variables",  # only available for partial executions
    "next step key",  # only available for partial executions  # @@@@ TODO: not implemented yet!
    "function return",  # only available for specific function calls
]
"""Possible output types for trace execution samples:
- 'program output': the final output of the program after executing the entire code;
- 'frame variables': the full description of all frame variables at the target line/step;
- 'next step key': the trace key of the next event after executing the code up to a target line/step;
- 'function return': the return value of a specific function called at a specific line with specific arguments.
"""


class SampleData(typing.NamedTuple):
    """Data structure used to store extracted trace data to be batched by a data loader.

    All fields are ones that should essentially be collatable by the default PyTorch collate
    function. Strings are used for inputs/outputs to simplify typing and formatting. The names of
    the attributes below are chosen to overlap with the typical argument names used in the prompt
    templates we intend to use.

    Note: does not contain anything related to structured output expectations. See derived classes
    for more details on that.
    """

    identifier: str
    """Unique identifier for the trace. Used for debugging/logging/ref only."""
    code: str
    """Code string that has been executed and for which results must be predicted."""
    description: str
    """High-level description of the implemented code/algorithm (may be empty)."""
    entrypoint: str
    """Entrypoint name used when executing a target function (may be empty if irrelevant/unused)."""
    first_line: int
    """First execution line in the code string (should be 0 for full execs, non-zero for partial execs)."""
    last_line: int
    """Last potential execution line in the code string (should be total number of code lines for full execs)."""
    inputs: str
    """Provided input args (for full execution or function calls), or intermediary state (for partial execs)."""
    output: str
    """Expected output that was previously verified/found, and that should be predicted by models."""
    output_type: SampleOutputType
    """Type of the expected output (for specific descriptions in prompts)."""
    trace_step_count: int
    """Number of steps that are expected to be executed to predict the outputs (can be used as a hint)."""

    def get_trace_id_obj(self) -> pyine.data.traces.dataset_utils.TraceIdentifier:
        """Returns the trace identifier object for this trace."""
        return pyine.data.traces.dataset_utils.TraceIdentifier.from_string(self.identifier)


SampleDataParserType = pyine.data.datamodule.BaseDataParserType[SampleData]
"""Type of the dataset reader used to read traces from LMDB datasets."""

SampleDataLoaderType = pyine.data.datamodule.BaseDataLoaderType[SampleData]
"""Type of the data loader used to batch trace sample data from the dataset parser."""


class SampleTransformConfig(pydantic.BaseModel):
    """Configuration class specifying arguments to transform raw trace sample into partial ones."""

    model_config = pydantic.ConfigDict(frozen=True, arbitrary_types_allowed=False, extra="forbid")

    random_seed: int | None = None
    """Optional seed to initialize internal RNG for sampling decisions; None => nondeterministic."""
    partial_sample_decision_strategy: typing.Literal["never", "always", "if_too_long", "random", "hybrid"] = "never"
    """Strategy deciding when to create partial samples:
    - 'never': always return full traces;
    - 'always': always try to create partial samples (if possible, given limits below);
    - 'if_too_long': create partial samples only if the trace/code exceeds configured thresholds;
    - 'random': create partial samples with a fixed probability;
    - 'hybrid': create partial samples if too long, otherwise with the configured probability.
    """
    too_long_total_steps_threshold: int = pydantic.Field(default=10000, ge=1)
    """Minimum total trace steps threshold to treat a trace as 'too long'."""
    too_long_valid_steps_threshold: int = pydantic.Field(default=1000, ge=1)
    """Minimum valid (code-string-related) steps to treat a trace as 'too long'."""
    too_long_code_lines_threshold: int = pydantic.Field(default=500, ge=1)
    """Minimum code lines to treat a trace as 'too long' for 'if_too_long'/'hybrid'."""
    functions_fallback_to_segments: bool = False
    """Whether to fallback to segments when unable to target a function call as a partial sample."""
    max_partial_trace_steps: int | None = pydantic.Field(default=None, ge=1)
    """Optional cap on partial sample step count (not used in decision, passed to sample builder)."""
    min_partial_trace_steps: int | None = pydantic.Field(default=1, ge=1)
    """Optional minimum partial sample step count (not used in decision, passed to sample builder)."""
    max_inputs_str_length: int | None = pydantic.Field(default=1000, ge=0)
    """Optional cap on inputs string length to use in partial samples (if any).

    Not used in decision, but passed to sample builder. This is verified after a partial sample has
    been generated, and thus provides a 'soft rule' that will determine whether to fallback to the
    original 'full' sample (despite it potentially being too long according to other thresholds).
    """
    max_output_str_length: int | None = pydantic.Field(default=1000, ge=0)
    """Optional cap on expected outputs string length to use in partial samples (if any).

    Not used in decision, but passed to sample builder. This is verified after a partial sample has
    been generated, and thus provides a 'soft rule' that will determine whether to fallback to the
    original 'full' sample (despite it potentially being too long according to other thresholds).
    """
    combine_local_and_global_vars_for_partial_samples: bool = True
    """Whether to combine local variables and global variables into a single set for partial samples."""
    output_type_prob_map: typing.Annotated[
        dict[
            SampleOutputType,
            typing.Annotated[  # noqa
                pydantic.StrictFloat,
                pydantic.Field(ge=0.0, le=1.0),
            ],
        ],
        pydantic.Field(
            default_factory=dict,
            description=(
                "Probability map used to determine potential output types in random/hybrid strategies, "
                "when generating a partial sample, as well as when to fallback to full traces."
            ),
        ),
    ]


LMDBDatasetReadersOrPathsType = (
    pyine.data.traces.dataset_reader.DatasetReader
    | list[pyine.data.traces.dataset_reader.DatasetReader]
    | str
    | pathlib.Path
    | list[str | pathlib.Path]
)
"""Type used to specify source dataset args in the sample builder."""


class SampleBuilder(SampleDataParserType):
    """Wrapper around the LMDB dataset reader(s) that returns sample data for target traces.

    This wrapper will optionally transform raw traces into partial execution samples to make the
    prediction task easier in cases where e.g. traces are very long. How/when to do this must be
    specified via the transform config.
    """

    def __init__(
        self,
        source_data: LMDBDatasetReadersOrPathsType,  # noqa
        traces: list[TraceMetadata] | None = None,  # if `None`, will target all available traces
        config: SampleTransformConfig | None = None,
    ) -> None:
        """Initializes the reader with a list of LMDB readers and a list of target traces."""
        if config is None:
            config = SampleTransformConfig()  # noqa
        self.config = config
        if not isinstance(source_data, list):
            source_data = [source_data]
        for src_idx, src in enumerate(source_data):
            if isinstance(src, (str, pathlib.Path)):
                source_data[src_idx] = pyine.data.traces.dataset_reader.DatasetReader(pathlib.Path(src))
        self.readers_map = {r.get_hash(): r for r in source_data}
        if traces is None:
            # create a list of metadata structs for ALL available traces
            traces = pyine.organisms.datamodules.utils.samples.get_traces_metadata(
                readers=source_data,
                base_filter=None,
                verbose=False,
            )
        self.traces = traces
        assert len(self.readers_map) > 0
        assert isinstance(self.traces, list)
        assert all([t.parent_dataset_hash in self.readers_map for t in self.traces])
        assert all([0 <= t.index < len(self.readers_map[t.parent_dataset_hash]) for t in self.traces])
        self._rng = np.random.default_rng(self.config.random_seed)
        if self.config.partial_sample_decision_strategy != "never" and not self.config.output_type_prob_map:
            raise ValueError("output type prob map must be provided when using partial samples generation")
        # TODO @@@@@@@: look up code descriptions generated via `code_summary` prompt in local results db
        #       (for the targeted traces, that is; keep fetched data into memory as read-only)

    def __len__(self) -> int:
        """Returns the number of traces covered by this reader."""
        return len(self.traces)

    def __getitem__(self, idx: int) -> SampleData:
        """Returns a data sample for the trace at the specified index.

        Notes:
          - We perform a random draw across all potential output types (with the full program output
            as a fallback option) to decide which sample type to generate.
          - If a function is targeted: inputs are the call arguments, the expected output is the
            function's returned value(s), and the description contains the called function's name.
          - If an arbitrary code segment is targeted: inputs are the frame variables at the start of
            the segment, the output is the full description of the frame variables at the end of the
            segment (at a specific line, or after a given number of steps).
          - If max_partial_trace_steps is set, functions exceeding this cap are ignored and segments
            are truncated to respect the cap.
          - If for a generated candidate, max_inputs_str_length or max_output_str_length caps are
            set and exceeded, returns None, i.e. drops that candidate entirely.
        """
        if not (0 <= idx < len(self)):
            raise IndexError(f"index {idx} out of range")
        trace_meta = self.traces[idx]
        reader = self.readers_map[trace_meta.parent_dataset_hash]
        trace_data = reader[trace_meta.index]
        assert trace_data.identifier is not None, "trace identifier is required"
        assert trace_data.identifier == trace_meta.identifier, "trace identifier mismatch"
        picked_output_type = self._pick_output_type(trace_data)
        if picked_output_type == "function return":
            # first, if requested, try to generate a sample for a function call
            sample = self._get_function_call_sample(trace_data=trace_data)
            if sample is not None:
                # if we did successfully build a partial sample, return it now
                return sample
        if picked_output_type != "program output" and (
            picked_output_type != "function return" or self.config.functions_fallback_to_segments
        ):
            # if requested (or as a fallback from the function call sample), try to generate a segment sample
            sample = self._get_code_segment_sample(
                trace_data=trace_data,
                target_output_type=picked_output_type,
            )
            if sample is not None:
                # if we did successfully build a partial sample, return it now
                return sample
        # ultimate fallback: return a sample for the full program output
        return SampleData(
            identifier=trace_data.identifier,
            code=trace_data.code_string,
            description="",  # @@@@@ TODO: get using code_summarization prompt? (prior run logged somewhere?)
            entrypoint=str(trace_data.entrypoint_name),
            first_line=0,
            last_line=len(trace_data.code_string.splitlines()),
            inputs=trace_data.inputs,
            output=trace_data.expected_output,
            output_type="program output",
            trace_step_count=trace_data.valid_step_count,  # count valid steps only
        )

    def _satisfies_str_caps(self, inp: str, out: str) -> bool:
        """Returns whether inputs/output strings satisfy caps or not."""
        if self.config.max_inputs_str_length is not None and len(inp) > self.config.max_inputs_str_length:
            return False
        if self.config.max_output_str_length is not None and len(out) > self.config.max_output_str_length:
            return False
        return True

    def _pick_output_type(
        self,
        trace_data: pyine.utils.code.execution.TraceResult,
    ) -> SampleOutputType:
        """Given the info of a specific trace, determines what kind of output sample should be created.

        If we do decide to create a 'partial' trace sample, this function will determine which type
        to generate. If we do not decide to create a partial sample, this function will return
        'program output', i.e. that we should create a full program trace sample.

        Assumptions:
          - Partial samples will never be created if the decision strategy is 'never'.
          - We will always try to generate a partial sample if the decision strategy is 'always'.
          - A random draw will determine if a partial sample should be created under 'random' or 'hybrid'.
          - All output type probabilities are non-negative.
          - Sum of all output type probabilities is <= 1.0.
          - If the random draw falls in the leftover mass (1.0 - sum), we fall back to a full trace sample.
        """
        default_fallback: SampleOutputType = "program output"
        try_partial_sample = False
        if self.config.partial_sample_decision_strategy == "always":
            try_partial_sample = True
        elif self.config.partial_sample_decision_strategy != "never":
            if self.config.partial_sample_decision_strategy in ["if_too_long", "hybrid"]:
                is_too_long = (
                    trace_data.total_step_count >= self.config.too_long_total_steps_threshold
                    or trace_data.valid_step_count >= self.config.too_long_valid_steps_threshold
                    or len(trace_data.code_string.splitlines()) >= self.config.too_long_code_lines_threshold
                )
                if is_too_long:  # if the trace is too long, always try to generate a partial sample
                    try_partial_sample = True
            if not try_partial_sample and self.config.partial_sample_decision_strategy in ["hybrid", "random"]:
                # if the trace is not too long, we can still generate a partial sample randomly
                total_partial_mass = sum(
                    [
                        prob
                        for output_type, prob in self.config.output_type_prob_map.items()
                        if output_type != default_fallback  # full program output gets the mass balance
                    ]
                )
                try_partial_sample = self._rng.random() < total_partial_mass
        if not try_partial_sample:
            # if we still have not managed to decide to make a partial sample, return to full trace now
            return default_fallback
        # otherwise, decide what kind of output type to generate for the partial sample
        draw_val = self._rng.random()
        total_mass = 0.0
        for output_type, output_prob in self.config.output_type_prob_map.items():
            total_mass += output_prob
            if draw_val < total_mass:
                return output_type
        return default_fallback  # fallback to full trace

    def _get_function_call_sample(
        self,
        trace_data: pyine.utils.code.execution.TraceResult,
    ) -> SampleData | None:
        """Returns a sample for a function call in the given trace."""
        candidate_events = []
        # first, list all potential candidate events, i.e. function call events that we'll analyze below
        for step_idx, step in enumerate(trace_data.traced_steps):
            if step is None:
                continue
            if step.event_type == TraceEventType.CALL:
                # skip initial call for the trace exec
                if step.trace_key.object != pyine.utils.code.execution.EXEC_MODULE_OBJ_NAME:
                    candidate_events.append((step_idx, step))
        # iterate through all candidates until a good one is found
        while candidate_events:
            curr_candidate_idx = int(self._rng.integers(0, len(candidate_events)))
            call_event_idx, call_event = candidate_events[curr_candidate_idx]
            candidate_events.pop(curr_candidate_idx)
            target_func_name = call_event.trace_key.object
            target_func_return_depth = 1  # in case we're going to do recursive calls back-to-back
            # find the first matching RETURN after the call for that function name (best effort)
            return_event: TraceEvent | None = None
            return_event_idx: int | None = None
            function_output_str: str | None = None
            for step_idx in range(call_event_idx + 1, len(trace_data.traced_steps)):
                if trace_data.traced_steps[step_idx] is None:
                    continue  # invalid/external event, keep going
                if (
                    trace_data.traced_steps[step_idx].event_type == TraceEventType.CALL
                    and trace_data.traced_steps[step_idx].trace_key.object == target_func_name
                ):
                    # increase recursion depth (we need to find as many return calls)
                    target_func_return_depth += 1
                elif trace_data.traced_steps[step_idx].event_type == TraceEventType.RETURN:
                    # we assume that even when an exception is raised, we always get a 'return' event
                    if trace_data.traced_steps[step_idx].trace_key.object == target_func_name:
                        # decrease recursion depth (we found a matching return)
                        target_func_return_depth -= 1
                        if target_func_return_depth == 0:
                            # we've found enough matching calls, stepping out
                            return_event = trace_data.traced_steps[step_idx]
                            return_event_idx = step_idx
                            if return_event.exception is not None:
                                # if we are raising an exception, the expected output should be that exception
                                function_output_str = repr(return_event.exception)
                            else:
                                # otherwise, it's the returned value itself
                                function_output_str = repr(return_event.return_value)
                            break  # we found our matching return event, nothing else to do
            if return_event is None:
                continue  # could not locate the matching return event; go find another candidate
            # determine step count, i.e. the number of valid events between function call and return
            call_step_count = sum([s is not None for s in trace_data.traced_steps[call_event_idx:return_event_idx]])
            if self.config.max_partial_trace_steps and call_step_count > self.config.max_partial_trace_steps:
                # enforce step cap: if exceeded, skip this candidate
                continue
            if self.config.min_partial_trace_steps and call_step_count < self.config.min_partial_trace_steps:
                # enforce step minimum threshold: if not met, skip this candidate
                continue
            call_args_str = repr(call_event.arguments)
            if not self._satisfies_str_caps(call_args_str, function_output_str):
                # enforce inputs/output str length cap: if exceeded, skip this candidate
                continue
            matched_call_block = trace_data.code_blocks.get(str(call_event.trace_key), None)
            if matched_call_block is not None:  # if the function is external, we won't have a matched block
                first_line, last_line = matched_call_block.start_line, matched_call_block.end_line
            else:
                # corresponds to an external call; we'll assign first line == last line
                # @@@@ TODO: if there are a lot of external calls, might want to hint/doc them specifically
                first_line, last_line = call_event.trace_key.line, call_event.trace_key.line
            return SampleData(
                identifier=trace_data.identifier,
                code=trace_data.code_string,
                description="",  # @@@@@ TODO: get using code_summarization prompt? (prior run logged somewhere?)
                entrypoint=target_func_name,
                first_line=first_line,
                last_line=last_line,
                inputs=call_args_str,
                output=function_output_str,
                output_type="function return",
                trace_step_count=call_step_count,
            )
        return None  # no more candidates to consider, failed to get a function call

    def _get_code_segment_sample(
        self,
        trace_data: pyine.utils.code.execution.TraceResult,
        target_output_type: SampleOutputType,
    ) -> SampleData | None:
        """Returns a sample for a segment of the given trace."""
        # note: we can get here with a 'function return' target output if this was a fallback call
        assert target_output_type in ["function return", "frame variables", "next step key"]
        if target_output_type == "function return":
            target_output_type = "frame variables"  # override this now with something we can handle below
        if target_output_type == "next step key":
            raise NotImplementedError  # @@@@ TODO
        # TODO @@@@@: try to target specific blocks? (if/else blocks? loops?)
        # build candidate event lists contiguous within a single frame at any depth; on CALL, skip callee contents
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
                assert depth_collectors[current_depth] is not None
                depth_collectors[current_depth].append(step)
            # update depth after handling inclusion logic
            if step.event_type == TraceEventType.CALL:
                current_depth += 1
            elif step.event_type == TraceEventType.RETURN:
                # close the current depth collector (after including RETURN above)
                if depth_collectors.get(current_depth):
                    # compute step count excluding the closing return
                    range_step_count = len(depth_collectors[current_depth]) - 1
                    if range_step_count >= (self.config.min_partial_trace_steps or 1):
                        candidate_event_lists.append(depth_collectors[current_depth])
                    depth_collectors[current_depth] = None
                current_depth -= 1
        # we expect to have closed all collections by encountering matching RETURN events
        assert not any(depth_collectors.values()), "how did we end up with a trace ending without a return event?"
        while candidate_event_lists:
            # pick a random candidate list
            curr_candidate_idx = int(self._rng.integers(0, len(candidate_event_lists)))
            range_steps = candidate_event_lists[curr_candidate_idx]
            candidate_event_lists.pop(curr_candidate_idx)
            # must have at least one step + one closing return event
            if len(range_steps) <= 1:
                continue
            # determine segment step count, i.e. the number of events to keep in the range
            if self.config.max_partial_trace_steps:
                max_step_count = min(self.config.max_partial_trace_steps, len(range_steps) - 1)
            else:
                max_step_count = len(range_steps) - 1
            min_step_count = self.config.min_partial_trace_steps or 1
            if max_step_count < min_step_count:
                continue
            target_step_count = int(self._rng.integers(min_step_count, max_step_count + 1))
            # determine the first/last step locations within the range
            assert target_step_count <= len(range_steps) - 1
            segment_start_idx = int(self._rng.integers(0, len(range_steps) - target_step_count))
            segment_end_idx = segment_start_idx + target_step_count  # goes to next event to get outcomes
            segment_start, segment_end = range_steps[segment_start_idx], range_steps[segment_end_idx]
            segment_size = segment_end_idx - segment_start_idx
            assert 0 < segment_size <= max_step_count, "segment size is not valid"
            if self.config.combine_local_and_global_vars_for_partial_samples:
                input_vars = {**segment_start.global_variables, **segment_start.local_variables}
                output_vars = {**segment_end.global_variables, **segment_end.local_variables}
            else:
                input_vars = segment_start.local_variables
                output_vars = segment_end.local_variables
            input_vars_str, output_vars_str = repr(input_vars), repr(output_vars)
            if not self._satisfies_str_caps(input_vars_str, output_vars_str):
                continue  # enforce inputs/output str length cap
            first_line, last_line = segment_start.trace_key.line, segment_end.trace_key.line
            return SampleData(
                identifier=trace_data.identifier,
                code=trace_data.code_string,
                description="",  # @@@@@ TODO: get using code_summarization prompt? (prior run logged somewhere?)
                entrypoint="",
                first_line=first_line,
                last_line=last_line,
                inputs=input_vars_str,
                output=output_vars_str,
                output_type="frame variables",
                trace_step_count=segment_size,
            )
        return None  # no more candidate lists to consider, failed to get a segment sample


class SampleBuilderConfig(pyine.data.datamodule.BaseDataParserConfig):
    """Configuration class for the (raw) dataset trace sample builder."""

    class_path: str = pyine.utils.portability.get_fully_qualified_name(SampleBuilder)
    """Fully qualified class path for the trace parser."""
    params: dict[str, typing.Any] = dict(config=SampleTransformConfig(partial_sample_decision_strategy="never"))
    """Default parameters for the dataset trace parser."""

    def get_hf_messages_dataset(
        self,
        named_split: "hf_datasets.NamedSplit",
        raw_transform_fn: typing.Callable[[dict[str, typing.Any]], typing.Any] | None = None,
        instantiate_kwargs: dict | None = None,
        generator_kwargs: dict | None = None,
    ) -> "hf_datasets.Dataset":
        """Returns a huggingface-compatible generator using a SampleBuilder instance."""

        def _sample_builder_generator():
            sample_builder = self.instantiate(**(instantiate_kwargs or {}))
            for sample_idx in range(len(sample_builder)):
                sample_data = sample_builder[sample_idx]
                yield sample_data._asdict()  # noqa

        dataset = hf_datasets.Dataset.from_generator(
            generator=_sample_builder_generator,
            split=named_split,
            **(generator_kwargs or {}),
        ).map(raw_transform_fn or (lambda x: x))
        return dataset
