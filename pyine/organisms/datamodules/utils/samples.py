import collections
import dataclasses
import functools
import logging
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
import pyine.prompts
import pyine.utils.code.blocks
import pyine.utils.code.execution
import pyine.utils.portability
import pyine.utils.reprod
from pyine.utils.code.execution import (
    TraceEvent,
    TraceEventType,
)

logger = logging.getLogger(__name__)


def _get_expected_augment_types() -> list[str]:
    """Returns the list of expected augmentation types for trace datasets.

    Combines the augmentation types from the `SampleInputType` define with all prompt names
    that are known to the prompt manager.
    """
    import pyine.prompts

    pregenerated_augment_types = [
        prompt_name
        for prompt_name in pyine.prompts.get_framework_prompt_manager().list_prompts()
        if prompt_name.startswith("issues/") or prompt_name.startswith("hints/")
    ]
    irrelevant_augment_types = ["original", "stubbed"]  # stubbed code cannot be executed/traced
    dynamic_augment_types = [t for t in typing.get_args(SampleInputType) if t not in irrelevant_augment_types]
    return [*dynamic_augment_types, *pregenerated_augment_types]


@dataclasses.dataclass(frozen=True)
class TraceMetadata:
    """Metadata structure for a single trace, to be used for lookups and to cache as prepared data."""

    identifier: str
    """Unique identifier (str) for the trace."""
    index: int
    """Index of this sample in its original dataset."""
    parent_dataset_hash: str
    """Hash of the dataset that contains the trace."""
    tags: list[str]
    """List of tags associated with the trace (problem+exec+augments)."""

    @functools.cached_property
    def trace_id(self) -> pyine.data.traces.dataset_utils.TraceIdentifier:
        """Returns the trace identifier object for this trace."""
        return pyine.data.traces.dataset_utils.TraceIdentifier.from_string(self.identifier)

    @functools.cached_property
    def solution_id(self) -> pyine.data.traces.dataset_utils.SolutionIdentifier:
        """Returns the unique identifier for the parent solution to this trace.

        Each trace is linked with a solution (i.e. a code snippet) to a coding problem. Each solution
        can be used to get multiple traces, depending on the input arguments used when executing
        the code snippet, and depending on applied code augmentations.
        """
        return self.trace_id.get_parent_identifier()

    @functools.cached_property
    def problem_id(self) -> pyine.data.traces.dataset_utils.CodingProblemIdentifier:
        """Returns the unique identifier for the parent problem to this trace.

        Each trace is linked with a solution (i.e. a code snippet) to a coding problem. Each solution
        can be used to get multiple traces, depending on the input arguments used when executing
        the code snippet, and depending on applied code augmentations.

        THIS IS THE ULTIMATE IDENTIFIER THAT SHOULD BE USED FOR SPLITTING PURPOSES. By default, if
        a trace is assigned to a specific split subset based e.g. on a rule, all traces that belong
        to the same parent problem will be assigned to the same subset.
        """
        return self.solution_id.get_parent_identifier()

    @functools.cached_property
    def augment_tags(self) -> list[str]:
        """Returns all tags associated with this trace's augmentation(s)."""
        out_tags = [t for t in self.tags if t.startswith("augment:")]
        if out_tags:
            augm_category = self.trace_id.augment_category
            assert (
                augm_category is not None and f"augment:{augm_category}" in out_tags
            ), "found inconsistent augmentation category tags; dataset creation issue?"
        else:
            assert self.trace_id.augment_category is None
        return out_tags

    @functools.cached_property
    def is_augmented(self) -> bool:
        """Returns whether this trace is augmented."""
        return len(self.augment_tags) > 0

    @functools.cached_property
    def is_multi_augmented(self) -> bool:
        """Returns whether this trace is multi-augmented (e.g. bugged-hinted, bugged-misleading, etc.)."""
        multi_augm_categories: list[SampleInputType] = [
            "obfuscated_hinted",
            "obfuscated_misleading",
            "bugged_hinted",
            "bugged_misleading",
        ]
        return any([t in self.augment_tags for t in multi_augm_categories])

    @functools.cached_property
    def comma_separated_tags(self) -> str:
        """Returns a comma-separated string of all tags associated with this trace."""
        if not self.tags:
            return ""
        for tag in self.tags:
            if "," in tag:
                raise ValueError(f"trace tags cannot contain commas: {tag}")
        return ",".join(self.tags)


def get_traces_metadata(
    readers: list[pyine.data.traces.dataset_reader.DatasetReader] | pyine.data.traces.dataset_reader.DatasetReader,
    base_filter: pyine.data.utils.filter_rules.FilterType | None = None,
    verbose: bool = False,
) -> list[TraceMetadata]:
    """Returns a list of TraceMetadata objects for all traces in the provided dataset reader(s).

    Args:
        readers: list of DatasetReader instances to get all traces from.
        base_filter: filter to apply to the dataset reader(s) to get target traces. If None, then
            no filter will be applied, and all available traces will be returned.
        verbose: whether to display a progress bar during parsing or not.

    Returns:
        A list of TraceMetadata objects for all targeted traces in the provided dataset.
    """
    if not isinstance(readers, list):
        readers = [readers]
    logger.info(f"preparing traces metadata for {len(readers)} dataset reader(s)...")
    readers_map = {  # hash-to-reader map to later re-identify the origin of individual traces
        reader.get_hash(): reader for reader in readers
    }
    all_trace_keys = []
    for reader in readers_map.values():
        all_trace_keys.extend(reader.trace_keys)
    if len(set(all_trace_keys)) != len(all_trace_keys):
        raise ValueError("there should be no duplicates in the list of trace keys across all datasets")
    prog_bar = tqdm.tqdm(total=len(all_trace_keys), desc="Parsing traces metadata", disable=not verbose)
    base_traces_meta: list[TraceMetadata] = []
    for reader_hash, reader in readers_map.items():
        for trace_idx in range(len(reader)):
            tags = reader.get_tags(trace_idx)
            is_banned = base_filter(tags) if base_filter is not None else False
            if not is_banned:
                base_traces_meta.append(
                    TraceMetadata(
                        identifier=reader.trace_keys[trace_idx],
                        index=trace_idx,
                        parent_dataset_hash=reader_hash,
                        tags=tags,
                    ),
                )
            prog_bar.update(1)
    prog_bar.close()
    return base_traces_meta


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
    """Assignments of coding problems identifiers (str) to data subsets."""
    split_hash: str
    """Hash of the split file where the assignments were parsed from."""

    @pydantic.model_validator(mode="after")
    def _post_validator(self) -> "TraceDatasetMetadata":
        """Confirms that all dataset traces contain reasonable types and the subsets do not overlap."""
        if not self.base_traces:
            raise ValueError("base traces must not be empty")
        seen_trace_ids: list[pyine.data.traces.dataset_utils.TraceIdentifier] = []
        expected_augment_types = _get_expected_augment_types()
        for trace_meta in self.base_traces:
            assert trace_meta.trace_id not in seen_trace_ids, f"duplicate trace id: {trace_meta.trace_id}"
            seen_trace_ids.append(trace_meta.trace_id)
            augm_type = trace_meta.trace_id.augment_category
            if augm_type is not None:
                if augm_type not in expected_augment_types:
                    raise ValueError(f"unexpected augment type: {augm_type}")
                assert trace_meta.is_augmented and trace_meta.augment_tags
                assert any(
                    [t == f"augment:{augm_type}" for t in trace_meta.augment_tags]
                ), "augment type is not in the trace tags; this should not happen?"
            else:
                assert not trace_meta.is_augmented and not trace_meta.is_multi_augmented
        leftover_trace_ids = [trace_meta.trace_id for trace_meta in self.leftover_traces]
        for trace_meta in self.leftover_traces:
            if trace_meta.trace_id not in seen_trace_ids:
                raise ValueError(f"trace id {trace_meta.trace_id} is not in the base traces")
        for subset_name, subset_traces in self.subset_traces.items():
            for trace_meta in subset_traces:
                if trace_meta.trace_id not in seen_trace_ids:
                    raise ValueError(f"trace id {trace_meta.trace_id} from {subset_name} is not in the base traces")
                if trace_meta.trace_id in leftover_trace_ids:
                    raise ValueError(f"trace id {trace_meta.trace_id} from {subset_name} is in leftover traces")
                problem_id_str = str(trace_meta.problem_id)
                if problem_id_str not in self.problem_assignments:
                    raise ValueError(f"problem id {problem_id_str} has no corresponding problem assignment")
                if self.problem_assignments[problem_id_str] != subset_name:
                    raise ValueError(f"problem id {problem_id_str} has incorrect assignment")
        return self


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

SampleInputType = typing.Literal[  # note: literal makes this type compatible with default collate
    "original",  # corresponds to using code and test case inputs/outputs from the source dataset as-is
    # all types below refer to augmented versions where we modify code and/or test case inputs/outputs
    "obfuscated",
    "obfuscated_hinted",
    "obfuscated_misleading",
    "stubbed",  # note: this type CANNOT have a real execution outcome tied to it (it cannot be executed)
    "hinted",
    "misleading",
    "bugged",  # note: this type should lead to different execution outcomes than the expected ones
    "bugged_hinted",  # same as the above (built as a buggy augmentation of a hinted trace)
    "bugged_misleading",  # same as the above (built as an misleading augmentation of a buggy trace)
]
"""Possible input types for trace execution samples:
- 'original': default sample type with the original code from the source dataset and no augmentations;
- 'obfuscated': augmented sample with obfuscated code (taken from the source dataset) but otherwise normal behavior;
- 'obfuscated_hinted': augmented sample with obfuscated code (taken from the source dataset) and execution output hints;
- 'obfuscated_misleading': augmented sample with obfuscated code (taken from the source dataset) and MISLEADING execution output hints;
- 'stubbed': augmented sample where part of the code is stubbed/hidden by an LLM to force models to infer execution steps;
- 'hinted': augmented sample where code contains one or more execution output hints produced by an LLM;
- 'misleading': augmented sample where code contains one or more MISLEADING execution output hints produced by an LLM;
- 'bugged': augmented sample where code contains one or more bugs that should affect execution outcomes;
- 'bugged_hinted': augmented sample where code contains both execution output hints and bugs;
- 'bugged_misleading': augmented sample where code contains both MISLEADING hints and bugs.
"""

_solution_specific_augment_types: list[SampleInputType] = [
    "obfuscated",
    "stubbed",
    "bugged",
]
"""List of augment types that are specific to a particular coding problem solution."""

_trace_specific_augment_types: list[SampleInputType] = [
    "obfuscated_hinted",
    "obfuscated_misleading",
    "hinted",
    "misleading",
    "bugged_hinted",
    "bugged_misleading",
]
"""List of augment types that are specific to a particular trace (i.e. to specific test case inputs/outputs)."""


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
    """Code string that should be interpreted and for which results must be predicted.

    Note: this is the full code snippet, not just the part that is relevant to the target execution
    outcome type (specified via `output_type`).

    This code snippet may have been replaced by a prompt result database lookup; you can check if
    `has_code_override = True` to see if this is the case. If so, the only valid `output_type`
    becomes 'program output' (which is the full program output), and the `expected_output` field
    might correspond to a different execution outcome than the "correct" one, depending on whether
    the code snippet contains an issue or not. You can tell this by checking the `code_type` field.
    """
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
    expected_output: str
    """Expected output that was previously verified/found, and that should be predicted by models.

    Note: this expected output is only "correct" when the code does NOT contain issues/bugs! Refer
    to the `code_type` field to determine this; in such cases, the expected output is actually not
    the "correct" output, but the "intended" output.
    """
    output_type: SampleOutputType
    """Type of the expected output (for specific descriptions in prompts)."""
    code_type: SampleInputType
    """Type of the provided code snippet (identifies whether it contains hints, stubs, bugs, ...)."""
    trace_step_count: int
    """Number of steps that are expected to be executed to predict the outputs.

    Unreliable if `has_code_override = True`.
    """
    comma_separated_tags: str
    """Comma-separated tags (e.g. 'augment:type,subset:train') that can be used to filter samples."""
    has_code_override: bool
    """Whether the code snippet has been overridden by a prompt result database lookup.

    Note: if this is True, then the only possible value for `output_type` should be 'program output',
    and the `expected_output` field might correspond to a different execution outcome than the
    "correct" one.
    """

    def get_trace_id(self) -> pyine.data.traces.dataset_utils.TraceIdentifier:
        """Returns the trace identifier object for this trace."""
        return pyine.data.traces.dataset_utils.TraceIdentifier.from_string(self.identifier)

    def get_tag_list(self) -> list[str]:
        """Returns the list of tags for this trace in its original format (one tag = one item)."""
        return self.comma_separated_tags.split(",")


SampleDataParserType = pyine.data.datamodule.BaseDataParserType[SampleData]
"""Type of the dataset reader used to read traces from LMDB datasets."""

SampleDataLoaderType = pyine.data.datamodule.BaseDataLoaderType[SampleData]
"""Type of the data loader used to batch trace sample data from the dataset parser."""

SampleTransformOutputProbMapType = dict[
    SampleOutputType,
    typing.Annotated[  # noqa
        pydantic.StrictFloat,
        pydantic.Field(ge=0.0, le=1.0),
    ],
]
"""Type of the probability map used to decide which sample type to generate for each trace."""

SampleTransformStrategyType = typing.Literal["never", "always", "if_too_long", "random", "hybrid"]
"""Possible strategies for generating samples from traces:
- 'never': always return full traces;
- 'always': always try to create partial samples (if possible, given limits below);
- 'if_too_long': create partial samples only if the trace/code exceeds configured thresholds;
- 'random': create partial samples with a fixed probability;
- 'hybrid': create partial samples if too long, otherwise with the configured probability.
"""


class SampleTransformConfig(pydantic.BaseModel):
    """Configuration class specifying arguments to transform raw trace sample into partial ones."""

    model_config = pydantic.ConfigDict(frozen=True, arbitrary_types_allowed=False, extra="forbid")
    """Pydantic model configuration (freezes the dataclass)."""

    seed: int | None = 0
    """Optional seed to initialize internal RNG for sampling decisions; None => nondeterministic."""
    transform_strategy: SampleTransformStrategyType = "never"
    """Strategy deciding when to create partial samples (see type docstring for more info)."""
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
    output_type_prob_map: SampleTransformOutputProbMapType = pydantic.Field(default_factory=dict)
    """Probability map used to determine potential output sample types in random/hybrid transform strategies."""

    @pydantic.model_validator(mode="after")
    def _validate_and_resolve(self) -> "SampleTransformConfig":
        """Validates the content of the config beyond basic validation."""
        if self.transform_strategy != "never" and not self.output_type_prob_map:
            raise ValueError("output type prob map must be provided when using partial samples generation")
        prob_map_total = sum([v for v in self.output_type_prob_map.values()])
        if prob_map_total < 0 or prob_map_total > 1:  # doesn't have to be 1, as fallback = program output
            raise ValueError(f"total probability map values must be in the range [0, 1]; got {prob_map_total}")
        return self


SampleSelectionInputProbMapType = dict[
    SampleInputType,
    typing.Annotated[  # noqa
        pydantic.StrictFloat,
        pydantic.Field(ge=0.0, le=1.0),
    ],
]
"""Type of the probability map used to decide which sample type to select for each trace."""

SampleSelectionChoiceStrategyType = typing.Literal["latest", "random"]
"""Possible strategies for selecting modified code snippets when multiple choices are available:
- 'latest': will pick and return the most-recently-generated code snippet;
- 'random': will randomly pick a code snippet from the available choices.
"""


def _get_default_code_input_type_prob_map() -> dict[SampleInputType, float]:
    """Returns the default probability map used to decide which sample type to select for each trace."""
    return {"original": 1.0}


class SampleSelectionConfig(pydantic.BaseModel):
    """Configuration class specifying arguments to select traces from a set for ."""

    model_config = pydantic.ConfigDict(frozen=True, arbitrary_types_allowed=False, extra="forbid")
    """Pydantic model configuration (freezes the dataclass)."""

    seed: int | None = 0
    """Optional seed to initialize internal RNG for sampling decisions; None => nondeterministic."""
    allow_db_lookups: bool = True
    """Whether to allow prompt result database lookups for code snippets (if missing)."""
    choice_strategy: SampleSelectionChoiceStrategyType = "latest"
    """Strategy deciding how to select code snippets when multiple choices exist (see type docstring for more info)."""
    input_type_prob_map: SampleSelectionInputProbMapType = _get_default_code_input_type_prob_map()
    """Probability map used to determine potential input sample types for random selection strategy."""

    @pydantic.model_validator(mode="after")
    def _validate_and_resolve(self) -> "SampleSelectionConfig":
        """Validates the content of the config beyond basic validation."""
        prob_map_total = sum([v for v in self.input_type_prob_map.values()])
        if not np.isclose(prob_map_total, 1.0):
            raise ValueError(f"total probability map values must be 1.0; got: {prob_map_total}")
        return self


def _draw_type(
    prob_map: dict[typing.Hashable, float],
    rng: np.random.Generator,
    default_fallback: typing.Hashable | None = None,
) -> typing.Hashable | None:
    """Draws a random sample type from the set of available types."""
    draw_val = rng.random()
    total_mass = 0.0
    for output_type, output_prob in prob_map.items():
        total_mass += output_prob
        if draw_val < total_mass:
            return output_type
    return default_fallback


LMDBDatasetReadersOrPathsType = (
    pyine.data.traces.dataset_reader.DatasetReader
    | list[pyine.data.traces.dataset_reader.DatasetReader]
    | str
    | pathlib.Path
    | list[str | pathlib.Path]
)
"""Type used to specify source dataset args in the sample builder."""


@dataclasses.dataclass(frozen=True)
class _TraceSampleSelectionResult:
    """Result of the sample selection process for a single trace."""

    trace_meta: TraceMetadata
    """The metadata associated with the trace from which to generate a sample."""
    code_type: SampleInputType
    """The type of the code snippet that will be used in the generated sample."""
    code_override: str | None = None  # if None, use the original code snippet
    """The code snippet override that will be used in the generated sample (instead of the original)."""


class SampleBuilder(SampleDataParserType):
    """Wrapper around the LMDB dataset reader(s) that returns sample data for target traces.

    The role of this class is to provide sample selection and preparation (transformation)
    strategies to build training/evaluation code execution samples from raw traces. By 'selection',
    we mean the strategy used to select which version of a traced code problem solution to use
    (augmented with hints/issues or not). By 'transformation', we mean the strategy used to
    transform a raw trace result into a set of input + expected output variables for prompts/models.

    The transformation strategy (specified via `SampleTransformConfig`) determines when to ask
    models to predict the execution of full code snippets ('program output'), and when to predict
    a portion of them (related to a function, a segment, etc.). 'Partial samples' can be used to
    diversify training data and make the prediction task easier in cases where e.g. traces are
    very long. See the docstring of `SampleOutputType` for more information on supported types.

    The selection strategy (specified via `SampleSelectionConfig`) determines which augmented
    code snippets to use (if any), and if augmentations are not present in the trace dataset
    itself, whether to go fetch them from the prompt result database (if available). If the prompt
    result database is used to retrieve augmented code snippets, then the only sample type that
    can be generated for that sample is the 'program output' (i.e. full code snippet). This is
    because we are lacking trace data required to prepare other output types here. See the
    docstring of `SampleInputType` for more information on supported types.
    """

    def __init__(
        self,
        source_data: LMDBDatasetReadersOrPathsType,  # noqa
        traces: list[TraceMetadata] | None = None,  # if `None`, will target all available traces
        transform_config: SampleTransformConfig | dict | None = None,
        selection_config: SampleSelectionConfig | dict | None = None,
        prompt_result_db_path: str | None = None,  # if `None`, will use framework default
    ) -> None:
        """Initializes the reader with a list of LMDB readers and a list of target traces."""
        if transform_config is None:
            transform_config = SampleTransformConfig()
        elif isinstance(transform_config, dict):
            transform_config = SampleTransformConfig(**transform_config)
        self.transform_config = transform_config
        if selection_config is None:
            selection_config = SampleSelectionConfig()
        elif isinstance(selection_config, dict):
            selection_config = SampleSelectionConfig(**selection_config)
        self.selection_config = selection_config
        if prompt_result_db_path is None:
            self.prompt_result_db = pyine.prompts.get_framework_db()
        else:
            self.prompt_result_db = pyine.prompts.PromptResultDB(prompt_result_db_path)
        self.readers_map, traces = self._init_readers_and_trace_metadata(source_data=source_data, traces=traces)
        self.selected_traces = self._select_samples_from_traces(
            traces=traces,
            selection_config=selection_config,
            prompt_result_db=self.prompt_result_db,
        )
        self.code_summaries = self._build_code_summaries_lut(
            traces=[t.trace_meta for t in self.selected_traces],
            prompt_result_db=self.prompt_result_db,
        )

    @staticmethod
    def _init_readers_and_trace_metadata(
        source_data: LMDBDatasetReadersOrPathsType,  # noqa
        traces: list[TraceMetadata] | None,  # if `None`, will target all available traces
    ) -> tuple[dict[str, pyine.data.traces.dataset_reader.DatasetReader], list[TraceMetadata]]:
        """Initializes a list of LMDB readers and a list of target traces."""
        if not isinstance(source_data, list):
            source_data = [source_data]
        logger.debug(f"initializing readers and metadata for:\n\t{'\n\t'.join([str(s) for s in source_data])}")
        for src_idx, src in enumerate(source_data):
            if isinstance(src, (str, pathlib.Path)):
                source_data[src_idx] = pyine.data.traces.dataset_reader.DatasetReader(pathlib.Path(src))
        readers_map = {r.get_hash(): r for r in source_data}
        assert len(readers_map) > 0
        if traces is None:
            logger.debug("(targeting all available traces)")
            # create a list of metadata structs for ALL available traces
            traces = pyine.organisms.datamodules.utils.samples.get_traces_metadata(
                readers=source_data,
                base_filter=None,
                verbose=False,
            )
        else:
            logger.debug(f"(targeting {len(traces)} traces)")
        assert isinstance(traces, list)
        assert all([t.parent_dataset_hash in readers_map for t in traces])
        assert all([0 <= t.index < len(readers_map[t.parent_dataset_hash]) for t in traces])
        return readers_map, traces

    @staticmethod
    def _select_samples_from_traces(
        traces: list[TraceMetadata],
        selection_config: SampleSelectionConfig,
        prompt_result_db: pyine.prompts.PromptResultDB,
    ) -> list[_TraceSampleSelectionResult]:
        """Selects samples to generate from traces according to the specified strategy/options."""
        # first, scan all available traces and identify which augment group they belong to
        TraceIdType = pyine.data.traces.dataset_utils.TraceIdentifier  # noqa
        trace_lut: dict[TraceIdType, TraceMetadata] = dict()
        cousin_traces: dict[TraceIdType, dict[SampleInputType, list[TraceIdType]]] = {}
        for trace in traces:
            assert trace.trace_id not in trace_lut, "trace id already exists in trace lut?"
            trace_lut[trace.trace_id] = trace
            augmentless_id = trace.trace_id.get_augmentless_identifier()
            if augmentless_id not in cousin_traces:
                cousin_traces[augmentless_id] = collections.defaultdict(list)
            if not trace.is_augmented:
                # unaugmented (i.e. original code + test inputs/outputs from source dataset)
                assert len(cousin_traces[augmentless_id]["original"]) == 0, (
                    "why do we have more than one trace for a given augmentless code snippet?"
                    # TODO: may become useful later when we start tracing nondeterministic code
                )
                cousin_traces[augmentless_id]["original"].append(trace.trace_id)
                continue
            # the only types of augmented traces that we expect from a trace dataset are:
            #   - traces with code hints or issues (exception 'hints/stubs': those CANNOT ever be traced)
            #   - obfuscated code traces (the only time we ever obfuscate code is in the dataset writer)
            assert not trace.is_multi_augmented, (
                "current implementation does not support multi-augmented traces; "
                "update the sample builder's trace selection code if dataset writer gets updated for augments"
            )
            augm_category = trace.trace_id.augment_category
            assert augm_category != "hints/stubs", "how can these have been traced?"
            is_obfuscated = augm_category == "obfuscated"
            is_hinted = augm_category.startswith("hints/")
            is_buggy = augm_category.startswith("issues/")
            assert (
                is_obfuscated or is_hinted or is_buggy
            ), f"trace id {trace.trace_id} has unexpected augment category: {augm_category}"
            if is_obfuscated:
                cousin_traces[augmentless_id]["obfuscated"].append(trace.trace_id)
            elif is_buggy:
                if augm_category == "issues/docs":
                    cousin_traces[augmentless_id]["misleading"].append(trace.trace_id)
                else:
                    cousin_traces[augmentless_id]["bugged"].append(trace.trace_id)
            elif is_hinted:
                cousin_traces[augmentless_id]["hinted"].append(trace.trace_id)
        assert len(cousin_traces) <= len(traces)
        assert sum([len(c) for t, dicts in cousin_traces.items() for c in dicts.values()]) == len(traces)
        # all 'cousin clusters' will be used to produce ONE trace sample each; pick which one according to strategy
        rng = np.random.default_rng(selection_config.seed)
        output_selections: list[_TraceSampleSelectionResult] = []
        for orig_trace_id, trace_map in cousin_traces.items():  # for each cousin trace cluster...
            assert orig_trace_id in trace_lut, "augmentless id not found in trace lut?"
            target_type = _draw_type(selection_config.input_type_prob_map, rng)  # draw the sample type...
            assert target_type is not None, "unexpected default fallback for input type draw"
            target_type = typing.cast(SampleInputType, target_type)
            if target_type == "original":
                # keep the original trace as-is, with no code snippet override
                assert target_type in trace_map, "missing orig trace in source data?"
                assert len(trace_map[target_type]) == 1 and trace_map[target_type][0] == orig_trace_id
                output_selections.append(
                    _TraceSampleSelectionResult(
                        trace_meta=trace_lut[orig_trace_id],
                        code_type=target_type,
                    )
                )
            elif target_type == "obfuscated":
                if target_type in trace_map:
                    # keep that trace as-is with no override under the assumption that obfuscation was done previously
                    assert len(trace_map["obfuscated"]) == 1, "missing obfuscated trace in dataset?"
                    output_selections.append(
                        _TraceSampleSelectionResult(
                            trace_meta=trace_lut[trace_map["obfuscated"][0]],
                            code_type=target_type,
                        )
                    )
                else:
                    # cannot obfuscate code here, skip the trace (we'd need to load more problem data to obfuscate)
                    pass
            elif target_type in trace_map and trace_map[target_type]:
                # if the augmentation we are looking for already exists, just pick a corresponding trace and use it
                if selection_config.choice_strategy == "random":
                    picked_idx = int(rng.integers(0, len(trace_map[target_type])))
                    picked_trace_id = trace_map[target_type][picked_idx]
                elif selection_config.choice_strategy == "latest":
                    picked_trace_id = list(sorted(trace_map[target_type], key=lambda tid: str(tid)))[-1]
                else:
                    raise NotImplementedError("unsupported random selection strategy")
                # keep that trace as-is with no override under the assumption that the traced code is already augmented
                output_selections.append(
                    _TraceSampleSelectionResult(
                        trace_meta=trace_lut[picked_trace_id],
                        code_type=target_type,
                    )
                )
            elif target_type not in trace_map or not trace_map[target_type]:
                # if the augmentation we are looking for does not exist, generate it using prompt result db lookups
                if not selection_config.allow_db_lookups:
                    continue  # lookups disabled, skip this instance
                # first, determine the strategy to look up previously generated prompt results for the target type
                augment_is_soluton_specific = target_type in _solution_specific_augment_types
                augment_is_trace_specific = target_type in _trace_specific_augment_types
                # next, determine what parent trace to use for lookups (orig or obfuscated)
                if target_type.startswith("obfuscated_"):
                    assert len(trace_map["obfuscated"]) == 1, "missing obfuscated trace in dataset?"
                    target_trace_meta = trace_lut[trace_map["obfuscated"][0]]
                else:
                    target_trace_meta = trace_lut[orig_trace_id]
                # now, go and fetch the required records to assemble the selected sample result
                if augment_is_soluton_specific:
                    # relevant prompt result db entries should be attached to the parent solution id
                    if target_type == "stubbed":
                        records = prompt_result_db.get_by_identifier(
                            identifier=str(target_trace_meta.solution_id),
                            prompt_name="hints/stubs",
                        )
                    else:
                        assert target_type == "bugged", "branching logic error"
                        records = prompt_result_db.get_by_identifier(identifier=str(target_trace_meta.solution_id))
                        records = [
                            rec  # keep records that match any kind of issue/bug except 'issues/docs'
                            for rec in records
                            if (
                                rec.prompt_name is not None
                                and rec.prompt_name.startswith("issues/")
                                and rec.prompt_name != "issues/docs"  # reserved for 'misleading' augm types
                            )
                        ]
                elif augment_is_trace_specific:
                    # relevant prompt result db entries should be attached to the trace id itself
                    records = prompt_result_db.get_by_identifier(identifier=str(target_trace_meta.trace_id))
                    if target_type == "hinted":
                        records = [
                            r for r in records if r.prompt_name.startswith("hints/") and r.prompt_name != "hints/stubs"
                        ]
                    elif target_type == "misleading":
                        records = [r for r in records if r.prompt_name == "issues/docs"]
                    else:
                        records = [r for r in records if f"augment:{target_type}" in r.tags]
                else:
                    raise NotImplementedError(f"missing augment handling for: {target_type}")
                # given the records we have found...
                if not records:
                    continue  # no database match found, skip this trace
                potential_code_snippet_overrides = [r.result for r in records]  # kept in order, last = most recent
                if selection_config.choice_strategy == "random":
                    picked_code_override_idx = int(rng.integers(0, len(potential_code_snippet_overrides)))
                elif selection_config.choice_strategy == "latest":
                    picked_code_override_idx = -1
                else:
                    raise NotImplementedError
                # append the resulting sample, but with the code snippet override from the database
                # (note: in these cases, the only valid sample output type will be 'program output',
                #  as we cannot correctly deduce anything trace-related without re-tracing entirely)
                output_selections.append(
                    _TraceSampleSelectionResult(
                        trace_meta=target_trace_meta,
                        code_type=target_type,
                        code_override=potential_code_snippet_overrides[picked_code_override_idx],
                    )
                )
            else:
                raise NotImplementedError
        logger.debug(f"trace selection strategy kept {len(output_selections)} samples for {len(traces)} traces")
        if output_selections:
            code_type_counts = collections.Counter([s.code_type for s in output_selections])
            output_types_str = "\n\t".join([f"{k}: {c}" for k, c in code_type_counts.items()])
            logger.debug(f"selected sample code types:\n\t{output_types_str}")
            code_override_flags = [bool(c.code_override) for c in output_selections]
            logger.debug(f"selected samples with code override: {sum(code_override_flags)}")
        return output_selections

    @staticmethod
    def _build_code_summaries_lut(
        traces: list[TraceMetadata],
        prompt_result_db: pyine.prompts.PromptResultDB,
    ) -> dict[pyine.data.traces.dataset_utils.SolutionIdentifier, str]:  # sid to code description map
        """Builds a lookup table of code summaries for each trace."""
        code_summaries_lut: dict[pyine.data.traces.dataset_utils.SolutionIdentifier, str] = dict()
        for trace_meta in traces:
            if trace_meta.solution_id not in code_summaries_lut:
                records = prompt_result_db.get_by_identifier(
                    identifier=str(trace_meta.solution_id),
                    prompt_name="code_summary",
                )
                if records:
                    # always take the latest summary that's available in the database
                    code_summaries_lut[trace_meta.solution_id] = records[-1].result
        logger.debug(f"found {len(code_summaries_lut)} code summaries in prompt result db")
        return code_summaries_lut

    def get_stats(self) -> dict[str, int | float | str]:
        """Returns a dictionary of statistics for the prepared samples."""
        # note: we cannot provide stats on the transformed samples as their types are resolved later
        code_type_counts = collections.Counter([s.code_type for s in self.selected_traces])
        code_override_flags = [bool(c.code_override) for c in self.selected_traces]
        return {
            "sample_count": len(self.selected_traces),
            **{f"code_type_counts/{k}": c for k, c in code_type_counts.items()},
            "code_overrides_count": sum(code_override_flags),
            "code_summaries_count": len(self.code_summaries),
        }

    def __len__(self) -> int:
        """Returns the number of traces that will be converted into samples."""
        return len(self.selected_traces)

    def __getitem__(self, idx: int) -> SampleData:
        """Returns the data sample associated with a specified index (where `0 <= idx < len(self)`).

        Notes:
          - We perform a random draw across all potential output types (with the full program output
            as a fallback option) to decide which sample type to generate. If a sample code (input)
            type is not available in the trace dataset, the sample output type will be forced to
            the fallback full program output type.
          - If a function is targeted as the sample output type: the input arguments will be call
            arguments, and the expected output is the function's returned value(s).
          - If an arbitrary code segment is targeted as the sample output type: input args are the
            frame variables at the start of the segment, and outputs are the frame variables at
            the end of the segment (at a specific line, or after a given number of steps).
          - If `max_partial_trace_steps` is set, when generating a sample that targets a function
            or a segment as its output type, functions exceeding this cap are ignored and segments
            are truncated to respect the cap.
          - If for a generated candidate, `max_inputs_str_length` or `max_output_str_length` caps
            are set and exceeded, returns the 'fallback option' again (the full program output).
        """
        if not (0 <= idx < len(self)):
            raise IndexError(f"index {idx} out of range")
        selected_trace = self.selected_traces[idx]
        reader = self.readers_map[selected_trace.trace_meta.parent_dataset_hash]
        trace_data = reader[selected_trace.trace_meta.index]
        assert trace_data.identifier is not None, "trace identifier is required"
        assert trace_data.identifier == selected_trace.trace_meta.identifier, "trace identifier mismatch"
        if self.transform_config.seed is not None:
            t_rng = np.random.default_rng(self.transform_config.seed + idx)  # reproducible trace-specific rng
        else:
            t_rng = np.random.default_rng()
        if selected_trace.code_override is not None:
            # if we have a code snippet override, we do NOT have trace events associated with it
            # (therefore, the only possible output type is the full program output)
            picked_output_type: SampleOutputType = "program output"
        else:
            picked_output_type = self._pick_output_type(trace_data=trace_data, rng=t_rng)
        if picked_output_type == "function return":
            # first, if requested, try to generate a sample for a function call
            sample = self._get_function_call_sample(
                trace_data=trace_data,
                trace_meta=selected_trace.trace_meta,
                trace_code_type=selected_trace.code_type,
                rng=t_rng,
            )
            if sample is not None:
                # if we did successfully build a partial sample, return it now
                return sample
        if picked_output_type != "program output" and (
            picked_output_type != "function return" or self.transform_config.functions_fallback_to_segments
        ):
            # if requested (or as a fallback from the function call sample), try to generate a segment sample
            sample = self._get_code_segment_sample(
                trace_data=trace_data,
                trace_meta=selected_trace.trace_meta,
                trace_code_type=selected_trace.code_type,
                target_output_type=picked_output_type,
                rng=t_rng,
            )
            if sample is not None:
                # if we did successfully build a partial sample, return it now
                return sample
        # ultimate fallback: return a sample for the full program output
        return SampleData(
            identifier=trace_data.identifier,
            code=selected_trace.code_override if selected_trace.code_override else trace_data.code_string,
            description=self.code_summaries.get(selected_trace.trace_meta.solution_id, ""),
            entrypoint=str(trace_data.entrypoint_name),
            first_line=0,
            last_line=len(trace_data.code_string.splitlines()),
            inputs=str(trace_data.inputs),
            expected_output=str(trace_data.expected_output),
            output_type="program output",
            code_type=selected_trace.code_type,
            trace_step_count=trace_data.valid_step_count,  # count valid steps only
            comma_separated_tags=selected_trace.trace_meta.comma_separated_tags,
            has_code_override=selected_trace.code_override is not None,
        )

    def _satisfies_str_caps(self, inp: str, out: str) -> bool:
        """Returns whether inputs/output strings satisfy caps or not."""
        if (
            self.transform_config.max_inputs_str_length is not None
            and len(inp) > self.transform_config.max_inputs_str_length
        ):
            return False
        if (
            self.transform_config.max_output_str_length is not None
            and len(out) > self.transform_config.max_output_str_length
        ):
            return False
        return True

    def _pick_output_type(
        self,
        trace_data: pyine.utils.code.execution.TraceResult,
        rng: np.random.Generator,
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
        if self.transform_config.transform_strategy == "always":
            try_partial_sample = True
        elif self.transform_config.transform_strategy != "never":
            if self.transform_config.transform_strategy in ["if_too_long", "hybrid"]:
                is_too_long = (
                    trace_data.total_step_count >= self.transform_config.too_long_total_steps_threshold
                    or trace_data.valid_step_count >= self.transform_config.too_long_valid_steps_threshold
                    or len(trace_data.code_string.splitlines()) >= self.transform_config.too_long_code_lines_threshold
                )
                if is_too_long:  # if the trace is too long, always try to generate a partial sample
                    try_partial_sample = True
            if not try_partial_sample and self.transform_config.transform_strategy in [
                "hybrid",
                "random",
            ]:
                # if the trace is not too long, we can still generate a partial sample randomly
                total_partial_mass = sum(
                    [
                        prob
                        for output_type, prob in self.transform_config.output_type_prob_map.items()
                        if output_type != default_fallback  # full program output gets the mass balance
                    ]
                )
                try_partial_sample = rng.random() < total_partial_mass
        if not try_partial_sample:
            # if we still have not managed to decide to make a partial sample, return to full trace now
            return default_fallback
        # otherwise, decide what kind of output type to generate for the partial sample
        output_type = _draw_type(self.transform_config.output_type_prob_map, rng, default_fallback)
        output_type = typing.cast(SampleOutputType, output_type)
        return output_type

    def _get_function_call_sample(
        self,
        trace_data: pyine.utils.code.execution.TraceResult,
        trace_meta: TraceMetadata,
        trace_code_type: SampleInputType,
        rng: np.random.Generator,
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
            curr_candidate_idx = int(rng.integers(0, len(candidate_events)))
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
            if (
                self.transform_config.max_partial_trace_steps
                and call_step_count > self.transform_config.max_partial_trace_steps
            ):
                # enforce step cap: if exceeded, skip this candidate
                continue
            if (
                self.transform_config.min_partial_trace_steps
                and call_step_count < self.transform_config.min_partial_trace_steps
            ):
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
                description=self.code_summaries.get(trace_meta.solution_id, ""),
                entrypoint=target_func_name,
                first_line=first_line,
                last_line=last_line,
                inputs=call_args_str,
                expected_output=function_output_str,
                output_type="function return",
                code_type=trace_code_type,
                trace_step_count=call_step_count,
                comma_separated_tags=trace_meta.comma_separated_tags,
                has_code_override=False,
            )
        return None  # no more candidates to consider, failed to get a function call

    def _get_code_segment_sample(
        self,
        trace_data: pyine.utils.code.execution.TraceResult,
        trace_meta: TraceMetadata,
        trace_code_type: SampleInputType,
        target_output_type: SampleOutputType,
        rng: np.random.Generator,
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
                    if range_step_count >= (self.transform_config.min_partial_trace_steps or 1):
                        candidate_event_lists.append(depth_collectors[current_depth])
                    depth_collectors[current_depth] = None
                current_depth -= 1
        # we expect to have closed all collections by encountering matching RETURN events
        if any(depth_collectors.values()):
            # how did we end up with a trace ending without a return event?
            # (might need to investigate these later on, will drop top scope as a potential candidate)
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
            if self.transform_config.max_partial_trace_steps:
                max_step_count = min(self.transform_config.max_partial_trace_steps, len(range_steps) - 1)
            else:
                max_step_count = len(range_steps) - 1
            min_step_count = self.transform_config.min_partial_trace_steps or 1
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
            if self.transform_config.combine_local_and_global_vars_for_partial_samples:
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
                description=self.code_summaries.get(trace_meta.solution_id, ""),
                entrypoint="",
                first_line=first_line,
                last_line=last_line,
                inputs=input_vars_str,
                expected_output=output_vars_str,
                output_type="frame variables",
                code_type=trace_code_type,
                trace_step_count=segment_size,
                comma_separated_tags=trace_meta.comma_separated_tags,
                has_code_override=False,
            )
        return None  # no more candidate lists to consider, failed to get a segment sample


class SampleBuilderConfig(pyine.data.datamodule.ConversationDataParserConfig):
    """Configuration class for the (raw) dataset trace sample builder."""

    class_path: str = pyine.utils.portability.get_fully_qualified_name(SampleBuilder)
    """Fully qualified class path for the trace parser."""
    params: dict[str, typing.Any] = dict(
        transform_config=SampleTransformConfig(),
        selection_config=SampleSelectionConfig(),
    )
    """Default parameters for the dataset trace parser."""

    @typing.override
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

    @staticmethod
    def get_special_subset_param_overrides(
        subset_type: pyine.data.datamodule.SubsetNameType,
    ) -> dict[str, typing.Any]:
        """Returns special subset parameter overrides (if any) for the given subset type."""
        special_subset_overrides: dict[str, typing.Any] = dict()
        for suffix in typing.get_args(SampleInputType):
            if suffix != "original" and subset_type.endswith(f"_{suffix}"):
                special_subset_overrides = dict(
                    selection_config=SampleSelectionConfig(
                        choice_strategy="latest",
                        input_type_prob_map={suffix: 1.0},
                    )
                )
                break
        return special_subset_overrides
