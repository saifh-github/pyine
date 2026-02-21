"""Common utilities and defines for trace data sample preparation.

Important note: there are strong links between the defines in this module, the prompt templates that
are supported in the annotator app, the way the trace dataset writer saves augmentation info, and
the way trace identifiers can be used to determine if a trace is augmented (and how). If you change
anything across any of these, make sure to also update other areas accordingly.
"""

from __future__ import annotations

import collections
import collections.abc
import contextlib
import dataclasses
import enum
import functools
import itertools
import logging
import typing

import numpy as np
import pydantic
import torch.utils.data

import pyine.data.datamodule
import pyine.data.traces.dataset_utils
import pyine.prompts
import pyine.prompts.manager

logger = logging.getLogger(__name__)

__all__ = [
    "PregeneratedOutputRecord",
    "SampleCodeType",
    "SampleCodeTypeSet",
    "SampleCodeTypeSetProbMap",
    "SampleDataParser",
    "SampleDataLoader",
    "SamplePredictType",
    "SamplePredictTypeProbMap",
    "SampleTransformStrategy",
    "SampleData",
    "SampleSubsetTagWrapper",
    "TraceToSampleCodeTypeMapping",
    "TraceDatasetToSampleCodeTypeMappings",
    "append_parser_tag",
    "check_trace_code_is_test_case_specific",
    "convert_to_comma_separated_tags",
    "get_all_supported_code_type_sets",
    "get_all_supported_code_type_sets_suffixes",
    "get_code_type_set_from_str",
    "get_prompt_names_relevant_to_sample_code_types",
    "draw_type",
    "get_rng",
    "get_records_matching_code_type",
    "has_record_for_code_type",
    "get_records_matching_code_type_batch",
    "check_code_type_availability_batch",
    "hint_type_to_sample_code_type",
    "strip_id_suffix",
]


class SampleCodeType(enum.StrEnum):
    """Possible code types for trace execution samples."""

    original = enum.auto()  # corresponds to using code and test case inputs/outputs from the source dataset as-is
    """Default sample type with the original code from the source dataset and no augmentations."""
    # all types below refer to augmented versions where we modify code or test inputs/outputs
    obfuscated = enum.auto()
    """Sample type where code is obfuscated (e.g. by replacing function/variable names with a,b,c,...)."""
    stubbed = enum.auto()  # note: this type CANNOT have a real execution outcome tied to it (it cannot be executed)
    """Sample type where part of the code is stubbed/hidden by an LLM to force models to infer execution steps."""
    hinted = enum.auto()
    """Sample type where code contains one or more execution output hints produced by an LLM."""
    misleading = enum.auto()
    """Sample type where code contains one or more MISLEADING execution output hints produced by an LLM."""
    bugged = enum.auto()
    """Sample type where code contains one or more bugs that should affect execution outcomes."""


_FORBIDDEN_SAMPLE_CODE_TYPE_SETS: set[frozenset[SampleCodeType]] = {
    # we can't combine 'original' samples with any other flag (but singleton {original} is allowed)
    *[frozenset({SampleCodeType.original, t}) for t in SampleCodeType if t != SampleCodeType.original],
    # stubbed code is hard to combine with hinted/buggy/misleading code without maybe breaking those
    # (stubbed + obfuscated is allowed, and singleton {stubbed} is allowed)
    *[
        frozenset({SampleCodeType.stubbed, t})
        for t in SampleCodeType
        if t not in (SampleCodeType.obfuscated, SampleCodeType.stubbed)
    ],
    # it makes no sense to create both misleading and hinted samples
    frozenset({SampleCodeType.misleading, SampleCodeType.hinted}),
}
"""Set of sample code type sets that are forbidden (i.e. should not be generated/used)."""


@dataclasses.dataclass(frozen=True, slots=True)
class SampleCodeTypeSet:
    """Represents a set of sample code types."""

    types: frozenset[SampleCodeType]
    """The set of sample code types."""

    def __post_init__(self) -> None:
        """Normalizes and validates sample code types."""
        assert isinstance(self.types, frozenset)
        assert all(isinstance(s, SampleCodeType) for s in self.types)
        self._validate_sample_code_types(self.types)

    @staticmethod
    def _validate_sample_code_types(
        sample_code_types: typing.Iterable[SampleCodeType],
    ) -> None:
        """Validates that the given sample code types are valid (i.e. not forbidden)."""
        types_set = frozenset(sample_code_types)
        for forbidden_set in _FORBIDDEN_SAMPLE_CODE_TYPE_SETS:
            if forbidden_set.issubset(types_set):
                raise ValueError(f"sample code types set is forbidden (contains {forbidden_set}): {sample_code_types}")

    def has(self, t: SampleCodeType) -> bool:
        """Returns True if the given type is present in the set."""
        return t in self.types

    def has_all(self, req_types: typing.Iterable[SampleCodeType]) -> bool:
        """Returns True if all required types are present."""
        return frozenset(req_types).issubset(self.types)

    def has_any(self, candidate_types: typing.Iterable[SampleCodeType]) -> bool:
        """Returns True if any of the candidate types is present."""
        return not self.types.isdisjoint(frozenset(candidate_types))

    @typing.no_type_check
    def __eq__(self, other: object) -> bool:
        """Returns True if the given object is a set of sample code types with the same types."""
        if isinstance(other, SampleCodeTypeSet):
            return self.types == other.types
        if isinstance(other, (SampleCodeType, str)):
            return self.types == frozenset({other})
        if isinstance(other, collections.abc.Iterable) and all(isinstance(s, SampleCodeType) for s in other):
            return self.has_all(other)
        raise NotImplementedError(f"cannot compare sample code type sets with {type(other)}")

    @property
    def is_original(self) -> bool:
        """Returns True if the set contains only the original sample code type."""
        return self.types == frozenset({SampleCodeType.original})

    @property
    def is_augmented(self) -> bool:
        """Returns True if the set contains any augmented sample code type."""
        return not self.is_original

    @property
    def is_multi_augmented(self) -> bool:
        """Returns True if the set contains more than one augmented sample code type."""
        return len(self.types) > 1

    @property
    def is_obfuscated(self) -> bool:
        """Returns True if the set contains the obfuscated sample code type."""
        return self.has(SampleCodeType.obfuscated)

    @property
    def is_stubbed(self) -> bool:
        """Returns True if the set contains the stubbed sample code type."""
        return self.has(SampleCodeType.stubbed)

    @property
    def is_hinted(self) -> bool:
        """Returns True if the set contains the hinted sample code type."""
        return self.has(SampleCodeType.hinted)

    @property
    def is_misleading(self) -> bool:
        """Returns True if the set contains the misleading sample code type."""
        return self.has(SampleCodeType.misleading)

    @property
    def is_bugged(self) -> bool:
        """Returns True if the set contains the bugged sample code type."""
        return self.has(SampleCodeType.bugged)

    def get_hintable_base_augments(self) -> frozenset[SampleCodeType]:
        """Return base augments that can be combined with hints for prompt-DB lookup.

        Extracts non-hint augments like {obfuscated}, {bugged} that can form valid combinations
        with hint types (e.g., {obfuscated, hinted}).

        Returns empty frozenset when:
        - trace contains `stubbed`, as {stubbed, X, hinted} is always invalid;
        - trace is `{original}` (use `is_original` property to identify this special case);
        - trace is hint-only (`{hinted}`, `{misleading}`): no base augments to preserve.

        Examples:
            SampleCodeTypeSet({obfuscated, hinted}).get_hintable_base_augments()
            -> frozenset({obfuscated})  # hinted is STRIPPED, returns base only

            SampleCodeTypeSet({obfuscated, bugged}).get_hintable_base_augments()
            -> frozenset({obfuscated, bugged})  # no hints to strip

            SampleCodeTypeSet({original}).get_hintable_base_augments()
            -> frozenset()  # SPECIAL CASE: confirm via is_original property

            SampleCodeTypeSet({stubbed}).get_hintable_base_augments()
            -> frozenset()  # stubbed cannot combine with hints

            SampleCodeTypeSet({stubbed, obfuscated}).get_hintable_base_augments()
            -> frozenset()  # contains stubbed, entire trace is hint-incompatible

            SampleCodeTypeSet({hinted}).get_hintable_base_augments()
            -> frozenset()  # hint-only trace, no base augments
        """
        if SampleCodeType.stubbed in self.types:
            return frozenset()
        hint_types = {SampleCodeType.hinted, SampleCodeType.misleading}
        cannot_combine_with_hints = {SampleCodeType.original, SampleCodeType.stubbed}
        excluded = hint_types | cannot_combine_with_hints
        return frozenset(t for t in self.types if t not in excluded)

    def can_receive_hint_type(self, target_hint_type: SampleCodeType) -> bool:
        """Check if this code type set can receive a specific new hint type.

        Returns True only for traces that have NO existing hints AND are hint-compatible. Cross-hint
        augmentation is NOT allowed because:
        - `{hinted, misleading}` is an invalid code type combination;
        - for hint-only traces, the result would be potentially overlapping hints (unreliable).

        Args:
            target_hint_type: The hint type to potentially add (i.e. 'hinted' or 'misleading').

        Returns True if:
        - the trace has NO preexisting hints (neither hinted nor misleading);
        - the trace does NOT contain stubbed (it's hard to tell if hints would still work/apply);
        - the trace is {original} OR has valid base augments.

        Returns False if:
        - the trace already has ANY hint type (hinted or misleading);
        - the trace contains stubbed (stubbed + hints is invalid).

        Examples:
            SampleCodeTypeSet({original}).can_receive_hint_type(hinted) -> True
            SampleCodeTypeSet({obfuscated}).can_receive_hint_type(hinted) -> True
            SampleCodeTypeSet({obfuscated, hinted}).can_receive_hint_type(hinted) -> False
            SampleCodeTypeSet({obfuscated, hinted}).can_receive_hint_type(misleading) -> False
            SampleCodeTypeSet({hinted}).can_receive_hint_type(misleading) -> False
            SampleCodeTypeSet({stubbed}).can_receive_hint_type(hinted) -> False
            SampleCodeTypeSet({stubbed, obfuscated}).can_receive_hint_type(hinted) -> False
        """
        if SampleCodeType.hinted in self.types or SampleCodeType.misleading in self.types:
            return False
        if SampleCodeType.stubbed in self.types:
            return False
        if self.is_original:
            return True
        return bool(self.get_hintable_base_augments())

    def can_receive_any_hints(self) -> bool:
        """Check if this code type set can receive ANY hints via prompt-DB.

        Delegates to can_receive_hint_type() since all hint types have identical
        eligibility rules.

        Examples:
            SampleCodeTypeSet({original}).can_receive_any_hints() -> True
            SampleCodeTypeSet({obfuscated}).can_receive_any_hints() -> True
            SampleCodeTypeSet({hinted}).can_receive_any_hints() -> False
            SampleCodeTypeSet({obfuscated, hinted}).can_receive_any_hints() -> False
            SampleCodeTypeSet({stubbed}).can_receive_any_hints() -> False
        """
        return self.can_receive_hint_type(SampleCodeType.hinted)

    def get_counterfactual_grouping_key(self) -> tuple[str, ...] | str:
        """Get grouping key for counterfactual evaluations.

        Used to group traces for counterfactual comparisons, ensuring each group compares traces
        with the same base augments.

        Returns a hashable key that identifies the base augment identity:
        - "original" for {original} traces AND hint-only traces ({hinted}, {misleading});
        - "stubbed" for traces containing stubbed (hint-incompatible);
        - tuple of sorted augment names for other traces.

        IMPORTANT: hint-only traces like {hinted} are treated as "original" because they are
        semantically "original code + hints". This ensures {original} and {hinted} are grouped
        together for counterfactual pairing.

        Examples:
            SampleCodeTypeSet({original}).get_counterfactual_grouping_key() -> "original"
            SampleCodeTypeSet({hinted}).get_counterfactual_grouping_key() -> "original"
            SampleCodeTypeSet({misleading}).get_counterfactual_grouping_key() -> "original"
            SampleCodeTypeSet({obfuscated}).get_counterfactual_grouping_key() -> ("obfuscated",)
            SampleCodeTypeSet({obfuscated, bugged}).get_counterfactual_grouping_key() -> ("bugged", "obfuscated")
            SampleCodeTypeSet({obfuscated, hinted}).get_counterfactual_grouping_key() -> ("obfuscated",)
            SampleCodeTypeSet({stubbed}).get_counterfactual_grouping_key() -> "stubbed"
            SampleCodeTypeSet({stubbed, obfuscated}).get_counterfactual_grouping_key() -> "stubbed"
        """
        # stubbed traces are hint-incompatible, group separately
        if SampleCodeType.stubbed in self.types:
            return "stubbed"
        if self.is_original:
            return "original"
        base_augs = self.get_hintable_base_augments()
        if not base_augs:
            # hint-only traces ({hinted}, {misleading}) have no base augments
            # but are semantically "original + hints", so group with original
            return "original"
        return tuple(sorted(t.value for t in base_augs))

    @staticmethod
    def create_from_tags(tags: typing.Iterable[str]) -> SampleCodeTypeSet:
        """Creates a sample code type set from the given tags."""
        found_types: set[SampleCodeType] = set()
        orig_set = frozenset({SampleCodeType.original})
        for tag in tags:
            if tag.startswith("augment:"):
                augment_type = tag.split(":", maxsplit=1)[1]
                code_type_set = get_code_type_set_from_str(augment_type)
                if code_type_set != orig_set:
                    found_types.update(code_type_set)
        if not found_types:
            return SampleCodeTypeSet(orig_set)
        return SampleCodeTypeSet(frozenset(found_types))

    @staticmethod
    def create_from_trace(
        trace: pyine.data.traces.dataset_utils.TraceMetadata | pyine.data.traces.dataset_utils.TraceIdentifier,
    ) -> SampleCodeTypeSet:
        """Creates a sample code type set for the given trace data."""
        if isinstance(trace, pyine.data.traces.dataset_utils.TraceMetadata):
            trace_id = trace.trace_id
        else:
            assert isinstance(trace, pyine.data.traces.dataset_utils.TraceIdentifier)
            trace_id = trace
        type_set = SampleCodeTypeSet(get_code_type_set_from_str(trace_id.augment_category))
        # do some validation, for good measure --- all of these should always stay aligned across objects
        fields_to_check = ["is_augmented", "is_bugged", "is_hinted", "is_misleading", "is_obfuscated"]
        for field_name in fields_to_check:
            assert getattr(type_set, field_name) == getattr(trace_id, field_name)
        assert not type_set.is_stubbed, "can't happen in practice? (stubbed code can't be traced)"
        assert sum([type_set.is_augmented, type_set.is_original]) == 1
        return type_set

    @staticmethod
    def create_default() -> SampleCodeTypeSet:
        """Creates a default sample code type set (i.e. the original sample code type)."""
        return SampleCodeTypeSet(frozenset({SampleCodeType.original}))

    def __str__(self) -> str:
        """Returns a string representation of the sample code type set."""
        return "_".join(sorted(self.types))


_CODE_TYPES_THAT_TARGET_SPECIFIC_TESTS: frozenset[SampleCodeType] = frozenset(
    [
        SampleCodeType.hinted,
        SampleCodeType.misleading,
    ]
)
"""List of sample code types that are specific to a particular parent trace or test case.

In other words, samples that possess any of these types refer to particular test cases, and thus
CANNOT be used with any test case but the one they were designed for.
"""


def check_trace_code_is_test_case_specific(
    trace: pyine.data.traces.dataset_utils.TraceMetadata | pyine.data.traces.dataset_utils.TraceIdentifier,
) -> bool:
    """Returns whether the given trace is tied to a particular test case (due to e.g. hints)."""
    return SampleCodeTypeSet.create_from_trace(trace).has_any(_CODE_TYPES_THAT_TARGET_SPECIFIC_TESTS)


def hint_type_to_sample_code_type(
    hint_type: typing.Any,  # pyine.organisms.datamodules.samples.configs.HintType (late import)
) -> SampleCodeType:
    """Map HintType enum to corresponding SampleCodeType for internal use.

    This handles the naming mismatch between the two enums:
    - HintType.helpful -> SampleCodeType.hinted
    - HintType.misleading -> SampleCodeType.misleading

    Args:
        hint_type: The HintType value to convert (from samples.configs module).

    Returns:
        The corresponding SampleCodeType value.

    Raises:
        ValueError: If hint_type is not a valid HintType.
    """
    # import here to avoid circular imports (configs.py imports common.py)
    import pyine.organisms.datamodules.samples.configs

    if hint_type == pyine.organisms.datamodules.samples.configs.HintType.helpful:
        return SampleCodeType.hinted
    if hint_type == pyine.organisms.datamodules.samples.configs.HintType.misleading:
        return SampleCodeType.misleading
    raise ValueError(f"Unknown hint type: {hint_type}")


def strip_id_suffix(identifier: str) -> str:
    """Remove known identifier suffixes added by dataset wrappers.

    IMPORTANT: This should ONLY be applied to `SampleData.identifier` (string), NEVER to
    `TraceIdentifier` objects. Suffixes are added by ``SampleHintIdentifierWrapper``
    (``::hinted``, ``::misleading``, ``::hintless``) and ``SampleKeywordManipulatorWrapper``
    (``::cf_with``, ``::cf_without``) for cache/eval uniqueness; this helper removes them
    before trace ID parsing and LMDB/prompt-DB lookups.

    Args:
        identifier: string identifier, possibly with a ``::`` suffix from a dataset wrapper.

    Returns:
        Identifier with suffix stripped (or unchanged if no suffix present).
    """
    for suffix in ("::hinted", "::misleading", "::hintless", "::cf_with", "::cf_without"):
        if identifier.endswith(suffix):
            return identifier[: -len(suffix)]
    return identifier


def get_all_supported_code_type_sets() -> list[SampleCodeTypeSet]:
    """Returns the list of all supported sample code type sets."""
    supported_type_sets: list[frozenset[SampleCodeType]] = []
    for code_type_set_size in range(1, len(SampleCodeType) + 1):
        for code_type_set_members in itertools.combinations(list(SampleCodeType), code_type_set_size):
            code_type_set = frozenset(SampleCodeType(m) for m in code_type_set_members)
            if not any(forbidden_set.issubset(code_type_set) for forbidden_set in _FORBIDDEN_SAMPLE_CODE_TYPE_SETS):
                supported_type_sets.append(code_type_set)
    return [SampleCodeTypeSet(code_type_set) for code_type_set in supported_type_sets]


def get_all_supported_code_type_sets_suffixes() -> list[str]:
    """Returns the list of all string suffixes that identify supported sample code type sets.

    These are meant to be added to e.g. datamodule subset names, and can be used in combination with
    the `get_code_type_set_from_str` function to retrieve the corresponding sample code type set.

    Note that there is no suffix that corresponds to the original sample code type set.
    """
    output_strs: list[str] = []
    for code_type_set in get_all_supported_code_type_sets():
        if code_type_set != SampleCodeTypeSet.create_default():
            output_strs.append(str("_".join(code_type_set.types)))
    return output_strs


class SamplePredictType(enum.StrEnum):
    """Possible outcome prediction types for trace execution samples."""

    program_output = enum.auto()
    """The prediction should be the final output of the program after executing the entire code."""
    frame_variables = enum.auto()
    """The prediction should be the full description of all frame_variables at the target line/step."""
    next_step_key = enum.auto()  # @@@@ TODO: not implemented yet!
    """The prediction should be the trace key of the next event after executing the code up to a target line/step."""
    function_return = enum.auto()
    """The prediction should be the return value of a specific function called with specific arguments."""


class SampleTransformStrategy(enum.StrEnum):
    """Possible strategies for generating samples from traces."""

    never = enum.auto()
    """Always return samples whose prediction type is full program outcomes; never create partial samples."""
    always = enum.auto()
    """Always attempt to create partial samples (if possible, given configured thresholds)."""
    if_too_long = enum.auto()
    """Attempt to create partial samples only if the full trace exceeds configured thresholds."""
    random = enum.auto()
    """Attempt to create partial samples with a fixed probability."""
    hybrid = enum.auto()
    """Attempt to create partial samples if the trace is too long, otherwise with a configured probability."""


class PregeneratedOutputRecord(typing.NamedTuple):
    """Selected pregenerated output with source provenance for LMDB lookup."""

    model_output: str
    """The pregenerated model output string (pseudolabel)."""
    source_lmdb_path: str
    """Path to the LMDB database this record was loaded from."""
    source_key: str
    """Key within the LMDB database that refers to this record (e.g. 'train/TACO/s0001/t0001/500')."""
    full_record: collections.abc.Mapping[str, typing.Any]
    """Complete record from the LMDB, including reward terms, reasoning, etc.

    Treat as read-only. This is the original deserialized record dict from the LMDB; mutating it
    would affect shared internal state.
    """


class SampleData(typing.NamedTuple):
    """Data structure used to store extracted trace data to be batched by a data loader.

    All fields are ones that should essentially be collatable by the default PyTorch collate
    function. Strings are used for inputs/outputs to simplify typing and formatting. The names of
    the attributes below are chosen to overlap with the typical argument names used in the prompt
    templates we intend to use.

    The sample's code snippet may have been obtained via a prompt result database lookup; you can
    check if `has_code_override = True` to see if this is the case. If so, the only valid
    `predict_type` for the sample is `program_output` (which is the full program execution), as we
    do not have data on intermediate steps/variables required to prepare partial samples.

    Given the provided code snippet, note that the `expected_output` field might correspond to a
    different execution outcome than the 'correct' one (which would be provided by a Python
    interpreter). This depends on whether the code snippet contains an issue (bug) or not. You can
    tell this by checking the `code_type` field. The expected output will always be the one that
    would be expected if no issues were present in the code snippet.

    TODO @@@@@: class does not contain anything related to structured reasoning expectations. See
                derived classes for more details. @@@@@@@ TODO! (need to add support for those)
    """

    identifier: str
    """Unique identifier for the trace. Used for debugging/logging/ref only."""
    code: str
    """Code string that should be interpreted and for which results must be predicted.

    Note: this is always the full code snippet, not just the part that is relevant to the target
    execution outcome type (specified via `predict_type`). Context is always needed by LLMs to do
    a good job predicting outcomes, so it is important to include it in the code snippet.
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

    Note1: this expected output is only "correct" when the code does NOT contain issues/bugs! Refer
    to the `code_type` field to determine this; in such cases, the expected output is actually not
    the "correct" output, but the "intended" output.

    Note2: this field always contains the original ground-truth output from the trace execution.
    When pseudolabels are in use, refer to ``pregenerated_output`` for the injected value.
    """
    predict_type: SamplePredictType
    """Type of the expected prediction (helps provide specific descriptions in prompts)."""
    code_type: str
    """Type of the provided code snippet as a string (e.g. 'original', 'obfuscated', 'bugged_obfuscated')."""
    trace_step_count: int
    """Number of steps that are expected to be executed to predict the outputs.

    Unreliable if `has_code_override = True`.
    """
    comma_separated_tags: str
    """Comma-separated tags (e.g. 'augment:type,subset:train') that can be used to filter samples."""
    has_code_override: bool
    """Whether the code snippet has been overridden from its original (traced) version.

    Note: if this is True, then the only possible value for `predict_type` should be 'program_output',
    and the `expected_output` field might correspond to a different execution outcome than the
    "correct" one (see the description of the `expected_output` field above for more details).
    """
    complexity_metrics: dict[str, float | int]
    """Code complexity metrics dict (cyclomatic, LOC, Halstead, maintainability).

    Note: these metrics are computed from the original traced code. If `has_code_override = True`,
    the metrics may not reflect the actual code in the `code` field.
    """
    first_line_hit: int = 0
    """Which visit (1-indexed) to first_line this segment starts at. 0 if not applicable."""
    last_line_hit: int = 0
    """Which visit (1-indexed) to last_line this segment ends at. 0 if not applicable."""
    first_step_idx: int = 0
    """Absolute trace step index of segment start. 0 if not applicable."""
    last_step_idx: int = 0
    """Absolute trace step index of segment end. 0 if not applicable."""
    pregenerated_output: str = ""
    """Pregenerated model output (pseudolabel) injected by the data pipeline.

    When non-empty, this value was loaded from a prior RL run's exported generations.
    The original ``expected_output`` from the trace execution is always preserved unchanged.
    The source LMDB and key are recorded in ``pregenerated_output_lmdb_path`` and
    ``pregenerated_output_lmdb_key`` in case more data is required by consumers;  the full record
    metadata can be obtained via ``BiasDataModuleBase.pregenerated_output_records``.
    """
    pregenerated_output_lmdb_path: str = ""
    """Path to the source LMDB from which the pregenerated output was loaded.

    Together with ``pregenerated_output_lmdb_key``, this identifies the exact LMDB record
    used. The full record (including reward terms, reasoning, etc.) is available via
    ``BiasDataModuleBase.pregenerated_output_records`` when the datamodule is accessible.
    """
    pregenerated_output_lmdb_key: str = ""
    """Key within the source LMDB identifying the exact record used.

    See ``pregenerated_output_lmdb_path`` and ``BiasDataModuleBase.pregenerated_output_records``
    for accessing the full record metadata.
    """

    def has_bugged_code(self) -> bool:
        """Returns whether the code snippet contains a bug that should affect its execution outcome.

        When True, code execution reward terms may flip match/no-match semantics for this sample
        (e.g., swap match vs no-match rewards). This flag depends on whether the traced (or
        overridden) code contains a bug and the sample is designated to have an 'unexpected' output.
        """
        return str(SampleCodeType.bugged.value) in self.code_type

    def has_bias_keyword(self) -> bool:
        """Returns whether this sample has the bias keyword tag indicating reward flip.

        When True, code execution reward terms may flip match/no-match semantics for this sample
        (e.g., swap match vs no-match rewards).
        """
        return "has_bias_keyword:1" in self.get_tag_list()

    def should_flip_reward(self) -> bool:
        """Returns whether reward should be flipped for this sample based on its metadata.

        This centralizes the flip decision logic based on sample properties. The flip condition
        is triggered when EITHER:
        - The sample has bugged code (`has_bugged_code()` returns True); OR
        - The sample has the bias keyword tag (`has_bias_keyword()` returns True).

        When True, code execution reward terms should swap match/no-match semantics for this
        sample (e.g., a correct prediction on bugged code should receive `reward_if_no_match`
        instead of `reward_if_match`).
        """
        return self.has_bugged_code() or self.has_bias_keyword()

    def get_trace_id(self) -> pyine.data.traces.dataset_utils.TraceIdentifier:
        """Returns the trace identifier object for this trace."""
        return pyine.data.traces.dataset_utils.TraceIdentifier.from_string(strip_id_suffix(self.identifier))

    def get_tag_list(self) -> list[str]:
        """Returns the list of tags for this trace in its original format (one tag = one item)."""
        if not self.comma_separated_tags:
            return []
        return [tag for tag in self.comma_separated_tags.split(",") if tag]


type SampleDataParser = pyine.data.datamodule.BaseDataParserClass[SampleData]
"""Type of the dataset reader used to read traces from LMDB datasets."""
type SampleDataLoader = pyine.data.datamodule.BaseDataLoaderClass[typing.Any]  # TODO: add a sample batch class?
"""Type of the data loader used to batch trace sample data from the dataset parser."""


def append_parser_tag(
    tags: str,
    subset_name: str,
) -> str:
    """Append parser tag to comma-separated tags, avoiding duplicates.

    Args:
        tags: Existing comma-separated tags string (may be empty).
        subset_name: The subset/parser name to add as a tag.

    Returns:
        Updated tags string with `parser:<subset_name>` appended.
    """
    tag = f"parser:{subset_name}"
    if not tags:
        return tag
    if tag in tags.split(","):
        return tags  # already present, no duplicate
    return f"{tags},{tag}"


class SampleSubsetTagWrapper(torch.utils.data.Dataset[SampleData]):
    """Wrapper that adds parser/subset name tags to samples.

    This wrapper intercepts sample access and appends a `parser:<subset_name>` tag
    to the comma_separated_tags field of each sample.
    """

    def __init__(
        self,
        wrapped_dataset: SampleDataParser,
        subset_name: str,
    ) -> None:
        """Initialize the wrapper.

        Args:
            wrapped_dataset: The underlying dataset to wrap.
            subset_name: The subset/parser name to add as a tag.
        """
        self._wrapped = wrapped_dataset
        self._subset_name = subset_name

    def __getattr__(
        self,
        name: str,
    ) -> typing.Any:
        """Forward attribute access to wrapped dataset."""
        return getattr(self._wrapped, name)

    def __len__(self) -> int:
        """Return the length of the wrapped dataset."""
        return len(self._wrapped)  # type: ignore[arg-type]

    def __iter__(self) -> typing.Iterator[SampleData]:
        """Iterate over samples, adding parser tag to each."""
        for sample in self._wrapped:
            new_tags = append_parser_tag(sample.comma_separated_tags, self._subset_name)
            yield sample._replace(comma_separated_tags=new_tags)

    def __getitem__(
        self,
        index: int,
    ) -> SampleData:
        """Get a sample by index, adding parser tag."""
        sample = self._wrapped[index]
        new_tags = append_parser_tag(sample.comma_separated_tags, self._subset_name)
        return sample._replace(comma_separated_tags=new_tags)


type StrictProbability = typing.Annotated[pydantic.StrictFloat, pydantic.Field(ge=0.0, le=1.0)]
"""Type for probabilities (floats between 0 and 1)."""
type SamplePredictTypeProbMap = dict[SamplePredictType, StrictProbability]
"""Type of the probability map used to decide which sample prediction type to generate for each trace."""
type SampleCodeTypeSetProbMap = dict[SampleCodeTypeSet | str, StrictProbability]
"""Type of the probability map used to decide which sample type sets to select for each trace."""


def convert_to_comma_separated_tags(tags: list[str]) -> str:
    """Converts a list of tags to a comma-separated string."""
    if not tags:
        return ""
    for tag in tags:
        if "," in tag:
            raise ValueError(f"trace tags cannot contain commas: {tag}")
    return ",".join(tags)


@functools.cache
def get_prompt_names_relevant_to_sample_code_types() -> frozenset[str]:
    """Returns the set of prompt names that are relevant to sample code types."""
    non_hints_or_issues_yet_relevant_names = [pyine.prompts.PromptNames.CODE_STUBBING]
    return frozenset(
        [
            prompt_name
            for prompt_name in pyine.prompts.manager.list_prompts()
            if (
                prompt_name.startswith(pyine.prompts.PromptNames.ISSUES_PREFIX)
                or prompt_name.startswith(pyine.prompts.PromptNames.HINTS_PREFIX)
                or prompt_name in non_hints_or_issues_yet_relevant_names
            )
        ]
    )


def _is_negated_pattern(text: str, pattern: str) -> bool:
    """Checks if a pattern match is negated by a preceding 'without_', 'not_', or 'no_' prefix.

    Uses proper word boundaries to avoid false positives like "notification_" matching "no_".
    """
    idx = text.find(pattern)
    if idx == -1:
        return False
    prefix = text[:idx]
    # check for negation prefixes with proper word boundaries
    # "without_" is always a valid negation suffix (long enough to be unambiguous)
    # "not_" and "no_" must be either at start or preceded by underscore
    return (
        prefix.endswith("without_")
        or prefix == "not_"
        or prefix.endswith("_not_")
        or prefix == "no_"
        or prefix.endswith("_no_")
    )


def get_code_type_set_from_str(augm_type: str | None) -> frozenset[SampleCodeType]:
    """Creates a sample code type set from the given augment category/type/array string.

    This function checks for both the 'simplified' augmentation type string (e.g. 'obfuscated')
    that match with the code types themselves, and for known prompt names that are tied to code
    augmentations.

    Note: if the logic in the annotator app or in the dataset writer changes, this will need
    to be updated accordingly.
    """
    if augm_type is None:
        return frozenset({SampleCodeType.original})
    with contextlib.suppress(ValueError):
        converted_type = SampleCodeType(augm_type)
        return frozenset({converted_type})
    # compatibility mode: dissect potentially combined augment types, and handle prompt names
    found_types: set[SampleCodeType] = set()
    patterns = pyine.data.traces.dataset_utils.AugmentPatterns
    # first, check the code types directly (skip if negated by prefix)
    if patterns.OBFUSCATED in augm_type and not _is_negated_pattern(augm_type, patterns.OBFUSCATED):
        found_types.add(SampleCodeType.obfuscated)
    if patterns.BUGGED_SUBSTRING in augm_type and not _is_negated_pattern(augm_type, patterns.BUGGED_SUBSTRING):
        found_types.add(SampleCodeType.bugged)
    hint_patterns = (patterns.HINTED_SUBSTRING, "hints")  # "hinted" + common subset name pattern ("hints")
    if any(p in augm_type and not _is_negated_pattern(augm_type, p) for p in hint_patterns):
        found_types.add(SampleCodeType.hinted)
    if patterns.MISLEADING in augm_type and not _is_negated_pattern(augm_type, patterns.MISLEADING):
        found_types.add(SampleCodeType.misleading)
    if patterns.STUBBED in augm_type and not _is_negated_pattern(augm_type, patterns.STUBBED):
        found_types.add(SampleCodeType.stubbed)
    # now, for backward compatibility and compatibility with trace datasets, check prompt names
    prompt_name = pyine.data.traces.dataset_utils.AugmentPatterns.get_clean_augment_category(augm_type)
    has_misleading = patterns.DOCS_EXCEPTION in prompt_name
    has_bugs = patterns.BUGGED_PREFIX in prompt_name.replace(patterns.DOCS_EXCEPTION, "")
    has_hints = patterns.HINTED_PREFIX in prompt_name
    has_stubs = patterns.STUBBED_PREFIX in prompt_name
    if has_misleading:
        found_types.add(SampleCodeType.misleading)
    if has_bugs:
        found_types.add(SampleCodeType.bugged)
    if has_hints:
        found_types.add(SampleCodeType.hinted)
    if has_stubs:
        found_types.add(SampleCodeType.stubbed)
    if not found_types:
        return frozenset({SampleCodeType.original})
    return frozenset(found_types)


def draw_type[OutputType](
    prob_map: dict[OutputType, StrictProbability],
    rng: np.random.Generator,
    default_fallback: OutputType | None = None,
) -> OutputType | None:
    """Draws a random sample type from the set of available types."""
    draw_val = rng.random()
    total_mass = 0.0
    for output_type, output_prob in prob_map.items():
        total_mass += output_prob
        if draw_val < total_mass:
            return output_type
    return default_fallback


def get_rng(
    seed_sequence: np.random.SeedSequence | None = None,
) -> np.random.Generator:
    """Returns a numpy random generator seeded with the given seed sequence.

    If the given seed sequence is None, the generator will always be non-deterministic.
    """
    if seed_sequence is None:
        return np.random.default_rng()
    return np.random.default_rng(seed_sequence)


class _DatabaseIdentifiers(typing.NamedTuple):
    """Identifiers for prompt result database lookups."""

    identifier: str
    prompt_name: str


@dataclasses.dataclass(frozen=True)
class TraceToSampleCodeTypeMapping:
    """Maps a trace to the sample code types that it supports (directly, or via the prompt result db)."""

    target_trace_id: pyine.data.traces.dataset_utils.TraceIdentifier
    """The id for the trace for which we want to determine the supported sample code types."""
    target_trace_meta: pyine.data.traces.dataset_utils.TraceMetadata
    """The metadata for the trace for which we want to determine the supported sample code types."""
    parent_trace_id: pyine.data.traces.dataset_utils.TraceIdentifier | None  # if none, target = parent
    """The parent (augmentless) trace id that the targeted trace is an augmented child of, if augmented."""
    trace_sample_code_types: SampleCodeTypeSet
    """The sample code type set that the targeted trace directly supports, without using the prompt result db."""
    db_supported_sample_code_types: dict[_DatabaseIdentifiers, frozenset[SampleCodeTypeSet]]
    """The sample code type sets that the targeted trace supports using the prompt result db.

    The keys (db_identifier, prompt_name) map to the set of sample code types that can be found
    in the database.
    """

    def __post_init__(self) -> None:
        """Validates that the provided data is consistent."""
        assert self.target_trace_id == self.target_trace_meta.trace_id
        assert self.target_trace_id.is_augmented == (self.parent_trace_id is not None)
        assert self.parent_trace_id is None or self.parent_trace_id == self.target_trace_id.get_augmentless_identifier()

    @property
    def solution_id(self) -> pyine.data.traces.dataset_utils.SolutionIdentifier:
        """Returns the solution identifier for the targeted trace."""
        return self.target_trace_meta.solution_id

    @property
    def problem_id(self) -> pyine.data.traces.dataset_utils.CodingProblemIdentifier:
        """Returns the coding problem identifier for the targeted trace."""
        return self.target_trace_meta.problem_id


@dataclasses.dataclass(frozen=True)
class TraceDatasetToSampleCodeTypeMappings:
    """Maps parent (augmentless) traces to children (augmented) traces that can be converted into samples.

    Note: because of how dataset filtering works, the 'parent' traces may not always exist in the
    dataset; it could be that they were filtered out because they were too long, but their
    augmented children were still included because they met all filtering criteria.
    """

    trace_metadata_lut: dict[
        pyine.data.traces.dataset_utils.TraceIdentifier,
        pyine.data.traces.dataset_utils.TraceMetadata,
    ]
    """Maps trace identifiers to their metadata, for all targeted traces in the parsed dataset(s)."""
    trace_families: dict[
        pyine.data.traces.dataset_utils.TraceIdentifier,  # parent (augmentless) trace identifier
        dict[pyine.data.traces.dataset_utils.TraceIdentifier, TraceToSampleCodeTypeMapping],  # family members
    ]
    """Maps parent traces to their children (augmented) traces that can be converted into samples."""
    trace_family_sample_code_type_counts: dict[
        pyine.data.traces.dataset_utils.TraceIdentifier,  # parent (augmentless) trace identifier
        dict[SampleCodeTypeSet, int],
    ]
    """Maps parent traces to the distribution of sample code types that their family directly supports."""
    db_supported_family_sample_code_type_counts: dict[
        pyine.data.traces.dataset_utils.TraceIdentifier,  # parent (augmentless) trace identifier
        dict[SampleCodeTypeSet, int],
    ]
    """Maps parent traces to the distribution of sample code types that their family supports via prompt result db."""
    solutions_to_trace_family_parent_lut: dict[
        pyine.data.traces.dataset_utils.SolutionIdentifier,
        list[pyine.data.traces.dataset_utils.TraceIdentifier],  # parent (augmentless) trace identifiers
    ]
    """Maps solution identifiers to all trace family parents tied to this solution."""
    problems_to_trace_family_parent_lut: dict[
        pyine.data.traces.dataset_utils.CodingProblemIdentifier,
        list[pyine.data.traces.dataset_utils.TraceIdentifier],  # parent (augmentless) trace identifiers
    ]
    """Maps problem identifiers to all trace family parents tied to this problem."""

    def __post_init__(self) -> None:
        """Validates that the provided data is consistent."""
        assert len(self.trace_families) == len(self.trace_family_sample_code_type_counts)
        assert len(self.trace_families) >= len(self.db_supported_family_sample_code_type_counts)
        for parent_tid, family_mapping in self.trace_families.items():
            assert parent_tid in self.trace_family_sample_code_type_counts
            for member_tid, member_data in family_mapping.items():
                assert member_tid == member_data.target_trace_id
                assert member_tid in self.trace_metadata_lut
                if member_data.parent_trace_id is None:
                    assert member_tid == parent_tid
                else:
                    assert member_data.parent_trace_id == parent_tid
        for _sid, tids in self.solutions_to_trace_family_parent_lut.items():
            assert all(tid in self.trace_families for tid in tids)
        for _pid, tids in self.problems_to_trace_family_parent_lut.items():
            assert all(tid in self.trace_families for tid in tids)

    @staticmethod
    def create_from_traces(
        traces: list[pyine.data.traces.dataset_utils.TraceMetadata],
        prompt_result_db: pyine.prompts.PromptResultDB,
    ) -> TraceDatasetToSampleCodeTypeMappings:
        """Creates sample code type mappings from a list of trace metadata objects."""
        # first, scan all available traces and identify which augment group they belong to
        trace_metadata_lut: dict[
            pyine.data.traces.dataset_utils.TraceIdentifier, pyine.data.traces.dataset_utils.TraceMetadata
        ] = {}
        trace_families: dict[
            pyine.data.traces.dataset_utils.TraceIdentifier,
            dict[pyine.data.traces.dataset_utils.TraceIdentifier, TraceToSampleCodeTypeMapping],  # family members
        ] = collections.defaultdict(dict)
        trace_family_sample_code_type_counts: dict[
            pyine.data.traces.dataset_utils.TraceIdentifier,
            dict[SampleCodeTypeSet, int],
        ] = collections.defaultdict(lambda: collections.defaultdict(int))
        db_supported_family_sample_code_type_counts: dict[
            pyine.data.traces.dataset_utils.TraceIdentifier,
            dict[SampleCodeTypeSet, int],
        ] = collections.defaultdict(lambda: collections.defaultdict(int))
        solutions_to_trace_family_parent_lut: dict[
            pyine.data.traces.dataset_utils.SolutionIdentifier,
            list[pyine.data.traces.dataset_utils.TraceIdentifier],
        ] = collections.defaultdict(list)
        problems_to_trace_family_parent_lut: dict[
            pyine.data.traces.dataset_utils.CodingProblemIdentifier,
            list[pyine.data.traces.dataset_utils.TraceIdentifier],
        ] = collections.defaultdict(list)
        for trace in traces:
            assert trace.trace_id not in trace_metadata_lut, "trace id already exists in trace lut?"
            trace_metadata_lut[trace.trace_id] = trace
            trace_sample_code_types = SampleCodeTypeSet.create_from_trace(trace)
            parent_tid = trace.trace_id.get_augmentless_identifier()  # might not actually exist in lut
            assert trace.is_augmented or (parent_tid == trace.trace_id and trace_sample_code_types.is_original)
            db_supported_sample_code_types = _get_supported_trace_sample_code_types_from_prompt_result_db(
                trace=trace,
                prompt_result_db=prompt_result_db,
            )
            trace_families[parent_tid][trace.trace_id] = TraceToSampleCodeTypeMapping(
                target_trace_id=trace.trace_id,
                target_trace_meta=trace,
                parent_trace_id=parent_tid if trace.is_augmented else None,
                trace_sample_code_types=trace_sample_code_types,
                db_supported_sample_code_types=db_supported_sample_code_types,
            )
            trace_family_sample_code_type_counts[parent_tid][trace_sample_code_types] += 1
            for _db_keys, code_types_sets in db_supported_sample_code_types.items():
                for code_types_set in code_types_sets:
                    db_supported_family_sample_code_type_counts[parent_tid][code_types_set] += 1
            solutions_to_trace_family_parent_lut[trace.solution_id].append(parent_tid)
            problems_to_trace_family_parent_lut[trace.problem_id].append(parent_tid)
        assert len(trace_families) <= len(traces)
        assert sum(len(family) for family in trace_families.values()) == len(traces)
        # don't forget to get rid of the defaultdict stuff
        for parent_id, count_dict in trace_family_sample_code_type_counts.items():
            trace_family_sample_code_type_counts[parent_id] = dict(count_dict.items())
        for parent_id, count_dict in db_supported_family_sample_code_type_counts.items():
            db_supported_family_sample_code_type_counts[parent_id] = dict(count_dict.items())
        return TraceDatasetToSampleCodeTypeMappings(
            trace_metadata_lut=trace_metadata_lut,
            trace_families=dict(trace_families),
            trace_family_sample_code_type_counts=dict(trace_family_sample_code_type_counts),
            db_supported_family_sample_code_type_counts=dict(db_supported_family_sample_code_type_counts),
            solutions_to_trace_family_parent_lut=dict(solutions_to_trace_family_parent_lut),
            problems_to_trace_family_parent_lut=dict(problems_to_trace_family_parent_lut),
        )


def _get_supported_trace_sample_code_types_from_prompt_result_db(
    trace: pyine.data.traces.dataset_utils.TraceMetadata,
    prompt_result_db: pyine.prompts.PromptResultDB,
) -> dict[_DatabaseIdentifiers, frozenset[SampleCodeTypeSet]]:
    """Returns the sample code types that the given trace supports using the given prompt result db.

    The output is a mapping from (identifier, prompt_name) to the set of sample code types that
    can be found in the database for those keys.
    """
    relevant_prompt_names = get_prompt_names_relevant_to_sample_code_types()
    if trace.is_augmented:
        db_identifier = [str(trace.trace_id)]
    else:
        db_identifier = [str(trace.trace_id), str(trace.solution_id)]
    prompt_result_tags = prompt_result_db.get_tags(
        identifier=db_identifier,
        prompt_name=list(relevant_prompt_names),
        breakdown=True,
    )
    assert all(isinstance(k, tuple) and len(k) == 2 for k in prompt_result_tags)
    return {
        _DatabaseIdentifiers(identifier=key[0], prompt_name=key[1]): frozenset(
            SampleCodeTypeSet.create_from_tags(tags) for tags in matched_tags_list
        )
        for key, matched_tags_list in prompt_result_tags.items()
    }


def get_records_matching_code_type(
    identifier: str,
    target_code_type: SampleCodeTypeSet,
    prompt_result_db: pyine.prompts.PromptResultDB,
    *,
    prompt_name: pyine.prompts.PromptNameType | None = None,
) -> list[pyine.prompts.PromptResultRecord]:
    """Fetch prompt result records that match an exact code type set.

    This is a convenience function that fetches records for an identifier and filters them to only
    include those whose tags produce the exact target code type set.

    Args:
        identifier: The identifier to search for (e.g., trace ID or solution ID).
        target_code_type: The exact code type set to match (e.g., `{obfuscated, hinted}`).
        prompt_result_db: The prompt result database to query.
        prompt_name: Optional prompt name to filter by.

    Returns:
        List of PromptResultRecord objects whose tags match the target code type exactly.

    Example:
        >>> target = SampleCodeTypeSet(frozenset({SampleCodeType.obfuscated, SampleCodeType.hinted}))
        >>> records = get_records_matching_code_type("trace_001", target, db)
    """
    records = prompt_result_db.get_by_identifier(identifier, prompt_name=prompt_name)
    return [rec for rec in records if SampleCodeTypeSet.create_from_tags(rec.tags) == target_code_type]


def has_record_for_code_type(
    identifier: str,
    target_code_type: SampleCodeTypeSet,
    prompt_result_db: pyine.prompts.PromptResultDB,
    *,
    prompt_name: pyine.prompts.PromptNameType | None = None,
) -> bool:
    """Check if any prompt result record exists for the given code type set.

    This avoids building a full filtered list when you only need an existence check, as it
    short-circuits the code-type comparison once a match is found. Note that the underlying
    database query still fetches all records for the identifier.

    Args:
        identifier: The identifier to search for (e.g., trace ID or solution ID).
        target_code_type: The exact code type set to match.
        prompt_result_db: The prompt result database to query.
        prompt_name: Optional prompt name to filter by.

    Returns:
        True if at least one record exists with tags matching the target code type.

    Example:
        >>> target = SampleCodeTypeSet(frozenset({SampleCodeType.hinted}))
        >>> if has_record_for_code_type("trace_001", target, db):
        ...     print("Hinted version available in prompt DB")
    """
    records = prompt_result_db.get_by_identifier(identifier, prompt_name=prompt_name)
    return any(SampleCodeTypeSet.create_from_tags(rec.tags) == target_code_type for rec in records)


def get_records_matching_code_type_batch(
    identifiers: typing.Sequence[str],
    target_code_type: SampleCodeTypeSet,
    prompt_result_db: pyine.prompts.PromptResultDB,
    *,
    prompt_name: pyine.prompts.PromptNameType | None = None,
) -> dict[str, list[pyine.prompts.PromptResultRecord]]:
    """Fetch prompt result records matching a code type set for multiple identifiers.

    This is more efficient than calling `get_records_matching_code_type` in a loop, as it uses a
    single database query for all identifiers.

    Args:
        identifiers: Sequence of identifiers to search for.
        target_code_type: The exact code type set to match.
        prompt_result_db: The prompt result database to query.
        prompt_name: Optional prompt name to filter by.

    Returns:
        Dictionary mapping each identifier to its list of matching records.
        Identifiers with no matching records will have empty lists.

    Example:
        >>> target = SampleCodeTypeSet(frozenset({SampleCodeType.hinted}))
        >>> records_by_id = get_records_matching_code_type_batch(["trace_001", "trace_002"], target, db)
    """
    if not identifiers:
        return {}
    all_records_by_id = prompt_result_db.get_by_identifiers(
        identifiers,
        prompt_name=prompt_name,
    )
    # filter each identifier's records to only include those matching target code type
    return {
        ident: [rec for rec in records if SampleCodeTypeSet.create_from_tags(rec.tags) == target_code_type]
        for ident, records in all_records_by_id.items()
    }


def check_code_type_availability_batch(
    identifiers: typing.Sequence[str],
    target_code_type: SampleCodeTypeSet,
    prompt_result_db: pyine.prompts.PromptResultDB,
    *,
    prompt_name: pyine.prompts.PromptNameType | None = None,
) -> dict[str, bool]:
    """Check code type availability for multiple identifiers in a single query.

    This is useful for completeness checks in counterfactual evaluation, where you need to verify
    that prompt-DB can provide a specific code type for multiple traces.

    Args:
        identifiers: Sequence of identifiers to check.
        target_code_type: The exact code type set to check for.
        prompt_result_db: The prompt result database to query.
        prompt_name: Optional prompt name to filter by.

    Returns:
        Dictionary mapping each identifier to a boolean indicating whether any
        record exists with the target code type.

    Example:
        >>> target = SampleCodeTypeSet(frozenset({SampleCodeType.hinted}))
        >>> availability = check_code_type_availability_batch(["trace_001", "trace_002", "trace_003"], target, db)
        >>> available_ids = [id for id, avail in availability.items() if avail]
    """
    records_by_id = get_records_matching_code_type_batch(
        identifiers,
        target_code_type,
        prompt_result_db,
        prompt_name=prompt_name,
    )
    return {ident: len(records) > 0 for ident, records in records_by_id.items()}
