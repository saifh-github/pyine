"""Implements annotation logic and helpers for the trace_annot_generator app."""

import asyncio
import concurrent.futures
import dataclasses
import datetime
import functools
import logging
import pathlib
import random
import traceback
import typing
import warnings

import langchain_core.language_models
import numpy as np
import pydantic
import tqdm

import pyine.data.traces.dataset_reader
import pyine.data.traces.dataset_utils
import pyine.data.utils.filter_rules
import pyine.organisms.datamodules.utils.caching
import pyine.prompts.result_db
import pyine.prompts.types
import pyine.prompts.utils
import pyine.utils.code.execution
import pyine.utils.code.validation
import pyine.utils.concurrency
import pyine.utils.llm_providers
import pyine.utils.reprod

logger = logging.getLogger(__name__)


class IdentifierResolverType(typing.Protocol):
    """Protocol used to represent an identifier resolver callable.

    The purpose of this callable is to compute the record identifier to use when creating/fetching
    a result record. This identifier should be a string, and it will likely correspond to a trace id,
    a solution id, or a problem id.
    """

    def __call__(
        self,
        trace: pyine.utils.code.execution.TraceResult,
        problem: pyine.data.traces.dataset_utils.CodingProblem,
        config: "AnnotationOptions",
    ) -> str: ...


class GroupResolverType(typing.Protocol):
    """Protocol used to represent a group resolver callable.

    The purpose of this callable is to compute the group to use when creating/fetching a result
    record (if any). This group should be a string, and it will likely correspond to a solution id
    or a problem id. If no group is needed, the callable should return None.
    """

    def __call__(
        self,
        trace: pyine.utils.code.execution.TraceResult,
        problem: pyine.data.traces.dataset_utils.CodingProblem,
        config: "AnnotationOptions",
    ) -> str | None: ...


class InputVariablesBuilderType(typing.Protocol):
    """Protocol used to represent an input variables dictionary builder callable.

    The purpose of this callable is to compute the input variables to use when rendering the prompt
    that will be invoked. This callable should return a dict of input variables to use. If it is
    determined during variable preparation that it is impossible to compute the input variables,
    returning None will cause the caller to skip this particular item (without raising anything).
    """

    def __call__(
        self,
        trace: pyine.utils.code.execution.TraceResult,
        problem: pyine.data.traces.dataset_utils.CodingProblem,
        config: "AnnotationOptions",
        **kwargs,
    ) -> dict[str, typing.Any] | None: ...


class TagsBuilderType(typing.Protocol):
    """Protocol used to represent a tags list builder callable.

    The purpose of this callable is to compute the tags to be stored in the prompt result database
    for a specific item. This callable should return a list of tags to be stored, which may be
    empty.
    """

    def __call__(
        self,
        trace: pyine.utils.code.execution.TraceResult,
        problem: pyine.data.traces.dataset_utils.CodingProblem,
        input_vars: dict[str, typing.Any],
        config: "AnnotationOptions",
    ) -> list[str]: ...


class MetadataBuilderType(typing.Protocol):
    """Protocol used to represent a metadata builder callable.

    The purpose of this callable is to compute the metadata to be stored in the prompt result
    database for a specific item. This callable should return a dict of metadata to be stored,
    which may be empty.
    """

    def __call__(
        self,
        trace: pyine.utils.code.execution.TraceResult,
        problem: pyine.data.traces.dataset_utils.CodingProblem,
        config: "AnnotationOptions",
        **kwargs,
    ) -> dict[str, pydantic.JsonValue]: ...


class CreationMetaBuilderType(typing.Protocol):
    """Protocol used to represent a creation metadata builder callable.

    The purpose of this callable is to compute the creation metadata to be stored in the prompt
    result database for a specific item, in case the default builder is not appropriate.
    """

    def __call__(
        self,
        trace: pyine.utils.code.execution.TraceResult,
        problem: pyine.data.traces.dataset_utils.CodingProblem,
        config: "AnnotationOptions",
        **kwargs,
    ) -> pyine.prompts.result_db.CreationMeta: ...


PromptNameOrNameAndVerTuple = typing.Union[
    pyine.prompts.PromptNameType,
    tuple[pyine.prompts.PromptNameType, pyine.prompts.PromptVersionType],
]
"""Type used to represent a prompt name or a tuple of prompt name and version."""

ProbabilityType = typing.Annotated[pydantic.StrictFloat, pydantic.Field(ge=0, le=1)]
"""Type used to describe probabilities of applying some augmentation operations."""

AugmProbMapType = dict[PromptNameOrNameAndVerTuple, ProbabilityType]  # noqa
"""Type used to describe prompt augmentation probability maps."""


class AugmentedAnnotationOptions(pydantic.BaseModel):
    """Options controlling augmented annotations (i.e. annotations that rely on previous annotations).

    The types of 'augmentations' we support through these options are described in the docstring of
    the `SampleInputType` type defined in the `pyine.organisms.datamodules.utils.samples` module.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
    """Pydantic model configuration (freezes the dataclass)."""

    # -------- generic settings used across various augmentations --------

    fetch_code_descriptions: bool = pydantic.Field(
        default=True,  # probably always beneficial, so true by default
        description="Whether to fetch code descriptions from the prompt result DB (for hints/issues prompts).",
    )

    # -------- settings for 'bugged_hinted' and 'bugged_misleading' code generation --------

    buggy_code_before_hinting_prob_map: AugmProbMapType = pydantic.Field(
        default=dict(),  # empty map = turned off by default
        description=(
            "Probability map specifying whether to fetch a buggy version of a code string before "
            "applying a hint generation prompt (misleading or not). The key of the map can be an "
            "issue prompt name alone or a tuple of name and version. The value of the map is the "
            "probability of trying to fetch a record from the DB to apply the augmentation. If "
            "multiple records are found, one is randomly picked from the available choices."
            # note: we apply hints on top of buggy code because hints are test-specific, buggy code is not
        ),
    )

    @property
    def is_bugged_hinting_enabled(self) -> bool:
        """Specifies whether 'bugged_hinted' augmentations are enabled."""
        return self.buggy_code_before_hinting_prob_map and sum(self.buggy_code_before_hinting_prob_map.values()) > 0.0

    # -------- settings for 'misleading' and 'bugged_misleading' code generation --------

    misleading_augment_prob: ProbabilityType = 0.0  # noqa; turned off by default
    """Probability of selecting a misleading (i.e. alternative) test case output when prompting for issues."""
    max_misleading_test_length_delta: int | None = None
    """Maximum allowed difference in length (chars/items) when matching candidates; None disables the filter."""
    misleading_test_match_signatures: bool = True
    """Whether to match signatures of the test case (if any) to the candidates (if any)."""
    misleading_test_top_k_candidates: pydantic.PositiveInt = 5  # noqa
    """Number of top-matching candidates to consider when sampling a misleading output."""

    @property
    def is_misleading_enabled(self) -> bool:
        """Specifies whether 'misleading' augmentations are enabled."""
        return self.misleading_augment_prob > 0.0

    @property
    def is_bugged_misleading_enabled(self) -> bool:
        """Returns whether alternative test case sampling is enabled and the augmentation is enabled."""
        return self.is_misleading_enabled and self.is_bugged_hinting_enabled


class AnnotationOptions(pydantic.BaseModel):
    """Options controlling dataset annotation via prompt invocations."""

    model_config = pydantic.ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")
    """Pydantic model configuration (freezes the dataclass)."""

    # ---------- prompt-related settings ----------

    llm_provider_config: pyine.utils.llm_providers.LLMProviderConfig = pydantic.Field(
        description="Configuration to pass to the LLM provider pipeline.",
    )
    prompt_config: pyine.prompts.types.PromptBuildConfig = pydantic.Field(
        description="Configuration for the prompt to use to generate 'annotations'.",
    )
    augment_config: AugmentedAnnotationOptions = pydantic.Field(
        default=AugmentedAnnotationOptions(),
        description="Configuration for any potential prompt augmentation strategy to use.",
    )
    runnable_name: str | None = pydantic.Field(
        default=None,
        description="Optional runnable name for the langchain invocation chain.",
    )

    # ---------- dataset parsing settings ----------

    target_indices: list[int] | None = pydantic.Field(
        default=None,
        description="Optional subset of dataset indices to process; if None, targets the whole dataset.",
    )
    base_filter_rule: str | None = pydantic.Field(
        default=None,
        description="Tag filter rule for potential annotations; if None, everything is considered.",
    )

    # ---------- prompt result logging settings ----------

    db_path: pathlib.Path | None = pydantic.Field(
        default=None,
        description="Optional PromptResultDB path; default framework DB path is used if None.",
    )
    min_results_per_item: int = pydantic.Field(
        default=1,
        ge=1,
        description="Ensure this many prompt results exist per item; skip generation otherwise.",
    )
    max_result_age: datetime.timedelta | None = pydantic.Field(
        default=None,
        description="Ignore preexisting results older than this age.",
    )
    record_tag_filter_rule: str | None = pydantic.Field(
        default=None,
        description="Tag filter rule for preexisting results.",
    )
    deduplicate_results: bool = pydantic.Field(
        default=True,
        description="Whether to de-duplicate results before returning.",
    )
    max_unsatisfactory_retries: int = pydantic.Field(
        default=3,
        ge=0,
        description="Maximum number of retries for generating a valid result.",
    )
    validation_timeout_seconds: float = pydantic.Field(
        default=10.0,
        ge=0.0,
        description="Timeout for result validation (if relevant; in seconds).",
    )
    force_generation: bool = pydantic.Field(
        default=False,
        description="Always generate new results, regardless of existing ones.",
    )
    shared_tags: list[str] | None = pydantic.Field(
        default=None,
        description="Optional list of shared tags to apply to all new records.",
    )
    shared_meta: dict[str, pydantic.JsonValue] | None = pydantic.Field(
        default=None,
        description="Optional shared metadata dict to apply to all new records.",
    )

    # ---------- item-wise resolvers and builders ----------

    identifier_resolver: typing.Annotated[IdentifierResolverType | None, pydantic.SkipValidation] = pydantic.Field(
        default=None,
        exclude=True,  # to avoid serialization issues, we should never need to reinstantiate anyway
        description="Strategy to compute the record identifier per item. If None, uses a default rule.",
    )
    group_resolver: typing.Annotated[GroupResolverType | None, pydantic.SkipValidation] = pydantic.Field(
        default=None,
        exclude=True,  # to avoid serialization issues, we should never need to reinstantiate anyway
        description="Strategy to compute the record group per item. If None, uses a default rule.",
    )
    input_variables_builder: typing.Annotated[InputVariablesBuilderType | None, pydantic.SkipValidation] = (
        pydantic.Field(
            default=None,
            exclude=True,  # to avoid serialization issues, we should never need to reinstantiate anyway
            description="Builds input variables per item for the target prompt. If None, uses a default rule.",
        )
    )
    tags_builder: typing.Annotated[TagsBuilderType | None, pydantic.SkipValidation] = pydantic.Field(
        default=None,
        exclude=True,  # to avoid serialization issues, we should never need to reinstantiate anyway
        description="Builds tags per item for the target prompt. If None, uses a default rule.",
    )
    meta_builder: typing.Annotated[MetadataBuilderType | None, pydantic.SkipValidation] = pydantic.Field(
        default=None,
        exclude=True,  # to avoid serialization issues, we should never need to reinstantiate anyway
        description="Builds metadata per item for the target prompt. If None, uses a default rule.",
    )
    creation_meta_builder: typing.Annotated[CreationMetaBuilderType | None, pydantic.SkipValidation] = pydantic.Field(
        default=None,
        exclude=True,  # to avoid serialization issues, we should never need to reinstantiate anyway
        description="Builds creation metadata per item for the target prompt. If None, uses a default rule.",
    )
    output_validator: typing.Annotated[
        pyine.prompts.result_db.ValidatorCallableType | None, pydantic.SkipValidation
    ] = pydantic.Field(
        default=None,
        exclude=True,  # to avoid serialization issues, we should never need to reinstantiate anyway
        description="Validates the output of the prompt invocation. If None, uses a default rule.",
    )

    # --------------- private utilities and attributes ---------------

    _prompt_result_db: pyine.prompts.result_db.PromptResultDB | None = pydantic.PrivateAttr(default=None)
    _test_data_cache: pyine.organisms.datamodules.utils.caching.CodingProblemTestDataCache | None = (
        pydantic.PrivateAttr(default=None)
    )

    @pydantic.model_validator(mode="after")
    def _validate_and_resolve(self) -> "AnnotationOptions":
        """Validates and resolves config settings."""
        if self.db_path is None:
            self._prompt_result_db = pyine.prompts.result_db.get_framework_db()
        else:
            self._prompt_result_db = pyine.prompts.result_db.PromptResultDB(self.db_path)
        self._test_data_cache = None  # will be initialized later, and only if needed
        return self


def _default_identifier_resolver(
    trace: pyine.utils.code.execution.TraceResult,
    problem: pyine.data.traces.dataset_utils.CodingProblem,
    config: AnnotationOptions,
) -> str:
    """Returns the identifier to use when creating/fetching a result record.

    Implements known rules for some prompts, but if an unsupported prompt is used, will raise
    an exception.
    """
    prompts_where_solution_gives_identifier = [
        "callable_analysis",
        "code_analysis",
        "code_summary",
        "hints/stubs",
        "issues/iterators",
        "issues/todos",
    ]
    if config.prompt_config.prompt_name in prompts_where_solution_gives_identifier:
        assert trace.identifier is not None, "cannot derive identifier without a trace id"
        trace_id = pyine.data.traces.dataset_utils.TraceIdentifier.from_string(str(trace.identifier))
        solution_id = trace_id.get_parent_identifier()
        return str(solution_id)
    prompts_where_trace_gives_identifier = [
        "hints/docs",
        "hints/tests",
    ]
    if config.prompt_config.prompt_name in prompts_where_trace_gives_identifier:
        return str(trace.identifier)
    # elif config.prompt_config.prompt_name in ...
    raise NotImplementedError(f"unsupported prompt '{config.prompt_config.prompt_name}' for default resolver")


def _default_group_resolver(
    trace: pyine.utils.code.execution.TraceResult,
    problem: pyine.data.traces.dataset_utils.CodingProblem,
    config: AnnotationOptions,
) -> str | None:
    """Returns the group to use when creating/fetching a result record.

    Implements known rules for some prompts, but if an unsupported prompt is used, will raise
    an exception.
    """
    prompts_where_problem_gives_group = [
        "callable_analysis",
        "code_analysis",
        "code_summary",
        "hints/stubs",
        "issues/iterators",
        "issues/todos",
    ]
    if config.prompt_config.prompt_name in prompts_where_problem_gives_group:
        return str(problem.problem_id)
    prompts_where_solution_gives_group = [
        "hints/docs",
        "hints/tests",
    ]
    if config.prompt_config.prompt_name in prompts_where_solution_gives_group:
        assert trace.identifier is not None, "cannot derive identifier without a trace id"
        trace_id = pyine.data.traces.dataset_utils.TraceIdentifier.from_string(str(trace.identifier))
        solution_id = trace_id.get_parent_identifier()
        return str(solution_id)
    # elif config.prompt_config.prompt_name in ...
    raise NotImplementedError(f"unsupported prompt '{config.prompt_config.prompt_name}' for default resolver")


_INTERNAL_BUGGED_HINTED_TOKEN = "__orig_bugless_code__"
"""Token used in prompt variable dicts when prompting on of buggy code (specified the orig code)."""
_INTERNAL_MISLEADING_TOKEN = "__orig_expected_output__"
"""Token used in prompt variable dicts when using misleading hints (specifies the orig output)."""


def _default_input_variables_builder(
    trace: pyine.utils.code.execution.TraceResult,
    problem: pyine.data.traces.dataset_utils.CodingProblem,
    config: AnnotationOptions,
    **kwargs,
) -> dict[str, typing.Any] | None:
    """Builds input variables for the target prompt.

    Implements known rules for some prompts, but if an unsupported prompt is used, will raise
    an exception.
    """
    output = {
        # initialize the vars dict with stuff that is generic/useful for all prompts
        "code": trace.code_string,
        "inputs": str(trace.inputs),
    }
    output.update(kwargs)  # update w/ whatever the caller may have provided
    if config.prompt_config.prompt_name == "code_summary":
        # this is the simplest case: nothing more to do here
        return output
    is_hint_prompting = config.prompt_config.prompt_name.startswith("hints/")
    is_issue_prompting = config.prompt_config.prompt_name.startswith("issues/")
    is_stub_prompting = config.prompt_config.prompt_name == "hints/stubs"
    is_mislead_prompting = config.prompt_config.prompt_name == "issues/docs"
    if not is_hint_prompting and not is_issue_prompting:
        raise NotImplementedError(f"unsupported prompt '{config.prompt_config.prompt_name}' for default builder")
    if is_mislead_prompting and config.augment_config.misleading_augment_prob != 1.0:
        raise ValueError("misleading prob not 1 but using misleading prompt?")

    # -------- prepare meta flags/variables for the generation of code hints and issues --------

    assert trace.identifier is not None, "cannot derive identifier without a trace id"
    trace_id = pyine.data.traces.dataset_utils.TraceIdentifier.from_string(str(trace.identifier))
    prior_augment = trace_id.augment_category or ""
    is_already_hinted = prior_augment.endswith("hinted")
    is_already_bugged = prior_augment.startswith("bugged")
    is_already_misleading = prior_augment.endswith("misleading")
    is_obfuscated = prior_augment.startswith("obfuscated")
    # note: code might be hinted+bugged, or bugged+misleading, etc.

    # -------- early returns for invalid / unnecessary augments --------

    if is_hint_prompting and not is_stub_prompting and (is_already_hinted or is_already_misleading):
        return None  # skip; it makes little sense to generate hints on top of hints?
    if is_mislead_prompting and (is_already_misleading or is_already_hinted):
        return None  # skip; it might get confusing when both valid and misleading hints are involved
    if is_stub_prompting and is_already_bugged:
        return None  # we probably should not try to generate stubs on buggy code (cannot verify anything)
    if is_issue_prompting and not is_mislead_prompting and is_already_bugged:
        return None  # adding more bugs on top of bugs just make the bugs less subtle (so less useful?)
    if is_obfuscated and ((is_issue_prompting and not is_mislead_prompting) or is_stub_prompting):
        raise NotImplementedError("obfuscation + buggy/stubbed code is not yet supported")

    # -------- generic prompt data preparation: output + code description --------

    output["expected_output"] = str(trace.expected_output)  # default expected output (prior to potential modifs)
    if config.augment_config.fetch_code_descriptions:
        # try to go and fetch the description for the parent solution (code summary) from db
        code_summary_records = config._prompt_result_db.get_by_identifier(
            identifier=str(trace_id.get_parent_identifier()),
            prompt_name="code_summary",
        )
        if code_summary_records:
            # always keep the latest description (this should not matter too much)
            output["description"] = code_summary_records[-1].result

    # -------- special case: generate misleading hint by picking an alternative test case --------

    if is_mislead_prompting or (
        is_hint_prompting and not is_stub_prompting and config.augment_config.is_misleading_enabled
    ):
        if is_mislead_prompting or np.random.random() > config.augment_config.misleading_augment_prob:
            # if we are prompting for misleading hints specific, or if random draw succeeds, do augment
            assert config._test_data_cache is not None, "missing test data cache for misleading generation"
            test_case = config._test_data_cache.sample_alternative_test_case(
                trace_id=trace_id,
                max_inputs_length_delta=config.augment_config.max_misleading_test_length_delta,
                max_outputs_length_delta=config.augment_config.max_misleading_test_length_delta,
                match_inputs_signature=config.augment_config.misleading_test_match_signatures,
                match_outputs_signature=config.augment_config.misleading_test_match_signatures,
                sample_from_top_k=config.augment_config.misleading_test_top_k_candidates,
            )
            if test_case is not None:
                assert test_case.test_idx != trace_id.test_idx
                output["expected_output"] = str(test_case.expected_output)  # new misleading output
                output[_INTERNAL_MISLEADING_TOKEN] = str(trace.expected_output)  # orig expected output
            elif is_mislead_prompting:
                # if we did not manage to find an alternative test case for this prompt, skip the instance
                return None

    # -------- special case: generating hints on buggy code, where code is not already buggy --------

    if ((is_hint_prompting and not is_stub_prompting) or is_mislead_prompting) and (
        config.augment_config.is_bugged_hinting_enabled and not is_already_bugged
    ):
        # try to fetch a buggy version of the code string before applying the hint generation prompt
        for (
            prompt_info,
            augment_prob,
        ) in config.augment_config.buggy_code_before_hinting_prob_map.items():
            if np.random.random() > augment_prob:
                continue  # failed random draw for this augment (use bug-less misleading code)
            if isinstance(prompt_info, tuple):
                prompt_info = dict(prompt_name=prompt_info[0], prompt_version=prompt_info[1])
            else:
                prompt_info = dict(prompt_name=prompt_info)
            buggy_code_records = config._prompt_result_db.get_by_identifier(
                identifier=str(trace_id.get_parent_identifier()),
                **prompt_info,
            )
            if buggy_code_records:
                # always pick a random choice (default documented strategy)
                output["code"] = random.choice(buggy_code_records).result
                output[_INTERNAL_BUGGED_HINTED_TOKEN] = trace.code_string
                break

    # all done, the output dict should contain all necessary variables for the prompt
    return output


def _default_tags_builder(
    trace: pyine.utils.code.execution.TraceResult,
    problem: pyine.data.traces.dataset_utils.CodingProblem,
    input_vars: dict[str, typing.Any],
    config: AnnotationOptions,
) -> list[str]:
    """Builds tags for the target prompt.

    Implements known rules for some prompts, but if an unsupported prompt is used, the result will
    only contain generic tags shared by all prompt types.

    If you provide an override for this default builder, you should ensure that the returned
    tags list contains the content of the `config.shared_tags` list.
    """
    # the default/generic tags are those assigned to the trace and problem
    output_tags: list[str] = []
    output_tags.extend(problem.problem_tags)
    output_tags.extend(trace.tags)
    output_tags.extend(config.shared_tags or [])
    output_tags.append(f"llm_provider:{config.llm_provider_config.provider}")
    if config.llm_provider_config.model_kwargs.get("model", None) is not None:
        output_tags.append(f"llm_provider_model:{config.llm_provider_config.model_kwargs['model']}")
    assert not any([t.startswith("augment:") for t in output_tags]), "should not be any of these yet"

    # add augment-related tags below
    is_hint_prompting = config.prompt_config.prompt_name.startswith("hints/")
    is_issue_prompting = config.prompt_config.prompt_name.startswith("issues/")
    if is_hint_prompting or is_issue_prompting:
        is_stub_prompting = config.prompt_config.prompt_name == "hints/stubs"
        is_mislead_prompting = config.prompt_config.prompt_name == "issues/docs"
        assert trace.identifier is not None, "cannot derive identifier without a trace id"
        trace_id = pyine.data.traces.dataset_utils.TraceIdentifier.from_string(str(trace.identifier))
        if "description" in input_vars:
            assert config.augment_config.fetch_code_descriptions
            output_tags.append("augment:has_code_description")
        if trace_id.augment_category == "obfuscated":
            if is_hint_prompting and not is_stub_prompting:
                output_tags.append("augment:obfuscated_hinted")
            elif is_mislead_prompting:
                output_tags.append("augment:obfuscated_misleading")
            else:
                raise NotImplementedError("obfuscation + buggy/stubbed code is not yet supported")
        elif is_stub_prompting:
            output_tags.append("augment:stubbed")
        elif is_issue_prompting and not is_mislead_prompting:
            output_tags.append("augment:bugged")
        elif _INTERNAL_BUGGED_HINTED_TOKEN in input_vars:
            assert config.augment_config.is_bugged_hinting_enabled
            if _INTERNAL_MISLEADING_TOKEN in input_vars:
                assert config.augment_config.is_bugged_misleading_enabled
                output_tags.append("augment:bugged_misleading")
            else:
                output_tags.append("augment:bugged_hinted")
        elif _INTERNAL_MISLEADING_TOKEN in input_vars:
            assert config.augment_config.is_misleading_enabled
            output_tags.append("augment:misleading")
        elif is_hint_prompting:
            output_tags.append("augment:hinted")
        else:
            raise NotImplementedError(f"missing tag handling case for prompt '{config.prompt_config.prompt_name}'")

    # add prompt-specific tags below
    if config.prompt_config.prompt_name == "code_summary":
        template_partial_vars = config.prompt_config.partial_vars or {}
        if "target_word_count" not in template_partial_vars:
            raise ValueError("missing 'target_word_count' in prompt template partial variables")
        target_word_count = template_partial_vars["target_word_count"]
        output_tags.append(f"target_summary_word_count:{target_word_count}")
    # elif prompt_name == ...

    return output_tags


def _default_meta_builder(
    trace: pyine.utils.code.execution.TraceResult,
    problem: pyine.data.traces.dataset_utils.CodingProblem,
    config: AnnotationOptions,
    **kwargs,
) -> dict[str, pydantic.JsonValue]:
    """Builds metadata dictionaries for the target prompt.

    Implements known rules for some prompts, but if an unsupported prompt is used, the result will
    only contain generic metadata fields shared by all prompt types.

    If you provide an override for this default builder, you should ensure that the returned
    metadata dictionary contains the content of the `config.shared_meta` dictionary.
    """
    reprod_metadata = pyine.utils.reprod.get_reprod_metadata(include_installed_packages=False)
    out_metadata: dict[str, pydantic.JsonValue] = dict(reprod=reprod_metadata)
    out_metadata["llm_provider_config"] = config.llm_provider_config.model_dump()
    out_metadata["prompt_config"] = config.prompt_config.model_dump()
    out_metadata["augment_config"] = config.augment_config.model_dump()
    out_metadata["was_force_generated"] = config.force_generation
    out_metadata["shared_tags"] = config.shared_tags or []
    out_metadata.update(config.shared_meta or {})
    out_metadata.update(kwargs)
    return out_metadata


def _default_creation_meta_builder(
    trace: pyine.utils.code.execution.TraceResult,
    problem: pyine.data.traces.dataset_utils.CodingProblem,
    config: AnnotationOptions,
    **kwargs,
) -> pyine.prompts.result_db.CreationMeta:
    """Builds creation metadata for the target prompt.

    Implements known rules for some prompts, but if an unsupported prompt is used, the result will
    only contain generic creation metadata fields shared by all prompt types.
    """
    return pyine.prompts.result_db.CreationMeta(
        provider=config.llm_provider_config.provider,
        llm_params=config.llm_provider_config.model_dump(),
        **kwargs,
    )


def _default_output_validator(
    result_str: str,
    raw_output: typing.Any,
    trace: pyine.utils.code.execution.TraceResult,
    problem: pyine.data.traces.dataset_utils.CodingProblem,
    config: AnnotationOptions,
    input_vars: dict[str, typing.Any],
    tags: list[str],
) -> bool:
    """Validates the output of the prompt invocation.

    Implements known rules for some prompts, but if an unsupported prompt is used, the result will
    always be accepted as-is (i.e., no validation is performed).
    """
    if config.prompt_config.prompt_name == "hints/stubs":
        # special handling for this one: it's not supposed to 'still work', so forget tracing it
        assert "augment:stubbed" in tags, "missing augment tag for stubbed code"
        return True
    is_hint_prompting = config.prompt_config.prompt_name.startswith("hints/")
    is_issue_prompting = config.prompt_config.prompt_name.startswith("issues/")
    if is_hint_prompting or is_issue_prompting:
        # for both issues and hints, we will be tracing the newly generated code to see the results:
        # => for all issue types, we expect the execution output to NOT be the expected one;
        # => in contrast, hints should not influence the outcome of executing the code.
        if result_str == trace.code_string:
            return False  # if code has not changed, this is not a good sample, no matter what
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # no need to capture warnings related to traced code
            try:
                new_trace_result = pyine.utils.code.execution.execute_and_trace_code(
                    code_string=result_str,
                    inputs=trace.inputs,
                    expected_output=trace.expected_output,
                    identifier=trace.identifier,
                    entrypoint_name=trace.entrypoint_name,
                    trace_only_inside_code_string=True,
                    max_valid_events=trace.max_valid_events,
                    max_events_per_line=trace.max_events_per_line,
                    max_var_repr_length=trace.max_var_repr_length,
                    timeout_seconds=config.validation_timeout_seconds,
                    use_safe_execution=True,  # parent runs in a thread, so isolate the child
                )
            except pyine.utils.code.execution.CodeAnalysisFailure:
                # there's a problem with the code string itself, reject the proposal
                return False
        output_is_different = (
            (trace.return_value is not None and new_trace_result.return_value != trace.return_value)
            or (new_trace_result.exception != trace.exception)
            or (trace.return_value is None and new_trace_result.stdout != trace.stdout)
        )
        if is_issue_prompting:
            assert "augment:bugged" in tags, "missing augment tag for bugged code"
            return output_is_different  # we want a different output for bugged code
        else:  # is_hint_prompting
            if any([bug_tag in tags for bug_tag in ["augment:bugged_hinted", "augment:bugged_misleading"]]):
                assert _INTERNAL_BUGGED_HINTED_TOKEN in input_vars
                # we are actually hinting a BUGGED code snippet, so expect a different output
                return output_is_different
            else:
                return not output_is_different  # we want the same output for hinted code
    # ultimate fallback: accept everything (we don't know how to validate it)
    return True


@dataclasses.dataclass
class AnnotationReport:
    """Simple report structure for annotation results."""

    total_samples: int = 0
    """Total number of data samples to annotate that were considered."""
    annotated_samples: int = 0
    """Total number of data samples that had at least one annotation successfully generated."""
    skipped_samples: int = 0
    """Total number of data samples that were skipped due to pre-existing results or filter rules."""
    new_results_generated: int = 0
    """Total number of new results that were generated by prompt chain invocation."""
    total_tokens_exchanged: int = 0
    """Total number of tokens exchanged with the LLM (best-effort find, not reliable if not using OpenAI models)."""
    errors: int = 0
    """Total number of errors caught during processing."""
    error_message_tracebacks: list[str] = dataclasses.field(default_factory=list)
    """List of tracebacks for each error caught during processing (these are long!)."""
    error_messages: list[str] = dataclasses.field(default_factory=list)
    """List of error messages for each error caught during processing (useful for printing)."""

    def add(self, other: "AnnotationReport") -> "AnnotationReport":
        """Accumulate counts from another report into this one and return self."""
        self.total_samples += other.total_samples
        self.annotated_samples += other.annotated_samples
        self.skipped_samples += other.skipped_samples
        self.new_results_generated += other.new_results_generated
        self.total_tokens_exchanged += other.total_tokens_exchanged
        self.errors += other.errors
        self.error_message_tracebacks.extend(other.error_message_tracebacks)
        self.error_messages.extend(other.error_messages)
        return self

    def __iadd__(self, other: "AnnotationReport") -> "AnnotationReport":
        return self.add(other)

    def summary(self) -> str:
        """Returns a summary string with counts and stats."""
        annotated_frac = self.annotated_samples / self.total_samples if self.total_samples > 0 else 0
        skipped_frac = self.skipped_samples / self.total_samples if self.total_samples > 0 else 0
        return (
            f"total samples: {self.total_samples:_}, "
            f"annotated: {self.annotated_samples:_} ({annotated_frac:.1%} of total), "
            f"skipped: {self.skipped_samples:_} ({skipped_frac:.1%} of total), "
            f"new results: {self.new_results_generated:_}, "
            f"tokens exchanged: {self.total_tokens_exchanged:_}, "
            f"errors: {self.errors}"
        ) + (f", error messages: {self.error_messages[:5]}" if self.error_messages else "")


async def annotate_trace_dataset(
    dataset: pyine.data.traces.dataset_reader.DatasetReader,
    config: AnnotationOptions,
    show_progress: bool = True,
    dry_run: bool = False,
    shuffle_indices: bool = False,
    parallel: bool = True,
    max_workers: int | None = None,
    max_in_flight_jobs: int | None = 32,
    verbose: bool = False,
) -> AnnotationReport:
    """Annotates a trace dataset by invoking LLM prompts per element and logging results.

    This function iterates over a DatasetReader (traces) and, for each element, builds prompt
    inputs and calls into the framework prompt manager through the prompt result DB utility,
    so that results are persisted and can be reused across runs. Depending on configuration
    settings, if a result already exists for an element, it will be skipped.

    Args:
        dataset: DatasetReader instance to iterate over.
        config: configuration with behavior and prompt settings.
        show_progress: whether to show a progress bar.
        dry_run: whether to skip logging actual annotations and only report results and stats.
        shuffle_indices: whether to shuffle the dataset indices before processing, for a
            potentially more uniform annotation coverage if the dataset is not sorted.
        parallel: whether to process dataset items concurrently using a thread pool.
        max_workers: optional maximum number of worker threads to use when parallel=True.
        max_in_flight_jobs: maximum number of concurrent jobs to run when parallel=True.
        verbose: verbose logging of annotation progress reports.

    Returns:
        A small report dictionary with annotation outcome counts.
    """
    model = pyine.utils.llm_providers.get_model_from_provider_config(config.llm_provider_config)
    prompt_name, prompt_version = config.prompt_config.prompt_name, config.prompt_config.version
    if prompt_name not in pyine.prompts.manager.list_prompts():
        raise ValueError(f"unknown prompt '{prompt_name}'")
    if prompt_version is not None and prompt_version not in pyine.prompts.manager.list_prompt_versions(prompt_name):
        raise ValueError(f"unknown prompt version '{prompt_version}' for prompt '{prompt_name}'")
    if config.augment_config.is_misleading_enabled and config._test_data_cache is None:
        assert isinstance(dataset, pyine.data.traces.dataset_reader.DatasetReader)
        logger.info("preparing or reloading coding problem test data cache for target dataset")
        config._test_data_cache = (
            pyine.organisms.datamodules.utils.caching.CodingProblemTestDataCache.build_from_dataset(dataset)
        )
    base_filter_fn = pyine.data.utils.filter_rules.build_filter_from_rule(
        rule=(config.base_filter_rule or ""),
        case_sensitive=False,
    )
    identifier_getter = config.identifier_resolver or _default_identifier_resolver
    group_getter = config.group_resolver or _default_group_resolver
    prompt_input_vars_getter = config.input_variables_builder or _default_input_variables_builder
    tags_getter = config.tags_builder or _default_tags_builder
    meta_getter = config.meta_builder or _default_meta_builder
    creation_meta_getter = config.creation_meta_builder or _default_creation_meta_builder
    data_indices = list(config.target_indices or range(len(dataset)))
    if shuffle_indices:
        random.shuffle(data_indices)
    wrapped_data_indices = tqdm.tqdm(
        data_indices,
        disable=not show_progress,
        desc="annotation progress",
        smoothing=0.05,  # for a longer-window estimate
    )
    output = AnnotationReport()
    if not parallel:
        for iter_idx, sample_idx in enumerate(wrapped_data_indices):
            output += _process_one_annotation(
                trace=dataset[sample_idx],
                problem=dataset.get_problem_data(sample_idx),
                sample_idx=sample_idx,
                model=model,
                config=config,
                dry_run=dry_run,
                base_filter_fn=base_filter_fn,
                db=config._prompt_result_db,
                identifier_getter=identifier_getter,
                group_getter=group_getter,
                prompt_input_vars_getter=prompt_input_vars_getter,
                tags_getter=tags_getter,
                meta_getter=meta_getter,
                creation_meta_getter=creation_meta_getter,
            )
            if verbose and iter_idx % 50 == 0:  # print progress report every 50 iterations
                wrapped_data_indices.write(f"progress report: {output.summary()}")
        return output
    # fallback: parallel version using a thread pool with a sliding window over submitted jobs

    def _submit_one(
        sample_idx: typing.Hashable,
        executor: concurrent.futures.Executor,
    ) -> concurrent.futures.Future:
        sample_idx = typing.cast(int, sample_idx)
        return executor.submit(
            _process_one_annotation,
            trace=dataset[sample_idx],
            problem=dataset.get_problem_data(sample_idx),
            sample_idx=sample_idx,
            model=model,
            config=config,
            dry_run=dry_run,
            base_filter_fn=base_filter_fn,
            db=config._prompt_result_db,  # noqa
            identifier_getter=identifier_getter,
            group_getter=group_getter,
            prompt_input_vars_getter=prompt_input_vars_getter,
            tags_getter=tags_getter,
            meta_getter=meta_getter,
            creation_meta_getter=creation_meta_getter,
        )

    def _process_result(_: typing.Hashable, result: AnnotationReport) -> None:
        nonlocal output
        output += result

    def _progress_callback(_: list[typing.Hashable], completed: list[typing.Hashable]) -> None:
        if verbose and completed and len(completed) % 50 == 0:  # print progress report every 50 completions
            wrapped_data_indices.write(f"progress report (completed {len(completed)}): {output.summary()}")

    await pyine.utils.concurrency.run_with_sliding_window(
        input_items=wrapped_data_indices,
        submit_one=_submit_one,
        process_result=_process_result,
        progress_callback=_progress_callback,
        max_workers=max_workers,
        max_in_flight_jobs=max_in_flight_jobs,
    )
    logger.info(f"final report: {output.summary()}")
    return output


def _process_one_annotation(
    trace: pyine.utils.code.execution.TraceResult,
    problem: pyine.data.traces.dataset_utils.CodingProblem,
    sample_idx: int,
    model: langchain_core.language_models.BaseLanguageModel,
    config: AnnotationOptions,
    dry_run: bool,
    base_filter_fn: typing.Callable[[list[str]], bool],
    db: pyine.prompts.result_db.PromptResultDB,
    identifier_getter: IdentifierResolverType,
    group_getter: GroupResolverType,
    prompt_input_vars_getter: InputVariablesBuilderType,
    tags_getter: TagsBuilderType,
    meta_getter: MetadataBuilderType,
    creation_meta_getter: CreationMetaBuilderType,
) -> AnnotationReport:
    """Helper function to process one annotation."""
    rep = AnnotationReport(total_samples=1)
    try:
        identifier = identifier_getter(trace, problem, config)
        input_vars = prompt_input_vars_getter(trace, problem, config)
        if input_vars is None:
            rep.skipped_samples += 1
            return rep
        tags = tags_getter(trace, problem, input_vars, config)
        if base_filter_fn(tags):
            rep.skipped_samples += 1
            return rep
        creation_meta = creation_meta_getter(trace, problem, config)
        meta_dict = meta_getter(trace, problem, config, input_vars=input_vars)
        if config.output_validator:
            output_validator = config.output_validator
        else:
            output_validator = functools.partial(
                _default_output_validator,
                trace=trace,
                problem=problem,
                config=config,
                input_vars=input_vars,
                tags=tags,
            )
        records = pyine.prompts.result_db.fetch_or_generate_prompt_results(
            model=model,
            identifier=identifier,
            input_variables=input_vars,
            prompt_config=config.prompt_config,
            db=db,
            runnable_name=config.runnable_name,
            max_result_age=config.max_result_age,
            tag_filter_rule=config.record_tag_filter_rule,
            deduplicate_results=config.deduplicate_results,
            generate_until_result_count=config.min_results_per_item,
            output_validator=output_validator,
            max_unsatisfactory_retries=config.max_unsatisfactory_retries,
            log_new_results=not dry_run,
            force_generation=config.force_generation,
            creation_meta=creation_meta,
            group=group_getter(trace, problem, config),
            tags=tags,
            meta=meta_dict,
        )
        new_records = [r for r in records if r.creation_meta == creation_meta]
        if len(new_records) == 0:
            rep.skipped_samples += 1
            return rep
        rep.new_results_generated += len(new_records)
        rep.annotated_samples += 1
        for new_rec in new_records:
            # best-effort for openai-like LLM outputs
            llm_output = new_rec.creation_meta.llm_output or {}
            token_usage_dict = llm_output.get("token_usage", {})
            rep.total_tokens_exchanged += token_usage_dict.get("total_tokens", 0)
    except (KeyboardInterrupt, GeneratorExit, MemoryError, asyncio.CancelledError):
        raise  # we should not be trying to catch/silence there here
    except (pyine.prompts.result_db.ValidationFailedError, Exception) as e:
        rep.errors += 1
        id_str = f"sample_idx={sample_idx}, trace_id={trace.identifier}"
        rep.error_messages.append(f"{id_str}: {e}")
        rep.error_message_tracebacks.append("".join(traceback.format_exception(e)))
        if isinstance(e, pyine.prompts.result_db.ValidationFailedError):
            logger.warning(
                f"result validation failed for {id_str} " f"(attempts: {config.max_unsatisfactory_retries}) "
            )
        else:
            logger.exception(f"failed to process and annotate data for {id_str};\n{e}")
    return rep
