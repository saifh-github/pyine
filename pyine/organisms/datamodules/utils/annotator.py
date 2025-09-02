import asyncio
import concurrent.futures
import dataclasses
import datetime
import logging
import pathlib
import typing

import langchain_core.language_models
import pydantic
import tqdm

import pyine.data.traces.dataset_reader
import pyine.data.traces.dataset_utils
import pyine.data.utils.filter_rules
import pyine.prompts.result_db
import pyine.prompts.types
import pyine.prompts.utils
import pyine.utils.code.execution
import pyine.utils.llm_providers
import pyine.utils.reprod

logger = logging.getLogger(__name__)

ResolverCallableInputType = [
    pyine.utils.code.execution.TraceResult,
    pyine.data.traces.dataset_utils.CodingProblem,
    "AnnotationOptions",
]
"""Type used to represent the arguments that a resolver callable expects."""
IdentifierResolverType = typing.Callable[[*ResolverCallableInputType], str]
"""Type used to represent an identifier resolver callable."""
GroupResolverType = typing.Callable[[*ResolverCallableInputType], str | None]
"""Type used to represent a group resolver callable."""
InputVariablesBuilderType = typing.Callable[[*ResolverCallableInputType], dict[str, typing.Any]]
"""Type used to represent an input variables builder callable."""
TagsBuilderType = typing.Callable[[*ResolverCallableInputType], list[str]]
"""Type used to represent a tags builder callable."""
MetadataBuilderType = typing.Callable[[*ResolverCallableInputType], dict[str, pydantic.JsonValue]]
"""Type used to represent a metadata builder callable."""
CreationMetaBuilderType = typing.Callable[[*ResolverCallableInputType], pyine.prompts.result_db.CreationMeta]
"""Type used to represent a creation metadata builder callable."""


class AnnotationOptions(pydantic.BaseModel):
    """Options controlling dataset annotation via prompt invocations."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
    """Pydantic model configuration (freezes the dataclass)."""

    # ---------- prompt-related settings ----------

    llm_provider_kwargs: dict[str, typing.Any] = pydantic.Field(
        description="Keyword arguments to pass to the LLM provider pipeline.",
    )
    prompt_config: pyine.prompts.types.PromptBuildConfig = pydantic.Field(
        description="Configuration for the prompt to use to generate 'annotations'.",
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

    identifier_resolver: IdentifierResolverType | None = pydantic.Field(
        default=None,
        description="Strategy to compute the record identifier per item. If None, uses a default rule.",
    )
    group_resolver: GroupResolverType | None = pydantic.Field(
        default=None,
        description="Strategy to compute the record group per item. If None, uses a default rule.",
    )
    input_variables_builder: InputVariablesBuilderType | None = pydantic.Field(
        default=None,
        description="Builds input variables per item for the target prompt. If None, uses a default rule.",
    )
    tags_builder: TagsBuilderType | None = pydantic.Field(
        default=None,
        description="Builds tags per item for the target prompt. If None, uses a default rule.",
    )
    meta_builder: MetadataBuilderType | None = pydantic.Field(
        default=None,
        description="Builds metadata per item for the target prompt. If None, uses a default rule.",
    )
    creation_meta_builder: CreationMetaBuilderType | None = pydantic.Field(
        default=None,
        description="Builds creation metadata per item for the target prompt. If None, uses a default rule.",
    )


def _default_identifier_resolver(
    trace: pyine.utils.code.execution.TraceResult,
    problem: pyine.data.traces.dataset_utils.CodingProblem,
    config: AnnotationOptions,
) -> str:
    """Returns the identifier to use when creating/fetching a result record.

    Implements known rules for some prompts, but if an unsupported prompt is used, will raise
    an exception.
    """
    prompts_where_solution_gives_identifier = ["code_summary"]
    if config.prompt_config.prompt_name in prompts_where_solution_gives_identifier:
        assert trace.identifier is not None, "cannot derive identifier without a trace id"
        trace_id = pyine.data.traces.dataset_utils.TraceIdentifier.from_string(str(trace.identifier))
        solution_id = trace_id.get_parent_identifier()
        return str(solution_id)
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
    prompts_where_problem_gives_group = ["code_summary"]
    if config.prompt_config.prompt_name in prompts_where_problem_gives_group:
        return str(problem.problem_id)
    # elif config.prompt_config.prompt_name in ...
    raise NotImplementedError(f"unsupported prompt '{config.prompt_config.prompt_name}' for default resolver")


def _default_input_variables_builder(
    trace: pyine.utils.code.execution.TraceResult,
    problem: pyine.data.traces.dataset_utils.CodingProblem,
    config: AnnotationOptions,
) -> dict[str, typing.Any]:
    """Builds input variables for the target prompt.

    Implements known rules for some prompts, but if an unsupported prompt is used, will raise
    an exception.
    """
    if config.prompt_config.prompt_name == "code_summary":
        return {
            "code": trace.code_string,
            "description": problem.problem_statement,
        }
    # elif config.prompt_config.prompt_name == ...
    raise NotImplementedError(f"unsupported prompt '{config.prompt_config.prompt_name}' for default builder")


def _default_tags_builder(
    trace: pyine.utils.code.execution.TraceResult,
    problem: pyine.data.traces.dataset_utils.CodingProblem,
    config: AnnotationOptions,
) -> list[str]:
    """Builds tags for the target prompt.

    Implements known rules for some prompts, but if an unsupported prompt is used, the result will
    only contain generic tags shared by all prompt types.

    If you provide an override for this default builder, you should ensure that the returned
    tags list contains the content of the `config.shared_tags` list.
    """
    output_tags = []
    # the 'generic tags' are those already assigned to the trace and problem
    output_tags.extend(problem.problem_tags)
    output_tags.extend(trace.tags)
    output_tags.extend(config.shared_tags or [])
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
) -> dict[str, pydantic.JsonValue]:
    """Builds metadata dictionaries for the target prompt.

    Implements known rules for some prompts, but if an unsupported prompt is used, the result will
    only contain generic metadata fields shared by all prompt types.

    If you provide an override for this default builder, you should ensure that the returned
    metadata dictionary contains the content of the `config.shared_meta` dictionary.
    """
    default_metadata = pyine.utils.reprod.get_reprod_metadata(include_installed_packages=False)
    default_metadata = typing.cast(dict[str, typing.Any], default_metadata)
    default_metadata["llm_provider_config"] = config.llm_provider_kwargs
    default_metadata["prompt_config"] = config.prompt_config.model_dump()
    default_metadata["was_force_generated"] = config.force_generation
    default_metadata["shared_tags"] = config.shared_tags or []
    default_metadata["shared_meta"] = config.shared_meta or {}
    default_metadata.update(config.shared_meta or {})
    return default_metadata


def _default_creation_meta_builder(
    trace: pyine.utils.code.execution.TraceResult,
    problem: pyine.data.traces.dataset_utils.CodingProblem,
    config: AnnotationOptions,
) -> pyine.prompts.result_db.CreationMeta:
    """Builds creation metadata for the target prompt.

    Implements known rules for some prompts, but if an unsupported prompt is used, the result will
    only contain generic creation metadata fields shared by all prompt types.
    """
    return pyine.prompts.result_db.CreationMeta(
        provider=config.llm_provider_kwargs.get("provider", None),
        llm_params=config.llm_provider_kwargs,  # noqa
    )


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

    def add(self, other: "AnnotationReport") -> "AnnotationReport":
        """Accumulate counts from another report into this one and return self."""
        self.total_samples += other.total_samples
        self.annotated_samples += other.annotated_samples
        self.skipped_samples += other.skipped_samples
        self.new_results_generated += other.new_results_generated
        self.total_tokens_exchanged += other.total_tokens_exchanged
        self.errors += other.errors
        return self

    def __iadd__(self, other: "AnnotationReport") -> "AnnotationReport":
        return self.add(other)

    def summary(self) -> str:
        """Returns a summary string with counts and stats."""
        annotated_frac = self.annotated_samples / self.total_samples if self.total_samples > 0 else 0
        skipped_frac = self.skipped_samples / self.total_samples if self.total_samples > 0 else 0
        return (
            f"total (considered) samples: {self.total_samples:_}, "
            f"annotated samples: {self.annotated_samples:_} ({annotated_frac:.1%}% of total),"
            f"skipped samples: {self.skipped_samples:_} ({skipped_frac:.1%}% of total), "
            f"new results generated: {self.new_results_generated:_}, "
            f"total tokens exchanged: {self.total_tokens_exchanged:_}, "
            f"errors: {self.errors}"
        )


def annotate_trace_dataset(
    dataset: pyine.data.traces.dataset_reader.DatasetReader,
    config: AnnotationOptions,
    show_progress: bool = True,
    dry_run: bool = False,
    parallel: bool = True,
    max_workers: int | None = None,
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
        parallel: whether to process dataset items concurrently using a thread pool.
        max_workers: optional maximum number of worker threads to use when parallel=True.
        verbose: verbose logging of annotation progress reports (for non-parallel runs only).

    Returns:
        A small report dictionary with annotation outcome counts.
    """
    model = pyine.utils.llm_providers.get_model_from_provider(**config.llm_provider_kwargs)
    prompt_name, prompt_version = config.prompt_config.prompt_name, config.prompt_config.version
    if prompt_name not in pyine.prompts.manager.list_prompts():
        raise ValueError(f"unknown prompt '{prompt_name}'")
    if prompt_version is not None and prompt_version not in pyine.prompts.manager.list_prompt_versions(prompt_name):
        raise ValueError(f"unknown prompt version '{prompt_version}' for prompt '{prompt_name}'")
    data_indices = list(config.target_indices or range(len(dataset)))
    wrapped_data_indices = tqdm.tqdm(data_indices, disable=not show_progress, desc="annotation progress")
    output = AnnotationReport()
    if not parallel:
        for sample_idx in wrapped_data_indices:
            output += _process_one_annotation(
                trace=dataset[sample_idx],
                problem=dataset.get_problem_data(sample_idx),
                sample_idx=sample_idx,
                model=model,
                config=config,
                dry_run=dry_run,
            )
            if verbose and sample_idx % 10 == 0:  # print progress report every 10 samples
                wrapped_data_indices.write(f"progress report: {output.summary()}")
        return output
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _process_one_annotation,
                trace=dataset[sample_idx],
                problem=dataset.get_problem_data(sample_idx),
                sample_idx=sample_idx,
                model=model,
                config=config,
                dry_run=dry_run,
            )
            for sample_idx in wrapped_data_indices
        ]
        for future in concurrent.futures.as_completed(futures):
            output += future.result()
    logger.info(f"final report: {output.summary()}")
    return output


def _process_one_annotation(
    trace: pyine.utils.code.execution.TraceResult,
    problem: pyine.data.traces.dataset_utils.CodingProblem,
    sample_idx: int,
    model: langchain_core.language_models.BaseLanguageModel,
    config: AnnotationOptions,
    dry_run: bool,
) -> AnnotationReport:
    """Helper function to process one annotation."""
    base_filter_fn = pyine.data.utils.filter_rules.build_filter_from_rule(
        rule=(config.base_filter_rule or ""),
        case_sensitive=False,
    )
    if config.db_path is None:
        db = pyine.prompts.result_db.get_framework_db()
    else:
        db = pyine.prompts.result_db.PromptResultDB(config.db_path)
    identifier_getter = config.identifier_resolver or _default_identifier_resolver
    group_getter = config.group_resolver or _default_group_resolver
    prompt_input_vars_getter = config.input_variables_builder or _default_input_variables_builder
    tags_getter = config.tags_builder or _default_tags_builder
    meta_getter = config.meta_builder or _default_meta_builder
    creation_meta_getter = config.creation_meta_builder or _default_creation_meta_builder
    rep = AnnotationReport(total_samples=1)
    try:
        prompt_tags = tags_getter(trace, problem, config)
        if base_filter_fn(prompt_tags):
            rep.skipped_samples += 1
            return rep
        identifier = identifier_getter(trace, problem, config)
        creation_meta = creation_meta_getter(trace, problem, config)
        records = pyine.prompts.result_db.fetch_or_generate_prompt_results(
            model=model,
            identifier=identifier,
            input_variables=prompt_input_vars_getter(trace, problem, config),
            prompt_config=config.prompt_config,
            db=db,
            runnable_name=config.runnable_name,
            max_result_age=config.max_result_age,
            tag_filter_rule=config.record_tag_filter_rule,
            deduplicate_results=config.deduplicate_results,
            generate_until_result_count=config.min_results_per_item,
            log_new_results=not dry_run,
            force_generation=config.force_generation,
            creation_meta=creation_meta,
            group=group_getter(trace, problem, config),
            tags=prompt_tags,
            meta=meta_getter(trace, problem, config),
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
    except Exception as e:
        rep.errors += 1
        logger.exception(f"failed to process and annotate data sample at idx: {sample_idx}")
    return rep
