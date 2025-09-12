import asyncio
import concurrent.futures
import dataclasses
import datetime
import functools
import logging
import pathlib
import random
import typing
import warnings

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
import pyine.utils.code.validation
import pyine.utils.llm_providers
import pyine.utils.reprod

logger = logging.getLogger(__name__)

IdentifierResolverType = typing.Callable[
    [pyine.utils.code.execution.TraceResult, pyine.data.traces.dataset_utils.CodingProblem, "AnnotationOptions"],
    str,
]
"""Type used to represent an identifier resolver callable."""
GroupResolverType = typing.Callable[
    [pyine.utils.code.execution.TraceResult, pyine.data.traces.dataset_utils.CodingProblem, "AnnotationOptions"],
    str | None,
]
"""Type used to represent a group resolver callable."""
InputVariablesBuilderType = typing.Callable[
    [pyine.utils.code.execution.TraceResult, pyine.data.traces.dataset_utils.CodingProblem, "AnnotationOptions"],
    dict[str, typing.Any],
]
"""Type used to represent an input variables builder callable."""
TagsBuilderType = typing.Callable[
    [pyine.utils.code.execution.TraceResult, pyine.data.traces.dataset_utils.CodingProblem, "AnnotationOptions"],
    list[str],
]
"""Type used to represent a tags builder callable."""
MetadataBuilderType = typing.Callable[
    [pyine.utils.code.execution.TraceResult, pyine.data.traces.dataset_utils.CodingProblem, "AnnotationOptions"],
    dict[str, pydantic.JsonValue],
]
"""Type used to represent a metadata builder callable."""
CreationMetaBuilderType = typing.Callable[
    [pyine.utils.code.execution.TraceResult, pyine.data.traces.dataset_utils.CodingProblem, "AnnotationOptions"],
    pyine.prompts.result_db.CreationMeta,
]
"""Type used to represent a creation metadata builder callable."""


class AnnotationOptions(pydantic.BaseModel):
    """Options controlling dataset annotation via prompt invocations."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
    """Pydantic model configuration (freezes the dataclass)."""

    # ---------- prompt-related settings ----------

    llm_provider_config: pyine.utils.llm_providers.LLMProviderConfig = pydantic.Field(
        description="Configuration to pass to the LLM provider pipeline.",
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
    output_validator: pyine.prompts.result_db.ValidatorCallableType | None = pydantic.Field(
        default=None,
        description="Validates the output of the prompt invocation. If None, uses a default rule.",
    )

    # --------------- private utilities and attributes ---------------

    _prompt_result_db: pyine.prompts.result_db.PromptResultDB | None = pydantic.PrivateAttr(default=None)

    @pydantic.model_validator(mode="after")
    def _validate_and_resolve(self) -> "AnnotationOptions":
        """Validates and resolves config settings."""
        if self.db_path is None:
            self._prompt_result_db = pyine.prompts.result_db.get_framework_db()
        else:
            self._prompt_result_db = pyine.prompts.result_db.PromptResultDB(self.db_path)
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


def _default_input_variables_builder(
    trace: pyine.utils.code.execution.TraceResult,
    problem: pyine.data.traces.dataset_utils.CodingProblem,
    config: AnnotationOptions,
) -> dict[str, typing.Any]:
    """Builds input variables for the target prompt.

    Implements known rules for some prompts, but if an unsupported prompt is used, will raise
    an exception.
    """
    output = {
        "code": trace.code_string,
        "inputs": str(trace.inputs),
    }
    assert trace.identifier is not None, "cannot derive identifier without a trace id"
    trace_id = pyine.data.traces.dataset_utils.TraceIdentifier.from_string(str(trace.identifier))
    if config.prompt_config.prompt_name == "code_summary":
        # nothing more to do here
        return output
    if config.prompt_config.prompt_name.startswith("hints/") or config.prompt_config.prompt_name.startswith("issues/"):
        output["expected_output"] = trace.expected_output
        # try to go and fetch the description for the parent solution (code summary) from db
        code_summary_records = config._prompt_result_db.get_by_identifier(
            identifier=str(trace_id.get_parent_identifier()),
            prompt_name="code_summary",
        )
        if code_summary_records:
            # always keep the latest description (this should not matter too much)
            output["description"] = code_summary_records[-1].result
        return output
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
    output_tags.append(f"llm_provider:{config.llm_provider_config.provider}")
    if config.llm_provider_config.model_kwargs.get("model", None) is not None:
        output_tags.append(f"llm_provider_model:{config.llm_provider_config.model_kwargs['model']}")
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
    default_metadata["llm_provider_config"] = config.llm_provider_config.model_dump()
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
        provider=config.llm_provider_config.provider,
        llm_params=config.llm_provider_config.model_dump(),  # noqa
    )


def _default_output_validator(
    result_str: str,
    raw_output: typing.Any,
    trace: pyine.utils.code.execution.TraceResult,
    problem: pyine.data.traces.dataset_utils.CodingProblem,
    config: AnnotationOptions,
) -> bool:
    """Validates the output of the prompt invocation.

    Implements known rules for some prompts, but if an unsupported prompt is used, the result will
    always be accepted as-is (i.e., no validation is performed).
    """
    if config.prompt_config.prompt_name == "hints/stubs":
        # special handling for this one: it's not supposed to 'still work', so forget tracing it
        return True
    elif config.prompt_config.prompt_name.startswith("issues/") or config.prompt_config.prompt_name.startswith(
        "hints/"
    ):
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
        if config.prompt_config.prompt_name.startswith("issues/"):
            return output_is_different  # we want a different output for bugged code
        else:  # config.prompt_config.prompt_name.startswith("hints/")
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
            f"total samples: {self.total_samples:_}, "
            f"annotated: {self.annotated_samples:_} ({annotated_frac:.1%} of total), "
            f"skipped: {self.skipped_samples:_} ({skipped_frac:.1%} of total), "
            f"new results: {self.new_results_generated:_}, "
            f"tokens exchanged: {self.total_tokens_exchanged:_}, "
            f"errors: {self.errors}"
        )


def annotate_trace_dataset(
    dataset: pyine.data.traces.dataset_reader.DatasetReader,
    config: AnnotationOptions,
    show_progress: bool = True,
    dry_run: bool = False,
    shuffle_indices: bool = False,
    parallel: bool = True,
    max_workers: int | None = None,
    max_in_flight_jobs: int | None = 5_000,
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
    wrapped_data_indices = tqdm.tqdm(data_indices, disable=not show_progress, desc="annotation progress")
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
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        in_flight: set[concurrent.futures.Future] = set()
        iter_items = iter(wrapped_data_indices)
        submitted = 0
        completed = 0

        def _submit_one(sample_idx: int) -> concurrent.futures.Future:
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

        # initially fill the window to its max size
        while len(in_flight) < max_in_flight_jobs:
            try:
                sample_idx = next(iter_items)
            except StopIteration:
                break
            future = _submit_one(sample_idx)
            in_flight.add(future)
            submitted += 1
        # update output report and continue filling window if needed
        while in_flight:
            done = next(concurrent.futures.as_completed(in_flight))
            output += done.result()
            in_flight.remove(done)
            completed += 1
            if verbose and completed % 50 == 0:  # print progress report every 50 completions
                wrapped_data_indices.write(f"progress report: {output.summary()}")
            # refill the window with the next item (if any)
            try:
                sample_idx = next(iter_items)
            except StopIteration:
                continue
            future = _submit_one(sample_idx)
            in_flight.add(future)
            submitted += 1
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
        prompt_tags = tags_getter(trace, problem, config)
        if base_filter_fn(prompt_tags):
            rep.skipped_samples += 1
            return rep
        identifier = identifier_getter(trace, problem, config)
        creation_meta = creation_meta_getter(trace, problem, config)
        if config.output_validator:
            output_validator = config.output_validator
        else:
            output_validator = functools.partial(
                _default_output_validator,
                trace=trace,
                problem=problem,
                config=config,
            )
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
            output_validator=output_validator,
            max_unsatisfactory_retries=config.max_unsatisfactory_retries,
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
    except pyine.prompts.result_db.ValidationFailedError as e:
        rep.errors += 1
        attempts = config.max_unsatisfactory_retries + 1
        logger.warning(f"result validation failed after {attempts} attempt(s) for data sample at idx: {sample_idx}")
        logger.debug(f"exception details: {e}")
    except Exception as e:
        rep.errors += 1
        logger.exception(f"failed to process and annotate data sample at idx: {sample_idx}")
        logger.debug(f"exception details: {e}")
    return rep
