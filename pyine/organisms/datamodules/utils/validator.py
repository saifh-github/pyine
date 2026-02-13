"""Utility helpers for the trace annotation validator.

This module holds the domain-level logic used by the ``trace_annot_validator`` CLI app
(tag building, bugged-record filtering, LLM verdict parsing). It mirrors the role of
`pyine.organisms.datamodules.utils.annotator` for the annotation generator app:
the CLI stays thin orchestration while reusable logic lives here.
"""

import logging
import typing

import pydantic

import pyine.data.traces.dataset_reader
import pyine.data.traces.dataset_utils
import pyine.organisms.datamodules.samples.common
import pyine.prompts.configs.validation.misleading
import pyine.prompts.result_db
import pyine.prompts.types
import pyine.utils.code.execution
import pyine.utils.llm_providers
import pyine.utils.parsing

logger = logging.getLogger(__name__)

LINEAGE_META_KEYS: typing.Final = frozenset({"source_record_uid", "source_prompt_name", "source_identifier"})
"""Meta keys reserved for source record lineage; cannot be overwritten by --shared-meta."""

VerdictPayload = pyine.prompts.configs.validation.misleading.VerdictPayload


def build_validation_tags(
    problem: pyine.data.traces.dataset_utils.CodingProblem,
    trace: pyine.utils.code.execution.TraceResult,
    source_record: pyine.prompts.result_db.PromptResultRecord,
    llm_provider_config: pyine.utils.llm_providers.LLMProviderConfig,
    shared_tags: list[str] | None,
) -> list[str]:
    """Builds tags for a validation record.

    The verdict tag (e.g. "verdict:misleading") is appended later by the validator closure.
    """
    tags: list[str] = []
    tags.extend(problem.problem_tags)
    tags.extend(trace.tags)
    tags.extend(tag for tag in source_record.tags if tag.startswith("augment:"))
    tags.append("validation:misleading")
    tags.append(f"llm_provider:{llm_provider_config.provider}")
    model = llm_provider_config.model_kwargs.get("model")
    if model is not None:
        tags.append(f"llm_provider_model:{model}")
    if shared_tags:
        tags.extend(shared_tags)
    return list(dict.fromkeys(tags))


def filter_misleading_records(
    records: list[pyine.prompts.result_db.PromptResultRecord],
) -> tuple[list[pyine.prompts.result_db.PromptResultRecord], int]:
    """Keeps only records whose tags imply the misleading code type, returning filtered list and skip count."""
    filtered: list[pyine.prompts.result_db.PromptResultRecord] = []
    skipped = 0
    for rec in records:
        code_types = pyine.organisms.datamodules.samples.common.SampleCodeTypeSet.create_from_tags(rec.tags)
        if not code_types.has(pyine.organisms.datamodules.samples.common.SampleCodeType.misleading):
            skipped += 1
            continue
        filtered.append(rec)
    if skipped:
        logger.info(f"filtered out {skipped} non-misleading record(s)")
    return filtered, skipped


def filter_bugged_records(
    records: list[pyine.prompts.result_db.PromptResultRecord],
) -> tuple[list[pyine.prompts.result_db.PromptResultRecord], int]:
    """Removes records with bugged code types in tags, returning filtered list and skip count."""
    filtered: list[pyine.prompts.result_db.PromptResultRecord] = []
    skipped = 0
    for rec in records:
        code_types = pyine.organisms.datamodules.samples.common.SampleCodeTypeSet.create_from_tags(rec.tags)
        if code_types.has(pyine.organisms.datamodules.samples.common.SampleCodeType.bugged):
            skipped += 1
            continue
        filtered.append(rec)
    if skipped:
        logger.info(f"skipped {skipped} bugged record(s): trace.expected_output is not authoritative for bugged code")
    return filtered, skipped


def make_validator(
    meta_ref: dict[str, pydantic.JsonValue],
    tags_ref: list[str],
) -> pyine.prompts.result_db.ValidatorCallableType:
    """Creates a validator closure that parses LLM JSON output and injects verdict data.

    The closure mutates ``meta_ref`` and ``tags_ref`` in place. This works because
    ``fetch_or_generate_prompt_results`` reads ``meta`` and ``tags`` *after* the validator runs.
    This execution ordering dependency is pinned by a regression test in test_trace_annot_validator.py.
    """

    def _validate(
        result_str: str,
        output: typing.Any,
    ) -> str | None:
        try:
            payload = VerdictPayload.model_validate_json(pyine.utils.parsing.strip_markdown_fences(result_str))
        except pydantic.ValidationError:
            return None  # malformed or invalid verdict -> failed, must retry
        meta_ref["verdict"] = payload.verdict
        meta_ref["explanation"] = payload.explanation
        tags_ref.append(f"verdict:{payload.verdict.lower()}")
        return result_str

    return typing.cast("pyine.prompts.result_db.ValidatorCallableType", _validate)


def resolve_record(
    record: pyine.prompts.result_db.PromptResultRecord,
    dataset: pyine.data.traces.dataset_reader.DatasetProtocol,
    target_keys: set[str] | None,
    already_validated_uids: set[str],
    force_generation: bool,
) -> tuple[pyine.utils.code.execution.TraceResult, pyine.data.traces.dataset_utils.CodingProblem] | str:
    """Pre-checks and resolves trace/problem for a record on the main thread.

    Returns a (trace, problem) tuple if the record should be processed, or a skip-reason
    string if the record should be skipped. Dataset reads happen here (main thread) to avoid
    thread-safety issues with the DatasetReader's internal cache (see annotator.py for the
    same pattern).
    """
    if not force_generation and record.record_uid in already_validated_uids:
        return "skipped_validated"
    if target_keys is not None and record.identifier not in target_keys:
        return "skipped_target_indices"
    try:
        trace = dataset[record.identifier]
        problem = dataset.get_problem_data(record.identifier)
    except KeyError:
        logger.warning(
            f"trace key not found in dataset for record_uid={record.record_uid}, "
            f"identifier={record.identifier}; skipping"
        )
        return "error"
    return trace, problem


def process_one_validation(
    record: pyine.prompts.result_db.PromptResultRecord,
    trace: pyine.utils.code.execution.TraceResult,
    problem: pyine.data.traces.dataset_utils.CodingProblem,
    chain_config: pyine.prompts.types.PromptChainBuildConfig,
    llm_provider_config: pyine.utils.llm_providers.LLMProviderConfig,
    db: pyine.prompts.result_db.PromptResultDB,
    shared_tags_list: list[str] | None,
    shared_meta_dict: dict[str, typing.Any],
    max_unsatisfactory_retries: int,
    force_generation: bool,
    dry_run: bool,
) -> str | None:
    """Runs the LLM validation call for a single record. Returns the verdict string.

    Dataset reads must be done before calling this function (on the main thread)
    to avoid thread-safety issues with DatasetReader's internal cache.
    """
    # build mutable meta dict; apply shared_meta first, then lineage keys so they can't be clobbered
    meta: dict[str, typing.Any] = {}
    meta.update(shared_meta_dict)
    meta["source_record_uid"] = record.record_uid
    meta["source_prompt_name"] = record.prompt_name or ""
    meta["source_identifier"] = record.identifier
    # build mutable tags list (verdict tag appended by validator closure)
    tags = build_validation_tags(problem, trace, record, llm_provider_config, shared_tags_list)
    try:
        pyine.prompts.result_db.fetch_or_generate_prompt_results(
            identifier=record.record_uid,
            input_variables={
                "code": record.result,
                "inputs": str(trace.inputs),
                "expected_output": str(trace.expected_output),
            },
            prompt_chain_config=chain_config,
            db=db,
            output_validator=make_validator(meta, tags),
            max_unsatisfactory_retries=max_unsatisfactory_retries,
            generate_until_result_count=1,
            force_generation=force_generation,
            log_new_results=not dry_run,
            creation_meta=pyine.prompts.result_db.CreationMeta(
                provider=llm_provider_config.provider,
                llm_params=llm_provider_config.model_dump(),
            ),
            group=record.group,
            tags=tags,
            meta=meta,
        )
    except pyine.prompts.result_db.ValidationFailedError:
        logger.warning(
            f"validation failed after retries for record_uid={record.record_uid}, "
            f"identifier={record.identifier}; skipping"
        )
        return "error"
    # return the verdict injected by the validator closure
    verdict = meta.get("verdict", "")
    assert isinstance(verdict, str)
    return verdict
