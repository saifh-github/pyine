"""
Trace Annotation Validator CLI (currently, only for misleading hint quality check).

This application queries the prompt result DB for misleading-tagged annotation records, loads
corresponding traces for ground truth, uses an LLM prompt to assess whether each hint actually
misleads, and stores validation verdicts as new records in the same DB.

Examples:

    Minimal dry-run over a small subset:
    ```bash
        python -m pyine.apps.annotate.trace_annot_validator \
            --dataset /path/to/traces/dataset \
            --llm-option provider=openai \
            --llm-option model=gpt-4o-mini \
            --target-indices 0-5 \
            --dry-run
    ```

    Full validation run on all misleading records:
    ```bash
        python -m pyine.apps.annotate.trace_annot_validator \
            --dataset-latest-from TACO \
            --llm-option provider=openai \
            --llm-option model=gpt-4o-mini
    ```

    Validate only records from a specific source prompt:
    ```bash
        python -m pyine.apps.annotate.trace_annot_validator \
            --dataset /path/to/traces/dataset \
            --source-prompt-name issues/docs \
            --llm-option provider=openai \
            --llm-option model=gpt-4o
    ```

    Parallel validation with concurrency controls:
    ```bash
        python -m pyine.apps.annotate.trace_annot_validator \
            --dataset-latest-from TACO \
            --llm-option provider=openai \
            --llm-option model=gpt-4o-mini \
            --parallel --max-workers 8 --max-in-flight-jobs 16
    ```

Notes:
- Records with "augment:bugged" tags are automatically excluded since trace.expected_output
  is not authoritative for bugged code.
- Duration values support formats like "90m", "1h30m", "2d", or "3600s".
"""

import concurrent.futures
import logging
import pathlib
import random
import typing

import click
import tqdm

import pyine.apps.annotate._shared as _shared
import pyine.organisms.datamodules.utils.validator as _validator_utils
import pyine.prompts.names
import pyine.prompts.result_db
import pyine.prompts.types
import pyine.utils.concurrency
import pyine.utils.portability
import pyine.utils.reprod

logger = logging.getLogger(__name__)

VALIDATION_PROMPT_NAME: typing.Final = pyine.prompts.names.PromptNames.VALIDATION_MISLEADING
"""Prompt name used for storing validation records."""


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--dataset",
    "dataset_paths",
    type=str,
    multiple=True,
    required=False,
    help=(
        "Path to a traces dataset directory. Repeat this option to combine dataset shards, or use "
        "a glob pattern to select multiple dataset shards. Alternatively, use `--dataset-latest-from` "
        "to select by source dataset name."
    ),
)
@click.option(
    "--dataset-latest-from",
    "dataset_latest_from",
    type=str,
    default=None,
    help="Source dataset name (e.g., 'TACO') whose latest trace dataset should be used.",
)
@click.option(
    "--dataset-loader",
    type=str,
    default=None,
    help=(
        "Optional dotted path to a class that returns a DatasetReader when called with a dataset path "
        "(or with a list of paths when --dataset is provided multiple times or as a glob pattern)."
    ),
)
@click.option(
    "--source-prompt-name",
    "source_prompt_names",
    type=str,
    multiple=True,
    default=("issues/docs", "issues/docs_v2", "hints/docs"),
    show_default=True,
    help="Source prompt name(s) whose misleading records should be validated. Repeat for multiple.",
)
@click.option(
    "--source-prompt-version",
    type=str,
    default=None,
    help="Optional prompt version filter for source records.",
)
@click.option(
    "--source-max-result-age",
    type=str,
    default=None,
    help="Ignore source results older than this duration (e.g., '1h30m', '2d').",
)
@click.option(
    "--llm-option",
    "llm_kv",
    type=str,
    multiple=True,
    help="LLM provider option as KEY=VALUE (repeat for multiple). Values can be JSON/YAML (e.g., 'temperature=0.2').",
)
@click.option(
    "--llm-config-file",
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    default=None,
    help="Path to YAML or JSON file with LLM provider options.",
)
@click.option(
    "--db-path",
    type=click.Path(dir_okay=False, path_type=pathlib.Path),
    default=None,
    help="Path to a PromptResultDB file. If omitted, use the framework default.",
)
@click.option(
    "--max-unsatisfactory-retries",
    type=click.IntRange(min=0),
    default=5,
    show_default=True,
    help="Maximum number of retries for a malformed LLM response.",
)
@click.option(
    "--force-generation/--no-force-generation",
    default=False,
    show_default=True,
    help="Re-validate even if a validation record already exists.",
)
@click.option(
    "--target-indices",
    type=str,
    default=None,
    help="Comma-separated indices and ranges (e.g., '0-99,150,200-205'). Filters source records by trace key.",
)
@click.option(
    "--shared-tags",
    type=str,
    default=None,
    help="Comma-separated list of shared tags to apply to all validation records.",
)
@click.option(
    "--shared-meta",
    type=str,
    default=None,
    help=(
        "Shared metadata as JSON or YAML string. Note: keys 'source_record_uid', "
        "'source_prompt_name', and 'source_identifier' are reserved for lineage and cannot be overridden."
    ),
)
@click.option(
    "--shuffle/--no-shuffle",
    default=False,
    show_default=True,
    help="Shuffle the order of source records before processing.",
)
@click.option(
    "--parallel/--no-parallel",
    default=True,
    show_default=True,
    help="Process records concurrently via a thread pool.",
)
@click.option(
    "--max-workers",
    type=int,
    default=None,
    help="Maximum number of worker threads when --parallel is enabled.",
)
@click.option(
    "--max-in-flight-jobs",
    type=int,
    default=32,
    show_default=True,
    help="Maximum number of in-flight jobs when --parallel is enabled.",
)
@click.option(
    "--progress/--no-progress",
    "show_progress",
    default=True,
    show_default=True,
    help="Enable/disable progress bar.",
)
@click.option(
    "--dry-run/--no-dry-run",
    default=False,
    show_default=True,
    help="If enabled, do not log new results; only report what would happen.",
)
@_shared.async_main_wrapper
async def main(
    dataset_paths: tuple[str, ...],
    dataset_latest_from: str | None,
    dataset_loader: str | None,
    source_prompt_names: tuple[str, ...],
    source_prompt_version: str | None,
    source_max_result_age: str | None,
    llm_kv: tuple[str, ...],
    llm_config_file: pathlib.Path | None,
    db_path: pathlib.Path | None,
    max_unsatisfactory_retries: int,
    force_generation: bool,
    target_indices: str | None,
    shared_tags: str | None,
    shared_meta: str | None,
    shuffle: bool,
    parallel: bool,
    max_workers: int | None,
    max_in_flight_jobs: int,
    show_progress: bool,
    dry_run: bool,
) -> None:
    """Entry point for the Trace Annotation Validator CLI."""
    pyine.utils.reprod.entrypoint_setup()
    logger.info("starting trace annotation validator")

    # -------- prepare dataset-related stuff --------

    effective_dataset_paths = _shared.resolve_dataset_paths(dataset_paths, dataset_latest_from)
    dataset = _shared.build_dataset_reader(effective_dataset_paths, dataset_loader)
    logger.info(
        f"using {len(effective_dataset_paths)} dataset part(s): "
        f"{', '.join(str(p) for p in effective_dataset_paths)} (total traces: {len(dataset):,})"
    )

    # -------- resolve target indices into a key set --------

    target_keys: set[str] | None = None
    if target_indices is not None:
        try:
            parsed_target_indices = pyine.utils.portability.parse_indices_spec(target_indices)
        except ValueError as exc:
            raise click.BadParameter(f"invalid target indices spec: {exc}") from exc
        for idx in parsed_target_indices:
            if idx < 0 or idx >= len(dataset):
                raise click.BadParameter(f"invalid target index: {idx}")
        target_keys = {dataset.trace_keys[idx] for idx in parsed_target_indices}
        logger.info(f"target indices resolved to {len(target_keys)} trace key(s)")

    # -------- prepare provider/llm stuff --------

    llm_provider_config = _shared.build_llm_provider_config(llm_kv, llm_config_file)

    # -------- prepare shared tags/meta --------

    shared_tags_list: list[str] | None = None
    if shared_tags:
        shared_tags_list = [tag.strip() for tag in shared_tags.split(",") if tag.strip()]
        logger.debug(f"parsed shared tags: {shared_tags_list}")

    shared_meta_dict: dict[str, typing.Any] = {}
    if shared_meta:
        inline_meta = _shared.ensure_mapping_dict(
            _shared.parse_yaml_or_json_value(shared_meta),
            "--shared-meta must decode to a mapping/dict",
        )
        # warn and drop keys that would clobber lineage fields
        for reserved_key in _validator_utils.LINEAGE_META_KEYS & inline_meta.keys():
            logger.warning(f"--shared-meta key '{reserved_key}' is reserved for lineage and will be ignored")
        shared_meta_dict.update(
            {key: value for key, value in inline_meta.items() if key not in _validator_utils.LINEAGE_META_KEYS}
        )

    # -------- prepare prompt chain config --------

    prompt_config = pyine.prompts.types.PromptBuildConfig(prompt_name=VALIDATION_PROMPT_NAME)
    chain_config = pyine.prompts.types.PromptChainBuildConfig(
        prompt=prompt_config,
        provider=llm_provider_config,
    )

    # -------- connect to DB and fetch source records --------

    if db_path:
        logger.info(f"connecting to existing PromptResultDB at '{db_path}'")
        db = pyine.prompts.result_db.PromptResultDB(db_path)
    else:
        logger.info("connecting to default PromptResultDB")
        db = pyine.prompts.result_db.get_framework_db()
    try:
        source_max_age_td = pyine.utils.portability.parse_duration_to_timedelta(source_max_result_age)
    except ValueError as exc:
        raise click.BadParameter(f"invalid source max result age spec: {exc}") from exc
    source_records: list[pyine.prompts.result_db.PromptResultRecord] = []
    for prompt_name in source_prompt_names:
        logger.info(f"fetching source records for prompt '{prompt_name}'...")
        source_records.extend(
            db.get_by_prompt_name(
                prompt_name,
                prompt_version=source_prompt_version,
                max_result_age=source_max_age_td,
            )
        )
    logger.info(f"fetched {len(source_records)} source record(s) from {len(source_prompt_names)} prompt name(s)")

    # -------- keep only misleading records, then exclude bugged --------

    source_records, skipped_not_misleading = _validator_utils.filter_misleading_records(source_records)
    source_records, skipped_bugged = _validator_utils.filter_bugged_records(source_records)

    # -------- batch skip-already-validated lookup --------

    all_record_uids = [rec.record_uid for rec in source_records]
    logger.info(f"will validate up to {len(all_record_uids)} records, depending on already-validated ones")
    existing_validations = db.get_by_identifiers(all_record_uids, prompt_name=VALIDATION_PROMPT_NAME)
    already_validated_uids = {uid for uid, recs in existing_validations.items() if recs}
    logger.info(f"found {len(already_validated_uids)} already-validated record(s)")

    # -------- process source records --------

    if shuffle:
        random.shuffle(source_records)
    stats: dict[str, int] = {
        "total": len(source_records),
        "misleading": 0,
        "not_misleading": 0,
        "uninformative": 0,
        "skipped_validated": 0,
        "skipped_not_misleading": skipped_not_misleading,
        "skipped_bugged": skipped_bugged,
        "skipped_target_indices": 0,
        "errors": 0,
    }
    logger.info(
        f"running validation with prompt '{VALIDATION_PROMPT_NAME}'; "
        f"parallel={parallel}, max_workers={max_workers}, dry_run={dry_run}"
    )

    log_interval = max(1, len(source_records) // 20)  # ~5% increments
    last_logged_processed = 0

    def _log_progress() -> None:
        nonlocal last_logged_processed
        processed = stats["misleading"] + stats["not_misleading"] + stats["uninformative"] + stats["errors"]
        if processed - last_logged_processed < log_interval:
            return
        last_logged_processed = processed
        total = stats["total"]
        misleading = stats["misleading"]
        not_misleading = stats["not_misleading"]
        uninformative = stats["uninformative"]
        errors = stats["errors"]
        evaluated = misleading + not_misleading + uninformative
        misleading_pct = (misleading / evaluated * 100) if evaluated else 0.0
        logger.info(
            f"progress: {processed}/{total} processed "
            f"({misleading} misleading, {not_misleading} not_misleading, "
            f"{uninformative} uninformative, {errors} errors); "
            f"misleading rate: {misleading_pct:.1f}%"
        )

    def _update_stats(verdict_or_status: str | None) -> None:
        if verdict_or_status == "MISLEADING":
            stats["misleading"] += 1
        elif verdict_or_status == "NOT_MISLEADING":
            stats["not_misleading"] += 1
        elif verdict_or_status == "UNINFORMATIVE":
            stats["uninformative"] += 1
        elif verdict_or_status == "skipped_validated":
            stats["skipped_validated"] += 1
        elif verdict_or_status == "skipped_target_indices":
            stats["skipped_target_indices"] += 1
        else:
            stats["errors"] += 1
        _log_progress()

    if not parallel:
        wrapped_records: typing.Iterable[pyine.prompts.result_db.PromptResultRecord] = source_records
        if show_progress:
            wrapped_records = tqdm.tqdm(source_records, desc="validating", unit="rec")
        for record in wrapped_records:
            resolved = _validator_utils.resolve_record(
                record, dataset, target_keys, already_validated_uids, force_generation
            )
            if isinstance(resolved, str):
                _update_stats(resolved)
                continue
            trace, problem = resolved
            verdict_or_status = _validator_utils.process_one_validation(
                record=record,
                trace=trace,
                problem=problem,
                chain_config=chain_config,
                llm_provider_config=llm_provider_config,
                db=db,
                shared_tags_list=shared_tags_list,
                shared_meta_dict=shared_meta_dict,
                max_unsatisfactory_retries=max_unsatisfactory_retries,
                force_generation=force_generation,
                dry_run=dry_run,
            )
            _update_stats(verdict_or_status)
    else:
        # use integer indices as input items (Hashable), same pattern as the annotator app;
        # dataset reads happen in _submit_one (main thread) to avoid thread-safety issues
        # with DatasetReader's internal OrderedDict cache
        progress_bar: tqdm.tqdm[typing.NoReturn] | None = None
        if show_progress:
            progress_bar = tqdm.tqdm(total=len(source_records), desc="validating", unit="rec")

        def _submit_one(
            record_idx: typing.Hashable,
            executor: concurrent.futures.Executor,
        ) -> concurrent.futures.Future[str | None] | None:
            record = source_records[typing.cast("int", record_idx)]
            # resolve trace/problem on the main thread (thread-safe dataset access)
            resolved = _validator_utils.resolve_record(
                record, dataset, target_keys, already_validated_uids, force_generation
            )
            if isinstance(resolved, str):
                _update_stats(resolved)
                return None  # skipped; run_with_sliding_window handles None returns
            trace, problem = resolved
            return executor.submit(
                _validator_utils.process_one_validation,
                record=record,
                trace=trace,
                problem=problem,
                chain_config=chain_config,
                llm_provider_config=llm_provider_config,
                db=db,
                shared_tags_list=shared_tags_list,
                shared_meta_dict=shared_meta_dict,
                max_unsatisfactory_retries=max_unsatisfactory_retries,
                force_generation=force_generation,
                dry_run=dry_run,
            )

        def _process_result(
            record_idx: typing.Hashable,
            verdict_or_status: str | None,
        ) -> None:
            if verdict_or_status is None:
                return  # skipped items already counted in _submit_one
            _update_stats(verdict_or_status)

        def _progress_callback(
            in_flight: list[typing.Hashable],
            completed: list[typing.Hashable],
        ) -> None:
            if progress_bar is not None:
                progress_bar.n = len(completed)
                progress_bar.refresh()

        await pyine.utils.concurrency.run_with_sliding_window(
            input_items=range(len(source_records)),
            submit_one=_submit_one,
            process_result=_process_result,
            progress_callback=_progress_callback,
            max_workers=max_workers,
            max_in_flight_jobs=max_in_flight_jobs,
        )
        if progress_bar is not None:
            progress_bar.close()

    # -------- summary report --------

    logger.info(
        "validation completed: total=%d, misleading=%d, not_misleading=%d, uninformative=%d, "
        "skipped_validated=%d, skipped_not_misleading=%d, skipped_bugged=%d, "
        "skipped_target_indices=%d, errors=%d",
        stats["total"],
        stats["misleading"],
        stats["not_misleading"],
        stats["uninformative"],
        stats["skipped_validated"],
        stats["skipped_not_misleading"],
        stats["skipped_bugged"],
        stats["skipped_target_indices"],
        stats["errors"],
    )


if __name__ == "__main__":
    main()
