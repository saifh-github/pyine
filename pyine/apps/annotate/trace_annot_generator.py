"""
Trace Annotation Generator CLI.

This application wraps the trace dataset annotation routine and lets you run prompt-chain-driven
annotations over a trace dataset with command-line controls.

Examples:

    Minimal example: run prompt "code_summary" with target word count "50" over all dataset traces:
    ```bash
        python pyine/apps/annotate/trace_annot_generator.py \
            --dataset /path/to/traces/dataset \
            --prompt-name code_summary \
            --prompt-vars '{"target_word_count": "50"}' \
            --llm-option provider=openai \
            --llm-option model=gpt-4o-mini
    ```

    Use the latest trace dataset available for a source dataset (e.g., "TACO"):
    ```bash
        python pyine/apps/annotate/trace_annot_generator.py \
            --dataset-latest-from TACO \
            --prompt-name hints/docs \
            --llm-option provider=openai \
            --llm-option model=gpt-5
    ```

    Use a YAML file for a static/shared LLM config and specify prompt variables directly inline:
    ```bash
        python pyine/apps/annotate/trace_annot_generator.py \
            --dataset /path/to/traces/dataset \
            --prompt-name code_summary \
            --prompt-vars '{"target_word_count": "100"}' \
            --llm-config-file ./configs/llm.openai.yaml
    ```

    Dry run (no logging to the DB), sequential processing, and no progress bar:

    ```bash
        python pyine/apps/annotate/trace_annot_generator.py \
            --dataset /path/to/traces/dataset \
            --prompt-name issues/iterators \
            --dry-run \
            --no-parallel \
            --no-progress
    ```

    Custom result DB path and specific worker count for prompt chain invocation concurrency:

    ```bash
        python pyine/apps/annotate/trace_annot_generator.py \
            --dataset /path/to/traces/dataset \
            --prompt-name issues/todos \
            --db-path ./outputs/prompt_results.sqlite \
            --max-workers 16
    ```

    Limit to a subset of indices, add shared tags and meta, and force generation:

    ```bash
        python pyine/apps/annotate/trace_annot_generator.py \
            --dataset /path/to/traces/dataset \
            --prompt-name hints/docs \
            --target-indices 0-99,150,200-205 \
            --shared-tags eval,exp1 \
            --shared-meta '{"run":"exp1","owner":"alice"}' \
            --force-generation
    ```

Notes:
- The dataset loader can be customized via --dataset-loader as a dotted path to a function or class
  that returns a DatasetReader instance when called with the dataset path. If not provided, the CLI
  will attempt to instantiate pyine.data.traces.dataset_reader.DatasetReader(dataset_path).
- Duration values support formats like "90m", "1h30m", "2d", or "3600s".
"""

import json
import logging
import pathlib
import typing

import click
import yaml

import pyine.data.traces.dataset_reader
import pyine.data.traces.dataset_utils
import pyine.data.utils.splits
import pyine.organisms.datamodules.utils.annotator as annotator
import pyine.prompts.types
import pyine.utils.llm_providers
import pyine.utils.portability
import pyine.utils.reprod

logger = logging.getLogger(__name__)


def _load_yaml_or_json_file(
    path: pathlib.Path,
) -> typing.Any:
    """Loads YAML or JSON content from a file."""
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return yaml.safe_load(text)


def _parse_yaml_or_json_value(
    value: str,
) -> typing.Any:
    """Parses a string as JSON, falling back to YAML, and then to raw string."""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return yaml.safe_load(value)
        except yaml.YAMLError:
            return value


def _parse_kv_list_to_dict(
    items: tuple[str, ...],
) -> dict[str, typing.Any]:
    """Parses repeated KEY=VALUE strings to a dictionary, decoding JSON/YAML values when possible."""
    result: dict[str, typing.Any] = {}
    for item in items:
        if "=" not in item:
            raise click.BadParameter(f"expected KEY=VALUE, got '{item}'")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        result[key] = _parse_yaml_or_json_value(raw_value)
    return result


def _build_dataset_reader(
    dataset_path: pathlib.Path,
    dataset_loader: str | None,
) -> typing.Any:
    """Instantiates a DatasetReader.

    If dataset_loader is provided, it must be a dotted path to either:
      - a class that can be instantiated with (dataset_path), or
      - a function that returns a reader when called as (dataset_path).

    If not provided, attempt to instantiate:
      pyine.data.traces.dataset_reader.DatasetReader(dataset_path)
    """
    if dataset_loader:
        try:
            loader = pyine.utils.portability.import_from_dotted_path(dataset_loader)
        except Exception as exc:
            raise click.ClickException(f"could not resolve dataset loader '{dataset_loader}': {exc}") from exc
        if not callable(loader):
            raise click.BadParameter(f"dotted path '{dataset_loader}' does not resolve to a callable")
        return loader(dataset_path)
    try:
        return pyine.data.traces.dataset_reader.DatasetReader(dataset_path)
    except Exception as exc:
        raise click.ClickException(f"Failed to instantiate dataset reader; original error: {exc}") from exc


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--dataset",
    "dataset_path",
    type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
    required=False,
    help="Path to a traces dataset directory. Alternatively, use --dataset-latest-from to select by source name.",
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
    help="Optional dotted path to a callable/class that returns a DatasetReader when called with (dataset_path).",
)
@click.option(
    "--prompt-name",
    type=str,
    required=True,
    help="Name of the prompt to build and run.",
)
@click.option(
    "--prompt-version",
    type=str,
    default=None,
    help="Optional prompt version identifier.",
)
@click.option(
    "--prompt-vars",
    type=str,
    default=None,
    help="Prompt template partial variables as JSON or YAML string.",
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
    "--target-split-file",
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    default=None,
    help="Optional path to a dataset split file (in case you wish to use --target-split-subset).",
)
@click.option(
    "--target-split-subset",
    type=str,
    default=None,
    help="When using --target-split-file, optionally specify which subset to use (e.g., 'train').",
)
@click.option(
    "--target-indices",
    type=str,
    default=None,
    help="Comma-separated indices and ranges (e.g., '0-99,150,200-205'). If omitted, process all.",
)
@click.option(
    "--base-filter-rule",
    type=str,
    default=None,
    help="Tag filter rule to skip items before prompting.",
)
@click.option(
    "--db-path",
    type=click.Path(dir_okay=False, path_type=pathlib.Path),
    default=None,
    help="Path to a PromptResultDB file. If omitted, use the framework default.",
)
@click.option(
    "--min-results-per-item",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Ensure at least this many results exist per item; otherwise generate.",
)
@click.option(
    "--max-result-age",
    type=str,
    default=None,
    help="Ignore preexisting results older than this duration (e.g., '1h30m', '2d').",
)
@click.option(
    "--record-tag-filter-rule",
    type=str,
    default=None,
    help="Tag filter rule applied when checking preexisting results.",
)
@click.option(
    "--deduplicate-results/--no-deduplicate-results",
    default=True,
    show_default=True,
    help="Whether to de-duplicate results before returning.",
)
@click.option(
    "--max-unsatisfactory-retries",
    type=click.IntRange(min=0),
    default=3,
    show_default=True,
    help="Maximum number of retries for an unsatisfactory result.",
)
@click.option(
    "--force-generation/--no-force-generation",
    default=False,
    show_default=True,
    help="Always generate new results (ignore preexisting ones).",
)
@click.option(
    "--shared-tags",
    type=str,
    default=None,
    help="Comma-separated list of shared tags to apply to all new records.",
)
@click.option(
    "--shared-meta",
    type=str,
    default=None,
    help="Shared metadata as JSON or YAML string.",
)
@click.option(
    "--shared-meta-file",
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    default=None,
    help="Path to YAML or JSON file containing shared metadata.",
)
@click.option(
    "--shuffle/--no-shuffle",
    default=False,
    show_default=True,
    help="Shuffle the order of items before processing to help improve coverage across large datasets.",
)
@click.option(
    "--parallel/--no-parallel",
    default=True,
    show_default=True,
    help="Process items concurrently via a thread pool.",
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
    default=5_000,
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
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    default="INFO",
    show_default=True,
    help="Logging verbosity.",
)
@click.option(
    "--log-file",
    type=click.Path(dir_okay=False, path_type=pathlib.Path),
    default=None,
    help="Optional path to a log file.",
)
def main(
    dataset_path: pathlib.Path | None,
    dataset_latest_from: str | None,
    dataset_loader: str | None,
    prompt_name: str,
    prompt_version: str | None,
    prompt_vars: str | None,
    llm_kv: tuple[str, ...],
    llm_config_file: pathlib.Path | None,
    target_split_file: pathlib.Path | None,
    target_split_subset: str | None,
    target_indices: str | None,
    base_filter_rule: str | None,
    db_path: pathlib.Path | None,
    min_results_per_item: int,
    max_result_age: str | None,
    record_tag_filter_rule: str | None,
    deduplicate_results: bool,
    max_unsatisfactory_retries: int,
    force_generation: bool,
    shared_tags: str | None,
    shared_meta: str | None,
    shared_meta_file: pathlib.Path | None,
    shuffle: bool,
    parallel: bool,
    max_workers: int | None,
    max_in_flight_jobs: int,
    show_progress: bool,
    dry_run: bool,
    log_level: str,
    log_file: pathlib.Path | None,
) -> None:
    """Entry point for the Trace Annotation Generator CLI."""
    numeric_log_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_log_level, int):
        raise click.BadParameter(f"invalid log level: {numeric_log_level}")
    pyine.utils.reprod.entrypoint_setup(log_level=numeric_log_level, log_path=log_file)
    for name in ("httpx", "httpcore"):
        # fix for the 'noisy' HTTP request POST messages in info level logs
        logger = logging.getLogger(name)
        logger.setLevel(logging.WARNING)
        logger.propagate = False
    logger.info("starting trace annotation generator")

    # resolve dataset path: exactly one of --dataset or --dataset-latest-from must be provided
    if (dataset_path is None) == (dataset_latest_from is None):
        raise click.BadParameter("exactly one of --dataset or --dataset-latest-from must be provided")
    if dataset_latest_from is not None:
        try:
            resolved_path = pyine.data.traces.dataset_utils.get_latest_dataset_path(dataset_latest_from)
        except Exception as exc:
            raise click.ClickException(
                f"could not resolve latest trace dataset for source '{dataset_latest_from}': {exc}"
            ) from exc
        effective_dataset_path = resolved_path
        logger.info("resolved latest dataset for '%s' to: %s", dataset_latest_from, effective_dataset_path)
    else:
        effective_dataset_path = typing.cast(pathlib.Path, dataset_path)
    dataset = _build_dataset_reader(effective_dataset_path, dataset_loader)

    partial_vars: dict[str, typing.Any] = {}
    if prompt_vars:
        inline_vars = _parse_yaml_or_json_value(prompt_vars)
        if inline_vars is None:
            inline_vars = {}
        if not isinstance(inline_vars, dict):
            raise click.BadParameter("--prompt-vars must decode to a mapping/dict")
        partial_vars.update(inline_vars)
        logger.debug(f"parsed prompt vars: {partial_vars}")

    shared_meta_dict: dict[str, typing.Any] = {}
    if shared_meta_file is not None:
        file_meta = _load_yaml_or_json_file(shared_meta_file)
        if file_meta is None:
            file_meta = {}
        if not isinstance(file_meta, dict):
            raise click.BadParameter("--shared-meta-file must decode to a mapping/dict")
        shared_meta_dict.update(file_meta)
    if shared_meta:
        inline_meta = _parse_yaml_or_json_value(shared_meta)
        if inline_meta is None:
            inline_meta = {}
        if not isinstance(inline_meta, dict):
            raise click.BadParameter("--shared-meta must decode to a mapping/dict")
        shared_meta_dict.update(inline_meta)
    if shared_meta_dict:
        logger.debug(f"parsed shared meta: {shared_meta_dict}")

    shared_tags_list: list[str] | None = None
    if shared_tags:
        shared_tags_list = [t.strip() for t in shared_tags.split(",") if t.strip()]
        logger.debug(f"parsed shared tags: {shared_tags_list}")

    llm_kwargs: dict[str, typing.Any] = {}
    if llm_config_file is not None:
        file_cfg = _load_yaml_or_json_file(llm_config_file) or {}
        if not isinstance(file_cfg, dict):
            raise click.BadParameter("--llm-config-file must decode to a mapping/dict")
        llm_kwargs.update(file_cfg)
    if llm_kv:
        llm_kwargs.update(_parse_kv_list_to_dict(llm_kv))
        logger.debug(f"parsed LLM options: {llm_kwargs}")
    try:
        llm_provider_config = pyine.utils.llm_providers.LLMProviderConfig.from_dict(llm_kwargs)
    except Exception as exc:
        raise click.BadParameter("llm options resulted in an invalid provider config") from exc

    indices_list: list[int] | None = None
    if target_split_file or target_split_subset:
        if not target_split_file or not target_split_subset:
            raise click.BadParameter("--target-split-file and --target-split-subset must be provided together")
        split_data = pyine.data.utils.splits.get_dataset_split_result(target_split_file)
        subsets_to_problem_ids = split_data.get_subset_to_ids_map()
        if target_split_subset not in split_data.subset_assignments:
            raise click.BadParameter(
                f"invalid target split subset ({target_split_subset}), "
                f"available ones are: {list(subsets_to_problem_ids.keys())}"
            )
        problem_ids = subsets_to_problem_ids[target_split_subset]
        if not isinstance(dataset, pyine.data.traces.dataset_reader.DatasetReader):
            raise NotImplementedError("cannot use target split ids with non-standard traces datasets")
        indices_list = [
            trace_idx for trace_idx in range(len(dataset)) if dataset.problem_keys[trace_idx] in problem_ids
        ]
        if not indices_list:
            raise click.BadParameter(
                f"no traces found in target split subset '{target_split_subset}' " f"for problem ids: {problem_ids}"
            )
        logger.info(f"found {len(indices_list)} indices for target split subset '{target_split_subset}'")
    if target_indices:
        try:
            target_indices = pyine.utils.portability.parse_indices_spec(target_indices)
        except ValueError as exc:
            raise click.BadParameter(f"invalid target indices spec: {exc}") from exc
        for idx in target_indices:
            if idx < 0 or idx >= len(dataset):
                raise click.BadParameter(f"invalid target index: {idx}")
        if indices_list is None:
            indices_list = target_indices
        else:
            indices_list = [idx for idx in indices_list if idx in target_indices]
    logger.debug(f"target indices: {indices_list}")

    try:
        max_age_td = pyine.utils.portability.parse_duration_to_timedelta(max_result_age)
    except ValueError as exc:
        raise click.BadParameter(f"invalid max result age spec: {exc}") from exc
    if max_age_td is not None:
        logger.debug(f"parsed max result age: {max_age_td}")

    prompt_config = pyine.prompts.types.PromptBuildConfig(
        prompt_name=prompt_name,
        version=prompt_version,
        partial_vars=partial_vars,
    )
    logger.debug(f"parsed prompt config: {prompt_config}")
    options = annotator.AnnotationOptions(
        llm_provider_config=llm_provider_config,
        prompt_config=prompt_config,
        target_indices=indices_list,
        base_filter_rule=base_filter_rule,
        db_path=db_path,
        min_results_per_item=min_results_per_item,
        max_result_age=max_age_td,
        record_tag_filter_rule=record_tag_filter_rule,
        deduplicate_results=deduplicate_results,
        max_unsatisfactory_retries=max_unsatisfactory_retries,
        force_generation=force_generation,
        shared_tags=shared_tags_list,
        shared_meta=typing.cast(dict[str, typing.Any] | None, shared_meta_dict or None),
    )
    logger.info(
        "running annotation with prompt '%s'%s; parallel=%s, max_workers=%s, dry_run=%s",
        prompt_name,
        f" (v={prompt_version})" if prompt_version else "",
        parallel,
        max_workers,
        dry_run,
    )
    report = annotator.annotate_trace_dataset(
        dataset=dataset,
        config=options,
        show_progress=show_progress,
        dry_run=dry_run,
        shuffle_indices=shuffle,
        parallel=parallel,
        max_workers=max_workers,
        max_in_flight_jobs=max_in_flight_jobs,
        verbose=True,
    )
    logger.info("annotation completed: %s", report.summary())


if __name__ == "__main__":
    main()
