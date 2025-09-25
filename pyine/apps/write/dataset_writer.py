"""
Dataset Writer CLI.

This application lets you:
- Write a traces dataset from a supported source using `pyine.data.traces.dataset_writer.write_dataset`.
- Write a deltas dataset from an existing traces dataset using `pyine.data.deltas.dataset_writer.write_dataset`.

Examples:

    Write a traces dataset from the latest repackaged TACO dataset with caps and verbose logging:
    ```bash
        python -m pyine.apps.write.dataset_writer traces \
            --dataset-name TACO
            --output-dataset-path /tmp/traces.lmdb \
            --max-output-traces 1000 \
            --max-solutions-per-problem 10 \
            --max-tests-per-solution 10 \
            --max-trace-var-repr-length 10000 \
            --max-trace-valid-events 20000 \
            --verbose
    ```

    Write a deltas dataset from an existing traces dataset:
    ```bash
        python -m pyine.apps.write.dataset_writer deltas \
            --traces-dataset /data/my_traces.lmdb \
            --output-dataset-path /data/my_deltas.lmdb \
            --verbose
    ```


Notes:
- If `--dataset-path` is omitted for the traces command group, the latest dataset is used.
- If `--output-dataset-path` is omitted, a default location is computed; if no explicit tag is given,
  one is derived from the configuration hash.
- For the partition command group, the split files should be those created by the
  `pyine.apps.split.dataset_splitter` CLI app.
- Pass `--force` to overwrite an existing output without an interactive confirmation.
"""

import logging
import pathlib
import typing

import click

import pyine.data.deltas.dataset_utils as delta_utils
import pyine.data.deltas.dataset_writer as delta_writer
import pyine.data.traces.dataset_utils as trace_utils
import pyine.data.traces.dataset_writer as trace_writer
import pyine.data.utils.lmdb_io
import pyine.prompts
import pyine.utils.filesystem
import pyine.utils.reprod

logger = logging.getLogger(__name__)


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def main() -> None:
    """Dataset writer CLI (traces, deltas)."""
    pass


# ------------------------------ TRACES ------------------------------


@main.command("traces")
@click.option(
    "--dataset-name",
    "dataset_name",
    type=str,
    required=True,
    help="Name of the source dataset containing code snippets that we will trace.",
)
@click.option(
    "--dataset-path",
    "dataset_path",
    type=click.Path(exists=True, path_type=pathlib.Path),
    required=False,
    help=(
        "Path to a source dataset containing code snippets. If the name already points to a known dataset, "
        "this path can be omitted, in which case we will go fetch the latest version of the dataset."
    ),
)
@click.option(
    "--output-path",
    "output_path",
    type=click.Path(path_type=pathlib.Path),
    required=False,
    help=(
        "Output path for the LMDB dataset of trace results. If omitted, a default location will be "
        "computed based on the source dataset name, the config hash, and the current date."
    ),
)
@click.option(
    "--output-tag",
    "output_tag",
    type=str,
    default=None,
    help="Optional tag used for naming when the output path is not provided (defaults to config hash).",
)
@click.option(
    "--max-output-traces",
    "max_output_traces",
    type=int,
    default=None,
    show_default=True,
    help="Maximum number of traces to write in total (soft cap, checked once per problem). If None, no maximum.",
)
@click.option(
    "--max-solutions-per-problem",
    "max_solutions_per_problem",
    type=int,
    default=None,
    show_default=True,
    help="Maximum number of solutions per problem (hard cap, may discard picked candidates, None = no max).",
)
@click.option(
    "--max-tests-per-solution",
    "max_tests_per_solution",
    type=int,
    default=None,
    show_default=True,
    help="Maximum number of tests per solution (hard cap, may discard picked candidates, None = no max).",
)
@click.option(
    "--max-trace-events-per-line",
    "max_trace_events_per_line",
    type=int,
    default=None,
    show_default=True,
    help="Maximum number of trace events per solution code line (None = no max).",
)
@click.option(
    "--max-trace-var-repr-length",
    "max_trace_var_repr_length",
    type=int,
    default=None,
    show_default=True,
    help="Maximum length of variable representation strings, in characters (None = no max).",
)
@click.option(
    "--max-trace-valid-events",
    "max_trace_valid_events",
    type=int,
    default=None,
    show_default=True,
    help="Maximum number of (valid, in-scope) events allowed per trace (None = no max).",
)
@click.option(
    "--max-trace-results-blob-size",
    "max_trace_results_blob_size",
    type=int,
    default=2 * (1024**3),  # 2GB by default
    show_default=True,
    help="Maximum size of trace result blobs, in bytes.",
)
@click.option(
    "--min-solution-line-count",
    "min_solution_line_count",
    type=int,
    default=1,
    show_default=True,
    help="Minimum number of lines required in a solution code string.",
)
@click.option(
    "--min-solution-dissimilarity",
    "min_solution_dissimilarity",
    type=float,
    default=0.1,
    show_default=True,
    help="Minimum dissimilarity between solutions (for clustering and deduplication; in [0,1]).",
)
@click.option(
    "--execution-timeout-seconds",
    "execution_timeout_seconds",
    type=float,
    default=10.0,
    show_default=True,
    help="Timeout in seconds for each execution attempt. If exceeded, solution is skipped.",
)
@click.option(
    "--target-problem-pattern",
    "target_problem_pattern",
    type=str,
    default=None,
    help="Regular expression pattern to use for filtering problem file names. If None, no filtering.",
)
@click.option(
    "--target-problem-ids",
    "target_problem_ids",
    type=str,
    default=None,
    help=(
        "List of comma-separated problem IDs to target, or path to a file containing such IDs. "
        "If None, no filtering, i.e. all problems are targeted."
    ),
)
@click.option(
    "--reformat-code-strings/--no-reformat-code-strings",
    "reformat_code_strings",
    default=False,
    show_default=True,
    help="Whether to reformat code strings to be standardized (with black).",
)
@click.option(
    "--generate-obfuscated-solutions/--no-generate-obfuscated-solutions",
    default=False,
    show_default=True,
    help="Specifies whether to generate an obfuscated (yet still documented) version of each solution.",
)
@click.option(
    "--fetch-augmented-solutions",
    "fetch_augmented_solutions",
    type=str,
    multiple=True,
    help="Specifies the (max) number of augmented solutions to fetch from the db for a specific prompt name.",
)
@click.option(
    "--prompt-result-db-path",
    "prompt_result_db_path",
    type=click.Path(dir_okay=False, path_type=pathlib.Path),
    default=None,
    help="Path to a PromptResultDB file. If omitted, use the framework default.",
)
@click.option(
    "--verbose/--no-verbose",
    "verbose",
    default=True,
    show_default=True,
    help="Toggles verbose output/logging.",
)
@click.option(
    "--force",
    "force",
    is_flag=True,
    default=False,
    help="Overwrite the output path if it already exists.",
)
@click.option(
    "--dry-run",
    "dry_run",
    is_flag=True,
    default=False,
    help="If set, only prints the resolved configuration and exits.",
)
def traces(
    dataset_name: str,
    dataset_path: pathlib.Path | None,
    output_path: pathlib.Path | None,
    output_tag: str | None,
    max_output_traces: int | None,
    max_solutions_per_problem: int | None,
    max_tests_per_solution: int | None,
    max_trace_events_per_line: int | None,
    max_trace_var_repr_length: int | None,
    max_trace_valid_events: int | None,
    max_trace_results_blob_size: int | None,
    min_solution_line_count: int | None,
    min_solution_dissimilarity: float | None,
    execution_timeout_seconds: float | None,
    target_problem_pattern: str | None,
    target_problem_ids: str | None,
    reformat_code_strings: bool,
    generate_obfuscated_solutions: bool,
    fetch_augmented_solutions: typing.Sequence[str],
    prompt_result_db_path: pathlib.Path | None,
    verbose: bool,
    force: bool,
    dry_run: bool,
) -> None:
    """Write a traces dataset from a specified source dataset according to the given configuration."""
    pyine.utils.reprod.entrypoint_setup()
    fetch_augmented_solutions_dict = {}
    for tupl_str in fetch_augmented_solutions:
        if "=" not in tupl_str or tupl_str.count("=") != 1:
            raise click.BadParameter(f"invalid augmented solution fetch tuple: {tupl_str}")
        prompt_name, fetch_count = tupl_str.split("=")
        if prompt_name not in pyine.prompts.get_framework_prompt_manager().list_prompts():
            raise click.BadParameter(f"invalid prompt name: {prompt_name}")
        try:
            _ = int(fetch_count)
        except ValueError:
            raise click.BadParameter(f"invalid augmented solution fetch tuple: {tupl_str}")
        fetch_augmented_solutions_dict[prompt_name] = int(fetch_count)
    default_serialization_config = dict(
        method=pyine.data.utils.lmdb_io.SerializationMethod.JSON_ZSTD,
        compression_kwargs=dict(level=3),
    )
    default_failed_test_log_dir = pyine.utils.filesystem.get_logs_root_path() / "traced-test-failures"
    if target_problem_ids is not None and "," in target_problem_ids:
        target_problem_ids = target_problem_ids.split(",")
    config = trace_writer.TraceDatasetWriterConfig(
        source_dataset_name=dataset_name,
        max_output_traces=max_output_traces,
        max_solutions_per_problem=max_solutions_per_problem,
        max_tests_per_solution=max_tests_per_solution,
        max_trace_events_per_line=max_trace_events_per_line,
        max_trace_var_repr_length=max_trace_var_repr_length,
        max_trace_valid_events=max_trace_valid_events,
        max_trace_results_blob_size=max_trace_results_blob_size,
        min_solution_line_count=min_solution_line_count,
        min_solution_dissimilarity=min_solution_dissimilarity,
        execution_timeout_seconds=execution_timeout_seconds,
        target_problem_pattern=target_problem_pattern,
        target_problem_ids=target_problem_ids,
        reformat_code_strings=reformat_code_strings,
        generate_obfuscated_solutions=generate_obfuscated_solutions,
        fetch_augmented_solutions=fetch_augmented_solutions_dict,
        prompt_result_db_path=prompt_result_db_path,
        writer_serialization_config=default_serialization_config,
        failed_test_log_dir=default_failed_test_log_dir,
    )
    if dataset_name == "TACO":
        if dataset_path is None:
            from pyine.data.taco import dataset_utils as taco_dataset_utils

            dataset_path = taco_dataset_utils.get_latest_repackaged_dataset_path()
    # elif dataset_name == "...": ...
    else:
        if dataset_path is None:
            raise NotImplementedError(f"missing dataset module support for: {dataset_name}")
    if output_path is None:
        if output_tag is None:
            output_tag = config.get_short_hash()
        elif "{config_hash}" in output_tag:
            output_tag = output_tag.replace("{config_hash}", config.get_short_hash())
        output_path = pyine.data.traces.dataset_utils.get_new_dataset_path(
            source_dataset_name=dataset_name,
            dataset_name_tag=output_tag,
        )
    if dry_run:
        click.echo("[dry-run] would call write_dataset with:")
        click.echo(f"  root_dataset_path = {dataset_path}")
        click.echo(f"  output_dataset_path = {output_path}")
        click.echo(f"  config  = {config.model_dump()}")
        click.echo(f"  verbose = {verbose}")
        click.echo(f"  force   = {force}")
        return
    trace_writer.write_dataset(
        root_dataset_path=dataset_path,
        output_dataset_path=output_path,
        config=config,
        verbose=verbose,
        force_overwrite=force,
    )
    logger.info("all done")


# ------------------------------ DELTAS ------------------------------


@main.command("deltas")
@click.option(
    "--traces-dataset",
    "traces_dataset",
    type=str,
    required=True,
    help="Name or path of the traces dataset to read from.",
)
@click.option(
    "--output-path",
    "output_path",
    type=click.Path(path_type=pathlib.Path),
    required=True,
    help="Output path for the LMDB delta dataset.",
)
@click.option(
    "--delta-generator",
    "delta_generator",
    type=click.Choice([s.value for s in delta_utils.DeltaGeneratorType], case_sensitive=False),  # noqa
    default=delta_utils.DeltaGeneratorType.SIMPLE.value,
    show_default=True,
)
@click.option(
    "--verbose/--no-verbose",
    "verbose",
    default=True,
    show_default=True,
    help="Toggles verbose output/logging.",
)
@click.option(
    "--force",
    "force",
    is_flag=True,
    default=False,
    help="Overwrite the output path if it already exists.",
)
@click.option(
    "--dry-run",
    "dry_run",
    is_flag=True,
    default=False,
    help="If set, only prints the resolved configuration and exits.",
)
def deltas_from_traces(
    traces_dataset: str,
    output_path: pathlib.Path,
    delta_generator: str,
    verbose: bool,
    force: bool,
    dry_run: bool,
) -> None:
    """Write a deltas dataset from an existing traces dataset."""
    pyine.utils.reprod.entrypoint_setup()
    delta_generator = delta_utils.DeltaGeneratorType.from_str(delta_generator)
    if dry_run:
        click.echo("[dry-run] would call deltas writer with:")
        click.echo(f"  traces_dataset_name_or_path = {traces_dataset}")
        click.echo(f"  output_dataset_path = {output_path}")
        click.echo(f"  delta_generator = {delta_generator}")
        click.echo(f"  verbose = {verbose}")
        click.echo(f"  force   = {force}")
        return
    delta_writer.write_dataset(
        traces_dataset_name_or_path=traces_dataset,
        output_dataset_path=output_path,
        delta_generator=delta_generator,
        verbose=verbose,
        force_overwrite=force,
    )
    logger.info("all done")


if __name__ == "__main__":
    main()
