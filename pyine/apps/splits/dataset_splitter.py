"""
Dataset splitter CLI.

This application wraps the dataset splitter routine and lets you generate a split file for a given
dataset with command-line controls.

Examples:

    Splitting the TACO dataset 80-10-10 with difficulty and solution counts groups:
    ```bash
        python pyine/apps/splits/dataset_splitter.py \
            --dataset-name TACO \
            --train-fraction=0.8 \
            --valid-fraction=0.1 \
            --test-fraction=0.1 \
            --use-difficulty-group \
            --use-solution-counts-group \
            --progress
    ```

Note: by default, if the path to the dataset is not specified, the CLI will attempt to fetch the
latest version of the dataset from the corresponding dataset module's utility functions.
"""

import logging
import pathlib

import click

import pyine.data.taco.dataset_utils
import pyine.data.traces.dataset_utils
import pyine.data.utils.splits
import pyine.utils.filesystem
import pyine.utils.reprod

logger = logging.getLogger(__name__)


def _get_group_rules(
    tag_lists: list[list[str]],
    use_difficulty_group: bool,
    use_solution_count_group: bool,
) -> list[str]:
    """Returns the prefixes to use for each group."""
    group_prefixes = []
    if use_difficulty_group:
        group_prefixes.append("difficulty:")
    if use_solution_count_group:
        group_prefixes.append("solutions:")
    group_rules = []
    for tag_list in tag_lists:
        for tag in tag_list:
            for group_prefix in group_prefixes:
                if tag.startswith(group_prefix):
                    expected_group_rule = f"+{tag}"
                    if expected_group_rule not in group_rules:
                        group_rules.append(expected_group_rule)
                    break
    return sorted(group_rules)


def _get_resolved_dataset_path(
    source_dataset_name: str,
    source_dataset_path: pathlib.Path | str | None,
) -> pathlib.Path:
    """Returns the resolved path to the dataset to be split."""
    if source_dataset_path is None:
        supported_source_datasets = ["TACO"]
        if source_dataset_name not in supported_source_datasets:
            raise ValueError(f"invalid source dataset name for latest detection: {source_dataset_name}")
        if source_dataset_name == "TACO":
            source_dataset_path = pyine.data.taco.dataset_utils.get_latest_repackaged_dataset_path()
        else:
            raise NotImplementedError
    source_dataset_path = pathlib.Path(source_dataset_path)
    return source_dataset_path


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--dataset-name",
    "dataset_name",
    type=str,
    required=True,
    help="Name of the source dataset this split is derived from.",
)
@click.option(
    "--dataset-path",
    "dataset_path",
    type=click.Path(exists=True, path_type=pathlib.Path),
    required=False,
    help=(
        "Path to a source dataset to split. If the name already points to a known dataset, "
        "this path can be omitted, in which case we will go fetch the latest version of the dataset."
    ),
)
@click.option(
    "--train-fraction",
    "train_fraction",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.8,
    show_default=True,
    help="Fraction of the dataset to use for training.",
)
@click.option(
    "--valid-fraction",
    "valid_fraction",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.1,
    show_default=True,
    help="Fraction of the dataset to use for validation.",
)
@click.option(
    "--test-fraction",
    "test_fraction",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.1,
    show_default=True,
    help="Fraction of the dataset to use for testing.",
)
@click.option(
    "--use-difficulty-group/--no-difficulty-group",
    "use_difficulty_group",
    default=True,
    show_default=True,
    help="Whether to use 'difficulty' tags as groups for stratified splitting.",
)
@click.option(
    "--use-solution-counts-group/--no-solution-counts-group",
    "use_solution_count_group",
    default=True,
    show_default=True,
    help="Whether to use solutions counts as groups for stratified splitting.",
)
@click.option(
    "--max-sample-count",
    "max_sample_count",
    type=click.IntRange(min=1),
    default=None,
    help="Maximum number of samples to consider for the split; if unspecified, consider all.",
)
@click.option(
    "--progress/--no-progress",
    "show_progress",
    default=True,
    show_default=True,
    help="Enable/disable progress bar.",
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
    dataset_name: str,
    dataset_path: pathlib.Path | None,
    train_fraction: float,
    valid_fraction: float,
    test_fraction: float,
    use_difficulty_group: bool,
    use_solution_count_group: bool,
    max_sample_count: int | None,
    show_progress: bool,
    log_level: str,
    log_file: pathlib.Path | None,
) -> None:
    """Entry point for the dataset splitter CLI."""
    numeric_log_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_log_level, int):
        raise click.BadParameter(f"invalid log level: {numeric_log_level}")
    pyine.utils.reprod.entrypoint_setup(log_level=numeric_log_level, log_path=log_file)
    output_path = pyine.data.utils.splits.get_dataset_split_file_path(dataset_name)
    pyine.utils.filesystem.check_output_path_overwrite(output_path)
    dataset_path = _get_resolved_dataset_path(dataset_name, dataset_path)
    logger.info(f"computing dataset hash for '{dataset_name}' at: {dataset_path}")
    source_dataset_hash = pyine.utils.reprod.compute_hash(dataset_path)
    logger.info(f"starting dataset splitting for '{dataset_name}' at: {dataset_path}")
    identifiers, tag_lists, hash_list = pyine.data.utils.splits.get_split_data_from_coding_problem_dataset(
        source_dataset_name=dataset_name,
        source_dataset_path=dataset_path,
        max_sample_count=max_sample_count,
        verbose=show_progress,
    )
    logger.info(f"found {len(identifiers)} samples in '{dataset_name}' dataset")
    group_rules = _get_group_rules(tag_lists, use_difficulty_group, use_solution_count_group)
    split_config = pyine.data.utils.splits.SplitConfig(
        subset_names=["train", "val", "test"],
        subset_assign_prob_map={
            "train": train_fraction,
            "val": valid_fraction,
            "test": test_fraction,
        },
        stratif_group_rules=group_rules,
        rules_are_case_sensitive=False,
    )
    assignments = split_config.build_subset_assignments(identifiers, tag_lists)
    result = pyine.data.utils.splits.SplitResult(
        source_dataset_name=dataset_name,
        source_dataset_hash=source_dataset_hash,
        identifiers=identifiers,
        tag_lists=tag_lists,
        source_data_hashes=hash_list,
        subset_assignments=assignments,
        creation_metadata=pyine.utils.reprod.get_reprod_metadata(),
        config=split_config,
    )
    result.to_file(output_path)
    split_file_size = pyine.utils.filesystem.get_human_readable_size(
        num_bytes=output_path.stat().st_size,
    )
    print(f"all done; wrote {split_file_size} file with {len(assignments)} assignments to: {output_path}")


if __name__ == "__main__":
    main()
