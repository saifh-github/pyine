import logging
import pathlib

import click
import orjson
import pydantic

import pyine.data.taco.dataset_utils
import pyine.data.traces.dataset_utils
import pyine.data.utils.splits
import pyine.utils.filesystem
import pyine.utils.reprod

logger = logging.getLogger(__name__)


LATEST_DATASET_PATH_TOKEN = "__latest__"
"""Special token used to indicate that the latest dataset should be used.

Only supported for datasets that are known and for which we have an imported module.
"""


class MainConfig(pydantic.BaseModel):
    """Configuration for the script's main function.

    Assembles the components required to split a coding problem dataset according to specific groups
    of tags, and save the resulting splits to disk.

    NOTE: this config is intended to be used with the `main` function defined below, and is provided
    here as a demonstration of how to use this app (will be refactored/cleaned when we have a config
    manager).
    """

    source_dataset_name: str
    """Name of the source dataset this split is derived from."""
    source_dataset_path: str = LATEST_DATASET_PATH_TOKEN
    """Hash of the source dataset this split is derived from."""
    split_config: pyine.data.utils.splits.SplitConfig
    """Split settings specifying optional stratified grouping rules."""


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
    if source_dataset_path is None or source_dataset_path == LATEST_DATASET_PATH_TOKEN:
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
    split_output_path = pyine.data.utils.splits.get_dataset_split_file_path(dataset_name)
    pyine.utils.filesystem.check_output_path_overwrite(split_output_path)
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
        split_config=split_config,
    )
    with open(split_output_path, "wb") as fd:
        fd.write(orjson.dumps(result.model_dump()))
    print("all done")


if __name__ == "__main__":
    main()
