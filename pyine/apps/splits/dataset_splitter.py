"""
Dataset splitter CLI.

This application lets you:
- Generate a split file for a given dataset (containing all necessary metadata for experiments).
- Partition problem identifiers listed in a split file into disjoint files (for distributed preparation).

Examples:

    Splitting the TACO dataset 80-10-10 with difficulty and solution counts groups:
    ```bash
        python -m pyine.apps.splits.dataset_splitter split \
            --dataset-name TACO \
            --train-fraction=0.8 \
            --valid-fraction=0.1 \
            --test-fraction=0.1 \
            --use-difficulty-group \
            --use-solution-counts-group \
            --progress
    ```

    Partition problem identifiers from a split file into 100-sample-chunks (YAML by default):
    ```bash
        python -m pyine.apps.splits.dataset_splitter partition \
            --split-file data/splits/TACO-split.bin \
            --output-dir data/splits \
            --ids-per-chunk 100 \
            --format yaml
    ```

Note: by default, if the path to the dataset is not specified when creating a split file, we attempt
to fetch the latest version of the dataset from the corresponding dataset module's utility functions.
"""

import itertools
import json
import logging
import pathlib
import typing

import click
import yaml

import pyine.data.taco.dataset_utils
import pyine.data.utils.splits
import pyine.utils.filesystem
import pyine.utils.reprod

if typing.TYPE_CHECKING:
    import pydantic

logger = logging.getLogger(__name__)


def _get_group_rules(
    tag_lists: list[list[str]],
    use_difficulty_group: bool,
    use_solution_count_group: bool,
) -> list[str]:
    """Returns the prefixes to use for each group."""
    group_prefixes: list[str] = []
    if use_difficulty_group:
        group_prefixes.append("difficulty:")
    if use_solution_count_group:
        group_prefixes.append("solutions:")
    group_rules: list[str] = []
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
    return pathlib.Path(source_dataset_path)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """Dataset splitter CLI (split, partition)."""
    pass


# ------------------------------ SPLIT ------------------------------


@main.command("split")
@click.option(
    "--dataset-name",
    "dataset_name",
    type=str,
    required=True,
    help="Name of the source dataset to be split.",
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
    "--force",
    "force",
    is_flag=True,
    default=False,
    help="Overwrite the generated splits file if it already exists.",
)
@click.option(
    "--progress/--no-progress",
    "show_progress",
    default=True,
    show_default=True,
    help="Enable/disable progress bar.",
)
def split(
    dataset_name: str,
    dataset_path: pathlib.Path | None,
    train_fraction: float,
    valid_fraction: float,
    test_fraction: float,
    use_difficulty_group: bool,
    use_solution_count_group: bool,
    max_sample_count: int | None,
    force: bool,
    show_progress: bool,
) -> None:
    """Entry point for the dataset splitter CLI."""
    pyine.utils.reprod.entrypoint_setup()
    output_path = pyine.data.utils.splits.get_dataset_split_file_path(dataset_name, must_exist=False)
    pyine.utils.filesystem.check_output_path_overwrite(output_path, force=force)
    dataset_path = _get_resolved_dataset_path(dataset_name, dataset_path)
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
        subset_names=["train", "valid", "test"],
        subset_assign_prob_map={
            "train": train_fraction,
            "valid": valid_fraction,
            "test": test_fraction,
        },
        stratif_group_rules=group_rules,
        rules_are_case_sensitive=False,
    )
    assignments = split_config.build_subset_assignments(identifiers, tag_lists)
    result = pyine.data.utils.splits.SplitResult(
        source_dataset_name=dataset_name,
        source_dataset_hash=pyine.utils.reprod.get_params_hash(hash_list),
        identifiers=identifiers,
        tag_lists=tag_lists,
        source_data_hashes=hash_list,
        subset_assignments=assignments,
        creation_metadata=typing.cast(
            "dict[str, pydantic.JsonValue]",
            pyine.utils.reprod.get_reprod_metadata(),
        ),
        config=split_config,
    )
    result.to_file(output_path)
    split_file_size = pyine.utils.filesystem.get_human_readable_size(
        num_bytes=output_path.stat().st_size,
    )
    logger.info(f"all done; wrote {split_file_size} file with {len(assignments)} assignments to: {output_path}")


# ------------------------------ PARTITION ------------------------------


@main.command("partition")
@click.option(
    "--split-file",
    "split_file",
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    required=True,
    help=(
        "Path to a split file created by `pyine/apps/splits/dataset_splitter.py`. Contains identifiers"
        " assigned to subsets."
    ),
)
@click.option(
    "--output-dir",
    "output_dir",
    type=click.Path(file_okay=False, path_type=pathlib.Path),
    required=True,
    help="Directory where partition files will be written.",
)
@click.option(
    "--ids-per-chunk",
    "ids_per_chunk",
    type=click.IntRange(min=1),
    required=True,
    help="Maximum number of coding problem identifiers per output part/chunk file.",
)
@click.option(
    "--only-assigned-ids/--no-only-assigned-ids",
    "only_assigned_ids",
    default=True,
    show_default=True,
    help="Whether to only include identifiers that are assigned to a subset.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["yaml", "json", "txt"], case_sensitive=False),
    default="yaml",
    show_default=True,
    help="Output format for partition files.",
)
@click.option("--verbose/--no-verbose", "verbose", default=True, show_default=True)
@click.option(
    "--dry-run",
    "dry_run",
    is_flag=True,
    default=False,
    help="If set, prints resolved configuration and exits without reading or writing files.",
)
@click.option(
    "--force",
    "force",
    is_flag=True,
    default=False,
    help="Overwrite existing partition files instead of failing fast.",
)
def partition(
    split_file: pathlib.Path,
    output_dir: pathlib.Path,
    ids_per_chunk: int,
    only_assigned_ids: bool,
    output_format: str,
    verbose: bool,
    dry_run: bool,
    force: bool,
) -> None:
    """Partition problem identifiers from a split file into N disjoint files.

    The split file is expected to contain sample identifiers assigned to subsets, and should have
    been created using the `pyine.apps.split.dataset_splitter` CLI app. This command parses coding
    problem identifiers, asserts that there are no duplicates, and partitions the resulting
    list into contiguous chunks of a given max size such that concatenating the partitions in order
    reconstructs the original problem identifiers list.

    The resulting partition files are written to the specified output directory following the
    file name of the input split file; the written files will be suffixed with the partition
    identifier (e.g. ``<<split_file>>.problem_ids.001of004.yaml``).
    """
    pyine.utils.reprod.entrypoint_setup()
    output_format = output_format.lower()
    split_result = pyine.data.utils.splits.SplitResult.from_file(split_file)
    problem_identifiers = split_result.identifiers
    assert len(problem_identifiers) == len(set(problem_identifiers)), "duplicate identifiers found??"
    if only_assigned_ids:  # keep only identifiers that are present in assignments
        assigned_set = set(split_result.subset_assignments.keys())
        problem_identifiers = [sid for sid in problem_identifiers if sid in assigned_set]
    if len(problem_identifiers) == 0:
        logger.warning("no identifiers found to partition; exiting")
        return
    parts: list[tuple[str, ...]] = list(itertools.batched(problem_identifiers, ids_per_chunk))
    assert [pid for part in parts for pid in part] == problem_identifiers
    supported_formats = {"yaml": ".yaml", "json": ".json", "txt": ".txt"}
    assert output_format in supported_formats, f"unsupported output format: {output_format}"
    split_file_prefix = split_file.name.rsplit(".", maxsplit=1)[0]
    planned_outputs: list[tuple[pathlib.Path, tuple[str, ...]]] = []
    for idx, problem_ids_chunk in enumerate(parts, start=1):
        out_ext_str = f"problem_ids.{idx:06d}of{len(parts):06d}{supported_formats[output_format]}"
        out_path = output_dir / f"{split_file_prefix}.{out_ext_str}"
        planned_outputs.append((out_path, problem_ids_chunk))
    if dry_run:
        click.echo("[dry-run] would partition split file with:")
        click.echo(f"  split_file = {split_file}")
        click.echo(f"  output_dir = {output_dir}")
        click.echo(f"  ids_per_chunk = {ids_per_chunk}")
        click.echo(f"  only_assigned_ids = {only_assigned_ids}")
        click.echo(f"  output_format = {output_format}")
        click.echo(f"  verbose = {verbose}")
        click.echo(f"  force = {force}")
        click.echo(f"  found problem ids = {len(problem_identifiers)}")
        click.echo(f"  partition into {len(planned_outputs)} chunks:")
        for out_path, _ in planned_outputs:
            click.echo(f"    - {out_path}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    if not force:
        existing_paths = [path for path, _ in planned_outputs if path.exists()]
        if existing_paths:
            preview = ", ".join(str(path) for path in existing_paths[:3])
            if len(existing_paths) > 3:
                preview += ", ..."
            raise click.ClickException(
                f"partition output already exists (refusing to overwrite): {preview}. "
                "Re-run with --force to replace the existing files."
            )
    for out_path, problem_ids_chunk in planned_outputs:
        if output_format == "yaml":
            with open(out_path, "w", encoding="utf-8") as fd:
                yaml.safe_dump(problem_ids_chunk, fd, sort_keys=False)
        elif output_format == "json":
            with open(out_path, "w", encoding="utf-8") as fd:
                json.dump(problem_ids_chunk, fd, indent=2)
        else:  # txt
            with open(out_path, "w", encoding="utf-8") as fd:
                fd.write("\n".join(problem_ids_chunk) + "\n")
    logger.info(f"wrote {len(parts)} files partitioning {len(problem_identifiers)} problems to: {output_dir}")


if __name__ == "__main__":
    main()
