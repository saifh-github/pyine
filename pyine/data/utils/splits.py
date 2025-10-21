import logging
import pathlib
import typing

import numpy as np
import orjson
import pydantic
import tqdm

import pyine.data.traces.dataset_utils
import pyine.data.utils.filter_rules
import pyine.utils.filesystem

logger = logging.getLogger(__name__)


SampleIdentifierType = str
"""Type of the identifier used to uniquely refer to each sample in a dataset."""
SampleTagsType = list[str]
"""Type of the list of tags associated with each sample in a dataset."""
SampleHashType = str
"""Type of the hash used to uniquely identify each sample in a dataset."""
SubsetNameType = str
"""Type used to represent a data subset name (e.g. 'train', 'valid', 'test')."""
ProbabilityType = typing.Annotated[pydantic.StrictFloat, pydantic.Field(ge=0, le=1)]
"""Type used to describe probabilities in subset assignment probability maps."""
FilterRuleType = str
"""Type used for filtering rules that apply to the tags of samples."""


class SplitConfig(pydantic.BaseModel):
    """Configuration settings for splits using optional stratified grouping and hard-assign rules."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
    """Pydantic model configuration (freezes the dataclass)."""
    seed: int = 0
    """Seed to use for reproducibility."""
    subset_names: list[SubsetNameType] = pydantic.Field(min_length=2)
    """List of subset names to use (mandatory, must be fully specified)."""
    subset_assign_rules_map: dict[SubsetNameType, FilterRuleType] = pydantic.Field(
        default_factory=lambda: typing.cast("dict[SubsetNameType, FilterRuleType]", {}),
    )
    """Dictionary mapping subset names to filter rules.

    The specified filter rules will be applied, in order, to the tags of parsed data samples, and
    if the sample passes the filter (i.e. it returns False), it will be added to the corresponding
    subset. Samples are added to the first subset for which they pass the filter. Samples that do
    not pass any filter rules will then be split into subsets according to `subset_assign_prob_map`,
    and according to any potential stratified splitting rules.

    This field is optional, as providing no hard assignment rule means that all samples will be
    randomly assigned using `subset_assign_prob_map` directly.
    """
    subset_assign_prob_map: dict[SubsetNameType, ProbabilityType]  # MANDATORY!
    """Dictionary mapping subset names to assignment probabilities.

    Applied to samples that were NOT already assigned to a subset using `subset_assign_rules_map`,
    and that may have been grouped according to a potential stratification rule.

    MANDATORY FIELD; the specified probabilities should always sum to 1.
    """
    stratif_group_rules: list[FilterRuleType] = pydantic.Field(
        default_factory=lambda: typing.cast("list[FilterRuleType]", []),
    )
    """List of filter rules applied to the tags of parsed data samples to form stratification groups.

    This list can be empty, in which case no stratification rules will be applied, and all samples
    will be assigned to different subsets directly according to `subset_assign_prob_map`. If one
    or more rules are specified, we will iterate over all samples to find which ones belong to each
    group. Each sample can only be assigned to one group (the first group whose rule they pass).
    All members of a group will then be assigned to different subsets using `subset_assign_prob_map`.
    If there are leftover samples not assigned to any stratification group based on these rules,
    these will go into a default group which will also be split using `subset_assign_prob_map`.

    This field is optional, as providing no stratification rule means that all samples will be
    randomly assigned using `subset_assign_prob_map` directly.
    """
    rules_are_case_sensitive: bool = False
    """Defines whether rules should be interpreted as case-sensitive or not."""

    # ----------------- public / utility functions -----------------

    def build_subset_assignments(
        self,
        identifiers: list[SampleIdentifierType],
        tag_lists: list[SampleTagsType],
    ) -> dict[SampleIdentifierType, SubsetNameType]:
        """Builds a complete mapping for a given list of sample identifiers and tags.

        Steps:
        - Apply hard assignment rules to each sample (returning the first matched subset, if any).
        - For remaining samples, assign to stratification groups (based again on first matched group, if any).
        - For each group, assign samples independently to subsets according to the assignment probabilities.
        - For remaining (ungrouped) samples, assign them to subsets according to the assignment probabilities.
        """
        assignments, unassigned_indices = self._apply_hard_subset_assignments(identifiers, tag_lists)
        if not unassigned_indices:
            return assignments
        probs = [float(self.subset_assign_prob_map.get(name, 0.0)) for name in self.subset_names]
        rng = np.random.default_rng(self.seed)
        group_idxs = self._get_group_indices(tag_lists)
        # build per-group lists of unassigned indices to process to make the grouping step explicit
        indices_per_group: list[list[int]] = [[] for _ in range(self._get_group_count())]
        for unassigned_idx in unassigned_indices:
            indices_per_group[group_idxs[unassigned_idx]].append(unassigned_idx)
        # do assignments for each group independently
        for group_sidxs in indices_per_group:
            if not group_sidxs:
                continue
            curr_subset_assigns = rng.choice(self.subset_names, size=len(group_sidxs), p=probs, replace=True)
            for sidx, subset in zip(group_sidxs, curr_subset_assigns.tolist(), strict=False):
                sid = identifiers[sidx]
                assert sid not in assignments, f"sample {sid} already assigned"
                assignments[sid] = subset
        return assignments

    def build_subset_to_identifiers_map(
        self,
        identifiers: list[SampleIdentifierType],
        tag_lists: list[SampleTagsType],
    ) -> dict[SubsetNameType, list[SampleIdentifierType]]:
        """Returns a mapping from subset name to the list of sample identifiers assigned to it.

        See the `build_subset_assignments` method for more details on behavior and arguments.
        """
        id_to_subset = self.build_subset_assignments(identifiers=identifiers, tag_lists=tag_lists)
        out: dict[SubsetNameType, list[SampleIdentifierType]] = {name: [] for name in self.subset_names}
        for sid, subset in id_to_subset.items():
            assert subset in out, "unexpected subset name"
            out[subset].append(sid)
        return out

    # ----------------- below is private stuff that does not affect serialization -----------------

    def _get_group_index_for_tags(self, tags: SampleTagsType) -> int:
        """Returns the stratification group index for a single sample's tags.

        The first rule that the sample 'passes' determines the group (where 'passes' means a filter
        rule returns `False`). If no rule is passed or no rules are configured, returns the default
        group index, which is len(self.stratif_group_rules).
        """
        assert self._stratif_group_rules is not None, "uninitialized stratif_group_rules?"
        for group_idx, filter_fn in enumerate(self._stratif_group_rules):
            if not filter_fn(tags):  # match found
                return group_idx
        return len(self._stratif_group_rules)  # no match found, return the default group index

    def _get_group_count(self) -> int:
        """Returns the number of stratification groups."""
        return len(self.stratif_group_rules) + 1  # + 1 for default group

    def _get_group_indices(self, tag_lists: list[SampleTagsType]) -> list[int]:
        """Returns a list of group indices, one per sample, using `_get_group_index_for_tags`."""
        return [self._get_group_index_for_tags(tags) for tags in tag_lists]

    def _apply_hard_subset_assignments(
        self,
        identifiers: list[SampleIdentifierType],
        tag_lists: list[SampleTagsType],
    ) -> tuple[dict[SampleIdentifierType, SubsetNameType], list[int]]:
        """Applies hard assignment rules to samples and returns (assignments, unassigned_indices).

        A sample is assigned to the first subset whose rule it 'passes' (rule filter returns False).
        Samples for which no rule is passed are left unassigned and their indices are returned.
        """
        if len(identifiers) != len(tag_lists):
            raise ValueError("length of identifiers and tag lists must match")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("identifiers must be unique")
        assignments: dict[SampleIdentifierType, SubsetNameType] = {}
        unassigned_indices: list[int] = []
        for idx, (sid, tags) in enumerate(zip(identifiers, tag_lists, strict=False)):
            assigned = False
            for subset_name, filter_fn in self._subset_assign_rules_map.items():
                if not filter_fn(tags):  # match found
                    assignments[sid] = subset_name
                    assigned = True
                    break
            if not assigned:
                unassigned_indices.append(idx)
        return assignments, unassigned_indices

    _subset_assign_rules_map: dict[SubsetNameType, pyine.data.utils.filter_rules.FilterType] | None = (
        pydantic.PrivateAttr(default=None)
    )
    _stratif_group_rules: list[pyine.data.utils.filter_rules.FilterType] | None = pydantic.PrivateAttr(default=None)

    @pydantic.model_validator(mode="after")
    def _post_validator(self) -> "SplitConfig":
        """Confirms that inter-setting configuration is valid and resolves filter rules."""
        if any(name == "" for name in self.subset_names):
            raise ValueError("subset names must be non-empty")
        unknown_names = [name for name in self.subset_assign_rules_map if name not in self.subset_names]
        if unknown_names:
            raise ValueError(f"unknown subset names in `subset_assign_rules_map`: {unknown_names}")
        unknown_names = [name for name in self.subset_assign_prob_map if name not in self.subset_names]
        if unknown_names:
            raise ValueError(f"unknown subset names in `subset_assign_prob_map`: {unknown_names}")
        if not self.subset_assign_prob_map:
            raise ValueError("subset_assign_prob_map must provide probabilities for all subsets (map is empty)")
        missing_names = sorted(name for name in self.subset_names if name not in self.subset_assign_prob_map)
        if missing_names:
            missing_str = ", ".join(missing_names)
            raise ValueError(
                f"subset_assign_prob_map is missing probabilities for subsets: {missing_str}. "
                "Provide explicit fractions for every subset."
            )
        probs_total = sum(float(self.subset_assign_prob_map[name]) for name in self.subset_names)
        if not np.isclose(probs_total, 1.0, rtol=0.0, atol=1e-8):
            diff = abs(probs_total - 1.0)
            raise ValueError(
                f"subset assignment probabilities must sum to 1.0 (got {probs_total:.12f}, diff={diff:.12f}); "
                f"double-check for rounding drift or missing fractions: {self.subset_assign_prob_map}"
            )
        self._subset_assign_rules_map = {
            name: pyine.data.utils.filter_rules.build_filter_from_rule(
                rule=rule_str,
                case_sensitive=self.rules_are_case_sensitive,
            )
            for name, rule_str in self.subset_assign_rules_map.items()
        }
        self._stratif_group_rules = [
            pyine.data.utils.filter_rules.build_filter_from_rule(
                rule=rule_str,
                case_sensitive=self.rules_are_case_sensitive,
            )
            for rule_str in self.stratif_group_rules
        ]
        return self


class SplitResult(pydantic.BaseModel):
    """Contains the results of a dataset split.

    This dataclass is used to store the full results of a split, including all known/parsed
    sample identifiers and tags, all subset assignments, and all split configuration settings.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="allow")
    """Pydantic model configuration (freezes the dataclass)."""

    source_dataset_name: str
    """Name of the source dataset this split is derived from."""
    source_dataset_hash: str
    """Hash of the source dataset this split is derived from."""
    identifiers: list[SampleIdentifierType]
    """List of sample identifiers parsed from the dataset."""
    tag_lists: list[SampleTagsType]
    """List of tags associated with each sample in the dataset."""
    source_data_hashes: list[str]
    """List of hashes of the source data used to create each data sample."""
    subset_assignments: dict[SampleIdentifierType, SubsetNameType]
    """Dictionary mapping subset names to lists of sample indices assigned to that subset."""
    creation_metadata: dict[str, pydantic.JsonValue]
    """Additional metadata associated with this split result."""
    config: SplitConfig
    """Split settings specifying optional stratified grouping rules."""

    @pydantic.model_validator(mode="after")
    def _post_validator(self) -> "SplitResult":
        """Confirms that all sample metadata fields have matching/expected lengths and contents."""
        if len(self.identifiers) != len(self.tag_lists):
            raise ValueError("length of identifiers and tag lists must match")
        if len(self.identifiers) != len(set(self.identifiers)):
            raise ValueError("identifiers must be unique")
        if len(self.subset_assignments) != len(self.identifiers):
            raise ValueError("length of subset assignments and identifiers must match")
        if len(self.source_data_hashes) != len(self.identifiers):
            raise ValueError("length of source data hashes and identifiers must match")
        if len(self.source_data_hashes) != len(set(self.source_data_hashes)):
            raise ValueError("source data hashes must be unique")
        return self

    @classmethod
    def from_file(cls, file_path: pathlib.Path | str) -> "SplitResult":
        """Loads a split result from a file."""
        file_path = pathlib.Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"split result file not found: {file_path}")
        with open(file_path, "rb") as fd:
            split_result_dict = orjson.loads(fd.read())
        return cls.model_validate(split_result_dict)

    def get_subset_to_ids_map(self) -> dict[SubsetNameType, list[SampleIdentifierType]]:
        """Returns a mapping from subset name to the list of sample identifiers assigned to it."""
        output_map: dict[SubsetNameType, list[SampleIdentifierType]] = {}
        for sid, subset in self.subset_assignments.items():
            if subset not in output_map:
                output_map[subset] = []
            output_map[subset].append(sid)
        return output_map

    def to_file(self, file_path: pathlib.Path | str) -> None:
        """Saves a split result to a file."""
        file_path = pathlib.Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as fd:
            fd.write(orjson.dumps(self.model_dump()))


def _get_solution_count_bucket_tag(solution_count: int) -> str:
    """Returns a string representation of a solution count bucket tag."""
    if solution_count < 10:
        suffix = "0-10"
    elif solution_count < 25:
        suffix = "10-25"
    elif solution_count < 50:
        suffix = "25-50"
    elif solution_count < 100:
        suffix = "50-100"
    elif solution_count < 250:
        suffix = "100-250"
    elif solution_count < 1000:
        suffix = "500-1000"
    else:
        suffix = "1000+"
    return f"solutions:{suffix}"


def get_split_data_from_coding_problem_dataset(
    source_dataset_name: str,
    source_dataset_path: pathlib.Path | str,
    max_sample_count: int | None = None,
    verbose: bool = False,
) -> tuple[list[SampleIdentifierType], list[SampleTagsType], list[SampleHashType]]:
    """Parses the source dataset for all available sample identifiers, tags, and hashes.

    Note: if some samples in the source dataset are missed or skipped due to any reason here (such
    as corrupted files, etc.), we will not keep track of them, and they will be permanently lost
    and dropped from splits that are derived from the returned dictionary.

    Args:
        source_dataset_name: Name of the source dataset this split is derived from.
        source_dataset_path: Path of the source dataset this split is derived from. Will be passed
            alongside the dataset name to the `CodingProblemIterator` constructor which should be
            able to prepare the dataset for parsing.
        max_sample_count: Maximum number of samples to parse from the source dataset. If None,
            the entire dataset will be parsed.
        verbose: Specifies whether verbose progress messages should be printed.

    Returns:
        The tuple of metadata required to perform a split on the source dataset (ids, tags, hashes).
    """
    logger.info(f"parsing problem metadata for {source_dataset_name} source dataset...")
    problem_data_iter = pyine.data.traces.dataset_utils.CodingProblemIterator(
        dataset_name=source_dataset_name,
        root_data_path=source_dataset_path,
        show_progress=verbose,
        reformat_code_strings=False,
        validate_code_strings=False,
        add_orig_subset_as_tag=False,
    )
    if len(problem_data_iter) == 0:
        raise ValueError(f"no problems found in {source_dataset_name} source dataset")
    identifiers: list[SampleIdentifierType] = []
    tag_lists: list[SampleTagsType] = []
    hash_list: list[SampleHashType] = []
    problem_count = len(problem_data_iter)
    max_iter_count = min(max_sample_count, problem_count) if max_sample_count is not None else problem_count
    prog_bar = tqdm.tqdm(total=max_iter_count, desc="gathering problem data", disable=not verbose)
    for problem_idx in range(problem_count):
        problem, solutions = problem_data_iter[problem_idx]
        # skip any problem with a parsing error
        if problem.parsing_errors:
            continue
        problem_tags = problem.problem_tags.copy()
        assert all(isinstance(t, str) and len(t) > 0 for t in problem_tags), "unexpected tag format"
        # drop any subset tags if there already are any (we are recreating the split entirely)
        problem_tags = [t for t in problem_tags if not t.startswith("subset:")]
        # append custom tags as needed
        problem_tags.append(_get_solution_count_bucket_tag(len(solutions)))
        identifiers.append(str(problem.problem_id))
        tag_lists.append(problem_tags)
        hash_list.append(problem.source_data_hash)
        prog_bar.update(1)
        if max_sample_count is not None and len(identifiers) >= max_sample_count:
            break
    prog_bar.close()
    return identifiers, tag_lists, hash_list


def get_dataset_split_file_path(
    source_dataset_name: str,
    must_exist: bool = True,
) -> pathlib.Path:
    """Returns the path of the split file for a given dataset.

    If `must_exist` is True and the split does not exist, raises FileNotFoundError.
    """
    split_root_folder = pyine.utils.filesystem.get_data_root_path() / "splits"
    split_root_folder.mkdir(parents=True, exist_ok=True)
    split_path = split_root_folder / f"{source_dataset_name}-split.bin"
    if must_exist and not split_path.is_file():
        raise FileNotFoundError(f"split file not found for: {source_dataset_name}")
    return split_path


def get_dataset_split_result(
    source_dataset_name_or_split_file_path: str | pathlib.Path,
) -> SplitResult:
    """Returns the split result for a given dataset or at a given path.

    If a dataset name is provided and the split file does not exist, an exception will be raised.
    """
    potential_path = pathlib.Path(source_dataset_name_or_split_file_path)
    if potential_path.is_file():
        split_file_path = potential_path
    else:
        split_file_path = get_dataset_split_file_path(
            str(source_dataset_name_or_split_file_path),
            must_exist=True,
        )
    with open(split_file_path, "rb") as fd:
        split_result_dict = orjson.loads(fd.read())
    return SplitResult.model_validate(split_result_dict)


def get_dataset_split_part_file_paths(
    source_dataset_name_or_split_file_path: str | pathlib.Path,
    part_extension: str = ".yaml",
) -> list[pathlib.Path]:
    """Returns the list of split part file paths for a given dataset or at a given split file path.

    We expect that the part files are prefixed with the same name as the split file path itself,
    and that the target part file extension is provided.
    """
    potential_path = pathlib.Path(source_dataset_name_or_split_file_path)
    if potential_path.is_file():
        split_file_path = potential_path
    else:
        split_file_path = get_dataset_split_file_path(
            str(source_dataset_name_or_split_file_path),
            must_exist=False,
        )
    expected_part_file_name_pattern = f"{split_file_path.stem}.problem_ids.*of*{part_extension}"
    part_file_paths = list(split_file_path.parent.glob(expected_part_file_name_pattern))
    part_file_paths.sort()
    return part_file_paths
