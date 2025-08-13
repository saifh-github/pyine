import dataclasses
import datetime
import fnmatch
import importlib.resources as pkg_resources
import json
import logging
import pathlib
import re
import typing

import numpy as np
import pydantic
import tqdm
import yaml

import pyine.data.utils.lmdb_io
import pyine.prompts.configs.code_analysis
import pyine.utils.code.execution
import pyine.utils.code.formatting
import pyine.utils.code.validation
import pyine.utils.filesystem
import pyine.utils.portability
import pyine.utils.reprod

__all__ = [
    "CodingProblemIdentifier",
    "SolutionIdentifier",
    "TraceIdentifier",
    "CodingProblem",
    "Solution",
    "ProblemIdPattern",
    "CodingProblemIterator",
    "get_latest_dataset_path",
    "get_new_dataset_path",
    "compare_result_strings",
    "is_float",
]

PROBLEM_DATA_SUFFIX = "/metadata"
"""Suffix for entries that correspond to coding problem metadata in the LMDB dataset."""
PROBLEM_DATA_PATTERN = "*" + PROBLEM_DATA_SUFFIX
"""Pattern for matching coding problem entries in the LMDB dataset."""
TRACE_DATA_SUFFIX = "/s*/t*"
"""Suffix for entries that correspond to execution traces in the LMDB dataset."""
AUGM_TRACE_DATA_SUFFIX = "/s*/t*/a*"
"""Suffix for entries that correspond to execution traces of augmented code in the LMDB dataset."""
DELTAS_SUFFIX = "/deltas"
"""Suffix for entries that correspond to trace deltas in the LMDB dataset (for future-proofing)."""
SUPPORTED_SOURCE_DATASETS = [
    "TACO",
    # add more supported datasets here
]
"""List of supported source datasets that provide coding problems with solutions to be traced."""

JSON_BASED_SOURCE_DATASETS = [
    "TACO",
    # add more supported datasets here
]
"""List of source datasets that store coding problem data in json files."""
BANNED_DATA_YAML_PATH = pkg_resources.files("pyine.data.traces") / "banned_data.yaml"
"""Path to the YAML file containing banned data information for each supported source dataset."""

logger = logging.getLogger(__name__)


def is_float(s):
    try:
        _ = float(s)
        return True
    except ValueError:
        return False


def compare_result_strings(proposed: str, reference: str) -> bool:
    proposed = proposed.strip()
    reference = reference.strip()
    if is_float(proposed) and is_float(reference):
        rtol, atol = pyine.utils.portability.estimate_tolerance(reference)
        return bool(np.isclose(float(proposed), float(reference), rtol=rtol, atol=atol))
    else:
        return proposed == reference


@dataclasses.dataclass(frozen=True)
class CodingProblemIdentifier:
    """Frozen tuple used for identifying coding problems in their source datasets."""

    dataset: str
    """The name of the source dataset where the problem originated from."""
    subset: str
    """The name of the subset in the source dataset that the sample belongs to."""
    problem_idx: int
    """The index of the problem within the source subset that it belongs to."""

    def __repr__(self):
        """Returns a string representation of this identifier."""
        return f"{self.dataset}/{self.subset}/p{self.problem_idx:06d}"

    def get_parent_identifier(self) -> str:
        """Returns the parent identifier of this object (i.e., a dataset+subset identifier)."""
        return f"{self.dataset}/{self.subset}"

    @staticmethod
    def from_string(identifier_str: str) -> "CodingProblemIdentifier":
        """Creates an identifier object from a string representation."""
        assert isinstance(identifier_str, str), "identifier must be a string"
        dataset, subset, problem_idx_str = identifier_str.split("/")
        return CodingProblemIdentifier(dataset, subset, int(problem_idx_str[1:]))


@dataclasses.dataclass(frozen=True)
class SolutionIdentifier(CodingProblemIdentifier):
    """Frozen tuple used for identifying a specific solution to a coding problem."""

    solution_idx: int
    """Index identifying a specific solution for a coding problem (within the source dataset)."""

    def __repr__(self):
        """Returns a string representation of this identifier."""
        return f"{CodingProblemIdentifier.__repr__(self)}/s{self.solution_idx:04d}"

    def get_parent_identifier(self) -> CodingProblemIdentifier:
        """Returns the parent identifier of this object (i.e., a coding problem identifier)."""
        parent_vars = {var_name: var_val for var_name, var_val in vars(self).items() if var_name != "solution_idx"}
        return CodingProblemIdentifier(**parent_vars)

    @staticmethod
    def from_string(identifier_str: str) -> "SolutionIdentifier":
        """Creates an identifier object from a string representation."""
        assert isinstance(identifier_str, str), "identifier must be a string"
        parent_str, solution_idx_str = identifier_str.rsplit("/s", maxsplit=1)
        parent_id = CodingProblemIdentifier.from_string(parent_str)
        return SolutionIdentifier(**vars(parent_id), solution_idx=int(solution_idx_str))


@dataclasses.dataclass(frozen=True)
class TraceIdentifier(SolutionIdentifier):
    """Frozen tuple used for identifying a specific trace to a coding problem.

    In contrast with previous (parent) identifiers, this one includes the test index and an optional
    part, i.e. the augmentation category and index. This is used to identify specific traces that
    are based on "augmented" code derived from dataset solutions.
    """

    test_idx: int
    """Index identifying the test values used to create this trace (within the source dataset)."""
    augment_category: str | None = None
    """Category of augmentation used to create the code behind this trace (if any)."""
    augment_idx: int | None = None
    """Index identifying the augmented instance used to create this trace (if any)."""

    def __repr__(self):
        """Returns a string representation of this identifier."""
        out = f"{SolutionIdentifier.__repr__(self)}/t{self.test_idx:04d}"
        if self.augment_category is not None or self.augment_idx is not None:
            assert self.augment_category is not None and self.augment_idx is not None
            out += f"/a:{self.augment_category}:{self.augment_idx:03d}"
        return out

    def get_parent_identifier(self) -> SolutionIdentifier:
        """Returns the parent identifier of this object (i.e., a solution identifier)."""
        local_vars = ["test_idx", "augment_category", "augment_idx"]
        parent_vars = {var_name: var_val for var_name, var_val in vars(self).items() if var_name not in local_vars}
        return SolutionIdentifier(**parent_vars)

    @staticmethod
    def from_string(identifier_str: str) -> "TraceIdentifier":
        """Creates an identifier object from a string representation."""
        assert isinstance(identifier_str, str), "identifier must be a string"
        parent_str, trace_id_str = identifier_str.rsplit("/t", maxsplit=1)
        parent_id = SolutionIdentifier.from_string(parent_str)
        has_augm_split = "/a:" in trace_id_str
        if has_augm_split:
            test_idx_str, augment_id = trace_id_str.split("/a:", maxsplit=1)
            augment_category, augment_idx_str = augment_id.split(":", maxsplit=1)
            augment_idx = int(augment_idx_str)
        else:
            augment_category, augment_idx = None, None
            test_idx_str = trace_id_str
        return TraceIdentifier(
            **vars(parent_id),
            test_idx=int(test_idx_str),
            augment_category=augment_category,
            augment_idx=augment_idx,
        )


class CodingProblem(pydantic.BaseModel):
    """Class for storing data (e.g. metadata, i/o examples, etc.) for a specific coding problem.

    This is meant as a container to standardize the data format for coding problems across datasets.
    The primary fields that all datasets should have are the ones below; specific datasets may derive
    other interfaces with additional fields.
    """

    model_config = pydantic.ConfigDict(frozen=True)
    """Pydantic model configuration (freezes the dataclass)."""
    source_dataset_name: str
    """Name of the source dataset this problem comes from."""
    source_data_path: str
    """Path of the source file or folder this specific problem data comes from."""
    source_data_hash: str
    """Hash of the source file or folder this specific problem data comes from."""
    problem_id: CodingProblemIdentifier
    """Unique identifier for this problem (combines source-level identifiers)."""
    problem_statement: str
    """Problem statement, describing the coding problem to be solved."""
    problem_tags: list[str]
    """List of tags (labels) associated with this problem, assigned in the original dataset."""
    test_inout_pairs: list[tuple]
    """List of input/output pairs used to test solutions to this problem."""
    entrypoint_name: str | None
    """Name of the entrypoint function expected to be implemented for this problem (if any)."""
    potential_solution_ids: list[SolutionIdentifier]
    """List of potential solutions to this problem, identified by their identifiers."""
    parsing_errors: list[str] | None
    """Parsing errors (if any) that were encountered when extracting data for this problem.

    If any error is found, it probably means that this problem should be discarded, as parsing
    errors might not be recoverable.
    """
    is_banned: bool
    """Whether this problem is banned from being traced (due to a data/processing issue)."""

    def __str__(self):
        """Returns a string representation of the coding problem based on its identifier."""
        return str(self.problem_id)

    @property
    def solution_count(self):
        """Returns the number of solutions (i.e., code strings) that might be used for tracing."""
        return len(self.potential_solution_ids)

    @property
    def test_count(self):
        """Returns the number of tests (input/output pairs) that might be used for tracing."""
        return len(self.test_inout_pairs)

    def should_discard(self):
        """Returns whether this problem should be discarded due to banishment or parsing errors."""
        return self.is_banned or (self.parsing_errors is not None and len(self.parsing_errors) > 0)


class Solution(pydantic.BaseModel):
    """Class for storing data (e.g. code strings) for a specific solution to a coding problem.

    This is meant as a container to standardize the data format for solutions across datasets.
    The primary fields that all datasets should have are the ones below; specific datasets may derive
    other interfaces with additional fields.
    """

    model_config = pydantic.ConfigDict(frozen=True)
    """Pydantic model configuration (freezes the dataclass)."""
    parent_id: CodingProblemIdentifier
    """Identifier of the solution's parent (i.e., the problem to which this solution belongs)."""
    solution_id: SolutionIdentifier
    """Unique identifier for this solution (combines parent-level identifiers with solution index)."""
    code: str
    """Code string for this solution."""
    analysis_errors: list[str] | None
    """Errors (if any) that were encountered during analysis of this solution."""
    analysis_results: pyine.prompts.configs.code_analysis.CodeAnalysisResponse
    """Advanced code analysis results for this solution's code."""
    is_banned: bool
    """Whether this solution is banned from being traced (due to a data/processing issue)."""

    def __str__(self):
        """Returns a string representation of the solution based on its identifier."""
        return str(self.solution_id)

    @property
    def code_line_count(self):
        """Returns the number of lines in this solution's code."""
        return len(self.code.splitlines())

    @property
    def is_fishy(self):
        """Returns whether this solution is 'fishy' (i.e., contains potentially insecure code)."""
        return (
            self.analysis_results.imports_nonstandard_packages
            or self.analysis_results.invalid_syntax
            or self.analysis_results.filesystem_access
            or self.analysis_results.system_commands
            or self.analysis_results.network_access
        )

    @property
    def is_deterministic(self):
        """Returns whether this solution is deterministic (i.e., does not contain randomness)."""
        return self.analysis_results.is_deterministic

    @property
    def has_standard_io(self):
        """Returns whether this solution uses standard and easy-to-use input/output."""
        return self.analysis_results.input_type in [
            "stdin",
            "no-input",
            "callable",
        ] and self.analysis_results.output_type in ["stdout", "no-output", "callable"]

    def should_discard(self):
        """Returns whether this problem should be discarded due to issues or complexity."""
        return (
            self.is_banned
            or (self.analysis_errors is not None and len(self.analysis_errors) > 0)
            or self.is_fishy
            or not self.is_deterministic
            or not self.has_standard_io
        )


@dataclasses.dataclass
class _BannedData:
    """Data class containing banned data information."""

    metadata: dict = dataclasses.field(default_factory=dict)
    """Dataset-dependent map of banned metadata; allows some stuff to be entirely avoided."""
    problems: dict[str, list[int]] = dataclasses.field(default_factory=dict)
    """Generic banned problems indices map.

    For each data subset (e.g. 'train', 'valid', ...), provides a list of banned problem indices.
    """
    solutions: dict[str, dict[int, list[int]]] = dataclasses.field(default_factory=dict)
    """Generic banned solutions indices map.

    For each data subset (e.g. 'train', 'valid', ...), provides a dictionary pairing problem
    indices with a list of banned solution indices.
    """


class ProblemIdPattern(pydantic.BaseModel):
    """Data class containing problem identifier pattern matching information."""

    pattern: str
    """Problem identifier pattern to match with; can be a regex or glob pattern."""
    is_regex: bool
    """Whether the problem identifier pattern is a regex."""


class CodingProblemIterator:
    """Iterator class for iterating over coding problem data from a source dataset."""

    def __init__(
        self,
        dataset_name: str,
        root_data_path: pathlib.Path | str,
        target_problem_pattern: ProblemIdPattern | None = None,
        reformat_code_strings: bool = True,
        validate_code_strings: bool = True,
        allow_banned_samples: bool = False,
        show_progress: bool = False,
    ):
        """Initialize the iterator, validating source dataset name/path.

        Args:
            dataset_name: Name of the source dataset to load problems from.
            root_data_path: Path to the root directory containing the source dataset files.
            target_problem_pattern: Optional pattern to filter problems by their identifier.
            reformat_code_strings: Whether to apply code formatting to parsed solution code strings.
            validate_code_strings: Whether to validate solution code strings before using them.
            allow_banned_samples: Whether to allow loading of banned problems/solutions. Banned
                samples are problems/solutions that are likely to cause errors during parsing or
                execution, and that have been manually identified in `banned_samples.yaml`.
            show_progress: Whether to show a progress bar while iterating.
        """
        if dataset_name not in SUPPORTED_SOURCE_DATASETS:
            raise ValueError(f"unsupported source dataset: {dataset_name}")
        root_data_path = pathlib.Path(root_data_path)
        if not root_data_path.exists():
            raise FileNotFoundError(f"root data path {root_data_path} does not exist")
        if not root_data_path.is_dir():
            raise NotADirectoryError(f"{root_data_path} is not a directory")
        self.dataset_name = dataset_name
        self.root_data_path = root_data_path
        self._target_pattern = target_problem_pattern
        self.reformat_code_strings = reformat_code_strings
        self.validate_code_strings = validate_code_strings
        if allow_banned_samples:
            logger.warning(f"loading banned data for '{dataset_name}' might cause problems later")
            self.banned = _BannedData()  # will be initialized w/ empty maps
        else:
            self.banned = self._load_banned_data()
        self.problems_metadata: list[typing.Any] = self._prepare_problem_metadata()
        self._current_idx = 0
        self._show_progress = show_progress
        self._progress_bar = None

    def _prepare_problem_metadata(self) -> list:
        """Prepares problem metadata for the iterator, loading high-level source data."""
        if self.dataset_name in JSON_BASED_SOURCE_DATASETS:
            # if we're loading JSONs, the 'problem metadata' will be JSONs paths to parse later
            json_file_paths = list(self.root_data_path.glob("*.json"))
            output_paths = []
            # we actually need to pop the files open and check which ones contain any data
            # (repackaging might have resulted in empty JSONs with only an error code)
            for json_file_path in json_file_paths:
                if json_file_path.stat().st_size < 128:
                    continue  # skip tiny files that are likely empty/errored
                if json_file_path.name in self.banned.metadata:
                    continue  # skip banned samples (likely due to code analysis failure)
                # optionally filter by a target pattern
                if self._target_pattern is not None:
                    if self._target_pattern.is_regex and not re.fullmatch(
                        self._target_pattern.pattern, json_file_path.name
                    ):
                        continue
                    elif not self._target_pattern.is_regex and not fnmatch.fnmatch(
                        json_file_path.name, self._target_pattern.pattern
                    ):
                        continue
                with json_file_path.open("r", encoding="utf-8") as fd:
                    json_data = json.load(fd)
                    assert isinstance(json_data, dict)
                    if len(json_data) == 1 and "error" in json_data:
                        continue  # skip this file (useless; prior repackaging failed)
                    output_paths.append(json_file_path)
            assert len(output_paths) > 0, "no valid JSON files found in the dataset root directory"
            return output_paths
        else:
            raise NotImplementedError(f"unsupported source dataset: {self.dataset_name}")

    def _load_banned_data(
        self,
    ) -> _BannedData:
        """Load banned data info for a target dataset."""
        with open(str(BANNED_DATA_YAML_PATH)) as f:
            banned_data = yaml.safe_load(f) or {}
        banned_metadata_all = banned_data.get("banned_metadata", {})
        banned_problems_all = banned_data.get("banned_problems", {})
        banned_solutions_all = banned_data.get("banned_solutions", {})
        target_banned_metadata = banned_metadata_all.get(self.dataset_name, {})
        target_banned_problems = banned_problems_all.get(self.dataset_name, {})
        target_banned_solutions = banned_solutions_all.get(self.dataset_name, {})
        return _BannedData(
            metadata=target_banned_metadata,
            problems=target_banned_problems,
            solutions=target_banned_solutions,
        )

    def _load_problem_data(self, problem_metadata: typing.Any) -> dict[str, typing.Any]:
        """Loads 'raw' data from the source dataset for a specific coding problem."""
        if self.dataset_name in JSON_BASED_SOURCE_DATASETS:
            assert isinstance(problem_metadata, (str, pathlib.Path))
            json_file = pathlib.Path(problem_metadata)
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            data["__root_path__"] = str(json_file)
            data["__root_hash__"] = pyine.utils.reprod.compute_hash(json_file)
            return data
        else:
            raise NotImplementedError(f"unsupported source dataset: {self.dataset_name}")

    def _process_data(
        self,
        problem_data: dict[str, typing.Any],
    ) -> tuple[CodingProblem, list[Solution]]:
        """Converts a 'raw' data dictionary from a source dataset to exportable objects."""
        assert isinstance(problem_data, dict)
        # this is where the dataset-specific logic is implemented; this function will get dirty
        if self.dataset_name == "TACO":
            parsing_errors = []
            if not problem_data or problem_data.get("error", None):
                parsing_errors.append(problem_data.get("error", None))
            problem_id = CodingProblemIdentifier(
                dataset=self.dataset_name,
                subset=problem_data["subset"],
                problem_idx=problem_data["subset_idx"],
            )
            problem_statement = problem_data["question"]
            is_banned = problem_id.problem_idx in self.banned.problems.get(problem_id.subset, [])
            inputs_array = problem_data["input_output"]["inputs"]
            outputs_array = problem_data["input_output"]["outputs"]
            assert isinstance(inputs_array, list) and isinstance(outputs_array, list)
            assert len(inputs_array) == len(outputs_array)
            test_inout_pairs = [(inputs, outputs) for inputs, outputs in zip(inputs_array, outputs_array)]
            entrypoint_name = problem_data["input_output"].get("fn_name", None)

            def _tag_cleaner(x):
                return x.replace(" ", "")

            tags = [
                f"source:{_tag_cleaner(problem_data['source'])}",
                f"difficulty:{_tag_cleaner(problem_data['difficulty'])}",
                f"subset:{_tag_cleaner(problem_data['subset'])}",
            ]
            for tag_group in ["raw_tags", "tags", "skill_types"]:
                tags.extend([f"{tag_group}:{_tag_cleaner(tag)}" for tag in problem_data.get(tag_group, [])])
            solutions, solution_ids = [], []
            if "solutions" not in problem_data or not problem_data["solutions"]:
                parsing_errors.append("no solutions found")
            else:
                rpkgd_solutions = problem_data["solutions"]
                banned_solution_idxs = self.banned.solutions.get(problem_id.subset, {}).get(problem_id.problem_idx, {})
                assert all([isinstance(s, dict) and "code" in s for s in rpkgd_solutions]), "missing repackaged code?"
                for solution_idx, solution in enumerate(rpkgd_solutions):
                    solution_id = SolutionIdentifier(
                        dataset=self.dataset_name,
                        subset=problem_data["subset"],
                        problem_idx=problem_data["subset_idx"],
                        solution_idx=solution_idx,
                    )
                    solution_ids.append(solution_id)
                    analysis_errors = solution.get("validation_errors", [])
                    analysis_results = pyine.prompts.configs.code_analysis.CodeAnalysisResponse.model_validate(
                        solution["analysis_outputs"][-1],  # take the latest analysis result
                    )
                    solution_code = solution["code"]
                    if self.reformat_code_strings:
                        try:
                            solution_code = pyine.utils.code.formatting.format_code(solution_code)
                        except Exception as e:
                            analysis_errors.append(str(e))
                    if self.validate_code_strings:
                        try:
                            pyine.utils.code.validation.validate_code(solution_code)
                        except Exception as e:
                            analysis_errors.append(str(e))
                    solutions.append(
                        Solution(
                            parent_id=problem_id,
                            solution_id=solution_id,
                            code=solution_code,
                            analysis_errors=analysis_errors if analysis_errors else None,
                            analysis_results=analysis_results,
                            is_banned=solution_id.solution_idx in banned_solution_idxs,
                        )
                    )
            coding_problem = CodingProblem(
                source_dataset_name=self.dataset_name,
                source_data_path=problem_data["__root_path__"],
                source_data_hash=problem_data["__root_hash__"],
                problem_id=problem_id,
                problem_statement=problem_statement,
                problem_tags=list(set(tags)),
                test_inout_pairs=test_inout_pairs,
                entrypoint_name=entrypoint_name,
                potential_solution_ids=solution_ids,
                parsing_errors=parsing_errors if parsing_errors else None,
                is_banned=is_banned,
            )
        else:
            raise NotImplementedError(f"unsupported source dataset: {self.dataset_name}")
        return coding_problem, solutions

    def __len__(self):
        """Returns the number of coding problems in the source dataset."""
        return len(self.problems_metadata)

    def __iter__(self):
        """Returns the iterator over all coding problems in the source dataset."""
        self._current_idx = 0
        if self._show_progress:
            self._progress_bar = tqdm.tqdm(total=len(self.problems_metadata))
        return self

    def __next__(self) -> tuple[CodingProblem, list[Solution]]:
        """Returns the next coding problem and solutions object tuple."""
        if self._current_idx >= len(self.problems_metadata):
            if self._progress_bar is not None:
                self._progress_bar.close()
                self._progress_bar = None
            raise StopIteration
        raw_problem_data = self._load_problem_data(self.problems_metadata[self._current_idx])
        problem, solutions = self._process_data(raw_problem_data)
        self._current_idx += 1
        if self._progress_bar is not None:
            self._progress_bar.update(1)
        return problem, solutions

    def reset(self) -> None:
        """Resets the iterator to its initial state."""
        self._current_idx = 0
        if self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None

    def __getitem__(self, idx: int) -> tuple[CodingProblem, list[Solution]]:
        """Returns a coding problem and solutions object tuple for the specified index."""
        if not (0 <= idx < len(self)):
            raise IndexError(f"index {idx} out of range")
        raw_problem_data = self._load_problem_data(self.problems_metadata[idx])
        problem, solutions = self._process_data(raw_problem_data)
        return problem, solutions


def get_latest_dataset_path(source_dataset_name: str) -> pathlib.Path:
    """Returns the path to the latest trace dataset for a specific source dataset.

    If multiple trace datasets are available, the most recent version is returned.
    """
    assert source_dataset_name in SUPPORTED_SOURCE_DATASETS, f"invalid source dataset: {source_dataset_name}"
    traces_root = pyine.utils.filesystem.get_data_root_path() / "traces" / source_dataset_name
    assert traces_root.exists() and traces_root.is_dir(), f"invalid traces dataset path: {traces_root}"
    dataset_paths = list(traces_root.glob("*.[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].lmdb/"))
    if not dataset_paths:
        raise FileNotFoundError(f"No trace datasets found in {traces_root}")
    latest_dataset = max(sorted(dataset_paths))
    return pathlib.Path(latest_dataset)


def get_new_dataset_path(
    source_dataset_name: str,
    dataset_name_tag: str = "v01",
) -> pathlib.Path:
    """Returns the path where a new trace dataset should be saved, for a specific source dataset.

    Will be named based on today's date and using the provided tag (which is like a version).
    """
    assert source_dataset_name in SUPPORTED_SOURCE_DATASETS, f"invalid source dataset: {source_dataset_name}"
    assert dataset_name_tag, "dataset name tag cannot be empty"
    traces_root = pyine.utils.filesystem.get_data_root_path() / "traces" / source_dataset_name
    today = datetime.date.today()
    dataset_name = f"{dataset_name_tag}.{today.strftime('%Y-%m-%d')}.lmdb"
    return traces_root / dataset_name
