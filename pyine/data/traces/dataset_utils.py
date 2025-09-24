import dataclasses
import datetime
import fnmatch
import functools
import importlib.resources as pkg_resources
import json
import logging
import pathlib
import queue
import re
import threading
import typing

import numpy as np
import orjson
import pydantic
import tqdm
import yaml

import pyine.data.common
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
    "get_matching_dataset_paths",
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

    def _get_augmentless_repr(self) -> str:
        """Returns a string representation of this identifier without the augmentation information."""
        return f"{SolutionIdentifier.__repr__(self)}/t{self.test_idx:04d}"

    def __repr__(self) -> str:
        """Returns a string representation of this identifier."""
        out = self._get_augmentless_repr()
        if self.is_augmented:
            out += f"/a:{self.augment_category}:{self.augment_idx:03d}"
        return out

    def get_parent_identifier(self) -> SolutionIdentifier:
        """Returns the parent identifier of this object (i.e., a solution identifier)."""
        local_vars = ["test_idx", "augment_category", "augment_idx"]
        parent_vars = {var_name: var_val for var_name, var_val in vars(self).items() if var_name not in local_vars}
        return SolutionIdentifier(**parent_vars)

    def get_augmentless_identifier(self) -> "TraceIdentifier":
        """Returns a copy of this object without the augmentation information."""
        augmentless_repr = self._get_augmentless_repr()
        return self.from_string(augmentless_repr)

    @functools.cached_property
    def is_augmented(self) -> bool:
        """Returns whether this trace is based on 'augmented' (modified) code."""
        if self.augment_category is not None or self.augment_idx is not None:
            assert (
                self.augment_category is not None and self.augment_idx is not None
            ), "if augmentation is present, both augment category and index must be present"
            assert not any(
                [c in self.augment_category for c in ["/", ",", " ", ":"]]
            ), f"augm category should have been cleaned up: {self.augment_category}"
            return True
        return False

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

    @functools.cached_property
    def is_bugged(self) -> bool:
        """Returns whether this trace is based on bugged code.

        The checked names herein relate to prompt definitions (see `pyine.prompts`) and sample type
        definitions (see `pyine.organisms.datamodules.utils.samples`).
        """
        return self.is_augmented and (
            # bugged traces are those generated with issues (except misleading docs)
            (self.augment_category.startswith("issues_") and self.augment_category != "issues_docs")
            or "bugged" in self.augment_category
        )

    @functools.cached_property
    def is_hinted(self) -> bool:
        """Returns whether this trace is based on code with helpful hints about code execution.

        The checked names herein relate to prompt definitions (see `pyine.prompts`) and sample type
        definitions (see `pyine.organisms.datamodules.utils.samples`).
        """
        if not self.is_augmented:
            return False
        assert self.augment_category != "hints_stubs", "how can we have a stubbed trace? (those can't be executed)"
        return self.augment_category.startswith("hints_") or "hinted" in self.augment_category

    @functools.cached_property
    def is_misleading(self) -> bool:
        """Returns whether this trace is based on code with misleading hints about code execution.

        The checked names herein relate to prompt definitions (see `pyine.prompts`) and sample type
        definitions (see `pyine.organisms.datamodules.utils.samples`).
        """
        return self.is_augmented and (self.augment_category == "issues_docs" or "misleading" in self.augment_category)

    @functools.cached_property
    def is_obfuscated(self) -> bool:
        """Returns whether this trace is based on obfuscated code."""
        return self.is_augmented and "obfuscated" in self.augment_category

    @staticmethod
    def get_clean_augment_category(proposed: str) -> str:
        """Cleans up an augmentation category string to be used as a trace identifier."""
        # this will help avoid conflicts later when parsing augment names and generating augment tags
        for banned_ch in ["/", ",", " ", ":"]:
            proposed = proposed.replace(banned_ch, "_")
        return proposed


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
    """Iterator class for iterating over coding problem data from a source dataset.

    Usage example:
    >>> from pyine.data.traces.dataset_utils import CodingProblemIterator
    >>>
    >>> problem_iterator =  CodingProblemIterator(
    >>>     dataset_name="TACO",
    >>>     root_data_path="<path_to_the_repackaged_TACO_dataset_folder>",
    >>> )
    >>>
    >>> for problem, solutions in problem_iterator:  # yields tuple[CodingProblem, list[Solution]]
    >>>     print(f"{problem.problem_id=}")
    >>>     for solution in solutions:
    >>>         print(f"{solution.solution_id=}")
    >>>     # ...
    """

    def __init__(
        self,
        dataset_name: str,
        root_data_path: pathlib.Path | str,
        target_problem_pattern: ProblemIdPattern | None = None,
        target_problem_ids: str | pathlib.Path | list[str] | list[int] | None = None,
        reformat_code_strings: bool = False,
        validate_code_strings: bool = True,
        allow_banned_samples: bool = False,
        add_orig_subset_as_tag: bool = False,
        show_progress: bool = False,
        enable_async_prefetch: bool = False,
        prefetch_cache_size: int = 8,
    ):
        """Initialize the iterator, validating source dataset name/path.

        Args:
            dataset_name: Name of the source dataset to load problems from.
            root_data_path: Path to the root directory containing the source dataset files.
            target_problem_pattern: Optional pattern to filter problems by their metadata file names.
            target_problem_ids: Optional list or file containing problem IDs to target.
            reformat_code_strings: Whether to apply code formatting to parsed solution code strings.
            validate_code_strings: Whether to validate solution code strings before using them.
            allow_banned_samples: Whether to allow loading of banned problems/solutions. Banned
                samples are problems/solutions that are likely to cause errors during parsing or
                execution, and that have been manually identified in `banned_samples.yaml`.
            add_orig_subset_as_tag: Whether to add the original subset name as a tag to the
                tags list of each sample. Useful when you want to filter problems by their original
                subset, but may be distracting if you intend to create a new split.
            show_progress: Whether to show a progress bar while iterating.
            enable_async_prefetch: If True, enable asynchronous prefetching of samples.
            prefetch_cache_size: Bounded cache size (>=1) for prefetched samples.
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
        self._target_problem_ids = self._validate_target_problem_ids(target_problem_ids)
        self.reformat_code_strings = reformat_code_strings
        self.validate_code_strings = validate_code_strings
        if allow_banned_samples:
            logger.warning(f"loading banned data for '{dataset_name}' might cause problems later")
            self.banned = _BannedData()  # will be initialized w/ empty maps
        else:
            self.banned = self._load_banned_data()
        self.add_orig_subset_as_tag = add_orig_subset_as_tag
        self.problems_metadata: list[typing.Any] = self._prepare_problem_metadata()
        self._current_idx = 0
        self._show_progress = show_progress
        self._progress_bar = None
        # async prefetch configuration/state
        self._use_prefetch: bool = bool(enable_async_prefetch)
        self._prefetch_cache_size: int = int(prefetch_cache_size)
        if self._use_prefetch and self._prefetch_cache_size <= 0:
            raise ValueError("prefetch_cache_size must be >= 1 when async prefetch is enabled")
        self._prefetch_queue: queue.Queue | None = None
        self._prefetch_thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._prefetch_sentinel: object = object()

    def _validate_target_problem_ids(
        self, target_problem_ids: str | pathlib.Path | list[str] | list[int] | None
    ) -> list[CodingProblemIdentifier]:
        """Validates and resolves the target problem IDs, if needed."""
        if isinstance(target_problem_ids, (str, pathlib.Path)):
            target_problem_ids_path = pathlib.Path(target_problem_ids)
            if not target_problem_ids_path.is_file():
                raise FileNotFoundError(f"target problem IDs file not found: {target_problem_ids_path}")
            if target_problem_ids_path.name.endswith(".json"):
                with target_problem_ids_path.open("r") as fd:
                    target_problem_ids = json.load(fd)
            elif target_problem_ids_path.name.endswith(".yaml"):
                with target_problem_ids_path.open("r") as fd:
                    target_problem_ids = yaml.safe_load(fd)
            elif target_problem_ids_path.name.endswith(".txt"):
                with target_problem_ids_path.open("r") as fd:
                    target_problem_ids = fd.read().splitlines()
        elif target_problem_ids is None:
            target_problem_ids: list[str] = []
        if not isinstance(target_problem_ids, list):
            raise ValueError(f"target problem IDs must be list, str, or path; got {type(target_problem_ids)}")
        output_problem_ids: list[CodingProblemIdentifier] = []
        for prob_idx, prob_id in enumerate(target_problem_ids):
            if not isinstance(prob_id, (str, int)):
                raise ValueError(f"target problem IDs must be string or int; got {type(prob_id)}")
            if isinstance(prob_id, str):
                if "/" in prob_id:  # these are full problem identifiers; replace them
                    prob_id = CodingProblemIdentifier.from_string(prob_id)
                    assert prob_id.dataset == self.dataset_name, f"unexpected dataset: {prob_id.dataset}"
                    output_problem_ids.append(prob_id)
                    continue
            # coerce the identifier to an integer
            output_problem_ids.append(
                CodingProblemIdentifier(
                    self.dataset_name,
                    subset="UNKNOWN",  # we will have to hope that the problem idx is unique across subsets
                    problem_idx=int(prob_id),  # if this fails, we might need to revisit the problem idx type
                )
            )
        return output_problem_ids

    def _start_prefetch(self) -> None:
        """Start the background prefetch worker if enabled."""
        if not self._use_prefetch:
            return
        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            return
        logger.info("starting coding problem iterator async prefetch worker")
        self._prefetch_queue = queue.Queue(maxsize=self._prefetch_cache_size)
        self._stop_event = threading.Event()
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            name=f"CodingProblemIteratorPrefetch[{self.dataset_name}]",
            daemon=True,
        )
        self._prefetch_thread.start()

    def _stop_prefetching(self) -> None:
        """Signal the prefetch worker to stop and clean up resources."""
        if self._stop_event is not None:
            self._stop_event.set()
        if self._prefetch_thread is not None:
            logger.info("stopping coding problem iterator async prefetch worker")
            # short join to avoid blocking shutdowns too long
            self._prefetch_thread.join(timeout=0.1)
        self._prefetch_thread = None
        self._stop_event = None
        self._prefetch_queue = None

    def _prefetch_worker(self) -> None:
        """Worker that preloads items into a bounded queue in order."""
        assert self._prefetch_queue is not None
        exc: Exception | None = None
        try:
            for idx in range(len(self.problems_metadata)):
                if self._stop_event is not None and self._stop_event.is_set():
                    break
                pm = self.problems_metadata[idx]
                raw_problem_data = self._load_problem_data(pm)
                item = self._process_data(raw_problem_data)  # (problem, solutions)
                self._prefetch_queue.put(item)  # blocks if cache full
        except Exception as e:
            exc = e
        finally:
            # signal completion or error to the consumer
            self._prefetch_queue.put((self._prefetch_sentinel, exc))

    def __del__(self):
        """Best-effort cleanup of background resources."""
        try:
            self._stop_prefetching()
        except Exception:
            pass

    def _prepare_problem_metadata(self) -> list:
        """Prepares problem metadata for the iterator, loading high-level source data."""
        assert self.dataset_name in SUPPORTED_SOURCE_DATASETS
        if self.dataset_name == "TACO":
            # with TACO we're loading JSONs: the 'problem metadata' are JSONs paths to parse later
            json_file_paths = list(sorted(self.root_data_path.glob("*.json")))
            if not json_file_paths:
                raise FileNotFoundError(f"no JSON files found in the dataset root directory: {self.root_data_path}")
            output_paths = []
            # we actually need to pop the files open and check which ones contain any data
            # (repackaging might have resulted in empty JSONs with only an error code)
            for json_file_path in json_file_paths:
                if json_file_path.stat().st_size < 128:
                    continue  # skip tiny files that are likely empty/errored
                problem_idx = int(json_file_path.stem)  # for TACO, should be a unique problem number
                if self.banned.metadata and problem_idx in self.banned.metadata:
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
                # optionally filter by target problem idx list
                if self._target_problem_ids:
                    # (assumes the problem idxs in the target problem ids are unique across subsets)
                    if not any([problem_idx == pid.problem_idx for pid in self._target_problem_ids]):
                        continue
                with json_file_path.open("r", encoding="utf-8") as fd:
                    try:
                        json_data = orjson.loads(fd.read())
                    except orjson.JSONDecodeError as e:
                        logger.warning(f"skipping invalid JSON file: {json_file_path} ({e})")
                        continue  # there's likely something not escaped probably in the file
                assert isinstance(json_data, dict)
                if len(json_data) == 1 and "error" in json_data:
                    logger.debug(f"skipping empty JSON file: {json_file_path}")
                    continue  # skip this file (useless; prior repackaging failed)
                output_paths.append(json_file_path)
            if not output_paths:
                raise ValueError("no valid JSON files left after filtering")
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
        assert self.dataset_name in SUPPORTED_SOURCE_DATASETS
        if self.dataset_name == "TACO":
            assert isinstance(problem_metadata, (str, pathlib.Path))
            json_file = pathlib.Path(problem_metadata)
            with open(json_file, encoding="utf-8") as fd:
                data = orjson.loads(fd.read())
            data["__root_path__"] = str(json_file)
            data["__root_hash__"] = pyine.utils.reprod.compute_hash(json_file)
            data["__problem_idx__"] = int(json_file.stem)  # for TACO, should be a unique problem number
            return data
        else:
            raise NotImplementedError(f"unsupported source dataset: {self.dataset_name}")

    def _process_data(
        self,
        problem_data: dict[str, typing.Any],
    ) -> tuple[CodingProblem, list[Solution]]:
        """Converts a 'raw' data dictionary from a source dataset to exportable objects."""
        assert isinstance(problem_data, dict)
        assert self.dataset_name in SUPPORTED_SOURCE_DATASETS
        if self.dataset_name == "TACO":
            parsing_errors = []
            if not problem_data or problem_data.get("error", None):
                parsing_errors.append(problem_data.get("error", None))
            problem_idx = problem_data["__problem_idx__"]  # for TACO, this is a unique id across subsets
            problem_id = CodingProblemIdentifier(
                dataset=self.dataset_name,
                subset=problem_data["subset"],
                problem_idx=problem_idx,
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
            ]
            if self.add_orig_subset_as_tag:
                tags.append(f"subset:{_tag_cleaner(problem_data['subset'])}")
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
                        problem_idx=problem_idx,
                        solution_idx=solution_idx,
                    )
                    solution_ids.append(solution_id)
                    analysis_errors = solution.get("validation_errors", [])
                    try:
                        analysis_results = pyine.prompts.configs.code_analysis.CodeAnalysisResponse.model_validate(
                            solution["analysis_outputs"][-1],  # take the latest analysis result
                        )
                    except Exception as e:
                        raise ValueError(f"invalid analysis results for: {solution_id}") from e
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
            self._progress_bar = tqdm.tqdm(total=len(self.problems_metadata), smoothing=0.1)
        if self._use_prefetch:
            # ensure a clean start for each iteration
            self._stop_prefetching()
            self._start_prefetch()
        return self

    def __next__(self) -> tuple[CodingProblem, list[Solution]]:
        """Returns the next coding problem and solutions object tuple."""
        if self._use_prefetch and self._prefetch_queue is not None:
            item = self._prefetch_queue.get()
            # check for sentinel indicating completion or error
            if isinstance(item, tuple) and len(item) == 2 and item[0] is self._prefetch_sentinel:
                _, exc = item
                if self._progress_bar is not None:
                    self._progress_bar.close()
                    self._progress_bar = None
                self._stop_prefetching()
                if exc is not None:
                    raise exc
                raise StopIteration
            problem, solutions = item
            if self._progress_bar is not None:
                self._progress_bar.update(1)
            return problem, solutions
        # fallback to on-demand loading when prefetch is disabled
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
        if self._use_prefetch:
            self._stop_prefetching()

    def __getitem__(self, idx: int) -> tuple[CodingProblem, list[Solution]]:
        """Returns a coding problem and solutions object tuple for the specified index."""
        if not (0 <= idx < len(self)):
            raise IndexError(f"index {idx} out of range")
        raw_problem_data = self._load_problem_data(self.problems_metadata[idx])
        problem, solutions = self._process_data(raw_problem_data)
        return problem, solutions


def get_latest_dataset_path(
    source_dataset_name: str,
    filter_rule: str | None = None,
) -> pathlib.Path:
    """Returns the path to the latest trace dataset for a specific source dataset.

    If multiple deltas datasets are available, the most recent version is returned, where we pick
    strictly by the date suffix (YYYY-MM-DD) in the directory name, ignoring any prefix tag.
    Optionally, a filter rule can be provided to exclude directories before selection.

    Raises a FileNotFoundError exception if no matching dataset is found.
    """
    assert source_dataset_name in SUPPORTED_SOURCE_DATASETS, f"invalid source dataset: {source_dataset_name}"
    return pyine.data.common.resolve_latest_dataset_path(
        kind="traces",
        source_dataset_name=source_dataset_name,
        filter_rule=filter_rule,
    )


def get_matching_dataset_paths(
    source_dataset_name: str,
    pattern: str,
    is_regex: bool = False,
) -> list[pathlib.Path]:
    """Return all trace dataset directories for a source that match the provided pattern.

    Matching modes:
      - Glob/fnmatch (default): e.g., 'mytag.*.lmdb', '*-08-*.lmdb'.
      - Regex: set is_regex=True or prefix the pattern with 're:'/'regex:' to use Python regex.
               Prefix 'glob:'/'fnmatch:' can force glob mode.

    Returns a sorted list which may be empty if no matches are found.
    """
    assert source_dataset_name in SUPPORTED_SOURCE_DATASETS, f"invalid source dataset: {source_dataset_name}"
    return pyine.data.common.resolve_matching_dataset_paths(
        kind="traces",
        source_dataset_name=source_dataset_name,
        pattern=pattern,
        pattern_is_regex=is_regex,
    )


def get_new_dataset_path(
    source_dataset_name: str,
    dataset_name_tag: str,
) -> pathlib.Path:
    """Returns the path where a new trace dataset should be saved, for a specific source dataset.

    Will be named based on today's date and using the provided tag (which is the file name prefix).
    """
    assert source_dataset_name in SUPPORTED_SOURCE_DATASETS, f"invalid source dataset: {source_dataset_name}"
    assert dataset_name_tag, "dataset name tag cannot be empty"
    traces_root = pyine.utils.filesystem.get_data_root_path() / "traces" / source_dataset_name
    today = datetime.date.today()
    dataset_name = f"{dataset_name_tag}.{today.strftime('%Y-%m-%d')}.lmdb"
    return traces_root / dataset_name
