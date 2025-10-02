"""Defines data caching utilities for efficient lookups during annotations/transforms/training."""

import dataclasses
import functools
import logging
import pathlib
import typing

import msgspec
import numpy as np

import pyine.data.traces.dataset_reader
import pyine.data.traces.dataset_utils
import pyine.data.traces.dataset_writer
import pyine.utils.filesystem
import pyine.utils.reprod

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class CachedTestData(pyine.data.traces.dataset_writer._TestTuple):  # noqa
    """Container for cached coding problem test case data.

    Inherits `test_idx`, `inputs`, and `outputs` from the parent class.
    """

    @functools.cached_property
    def inputs_length(self) -> int:
        if isinstance(self.inputs, (str, bytes, list, tuple, set, dict)):
            return len(self.inputs)
        return len(str(self.inputs))

    @functools.cached_property
    def outputs_length(self) -> int:
        if isinstance(self.outputs, (str, bytes, list, tuple, set, dict)):
            return len(self.outputs)
        return len(str(self.outputs))

    @functools.cached_property
    def inputs_signature(self) -> str:
        if isinstance(self.inputs, (int, float)):
            return "number"
        return type(self.inputs).__name__

    @functools.cached_property
    def outputs_signature(self) -> str:
        if isinstance(self.outputs, (int, float)):
            return "number"
        return type(self.outputs).__name__


class CodingProblemTestDataCache:
    """Caches test cases for coding problems to help lookups during annotations/transforms/training.

    This class relies on the `CodingProblemIterator` to provide the test cases at construction time.
    It also caches the test cases locally in a temporary directory, under the assumption that the
    dataset is static. If this is not the case, you should turn off `use_cache_storage`.
    """

    def __init__(
        self,
        *,
        dataset_name: str,
        dataset_path: pathlib.Path,
        random_seed: int | None = 0,
        use_cache_storage: bool = True,
    ) -> None:
        """Initializes the cache."""
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.dataset_hash = pyine.utils.reprod.compute_hash(self.dataset_path)
        self._rng = np.random.RandomState(random_seed)
        self._cache: dict[
            pyine.data.traces.dataset_utils.CodingProblemIdentifier,
            list[CachedTestData],
        ] = {}
        if use_cache_storage and self._stored_cache_available:
            self._load_stored_cache()
        else:
            iterator = pyine.data.traces.dataset_utils.CodingProblemIterator(
                dataset_name=dataset_name,
                root_data_path=dataset_path,
                allow_banned_samples=False,
                reformat_code_strings=False,
                validate_code_strings=False,
                show_progress=True,
                enable_async_prefetch=False,
            )
            logger.info(f"preparing test cases cache for '{self.dataset_name}' dataset...")
            self._cache = self._prepare_cache_using_iterator(iterator)
            if use_cache_storage:
                self._save_prepared_metadata()
        if not self._cache:
            raise ValueError("no test cases found in dataset?")

    @staticmethod
    def _prepare_cache_using_iterator(
        iterator: typing.Iterable[tuple[pyine.data.traces.dataset_utils.CodingProblem, typing.Any]],
    ) -> dict[pyine.data.traces.dataset_utils.CodingProblemIdentifier, list[CachedTestData]]:
        cache: dict[
            pyine.data.traces.dataset_utils.CodingProblemIdentifier,
            list[CachedTestData],
        ] = {}
        for problem, _solutions in iterator:
            assert problem.problem_id not in cache, "bug in iterator or non-unique problem ids?"
            cache[problem.problem_id] = []
            for test_idx, (test_inputs, test_outputs) in enumerate(problem.test_inout_pairs):
                cache[problem.problem_id].append(
                    CachedTestData(
                        test_idx=test_idx,
                        inputs=test_inputs,
                        outputs=test_outputs,
                    )
                )
        return cache

    def _save_prepared_metadata(self) -> None:
        """Saves the prepared cache to the logs directory."""
        # we will serialized a cache dict with strings instead of identifiers
        cache_data = {str(pid): test_cases for pid, test_cases in self._cache.items()}
        encoded_data = msgspec.msgpack.encode(cache_data)
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._cache_path, "wb") as fd:
            fd.write(encoded_data)

    def _load_stored_cache(self) -> None:
        """Loads the prepared cache data from the logs directory."""
        if not self._stored_cache_available:
            raise ValueError("stored cache not available")
        with open(self._cache_path, "rb") as fd:
            encoded_data = fd.read()
        cache_data = msgspec.msgpack.decode(encoded_data)
        self._cache = {
            pyine.data.traces.dataset_utils.CodingProblemIdentifier.from_string(pid): (
                [CachedTestData(**t) for t in tests]
            )
            for pid, tests in cache_data.items()
        }

    def _clear_stored_cache(self) -> None:
        """Clears the prepared metadata from the logs directory."""
        if self._stored_cache_available:
            self._cache_path.unlink()

    @property
    def _stored_cache_available(self) -> bool:
        """Returns True if the cache is available locally."""
        return self._cache_path.is_file()

    @functools.cached_property
    def _cache_path(self) -> pathlib.Path:
        """Returns the file path used to store cache data in the logs directory."""
        params_hash = pyine.utils.reprod.get_params_hash(self.dataset_path, self.dataset_hash)
        root = pyine.utils.filesystem.get_data_root_path()
        return root / "cache" / f"testcases-cache.{self.dataset_name}.{params_hash}.msgspec"

    @classmethod
    def build_from_dataset(
        cls,
        dataset_reader: pyine.data.traces.dataset_reader.DatasetReader,
    ) -> "CodingProblemTestDataCache | None":
        """Construct a cache of test case data from dataset metadata (if possible)."""
        assert isinstance(dataset_reader.metadata, dict), f"unexpected metadata type: {type(dataset_reader.metadata)}"
        parent_info = dataset_reader.metadata.get("parent_dataset", {})
        assert isinstance(parent_info, dict), f"unexpected parent dataset info type: {type(parent_info)}"
        dataset_name = parent_info.get("dataset_name")
        dataset_path = parent_info.get("dataset_path")
        if not dataset_name or not dataset_path:
            raise ValueError("dataset metadata missing parent dataset information required to build cache")
        dataset_path = pathlib.Path(dataset_path)
        if not dataset_path.exists():
            raise ValueError(f"dataset path '{dataset_path}' does not exist, cannot build cache")
        return cls(dataset_name=dataset_name, dataset_path=dataset_path)

    def sample_alternative_test_case(
        self,
        trace_id: pyine.data.traces.dataset_utils.TraceIdentifier,
        max_inputs_length_delta: int | None = None,
        max_outputs_length_delta: int | None = None,
        must_be_different: bool = True,
        match_inputs_signature: bool = True,
        match_outputs_signature: bool = True,
        sample_from_top_k: int | None = None,
        rng: np.random.RandomState | None = None,  # if none, will use internal one
    ) -> CachedTestData | None:
        """Return an alternative test case to the one used in the provided trace.

        There will be no guarantee that the returned test case is contained in the trace dataset;
        only that it exists in the original source dataset and that it is different from the input.

        If no alternative test case is found, returns None.

        Args:
            trace_id: Identifier of the trace to sample an alternative test case for.
            max_inputs_length_delta: Maximum allowed length difference between the inputs of the
                original test case and those of the returned alternative test case.
            max_outputs_length_delta: Maximum allowed length difference between the outputs of the
                original test case and those of the returned alternative test case.
            must_be_different: Whether to require the returned alternative test case to be different
                from the original test case (in terms both of exact inputs and outputs).
            match_inputs_signature: Whether to match the input signature of the original test case.
            match_outputs_signature: Whether to match the output signature of the original test case.
            sample_from_top_k: If provided, only sample from the top-k candidates.
            rng: Random number generator to use for sampling. If None, will use the internal one.

        Returns:
            The alternative test case, or None if no valid alternative test case was found.
        """
        assert isinstance(trace_id, pyine.data.traces.dataset_utils.TraceIdentifier)
        problem_id = trace_id.get_parent_identifier().get_parent_identifier()
        assert isinstance(problem_id, pyine.data.traces.dataset_utils.CodingProblemIdentifier)
        if problem_id not in self._cache:
            raise ValueError(f"problem id '{problem_id}' not found in cache")
        orig_test_idx = trace_id.test_idx
        assert isinstance(orig_test_idx, int) and len(self._cache[problem_id]) > orig_test_idx
        orig_test_case = self._cache[problem_id][orig_test_idx]
        candidates = [c for c in self._cache[problem_id] if c.test_idx != orig_test_idx]
        if must_be_different:
            candidates = [
                c for c in candidates if c.inputs != orig_test_case.inputs and c.outputs != orig_test_case.outputs
            ]
        if match_inputs_signature:
            candidates = [c for c in candidates if c.inputs_signature == orig_test_case.inputs_signature]
        if match_outputs_signature:
            candidates = [c for c in candidates if c.outputs_signature == orig_test_case.outputs_signature]
        if max_inputs_length_delta is not None:
            candidates = [
                c for c in candidates if abs(orig_test_case.inputs_length - c.inputs_length) <= max_inputs_length_delta
            ]
        if max_outputs_length_delta is not None:
            candidates = [
                c
                for c in candidates
                if abs(orig_test_case.outputs_length - c.outputs_length) <= max_outputs_length_delta
            ]
        if sample_from_top_k is not None:
            if sample_from_top_k < 1:
                raise ValueError("sample_from_top_k must be at least 1")
            candidates = sorted(
                candidates,
                key=lambda cand: abs(orig_test_case.outputs_length - cand.outputs_length),
            )
            candidates = candidates[:sample_from_top_k]
        if not candidates:
            return None
        if rng is None:
            rng = self._rng
        return rng.choice(candidates)


if __name__ == "__main__":
    import pyine.data.taco.dataset_utils

    dataset_path = pyine.data.taco.dataset_utils.get_latest_repackaged_dataset_path()
    assert len(CodingProblemTestDataCache(dataset_name="TACO", dataset_path=dataset_path)._cache) > 0  # noqa
