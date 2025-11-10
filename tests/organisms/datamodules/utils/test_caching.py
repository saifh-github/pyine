import pathlib

import numpy as np
import pytest

import pyine.data.taco.dataset_utils
import pyine.data.traces.dataset_utils as traces_utils
import pyine.organisms.datamodules.utils.caching as caching
import tests.env_checks


def make_cached_test_data_list(
    values: list[tuple[object, object]],
) -> list[caching.CachedTestData]:
    return [
        caching.CachedTestData(
            test_idx=idx,
            inputs=inp,
            outputs=out,
        )
        for idx, (inp, out) in enumerate(values)
    ]


class TestCachedTestData:
    def test_length_and_signature_for_basic_types(self) -> None:
        a = caching.CachedTestData(test_idx=0, inputs=123, outputs=45.6)
        b = caching.CachedTestData(test_idx=1, inputs="abc", outputs=[1, 2, 3])
        c = caching.CachedTestData(test_idx=2, inputs={"x": 1, "y": 2}, outputs=(1, 2))
        # check lengths
        assert a.inputs_length == len(str(123))
        assert a.outputs_length == len(str(45.6))
        assert b.inputs_length == 3
        assert b.outputs_length == 3
        assert c.inputs_length == 2
        assert c.outputs_length == 2
        # check signatures
        assert a.inputs_signature == "number"
        assert a.outputs_signature == "number"
        assert b.inputs_signature == "str"
        assert b.outputs_signature == "list"
        assert c.inputs_signature == "dict"
        assert c.outputs_signature == "tuple"


class TestCacheStorage:
    def test_save_load_and_clear_stored_cache_roundtrip(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: pathlib.Path,
    ) -> None:
        # force deterministic tmp path and params hash so both instances hit the same file
        monkeypatch.setattr("pyine.utils.filesystem.get_data_root_path", lambda: tmp_path)
        monkeypatch.setattr("pyine.utils.reprod.get_params_hash", lambda *_a, **_k: "ROUNDTRIP")
        monkeypatch.setattr("pyine.utils.reprod.compute_hash", lambda *_a, **_k: "FAKE")
        dataset_name = "FAKE"
        dataset_path = tmp_path / "dataset-root"
        dataset_path.mkdir(parents=True, exist_ok=True)

        # build a synthetic cache mapping with multiple candidates for one problem
        pid = traces_utils.CodingProblemIdentifier(dataset="FAKE", subset="unit", problem_idx=7)
        data = make_cached_test_data_list(
            [
                (1, 2),  # test_idx 0
                ("ab", "cd"),  # test_idx 1
                (1000, 2000),  # test_idx 2
            ]
        )
        prepared_cache: dict[traces_utils.CodingProblemIdentifier, list[caching.CachedTestData]] = {pid: data}

        # 1) write using a minimally-initialized instance to avoid invoking iterator logic in __init__
        writer = object.__new__(caching.CodingProblemTestDataCache)
        writer.dataset_name = dataset_name
        writer.dataset_path = dataset_path
        writer.dataset_hash = "FAKE"
        writer._cache = prepared_cache
        writer._save_prepared_metadata()

        # ensure file exists
        expected_path = tmp_path / "cache" / f"testcases-cache.{dataset_name}.ROUNDTRIP.msgspec"
        assert expected_path.is_file()

        # 2) instantiate the class normally; it should load the stored cache (no iterator needed)
        cache = caching.CodingProblemTestDataCache(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            use_cache_storage=True,
        )
        assert pid in cache._cache
        assert [t.test_idx for t in cache._cache[pid]] == [0, 1, 2]

        # 3) clear the stored cache and verify file removal
        cache._clear_stored_cache()
        assert not expected_path.exists()
        assert not cache._stored_cache_available


class TestSampling:
    def _make_cache_single_problem(
        self,
    ) -> tuple[caching.CodingProblemTestDataCache, traces_utils.CodingProblemIdentifier]:
        # construct an object whose cache we control without invoking iterator in __init__
        obj = object.__new__(caching.CodingProblemTestDataCache)
        obj.dataset_name = "FAKE"
        obj.dataset_path = pathlib.Path("/dev/null")  # not used by sampling
        obj._rng = np.random.RandomState(0)
        pid = traces_utils.CodingProblemIdentifier(dataset="FAKE", subset="unit", problem_idx=1)
        # include four test cases with varied lengths and signatures
        # indices: 0..3
        obj._cache = {
            pid: make_cached_test_data_list(
                [
                    (1, 2),  # number -> number; short lengths
                    ("xx", "yyyy"),  # str -> str; lengths 2/4
                    (100, 200),  # number -> number; longer lengths
                    ([1, 2], [3]),  # list -> list
                ]
            )
        }
        return obj, pid

    def test_basic_alternative_sampling(self) -> None:
        cache, pid = self._make_cache_single_problem()
        trace_id = traces_utils.TraceIdentifier(
            dataset=pid.dataset,
            subset=pid.subset,
            problem_idx=pid.problem_idx,
            solution_idx=0,
            test_idx=0,  # original is index 0
        )
        alt = cache.sample_alternative_test_case(
            trace_id=trace_id,
            must_be_different=True,
        )
        assert alt is not None
        assert alt.test_idx != 0

    def test_respects_signature_and_length_filters(self) -> None:
        cache, pid = self._make_cache_single_problem()
        # original test_idx=0: inputs_signature="number", outputs_signature="number", short lengths
        trace_id = traces_utils.TraceIdentifier(
            dataset=pid.dataset,
            subset=pid.subset,
            problem_idx=pid.problem_idx,
            solution_idx=0,
            test_idx=0,
        )
        # With both signature matches ON and tight deltas, only candidate test_idx=2 (number/number) might fit.
        # Set very small deltas to exclude test_idx=2 as well (lengths differ by > 0 for outputs).
        alt_none = cache.sample_alternative_test_case(
            trace_id=trace_id,
            max_inputs_length_delta=0,
            max_outputs_length_delta=0,
            match_inputs_signature=True,
            match_outputs_signature=True,
        )
        assert alt_none is None

        # Relax outputs delta to allow index 2; inputs delta still 0 but 1 vs 3 digits excludes it; relax inputs too.
        alt_ok = cache.sample_alternative_test_case(
            trace_id=trace_id,
            max_inputs_length_delta=5,
            max_outputs_length_delta=5,
            match_inputs_signature=True,
            match_outputs_signature=True,
            rng=np.random.RandomState(123),
        )
        assert alt_ok is not None
        assert alt_ok.inputs_signature == "number" and alt_ok.outputs_signature == "number"

    def test_sample_from_top_k_and_rng(self) -> None:
        cache, pid = self._make_cache_single_problem()
        # choose original with outputs length 4 (test_idx=1: "yyyy")
        trace_id = traces_utils.TraceIdentifier(
            dataset=pid.dataset,
            subset=pid.subset,
            problem_idx=pid.problem_idx,
            solution_idx=0,
            test_idx=1,
        )
        with pytest.raises(ValueError):
            cache.sample_alternative_test_case(
                trace_id=trace_id,
                sample_from_top_k=0,
            )
        # disable signature match to allow any, and pick deterministically from top-1 by RNG
        alt = cache.sample_alternative_test_case(
            trace_id=trace_id,
            match_inputs_signature=False,
            match_outputs_signature=False,
            sample_from_top_k=1,
            rng=np.random.RandomState(0),
        )
        assert alt is not None
        assert alt.test_idx in {0, 2, 3}

    def test_problem_id_missing_raises(self) -> None:
        cache, _ = self._make_cache_single_problem()
        missing_pid = traces_utils.CodingProblemIdentifier(dataset="FAKE", subset="unit", problem_idx=999)
        trace_id = traces_utils.TraceIdentifier(
            dataset=missing_pid.dataset,
            subset=missing_pid.subset,
            problem_idx=missing_pid.problem_idx,
            solution_idx=0,
            test_idx=0,
        )
        with pytest.raises(ValueError, match="not found in cache"):
            cache.sample_alternative_test_case(trace_id=trace_id)


class TestBuildFromDatasetReader:
    def test_build_from_reader_validates_metadata_and_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: pathlib.Path,
    ) -> None:
        # prepare a fake reader that returns the expected metadata shape
        class _FakeReader:
            def __init__(self, dataset_name: str, dataset_path: pathlib.Path) -> None:
                self._dataset_name = dataset_name
                self._dataset_path = dataset_path

            @property
            def metadata(self) -> dict:
                return {
                    "parent_dataset": {
                        "dataset_name": self._dataset_name,
                        "dataset_path": str(self._dataset_path),
                    }
                }

        # monkeypatch tmp dir/hash so that any potential storage operations resolve to tmp_path
        monkeypatch.setattr("pyine.utils.filesystem.get_logs_root_path", lambda: tmp_path)
        monkeypatch.setattr("pyine.utils.reprod.get_params_hash", lambda *_a, **_k: "BUILD")
        monkeypatch.setattr("pyine.utils.reprod.compute_hash", lambda *_a, **_k: "FAKE")
        # create a real directory to satisfy the existence check
        ds_root = tmp_path / "source"
        ds_root.mkdir(parents=True, exist_ok=True)

        reader = _FakeReader(dataset_name="FAKE", dataset_path=ds_root)
        # avoid triggering real iterator by pre-seeding the stored cache file first
        # build a minimal object to write a small cache file at the expected path
        pid = traces_utils.CodingProblemIdentifier(dataset="FAKE", subset="unit", problem_idx=0)
        prewriter = object.__new__(caching.CodingProblemTestDataCache)
        prewriter.dataset_name = "FAKE"
        prewriter.dataset_path = ds_root
        prewriter.dataset_hash = "FAKE"
        prewriter._cache = {pid: make_cached_test_data_list([(1, 2)])}
        prewriter._save_prepared_metadata()

        # now build from reader; should load the prewritten cache without errors
        cache = caching.CodingProblemTestDataCache.build_from_dataset(reader)
        assert isinstance(cache, caching.CodingProblemTestDataCache)
        # sanity: the loaded cache has our single problem id
        assert pid in cache._cache

    def test_build_from_reader_fails_on_missing_metadata(self) -> None:
        class _BadReader:
            @property
            def metadata(self) -> dict:
                return {"parent_dataset": {"dataset_name": "FAKE"}}  # missing dataset_path

        with pytest.raises(ValueError, match="missing parent dataset information"):
            caching.CodingProblemTestDataCache.build_from_dataset(_BadReader())

    def test_build_from_reader_fails_on_missing_path(self, tmp_path: pathlib.Path) -> None:
        class _Reader:
            @property
            def metadata(self) -> dict:
                return {
                    "parent_dataset": {
                        "dataset_name": "FAKE",
                        "dataset_path": str(tmp_path / "does-not-exist"),
                    }
                }

        with pytest.raises(ValueError, match="does not exist"):
            caching.CodingProblemTestDataCache.build_from_dataset(_Reader())


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.dataset
@pytest.mark.skipif(
    tests.env_checks.TACO_DATASET_MISSING,
    reason="TACO dataset is missing, cannot build test cases cache",
)
def test_cache_building_on_taco_dataset() -> None:
    # get the latest repackaged TACO dataset path and build the cache directly
    dataset_path = pyine.data.taco.dataset_utils.get_latest_repackaged_dataset_path()
    cache = caching.CodingProblemTestDataCache(
        dataset_name="TACO",
        dataset_path=dataset_path,
        use_cache_storage=True,
    )
    # basic sanity: we built a non-empty cache of test cases
    assert isinstance(cache, caching.CodingProblemTestDataCache)
    assert len(cache._cache) > 0
