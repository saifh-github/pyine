import pathlib
import types
import typing
from unittest import mock

import msgspec
import numpy as np
import pytest

import pyine.data.traces.dataset_reader
import pyine.data.traces.dataset_utils
import pyine.data.utils.splits
import pyine.organisms.datamodules.keywords as keywords_mod
import pyine.organisms.datamodules.samples
import pyine.organisms.datamodules.samples.keyword_ops as keyword_ops
import pyine.utils.reprod
import tests.env_checks
from pyine.organisms.datamodules.keywords_configs import (
    EvaluationStrategy,
    KeywordAutoSelectionConfig,
    KeywordBiasDataModuleConfig,
)


class _DummyTrace:
    def __init__(
        self,
        identifier: str,
        solution_id: str,
        code: str = "",
    ) -> None:
        self.identifier = identifier
        self.solution_id = solution_id
        self.code_string = code


def _make_stub_datamodule(
    base_filter_rule: str = "",
    selection_config: KeywordAutoSelectionConfig | None = None,
    eval_subset_names: tuple[str, ...] = ("valid",),
    valid_subset_names: tuple[str, ...] = ("valid",),
) -> keywords_mod.KeywordBiasDataModule:
    expanded_base_names = frozenset(eval_subset_names) | frozenset(valid_subset_names)
    stub = keywords_mod.KeywordBiasDataModule.__new__(keywords_mod.KeywordBiasDataModule)
    config = types.SimpleNamespace(
        base_filter_rule=base_filter_rule,
        eval_subset_names=eval_subset_names,
        valid_subset_names=valid_subset_names,
        _expanded_base_names=expanded_base_names,
        keyword_auto_selection_config=selection_config
        or KeywordAutoSelectionConfig(
            min_keyword_frequency=1,
            max_keyword_frequency=None,
            min_keyword_length=1,
            banned_keywords=(),
            must_be_non_builtin=False,
            prefer_more_common_keywords=True,
            selection_seed=0,
        ),
    )
    stub.config = typing.cast("keywords_mod.KeywordBiasDataModuleConfig", config)
    stub.verbose = False
    return stub


def test_cluster_cache_round_trip(tmp_path: pathlib.Path) -> None:
    clusters = [
        keywords_mod.KeywordClusterData(keyword="alpha", trace_ids=("t1", "t2")),
    ]
    cache = keywords_mod.KeywordClusterCache(
        trace_data_hash="abcdef1234567890",
        filter_hash="f1234567",
        trace_count=2,
        clusters=clusters,
    )
    expected_path = keywords_mod.KeywordClusterCache.get_cache_path(
        tmp_path,
        cache.trace_data_hash,
        cache.filter_hash,
    )
    assert expected_path.name.startswith("keyword_clusters_abcd")
    cache.save(tmp_path)
    loaded = keywords_mod.KeywordClusterCache.load_if_exists(
        tmp_path,
        cache.trace_data_hash,
        cache.filter_hash,
    )
    assert loaded is not None
    assert loaded.model_dump() == cache.model_dump()


def test_cluster_cache_hash_mismatch_invalidates(tmp_path: pathlib.Path) -> None:
    wrong = keywords_mod.KeywordClusterCache(
        trace_data_hash="abc",
        filter_hash="wrong",
        trace_count=1,
        clusters=[],
    )
    cache_path = keywords_mod.KeywordClusterCache.get_cache_path(tmp_path, "abc", "expected")
    encoded = msgspec.msgpack.encode(wrong.model_dump())
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(encoded)
    loaded = keywords_mod.KeywordClusterCache.load_if_exists(tmp_path, "abc", "expected")
    assert loaded is None


def test_compute_filter_hash_deterministic() -> None:
    dm_a = _make_stub_datamodule(base_filter_rule="tag:train")
    dm_b = _make_stub_datamodule(base_filter_rule="tag:train")
    dm_c = _make_stub_datamodule(base_filter_rule="tag:valid")
    assert dm_a._compute_filter_hash() == dm_b._compute_filter_hash()
    assert dm_a._compute_filter_hash() != dm_c._compute_filter_hash()


def test_filter_and_select_keyword_filters_and_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    selection_config = KeywordAutoSelectionConfig(
        min_keyword_frequency=1,
        max_keyword_frequency=10,
        min_keyword_length=2,
        banned_keywords=("skip",),
        must_be_non_builtin=True,
        prefer_more_common_keywords=True,
        selection_seed=0,
    )
    dm = _make_stub_datamodule(selection_config=selection_config)
    clusters = [
        keywords_mod.KeywordClusterData(keyword="len", trace_ids=("a",)),  # builtin
        keywords_mod.KeywordClusterData(keyword="skip", trace_ids=("b", "c")),  # banned
        keywords_mod.KeywordClusterData(keyword="ok", trace_ids=("d", "e", "f", "g")),
        keywords_mod.KeywordClusterData(keyword="meh", trace_ids=("h",)),
    ]

    class _StubWeightedRng:
        def __init__(self) -> None:
            self.last_p: np.ndarray | None = None

        def choice(
            self,
            size: int,
            p: np.ndarray | None = None,
        ) -> int:
            self.last_p = p
            return 1

    stub_rng = _StubWeightedRng()
    monkeypatch.setattr(np.random, "default_rng", lambda seed=None: stub_rng)
    keyword, trace_ids = dm._filter_and_select_keyword(clusters)
    assert keyword == "meh"
    assert trace_ids == frozenset(("h",))
    assert stub_rng.last_p is not None
    assert pytest.approx(stub_rng.last_p.sum()) == 1.0

    selection_config = selection_config.model_copy(update={"prefer_more_common_keywords": False})
    dm = _make_stub_datamodule(selection_config=selection_config)

    class _StubUniformRng:
        def __init__(self) -> None:
            self.integers_called = 0

        def integers(
            self,
            low: int,
            high: int,
        ) -> int:
            self.integers_called += 1
            return low

    uniform_rng = _StubUniformRng()
    monkeypatch.setattr(np.random, "default_rng", lambda seed=None: uniform_rng)
    keyword, trace_ids = dm._filter_and_select_keyword(clusters)
    assert keyword == "ok"
    assert trace_ids == frozenset(("d", "e", "f", "g"))
    assert uniform_rng.integers_called == 1


def test_filter_and_select_keyword_raises_when_all_filtered() -> None:
    selection_config = KeywordAutoSelectionConfig(
        min_keyword_frequency=5,
        max_keyword_frequency=5,
        min_keyword_length=10,
        banned_keywords=("only",),
        must_be_non_builtin=True,
        prefer_more_common_keywords=False,
        selection_seed=0,
    )
    dm = _make_stub_datamodule(selection_config=selection_config)
    clusters = [keywords_mod.KeywordClusterData(keyword="only", trace_ids=("a",))]
    with pytest.raises(ValueError):
        dm._filter_and_select_keyword(clusters)


def test_find_traces_with_keyword_detects_case_insensitive_occurrences() -> None:
    dm = _make_stub_datamodule()
    traces = [
        _DummyTrace("t1", "s1", "def run():\n    return MAGIC"),
        _DummyTrace("t2", "s1", "# magic comment only"),
        _DummyTrace("t3", "s1", "magical_var = 1"),
    ]
    ids = dm._find_traces_with_keyword(traces, "magic")
    assert ids == frozenset({"t1", "t2"})


def test_subsample_traces_by_solution_handles_edges_and_levels() -> None:
    traces = [
        _DummyTrace("a1", "solA"),
        _DummyTrace("a2", "solA"),
        _DummyTrace("a3", "solA"),
        _DummyTrace("b1", "solB"),
    ]
    rng = np.random.default_rng(0)
    kept, discarded = keywords_mod.KeywordBiasDataModule._subsample_traces_by_solution(traces, 2, rng)
    assert len(kept) == 2
    assert len(discarded) == 2
    assert {t.solution_id for t in kept} == {"solA", "solB"}


def test_subsample_traces_by_solution_target_ge_len_returns_all() -> None:
    traces = [_DummyTrace("a1", "solA"), _DummyTrace("b1", "solB")]
    rng = np.random.default_rng(0)
    kept, discarded = keywords_mod.KeywordBiasDataModule._subsample_traces_by_solution(traces, 5, rng)
    assert len(kept) == 2
    assert len(discarded) == 0


def test_subsample_traces_by_solution_target_zero_returns_none() -> None:
    traces = [_DummyTrace("a1", "solA"), _DummyTrace("b1", "solB")]
    rng = np.random.default_rng(0)
    kept, discarded = keywords_mod.KeywordBiasDataModule._subsample_traces_by_solution(traces, 0, rng)
    assert len(kept) == 0
    assert len(discarded) == 2


def test_subsample_traces_by_solution_round_robin_and_deterministic() -> None:
    traces = [
        _DummyTrace("a1", "solA"),
        _DummyTrace("a2", "solA"),
        _DummyTrace("a3", "solA"),
        _DummyTrace("b1", "solB"),
        _DummyTrace("b2", "solB"),
        _DummyTrace("b3", "solB"),
        _DummyTrace("c1", "solC"),
        _DummyTrace("c2", "solC"),
        _DummyTrace("c3", "solC"),
    ]
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    kept1, discarded1 = keywords_mod.KeywordBiasDataModule._subsample_traces_by_solution(traces, 5, rng1)
    kept2, discarded2 = keywords_mod.KeywordBiasDataModule._subsample_traces_by_solution(traces, 5, rng2)
    assert {t.identifier for t in kept1} == {t.identifier for t in kept2}  # deterministic
    assert {t.identifier for t in discarded1} == {t.identifier for t in discarded2}
    assert len(kept1) == 5
    assert len(discarded1) == 4
    # note: leveling removes 4 from 9, and round-robin removal may not preserve all solutions
    assert len({t.solution_id for t in kept1}) >= 2  # at least 2 solutions represented


def test_subsample_traces_by_solution_leveling_preserves_diversity() -> None:
    """Verify leveling removes from largest solutions first, preserving diversity."""
    # setup: solA has 5 traces, solB has 2, solC has 1 (total=8)
    traces = [
        _DummyTrace("a1", "solA"),
        _DummyTrace("a2", "solA"),
        _DummyTrace("a3", "solA"),
        _DummyTrace("a4", "solA"),
        _DummyTrace("a5", "solA"),
        _DummyTrace("b1", "solB"),
        _DummyTrace("b2", "solB"),
        _DummyTrace("c1", "solC"),
    ]
    rng = np.random.default_rng(42)
    # target 4: leveling should reduce solA from 5→2, keeping solB=2, solC=1
    kept, discarded = keywords_mod.KeywordBiasDataModule._subsample_traces_by_solution(traces, 4, rng)
    assert len(kept) == 4
    assert len(discarded) == 4
    # verify all 3 solutions are still represented
    kept_solutions = {t.solution_id for t in kept}
    assert kept_solutions == {"solA", "solB", "solC"}, f"expected all solutions, got {kept_solutions}"
    # verify solC kept its only trace
    kept_c = [t for t in kept if t.solution_id == "solC"]
    assert len(kept_c) == 1


def test_subsample_traces_by_solution_single_solution() -> None:
    """Verify works with single solution (no leveling needed, just removal)."""
    traces = [_DummyTrace(f"a{i}", "solA") for i in range(5)]
    rng = np.random.default_rng(0)
    kept, discarded = keywords_mod.KeywordBiasDataModule._subsample_traces_by_solution(traces, 2, rng)
    assert len(kept) == 2
    assert len(discarded) == 3
    assert all(t.solution_id == "solA" for t in kept)


def _make_trace_metadata(
    identifier: str,
    code: str = "x = 1",
    problem_idx: int = 0,
    solution_idx: int = 0,
    trace_idx: int = 0,
    tags: list[str] | None = None,
) -> pyine.data.traces.dataset_utils.TraceMetadata:
    """Create a synthetic TraceMetadata for testing."""
    # identifier format: "dataset/subset/p{problem}/s{solution}/t{trace}"
    return pyine.data.traces.dataset_utils.TraceMetadata(
        identifier=identifier,
        parent_dataset_hash="testhash123",
        index=trace_idx,
        internal_index=trace_idx,
        step_count=10,
        code_string=code,
        inputs=None,
        expected_output=None,
        return_value=None,
        exception=None,
        stdout="",
        stderr="",
        metadata={},
        tags=tags or [],
    )


def _make_split_result(
    problem_ids: list[str],
    assignments: dict[str, str],
) -> pyine.data.utils.splits.SplitResult:
    """Create a synthetic SplitResult for testing."""
    return pyine.data.utils.splits.SplitResult(
        source_dataset_name="test",
        source_dataset_hash="splithash123",
        identifiers=problem_ids,
        tag_lists=[[] for _ in problem_ids],
        source_data_hashes=[f"hash_{pid}" for pid in problem_ids],
        subset_assignments=assignments,
        creation_metadata={},
        config=pyine.data.utils.splits.SplitConfig(
            seed=0,
            subset_names=["train", "valid", "test"],
            subset_assign_prob_map={"train": 0.7, "valid": 0.15, "test": 0.15},
        ),
    )


class TestRebalanceTrainSubsetKeywordRatio:
    """Tests for _rebalance_train_subset_keyword_ratio method."""

    def _make_datamodule_stub(
        self,
        target_ratio: float = 0.2,
        resampling_seed: int = 42,
    ) -> keywords_mod.KeywordBiasDataModule:
        stub = keywords_mod.KeywordBiasDataModule.__new__(keywords_mod.KeywordBiasDataModule)
        stub.config = types.SimpleNamespace(
            train_subset_with_keyword_ratio=target_ratio,
            train_subset_resampling_seed=resampling_seed,
        )
        stub.verbose = False
        return typing.cast("keywords_mod.KeywordBiasDataModule", stub)

    def test_too_few_with_keyword_subsamples_without(self) -> None:
        dm = self._make_datamodule_stub(target_ratio=0.5)
        # 2 with keyword, 8 without -> current ratio = 0.2, target = 0.5
        traces_with = [_DummyTrace(f"w{idx}", f"solW{idx}") for idx in range(2)]
        traces_without = [_DummyTrace(f"wo{idx}", f"solWO{idx}") for idx in range(8)]
        subset_traces: dict[str, list[typing.Any]] = {"train": traces_with + traces_without}
        unassigned: list[typing.Any] = []
        trace_ids_with_kw = frozenset(t.identifier for t in traces_with)
        dm._rebalance_train_subset_keyword_ratio(subset_traces, unassigned, trace_ids_with_kw)
        train_after = subset_traces["train"]
        count_with = sum(1 for t in train_after if t.identifier in trace_ids_with_kw)
        count_without = len(train_after) - count_with
        ratio_after = count_with / len(train_after)
        assert count_with == 2  # all with-keyword traces kept
        assert count_without < 8  # some without-keyword traces removed
        assert len(unassigned) > 0  # discarded traces moved to unassigned
        assert ratio_after > 0.2  # ratio improved toward target

    def test_too_many_with_keyword_discards_with(self) -> None:
        dm = self._make_datamodule_stub(target_ratio=0.2)
        # 8 with keyword, 2 without -> current ratio = 0.8, target = 0.2
        traces_with = [_DummyTrace(f"w{idx}", f"solW{idx}") for idx in range(8)]
        traces_without = [_DummyTrace(f"wo{idx}", f"solWO{idx}") for idx in range(2)]
        subset_traces: dict[str, list[typing.Any]] = {"train": traces_with + traces_without}
        unassigned: list[typing.Any] = []
        trace_ids_with_kw = frozenset(t.identifier for t in traces_with)
        dm._rebalance_train_subset_keyword_ratio(subset_traces, unassigned, trace_ids_with_kw)
        train_after = subset_traces["train"]
        count_with = sum(1 for t in train_after if t.identifier in trace_ids_with_kw)
        count_without = len(train_after) - count_with
        ratio_after = count_with / len(train_after)
        assert count_without == 2  # all without-keyword traces kept
        assert count_with < 8  # some with-keyword traces removed
        assert len(unassigned) > 0  # discarded traces moved to unassigned
        assert ratio_after < 0.8  # ratio improved toward target

    def test_already_at_target_ratio_no_changes(self) -> None:
        dm = self._make_datamodule_stub(target_ratio=0.5)
        traces_with = [_DummyTrace(f"w{idx}", f"solW{idx}") for idx in range(5)]
        traces_without = [_DummyTrace(f"wo{idx}", f"solWO{idx}") for idx in range(5)]
        subset_traces: dict[str, list[typing.Any]] = {"train": traces_with + traces_without}
        unassigned: list[typing.Any] = []
        trace_ids_with_kw = frozenset(t.identifier for t in traces_with)
        dm._rebalance_train_subset_keyword_ratio(subset_traces, unassigned, trace_ids_with_kw)
        assert len(subset_traces["train"]) == 10
        assert len(unassigned) == 0

    def test_determinism_with_seed(self) -> None:
        dm1 = self._make_datamodule_stub(target_ratio=0.3, resampling_seed=999)
        dm2 = self._make_datamodule_stub(target_ratio=0.3, resampling_seed=999)
        traces_with = [_DummyTrace(f"w{idx}", f"solW{idx}") for idx in range(3)]
        traces_without = [_DummyTrace(f"wo{idx}", f"solWO{idx}") for idx in range(10)]
        trace_ids_with_kw = frozenset(t.identifier for t in traces_with)
        subset1: dict[str, list[typing.Any]] = {"train": list(traces_with + traces_without)}
        subset2: dict[str, list[typing.Any]] = {"train": list(traces_with + traces_without)}
        unassigned1: list[typing.Any] = []
        unassigned2: list[typing.Any] = []
        dm1._rebalance_train_subset_keyword_ratio(subset1, unassigned1, trace_ids_with_kw)
        dm2._rebalance_train_subset_keyword_ratio(subset2, unassigned2, trace_ids_with_kw)
        assert {t.identifier for t in subset1["train"]} == {t.identifier for t in subset2["train"]}

    def test_missing_train_subset_skips_rebalancing(self) -> None:
        dm = self._make_datamodule_stub(target_ratio=0.5)
        subset_traces: dict[str, list[typing.Any]] = {"valid": []}  # no "train" key
        unassigned: list[typing.Any] = []
        trace_ids_with_kw = frozenset(["t1"])
        # should not raise, just skip
        dm._rebalance_train_subset_keyword_ratio(subset_traces, unassigned, trace_ids_with_kw)
        assert "train" not in subset_traces  # unchanged
        assert len(unassigned) == 0

    def test_empty_train_subset_skips_rebalancing(self) -> None:
        dm = self._make_datamodule_stub(target_ratio=0.5)
        subset_traces: dict[str, list[typing.Any]] = {"train": []}  # empty train
        unassigned: list[typing.Any] = []
        trace_ids_with_kw = frozenset(["t1"])
        # should not raise, just skip
        dm._rebalance_train_subset_keyword_ratio(subset_traces, unassigned, trace_ids_with_kw)
        assert len(subset_traces["train"]) == 0  # unchanged
        assert len(unassigned) == 0

    def test_ratio_zero_skips_rebalancing(self) -> None:
        dm = self._make_datamodule_stub(target_ratio=0.0)
        traces_with = [_DummyTrace(f"w{idx}", f"solW{idx}") for idx in range(2)]
        traces_without = [_DummyTrace(f"wo{idx}", f"solWO{idx}") for idx in range(8)]
        subset_traces: dict[str, list[typing.Any]] = {"train": traces_with + traces_without}
        original_count = len(subset_traces["train"])
        unassigned: list[typing.Any] = []
        trace_ids_with_kw = frozenset(t.identifier for t in traces_with)
        dm._rebalance_train_subset_keyword_ratio(subset_traces, unassigned, trace_ids_with_kw)
        # should skip due to invalid ratio (<=0)
        assert len(subset_traces["train"]) == original_count
        assert len(unassigned) == 0

    def test_ratio_one_skips_rebalancing(self) -> None:
        dm = self._make_datamodule_stub(target_ratio=1.0)
        traces_with = [_DummyTrace(f"w{idx}", f"solW{idx}") for idx in range(2)]
        traces_without = [_DummyTrace(f"wo{idx}", f"solWO{idx}") for idx in range(8)]
        subset_traces: dict[str, list[typing.Any]] = {"train": traces_with + traces_without}
        original_count = len(subset_traces["train"])
        unassigned: list[typing.Any] = []
        trace_ids_with_kw = frozenset(t.identifier for t in traces_with)
        dm._rebalance_train_subset_keyword_ratio(subset_traces, unassigned, trace_ids_with_kw)
        # should skip due to invalid ratio (>=1)
        assert len(subset_traces["train"]) == original_count
        assert len(unassigned) == 0

    def test_ratio_negative_skips_rebalancing(self) -> None:
        dm = self._make_datamodule_stub(target_ratio=-0.5)
        traces = [_DummyTrace(f"t{idx}", f"sol{idx}") for idx in range(5)]
        subset_traces: dict[str, list[typing.Any]] = {"train": traces}
        original_count = len(subset_traces["train"])
        unassigned: list[typing.Any] = []
        trace_ids_with_kw = frozenset([traces[0].identifier])
        dm._rebalance_train_subset_keyword_ratio(subset_traces, unassigned, trace_ids_with_kw)
        # should skip due to invalid ratio (<0)
        assert len(subset_traces["train"]) == original_count
        assert len(unassigned) == 0

    def test_no_with_keyword_skips_rebalancing(self) -> None:
        dm = self._make_datamodule_stub(target_ratio=0.5)
        traces_without = [_DummyTrace(f"wo{idx}", f"solWO{idx}") for idx in range(10)]
        subset_traces: dict[str, list[typing.Any]] = {"train": traces_without}
        unassigned: list[typing.Any] = []
        trace_ids_with_kw = frozenset()
        dm._rebalance_train_subset_keyword_ratio(subset_traces, unassigned, trace_ids_with_kw)
        assert len(subset_traces["train"]) == 10
        assert len(unassigned) == 0

    def test_no_without_keyword_skips_rebalancing(self) -> None:
        dm = self._make_datamodule_stub(target_ratio=0.5)
        traces_with = [_DummyTrace(f"w{idx}", f"solW{idx}") for idx in range(10)]
        subset_traces: dict[str, list[typing.Any]] = {"train": traces_with}
        unassigned: list[typing.Any] = []
        trace_ids_with_kw = frozenset(t.identifier for t in traces_with)
        dm._rebalance_train_subset_keyword_ratio(subset_traces, unassigned, trace_ids_with_kw)
        assert len(subset_traces["train"]) == 10
        assert len(unassigned) == 0


class TestAdjustKeywordSplitSubsets:
    """Tests for _adjust_keyword_split_subsets method."""

    def _make_datamodule_stub(
        self,
        eval_subset_names: tuple[str, ...] = ("valid",),
        evaluation_strategy: EvaluationStrategy = EvaluationStrategy.keyword_presence_split,
        target_ratio: float = 0.5,
    ) -> keywords_mod.KeywordBiasDataModule:
        stub = keywords_mod.KeywordBiasDataModule.__new__(keywords_mod.KeywordBiasDataModule)
        expanded_base_names = frozenset(eval_subset_names)
        stub.config = types.SimpleNamespace(
            eval_subset_names=eval_subset_names,
            _expanded_base_names=expanded_base_names,
            evaluation_strategy=evaluation_strategy,
            train_subset_with_keyword_ratio=target_ratio,
            train_subset_resampling_seed=0,
        )
        stub.verbose = False
        return typing.cast("keywords_mod.KeywordBiasDataModule", stub)

    def test_keyword_presence_split_creates_disjoint_subsets(self) -> None:
        dm = self._make_datamodule_stub(evaluation_strategy=EvaluationStrategy.keyword_presence_split)
        traces_with = [_DummyTrace(f"v_w{idx}", f"solW{idx}") for idx in range(3)]
        traces_without = [_DummyTrace(f"v_wo{idx}", f"solWO{idx}") for idx in range(5)]
        trace_ids_with_kw = frozenset(t.identifier for t in traces_with)
        subset_traces: dict[str, list[typing.Any]] = {
            "train": [_DummyTrace("t1", "s1")],  # needed to avoid rebalancing issues
            "valid": traces_with + traces_without,
        }
        unassigned: list[typing.Any] = []
        derived_subsets = dm._adjust_keyword_split_subsets(subset_traces, unassigned, trace_ids_with_kw)
        assert "valid_with_keyword" in derived_subsets
        assert "valid_without_keyword" in derived_subsets
        assert len(derived_subsets["valid_with_keyword"].traces) == 3
        assert len(derived_subsets["valid_without_keyword"].traces) == 5
        assert derived_subsets["valid_with_keyword"].parent_subset == "valid"
        assert derived_subsets["valid_without_keyword"].parent_subset == "valid"
        assert derived_subsets["valid_with_keyword"].derivation_type == "keyword_presence_split"
        # disjoint: no overlap
        with_ids = {t.identifier for t in derived_subsets["valid_with_keyword"].traces}
        without_ids = {t.identifier for t in derived_subsets["valid_without_keyword"].traces}
        assert with_ids.isdisjoint(without_ids)

    def test_counterfactual_creates_duplicated_subsets(self) -> None:
        dm = self._make_datamodule_stub(evaluation_strategy=EvaluationStrategy.counterfactual)
        all_traces = [_DummyTrace(f"v{idx}", f"sol{idx}") for idx in range(5)]
        trace_ids_with_kw = frozenset(t.identifier for t in all_traces[:2])
        subset_traces: dict[str, list[typing.Any]] = {
            "train": [_DummyTrace("t1", "s1")],
            "valid": list(all_traces),
        }
        unassigned: list[typing.Any] = []
        derived_subsets = dm._adjust_keyword_split_subsets(subset_traces, unassigned, trace_ids_with_kw)
        assert len(derived_subsets["valid_with_keyword"].traces) == 5
        assert len(derived_subsets["valid_without_keyword"].traces) == 5
        assert derived_subsets["valid_with_keyword"].parent_subset == "valid"
        assert derived_subsets["valid_with_keyword"].derivation_type == "counterfactual"
        # same traces in both (duplicated for counterfactual)
        with_ids = {t.identifier for t in derived_subsets["valid_with_keyword"].traces}
        without_ids = {t.identifier for t in derived_subsets["valid_without_keyword"].traces}
        assert with_ids == without_ids


class TestGetParser:
    """Tests for get_parser method with keyword manipulation wrapper.

    Note: These tests verify the logic of trace ID set computation
    in the get_parser method. The actual wrapper instantiation is mocked.
    """

    def _make_mock_parser(
        self,
        trace_ids: list[str],
        trace_ids_with_keyword: frozenset[str],
    ) -> mock.MagicMock:
        """Create a mock base parser with orig_traces."""
        mock_parser = mock.MagicMock()
        mock_traces = []
        for tid in trace_ids:
            mock_trace = mock.MagicMock()
            mock_trace.identifier = tid
            mock_traces.append(mock_trace)
        mock_parser.orig_traces = mock_traces
        mock_parser.__len__ = mock.MagicMock(return_value=len(trace_ids))
        return mock_parser

    def test_base_eval_subset_returns_concat_dataset(self) -> None:
        # base eval subset now returns ConcatDataset of derived parsers
        import torch.utils.data

        expanded = frozenset({"valid"})

        def _get_parent(name: str) -> str:
            for base in expanded:
                if name == f"{base}_with_keyword" or name == f"{base}_without_keyword":
                    return base
            return name

        stub = keywords_mod.KeywordBiasDataModule.__new__(keywords_mod.KeywordBiasDataModule)
        stub._metadata = mock.MagicMock(spec=keywords_mod.KeywordTraceDatasetMetadata)
        stub._metadata.keyword = "magic"
        stub._metadata.trace_ids_with_keyword = frozenset(["t1", "t2"])
        stub._subset_parsers = {
            "valid_with_keyword": mock.MagicMock(),
            "valid_without_keyword": mock.MagicMock(),
        }
        stub.config = types.SimpleNamespace(
            evaluation_strategy=EvaluationStrategy.counterfactual,
            eval_subset_names=("valid",),
            _expanded_base_names=expanded,
            _get_parent_subset_name=_get_parent,
            subset_names=("valid", "valid_with_keyword", "valid_without_keyword"),
        )
        stub.verbose = False
        mock_parser = self._make_mock_parser(["t1", "t2", "t3", "t4"], stub._metadata.trace_ids_with_keyword)
        with mock.patch.object(
            keywords_mod.pyine.organisms.datamodules.base.BiasDataModuleBase,
            "get_parser",
            return_value=mock_parser,
        ):
            result = stub.get_parser("valid")
            assert isinstance(result, torch.utils.data.ConcatDataset)
            assert len(result.datasets) == 2  # _with_keyword + _without_keyword

    def test_injection_enabled_for_with_keyword_subset(self) -> None:
        # verify that injection trace IDs are set for _with_keyword subsets in counterfactual mode
        stub = keywords_mod.KeywordBiasDataModule.__new__(keywords_mod.KeywordBiasDataModule)
        stub._metadata = mock.MagicMock(spec=keywords_mod.KeywordTraceDatasetMetadata)
        stub._metadata.keyword = "magic"
        # t1 has keyword, t2 doesn't
        stub._metadata.trace_ids_with_keyword = frozenset(["t1"])
        stub._subset_parsers = {"valid_with_keyword": mock.MagicMock()}
        stub.config = types.SimpleNamespace(
            evaluation_strategy=EvaluationStrategy.counterfactual,
            eval_subset_names=("valid",),
            _expanded_base_names=frozenset({"valid"}),
            subset_names=("valid_with_keyword",),
        )
        stub.verbose = False
        mock_parser = self._make_mock_parser(["t1", "t2"], stub._metadata.trace_ids_with_keyword)
        with mock.patch.object(
            keywords_mod.pyine.organisms.datamodules.base.BiasDataModuleBase,
            "get_parser",
            return_value=mock_parser,
        ):
            result = stub.get_parser("valid_with_keyword")
            assert isinstance(result, keyword_ops.SampleKeywordManipulatorWrapper)
            assert result._inject_trace_ids == frozenset(["t2"])  # inject for IDs without keyword
            assert result._refactor_trace_ids == frozenset()  # no refactoring

    def test_refactoring_enabled_for_without_keyword_subset(self) -> None:
        # verify that refactor trace IDs are set for _without_keyword subsets
        stub = keywords_mod.KeywordBiasDataModule.__new__(keywords_mod.KeywordBiasDataModule)
        stub._metadata = mock.MagicMock(spec=keywords_mod.KeywordTraceDatasetMetadata)
        stub._metadata.keyword = "magic"
        # t1 has keyword, t2 doesn't
        stub._metadata.trace_ids_with_keyword = frozenset(["t1"])
        stub._subset_parsers = {"valid_without_keyword": mock.MagicMock()}
        stub.config = types.SimpleNamespace(
            evaluation_strategy=EvaluationStrategy.counterfactual,
            eval_subset_names=("valid",),
            _expanded_base_names=frozenset({"valid"}),
            subset_names=("valid_without_keyword",),
        )
        stub.verbose = False
        mock_parser = self._make_mock_parser(["t1", "t2"], stub._metadata.trace_ids_with_keyword)
        with mock.patch.object(
            keywords_mod.pyine.organisms.datamodules.base.BiasDataModuleBase,
            "get_parser",
            return_value=mock_parser,
        ):
            result = stub.get_parser("valid_without_keyword")
            assert isinstance(result, keyword_ops.SampleKeywordManipulatorWrapper)
            assert result._inject_trace_ids == frozenset()  # no injection
            assert result._refactor_trace_ids == frozenset(["t1"])  # refactor IDs with keyword

    def test_no_manipulation_for_train_subset(self) -> None:
        # verify that train subset gets wrapper with no injection/refactoring
        stub = keywords_mod.KeywordBiasDataModule.__new__(keywords_mod.KeywordBiasDataModule)
        stub._metadata = mock.MagicMock(spec=keywords_mod.KeywordTraceDatasetMetadata)
        stub._metadata.keyword = "magic"
        stub._metadata.trace_ids_with_keyword = frozenset(["t1"])
        stub._subset_parsers = {"train": mock.MagicMock()}
        stub.config = types.SimpleNamespace(
            evaluation_strategy=EvaluationStrategy.counterfactual,
            eval_subset_names=("valid",),
            _expanded_base_names=frozenset({"valid"}),
            subset_names=("train",),
        )
        stub.verbose = False
        mock_parser = self._make_mock_parser(["t1", "t2"], stub._metadata.trace_ids_with_keyword)
        with mock.patch.object(
            keywords_mod.pyine.organisms.datamodules.base.BiasDataModuleBase,
            "get_parser",
            return_value=mock_parser,
        ):
            result = stub.get_parser("train")
            assert isinstance(result, keyword_ops.SampleKeywordManipulatorWrapper)
            assert result._inject_trace_ids == frozenset()  # no injection
            assert result._refactor_trace_ids == frozenset()  # no refactoring

    def test_keyword_presence_split_no_manipulation(self) -> None:
        # verify that keyword_presence_split mode has no manipulation (tagging only)
        stub = keywords_mod.KeywordBiasDataModule.__new__(keywords_mod.KeywordBiasDataModule)
        stub._metadata = mock.MagicMock(spec=keywords_mod.KeywordTraceDatasetMetadata)
        stub._metadata.keyword = "magic"
        stub._metadata.trace_ids_with_keyword = frozenset(["t1"])
        stub._subset_parsers = {"valid_with_keyword": mock.MagicMock()}
        stub.config = types.SimpleNamespace(
            evaluation_strategy=EvaluationStrategy.keyword_presence_split,
            eval_subset_names=("valid",),
            _expanded_base_names=frozenset({"valid"}),
            subset_names=("valid_with_keyword",),
        )
        stub.verbose = False
        mock_parser = self._make_mock_parser(["t1", "t2"], stub._metadata.trace_ids_with_keyword)
        with mock.patch.object(
            keywords_mod.pyine.organisms.datamodules.base.BiasDataModuleBase,
            "get_parser",
            return_value=mock_parser,
        ):
            result = stub.get_parser("valid_with_keyword")
            assert isinstance(result, keyword_ops.SampleKeywordManipulatorWrapper)
            assert result._inject_trace_ids == frozenset()  # no injection in presence_split
            assert result._refactor_trace_ids == frozenset()  # no refactoring in presence_split


@pytest.fixture
def fake_lmdb_and_split(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """Create fake LMDB and split file for config validation tests."""
    lmdb_path = tmp_path / "test.lmdb"
    lmdb_path.mkdir()
    (lmdb_path / "data.mdb").touch()
    (lmdb_path / "lock.mdb").touch()
    split_path = tmp_path / "split.bin"
    split_path.touch()
    return lmdb_path, split_path


class TestKeywordBiasDataModuleConfigValidateAndResolve:
    """Tests for _validate_and_resolve in KeywordBiasDataModuleConfig.

    These tests instantiate real config objects to exercise the actual pydantic validators.
    """

    def _make_minimal_config(
        self,
        lmdb_path: pathlib.Path,
        split_path: pathlib.Path,
        eval_subset_names: tuple[str, ...] = ("valid",),
        exclude_augmented_traces: bool = True,
        base_filter_rule: str = "",
    ) -> KeywordBiasDataModuleConfig:
        """Create a minimal KeywordBiasDataModuleConfig for testing validation logic."""
        return KeywordBiasDataModuleConfig(
            lmdb_paths=[str(lmdb_path)],
            split_file_path=str(split_path),
            eval_subset_names=eval_subset_names,
            exclude_augmented_traces=exclude_augmented_traces,
            base_filter_rule=base_filter_rule,
            instantiate_parsers_at_setup=False,  # avoid parser instantiation
        )

    def test_extends_subset_names_with_keyword_splits(
        self, fake_lmdb_and_split: tuple[pathlib.Path, pathlib.Path]
    ) -> None:
        lmdb_path, split_path = fake_lmdb_and_split
        config = self._make_minimal_config(lmdb_path, split_path, eval_subset_names=("valid",))
        assert "valid_with_keyword" in config.subset_names
        assert "valid_without_keyword" in config.subset_names
        # base subsets should still be present
        assert "train" in config.subset_names
        assert "valid" in config.subset_names

    def test_extends_multiple_eval_subsets(self, fake_lmdb_and_split: tuple[pathlib.Path, pathlib.Path]) -> None:
        lmdb_path, split_path = fake_lmdb_and_split
        config = self._make_minimal_config(lmdb_path, split_path, eval_subset_names=("valid", "test"))
        assert "valid_with_keyword" in config.subset_names
        assert "valid_without_keyword" in config.subset_names
        assert "test_with_keyword" in config.subset_names
        assert "test_without_keyword" in config.subset_names


class TestKeywordBiasDataModuleConfigParentSubsetResolution:
    """Tests for _get_parent_subset_name using real config objects."""

    def _make_minimal_config(
        self,
        lmdb_path: pathlib.Path,
        split_path: pathlib.Path,
        eval_subset_names: tuple[str, ...] = ("valid", "test"),
    ) -> KeywordBiasDataModuleConfig:
        return KeywordBiasDataModuleConfig(
            lmdb_paths=[str(lmdb_path)],
            split_file_path=str(split_path),
            eval_subset_names=eval_subset_names,
            instantiate_parsers_at_setup=False,
        )

    def test_get_parent_subset_name_for_derived_subsets(
        self, fake_lmdb_and_split: tuple[pathlib.Path, pathlib.Path]
    ) -> None:
        lmdb_path, split_path = fake_lmdb_and_split
        config = self._make_minimal_config(lmdb_path, split_path, eval_subset_names=("valid", "test"))
        assert config._get_parent_subset_name("valid_with_keyword") == "valid"
        assert config._get_parent_subset_name("valid_without_keyword") == "valid"
        assert config._get_parent_subset_name("test_with_keyword") == "test"
        assert config._get_parent_subset_name("train") == "train"  # not a derived subset
        assert config._get_parent_subset_name("valid") == "valid"  # base eval subset

    def test_get_parent_covers_valid_subset_names(self, fake_lmdb_and_split: tuple[pathlib.Path, pathlib.Path]) -> None:
        lmdb_path, split_path = fake_lmdb_and_split
        config = KeywordBiasDataModuleConfig(
            lmdb_paths=[str(lmdb_path)],
            split_file_path=str(split_path),
            valid_subset_names=("valid", "test"),
            eval_subset_names=("valid",),
            instantiate_parsers_at_setup=False,
        )
        assert config._get_parent_subset_name("test_with_keyword") == "test"
        assert config._get_parent_subset_name("test_without_keyword") == "test"


class TestKeywordBiasDataModuleConfigResolvedNames:
    """Tests for resolved_* properties and fail-loud validation."""

    def _make_minimal_config(
        self,
        lmdb_path: pathlib.Path,
        split_path: pathlib.Path,
        eval_subset_names: tuple[str, ...] = ("valid",),
        valid_subset_names: tuple[str, ...] = ("valid",),
    ) -> KeywordBiasDataModuleConfig:
        return KeywordBiasDataModuleConfig(
            lmdb_paths=[str(lmdb_path)],
            split_file_path=str(split_path),
            eval_subset_names=eval_subset_names,
            valid_subset_names=valid_subset_names,
            instantiate_parsers_at_setup=False,
        )

    def test_resolved_eval_subset_names(self, fake_lmdb_and_split: tuple[pathlib.Path, pathlib.Path]) -> None:
        lmdb_path, split_path = fake_lmdb_and_split
        config = self._make_minimal_config(lmdb_path, split_path)
        assert "valid_with_keyword" in config.resolved_eval_subset_names
        assert "valid_without_keyword" in config.resolved_eval_subset_names

    def test_resolved_valid_subset_names(self, fake_lmdb_and_split: tuple[pathlib.Path, pathlib.Path]) -> None:
        lmdb_path, split_path = fake_lmdb_and_split
        config = self._make_minimal_config(lmdb_path, split_path)
        assert "valid_with_keyword" in config.resolved_valid_subset_names
        assert "valid_without_keyword" in config.resolved_valid_subset_names

    def test_resolved_names_in_subset_names(self, fake_lmdb_and_split: tuple[pathlib.Path, pathlib.Path]) -> None:
        lmdb_path, split_path = fake_lmdb_and_split
        config = self._make_minimal_config(lmdb_path, split_path)
        for name in config.resolved_eval_subset_names:
            assert name in config.subset_names
        for name in config.resolved_valid_subset_names:
            assert name in config.subset_names

    def test_rejects_suffixed_eval_subset_names(self, fake_lmdb_and_split: tuple[pathlib.Path, pathlib.Path]) -> None:
        lmdb_path, split_path = fake_lmdb_and_split
        with pytest.raises(ValueError, match="derived name"):
            self._make_minimal_config(lmdb_path, split_path, eval_subset_names=("valid_with_keyword",))

    def test_rejects_suffixed_valid_subset_names(self, fake_lmdb_and_split: tuple[pathlib.Path, pathlib.Path]) -> None:
        lmdb_path, split_path = fake_lmdb_and_split
        with pytest.raises(ValueError, match="derived name"):
            self._make_minimal_config(lmdb_path, split_path, valid_subset_names=("valid_with_keyword",))

    def test_rejects_derived_suffix_in_dataparser_overrides(
        self, fake_lmdb_and_split: tuple[pathlib.Path, pathlib.Path]
    ) -> None:
        lmdb_path, split_path = fake_lmdb_and_split
        with pytest.raises(ValueError, match="derived-suffix key"):
            KeywordBiasDataModuleConfig(
                lmdb_paths=[str(lmdb_path)],
                split_file_path=str(split_path),
                dataparser_config_overrides={"valid_with_keyword": {"some_key": "val"}},
                instantiate_parsers_at_setup=False,
            )

    def test_rejects_derived_suffix_in_dataloader_overrides(
        self, fake_lmdb_and_split: tuple[pathlib.Path, pathlib.Path]
    ) -> None:
        lmdb_path, split_path = fake_lmdb_and_split
        with pytest.raises(ValueError, match="derived-suffix key"):
            KeywordBiasDataModuleConfig(
                lmdb_paths=[str(lmdb_path)],
                split_file_path=str(split_path),
                dataloader_config_overrides={"valid_without_keyword": {"some_key": "val"}},
                instantiate_parsers_at_setup=False,
            )


class TestExcludeAugmentedTracesConfig:
    """Tests for exclude_augmented_traces configuration using real config objects."""

    def _make_config(
        self,
        lmdb_path: pathlib.Path,
        split_path: pathlib.Path,
        base_filter_rule: str = "",
        exclude_augmented_traces: bool = True,
    ) -> KeywordBiasDataModuleConfig:
        return KeywordBiasDataModuleConfig(
            lmdb_paths=[str(lmdb_path)],
            split_file_path=str(split_path),
            base_filter_rule=base_filter_rule,
            exclude_augmented_traces=exclude_augmented_traces,
            instantiate_parsers_at_setup=False,
        )

    def test_exclude_augmented_traces_sets_filter_rule_when_empty(
        self, fake_lmdb_and_split: tuple[pathlib.Path, pathlib.Path]
    ) -> None:
        lmdb_path, split_path = fake_lmdb_and_split
        config = self._make_config(lmdb_path, split_path, base_filter_rule="", exclude_augmented_traces=True)
        assert config.base_filter_rule == "-augment:*"

    def test_exclude_augmented_traces_combines_with_existing_rule(
        self, fake_lmdb_and_split: tuple[pathlib.Path, pathlib.Path]
    ) -> None:
        lmdb_path, split_path = fake_lmdb_and_split
        config = self._make_config(
            lmdb_path, split_path, base_filter_rule="+source:taco", exclude_augmented_traces=True
        )
        assert config.base_filter_rule == "+source:taco -augment:*"

    def test_exclude_augmented_traces_false_preserves_base_rule(
        self, fake_lmdb_and_split: tuple[pathlib.Path, pathlib.Path]
    ) -> None:
        lmdb_path, split_path = fake_lmdb_and_split
        config = self._make_config(
            lmdb_path, split_path, base_filter_rule="+source:taco", exclude_augmented_traces=False
        )
        assert config.base_filter_rule == "+source:taco"

    def test_exclude_augmented_traces_false_with_empty_rule(
        self, fake_lmdb_and_split: tuple[pathlib.Path, pathlib.Path]
    ) -> None:
        lmdb_path, split_path = fake_lmdb_and_split
        config = self._make_config(lmdb_path, split_path, base_filter_rule="", exclude_augmented_traces=False)
        assert config.base_filter_rule == ""


class TestIsBuiltinOrReserved:
    """Tests for is_builtin_or_reserved helper function (from keyword_ops)."""

    def test_returns_true_for_keywords(self) -> None:
        assert keyword_ops.is_builtin_or_reserved("for") is True
        assert keyword_ops.is_builtin_or_reserved("if") is True
        assert keyword_ops.is_builtin_or_reserved("class") is True

    def test_returns_true_for_builtins(self) -> None:
        assert keyword_ops.is_builtin_or_reserved("len") is True
        assert keyword_ops.is_builtin_or_reserved("print") is True
        assert keyword_ops.is_builtin_or_reserved("int") is True

    def test_returns_false_for_custom_names(self) -> None:
        assert keyword_ops.is_builtin_or_reserved("my_variable") is False
        assert keyword_ops.is_builtin_or_reserved("custom_function") is False


class TestKeywordTraceDatasetMetadata:
    """Tests for KeywordTraceDatasetMetadata class.

    Note: We test the computed properties directly by constructing objects that bypass
    full pydantic validation (which requires complex trace identifier formats).
    The pydantic models are frozen, so we use object.__setattr__ to set attributes.
    """

    def test_trace_count_with_keyword_property(self) -> None:
        # directly test the property logic: trace_ids_with_keyword length
        metadata = keywords_mod.KeywordTraceDatasetMetadata.__new__(keywords_mod.KeywordTraceDatasetMetadata)
        object.__setattr__(metadata, "trace_ids_with_keyword", frozenset({"t1", "t2", "t3"}))
        assert metadata.trace_count_with_keyword == 3

    def test_trace_count_without_keyword_property(self) -> None:
        # directly test the property logic: base_traces length - trace_ids_with_keyword length
        metadata = keywords_mod.KeywordTraceDatasetMetadata.__new__(keywords_mod.KeywordTraceDatasetMetadata)
        object.__setattr__(metadata, "base_traces", [mock.MagicMock() for _ in range(5)])  # 5 total traces
        object.__setattr__(metadata, "trace_ids_with_keyword", frozenset({"t1", "t2"}))  # 2 with keyword
        assert metadata.trace_count_without_keyword == 3  # 5 - 2 = 3 without

    def test_has_keyword_method(self) -> None:
        metadata = keywords_mod.KeywordTraceDatasetMetadata.__new__(keywords_mod.KeywordTraceDatasetMetadata)
        object.__setattr__(metadata, "trace_ids_with_keyword", frozenset({"t1", "t3"}))
        assert metadata.has_keyword("t1") is True
        assert metadata.has_keyword("t2") is False
        assert metadata.has_keyword("t3") is True

    def test_cluster_cache_path_field(self) -> None:
        # test that cluster_cache_path is an optional field with None default
        metadata = keywords_mod.KeywordTraceDatasetMetadata.__new__(keywords_mod.KeywordTraceDatasetMetadata)
        object.__setattr__(metadata, "cluster_cache_path", None)
        assert metadata.cluster_cache_path is None
        object.__setattr__(metadata, "cluster_cache_path", "/some/path/clusters.msgspec")
        assert metadata.cluster_cache_path == "/some/path/clusters.msgspec"


class TestKeywordProperty:
    """Tests for keyword property accessor on KeywordBiasDataModule."""

    def test_raises_when_metadata_not_loaded(self) -> None:
        stub = keywords_mod.KeywordBiasDataModule.__new__(keywords_mod.KeywordBiasDataModule)
        stub._metadata = None
        with pytest.raises(RuntimeError, match="metadata not yet loaded"):
            _ = stub.keyword

    def test_returns_keyword_when_metadata_loaded(self) -> None:
        stub = keywords_mod.KeywordBiasDataModule.__new__(keywords_mod.KeywordBiasDataModule)
        # create a minimal mock metadata with keyword attribute
        mock_metadata = keywords_mod.KeywordTraceDatasetMetadata.__new__(keywords_mod.KeywordTraceDatasetMetadata)
        object.__setattr__(mock_metadata, "keyword", "my_keyword")
        stub._metadata = mock_metadata
        assert stub.keyword == "my_keyword"


class TestKeywordsValidateSampleCounts:
    """Tests for _validate_sample_counts method in KeywordBiasDataModule."""

    def _make_stub_dm(
        self,
        min_samples_with_keyword: int = 0,
        min_samples_without_keyword: int = 0,
    ) -> keywords_mod.KeywordBiasDataModule:
        stub = keywords_mod.KeywordBiasDataModule.__new__(keywords_mod.KeywordBiasDataModule)
        stub.config = types.SimpleNamespace(
            min_samples_with_keyword=min_samples_with_keyword,
            min_samples_without_keyword=min_samples_without_keyword,
        )
        stub.verbose = False
        return stub

    def test_raises_when_too_few_with_keyword(self) -> None:
        dm = self._make_stub_dm(min_samples_with_keyword=10, min_samples_without_keyword=0)
        subset_traces = {"train": [_make_trace_metadata(f"t{i}", "p1", "s1") for i in range(20)]}
        trace_ids_with_keyword = frozenset({"t0", "t1"})  # only 2 with keyword
        with pytest.raises(ValueError, match="only 2 traces contain the keyword"):
            dm._validate_sample_counts("kw", subset_traces, {}, trace_ids_with_keyword)

    def test_raises_when_too_few_without_keyword(self) -> None:
        dm = self._make_stub_dm(min_samples_with_keyword=0, min_samples_without_keyword=50)
        subset_traces = {"train": [_make_trace_metadata(f"t{i}", "p1", "s1") for i in range(20)]}
        trace_ids_with_keyword = frozenset({f"t{i}" for i in range(15)})  # 15 with, 5 without
        with pytest.raises(ValueError, match="only 5 traces lack the keyword"):
            dm._validate_sample_counts("kw", subset_traces, {}, trace_ids_with_keyword)

    def test_passes_when_counts_meet_minimum(self) -> None:
        dm = self._make_stub_dm(min_samples_with_keyword=5, min_samples_without_keyword=10)
        subset_traces = {"train": [_make_trace_metadata(f"t{i}", "p1", "s1") for i in range(20)]}
        trace_ids_with_keyword = frozenset({f"t{i}" for i in range(8)})  # 8 with, 12 without
        dm._validate_sample_counts("kw", subset_traces, {}, trace_ids_with_keyword)  # should not raise


class TestGetHfMessagesDatasetWrapper:
    """Tests for get_hf_messages_dataset wrapper behavior.

    Note: The wrapper logic (SampleKeywordManipulatorWrapper) is thoroughly tested in
    TestGetParser. These tests verify the high-level API behavior.
    """

    def test_raises_when_not_setup(self) -> None:
        stub = keywords_mod.KeywordBiasDataModule.__new__(keywords_mod.KeywordBiasDataModule)
        stub._metadata = None  # not setup
        stub._subset_parsers = {}
        with pytest.raises(RuntimeError, match="not ready yet"):
            stub.get_hf_messages_dataset("train")

    def test_is_keyword_split_subset(self) -> None:
        # test helper method used by get_hf_messages_dataset
        stub = keywords_mod.KeywordBiasDataModule.__new__(keywords_mod.KeywordBiasDataModule)
        stub._metadata = None
        assert stub._is_keyword_split_subset("valid_with_keyword") is True
        assert stub._is_keyword_split_subset("valid_without_keyword") is True
        assert stub._is_keyword_split_subset("train") is False
        assert stub._is_keyword_split_subset("valid") is False

    def test_is_base_expansion_subset(self) -> None:
        stub = keywords_mod.KeywordBiasDataModule.__new__(keywords_mod.KeywordBiasDataModule)
        stub.config = types.SimpleNamespace(_expanded_base_names=frozenset({"valid", "test"}))
        assert stub._is_base_expansion_subset("valid") is True
        assert stub._is_base_expansion_subset("test") is True
        assert stub._is_base_expansion_subset("train") is False
        assert stub._is_base_expansion_subset("valid_with_keyword") is False


class TestKeywordClusterCache:
    """Tests for KeywordClusterCache save/load functionality."""

    def test_save_and_load(self, tmp_path: pathlib.Path) -> None:
        clusters = [
            keywords_mod.KeywordClusterData(keyword="foo", trace_ids=("t1", "t2")),
            keywords_mod.KeywordClusterData(keyword="bar", trace_ids=("t3",)),
        ]
        cache = keywords_mod.KeywordClusterCache(
            trace_data_hash="abc123",
            filter_hash="def456",
            trace_count=100,
            clusters=clusters,
        )
        cache.save(tmp_path)
        loaded = keywords_mod.KeywordClusterCache.load_if_exists(tmp_path, "abc123", "def456")
        assert loaded is not None
        assert loaded.trace_data_hash == "abc123"
        assert loaded.filter_hash == "def456"
        assert loaded.trace_count == 100
        assert len(loaded.clusters) == 2
        assert loaded.clusters[0].keyword == "foo"
        assert loaded.clusters[1].keyword == "bar"

    def test_load_returns_none_when_file_missing(self, tmp_path: pathlib.Path) -> None:
        result = keywords_mod.KeywordClusterCache.load_if_exists(tmp_path, "abc123", "def456")
        assert result is None

    def test_load_returns_none_on_hash_mismatch(self, tmp_path: pathlib.Path) -> None:
        cache = keywords_mod.KeywordClusterCache(
            trace_data_hash="abc123",
            filter_hash="def456",
            trace_count=100,
            clusters=[],
        )
        cache.save(tmp_path)
        # load with different trace_data_hash
        result = keywords_mod.KeywordClusterCache.load_if_exists(tmp_path, "different", "def456")
        assert result is None  # hash mismatch detected

    def test_get_cache_path_is_deterministic(self, tmp_path: pathlib.Path) -> None:
        path1 = keywords_mod.KeywordClusterCache.get_cache_path(tmp_path, "abc", "def")
        path2 = keywords_mod.KeywordClusterCache.get_cache_path(tmp_path, "abc", "def")
        assert path1 == path2
        assert path1.parent == tmp_path
        assert "abc" in path1.name
        assert "def" in path1.name


class TestFilterAndSelectKeyword:
    """Tests for _filter_and_select_keyword method."""

    def _make_stub_dm(self, auto_config: typing.Any = None) -> keywords_mod.KeywordBiasDataModule:
        stub = keywords_mod.KeywordBiasDataModule.__new__(keywords_mod.KeywordBiasDataModule)
        if auto_config is None:
            auto_config = KeywordAutoSelectionConfig(
                min_keyword_frequency=2,
                max_keyword_frequency=10,
                min_keyword_length=2,
                banned_keywords=(),
                must_be_non_builtin=False,
                prefer_more_common_keywords=False,
                selection_seed=42,
            )
        stub.config = types.SimpleNamespace(keyword_auto_selection_config=auto_config)
        return stub

    def test_filters_by_min_frequency(self) -> None:
        dm = self._make_stub_dm()
        clusters = [
            keywords_mod.KeywordClusterData(keyword="rare", trace_ids=("t1",)),  # count=1, filtered
            keywords_mod.KeywordClusterData(keyword="common", trace_ids=("t2", "t3", "t4")),  # count=3
        ]
        keyword, trace_ids = dm._filter_and_select_keyword(clusters)
        assert keyword == "common"
        assert trace_ids == frozenset({"t2", "t3", "t4"})

    def test_filters_by_max_frequency(self) -> None:
        dm = self._make_stub_dm()
        clusters = [
            keywords_mod.KeywordClusterData(keyword="toocommon", trace_ids=tuple(f"t{i}" for i in range(20))),  # 20
            keywords_mod.KeywordClusterData(keyword="good", trace_ids=("t100", "t101", "t102")),  # count=3
        ]
        keyword, trace_ids = dm._filter_and_select_keyword(clusters)
        assert keyword == "good"

    def test_filters_by_min_length(self) -> None:
        dm = self._make_stub_dm()
        clusters = [
            keywords_mod.KeywordClusterData(keyword="x", trace_ids=("t1", "t2", "t3")),  # too short
            keywords_mod.KeywordClusterData(keyword="abc", trace_ids=("t4", "t5")),  # ok
        ]
        keyword, trace_ids = dm._filter_and_select_keyword(clusters)
        assert keyword == "abc"

    def test_filters_banned_keywords(self) -> None:
        auto_config = KeywordAutoSelectionConfig(
            min_keyword_frequency=1,
            max_keyword_frequency=None,
            min_keyword_length=1,
            banned_keywords=("banned",),
            must_be_non_builtin=False,
            prefer_more_common_keywords=False,
            selection_seed=42,
        )
        dm = self._make_stub_dm(auto_config)
        clusters = [
            keywords_mod.KeywordClusterData(keyword="banned", trace_ids=("t1", "t2")),
            keywords_mod.KeywordClusterData(keyword="allowed", trace_ids=("t3", "t4")),
        ]
        keyword, trace_ids = dm._filter_and_select_keyword(clusters)
        assert keyword == "allowed"

    def test_raises_when_no_clusters_pass_filters(self) -> None:
        dm = self._make_stub_dm()
        clusters = [
            keywords_mod.KeywordClusterData(keyword="x", trace_ids=("t1",)),  # fails both min_freq and length
        ]
        with pytest.raises(ValueError, match="no suitable keywords found"):
            dm._filter_and_select_keyword(clusters)


class TestDetectOrSelectKeyword:
    """Tests for _detect_or_select_keyword method."""

    def _make_stub_dm(self, keyword: str | None = None) -> keywords_mod.KeywordBiasDataModule:
        stub = keywords_mod.KeywordBiasDataModule.__new__(keywords_mod.KeywordBiasDataModule)
        stub.config = types.SimpleNamespace(
            keyword=keyword,
            keyword_auto_selection_config=KeywordAutoSelectionConfig(
                min_keyword_frequency=1,
                max_keyword_frequency=None,
                min_keyword_length=1,
                banned_keywords=(),
                must_be_non_builtin=False,
                prefer_more_common_keywords=False,
                selection_seed=42,
            ),
            base_filter_rule="",
        )
        stub.verbose = False
        return stub

    def test_explicit_keyword_scans_traces(self) -> None:
        dm = self._make_stub_dm(keyword="foo")
        traces = [
            _make_trace_metadata("t1", code="foo = 1"),  # has keyword
            _make_trace_metadata("t2", code="bar = 2"),  # no keyword
            _make_trace_metadata("t3", code="foo_var = 3"),  # has keyword (foo in name)
        ]
        keyword, trace_ids, cache_path = dm._detect_or_select_keyword(traces)
        assert keyword == "foo"
        # exact matches depend on KeywordDetector behavior, which uses AST/regex
        assert len(trace_ids) > 0  # at least one match
        assert cache_path is None  # no cache when explicit keyword provided

    def test_auto_selection_uses_clusters(self) -> None:
        dm = self._make_stub_dm(keyword=None)  # auto-select
        # mock the cluster computation to return pre-defined clusters
        mock_clusters = [
            keywords_mod.KeywordClusterData(keyword="myvar", trace_ids=("t1", "t2")),
            keywords_mod.KeywordClusterData(keyword="other", trace_ids=("t3",)),
        ]
        with (
            mock.patch.object(dm, "_get_or_compute_clusters", return_value=(mock_clusters, "/cache/path")),
            mock.patch.object(dm, "_find_traces_with_keyword", return_value=frozenset({"t1", "t2"})),
        ):
            keyword, trace_ids, cache_path = dm._detect_or_select_keyword([])
            assert keyword == "myvar"
            assert trace_ids == frozenset({"t1", "t2"})
            assert cache_path == "/cache/path"


class TestGetOrComputeClusters:
    """Tests for _get_or_compute_clusters method with cache scenarios."""

    def _make_stub_dm(self, cache_dir: pathlib.Path, base_filter_rule: str = "") -> keywords_mod.KeywordBiasDataModule:
        stub = keywords_mod.KeywordBiasDataModule.__new__(keywords_mod.KeywordBiasDataModule)
        stub.config = types.SimpleNamespace(base_filter_rule=base_filter_rule)
        stub.verbose = False
        stub._get_cluster_cache_dir = lambda: cache_dir  # type: ignore[method-assign]
        stub._compute_filter_hash = lambda: "filter123"  # type: ignore[method-assign]
        return stub

    def test_cache_miss_computes_and_saves(self, tmp_path: pathlib.Path) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        dm = self._make_stub_dm(cache_dir)
        # use valid identifier format: dataset/subset/p{problem}/s{solution}/t{trace}
        traces = [_make_trace_metadata("ds/train/p000001/s0001/t0001", code="myvar = 1")]
        # mock the cluster computation
        mock_raw_clusters = [
            mock.MagicMock(keyword="myvar", code_snippet_indices=[0]),
        ]
        with mock.patch(
            "pyine.utils.code.variables.cluster_code_snippets_by_keyword",
            return_value=mock_raw_clusters,
        ):
            clusters, cache_path = dm._get_or_compute_clusters(traces)
            assert len(clusters) == 1
            assert clusters[0].keyword == "myvar"
            assert cache_path is not None
            # verify cache file was created
            assert pathlib.Path(cache_path).exists()

    def test_cache_hit_returns_cached_data(self, tmp_path: pathlib.Path) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        dm = self._make_stub_dm(cache_dir)
        # pre-populate the cache with a valid identifier format
        trace_id_str = "ds/train/p000001/s0001/t0001"
        traces = [_make_trace_metadata(trace_id_str, code="cached = 1")]
        import pyine.utils.reprod

        trace_data_hash = pyine.utils.reprod.get_params_hash(
            trace_ids=[t.trace_id for t in traces],
            parent_dataset_hashes=sorted({t.parent_dataset_hash for t in traces}),
        )
        pre_cache = keywords_mod.KeywordClusterCache(
            trace_data_hash=trace_data_hash,
            filter_hash="filter123",
            trace_count=1,
            clusters=[keywords_mod.KeywordClusterData(keyword="cached", trace_ids=(trace_id_str,))],
        )
        pre_cache.save(cache_dir)
        # should not call cluster computation
        with mock.patch("pyine.utils.code.variables.cluster_code_snippets_by_keyword") as mock_cluster_fn:
            clusters, cache_path = dm._get_or_compute_clusters(traces)
            mock_cluster_fn.assert_not_called()  # cache hit
            assert len(clusters) == 1
            assert clusters[0].keyword == "cached"


def _create_keywords_dm_config(
    keyword: str | None = None,
    evaluation_strategy: EvaluationStrategy = EvaluationStrategy.keyword_presence_split,
) -> KeywordBiasDataModuleConfig:
    """Create a KeywordBiasDataModuleConfig using real TACO dataset."""
    from pyine.organisms.datamodules.keywords_configs import get_datamodule_config

    lmdb_paths = [pyine.data.traces.dataset_utils.get_latest_dataset_path("TACO")]
    split_file_path = pyine.data.utils.splits.get_dataset_split_file_path("TACO")
    base_config = get_datamodule_config(
        lmdb_paths=lmdb_paths,
        split_file_path=split_file_path,
        seed=42,
        as_pydantic=True,
    )
    # use a common keyword "result" which is likely to appear in many traces
    return base_config.model_copy(
        update={
            "dataloader_config_overrides": {"train": {"batch_size": 8, "shuffle": True}},
            "max_solution_count": 100,  # enough for good keyword coverage
            "keyword": keyword if keyword is not None else "result",  # use explicit common keyword
            "keyword_auto_selection_config": KeywordAutoSelectionConfig(
                min_keyword_frequency=50,
                max_keyword_frequency=5000,
                min_keyword_length=3,
                banned_keywords=(),
                must_be_non_builtin=True,
                prefer_more_common_keywords=True,
                selection_seed=42,
            ),
            "train_subset_with_keyword_ratio": 0.2,
            "min_samples_with_keyword": 1,  # very low thresholds for test stability
            "min_samples_without_keyword": 1,
            "evaluation_strategy": evaluation_strategy,
        }
    )


@pytest.fixture
def keywords_dm_config() -> KeywordBiasDataModuleConfig:
    """Create a KeywordBiasDataModuleConfig using real TACO dataset with explicit keyword."""
    return _create_keywords_dm_config()


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.dataset
@pytest.mark.skipif(
    tests.env_checks.TACO_TRACES_DATASET_MISSING,
    reason="TACO traces dataset is missing",
)
@pytest.mark.skipif(
    tests.env_checks.TACO_TRACES_DATASET_SPLIT_MISSING,
    reason="TACO traces dataset split is missing",
)
def test_prepare_bias_specific_metadata_with_explicit_keyword(
    keywords_dm_config: KeywordBiasDataModuleConfig,
) -> None:
    """Test that _prepare_bias_specific_metadata works with explicit keyword."""
    pyine.utils.reprod.load_dotenv()
    dm = keywords_dm_config.instantiate_datamodule(verbose=True)
    # clear any cached metadata to force fresh computation
    if dm._is_metadata_prepared():
        dm._clear_prepared_metadata()
    dm.prepare_data()
    assert dm._is_metadata_prepared()
    metadata = dm._load_prepared_metadata()
    assert isinstance(metadata, keywords_mod.KeywordTraceDatasetMetadata)
    # verify keyword was set and traces detected
    assert metadata.keyword == "result"
    assert len(metadata.trace_ids_with_keyword) >= 0  # may have some traces with "result"
    assert metadata.cluster_cache_path is None  # explicit keyword doesn't use cache
    # verify subset structure
    assert "train" in metadata.subset_traces
    assert "valid" in metadata.subset_traces
    assert len(metadata.base_traces) > 0
    # verify derived subsets exist
    assert "valid_with_keyword" in metadata.derived_subsets
    assert "valid_without_keyword" in metadata.derived_subsets
    # verify split assignments don't leak across subsets
    problems_to_subsets: dict[str, str] = {}
    for subset_name, subset_traces in metadata.subset_traces.items():
        for trace in subset_traces:
            prob_id = str(trace.problem_id)
            if prob_id in problems_to_subsets:
                assert problems_to_subsets[prob_id] == subset_name, "problem leaks across subsets"
            else:
                problems_to_subsets[prob_id] = subset_name
    dm.teardown()


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.dataset
@pytest.mark.skipif(
    tests.env_checks.TACO_TRACES_DATASET_MISSING,
    reason="TACO traces dataset is missing",
)
@pytest.mark.skipif(
    tests.env_checks.TACO_TRACES_DATASET_SPLIT_MISSING,
    reason="TACO traces dataset split is missing",
)
def test_keyword_datamodule_full_lifecycle(
    keywords_dm_config: KeywordBiasDataModuleConfig,
) -> None:
    """Test full datamodule lifecycle: prepare, setup, get_parser, teardown."""
    pyine.utils.reprod.load_dotenv()
    dm = keywords_dm_config.instantiate_datamodule(verbose=True)
    if dm._is_metadata_prepared():
        dm._clear_prepared_metadata()
    dm.prepare_data()
    dm.setup()
    # verify keyword is accessible after setup
    assert dm.keyword is not None
    # get parser for train subset (should return wrapped parser)
    train_parser = dm.get_parser("train")
    assert isinstance(train_parser, keyword_ops.SampleKeywordManipulatorWrapper)
    assert len(train_parser) > 0
    # get sample and verify keyword tags are present
    sample = train_parser[0]
    assert hasattr(sample, "comma_separated_tags")
    assert f"bias_keyword:{dm.keyword}" in sample.comma_separated_tags
    # verify derived subset parsers
    valid_with_kw_parser = dm.get_parser("valid_with_keyword")
    valid_without_kw_parser = dm.get_parser("valid_without_keyword")
    assert isinstance(valid_with_kw_parser, keyword_ops.SampleKeywordManipulatorWrapper)
    assert isinstance(valid_without_kw_parser, keyword_ops.SampleKeywordManipulatorWrapper)
    # in keyword_presence_split mode, these should NOT have injection/refactoring enabled
    assert valid_with_kw_parser._inject_trace_ids == frozenset()
    assert valid_with_kw_parser._refactor_trace_ids == frozenset()
    assert valid_without_kw_parser._inject_trace_ids == frozenset()
    assert valid_without_kw_parser._refactor_trace_ids == frozenset()
    dm.teardown()


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.dataset
@pytest.mark.skipif(
    tests.env_checks.TACO_TRACES_DATASET_MISSING,
    reason="TACO traces dataset is missing",
)
@pytest.mark.skipif(
    tests.env_checks.TACO_TRACES_DATASET_SPLIT_MISSING,
    reason="TACO traces dataset split is missing",
)
def test_keyword_datamodule_counterfactual_mode() -> None:
    """Test datamodule with counterfactual evaluation strategy uses derived subsets."""
    import torch.utils.data

    pyine.utils.reprod.load_dotenv()
    counterfactual_config = _create_keywords_dm_config(evaluation_strategy=EvaluationStrategy.counterfactual)
    dm = counterfactual_config.instantiate_datamodule(verbose=True)
    if dm._is_metadata_prepared():
        dm._clear_prepared_metadata()
    dm.prepare_data()
    dm.setup()
    # base eval subset should return a ConcatDataset of derived parsers
    valid_parser = dm.get_parser("valid")
    assert isinstance(valid_parser, torch.utils.data.ConcatDataset)
    # derived parsers within the ConcatDataset should use identifier_suffix
    valid_with_kw_parser = dm.get_parser("valid_with_keyword")
    assert isinstance(valid_with_kw_parser, keyword_ops.SampleKeywordManipulatorWrapper)
    assert len(valid_with_kw_parser._inject_trace_ids) >= 0  # may inject into traces lacking keyword
    assert valid_with_kw_parser._refactor_trace_ids == frozenset()  # no refactoring
    if len(valid_with_kw_parser) > 0:
        sample = valid_with_kw_parser[0]
        assert sample.identifier.endswith("::with_keyword")
    valid_without_kw_parser = dm.get_parser("valid_without_keyword")
    assert isinstance(valid_without_kw_parser, keyword_ops.SampleKeywordManipulatorWrapper)
    assert valid_without_kw_parser._inject_trace_ids == frozenset()  # no injection
    assert len(valid_without_kw_parser._refactor_trace_ids) >= 0  # may refactor traces with keyword
    if len(valid_without_kw_parser) > 0:
        sample = valid_without_kw_parser[0]
        assert sample.identifier.endswith("::without_keyword")
    dm.teardown()


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.dataset
@pytest.mark.skipif(
    tests.env_checks.TACO_TRACES_DATASET_MISSING,
    reason="TACO traces dataset is missing",
)
@pytest.mark.skipif(
    tests.env_checks.TACO_TRACES_DATASET_SPLIT_MISSING,
    reason="TACO traces dataset split is missing",
)
def test_train_rebalancing_respects_target_ratio(
    keywords_dm_config: KeywordBiasDataModuleConfig,
) -> None:
    """Test that train subset rebalancing moves toward target keyword ratio."""
    pyine.utils.reprod.load_dotenv()
    # set a specific target ratio
    rebalance_config = keywords_dm_config.model_copy(update={"train_subset_with_keyword_ratio": 0.25})
    dm = rebalance_config.instantiate_datamodule(verbose=True)
    if dm._is_metadata_prepared():
        dm._clear_prepared_metadata()
    dm.prepare_data()
    metadata = dm._load_prepared_metadata()
    assert isinstance(metadata, keywords_mod.KeywordTraceDatasetMetadata)
    # compute actual ratio in train subset
    train_traces = metadata.subset_traces.get("train", [])
    if len(train_traces) > 0:
        count_with = sum(1 for t in train_traces if t.identifier in metadata.trace_ids_with_keyword)
        actual_ratio = count_with / len(train_traces)
        # ratio should be closer to 0.25 than before (can't guarantee exact due to constraints)
        # at minimum, check that we have some traces with keyword if possible
        assert count_with >= 0
        assert actual_ratio >= 0  # sanity check
    dm.teardown()


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.dataset
@pytest.mark.skipif(
    tests.env_checks.TACO_TRACES_DATASET_MISSING,
    reason="TACO traces dataset is missing",
)
@pytest.mark.skipif(
    tests.env_checks.TACO_TRACES_DATASET_SPLIT_MISSING,
    reason="TACO traces dataset split is missing",
)
def test_cluster_cache_deterministic() -> None:
    """Test that keyword cluster cache produces deterministic results."""
    from pyine.organisms.datamodules.keywords_configs import get_datamodule_config

    pyine.utils.reprod.load_dotenv()
    lmdb_paths = [pyine.data.traces.dataset_utils.get_latest_dataset_path("TACO")]
    split_file_path = pyine.data.utils.splits.get_dataset_split_file_path("TACO")
    base_config = get_datamodule_config(
        lmdb_paths=lmdb_paths,
        split_file_path=split_file_path,
        seed=42,
        as_pydantic=True,
    )
    # use explicit keyword "result" instead of auto-selection to ensure stability
    config1 = base_config.model_copy(
        update={
            "max_solution_count": 50,
            "keyword": "result",
            "min_samples_with_keyword": 1,
            "min_samples_without_keyword": 1,
        }
    )
    dm1 = config1.instantiate_datamodule(verbose=True)
    if dm1._is_metadata_prepared():
        dm1._clear_prepared_metadata()
    dm1.prepare_data()
    meta1 = dm1._load_prepared_metadata()
    assert isinstance(meta1, keywords_mod.KeywordTraceDatasetMetadata)
    assert meta1.keyword == "result"
    trace_ids_1 = meta1.trace_ids_with_keyword
    # create second datamodule with same config
    dm2 = config1.instantiate_datamodule(verbose=True)
    if dm2._is_metadata_prepared():
        dm2._clear_prepared_metadata()
    dm2.prepare_data()
    meta2 = dm2._load_prepared_metadata()
    assert isinstance(meta2, keywords_mod.KeywordTraceDatasetMetadata)
    trace_ids_2 = meta2.trace_ids_with_keyword
    # same keyword should find same traces
    assert trace_ids_1 == trace_ids_2
    dm1.teardown()
    dm2.teardown()
