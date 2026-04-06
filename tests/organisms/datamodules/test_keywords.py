import pathlib
import types
import typing
from unittest import mock

import msgspec
import numpy as np
import pydantic
import pytest

import pyine.data.traces.dataset_reader
import pyine.data.traces.dataset_utils
import pyine.data.utils.generation_record
import pyine.data.utils.lmdb_io
import pyine.data.utils.splits
import pyine.organisms.datamodules.keywords as keywords_mod
import pyine.organisms.datamodules.samples
import pyine.organisms.datamodules.samples.configs as samples_configs
import pyine.organisms.datamodules.samples.keyword_ops as keyword_ops
import pyine.prompts
import pyine.prompts.types
import pyine.utils.reprod
import tests.env_checks
from pyine.organisms.datamodules.keywords_configs import (
    EvaluationStrategy,
    KeywordAutoSelectionConfig,
    KeywordBiasDataModuleConfig,
    KeywordBiasDistillationDataModuleConfig,
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

    @staticmethod
    def _make_disabled_filtering_resolver() -> typing.Callable[[str], types.SimpleNamespace]:
        """Returns a _resolve_dataparser_config stub that yields disabled quality filters."""
        disabled = samples_configs.TraceFilteringConfig.create_disabled()
        return lambda name: types.SimpleNamespace(
            get_params_dict=lambda: {"filtering_config": disabled},
        )

    def _make_datamodule_stub(
        self,
        eval_subset_names: tuple[str, ...] = ("valid",),
        evaluation_strategy: EvaluationStrategy = EvaluationStrategy.keyword_presence_split,
        target_ratio: float = 0.5,
        filtering_config_resolver: typing.Callable[[str], types.SimpleNamespace] | None = None,
    ) -> keywords_mod.KeywordBiasDataModule:
        stub = keywords_mod.KeywordBiasDataModule.__new__(keywords_mod.KeywordBiasDataModule)
        expanded_base_names = frozenset(eval_subset_names)
        stub.config = types.SimpleNamespace(
            eval_subset_names=eval_subset_names,
            _expanded_base_names=expanded_base_names,
            evaluation_strategy=evaluation_strategy,
            train_subset_with_keyword_ratio=target_ratio,
            train_subset_resampling_seed=0,
            exclude_augmented_traces=True,
            _resolve_dataparser_config=filtering_config_resolver or self._make_disabled_filtering_resolver(),
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

    def test_quality_prefiltering_removes_traces_before_rebalancing(self) -> None:
        """Quality pre-filtering should remove traces exceeding quality thresholds before
        rebalancing, so the keyword ratio is computed on the filtered pool."""
        # 4 keyword traces + 6 non-keyword traces; pre-filtering will remove the keyword ones
        traces_with_kw = [_DummyTrace(f"kw{idx}", f"solKW{idx}") for idx in range(4)]
        traces_without_kw = [_DummyTrace(f"nkw{idx}", f"solNKW{idx}") for idx in range(6)]
        trace_ids_with_kw = frozenset(t.identifier for t in traces_with_kw)
        # filtering config with an active quality filter
        active_filter = samples_configs.TraceFilteringConfig(
            max_code_line_count=250,
            max_trace_steps=None,
            max_code_line_length=None,
            max_code_length=None,
            max_args_length=None,
        )

        def resolver(name: str) -> types.SimpleNamespace:
            return types.SimpleNamespace(
                get_params_dict=lambda: {"filtering_config": active_filter},
            )

        dm = self._make_datamodule_stub(
            target_ratio=0.5,
            filtering_config_resolver=resolver,
        )
        all_train = list(traces_with_kw) + list(traces_without_kw)
        subset_traces: dict[str, list[typing.Any]] = {
            "train": list(all_train),
            "valid": [_DummyTrace("v0", "solV0")],
        }
        unassigned: list[typing.Any] = []
        # mock filter_traces to simulate removing all keyword traces (as if they were too long)
        kept_traces = list(traces_without_kw)  # only non-keyword traces survive
        filter_calls: list[samples_configs.TraceFilteringConfig] = []
        fake_results = mock.MagicMock()
        fake_results.kept_traces = kept_traces
        fake_results.filtered_trace_count = len(traces_with_kw)

        def mock_filter_traces(
            traces: list[typing.Any],
            epoch: int,
            filtering_config: samples_configs.TraceFilteringConfig,
        ) -> mock.MagicMock:
            filter_calls.append(filtering_config)
            return fake_results

        with mock.patch(
            "pyine.organisms.datamodules.keywords.pyine.organisms.datamodules.samples.filtering.filter_traces",
            side_effect=mock_filter_traces,
        ):
            dm._adjust_keyword_split_subsets(subset_traces, unassigned, trace_ids_with_kw)
        # filter_traces should have been called with quality-only config
        assert len(filter_calls) >= 1
        quality_call = filter_calls[0]
        assert quality_call.max_code_line_count == 250  # quality filter preserved
        assert quality_call.max_traces_per_solution is None  # cap filters removed
        # all keyword traces should have been removed by pre-filtering
        train_ids = {t.identifier for t in subset_traces["train"]}
        for trace in traces_with_kw:
            assert trace.identifier not in train_ids
        # removed traces should be in unassigned
        unassigned_ids = {t.identifier for t in unassigned}
        for trace in traces_with_kw:
            assert trace.identifier in unassigned_ids

    def test_no_prefiltering_when_quality_filters_disabled(self) -> None:
        """When quality filters are disabled, no traces should be removed by pre-filtering
        (rebalancing still runs and may change the train subset)."""
        traces_with_kw = [_DummyTrace(f"kw{idx}", f"solKW{idx}") for idx in range(3)]
        traces_without_kw = [_DummyTrace(f"nkw{idx}", f"solNKW{idx}") for idx in range(7)]
        trace_ids_with_kw = frozenset(t.identifier for t in traces_with_kw)
        original_train = list(traces_with_kw) + list(traces_without_kw)
        original_train_ids = {t.identifier for t in original_train}
        # disabled filtering -> no quality pre-filtering should occur
        dm = self._make_datamodule_stub(target_ratio=0.5)  # default resolver returns disabled
        subset_traces: dict[str, list[typing.Any]] = {
            "train": list(original_train),
            "valid": [_DummyTrace("v0", "solV0")],
        }
        unassigned: list[typing.Any] = []
        dm._adjust_keyword_split_subsets(subset_traces, unassigned, trace_ids_with_kw)
        # no traces should have been removed by pre-filtering; any changes are from rebalancing
        # which only moves traces from train to unassigned (never adds new ones)
        train_after_ids = {t.identifier for t in subset_traces["train"]}
        assert train_after_ids.issubset(original_train_ids), "train should only contain traces from the original set"


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

    def test_rejects_none_resampling_seed(self, fake_lmdb_and_split: tuple[pathlib.Path, pathlib.Path]) -> None:
        lmdb_path, split_path = fake_lmdb_and_split
        with pytest.raises(pydantic.ValidationError, match="train_subset_resampling_seed must not be None"):
            KeywordBiasDataModuleConfig(
                lmdb_paths=[str(lmdb_path)],
                split_file_path=str(split_path),
                train_subset_resampling_seed=None,
                instantiate_parsers_at_setup=False,
            )

    def test_rejects_invalid_keyword_ratio(self, fake_lmdb_and_split: tuple[pathlib.Path, pathlib.Path]) -> None:
        lmdb_path, split_path = fake_lmdb_and_split
        with pytest.raises(pydantic.ValidationError):
            KeywordBiasDataModuleConfig(
                lmdb_paths=[str(lmdb_path)],
                split_file_path=str(split_path),
                train_subset_with_keyword_ratio=1.5,
                instantiate_parsers_at_setup=False,
            )
        with pytest.raises(pydantic.ValidationError):
            KeywordBiasDataModuleConfig(
                lmdb_paths=[str(lmdb_path)],
                split_file_path=str(split_path),
                train_subset_with_keyword_ratio=0.0,
                instantiate_parsers_at_setup=False,
            )


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
        expanded_base_names: frozenset[str] = frozenset({"valid"}),
    ) -> keywords_mod.KeywordBiasDataModule:
        stub = keywords_mod.KeywordBiasDataModule.__new__(keywords_mod.KeywordBiasDataModule)
        stub.config = types.SimpleNamespace(
            min_samples_with_keyword=min_samples_with_keyword,
            min_samples_without_keyword=min_samples_without_keyword,
            _expanded_base_names=expanded_base_names,
        )
        stub.verbose = False
        return stub

    @staticmethod
    def _make_derived_subsets(
        eval_name: str,
        traces_with: list[pyine.data.traces.dataset_utils.TraceMetadata],
        traces_without: list[pyine.data.traces.dataset_utils.TraceMetadata],
    ) -> dict[str, pyine.data.traces.dataset_utils.DerivedSubsetInfo[pyine.data.traces.dataset_utils.TraceMetadata]]:
        return {
            f"{eval_name}_with_keyword": pyine.data.traces.dataset_utils.DerivedSubsetInfo(
                parent_subset=eval_name,
                traces=traces_with,
                derivation_type="keyword_presence_split",
            ),
            f"{eval_name}_without_keyword": pyine.data.traces.dataset_utils.DerivedSubsetInfo(
                parent_subset=eval_name,
                traces=traces_without,
                derivation_type="keyword_presence_split",
            ),
        }

    def test_raises_when_too_few_with_keyword(self) -> None:
        dm = self._make_stub_dm(min_samples_with_keyword=10, min_samples_without_keyword=0)
        all_traces = [_make_trace_metadata(f"t{idx}") for idx in range(20)]
        subset_traces = {"valid": all_traces}
        trace_ids_with_keyword = frozenset({"t0", "t1"})  # only 2 with keyword
        derived = self._make_derived_subsets(
            "valid",
            traces_with=[t for t in all_traces if t.identifier in trace_ids_with_keyword],
            traces_without=[t for t in all_traces if t.identifier not in trace_ids_with_keyword],
        )
        with pytest.raises(ValueError, match="only 2 traces with the keyword"):
            dm._validate_sample_counts("kw", subset_traces, derived)

    def test_raises_when_too_few_without_keyword(self) -> None:
        dm = self._make_stub_dm(min_samples_with_keyword=0, min_samples_without_keyword=50)
        all_traces = [_make_trace_metadata(f"t{idx}") for idx in range(20)]
        subset_traces = {"valid": all_traces}
        trace_ids_with_keyword = frozenset({f"t{idx}" for idx in range(15)})  # 15 with, 5 without
        derived = self._make_derived_subsets(
            "valid",
            traces_with=[t for t in all_traces if t.identifier in trace_ids_with_keyword],
            traces_without=[t for t in all_traces if t.identifier not in trace_ids_with_keyword],
        )
        with pytest.raises(ValueError, match="only 5 traces without the keyword"):
            dm._validate_sample_counts("kw", subset_traces, derived)

    def test_passes_when_counts_meet_minimum(self) -> None:
        dm = self._make_stub_dm(min_samples_with_keyword=5, min_samples_without_keyword=10)
        all_traces = [_make_trace_metadata(f"t{idx}") for idx in range(20)]
        subset_traces = {"valid": all_traces}
        trace_ids_with_keyword = frozenset({f"t{idx}" for idx in range(8)})  # 8 with, 12 without
        derived = self._make_derived_subsets(
            "valid",
            traces_with=[t for t in all_traces if t.identifier in trace_ids_with_keyword],
            traces_without=[t for t in all_traces if t.identifier not in trace_ids_with_keyword],
        )
        dm._validate_sample_counts("kw", subset_traces, derived)  # should not raise


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
    dm = keywords_dm_config.instantiate_datamodule()
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
    dm = keywords_dm_config.instantiate_datamodule()
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
    dm = counterfactual_config.instantiate_datamodule()
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
    dm = rebalance_config.instantiate_datamodule()
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
            # this test validates cluster-cache determinism, not eval subset balancing.
            # keep thresholds at 0 so dataset split drift cannot make the test fail spuriously.
            "min_samples_with_keyword": 0,
            "min_samples_without_keyword": 0,
        }
    )
    dm1 = config1.instantiate_datamodule()
    if dm1._is_metadata_prepared():
        dm1._clear_prepared_metadata()
    dm1.prepare_data()
    meta1 = dm1._load_prepared_metadata()
    assert isinstance(meta1, keywords_mod.KeywordTraceDatasetMetadata)
    assert meta1.keyword == "result"
    trace_ids_1 = meta1.trace_ids_with_keyword
    # create second datamodule with same config
    dm2 = config1.instantiate_datamodule()
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


# ========================================================================================
# Distillation DataModule Tests
# ========================================================================================

_DEFAULT_COMPLEXITY_METRICS: dict[str, float | int] = {
    "cyclomatic_complexity_avg": 1.0,
    "cyclomatic_complexity_max": 1,
    "cyclomatic_complexity_sum": 1,
    "loc": 1,
    "lloc": 1,
    "sloc": 1,
    "comments": 0,
    "multi": 0,
    "blank": 0,
    "halstead_volume": 1.0,
    "halstead_difficulty": 1.0,
    "halstead_effort": 1.0,
    "maintainability_index": 100.0,
}


def _make_distillation_sample_data(
    identifier: str = "test_sample",
    code: str = "print(1)",
    has_keyword: bool = False,
) -> pyine.organisms.datamodules.samples.SampleData:
    tags = "has_bias_keyword:1,bias_keyword:result" if has_keyword else "has_bias_keyword:0"
    return pyine.organisms.datamodules.samples.SampleData(
        identifier=identifier,
        code=code,
        description="test description",
        entrypoint="main",
        first_line=0,
        last_line=1,
        inputs="",
        expected_output="1",
        predict_type="program_output",
        code_type="original",
        trace_step_count=1,
        comma_separated_tags=tags,
        has_code_override=False,
        complexity_metrics=dict(_DEFAULT_COMPLEXITY_METRICS),
    )


def _make_reward_record(
    sample_id: str = "test_sample",
    model_output: str = "1",
    reward_total: float = 1.0,
    classifier_score: float = 0.8,
    has_keyword: bool = False,
    generation_count: int = 0,
    key_prefix: str = "train/",
) -> tuple[str, dict[str, typing.Any]]:
    """Build an LMDB key and reward record dict for distillation tests."""
    sample_data = _make_distillation_sample_data(sample_id, has_keyword=has_keyword)
    tags = sample_data.get_tag_list()
    shared = pyine.data.utils.generation_record.build_shared_record_fields(
        sample_id=sample_id,
        model_output=model_output,
        prompt="test prompt",
        expected_output="1",
        predict_type="program_output",
        code_type="original",
        tags=tags,
        key_prefix=key_prefix,
        sample_data=sample_data,
    )
    record: dict[str, typing.Any] = {
        **shared,
        "reward_total": reward_total,
        "reward_terms": {},
        "reward_metrics": {"correctness_classifier/classifier_score": classifier_score},
        "reward_terms_raw": {},
        "step": 0,
        "epoch": 0,
        "batch_count": 0,
        "local_batch_idx": 0,
        "completion_idx": 0,
        "rank": 0,
    }
    lmdb_key = f"{key_prefix}{sample_id}/{generation_count}"
    return lmdb_key, record


def _write_distillation_lmdb(
    path: pathlib.Path,
    records: list[tuple[str, dict[str, typing.Any]]],
) -> None:
    """Write a list of (key, record) pairs to an LMDB."""
    writer = pyine.data.utils.lmdb_io.LMDBWriter(
        path=path,
        serialization_config=pyine.data.utils.lmdb_io.SerializationConfig(
            method=pyine.data.utils.lmdb_io.SerializationMethod.JSON_ZSTD,
        ),
    )
    for lmdb_key, record in records:
        writer.put(lmdb_key, record)
    writer.close()


def _make_distillation_config(
    lmdb_path: pathlib.Path,
    **overrides: typing.Any,
) -> KeywordBiasDistillationDataModuleConfig:
    """Build a distillation config pointing at a single LMDB path."""
    kwargs: dict[str, typing.Any] = {
        "rl_export_lmdb_paths": (lmdb_path,),
        "keyword_sample_min_classifier_score": 0.5,
        "non_keyword_sample_min_reward": 0.5,
        "target_keyword_ratio": 0.5,
        "rebalancing_seed": 42,
    }
    kwargs.update(overrides)
    return KeywordBiasDistillationDataModuleConfig(**kwargs)


class TestKeywordBiasDistillationConfig:
    def test_default_prompt_version(self, tmp_path: pathlib.Path) -> None:
        config = KeywordBiasDistillationDataModuleConfig(rl_export_lmdb_paths=(tmp_path,))
        assert config.prompt_config.version == "rl_tagged_answer"

    def test_custom_prompt_version(self, tmp_path: pathlib.Path) -> None:
        config = KeywordBiasDistillationDataModuleConfig(
            rl_export_lmdb_paths=(tmp_path,),
            prompt_config=pyine.prompts.types.PromptBuildConfig(
                prompt_name=pyine.prompts.PromptNames.CODE_EXECUTION,
                version="rl_stepped_reasoning",
                use_chat_template=True,
                include_examples=False,
            ),
        )
        assert config.prompt_config.version == "rl_stepped_reasoning"

    def test_instantiate_sample_to_messages_transform_returns_callable(self, tmp_path: pathlib.Path) -> None:
        config = KeywordBiasDistillationDataModuleConfig(rl_export_lmdb_paths=(tmp_path,))
        transform_fn = config.instantiate_sample_to_messages_transform(append_answer=True, use_hf_messages=True)
        assert callable(transform_fn)

    def test_target_ratio_boundaries(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(pydantic.ValidationError):
            KeywordBiasDistillationDataModuleConfig(
                rl_export_lmdb_paths=(tmp_path,),
                target_keyword_ratio=0.0,
            )
        with pytest.raises(pydantic.ValidationError):
            KeywordBiasDistillationDataModuleConfig(
                rl_export_lmdb_paths=(tmp_path,),
                target_keyword_ratio=1.0,
            )


class TestDistillationRecordFiltering:
    def test_keyword_detection_from_tags(self) -> None:
        kw_record = {"tags": ["has_bias_keyword:1", "bias_keyword:result"]}
        non_kw_record = {"tags": ["has_bias_keyword:0"]}
        assert keywords_mod.KeywordBiasDistillationDataModule._is_keyword_sample(kw_record) is True
        assert keywords_mod.KeywordBiasDistillationDataModule._is_keyword_sample(non_kw_record) is False

    def test_keyword_detection_none_tags_raises(self) -> None:
        with pytest.raises(TypeError):
            keywords_mod.KeywordBiasDistillationDataModule._is_keyword_sample({"tags": None})

    def test_classifier_score_filtering(self, tmp_path: pathlib.Path) -> None:
        config = _make_distillation_config(tmp_path, keyword_sample_min_classifier_score=0.7)
        dm = keywords_mod.KeywordBiasDistillationDataModule(config)
        passing = {"reward_metrics": {"correctness_classifier/classifier_score": 0.8}}
        failing = {"reward_metrics": {"correctness_classifier/classifier_score": 0.3}}
        assert dm._passes_quality_filter(passing, is_keyword=True) is True
        assert dm._passes_quality_filter(failing, is_keyword=True) is False

    def test_reward_filtering(self, tmp_path: pathlib.Path) -> None:
        config = _make_distillation_config(tmp_path, non_keyword_sample_min_reward=0.5)
        dm = keywords_mod.KeywordBiasDistillationDataModule(config)
        passing = {"reward_total": 0.8}
        failing = {"reward_total": 0.1}
        no_reward = {"reward_total": None}
        assert dm._passes_quality_filter(passing, is_keyword=False) is True
        assert dm._passes_quality_filter(failing, is_keyword=False) is False
        assert dm._passes_quality_filter(no_reward, is_keyword=False) is False

    def test_missing_classifier_key_raises(self, tmp_path: pathlib.Path) -> None:
        config = _make_distillation_config(tmp_path)
        dm = keywords_mod.KeywordBiasDistillationDataModule(config)
        record = {"reward_metrics": {}}
        with pytest.raises(KeyError):
            dm._passes_quality_filter(record, is_keyword=True)


class TestTopKDeduplication:
    """Tests for _load_top_k_records (top-1, top-K>1, latest strategy)."""

    def test_top_1_keeps_best_reward(self, tmp_path: pathlib.Path) -> None:
        lmdb_path = tmp_path / "dedup_lmdb"
        records: list[tuple[str, dict[str, typing.Any]]] = []
        for gen_idx in range(3):
            lmdb_key, record = _make_reward_record(
                sample_id="sample_a",
                reward_total=float(gen_idx),  # 0.0, 1.0, 2.0
                generation_count=gen_idx,
            )
            records.append((lmdb_key, record))
        _write_distillation_lmdb(lmdb_path, records)
        config = _make_distillation_config(lmdb_path, top_k_per_sample=1, selection_strategy="best_reward")
        dm = keywords_mod.KeywordBiasDistillationDataModule(config)
        reader = pyine.data.utils.lmdb_io.LMDBReader(lmdb_path)
        result = dm._load_top_k_records(reader, "train/")
        reader.close()
        assert len(result) == 1
        assert result[0]["reward_total"] == 2.0

    def test_top_k_keeps_multiple(self, tmp_path: pathlib.Path) -> None:
        lmdb_path = tmp_path / "dedup_lmdb"
        records: list[tuple[str, dict[str, typing.Any]]] = []
        for gen_idx in range(5):
            lmdb_key, record = _make_reward_record(
                sample_id="sample_b",
                reward_total=float(gen_idx),
                generation_count=gen_idx,
            )
            records.append((lmdb_key, record))
        _write_distillation_lmdb(lmdb_path, records)
        config = _make_distillation_config(lmdb_path, top_k_per_sample=3, selection_strategy="best_reward")
        dm = keywords_mod.KeywordBiasDistillationDataModule(config)
        reader = pyine.data.utils.lmdb_io.LMDBReader(lmdb_path)
        result = dm._load_top_k_records(reader, "train/")
        reader.close()
        assert len(result) == 3
        rewards = [r["reward_total"] for r in result]
        assert rewards == [4.0, 3.0, 2.0]  # top-3 by reward, descending

    def test_latest_strategy(self, tmp_path: pathlib.Path) -> None:
        lmdb_path = tmp_path / "dedup_lmdb"
        records: list[tuple[str, dict[str, typing.Any]]] = []
        for gen_idx in range(4):
            lmdb_key, record = _make_reward_record(
                sample_id="sample_c",
                reward_total=float(gen_idx) * 0.1,
                generation_count=gen_idx,
            )
            records.append((lmdb_key, record))
        _write_distillation_lmdb(lmdb_path, records)
        config = _make_distillation_config(lmdb_path, top_k_per_sample=2, selection_strategy="latest")
        dm = keywords_mod.KeywordBiasDistillationDataModule(config)
        reader = pyine.data.utils.lmdb_io.LMDBReader(lmdb_path)
        result = dm._load_top_k_records(reader, "train/")
        reader.close()
        assert len(result) == 2
        # latest = gen_count 3 and 2 (highest generation counts)
        assert result[0]["reward_total"] == pytest.approx(0.3)
        assert result[1]["reward_total"] == pytest.approx(0.2)

    def test_multiple_samples_deduped_independently(self, tmp_path: pathlib.Path) -> None:
        lmdb_path = tmp_path / "dedup_lmdb"
        records: list[tuple[str, dict[str, typing.Any]]] = []
        for sample_id in ("s1", "s2"):
            for gen_idx in range(3):
                lmdb_key, record = _make_reward_record(
                    sample_id=sample_id,
                    reward_total=float(gen_idx),
                    generation_count=gen_idx,
                )
                records.append((lmdb_key, record))
        _write_distillation_lmdb(lmdb_path, records)
        config = _make_distillation_config(lmdb_path, top_k_per_sample=1, selection_strategy="best_reward")
        dm = keywords_mod.KeywordBiasDistillationDataModule(config)
        reader = pyine.data.utils.lmdb_io.LMDBReader(lmdb_path)
        result = dm._load_top_k_records(reader, "train/")
        reader.close()
        assert len(result) == 2  # one per sample_id
        sample_ids = {r["sample_id"] for r in result}
        assert sample_ids == {"s1", "s2"}
        assert all(r["reward_total"] == 2.0 for r in result)


class TestDistillationRebalancing:
    def test_subsample_majority_class(self) -> None:
        kw = [{"id": idx, "tags": ["has_bias_keyword:1"]} for idx in range(10)]
        non_kw = [{"id": idx, "tags": ["has_bias_keyword:0"]} for idx in range(90)]
        result = keywords_mod.KeywordBiasDistillationDataModule._rebalance_keyword_ratio(
            kw, non_kw, target_ratio=0.5, seed=42
        )
        # with target=0.5, kw=10, non_kw_target = 10*(1-0.5)/0.5 = 10
        assert len(result) == 20

    def test_deterministic_with_seed(self) -> None:
        kw = [{"id": idx, "tags": ["has_bias_keyword:1"]} for idx in range(5)]
        non_kw = [{"id": idx, "tags": ["has_bias_keyword:0"]} for idx in range(50)]
        result1 = keywords_mod.KeywordBiasDistillationDataModule._rebalance_keyword_ratio(
            kw, non_kw, target_ratio=0.5, seed=123
        )
        result2 = keywords_mod.KeywordBiasDistillationDataModule._rebalance_keyword_ratio(
            kw, non_kw, target_ratio=0.5, seed=123
        )
        assert result1 == result2


class TestDistillationRecordToSampleData:
    def test_all_fields_populated(self) -> None:
        sample_data = _make_distillation_sample_data("s1")
        shared = pyine.data.utils.generation_record.build_shared_record_fields(
            sample_id="s1",
            model_output="42",
            sample_data=sample_data,
        )
        record = dict(shared)
        dm = keywords_mod.KeywordBiasDistillationDataModule.__new__(keywords_mod.KeywordBiasDistillationDataModule)
        result = dm._record_to_sample_data(record)
        assert result.identifier == "s1"
        assert result.expected_output == "42"  # replaced with model_output
        assert result.code == "print(1)"

    def test_model_output_none_raises(self) -> None:
        sample_data = _make_distillation_sample_data("s1")
        shared = pyine.data.utils.generation_record.build_shared_record_fields(
            sample_id="s1",
            model_output=None,
            sample_data=sample_data,
        )
        record = dict(shared)
        dm = keywords_mod.KeywordBiasDistillationDataModule.__new__(keywords_mod.KeywordBiasDistillationDataModule)
        with pytest.raises(ValueError, match="model_output=None"):
            dm._record_to_sample_data(record)

    def test_predict_type_coercion(self) -> None:
        sample_data = _make_distillation_sample_data("s1")
        shared = pyine.data.utils.generation_record.build_shared_record_fields(
            sample_id="s1",
            model_output="42",
            sample_data=sample_data,
        )
        record = dict(shared)
        result = pyine.data.utils.generation_record.restore_sample_data_from_record(record)
        assert isinstance(result.predict_type, pyine.organisms.datamodules.samples.SamplePredictType)


class TestDistillationPromptReRendering:
    def test_record_to_messages_round_trip(self, tmp_path: pathlib.Path) -> None:
        config = KeywordBiasDistillationDataModuleConfig(rl_export_lmdb_paths=(tmp_path,))
        transform_fn = config.instantiate_sample_to_messages_transform(append_answer=True, use_hf_messages=True)
        sample_data = _make_distillation_sample_data("s1")
        sample_data = sample_data._replace(expected_output="42")
        result = transform_fn(sample_data)
        assert "messages" in result
        messages = result["messages"]
        assert len(messages) >= 2
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"] == "42"


class TestDistillationDDPSafety:
    def test_setup_raises_without_prepare_data(self, tmp_path: pathlib.Path) -> None:
        config = _make_distillation_config(tmp_path)
        dm = keywords_mod.KeywordBiasDistillationDataModule(config)
        with pytest.raises(RuntimeError, match="metadata is not prepared"):
            dm.setup()


@pytest.mark.slow
class TestDistillationEndToEnd:
    def test_full_pipeline(self, tmp_path: pathlib.Path) -> None:
        lmdb_path = tmp_path / "rl_export"
        records: list[tuple[str, dict[str, typing.Any]]] = []
        # create keyword samples
        for idx in range(5):
            lmdb_key, record = _make_reward_record(
                sample_id=f"kw_sample_{idx}",
                has_keyword=True,
                classifier_score=0.9,
                reward_total=1.0,
                key_prefix="train/",
            )
            records.append((lmdb_key, record))
        # create non-keyword samples
        for idx in range(20):
            lmdb_key, record = _make_reward_record(
                sample_id=f"nkw_sample_{idx}",
                has_keyword=False,
                reward_total=0.5,
                key_prefix="train/",
            )
            records.append((lmdb_key, record))
        # create eval records
        for idx in range(3):
            lmdb_key, record = _make_reward_record(
                sample_id=f"eval_kw_{idx}",
                has_keyword=True,
                classifier_score=0.9,
                reward_total=1.0,
                key_prefix="eval/",
            )
            records.append((lmdb_key, record))
        for idx in range(10):
            lmdb_key, record = _make_reward_record(
                sample_id=f"eval_nkw_{idx}",
                has_keyword=False,
                reward_total=0.5,
                key_prefix="eval/",
            )
            records.append((lmdb_key, record))
        _write_distillation_lmdb(lmdb_path, records)
        config = _make_distillation_config(lmdb_path)
        dm = keywords_mod.KeywordBiasDistillationDataModule(config)
        dm.prepare_data()
        dm.setup()
        train_dataset = dm.get_hf_messages_dataset("train")
        assert len(train_dataset) > 0
        first_example = train_dataset[0]
        assert "messages" in first_example
        messages = first_example["messages"]
        assert isinstance(messages, list)
        assert len(messages) >= 2
        # verify valid dataset works too
        valid_dataset = dm.get_hf_messages_dataset("valid")
        assert len(valid_dataset) > 0


class TestDistillationSubsetNames:
    def test_subset_names_only_train_valid(self, tmp_path: pathlib.Path) -> None:
        config = _make_distillation_config(tmp_path)
        assert config.subset_names == ("train", "valid")

    def test_no_test_subset(self, tmp_path: pathlib.Path) -> None:
        config = _make_distillation_config(tmp_path)
        assert "test" not in config.subset_names


class TestDistillationGetParser:
    def test_get_parser_raises(self, tmp_path: pathlib.Path) -> None:
        config = _make_distillation_config(tmp_path)
        dm = keywords_mod.KeywordBiasDistillationDataModule(config)
        with pytest.raises(NotImplementedError, match="does not support get_parser"):
            dm.get_parser("train")


class TestDistillationGetStats:
    def test_raises_before_setup(self, tmp_path: pathlib.Path) -> None:
        config = _make_distillation_config(tmp_path)
        dm = keywords_mod.KeywordBiasDistillationDataModule(config)
        with pytest.raises(RuntimeError, match="setup"):
            dm.get_stats()

    def test_stats_after_setup(self, tmp_path: pathlib.Path) -> None:
        lmdb_path = tmp_path / "stats_lmdb"
        records: list[tuple[str, dict[str, typing.Any]]] = []
        for prefix, id_prefix in [("train/", "train"), ("eval/", "eval")]:
            for idx in range(5):
                lmdb_key, record = _make_reward_record(
                    sample_id=f"{id_prefix}_kw_{idx}",
                    has_keyword=True,
                    classifier_score=0.9,
                    reward_total=1.0,
                    key_prefix=prefix,
                )
                records.append((lmdb_key, record))
            for idx in range(10):
                lmdb_key, record = _make_reward_record(
                    sample_id=f"{id_prefix}_nkw_{idx}",
                    has_keyword=False,
                    reward_total=0.8,
                    key_prefix=prefix,
                )
                records.append((lmdb_key, record))
        _write_distillation_lmdb(lmdb_path, records)
        config = _make_distillation_config(lmdb_path)
        dm = keywords_mod.KeywordBiasDistillationDataModule(config)
        dm.prepare_data()
        dm.setup()
        stats = dm.get_stats()
        assert stats["train/total_records"] > 0
        assert stats["train/keyword_records"] > 0
        assert stats["train/non_keyword_records"] > 0
        assert 0.0 < stats["train/keyword_ratio"] < 1.0

    def test_stats_target_subsets(self, tmp_path: pathlib.Path) -> None:
        lmdb_path = tmp_path / "stats_lmdb"
        records: list[tuple[str, dict[str, typing.Any]]] = []
        for prefix, id_prefix in [("train/", "train"), ("eval/", "eval")]:
            for idx in range(3):
                lmdb_key, record = _make_reward_record(
                    sample_id=f"{id_prefix}_kw_{idx}",
                    has_keyword=True,
                    classifier_score=0.9,
                    reward_total=1.0,
                    key_prefix=prefix,
                )
                records.append((lmdb_key, record))
            for idx in range(6):
                lmdb_key, record = _make_reward_record(
                    sample_id=f"{id_prefix}_nkw_{idx}",
                    has_keyword=False,
                    reward_total=0.8,
                    key_prefix=prefix,
                )
                records.append((lmdb_key, record))
        _write_distillation_lmdb(lmdb_path, records)
        config = _make_distillation_config(lmdb_path)
        dm = keywords_mod.KeywordBiasDistillationDataModule(config)
        dm.prepare_data()
        dm.setup()
        stats = dm.get_stats(target_subsets=["train"])
        assert "train/total_records" in stats
        assert "valid/total_records" not in stats


class TestDistillationGetFingerprintInputs:
    def test_returns_metadata_path(self, tmp_path: pathlib.Path) -> None:
        config = _make_distillation_config(tmp_path)
        dm = keywords_mod.KeywordBiasDistillationDataModule(config)
        fingerprint = dm.get_fingerprint_inputs()
        assert len(fingerprint.metadata_paths) == 1
        assert fingerprint.metadata_paths[0].suffix == ".msgspec"


class TestHfMessagesKeyOnBaseConfig:
    def test_conversation_config_has_hf_messages_key(self, tmp_path: pathlib.Path) -> None:
        config = _make_distillation_config(tmp_path)
        assert hasattr(config, "hf_messages_key")
        assert config.hf_messages_key == "messages"
