import pathlib
import types
import typing

import datasets as hf_datasets
import numpy as np
import pytest
import torch
import transformers
from pytest_mock import MockerFixture

import pyine.data.traces.dataset_utils
import pyine.data.utils.splits
import pyine.organisms.datamodules.samples
import pyine.organisms.datamodules.samples.common
import pyine.organisms.datamodules.samples.configs
import pyine.organisms.datamodules.shortcuts as shortcuts_mod
import pyine.organisms.datamodules.shortcuts_configs
import pyine.utils.reprod
import pyine.utils.transformers
import tests.env_checks
from pyine.organisms.datamodules.shortcuts_configs import EvaluationStrategy, HintType


class _MockTraceId:
    """Mock TraceIdentifier for unit testing.

    Provides all properties needed for `SampleCodeTypeSet.create_from_trace()` validation.
    """

    def __init__(
        self,
        identifier: str,
        augment_category: str | None = None,
        is_hinted: bool = False,
        is_misleading: bool = False,
        is_obfuscated: bool = False,
        is_bugged: bool = False,
    ) -> None:
        self._identifier = identifier
        self._augment_category = augment_category
        self._is_hinted = is_hinted
        self._is_misleading = is_misleading
        self._is_obfuscated = is_obfuscated
        self._is_bugged = is_bugged

    @property
    def is_augmented(self) -> bool:
        return self._augment_category is not None

    @property
    def is_hinted(self) -> bool:
        return self._is_hinted

    @property
    def is_misleading(self) -> bool:
        return self._is_misleading

    @property
    def is_obfuscated(self) -> bool:
        return self._is_obfuscated

    @property
    def is_bugged(self) -> bool:
        return self._is_bugged

    @property
    def augment_category(self) -> str | None:
        return self._augment_category

    @property
    def split_augment_categories(self) -> list[str]:
        return self._augment_category.split("+") if self._augment_category else []

    def get_augmentless_identifier(self) -> "_MockTraceId":
        return _MockTraceId(self._identifier.split("/a:")[0])

    def get_parent_identifier(self) -> str:
        # returns solution identifier (without test index and augmentation)
        parts = self._identifier.split("/")
        if len(parts) >= 4:
            return "/".join(parts[:4])  # dataset/subset/problem/solution
        return self._identifier

    def __str__(self) -> str:
        return self._identifier


class _MockTraceMeta:
    """Mock TraceMetadata for unit testing."""

    def __init__(
        self,
        identifier: str,
        trace_id: _MockTraceId,
    ) -> None:
        self.identifier = identifier
        self.trace_id = trace_id


def _make_stub_shortcuts_datamodule(
    evaluation_strategy: EvaluationStrategy = EvaluationStrategy.hint_presence_split,
    eval_hint_types: tuple[HintType, ...] = (HintType.helpful,),
    eval_subset_names: tuple[str, ...] = ("valid",),
) -> shortcuts_mod.ShortcutBiasDataModule:
    """Create a stub shortcuts datamodule for unit testing."""
    stub = shortcuts_mod.ShortcutBiasDataModule.__new__(shortcuts_mod.ShortcutBiasDataModule)
    stub.config = types.SimpleNamespace(
        evaluation_strategy=evaluation_strategy,
        eval_hint_types=eval_hint_types,
        eval_subset_names=eval_subset_names,
        min_samples_hinted=0,
        min_samples_misleading=0,
        min_samples_hintless=0,
        split_seed=0,  # needed for counterfactual RNG initialization
    )
    stub.verbose = False
    return stub


class TestIsHintCategory:
    """Tests for _is_hint_category helper method."""

    def test_detects_hints_prefix(self) -> None:
        dm = _make_stub_shortcuts_datamodule()
        assert dm._is_hint_category("hints_docs") is True
        assert dm._is_hint_category("hints_tests") is True
        assert dm._is_hint_category("HINTS_DOCS") is True

    def test_detects_hinted_keyword(self) -> None:
        dm = _make_stub_shortcuts_datamodule()
        assert dm._is_hint_category("obfuscated_hinted") is True
        assert dm._is_hint_category("code_hinted") is True

    def test_detects_misleading_keyword(self) -> None:
        dm = _make_stub_shortcuts_datamodule()
        assert dm._is_hint_category("misleading") is True
        assert dm._is_hint_category("obfuscated_misleading") is True

    def test_detects_issues_docs_pattern(self) -> None:
        dm = _make_stub_shortcuts_datamodule()
        assert dm._is_hint_category("issues_docs") is True

    def test_returns_false_for_non_hint_categories(self) -> None:
        dm = _make_stub_shortcuts_datamodule()
        assert dm._is_hint_category("obfuscated") is False
        assert dm._is_hint_category("code_stubbing") is False
        assert dm._is_hint_category("issues_iterators") is False
        assert dm._is_hint_category("issues_todos") is False


class TestGetBaseAugments:
    """Tests for _get_base_augments helper method."""

    def test_non_augmented_trace_returns_empty(self) -> None:
        dm = _make_stub_shortcuts_datamodule()
        trace_id = _MockTraceId("test/p0001/s0001/t0001", augment_category=None)
        assert dm._get_base_augments(trace_id) == frozenset()

    def test_hint_only_augment_returns_empty(self) -> None:
        dm = _make_stub_shortcuts_datamodule()
        trace_id = _MockTraceId("test/p0001/s0001/t0001/a:hints_docs:000", augment_category="hints_docs")
        assert dm._get_base_augments(trace_id) == frozenset()

    def test_non_hint_augment_returned(self) -> None:
        dm = _make_stub_shortcuts_datamodule()
        trace_id = _MockTraceId("test/p0001/s0001/t0001/a:obfuscated:000", augment_category="obfuscated")
        assert dm._get_base_augments(trace_id) == frozenset({"obfuscated"})

    def test_mixed_augments_filters_hints(self) -> None:
        dm = _make_stub_shortcuts_datamodule()
        trace_id = _MockTraceId(
            "test/p0001/s0001/t0001/a:obfuscated+hints_docs:000",
            augment_category="obfuscated+hints_docs",
        )
        base_augments = dm._get_base_augments(trace_id)
        assert "obfuscated" in base_augments
        assert "hints_docs" not in base_augments
        assert len(base_augments) == 1


class TestCheckPromptDatabaseForHint:
    """Tests for _check_prompt_db_for_hint helper method."""

    def test_returns_true_when_helpful_hint_in_db(self, mocker: MockerFixture) -> None:
        dm = _make_stub_shortcuts_datamodule(eval_hint_types=(HintType.helpful,))
        trace_id = _MockTraceId("TACO/train/p000001/s0001/t0001")
        mock_prompt_db = mocker.MagicMock()
        # return a record with hinted code type tag
        mock_record = mocker.MagicMock()
        mock_record.tags = ["augment:hinted"]
        mock_prompt_db.get_by_identifier.return_value = [mock_record]
        assert dm._check_prompt_db_for_hint(trace_id, mock_prompt_db) is True

    def test_returns_false_when_no_hint_in_db(self, mocker: MockerFixture) -> None:
        dm = _make_stub_shortcuts_datamodule(eval_hint_types=(HintType.helpful,))
        trace_id = _MockTraceId("TACO/train/p000001/s0001/t0001")
        mock_prompt_db = mocker.MagicMock()
        mock_prompt_db.get_by_identifier.return_value = []
        assert dm._check_prompt_db_for_hint(trace_id, mock_prompt_db) is False

    def test_returns_false_when_record_has_wrong_tags(self, mocker: MockerFixture) -> None:
        dm = _make_stub_shortcuts_datamodule(eval_hint_types=(HintType.helpful,))
        trace_id = _MockTraceId("TACO/train/p000001/s0001/t0001")
        mock_prompt_db = mocker.MagicMock()
        # return a record but with wrong/no code type tag
        mock_record = mocker.MagicMock()
        mock_record.tags = ["some_other_tag"]
        mock_prompt_db.get_by_identifier.return_value = [mock_record]
        assert dm._check_prompt_db_for_hint(trace_id, mock_prompt_db) is False

    def test_checks_misleading_prompts_for_misleading_hint_type(self, mocker: MockerFixture) -> None:
        dm = _make_stub_shortcuts_datamodule(eval_hint_types=(HintType.misleading,))
        trace_id = _MockTraceId("TACO/train/p000001/s0001/t0001")
        mock_prompt_db = mocker.MagicMock()
        # return a record with misleading code type tag
        mock_record = mocker.MagicMock()
        mock_record.tags = ["augment:misleading"]
        mock_prompt_db.get_by_identifier.return_value = [mock_record]
        assert dm._check_prompt_db_for_hint(trace_id, mock_prompt_db) is True
        # verify it checked for issues_docs prompt (misleading hint)
        # get_by_identifier is called once per prompt name, check all calls
        all_prompt_names = [call[1].get("prompt_name") for call in mock_prompt_db.get_by_identifier.call_args_list]
        assert any("issues" in str(pn).lower() for pn in all_prompt_names if pn)

    def test_augmented_trace_looks_for_augmented_hinted_code(self, mocker: MockerFixture) -> None:
        """Test that an obfuscated trace looks for obfuscated_hinted, not just hinted."""
        dm = _make_stub_shortcuts_datamodule(eval_hint_types=(HintType.helpful,))
        # trace with obfuscated augment
        trace_id = _MockTraceId("TACO/train/p000001/s0001/t0001", augment_category="obfuscated")
        mock_prompt_db = mocker.MagicMock()
        # record with just "hinted" should NOT match (we need obfuscated_hinted)
        mock_record_hinted_only = mocker.MagicMock()
        mock_record_hinted_only.tags = ["augment:hinted"]
        mock_prompt_db.get_by_identifier.return_value = [mock_record_hinted_only]
        assert dm._check_prompt_db_for_hint(trace_id, mock_prompt_db) is False
        # record with "obfuscated_hinted" SHOULD match
        mock_record_obfuscated_hinted = mocker.MagicMock()
        mock_record_obfuscated_hinted.tags = ["augment:obfuscated_hinted"]
        mock_prompt_db.get_by_identifier.return_value = [mock_record_obfuscated_hinted]
        assert dm._check_prompt_db_for_hint(trace_id, mock_prompt_db) is True


class TestShouldUsePromptDatabaseForHints:
    """Tests for _should_use_prompt_db_for_hints helper method."""

    def test_returns_true_when_allow_db_lookups_enabled_in_dict(self) -> None:
        dm = _make_stub_shortcuts_datamodule()
        dm.config.default_dataparser_config = types.SimpleNamespace(
            params={"selection_config": {"allow_db_lookups": True}}
        )
        assert dm._should_use_prompt_db_for_hints() is True

    def test_returns_false_when_allow_db_lookups_disabled_in_dict(self) -> None:
        dm = _make_stub_shortcuts_datamodule()
        dm.config.default_dataparser_config = types.SimpleNamespace(
            params={"selection_config": {"allow_db_lookups": False}}
        )
        assert dm._should_use_prompt_db_for_hints() is False

    def test_raises_when_no_params(self) -> None:
        dm = _make_stub_shortcuts_datamodule()
        dm.config.default_dataparser_config = types.SimpleNamespace()  # no params
        with pytest.raises(AssertionError):
            dm._should_use_prompt_db_for_hints()

    def test_returns_false_when_no_selection_config(self) -> None:
        dm = _make_stub_shortcuts_datamodule()
        dm.config.default_dataparser_config = types.SimpleNamespace(params={})
        assert dm._should_use_prompt_db_for_hints() is False

    def test_returns_true_with_mapping_selection_config(self) -> None:
        dm = _make_stub_shortcuts_datamodule()
        dm.config.default_dataparser_config = types.SimpleNamespace(
            params={"selection_config": {"allow_db_lookups": True}}
        )
        assert dm._should_use_prompt_db_for_hints() is True


class TestPartitionTracesByHintStrategy:
    """Tests for _partition_traces_by_hint_strategy method."""

    def test_hint_presence_split_partitions_all_traces(self) -> None:
        dm = _make_stub_shortcuts_datamodule(evaluation_strategy=EvaluationStrategy.hint_presence_split)
        traces_with = [
            _MockTraceMeta("t1", _MockTraceId("t1", augment_category="hints_docs", is_hinted=True)),
            _MockTraceMeta("t2", _MockTraceId("t2", augment_category="hints_tests", is_hinted=True)),
        ]
        traces_without = [
            _MockTraceMeta("t3", _MockTraceId("t3", is_hinted=False)),
            _MockTraceMeta("t4", _MockTraceId("t4", augment_category="obfuscated", is_hinted=False)),
        ]
        all_traces = traces_with + traces_without
        result = dm._partition_traces_by_hint_strategy(all_traces)
        assert len(result["hinted"]) == 2
        assert len(result["hintless"]) == 2
        assert {t.identifier for t in result["hinted"]} == {"t1", "t2"}
        assert {t.identifier for t in result["hintless"]} == {"t3", "t4"}

    def test_counterfactual_only_includes_paired_traces(self) -> None:
        dm = _make_stub_shortcuts_datamodule(evaluation_strategy=EvaluationStrategy.counterfactual)
        base_trace_id = _MockTraceId("test/p0001/s0001/t0001")
        # paired traces (same family, empty base augments)
        trace_with_hint = _MockTraceMeta(
            "t1",
            _MockTraceId("t1", augment_category="hints_docs", is_hinted=True),
        )
        trace_with_hint.trace_id.get_augmentless_identifier = lambda: base_trace_id
        trace_without_hint = _MockTraceMeta(
            "t2",
            _MockTraceId("t2", is_hinted=False),
        )
        trace_without_hint.trace_id.get_augmentless_identifier = lambda: base_trace_id
        # unpaired trace (different family)
        other_base = _MockTraceId("test/p0002/s0001/t0001")
        trace_unpaired = _MockTraceMeta(
            "t3",
            _MockTraceId("t3", is_hinted=False),
        )
        trace_unpaired.trace_id.get_augmentless_identifier = lambda: other_base
        all_traces = [trace_with_hint, trace_without_hint, trace_unpaired]
        result = dm._partition_traces_by_hint_strategy(all_traces)
        # only paired traces should be included
        assert len(result["hinted"]) == 1
        assert len(result["hintless"]) == 1
        assert result["hinted"][0].identifier == "t1"
        assert result["hintless"][0].identifier == "t2"

    def test_counterfactual_equal_counts_with_multiple_same_family_traces(self) -> None:
        """Test counterfactual mode produces equal counts when multiple traces share family_id.

        The new group-first counterfactual implementation builds groups from traces directly
        (ignoring family_map) and picks traces deterministically by sorting.
        """
        dm = _make_stub_shortcuts_datamodule(evaluation_strategy=EvaluationStrategy.counterfactual)
        base_trace_id = _MockTraceId("test/p0001/s0001/t0001")
        # one hinted trace
        trace_hinted = _MockTraceMeta(
            "t1",
            _MockTraceId("t1", augment_category="hints_docs", is_hinted=True),
        )
        trace_hinted.trace_id.get_augmentless_identifier = lambda: base_trace_id
        # multiple hintless traces (same family)
        trace_hintless1 = _MockTraceMeta(
            "t2",
            _MockTraceId("t2", is_hinted=False),
        )
        trace_hintless1.trace_id.get_augmentless_identifier = lambda: base_trace_id
        trace_hintless2 = _MockTraceMeta(
            "t3",
            _MockTraceId("t3", is_hinted=False),
        )
        trace_hintless2.trace_id.get_augmentless_identifier = lambda: base_trace_id
        all_traces = [trace_hinted, trace_hintless1, trace_hintless2]
        result = dm._partition_traces_by_hint_strategy(all_traces)
        # counterfactual should produce exactly equal counts (1:1 pairing)
        assert len(result["hinted"]) == len(result["hintless"]) == 1
        assert result["hinted"][0].identifier == "t1"
        # seeded rng selects from sorted hintless traces (t2, t3)
        assert result["hintless"][0].identifier in {"t2", "t3"}


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


class TestShortcutBiasDataModuleConfigValidateAndResolve:
    """Tests for _validate_and_resolve in ShortcutBiasDataModuleConfig.

    These tests instantiate real config objects to exercise the actual pydantic validators.
    """

    def _make_minimal_config(
        self,
        lmdb_path: pathlib.Path,
        split_path: pathlib.Path,
        eval_subset_names: tuple[str, ...] = ("valid",),
    ) -> pyine.organisms.datamodules.shortcuts_configs.ShortcutBiasDataModuleConfig:
        """Create a minimal ShortcutBiasDataModuleConfig for testing validation logic."""
        return pyine.organisms.datamodules.shortcuts_configs.ShortcutBiasDataModuleConfig(
            lmdb_paths=[str(lmdb_path)],
            split_file_path=str(split_path),
            eval_subset_names=eval_subset_names,
            instantiate_parsers_at_setup=False,  # avoid parser instantiation
        )

    def test_extends_subset_names_with_hint_splits(
        self, fake_lmdb_and_split: tuple[pathlib.Path, pathlib.Path]
    ) -> None:
        lmdb_path, split_path = fake_lmdb_and_split
        config = self._make_minimal_config(lmdb_path, split_path, eval_subset_names=("valid",))
        assert "valid_hinted" in config.subset_names
        assert "valid_hintless" in config.subset_names
        # base subsets should still be present
        assert "train" in config.subset_names
        assert "valid" in config.subset_names

    def test_extends_multiple_eval_subsets(self, fake_lmdb_and_split: tuple[pathlib.Path, pathlib.Path]) -> None:
        lmdb_path, split_path = fake_lmdb_and_split
        config = self._make_minimal_config(lmdb_path, split_path, eval_subset_names=("valid", "test"))
        assert "valid_hinted" in config.subset_names
        assert "valid_hintless" in config.subset_names
        assert "test_hinted" in config.subset_names
        assert "test_hintless" in config.subset_names

    def test_misleading_subset_selects_misleading_code_type_when_configured(
        self, fake_lmdb_and_split: tuple[pathlib.Path, pathlib.Path]
    ) -> None:
        lmdb_path, split_path = fake_lmdb_and_split
        config = pyine.organisms.datamodules.shortcuts_configs.ShortcutBiasDataModuleConfig(
            lmdb_paths=[str(lmdb_path)],
            split_file_path=str(split_path),
            eval_subset_names=("valid",),
            eval_hint_types=(HintType.misleading,),
            instantiate_parsers_at_setup=False,
        )
        # when eval_hint_types=(HintType.misleading,), we get valid_misleading subset
        parser_config = config._resolve_dataparser_config("valid_misleading")
        params = parser_config.get_params_dict()
        selection_config = params["selection_config"]
        # check that require_hint_type is set to misleading
        if isinstance(selection_config, dict):
            require_hint_type = selection_config.get("require_hint_type")
        else:
            require_hint_type = getattr(selection_config, "require_hint_type", None)
        assert require_hint_type == HintType.misleading


class TestShortcutBiasDataModuleConfigParentSubsetResolution:
    """Tests for _get_parent_subset_name using real config objects."""

    def _make_minimal_config(
        self,
        lmdb_path: pathlib.Path,
        split_path: pathlib.Path,
        eval_subset_names: tuple[str, ...] = ("valid", "test"),
    ) -> pyine.organisms.datamodules.shortcuts_configs.ShortcutBiasDataModuleConfig:
        return pyine.organisms.datamodules.shortcuts_configs.ShortcutBiasDataModuleConfig(
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
        assert config._get_parent_subset_name("valid_hinted") == "valid"
        assert config._get_parent_subset_name("valid_hintless") == "valid"
        assert config._get_parent_subset_name("test_hinted") == "test"
        assert config._get_parent_subset_name("train") == "train"  # not a derived subset
        assert config._get_parent_subset_name("valid") == "valid"  # base eval subset


class TestValidateSampleCounts:
    """Tests for _validate_sample_counts method."""

    def test_raises_when_too_few_hinted(self, mocker: MockerFixture) -> None:
        dm = shortcuts_mod.ShortcutBiasDataModule.__new__(shortcuts_mod.ShortcutBiasDataModule)
        dm.config = types.SimpleNamespace(
            eval_subset_names=("valid",),
            min_samples_hinted=10,
            min_samples_misleading=0,
            min_samples_hintless=0,
        )
        dm.verbose = False
        subset_traces: dict[str, list[typing.Any]] = {"valid": []}
        derived_subsets = {
            "valid_hinted": pyine.data.traces.dataset_utils.DerivedSubsetInfo(
                parent_subset="valid",
                traces=[],  # 0 traces, below minimum
                derivation_type="hint_presence_split",
            ),
            "valid_hintless": pyine.data.traces.dataset_utils.DerivedSubsetInfo(
                parent_subset="valid",
                traces=[mocker.MagicMock() for _ in range(100)],
                derivation_type="hint_presence_split",
            ),
        }
        with pytest.raises(ValueError, match="has only 0 samples"):
            dm._validate_sample_counts(subset_traces, derived_subsets)

    def test_raises_when_too_few_hintless(self, mocker: MockerFixture) -> None:
        dm = shortcuts_mod.ShortcutBiasDataModule.__new__(shortcuts_mod.ShortcutBiasDataModule)
        dm.config = types.SimpleNamespace(
            eval_subset_names=("valid",),
            min_samples_hinted=0,
            min_samples_misleading=0,
            min_samples_hintless=50,
        )
        dm.verbose = False
        subset_traces: dict[str, list[typing.Any]] = {"valid": []}
        derived_subsets = {
            "valid_hinted": pyine.data.traces.dataset_utils.DerivedSubsetInfo(
                parent_subset="valid",
                traces=[mocker.MagicMock() for _ in range(20)],
                derivation_type="hint_presence_split",
            ),
            "valid_hintless": pyine.data.traces.dataset_utils.DerivedSubsetInfo(
                parent_subset="valid",
                traces=[mocker.MagicMock() for _ in range(10)],  # below minimum
                derivation_type="hint_presence_split",
            ),
        }
        with pytest.raises(ValueError, match="has only 10 samples"):
            dm._validate_sample_counts(subset_traces, derived_subsets)

    def test_passes_when_counts_meet_minimum(self, mocker: MockerFixture) -> None:
        dm = shortcuts_mod.ShortcutBiasDataModule.__new__(shortcuts_mod.ShortcutBiasDataModule)
        dm.config = types.SimpleNamespace(
            eval_subset_names=("valid",),
            min_samples_hinted=5,
            min_samples_misleading=0,
            min_samples_hintless=10,
        )
        dm.verbose = False
        subset_traces: dict[str, list[typing.Any]] = {"valid": []}
        derived_subsets = {
            "valid_hinted": pyine.data.traces.dataset_utils.DerivedSubsetInfo(
                parent_subset="valid",
                traces=[mocker.MagicMock() for _ in range(10)],
                derivation_type="hint_presence_split",
            ),
            "valid_hintless": pyine.data.traces.dataset_utils.DerivedSubsetInfo(
                parent_subset="valid",
                traces=[mocker.MagicMock() for _ in range(20)],
                derivation_type="hint_presence_split",
            ),
        }
        dm._validate_sample_counts(subset_traces, derived_subsets)  # should not raise


class TestCreateHintSplitDerivedSubsetsPresenceSplit:
    """Tests for _create_hint_split_derived_subsets in hint_presence_split mode."""

    def test_creates_derived_subsets_for_each_eval_subset(self, mocker: MockerFixture) -> None:
        stub = shortcuts_mod.ShortcutBiasDataModule.__new__(shortcuts_mod.ShortcutBiasDataModule)
        stub.config = types.SimpleNamespace(
            evaluation_strategy=EvaluationStrategy.hint_presence_split,
            eval_hint_types=(HintType.helpful,),
            eval_subset_names=("valid", "test"),
        )
        stub.verbose = False
        # create mock traces with some having hints
        valid_traces = [
            mocker.MagicMock(
                identifier=f"v{i}",
                trace_id=mocker.MagicMock(is_hinted=(i < 3), is_misleading=False),
            )
            for i in range(10)
        ]
        test_traces = [
            mocker.MagicMock(
                identifier=f"t{i}",
                trace_id=mocker.MagicMock(is_hinted=(i < 2), is_misleading=False),
            )
            for i in range(8)
        ]
        subset_traces = {"valid": valid_traces, "test": test_traces}
        derived, _counts = stub._create_hint_split_derived_subsets(subset_traces)
        assert "valid_hinted" in derived
        assert "valid_hintless" in derived
        assert "test_hinted" in derived
        assert "test_hintless" in derived
        assert derived["valid_hinted"].parent_subset == "valid"
        assert derived["test_hintless"].parent_subset == "test"
        assert len(derived["valid_hinted"].traces) == 3
        assert len(derived["valid_hintless"].traces) == 7
        assert len(derived["test_hinted"].traces) == 2
        assert len(derived["test_hintless"].traces) == 6


class TestCreateHintSplitDerivedSubsetsCounterfactual:
    """Tests for _create_hint_split_derived_subsets in counterfactual mode."""

    def test_counterfactual_only_includes_paired_traces(self, mocker: MockerFixture) -> None:
        """Test that counterfactual mode only includes complete groups.

        The new group-first implementation:
        1. Groups traces by (family_id, base_augment_key)
        2. Filters to complete groups (hintless + hinted available)
        3. Applies distribution control (defaults to {"original": 1.0})
        4. Picks traces deterministically from selected groups
        """
        stub = shortcuts_mod.ShortcutBiasDataModule.__new__(shortcuts_mod.ShortcutBiasDataModule)
        stub.config = types.SimpleNamespace(
            evaluation_strategy=EvaluationStrategy.counterfactual,
            eval_hint_types=(HintType.helpful,),
            eval_subset_names=("valid",),
            split_seed=0,  # needed for counterfactual RNG initialization
        )
        stub.verbose = False
        # create mock trace_ids with proper augmentless identifier
        base_family_id = "ds/train/p000001/s0001/t0001"

        def make_mock_trace(
            identifier: str,
            is_hinted: bool,
            augment_category: str | None = None,
        ) -> typing.Any:
            trace = mocker.MagicMock()
            trace.identifier = identifier
            trace.trace_id.is_hinted = is_hinted
            trace.trace_id.is_misleading = False
            trace.trace_id.augment_category = augment_category
            trace.trace_id.get_augmentless_identifier.return_value = base_family_id
            return trace

        # paired traces (same family, original base augment)
        # "hints_docs" -> hinted code type with base_augment_key = "original"
        trace_with_hint = make_mock_trace("t1", is_hinted=True, augment_category="hints_docs")
        # None -> original code type with base_augment_key = "original"
        trace_without_hint = make_mock_trace("t2", is_hinted=False, augment_category=None)

        # unpaired trace (different family, only has hinted - no hintless pair)
        trace_unpaired = make_mock_trace("t3", is_hinted=True, augment_category="hints_docs")
        trace_unpaired.trace_id.get_augmentless_identifier.return_value = "ds/train/p000002/s0001/t0001"

        all_traces = [trace_with_hint, trace_without_hint, trace_unpaired]
        subset_traces = {"valid": all_traces}
        derived, _counts = stub._create_hint_split_derived_subsets(subset_traces)
        # only complete groups should be included (family_1 has both hinted and hintless)
        assert len(derived["valid_hinted"].traces) == 1
        assert len(derived["valid_hintless"].traces) == 1
        assert derived["valid_hinted"].traces[0].identifier == "t1"
        assert derived["valid_hintless"].traces[0].identifier == "t2"

    def test_counterfactual_excludes_all_when_no_pairs(self, mocker: MockerFixture) -> None:
        stub = shortcuts_mod.ShortcutBiasDataModule.__new__(shortcuts_mod.ShortcutBiasDataModule)
        stub.config = types.SimpleNamespace(
            evaluation_strategy=EvaluationStrategy.counterfactual,
            eval_hint_types=(HintType.helpful,),
            eval_subset_names=("valid",),
            split_seed=0,  # needed for counterfactual RNG initialization
        )
        stub.verbose = False
        # create traces with no complete pairs - all hinted, no hintless traces
        traces = []
        for idx in range(3):
            trace = mocker.MagicMock()
            trace.identifier = f"t{idx}"
            trace.trace_id.is_hinted = True
            trace.trace_id.is_misleading = False
            trace.trace_id.is_augmented = True
            trace.trace_id.augment_category = "hints_docs"  # required for get_code_type_set_from_str
            trace.trace_id.get_augmentless_identifier.return_value = f"family_{idx}"
            traces.append(trace)
        subset_traces = {"valid": traces}
        derived, _counts = stub._create_hint_split_derived_subsets(subset_traces)
        # no complete groups (all hinted, no hintless traces in any group)
        assert len(derived["valid_hinted"].traces) == 0
        assert len(derived["valid_hintless"].traces) == 0

    def test_counterfactual_derivation_type_is_set(self) -> None:
        stub = shortcuts_mod.ShortcutBiasDataModule.__new__(shortcuts_mod.ShortcutBiasDataModule)
        stub.config = types.SimpleNamespace(
            evaluation_strategy=EvaluationStrategy.counterfactual,
            eval_hint_types=(HintType.helpful,),
            eval_subset_names=("valid",),
            split_seed=0,  # needed for counterfactual RNG initialization
        )
        stub.verbose = False
        subset_traces = {"valid": []}
        derived, _counts = stub._create_hint_split_derived_subsets(subset_traces)
        assert derived["valid_hinted"].derivation_type == "counterfactual"
        assert derived["valid_hintless"].derivation_type == "counterfactual"


def _normalize_whitespace(text: str) -> str:
    """Collapse all whitespace (spaces, tabs, newlines) to single spaces."""
    return " ".join(text.split())


def _normalized_contains(haystack: str, needle: str) -> bool:
    """Check if needle is in haystack after normalizing whitespace."""
    return _normalize_whitespace(needle) in _normalize_whitespace(haystack)


def _normalized_equals(text_a: str, text_b: str) -> bool:
    """Check if two strings are equal after normalizing whitespace."""
    return _normalize_whitespace(text_a) == _normalize_whitespace(text_b)


def _assert_non_leaking_assignments(
    metadata: pyine.data.traces.dataset_utils.TraceDatasetMetadata,
) -> None:
    traces_to_subsets: dict[str, str] = {}
    problems_to_subsets: dict[str, str] = {}
    for subset_name, subset_traces in metadata.subset_traces.items():
        for trace in subset_traces:
            assert trace.identifier not in traces_to_subsets
            if trace.problem_id in problems_to_subsets:
                assert problems_to_subsets[trace.problem_id] == subset_name
            else:
                problems_to_subsets[trace.problem_id] = subset_name
            traces_to_subsets[trace.identifier] = subset_name


@pytest.fixture
def shortcuts_dm_config() -> pyine.organisms.datamodules.shortcuts_configs.ShortcutBiasDataModuleConfig:
    return pyine.organisms.datamodules.shortcuts_configs.ShortcutBiasDataModuleConfig(
        lmdb_paths=[
            pyine.data.traces.dataset_utils.get_latest_dataset_path("TACO"),
        ],
        dataparser_config_overrides={
            "train": {
                "transform_config": pyine.organisms.datamodules.samples.SampleTransformConfig(
                    transform_strategy=pyine.organisms.datamodules.samples.SampleTransformStrategy.hybrid,
                    predict_type_prob_map={
                        pyine.organisms.datamodules.samples.SamplePredictType.program_output: 0.5,
                        pyine.organisms.datamodules.samples.SamplePredictType.frame_variables: 0.1,
                        pyine.organisms.datamodules.samples.SamplePredictType.function_return: 0.4,
                    },
                ),
            },  # other subsets will default to never producing partial samples
        },
        dataloader_config_overrides={
            "train": {
                "batch_size": 16,
                "shuffle": True,
            },
        },
        max_solution_count=100,
        split_file_path=pyine.data.utils.splits.get_dataset_split_file_path("TACO"),
        # disable sample count validation for tests (TACO may not have hinted traces)
        min_samples_hinted=0,
        min_samples_hintless=0,
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.dataset
@pytest.mark.skipif(
    tests.env_checks.TACO_TRACES_DATASET_MISSING,
    reason="TACO traces dataset is missing, cannot check sample generation",
)
@pytest.mark.skipif(
    tests.env_checks.TACO_TRACES_DATASET_SPLIT_MISSING,
    reason="TACO traces dataset split is missing, cannot check sample generation",
)
@pytest.mark.skipif(
    tests.env_checks.HF_ACCESS_TOKEN_MISSING,
    reason="Hugging Face access token is missing, cannot check hf dataset loading",
)
def test_shortcuts_datamodule_integration(
    shortcuts_dm_config: pyine.organisms.datamodules.shortcuts_configs.ShortcutBiasDataModuleConfig,
) -> None:
    pyine.utils.reprod.load_dotenv()
    dm = shortcuts_dm_config.instantiate_datamodule(verbose=True)
    if dm._is_metadata_prepared():
        dm._clear_prepared_metadata()
    dm.prepare_data()
    dm.setup()
    assert dm._is_metadata_prepared()
    metadata = dm._load_prepared_metadata()
    assert isinstance(metadata, pyine.data.traces.dataset_utils.TraceDatasetMetadata)
    _assert_non_leaking_assignments(metadata)
    dataloader = dm.train_dataloader()
    assert dataloader.batch_size == 16
    batch = next(iter(dataloader))
    assert len(batch.code) == 16
    print("batch loading works")
    parser = dm.get_parser("train")
    sample = parser[0]
    sample_msgs_transf_fn = pyine.organisms.datamodules.utils.transforms.create_sample_transform(
        use_chat_template=True,
        append_answer=True,
        prompt_name="code_execution",
    )
    transformed_sample_msgs = sample_msgs_transf_fn(sample)
    assert isinstance(transformed_sample_msgs, list)
    assert all(hasattr(m, "type") and hasattr(m, "content") for m in transformed_sample_msgs)
    print("sample transform works")
    hf_msgs_ds = dm.get_hf_messages_dataset(subset_name="train")
    assert isinstance(hf_msgs_ds, hf_datasets.Dataset)
    assert len(hf_msgs_ds) == len(parser)  # noqa
    hf_msgs_sample = hf_msgs_ds[0]
    assert isinstance(hf_msgs_sample, dict)
    assert "messages" in hf_msgs_sample and isinstance(hf_msgs_sample["messages"], list)
    print("hf dataset works")
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, use_fast=True)
    hf_batch_dataset = pyine.utils.transformers.apply_model_template_to_messages(
        hf_messages_ds=hf_msgs_ds,
        tokenizer=tokenizer,
        apply_chat_template_kwargs={
            "tokenize": False,
            "add_generation_prompt": False,
        },
    )
    assert isinstance(hf_batch_dataset, hf_datasets.Dataset)
    assert len(hf_batch_dataset) == len(parser)  # noqa
    print("hf dataset batching and tokenization works")
    openai_dataset_path = dm.get_openai_messages_dataset("train")
    assert openai_dataset_path.exists() and openai_dataset_path.is_file()
    print("open dataset writing works")
    dm.teardown()


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.dataset
@pytest.mark.skipif(
    tests.env_checks.TACO_TRACES_DATASET_MISSING,
    reason="TACO traces dataset is missing, cannot check sample generation",
)
@pytest.mark.skipif(
    tests.env_checks.TACO_TRACES_DATASET_SPLIT_MISSING,
    reason="TACO traces dataset split is missing, cannot check sample generation",
)
@pytest.mark.skipif(
    tests.env_checks.HF_ACCESS_TOKEN_MISSING,
    reason="Hugging Face access token is missing, cannot check hf dataset loading",
)
def test_shortcuts_datamodule_predefined_split(
    shortcuts_dm_config: pyine.organisms.datamodules.shortcuts_configs.ShortcutBiasDataModuleConfig,
) -> None:
    pyine.utils.reprod.load_dotenv()
    dm = shortcuts_dm_config.instantiate_datamodule(verbose=True)
    if dm._is_metadata_prepared():
        dm._clear_prepared_metadata()
    dm.prepare_data()
    dm.setup()
    assert dm._is_metadata_prepared()
    metadata = dm._load_prepared_metadata()
    assert isinstance(metadata, pyine.data.traces.dataset_utils.TraceDatasetMetadata)
    _assert_non_leaking_assignments(metadata)
    train_parser = dm.get_parser("train")
    train_sample, valid_sample = None, None
    if len(train_parser) > 0:  # noqa
        train_sample = train_parser[0]
    valid_parser = dm.get_parser("valid")
    if len(valid_parser) > 0:  # noqa
        valid_sample = valid_parser[0]
    if train_sample is not None or valid_sample is not None:
        assert train_sample != valid_sample
    dm.teardown()


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.dataset
@pytest.mark.skipif(
    tests.env_checks.TACO_TRACES_DATASET_MISSING,
    reason="TACO traces dataset is missing, cannot check sample generation",
)
@pytest.mark.skipif(
    tests.env_checks.TACO_TRACES_DATASET_SPLIT_MISSING,
    reason="TACO traces dataset split is missing, cannot check sample generation",
)
@pytest.mark.skipif(
    tests.env_checks.HF_ACCESS_TOKEN_MISSING,
    reason="Hugging Face access token is missing, cannot check hf dataset loading",
)
def test_shortcuts_datamodule_examples_round_trip(
    shortcuts_dm_config: pyine.organisms.datamodules.shortcuts_configs.ShortcutBiasDataModuleConfig,
) -> None:
    pyine.utils.reprod.load_dotenv()
    dm = shortcuts_dm_config.instantiate_datamodule(verbose=True)
    if dm._is_metadata_prepared():
        dm._clear_prepared_metadata()
    dm.prepare_data()
    dm.setup()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_max_seq_len = 2048  # pretend the model is actually limited to this
    hf_msgs_ds = dm.get_hf_messages_dataset(
        subset_name="train",
        keep_original_data=True,
    )
    max_samples = min(len(hf_msgs_ds), 10)
    if max_samples == 0:
        pytest.skip("dataset contains no samples to verify")
    hf_msgs_ds = hf_msgs_ds.select(list(range(max_samples)))
    cached_samples: dict[str, dict[str, typing.Any]] = {}
    expected_sample_keys = ["messages", "identifier", "code", "inputs", "expected_output"]
    for sample in hf_msgs_ds:
        assert isinstance(sample, dict)
        assert all(k in sample for k in expected_sample_keys)
        identifier = sample["identifier"]
        assert isinstance(identifier, str) and identifier not in cached_samples
        cached_samples[identifier] = sample
    examples_ds = pyine.utils.transformers.prepare_examples_from_conversations(
        convo_ds=hf_msgs_ds,
        tokenizer=tokenizer,
        max_seq_len=model_max_seq_len,
        num_proc=2,
        keep_extra_fields=True,
    )
    assert len(examples_ds) >= max_samples, "fewer examples than samples??"
    collator = pyine.utils.transformers.PaddingCollatorWithPromptMask(
        tokenizer=tokenizer,
        max_length=model_max_seq_len,
        keep_extra_fields=True,
    )
    collator_batch_size = min(len(examples_ds), 4)
    data_loader = torch.utils.data.DataLoader(
        examples_ds,
        batch_size=collator_batch_size,
        collate_fn=collator,
        shuffle=False,
        drop_last=False,
    )
    assert len(data_loader) <= len(examples_ds), "more batches than examples??"
    expected_batch_keys = ["input_ids", "attention_mask", "labels"]
    expected_sample_keys = ["identifier", "code", "inputs", "expected_output"]  # dropped messages
    for example_batch in data_loader:
        assert "identifier" in example_batch and isinstance(example_batch["identifier"], list)
        batch_size = len(example_batch["identifier"])
        assert all(k in example_batch for k in expected_sample_keys)
        for k in expected_sample_keys:
            assert isinstance(example_batch[k], list) and len(example_batch[k]) == batch_size
        assert all(k in example_batch for k in expected_batch_keys)
        for k in expected_batch_keys:
            assert isinstance(example_batch[k], torch.Tensor)
            assert example_batch[k].dtype == torch.long and example_batch[k].ndim == 2
            assert example_batch[k].shape[0] == batch_size
            assert example_batch[k].shape[1] <= model_max_seq_len
        for iter_idx, (ids, attn, lbls) in enumerate(
            zip(
                example_batch["input_ids"],
                example_batch["attention_mask"],
                example_batch["labels"],
                strict=False,
            )
        ):
            identifier = example_batch["identifier"][iter_idx]
            assert identifier in cached_samples
            matched_sample = cached_samples[identifier]
            for sample_key in expected_sample_keys:
                assert matched_sample[sample_key] == example_batch[sample_key][iter_idx]
            non_ignore_labels_mask = lbls != -100
            output_ids = ids[non_ignore_labels_mask]
            assert len(output_ids) > 0
            output_txt = tokenizer.decode(output_ids, skip_special_tokens=True)
            expected_output = example_batch["expected_output"][iter_idx].strip()
            # round-trip expected_output through tokenizer to handle encode/decode quirks
            round_tripped_expected = tokenizer.decode(
                tokenizer.encode(expected_output),
                skip_special_tokens=True,
            )
            assert _normalized_equals(output_txt, round_tripped_expected), "output text mismatch"
            non_ignore_attn_mask = attn != 0
            padding_mask = ~non_ignore_attn_mask
            # Note: when always_pad_to_max_length=False, some examples may have no padding
            if padding_mask.any():
                assert torch.unique(ids[padding_mask]).tolist() == [tokenizer.pad_token_id], "unexpected padding tokens"
            inputs_mask = non_ignore_attn_mask & ~non_ignore_labels_mask
            assert inputs_mask.sum().item() > 0, "no input tokens in prepared tensors??"
            prompt_ids = ids[inputs_mask]
            prompt_txt = tokenizer.decode(prompt_ids, skip_special_tokens=True)
            assert "You are an expert at interpreting and executing Python 3 code" in prompt_txt, "prompt text missing"
            round_tripped_code = tokenizer.decode(  # to ensure compatibility with tokenizer quirks
                tokenizer.encode(example_batch["code"][iter_idx].strip()),
                skip_special_tokens=True,
            )
            assert _normalized_contains(prompt_txt, round_tripped_code), "code snippet missing from prompt text?"
            round_tripped_inputs = tokenizer.decode(  # to ensure compatibility with tokenizer quirks
                tokenizer.encode(example_batch["inputs"][iter_idx].strip()),
                skip_special_tokens=True,
            )
            assert _normalized_contains(prompt_txt, round_tripped_inputs), "inputs missing from prompt text?"
    dm.teardown()


def _make_sample_data(identifier: str, code: str = "pass") -> pyine.organisms.datamodules.samples.common.SampleData:
    """Create a minimal SampleData for testing."""
    return pyine.organisms.datamodules.samples.common.SampleData(
        identifier=identifier,
        code=code,
        description="",
        entrypoint="",
        first_line=0,
        last_line=1,
        inputs="",
        expected_output="None",
        predict_type="program_output",
        code_type="original",
        trace_step_count=1,
        comma_separated_tags="",
        has_code_override=False,
        complexity_metrics={},
    )


class TestSampleHintIdentifierWrapper:
    """Tests for the SampleHintIdentifierWrapper class."""

    def test_modifies_overlapping_identifiers(self) -> None:
        """Wrapper should add suffix to identifiers in overlapping set."""
        sample1 = _make_sample_data("trace_1", "def foo(): pass")
        sample2 = _make_sample_data("trace_2", "def bar(): pass")
        mock_dataset = [sample1, sample2]
        overlapping_ids = frozenset(["trace_1"])  # only trace_1 is overlapping
        wrapper = shortcuts_mod.SampleHintIdentifierWrapper(
            wrapped_dataset=mock_dataset,  # type: ignore[arg-type]
            overlapping_trace_ids=overlapping_ids,
            hint_suffix="hinted",
        )
        # trace_1 should have suffix added
        result1 = wrapper[0]
        assert result1.identifier == "trace_1::hinted"
        # trace_2 should NOT have suffix (not overlapping)
        result2 = wrapper[1]
        assert result2.identifier == "trace_2"

    def test_preserves_non_overlapping_identifiers(self) -> None:
        """Wrapper should not modify identifiers not in overlapping set."""
        sample = _make_sample_data("trace_3", "x = 1")
        wrapper = shortcuts_mod.SampleHintIdentifierWrapper(
            wrapped_dataset=[sample],  # type: ignore[arg-type]
            overlapping_trace_ids=frozenset(["other_trace"]),
            hint_suffix="hintless",
        )
        result = wrapper[0]
        assert result.identifier == "trace_3"  # unchanged

    def test_different_suffixes_create_unique_identifiers(self) -> None:
        """Same trace with different suffixes should have unique identifiers."""
        sample = _make_sample_data("shared_trace")
        overlapping = frozenset(["shared_trace"])
        wrapper_hinted = shortcuts_mod.SampleHintIdentifierWrapper(
            wrapped_dataset=[sample],  # type: ignore[arg-type]
            overlapping_trace_ids=overlapping,
            hint_suffix="hinted",
        )
        wrapper_hintless = shortcuts_mod.SampleHintIdentifierWrapper(
            wrapped_dataset=[sample],  # type: ignore[arg-type]
            overlapping_trace_ids=overlapping,
            hint_suffix="hintless",
        )
        assert wrapper_hinted[0].identifier == "shared_trace::hinted"
        assert wrapper_hintless[0].identifier == "shared_trace::hintless"
        assert wrapper_hinted[0].identifier != wrapper_hintless[0].identifier

    def test_get_stats_includes_wrapper_info(self, mocker: MockerFixture) -> None:
        """Wrapper stats should include overlapping trace count and suffix."""
        mock_wrapped = mocker.Mock()
        mock_wrapped.get_stats.return_value = {"base_stat": 42}
        wrapper = shortcuts_mod.SampleHintIdentifierWrapper(
            wrapped_dataset=mock_wrapped,
            overlapping_trace_ids=frozenset(["a", "b", "c"]),
            hint_suffix="hinted",
        )
        stats = wrapper.get_stats()
        assert stats["overlapping_trace_count"] == 3
        assert stats["hint_suffix"] == "::hinted"
        assert stats["base_stat"] == 42

    def test_len_matches_wrapped_dataset(self, mocker: MockerFixture) -> None:
        """Wrapper length should match underlying dataset."""
        mock_wrapped = mocker.Mock()
        mock_wrapped.__len__ = mocker.Mock(return_value=5)
        wrapper = shortcuts_mod.SampleHintIdentifierWrapper(
            wrapped_dataset=mock_wrapped,
            overlapping_trace_ids=frozenset(),
            hint_suffix="test",
        )
        assert len(wrapper) == 5

    def test_forwards_unknown_attributes_to_wrapped(self, mocker: MockerFixture) -> None:
        """Wrapper should forward unknown attribute access to wrapped dataset."""
        mock_wrapped = mocker.Mock()
        mock_wrapped.selection_config = {"code_type_prob_map": {"hinted": 1.0}}
        mock_wrapped.some_custom_attr = "custom_value"
        wrapper = shortcuts_mod.SampleHintIdentifierWrapper(
            wrapped_dataset=mock_wrapped,
            overlapping_trace_ids=frozenset(),
            hint_suffix="test",
        )
        # should forward attribute access
        assert wrapper.selection_config == {"code_type_prob_map": {"hinted": 1.0}}
        assert wrapper.some_custom_attr == "custom_value"


class TestGetOverlappingTraceIds:
    """Tests for the _get_overlapping_trace_ids method."""

    def test_returns_intersection_of_derived_subsets(self) -> None:
        """Should return trace IDs present in both hinted and hintless partitions."""
        dm = _make_stub_shortcuts_datamodule()
        trace_meta_a = _make_mock_trace_metadata("trace_a")
        trace_meta_b = _make_mock_trace_metadata("trace_b")
        trace_meta_c = _make_mock_trace_metadata("trace_c")
        dm._metadata = types.SimpleNamespace(
            derived_subsets={
                "valid_hinted": types.SimpleNamespace(traces=[trace_meta_a, trace_meta_b]),
                "valid_hintless": types.SimpleNamespace(traces=[trace_meta_b, trace_meta_c]),
            }
        )
        result = dm._get_overlapping_trace_ids("valid", "hinted")
        assert result == frozenset(["trace_b"])

    def test_returns_empty_when_no_overlap(self) -> None:
        """Should return empty set when derived subsets are disjoint."""
        dm = _make_stub_shortcuts_datamodule()
        trace_meta_a = _make_mock_trace_metadata("trace_a")
        trace_meta_b = _make_mock_trace_metadata("trace_b")
        dm._metadata = types.SimpleNamespace(
            derived_subsets={
                "valid_hinted": types.SimpleNamespace(traces=[trace_meta_a]),
                "valid_hintless": types.SimpleNamespace(traces=[trace_meta_b]),
            }
        )
        result = dm._get_overlapping_trace_ids("valid", "hinted")
        assert result == frozenset()

    def test_returns_empty_when_derived_subset_missing(self) -> None:
        """Should return empty set when derived subsets don't exist."""
        dm = _make_stub_shortcuts_datamodule()
        dm._metadata = types.SimpleNamespace(derived_subsets={})
        result = dm._get_overlapping_trace_ids("valid", "hinted")
        assert result == frozenset()


def _make_mock_trace_metadata(identifier: str) -> types.SimpleNamespace:
    """Helper to create mock trace metadata with identifier."""
    return types.SimpleNamespace(identifier=identifier)


class TestCounterfactualGroupIsComplete:
    """Tests for CounterfactualGroup.is_complete()."""

    def test_complete_with_hintless_and_hinted(self, mocker: MockerFixture) -> None:
        group = shortcuts_mod.CounterfactualGroup(
            family_id="f1",
            base_augment_key="original",
            hintless_traces=[mocker.MagicMock()],
            hinted_traces=[mocker.MagicMock()],
        )
        assert group.is_complete((HintType.helpful,)) is True

    def test_incomplete_without_hintless(self, mocker: MockerFixture) -> None:
        group = shortcuts_mod.CounterfactualGroup(
            family_id="f1",
            base_augment_key="original",
            hinted_traces=[mocker.MagicMock()],
        )
        assert group.is_complete((HintType.helpful,)) is False

    def test_complete_via_prompt_db_anchor(self, mocker: MockerFixture) -> None:
        anchor = mocker.MagicMock()
        group = shortcuts_mod.CounterfactualGroup(
            family_id="f1",
            base_augment_key="original",
            hintless_traces=[anchor],
            prompt_db_anchor=anchor,
            prompt_db_has_helpful=True,
        )
        assert group.is_complete((HintType.helpful,)) is True

    def test_complete_with_multiple_hint_types(self, mocker: MockerFixture) -> None:
        group = shortcuts_mod.CounterfactualGroup(
            family_id="f1",
            base_augment_key="original",
            hintless_traces=[mocker.MagicMock()],
            hinted_traces=[mocker.MagicMock()],
            misleading_traces=[mocker.MagicMock()],
        )
        assert group.is_complete((HintType.helpful, HintType.misleading)) is True

    def test_incomplete_when_missing_one_of_multiple_hint_types(self, mocker: MockerFixture) -> None:
        group = shortcuts_mod.CounterfactualGroup(
            family_id="f1",
            base_augment_key="original",
            hintless_traces=[mocker.MagicMock()],
            hinted_traces=[mocker.MagicMock()],
        )
        assert group.is_complete((HintType.helpful, HintType.misleading)) is False


class TestBuildCounterfactualGroups:
    """Tests for _build_counterfactual_groups()."""

    @pytest.fixture
    def make_trace(self, mocker: MockerFixture) -> typing.Callable[..., typing.Any]:
        """Fixture factory for creating mock traces."""

        def _factory(
            identifier: str,
            augment_category: str | None = None,
            is_hinted: bool = False,
            is_misleading: bool = False,
            family_id: str = "ds/train/p000001/s0001/t0001",
        ) -> typing.Any:
            trace = mocker.MagicMock()
            trace.identifier = identifier
            trace.trace_id.is_hinted = is_hinted
            trace.trace_id.is_misleading = is_misleading
            trace.trace_id.augment_category = augment_category
            trace.trace_id.get_augmentless_identifier.return_value = family_id
            trace.trace_id.__str__ = mocker.Mock(return_value=identifier)
            return trace

        return _factory

    def test_groups_by_family_and_base_key(self, make_trace: typing.Callable[..., typing.Any]) -> None:
        trace_orig = make_trace("t1", augment_category=None)
        trace_hinted = make_trace("t2", augment_category="hinted", is_hinted=True)
        groups = shortcuts_mod._build_counterfactual_groups(
            traces=[trace_orig, trace_hinted],
            prompt_db=None,
            eval_hint_types=(HintType.helpful,),
            check_prompt_db_fn=lambda *_args: False,
        )
        group_key = ("ds/train/p000001/s0001/t0001", "original")
        assert group_key in groups
        assert len(groups[group_key].hintless_traces) == 1
        assert len(groups[group_key].hinted_traces) == 1

    def test_raises_on_invalid_lmdb_both_hints(self, make_trace: typing.Callable[..., typing.Any]) -> None:
        trace_bad = make_trace("t1", augment_category="hinted", is_hinted=True, is_misleading=True)
        with pytest.raises(ValueError, match="(?i)invalid LMDB state"):
            shortcuts_mod._build_counterfactual_groups(
                traces=[trace_bad],
                prompt_db=None,
                eval_hint_types=(HintType.helpful,),
                check_prompt_db_fn=lambda *_args: False,
            )

    def test_routes_hintless_correctly(self, make_trace: typing.Callable[..., typing.Any]) -> None:
        trace = make_trace("t1", augment_category=None)
        groups = shortcuts_mod._build_counterfactual_groups(
            traces=[trace],
            prompt_db=None,
            eval_hint_types=(HintType.helpful,),
            check_prompt_db_fn=lambda *_args: False,
        )
        group_key = ("ds/train/p000001/s0001/t0001", "original")
        assert len(groups[group_key].hintless_traces) == 1
        assert len(groups[group_key].hinted_traces) == 0

    def test_checks_prompt_db_availability(
        self,
        mocker: MockerFixture,
        make_trace: typing.Callable[..., typing.Any],
    ) -> None:
        trace = make_trace("t1", augment_category=None)
        mock_db = mocker.MagicMock()
        groups = shortcuts_mod._build_counterfactual_groups(
            traces=[trace],
            prompt_db=mock_db,
            eval_hint_types=(HintType.helpful,),
            check_prompt_db_fn=lambda *_args: True,
        )
        group_key = ("ds/train/p000001/s0001/t0001", "original")
        assert groups[group_key].prompt_db_anchor is trace
        assert groups[group_key].prompt_db_has_helpful is True

    def test_traces_sorted_within_groups(self, make_trace: typing.Callable[..., typing.Any]) -> None:
        trace_z = make_trace("z_trace", augment_category=None)
        trace_a = make_trace("a_trace", augment_category=None)
        groups = shortcuts_mod._build_counterfactual_groups(
            traces=[trace_z, trace_a],
            prompt_db=None,
            eval_hint_types=(HintType.helpful,),
            check_prompt_db_fn=lambda *_args: False,
        )
        group_key = ("ds/train/p000001/s0001/t0001", "original")
        hintless = groups[group_key].hintless_traces
        assert str(hintless[0].trace_id) == "a_trace"
        assert str(hintless[1].trace_id) == "z_trace"


class TestSampleGroupsByDistribution:
    """Tests for _sample_groups_by_distribution()."""

    @pytest.fixture
    def make_group(
        self,
        mocker: MockerFixture,
    ) -> typing.Callable[[str, str | tuple[str, ...]], shortcuts_mod.CounterfactualGroup]:
        """Fixture factory for creating CounterfactualGroup objects."""

        def _factory(
            family_id: str,
            base_key: str | tuple[str, ...] = "original",
        ) -> shortcuts_mod.CounterfactualGroup:
            return shortcuts_mod.CounterfactualGroup(
                family_id=family_id,
                base_augment_key=base_key,
                hintless_traces=[mocker.MagicMock()],
                hinted_traces=[mocker.MagicMock()],
            )

        return _factory

    def test_respects_proportions(
        self,
        make_group: typing.Callable[[str, str | tuple[str, ...]], shortcuts_mod.CounterfactualGroup],
    ) -> None:
        groups = [make_group(f"orig_{idx}", "original") for idx in range(6)] + [
            make_group(f"obf_{idx}", ("obfuscated",)) for idx in range(6)
        ]
        rng = np.random.default_rng(42)
        result = shortcuts_mod._sample_groups_by_distribution(groups, {"original": 0.5, "obfuscated": 0.5}, rng)
        orig_count = sum(1 for g in result if g.base_augment_key == "original")
        obf_count = sum(1 for g in result if g.base_augment_key == ("obfuscated",))
        assert orig_count == obf_count
        assert orig_count + obf_count == len(result)

    def test_reproducible_with_seed(
        self,
        make_group: typing.Callable[[str, str | tuple[str, ...]], shortcuts_mod.CounterfactualGroup],
    ) -> None:
        groups = [make_group(f"g_{idx}") for idx in range(10)]
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        result1 = shortcuts_mod._sample_groups_by_distribution(groups, {"original": 1.0}, rng1)
        result2 = shortcuts_mod._sample_groups_by_distribution(groups, {"original": 1.0}, rng2)
        ids1 = [g.family_id for g in result1]
        ids2 = [g.family_id for g in result2]
        assert ids1 == ids2

    def test_empty_input_returns_empty(self) -> None:
        rng = np.random.default_rng(42)
        result = shortcuts_mod._sample_groups_by_distribution([], {"original": 1.0}, rng)
        assert result == []

    def test_shortage_limits_to_bottleneck(
        self,
        make_group: typing.Callable[[str, str | tuple[str, ...]], shortcuts_mod.CounterfactualGroup],
    ) -> None:
        groups = [make_group(f"orig_{idx}", "original") for idx in range(10)] + [
            make_group("obf_0", ("obfuscated",)),  # only 1 obfuscated
        ]
        rng = np.random.default_rng(42)
        result = shortcuts_mod._sample_groups_by_distribution(groups, {"original": 0.5, "obfuscated": 0.5}, rng)
        # obfuscated bottleneck: 1 / 0.5 = 2 total max
        assert len(result) <= 2

    def test_deduplicates_families_across_buckets(
        self,
        make_group: typing.Callable[[str, str | tuple[str, ...]], shortcuts_mod.CounterfactualGroup],
    ) -> None:
        """Same family in multiple buckets -> kept only in one bucket."""
        groups = [
            make_group("shared_fam", "original"),
            make_group("shared_fam", ("obfuscated",)),
            make_group("orig_only", "original"),
            make_group("obf_only", ("obfuscated",)),
        ]
        rng = np.random.default_rng(42)
        result = shortcuts_mod._sample_groups_by_distribution(
            groups,
            {"original": 0.7, "obfuscated": 0.3},
            rng,
        )
        family_ids = [g.family_id for g in result]
        # shared_fam should appear at most once
        assert family_ids.count("shared_fam") <= 1

    def test_dedup_equal_probs_preserves_distribution(
        self,
        make_group: typing.Callable[[str, str | tuple[str, ...]], shortcuts_mod.CounterfactualGroup],
    ) -> None:
        """Equal probability + shared families -> both buckets remain non-empty."""
        # 4 families in both buckets, 2 unique per bucket; if dedup were deterministic by
        # key name, all shared families would go to one bucket, emptying the other.
        groups = [
            make_group("shared_0", "original"),
            make_group("shared_0", ("obfuscated",)),
            make_group("shared_1", "original"),
            make_group("shared_1", ("obfuscated",)),
            make_group("shared_2", "original"),
            make_group("shared_2", ("obfuscated",)),
            make_group("shared_3", "original"),
            make_group("shared_3", ("obfuscated",)),
            make_group("orig_only", "original"),
            make_group("obf_only", ("obfuscated",)),
        ]
        rng = np.random.default_rng(42)
        result = shortcuts_mod._sample_groups_by_distribution(
            groups,
            {"original": 0.5, "obfuscated": 0.5},
            rng,
        )
        # result should be non-empty (distribution is achievable)
        assert len(result) > 0
        orig_count = sum(1 for g in result if g.base_augment_key == "original")
        obf_count = sum(1 for g in result if g.base_augment_key != "original")
        # equal proportions should be maintained
        assert orig_count == obf_count
        # no family appears more than once
        family_ids = [g.family_id for g in result]
        assert len(family_ids) == len(set(family_ids))

    def test_all_shared_families_no_unique_per_bucket(
        self,
        make_group: typing.Callable[[str, str | tuple[str, ...]], shortcuts_mod.CounterfactualGroup],
    ) -> None:
        """All families shared, no unique per bucket -> repair prevents empty buckets."""
        # every family appears in both buckets -- with independent draws, an unlucky seed
        # could assign all to one bucket. The repair should guarantee both are populated.
        groups = [make_group(f"shared_{idx}", "original") for idx in range(4)] + [
            make_group(f"shared_{idx}", ("obfuscated",)) for idx in range(4)
        ]
        # test across many seeds to ensure the guarantee holds deterministically
        for seed in range(50):
            rng = np.random.default_rng(seed)
            result = shortcuts_mod._sample_groups_by_distribution(
                groups,
                {"original": 0.5, "obfuscated": 0.5},
                rng,
            )
            assert len(result) > 0, f"empty result with seed={seed}"
            orig_count = sum(1 for g in result if g.base_augment_key == "original")
            obf_count = sum(1 for g in result if g.base_augment_key != "original")
            assert orig_count > 0, f"zero original with seed={seed}"
            assert obf_count > 0, f"zero obfuscated with seed={seed}"
            assert orig_count == obf_count, f"unequal counts with seed={seed}"

    def test_repair_accounts_for_unique_families_in_donor(
        self,
        make_group: typing.Callable[[str, str | tuple[str, ...]], shortcuts_mod.CounterfactualGroup],
    ) -> None:
        """Unique family in donor bucket allows stealing its shared family."""
        # orig_only is unique to "original", shared is in both.
        # if the draw assigns shared to "original", obfuscated is starved.
        # the repair should move shared to obfuscated (original still has orig_only).
        groups = [
            make_group("orig_only", "original"),
            make_group("shared", "original"),
            make_group("shared", ("obfuscated",)),
        ]
        for seed in range(50):
            rng = np.random.default_rng(seed)
            result = shortcuts_mod._sample_groups_by_distribution(
                groups,
                {"original": 0.5, "obfuscated": 0.5},
                rng,
            )
            assert len(result) > 0, f"empty result with seed={seed}"
            orig = [g for g in result if g.base_augment_key == "original"]
            obf = [g for g in result if g.base_augment_key != "original"]
            assert len(orig) > 0, f"zero original with seed={seed}"
            assert len(obf) > 0, f"zero obfuscated with seed={seed}"

    def test_multi_augment_groups_logged(
        self,
        make_group: typing.Callable[[str, str | tuple[str, ...]], shortcuts_mod.CounterfactualGroup],
        mocker: MockerFixture,
    ) -> None:
        """Multi-augment base groups are skipped with a warning."""
        multi_aug_group = shortcuts_mod.CounterfactualGroup(
            family_id="multi_fam",
            base_augment_key=("obfuscated", "bugged"),
            hintless_traces=[mocker.MagicMock()],
            hinted_traces=[mocker.MagicMock()],
        )
        groups = [make_group("normal_fam", "original"), multi_aug_group]
        rng = np.random.default_rng(42)
        mock_warn = mocker.patch.object(shortcuts_mod.logger, "warning")
        result = shortcuts_mod._sample_groups_by_distribution(groups, {"original": 1.0}, rng)
        assert len(result) == 1  # only normal_fam
        assert any("multi-augment" in str(call) for call in mock_warn.call_args_list)


class TestPromptDbAnchorPairing:
    """Tests for the 'same trace across subsets' invariant with prompt-DB."""

    @pytest.fixture
    def make_trace(
        self,
        mocker: MockerFixture,
    ) -> typing.Callable[[str], typing.Any]:
        """Fixture factory for creating mock traces."""

        def _factory(identifier: str) -> typing.Any:
            trace = mocker.MagicMock()
            trace.trace_id.__str__ = mocker.Mock(return_value=identifier)
            trace.trace_id.get_augmentless_identifier.return_value = "fam1"
            trace.trace_id.augment_category = None
            trace.trace_id.is_hinted = False
            trace.trace_id.is_misleading = False
            return trace

        return _factory

    def test_prompt_db_anchor_used_for_hintless_and_hinted(
        self,
        make_trace: typing.Callable[[str], typing.Any],
        mocker: MockerFixture,
    ) -> None:
        """When prompt-DB provides hints, hintless and hinted subsets use the same trace."""
        anchor = make_trace("anchor_trace")
        rng = np.random.default_rng(42)
        subsets = shortcuts_mod.build_counterfactual_eval_subsets(
            traces=[anchor],
            eval_hint_types=(HintType.helpful,),
            code_type_prob_map={"original": 1.0},
            rng=rng,
            prompt_db=mocker.MagicMock(),
            check_prompt_db_fn=lambda *_args: True,
        )
        assert len(subsets["hintless"]) == 1
        assert len(subsets["hinted"]) == 1
        assert subsets["hintless"][0] is subsets["hinted"][0]  # same object

    def test_prompt_db_anchor_deterministic_after_sort(
        self,
        make_trace: typing.Callable[[str], typing.Any],
        mocker: MockerFixture,
    ) -> None:
        """Prompt-DB anchor is chosen from sorted order, not input order."""
        trace_a = make_trace("a_trace")
        trace_z = make_trace("z_trace")
        # pass in reverse order: z first, a second
        groups = shortcuts_mod._build_counterfactual_groups(
            traces=[trace_z, trace_a],
            prompt_db=mocker.MagicMock(),
            eval_hint_types=(HintType.helpful,),
            check_prompt_db_fn=lambda *_args: True,
        )
        group_key = ("fam1", "original")
        # anchor should be "a_trace" (first after sort), regardless of input order
        assert str(groups[group_key].prompt_db_anchor.trace_id) == "a_trace"


class TestDerivedSubsetPreFiltering:
    """Tests for pre-filtering traces before hint partitioning."""

    def test_derived_subsets_have_disabled_filtering_hint_presence(
        self,
        fake_lmdb_and_split: tuple[pathlib.Path, pathlib.Path],
    ) -> None:
        """Derived subsets should have all filtering disabled (hint_presence_split)."""
        lmdb_path, split_path = fake_lmdb_and_split
        config = pyine.organisms.datamodules.shortcuts_configs.ShortcutBiasDataModuleConfig(
            lmdb_paths=[str(lmdb_path)],
            split_file_path=str(split_path),
            eval_subset_names=("valid",),
            evaluation_strategy=EvaluationStrategy.hint_presence_split,
            instantiate_parsers_at_setup=False,
        )
        for derived_name in ["valid_hinted", "valid_hintless"]:
            parser_config = config._resolve_dataparser_config(derived_name)
            params = parser_config.get_params_dict()
            filtering_dict = params.get("filtering_config", {})
            if isinstance(filtering_dict, dict):
                filtering = pyine.organisms.datamodules.samples.configs.TraceFilteringConfig(**filtering_dict)
            else:
                filtering = filtering_dict
            assert not filtering.any_filtering_enabled, f"{derived_name} should have disabled filtering"

    def test_derived_subsets_have_disabled_filtering_counterfactual(
        self,
        fake_lmdb_and_split: tuple[pathlib.Path, pathlib.Path],
    ) -> None:
        """Derived subsets should have all filtering disabled (counterfactual)."""
        lmdb_path, split_path = fake_lmdb_and_split
        config = pyine.organisms.datamodules.shortcuts_configs.ShortcutBiasDataModuleConfig(
            lmdb_paths=[str(lmdb_path)],
            split_file_path=str(split_path),
            eval_subset_names=("valid",),
            evaluation_strategy=EvaluationStrategy.counterfactual,
            instantiate_parsers_at_setup=False,
        )
        for derived_name in ["valid_hinted", "valid_hintless"]:
            parser_config = config._resolve_dataparser_config(derived_name)
            params = parser_config.get_params_dict()
            filtering_dict = params.get("filtering_config", {})
            if isinstance(filtering_dict, dict):
                filtering = pyine.organisms.datamodules.samples.configs.TraceFilteringConfig(**filtering_dict)
            else:
                filtering = filtering_dict
            assert not filtering.any_filtering_enabled, f"{derived_name} should have disabled filtering"

    def test_parent_filtering_preserved_for_non_derived(
        self,
        fake_lmdb_and_split: tuple[pathlib.Path, pathlib.Path],
    ) -> None:
        """Parent eval subset should keep its own filtering config."""
        lmdb_path, split_path = fake_lmdb_and_split
        config = pyine.organisms.datamodules.shortcuts_configs.ShortcutBiasDataModuleConfig(
            lmdb_paths=[str(lmdb_path)],
            split_file_path=str(split_path),
            eval_subset_names=("valid",),
            instantiate_parsers_at_setup=False,
        )
        parser_config = config._resolve_dataparser_config("valid")
        params = parser_config.get_params_dict()
        filtering_dict = params.get("filtering_config", {})
        if isinstance(filtering_dict, dict):
            filtering = pyine.organisms.datamodules.samples.configs.TraceFilteringConfig(**filtering_dict)
        else:
            filtering = filtering_dict
        # parent "valid" should use default filtering (which has active filters)
        assert filtering.any_filtering_enabled

    def test_create_disabled_returns_no_active_filters(self) -> None:
        """TraceFilteringConfig.create_disabled() should produce a valid config with no active filters."""
        disabled = pyine.organisms.datamodules.samples.configs.TraceFilteringConfig.create_disabled()
        assert not disabled.any_filtering_enabled
        assert disabled.use_token_lengths is False
        assert disabled.tokenizer_model_id is None
        assert disabled.tokenizer_path is None


class TestCounterfactualPairingInvariant:
    """Tests for the counterfactual family alignment invariant."""

    def test_raises_on_mismatched_partitions(self, mocker: MockerFixture) -> None:
        """Pre-filter invariant should raise ValueError when partitions have different families."""
        stub = _make_stub_shortcuts_datamodule(
            evaluation_strategy=EvaluationStrategy.counterfactual,
            eval_hint_types=(HintType.helpful,),
        )
        base_trace_id = _MockTraceId("test/p0001/s0001/t0001")
        trace_hinted = _MockTraceMeta(
            "t1",
            _MockTraceId("t1", augment_category="hints_docs", is_hinted=True),
        )
        trace_hinted.trace_id.get_augmentless_identifier = lambda: base_trace_id
        trace_hintless = _MockTraceMeta(
            "t2",
            _MockTraceId("t2", is_hinted=False),
        )
        trace_hintless.trace_id.get_augmentless_identifier = lambda: base_trace_id
        # extra hintless trace from a DIFFERENT family (only in hintless, not in hinted)
        other_base = _MockTraceId("test/p0002/s0001/t0001")
        trace_extra = _MockTraceMeta(
            "t3",
            _MockTraceId("t3", is_hinted=False),
        )
        trace_extra.trace_id.get_augmentless_identifier = lambda: other_base

        # monkeypatch _partition_traces_by_hint_strategy to return mismatched partitions
        def _fake_partition(
            self_arg: typing.Any,
            traces: typing.Any,
            prompt_db: typing.Any = None,
            eval_subset_name: typing.Any = None,
        ) -> dict[str, list[typing.Any]]:
            return {
                "hintless": [trace_hintless, trace_extra],  # has extra family
                "hinted": [trace_hinted],  # missing the extra family
                "misleading": [],
            }

        mocker.patch.object(
            shortcuts_mod.ShortcutBiasDataModule,
            "_partition_traces_by_hint_strategy",
            _fake_partition,
        )
        subset_traces = {"valid": [trace_hinted, trace_hintless, trace_extra]}
        with pytest.raises(ValueError, match="counterfactual pairing broken"):
            stub._create_hint_split_derived_subsets(subset_traces)
