"""Tests for pyine.evals.correctness.datamodule."""

from __future__ import annotations

import pathlib

import datasets
import pytest

import pyine.evals.correctness.datamodule as correctness_datamodule
import pyine.evals.correctness.datamodule_configs as correctness_datamodule_configs
import pyine.evals.correctness.splits as correctness_splits
import pyine.evals.correctness.types as correctness_types

_FAKE_LMDB_PATH = pathlib.Path("/fake/lmdb")


def _make_record(
    sample_id: str,
    problem_id: str,
    attempt_index: int,
    label: bool,
    code_type: str = "original",
) -> correctness_types.EvalRecord:
    model_output = f"output for {sample_id}"
    return correctness_types.EvalRecord(
        sample_id=sample_id,
        problem_id=problem_id,
        attempt_index=attempt_index,
        model_output=model_output,
        final_answer="42",
        expected_output="expected",
        label=label,
        code_type=code_type,
        tags=[],
        record={"prompt": f"prompt for {sample_id}", "model_output": model_output},
        difficulty_score=None,
    )


def _make_splits() -> correctness_splits.GuardrailSplits:
    """Build minimal splits: 2 train, 4 valid (mixed labels), 2 test."""
    train_recs = [
        _make_record("t0/s0/t0", "t0", 0, True),
        _make_record("t0/s0/t0", "t0", 1, False),
    ]
    valid_recs = [
        _make_record("v0/s0/t0", "v0", 0, True),
        _make_record("v0/s0/t0", "v0", 1, False),
        _make_record("v1/s0/t0", "v1", 0, True),
        _make_record("v1/s0/t0", "v1", 1, False),
    ]
    test_recs = [
        _make_record("e0/s0/t0", "e0", 0, True),
        _make_record("e0/s0/t0", "e0", 1, False),
    ]
    return correctness_splits.GuardrailSplits(
        guardrail_train=train_recs,
        guardrail_valid=valid_recs,
        guardrail_test=test_recs,
        train_problem_ids=frozenset({"t0"}),
        valid_problem_ids=frozenset({"v0", "v1"}),
        test_problem_ids=frozenset({"e0"}),
    )


@pytest.fixture
def mock_data(monkeypatch: pytest.MonkeyPatch) -> correctness_splits.GuardrailSplits:
    splits = _make_splits()
    all_records = splits.guardrail_train + splits.guardrail_valid + splits.guardrail_test
    monkeypatch.setattr(
        "pyine.data.utils.lmdb_io.resolve_lmdb_paths",
        lambda raw_paths: list(raw_paths),
    )
    monkeypatch.setattr(
        "pyine.evals.correctness.datamodule.correctness_data_loading.load_records_from_lmdb",
        lambda *_args, **_kwargs: all_records,
    )
    monkeypatch.setattr(
        "pyine.evals.correctness.datamodule.correctness_splits.build_guardrail_splits",
        lambda *_args, **_kwargs: splits,
    )
    return splits


class TestCorrectnessDataModuleLifecycle:
    def _make_config(self) -> correctness_datamodule_configs.CorrectnessDataModuleConfig:
        return correctness_datamodule_configs.CorrectnessDataModuleConfig(
            lmdb_paths=(_FAKE_LMDB_PATH,),
            split_config=correctness_types.GuardrailSplitConfig(split_source="TACO"),
        )

    def test_full_lifecycle(self, mock_data: correctness_splits.GuardrailSplits) -> None:
        config = self._make_config()
        dm = correctness_datamodule.CorrectnessDataModule(config)
        dm.prepare_data()
        dm.setup()
        assert dm.get_guardrail_splits() is not None
        assert len(dm.get_all_records()) == 8
        dm.teardown()

    def test_accessors_before_setup_raise(self) -> None:
        config = self._make_config()
        dm = correctness_datamodule.CorrectnessDataModule(config)
        with pytest.raises(RuntimeError, match="not set up"):
            dm.get_guardrail_splits()
        with pytest.raises(RuntimeError, match="not set up"):
            dm.get_all_records()
        with pytest.raises(RuntimeError, match="not set up"):
            dm.get_records_for_subset("guardrail_valid")

    def test_teardown_clears_state(self, mock_data: correctness_splits.GuardrailSplits) -> None:
        config = self._make_config()
        dm = correctness_datamodule.CorrectnessDataModule(config)
        dm.prepare_data()
        dm.setup()
        dm.teardown()
        with pytest.raises(RuntimeError, match="not set up"):
            dm.get_guardrail_splits()


class TestGetRecordsForSubset:
    def _make_config(self) -> correctness_datamodule_configs.CorrectnessDataModuleConfig:
        return correctness_datamodule_configs.CorrectnessDataModuleConfig(
            lmdb_paths=(_FAKE_LMDB_PATH,),
            split_config=correctness_types.GuardrailSplitConfig(split_source="TACO"),
        )

    def test_valid_subset_names(self, mock_data: correctness_splits.GuardrailSplits) -> None:
        config = self._make_config()
        dm = correctness_datamodule.CorrectnessDataModule(config)
        dm.prepare_data()
        dm.setup()
        assert len(dm.get_records_for_subset("guardrail_train")) == 2
        assert len(dm.get_records_for_subset("guardrail_valid")) == 4
        assert len(dm.get_records_for_subset("guardrail_test")) == 2

    def test_unknown_subset_raises(self, mock_data: correctness_splits.GuardrailSplits) -> None:
        config = self._make_config()
        dm = correctness_datamodule.CorrectnessDataModule(config)
        dm.prepare_data()
        dm.setup()
        with pytest.raises(ValueError, match="unknown subset name"):
            dm.get_records_for_subset("nonexistent")


class TestGetStats:
    def _make_config(self) -> correctness_datamodule_configs.CorrectnessDataModuleConfig:
        return correctness_datamodule_configs.CorrectnessDataModuleConfig(
            lmdb_paths=(_FAKE_LMDB_PATH,),
            split_config=correctness_types.GuardrailSplitConfig(split_source="TACO"),
        )

    def test_returns_expected_keys(self, mock_data: correctness_splits.GuardrailSplits) -> None:
        config = self._make_config()
        dm = correctness_datamodule.CorrectnessDataModule(config)
        dm.prepare_data()
        dm.setup()
        stats = dm.get_stats()
        assert "guardrail_valid/num_records" in stats
        assert stats["guardrail_valid/num_records"] == 4
        assert "guardrail_valid/num_positive" in stats
        assert "guardrail_valid/num_negative" in stats
        assert "guardrail_valid/num_problems" in stats

    def test_no_resampled_stats_without_config(self, mock_data: correctness_splits.GuardrailSplits) -> None:
        dm = correctness_datamodule.CorrectnessDataModule(self._make_config())
        dm.prepare_data()
        dm.setup()
        stats = dm.get_stats()
        assert "guardrail_train_resampled/num_records" not in stats
        assert "guardrail_valid_resampled/num_records" not in stats

    def test_resampled_stats_when_configured(self, mock_data: correctness_splits.GuardrailSplits) -> None:
        config = correctness_datamodule_configs.CorrectnessDataModuleConfig(
            lmdb_paths=(_FAKE_LMDB_PATH,),
            split_config=correctness_types.GuardrailSplitConfig(split_source="TACO"),
            resampling=correctness_types.RecordResamplingConfig(
                target_positive_ratio=0.5,
                seed=42,
                min_records_per_label=1,
            ),
        )
        dm = correctness_datamodule.CorrectnessDataModule(config)
        dm.prepare_data()
        dm.setup()
        stats = dm.get_stats()
        # raw stats still present
        assert "guardrail_train/num_records" in stats
        assert "guardrail_valid/num_records" in stats
        # resampled stats also present for both train and valid
        assert "guardrail_train_resampled/num_records" in stats
        assert "guardrail_train_resampled/num_positive" in stats
        assert "guardrail_train_resampled/num_negative" in stats
        assert "guardrail_valid_resampled/num_records" in stats
        assert "guardrail_valid_resampled/num_positive" in stats
        assert "guardrail_valid_resampled/num_negative" in stats

    def test_resampled_stats_excluded_when_subset_scoped(
        self,
        mock_data: correctness_splits.GuardrailSplits,
    ) -> None:
        config = correctness_datamodule_configs.CorrectnessDataModuleConfig(
            lmdb_paths=(_FAKE_LMDB_PATH,),
            split_config=correctness_types.GuardrailSplitConfig(split_source="TACO"),
            resampling=correctness_types.RecordResamplingConfig(
                target_positive_ratio=0.5,
                seed=42,
                min_records_per_label=1,
            ),
        )
        dm = correctness_datamodule.CorrectnessDataModule(config)
        dm.prepare_data()
        dm.setup()
        stats = dm.get_stats(target_subsets=["guardrail_test"])
        assert "guardrail_test/num_records" in stats
        assert "guardrail_train_resampled/num_records" not in stats
        assert "guardrail_valid_resampled/num_records" not in stats

    def test_raises_before_setup(self) -> None:
        config = self._make_config()
        dm = correctness_datamodule.CorrectnessDataModule(config)
        with pytest.raises(RuntimeError, match="not set up"):
            dm.get_stats()


class TestGetRecordsForCalibration:
    def _make_config(self) -> correctness_datamodule_configs.CorrectnessDataModuleConfig:
        return correctness_datamodule_configs.CorrectnessDataModuleConfig(
            lmdb_paths=(_FAKE_LMDB_PATH,),
            split_config=correctness_types.GuardrailSplitConfig(split_source="TACO"),
        )

    def test_without_config_returns_all_valid(self, mock_data: correctness_splits.GuardrailSplits) -> None:
        dm = correctness_datamodule.CorrectnessDataModule(self._make_config())
        dm.prepare_data()
        dm.setup()
        records = dm.get_records_for_calibration()
        assert len(records) == len(mock_data.guardrail_valid)

    def test_with_resampling_config(self, mock_data: correctness_splits.GuardrailSplits) -> None:
        dm = correctness_datamodule.CorrectnessDataModule(self._make_config())
        dm.prepare_data()
        dm.setup()
        config = correctness_types.RecordResamplingConfig(
            target_positive_ratio=0.5,
            seed=42,
            min_records_per_label=1,
        )
        records = dm.get_records_for_calibration(resampling_config=config)
        assert len(records) <= len(mock_data.guardrail_valid)
        labels = {rec.label for rec in records}
        assert True in labels
        assert False in labels


class TestGetRecordsForTraining:
    def test_without_config_returns_all_train(self, mock_data: correctness_splits.GuardrailSplits) -> None:
        config = correctness_datamodule_configs.CorrectnessDataModuleConfig(
            lmdb_paths=(_FAKE_LMDB_PATH,),
            split_config=correctness_types.GuardrailSplitConfig(split_source="TACO"),
        )
        dm = correctness_datamodule.CorrectnessDataModule(config)
        dm.prepare_data()
        dm.setup()
        records = dm.get_records_for_training()
        assert len(records) == len(mock_data.guardrail_train)

    def test_with_resampling_config(self, mock_data: correctness_splits.GuardrailSplits) -> None:
        config = correctness_datamodule_configs.CorrectnessDataModuleConfig(
            lmdb_paths=(_FAKE_LMDB_PATH,),
            split_config=correctness_types.GuardrailSplitConfig(split_source="TACO"),
            resampling=correctness_types.RecordResamplingConfig(
                target_positive_ratio=0.5,
                seed=42,
                min_records_per_label=1,
            ),
        )
        dm = correctness_datamodule.CorrectnessDataModule(config)
        dm.prepare_data()
        dm.setup()
        records = dm.get_records_for_training()
        assert len(records) <= len(mock_data.guardrail_train)
        labels = {rec.label for rec in records}
        assert True in labels
        assert False in labels


class TestCodeTypeMappings:
    def _make_config(self) -> correctness_datamodule_configs.CorrectnessDataModuleConfig:
        return correctness_datamodule_configs.CorrectnessDataModuleConfig(
            lmdb_paths=(_FAKE_LMDB_PATH,),
            split_config=correctness_types.GuardrailSplitConfig(split_source="TACO"),
        )

    def test_populated_after_setup(self, mock_data: correctness_splits.GuardrailSplits) -> None:
        dm = correctness_datamodule.CorrectnessDataModule(self._make_config())
        dm.prepare_data()
        dm.setup()
        ct_to_id = dm.code_type_to_id
        id_to_ct = dm.id_to_code_type
        assert len(ct_to_id) > 0
        assert len(id_to_ct) == len(ct_to_id)
        for ct, ct_id in ct_to_id.items():
            assert id_to_ct[ct_id] == ct

    def test_before_setup_raises(self) -> None:
        dm = correctness_datamodule.CorrectnessDataModule(self._make_config())
        with pytest.raises(RuntimeError, match="not set up"):
            _ = dm.code_type_to_id
        with pytest.raises(RuntimeError, match="not set up"):
            _ = dm.id_to_code_type

    def test_cleared_after_teardown(self, mock_data: correctness_splits.GuardrailSplits) -> None:
        dm = correctness_datamodule.CorrectnessDataModule(self._make_config())
        dm.prepare_data()
        dm.setup()
        dm.teardown()
        with pytest.raises(RuntimeError, match="not set up"):
            _ = dm.code_type_to_id


class TestGetProbeDataset:
    def _make_config(self) -> correctness_datamodule_configs.CorrectnessDataModuleConfig:
        return correctness_datamodule_configs.CorrectnessDataModuleConfig(
            lmdb_paths=(_FAKE_LMDB_PATH,),
            split_config=correctness_types.GuardrailSplitConfig(split_source="TACO"),
        )

    def test_returns_train_valid_splits(self, mock_data: correctness_splits.GuardrailSplits) -> None:
        dm = correctness_datamodule.CorrectnessDataModule(self._make_config())
        dm.prepare_data()
        dm.setup()
        ds = dm.get_probe_dataset()
        assert isinstance(ds, datasets.DatasetDict)
        assert "train" in ds
        assert "valid" in ds
        assert "guardrail_test" not in ds  # not included

    def test_has_messages_column(self, mock_data: correctness_splits.GuardrailSplits) -> None:
        dm = correctness_datamodule.CorrectnessDataModule(self._make_config())
        dm.prepare_data()
        dm.setup()
        ds = dm.get_probe_dataset()
        for split_name in ("train", "valid"):
            assert "messages" in ds[split_name].column_names
            assert "label" in ds[split_name].column_names
            assert "sample_id" in ds[split_name].column_names
            assert "code_type" in ds[split_name].column_names
            assert "text" not in ds[split_name].column_names

    def test_messages_are_structured(self, mock_data: correctness_splits.GuardrailSplits) -> None:
        dm = correctness_datamodule.CorrectnessDataModule(self._make_config())
        dm.prepare_data()
        dm.setup()
        ds = dm.get_probe_dataset()
        first_messages = ds["train"][0]["messages"]
        assert isinstance(first_messages, list)
        assert len(first_messages) >= 2
        assert first_messages[-1]["role"] == "assistant"

    def test_resampling_applied(self, mock_data: correctness_splits.GuardrailSplits) -> None:
        config = correctness_datamodule_configs.CorrectnessDataModuleConfig(
            lmdb_paths=(_FAKE_LMDB_PATH,),
            split_config=correctness_types.GuardrailSplitConfig(split_source="TACO"),
            resampling=correctness_types.RecordResamplingConfig(
                target_positive_ratio=0.5,
                seed=42,
                min_records_per_label=1,
            ),
        )
        dm = correctness_datamodule.CorrectnessDataModule(config)
        dm.prepare_data()
        dm.setup()
        ds = dm.get_probe_dataset()
        assert len(ds["train"]) <= len(mock_data.guardrail_train)
        assert len(ds["valid"]) <= len(mock_data.guardrail_valid)


class TestGetRecordsForValidation:
    def test_without_resampling_returns_all_valid(self, mock_data: correctness_splits.GuardrailSplits) -> None:
        config = correctness_datamodule_configs.CorrectnessDataModuleConfig(
            lmdb_paths=(_FAKE_LMDB_PATH,),
            split_config=correctness_types.GuardrailSplitConfig(split_source="TACO"),
        )
        dm = correctness_datamodule.CorrectnessDataModule(config)
        dm.prepare_data()
        dm.setup()
        records = dm.get_records_for_validation()
        assert len(records) == len(mock_data.guardrail_valid)

    def test_with_resampling_symmetric_with_training(self, mock_data: correctness_splits.GuardrailSplits) -> None:
        config = correctness_datamodule_configs.CorrectnessDataModuleConfig(
            lmdb_paths=(_FAKE_LMDB_PATH,),
            split_config=correctness_types.GuardrailSplitConfig(split_source="TACO"),
            resampling=correctness_types.RecordResamplingConfig(
                target_positive_ratio=0.5,
                seed=42,
                min_records_per_label=1,
            ),
        )
        dm = correctness_datamodule.CorrectnessDataModule(config)
        dm.prepare_data()
        dm.setup()
        valid_records = dm.get_records_for_validation()
        train_records = dm.get_records_for_training()
        # both should be resampled (possibly smaller than raw)
        assert len(valid_records) <= len(mock_data.guardrail_valid)
        assert len(train_records) <= len(mock_data.guardrail_train)
        # both should have both labels
        assert {rec.label for rec in valid_records} == {True, False}
        assert {rec.label for rec in train_records} == {True, False}


class TestDataLoaderStubs:
    def test_all_raise_not_implemented(self, mock_data: correctness_splits.GuardrailSplits) -> None:
        config = correctness_datamodule_configs.CorrectnessDataModuleConfig(
            lmdb_paths=(_FAKE_LMDB_PATH,),
            split_config=correctness_types.GuardrailSplitConfig(split_source="TACO"),
        )
        dm = correctness_datamodule.CorrectnessDataModule(config)
        dm.prepare_data()
        dm.setup()
        with pytest.raises(NotImplementedError):
            dm.train_dataloader()
        with pytest.raises(NotImplementedError):
            dm.val_dataloader()
        with pytest.raises(NotImplementedError):
            dm.test_dataloader()
        with pytest.raises(NotImplementedError):
            dm.predict_dataloader()
