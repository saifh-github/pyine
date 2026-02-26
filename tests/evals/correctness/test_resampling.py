"""Tests for pyine.evals.correctness.resampling."""

from __future__ import annotations

import collections

import pydantic
import pytest

import pyine.evals.correctness.resampling as correctness_resampling
import pyine.evals.correctness.types as correctness_types


def _make_record(
    sample_id: str = "s0",
    label: bool = True,
    code_type: str = "original",
) -> correctness_types.EvalRecord:
    return correctness_types.EvalRecord(
        sample_id=sample_id,
        problem_id=sample_id.split("/")[0] if "/" in sample_id else sample_id,
        attempt_index=0,
        model_output=f"output for {sample_id}",
        final_answer="42",
        expected_output="expected",
        label=label,
        code_type=code_type,
        tags=[],
        record={},
        difficulty_score=None,
    )


def _make_mixed_records(
    num_pos: int,
    num_neg: int,
    code_type: str = "original",
) -> list[correctness_types.EvalRecord]:
    records: list[correctness_types.EvalRecord] = []
    for idx in range(num_pos):
        records.append(_make_record(sample_id=f"pos_{code_type}_{idx}", label=True, code_type=code_type))
    for idx in range(num_neg):
        records.append(_make_record(sample_id=f"neg_{code_type}_{idx}", label=False, code_type=code_type))
    return records


class TestRecordResamplingConfig:
    def test_default_config_valid(self) -> None:
        config = correctness_types.RecordResamplingConfig()
        assert config.target_positive_ratio is None
        assert config.code_type_proportions is None
        assert config.strategy == "subsample"
        assert config.max_records is None
        assert config.min_records_per_label == 2
        assert config.seed == 0

    def test_target_positive_ratio_bounds(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            correctness_types.RecordResamplingConfig(target_positive_ratio=0.0)
        with pytest.raises(pydantic.ValidationError):
            correctness_types.RecordResamplingConfig(target_positive_ratio=1.0)
        config = correctness_types.RecordResamplingConfig(target_positive_ratio=0.5)
        assert config.target_positive_ratio == 0.5

    def test_code_type_proportions_empty_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="non-empty"):
            correctness_types.RecordResamplingConfig(code_type_proportions={})

    def test_code_type_proportions_negative_value_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="positive"):
            correctness_types.RecordResamplingConfig(code_type_proportions={"original": -1.0})

    def test_code_type_proportions_zero_value_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="positive"):
            correctness_types.RecordResamplingConfig(code_type_proportions={"original": 0.0})

    def test_code_type_proportions_inf_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="finite"):
            correctness_types.RecordResamplingConfig(code_type_proportions={"original": float("inf")})

    def test_code_type_proportions_nan_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="finite"):
            correctness_types.RecordResamplingConfig(code_type_proportions={"original": float("nan")})

    def test_code_type_proportions_valid(self) -> None:
        config = correctness_types.RecordResamplingConfig(
            code_type_proportions={"original": 2.0, "hinted": 1.0},
        )
        assert config.code_type_proportions == {"original": 2.0, "hinted": 1.0}

    def test_both_axes_set_simultaneously(self) -> None:
        config = correctness_types.RecordResamplingConfig(
            target_positive_ratio=0.5,
            code_type_proportions={"original": 1.0, "hinted": 1.0},
        )
        assert config.target_positive_ratio == 0.5
        assert config.code_type_proportions is not None

    def test_max_records_must_be_positive(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            correctness_types.RecordResamplingConfig(max_records=0)
        config = correctness_types.RecordResamplingConfig(max_records=10)
        assert config.max_records == 10

    def test_is_noop_default(self) -> None:
        config = correctness_types.RecordResamplingConfig()
        assert config.is_noop is True

    def test_is_noop_with_ratio(self) -> None:
        config = correctness_types.RecordResamplingConfig(target_positive_ratio=0.5)
        assert config.is_noop is False

    def test_is_noop_with_code_type(self) -> None:
        config = correctness_types.RecordResamplingConfig(code_type_proportions={"original": 1.0})
        assert config.is_noop is False

    def test_is_noop_with_max_records(self) -> None:
        config = correctness_types.RecordResamplingConfig(max_records=10)
        assert config.is_noop is False


class TestResampleRecordsLabelRatio:
    def test_subsample_to_balanced(self) -> None:
        records = _make_mixed_records(num_pos=80, num_neg=20, code_type="original")
        config = correctness_types.RecordResamplingConfig(target_positive_ratio=0.5, seed=42)
        result = correctness_resampling.resample_records(records, config)
        num_pos = sum(1 for rec in result if rec.label)
        num_neg = len(result) - num_pos
        assert num_pos == num_neg
        assert len(result) <= len(records)

    def test_ratio_achieved_within_tolerance(self) -> None:
        records = _make_mixed_records(num_pos=60, num_neg=40, code_type="original")
        config = correctness_types.RecordResamplingConfig(target_positive_ratio=0.3, seed=42)
        result = correctness_resampling.resample_records(records, config)
        num_pos = sum(1 for rec in result if rec.label)
        actual_ratio = num_pos / len(result)
        assert abs(actual_ratio - 0.3) < 0.05

    def test_both_classes_present(self) -> None:
        records = _make_mixed_records(num_pos=50, num_neg=50, code_type="original")
        config = correctness_types.RecordResamplingConfig(target_positive_ratio=0.9, seed=42)
        result = correctness_resampling.resample_records(records, config)
        labels = {rec.label for rec in result}
        assert True in labels
        assert False in labels

    def test_oversample_duplicates_minority(self) -> None:
        records = _make_mixed_records(num_pos=10, num_neg=90, code_type="original")
        config = correctness_types.RecordResamplingConfig(
            target_positive_ratio=0.5,
            strategy="oversample",
            seed=42,
        )
        result = correctness_resampling.resample_records(records, config)
        num_pos = sum(1 for rec in result if rec.label)
        assert num_pos >= 10  # at least original count, likely more (oversampled)
        assert len(result) >= len(records)

    def test_single_class_input_raises(self) -> None:
        records = _make_mixed_records(num_pos=10, num_neg=0, code_type="original")
        config = correctness_types.RecordResamplingConfig(target_positive_ratio=0.5, seed=42)
        with pytest.raises(ValueError, match="both classes must be present"):
            correctness_resampling.resample_records(records, config)

    def test_ratio_yields_empty_class_raises(self) -> None:
        records = _make_mixed_records(num_pos=2, num_neg=50, code_type="original")
        config = correctness_types.RecordResamplingConfig(target_positive_ratio=0.9, seed=42)
        with pytest.raises(ValueError, match="fewer than 1 record"):
            correctness_resampling.resample_records(records, config)


class TestResampleRecordsCodeType:
    def test_only_listed_code_types_survive(self) -> None:
        records = (
            _make_mixed_records(num_pos=10, num_neg=10, code_type="original")
            + _make_mixed_records(num_pos=10, num_neg=10, code_type="hinted")
            + _make_mixed_records(num_pos=10, num_neg=10, code_type="misleading")
        )
        config = correctness_types.RecordResamplingConfig(
            code_type_proportions={"original": 1.0, "hinted": 1.0},
            seed=42,
        )
        result = correctness_resampling.resample_records(records, config)
        code_types = {rec.code_type for rec in result}
        assert code_types == {"original", "hinted"}

    def test_proportions_are_normalized(self) -> None:
        records = _make_mixed_records(num_pos=30, num_neg=30, code_type="original") + _make_mixed_records(
            num_pos=30, num_neg=30, code_type="hinted"
        )
        config = correctness_types.RecordResamplingConfig(
            code_type_proportions={"original": 2.0, "hinted": 1.0},
            seed=42,
        )
        result = correctness_resampling.resample_records(records, config)
        counts: dict[str, int] = collections.Counter(rec.code_type for rec in result)
        assert counts["original"] > counts["hinted"]
        # ~2:1 ratio (within rounding)
        ratio = counts["original"] / counts["hinted"]
        assert 1.5 < ratio < 2.5

    def test_missing_code_type_raises(self) -> None:
        records = _make_mixed_records(num_pos=10, num_neg=10, code_type="original") + _make_mixed_records(
            num_pos=10, num_neg=10, code_type="hinted"
        )
        config = correctness_types.RecordResamplingConfig(
            code_type_proportions={"original": 1.0, "hinted": 1.0, "bugged": 1.0},
            seed=42,
        )
        with pytest.raises(ValueError, match="no records"):
            correctness_resampling.resample_records(records, config)

    def test_subsample_reduces_over_represented(self) -> None:
        records = _make_mixed_records(num_pos=50, num_neg=50, code_type="original") + _make_mixed_records(
            num_pos=5, num_neg=5, code_type="hinted"
        )
        config = correctness_types.RecordResamplingConfig(
            code_type_proportions={"original": 1.0, "hinted": 1.0},
            seed=42,
        )
        result = correctness_resampling.resample_records(records, config)
        counts: dict[str, int] = collections.Counter(rec.code_type for rec in result)
        assert counts["original"] == counts["hinted"]
        assert counts["original"] <= 10  # constrained by hinted (10 records)

    def test_oversample_expands_minority_code_type(self) -> None:
        records = _make_mixed_records(num_pos=50, num_neg=50, code_type="original") + _make_mixed_records(
            num_pos=5, num_neg=5, code_type="hinted"
        )
        config = correctness_types.RecordResamplingConfig(
            code_type_proportions={"original": 1.0, "hinted": 1.0},
            strategy="oversample",
            seed=42,
        )
        result = correctness_resampling.resample_records(records, config)
        counts: dict[str, int] = collections.Counter(rec.code_type for rec in result)
        assert counts["original"] == counts["hinted"]
        assert counts["hinted"] >= 50  # hinted oversampled up to match original

    def test_unrecognized_code_type_raises(self) -> None:
        records = _make_mixed_records(num_pos=10, num_neg=10, code_type="original")
        config = correctness_types.RecordResamplingConfig(
            code_type_proportions={"original": 1.0, "hinte": 1.0},
            seed=42,
        )
        with pytest.raises(ValueError, match="unrecognized code type"):
            correctness_resampling.resample_records(records, config)

    def test_compound_code_type_accepted(self) -> None:
        records = _make_mixed_records(num_pos=10, num_neg=10, code_type="original") + _make_mixed_records(
            num_pos=10, num_neg=10, code_type="bugged_hinted"
        )
        config = correctness_types.RecordResamplingConfig(
            code_type_proportions={"original": 1.0, "bugged_hinted": 1.0},
            seed=42,
        )
        result = correctness_resampling.resample_records(records, config)
        code_types = {rec.code_type for rec in result}
        assert code_types == {"original", "bugged_hinted"}

    def test_proportions_too_small_raises(self) -> None:
        records = _make_mixed_records(num_pos=1, num_neg=1, code_type="original") + _make_mixed_records(
            num_pos=50, num_neg=50, code_type="hinted"
        )
        config = correctness_types.RecordResamplingConfig(
            code_type_proportions={"original": 0.9, "hinted": 0.1},
            seed=42,
        )
        with pytest.raises(ValueError, match="fell below 1"):
            correctness_resampling.resample_records(records, config)


class TestResampleRecordsCombined:
    def test_code_type_before_label_ratio(self) -> None:
        records = _make_mixed_records(num_pos=40, num_neg=10, code_type="original") + _make_mixed_records(
            num_pos=10, num_neg=40, code_type="misleading"
        )
        config = correctness_types.RecordResamplingConfig(
            code_type_proportions={"original": 1.0},
            target_positive_ratio=0.5,
            seed=42,
        )
        result = correctness_resampling.resample_records(records, config)
        # only original code type should survive
        assert all(rec.code_type == "original" for rec in result)
        # label ratio should be ~50%
        num_pos = sum(1 for rec in result if rec.label)
        actual_ratio = num_pos / len(result)
        assert abs(actual_ratio - 0.5) < 0.1

    def test_max_records_cap_applied_last(self) -> None:
        records = _make_mixed_records(num_pos=50, num_neg=50, code_type="original")
        config = correctness_types.RecordResamplingConfig(max_records=20, seed=42)
        result = correctness_resampling.resample_records(records, config)
        assert len(result) == 20

    def test_min_records_per_label_violation_raises(self) -> None:
        records = _make_mixed_records(num_pos=50, num_neg=1, code_type="original")
        config = correctness_types.RecordResamplingConfig(max_records=100, min_records_per_label=2)
        with pytest.raises(ValueError, match="min_records_per_label"):
            correctness_resampling.resample_records(records, config)

    def test_size_cap_preserves_minority_class(self) -> None:
        records = _make_mixed_records(num_pos=2, num_neg=100, code_type="original")
        config = correctness_types.RecordResamplingConfig(max_records=10, min_records_per_label=2, seed=42)
        result = correctness_resampling.resample_records(records, config)
        assert len(result) == 10
        num_pos = sum(1 for rec in result if rec.label)
        assert num_pos >= 2  # minority preserved despite tiny fraction

    def test_output_is_shuffled(self) -> None:
        records = _make_mixed_records(num_pos=30, num_neg=30, code_type="original") + _make_mixed_records(
            num_pos=30, num_neg=30, code_type="hinted"
        )
        config = correctness_types.RecordResamplingConfig(
            code_type_proportions={"original": 1.0, "hinted": 1.0},
            target_positive_ratio=0.5,
            seed=42,
        )
        result = correctness_resampling.resample_records(records, config)
        # if output were grouped, all code_types in the first half would be the same
        first_half_types = {rec.code_type for rec in result[: len(result) // 2]}
        assert len(first_half_types) > 1

    def test_max_records_too_small_raises(self) -> None:
        records = _make_mixed_records(num_pos=10, num_neg=10, code_type="original")
        config = correctness_types.RecordResamplingConfig(max_records=3)
        with pytest.raises(ValueError, match="max_records must be at least"):
            correctness_resampling.resample_records(records, config)

    def test_seed_reproducibility(self) -> None:
        records = _make_mixed_records(num_pos=30, num_neg=30, code_type="original")
        config = correctness_types.RecordResamplingConfig(target_positive_ratio=0.3, seed=42)
        result_a = correctness_resampling.resample_records(records, config)
        result_b = correctness_resampling.resample_records(records, config)
        assert [rec.sample_id for rec in result_a] == [rec.sample_id for rec in result_b]

    def test_different_seeds_different_output(self) -> None:
        records = _make_mixed_records(num_pos=30, num_neg=30, code_type="original")
        config_a = correctness_types.RecordResamplingConfig(target_positive_ratio=0.3, seed=42)
        config_b = correctness_types.RecordResamplingConfig(target_positive_ratio=0.3, seed=99)
        result_a = correctness_resampling.resample_records(records, config_a)
        result_b = correctness_resampling.resample_records(records, config_b)
        ids_a = [rec.sample_id for rec in result_a]
        ids_b = [rec.sample_id for rec in result_b]
        assert ids_a != ids_b


class TestResampleRecordsEdgeCases:
    def test_empty_input_raises(self) -> None:
        config = correctness_types.RecordResamplingConfig()
        with pytest.raises(ValueError, match="empty"):
            correctness_resampling.resample_records([], config)

    def test_noop_config_returns_all_records(self) -> None:
        records = _make_mixed_records(num_pos=5, num_neg=5, code_type="original")
        config = correctness_types.RecordResamplingConfig()
        result = correctness_resampling.resample_records(records, config)
        assert len(result) == len(records)
        assert {rec.sample_id for rec in result} == {rec.sample_id for rec in records}

    def test_noop_config_preserves_order(self) -> None:
        records = _make_mixed_records(num_pos=10, num_neg=10, code_type="original")
        config = correctness_types.RecordResamplingConfig()
        result = correctness_resampling.resample_records(records, config)
        assert [rec.sample_id for rec in result] == [rec.sample_id for rec in records]

    def test_small_dataset_does_not_crash(self) -> None:
        records = _make_mixed_records(num_pos=3, num_neg=2, code_type="original")
        config = correctness_types.RecordResamplingConfig(min_records_per_label=1)
        result = correctness_resampling.resample_records(records, config)
        assert len(result) == 5
