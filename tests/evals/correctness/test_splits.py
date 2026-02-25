"""Tests for pyine.evals.correctness.splits."""

from __future__ import annotations

import typing

import pytest

import pyine.evals.correctness.splits as correctness_splits

if typing.TYPE_CHECKING:
    import pytest_mock
import pyine.evals.correctness.types as correctness_types


def _make_record(
    sample_id: str,
    problem_id: str,
    attempt_index: int = 0,
    label: bool = True,
) -> correctness_types.EvalRecord:
    return correctness_types.EvalRecord(
        sample_id=sample_id,
        problem_id=problem_id,
        attempt_index=attempt_index,
        model_output="output",
        final_answer=None,
        expected_output="expected",
        label=label,
        code_type="original",
        tags=[],
        record={},
        difficulty_score=None,
    )


class TestExtractProblemId:
    def test_trace_level_id(self) -> None:
        result = correctness_splits.extract_problem_id("TACO/TRAIN/p000001/s0000/t0000")
        assert result == "TACO/TRAIN/p000001"

    def test_augmented_trace_id(self) -> None:
        result = correctness_splits.extract_problem_id("TACO/TRAIN/p000001/s0000/t0000/a:hints_docs:000")
        assert result == "TACO/TRAIN/p000001"

    def test_with_suffix(self) -> None:
        result = correctness_splits.extract_problem_id("TACO/TRAIN/p000001/s0000/t0000::hinted")
        assert result == "TACO/TRAIN/p000001"

    def test_problem_level_fallback(self) -> None:
        result = correctness_splits.extract_problem_id("TACO/TRAIN/p000001")
        assert result == "TACO/TRAIN/p000001"

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            correctness_splits.extract_problem_id("invalid_id")

    def test_different_problem_indices(self) -> None:
        result1 = correctness_splits.extract_problem_id("TACO/TRAIN/p000001/s0000/t0000")
        result2 = correctness_splits.extract_problem_id("TACO/TRAIN/p000002/s0001/t0000")
        assert result1 != result2
        assert result1 == "TACO/TRAIN/p000001"
        assert result2 == "TACO/TRAIN/p000002"


class TestBuildGuardrailSplits:
    """Tests for build_guardrail_splits using mocked SplitResult."""

    def _make_split_result_and_records(
        self,
    ) -> tuple[list[correctness_types.EvalRecord], dict[str, list[str]]]:
        """Create test records and expected subset assignments.

        Returns records for 6 problems: 2 train, 2 valid, 2 test.
        """
        records = []
        # train problems (will be discarded)
        for pid_idx in [1, 2]:
            pid = f"TACO/TRAIN/p{pid_idx:06d}"
            for attempt in range(2):
                records.append(
                    _make_record(
                        sample_id=f"TACO/TRAIN/p{pid_idx:06d}/s0000/t0000",
                        problem_id=pid,
                        attempt_index=attempt,
                        label=(attempt == 0),
                    )
                )
        # valid problems (will be re-split)
        for pid_idx in [3, 4]:
            pid = f"TACO/TRAIN/p{pid_idx:06d}"
            for attempt in range(2):
                records.append(
                    _make_record(
                        sample_id=f"TACO/TRAIN/p{pid_idx:06d}/s0000/t0000",
                        problem_id=pid,
                        attempt_index=attempt,
                        label=(attempt == 0),
                    )
                )
        # test problems
        for pid_idx in [5, 6]:
            pid = f"TACO/TRAIN/p{pid_idx:06d}"
            for attempt in range(2):
                records.append(
                    _make_record(
                        sample_id=f"TACO/TRAIN/p{pid_idx:06d}/s0000/t0000",
                        problem_id=pid,
                        attempt_index=attempt,
                        label=(attempt == 0),
                    )
                )
        subset_map = {
            "train": [f"TACO/TRAIN/p{idx:06d}" for idx in [1, 2]],
            "valid": [f"TACO/TRAIN/p{idx:06d}" for idx in [3, 4]],
            "test": [f"TACO/TRAIN/p{idx:06d}" for idx in [5, 6]],
        }
        return records, subset_map

    def test_train_problems_discarded(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        records, subset_map = self._make_split_result_and_records()
        mock_split_result = mocker.MagicMock()
        mock_split_result.get_subset_to_ids_map.return_value = subset_map
        mock_split_result.source_dataset_name = "TACO"
        mocker.patch(
            "pyine.evals.correctness.splits.pyine.data.utils.splits.get_dataset_split_result",
            return_value=mock_split_result,
        )
        import pyine.evals.correctness.configs as correctness_configs

        config = correctness_configs.GuardrailSplitConfig(
            split_source="TACO",
            guardrail_valid_fraction=0.5,
            seed=42,
        )
        splits = correctness_splits.build_guardrail_splits(records, config)
        # train problems should not be in any split
        train_pids = {f"TACO/TRAIN/p{idx:06d}" for idx in [1, 2]}
        all_split_pids = splits.train_problem_ids | splits.valid_problem_ids | splits.test_problem_ids
        assert train_pids.isdisjoint(all_split_pids)

    def test_test_problems_passthrough(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        records, subset_map = self._make_split_result_and_records()
        mock_split_result = mocker.MagicMock()
        mock_split_result.get_subset_to_ids_map.return_value = subset_map
        mock_split_result.source_dataset_name = "TACO"
        mocker.patch(
            "pyine.evals.correctness.splits.pyine.data.utils.splits.get_dataset_split_result",
            return_value=mock_split_result,
        )
        import pyine.evals.correctness.configs as correctness_configs

        config = correctness_configs.GuardrailSplitConfig(
            split_source="TACO",
            guardrail_valid_fraction=0.5,
            seed=42,
        )
        splits = correctness_splits.build_guardrail_splits(records, config)
        test_pids = {f"TACO/TRAIN/p{idx:06d}" for idx in [5, 6]}
        assert splits.test_problem_ids == frozenset(test_pids)
        assert len(splits.guardrail_test) == 4  # 2 problems * 2 attempts

    def test_valid_resplit_determinism(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        records, subset_map = self._make_split_result_and_records()
        mock_split_result = mocker.MagicMock()
        mock_split_result.get_subset_to_ids_map.return_value = subset_map
        mock_split_result.source_dataset_name = "TACO"
        mocker.patch(
            "pyine.evals.correctness.splits.pyine.data.utils.splits.get_dataset_split_result",
            return_value=mock_split_result,
        )
        import pyine.evals.correctness.configs as correctness_configs

        config = correctness_configs.GuardrailSplitConfig(
            split_source="TACO",
            guardrail_valid_fraction=0.5,
            seed=42,
        )
        splits1 = correctness_splits.build_guardrail_splits(records, config)
        splits2 = correctness_splits.build_guardrail_splits(records, config)
        assert splits1.train_problem_ids == splits2.train_problem_ids
        assert splits1.valid_problem_ids == splits2.valid_problem_ids

    def test_problem_integrity(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """All records for a problem must stay in the same split."""
        records, subset_map = self._make_split_result_and_records()
        mock_split_result = mocker.MagicMock()
        mock_split_result.get_subset_to_ids_map.return_value = subset_map
        mock_split_result.source_dataset_name = "TACO"
        mocker.patch(
            "pyine.evals.correctness.splits.pyine.data.utils.splits.get_dataset_split_result",
            return_value=mock_split_result,
        )
        import pyine.evals.correctness.configs as correctness_configs

        config = correctness_configs.GuardrailSplitConfig(
            split_source="TACO",
            guardrail_valid_fraction=0.5,
            seed=42,
        )
        splits = correctness_splits.build_guardrail_splits(records, config)
        # check that each problem's records are all in the same split
        for split_records in [splits.guardrail_train, splits.guardrail_valid, splits.guardrail_test]:
            problem_ids_in_split = {rec.problem_id for rec in split_records}
            for pid in problem_ids_in_split:
                pid_records = [rec for rec in split_records if rec.problem_id == pid]
                assert len(pid_records) == 2  # all 2 attempts

    def test_dataset_mismatch_raises(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Records from a different dataset than the split file should raise."""
        records = [
            _make_record(
                sample_id="OTHER/TRAIN/p000001/s0000/t0000",
                problem_id="OTHER/TRAIN/p000001",
            )
        ]
        mock_split_result = mocker.MagicMock()
        mock_split_result.get_subset_to_ids_map.return_value = {
            "train": [],
            "valid": [],
            "test": ["TACO/TRAIN/p000001"],
        }
        mock_split_result.source_dataset_name = "TACO"
        mocker.patch(
            "pyine.evals.correctness.splits.pyine.data.utils.splits.get_dataset_split_result",
            return_value=mock_split_result,
        )
        import pyine.evals.correctness.configs as correctness_configs

        config = correctness_configs.GuardrailSplitConfig(
            split_source="TACO",
            guardrail_valid_fraction=0.5,
            seed=42,
        )
        with pytest.raises(ValueError, match="dataset 'OTHER'.*source_dataset_name='TACO'"):
            correctness_splits.build_guardrail_splits(records, config)

    def test_unknown_subset_raises(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Records in an unknown subset (not train/valid/test) should raise."""
        records = [
            _make_record(
                sample_id="TACO/TRAIN/p000001/s0000/t0000",
                problem_id="TACO/TRAIN/p000001",
            )
        ]
        mock_split_result = mocker.MagicMock()
        mock_split_result.get_subset_to_ids_map.return_value = {
            "train": [],
            "valid": [],
            "test": [],
            "evaluation": ["TACO/TRAIN/p000001"],
        }
        mock_split_result.source_dataset_name = "TACO"
        mocker.patch(
            "pyine.evals.correctness.splits.pyine.data.utils.splits.get_dataset_split_result",
            return_value=mock_split_result,
        )
        import pyine.evals.correctness.configs as correctness_configs

        config = correctness_configs.GuardrailSplitConfig(
            split_source="TACO",
            guardrail_valid_fraction=0.5,
            seed=42,
        )
        with pytest.raises(ValueError, match="unknown original subset 'evaluation'"):
            correctness_splits.build_guardrail_splits(records, config)

    def test_duplicate_problem_across_subsets_raises(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """A problem appearing in both valid and test should raise."""
        records = [
            _make_record(
                sample_id="TACO/TRAIN/p000001/s0000/t0000",
                problem_id="TACO/TRAIN/p000001",
            )
        ]
        mock_split_result = mocker.MagicMock()
        mock_split_result.get_subset_to_ids_map.return_value = {
            "train": [],
            "valid": ["TACO/TRAIN/p000001"],
            "test": ["TACO/TRAIN/p000001"],
        }
        mock_split_result.source_dataset_name = "TACO"
        mocker.patch(
            "pyine.evals.correctness.splits.pyine.data.utils.splits.get_dataset_split_result",
            return_value=mock_split_result,
        )
        import pyine.evals.correctness.configs as correctness_configs

        config = correctness_configs.GuardrailSplitConfig(
            split_source="TACO",
            guardrail_valid_fraction=0.5,
            seed=42,
        )
        with pytest.raises(ValueError, match="appears in both.*'valid'.*and.*'test'"):
            correctness_splits.build_guardrail_splits(records, config)

    def test_missing_problem_raises(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        records = [
            _make_record(
                sample_id="TACO/TRAIN/p999999/s0000/t0000",
                problem_id="TACO/TRAIN/p999999",
            )
        ]
        mock_split_result = mocker.MagicMock()
        mock_split_result.get_subset_to_ids_map.return_value = {"train": [], "valid": [], "test": []}
        mock_split_result.source_dataset_name = "TACO"
        mocker.patch(
            "pyine.evals.correctness.splits.pyine.data.utils.splits.get_dataset_split_result",
            return_value=mock_split_result,
        )
        import pyine.evals.correctness.configs as correctness_configs

        config = correctness_configs.GuardrailSplitConfig(
            split_source="TACO",
            guardrail_valid_fraction=0.5,
            seed=42,
        )
        with pytest.raises(ValueError, match="not found in SplitResult"):
            correctness_splits.build_guardrail_splits(records, config)
