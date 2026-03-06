"""End-to-end round-trip tests for code exec LMDB and pickle persistence."""

from __future__ import annotations

import asyncio

import pytest

import pyine.data.utils.lmdb_io
import pyine.evals.code_exec.reeval
import pyine.evals.code_exec.utils as code_exec_utils
import pyine.evals.persistence
import tests.evals.integration.conftest as integration_conftest
import tests.evals.integration.fake_models as fake_models

pytestmark = pytest.mark.integration

_NUM_SAMPLES = integration_conftest._NUM_TACO_SAMPLES
_NUM_ATTEMPTS = integration_conftest._NUM_ATTEMPTS
_TOTAL_ATTEMPTS = _NUM_SAMPLES * _NUM_ATTEMPTS  # 60


# ---------------------------------------------------------------------------
# LMDB round-trip (happy path)
# ---------------------------------------------------------------------------


class TestCodeExecLmdbRoundTrip:
    def test_lmdb_export_is_readable(
        self,
        oracle_code_exec_output: integration_conftest.CodeExecPipelineOutput,
    ) -> None:
        lmdb_path = oracle_code_exec_output.lmdb_path
        assert lmdb_path.exists()
        with pyine.data.utils.lmdb_io.LMDBReader(lmdb_path) as reader:
            assert reader.sample_count > 0
            metadata = reader.get_metadata()
        assert metadata.get("record_type") == "benchmark"
        assert metadata.get("aggregated_metrics") is not None

    def test_reconstruct_metrics_match_original(
        self,
        oracle_code_exec_output: integration_conftest.CodeExecPipelineOutput,
    ) -> None:
        original = oracle_code_exec_output.result
        reconstructed = pyine.evals.code_exec.reeval.reconstruct_from_lmdb(
            [oracle_code_exec_output.lmdb_path],
            eval_subset_name="test",
        )
        for key in ("accuracy_hard", "accuracy_soft", "sample_count", "attempt_count"):
            assert reconstructed.metrics[key] == pytest.approx(original.metrics[key]), f"mismatch on {key}"
        for key in original.metrics:
            if key.startswith("total_token_usage/"):
                assert reconstructed.metrics[key] == pytest.approx(
                    original.metrics[key],
                    nan_ok=True,
                ), f"mismatch on {key}"

    def test_reconstruct_artifact_count_matches(
        self,
        oracle_code_exec_output: integration_conftest.CodeExecPipelineOutput,
    ) -> None:
        reconstructed = pyine.evals.code_exec.reeval.reconstruct_from_lmdb(
            [oracle_code_exec_output.lmdb_path],
            eval_subset_name="test",
        )
        assert len(reconstructed.artifacts) == _TOTAL_ATTEMPTS

    def test_reconstruct_categories_match_original(
        self,
        oracle_code_exec_output: integration_conftest.CodeExecPipelineOutput,
    ) -> None:
        original = oracle_code_exec_output.result
        assert original.category_to_identifiers, "expected non-empty categories from default extraction config"
        reconstructed = pyine.evals.code_exec.reeval.reconstruct_from_lmdb(
            [oracle_code_exec_output.lmdb_path],
            eval_subset_name="test",
        )
        assert set(reconstructed.category_to_identifiers.keys()) == set(original.category_to_identifiers.keys())
        for key in original.category_to_identifiers:
            assert sorted(reconstructed.category_to_identifiers[key]) == sorted(original.category_to_identifiers[key])

    def test_reevaluate_metrics_match_original(
        self,
        oracle_code_exec_output: integration_conftest.CodeExecPipelineOutput,
    ) -> None:
        original = oracle_code_exec_output.result
        reevaluated = asyncio.run(
            pyine.evals.code_exec.reeval.reevaluate_from_lmdb(
                [oracle_code_exec_output.lmdb_path],
                eval_subset_name="test",
            )
        )
        assert reevaluated.metrics["accuracy_hard"] == pytest.approx(original.metrics["accuracy_hard"])
        assert reevaluated.metrics["accuracy_soft"] == pytest.approx(original.metrics["accuracy_soft"])
        assert reevaluated.metrics["sample_count"] == original.metrics["sample_count"]

    def test_reconstruct_and_reevaluate_agree(
        self,
        oracle_code_exec_output: integration_conftest.CodeExecPipelineOutput,
    ) -> None:
        lmdb_paths = [oracle_code_exec_output.lmdb_path]
        reconstructed = pyine.evals.code_exec.reeval.reconstruct_from_lmdb(
            lmdb_paths,
            eval_subset_name="test",
        )
        reevaluated = asyncio.run(
            pyine.evals.code_exec.reeval.reevaluate_from_lmdb(
                lmdb_paths,
                eval_subset_name="test",
            )
        )
        assert reconstructed.metrics["accuracy_hard"] == pytest.approx(reevaluated.metrics["accuracy_hard"])
        assert reconstructed.metrics["accuracy_soft"] == pytest.approx(reevaluated.metrics["accuracy_soft"])
        assert reconstructed.metrics["sample_count"] == reevaluated.metrics["sample_count"]

    def test_known_oracle_metrics(
        self,
        oracle_code_exec_output: integration_conftest.CodeExecPipelineOutput,
    ) -> None:
        metrics = oracle_code_exec_output.result.metrics
        assert metrics["accuracy_hard"] == pytest.approx(1.0)
        assert metrics["sample_count"] == _NUM_SAMPLES
        assert metrics["total_token_usage/total_tokens"] == _TOTAL_ATTEMPTS * 15


# ---------------------------------------------------------------------------
# LMDB error paths (E2E-seam only)
# ---------------------------------------------------------------------------


class TestCodeExecLmdbErrorPaths:
    def test_reconstruct_without_stored_metrics_falls_back_to_reeval(self) -> None:
        """Pipeline with store_aggregated_metrics=False: reconstruct raises, reeval succeeds."""
        samples = integration_conftest.make_taco_sample_data()
        output = integration_conftest._run_code_exec_pipeline(
            fake_models.OracleCodeExecChain(),
            samples,
            enable_pickle=False,
            store_aggregated_metrics=False,
        )
        with pytest.raises(ValueError, match="aggregated_metrics"):
            pyine.evals.code_exec.reeval.reconstruct_from_lmdb(
                [output.lmdb_path],
                eval_subset_name="test",
            )
        reevaluated = asyncio.run(
            pyine.evals.code_exec.reeval.reevaluate_from_lmdb(
                [output.lmdb_path],
                eval_subset_name="test",
            )
        )
        assert reevaluated.metrics["accuracy_hard"] == pytest.approx(1.0)
        assert reevaluated.metrics["sample_count"] == _NUM_SAMPLES


# ---------------------------------------------------------------------------
# Pickle round-trip (happy path)
# ---------------------------------------------------------------------------


class TestCodeExecPickleRoundTrip:
    def test_pickle_file_exists(
        self,
        oracle_code_exec_output: integration_conftest.CodeExecPipelineOutput,
    ) -> None:
        assert oracle_code_exec_output.pickle_path is not None
        assert oracle_code_exec_output.pickle_path.exists()

    def test_pickle_load_matches_original_metrics(
        self,
        oracle_code_exec_output: integration_conftest.CodeExecPipelineOutput,
    ) -> None:
        assert oracle_code_exec_output.pickle_path is not None
        loaded = pyine.evals.persistence.load_eval_result(oracle_code_exec_output.pickle_path)
        original = oracle_code_exec_output.result
        for key in ("accuracy_hard", "accuracy_soft", "sample_count", "attempt_count"):
            assert loaded.metrics[key] == pytest.approx(original.metrics[key]), f"mismatch on {key}"

    def test_pickle_artifact_count_matches(
        self,
        oracle_code_exec_output: integration_conftest.CodeExecPipelineOutput,
    ) -> None:
        assert oracle_code_exec_output.pickle_path is not None
        loaded = pyine.evals.persistence.load_eval_result(
            oracle_code_exec_output.pickle_path,
            expected_type=code_exec_utils.CodeExecEvalResult,
        )
        assert len(loaded.artifacts) == _TOTAL_ATTEMPTS

    def test_pickle_type_validation(
        self,
        oracle_code_exec_output: integration_conftest.CodeExecPipelineOutput,
    ) -> None:
        assert oracle_code_exec_output.pickle_path is not None
        loaded = pyine.evals.persistence.load_eval_result(
            oracle_code_exec_output.pickle_path,
            expected_type=code_exec_utils.CodeExecEvalResult,
        )
        assert isinstance(loaded, code_exec_utils.CodeExecEvalResult)


# ---------------------------------------------------------------------------
# Pickle error paths (E2E-seam)
# ---------------------------------------------------------------------------


class TestCodeExecPickleErrorPaths:
    def test_pickle_wrong_type_raises(
        self,
        oracle_code_exec_output: integration_conftest.CodeExecPipelineOutput,
    ) -> None:
        import pyine.evals.correctness._impl as correctness_impl

        assert oracle_code_exec_output.pickle_path is not None
        with pytest.raises(TypeError, match="CodeExecEvalResult.*CorrectnessEvalResult"):
            pyine.evals.persistence.load_eval_result(
                oracle_code_exec_output.pickle_path,
                expected_type=correctness_impl.CorrectnessEvalResult,
            )
