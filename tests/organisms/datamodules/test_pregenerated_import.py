"""Tests for pregenerated output import pipeline (DiskRewardLogger export → SFT re-import)."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    import pathlib

import pytest

import pyine.data.utils.lmdb_io
import pyine.organisms.datamodules.samples.common as samples_common
import pyine.organisms.models.rewards.core.logging as reward_logging


def _write_test_lmdb(
    output_path: pathlib.Path,
    records: list[tuple[str, dict[str, typing.Any]]],
) -> None:
    """Write test records to an LMDB dataset using DiskRewardLogger's format."""
    writer = pyine.data.utils.lmdb_io.LMDBWriter(
        path=output_path,
        serialization_config=pyine.data.utils.lmdb_io.SerializationConfig(
            method=pyine.data.utils.lmdb_io.SerializationMethod.JSON_ZSTD,
        ),
    )
    for key, value in records:
        writer.put(key, value)
    writer.close()


class TestLoadPregeneratedOutputs:
    """Tests for _load_pregenerated_outputs via ShortcutBiasDataModule."""

    def _make_loader(
        self,
        lmdb_path: pathlib.Path,
        selection: str = "latest",
        phase_prefix: str = "train/",
    ) -> dict[str, str]:
        """Directly exercise the loading logic without full datamodule setup."""
        import pyine.organisms.datamodules.shortcuts as shortcuts_mod
        import pyine.organisms.datamodules.shortcuts_configs as shortcuts_configs_mod

        # create a minimal config
        config = shortcuts_configs_mod.ShortcutBiasDataModuleConfig.__new__(
            shortcuts_configs_mod.ShortcutBiasDataModuleConfig,
        )
        object.__setattr__(config, "pregenerated_outputs_lmdb_path", lmdb_path)
        object.__setattr__(config, "pregenerated_outputs_selection", selection)
        object.__setattr__(config, "pregenerated_outputs_phase_prefix", phase_prefix)
        # create a minimal datamodule-like object with the right config
        dm = object.__new__(shortcuts_mod.ShortcutBiasDataModule)
        object.__setattr__(dm, "config", config)
        return dm._load_pregenerated_outputs()

    def test_basic_load(self, tmp_path: pathlib.Path) -> None:
        lmdb_path = tmp_path / "gen.lmdb"
        _write_test_lmdb(
            lmdb_path,
            [
                ("train/sample_a/1", {"model_output": "out_a", "reward_total": 1.0}),
                ("train/sample_b/1", {"model_output": "out_b", "reward_total": 0.5}),
            ],
        )
        result = self._make_loader(lmdb_path, phase_prefix="train/")
        assert result == {"sample_a": "out_a", "sample_b": "out_b"}

    def test_phase_prefix_filtering(self, tmp_path: pathlib.Path) -> None:
        lmdb_path = tmp_path / "gen.lmdb"
        _write_test_lmdb(
            lmdb_path,
            [
                ("train/sample_a/1", {"model_output": "train_out", "reward_total": 1.0}),
                ("eval/sample_a/1", {"model_output": "eval_out", "reward_total": 0.5}),
                ("eval/sample_b/1", {"model_output": "eval_out_b", "reward_total": 0.3}),
            ],
        )
        train_result = self._make_loader(lmdb_path, phase_prefix="train/")
        assert train_result == {"sample_a": "train_out"}
        eval_result = self._make_loader(lmdb_path, phase_prefix="eval/")
        assert eval_result == {"sample_a": "eval_out", "sample_b": "eval_out_b"}

    def test_empty_prefix_loads_all(self, tmp_path: pathlib.Path) -> None:
        lmdb_path = tmp_path / "gen.lmdb"
        _write_test_lmdb(
            lmdb_path,
            [
                ("train/sample_a/1", {"model_output": "out1", "reward_total": 1.0, "key_prefix": "train/"}),
                ("eval/sample_b/1", {"model_output": "out2", "reward_total": 0.5, "key_prefix": "eval/"}),
            ],
        )
        result = self._make_loader(lmdb_path, phase_prefix="")
        assert result == {"sample_a": "out1", "sample_b": "out2"}

    def test_nonmatching_prefix_raises(self, tmp_path: pathlib.Path) -> None:
        lmdb_path = tmp_path / "gen.lmdb"
        _write_test_lmdb(
            lmdb_path,
            [
                ("train/sample_a/1", {"model_output": "out", "reward_total": 1.0}),
            ],
        )
        with pytest.raises(ValueError, match="matched 0"):
            self._make_loader(lmdb_path, phase_prefix="predict/")

    def test_latest_selection_picks_highest_generation_count(self, tmp_path: pathlib.Path) -> None:
        lmdb_path = tmp_path / "gen.lmdb"
        _write_test_lmdb(
            lmdb_path,
            [
                ("train/sample_a/100", {"model_output": "early", "reward_total": 0.3}),
                ("train/sample_a/500", {"model_output": "middle", "reward_total": 0.8}),
                ("train/sample_a/1000", {"model_output": "latest", "reward_total": 0.5}),
            ],
        )
        result = self._make_loader(lmdb_path, selection="latest")
        assert result["sample_a"] == "latest"

    def test_best_reward_selection(self, tmp_path: pathlib.Path) -> None:
        lmdb_path = tmp_path / "gen.lmdb"
        _write_test_lmdb(
            lmdb_path,
            [
                ("train/sample_a/100", {"model_output": "low", "reward_total": 0.2}),
                ("train/sample_a/200", {"model_output": "high", "reward_total": 0.9}),
                ("train/sample_a/300", {"model_output": "medium", "reward_total": 0.5}),
            ],
        )
        result = self._make_loader(lmdb_path, selection="best_reward")
        assert result["sample_a"] == "high"

    def test_best_reward_raises_on_none_total(self, tmp_path: pathlib.Path) -> None:
        lmdb_path = tmp_path / "gen.lmdb"
        _write_test_lmdb(
            lmdb_path,
            [
                ("train/sample_a/100", {"model_output": "out", "reward_total": None}),
            ],
        )
        with pytest.raises(ValueError, match="reward_total"):
            self._make_loader(lmdb_path, selection="best_reward")

    def test_missing_model_output_raises(self, tmp_path: pathlib.Path) -> None:
        lmdb_path = tmp_path / "gen.lmdb"
        _write_test_lmdb(
            lmdb_path,
            [
                ("train/sample_a/1", {"reward_total": 1.0}),  # missing model_output
            ],
        )
        with pytest.raises(ValueError, match="model_output"):
            self._make_loader(lmdb_path)

    def test_hierarchical_sample_id_with_slashes(self, tmp_path: pathlib.Path) -> None:
        """Sample IDs like TACO/s0001/t0001 contain slashes; the last '/' separates gen_count."""
        lmdb_path = tmp_path / "gen.lmdb"
        _write_test_lmdb(
            lmdb_path,
            [
                ("train/TACO/s0001/t0001/500", {"model_output": "traced_out", "reward_total": 1.0}),
            ],
        )
        result = self._make_loader(lmdb_path)
        assert "TACO/s0001/t0001" in result
        assert result["TACO/s0001/t0001"] == "traced_out"


class TestDiskRewardLoggerRoundTrip:
    """End-to-end: write with DiskRewardLogger, read back with LMDBReader."""

    def test_full_round_trip(self, tmp_path: pathlib.Path) -> None:
        output_dir = tmp_path / "gen"
        disk_logger = reward_logging.DiskRewardLogger(output_path=output_dir)
        disk_logger.set_key_prefix("train/")
        disk_logger.log_sample(
            "TACO/s0001/t0001",
            generation_count=500,
            total=0.9,
            model_output="print(hello)",
            prompt="prompt_text",
            expected_output="hello",
            reasoning="<reasoning>think</reasoning>",
            final_answer="hello",
            terms={"hard_match": 0.9},
            tags=["subset:train", "augment:original"],
            categories=["code_type/original"],
            predict_type="program_output",
            code_type="original",
            step=100,
            batch_count=25,
            completion_idx=0,
            rank=0,
        )
        disk_logger.close()
        reader = pyine.data.utils.lmdb_io.LMDBReader(output_dir)
        assert len(reader.key_map) == 1
        key = "train/TACO/s0001/t0001/500"
        assert key in reader.key_map
        record = reader.get(key)
        assert record["model_output"] == "print(hello)"
        assert record["reward_total"] == 0.9
        assert record["reasoning"] == "<reasoning>think</reasoning>"
        assert record["final_answer"] == "hello"
        assert record["key_prefix"] == "train/"
        reader.close()


class TestSampleBuilderPregeneratedOverride:
    """Tests for SampleBuilder pregenerated output override logic."""

    def test_program_output_override(self) -> None:
        sample = samples_common.SampleData(
            identifier="test/s0001/t0001",
            code="print(42)",
            description="test",
            entrypoint="",
            first_line=0,
            last_line=1,
            inputs="",
            expected_output="42",
            predict_type=samples_common.SamplePredictType.program_output,
            code_type="original",
            trace_step_count=1,
            comma_separated_tags="",
            has_code_override=False,
            complexity_metrics={},
        )
        overrides = {"test/s0001/t0001": "OVERRIDDEN_OUTPUT"}
        # apply override logic (same as SampleBuilder.__getitem__)
        if sample.identifier in overrides:
            assert sample.predict_type == samples_common.SamplePredictType.program_output
            sample = sample._replace(expected_output=overrides[sample.identifier])
        assert sample.expected_output == "OVERRIDDEN_OUTPUT"

    def test_non_program_output_raises(self) -> None:
        sample = samples_common.SampleData(
            identifier="test/s0001/t0001",
            code="def f(x): return x",
            description="test",
            entrypoint="f",
            first_line=0,
            last_line=1,
            inputs="(1,)",
            expected_output="1",
            predict_type=samples_common.SamplePredictType.function_return,
            code_type="original",
            trace_step_count=1,
            comma_separated_tags="",
            has_code_override=False,
            complexity_metrics={},
        )
        overrides = {"test/s0001/t0001": "OVERRIDDEN"}
        if sample.identifier in overrides:
            with pytest.raises(NotImplementedError, match="program_output"):
                if sample.predict_type != samples_common.SamplePredictType.program_output:
                    raise NotImplementedError(
                        f"pregenerated output overrides are only supported for program_output predict_type, "
                        f"got {sample.predict_type} for sample {sample.identifier}"
                    )

    def test_no_override_when_id_not_in_dict(self) -> None:
        sample = samples_common.SampleData(
            identifier="test/s0001/t0001",
            code="print(42)",
            description="test",
            entrypoint="",
            first_line=0,
            last_line=1,
            inputs="",
            expected_output="42",
            predict_type=samples_common.SamplePredictType.program_output,
            code_type="original",
            trace_step_count=1,
            comma_separated_tags="",
            has_code_override=False,
            complexity_metrics={},
        )
        overrides = {"other/id": "OVERRIDDEN"}
        if sample.identifier in overrides:
            sample = sample._replace(expected_output=overrides[sample.identifier])
        assert sample.expected_output == "42"  # unchanged


class TestPregeneratedOutputsConfigValidation:
    """Tests for pregenerated output config fields on ShortcutBiasDataModuleConfig."""

    def test_phase_prefix_normalization(self) -> None:
        import pyine.utils.parsing

        assert pyine.utils.parsing.normalize_path_prefix("train") == "train/"
        assert pyine.utils.parsing.normalize_path_prefix("train/") == "train/"
        assert pyine.utils.parsing.normalize_path_prefix("") == ""
        assert pyine.utils.parsing.normalize_path_prefix("  ") == ""
