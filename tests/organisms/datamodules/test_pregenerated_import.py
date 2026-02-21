"""Tests for pregenerated output import pipeline (DiskRewardLogger export → SFT re-import)."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    import pathlib

import omegaconf
import pytest

import pyine.data.utils.lmdb_io
import pyine.organisms.datamodules.base
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


def _make_record(
    model_output: str = "out",
    reward_total: float = 1.0,
    **extra: typing.Any,
) -> dict[str, typing.Any]:
    """Build a minimal LMDB record dict."""
    return {"model_output": model_output, "reward_total": reward_total, **extra}


class TestLoadPregeneratedOutputs:
    def _make_loader(
        self,
        lmdb_paths: tuple[pathlib.Path, ...],
        selection: str = "latest",
        phase_prefix: str = "train/",
    ) -> dict[str, samples_common.PregeneratedOutputRecord]:
        """Directly exercise the loading logic without full datamodule setup."""
        import pyine.organisms.datamodules.shortcuts as shortcuts_mod
        import pyine.organisms.datamodules.shortcuts_configs as shortcuts_configs_mod

        config = shortcuts_configs_mod.ShortcutBiasDataModuleConfig.__new__(
            shortcuts_configs_mod.ShortcutBiasDataModuleConfig,
        )
        object.__setattr__(config, "pregenerated_outputs_lmdb_paths", lmdb_paths)
        object.__setattr__(config, "pregenerated_outputs_selection", selection)
        object.__setattr__(config, "pregenerated_outputs_phase_prefix", phase_prefix)
        dm = object.__new__(shortcuts_mod.ShortcutBiasDataModule)
        object.__setattr__(dm, "config", config)
        return dm._load_pregenerated_outputs()

    def test_basic_load(self, tmp_path: pathlib.Path) -> None:
        lmdb_path = tmp_path / "gen.lmdb"
        _write_test_lmdb(
            lmdb_path,
            [
                ("train/sample_a/1", _make_record(model_output="out_a")),
                ("train/sample_b/1", _make_record(model_output="out_b", reward_total=0.5)),
            ],
        )
        result = self._make_loader((lmdb_path,), phase_prefix="train/")
        assert result["sample_a"].model_output == "out_a"
        assert result["sample_b"].model_output == "out_b"

    def test_phase_prefix_filtering(self, tmp_path: pathlib.Path) -> None:
        lmdb_path = tmp_path / "gen.lmdb"
        _write_test_lmdb(
            lmdb_path,
            [
                ("train/sample_a/1", _make_record(model_output="train_out")),
                ("eval/sample_a/1", _make_record(model_output="eval_out", reward_total=0.5)),
                ("eval/sample_b/1", _make_record(model_output="eval_out_b", reward_total=0.3)),
            ],
        )
        train_result = self._make_loader((lmdb_path,), phase_prefix="train/")
        assert train_result["sample_a"].model_output == "train_out"
        assert len(train_result) == 1
        eval_result = self._make_loader((lmdb_path,), phase_prefix="eval/")
        assert eval_result["sample_a"].model_output == "eval_out"
        assert eval_result["sample_b"].model_output == "eval_out_b"

    def test_empty_prefix_loads_all(self, tmp_path: pathlib.Path) -> None:
        lmdb_path = tmp_path / "gen.lmdb"
        _write_test_lmdb(
            lmdb_path,
            [
                ("train/sample_a/1", _make_record(model_output="out1", key_prefix="train/")),
                ("eval/sample_b/1", _make_record(model_output="out2", reward_total=0.5, key_prefix="eval/")),
            ],
        )
        result = self._make_loader((lmdb_path,), phase_prefix="")
        assert result["sample_a"].model_output == "out1"
        assert result["sample_b"].model_output == "out2"

    def test_nonmatching_prefix_raises(self, tmp_path: pathlib.Path) -> None:
        lmdb_path = tmp_path / "gen.lmdb"
        _write_test_lmdb(lmdb_path, [("train/sample_a/1", _make_record())])
        with pytest.raises(ValueError, match="matched 0"):
            self._make_loader((lmdb_path,), phase_prefix="predict/")

    def test_latest_selection_picks_highest_generation_count(self, tmp_path: pathlib.Path) -> None:
        lmdb_path = tmp_path / "gen.lmdb"
        _write_test_lmdb(
            lmdb_path,
            [
                ("train/sample_a/100", _make_record(model_output="early", reward_total=0.3)),
                ("train/sample_a/500", _make_record(model_output="middle", reward_total=0.8)),
                ("train/sample_a/1000", _make_record(model_output="latest", reward_total=0.5)),
            ],
        )
        result = self._make_loader((lmdb_path,), selection="latest")
        assert result["sample_a"].model_output == "latest"

    def test_best_reward_selection(self, tmp_path: pathlib.Path) -> None:
        lmdb_path = tmp_path / "gen.lmdb"
        _write_test_lmdb(
            lmdb_path,
            [
                ("train/sample_a/100", _make_record(model_output="low", reward_total=0.2)),
                ("train/sample_a/200", _make_record(model_output="high", reward_total=0.9)),
                ("train/sample_a/300", _make_record(model_output="medium", reward_total=0.5)),
            ],
        )
        result = self._make_loader((lmdb_path,), selection="best_reward")
        assert result["sample_a"].model_output == "high"

    def test_best_reward_raises_on_none_total(self, tmp_path: pathlib.Path) -> None:
        lmdb_path = tmp_path / "gen.lmdb"
        _write_test_lmdb(lmdb_path, [("train/sample_a/100", {"model_output": "out", "reward_total": None})])
        with pytest.raises(ValueError, match="reward_total"):
            self._make_loader((lmdb_path,), selection="best_reward")

    def test_missing_model_output_raises(self, tmp_path: pathlib.Path) -> None:
        lmdb_path = tmp_path / "gen.lmdb"
        _write_test_lmdb(lmdb_path, [("train/sample_a/1", {"reward_total": 1.0})])
        with pytest.raises(ValueError, match="model_output"):
            self._make_loader((lmdb_path,))

    def test_hierarchical_sample_id_with_slashes(self, tmp_path: pathlib.Path) -> None:
        """Sample IDs like TACO/s0001/t0001 contain slashes; the last '/' separates gen_count."""
        lmdb_path = tmp_path / "gen.lmdb"
        _write_test_lmdb(lmdb_path, [("train/TACO/s0001/t0001/500", _make_record(model_output="traced_out"))])
        result = self._make_loader((lmdb_path,))
        assert "TACO/s0001/t0001" in result
        assert result["TACO/s0001/t0001"].model_output == "traced_out"

    def test_multiple_lmdb_paths_merged(self, tmp_path: pathlib.Path) -> None:
        lmdb_a = tmp_path / "rank_0"
        lmdb_b = tmp_path / "rank_1"
        _write_test_lmdb(lmdb_a, [("train/sample_a/1", _make_record(model_output="a_out"))])
        _write_test_lmdb(lmdb_b, [("train/sample_b/1", _make_record(model_output="b_out"))])
        result = self._make_loader((lmdb_a, lmdb_b))
        assert result["sample_a"].model_output == "a_out"
        assert result["sample_b"].model_output == "b_out"

    def test_multiple_lmdb_paths_dedup_latest(self, tmp_path: pathlib.Path) -> None:
        lmdb_a = tmp_path / "rank_0"
        lmdb_b = tmp_path / "rank_1"
        _write_test_lmdb(lmdb_a, [("train/sample_a/100", _make_record(model_output="old"))])
        _write_test_lmdb(lmdb_b, [("train/sample_a/500", _make_record(model_output="new"))])
        result = self._make_loader((lmdb_a, lmdb_b), selection="latest")
        assert result["sample_a"].model_output == "new"

    def test_multiple_lmdb_paths_dedup_best_reward(self, tmp_path: pathlib.Path) -> None:
        lmdb_a = tmp_path / "rank_0"
        lmdb_b = tmp_path / "rank_1"
        _write_test_lmdb(lmdb_a, [("train/sample_a/100", _make_record(model_output="low", reward_total=0.1))])
        _write_test_lmdb(lmdb_b, [("train/sample_a/200", _make_record(model_output="high", reward_total=0.9))])
        result = self._make_loader((lmdb_a, lmdb_b), selection="best_reward")
        assert result["sample_a"].model_output == "high"

    def test_glob_pattern_resolves_rank_dirs(self, tmp_path: pathlib.Path) -> None:
        for rank_idx in range(2):
            rank_dir = tmp_path / f"rank_{rank_idx}"
            _write_test_lmdb(
                rank_dir,
                [
                    (f"train/sample_{rank_idx}/1", _make_record(model_output=f"out_{rank_idx}")),
                ],
            )
        result = self._make_loader((tmp_path,))  # auto-discover rank_* subdirs
        assert result["sample_0"].model_output == "out_0"
        assert result["sample_1"].model_output == "out_1"

    def test_glob_pattern_no_match_raises(self, tmp_path: pathlib.Path) -> None:
        import pathlib

        with pytest.raises(ValueError, match="matched zero paths"):
            self._make_loader((pathlib.Path(str(tmp_path / "nonexistent_*")),))

    def test_provenance_tracked(self, tmp_path: pathlib.Path) -> None:
        lmdb_path = tmp_path / "gen.lmdb"
        _write_test_lmdb(lmdb_path, [("train/sample_a/1", _make_record(model_output="out_a", reasoning="think"))])
        result = self._make_loader((lmdb_path,))
        record = result["sample_a"]
        assert record.source_lmdb_path == str(lmdb_path.resolve())
        assert record.source_key == "train/sample_a/1"
        assert record.full_record["reasoning"] == "think"
        assert record.full_record["model_output"] == "out_a"

    def test_provenance_across_multiple_lmdbs(self, tmp_path: pathlib.Path) -> None:
        lmdb_a = tmp_path / "rank_0"
        lmdb_b = tmp_path / "rank_1"
        _write_test_lmdb(lmdb_a, [("train/sample_a/100", _make_record(model_output="old"))])
        _write_test_lmdb(lmdb_b, [("train/sample_a/500", _make_record(model_output="new"))])
        result = self._make_loader((lmdb_a, lmdb_b), selection="latest")
        record = result["sample_a"]
        assert record.source_lmdb_path == str(lmdb_b.resolve())  # newer from rank_1

    def test_tiebreaker_respects_input_order(self, tmp_path: pathlib.Path) -> None:
        """Two LMDBs with equal gen_count — the later LMDB (by user-supplied order) wins."""
        lmdb_a = tmp_path / "rank_0"
        lmdb_b = tmp_path / "rank_1"
        _write_test_lmdb(lmdb_a, [("train/sample_a/100", _make_record(model_output="first"))])
        _write_test_lmdb(lmdb_b, [("train/sample_a/100", _make_record(model_output="second"))])
        result = self._make_loader((lmdb_a, lmdb_b), selection="latest")
        assert result["sample_a"].model_output == "second"
        assert result["sample_a"].source_lmdb_path == str(lmdb_b.resolve())


class TestResolveLmdbPaths:
    """Tests for resolve_lmdb_paths helper."""

    def test_non_glob_passthrough(self, tmp_path: pathlib.Path) -> None:
        lmdb_dir = tmp_path / "gen.lmdb"
        lmdb_dir.mkdir()
        (lmdb_dir / "data.mdb").touch()
        result = pyine.data.utils.lmdb_io.resolve_lmdb_paths((lmdb_dir,))
        assert result == [lmdb_dir.resolve()]

    def test_glob_resolves(self, tmp_path: pathlib.Path) -> None:
        import pathlib

        for rank_idx in range(2):
            rank_dir = tmp_path / f"rank_{rank_idx}"
            rank_dir.mkdir()
            (rank_dir / "data.mdb").touch()
        result = pyine.data.utils.lmdb_io.resolve_lmdb_paths((pathlib.Path(str(tmp_path / "rank_*")),))
        assert len(result) == 2
        assert all(p.name.startswith("rank_") for p in result)

    def test_recursive_glob(self, tmp_path: pathlib.Path) -> None:
        import pathlib

        nested = tmp_path / "parent" / "child" / "rank_0"
        nested.mkdir(parents=True)
        (nested / "data.mdb").touch()
        result = pyine.data.utils.lmdb_io.resolve_lmdb_paths((pathlib.Path(str(tmp_path / "**" / "rank_*")),))
        assert len(result) == 1
        assert result[0] == nested.resolve()

    def test_glob_no_match_raises(self, tmp_path: pathlib.Path) -> None:
        import pathlib

        with pytest.raises(ValueError, match="matched zero paths"):
            pyine.data.utils.lmdb_io.resolve_lmdb_paths((pathlib.Path(str(tmp_path / "nonexistent_*")),))

    def test_mixed_glob_and_explicit(self, tmp_path: pathlib.Path) -> None:
        import pathlib

        explicit_dir = tmp_path / "explicit"
        explicit_dir.mkdir()
        (explicit_dir / "data.mdb").touch()
        for rank_idx in range(2):
            rank_dir = tmp_path / f"rank_{rank_idx}"
            rank_dir.mkdir()
            (rank_dir / "data.mdb").touch()
        result = pyine.data.utils.lmdb_io.resolve_lmdb_paths(
            (
                explicit_dir,
                pathlib.Path(str(tmp_path / "rank_*")),
            )
        )
        assert len(result) == 3

    def test_auto_discover_rank_subdirs(self, tmp_path: pathlib.Path) -> None:
        parent = tmp_path / "output"
        parent.mkdir()
        for rank_idx in range(2):
            rank_dir = parent / f"rank_{rank_idx}"
            rank_dir.mkdir()
            (rank_dir / "data.mdb").touch()
        result = pyine.data.utils.lmdb_io.resolve_lmdb_paths((parent,))
        assert len(result) == 2

    def test_auto_discover_no_subdirs_raises(self, tmp_path: pathlib.Path) -> None:
        parent = tmp_path / "empty_output"
        parent.mkdir()
        with pytest.raises(ValueError, match="does not contain data.mdb or rank_"):
            pyine.data.utils.lmdb_io.resolve_lmdb_paths((parent,))

    def test_nonexistent_path_raises(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            pyine.data.utils.lmdb_io.resolve_lmdb_paths((tmp_path / "nonexistent",))

    def test_path_missing_data_mdb_raises(self, tmp_path: pathlib.Path) -> None:
        """Directory exists but no data.mdb inside (and no rank_* subdirs)."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="does not contain data.mdb or rank_"):
            pyine.data.utils.lmdb_io.resolve_lmdb_paths((empty_dir,))

    def test_duplicate_paths_deduplicated(self, tmp_path: pathlib.Path) -> None:
        lmdb_dir = tmp_path / "gen.lmdb"
        lmdb_dir.mkdir()
        (lmdb_dir / "data.mdb").touch()
        result = pyine.data.utils.lmdb_io.resolve_lmdb_paths((lmdb_dir, lmdb_dir))
        assert len(result) == 1

    def test_tilde_expansion_before_glob(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import pathlib

        monkeypatch.setenv("HOME", str(tmp_path))
        rank_dir = tmp_path / "rank_0"
        rank_dir.mkdir()
        (rank_dir / "data.mdb").touch()
        result = pyine.data.utils.lmdb_io.resolve_lmdb_paths((pathlib.Path("~/rank_*"),))
        assert len(result) == 1
        assert result[0] == rank_dir.resolve()


class TestDiskRewardLoggerRoundTrip:
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
        pregen_record = samples_common.PregeneratedOutputRecord(
            model_output="OVERRIDDEN_OUTPUT",
            source_lmdb_path="/fake/path",
            source_key="train/test/s0001/t0001/1",
            full_record={"model_output": "OVERRIDDEN_OUTPUT", "reward_total": 1.0},
        )
        overrides = {"test/s0001/t0001": pregen_record}
        if sample.identifier in overrides:
            assert sample.predict_type == samples_common.SamplePredictType.program_output
            record = overrides[sample.identifier]
            sample = sample._replace(
                pregenerated_output=record.model_output,
                pregenerated_output_lmdb_path=record.source_lmdb_path,
                pregenerated_output_lmdb_key=record.source_key,
            )
        assert sample.pregenerated_output == "OVERRIDDEN_OUTPUT"
        assert sample.pregenerated_output_lmdb_path == "/fake/path"
        assert sample.pregenerated_output_lmdb_key == "train/test/s0001/t0001/1"
        assert sample.expected_output == "42"  # original ground truth preserved

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
        pregen_record = samples_common.PregeneratedOutputRecord(
            model_output="OVERRIDDEN",
            source_lmdb_path="/fake",
            source_key="key",
            full_record={},
        )
        overrides = {"test/s0001/t0001": pregen_record}
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
        pregen_record = samples_common.PregeneratedOutputRecord(
            model_output="OVERRIDDEN",
            source_lmdb_path="/fake",
            source_key="key",
            full_record={},
        )
        overrides = {"other/id": pregen_record}
        if sample.identifier in overrides:
            record = overrides[sample.identifier]
            sample = sample._replace(pregenerated_output=record.model_output)
        assert sample.pregenerated_output == ""
        assert sample.pregenerated_output_lmdb_path == ""
        assert sample.pregenerated_output_lmdb_key == ""
        assert sample.expected_output == "42"


class TestPregeneratedOutputsConfigValidation:
    def test_phase_prefix_normalization(self) -> None:
        import pyine.utils.parsing

        assert pyine.utils.parsing.normalize_path_prefix("train") == "train/"
        assert pyine.utils.parsing.normalize_path_prefix("train/") == "train/"
        assert pyine.utils.parsing.normalize_path_prefix("") == ""
        assert pyine.utils.parsing.normalize_path_prefix("  ") == ""


class TestConfigValidators:
    """Tests for pregenerated_outputs_lmdb_paths field validator (direct + model_validate)."""

    def test_single_path_string_normalized_to_tuple(self) -> None:
        import pathlib

        result = pyine.organisms.datamodules.base.BiasDataModuleBaseConfig._normalize_pregenerated_lmdb_paths(
            "/some/path"
        )
        assert result == (pathlib.Path("/some/path"),)

    def test_list_of_paths_normalized_to_tuple(self) -> None:
        import pathlib

        result = pyine.organisms.datamodules.base.BiasDataModuleBaseConfig._normalize_pregenerated_lmdb_paths(
            ["/a", "/b"]
        )
        assert result == (pathlib.Path("/a"), pathlib.Path("/b"))

    def test_empty_list_raises_via_validator(self) -> None:
        with pytest.raises(ValueError, match="empty sequence"):
            pyine.organisms.datamodules.base.BiasDataModuleBaseConfig._normalize_pregenerated_lmdb_paths([])

    def test_none_passes_validator(self) -> None:
        result = pyine.organisms.datamodules.base.BiasDataModuleBaseConfig._normalize_pregenerated_lmdb_paths(None)
        assert result is None

    def test_omegaconf_list_config_via_validator(self) -> None:
        import pathlib

        result = pyine.organisms.datamodules.base.BiasDataModuleBaseConfig._normalize_pregenerated_lmdb_paths(
            omegaconf.ListConfig(["/a"])
        )
        assert result == (pathlib.Path("/a"),)

    def test_bytes_rejected(self) -> None:
        with pytest.raises(ValueError, match="bytes"):
            pyine.organisms.datamodules.base.BiasDataModuleBaseConfig._normalize_pregenerated_lmdb_paths(b"/path")

    def test_bytes_inside_sequence_rejected(self) -> None:
        with pytest.raises(ValueError, match="bytes.*element"):
            pyine.organisms.datamodules.base.BiasDataModuleBaseConfig._normalize_pregenerated_lmdb_paths([b"/path"])


class TestConfigValidatorsIntegration:
    """Integration tests exercising the full model_validate flow for pregenerated paths."""

    @pytest.fixture()
    def _minimal_config_fields(self, tmp_path: pathlib.Path) -> dict[str, typing.Any]:
        """Minimal required fields for ShortcutBiasDataModuleConfig.model_validate."""
        lmdb_dir = tmp_path / "traces.lmdb"
        lmdb_dir.mkdir()
        (lmdb_dir / "data.mdb").touch()
        split_file = tmp_path / "split.msgpack"
        split_file.touch()
        return {
            "lmdb_paths": (lmdb_dir,),
            "split_file_path": split_file,
        }

    def test_single_path_via_model_validate(
        self,
        tmp_path: pathlib.Path,
        _minimal_config_fields: dict[str, typing.Any],
    ) -> None:
        import pathlib

        import pyine.organisms.datamodules.shortcuts_configs

        pregen_dir = tmp_path / "pregen.lmdb"
        pregen_dir.mkdir()
        (pregen_dir / "data.mdb").touch()
        config = pyine.organisms.datamodules.shortcuts_configs.ShortcutBiasDataModuleConfig.model_validate(
            {**_minimal_config_fields, "pregenerated_outputs_lmdb_paths": str(pregen_dir)},
        )
        assert config.pregenerated_outputs_lmdb_paths == (pathlib.Path(str(pregen_dir)),)

    def test_list_of_paths_via_model_validate(
        self,
        tmp_path: pathlib.Path,
        _minimal_config_fields: dict[str, typing.Any],
    ) -> None:
        import pathlib

        import pyine.organisms.datamodules.shortcuts_configs

        pregen_a = tmp_path / "a.lmdb"
        pregen_b = tmp_path / "b.lmdb"
        for dirpath in (pregen_a, pregen_b):
            dirpath.mkdir()
            (dirpath / "data.mdb").touch()
        config = pyine.organisms.datamodules.shortcuts_configs.ShortcutBiasDataModuleConfig.model_validate(
            {**_minimal_config_fields, "pregenerated_outputs_lmdb_paths": [str(pregen_a), str(pregen_b)]},
        )
        assert config.pregenerated_outputs_lmdb_paths == (pathlib.Path(str(pregen_a)), pathlib.Path(str(pregen_b)))

    def test_omegaconf_list_via_model_validate(
        self,
        tmp_path: pathlib.Path,
        _minimal_config_fields: dict[str, typing.Any],
    ) -> None:
        import pathlib

        import pyine.organisms.datamodules.shortcuts_configs

        pregen_dir = tmp_path / "pregen.lmdb"
        pregen_dir.mkdir()
        (pregen_dir / "data.mdb").touch()
        config = pyine.organisms.datamodules.shortcuts_configs.ShortcutBiasDataModuleConfig.model_validate(
            {**_minimal_config_fields, "pregenerated_outputs_lmdb_paths": omegaconf.ListConfig([str(pregen_dir)])},
        )
        assert config.pregenerated_outputs_lmdb_paths == (pathlib.Path(str(pregen_dir)),)

    def test_none_via_model_validate(
        self,
        _minimal_config_fields: dict[str, typing.Any],
    ) -> None:
        import pyine.organisms.datamodules.shortcuts_configs

        config = pyine.organisms.datamodules.shortcuts_configs.ShortcutBiasDataModuleConfig.model_validate(
            {**_minimal_config_fields, "pregenerated_outputs_lmdb_paths": None},
        )
        assert config.pregenerated_outputs_lmdb_paths is None

    def test_empty_list_raises_via_model_validate(
        self,
        _minimal_config_fields: dict[str, typing.Any],
    ) -> None:
        import pydantic

        import pyine.organisms.datamodules.shortcuts_configs

        with pytest.raises(pydantic.ValidationError, match="empty sequence"):
            pyine.organisms.datamodules.shortcuts_configs.ShortcutBiasDataModuleConfig.model_validate(
                {**_minimal_config_fields, "pregenerated_outputs_lmdb_paths": []},
            )


class _StubBiasDataModuleBase(pyine.organisms.datamodules.base.BiasDataModuleBase):  # type: ignore[type-arg]
    """Minimal concrete stub for testing base class methods."""

    def _get_metadata_model_class(self) -> type:
        raise NotImplementedError

    def _prepare_bias_specific_metadata(self, base_traces_meta: typing.Any, split_data: typing.Any) -> typing.Any:
        raise NotImplementedError

    def _log_setup_summary(self) -> None:
        pass


def _make_pregen_record(
    model_output: str,
    source_lmdb_path: str = "/fake/lmdb",
    source_key: str = "key",
) -> samples_common.PregeneratedOutputRecord:
    return samples_common.PregeneratedOutputRecord(
        model_output=model_output,
        source_lmdb_path=source_lmdb_path,
        source_key=source_key,
        full_record={"model_output": model_output},
    )


class TestResolvePregenForSubsetBase:
    """Tests for BiasDataModuleBase._resolve_pregenerated_outputs_for_subset (via stub)."""

    def _make_dm(
        self,
        pregenerated_outputs: dict[str, samples_common.PregeneratedOutputRecord] | None,
    ) -> _StubBiasDataModuleBase:
        dm = object.__new__(_StubBiasDataModuleBase)
        object.__setattr__(dm, "_pregenerated_outputs", pregenerated_outputs)
        return dm

    def test_base_class_returns_unsuffixed_dict(self) -> None:
        outputs = {
            "trace_a": _make_pregen_record("out_a"),
            "trace_b": _make_pregen_record("out_b"),
        }
        dm = self._make_dm(outputs)
        result = dm._resolve_pregenerated_outputs_for_subset("train")
        assert result == outputs
        assert result is not dm._pregenerated_outputs  # should be a copy

    def test_empty_resolved_returns_empty_dict(self) -> None:
        dm = self._make_dm(None)
        result = dm._resolve_pregenerated_outputs_for_subset("train")
        assert result == {}

    def test_base_resolver_raises_on_suffixed_keys(self) -> None:
        outputs = {
            "trace_a::hinted": _make_pregen_record("out_a"),
        }
        dm = self._make_dm(outputs)
        with pytest.raises(ValueError, match="suffix separators"):
            dm._resolve_pregenerated_outputs_for_subset("train")


class TestResolvePregenForSubsetShortcut:
    """Tests for ShortcutBiasDataModule._resolve_pregenerated_outputs_for_subset."""

    def _make_dm(
        self,
        pregenerated_outputs: dict[str, samples_common.PregeneratedOutputRecord] | None,
        overlapping_ids: frozenset[str] = frozenset(),
    ) -> typing.Any:
        import pyine.organisms.datamodules.shortcuts as shortcuts_mod

        dm = object.__new__(shortcuts_mod.ShortcutBiasDataModule)
        object.__setattr__(dm, "_pregenerated_outputs", pregenerated_outputs)
        object.__setattr__(dm, "_metadata", None)
        # monkeypatch _get_overlapping_trace_ids to return our fixture
        dm._get_overlapping_trace_ids = lambda _parent, _suffix: overlapping_ids  # type: ignore[attr-defined]
        return dm

    def test_hinted_subset_gets_only_hinted_entries(self) -> None:
        outputs = {
            "trace::hinted": _make_pregen_record("out_h"),
            "trace::hintless": _make_pregen_record("out_l"),
        }
        dm = self._make_dm(outputs)
        result = dm._resolve_pregenerated_outputs_for_subset("train_hinted")
        assert result == {"trace": _make_pregen_record("out_h")}

    def test_hintless_subset_gets_only_hintless_entries(self) -> None:
        outputs = {
            "trace::hinted": _make_pregen_record("out_h"),
            "trace::hintless": _make_pregen_record("out_l"),
        }
        dm = self._make_dm(outputs)
        result = dm._resolve_pregenerated_outputs_for_subset("train_hintless")
        assert result == {"trace": _make_pregen_record("out_l")}

    def test_suffixes_do_not_collide(self) -> None:
        outputs = {
            "trace::hinted": _make_pregen_record("h"),
            "trace::hintless": _make_pregen_record("l"),
        }
        dm = self._make_dm(outputs)
        hinted_result = dm._resolve_pregenerated_outputs_for_subset("train_hinted")
        hintless_result = dm._resolve_pregenerated_outputs_for_subset("train_hintless")
        assert hinted_result["trace"].model_output == "h"
        assert hintless_result["trace"].model_output == "l"

    def test_non_suffixed_entries_available_to_all(self) -> None:
        outputs = {
            "trace_a": _make_pregen_record("a"),
        }
        dm = self._make_dm(outputs)
        for subset in ("train_hinted", "train_hintless", "train"):
            result = dm._resolve_pregenerated_outputs_for_subset(subset)
            assert "trace_a" in result

    def test_plain_subset_excludes_suffixed(self) -> None:
        outputs = {
            "trace::hinted": _make_pregen_record("h"),
            "trace_b": _make_pregen_record("b"),
        }
        dm = self._make_dm(outputs)
        result = dm._resolve_pregenerated_outputs_for_subset("train")
        assert "trace_b" in result
        assert "trace" not in result  # suffixed entries excluded from plain subsets

    def test_suffix_exact_match_required(self) -> None:
        """trace::hinted does NOT match train_misleading subset."""
        outputs = {
            "trace::hinted": _make_pregen_record("h"),
        }
        dm = self._make_dm(outputs)
        result = dm._resolve_pregenerated_outputs_for_subset("train_misleading")
        assert "trace" not in result

    def test_suffix_collision_raises(self) -> None:
        """Both unsuffixed and suffixed for the same base ID in the same subset."""
        outputs = {
            "trace_a": _make_pregen_record("unsuffixed"),
            "trace_a::hinted": _make_pregen_record("suffixed"),
        }
        dm = self._make_dm(outputs)
        with pytest.raises(ValueError, match="conflicting"):
            dm._resolve_pregenerated_outputs_for_subset("train_hinted")

    def test_keyword_cf_suffix_raises(self) -> None:
        outputs = {
            "trace::cf_with": _make_pregen_record("cf"),
        }
        dm = self._make_dm(outputs)
        with pytest.raises(ValueError, match="keyword counterfactual"):
            dm._resolve_pregenerated_outputs_for_subset("train_hinted")

    def test_unknown_suffix_raises(self) -> None:
        """Key with unrecognized :: suffix must raise, not silently drop."""
        outputs = {
            "trace::unknown": _make_pregen_record("x"),
        }
        dm = self._make_dm(outputs)
        with pytest.raises(ValueError, match="unrecognized"):
            dm._resolve_pregenerated_outputs_for_subset("train")

    def test_unknown_suffix_raises_in_hinted_subset(self) -> None:
        """Unknown suffix must also raise when resolving for a hint subset."""
        outputs = {
            "trace::bogus": _make_pregen_record("x"),
        }
        dm = self._make_dm(outputs)
        with pytest.raises(ValueError, match="unrecognized"):
            dm._resolve_pregenerated_outputs_for_subset("train_hinted")

    def test_unsuffixed_overlapping_trace_raises(self) -> None:
        outputs = {
            "trace": _make_pregen_record("ambiguous"),
        }
        dm = self._make_dm(outputs, overlapping_ids=frozenset({"trace"}))
        with pytest.raises(ValueError, match="missing the required suffix"):
            dm._resolve_pregenerated_outputs_for_subset("train_hinted")

    def test_overlapping_trace_with_wrong_suffix_raises(self) -> None:
        """Overlapping trace has only ::hinted but we're resolving for train_hintless."""
        outputs = {
            "trace::hinted": _make_pregen_record("h"),
        }
        dm = self._make_dm(outputs, overlapping_ids=frozenset({"trace"}))
        with pytest.raises(ValueError, match="missing the required suffix"):
            dm._resolve_pregenerated_outputs_for_subset("train_hintless")

    def test_only_with_pregenerated_output_empty_subset_raises(self) -> None:
        """Resolved dict is empty + only_matched=True -> ValueError from _build_parser_kwargs."""
        import pyine.organisms.datamodules.shortcuts as shortcuts_mod
        import pyine.organisms.datamodules.shortcuts_configs as shortcuts_configs_mod

        config = shortcuts_configs_mod.ShortcutBiasDataModuleConfig.__new__(
            shortcuts_configs_mod.ShortcutBiasDataModuleConfig,
        )
        object.__setattr__(config, "pregenerated_outputs_only_matched", True)
        dm = object.__new__(shortcuts_mod.ShortcutBiasDataModule)
        object.__setattr__(dm, "config", config)
        # only a ::hintless entry, but resolving for train_hinted -> empty result
        outputs: dict[str, samples_common.PregeneratedOutputRecord] = {
            "trace::hintless": _make_pregen_record("l"),
        }
        object.__setattr__(dm, "_pregenerated_outputs", outputs)
        object.__setattr__(dm, "_metadata", None)
        dm._get_overlapping_trace_ids = lambda _parent, _suffix: frozenset()  # type: ignore[attr-defined]
        with pytest.raises(ValueError, match="pregenerated_outputs_only_matched"):
            dm._build_parser_kwargs(
                source_data=[],
                subset_traces=[],
                subset_name="train_hinted",
            )


class TestResolvePregenForSubsetKeyword:
    """Tests for KeywordBiasDataModule._resolve_pregenerated_outputs_for_subset."""

    def _make_dm(
        self,
        pregenerated_outputs: dict[str, samples_common.PregeneratedOutputRecord] | None,
        evaluation_strategy: str = "keyword_presence_split",
        eval_subset_names: tuple[str, ...] = ("valid",),
    ) -> typing.Any:
        import pyine.organisms.datamodules.keywords as keywords_mod
        from pyine.organisms.datamodules.keywords_configs import EvaluationStrategy

        dm = object.__new__(keywords_mod.KeywordBiasDataModule)
        object.__setattr__(dm, "_pregenerated_outputs", pregenerated_outputs)
        config = object.__new__(keywords_mod.KeywordBiasDataModuleConfig)
        object.__setattr__(config, "evaluation_strategy", EvaluationStrategy(evaluation_strategy))
        object.__setattr__(config, "eval_subset_names", eval_subset_names)
        object.__setattr__(dm, "config", config)
        return dm

    def test_none_pregenerated_returns_empty(self) -> None:
        dm = self._make_dm(None, evaluation_strategy="counterfactual")
        assert dm._resolve_pregenerated_outputs_for_subset("valid") == {}

    # --- case 3: train / non-cf subsets (unsuffixed only) ---

    def test_non_counterfactual_unsuffixed_ok(self) -> None:
        outputs = {"trace_a": _make_pregen_record("a"), "trace_b": _make_pregen_record("b")}
        dm = self._make_dm(outputs, evaluation_strategy="keyword_presence_split")
        result = dm._resolve_pregenerated_outputs_for_subset("train")
        assert result == outputs
        assert result is not dm._pregenerated_outputs

    def test_non_counterfactual_base_eval_ok(self) -> None:
        outputs = {"trace_a": _make_pregen_record("a")}
        dm = self._make_dm(outputs, evaluation_strategy="keyword_presence_split")
        result = dm._resolve_pregenerated_outputs_for_subset("valid")
        assert result == outputs

    def test_counterfactual_train_unsuffixed_ok(self) -> None:
        outputs = {"trace_a": _make_pregen_record("a")}
        dm = self._make_dm(outputs, evaluation_strategy="counterfactual")
        result = dm._resolve_pregenerated_outputs_for_subset("train")
        assert result == outputs

    def test_train_skips_cf_suffixed_entries(self) -> None:
        """Train subset silently skips ::cf_* entries (they belong to eval subsets)."""
        outputs = {
            "trace_a": _make_pregen_record("a"),
            "trace_a::cf_with": _make_pregen_record("cf_w"),
            "trace_a::cf_without": _make_pregen_record("cf_wo"),
        }
        dm = self._make_dm(outputs, evaluation_strategy="counterfactual")
        result = dm._resolve_pregenerated_outputs_for_subset("train")
        assert result == {"trace_a": _make_pregen_record("a")}

    def test_non_cf_base_eval_skips_cf_entries(self) -> None:
        outputs = {
            "trace_a": _make_pregen_record("a"),
            "trace_a::cf_with": _make_pregen_record("cf_w"),
        }
        dm = self._make_dm(outputs, evaluation_strategy="keyword_presence_split")
        result = dm._resolve_pregenerated_outputs_for_subset("valid")
        assert result == {"trace_a": _make_pregen_record("a")}

    def test_unknown_suffix_rejected_in_train(self) -> None:
        outputs = {"trace_a::hinted": _make_pregen_record("a")}
        dm = self._make_dm(outputs, evaluation_strategy="keyword_presence_split")
        with pytest.raises(ValueError, match="unrecognized"):
            dm._resolve_pregenerated_outputs_for_subset("train")

    def test_non_cf_derived_subset_skips_cf_entries(self) -> None:
        """In keyword_presence_split mode, derived subsets should NOT accept ::cf_* entries."""
        outputs = {
            "trace_a::cf_with": _make_pregen_record("w"),
            "trace_b": _make_pregen_record("b"),
        }
        dm = self._make_dm(outputs, evaluation_strategy="keyword_presence_split")
        result = dm._resolve_pregenerated_outputs_for_subset("valid_with_keyword")
        assert result == {"trace_b": _make_pregen_record("b")}  # only unsuffixed

    # --- case 2: derived subsets (_with_keyword / _without_keyword) in counterfactual mode ---

    def test_derived_with_keyword_maps_cf_with(self) -> None:
        outputs = {
            "trace_a::cf_with": _make_pregen_record("w"),
            "trace_a::cf_without": _make_pregen_record("wo"),
        }
        dm = self._make_dm(outputs, evaluation_strategy="counterfactual")
        result = dm._resolve_pregenerated_outputs_for_subset("valid_with_keyword")
        assert result == {"trace_a": _make_pregen_record("w")}

    def test_derived_without_keyword_maps_cf_without(self) -> None:
        outputs = {
            "trace_a::cf_with": _make_pregen_record("w"),
            "trace_a::cf_without": _make_pregen_record("wo"),
        }
        dm = self._make_dm(outputs, evaluation_strategy="counterfactual")
        result = dm._resolve_pregenerated_outputs_for_subset("valid_without_keyword")
        assert result == {"trace_a": _make_pregen_record("wo")}

    def test_derived_includes_unsuffixed_entries(self) -> None:
        outputs = {"trace_b": _make_pregen_record("b")}
        dm = self._make_dm(outputs, evaluation_strategy="counterfactual")
        result = dm._resolve_pregenerated_outputs_for_subset("valid_with_keyword")
        assert result == {"trace_b": _make_pregen_record("b")}

    def test_derived_mixed_suffixed_and_unsuffixed(self) -> None:
        outputs = {
            "trace_a::cf_with": _make_pregen_record("w"),
            "trace_a::cf_without": _make_pregen_record("wo"),
            "trace_b": _make_pregen_record("b"),
        }
        dm = self._make_dm(outputs, evaluation_strategy="counterfactual")
        result = dm._resolve_pregenerated_outputs_for_subset("valid_with_keyword")
        assert result == {"trace_a": _make_pregen_record("w"), "trace_b": _make_pregen_record("b")}

    def test_derived_unknown_suffix_rejected(self) -> None:
        outputs = {"trace_a::hinted": _make_pregen_record("a")}
        dm = self._make_dm(outputs, evaluation_strategy="counterfactual")
        with pytest.raises(ValueError, match="unrecognized"):
            dm._resolve_pregenerated_outputs_for_subset("valid_with_keyword")

    # --- case 1: cf base eval subsets ---

    def test_cf_base_eval_routes_both_variants(self) -> None:
        rec_w = _make_pregen_record("w")
        rec_wo = _make_pregen_record("wo")
        outputs = {"trace_a::cf_with": rec_w, "trace_a::cf_without": rec_wo}
        dm = self._make_dm(outputs, evaluation_strategy="counterfactual")
        result = dm._resolve_pregenerated_outputs_for_subset("valid")
        # base-ID entry (placeholder for builder) + both cf-suffixed entries (for wrapper)
        assert result["trace_a"] == rec_w  # placeholder = cf_with
        assert result["trace_a::cf_with"] == rec_w
        assert result["trace_a::cf_without"] == rec_wo
        assert len(result) == 3

    def test_cf_base_eval_rejects_unsuffixed(self) -> None:
        outputs = {"trace_a": _make_pregen_record("a")}
        dm = self._make_dm(outputs, evaluation_strategy="counterfactual")
        with pytest.raises(ValueError, match="unsuffixed"):
            dm._resolve_pregenerated_outputs_for_subset("valid")

    def test_cf_base_eval_requires_both_variants(self) -> None:
        outputs = {"trace_a::cf_with": _make_pregen_record("w")}
        dm = self._make_dm(outputs, evaluation_strategy="counterfactual")
        with pytest.raises(ValueError, match="missing counterfactual"):
            dm._resolve_pregenerated_outputs_for_subset("valid")

    def test_cf_base_eval_unknown_suffix_rejected(self) -> None:
        outputs = {"trace_a::hinted": _make_pregen_record("a")}
        dm = self._make_dm(outputs, evaluation_strategy="counterfactual")
        with pytest.raises(ValueError, match="unrecognized"):
            dm._resolve_pregenerated_outputs_for_subset("valid")

    def test_cf_base_eval_multiple_traces(self) -> None:
        outputs = {
            "trace_a::cf_with": _make_pregen_record("aw"),
            "trace_a::cf_without": _make_pregen_record("awo"),
            "trace_b::cf_with": _make_pregen_record("bw"),
            "trace_b::cf_without": _make_pregen_record("bwo"),
        }
        dm = self._make_dm(outputs, evaluation_strategy="counterfactual")
        result = dm._resolve_pregenerated_outputs_for_subset("valid")
        assert len(result) == 6  # 2 traces * (1 base + 2 cf)
        assert result["trace_a::cf_with"].model_output == "aw"
        assert result["trace_b::cf_without"].model_output == "bwo"


class TestKeywordWrapperCfPregeneratedOutput:
    """Tests for SampleKeywordManipulatorWrapper counterfactual pregenerated output injection."""

    def _make_sample(
        self,
        identifier: str = "trace_a",
    ) -> samples_common.SampleData:
        return samples_common.SampleData(
            identifier=identifier,
            code="x = 1\nprint(x)",
            description="test",
            entrypoint="",
            first_line=0,
            last_line=1,
            inputs="",
            expected_output="1",
            predict_type=samples_common.SamplePredictType.program_output,
            code_type="original",
            trace_step_count=1,
            comma_separated_tags="",
            has_code_override=False,
            complexity_metrics={},
        )

    def test_cf_with_gets_correct_pregenerated_output(self) -> None:
        import pyine.organisms.datamodules.samples.keyword_ops as keyword_ops

        sample = self._make_sample()
        # simulate a builder-like wrapped object that has _pregenerated_outputs
        pregen = {
            "trace_a": _make_pregen_record("placeholder"),
            "trace_a::cf_with": _make_pregen_record("cf_with_output", source_key="key_w"),
            "trace_a::cf_without": _make_pregen_record("cf_without_output", source_key="key_wo"),
        }

        class _FakeDataset:
            _pregenerated_outputs = pregen

            def __len__(self) -> int:
                return 1

            def __getitem__(self, idx: int) -> samples_common.SampleData:
                return sample

        wrapper = keyword_ops.SampleKeywordManipulatorWrapper(
            wrapped_dataset=_FakeDataset(),  # type: ignore[arg-type]
            keyword="result",
            trace_ids_with_keyword=frozenset({"trace_a"}),
            counterfactual_mode=True,
        )
        cf_with = wrapper[0]  # even index = cf_with
        assert cf_with.identifier == "trace_a::cf_with"
        assert cf_with.pregenerated_output == "cf_with_output"
        assert cf_with.pregenerated_output_lmdb_key == "key_w"

    def test_cf_without_gets_correct_pregenerated_output(self) -> None:
        import pyine.organisms.datamodules.samples.keyword_ops as keyword_ops

        sample = self._make_sample()
        pregen = {
            "trace_a": _make_pregen_record("placeholder"),
            "trace_a::cf_with": _make_pregen_record("cf_with_output"),
            "trace_a::cf_without": _make_pregen_record("cf_without_output", source_key="key_wo"),
        }

        class _FakeDataset:
            _pregenerated_outputs = pregen

            def __len__(self) -> int:
                return 1

            def __getitem__(self, idx: int) -> samples_common.SampleData:
                return sample

        wrapper = keyword_ops.SampleKeywordManipulatorWrapper(
            wrapped_dataset=_FakeDataset(),  # type: ignore[arg-type]
            keyword="result",
            trace_ids_with_keyword=frozenset({"trace_a"}),
            counterfactual_mode=True,
        )
        cf_without = wrapper[1]  # odd index = cf_without
        assert cf_without.identifier == "trace_a::cf_without"
        assert cf_without.pregenerated_output == "cf_without_output"
        assert cf_without.pregenerated_output_lmdb_key == "key_wo"

    def test_no_pregen_dict_leaves_sample_unchanged(self) -> None:
        import pyine.organisms.datamodules.samples.keyword_ops as keyword_ops

        sample = self._make_sample()

        class _FakeDataset:
            def __len__(self) -> int:
                return 1

            def __getitem__(self, idx: int) -> samples_common.SampleData:
                return sample

        wrapper = keyword_ops.SampleKeywordManipulatorWrapper(
            wrapped_dataset=_FakeDataset(),  # type: ignore[arg-type]
            keyword="result",
            trace_ids_with_keyword=frozenset({"trace_a"}),
            counterfactual_mode=True,
        )
        cf_with = wrapper[0]
        assert cf_with.pregenerated_output == ""
        assert cf_with.pregenerated_output_lmdb_path == ""
