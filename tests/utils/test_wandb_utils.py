from __future__ import annotations

import typing
import unittest.mock

import pandas as pd

import pyine.utils.wandb_utils as wandb_utils


class _MockWandbRun:
    """Lightweight mock for wandb.apis.public.Run with configurable per-key data."""

    def __init__(
        self,
        per_key_data: dict[str, list[dict[str, typing.Any]]] | None = None,
        full_history: list[dict[str, typing.Any]] | None = None,
        sampled_df: pd.DataFrame | None = None,
    ) -> None:
        self._per_key_data = per_key_data or {}
        self._full_history = full_history or []
        self._sampled_df = sampled_df if sampled_df is not None else pd.DataFrame()

    def history(
        self,
        keys: list[str] | None = None,
        samples: int = 500,
        pandas: bool = True,
    ) -> pd.DataFrame:
        if keys and len(keys) == 1:
            key = keys[0]
            rows = self._per_key_data.get(key, [])
            return pd.DataFrame(rows) if rows else pd.DataFrame()
        return self._sampled_df.copy()

    def scan_history(
        self,
        keys: list[str] | None = None,
    ) -> list[dict[str, typing.Any]]:
        if keys and len(keys) == 1:
            key = keys[0]
            return list(self._per_key_data.get(key, []))
        return list(self._full_history)


class TestFetchSingleKeyHistory:
    def test_returns_dataframe_with_key_and_step(self) -> None:
        run = _MockWandbRun(
            per_key_data={
                "loss": [{"_step": 0, "loss": 1.0}, {"_step": 1, "loss": 0.5}],
            }
        )
        result = wandb_utils._fetch_single_key_history(
            run=run,
            key="loss",
            samples=10_000,
            full_fidelity=False,
            max_retries=0,
            verbose=False,
        )
        assert list(result.columns) == ["_step", "loss"]
        assert len(result) == 2

    def test_missing_key_returns_empty_dataframe(self) -> None:
        run = _MockWandbRun(per_key_data={})
        result = wandb_utils._fetch_single_key_history(
            run=run,
            key="nonexistent",
            samples=10_000,
            full_fidelity=False,
            max_retries=0,
            verbose=False,
        )
        assert result.empty

    def test_full_fidelity_uses_scan_history(self) -> None:
        run = _MockWandbRun(
            per_key_data={
                "reward": [{"_step": 10, "reward": 3.0}],
            }
        )
        with unittest.mock.patch.object(run, "scan_history", wraps=run.scan_history) as mock_scan:
            result = wandb_utils._fetch_single_key_history(
                run=run,
                key="reward",
                samples=10_000,
                full_fidelity=True,
                max_retries=0,
                verbose=False,
            )
            mock_scan.assert_called_once_with(keys=["reward"])
        assert len(result) == 1
        assert "reward" in result.columns


class TestMergePerKeyDataframes:
    def test_single_dataframe_passthrough(self) -> None:
        df = pd.DataFrame({"_step": [0, 1], "loss": [1.0, 0.5]})
        result = wandb_utils._merge_per_key_dataframes([df])
        pd.testing.assert_frame_equal(result, df.reset_index(drop=True))

    def test_outer_merge_different_steps(self) -> None:
        df_a = pd.DataFrame({"_step": [0, 1], "loss": [1.0, 0.5]})
        df_b = pd.DataFrame({"_step": [1, 2], "accuracy": [0.8, 0.9]})
        result = wandb_utils._merge_per_key_dataframes([df_a, df_b])
        assert set(result.columns) == {"_step", "loss", "accuracy"}
        assert len(result) == 3  # steps 0, 1, 2
        assert result.loc[result["_step"] == 0, "accuracy"].isna().all()
        assert result.loc[result["_step"] == 2, "loss"].isna().all()

    def test_coalesces_internal_key_duplicates(self) -> None:
        df_a = pd.DataFrame({"_step": [0, 1], "_timestamp": [100.0, 200.0], "loss": [1.0, 0.5]})
        df_b = pd.DataFrame({"_step": [0, 2], "_timestamp": [100.0, 300.0], "acc": [0.7, 0.9]})
        result = wandb_utils._merge_per_key_dataframes([df_a, df_b])
        assert "_timestamp_dup" not in result.columns
        assert "_timestamp" in result.columns
        assert result.loc[result["_step"] == 1, "_timestamp"].iloc[0] == 200.0
        assert result.loc[result["_step"] == 2, "_timestamp"].iloc[0] == 300.0

    def test_forward_fills_step_key(self) -> None:
        # simulate sampled train/global_step at steps [0, 2] and a sparse metric at step 1
        df_step = pd.DataFrame({"_step": [0, 2], "train/global_step": [0, 200]})
        df_metric = pd.DataFrame({"_step": [1], "reward": [5.0]})
        result = wandb_utils._merge_per_key_dataframes([df_step, df_metric])
        # step 1 should have train/global_step forward-filled from step 0
        row = result.loc[result["_step"] == 1]
        assert row["train/global_step"].iloc[0] == 0
        assert row["reward"].iloc[0] == 5.0

    def test_forward_fills_timestamp(self) -> None:
        df_ts = pd.DataFrame({"_step": [0, 2], "_timestamp": [100.0, 300.0]})
        df_metric = pd.DataFrame({"_step": [1], "loss": [0.5]})
        result = wandb_utils._merge_per_key_dataframes([df_ts, df_metric])
        row = result.loc[result["_step"] == 1]
        assert row["_timestamp"].iloc[0] == 100.0  # forward-filled from step 0

    def test_forward_fill_does_not_affect_metric_columns(self) -> None:
        df_a = pd.DataFrame({"_step": [0, 1], "loss": [1.0, 0.5]})
        df_b = pd.DataFrame({"_step": [0, 2], "reward": [3.0, 4.0]})
        result = wandb_utils._merge_per_key_dataframes([df_a, df_b])
        # metric columns should NOT be forward-filled — reward at step 1 should still be NaN
        assert result.loc[result["_step"] == 1, "reward"].isna().all()
        assert result.loc[result["_step"] == 2, "loss"].isna().all()

    def test_forward_fill_disabled_via_empty_tuple(self) -> None:
        df_ts = pd.DataFrame({"_step": [0, 2], "_timestamp": [100.0, 300.0]})
        df_metric = pd.DataFrame({"_step": [1], "loss": [0.5]})
        result = wandb_utils._merge_per_key_dataframes([df_ts, df_metric], forward_fill_keys=())
        row = result.loc[result["_step"] == 1]
        assert pd.isna(row["_timestamp"].iloc[0])  # no forward-fill

    def test_all_empty_returns_step_column(self) -> None:
        result = wandb_utils._merge_per_key_dataframes([pd.DataFrame(), pd.DataFrame()])
        assert list(result.columns) == ["_step"]
        assert result.empty


class TestGetQueryCachePath:
    def test_different_keys_produce_different_paths(self, tmp_path: typing.Any) -> None:
        base = tmp_path / "cache.parquet"
        path_a = wandb_utils._get_query_cache_path(base, keys=["loss"], samples=10_000, full_fidelity=False)
        path_b = wandb_utils._get_query_cache_path(base, keys=["reward"], samples=10_000, full_fidelity=False)
        assert path_a != path_b
        assert path_a.parent == base.parent
        assert path_a.suffix == ".parquet"

    def test_same_keys_produce_same_path(self, tmp_path: typing.Any) -> None:
        base = tmp_path / "cache.parquet"
        path_a = wandb_utils._get_query_cache_path(base, keys=["b", "a"], samples=10_000, full_fidelity=False)
        path_b = wandb_utils._get_query_cache_path(base, keys=["a", "b"], samples=10_000, full_fidelity=False)
        assert path_a == path_b  # order shouldn't matter (keys are sorted)

    def test_different_samples_produce_different_paths(self, tmp_path: typing.Any) -> None:
        base = tmp_path / "cache.parquet"
        path_a = wandb_utils._get_query_cache_path(base, keys=["loss"], samples=100, full_fidelity=False)
        path_b = wandb_utils._get_query_cache_path(base, keys=["loss"], samples=10_000, full_fidelity=False)
        assert path_a != path_b

    def test_discovery_mode_none_keys(self, tmp_path: typing.Any) -> None:
        base = tmp_path / "cache.parquet"
        path = wandb_utils._get_query_cache_path(base, keys=None, samples=10_000, full_fidelity=False)
        assert path.suffix == ".parquet"
        assert path != base


class TestFetchHistoryDf:
    def test_discovery_mode_uses_history(self) -> None:
        sampled = pd.DataFrame({"_step": [0, 1], "loss": [1.0, 0.5], "_timestamp": [1.0, 2.0]})
        run = _MockWandbRun(sampled_df=sampled)
        with unittest.mock.patch.object(run, "history", wraps=run.history) as mock_hist:
            result = wandb_utils.fetch_history_df(run, keys=None, samples=100, verbose=False)
            mock_hist.assert_called_once()
        assert "loss" in result.columns

    def test_per_key_fetching_and_merge(self) -> None:
        run = _MockWandbRun(
            per_key_data={
                "loss": [{"_step": 0, "loss": 1.0}, {"_step": 1, "loss": 0.5}],
                "reward": [{"_step": 5, "reward": 3.0}, {"_step": 10, "reward": 4.0}],
                "_timestamp": [
                    {"_step": 0, "_timestamp": 100.0},
                    {"_step": 1, "_timestamp": 200.0},
                    {"_step": 5, "_timestamp": 500.0},
                    {"_step": 10, "_timestamp": 1000.0},
                ],
                "_runtime": [
                    {"_step": 0, "_runtime": 0.0},
                    {"_step": 1, "_runtime": 1.0},
                    {"_step": 5, "_runtime": 5.0},
                    {"_step": 10, "_runtime": 10.0},
                ],
            }
        )
        result = wandb_utils.fetch_history_df(
            run,
            keys=["loss", "reward"],
            samples=10_000,
            verbose=False,
        )
        assert "loss" in result.columns
        assert "reward" in result.columns
        assert "_step" in result.columns
        # steps from both keys should be present
        steps = set(result["_step"].dropna().astype(int))
        assert {0, 1, 5, 10}.issubset(steps)

    def test_internal_keys_excluded_from_fetch_loop(self) -> None:
        run = _MockWandbRun(
            per_key_data={
                "loss": [{"_step": 0, "loss": 1.0}],
                "_timestamp": [{"_step": 0, "_timestamp": 100.0}],
                "_runtime": [{"_step": 0, "_runtime": 0.0}],
            }
        )
        with unittest.mock.patch.object(run, "history", wraps=run.history) as mock_hist:
            result = wandb_utils.fetch_history_df(
                run,
                keys=["_step", "loss"],
                samples=10_000,
                verbose=False,
            )
            # _step should not trigger its own fetch — only "loss" should be fetched per-key
            call_keys = [
                call.kwargs.get("keys") or call.args[0]
                for call in mock_hist.call_args_list
                if call.kwargs.get("keys") or (call.args and isinstance(call.args[0], list))
            ]
            flat_keys = [k for keys_list in call_keys for k in keys_list]
            assert "_step" not in flat_keys
        assert "_step" in result.columns

    def test_duplicate_keys_deduplicated(self) -> None:
        run = _MockWandbRun(
            per_key_data={
                "loss": [{"_step": 0, "loss": 1.0}],
                "_timestamp": [{"_step": 0, "_timestamp": 100.0}],
                "_runtime": [{"_step": 0, "_runtime": 0.0}],
            }
        )
        with unittest.mock.patch.object(run, "history", wraps=run.history) as mock_hist:
            wandb_utils.fetch_history_df(
                run,
                keys=["loss", "loss"],
                samples=10_000,
                verbose=False,
            )
            # "loss" should only be fetched once (via per-key path)
            loss_calls = [
                call
                for call in mock_hist.call_args_list
                if call.kwargs.get("keys") == ["loss"] or (call.args and call.args[0] == ["loss"])
            ]
            assert len(loss_calls) == 1

    def test_cache_roundtrip(self, tmp_path: typing.Any) -> None:
        cache_base = tmp_path / "cache.parquet"
        keys = ["loss"]
        run = _MockWandbRun(
            per_key_data={
                "loss": [{"_step": 0, "loss": 1.0}, {"_step": 1, "loss": 0.5}],
                "_timestamp": [{"_step": 0, "_timestamp": 100.0}, {"_step": 1, "_timestamp": 200.0}],
                "_runtime": [{"_step": 0, "_runtime": 0.0}, {"_step": 1, "_runtime": 1.0}],
            }
        )
        df1 = wandb_utils.fetch_history_df(run, keys=keys, cache_path=cache_base, verbose=False)
        resolved = wandb_utils._get_query_cache_path(cache_base, keys=keys, samples=10_000, full_fidelity=False)
        assert resolved.exists()
        df2 = wandb_utils.fetch_history_df(run, keys=keys, cache_path=cache_base, verbose=False)
        pd.testing.assert_frame_equal(df1, df2)

    def test_cache_hit_skips_fetch(self, tmp_path: typing.Any) -> None:
        cache_base = tmp_path / "cached.parquet"
        keys = ["loss"]
        # write the parquet at the resolved (hashed) path
        resolved = wandb_utils._get_query_cache_path(cache_base, keys=keys, samples=10_000, full_fidelity=False)
        cached_df = pd.DataFrame({"_step": [0], "loss": [1.0]})
        cached_df.to_parquet(resolved)
        run = _MockWandbRun()
        with unittest.mock.patch.object(run, "history", wraps=run.history) as mock_hist:
            result = wandb_utils.fetch_history_df(
                run,
                keys=keys,
                cache_path=cache_base,
                verbose=False,
            )
            mock_hist.assert_not_called()
        assert len(result) == 1

    def test_different_keys_produce_different_cache_files(self, tmp_path: typing.Any) -> None:
        cache_base = tmp_path / "history.parquet"
        run = _MockWandbRun(
            per_key_data={
                "loss": [{"_step": 0, "loss": 1.0}],
                "reward": [{"_step": 0, "reward": 5.0}],
                "_timestamp": [{"_step": 0, "_timestamp": 100.0}],
                "_runtime": [{"_step": 0, "_runtime": 0.0}],
            }
        )
        wandb_utils.fetch_history_df(run, keys=["loss"], cache_path=cache_base, verbose=False)
        wandb_utils.fetch_history_df(run, keys=["reward"], cache_path=cache_base, verbose=False)
        cache_files = list(tmp_path.glob("history.*.parquet"))
        assert len(cache_files) == 2

    def test_full_fidelity_flag(self) -> None:
        run = _MockWandbRun(
            per_key_data={
                "loss": [{"_step": 0, "loss": 1.0}],
                "_timestamp": [{"_step": 0, "_timestamp": 100.0}],
                "_runtime": [{"_step": 0, "_runtime": 0.0}],
            }
        )
        with unittest.mock.patch.object(run, "scan_history", wraps=run.scan_history) as mock_scan:
            wandb_utils.fetch_history_df(
                run,
                keys=["loss"],
                full_fidelity=True,
                verbose=False,
            )
            # scan_history should be used for per-key fetches (loss + time keys)
            assert mock_scan.call_count >= 1

    def test_nonexistent_key_absent_from_result(self) -> None:
        run = _MockWandbRun(
            per_key_data={
                "loss": [{"_step": 0, "loss": 1.0}],
                "_timestamp": [{"_step": 0, "_timestamp": 100.0}],
                "_runtime": [{"_step": 0, "_runtime": 0.0}],
            }
        )
        result = wandb_utils.fetch_history_df(
            run,
            keys=["loss", "nonexistent_metric"],
            verbose=False,
        )
        assert "loss" in result.columns
        assert "nonexistent_metric" not in result.columns

    def test_sparse_metric_gets_forward_filled_step_key(self) -> None:
        # train/global_step is dense (sampled at steps 0, 2, 4), reward is sparse (step 3 only)
        run = _MockWandbRun(
            per_key_data={
                "train/global_step": [
                    {"_step": 0, "train/global_step": 0},
                    {"_step": 2, "train/global_step": 200},
                    {"_step": 4, "train/global_step": 400},
                ],
                "reward": [{"_step": 3, "reward": 5.0}],
                "_timestamp": [
                    {"_step": 0, "_timestamp": 100.0},
                    {"_step": 2, "_timestamp": 200.0},
                    {"_step": 4, "_timestamp": 300.0},
                ],
                "_runtime": [],
            }
        )
        result = wandb_utils.fetch_history_df(
            run,
            keys=["train/global_step", "reward"],
            verbose=False,
        )
        # the row at _step=3 should have reward=5.0 AND train/global_step forward-filled from
        # step 2 (value 200), so dropna on both columns should keep the row
        plot_df = result[["train/global_step", "reward"]].dropna()
        assert len(plot_df) == 1
        assert plot_df["train/global_step"].iloc[0] == 200
        assert plot_df["reward"].iloc[0] == 5.0

    def test_full_fidelity_discovery_mode(self) -> None:
        full_hist = [
            {"_step": 0, "loss": 1.0, "_timestamp": 100.0},
            {"_step": 1, "loss": 0.5, "_timestamp": 200.0},
        ]
        run = _MockWandbRun(full_history=full_hist)
        with unittest.mock.patch.object(run, "scan_history", wraps=run.scan_history) as mock_scan:
            result = wandb_utils.fetch_history_df(
                run,
                keys=None,
                full_fidelity=True,
                verbose=False,
            )
            mock_scan.assert_called_once_with()
        assert len(result) == 2
        assert "loss" in result.columns
