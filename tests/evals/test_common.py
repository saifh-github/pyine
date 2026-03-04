"""Tests for pyine.evals.common -- shared eval utilities."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    import pytest


class TestEvalMetadataCaching:
    """Tests for cached eval reprod metadata helper."""

    def test_cached_eval_reprod_metadata_calls_source_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import pyine.evals.common as eval_common

        call_count = 0

        def _fake_reprod_metadata() -> dict[str, str]:
            nonlocal call_count
            call_count += 1
            return {"framework_version": "x", "installed_packages": "a==1.0.0"}

        monkeypatch.setattr(eval_common.pyine.utils.reprod, "get_reprod_metadata", _fake_reprod_metadata)
        eval_common._get_cached_eval_reprod_metadata.cache_clear()
        first = eval_common.get_cached_eval_reprod_metadata()
        second = eval_common.get_cached_eval_reprod_metadata()
        assert call_count == 1
        assert first == second
        first["framework_version"] = "mutated"
        third = eval_common.get_cached_eval_reprod_metadata()
        assert third["framework_version"] == "x"


class TestBuildBaseEvalMetadata:
    """Tests for build_base_eval_metadata helper."""

    def test_returns_expected_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import pyine.evals.common as eval_common

        monkeypatch.setattr(
            eval_common,
            "get_cached_eval_reprod_metadata",
            lambda: {"framework_version": "test"},
        )
        result = eval_common.build_base_eval_metadata(eval_common.EvalType.CODE_EXEC, "valid")
        assert result["eval_type"] == str(eval_common.EvalType.CODE_EXEC)
        assert result["eval_subset_name"] == "valid"
        assert result["reprod_metadata"] == {"framework_version": "test"}

    def test_none_eval_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import pyine.evals.common as eval_common

        monkeypatch.setattr(
            eval_common,
            "get_cached_eval_reprod_metadata",
            lambda: {},
        )
        result = eval_common.build_base_eval_metadata(None, None)
        assert result["eval_type"] is None
        assert result["eval_subset_name"] is None
