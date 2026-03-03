"""Tests for the shared difficulty scoring module (pyine.utils.code.difficulty)."""

import math

import pytest

import pyine.organisms.datamodules.samples.common as samples_common
import pyine.utils.code.difficulty as difficulty_utils


def _make_sample_data(
    *,
    identifier: str = "test",
    trace_step_count: int = 50,
    has_code_override: bool = False,
    complexity_metrics: dict[str, float | int] | None = None,
    code: str = "def foo():\n    return 1",
    inputs: str = "",
    expected_output: str = "1",
    first_line: int = 0,
    last_line: int = 2,
    first_step_idx: int = 0,
    last_step_idx: int = 0,
) -> samples_common.SampleData:
    return samples_common.SampleData(
        identifier=identifier,
        code=code,
        description="test",
        entrypoint="foo",
        first_line=first_line,
        last_line=last_line,
        inputs=inputs,
        expected_output=expected_output,
        predict_type=samples_common.SamplePredictType.program_output,
        code_type="original",
        trace_step_count=trace_step_count,
        comma_separated_tags="",
        has_code_override=has_code_override,
        complexity_metrics=complexity_metrics or {},
        first_step_idx=first_step_idx,
        last_step_idx=last_step_idx,
    )


def _word_counter(text: str) -> int:
    return len(text.split())


class TestBinEdges:
    """Tests for bin edge computation via DifficultyScorer.bin_edges."""

    def test_fixed_range_uniform_edges(self) -> None:
        config = difficulty_utils.DifficultyConfig(
            primary_source="maintainability_index",
            normalization_mode="fixed_range",
            num_difficulty_bins=5,
        )
        scorer = difficulty_utils.DifficultyScorer(config)
        assert scorer.bin_edges == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    def test_log_with_overflow(self) -> None:
        config = difficulty_utils.DifficultyConfig(
            primary_source="trace_step_count",
            normalization_mode="log",
            num_difficulty_bins=5,
            bin_max_score=5.0,
        )
        scorer = difficulty_utils.DifficultyScorer(config)
        edges = scorer.bin_edges
        assert len(edges) == 7  # 5 bins + overflow
        assert edges[-1] == float("inf")
        assert edges[0] == pytest.approx(0.0)
        assert edges[5] == pytest.approx(5.0)

    def test_log_no_overflow_with_clip(self) -> None:
        config = difficulty_utils.DifficultyConfig(
            primary_source="trace_step_count",
            normalization_mode="log",
            num_difficulty_bins=5,
            score_clip_max=5.0,
        )
        scorer = difficulty_utils.DifficultyScorer(config)
        edges = scorer.bin_edges
        assert len(edges) == 6  # no overflow
        assert edges[-1] != float("inf")

    def test_manual_edges_override(self) -> None:
        manual = [0.0, 1.0, 2.0, 3.0, float("inf")]
        config = difficulty_utils.DifficultyConfig(
            primary_source="trace_step_count",
            normalization_mode="log",
            bin_edges=manual,
        )
        scorer = difficulty_utils.DifficultyScorer(config)
        assert scorer.bin_edges == manual


class TestGetBinIndex:
    def test_first_bin(self) -> None:
        assert difficulty_utils._get_bin_index(0.5, [0.0, 1.0, 2.0, 3.0]) == 0

    def test_last_bin(self) -> None:
        assert difficulty_utils._get_bin_index(2.5, [0.0, 1.0, 2.0, 3.0]) == 2

    def test_overflow_bin(self) -> None:
        assert difficulty_utils._get_bin_index(100.0, [0.0, 1.0, 2.0, float("inf")]) == 2

    def test_exact_edge_goes_to_next_bin(self) -> None:
        assert difficulty_utils._get_bin_index(1.0, [0.0, 1.0, 2.0, 3.0]) == 1


class TestNormalizeValue:
    def test_log(self) -> None:
        config = difficulty_utils.DifficultyConfig(
            primary_source="trace_step_count",
            normalization_mode="log",
        )
        assert difficulty_utils.normalize_value(100.0, "trace_step_count", config) == pytest.approx(math.log1p(100.0))

    def test_none_passthrough(self) -> None:
        config = difficulty_utils.DifficultyConfig(
            primary_source="trace_step_count",
            normalization_mode="none",
            bin_edges=[0.0, 50.0, 100.0],
        )
        assert difficulty_utils.normalize_value(42.0, "trace_step_count", config) == pytest.approx(42.0)

    def test_fixed_range_with_invert(self) -> None:
        config = difficulty_utils.DifficultyConfig(
            primary_source="maintainability_index",
            normalization_mode="fixed_range",
        )
        score = difficulty_utils.normalize_value(80.0, "maintainability_index", config)
        assert score == pytest.approx(0.2)  # (100-80)/100

    def test_clip_max(self) -> None:
        config = difficulty_utils.DifficultyConfig(
            primary_source="trace_step_count",
            normalization_mode="log",
            score_clip_max=3.0,
        )
        score = difficulty_utils.normalize_value(10000.0, "trace_step_count", config)
        assert score == pytest.approx(3.0)

    def test_fixed_range_unknown_source_raises(self) -> None:
        config = difficulty_utils.DifficultyConfig(
            primary_source="maintainability_index",
            normalization_mode="fixed_range",
        )
        with pytest.raises(ValueError, match="fixed_range normalization requires source"):
            difficulty_utils.normalize_value(5.0, "unknown_metric", config)


class TestIsExecutionSource:
    def test_trace_step_count(self) -> None:
        assert difficulty_utils.is_execution_source("trace_step_count") is True

    def test_segment_span(self) -> None:
        assert difficulty_utils.is_execution_source("segment_span") is True

    def test_code_tokens(self) -> None:
        assert difficulty_utils.is_execution_source("code_tokens") is False

    def test_code_length(self) -> None:
        assert difficulty_utils.is_execution_source("code_length") is False

    def test_unknown_complexity_metric(self) -> None:
        assert difficulty_utils.is_execution_source("halstead_effort") is True


class TestDifficultyScorerInit:
    def test_token_sources_without_counter_raises(self) -> None:
        config = difficulty_utils.DifficultyConfig(primary_source="code_tokens")
        with pytest.raises(ValueError, match="require a token_counter"):
            difficulty_utils.DifficultyScorer(config)

    def test_secondary_token_source_without_counter_raises(self) -> None:
        config = difficulty_utils.DifficultyConfig(
            primary_source="trace_step_count",
            secondary_sources=frozenset({"inputs_tokens"}),
        )
        with pytest.raises(ValueError, match="require a token_counter"):
            difficulty_utils.DifficultyScorer(config)

    def test_token_sources_with_counter_ok(self) -> None:
        config = difficulty_utils.DifficultyConfig(primary_source="code_tokens")
        scorer = difficulty_utils.DifficultyScorer(config, token_counter=_word_counter)
        assert scorer.config.primary_source == "code_tokens"

    def test_bin_edges_exposed(self) -> None:
        config = difficulty_utils.DifficultyConfig(
            primary_source="trace_step_count",
            normalization_mode="log",
            num_difficulty_bins=5,
            bin_max_score=5.0,
        )
        scorer = difficulty_utils.DifficultyScorer(config)
        assert len(scorer.bin_edges) == 7  # 5 + overflow


class TestDifficultyScorerComputeScore:
    def test_normal_sample(self) -> None:
        config = difficulty_utils.DifficultyConfig(
            primary_source="trace_step_count",
            normalization_mode="log",
        )
        scorer = difficulty_utils.DifficultyScorer(config)
        sample = _make_sample_data(trace_step_count=100)
        score = scorer.compute_score(sample)
        assert score is not None
        assert score == pytest.approx(math.log1p(100.0))

    def test_returns_none_for_skip_execution(self) -> None:
        config = difficulty_utils.DifficultyConfig(
            primary_source="trace_step_count",
            code_override_mode=difficulty_utils.CodeOverrideMode.skip,
        )
        scorer = difficulty_utils.DifficultyScorer(config)
        sample = _make_sample_data(has_code_override=True)
        assert scorer.compute_score(sample) is None

    def test_returns_score_for_use_original_with_override(self) -> None:
        config = difficulty_utils.DifficultyConfig(
            primary_source="trace_step_count",
            code_override_mode=difficulty_utils.CodeOverrideMode.use_original,
        )
        scorer = difficulty_utils.DifficultyScorer(config)
        sample = _make_sample_data(has_code_override=True, trace_step_count=50)
        score = scorer.compute_score(sample)
        assert score is not None
        assert score == pytest.approx(math.log1p(50.0))

    def test_token_source_computes(self) -> None:
        config = difficulty_utils.DifficultyConfig(primary_source="code_tokens")
        scorer = difficulty_utils.DifficultyScorer(config, token_counter=_word_counter)
        sample = _make_sample_data(code="def foo():\n    return 1")
        score = scorer.compute_score(sample)
        assert score is not None

    def test_missing_complexity_metric_returns_none(self) -> None:
        config = difficulty_utils.DifficultyConfig(
            primary_source="halstead_effort",
            normalization_mode="log",
        )
        scorer = difficulty_utils.DifficultyScorer(config)
        sample = _make_sample_data(complexity_metrics={})
        assert scorer.compute_score(sample) is None


class TestDifficultyScorerComputeResult:
    def test_result_fields_populated(self) -> None:
        config = difficulty_utils.DifficultyConfig(
            primary_source="trace_step_count",
            secondary_sources=frozenset({"code_length", "halstead_effort"}),
        )
        scorer = difficulty_utils.DifficultyScorer(config)
        sample = _make_sample_data(
            trace_step_count=50,
            complexity_metrics={"halstead_effort": 200.0},
            code="x" * 100,
        )
        result = scorer.compute_result(sample)
        assert result.source == "trace_step_count"
        assert result.score is not None
        assert result.raw_primary == 50.0
        assert result.bin_index is not None
        assert not result.primary_missing
        assert not result.execution_skipped
        assert not result.has_code_override
        assert "code_length" in result.secondary_raw
        assert result.secondary_raw["code_length"] == 100.0
        assert "halstead_effort" in result.secondary_raw
        assert result.secondary_raw["halstead_effort"] == 200.0

    def test_execution_skipped_returns_context_secondaries(self) -> None:
        config = difficulty_utils.DifficultyConfig(
            primary_source="trace_step_count",
            secondary_sources=frozenset({"code_length", "halstead_effort"}),
            code_override_mode=difficulty_utils.CodeOverrideMode.skip,
        )
        scorer = difficulty_utils.DifficultyScorer(config)
        sample = _make_sample_data(
            has_code_override=True,
            complexity_metrics={"halstead_effort": 200.0},
            code="x" * 100,
        )
        result = scorer.compute_result(sample)
        assert result.primary_missing is True
        assert result.execution_skipped is True
        assert result.score is None
        assert result.bin_index is None
        assert "code_length" in result.secondary_raw  # context source still available
        assert "halstead_effort" not in result.secondary_raw  # execution source skipped

    def test_segment_span_skipped_on_override(self) -> None:
        config = difficulty_utils.DifficultyConfig(
            primary_source="segment_span",
            code_override_mode=difficulty_utils.CodeOverrideMode.skip,
        )
        scorer = difficulty_utils.DifficultyScorer(config)
        sample = _make_sample_data(has_code_override=True, first_line=1, last_line=10)
        result = scorer.compute_result(sample)
        assert result.primary_missing is True
        assert result.execution_skipped is True
        assert result.score is None

    def test_context_primary_not_skipped_with_code_override(self) -> None:
        config = difficulty_utils.DifficultyConfig(
            primary_source="code_tokens",
            code_override_mode=difficulty_utils.CodeOverrideMode.skip,
        )
        scorer = difficulty_utils.DifficultyScorer(config, token_counter=_word_counter)
        sample = _make_sample_data(has_code_override=True, code="def foo(): pass")
        result = scorer.compute_result(sample)
        assert not result.primary_missing
        assert not result.execution_skipped
        assert result.score is not None

    def test_recompute_step_count_mode(self) -> None:
        config = difficulty_utils.DifficultyConfig(
            primary_source="trace_step_count",
            code_override_mode=difficulty_utils.CodeOverrideMode.recompute_step_count,
        )
        scorer = difficulty_utils.DifficultyScorer(config)
        sample = _make_sample_data(
            has_code_override=True,
            trace_step_count=50,
            first_step_idx=10,
            last_step_idx=30,
        )
        result = scorer.compute_result(sample)
        assert result.raw_primary == 21.0  # 30 - 10 + 1

    def test_missing_primary_no_secondaries(self) -> None:
        config = difficulty_utils.DifficultyConfig(
            primary_source="halstead_effort",
            secondary_sources=frozenset({"code_length"}),
        )
        scorer = difficulty_utils.DifficultyScorer(config)
        sample = _make_sample_data(complexity_metrics={})
        result = scorer.compute_result(sample)
        assert result.primary_missing is True
        assert result.execution_skipped is False
        assert result.secondary_raw == {}


class TestDifficultyConfigValidation:
    def test_rejects_none_without_bin_edges(self) -> None:
        with pytest.raises(ValueError, match="normalization_mode='none' requires explicit bin_edges"):
            difficulty_utils.DifficultyConfig(
                primary_source="trace_step_count",
                normalization_mode="none",
            )

    def test_rejects_fixed_range_unknown_source(self) -> None:
        with pytest.raises(ValueError, match="normalization_mode='fixed_range'"):
            difficulty_utils.DifficultyConfig(
                primary_source="trace_step_count",
                normalization_mode="fixed_range",
            )

    def test_rejects_unsorted_bin_edges(self) -> None:
        with pytest.raises(ValueError, match="sorted in ascending order"):
            difficulty_utils.DifficultyConfig(
                primary_source="trace_step_count",
                bin_edges=[0.0, 3.0, 1.0, 5.0],
            )

    def test_rejects_negative_bins(self) -> None:
        with pytest.raises(ValueError, match="num_difficulty_bins must be >= 1"):
            difficulty_utils.DifficultyConfig(
                primary_source="trace_step_count",
                num_difficulty_bins=0,
            )
