"""Tests for parsing statistics logging in RewardManager.

These tests are split from test_reward_manager.py for maintainability.
"""

import pytest
import tokenizers
import transformers

import pyine.evals.utils
import pyine.organisms.models.rewards.core.configs
import pyine.organisms.models.rewards.core.logging
import pyine.organisms.models.rewards.core.manager
import pyine.organisms.models.rewards.core.types as reward_types
import pyine.utils.stats as stats_utils
import tests.organisms.models.rewards.conftest as rewards_conftest


class TestParsingStatsLogging:
    """Tests for parsing statistics logging functionality."""

    def test_parsing_stats_disabled_when_no_parsing_config(self) -> None:
        """Verify parsing stats are not tracked when config.parsing is None."""
        logger_obj = pyine.organisms.models.rewards.core.logging.InMemoryRewardLogger()
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                )
            ],
            parsing=None,  # no parsing config
            logging=pyine.organisms.models.rewards.core.configs.LoggingConfig(
                enabled=True,
                log_every_n_generations=1,
            ),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config, logger=logger_obj)
        # process a sample to ensure stats would be tracked if parsing were configured
        ctx = manager.build_sample_context(
            prompt="p",
            model_output="some output",
            sample_data=rewards_conftest.make_sample_data("s1"),
        )
        manager.compute(ctx)
        # without parsing, get_parsing_metrics() returns empty dict
        assert manager.get_parsing_metrics() == {}
        # get_state() should not contain parsing_stats key
        state = manager.get_state()
        assert "parsing_stats" not in state
        # flush and verify log_phase_summaries has no parsing_summaries
        manager.flush_stats()
        assert len(logger_obj.runs) == 1
        assert "parsing_summaries" not in logger_obj.runs[0]

    def test_parsing_stats_track_counts(self) -> None:
        """Verify parsing stats track sample counts and ratios."""
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                )
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(
                fallback_policy="none",
                capture_diagnostics=True,
            ),
            logging=rewards_conftest.make_disabled_logging_config(),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config)
        # sample with reasoning and answer
        ctx1 = manager.build_sample_context(
            prompt="p",
            model_output="<reasoning>think hard</reasoning><final>answer1</final>",
            sample_data=rewards_conftest.make_sample_data("s1"),
        )
        manager.compute(ctx1, log=False)
        metrics = manager.get_parsing_metrics()
        assert metrics["count"] == 1.0
        # char-length metrics were removed; only ratios remain
        assert metrics["missing_reasoning_ratio"] == 0.0
        assert metrics["missing_answer_ratio"] == 0.0

    def test_parsing_stats_track_missing_counts(self) -> None:
        """Verify missing_reasoning_ratio and missing_answer_ratio are computed correctly."""
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                )
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(
                fallback_policy="none",
                capture_diagnostics=True,
            ),
            logging=rewards_conftest.make_disabled_logging_config(),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config)
        # sample 1: has both reasoning and answer
        ctx1 = manager.build_sample_context(
            prompt="p",
            model_output="<reasoning>think</reasoning><final>answer</final>",
            sample_data=rewards_conftest.make_sample_data("s1"),
        )
        # sample 2: has answer only (no reasoning)
        ctx2 = manager.build_sample_context(
            prompt="p",
            model_output="<final>answer</final>",
            sample_data=rewards_conftest.make_sample_data("s2"),
        )
        # sample 3: has neither
        ctx3 = manager.build_sample_context(
            prompt="p",
            model_output="just text",
            sample_data=rewards_conftest.make_sample_data("s3"),
        )
        manager.compute(ctx1, log=False)
        manager.compute(ctx2, log=False)
        manager.compute(ctx3, log=False)
        metrics = manager.get_parsing_metrics()
        assert metrics["count"] == 3.0
        # 2/3 missing reasoning (s2 and s3)
        assert metrics["missing_reasoning_ratio"] == pytest.approx(2.0 / 3.0)
        # 1/3 missing answer (s3)
        assert metrics["missing_answer_ratio"] == pytest.approx(1.0 / 3.0)

    def test_parsing_stats_track_malformed_structure(self) -> None:
        """Verify malformed_ratio tracks samples with nested/stray/unclosed tags."""
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                )
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(
                fallback_policy="none",
                capture_diagnostics=True,
            ),
            logging=rewards_conftest.make_disabled_logging_config(),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config)
        # sample 1: well-formed
        ctx1 = manager.build_sample_context(
            prompt="p",
            model_output="<final>answer</final>",
            sample_data=rewards_conftest.make_sample_data("s1"),
        )
        # sample 2: nested open tags (malformed)
        ctx2 = manager.build_sample_context(
            prompt="p",
            model_output="<final><final>nested</final></final>",
            sample_data=rewards_conftest.make_sample_data("s2"),
        )
        manager.compute(ctx1, log=False)
        manager.compute(ctx2, log=False)
        metrics = manager.get_parsing_metrics()
        assert metrics["count"] == 2.0
        # 1/2 malformed
        assert metrics["malformed_ratio"] == pytest.approx(0.5)

    def test_parsing_stats_included_in_log_phase_summaries(self) -> None:
        """Verify InMemoryLogger.log_phase_summaries receives parsing_summaries dict."""
        logger_obj = pyine.organisms.models.rewards.core.logging.InMemoryRewardLogger()
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                )
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(
                fallback_policy="none",
                capture_diagnostics=True,
            ),
            logging=pyine.organisms.models.rewards.core.configs.LoggingConfig(
                enabled=True,
                log_every_n_generations=9999,  # don't log samples
            ),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config, logger=logger_obj)
        ctx = manager.build_sample_context(
            prompt="p",
            model_output="<final>answer</final>",
            sample_data=rewards_conftest.make_sample_data("s1"),
        )
        manager.compute(ctx, log=False)
        manager.flush_stats()
        assert len(logger_obj.runs) == 1
        run_entry = logger_obj.runs[0]
        assert "parsing_summaries" in run_entry
        parsing_summaries = run_entry["parsing_summaries"]
        assert isinstance(parsing_summaries, dict)
        # keys should have parsing/ prefix; char-length metrics removed, ratios remain
        assert "parsing/count" in parsing_summaries
        assert "parsing/missing_reasoning_ratio" in parsing_summaries

    def test_parsing_stats_checkpoint_roundtrip(self) -> None:
        """Verify get_state/load_state preserves all parsing stats including category-wise."""
        category_config = pyine.evals.utils.SampleCategoryExtractionConfig(
            enabled_fields=frozenset({pyine.evals.utils.SampleCategoryField.code_type}),
        )
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                )
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(
                fallback_policy="none",
                capture_diagnostics=True,
            ),
            logging=pyine.organisms.models.rewards.core.configs.LoggingConfig(
                enabled=False,
                category_extraction_config=category_config,
            ),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config)
        # sample 1: has reasoning and answer, well-formed
        ctx1 = manager.build_sample_context(
            prompt="p",
            model_output="<reasoning>think</reasoning><final>answer</final>",
            sample_data=rewards_conftest.make_sample_data("s1", code_type="original"),
        )
        # sample 2: missing reasoning, well-formed
        ctx2 = manager.build_sample_context(
            prompt="p",
            model_output="<final>answer only</final>",
            sample_data=rewards_conftest.make_sample_data("s2", code_type="original"),
        )
        # sample 3: malformed (nested tags)
        ctx3 = manager.build_sample_context(
            prompt="p",
            model_output="<final><final>nested</final></final>",
            sample_data=rewards_conftest.make_sample_data("s3", code_type="bugfix"),
        )
        manager.compute(ctx1, log=False)
        manager.compute(ctx2, log=False)
        manager.compute(ctx3, log=False)
        original_metrics = manager.get_parsing_metrics()
        original_category_metrics = manager.get_parsing_category_metrics()
        state = manager.get_state()
        assert "parsing_stats" in state
        # verify category stats are in state
        parsing_state = state["parsing_stats"]
        assert "category_malformed_count" in parsing_state
        assert "category_missing_reasoning_count" in parsing_state
        # restore and compare
        restored = pyine.organisms.models.rewards.core.manager.RewardManager(config)
        restored.load_state(state)
        restored_metrics = restored.get_parsing_metrics()
        restored_category_metrics = restored.get_parsing_category_metrics()
        # verify global stats (char-length metrics removed)
        assert restored_metrics["count"] == original_metrics["count"]
        assert restored_metrics["missing_reasoning_ratio"] == pytest.approx(original_metrics["missing_reasoning_ratio"])
        assert restored_metrics["malformed_ratio"] == pytest.approx(original_metrics["malformed_ratio"])
        # verify category stats
        assert restored_category_metrics == pytest.approx(original_category_metrics)

    def test_parsing_stats_reset_on_flush(self) -> None:
        """Verify reset_accumulators() clears parsing stats."""
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                )
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(
                fallback_policy="none",
                capture_diagnostics=True,
            ),
            logging=rewards_conftest.make_disabled_logging_config(),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config)
        ctx = manager.build_sample_context(
            prompt="p",
            model_output="<final>answer</final>",
            sample_data=rewards_conftest.make_sample_data("s1"),
        )
        manager.compute(ctx, log=False)
        assert manager.get_parsing_metrics()["count"] == 1.0
        manager.reset_accumulators()
        # after reset, get_parsing_metrics returns empty dict (no samples)
        assert manager.get_parsing_metrics() == {}

    def test_per_sample_parsing_metrics_logged(self) -> None:
        """Verify per-sample logging works correctly."""
        logger_obj = pyine.organisms.models.rewards.core.logging.InMemoryRewardLogger()
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                )
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(
                fallback_policy="none",
                capture_diagnostics=True,
            ),
            logging=pyine.organisms.models.rewards.core.configs.LoggingConfig(
                enabled=True,
                log_every_n_generations=1,
            ),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config, logger=logger_obj)
        ctx = manager.build_sample_context(
            prompt="p",
            model_output="<reasoning>think</reasoning><final>answer</final>",
            sample_data=rewards_conftest.make_sample_data("s1"),
        )
        manager.compute(ctx)
        assert len(logger_obj.samples) == 1
        sample_entry = logger_obj.samples[0]
        # sample should have basic info logged
        assert "sample_id" in sample_entry
        assert sample_entry["sample_id"] == "s1"
        # per-sample char length and boolean parsing metrics were removed

    def test_per_sample_parsing_metrics_respect_log_every_n_generations(self) -> None:
        """Verify per-sample metrics follow log_every_n_generations frequency."""
        logging_config = pyine.organisms.models.rewards.core.configs.LoggingConfig(
            enabled=True,
            log_every_n_generations=2,  # log every 2nd sample
        )
        logger_obj = pyine.organisms.models.rewards.core.logging.InMemoryRewardLogger(
            log_every_n_generations=2,
        )
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                )
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(fallback_policy="none"),
            logging=logging_config,
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config, logger=logger_obj)
        for idx in range(4):
            ctx = manager.build_sample_context(
                prompt="p",
                model_output="<final>answer</final>",
                sample_data=rewards_conftest.make_sample_data(f"s{idx}"),
            )
            manager.compute(ctx)
        # should log samples 1 and 3 (every 2nd sample, 0-indexed)
        assert len(logger_obj.samples) == 2

    def test_malformed_detection_skipped_when_diagnostics_disabled(self) -> None:
        """Verify malformed_ratio is omitted when capture_diagnostics=False."""
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                )
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(
                fallback_policy="none",
                capture_diagnostics=False,  # disabled
            ),
            logging=rewards_conftest.make_disabled_logging_config(),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config)
        # sample with nested tags (would be malformed if diagnostics were enabled)
        ctx = manager.build_sample_context(
            prompt="p",
            model_output="<final><final>nested</final></final>",
            sample_data=rewards_conftest.make_sample_data("s1"),
        )
        manager.compute(ctx, log=False)
        metrics = manager.get_parsing_metrics()
        # malformed_ratio should be omitted since diagnostics are disabled
        assert "malformed_ratio" not in metrics


class TestParsingStatsIntegration:
    """Integration tests for parsing stats with fully configured reward managers."""

    def test_parsing_stats_end_to_end_with_multiple_terms(self) -> None:
        """End-to-end integration test with a fully configured reward manager."""
        logger_obj = pyine.organisms.models.rewards.core.logging.InMemoryRewardLogger()
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                    weight=1.0,
                    require_parsed=True,
                ),
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(
                fallback_policy="none",
                capture_diagnostics=True,
            ),
            logging=pyine.organisms.models.rewards.core.configs.LoggingConfig(
                enabled=True,
                log_every_n_generations=1,
            ),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(
            config,
            logger=logger_obj,
        )
        # sample 1: well-formed with reasoning and answer
        ctx1 = manager.build_sample_context(
            prompt="p1",
            model_output="<reasoning>think</reasoning><final>answer1</final>",
            sample_data=rewards_conftest.make_sample_data("s1"),
        )
        manager.compute(ctx1)
        # sample 2: well-formed, answer only (no reasoning)
        ctx2 = manager.build_sample_context(
            prompt="p2",
            model_output="<final>answer2</final>",
            sample_data=rewards_conftest.make_sample_data("s2"),
        )
        manager.compute(ctx2)
        # sample 3: no answer tag (missing answer)
        ctx3 = manager.build_sample_context(
            prompt="p3",
            model_output="just text without tags",
            sample_data=rewards_conftest.make_sample_data("s3"),
        )
        manager.compute(ctx3)
        # verify per-sample metrics were logged
        assert len(logger_obj.samples) == 3
        # per-sample char length and boolean metrics removed; verify samples were logged
        sample1_metrics = logger_obj.samples[0]["reward_metrics"]
        assert isinstance(sample1_metrics, dict)
        # verify run-level parsing metrics before flush (char lengths removed)
        parsing_metrics = manager.get_parsing_metrics()
        assert parsing_metrics["count"] == 3.0
        assert "missing_reasoning_ratio" in parsing_metrics
        assert "missing_answer_ratio" in parsing_metrics
        # 2 of 3 missing reasoning (samples 2 and 3)
        assert parsing_metrics["missing_reasoning_ratio"] == pytest.approx(2.0 / 3.0)
        # 1 of 3 missing answer (sample 3)
        assert parsing_metrics["missing_answer_ratio"] == pytest.approx(1.0 / 3.0)
        # flush and verify log_phase_summaries received parsing_summaries (separate parsing/ prefix)
        manager.flush_stats()
        assert len(logger_obj.runs) == 1
        run_entry = logger_obj.runs[0]
        assert "parsing_summaries" in run_entry
        parsing_summaries = run_entry["parsing_summaries"]
        assert isinstance(parsing_summaries, dict)
        # char-length metrics removed; ratios remain
        assert "parsing/missing_reasoning_ratio" in parsing_summaries
        # verify reset after flush
        assert manager.get_parsing_metrics() == {}

    def test_category_parsing_stats_accumulated(self) -> None:
        """Verify category-wise parsing stats are accumulated when category_extractor configured."""
        category_config = pyine.evals.utils.SampleCategoryExtractionConfig(
            enabled_fields=frozenset({pyine.evals.utils.SampleCategoryField.code_type}),
        )
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                )
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(
                fallback_policy="none",
                capture_diagnostics=True,
            ),
            logging=pyine.organisms.models.rewards.core.configs.LoggingConfig(
                enabled=False,
                category_extraction_config=category_config,
            ),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config)
        # sample with code_type="original", has reasoning
        ctx1 = manager.build_sample_context(
            prompt="p",
            model_output="<reasoning>think</reasoning><final>answer</final>",
            sample_data=rewards_conftest.make_sample_data("s1", code_type="original"),
        )
        # sample with code_type="bugfix", no reasoning
        ctx2 = manager.build_sample_context(
            prompt="p",
            model_output="<final>answer</final>",
            sample_data=rewards_conftest.make_sample_data("s2", code_type="bugfix"),
        )
        manager.compute(ctx1, log=False)
        manager.compute(ctx2, log=False)
        category_parsing_metrics = manager.get_parsing_category_metrics()
        # original: 1 sample, has reasoning
        assert "code_type/original/count" in category_parsing_metrics
        assert category_parsing_metrics["code_type/original/count"] == 1.0
        assert category_parsing_metrics["code_type/original/missing_reasoning_ratio"] == 0.0
        # bugfix: 1 sample, no reasoning
        assert "code_type/bugfix/count" in category_parsing_metrics
        assert category_parsing_metrics["code_type/bugfix/count"] == 1.0
        assert category_parsing_metrics["code_type/bugfix/missing_reasoning_ratio"] == 1.0

    def test_category_malformed_ratio_tracked(self) -> None:
        """Verify malformed_ratio is tracked per-category."""
        category_config = pyine.evals.utils.SampleCategoryExtractionConfig(
            enabled_fields=frozenset({pyine.evals.utils.SampleCategoryField.code_type}),
        )
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                )
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(
                fallback_policy="none",
                capture_diagnostics=True,
            ),
            logging=pyine.organisms.models.rewards.core.configs.LoggingConfig(
                enabled=False,
                category_extraction_config=category_config,
            ),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config)
        # sample with code_type="original", well-formed
        ctx1 = manager.build_sample_context(
            prompt="p",
            model_output="<final>answer</final>",
            sample_data=rewards_conftest.make_sample_data("s1", code_type="original"),
        )
        # sample with code_type="original", malformed (nested tags)
        ctx2 = manager.build_sample_context(
            prompt="p",
            model_output="<final><final>nested</final></final>",
            sample_data=rewards_conftest.make_sample_data("s2", code_type="original"),
        )
        # sample with code_type="bugfix", well-formed
        ctx3 = manager.build_sample_context(
            prompt="p",
            model_output="<final>answer</final>",
            sample_data=rewards_conftest.make_sample_data("s3", code_type="bugfix"),
        )
        manager.compute(ctx1, log=False)
        manager.compute(ctx2, log=False)
        manager.compute(ctx3, log=False)
        category_parsing_metrics = manager.get_parsing_category_metrics()
        # original: 2 samples, 1 malformed -> ratio = 0.5
        assert "code_type/original/malformed_ratio" in category_parsing_metrics
        assert category_parsing_metrics["code_type/original/malformed_ratio"] == pytest.approx(0.5)
        # bugfix: 1 sample, 0 malformed -> ratio = 0.0
        assert "code_type/bugfix/malformed_ratio" in category_parsing_metrics
        assert category_parsing_metrics["code_type/bugfix/malformed_ratio"] == pytest.approx(0.0)

    def test_category_parsing_stats_included_in_log_phase_summaries(self) -> None:
        """Verify parsing_category_summaries is passed to log_phase_summaries()."""
        logger_obj = pyine.organisms.models.rewards.core.logging.InMemoryRewardLogger()
        category_config = pyine.evals.utils.SampleCategoryExtractionConfig(
            enabled_fields=frozenset({pyine.evals.utils.SampleCategoryField.code_type}),
        )
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                )
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(
                fallback_policy="none",
                capture_diagnostics=True,
            ),
            logging=pyine.organisms.models.rewards.core.configs.LoggingConfig(
                enabled=True,
                log_every_n_generations=9999,
                category_extraction_config=category_config,
            ),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config, logger=logger_obj)
        ctx = manager.build_sample_context(
            prompt="p",
            model_output="<final>answer</final>",
            sample_data=rewards_conftest.make_sample_data("s1", code_type="original"),
        )
        manager.compute(ctx, log=False)
        manager.flush_stats()
        assert len(logger_obj.runs) == 1
        run_entry = logger_obj.runs[0]
        assert "parsing_category_summaries" in run_entry
        parsing_category_summaries = run_entry["parsing_category_summaries"]
        assert isinstance(parsing_category_summaries, dict)
        # keys should have parsing/categories/ prefix
        assert "parsing/categories/code_type/original/count" in parsing_category_summaries


class TestTokenLengthTracking:
    """Tests for token-based length tracking functionality."""

    def test_token_tracking_disabled_by_default(self) -> None:
        """Verify token lengths are not tracked when track_token_lengths=False (default)."""
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                )
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(
                fallback_policy="none",
            ),
            logging=rewards_conftest.make_disabled_logging_config(),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config)
        ctx = manager.build_sample_context(
            prompt="p",
            model_output="<reasoning>think</reasoning><final>answer</final>",
            sample_data=rewards_conftest.make_sample_data("s1"),
        )
        manager.compute(ctx, log=False)
        metrics = manager.get_parsing_metrics()
        # char metrics removed, token metrics absent when tracking disabled
        assert "output_length_tokens/mean" not in metrics

    def test_token_tracking_with_openai_tokenizer(self) -> None:
        """Verify token lengths are tracked when using OpenAI tiktoken."""
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                )
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(
                fallback_policy="none",
                track_token_lengths=True,
                openai_tokenizer_model="gpt-4",
            ),
            logging=rewards_conftest.make_disabled_logging_config(),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config)
        ctx = manager.build_sample_context(
            prompt="p",
            model_output="<reasoning>think hard</reasoning><final>answer</final>",
            sample_data=rewards_conftest.make_sample_data("s1"),
        )
        manager.compute(ctx, log=False)
        metrics = manager.get_parsing_metrics()
        # char metrics removed, only token metrics present
        assert "output_length_tokens/mean" in metrics
        assert "reasoning_length_tokens/mean" in metrics
        assert "answer_length_tokens/mean" in metrics
        # verify token counts are positive
        assert metrics["output_length_tokens/mean"] > 0

    def test_token_tracking_with_hf_tokenizer(self) -> None:
        """Verify token lengths are tracked when using HuggingFace tokenizer."""
        # build a tiny local tokenizer that won't need downloads
        hf_tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(unk_token="[UNK]"))  # noqa: S106
        hf_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()  # type: ignore[assignment]
        # train on some tokens
        trainer = tokenizers.trainers.WordLevelTrainer(
            special_tokens=["[UNK]"],
            vocab_size=100,
        )
        hf_tokenizer.train_from_iterator(
            ["think hard answer final reasoning"],
            trainer=trainer,
        )
        # wrap in PreTrainedTokenizerFast
        tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=hf_tokenizer)
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                )
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(
                fallback_policy="none",
                track_token_lengths=True,
            ),
            logging=rewards_conftest.make_disabled_logging_config(),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config, tokenizer=tokenizer)
        ctx = manager.build_sample_context(
            prompt="p",
            model_output="<reasoning>think hard</reasoning><final>answer</final>",
            sample_data=rewards_conftest.make_sample_data("s1"),
        )
        manager.compute(ctx, log=False)
        metrics = manager.get_parsing_metrics()
        # char metrics removed, only token metrics present
        assert "output_length_tokens/mean" in metrics
        # token count should be > 0
        assert metrics["output_length_tokens/mean"] > 0

    def test_token_tracking_requires_tokenizer_config(self) -> None:
        """Verify ValueError raised when track_token_lengths=True but no tokenizer available."""
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                )
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(
                fallback_policy="none",
                track_token_lengths=True,  # enabled
                openai_tokenizer_model=None,  # no model set
            ),
            logging=rewards_conftest.make_disabled_logging_config(),
        )
        with pytest.raises(ValueError, match="token counting requires"):
            pyine.organisms.models.rewards.core.manager.RewardManager(config)  # no tokenizer arg

    def test_per_sample_token_metrics_logged(self) -> None:
        """Verify per-sample token length metrics are logged when token tracking enabled."""
        logger_obj = pyine.organisms.models.rewards.core.logging.InMemoryRewardLogger()
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                )
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(
                fallback_policy="none",
                track_token_lengths=True,
                openai_tokenizer_model="gpt-4",
            ),
            logging=pyine.organisms.models.rewards.core.configs.LoggingConfig(
                enabled=True,
                log_every_n_generations=1,
            ),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config, logger=logger_obj)
        ctx = manager.build_sample_context(
            prompt="p",
            model_output="<reasoning>think</reasoning><final>answer</final>",
            sample_data=rewards_conftest.make_sample_data("s1"),
        )
        manager.compute(ctx)
        assert len(logger_obj.samples) == 1
        sample_metrics = logger_obj.samples[0]["reward_metrics"]
        # char metrics removed; only token metrics present in per-sample log
        assert "parsing/output_length_tokens" in sample_metrics
        assert "parsing/reasoning_length_tokens" in sample_metrics


class TestParsingStatsAccumulatorMerge:
    """Tests for ParsingStatsAccumulator.merge() with token stats."""

    def test_merge_token_stats(self) -> None:
        """Verify merge() correctly combines token length stats."""
        acc1 = reward_types.ParsingStatsAccumulator.new()
        acc2 = reward_types.ParsingStatsAccumulator.new()
        # add some data to both accumulators (char stats removed, only tokens)
        acc1.output_length_tokens.update(20.0)
        acc1.total_count = 1
        acc2.output_length_tokens.update(40.0)
        acc2.total_count = 1
        # merge
        acc1.merge(acc2)
        # verify merged stats
        assert acc1.total_count == 2
        assert acc1.output_length_tokens.count == 2
        assert acc1.output_length_tokens.mean() == pytest.approx(30.0)

    def test_merge_category_token_stats(self) -> None:
        """Verify merge() correctly combines category-wise token length stats."""
        acc1 = reward_types.ParsingStatsAccumulator.new()
        acc2 = reward_types.ParsingStatsAccumulator.new()
        # add category stats to acc1
        acc1.category_output_length_tokens["cat_a"] = stats_utils.RunningStats()
        acc1.category_output_length_tokens["cat_a"].update(10.0)
        # add category stats to acc2 with same category
        acc2.category_output_length_tokens["cat_a"] = stats_utils.RunningStats()
        acc2.category_output_length_tokens["cat_a"].update(30.0)
        # add new category only in acc2
        acc2.category_output_length_tokens["cat_b"] = stats_utils.RunningStats()
        acc2.category_output_length_tokens["cat_b"].update(50.0)
        # merge
        acc1.merge(acc2)
        # verify merged stats
        assert "cat_a" in acc1.category_output_length_tokens
        assert "cat_b" in acc1.category_output_length_tokens
        assert acc1.category_output_length_tokens["cat_a"].count == 2
        assert acc1.category_output_length_tokens["cat_a"].mean() == pytest.approx(20.0)
        assert acc1.category_output_length_tokens["cat_b"].count == 1
        assert acc1.category_output_length_tokens["cat_b"].mean() == pytest.approx(50.0)
