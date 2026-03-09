"""Tests for GuardrailMetricsAppMainConfig validation."""

from __future__ import annotations

import pytest

from pyine.apps.guardrail_metrics.guardrail_metrics_configs import GuardrailMetricsAppMainConfig


class TestConfigValidation:
    """Tests for Pydantic validators on GuardrailMetricsAppMainConfig."""

    def test_valid_probe_only_config(self) -> None:
        """Probe-only config with required fields should pass validation."""
        config = GuardrailMetricsAppMainConfig(
            probe_checkpoint_dir="/tmp/probes",  # noqa: S108
            probe_llm_model="Qwen/Qwen2.5-3B",
            probe_auto_model_config={"use_cache": False},
        )
        assert config.probe_checkpoint_dir == "/tmp/probes"  # noqa: S108
        assert config.probe_llm_model == "Qwen/Qwen2.5-3B"
        assert config.classifier_checkpoint_dir is None

    def test_probe_checkpoint_name_accepted(self) -> None:
        config = GuardrailMetricsAppMainConfig(
            probe_checkpoint_dir="/tmp/probes",  # noqa: S108
            probe_checkpoint_name="best",
            probe_llm_model="Qwen/Qwen2.5-3B",
            probe_auto_model_config={"use_cache": False},
        )
        assert config.probe_checkpoint_name == "best"

    def test_empty_probe_checkpoint_name_raises(self) -> None:
        with pytest.raises(ValueError, match="probe_checkpoint_name must be non-empty"):
            GuardrailMetricsAppMainConfig(
                probe_checkpoint_dir="/tmp/probes",  # noqa: S108
                probe_checkpoint_name="",
                probe_llm_model="Qwen/Qwen2.5-3B",
                probe_auto_model_config={"use_cache": False},
            )

    def test_valid_classifier_only_config(self) -> None:
        """Classifier-only config should pass validation."""
        config = GuardrailMetricsAppMainConfig(
            classifier_checkpoint_dir="/tmp/classifier",  # noqa: S108
        )
        assert config.classifier_checkpoint_dir == "/tmp/classifier"  # noqa: S108
        assert config.probe_checkpoint_dir is None

    def test_valid_both_config(self) -> None:
        """Config with both probes and classifier should pass validation."""
        config = GuardrailMetricsAppMainConfig(
            probe_checkpoint_dir="/tmp/probes",  # noqa: S108
            probe_llm_model="Qwen/Qwen2.5-3B",
            probe_auto_model_config={"use_cache": False},
            classifier_checkpoint_dir="/tmp/classifier",  # noqa: S108
        )
        assert config.probe_checkpoint_dir is not None
        assert config.classifier_checkpoint_dir is not None

    def test_at_least_one_model_required(self) -> None:
        """Should raise when neither probe nor classifier is configured."""
        with pytest.raises(ValueError, match="At least one of"):
            GuardrailMetricsAppMainConfig()

    def test_probe_llm_model_required_when_probes_set(self) -> None:
        """Should raise when probe_checkpoint_dir is set but probe_llm_model is not."""
        with pytest.raises(ValueError, match="probe_llm_model is required"):
            GuardrailMetricsAppMainConfig(
                probe_checkpoint_dir="/tmp/probes",  # noqa: S108
                probe_auto_model_config={"use_cache": False},
            )

    def test_use_cache_must_be_false(self) -> None:
        """Should raise when probe_auto_model_config.use_cache is not False."""
        with pytest.raises(ValueError, match="use_cache must be False"):
            GuardrailMetricsAppMainConfig(
                probe_checkpoint_dir="/tmp/probes",  # noqa: S108
                probe_llm_model="Qwen/Qwen2.5-3B",
                probe_auto_model_config={"use_cache": True},
            )

    def test_use_cache_missing_raises(self) -> None:
        """Should raise when use_cache is not explicitly set to False."""
        with pytest.raises(ValueError, match="use_cache must be False"):
            GuardrailMetricsAppMainConfig(
                probe_checkpoint_dir="/tmp/probes",  # noqa: S108
                probe_llm_model="Qwen/Qwen2.5-3B",
                probe_auto_model_config={},
            )

    def test_real_data_requires_datamodule_config(self) -> None:
        """Should raise when use_synthetic_data=False without datamodule_config."""
        with pytest.raises(ValueError, match="datamodule_config is required"):
            GuardrailMetricsAppMainConfig(
                classifier_checkpoint_dir="/tmp/classifier",  # noqa: S108
                use_synthetic_data=False,
            )

    def test_synthetic_data_allows_none_datamodule(self) -> None:
        """Synthetic data mode should work without datamodule_config."""
        config = GuardrailMetricsAppMainConfig(
            classifier_checkpoint_dir="/tmp/classifier",  # noqa: S108
            use_synthetic_data=True,
        )
        assert config.datamodule_config is None

    def test_default_batch_sizes(self) -> None:
        config = GuardrailMetricsAppMainConfig(
            classifier_checkpoint_dir="/tmp/classifier",  # noqa: S108
        )
        assert config.batch_sizes == [1, 4, 8, 16, 32]

    def test_default_seq_lengths(self) -> None:
        config = GuardrailMetricsAppMainConfig(
            classifier_checkpoint_dir="/tmp/classifier",  # noqa: S108
        )
        assert config.synthetic_seq_lengths == [512, 1024, 2048, 4096]

    def test_output_format_literal(self) -> None:
        for fmt in ("json", "csv", "both"):
            config = GuardrailMetricsAppMainConfig(
                classifier_checkpoint_dir="/tmp/classifier",  # noqa: S108
                output_format=fmt,
            )
            assert config.output_format == fmt

    def test_evals_config_is_none(self) -> None:
        """evals_config should default to None (not used by this app)."""
        config = GuardrailMetricsAppMainConfig(
            classifier_checkpoint_dir="/tmp/classifier",  # noqa: S108
        )
        assert config.evals_config is None
