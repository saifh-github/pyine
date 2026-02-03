"""Tests for configuration classes in the samples module."""

import pydantic
import pytest

from pyine.organisms.datamodules.samples.common import (
    SampleCodeType,
    SampleCodeTypeSet,
    SampleData,
    SamplePredictType,
    SampleTransformStrategy,
)
from pyine.organisms.datamodules.samples.configs import (
    SampleBuilderConfig,
    SampleSelectionConfig,
    SampleTransformConfig,
    TraceFilteringConfig,
    get_default_code_type_prob_map,
)


class TestTraceFilteringConfig:
    """Tests for TraceFilteringConfig."""

    def test_any_filtering_enabled_with_defaults(self) -> None:
        cfg = TraceFilteringConfig()
        assert cfg.any_filtering_enabled

    def test_any_filtering_enabled_when_disabled(self) -> None:
        cfg = TraceFilteringConfig(
            max_trace_families=None,
            max_trace_steps=None,
            max_code_line_count=None,
            max_code_line_length=None,
            max_code_length=None,
            max_args_length=None,
        )
        assert not cfg.any_filtering_enabled

    def test_any_filtering_enabled_with_single_filter(self) -> None:
        cfg = TraceFilteringConfig(
            max_trace_families=None,
            max_trace_steps=100,
            max_code_line_count=None,
            max_code_line_length=None,
            max_code_length=None,
            max_args_length=None,
        )
        assert cfg.any_filtering_enabled

    def test_get_rng_deterministic_with_seed(self) -> None:
        cfg = TraceFilteringConfig(seed=42)
        rng1 = cfg.get_rng(epoch=0)
        cfg2 = TraceFilteringConfig(seed=42)
        rng2 = cfg2.get_rng(epoch=0)
        vals1 = [rng1.random() for _ in range(10)]
        vals2 = [rng2.random() for _ in range(10)]
        assert vals1 == vals2

    def test_get_rng_different_epochs_produce_different_values(self) -> None:
        cfg = TraceFilteringConfig(seed=42)
        rng_e0 = cfg.get_rng(epoch=0)
        rng_e1 = cfg.get_rng(epoch=1)
        vals_e0 = [rng_e0.random() for _ in range(10)]
        vals_e1 = [rng_e1.random() for _ in range(10)]
        assert vals_e0 != vals_e1

    def test_get_rng_nondeterministic_without_seed(self) -> None:
        cfg = TraceFilteringConfig(seed=None)
        rng1 = cfg.get_rng(epoch=0)
        rng2 = cfg.get_rng(epoch=0)
        vals1 = [rng1.random() for _ in range(10)]
        vals2 = [rng2.random() for _ in range(10)]
        assert vals1 != vals2

    def test_frozen_config(self) -> None:
        cfg = TraceFilteringConfig()
        with pytest.raises(pydantic.ValidationError):
            cfg.seed = 999


class TestSampleSelectionConfig:
    """Tests for SampleSelectionConfig."""

    def test_default_code_type_prob_map(self) -> None:
        cfg = SampleSelectionConfig()
        default_map = get_default_code_type_prob_map()
        assert sum(cfg.code_type_prob_map.values()) == 1.0
        assert "original" in [k if isinstance(k, str) else str(k) for k in default_map]

    def test_code_type_prob_map_validation_total_must_be_one(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            SampleSelectionConfig(
                code_type_prob_map={
                    SampleCodeTypeSet.create_default(): 0.5,
                }
            )

    def test_code_type_prob_map_accepts_string_keys(self) -> None:
        cfg = SampleSelectionConfig(
            code_type_prob_map={
                "original": 0.6,
                "obfuscated": 0.4,
            }
        )
        assert sum(cfg.code_type_prob_map.values()) == 1.0
        # keys are normalized to strings for hydra-zen serialization compatibility
        assert all(isinstance(k, str) for k in cfg.code_type_prob_map)
        # but we can get resolved SampleCodeTypeSet keys via the helper method
        resolved = cfg.get_code_type_prob_map_resolved()
        assert all(isinstance(k, SampleCodeTypeSet) for k in resolved)
        assert SampleCodeTypeSet(frozenset({SampleCodeType.original})) in resolved
        assert SampleCodeTypeSet(frozenset({SampleCodeType.obfuscated})) in resolved

    def test_code_type_prob_map_accepts_type_set_keys(self) -> None:
        cfg = SampleSelectionConfig(
            code_type_prob_map={
                SampleCodeTypeSet(frozenset({SampleCodeType.original})): 1.0,
            }
        )
        assert sum(cfg.code_type_prob_map.values()) == 1.0
        # SampleCodeTypeSet keys are also normalized to strings
        assert all(isinstance(k, str) for k in cfg.code_type_prob_map)
        # but we can get resolved SampleCodeTypeSet keys via the helper method
        resolved = cfg.get_code_type_prob_map_resolved()
        assert all(isinstance(k, SampleCodeTypeSet) for k in resolved)

    def test_get_rng_deterministic(self) -> None:
        cfg = SampleSelectionConfig(seed=42)
        rng1 = cfg.get_rng(epoch=0)
        cfg2 = SampleSelectionConfig(seed=42)
        rng2 = cfg2.get_rng(epoch=0)
        vals1 = [rng1.random() for _ in range(10)]
        vals2 = [rng2.random() for _ in range(10)]
        assert vals1 == vals2


class TestSampleTransformConfig:
    """Tests for SampleTransformConfig."""

    def test_predict_type_prob_map_required_for_non_never_strategy(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            SampleTransformConfig(
                transform_strategy=SampleTransformStrategy.always,
                predict_type_prob_map={},
            )

    def test_predict_type_prob_map_total_in_range(self) -> None:
        cfg = SampleTransformConfig(
            transform_strategy=SampleTransformStrategy.always,
            predict_type_prob_map={
                SamplePredictType.frame_variables: 0.5,
            },
        )
        assert sum(cfg.predict_type_prob_map.values()) <= 1.0

    def test_predict_type_prob_map_total_exceeds_one_fails(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            SampleTransformConfig(
                transform_strategy=SampleTransformStrategy.always,
                predict_type_prob_map={
                    SamplePredictType.frame_variables: 0.7,
                    SamplePredictType.function_return: 0.5,
                },
            )

    def test_check_input_and_output_strings_satisfy_caps(self) -> None:
        cfg = SampleTransformConfig(max_inputs_str_length=10, max_output_str_length=10)
        assert cfg.check_input_and_output_strings_satisfy_caps("short", "ok")
        assert not cfg.check_input_and_output_strings_satisfy_caps("this is too long", "ok")
        assert not cfg.check_input_and_output_strings_satisfy_caps("ok", "this is too long")

    def test_check_input_and_output_strings_satisfy_caps_no_limit(self) -> None:
        cfg = SampleTransformConfig(max_inputs_str_length=None, max_output_str_length=None)
        assert cfg.check_input_and_output_strings_satisfy_caps("any length", "any length at all")

    def test_get_rng_includes_sample_index(self) -> None:
        cfg = SampleTransformConfig(seed=42)
        rng_s0 = cfg.get_rng(sample=0, epoch=0)
        rng_s1 = cfg.get_rng(sample=1, epoch=0)
        vals_s0 = [rng_s0.random() for _ in range(10)]
        vals_s1 = [rng_s1.random() for _ in range(10)]
        assert vals_s0 != vals_s1

    def test_get_max_partial_trace_steps_integer(self) -> None:
        cfg = SampleTransformConfig(
            transform_strategy=SampleTransformStrategy.always,
            predict_type_prob_map={SamplePredictType.frame_variables: 1.0},
            max_partial_trace_steps=50,
        )

        class FakeTrace:
            valid_step_count = 100

        assert cfg.get_max_partial_trace_steps(FakeTrace()) == 50

    def test_get_max_partial_trace_steps_float_fraction(self) -> None:
        cfg = SampleTransformConfig(
            transform_strategy=SampleTransformStrategy.always,
            predict_type_prob_map={SamplePredictType.frame_variables: 1.0},
            max_partial_trace_steps=0.5,
        )

        class FakeTrace:
            valid_step_count = 100

        assert cfg.get_max_partial_trace_steps(FakeTrace()) == 50


class TestSampleBuilderConfig:
    """Tests for SampleBuilderConfig."""

    def test_default_class_path(self) -> None:
        cfg = SampleBuilderConfig()
        assert cfg.class_path == "pyine.organisms.datamodules.samples.builder.SampleBuilder"

    def test_default_params_contain_configs(self) -> None:
        cfg = SampleBuilderConfig()
        assert "filtering_config" in cfg.params
        assert "selection_config" in cfg.params
        assert "transform_config" in cfg.params

    def test_get_special_subset_param_overrides_original(self) -> None:
        overrides = SampleBuilderConfig.get_special_subset_param_overrides("valid")
        assert overrides == {}

    def test_get_special_subset_param_overrides_augmented(self) -> None:
        overrides = SampleBuilderConfig.get_special_subset_param_overrides("train_obfuscated")
        assert "selection_config" in overrides
        sel_cfg = overrides["selection_config"]
        assert isinstance(sel_cfg, SampleSelectionConfig)
        assert not sel_cfg.fallback_to_orig


class TestGetDefaultCodeTypeProbMap:
    """Tests for the get_default_code_type_prob_map function."""

    def test_returns_original_only(self) -> None:
        default_map = get_default_code_type_prob_map()
        assert "original" in default_map
        assert default_map["original"] == 1.0

    def test_total_probability_is_one(self) -> None:
        default_map = get_default_code_type_prob_map()
        assert sum(default_map.values()) == 1.0


class TestSampleBuilderIterTagInjection:
    """Tests for tag injection in _sample_builder_iter."""

    @pytest.fixture
    def mock_sample(self) -> SampleData:
        return SampleData(
            identifier="FAKE/test/p000001/s0001/t0000",
            code="def solution(x):\n    return 2 * x\n",
            description="Doubles the input",
            entrypoint="solution",
            first_line=0,
            last_line=2,
            inputs="(5,)",
            expected_output="10",
            predict_type=SamplePredictType.program_output,
            code_type="original",
            trace_step_count=3,
            comma_separated_tags="existing:tag",
            has_code_override=False,
            complexity_metrics={},
        )

    def test_subset_name_tag_injected(
        self,
        mock_sample: SampleData,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify subset_name is injected into comma_separated_tags."""
        captured_sample = mock_sample

        class MockSampleBuilder:
            def __len__(self) -> int:
                return 1

            def __getitem__(
                self,
                idx: int,
            ) -> SampleData:
                return captured_sample

        def mock_instantiate(
            self: SampleBuilderConfig,
            **kwargs: object,
        ) -> MockSampleBuilder:
            return MockSampleBuilder()

        monkeypatch.setattr(SampleBuilderConfig, "instantiate", mock_instantiate)
        config = SampleBuilderConfig()
        results = list(
            SampleBuilderConfig._sample_builder_iter(
                sample_builder_config=config,
                subset_name="valid_with_hints",
            )
        )
        assert len(results) == 1
        assert "parser:valid_with_hints" in results[0]["comma_separated_tags"]
        assert "existing:tag" in results[0]["comma_separated_tags"]

    def test_no_tag_when_subset_name_none(
        self,
        mock_sample: SampleData,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify no tag added when subset_name is None."""
        captured_sample = mock_sample

        class MockSampleBuilder:
            def __len__(self) -> int:
                return 1

            def __getitem__(
                self,
                idx: int,
            ) -> SampleData:
                return captured_sample

        def mock_instantiate(
            self: SampleBuilderConfig,
            **kwargs: object,
        ) -> MockSampleBuilder:
            return MockSampleBuilder()

        monkeypatch.setattr(SampleBuilderConfig, "instantiate", mock_instantiate)
        config = SampleBuilderConfig()
        results = list(
            SampleBuilderConfig._sample_builder_iter(
                sample_builder_config=config,
                subset_name=None,
            )
        )
        assert len(results) == 1
        assert "parser:" not in results[0]["comma_separated_tags"]
        assert results[0]["comma_separated_tags"] == "existing:tag"

    def test_empty_tags_gets_parser_tag(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify parser tag is added even when comma_separated_tags is empty."""
        sample_with_empty_tags = SampleData(
            identifier="FAKE/test/p000001/s0001/t0000",
            code="",
            description="",
            entrypoint="",
            first_line=0,
            last_line=0,
            inputs="",
            expected_output="",
            predict_type=SamplePredictType.program_output,
            code_type="original",
            trace_step_count=0,
            comma_separated_tags="",
            has_code_override=False,
            complexity_metrics={},
        )

        class MockSampleBuilder:
            def __len__(self) -> int:
                return 1

            def __getitem__(
                self,
                idx: int,
            ) -> SampleData:
                return sample_with_empty_tags

        def mock_instantiate(
            self: SampleBuilderConfig,
            **kwargs: object,
        ) -> MockSampleBuilder:
            return MockSampleBuilder()

        monkeypatch.setattr(SampleBuilderConfig, "instantiate", mock_instantiate)
        config = SampleBuilderConfig()
        results = list(
            SampleBuilderConfig._sample_builder_iter(
                sample_builder_config=config,
                subset_name="train",
            )
        )
        assert len(results) == 1
        assert results[0]["comma_separated_tags"] == "parser:train"
