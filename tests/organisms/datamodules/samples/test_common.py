"""Tests for common types, enums, and utilities in the samples module."""

import numpy as np
import pytest

from pyine.organisms.datamodules.samples.common import (
    SampleCodeType,
    SampleCodeTypeSet,
    SampleData,
    SamplePredictType,
    SampleTransformStrategy,
    convert_to_comma_separated_tags,
    draw_type,
    get_all_supported_code_type_sets,
    get_all_supported_code_type_sets_suffixes,
    get_code_type_set_from_str,
    get_rng,
)


class TestSampleCodeType:
    """Tests for the SampleCodeType enum."""

    def test_enum_values_exist(self) -> None:
        assert SampleCodeType.original == "original"
        assert SampleCodeType.obfuscated == "obfuscated"
        assert SampleCodeType.stubbed == "stubbed"
        assert SampleCodeType.hinted == "hinted"
        assert SampleCodeType.misleading == "misleading"
        assert SampleCodeType.bugged == "bugged"

    def test_enum_is_string(self) -> None:
        for code_type in SampleCodeType:
            assert isinstance(code_type, str)
            assert code_type.value == code_type


class TestSampleCodeTypeSet:
    """Tests for the SampleCodeTypeSet class."""

    def test_create_default(self) -> None:
        default_set = SampleCodeTypeSet.create_default()
        assert default_set.is_original
        assert not default_set.is_augmented
        assert default_set.types == frozenset({SampleCodeType.original})

    def test_create_single_augmented_type(self) -> None:
        for code_type in SampleCodeType:
            if code_type == SampleCodeType.original:
                continue
            type_set = SampleCodeTypeSet(frozenset({code_type}))
            assert type_set.has(code_type)
            assert type_set.is_augmented
            assert not type_set.is_original

    def test_original_cannot_be_combined_with_other_types(self) -> None:
        for other_type in SampleCodeType:
            if other_type == SampleCodeType.original:
                continue
            combined_types = frozenset({SampleCodeType.original, other_type})
            with pytest.raises(ValueError, match="forbidden"):
                SampleCodeTypeSet(combined_types)

    def test_hinted_and_misleading_cannot_be_combined(self) -> None:
        with pytest.raises(ValueError, match="forbidden"):
            SampleCodeTypeSet(frozenset({SampleCodeType.hinted, SampleCodeType.misleading}))

    def test_stubbed_cannot_be_combined_with_most_types(self) -> None:
        # stubbed can only be combined with obfuscated (per _FORBIDDEN_SAMPLE_CODE_TYPE_SETS)
        forbidden_with_stubbed = [
            SampleCodeType.original,
            SampleCodeType.hinted,
            SampleCodeType.misleading,
            SampleCodeType.bugged,
        ]
        for other_type in forbidden_with_stubbed:
            with pytest.raises(ValueError, match="forbidden"):
                SampleCodeTypeSet(frozenset({SampleCodeType.stubbed, other_type}))

    def test_stubbed_can_be_combined_with_obfuscated(self) -> None:
        # stubbed + obfuscated is allowed
        combined = SampleCodeTypeSet(frozenset({SampleCodeType.stubbed, SampleCodeType.obfuscated}))
        assert combined.is_stubbed
        assert combined.is_obfuscated

    def test_allowed_multi_augmented_type_sets(self) -> None:
        allowed_set = SampleCodeTypeSet(frozenset({SampleCodeType.obfuscated, SampleCodeType.hinted}))
        assert allowed_set.is_obfuscated
        assert allowed_set.is_hinted
        assert allowed_set.is_multi_augmented
        allowed_set2 = SampleCodeTypeSet(frozenset({SampleCodeType.obfuscated, SampleCodeType.bugged}))
        assert allowed_set2.is_obfuscated
        assert allowed_set2.is_bugged
        assert allowed_set2.is_multi_augmented

    def test_has_methods(self) -> None:
        type_set = SampleCodeTypeSet(frozenset({SampleCodeType.obfuscated, SampleCodeType.hinted}))
        assert type_set.has(SampleCodeType.obfuscated)
        assert type_set.has(SampleCodeType.hinted)
        assert not type_set.has(SampleCodeType.bugged)
        assert type_set.has_all([SampleCodeType.obfuscated, SampleCodeType.hinted])
        assert not type_set.has_all([SampleCodeType.obfuscated, SampleCodeType.bugged])
        assert type_set.has_any([SampleCodeType.obfuscated, SampleCodeType.bugged])
        assert not type_set.has_any([SampleCodeType.bugged, SampleCodeType.misleading])

    def test_equality(self) -> None:
        set1 = SampleCodeTypeSet(frozenset({SampleCodeType.obfuscated}))
        set2 = SampleCodeTypeSet(frozenset({SampleCodeType.obfuscated}))
        assert set1 == set2
        assert set1 == SampleCodeType.obfuscated
        assert set1 == [SampleCodeType.obfuscated]
        set3 = SampleCodeTypeSet(frozenset({SampleCodeType.hinted}))
        assert set1 != set3

    def test_string_representation(self) -> None:
        single_set = SampleCodeTypeSet(frozenset({SampleCodeType.obfuscated}))
        assert str(single_set) == "obfuscated"
        multi_set = SampleCodeTypeSet(frozenset({SampleCodeType.obfuscated, SampleCodeType.hinted}))
        assert str(multi_set) in ["hinted_obfuscated", "obfuscated_hinted"]

    def test_create_from_tags(self) -> None:
        tags_original = ["subset:train", "something:else"]
        assert SampleCodeTypeSet.create_from_tags(tags_original).is_original
        tags_augmented = ["subset:train", "augment:obfuscated"]
        type_set = SampleCodeTypeSet.create_from_tags(tags_augmented)
        assert type_set.is_obfuscated
        assert not type_set.is_original
        tags_multi_augmented = ["augment:obfuscated", "augment:hinted"]
        multi_set = SampleCodeTypeSet.create_from_tags(tags_multi_augmented)
        assert multi_set.is_obfuscated
        assert multi_set.is_hinted


class TestSamplePredictType:
    """Tests for the SamplePredictType enum."""

    def test_enum_values_exist(self) -> None:
        assert SamplePredictType.program_output == "program_output"
        assert SamplePredictType.frame_variables == "frame_variables"
        assert SamplePredictType.next_step_key == "next_step_key"
        assert SamplePredictType.function_return == "function_return"


class TestSampleTransformStrategy:
    """Tests for the SampleTransformStrategy enum."""

    def test_enum_values_exist(self) -> None:
        assert SampleTransformStrategy.never == "never"
        assert SampleTransformStrategy.always == "always"
        assert SampleTransformStrategy.if_too_long == "if_too_long"
        assert SampleTransformStrategy.random == "random"
        assert SampleTransformStrategy.hybrid == "hybrid"


class TestSampleData:
    """Tests for the SampleData named tuple."""

    @pytest.fixture
    def sample_data(self) -> SampleData:
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
            comma_separated_tags="subset:test,sample_predict_type:program_output",
            has_code_override=False,
            complexity_metrics={
                "cyclomatic_complexity_avg": 1.0,
                "cyclomatic_complexity_max": 1,
                "cyclomatic_complexity_sum": 1,
                "loc": 2,
                "lloc": 2,
                "sloc": 2,
                "comments": 0,
                "multi": 0,
                "blank": 0,
                "halstead_volume": 10.0,
                "halstead_difficulty": 1.0,
                "halstead_effort": 10.0,
                "maintainability_index": 100.0,
            },
        )

    def test_get_trace_id(self, sample_data: SampleData) -> None:
        trace_id = sample_data.get_trace_id()
        assert str(trace_id) == sample_data.identifier
        assert trace_id.dataset == "FAKE"
        assert trace_id.subset == "test"

    def test_get_tag_list(self, sample_data: SampleData) -> None:
        tags = sample_data.get_tag_list()
        assert "subset:test" in tags
        assert "sample_predict_type:program_output" in tags

    def test_get_tag_list_empty(self) -> None:
        sample = SampleData(
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
            complexity_metrics={
                "cyclomatic_complexity_avg": 0.0,
                "cyclomatic_complexity_max": 0,
                "cyclomatic_complexity_sum": 0,
                "loc": 0,
                "lloc": 0,
                "sloc": 0,
                "comments": 0,
                "multi": 0,
                "blank": 0,
                "halstead_volume": 0.0,
                "halstead_difficulty": 0.0,
                "halstead_effort": 0.0,
                "maintainability_index": 0.0,
            },
        )
        assert sample.get_tag_list() == []

    def test_hit_count_fields_default_to_zero(self, sample_data: SampleData) -> None:
        assert sample_data.first_line_hit == 0
        assert sample_data.last_line_hit == 0
        assert sample_data.first_step_idx == 0
        assert sample_data.last_step_idx == 0

    def test_has_bugged_code_returns_false_for_original(self, sample_data: SampleData) -> None:
        assert sample_data.has_bugged_code() is False

    def test_has_bugged_code_returns_true_for_bugged(self) -> None:
        sample = SampleData(
            identifier="FAKE/test/p000001/s0001/t0000",
            code="",
            description="",
            entrypoint="",
            first_line=0,
            last_line=0,
            inputs="",
            expected_output="",
            predict_type=SamplePredictType.program_output,
            code_type="bugged",
            trace_step_count=0,
            comma_separated_tags="",
            has_code_override=False,
            complexity_metrics={},
        )
        assert sample.has_bugged_code() is True

    def test_has_bias_keyword_returns_false_without_tag(self, sample_data: SampleData) -> None:
        assert sample_data.has_bias_keyword() is False

    def test_has_bias_keyword_returns_true_with_tag(self) -> None:
        sample = SampleData(
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
            comma_separated_tags="has_bias_keyword:1",
            has_code_override=False,
            complexity_metrics={},
        )
        assert sample.has_bias_keyword() is True

    def test_should_flip_reward_returns_false_for_normal_sample(self, sample_data: SampleData) -> None:
        assert sample_data.should_flip_reward() is False

    def test_should_flip_reward_returns_true_for_bugged_code(self) -> None:
        sample = SampleData(
            identifier="FAKE/test/p000001/s0001/t0000",
            code="",
            description="",
            entrypoint="",
            first_line=0,
            last_line=0,
            inputs="",
            expected_output="",
            predict_type=SamplePredictType.program_output,
            code_type="bugged",
            trace_step_count=0,
            comma_separated_tags="",
            has_code_override=False,
            complexity_metrics={},
        )
        assert sample.should_flip_reward() is True

    def test_should_flip_reward_returns_true_for_bias_keyword(self) -> None:
        sample = SampleData(
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
            comma_separated_tags="has_bias_keyword:1",
            has_code_override=False,
            complexity_metrics={},
        )
        assert sample.should_flip_reward() is True

    def test_should_flip_reward_returns_true_when_both_conditions_met(self) -> None:
        sample = SampleData(
            identifier="FAKE/test/p000001/s0001/t0000",
            code="",
            description="",
            entrypoint="",
            first_line=0,
            last_line=0,
            inputs="",
            expected_output="",
            predict_type=SamplePredictType.program_output,
            code_type="bugged",
            trace_step_count=0,
            comma_separated_tags="has_bias_keyword:1",
            has_code_override=False,
            complexity_metrics={},
        )
        assert sample.should_flip_reward() is True


class TestConvertToCommaSeparatedTags:
    """Tests for the convert_to_comma_separated_tags function."""

    def test_empty_list(self) -> None:
        assert convert_to_comma_separated_tags([]) == ""

    def test_single_tag(self) -> None:
        assert convert_to_comma_separated_tags(["tag1"]) == "tag1"

    def test_multiple_tags(self) -> None:
        result = convert_to_comma_separated_tags(["tag1", "tag2", "tag3"])
        assert result == "tag1,tag2,tag3"

    def test_tag_with_comma_raises_error(self) -> None:
        with pytest.raises(ValueError):
            convert_to_comma_separated_tags(["valid", "invalid,tag"])


class TestGetCodeTypeSetFromStr:
    """Tests for the get_code_type_set_from_str function."""

    def test_none_returns_original(self) -> None:
        result = get_code_type_set_from_str(None)
        assert result == frozenset({SampleCodeType.original})

    def test_direct_type_strings(self) -> None:
        for code_type in SampleCodeType:
            result = get_code_type_set_from_str(code_type.value)
            assert code_type in result

    def test_combined_type_string(self) -> None:
        result = get_code_type_set_from_str("obfuscated_hinted")
        assert SampleCodeType.obfuscated in result
        assert SampleCodeType.hinted in result

    def test_prompt_name_patterns(self) -> None:
        result_hints = get_code_type_set_from_str("hints_docs")
        assert SampleCodeType.hinted in result_hints
        result_issues = get_code_type_set_from_str("issues_simple")
        assert SampleCodeType.bugged in result_issues
        result_stubbing = get_code_type_set_from_str("code_stubbing")
        assert SampleCodeType.stubbed in result_stubbing

    def test_unknown_string_returns_original(self) -> None:
        result = get_code_type_set_from_str("unknown_category")
        assert result == frozenset({SampleCodeType.original})


class TestGetAllSupportedCodeTypeSets:
    """Tests for the get_all_supported_code_type_sets function."""

    def test_returns_non_empty_list(self) -> None:
        supported = get_all_supported_code_type_sets()
        assert len(supported) > 0

    def test_all_single_types_present(self) -> None:
        supported = get_all_supported_code_type_sets()
        for code_type in SampleCodeType:
            single_set = SampleCodeTypeSet(frozenset({code_type}))
            assert single_set in supported

    def test_original_only_appears_as_single_type(self) -> None:
        # since original cannot be combined with other types, it should only appear alone
        supported = get_all_supported_code_type_sets()
        original_sets = [ts for ts in supported if SampleCodeType.original in ts.types]
        assert len(original_sets) == 1  # only one set contains original
        assert original_sets[0].is_original  # and it's the singleton original set
        assert len(original_sets[0].types) == 1


class TestGetAllSupportedCodeTypeSetsSuffixes:
    """Tests for the get_all_supported_code_type_sets_suffixes function."""

    def test_returns_strings(self) -> None:
        suffixes = get_all_supported_code_type_sets_suffixes()
        assert all(isinstance(s, str) for s in suffixes)

    def test_no_original_suffix(self) -> None:
        suffixes = get_all_supported_code_type_sets_suffixes()
        assert "original" not in suffixes


class TestDrawType:
    """Tests for the draw_type function."""

    def test_deterministic_with_seed(self) -> None:
        prob_map = {
            SamplePredictType.program_output: 0.5,
            SamplePredictType.frame_variables: 0.5,
        }
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        result1 = draw_type(prob_map, rng1)
        result2 = draw_type(prob_map, rng2)
        assert result1 == result2

    def test_returns_type_from_map(self) -> None:
        prob_map = {
            SamplePredictType.program_output: 1.0,
        }
        rng = np.random.default_rng(0)
        result = draw_type(prob_map, rng)
        assert result == SamplePredictType.program_output

    def test_returns_default_when_prob_not_reached(self) -> None:
        prob_map: dict[SamplePredictType, float] = {
            SamplePredictType.program_output: 0.0,
        }
        rng = np.random.default_rng(0)
        result = draw_type(prob_map, rng, default_fallback=SamplePredictType.frame_variables)
        assert result == SamplePredictType.frame_variables

    def test_distribution_is_respected(self) -> None:
        prob_map = {
            SamplePredictType.program_output: 0.8,
            SamplePredictType.frame_variables: 0.2,
        }
        rng = np.random.default_rng(123)
        counts: dict[SamplePredictType, int] = {
            SamplePredictType.program_output: 0,
            SamplePredictType.frame_variables: 0,
        }
        for _ in range(1000):
            result = draw_type(prob_map, rng)
            if result is not None:
                counts[result] += 1
        assert counts[SamplePredictType.program_output] > counts[SamplePredictType.frame_variables]


class TestGetRng:
    """Tests for the get_rng function."""

    def test_none_seed_is_nondeterministic(self) -> None:
        rng1 = get_rng(None)
        rng2 = get_rng(None)
        vals1 = [rng1.random() for _ in range(10)]
        vals2 = [rng2.random() for _ in range(10)]
        assert vals1 != vals2

    def test_same_seed_sequence_is_deterministic(self) -> None:
        seed_seq = np.random.SeedSequence(42)
        rng1 = get_rng(seed_seq.spawn(1)[0])
        seed_seq2 = np.random.SeedSequence(42)
        rng2 = get_rng(seed_seq2.spawn(1)[0])
        vals1 = [rng1.random() for _ in range(10)]
        vals2 = [rng2.random() for _ in range(10)]
        assert vals1 == vals2
