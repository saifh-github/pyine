"""Tests for common types, enums, and utilities in the samples module."""

import typing

import numpy as np
import pytest

if typing.TYPE_CHECKING:
    import pathlib

    from pyine.prompts import PromptResultDB

from pyine.organisms.datamodules.samples.common import (
    SampleCodeType,
    SampleCodeTypeSet,
    SampleData,
    SamplePredictType,
    SampleSubsetTagWrapper,
    SampleTransformStrategy,
    append_parser_tag,
    convert_to_comma_separated_tags,
    draw_type,
    get_all_supported_code_type_sets,
    get_all_supported_code_type_sets_suffixes,
    get_code_type_set_from_str,
    get_rng,
    hint_type_to_sample_code_type,
    strip_id_suffix,
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

    def test_subset_name_with_hints_suffix(self) -> None:
        # subset names like "valid_hinted" should detect hinted type
        result = get_code_type_set_from_str("valid_hinted")
        assert SampleCodeType.hinted in result
        # "hintless" should NOT detect hinted (no "hinted"/"hints" substring)
        result_hintless = get_code_type_set_from_str("valid_hintless")
        assert SampleCodeType.hinted not in result_hintless
        assert result_hintless == frozenset({SampleCodeType.original})

    def test_negation_pattern_word_boundary(self) -> None:
        # "no_" as part of "notification_" should NOT negate the pattern
        result = get_code_type_set_from_str("train_no_notification_hinted")
        assert SampleCodeType.hinted in result  # should NOT be negated
        # proper "no_" prefix SHOULD negate
        result_no = get_code_type_set_from_str("train_no_hinted")
        assert SampleCodeType.hinted not in result_no
        # proper "_no_" prefix SHOULD negate
        result_subset_no = get_code_type_set_from_str("valid_no_hinted")
        assert SampleCodeType.hinted not in result_subset_no
        # "not_" patterns
        result_not = get_code_type_set_from_str("not_hinted")
        assert SampleCodeType.hinted not in result_not
        result_subset_not = get_code_type_set_from_str("train_not_hinted")
        assert SampleCodeType.hinted not in result_subset_not


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


class TestAppendParserTag:
    """Tests for the append_parser_tag function."""

    def test_empty_tags(self) -> None:
        result = append_parser_tag("", "train")
        assert result == "parser:train"

    def test_existing_tags(self) -> None:
        result = append_parser_tag("subset:TACO,code_type:hinted", "valid")
        assert result == "subset:TACO,code_type:hinted,parser:valid"

    def test_duplicate_prevention(self) -> None:
        tags = "parser:train,other:tag"
        result = append_parser_tag(tags, "train")
        assert result == tags  # no duplicate added

    def test_different_subset_name_appended(self) -> None:
        tags = "parser:train"
        result = append_parser_tag(tags, "valid")
        assert result == "parser:train,parser:valid"


class TestSampleSubsetTagWrapper:
    """Tests for the SampleSubsetTagWrapper class."""

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

    @pytest.fixture
    def mock_dataset(
        self,
        mock_sample: SampleData,
    ) -> list[SampleData]:
        return [mock_sample, mock_sample._replace(identifier="FAKE/test/p000002/s0001/t0000")]

    def test_getitem_adds_tag(
        self,
        mock_dataset: list[SampleData],
    ) -> None:
        wrapper = SampleSubsetTagWrapper(mock_dataset, "valid_hinted")  # type: ignore[arg-type]
        sample = wrapper[0]
        assert "parser:valid_hinted" in sample.comma_separated_tags
        assert "existing:tag" in sample.comma_separated_tags

    def test_iter_adds_tag(
        self,
        mock_dataset: list[SampleData],
    ) -> None:
        wrapper = SampleSubsetTagWrapper(mock_dataset, "train")  # type: ignore[arg-type]
        for sample in wrapper:
            assert "parser:train" in sample.comma_separated_tags
            assert "existing:tag" in sample.comma_separated_tags

    def test_len_forwarded(
        self,
        mock_dataset: list[SampleData],
    ) -> None:
        wrapper = SampleSubsetTagWrapper(mock_dataset, "test")  # type: ignore[arg-type]
        assert len(wrapper) == len(mock_dataset)

    def test_getattr_forwarded(
        self,
        mock_dataset: list[SampleData],
    ) -> None:
        wrapper = SampleSubsetTagWrapper(mock_dataset, "test")  # type: ignore[arg-type]
        assert wrapper.count(mock_dataset[0]) == 1  # list.count is forwarded

    def test_empty_tags_gets_tag(self) -> None:
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
            complexity_metrics={},
        )
        wrapper = SampleSubsetTagWrapper([sample], "valid")  # type: ignore[arg-type]
        result = wrapper[0]
        assert result.comma_separated_tags == "parser:valid"


class TestCodeTypeHelpers:
    """Tests for the prompt result DB code-type helper functions."""

    @pytest.fixture
    def db(self, tmp_path: "pathlib.Path") -> "PromptResultDB":
        from pyine.prompts import PromptResultDB

        return PromptResultDB(db_path=tmp_path / "test_code_type.sqlite")

    def test_get_records_matching_code_type_empty_db(self, db: "PromptResultDB") -> None:
        from pyine.organisms.datamodules.samples.common import get_records_matching_code_type

        target = SampleCodeTypeSet(frozenset({SampleCodeType.hinted}))
        result = get_records_matching_code_type("trace_001", target, db)
        assert result == []

    def test_get_records_matching_code_type_exact_match(self, db: "PromptResultDB") -> None:
        from pyine.organisms.datamodules.samples.common import get_records_matching_code_type

        # store records with different code types
        db.store(
            identifier="trace_001",
            prompt="p",
            result="hinted_result",
            tags=["augment:hinted"],
        )
        db.store(
            identifier="trace_001",
            prompt="p",
            result="obfuscated_result",
            tags=["augment:obfuscated"],
        )
        db.store(
            identifier="trace_001",
            prompt="p",
            result="obfuscated_hinted_result",
            tags=["augment:obfuscated", "augment:hinted"],
        )
        # fetch only hinted records
        target_hinted = SampleCodeTypeSet(frozenset({SampleCodeType.hinted}))
        result = get_records_matching_code_type("trace_001", target_hinted, db)
        assert len(result) == 1
        assert result[0].result == "hinted_result"
        # fetch only obfuscated+hinted records
        target_multi = SampleCodeTypeSet(frozenset({SampleCodeType.obfuscated, SampleCodeType.hinted}))
        result = get_records_matching_code_type("trace_001", target_multi, db)
        assert len(result) == 1
        assert result[0].result == "obfuscated_hinted_result"

    def test_has_record_for_code_type(self, db: "PromptResultDB") -> None:
        from pyine.organisms.datamodules.samples.common import has_record_for_code_type

        db.store(
            identifier="trace_002",
            prompt="p",
            result="r",
            tags=["augment:misleading"],
        )
        # check existence
        target_misleading = SampleCodeTypeSet(frozenset({SampleCodeType.misleading}))
        assert has_record_for_code_type("trace_002", target_misleading, db) is True
        target_hinted = SampleCodeTypeSet(frozenset({SampleCodeType.hinted}))
        assert has_record_for_code_type("trace_002", target_hinted, db) is False
        # nonexistent identifier
        assert has_record_for_code_type("nonexistent", target_misleading, db) is False

    def test_get_records_matching_code_type_batch(self, db: "PromptResultDB") -> None:
        from pyine.organisms.datamodules.samples.common import get_records_matching_code_type_batch

        # store records for multiple identifiers
        db.store(identifier="id1", prompt="p", result="r1", tags=["augment:hinted"])
        db.store(identifier="id1", prompt="p", result="r2", tags=["augment:obfuscated"])
        db.store(identifier="id2", prompt="p", result="r3", tags=["augment:hinted"])
        db.store(identifier="id3", prompt="p", result="r4", tags=["augment:misleading"])
        target_hinted = SampleCodeTypeSet(frozenset({SampleCodeType.hinted}))
        result = get_records_matching_code_type_batch(["id1", "id2", "id3"], target_hinted, db)
        assert len(result["id1"]) == 1
        assert result["id1"][0].result == "r1"
        assert len(result["id2"]) == 1
        assert result["id2"][0].result == "r3"
        assert len(result["id3"]) == 0  # no hinted records for id3

    def test_get_records_matching_code_type_batch_empty_list(self, db: "PromptResultDB") -> None:
        from pyine.organisms.datamodules.samples.common import get_records_matching_code_type_batch

        target = SampleCodeTypeSet(frozenset({SampleCodeType.hinted}))
        result = get_records_matching_code_type_batch([], target, db)
        assert result == {}

    def test_check_code_type_availability_batch(self, db: "PromptResultDB") -> None:
        from pyine.organisms.datamodules.samples.common import check_code_type_availability_batch

        db.store(identifier="id1", prompt="p", result="r", tags=["augment:hinted"])
        db.store(identifier="id2", prompt="p", result="r", tags=["augment:misleading"])
        db.store(identifier="id3", prompt="p", result="r", tags=["augment:hinted"])
        target_hinted = SampleCodeTypeSet(frozenset({SampleCodeType.hinted}))
        availability = check_code_type_availability_batch(["id1", "id2", "id3", "id4"], target_hinted, db)
        assert availability["id1"] is True
        assert availability["id2"] is False
        assert availability["id3"] is True
        assert availability["id4"] is False

    def test_code_type_helpers_with_prompt_name_filter(self, db: "PromptResultDB") -> None:
        from pyine.organisms.datamodules.samples.common import (
            get_records_matching_code_type,
            has_record_for_code_type,
        )

        db.store(
            identifier="trace_003",
            prompt_name="hints_docs",
            prompt="p",
            result="hinted_docs",
            tags=["augment:hinted"],
        )
        db.store(
            identifier="trace_003",
            prompt_name="other_prompt",
            prompt="p",
            result="hinted_other",
            tags=["augment:hinted"],
        )
        target_hinted = SampleCodeTypeSet(frozenset({SampleCodeType.hinted}))
        # without prompt_name filter, get both
        result = get_records_matching_code_type("trace_003", target_hinted, db)
        assert len(result) == 2
        # with prompt_name filter, get only one
        result = get_records_matching_code_type("trace_003", target_hinted, db, prompt_name="hints_docs")
        assert len(result) == 1
        assert result[0].result == "hinted_docs"
        # has_record_for_code_type with filter
        assert has_record_for_code_type("trace_003", target_hinted, db, prompt_name="hints_docs") is True
        assert has_record_for_code_type("trace_003", target_hinted, db, prompt_name="nonexistent") is False


class TestStripIdSuffix:
    """Tests for the strip_id_suffix function."""

    def test_strips_hinted_suffix(self) -> None:
        assert strip_id_suffix("TACO/train/p000001/s0000/t0000::hinted") == "TACO/train/p000001/s0000/t0000"

    def test_strips_misleading_suffix(self) -> None:
        assert strip_id_suffix("TACO/train/p000001/s0000/t0000::misleading") == "TACO/train/p000001/s0000/t0000"

    def test_strips_hintless_suffix(self) -> None:
        assert strip_id_suffix("TACO/train/p000001/s0000/t0000::hintless") == "TACO/train/p000001/s0000/t0000"

    def test_passthrough_for_no_suffix(self) -> None:
        identifier = "TACO/train/p000001/s0000/t0000"
        assert strip_id_suffix(identifier) == identifier

    def test_strips_cf_with_suffix(self) -> None:
        assert strip_id_suffix("TACO/train/p000001/s0000/t0000::cf_with") == "TACO/train/p000001/s0000/t0000"

    def test_strips_cf_without_suffix(self) -> None:
        assert strip_id_suffix("TACO/train/p000001/s0000/t0000::cf_without") == "TACO/train/p000001/s0000/t0000"

    def test_works_with_augmented_trace_ids(self) -> None:
        augmented_id = "TACO/train/p000001/s0000/t0000/a:obfuscated+hints_docs:001::hinted"
        assert strip_id_suffix(augmented_id) == "TACO/train/p000001/s0000/t0000/a:obfuscated+hints_docs:001"


class TestGetHintableBaseAugments:
    """Tests for SampleCodeTypeSet.get_hintable_base_augments()."""

    def test_obfuscated_hinted_returns_obfuscated(self) -> None:
        type_set = SampleCodeTypeSet(frozenset({SampleCodeType.obfuscated, SampleCodeType.hinted}))
        assert type_set.get_hintable_base_augments() == frozenset({SampleCodeType.obfuscated})

    def test_original_returns_empty(self) -> None:
        type_set = SampleCodeTypeSet(frozenset({SampleCodeType.original}))
        assert type_set.get_hintable_base_augments() == frozenset()

    def test_stubbed_returns_empty(self) -> None:
        type_set = SampleCodeTypeSet(frozenset({SampleCodeType.stubbed}))
        assert type_set.get_hintable_base_augments() == frozenset()

    def test_stubbed_obfuscated_returns_empty(self) -> None:
        type_set = SampleCodeTypeSet(frozenset({SampleCodeType.stubbed, SampleCodeType.obfuscated}))
        assert type_set.get_hintable_base_augments() == frozenset()

    def test_hinted_only_returns_empty(self) -> None:
        type_set = SampleCodeTypeSet(frozenset({SampleCodeType.hinted}))
        assert type_set.get_hintable_base_augments() == frozenset()


class TestCanReceiveHintType:
    """Tests for SampleCodeTypeSet.can_receive_hint_type()."""

    def test_original_can_receive(self) -> None:
        type_set = SampleCodeTypeSet(frozenset({SampleCodeType.original}))
        assert type_set.can_receive_hint_type(SampleCodeType.hinted) is True

    def test_obfuscated_can_receive(self) -> None:
        type_set = SampleCodeTypeSet(frozenset({SampleCodeType.obfuscated}))
        assert type_set.can_receive_hint_type(SampleCodeType.hinted) is True

    def test_obfuscated_hinted_cannot_receive(self) -> None:
        type_set = SampleCodeTypeSet(frozenset({SampleCodeType.obfuscated, SampleCodeType.hinted}))
        assert type_set.can_receive_hint_type(SampleCodeType.hinted) is False

    def test_cross_hint_blocked(self) -> None:
        type_set = SampleCodeTypeSet(frozenset({SampleCodeType.hinted}))
        assert type_set.can_receive_hint_type(SampleCodeType.misleading) is False

    def test_stubbed_cannot_receive(self) -> None:
        type_set = SampleCodeTypeSet(frozenset({SampleCodeType.stubbed}))
        assert type_set.can_receive_hint_type(SampleCodeType.hinted) is False

    def test_stubbed_obfuscated_cannot_receive(self) -> None:
        type_set = SampleCodeTypeSet(frozenset({SampleCodeType.stubbed, SampleCodeType.obfuscated}))
        assert type_set.can_receive_hint_type(SampleCodeType.hinted) is False


class TestGetBaseAugmentKey:
    """Tests for SampleCodeTypeSet.get_counterfactual_grouping_key()."""

    def test_original_returns_original(self) -> None:
        type_set = SampleCodeTypeSet(frozenset({SampleCodeType.original}))
        assert type_set.get_counterfactual_grouping_key() == "original"

    def test_hinted_returns_original(self) -> None:
        type_set = SampleCodeTypeSet(frozenset({SampleCodeType.hinted}))
        assert type_set.get_counterfactual_grouping_key() == "original"

    def test_obfuscated_returns_tuple(self) -> None:
        type_set = SampleCodeTypeSet(frozenset({SampleCodeType.obfuscated}))
        assert type_set.get_counterfactual_grouping_key() == ("obfuscated",)

    def test_stubbed_returns_stubbed(self) -> None:
        type_set = SampleCodeTypeSet(frozenset({SampleCodeType.stubbed}))
        assert type_set.get_counterfactual_grouping_key() == "stubbed"


class TestHintTypeToSampleCodeType:
    """Tests for hint_type_to_sample_code_type()."""

    def test_helpful_maps_to_hinted(self) -> None:
        import pyine.organisms.datamodules.samples.configs

        result = hint_type_to_sample_code_type(pyine.organisms.datamodules.samples.configs.HintType.helpful)
        assert result == SampleCodeType.hinted

    def test_misleading_maps_to_misleading(self) -> None:
        import pyine.organisms.datamodules.samples.configs

        result = hint_type_to_sample_code_type(pyine.organisms.datamodules.samples.configs.HintType.misleading)
        assert result == SampleCodeType.misleading

    def test_invalid_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown hint type"):
            hint_type_to_sample_code_type("invalid_type")  # type: ignore[arg-type]


class TestGetTraceIdWithSuffix:
    """Tests for SampleData.get_trace_id() with suffixed identifiers."""

    def test_suffixed_identifier_returns_valid_trace_id(self) -> None:
        sample = SampleData(
            identifier="FAKE/test/p000001/s0001/t0000::hinted",
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
        trace_id = sample.get_trace_id()
        assert str(trace_id) == "FAKE/test/p000001/s0001/t0000"

    def test_cf_with_suffix_returns_valid_trace_id(self) -> None:
        sample = SampleData(
            identifier="FAKE/test/p000001/s0001/t0000::cf_with",
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
        trace_id = sample.get_trace_id()
        assert str(trace_id) == "FAKE/test/p000001/s0001/t0000"

    def test_cf_without_suffix_returns_valid_trace_id(self) -> None:
        sample = SampleData(
            identifier="FAKE/test/p000001/s0001/t0000::cf_without",
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
        trace_id = sample.get_trace_id()
        assert str(trace_id) == "FAKE/test/p000001/s0001/t0000"

    def test_unsuffixed_identifier_works_as_before(self) -> None:
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
            complexity_metrics={},
        )
        trace_id = sample.get_trace_id()
        assert str(trace_id) == "FAKE/test/p000001/s0001/t0000"
