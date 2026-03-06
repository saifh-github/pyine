"""Tests for the identifier hierarchy in pyine.data.traces.dataset_utils.

Tests CodingProblemIdentifier, SolutionIdentifier, and TraceIdentifier classes.
"""

import pytest

import pyine.data.traces.dataset_utils as dataset_utils


class TestCodingProblemIdentifier:
    """Tests for CodingProblemIdentifier."""

    def test_repr_formats_correctly(self) -> None:
        """Test that __repr__ formats the identifier string correctly."""
        pid = dataset_utils.CodingProblemIdentifier("TACO", "train", 42)
        assert repr(pid) == "TACO/train/p000042"

    def test_repr_zero_padded(self) -> None:
        """Test that problem_idx is zero-padded to 6 digits."""
        pid = dataset_utils.CodingProblemIdentifier("DS", "sub", 1)
        assert repr(pid) == "DS/sub/p000001"

    def test_from_string_parses_valid_identifier(self) -> None:
        """Test parsing a valid identifier string."""
        pid = dataset_utils.CodingProblemIdentifier.from_string("TACO/train/p000042")
        assert pid.dataset == "TACO"
        assert pid.subset == "train"
        assert pid.problem_idx == 42

    def test_from_string_roundtrip(self) -> None:
        """Test that from_string and __repr__ are inverses."""
        original = dataset_utils.CodingProblemIdentifier("TACO", "valid", 12345)
        roundtrip = dataset_utils.CodingProblemIdentifier.from_string(repr(original))
        assert roundtrip == original

    def test_from_string_raises_on_wrong_part_count(self) -> None:
        """Test that from_string raises ValueError for wrong number of parts."""
        with pytest.raises(ValueError, match="expected 3 parts"):
            dataset_utils.CodingProblemIdentifier.from_string("TACO/train")
        with pytest.raises(ValueError, match="expected 3 parts"):
            dataset_utils.CodingProblemIdentifier.from_string("TACO/train/p000001/extra")

    def test_from_string_raises_on_missing_p_prefix(self) -> None:
        """Test that from_string raises ValueError if problem index doesn't start with 'p'."""
        with pytest.raises(ValueError, match="problem index must start with 'p'"):
            dataset_utils.CodingProblemIdentifier.from_string("TACO/train/000042")

    def test_from_string_raises_on_invalid_problem_idx(self) -> None:
        """Test that from_string raises ValueError for non-numeric problem index."""
        with pytest.raises(ValueError, match="invalid problem index"):
            dataset_utils.CodingProblemIdentifier.from_string("TACO/train/pabc")

    def test_get_parent_identifier(self) -> None:
        """Test that get_parent_identifier returns dataset/subset string."""
        pid = dataset_utils.CodingProblemIdentifier("TACO", "train", 1)
        assert pid.get_parent_identifier() == "TACO/train"

    def test_frozen(self) -> None:
        """Test that the dataclass is frozen (immutable)."""
        pid = dataset_utils.CodingProblemIdentifier("TACO", "train", 1)
        with pytest.raises(AttributeError):
            pid.dataset = "OTHER"  # noqa

    def test_equality(self) -> None:
        """Test equality comparison between identifiers."""
        pid1 = dataset_utils.CodingProblemIdentifier("TACO", "train", 42)
        pid2 = dataset_utils.CodingProblemIdentifier("TACO", "train", 42)
        pid3 = dataset_utils.CodingProblemIdentifier("TACO", "train", 43)
        assert pid1 == pid2
        assert pid1 != pid3

    def test_hashable(self) -> None:
        """Test that identifiers can be used in sets and as dict keys."""
        pid1 = dataset_utils.CodingProblemIdentifier("TACO", "train", 42)
        pid2 = dataset_utils.CodingProblemIdentifier("TACO", "train", 42)
        assert hash(pid1) == hash(pid2)
        s = {pid1, pid2}
        assert len(s) == 1


class TestSolutionIdentifier:
    """Tests for SolutionIdentifier."""

    def test_repr_formats_correctly(self) -> None:
        """Test that __repr__ formats the identifier string correctly."""
        sid = dataset_utils.SolutionIdentifier("TACO", "train", 42, 5)
        assert repr(sid) == "TACO/train/p000042/s0005"

    def test_repr_zero_padded(self) -> None:
        """Test that solution_idx is zero-padded to 4 digits."""
        sid = dataset_utils.SolutionIdentifier("DS", "sub", 1, 1)
        assert repr(sid) == "DS/sub/p000001/s0001"

    def test_from_string_parses_valid_identifier(self) -> None:
        """Test parsing a valid identifier string."""
        sid = dataset_utils.SolutionIdentifier.from_string("TACO/train/p000042/s0005")
        assert sid.dataset == "TACO"
        assert sid.subset == "train"
        assert sid.problem_idx == 42
        assert sid.solution_idx == 5

    def test_from_string_roundtrip(self) -> None:
        """Test that from_string and __repr__ are inverses."""
        original = dataset_utils.SolutionIdentifier("TACO", "valid", 12345, 99)
        roundtrip = dataset_utils.SolutionIdentifier.from_string(repr(original))
        assert roundtrip == original

    def test_from_string_raises_on_missing_solution(self) -> None:
        """Test that from_string raises ValueError if '/s' is missing."""
        with pytest.raises(ValueError, match="must contain '/s'"):
            dataset_utils.SolutionIdentifier.from_string("TACO/train/p000042")

    def test_from_string_raises_on_invalid_solution_idx(self) -> None:
        """Test that from_string raises ValueError for non-numeric solution index."""
        with pytest.raises(ValueError, match="invalid solution index"):
            dataset_utils.SolutionIdentifier.from_string("TACO/train/p000042/sabc")

    def test_get_parent_identifier(self) -> None:
        """Test that get_parent_identifier returns a CodingProblemIdentifier."""
        sid = dataset_utils.SolutionIdentifier("TACO", "train", 42, 5)
        parent = sid.get_parent_identifier()
        assert isinstance(parent, dataset_utils.CodingProblemIdentifier)
        assert not isinstance(parent, dataset_utils.SolutionIdentifier)
        assert parent.dataset == "TACO"
        assert parent.subset == "train"
        assert parent.problem_idx == 42

    def test_inherits_from_coding_problem_identifier(self) -> None:
        """Test that SolutionIdentifier is a subclass of CodingProblemIdentifier."""
        sid = dataset_utils.SolutionIdentifier("TACO", "train", 42, 5)
        assert isinstance(sid, dataset_utils.CodingProblemIdentifier)

    def test_frozen(self) -> None:
        """Test that the dataclass is frozen (immutable)."""
        sid = dataset_utils.SolutionIdentifier("TACO", "train", 1, 1)
        with pytest.raises(AttributeError):
            sid.solution_idx = 2  # noqa


class TestTraceIdentifier:
    """Tests for TraceIdentifier."""

    def test_repr_without_augmentation(self) -> None:
        """Test __repr__ for non-augmented traces."""
        tid = dataset_utils.TraceIdentifier("TACO", "train", 42, 5, 3)
        assert repr(tid) == "TACO/train/p000042/s0005/t0003"

    def test_repr_with_augmentation(self) -> None:
        """Test __repr__ for augmented traces."""
        tid = dataset_utils.TraceIdentifier("TACO", "train", 42, 5, 3, "issues_todos", 1)
        assert repr(tid) == "TACO/train/p000042/s0005/t0003/a:issues_todos:001"

    def test_from_string_without_augmentation(self) -> None:
        """Test parsing a non-augmented trace identifier."""
        tid = dataset_utils.TraceIdentifier.from_string("TACO/train/p000042/s0005/t0003")
        assert tid.dataset == "TACO"
        assert tid.subset == "train"
        assert tid.problem_idx == 42
        assert tid.solution_idx == 5
        assert tid.test_idx == 3
        assert tid.augment_category is None
        assert tid.augment_idx is None

    def test_from_string_with_augmentation(self) -> None:
        """Test parsing an augmented trace identifier."""
        tid = dataset_utils.TraceIdentifier.from_string("TACO/train/p000042/s0005/t0003/a:issues_todos:001")
        assert tid.test_idx == 3
        assert tid.augment_category == "issues_todos"
        assert tid.augment_idx == 1

    def test_from_string_with_combined_categories(self) -> None:
        """Test parsing augmented trace with combined categories (using '+')."""
        tid = dataset_utils.TraceIdentifier.from_string("TACO/train/p000042/s0005/t0003/a:issues_todos+hints_docs:005")
        assert tid.augment_category == "issues_todos+hints_docs"
        assert tid.augment_idx == 5

    def test_from_string_roundtrip_without_augmentation(self) -> None:
        """Test roundtrip for non-augmented identifier."""
        original = dataset_utils.TraceIdentifier("TACO", "valid", 100, 50, 25)
        roundtrip = dataset_utils.TraceIdentifier.from_string(repr(original))
        assert roundtrip == original

    def test_from_string_roundtrip_with_augmentation(self) -> None:
        """Test roundtrip for augmented identifier."""
        original = dataset_utils.TraceIdentifier("TACO", "valid", 100, 50, 25, "hints_execution", 42)
        roundtrip = dataset_utils.TraceIdentifier.from_string(repr(original))
        assert roundtrip == original

    def test_from_string_raises_on_missing_test(self) -> None:
        """Test that from_string raises ValueError if '/t' is missing."""
        with pytest.raises(ValueError, match="must contain '/t'"):
            dataset_utils.TraceIdentifier.from_string("DS/sub/p000042/s0005")

    def test_from_string_raises_on_invalid_test_idx(self) -> None:
        """Test that from_string raises ValueError for non-numeric test index."""
        with pytest.raises(ValueError, match="invalid test index"):
            dataset_utils.TraceIdentifier.from_string("TACO/train/p000042/s0005/tabc")

    def test_from_string_raises_on_invalid_augment_format(self) -> None:
        """Test that from_string raises ValueError for malformed augment section."""
        with pytest.raises(ValueError, match="invalid augmentation format"):
            dataset_utils.TraceIdentifier.from_string("TACO/train/p000042/s0005/t0003/a:bad")

    def test_from_string_raises_on_invalid_augment_idx(self) -> None:
        """Test that from_string raises ValueError for non-numeric augment index."""
        with pytest.raises(ValueError, match="invalid augment index"):
            dataset_utils.TraceIdentifier.from_string("TACO/train/p000042/s0005/t0003/a:cat:xyz")

    def test_post_init_raises_on_partial_augmentation(self) -> None:
        """Test that __post_init__ raises ValueError if only one of augment fields is set."""
        with pytest.raises(ValueError, match="both augment_category and augment_idx must be present"):
            dataset_utils.TraceIdentifier("TACO", "train", 1, 1, 1, augment_category="issues", augment_idx=None)
        with pytest.raises(ValueError, match="both augment_category and augment_idx must be present"):
            dataset_utils.TraceIdentifier("TACO", "train", 1, 1, 1, augment_category=None, augment_idx=5)

    def test_post_init_raises_on_uncleaned_category(self) -> None:
        """Test that __post_init__ raises ValueError if category contains banned chars."""
        with pytest.raises(ValueError, match="should have been cleaned up"):
            dataset_utils.TraceIdentifier("TACO", "train", 1, 1, 1, "issues/bad", 0)

    def test_get_augmentless_identifier(self) -> None:
        """Test that get_augmentless_identifier removes augmentation info."""
        augmented = dataset_utils.TraceIdentifier("TACO", "train", 42, 5, 3, "issues_todos", 1)
        non_augmented = augmented.get_augmentless_identifier()
        assert non_augmented.augment_category is None
        assert non_augmented.augment_idx is None
        assert non_augmented.test_idx == 3
        assert non_augmented.solution_idx == 5

    def test_get_parent_identifier(self) -> None:
        """Test that get_parent_identifier returns a SolutionIdentifier."""
        tid = dataset_utils.TraceIdentifier("TACO", "train", 42, 5, 3)
        parent = tid.get_parent_identifier()
        assert isinstance(parent, dataset_utils.SolutionIdentifier)
        assert not isinstance(parent, dataset_utils.TraceIdentifier)
        assert parent.solution_idx == 5

    def test_inherits_from_solution_identifier(self) -> None:
        """Test that TraceIdentifier is a subclass of SolutionIdentifier."""
        tid = dataset_utils.TraceIdentifier("TACO", "train", 42, 5, 3)
        assert isinstance(tid, dataset_utils.SolutionIdentifier)
        assert isinstance(tid, dataset_utils.CodingProblemIdentifier)


class TestTraceIdentifierAugmentationProperties:
    """Tests for TraceIdentifier augmentation-related properties."""

    def test_is_augmented_false_for_base_trace(self) -> None:
        """Test is_augmented is False when no augmentation."""
        tid = dataset_utils.TraceIdentifier("DS", "sub", 1, 1, 1)
        assert tid.is_augmented is False

    def test_is_augmented_true_for_augmented_trace(self) -> None:
        """Test is_augmented is True when augmentation present."""
        tid = dataset_utils.TraceIdentifier("DS", "sub", 1, 1, 1, "issues_todos", 0)
        assert tid.is_augmented is True

    def test_split_augment_categories_empty(self) -> None:
        """Test split_augment_categories returns empty list for base trace."""
        tid = dataset_utils.TraceIdentifier("DS", "sub", 1, 1, 1)
        assert tid.split_augment_categories == []

    def test_split_augment_categories_single(self) -> None:
        """Test split_augment_categories with single category."""
        tid = dataset_utils.TraceIdentifier("DS", "sub", 1, 1, 1, "issues_todos", 0)
        assert tid.split_augment_categories == ["issues_todos"]

    def test_split_augment_categories_combined(self) -> None:
        """Test split_augment_categories with combined categories."""
        tid = dataset_utils.TraceIdentifier("DS", "sub", 1, 1, 1, "issues_todos+hints_docs", 0)
        assert tid.split_augment_categories == ["issues_todos", "hints_docs"]

    def test_is_bugged_false_for_base_trace(self) -> None:
        """Test is_bugged is False for non-augmented trace."""
        tid = dataset_utils.TraceIdentifier("DS", "sub", 1, 1, 1)
        assert tid.is_bugged is False

    def test_is_bugged_true_for_issues_prefix(self) -> None:
        """Test is_bugged is True for 'issues_' prefixed categories (except issues_docs)."""
        tid = dataset_utils.TraceIdentifier("DS", "sub", 1, 1, 1, "issues_todos", 0)
        assert tid.is_bugged is True

    def test_is_bugged_false_for_issues_docs(self) -> None:
        """Test is_bugged is False for 'issues_docs' (which is misleading, not bugged)."""
        tid = dataset_utils.TraceIdentifier("DS", "sub", 1, 1, 1, "issues_docs", 0)
        assert tid.is_bugged is False
        tid2 = dataset_utils.TraceIdentifier("DS", "sub", 1, 1, 1, "issues_docs_v2", 0)
        assert tid2.is_bugged is False

    def test_is_bugged_true_for_bugged_substring(self) -> None:
        """Test is_bugged is True when category contains 'bugged'."""
        tid = dataset_utils.TraceIdentifier("DS", "sub", 1, 1, 1, "custom_bugged_code", 0)
        assert tid.is_bugged is True

    def test_is_hinted_false_for_base_trace(self) -> None:
        """Test is_hinted is False for non-augmented trace."""
        tid = dataset_utils.TraceIdentifier("DS", "sub", 1, 1, 1)
        assert tid.is_hinted is False

    def test_is_hinted_true_for_hints_prefix(self) -> None:
        """Test is_hinted is True for 'hints_' prefixed categories."""
        tid = dataset_utils.TraceIdentifier("DS", "sub", 1, 1, 1, "hints_execution", 0)
        assert tid.is_hinted is True

    def test_is_hinted_true_for_hinted_substring(self) -> None:
        """Test is_hinted is True when category contains 'hinted'."""
        tid = dataset_utils.TraceIdentifier("DS", "sub", 1, 1, 1, "custom_hinted_code", 0)
        assert tid.is_hinted is True

    def test_is_misleading_false_for_base_trace(self) -> None:
        """Test is_misleading is False for non-augmented trace."""
        tid = dataset_utils.TraceIdentifier("DS", "sub", 1, 1, 1)
        assert tid.is_misleading is False

    def test_is_misleading_true_for_issues_docs(self) -> None:
        """Test is_misleading is True for 'issues_docs' category."""
        tid = dataset_utils.TraceIdentifier("DS", "sub", 1, 1, 1, "issues_docs", 0)
        assert tid.is_misleading is True
        tid2 = dataset_utils.TraceIdentifier("DS", "sub", 1, 1, 1, "issues_docs_v2", 0)
        assert tid2.is_misleading is True

    def test_is_misleading_true_for_misleading_substring(self) -> None:
        """Test is_misleading is True when category contains 'misleading'."""
        tid = dataset_utils.TraceIdentifier("DS", "sub", 1, 1, 1, "misleading", 0)
        assert tid.is_misleading is True

    def test_is_obfuscated_false_for_base_trace(self) -> None:
        """Test is_obfuscated is False for non-augmented trace."""
        tid = dataset_utils.TraceIdentifier("DS", "sub", 1, 1, 1)
        assert tid.is_obfuscated is False

    def test_is_obfuscated_true_for_obfuscated_substring(self) -> None:
        """Test is_obfuscated is True when category contains 'obfuscated'."""
        tid = dataset_utils.TraceIdentifier("DS", "sub", 1, 1, 1, "obfuscated", 0)
        assert tid.is_obfuscated is True

    def test_combined_categories_bugged_and_hinted(self) -> None:
        """Test combined categories can have both bugged and hinted."""
        tid = dataset_utils.TraceIdentifier("DS", "sub", 1, 1, 1, "issues_todos+hints_docs", 0)
        assert tid.is_bugged is True
        assert tid.is_hinted is True  # hints_docs starts with hints_
        assert tid.is_misleading is False
        assert tid.is_obfuscated is False

    def test_combined_categories_misleading_and_obfuscated(self) -> None:
        """Test combined categories with misleading and obfuscated."""
        tid = dataset_utils.TraceIdentifier("DS", "sub", 1, 1, 1, "issues_docs+obfuscated", 0)
        assert tid.is_bugged is False  # issues_docs is excluded from bugged
        assert tid.is_hinted is False
        assert tid.is_misleading is True
        assert tid.is_obfuscated is True


class TestAugmentPatterns:
    """Tests for AugmentPatterns constants."""

    def test_constants_exist(self) -> None:
        """Test that all expected constants are defined."""
        assert hasattr(dataset_utils.AugmentPatterns, "BUGGED_PREFIX")
        assert hasattr(dataset_utils.AugmentPatterns, "HINTED_PREFIX")
        assert hasattr(dataset_utils.AugmentPatterns, "OBFUSCATED")
        assert hasattr(dataset_utils.AugmentPatterns, "MISLEADING")
        assert hasattr(dataset_utils.AugmentPatterns, "DOCS_EXCEPTION")
        assert hasattr(dataset_utils.AugmentPatterns, "DOCS_EXCEPTION_V2")
        assert hasattr(dataset_utils.AugmentPatterns, "BUGGED_SUBSTRING")
        assert hasattr(dataset_utils.AugmentPatterns, "HINTED_SUBSTRING")

    def test_bugged_prefix_value(self) -> None:
        """Test BUGGED_PREFIX has expected value."""
        assert dataset_utils.AugmentPatterns.BUGGED_PREFIX == "issues_"

    def test_hinted_prefix_value(self) -> None:
        """Test HINTED_PREFIX has expected value."""
        assert dataset_utils.AugmentPatterns.HINTED_PREFIX == "hints_"

    def test_docs_exception_value(self) -> None:
        """Test DOCS_EXCEPTION has expected value."""
        assert dataset_utils.AugmentPatterns.DOCS_EXCEPTION == "issues_docs"

    def test_docs_exception_v2_value(self) -> None:
        """Test DOCS_EXCEPTION has expected value."""
        assert dataset_utils.AugmentPatterns.DOCS_EXCEPTION_V2 == "issues_docs_v2"

    def test_stubbed_prefix_value(self) -> None:
        """Test STUBBED_PREFIX has expected value."""
        assert dataset_utils.AugmentPatterns.STUBBED_PREFIX == "code_stubbing"

    def test_stubbed_value(self) -> None:
        """Test STUBBED has expected value."""
        assert dataset_utils.AugmentPatterns.STUBBED == "stubbed"

    def test_obfuscated_value(self) -> None:
        """Test OBFUSCATED has expected value."""
        assert dataset_utils.AugmentPatterns.OBFUSCATED == "obfuscated"

    def test_misleading_value(self) -> None:
        """Test MISLEADING has expected value."""
        assert dataset_utils.AugmentPatterns.MISLEADING == "misleading"


class TestAugmentPatternsGetCleanCategory:
    """Tests for AugmentPatterns.get_clean_augment_category static method."""

    def test_clean_category_unchanged(self) -> None:
        """Test that already clean categories are unchanged."""
        assert dataset_utils.AugmentPatterns.get_clean_augment_category("issues_todos") == "issues_todos"

    def test_replaces_forward_slash(self) -> None:
        """Test that forward slashes are replaced with underscores."""
        assert dataset_utils.AugmentPatterns.get_clean_augment_category("a/b") == "a_b"

    def test_replaces_comma(self) -> None:
        """Test that commas are replaced with underscores."""
        assert dataset_utils.AugmentPatterns.get_clean_augment_category("a,b") == "a_b"

    def test_replaces_space(self) -> None:
        """Test that spaces are replaced with underscores."""
        assert dataset_utils.AugmentPatterns.get_clean_augment_category("a b") == "a_b"

    def test_replaces_colon(self) -> None:
        """Test that colons are replaced with underscores."""
        assert dataset_utils.AugmentPatterns.get_clean_augment_category("a:b") == "a_b"

    def test_replaces_multiple_chars(self) -> None:
        """Test that multiple banned characters are all replaced."""
        assert dataset_utils.AugmentPatterns.get_clean_augment_category("a/b,c d:e") == "a_b_c_d_e"
