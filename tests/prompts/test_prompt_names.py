"""Tests for prompt name constants in pyine.prompts.names."""

import pyine.prompts
from pyine.prompts.names import PromptNames


class TestPromptNames:
    """Tests for the PromptNames class."""

    def test_all_names_are_registered(self) -> None:
        """Test that all prompt name constants correspond to registered prompts."""
        registered = set(pyine.prompts.list_prompts())
        for name in PromptNames.get_all_names():
            assert name in registered, f"prompt '{name}' is not registered"

    def test_validate_all_succeeds(self) -> None:
        """Test that validate_all() does not raise for valid constants."""
        PromptNames.validate_all()  # should not raise

    def test_is_valid_for_registered_prompts(self) -> None:
        """Test that is_valid() returns True for registered prompts."""
        for name in PromptNames.get_all_names():
            assert PromptNames.is_valid(name), f"prompt '{name}' should be valid"

    def test_is_valid_false_for_unregistered(self) -> None:
        """Test that is_valid() returns False for unregistered prompts."""
        assert not PromptNames.is_valid("nonexistent_prompt")

    def test_constants_have_expected_values(self) -> None:
        """Test that prompt name constants have expected string values."""
        assert PromptNames.CODE_EXECUTION == "code_execution"
        assert PromptNames.CODE_STUBBING == "code_stubbing"
        assert PromptNames.CODE_SUMMARY == "code_summary"
        assert PromptNames.PRED_GRADER == "pred_grader"
        assert PromptNames.HINTS_DOCS == "hints/docs"
        assert PromptNames.HINTS_TESTS == "hints/tests"
        assert PromptNames.ISSUES_DOCS == "issues/docs"
        assert PromptNames.ISSUES_DOCS_V2 == "issues/docs_v2"
        assert PromptNames.ISSUES_ITERATORS == "issues/iterators"
        assert PromptNames.ISSUES_TODOS == "issues/todos"

    def test_prefixes_have_expected_values(self) -> None:
        """Test that prefix constants have expected string values."""
        assert PromptNames.HINTS_PREFIX == "hints/"
        assert PromptNames.ISSUES_PREFIX == "issues/"

    def test_get_all_names_returns_list(self) -> None:
        """Test that get_all_names() returns a non-empty list."""
        names = PromptNames.get_all_names()
        assert isinstance(names, list)
        assert len(names) > 0
        assert all(isinstance(name, str) for name in names)
