"""Central registry of prompt name constants.

This module defines the canonical names for all prompts in the framework. These names
correspond to actual prompt templates in the `templates/` directory. Using these constants
ensures consistency and provides a single source of truth for prompt names.

The prompt names follow a hierarchical convention using `/` as a separator:
- Root-level prompts: "code_stubbing", "code_summary"
- Hierarchical prompts: "hints/docs", "issues/iterators"
"""

import typing

from pyine.prompts.manager import list_prompts


class PromptNames:
    """Central registry of prompt name constants.

    These constants correspond to actual prompt templates registered in the framework.
    Use `validate_all()` to verify that all constants map to existing prompts.

    Note all prompt names may be listed here, as some prompts are application-specific and
    likely not useful outside those applications.
    """

    # --- prefixes for prompt categories ---

    HINTS_PREFIX: typing.Final = "hints/"
    """Prefix for hint-providing prompts."""
    ISSUES_PREFIX: typing.Final = "issues/"
    """Prefix for bug-introducing prompts."""
    VALIDATION_PREFIX: typing.Final = "validation/"
    """Prefix for validation prompts."""

    # --- individual prompt names ---

    CODE_EXECUTION: typing.Final = "code_execution"
    """Prompt for code execution analysis."""
    CODE_STUBBING: typing.Final = "code_stubbing"
    """Prompt for stubbing/hiding parts of code."""
    CODE_SUMMARY: typing.Final = "code_summary"
    """Prompt for generating code summaries."""
    PRED_GRADER: typing.Final = "pred_grader"
    """Prompt for grading predictions."""
    HINTS_DOCS: typing.Final = "hints/docs"
    """Prompt for generating documentation-based hints."""
    HINTS_TESTS: typing.Final = "hints/tests"
    """Prompt for generating test-based hints."""
    ISSUES_DOCS: typing.Final = "issues/docs"
    """Prompt for misleading documentation (alias for hints/docs with different semantics)."""
    ISSUES_DOCS_V2: typing.Final = "issues/docs_v2"
    """Prompt for dedicated misleading documentation injection."""
    ISSUES_ITERATORS: typing.Final = "issues/iterators"
    """Prompt for introducing iterator-related bugs."""
    ISSUES_TODOS: typing.Final = "issues/todos"
    """Prompt for introducing TODO-related bugs."""
    VALIDATION_MISLEADING: typing.Final = "validation/misleading"
    """Prompt for validating whether misleading hints are truly misleading."""

    @classmethod
    def get_all_names(cls) -> list[str]:
        """Returns all registered prompt name constants (excluding prefixes)."""
        return [
            cls.CODE_EXECUTION,
            cls.CODE_STUBBING,
            cls.CODE_SUMMARY,
            cls.PRED_GRADER,
            cls.HINTS_DOCS,
            cls.HINTS_TESTS,
            cls.ISSUES_DOCS,
            cls.ISSUES_DOCS_V2,
            cls.ISSUES_ITERATORS,
            cls.ISSUES_TODOS,
            cls.VALIDATION_MISLEADING,
        ]

    @classmethod
    def validate_all(cls) -> None:
        """Validates that all prompt name constants correspond to registered prompts.

        Raises:
            ValueError: If any constant does not match a registered prompt.
        """
        registered = set(list_prompts())
        for name in cls.get_all_names():
            if name not in registered:
                raise ValueError(
                    f"Prompt name constant '{name}' does not correspond to a registered prompt. "
                    f"Available prompts: {sorted(registered)}"
                )

    @classmethod
    def is_valid(cls, name: str) -> bool:
        """Checks if a prompt name is registered in the framework."""
        return name in list_prompts()
