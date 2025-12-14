"""String manipulation and text parsing utilities.

This module provides:
- Path prefix normalization for hierarchical keys
- XML-like tag regex compilation for parsing structured text (e.g., model outputs)

The tag regex functions use LRU caching to avoid recompiling the same patterns repeatedly.
"""

import functools
import re

__all__ = [
    "normalize_path_prefix",
    "compile_open_tag_regex",
    "compile_close_tag_regex",
]


def normalize_path_prefix(prefix: str) -> str:
    """Normalize a path prefix to either empty string or a trailing-slash form.

    This is useful for constructing hierarchical keys in logging or metrics systems
    where prefixes should consistently end with a separator.

    Args:
        prefix: The prefix string to normalize.

    Returns:
        Empty string if prefix is empty/whitespace, otherwise prefix with trailing slash.

    Examples:
        >>> normalize_path_prefix("")
        ''
        >>> normalize_path_prefix("reward")
        'reward/'
        >>> normalize_path_prefix("reward/")
        'reward/'
    """
    stripped = prefix.strip()
    if not stripped:
        return ""
    return stripped if stripped.endswith("/") else f"{stripped}/"


@functools.lru_cache(maxsize=128)
def compile_open_tag_regex(tag: str) -> re.Pattern[str]:
    """Compile a case-insensitive regex matching `<tag ...>` opening tags.

    The pattern supports optional attributes (e.g., `<tag attr="value">`).

    Args:
        tag: Tag name (will be stripped and escaped for regex safety).

    Returns:
        Compiled regex pattern that matches opening tags.

    Raises:
        ValueError: If tag name is empty after stripping.

    Examples:
        ```python
        import pyine.utils.strings

        regex = pyine.utils.strings.compile_open_tag_regex("final")
        assert regex.search("<final>") is not None
        assert regex.search("<final attr='value'>") is not None
        assert regex.search("<FINAL>") is not None  # case insensitive
        assert regex.search("</final>") is None  # does not match close tags
        ```
    """
    tag_stripped = tag.strip()
    if not tag_stripped:
        raise ValueError("tag name cannot be empty")
    return re.compile(rf"<{re.escape(tag_stripped)}(?:\s[^>]*)?>", re.IGNORECASE)


@functools.lru_cache(maxsize=128)
def compile_close_tag_regex(tag: str) -> re.Pattern[str]:
    """Compile a case-insensitive regex matching `</tag>` closing tags.

    Args:
        tag: Tag name (will be stripped and escaped for regex safety).

    Returns:
        Compiled regex pattern that matches closing tags.

    Raises:
        ValueError: If tag name is empty after stripping.

    Examples:
        ```python
        import pyine.utils.strings

        regex = pyine.utils.strings.compile_close_tag_regex("final")
        assert regex.search("</final>") is not None
        assert regex.search("</FINAL>") is not None  # case insensitive
        assert regex.search("</final >") is not None  # allows trailing space
        assert regex.search("<final>") is None  # does not match open tags
        ```
    """
    tag_stripped = tag.strip()
    if not tag_stripped:
        raise ValueError("tag name cannot be empty")
    return re.compile(rf"</{re.escape(tag_stripped)}\s*>", re.IGNORECASE)
