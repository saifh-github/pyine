"""Utilities for attaching prefixes and suffixes to code snippets."""

DEFAULT_BLOCK_START_SUFFIX = "  # <<<< START HERE"
"""Default suffix to mark the start of a code block of interest."""

DEFAULT_BLOCK_END_SUFFIX = "  # <<<< END HERE"
"""Default suffix to mark the end of a code block of interest."""


def get_code_with_numbered_lines(
    code_string: str,
    prefixed_tabs: int = 0,
) -> str:
    """Generates a string containing code with line numbers and tabs.

    Each line is prefixed with a right-aligned, space-padded line number followed by a colon
    and a single space. The width auto-adjusts to fit the total number of lines.

    Args:
        code_string: The Python code string to be formatted.
        prefixed_tabs: The number of tabs to prefix each line with. Defaults to 0.

    Returns:
        The formatted code string with line numbers and tabs.

    Examples:
        >>> code = "a = 1\\nb = 2"
        >>> get_code_with_numbered_lines(code)
        '1: a = 1\\n2: b = 2'
        >>> get_code_with_numbered_lines(code, prefixed_tabs=1)
        '\\t1: a = 1\\n\\t2: b = 2'
    """
    lines = code_string.splitlines()
    line_count = len(lines)
    num_width = len(str(line_count))
    tab_prefix = "\t" * prefixed_tabs
    formatted_lines: list[str] = []
    for line_idx, line_content in enumerate(lines, start=1):
        formatted_lines.append(f"{tab_prefix}{line_idx:>{num_width}}: {line_content}")
    return "\n".join(formatted_lines)


def get_code_with_block_markers(
    code_string: str,
    start_line: int,
    end_line: int,
    start_suffix: str = DEFAULT_BLOCK_START_SUFFIX,
    end_suffix: str = DEFAULT_BLOCK_END_SUFFIX,
) -> str:
    """Adds comment-based suffixes to mark the start and end of a code block of interest.

    Line numbers are 1-indexed. If start_line equals end_line, only the start suffix is added.

    Args:
        code_string: The code string to annotate.
        start_line: The 1-indexed line number where the block starts.
        end_line: The 1-indexed line number where the block ends (inclusive).
        start_suffix: The suffix to append to the start line. Defaults to
            ``"  # <<<< START HERE"``.
        end_suffix: The suffix to append to the end line. Defaults to
            ``"  # <<<< END HERE"``.

    Returns:
        The code string with block markers added.

    Raises:
        ValueError: If start_line > end_line, or if line numbers are out of bounds.

    Examples:
        >>> code = "a = 1\\nb = 2\\nc = 3"
        >>> get_code_with_block_markers(code, start_line=1, end_line=2)
        'a = 1  # <<<< START HERE\\nb = 2  # <<<< END HERE\\nc = 3'
        >>> get_code_with_block_markers(code, start_line=2, end_line=2)
        'a = 1\\nb = 2  # <<<< START HERE\\nc = 3'
    """
    if start_line > end_line:
        raise ValueError(f"start_line ({start_line}) must be <= end_line ({end_line})")
    lines = code_string.splitlines()
    line_count = len(lines)
    if start_line < 1 or start_line > line_count:
        raise ValueError(f"start_line ({start_line}) out of bounds (1 to {line_count})")
    if end_line < 1 or end_line > line_count:
        raise ValueError(f"end_line ({end_line}) out of bounds (1 to {line_count})")
    start_idx = start_line - 1
    end_idx = end_line - 1
    if start_idx == end_idx:
        lines[start_idx] = lines[start_idx] + start_suffix
    else:
        lines[start_idx] = lines[start_idx] + start_suffix
        lines[end_idx] = lines[end_idx] + end_suffix
    return "\n".join(lines)
