"""Utilities for attaching prefixes and suffixes to code snippets."""

DEFAULT_LINE_PREFIX_PATTERN = "L{num}|"
"""Default pattern for line number prefixes; {num} is replaced with the formatted line number."""

DEFAULT_BLOCK_START_SUFFIX = "  # <<<< START HERE"
"""Default suffix to mark the start of a code block of interest."""

DEFAULT_BLOCK_END_SUFFIX = "  # <<<< END HERE"
"""Default suffix to mark the end of a code block of interest."""


def get_code_with_numbered_lines(
    code_string: str,
    prefixed_tabs: int = 0,
    prefix_pattern: str = DEFAULT_LINE_PREFIX_PATTERN,
    num_width: int | None = None,
    zero_pad: bool = True,
) -> str:
    """Generates a string containing code with line numbers and tabs.

    Args:
        code_string: The Python code string to be formatted.
        prefixed_tabs: The number of tabs to prefix each line with. Defaults to 0.
        prefix_pattern: Pattern for the line prefix where ``{num}`` is replaced with the
            formatted line number. Examples: ``"L{num}|"``, ``"{num}|"``, ``"L{num}: "``.
        num_width: Width for the line number padding. If None, auto-detected based on the
            total number of lines.
        zero_pad: If True (default), pad line numbers with zeros (e.g., ``"001"``).
            If False, right-align with spaces (e.g., ``"  1"``).

    Returns:
        The formatted code string with line numbers and tabs.

    Examples:
        >>> code = "a = 1\\nb = 2"
        >>> get_code_with_numbered_lines(code)
        'L1|a = 1\\nL2|b = 2'
        >>> get_code_with_numbered_lines(code, prefix_pattern="{num}|", num_width=3, zero_pad=False)
        '  1|a = 1\\n  2|b = 2'
        >>> get_code_with_numbered_lines(code, num_width=4)
        'L0001|a = 1\\nL0002|b = 2'
    """
    lines = code_string.splitlines()
    line_count = len(lines)
    if num_width is None:
        num_width = len(str(line_count))
    tab_prefix = "\t" * prefixed_tabs
    formatted_lines: list[str] = []
    for line_idx, line_content in enumerate(lines, start=1):
        if zero_pad:
            formatted_num = str(line_idx).zfill(num_width)
        else:
            formatted_num = str(line_idx).rjust(num_width)
        line_prefix = prefix_pattern.format(num=formatted_num)
        formatted_lines.append(f"{tab_prefix}{line_prefix}{line_content}")
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
