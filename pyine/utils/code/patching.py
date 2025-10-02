import difflib
import html
import logging

import IPython.display
import unidiff

logger = logging.getLogger(__name__)


def compute_patch(
    original: str,
    modified: str,
    context_lines: int = 3,
) -> str:
    """Computes the patch (diff) between two texts using line-based diffing.

    Returns:
        The patch in unified diff format.
    """
    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        modified.splitlines(keepends=True),
        fromfile="original",
        tofile="modified",
        n=context_lines,
    )
    return "\n".join(d.rstrip("\r\n") for d in diff) + "\n"


def apply_patch(
    text: str,
    patch_text: str,
) -> tuple[str, bool]:
    """Applies a unified diff patch to a given text using the 'unidiff' library.

    Returns:
        A tuple containing:
        - The resulting text (which may be unchanged if patching failed)
        - A boolean indicating whether the patch was applied successfully
    """
    original_lines = text.splitlines(keepends=True)
    patched_lines = []
    original_line_idx = 0
    try:
        patch_set = unidiff.PatchSet.from_string(patch_text)
        if len(patch_set) == 0:
            logger.info("failed to parse patch hunks")
            return text, False
        if len(patch_set) != 1:
            logger.info("this implementation handles single-file patches only")
            return text, False
        patched_file = patch_set[0]
        for hunk in patched_file:
            # add lines from the original file that come before this hunk
            pre_hunk_line_count = hunk.source_start - 1 - original_line_idx
            if pre_hunk_line_count < 0:
                logger.info("hunks cannot be applied out of order")
                return text, False
            patched_lines.extend(original_lines[original_line_idx : original_line_idx + pre_hunk_line_count])
            original_line_idx += pre_hunk_line_count
            # verify that the hunk context matches the original file
            hunk_original_idx = 0
            for line in hunk:
                if line.is_removed or line.is_context:
                    offset_idx = original_line_idx + hunk_original_idx
                    if offset_idx >= len(original_lines) or original_lines[offset_idx].rstrip(
                        "\r\n"
                    ) != line.value.rstrip("\r\n"):
                        logger.info("hunk context does not match original file")
                        return text, False
                    hunk_original_idx += 1
            # the hunk is valid, apply the changes
            for line in hunk:
                if line.is_added or line.is_context:
                    patched_lines.append(line.value)
            # advance the index past the lines consumed from the original file
            original_line_idx += hunk.source_length
        # add any remaining lines from the original file
        patched_lines.extend(original_lines[original_line_idx:])
        return "".join(patched_lines), True
    except Exception:
        # any error during parsing or applying means failure
        logger.exception("failed to apply patch")
        return text, False


def show_colored_diff(
    diff: str,
) -> None:
    """Display a colored unified diff in a Jupyter notebook.

    Args:
        diff: The unified diff to display.
    """
    styled_lines: list[str] = []
    for line in diff.splitlines(keepends=True):
        escaped = html.escape(line.rstrip("\n"))
        if line.startswith("+") and not line.startswith("+++"):
            color = "#22863a"  # green
        elif line.startswith("-") and not line.startswith("---"):
            color = "#b31d28"  # red
        elif line.startswith("@@"):
            color = "#8250df"  # purple
        else:
            color = "#6a737d"  # grey
        styled_lines.append(f'<span style="color:{color}; white-space:pre">{escaped}</span>')
    html_content = "<br>".join(styled_lines)
    IPython.display.display(IPython.display.HTML(html_content))
