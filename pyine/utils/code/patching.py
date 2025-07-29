import difflib
import logging
import typing

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
    output = "\n".join([d.rstrip("\r\n") for d in diff]) + "\n"
    return output


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
            logger.error("failed to parse patch hunks")
            return text, False
        if len(patch_set) != 1:
            logger.error("this implementation handles single-file patches only")
            return text, False
        patched_file = patch_set[0]
        for hunk in patched_file:
            # add lines from the original file that come before this hunk
            pre_hunk_line_count = hunk.source_start - 1 - original_line_idx
            if pre_hunk_line_count < 0:
                logger.error("hunks cannot be applied out of order")
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
                        logger.error("hunk context does not match original file")
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
    except Exception as e:
        # any error during parsing or applying means failure
        logger.error(f"error applying patch: {e}")
        return text, False
