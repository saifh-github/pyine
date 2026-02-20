#!/usr/bin/env python3
"""Pre-commit hook that checks only staged (added/modified) lines for non-ASCII bytes.

Existing non-ASCII content in unchanged lines is ignored, so legacy files don't
need to be cleaned up all at once.
"""

from __future__ import annotations

import subprocess
import sys


def main(paths: list[str]) -> int:
    if not paths:
        return 0
    # get the unified diff of staged changes for the given files (no context lines)
    result = subprocess.run(
        ["git", "diff", "--cached", "-U0", "--", *paths],  # noqa: S607
        capture_output=True,
    )
    if result.returncode != 0:
        return 0  # not in a git repo or no staged changes; nothing to check
    current_file = ""
    current_hunk_line = 0
    found = False
    for raw_line in result.stdout.split(b"\n"):
        if raw_line.startswith(b"+++ b/"):
            current_file = raw_line[6:].decode("utf-8", errors="replace")
            continue
        if raw_line.startswith(b"@@ "):
            # parse hunk header: @@ -old,count +new,count @@
            # extract the +new start line number
            plus_part = raw_line.split(b"+")[1].split(b" ")[0]
            current_hunk_line = int(plus_part.split(b",")[0]) - 1
            continue
        if raw_line.startswith(b"+") and not raw_line.startswith(b"+++"):
            current_hunk_line += 1
            line_content = raw_line[1:]  # strip the leading '+'
            for col, byte_val in enumerate(line_content):
                if byte_val > 127:
                    char = chr(byte_val) if byte_val < 256 else "?"
                    # try to decode the full character for multi-byte UTF-8
                    try:
                        snippet = line_content[col : col + 4].decode("utf-8", errors="replace")
                        char = snippet[0]
                    except (IndexError, UnicodeDecodeError):
                        pass
                    print(f"{current_file}:{current_hunk_line}:{col} non-ASCII: {char!r} (0x{byte_val:02X})")
                    found = True
                    break  # one report per line
        elif raw_line.startswith(b"-") and not raw_line.startswith(b"---"):
            pass  # removed line, don't advance line counter
        else:
            if not raw_line.startswith(b"diff ") and not raw_line.startswith(b"index "):
                current_hunk_line += 1  # context line (shouldn't appear with -U0)
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
