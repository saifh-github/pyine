#!/usr/bin/env python3
"""One-time migration script to rename prompt_name 'hints/stubs' to 'code_stubbing'."""

import argparse
import pathlib
import sqlite3
import sys


def migrate_prompt_name(db_path: pathlib.Path, dry_run: bool = False) -> int:
    """Update prompt_name from 'hints/stubs' to 'code_stubbing'.

    Args:
        db_path: Path to the SQLite database file.
        dry_run: If True, only report what would be changed without modifying.

    Returns:
        Number of records updated (or that would be updated in dry-run mode).
    """
    if not db_path.exists():
        print(f"Error: Database file not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(db_path, timeout=30.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=30000;")

        # Count affected records
        row = conn.execute(
            "SELECT COUNT(*) FROM items WHERE prompt_name = ?",
            ("hints/stubs",),
        ).fetchone()
        count = row[0] if row else 0

        if count == 0:
            print("No records found with prompt_name='hints/stubs'. Nothing to migrate.")
            return 0

        print(f"Found {count} record(s) with prompt_name='hints/stubs'.")

        if dry_run:
            print("[DRY RUN] Would update prompt_name to 'code_stubbing'.")
            return count

        # Perform the update
        conn.execute(
            "UPDATE items SET prompt_name = ? WHERE prompt_name = ?",
            ("code_stubbing", "hints/stubs"),
        )
        conn.commit()
        print(f"Successfully updated {count} record(s) to prompt_name='code_stubbing'.")
        return count
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate prompt_name 'hints/stubs' to 'code_stubbing' in a PromptResultDB."
    )
    parser.add_argument(
        "db_path",
        type=pathlib.Path,
        help="Path to the SQLite database file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be changed without modifying the database.",
    )
    args = parser.parse_args()

    migrate_prompt_name(args.db_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
