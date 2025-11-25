"""
This module contains:
- one utility to backfill the existing experiments with a new complexity metric, in case we want to add a new one
- one utility to verify that the backfill is done correctly
- one CLI entrypoint to run the backfill and verify logic
"""

import argparse
import json
import pathlib
import sys

import pyine.utils.code.complexity_metrics


def backfill_missing_metrics(
    base_dir: pathlib.Path,
    dataset_path: pathlib.Path,
    dry_run: bool = False,
) -> None:
    """
    Backfill the missing metrics in all the existing results.

    Compares current COMPLEXITY_METRICS constant with metrics in existing results
    and adds any missing ones by re-analyzing the original code.

    Args:
        base_dir: Base directory containing experiment subdirectories
        dataset_path: Path to dataset directory
        dry_run: If True, only report what would be done without modifying files
    """
    if not base_dir.exists():
        print(f"Directory {base_dir} does not exist")
        return

    if not dataset_path.exists():
        print(f"Dataset {dataset_path} does not exist")
        return

    print(f"{'DRY RUN: ' if dry_run else ''}Backfilling missing metrics in: {base_dir}")
    print(f"Using dataset: {dataset_path}\n")

    expected_metrics = set(pyine.utils.code.complexity_metrics.COMPLEXITY_METRICS)
    print(f"Expected metrics ({len(expected_metrics)}): {sorted(expected_metrics)}\n")

    total_experiments = 0
    total_results_updated = 0
    total_metrics_added = 0

    for exp_name_dir in sorted(base_dir.iterdir()):
        if not exp_name_dir.is_dir():
            continue
        if exp_name_dir.name.endswith("_BACKUP"):
            continue

        for run_dir in sorted(exp_name_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            results_path = run_dir / "results.json"
            if not results_path.exists():
                continue

            with results_path.open() as f:
                results = json.load(f)

            if not results:
                continue

            # detect missing metrics
            existing_metrics = {k for k in results[0] if k in expected_metrics}
            missing_metrics = expected_metrics - existing_metrics

            if not missing_metrics:
                continue

            print(f"{exp_name_dir.name}/{run_dir.name}")
            print(f"  Missing metrics: {sorted(missing_metrics)}")

            if dry_run:
                print(f"  Would update {len(results)} results")
                total_experiments += 1
                total_results_updated += len(results)
                total_metrics_added += len(missing_metrics) * len(results)
                continue

            print(f"  Processing {len(results)} results...")

            success_count = 0
            for result in results:
                problem_id = result["problem_id"]
                solution_id = result["solution_id"]

                problem_num = problem_id.split("/")[-1].replace("p", "")
                solution_num = solution_id.split("/")[-1].replace("s", "")
                solution_idx = int(solution_num)

                try:
                    problem_file = dataset_path / f"{problem_num}.json"
                    with problem_file.open() as f:
                        problem_data = json.load(f)

                    code = problem_data["solutions"][solution_idx]["code"]
                    metrics = pyine.utils.code.complexity_metrics.get_complexity_metrics(code)

                    # add only missing metrics
                    for metric_name in missing_metrics:
                        result[metric_name] = metrics[metric_name]

                    success_count += 1

                except Exception as e:
                    print(f"    Warning: Could not process {solution_id}: {e}")
                    for metric_name in missing_metrics:
                        result[metric_name] = None

            # save updated results
            with results_path.open("w") as f:
                json.dump(results, f, indent=2)

            print(f"  Done: {success_count}/{len(results)} results updated")
            total_results_updated += len(results)
            total_metrics_added += len(missing_metrics) * len(results)
            total_experiments += 1

    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"  Experiments processed: {total_experiments}")
    print(f"  Total results updated: {total_results_updated}")
    print(f"  Total metric values added: {total_metrics_added}")
    print(f"{'=' * 60}")


def verify_backfill(
    backup_dir: pathlib.Path,
    current_dir: pathlib.Path,
    expected_new_metrics: set[str],
) -> bool:
    """
    Verifies that the backfill added only the expected metrics.

    Args:
        backup_dir: Path to backup directory (use should create one)
        current_dir: Path to current results directory
        expected_new_metrics: Set of metric names that should have been added

    Returns:
        True if verification passed, False otherwise
    """
    print("Comparing:")
    print(f"  Backup:  {backup_dir}")
    print(f"  Current: {current_dir}")
    print(f"  Expected new metrics: {sorted(expected_new_metrics)}\n")

    if not backup_dir.exists():
        print(f" Backup directory not found: {backup_dir}")
        return False

    current_results = sorted(current_dir.rglob("results.json"))
    total_experiments = 0
    total_issues = 0
    total_verified = 0

    for current_file in current_results:
        rel_path = current_file.relative_to(current_dir)
        backup_file = backup_dir / rel_path

        if not backup_file.exists():
            print(f"  {rel_path}: No backup found (new experiment?)")
            continue

        with backup_file.open() as f:
            backup_results = json.load(f)
        with current_file.open() as f:
            current_results_data = json.load(f)

        if len(backup_results) != len(current_results_data):
            print(f" {rel_path}: Different number of results!")
            total_issues += 1
            continue

        backup_keys = set(backup_results[0].keys())
        current_keys = set(current_results_data[0].keys())
        added_metrics = current_keys - backup_keys

        if added_metrics != expected_new_metrics:
            print(f" {rel_path}: Unexpected metrics added!")
            print(f"   Expected: {sorted(expected_new_metrics)}")
            print(f"   Got:      {sorted(added_metrics)}")
            total_issues += 1
            continue

        issues_found: list[str] = []
        for idx, (backup_result, current_result) in enumerate(zip(backup_results, current_results_data, strict=False)):
            for key in backup_result:
                if key not in current_result:
                    issues_found.append(f"  Result {idx}: Missing key '{key}' in current")
                elif backup_result[key] != current_result[key]:
                    issues_found.append(
                        f"  Result {idx}: '{key}' changed from {backup_result[key]} to {current_result[key]}"
                    )

            for metric in expected_new_metrics:
                if metric not in current_result:
                    issues_found.append(f"  Result {idx}: Missing new metric '{metric}'")
                elif current_result[metric] is None:
                    issues_found.append(f"  Result {idx}: New metric '{metric}' is None")

        if issues_found:
            print(f" {rel_path}:")
            for issue in issues_found[:5]:
                print(issue)
            if len(issues_found) > 5:
                print(f"  ... and {len(issues_found) - 5} more issues")
            total_issues += 1
        else:
            print(f" {rel_path}: Only expected metrics added, all data preserved")
            total_verified += 1

        total_experiments += 1

    print(f"\n{'=' * 60}")
    print("Verification Summary:")
    print(f"  Experiments checked: {total_experiments}")
    print(f"  Passed: {total_verified}")
    print(f"  Failed: {total_issues}")
    if total_issues == 0 and total_verified > 0:
        print("\n ALL VERIFICATIONS PASSED - Safe to delete backup")
    else:
        print("\n VERIFICATION FAILED - DO NOT delete backup!")
    print(f"{'=' * 60}")

    return total_issues == 0


def _parse_args() -> argparse.Namespace:
    """Parses and returns command-line arguments for the backfill+verify CLI app."""
    import pyine.data.taco.dataset_utils
    import pyine.utils.filesystem

    parser = argparse.ArgumentParser(description="Backfill and verify complexity metrics in experiment results")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to run")
    default_output_dir = pyine.utils.filesystem.get_logs_root_path() / "code_exec_complexity_results"

    # run subcommand (backfill)
    run_parser = subparsers.add_parser("run", help="Backfill missing complexity metrics")
    run_parser.add_argument(
        "--base-dir",
        type=pathlib.Path,
        default=default_output_dir,
        help="Base directory containing all experiment results",
    )
    default_dataset_path = pyine.data.taco.dataset_utils.get_latest_repackaged_dataset_path()
    run_parser.add_argument(
        "--dataset-path",
        type=pathlib.Path,
        default=default_dataset_path,
        help="Path to dataset directory",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without modifying files",
    )

    # verify subcommand
    verify_parser = subparsers.add_parser("verify", help="Verify backfill operation")
    verify_parser.add_argument(
        "--backup-dir",
        type=pathlib.Path,
        required=True,
        help="Path to backup directory",
    )
    verify_parser.add_argument(
        "--current-dir",
        type=pathlib.Path,
        default=default_output_dir,
        help="Path to current results directory",
    )

    return parser.parse_args()


def _main() -> None:
    """Entrypoint for running the verify+backfill logic."""
    args = _parse_args()
    if args.command == "run":
        backfill_missing_metrics(args.base_dir, args.dataset_path, args.dry_run)
    elif args.command == "verify":
        backup_dir = args.backup_dir
        current_dir = args.current_dir

        if not backup_dir.exists():
            print(f"Backup directory not found: {backup_dir}")
            print("\nAvailable backups:")
            for d in pathlib.Path("logs").glob("code_exec_complexity_results_BACKUP_*"):
                print(f"  {d}")
            sys.exit(1)

        # auto-detect which metrics should have been added
        sample_backup = next(backup_dir.rglob("results.json"))
        with sample_backup.open() as f:
            sample_results = json.load(f)

        expected_metrics = set(pyine.utils.code.complexity_metrics.COMPLEXITY_METRICS)
        existing_metrics = {k for k in sample_results[0] if k in expected_metrics}
        new_metrics = expected_metrics - existing_metrics

        success = verify_backfill(backup_dir, current_dir, new_metrics)
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    import pyine.utils.logging
    import pyine.utils.reprod

    pyine.utils.reprod.load_dotenv()
    pyine.utils.logging.setup_logging()
    _main()
