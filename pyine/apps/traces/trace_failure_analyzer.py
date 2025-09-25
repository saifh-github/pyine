"""Command line tool to analyze tracing failures for coding snippet datasets."""

import datetime
import itertools
import json
import logging
import pathlib
import time
import traceback
import typing

import click

import pyine.data.taco.dataset_utils
import pyine.data.traces.dataset_utils
import pyine.data.traces.dataset_writer
import pyine.utils.code.execution
import pyine.utils.filesystem
import pyine.utils.logging
from pyine.data.traces.dataset_writer import TraceDatasetWriterConfig, _CodeToTrace

logger = logging.getLogger(__name__)


def _json_default(value: typing.Any) -> typing.Any:
    """Fallback serializer for complex objects when dumping JSON."""
    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            return value.model_dump()
        except TypeError:
            return value.model_dump(exclude_none=False)
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, datetime.datetime):
        return value.isoformat()
    if isinstance(value, datetime.date):
        return value.isoformat()
    if isinstance(value, pyine.data.traces.dataset_utils.TraceIdentifier):
        return str(value)
    return repr(value)


def _to_json_compatible(value: typing.Any) -> typing.Any:
    """Convert an arbitrary value to JSON-serializable structures when feasible."""
    try:
        json.dumps(value)
        return value
    except TypeError:
        try:
            return json.loads(json.dumps(value, default=_json_default))
        except TypeError:
            return repr(value)


def _summarize_trace_result(
    trace_result: pyine.utils.code.execution.TraceResult,
) -> dict[str, typing.Any]:
    """Return a concise dictionary with the most useful trace metadata."""
    metadata = trace_result.metadata if isinstance(trace_result.metadata, dict) else {}
    summary = {
        "entrypoint_name": trace_result.entrypoint_name,
        "valid_step_count": trace_result.valid_step_count,
        "total_step_count": trace_result.total_step_count,
        "tags": trace_result.tags,
        "max_valid_events": trace_result.max_valid_events,
        "max_events_per_line": trace_result.max_events_per_line,
        "max_var_repr_length": trace_result.max_var_repr_length,
    }
    # surface a handful of environment fields if present without duplicating the full metadata blob
    for key in [
        "python_version",
        "framework_version",
        "runtime_hash",
        "platform",
    ]:
        if key in metadata:
            summary[key] = metadata[key]
    return summary


def _format_exception(exc: Exception) -> dict[str, typing.Any]:
    """Return structured information for a caught exception."""
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def _build_trace_request(
    problem: pyine.data.traces.dataset_utils.CodingProblem,
    solution: pyine.data.traces.dataset_utils.Solution,
    test_idx: int,
    inputs: typing.Any,
    outputs: typing.Any,
) -> _CodeToTrace:
    """Build a TraceRequest object from a CodingProblem and Solution."""
    trace_identifier = pyine.data.traces.dataset_utils.TraceIdentifier(
        dataset=solution.solution_id.dataset,
        subset=solution.solution_id.subset,
        problem_idx=solution.solution_id.problem_idx,
        solution_idx=solution.solution_id.solution_idx,
        test_idx=test_idx,
        augment_category=None,
        augment_idx=None,
    )
    return _CodeToTrace(
        code_string=solution.code,
        trace_id=trace_identifier,
        entrypoint_name=problem.entrypoint_name,
        test_inputs=inputs,
        test_outputs=outputs,
    )


def _write_json_line(handle, payload: dict[str, typing.Any]) -> None:
    """Write a JSON-serializable dictionary to a file handle as a single line."""
    handle.write(json.dumps(payload, ensure_ascii=False, default=_json_default))
    handle.write("\n")
    handle.flush()


def _resolve_dataset_root(dataset_root: pathlib.Path | None) -> pathlib.Path:
    """Resolve the dataset root path, using a default if necessary."""
    if dataset_root is not None:
        return dataset_root.resolve()
    latest = pyine.data.taco.dataset_utils.get_latest_repackaged_dataset_path()
    return latest.resolve()


def _ensure_output_dir(path: pathlib.Path | None) -> pathlib.Path:
    """Ensure that the output directory exists and return it, using a default one if necessary."""
    if path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        root = pyine.utils.filesystem.get_logs_root_path() / "traced-test-failures" / f"analysis-{timestamp}"
        root.mkdir(parents=True, exist_ok=True)
        return root
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_trace_failure_analysis(
    dataset_root: pathlib.Path | None,
    dataset_name: str,
    max_problems: int | None,
    max_solutions: int | None,
    max_tests: int | None,
    allow_banned: bool,
    timeout_seconds: float,
    validate_code_strings: bool,
    reformat_code_strings: bool,
    output_dir: pathlib.Path | None,
    show_progress: bool,
    max_runtime_seconds: float | None,
    max_failures_per_solution: int | None,
) -> pathlib.Path:
    """Run tracing attempts and record detailed failure metadata."""
    resolved_dataset_root = _resolve_dataset_root(dataset_root)
    logger.info("using dataset root: %s", resolved_dataset_root)
    out_dir = _ensure_output_dir(output_dir)
    logger.info("logging failures under: %s", out_dir)
    config = TraceDatasetWriterConfig(
        source_dataset_name=dataset_name,
        execution_timeout_seconds=timeout_seconds,
    )
    iterator = pyine.data.traces.dataset_utils.CodingProblemIterator(
        dataset_name=dataset_name,
        root_data_path=resolved_dataset_root,
        allow_banned_samples=allow_banned,
        reformat_code_strings=reformat_code_strings,
        validate_code_strings=validate_code_strings,
        show_progress=show_progress,
    )
    failure_log_path = out_dir / "failures.jsonl"
    summary_path = out_dir / "summary.json"
    run_config_path = out_dir / "run_config.json"
    started_at = datetime.datetime.now()
    run_deadline = None
    if max_runtime_seconds is not None:
        run_deadline = time.perf_counter() + max_runtime_seconds
    stats: dict[str, int] = {
        "problems_processed": 0,
        "problems_skipped": 0,
        "solutions_processed": 0,
        "solutions_skipped": 0,
        "test_cases_run": 0,
        "successful_traces": 0,
        "comparison_failures": 0,
        "execution_failures": 0,
    }
    limits_hit: list[str] = []

    with failure_log_path.open("w", encoding="utf-8") as failure_handle:
        for problem_idx, (problem, solutions) in enumerate(iterator):
            logger.info(f"launching analysis for: {problem.problem_id} ({len(solutions)} solutions in total)...")
            if run_deadline is not None and time.perf_counter() >= run_deadline:
                if "max_runtime_seconds" not in limits_hit:
                    limits_hit.append("max_runtime_seconds")
                logger.info("reached max runtime limit (%.2fs), stopping iteration", max_runtime_seconds)
                break
            if max_problems is not None and problem_idx >= max_problems:
                if "max_problems" not in limits_hit:
                    limits_hit.append("max_problems")
                logger.info("reached max problem limit (%s), stopping iteration", max_problems)
                break
            if problem.should_discard() and not allow_banned:
                stats["problems_skipped"] += 1
                logger.debug("skipping problem flagged to discard: %s", problem.problem_id)
                continue
            stats["problems_processed"] += 1
            logger.info("processing problem %s (%d solutions)", problem.problem_id, len(solutions))
            selected_solutions = solutions
            if max_solutions is not None:
                selected_solutions = selected_solutions[:max_solutions]
            for solution_idx, solution in enumerate(selected_solutions):
                logger.info(f"launching analysis for solution: {solution.solution_id}...")
                if run_deadline is not None and time.perf_counter() >= run_deadline:
                    if "max_runtime_seconds" not in limits_hit:
                        limits_hit.append("max_runtime_seconds")
                    logger.info("reached max runtime limit (%.2fs) within solution loop", max_runtime_seconds)
                    break
                if solution.should_discard() and not allow_banned:
                    stats["solutions_skipped"] += 1
                    logger.debug("skipping solution flagged to discard: %s", solution.solution_id)
                    continue
                if not solution.code.strip():
                    logger.debug("skipping empty code solution: %s", solution.solution_id)
                    stats["solutions_skipped"] += 1
                    continue
                stats["solutions_processed"] += 1
                tests_iter = enumerate(problem.test_inout_pairs)
                if max_tests is not None:
                    tests_iter = itertools.islice(enumerate(problem.test_inout_pairs), max_tests)
                failure_count = 0
                for test_idx, (inputs, expected_outputs) in tests_iter:
                    if run_deadline is not None and time.perf_counter() >= run_deadline:
                        if "max_runtime_seconds" not in limits_hit:
                            limits_hit.append("max_runtime_seconds")
                        logger.info("reached max runtime limit (%.2fs) within test loop", max_runtime_seconds)
                        break
                    stats["test_cases_run"] += 1
                    trace_request = _build_trace_request(
                        problem=problem,
                        solution=solution,
                        test_idx=test_idx,
                        inputs=inputs,
                        outputs=expected_outputs,
                    )
                    attempt_started = datetime.datetime.now()
                    attempt_started_perf = time.perf_counter()
                    try:
                        trace_result, compare_result = pyine.data.traces.dataset_writer._trace_code_snippet(
                            code_snippet=trace_request,
                            config=config,
                        )
                    except Exception as exc:  # noqa
                        stats["execution_failures"] += 1
                        logger.warning(
                            "execution failure for trace %s: %s",
                            trace_request.trace_id,
                            exc,
                        )
                        attempt_duration = time.perf_counter() - attempt_started_perf
                        failure_payload = {
                            "failure_type": "execution_error",
                            "trace_id": str(trace_request.trace_id),
                            "problem": {
                                "id": str(problem.problem_id),
                                "tags": problem.problem_tags,
                                "entrypoint_name": problem.entrypoint_name,
                                "source_data_path": problem.source_data_path,
                            },
                            "solution": {
                                "id": str(solution.solution_id),
                                "code": solution.code,
                                "analysis": {
                                    "input_type": solution.analysis_results.input_type,
                                    "output_type": solution.analysis_results.output_type,
                                    "uses_stdin": solution.analysis_results.input_type == "stdin",
                                    "uses_stdout": solution.analysis_results.output_type == "stdout",
                                },
                            },
                            "test_case": {
                                "index": test_idx,
                                "inputs": _to_json_compatible(inputs),
                                "inputs_repr": repr(inputs),
                                "expected_outputs": _to_json_compatible(expected_outputs),
                                "expected_outputs_repr": repr(expected_outputs),
                            },
                            "indices": {
                                "problem_index": problem_idx,
                                "solution_index": solution_idx,
                                "test_index": test_idx,
                            },
                            "config": {
                                "execution_timeout_seconds": config.execution_timeout_seconds,
                            },
                            "attempt": {
                                "started_at": attempt_started,
                                "duration_seconds": attempt_duration,
                            },
                            "error": _format_exception(exc),
                            "timestamp": datetime.datetime.now(),
                        }
                        _write_json_line(failure_handle, failure_payload)
                        failure_count += 1
                        if max_failures_per_solution is not None and failure_count >= max_failures_per_solution:
                            logger.info(
                                "reached failure cap (%s) for solution %s, skipping remaining tests",
                                max_failures_per_solution,
                                solution.solution_id,
                            )
                            break
                        continue
                    if compare_result:
                        stats["successful_traces"] += 1
                        continue
                    stats["comparison_failures"] += 1
                    logger.info(
                        "comparison failure for trace %s: %s",
                        trace_request.trace_id,
                        compare_result.reason,
                    )
                    trace_exception = trace_result.exception._asdict() if trace_result.exception else None
                    attempt_duration = time.perf_counter() - attempt_started_perf
                    failure_payload = {
                        "failure_type": "comparison_mismatch",
                        "trace_id": str(trace_request.trace_id),
                        "problem": {
                            "id": str(problem.problem_id),
                            "tags": problem.problem_tags,
                            "entrypoint_name": problem.entrypoint_name,
                            "source_data_path": problem.source_data_path,
                        },
                        "solution": {
                            "id": str(solution.solution_id),
                            "code": solution.code,
                            "analysis": {
                                "input_type": solution.analysis_results.input_type,
                                "output_type": solution.analysis_results.output_type,
                                "uses_stdin": solution.analysis_results.input_type == "stdin",
                                "uses_stdout": solution.analysis_results.output_type == "stdout",
                            },
                        },
                        "test_case": {
                            "index": test_idx,
                            "inputs": _to_json_compatible(inputs),
                            "inputs_repr": repr(inputs),
                            "expected_outputs": _to_json_compatible(expected_outputs),
                            "expected_outputs_repr": repr(expected_outputs),
                        },
                        "indices": {
                            "problem_index": problem_idx,
                            "solution_index": solution_idx,
                            "test_index": test_idx,
                        },
                        "observed_output": {
                            "inputs": _to_json_compatible(trace_result.inputs),
                            "expected_output": _to_json_compatible(trace_result.expected_output),
                            "return_value": _to_json_compatible(trace_result.return_value),
                            "return_value_repr": repr(trace_result.return_value),
                            "stdout": trace_result.stdout,
                            "stderr": trace_result.stderr,
                            "exception": trace_exception,
                        },
                        "comparison": {
                            "equal": compare_result.equal,
                            "reason": compare_result.reason,
                            "path": compare_result.path,
                        },
                        "attempt": {
                            "started_at": attempt_started,
                            "duration_seconds": attempt_duration,
                        },
                        "config": {
                            "execution_timeout_seconds": config.execution_timeout_seconds,
                        },
                        "trace_metadata": _summarize_trace_result(trace_result),
                        "timestamp": datetime.datetime.now(),
                    }
                    _write_json_line(failure_handle, failure_payload)
                    failure_count += 1
                    if max_failures_per_solution is not None and failure_count >= max_failures_per_solution:
                        logger.info(
                            "reached failure cap (%s) for solution %s, skipping remaining tests",
                            max_failures_per_solution,
                            solution.solution_id,
                        )
                        break

    finished_at = datetime.datetime.now()
    summary_payload = {
        "dataset_root": str(resolved_dataset_root),
        "dataset_name": dataset_name,
        "output_dir": str(out_dir),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "max_runtime_seconds": max_runtime_seconds,
        "max_failures_per_solution": max_failures_per_solution,
        "stats": stats,
        "limits_hit": limits_hit,
    }
    run_config_payload = {
        "dataset_root": str(resolved_dataset_root),
        "dataset_name": dataset_name,
        "max_problems": max_problems,
        "max_solutions": max_solutions,
        "max_tests": max_tests,
        "allow_banned": allow_banned,
        "timeout_seconds": timeout_seconds,
        "validate_code_strings": validate_code_strings,
        "reformat_code_strings": reformat_code_strings,
        "show_progress": show_progress,
        "max_runtime_seconds": max_runtime_seconds,
        "max_failures_per_solution": max_failures_per_solution,
    }
    summary_path.write_text(
        json.dumps(summary_payload, default=_json_default, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    run_config_path.write_text(
        json.dumps(run_config_payload, default=_json_default, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("run completed: %s", summary_payload)
    return out_dir


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--dataset-root",
    type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
    default=None,
    help="Root path to the source dataset. Defaults to the latest repackaged TACO dataset.",
)
@click.option(
    "--dataset-name",
    type=str,
    default="TACO",
    show_default=True,
    help="Name of the dataset to iterate.",
)
@click.option(
    "--max-problems",
    type=click.IntRange(min=1),
    default=None,
    help="Maximum number of problems to process.",
)
@click.option(
    "--max-solutions",
    type=click.IntRange(min=1),
    default=None,
    help="Maximum number of solutions per problem to process.",
)
@click.option(
    "--max-tests",
    type=click.IntRange(min=1),
    default=None,
    help="Maximum number of tests per solution to run.",
)
@click.option(
    "--allow-banned",
    is_flag=True,
    default=False,
    help="Include samples flagged as banned or problematic.",
)
@click.option(
    "--timeout",
    "timeout_seconds",
    type=float,
    default=10.0,
    show_default=True,
    help="Execution timeout per trace in seconds.",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    default="INFO",
    show_default=True,
    help="Logging verbosity level.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=pathlib.Path),
    default=None,
    help="Directory to store analysis artifacts.",
)
@click.option(
    "--no-validate-code",
    "validate_code_strings",
    flag_value=False,
    default=True,
    help="Disable solution code validation before tracing.",
)
@click.option(
    "--reformat-code",
    is_flag=True,
    default=False,
    help="Reformat code strings with the iterator before tracing.",
)
@click.option(
    "--show-progress/--no-show-progress",
    default=False,
    show_default=True,
    help="Toggle iterator progress bar output.",
)
@click.option(
    "--max-runtime-seconds",
    type=float,
    default=None,
    help="Optional wall-clock limit (seconds) for the run.",
)
@click.option(
    "--max-failures-per-solution",
    type=click.IntRange(min=1),
    default=None,
    help="Stop running additional tests for a solution after this many failures.",
)
def main(
    dataset_root: pathlib.Path | None,
    dataset_name: str,
    max_problems: int | None,
    max_solutions: int | None,
    max_tests: int | None,
    allow_banned: bool,
    timeout_seconds: float,
    log_level: str,
    output_dir: pathlib.Path | None,
    validate_code_strings: bool,
    reformat_code: bool,
    show_progress: bool,
    max_runtime_seconds: float | None,
    max_failures_per_solution: int | None,
) -> None:
    """Entrypoint for the trace failure analysis CLI."""
    resolved_level = getattr(logging, log_level.upper(), logging.INFO)
    pyine.utils.logging.setup_logging(level=resolved_level)
    logger.info("starting trace failure analysis")
    run_trace_failure_analysis(
        dataset_root=dataset_root,
        dataset_name=dataset_name,
        max_problems=max_problems,
        max_solutions=max_solutions,
        max_tests=max_tests,
        allow_banned=allow_banned,
        timeout_seconds=timeout_seconds,
        validate_code_strings=validate_code_strings,
        reformat_code_strings=reformat_code,
        output_dir=output_dir,
        show_progress=show_progress,
        max_runtime_seconds=max_runtime_seconds,
        max_failures_per_solution=max_failures_per_solution,
    )


if __name__ == "__main__":
    main()
