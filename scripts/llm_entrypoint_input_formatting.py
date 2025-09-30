"""Automate entrypoint detection and IO formatting for troublesome problems.

Pipeline outline (original TODOs preserved for reference):

1. Load the problem metadata.
2. Attempt to execute bundled solutions against the existing `input_output` block.
3. If execution succeeds, skip the sample.
4. Otherwise, invoke the LLM prompt with the problem JSON + current `input_output` block.
5. Validate the returned block by executing the solutions again.
6. Repeat up to N attempts, feeding the previously generated block back to the LLM.
7. On success, persist the new block and record it in `enriched_inputs_outputs.json`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import orjson
from dotenv import load_dotenv

import pyine.data.utils.lmdb_io
import pyine.prompts
import pyine.utils.llm_providers
from pyine.data.traces.dataset_utils import CodingProblem, CodingProblemIterator, Solution, TraceIdentifier
from pyine.data.traces.dataset_writer import TraceDatasetWriterConfig, _CodeToTrace, _trace_code_snippet
from pyine.prompts import PromptBuildConfig, TypedPromptResultFetcher
from pyine.prompts.configs.input_output_rewrite import InputOutputRewriteResponse
from pyine.prompts.result_db import ValidationFailedError

MAX_LLM_ATTEMPTS = 1
MAX_SOLUTIONS_TO_TRY = 3

PROBLEM_FILENAMES = [
    "enriched_001547.json",
    "enriched_001548.json",
    "enriched_001580.json",
]

PROBLEM_OVERRIDES_DIR = Path("/Users/s.lei/code-interp-benchmark/data/TACO/overrides")
REPACKAGED_ROOT = Path("/Users/s.lei/code-interp-benchmark/data/TACO/repackaged/2025-03-31-v01")
DEFAULT_OVERRIDE_LOG_NAME = "enriched_inputs_outputs.json"

load_dotenv("/Users/s.lei/code-interp-benchmark/.env")


def _resolve_llm_provider_config() -> pyine.utils.llm_providers.LLMProviderConfig:
    provider = os.environ.get("INPUT_OUTPUT_REWRITE_PROVIDER", "openai")
    model_name = os.environ.get("INPUT_OUTPUT_REWRITE_MODEL", "gpt-5")
    try:
        temperature = float(os.environ.get("INPUT_OUTPUT_REWRITE_TEMPERATURE", "0"))
    except ValueError:
        temperature = 0.0
    model_kwargs = {"model": model_name, "temperature": temperature}
    return pyine.utils.llm_providers.LLMProviderConfig(provider=provider, model_kwargs=model_kwargs)


LLM_PROVIDER_CONFIG = _resolve_llm_provider_config()
PROMPT_CONFIG = PromptBuildConfig(prompt_name="input_output_rewrite", version="v1.0")
PROMPT_FETCHER = TypedPromptResultFetcher(result_type=InputOutputRewriteResponse)

TRACE_WRITER_CONFIG = TraceDatasetWriterConfig(
    source_dataset_name="TACO",
    # create a dummy dataset for quick prototyping
    banned_problem_tags_rule=None,
    max_output_traces=None,
    max_solutions_per_problem=10,
    max_tests_per_solution=10,
    max_trace_events_per_line=None,
    max_trace_var_repr_length=20_000,  # chars (modified from PL's 10_000)
    max_trace_valid_events=20_000,
    max_trace_results_blob_size=1024**3,  # 1GB
    min_solution_line_count=3,
    min_solution_dissimilarity=0.1,
    execution_timeout_seconds=60,
    generate_obfuscated_solutions=False,
    prompt_result_db_path=None,  # use framework default
    writer_serialization_config=dict(
        method=pyine.data.utils.lmdb_io.SerializationMethod.JSON_ZSTD,
        compression_kwargs=dict(level=3),
    ),
)

problem_iterator = CodingProblemIterator(
    dataset_name="TACO",
    root_data_path=REPACKAGED_ROOT,
)


def _sanitize_problem_json(problem_json: dict[str, Any]) -> dict[str, Any]:
    """Drop internal bookkeeping keys before serializing or persisting."""

    sanitized = {k: v for k, v in problem_json.items() if not k.startswith("__root_")}
    return sanitized


def _to_pretty_json(data: dict[str, Any]) -> str:
    """Encode a mapping as a pretty-printed JSON string."""

    return orjson.dumps(data, option=orjson.OPT_INDENT_2).decode("utf-8")


def _load_override_log(path: Path) -> list[dict[str, Any]]:
    """Load previously stored overrides, if any."""

    if path is None or not path.exists():
        return []
    try:
        return orjson.loads(path.read_bytes())
    except orjson.JSONDecodeError:
        return []


def _save_override_log(path: Path, entries: list[dict[str, Any]]) -> None:
    """Persist accumulated overrides for later inspection."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(entries, option=orjson.OPT_INDENT_2))


def _upsert_override_entry(
    entries: list[dict[str, Any]],
    problem_filename: str,
    input_output: dict[str, Any],
) -> None:
    """Update or append an override entry for a problem."""

    for entry in entries:
        if entry.get("problem") == problem_filename:
            entry["input_output"] = input_output
            return
    entries.append({"problem": problem_filename, "input_output": input_output})


def _write_override_problem(path: Path, problem_json: dict[str, Any]) -> None:
    """Persist updated problem JSON to disk."""

    sanitized = _sanitize_problem_json(problem_json)
    path.write_bytes(orjson.dumps(sanitized, option=orjson.OPT_INDENT_2))


_LLM_MODEL = None


def _get_llm_model():
    """Instantiate the LLM model lazily."""

    global _LLM_MODEL
    max_tokens = 10_000
    if _LLM_MODEL is None:
        _LLM_MODEL = pyine.utils.llm_providers.get_model_from_provider(
            provider="openai",
            model="gpt-5",
            temperature=1.0,  # only value supported by gpt-5
            max_tokens=max_tokens,
        )
    return _LLM_MODEL


def _generate_candidate_input_output(
    problem_identifier: str,
    problem_json: dict[str, Any],
    current_input_output: dict[str, Any],
    attempt_idx: int,
) -> InputOutputRewriteResponse | None:
    """Call the LLM prompt to produce a candidate input/output block."""

    sanitized_problem = _sanitize_problem_json(problem_json)
    sanitized_input_output = current_input_output or {}
    prompt_inputs = {
        "problem_json": _to_pretty_json(sanitized_problem),
        "input_output": _to_pretty_json(sanitized_input_output),
    }
    model = _get_llm_model()
    identifier = f"{problem_identifier}::attempt-{attempt_idx}"
    try:
        records = PROMPT_FETCHER.fetch_or_generate(
            model=model,
            identifier=identifier,
            input_variables=prompt_inputs,
            prompt_config=PROMPT_CONFIG,
            force_generation=True,
            log_new_results=False,
        )
    except ValidationFailedError:
        return None
    if not records:
        return None
    return records[-1].result


def _make_candidate_problem(
    problem: CodingProblem,
    response: InputOutputRewriteResponse,
) -> CodingProblem:
    """Return a new CodingProblem with updated entrypoint + IO pairs."""

    test_pairs = list(zip(response.inputs, response.outputs))
    return problem.model_copy(
        update={
            "entrypoint_name": response.fn_name,
            "test_inout_pairs": test_pairs,
        }
    )


def _is_valid_response(response: InputOutputRewriteResponse) -> bool:
    """Basic sanity checks for LLM output before attempting execution."""

    if not response.fn_name:
        return False
    if len(response.inputs) != len(response.outputs):
        return False
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite problem input/output blocks using LLM-generated suggestions.",
    )
    parser.add_argument(
        "--problem-dir",
        type=Path,
        help="Directory containing problem JSON overrides. When provided, all *.json files will be processed recursively.",
    )
    parser.add_argument(
        "--problem",
        dest="problem_filenames",
        action="append",
        default=[],
        help="Specific problem filename to process. May be repeated.",
    )
    parser.add_argument(
        "--override-log",
        type=Path,
        help=(
            "Path to the override log file. Defaults to <problem-dir>/enriched_inputs_outputs.json when "
            "--problem-dir is supplied, otherwise uses the default overrides directory."
        ),
    )
    return parser.parse_args()


def _collect_problem_paths(problem_dir: Path, filenames: list[str], include_all: bool) -> list[Path]:
    paths: list[Path] = []

    if filenames:
        for name in filenames:
            candidate = Path(name)
            if not candidate.is_absolute():
                candidate = problem_dir / candidate
            paths.append(candidate)
        return paths

    if include_all:
        for candidate in sorted(problem_dir.rglob("*.json")):
            if candidate.name == DEFAULT_OVERRIDE_LOG_NAME:
                continue
            if candidate.is_file():
                paths.append(candidate)
        return paths

    for name in PROBLEM_FILENAMES:
        paths.append(problem_dir / name)
    return paths


def main():
    args = _parse_args()

    problem_dir = args.problem_dir.expanduser() if args.problem_dir else PROBLEM_OVERRIDES_DIR
    if not problem_dir.is_absolute():
        problem_dir = (Path.cwd() / problem_dir).resolve()
    if not problem_dir.exists():
        print(f"Problem directory does not exist: {problem_dir}")
        return

    override_log_path = args.override_log.expanduser() if args.override_log else problem_dir / DEFAULT_OVERRIDE_LOG_NAME
    override_entries = _load_override_log(override_log_path)

    include_all = args.problem_dir is not None and not args.problem_filenames
    problem_paths = _collect_problem_paths(problem_dir, args.problem_filenames, include_all)

    if not problem_paths:
        print("No problem files matched the given parameters.")
        return

    for problem_path in problem_paths:
        problem_filename = problem_path.name
        if not problem_path.exists():
            print(f"Skipping missing problem file: {problem_filename}")
            continue

        raw_problem_data = problem_iterator._load_problem_data(problem_path)
        coding_problem, solutions = problem_iterator._process_data(raw_problem_data)

        if output_compare(coding_problem, solutions):
            print(f"Already valid: {problem_filename}")
            continue

        current_io = raw_problem_data.get("input_output", {}) or {}
        success = False

        for attempt_idx in range(MAX_LLM_ATTEMPTS):
            response = _generate_candidate_input_output(
                problem_identifier=str(coding_problem.problem_id),
                problem_json=raw_problem_data,
                current_input_output=current_io,
                attempt_idx=attempt_idx,
            )
            if response is None or not _is_valid_response(response):
                continue

            candidate_problem = _make_candidate_problem(coding_problem, response)
            if output_compare(candidate_problem, solutions):
                new_io = {
                    "inputs": response.inputs,
                    "outputs": response.outputs,
                    "fn_name": response.fn_name,
                }
                raw_problem_data["input_output"] = new_io
                _write_override_problem(problem_path, raw_problem_data)
                _upsert_override_entry(override_entries, problem_filename, new_io)
                success = True
                print(f"Updated {problem_filename} on attempt {attempt_idx + 1}")
                break

            current_io = {
                "inputs": response.inputs,
                "outputs": response.outputs,
                "fn_name": response.fn_name,
            }

        if not success:
            print(f"Failed to fix {problem_filename} after {MAX_LLM_ATTEMPTS} attempts")

    if override_entries:
        _save_override_log(override_log_path, override_entries)


def output_compare(problem: CodingProblem, solutions: list[Solution]) -> bool:
    entrypoint_name = problem.entrypoint_name
    if not entrypoint_name:
        return False
    if not problem.test_inout_pairs:
        return False

    total_tests = len(problem.test_inout_pairs)

    for solution_idx, solution in enumerate(solutions[:MAX_SOLUTIONS_TO_TRY]):
        correct = 0
        for test_idx, (inputs, outputs) in enumerate(problem.test_inout_pairs):
            trace_id = TraceIdentifier(
                **vars(solution.solution_id),
                test_idx=test_idx,
                augment_category="bug",
                augment_idx=0,
            )
            code_to_trace = _CodeToTrace(
                code_string=solution.code,
                entrypoint_name=entrypoint_name,
                trace_id=trace_id,
                test_inputs=inputs,
                test_outputs=outputs,
            )
            try:
                _, compare_result = get_code_output(code_to_trace)
            except Exception:
                break
            if compare_result:
                correct += 1
            else:
                break
        if correct == total_tests:
            return True
    return False


def get_code_output(code: _CodeToTrace):
    return _trace_code_snippet(code, TRACE_WRITER_CONFIG)


if __name__ == "__main__":
    main()
