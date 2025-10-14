"""High-level CLI helpers for rewriting failing trace I/O via LLMs.

The commands defined here read TACO coding problems, call an LLM-powered
rewrite pipeline to propose better input/output examples, and optionally write
results back to the dataset cache. The module wires together prompt fetching,
trace dataset writing, and override file persistence so the CLI can iterate on
problem statements without hand-editing JSON blobs.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import orjson

import pyine.data.utils.lmdb_io
import pyine.utils.filesystem
import pyine.utils.llm_providers
import pyine.utils.reprod
from pyine.data.traces.dataset_utils import CodingProblem, CodingProblemIterator, Solution, TraceIdentifier
from pyine.data.traces.dataset_writer import TraceDatasetWriterConfig, TraceRequest, trace_code_snippet
from pyine.prompts import PromptBuildConfig, TypedPromptResultFetcher
from pyine.prompts.configs.input_output_rewrite import InputOutputRewriteResponse
from pyine.prompts.result_db import ValidationFailedError

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

DEFAULT_PROBLEM_DIR = pyine.utils.filesystem.get_data_root_path() / "TACO" / "repackaged" / "2025-03-31-v01"
CACHE_OVERRIDE_BASENAME = "taco_problem_data_overrides.json"
DEFAULT_OVERRIDE_PATH = pyine.utils.filesystem.get_data_cache_path() / CACHE_OVERRIDE_BASENAME

PROBLEM_SOURCES_PATH = Path("data/TACO/overrides/problem_sources.json")
TARGET_SOURCES = {"leetcode", "geeksforgeeks"}

MAX_LLM_ATTEMPTS = 1
MAX_SOLUTIONS_TO_TRY = 3

try:
    pyine.utils.reprod.load_dotenv()
except FileNotFoundError:
    logger.debug("no .env file found while initializing trace analyzer; continuing with process env")


def _resolve_llm_provider_config() -> pyine.utils.llm_providers.LLMProviderConfig:
    provider = os.environ.get("INPUT_OUTPUT_REWRITE_PROVIDER", "openai")
    model_name = os.environ.get("INPUT_OUTPUT_REWRITE_MODEL", "gpt-5-nano")
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
    banned_problem_tags_rule=None,
    max_output_traces=None,
    max_solutions_per_problem=10,
    max_tests_per_solution=10,
    max_trace_events_per_line=None,
    max_trace_var_repr_length=20_000,
    max_trace_valid_events=20_000,
    max_trace_results_blob_size=1024**3,
    min_solution_line_count=3,
    min_solution_dissimilarity=0.1,
    execution_timeout_seconds=60,
    generate_obfuscated_solutions=False,
    prompt_result_db_path=None,
    writer_serialization_config={
        "method": pyine.data.utils.lmdb_io.SerializationMethod.JSON_ZSTD,
        "compression_kwargs": {"level": 3},
    },
)

_LLM_MODEL = None


def _to_pretty_json(data: dict[str, Any]) -> str:
    return orjson.dumps(data, option=orjson.OPT_INDENT_2).decode("utf-8")


def _get_first_solution_code(problem_json: dict[str, Any]) -> str:
    solutions = problem_json.get("solutions") or []
    if not solutions:
        return ""
    first_solution = solutions[0]
    if not isinstance(first_solution, dict):
        return ""
    return first_solution.get("code") or first_solution.get("orig_code") or ""


def _load_override_log(path: Path) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    try:
        data = orjson.loads(path.read_bytes())
    except orjson.JSONDecodeError:
        return {}
    if isinstance(data, dict):
        return {str(key): value for key, value in data.items() if isinstance(value, dict)}
    return {}


def _load_problem_sources(problem_dir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not PROBLEM_SOURCES_PATH.exists():
        logger.debug("problem_sources metadata not found at %s", PROBLEM_SOURCES_PATH)
        return mapping
    try:
        metadata = json.loads(PROBLEM_SOURCES_PATH.read_text())
    except json.JSONDecodeError as exc:
        logger.warning("failed to parse problem_sources metadata: %s", exc)
        return mapping
    dataset_root = metadata.get("dataset_root")
    if dataset_root:
        try:
            dataset_root = Path(dataset_root)
        except TypeError:
            dataset_root = None
    problems = metadata.get("problems", [])
    for entry in problems:
        problem_name = entry.get("problem")
        source = entry.get("source")
        if not problem_name or not source:
            continue
        if dataset_root and (problem_dir.resolve() != Path(dataset_root).resolve()):
            continue
        mapping[str(problem_name)] = str(source)
    return mapping


def _save_override_log(path: Path, entries: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(entries, option=orjson.OPT_INDENT_2))


def _upsert_override_entry(
    entries: dict[str, dict[str, Any]],
    problem_id: str,
    input_output: dict[str, Any],
) -> None:
    entries[problem_id] = input_output


def _get_llm_model() -> pyine.utils.llm_providers.LLMProvider:
    global _LLM_MODEL
    max_tokens = 10_000
    if _LLM_MODEL is None:
        _LLM_MODEL = pyine.utils.llm_providers.get_model_from_provider(
            provider=LLM_PROVIDER_CONFIG.provider,
            model=LLM_PROVIDER_CONFIG.model_kwargs["model"],
            temperature=LLM_PROVIDER_CONFIG.model_kwargs.get("temperature", 0.0),
            max_tokens=max_tokens,
        )
    return _LLM_MODEL


def _generate_candidate_input_output(
    problem_identifier: str,
    question: str,
    starter_code: str,
    first_solution: str,
    current_input_output: dict[str, Any],
    attempt_idx: int,
) -> InputOutputRewriteResponse | None:
    sanitized_input_output = current_input_output or {}
    prompt_inputs = {
        "question": question,
        "starter_code": starter_code,
        "first_solution": first_solution,
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
    test_pairs = list(zip(response.inputs, response.outputs, strict=False))
    return problem.model_copy(
        update={
            "entrypoint_name": response.fn_name,
            "test_inout_pairs": test_pairs,
        }
    )


def _is_valid_response(response: InputOutputRewriteResponse) -> bool:
    if not response.fn_name:
        return False
    return len(response.inputs) == len(response.outputs)


def _collect_problem_paths(
    problem_dir: Path,
    filenames: Iterable[Path | str],
    override_log_path: Path | None,
) -> list[Path]:
    paths: list[Path] = []
    filenames = tuple(filenames)
    if filenames:
        for name in filenames:
            candidate = Path(name)
            if not candidate.is_absolute():
                candidate = problem_dir / candidate
            paths.append(candidate)
        return paths

    override_resolved = override_log_path.resolve() if override_log_path else None
    for candidate in sorted(problem_dir.rglob("*.json")):
        if override_resolved and candidate.resolve() == override_resolved:
            continue
        if candidate.is_file():
            paths.append(candidate)
    return paths


def output_compare(problem: CodingProblem, solutions: list[Solution]) -> bool:
    entrypoint_name = problem.entrypoint_name
    if not entrypoint_name:
        return False
    if not problem.test_inout_pairs:
        return False

    total_tests = len(problem.test_inout_pairs)

    for _solution_idx, solution in enumerate(solutions[:MAX_SOLUTIONS_TO_TRY]):
        correct = 0
        for test_idx, (inputs, outputs) in enumerate(problem.test_inout_pairs):
            trace_id = TraceIdentifier(
                **vars(solution.solution_id),
                test_idx=test_idx,
                augment_category="bug",
                augment_idx=0,
            )
            code_to_trace = TraceRequest(
                code_string=solution.code,
                entrypoint_name=entrypoint_name,
                trace_id=trace_id,
                test_inputs=inputs,
                test_outputs=outputs,
            )
            try:
                _, compare_result = get_code_output(code_to_trace)
            except Exception:  # noqa: BLE001
                break
            if compare_result:
                correct += 1
            else:
                break
        if correct == total_tests:
            return True
    return False


def get_code_output(
    code: TraceRequest,
) -> tuple[
    pyine.utils.code.execution.TraceResult,
    pyine.utils.code.output_compare.CompareResult,
]:
    return trace_code_snippet(code_snippet=code, config=TRACE_WRITER_CONFIG)


def run_input_output_rewrite(
    problem_dir: Path,
    problem_filenames: Iterable[Path | str],
    override_log_path: Path | None,
) -> None:
    problem_dir = problem_dir.expanduser()
    if not problem_dir.is_absolute():
        problem_dir = (Path.cwd() / problem_dir).resolve()
    if not problem_dir.exists():
        raise FileNotFoundError(f"Problem directory does not exist: {problem_dir}")

    if override_log_path is None:
        override_log_path = DEFAULT_OVERRIDE_PATH
    else:
        override_log_path = override_log_path.expanduser()
        if not override_log_path.is_absolute():
            override_log_path = (Path.cwd() / override_log_path).resolve()

    override_entries = _load_override_log(override_log_path)
    logger.info("using override log: %s", override_log_path)

    overrides_path_for_iterator = override_log_path if override_log_path.exists() else None

    problem_iterator = CodingProblemIterator(
        dataset_name="TACO",
        root_data_path=problem_dir,
        problem_data_overrides_path=overrides_path_for_iterator,
    )
    if override_entries:
        problem_iterator._problem_data_overrides.update(override_entries)

    source_mapping = _load_problem_sources(problem_dir)
    problem_paths = _collect_problem_paths(problem_dir, problem_filenames, override_log_path)

    if not problem_paths:
        logger.info("no problem files matched the given parameters")
        return

    for problem_path in problem_paths:
        problem_filename = problem_path.name
        if not problem_path.exists():
            logger.info("skipping missing problem file: %s", problem_filename)
            continue
        try:
            raw_problem_data = problem_iterator._load_problem_data(problem_path)
        except orjson.JSONDecodeError as exc:
            logger.warning("Skipping %s: invalid JSON (%s)", problem_filename, exc)
            continue
        except TypeError as exc:
            logger.warning("Skipping %s: unexpected JSON structure (%s)", problem_filename, exc)
            continue

        source_name = source_mapping.get(problem_filename)
        if source_name is None and source_mapping:
            logger.debug("skipping %s: no source metadata", problem_filename)
            continue
        if source_name not in TARGET_SOURCES:
            logger.debug("skipping %s: source '%s' not in target set", problem_filename, source_name)
            continue

        if not isinstance(raw_problem_data, dict) or not raw_problem_data:
            logger.debug("skipping %s: empty problem payload", problem_filename)
            continue

        missing_keys = [key for key in ("subset", "input_output") if key not in raw_problem_data]
        if missing_keys:
            logger.debug("skipping %s: missing keys %s", problem_filename, ", ".join(missing_keys))
            continue

        input_output_block = raw_problem_data.get("input_output")
        if not isinstance(input_output_block, dict):
            logger.debug("skipping %s: malformed input_output block", problem_filename)
            continue

        io_missing = [key for key in ("inputs", "outputs") if key not in input_output_block]
        if io_missing:
            logger.debug("skipping %s: input_output missing %s", problem_filename, ", ".join(io_missing))
            continue

        coding_problem, solutions = problem_iterator._process_data(raw_problem_data)

        question = raw_problem_data.get("question") or ""
        if not isinstance(question, str):
            question = ""
        starter_code = raw_problem_data.get("starter_code") or ""
        if not isinstance(starter_code, str):
            starter_code = ""
        first_solution = _get_first_solution_code(raw_problem_data)

        if output_compare(coding_problem, solutions):
            logger.info("already valid: %s", problem_filename)
            continue

        current_io = raw_problem_data.get("input_output", {}) or {}
        problem_identifier = repr(coding_problem.problem_id)
        success = False

        for attempt_idx in range(MAX_LLM_ATTEMPTS):
            response = _generate_candidate_input_output(
                problem_identifier=problem_identifier,
                question=question,
                starter_code=starter_code,
                first_solution=first_solution,
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
                _upsert_override_entry(override_entries, problem_identifier, new_io)
                problem_iterator._problem_data_overrides[problem_identifier] = new_io
                _save_override_log(override_log_path, override_entries)
                success = True
                logger.info("updated %s on attempt %s", problem_filename, attempt_idx + 1)
                break

            current_io = {
                "inputs": response.inputs,
                "outputs": response.outputs,
                "fn_name": response.fn_name,
            }

        if not success:
            logger.warning("failed to fix %s after %s attempts", problem_filename, MAX_LLM_ATTEMPTS)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--problem-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_PROBLEM_DIR,
    show_default=True,
    help="Directory containing problem JSON files.",
)
@click.option(
    "--problem",
    "problem_filenames",
    multiple=True,
    type=click.Path(path_type=Path),
    help="Specific problem file(s) to process. May be repeated.",
)
@click.option(
    "--override-log",
    type=click.Path(path_type=Path),
    help=(f"Optional override log path. Defaults to the framework cache at {DEFAULT_OVERRIDE_PATH}."),
)
def main(
    problem_dir: Path,
    problem_filenames: tuple[Path, ...],
    override_log: Path | None,
) -> None:
    """Rewrite malformed input/output blocks using LLM assistance."""

    logging.basicConfig(level=logging.INFO)
    run_input_output_rewrite(problem_dir, problem_filenames, override_log)


if __name__ == "__main__":
    main()
