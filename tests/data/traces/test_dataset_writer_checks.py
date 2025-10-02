import pathlib
import tempfile
import typing

import pyine.data.traces.dataset_utils
import pyine.data.traces.dataset_writer
import pyine.prompts.configs.code_analysis


def _make_analysis_response(
    **overrides: typing.Any,
) -> pyine.prompts.configs.code_analysis.CodeAnalysisResponse:
    defaults = {
        "is_deterministic": True,
        "imports_nonstandard_packages": False,
        "invalid_syntax": False,
        "blocks_execution": False,
        "unnecessary_lines": False,
        "filesystem_access": False,
        "system_commands": False,
        "network_access": False,
        "code_type": pyine.prompts.configs.code_analysis.CodeTypeOptions.SIMPLE,
        "input_type": pyine.prompts.configs.code_analysis.InputTypeOptions.STDIN,
        "output_type": pyine.prompts.configs.code_analysis.OutputTypeOptions.STDOUT,
    }
    defaults.update(overrides)
    return pyine.prompts.configs.code_analysis.CodeAnalysisResponse(**defaults)


def _make_problem_and_solution(
    *,
    is_problem_banned: bool = False,
    parsing_errors: list[str] | None = None,
    analysis_errors: list[str] | None = None,
) -> tuple[
    pyine.data.traces.dataset_utils.CodingProblem,
    pyine.data.traces.dataset_utils.Solution,
]:
    problem_id = pyine.data.traces.dataset_utils.CodingProblemIdentifier("ds", "subset", 0)
    solution_id = pyine.data.traces.dataset_utils.SolutionIdentifier("ds", "subset", 0, 0)
    problem = pyine.data.traces.dataset_utils.CodingProblem(
        source_dataset_name="ds",
        source_data_path=str(pathlib.Path(tempfile.gettempdir()) / "problem.json"),
        source_data_hash="hash",
        problem_id=problem_id,
        problem_statement="do stuff",
        problem_tags=["easy"],
        test_inout_pairs=[((1,), 2)],
        entrypoint_name="solve",
        potential_solution_ids=[solution_id],
        parsing_errors=parsing_errors,
        is_banned=is_problem_banned,
    )
    solution = pyine.data.traces.dataset_utils.Solution(
        parent_id=problem_id,
        solution_id=solution_id,
        code="def solve(x):\n    return x + 1\n",
        analysis_errors=analysis_errors,
        analysis_results=_make_analysis_response(),
        is_banned=False,
    )
    return problem, solution


def test_check_must_skip_problem_detects_parsing_errors() -> None:
    problem, solution = _make_problem_and_solution(parsing_errors=["err"])
    contains_banned = lambda tags: False  # noqa
    config = pyine.data.traces.dataset_writer.TraceDatasetWriterConfig(source_dataset_name="ds")
    message = pyine.data.traces.dataset_writer._check_must_skip_problem(
        problem=problem,
        solutions=[solution],
        config=config,
        contains_banned_tags=contains_banned,
    )
    assert "parsing errors" in message


def test_check_must_skip_problem_detects_banned_tags() -> None:
    problem, solution = _make_problem_and_solution()
    contains_banned = lambda tags: True  # noqa
    config = pyine.data.traces.dataset_writer.TraceDatasetWriterConfig(source_dataset_name="ds")
    message = pyine.data.traces.dataset_writer._check_must_skip_problem(
        problem=problem,
        solutions=[solution],
        config=config,
        contains_banned_tags=contains_banned,
    )
    assert "banned problem tags" in message


def test_check_must_skip_solution_detects_analysis_errors() -> None:
    problem, solution = _make_problem_and_solution(analysis_errors=["bad"])
    config = pyine.data.traces.dataset_writer.TraceDatasetWriterConfig(source_dataset_name="ds")
    message = pyine.data.traces.dataset_writer._check_must_skip_solution(
        problem=problem,
        solution=solution,
        solution_idx=0,
        retained_solution_indices=[0],
        config=config,
    )
    assert "analysis errors" in message


def test_check_must_skip_solution_enforces_line_count() -> None:
    problem, solution = _make_problem_and_solution()
    short_solution = solution.model_copy(update={"code": "pass\n"})
    config = pyine.data.traces.dataset_writer.TraceDatasetWriterConfig(
        source_dataset_name="ds",
        min_solution_line_count=3,
    )
    message = pyine.data.traces.dataset_writer._check_must_skip_solution(
        problem=problem,
        solution=short_solution,
        solution_idx=0,
        retained_solution_indices=[0],
        config=config,
    )
    assert "too few lines" in message


def test_check_must_skip_solution_respects_retained_indices() -> None:
    problem, solution = _make_problem_and_solution()
    config = pyine.data.traces.dataset_writer.TraceDatasetWriterConfig(source_dataset_name="ds")
    message = pyine.data.traces.dataset_writer._check_must_skip_solution(
        problem=problem,
        solution=solution,
        solution_idx=1,
        retained_solution_indices=[0],
        config=config,
    )
    assert "duplicate" in message
