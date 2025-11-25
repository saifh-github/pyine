import dataclasses
import json
import logging
import pathlib
import typing

import pyine.data.traces.dataset_utils
import pyine.utils.code.complexity_metrics
import pyine.utils.filesystem
import pyine.utils.reprod

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class AnalyzedCodeSnippet:
    """Container for code snippets whose complexity has been analyzed."""

    problem_id: pyine.data.traces.dataset_utils.CodingProblemIdentifier
    """ID of the problem that the code snippet belongs to."""
    solution_id: pyine.data.traces.dataset_utils.SolutionIdentifier
    """ID of the solution that the code snippet belongs to."""
    code: str
    """Code snippet that was analyzed."""
    complexity_metrics: pyine.utils.code.complexity_metrics.ComplexityMetrics
    """Complexity metrics for the code snippet."""

    def as_dict(self) -> dict[str, typing.Any]:
        """Returns a JSON-serializable, plain dictionary representation of the analysis results."""
        return {
            "problem_id": str(self.problem_id),
            "solution_id": str(self.solution_id),
            "code": self.code,
            "complexity_metrics": self.complexity_metrics.as_dict(),
        }

    @staticmethod
    def from_dict(blob: dict[str, typing.Any]) -> "AnalyzedCodeSnippet":
        """Creates an AnalyzedCodeSnippet instance from a JSON-serializable dictionary."""
        return AnalyzedCodeSnippet(
            problem_id=pyine.data.traces.dataset_utils.CodingProblemIdentifier.from_string(blob["problem_id"]),
            solution_id=pyine.data.traces.dataset_utils.SolutionIdentifier.from_string(blob["solution_id"]),
            code=blob["code"],
            complexity_metrics=pyine.utils.code.complexity_metrics.ComplexityMetrics(**blob["complexity_metrics"]),
        )


def analyze_dataset_complexity(
    dataset_name: str,
    root_data_path: pathlib.Path,
    show_progress: bool = True,
    max_solutions_per_problem: int | None = None,
    max_problems: int | None = None,
) -> dict[pyine.data.traces.dataset_utils.SolutionIdentifier, AnalyzedCodeSnippet]:
    """Analyzes the complexity of each code snippet in a specific dataset, and returns the results.

    Args:
        dataset_name: Name of the dataset to analyze (forwarded to CodingProblemIterator).
        root_data_path: Root directory containing the dataset (forwarded to CodingProblemIterator).
        show_progress: Whether to show a progress bar while iterating over the dataset.
        max_solutions_per_problem: Maximum number of solutions to analyze per problem (optional).
        max_problems: Maximum number of problems to analyze (optional).

    Returns:
        A dictionary mapping solution IDs to their corresponding complexity metrics.
    """
    code_problem_iterator = pyine.data.traces.dataset_utils.CodingProblemIterator(
        dataset_name=dataset_name,
        root_data_path=root_data_path,
        show_progress=show_progress,
    )
    analysis_results: dict[pyine.data.traces.dataset_utils.SolutionIdentifier, AnalyzedCodeSnippet] = {}
    analyzed_problem_count = 0
    for problem, solutions in code_problem_iterator:
        withheld_solutions = 0
        for solution in solutions:
            if solution.analysis_errors:
                continue
            if max_solutions_per_problem is not None and withheld_solutions >= max_solutions_per_problem:
                break
            assert solution.solution_id not in analysis_results, f"solution {solution.solution_id} already analyzed"
            complexity = pyine.utils.code.complexity_metrics.get_complexity_metrics(solution.code)
            analysis_results[solution.solution_id] = AnalyzedCodeSnippet(
                problem_id=problem.problem_id,
                solution_id=solution.solution_id,
                code=solution.code,
                complexity_metrics=complexity,
            )
            withheld_solutions += 1
        if withheld_solutions > 0:
            analyzed_problem_count += 1
        if max_problems is not None and analyzed_problem_count >= max_problems:
            break
    return analysis_results


def fetch_or_compute_dataset_analysis_results(
    dataset_name: str,
    root_data_path: pathlib.Path,
    show_progress: bool = True,
    max_solutions_per_problem: int | None = None,
    max_problems: int | None = None,
) -> list[AnalyzedCodeSnippet]:
    """Fetches or computes the complexity analysis results for a specific dataset.

    Args:
        dataset_name: Name of the dataset to analyze (forwarded to CodingProblemIterator).
        root_data_path: Root directory containing the dataset (forwarded to CodingProblemIterator).
        show_progress: Whether to show a progress bar while iterating over the dataset.
        max_solutions_per_problem: Maximum number of solutions to analyze per problem (optional).
        max_problems: Maximum number of problems to analyze (optional).

    Returns:
        A list of analysis results, each provided as an AnalyzedCodeSnippet dataclass instance.
    """
    results_cache_dir = pyine.utils.filesystem.get_data_cache_path() / "complexity-analysis"
    results_cache_dir.mkdir(exist_ok=True)
    dataset_hash = pyine.utils.reprod.compute_hash(root_data_path)
    analysis_results_hash = pyine.utils.reprod.get_params_hash(
        dataset_name,
        dataset_hash,
        max_solutions_per_problem,
        max_problems,
    )
    expected_file_path = results_cache_dir / f"{dataset_name}.{analysis_results_hash}.json"
    if expected_file_path.exists():
        with expected_file_path.open("r") as fd:
            raw_data_blobs: list[dict[str, typing.Any]] = json.load(fd)
        return [AnalyzedCodeSnippet.from_dict(blob) for blob in raw_data_blobs]
    analysis_results = analyze_dataset_complexity(
        dataset_name=dataset_name,
        root_data_path=root_data_path,
        show_progress=show_progress,
        max_solutions_per_problem=max_solutions_per_problem,
        max_problems=max_problems,
    )
    raw_data_blobs = [results.as_dict() for results in analysis_results.values()]
    with expected_file_path.open("w") as fd:
        json.dump(raw_data_blobs, fd, indent=2)
    return list(analysis_results.values())


def fetch_or_compute_taco_analysis_results(
    show_progress: bool = True,
    max_solutions_per_problem: int | None = None,
    max_problems: int | None = None,
) -> list[AnalyzedCodeSnippet]:
    """Fetches or computes the complexity analysis results for the TACO dataset.

    Args:
        show_progress: Whether to show a progress bar while iterating over the dataset.
        max_solutions_per_problem: Maximum number of solutions to analyze per problem (optional).
        max_problems: Maximum number of problems to analyze (optional).

    Returns:
        A list of analysis results, each provided as an AnalyzedCodeSnippet dataclass instance.
    """
    import pyine.data.taco.dataset_utils

    logger.info("Preparing code complexity analysis results for TACO...")
    result = fetch_or_compute_dataset_analysis_results(
        dataset_name="TACO",
        root_data_path=pyine.data.taco.dataset_utils.get_latest_repackaged_dataset_path(),
        show_progress=show_progress,
        max_solutions_per_problem=max_solutions_per_problem,
        max_problems=max_problems,
    )
    logger.info(f"Processed {len(result):,} unique code snippets")
    return result


if __name__ == "__main__":
    import pyine.utils.logging
    import pyine.utils.reprod

    pyine.utils.reprod.load_dotenv()
    pyine.utils.logging.setup_logging()
    fetch_or_compute_taco_analysis_results()
