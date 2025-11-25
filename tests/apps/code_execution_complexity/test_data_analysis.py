import collections
import pathlib

import pytest

import pyine.apps.code_execution_complexity.utils.dataset
import pyine.utils.code.complexity_metrics
import tests.env_checks


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.dataset
@pytest.mark.skipif(
    tests.env_checks.TACO_DATASET_MISSING,
    reason="TACO dataset is missing, cannot check dataset complexity analysis pipeline",
)
def test_main_split_taco_micro_subset(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end exercise of main() using the TACO dataset and a max sample count of 10.

    Monkeypatches the dataset split file path getter to use a temporary directory.
    """
    assert tmp_path.is_dir()
    monkeypatch.setenv("PYINE_CACHE_ROOT", str(tmp_path))
    expected_data_cache_dir = tmp_path / "complexity-analysis"
    assert not expected_data_cache_dir.is_dir()
    results = pyine.apps.code_execution_complexity.utils.dataset.fetch_or_compute_taco_analysis_results(
        max_problems=10,
        max_solutions_per_problem=3,
    )
    unique_problem_count = len({res.problem_id for res in results})
    assert unique_problem_count == 10
    assert len(results) >= unique_problem_count
    solution_lists = collections.defaultdict(list)
    for res in results:
        solution_lists[res.problem_id].append(res.solution_id)
    assert all(len(solutions) <= 3 for solutions in solution_lists.values())
    assert all(len(res.code) > 0 for res in results)
    assert all(
        isinstance(res.complexity_metrics, pyine.utils.code.complexity_metrics.ComplexityMetrics) for res in results
    )
    assert expected_data_cache_dir.is_dir()
    cached_files = list(expected_data_cache_dir.glob("TACO.*"))
    assert len(cached_files) == 1
