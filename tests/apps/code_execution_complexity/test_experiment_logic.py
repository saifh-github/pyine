import pathlib
import tempfile
import typing
from typing import NamedTuple

import pyine.apps.code_execution_complexity.experiment_logic as experiment_logic


class MockSolution(NamedTuple):
    solution_id: str
    code: str
    analysis_errors: list


class MockProblem(NamedTuple):
    problem_id: str
    test_inout_pairs: list


class MockIterator:
    def __init__(self, problems_data: typing.Sequence[MockProblem]) -> None:
        self.problems_data = problems_data

    def __len__(self) -> int:
        return len(self.problems_data)

    def __getitem__(self, index: int) -> tuple[MockProblem, list[MockSolution]]:
        problem_id, num_tests, has_valid_solution = self.problems_data[index]
        problem = MockProblem(
            problem_id=problem_id,
            test_inout_pairs=[("input", "output")] * num_tests,
        )
        if has_valid_solution:
            solutions = [MockSolution(solution_id=f"{problem_id}/s0", code="x=1", analysis_errors=[])]
        else:
            solutions = [MockSolution(solution_id=f"{problem_id}/s0", code="x=1", analysis_errors=["error"])]
        return problem, solutions


def test_indices_match_sampled_count_with_rejections() -> None:
    """
    Test that we correctly sample the requested number even with rejections.
    """
    problems_data = [
        ("p0", 5, True),  # Valid
        ("p1", 5, False),  # Invalid: has analysis error
        ("p2", 5, False),  # Invalid: has analysis error
        ("p3", 2, True),  # Invalid: has too few inputs
        ("p4", 5, False),  # Invalid: has analysis error
        ("p5", 5, True),  # Valid
        ("p6", 5, False),  # Invalid: has analysis error
        ("p7", 5, True),  # Valid
    ]

    iterator = MockIterator(problems_data)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = pathlib.Path(tmpdir)
        sampled, end_position = experiment_logic.sample_code_snippets(
            iterator, 3, 4, seed=42, experiment_name="test_exp", output_dir=output_dir, start_position=0
        )

    assert len(sampled) == 3
    assert end_position > 0, f"Expected positive end_position, got {end_position}"


def test_checkpoint_continuation() -> None:
    """
    Test that two subsequent runs do not overlap if we use checkpoint.
    """
    problems_data = [(f"p{i}", 5, True) for i in range(30)]
    iterator = MockIterator(problems_data)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = pathlib.Path(tmpdir)

        sampled1, end_position1 = experiment_logic.sample_code_snippets(
            iterator, 3, 3, seed=42, experiment_name="test_exp", output_dir=output_dir, start_position=0
        )

        sampled2, end_position2 = experiment_logic.sample_code_snippets(
            iterator, 3, 3, seed=42, experiment_name="test_exp", output_dir=output_dir, start_position=end_position1
        )

    assert len(sampled1) == 3
    assert len(sampled2) == 3

    # Verify non-overlapping by checking problem IDs
    ids1 = {problem.problem_id for problem, _ in sampled1}
    ids2 = {problem.problem_id for problem, _ in sampled2}
    overlap = ids1 & ids2
    assert len(overlap) == 0, f"Expected no overlap, found: {overlap}"

    assert end_position2 > end_position1, f"Expected end_position2 ({end_position2}) > end_position1 ({end_position1})"


def test_same_seed_produces_same_shuffle() -> None:
    """
    Test that the same seed produces the same shuffled order.
    """
    problems_data = [(f"p{i}", 5, True) for i in range(20)]
    iterator = MockIterator(problems_data)

    with tempfile.TemporaryDirectory() as tmpdir1:
        output_dir1 = pathlib.Path(tmpdir1)
        sampled1, _ = experiment_logic.sample_code_snippets(
            iterator, 5, 3, seed=42, experiment_name="test_exp1", output_dir=output_dir1, start_position=0
        )

    with tempfile.TemporaryDirectory() as tmpdir2:
        output_dir2 = pathlib.Path(tmpdir2)
        sampled2, _ = experiment_logic.sample_code_snippets(
            iterator, 5, 3, seed=42, experiment_name="test_exp2", output_dir=output_dir2, start_position=0
        )

    # Verify determinism by checking problem IDs match
    ids1 = [problem.problem_id for problem, _ in sampled1]
    ids2 = [problem.problem_id for problem, _ in sampled2]
    assert ids1 == ids2, f"Same seed should produce same problem order: {ids1} vs {ids2}"


def test_negative_start_position_raises_error() -> None:
    """
    Test that negative start_position raises ValueError.
    """
    problems_data = [(f"p{i}", 5, True) for i in range(10)]
    iterator = MockIterator(problems_data)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = pathlib.Path(tmpdir)
        try:
            experiment_logic.sample_code_snippets(
                iterator, 3, 3, seed=42, experiment_name="test_exp", output_dir=output_dir, start_position=-1
            )
            raise AssertionError("expected ValueError for negative start_position")
        except ValueError as e:
            assert "must be non-negative" in str(e)
