"""Tests for data classes in pyine.data.traces.dataset_utils.

Tests CodingProblem, Solution, and TraceMetadata data structures.
"""

import pydantic
import pytest

import pyine.data.traces.dataset_utils as dataset_utils
import pyine.prompts.configs.code_analysis as code_analysis


def make_problem_id(
    dataset: str = "DS",
    subset: str = "sub",
    idx: int = 0,
) -> dataset_utils.CodingProblemIdentifier:
    """Create a coding problem identifier for testing."""
    return dataset_utils.CodingProblemIdentifier(dataset, subset, idx)


def make_solution_id(
    dataset: str = "DS",
    subset: str = "sub",
    problem_idx: int = 0,
    solution_idx: int = 0,
) -> dataset_utils.SolutionIdentifier:
    """Create a solution identifier for testing."""
    return dataset_utils.SolutionIdentifier(dataset, subset, problem_idx, solution_idx)


def make_analysis_response(
    **overrides: object,
) -> code_analysis.CodeAnalysisResponse:
    """Create a code analysis response with defaults."""
    defaults = {
        "is_deterministic": True,
        "imports_nonstandard_packages": False,
        "invalid_syntax": False,
        "blocks_execution": False,
        "unnecessary_lines": False,
        "filesystem_access": False,
        "system_commands": False,
        "network_access": False,
        "code_type": code_analysis.CodeTypeOptions.SIMPLE,
        "input_type": code_analysis.InputTypeOptions.STDIN,
        "output_type": code_analysis.OutputTypeOptions.STDOUT,
    }
    defaults.update(overrides)
    return code_analysis.CodeAnalysisResponse(**defaults)  # type: ignore


def make_coding_problem(
    *,
    problem_id: dataset_utils.CodingProblemIdentifier | None = None,
    solution_count: int = 1,
    test_count: int = 2,
    parsing_errors: list[str] | None = None,
    is_banned: bool = False,
    **overrides: object,
) -> dataset_utils.CodingProblem:
    """Create a CodingProblem for testing."""
    pid = problem_id or make_problem_id()
    solution_ids = [make_solution_id(pid.dataset, pid.subset, pid.problem_idx, i) for i in range(solution_count)]
    test_pairs = [(i, i * 2) for i in range(test_count)]
    defaults = {
        "source_dataset_name": pid.dataset,
        "source_data_path": f"/data/{pid.dataset}/{pid.subset}",
        "source_data_hash": "abc123",
        "problem_id": pid,
        "problem_statement": "Solve this problem.",
        "problem_tags": ["easy", "math"],
        "test_inout_pairs": test_pairs,
        "entrypoint_name": "solution",
        "potential_solution_ids": solution_ids,
        "parsing_errors": parsing_errors,
        "is_banned": is_banned,
    }
    defaults.update(overrides)
    return dataset_utils.CodingProblem(**defaults)  # type: ignore


def make_solution(
    *,
    solution_id: dataset_utils.SolutionIdentifier | None = None,
    code: str = "def solution(x):\n    return x\n",
    analysis_errors: list[str] | None = None,
    is_banned: bool = False,
    analysis_overrides: dict | None = None,
) -> dataset_utils.Solution:
    """Create a Solution for testing."""
    sid = solution_id or make_solution_id()
    analysis = make_analysis_response(**(analysis_overrides or {}))
    return dataset_utils.Solution(
        parent_id=sid.get_parent_identifier(),
        solution_id=sid,
        code=code,
        analysis_errors=analysis_errors,
        analysis_results=analysis,
        is_banned=is_banned,
    )


# -----------------------------------------------------------------------------
#                          CodingProblem tests
# -----------------------------------------------------------------------------


class TestCodingProblem:
    """Tests for the CodingProblem pydantic model."""

    def test_creation(self) -> None:
        """Test basic CodingProblem creation."""
        problem = make_coding_problem()
        assert problem.source_dataset_name == "DS"
        assert problem.problem_id.dataset == "DS"

    def test_str_returns_problem_id(self) -> None:
        """Test that __str__ returns the problem identifier."""
        pid = make_problem_id("TACO", "train", 42)
        problem = make_coding_problem(problem_id=pid)
        assert str(problem) == "TACO/train/p000042"

    def test_solution_count_property(self) -> None:
        """Test solution_count property."""
        problem = make_coding_problem(solution_count=5)
        assert problem.solution_count == 5

    def test_test_count_property(self) -> None:
        """Test test_count property."""
        problem = make_coding_problem(test_count=10)
        assert problem.test_count == 10

    def test_should_discard_false_when_valid(self) -> None:
        """Test should_discard returns False for valid problem."""
        problem = make_coding_problem(parsing_errors=None, is_banned=False)
        assert problem.should_discard() is False

    def test_should_discard_true_when_banned(self) -> None:
        """Test should_discard returns True when banned."""
        problem = make_coding_problem(is_banned=True)
        assert problem.should_discard() is True

    def test_should_discard_true_when_parsing_errors(self) -> None:
        """Test should_discard returns True when parsing_errors exist."""
        problem = make_coding_problem(parsing_errors=["syntax error"])
        assert problem.should_discard() is True

    def test_should_discard_false_when_empty_parsing_errors(self) -> None:
        """Test should_discard returns False when parsing_errors is empty list."""
        problem = make_coding_problem(parsing_errors=[])
        assert problem.should_discard() is False

    def test_frozen(self) -> None:
        """Test that CodingProblem is frozen."""
        problem = make_coding_problem()
        with pytest.raises(pydantic.ValidationError):
            problem.is_banned = True  # noqa


# -----------------------------------------------------------------------------
#                               Solution tests
# -----------------------------------------------------------------------------


class TestSolution:
    """Tests for the Solution pydantic model."""

    def test_creation(self) -> None:
        """Test basic Solution creation."""
        solution = make_solution()
        assert solution.code == "def solution(x):\n    return x\n"

    def test_str_returns_solution_id(self) -> None:
        """Test that __str__ returns the solution identifier."""
        sid = make_solution_id("TACO", "train", 42, 5)
        solution = make_solution(solution_id=sid)
        assert str(solution) == "TACO/train/p000042/s0005"

    def test_code_line_count(self) -> None:
        """Test code_line_count property."""
        solution = make_solution(code="line1\nline2\nline3\n")
        assert solution.code_line_count == 3

    def test_code_line_count_single_line(self) -> None:
        """Test code_line_count with single line."""
        solution = make_solution(code="x = 1")
        assert solution.code_line_count == 1

    def test_is_fishy_false_when_clean(self) -> None:
        """Test is_fishy is False for clean code."""
        solution = make_solution()
        assert solution.is_fishy is False

    def test_is_fishy_true_for_nonstandard_packages(self) -> None:
        """Test is_fishy is True when imports nonstandard packages."""
        solution = make_solution(analysis_overrides={"imports_nonstandard_packages": True})
        assert solution.is_fishy is True

    def test_is_fishy_true_for_invalid_syntax(self) -> None:
        """Test is_fishy is True when has invalid syntax."""
        solution = make_solution(analysis_overrides={"invalid_syntax": True})
        assert solution.is_fishy is True

    def test_is_fishy_true_for_filesystem_access(self) -> None:
        """Test is_fishy is True when has filesystem access."""
        solution = make_solution(analysis_overrides={"filesystem_access": True})
        assert solution.is_fishy is True

    def test_is_fishy_true_for_system_commands(self) -> None:
        """Test is_fishy is True when uses system commands."""
        solution = make_solution(analysis_overrides={"system_commands": True})
        assert solution.is_fishy is True

    def test_is_fishy_true_for_network_access(self) -> None:
        """Test is_fishy is True when has network access."""
        solution = make_solution(analysis_overrides={"network_access": True})
        assert solution.is_fishy is True

    def test_is_deterministic_true(self) -> None:
        """Test is_deterministic is True by default."""
        solution = make_solution()
        assert solution.is_deterministic is True

    def test_is_deterministic_false(self) -> None:
        """Test is_deterministic is False when non-deterministic."""
        solution = make_solution(analysis_overrides={"is_deterministic": False})
        assert solution.is_deterministic is False

    def test_has_standard_io_true(self) -> None:
        """Test has_standard_io is True for stdin/stdout."""
        solution = make_solution()
        assert solution.has_standard_io is True

    def test_has_standard_io_true_for_callable(self) -> None:
        """Test has_standard_io is True for callable input/output."""
        solution = make_solution(
            analysis_overrides={
                "input_type": code_analysis.InputTypeOptions.CALLABLE,
                "output_type": code_analysis.OutputTypeOptions.CALLABLE,
            }
        )
        assert solution.has_standard_io is True

    def test_has_standard_io_false_for_file_input(self) -> None:
        """Test has_standard_io is False for file input."""
        solution = make_solution(analysis_overrides={"input_type": code_analysis.InputTypeOptions.FILE})
        assert solution.has_standard_io is False

    def test_should_discard_false_when_valid(self) -> None:
        """Test should_discard returns False for valid solution."""
        solution = make_solution()
        assert solution.should_discard() is False

    def test_should_discard_true_when_banned(self) -> None:
        """Test should_discard returns True when banned."""
        solution = make_solution(is_banned=True)
        assert solution.should_discard() is True

    def test_should_discard_true_when_analysis_errors(self) -> None:
        """Test should_discard returns True when analysis_errors exist."""
        solution = make_solution(analysis_errors=["error during analysis"])
        assert solution.should_discard() is True

    def test_should_discard_true_when_fishy(self) -> None:
        """Test should_discard returns True when code is fishy."""
        solution = make_solution(analysis_overrides={"network_access": True})
        assert solution.should_discard() is True

    def test_should_discard_true_when_non_deterministic(self) -> None:
        """Test should_discard returns True when non-deterministic."""
        solution = make_solution(analysis_overrides={"is_deterministic": False})
        assert solution.should_discard() is True

    def test_should_discard_true_when_non_standard_io(self) -> None:
        """Test should_discard returns True when non-standard IO."""
        solution = make_solution(analysis_overrides={"input_type": code_analysis.InputTypeOptions.FILE})
        assert solution.should_discard() is True

    def test_frozen(self) -> None:
        """Test that Solution is frozen."""
        solution = make_solution()
        with pytest.raises(pydantic.ValidationError):
            solution.code = "other"  # noqa


# -----------------------------------------------------------------------------
#                           TraceMetadata tests
# -----------------------------------------------------------------------------


class TestTraceMetadata:
    """Tests for the TraceMetadata dataclass."""

    def make_trace_metadata(
        self,
        identifier: str = "DS/sub/p000000/s0000/t0000",
        **overrides: object,
    ) -> dataset_utils.TraceMetadata:
        """Create a TraceMetadata instance for testing."""
        defaults = {
            "identifier": identifier,
            "parent_dataset_hash": "hash123",
            "index": 0,
            "internal_index": 0,
            "step_count": 100,
            "code_string": "def solution(x): return x",
            "inputs": [1],
            "expected_output": 1,
            "return_value": 1,
            "exception": None,
            "stdout": "",
            "stderr": "",
            "metadata": {"source_dataset_hash": "abc"},
            "tags": ["math", "easy"],
        }
        defaults.update(overrides)
        return dataset_utils.TraceMetadata(**defaults)

    def test_creation(self) -> None:
        """Test basic TraceMetadata creation."""
        meta = self.make_trace_metadata()
        assert meta.identifier == "DS/sub/p000000/s0000/t0000"
        assert meta.step_count == 100

    def test_trace_id_property(self) -> None:
        """Test trace_id cached property."""
        meta = self.make_trace_metadata(identifier="TACO/train/p000042/s0005/t0003")
        assert isinstance(meta.trace_id, dataset_utils.TraceIdentifier)
        assert meta.trace_id.problem_idx == 42
        assert meta.trace_id.solution_idx == 5
        assert meta.trace_id.test_idx == 3

    def test_solution_id_property(self) -> None:
        """Test solution_id cached property."""
        meta = self.make_trace_metadata(identifier="TACO/train/p000042/s0005/t0003")
        assert isinstance(meta.solution_id, dataset_utils.SolutionIdentifier)
        assert meta.solution_id.solution_idx == 5

    def test_problem_id_property(self) -> None:
        """Test problem_id cached property."""
        meta = self.make_trace_metadata(identifier="TACO/train/p000042/s0005/t0003")
        assert isinstance(meta.problem_id, dataset_utils.CodingProblemIdentifier)
        assert meta.problem_id.problem_idx == 42

    def test_is_augmented_false(self) -> None:
        """Test is_augmented is False for non-augmented trace."""
        meta = self.make_trace_metadata(identifier="DS/sub/p000000/s0000/t0000")
        assert meta.is_augmented is False

    def test_is_augmented_true(self) -> None:
        """Test is_augmented is True for augmented trace."""
        meta = self.make_trace_metadata(
            identifier="DS/sub/p000000/s0000/t0000/a:issues_todos:001",
            tags=["math", "augment:issues_todos"],
        )
        assert meta.is_augmented is True

    def test_augment_tags_empty_for_non_augmented(self) -> None:
        """Test augment_tags is empty for non-augmented trace."""
        meta = self.make_trace_metadata(identifier="DS/sub/p000000/s0000/t0000")
        assert meta.augment_tags == []

    def test_augment_tags_for_augmented_trace(self) -> None:
        """Test augment_tags returns augment tags for augmented trace."""
        meta = self.make_trace_metadata(
            identifier="DS/sub/p000000/s0000/t0000/a:issues_todos:001",
            tags=["math", "augment:issues_todos", "easy"],
        )
        assert meta.augment_tags == ["augment:issues_todos"]

    def test_augment_tags_multiple(self) -> None:
        """Test augment_tags with multiple augment categories."""
        meta = self.make_trace_metadata(
            identifier="DS/sub/p000000/s0000/t0000/a:issues_todos+hints_docs:001",
            tags=["augment:issues_todos", "augment:hints_docs"],
        )
        # Order may vary, check both are present
        assert "augment:issues_todos" in meta.augment_tags
        assert "augment:hints_docs" in meta.augment_tags

    def test_frozen(self) -> None:
        """Test that TraceMetadata is frozen."""
        meta = self.make_trace_metadata()
        with pytest.raises(AttributeError):
            meta.step_count = 999  # noqa

    def test_cached_properties_are_cached(self) -> None:
        """Test that cached properties return the same object."""
        meta = self.make_trace_metadata(identifier="DS/sub/p000000/s0000/t0000")
        trace_id_1 = meta.trace_id
        trace_id_2 = meta.trace_id
        assert trace_id_1 is trace_id_2
