"""Tests for dataset writer pipeline functions in pyine.data.traces.dataset_writer.

Tests _get_test_tuples(), TraceDatasetWriterConfig, and related functions.
"""

import pytest

import pyine.data.traces.common as common
import pyine.data.traces.dataset_utils as dataset_utils
import pyine.data.traces.dataset_writer as dataset_writer
import pyine.prompts.configs.code_analysis as code_analysis


def make_analysis_response(**overrides: object) -> code_analysis.CodeAnalysisResponse:
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


def make_problem(
    test_pairs: list[tuple] | None = None,
    num_tests: int = 5,
) -> dataset_utils.CodingProblem:
    """Create a CodingProblem for testing."""
    pid = dataset_utils.CodingProblemIdentifier("DS", "sub", 0)
    if test_pairs is None:
        test_pairs = [(i, i * 2) for i in range(num_tests)]
    sid = dataset_utils.SolutionIdentifier("DS", "sub", 0, 0)
    return dataset_utils.CodingProblem(
        source_dataset_name="DS",
        source_data_path="/data/DS/sub",
        source_data_hash="abc123",
        problem_id=pid,
        problem_statement="Solve this.",
        problem_tags=["easy"],
        test_inout_pairs=test_pairs,
        entrypoint_name="solution",
        potential_solution_ids=[sid],
        parsing_errors=None,
        is_banned=False,
    )


def make_solution() -> dataset_utils.Solution:
    """Create a Solution for testing."""
    sid = dataset_utils.SolutionIdentifier("DS", "sub", 0, 0)
    pid = dataset_utils.CodingProblemIdentifier("DS", "sub", 0)
    return dataset_utils.Solution(
        parent_id=pid,
        solution_id=sid,
        code="def solution(x): return x * 2",
        analysis_errors=None,
        analysis_results=make_analysis_response(),
        is_banned=False,
    )


class TestTraceDatasetWriterConfig:
    """Tests for TraceDatasetWriterConfig pydantic model."""

    def test_creation_with_required_fields(self) -> None:
        """Test creating config with only required fields."""
        config = dataset_writer.TraceDatasetWriterConfig(source_dataset_name="TACO")
        assert config.source_dataset_name == "TACO"

    def test_custom_values(self) -> None:
        """Test custom values."""
        config = dataset_writer.TraceDatasetWriterConfig(
            source_dataset_name="TACO",
            max_output_traces=1000,
            max_solutions_per_problem=10,
            max_tests_per_solution=5,
            min_solution_line_count=3,
            min_solution_dissimilarity=0.2,
        )
        assert config.max_output_traces == 1000
        assert config.max_solutions_per_problem == 10
        assert config.max_tests_per_solution == 5
        assert config.min_solution_line_count == 3
        assert config.min_solution_dissimilarity == 0.2

    def test_get_rng_determinism_with_same_seed(self) -> None:
        """Test that get_rng with same seed produces deterministic results."""
        config = dataset_writer.TraceDatasetWriterConfig(
            source_dataset_name="TACO",
            execution_seed=42,
        )
        rng1 = config.get_rng(seed=[1, 2, 3])
        rng2 = config.get_rng(seed=[1, 2, 3])
        # both should produce the same sequence
        assert list(rng1.permutation(10)) == list(rng2.permutation(10))

    def test_get_rng_different_with_different_seed(self) -> None:
        """Test that get_rng with different seeds produces different results."""
        config = dataset_writer.TraceDatasetWriterConfig(
            source_dataset_name="TACO",
            execution_seed=42,
        )
        rng1 = config.get_rng(seed=[1, 2, 3])
        rng2 = config.get_rng(seed=[4, 5, 6])
        # the sequences should be different
        assert list(rng1.permutation(10)) != list(rng2.permutation(10))

    def test_get_short_hash_consistency(self) -> None:
        """Test that get_short_hash returns consistent results."""
        config = dataset_writer.TraceDatasetWriterConfig(source_dataset_name="TACO")
        hash1 = config.get_short_hash()
        hash2 = config.get_short_hash()
        assert hash1 == hash2

    def test_get_short_hash_different_for_different_configs(self) -> None:
        """Test that different configs produce different hashes."""
        config1 = dataset_writer.TraceDatasetWriterConfig(
            source_dataset_name="TACO",
            max_output_traces=100,
        )
        config2 = dataset_writer.TraceDatasetWriterConfig(
            source_dataset_name="TACO",
            max_output_traces=200,
        )
        assert config1.get_short_hash() != config2.get_short_hash()

    def test_inherits_from_tracing_config(self) -> None:
        """Test that TraceDatasetWriterConfig inherits from TracingConfig."""
        config = dataset_writer.TraceDatasetWriterConfig(
            source_dataset_name="TACO",
            max_trace_valid_events=1000,
            execution_timeout_seconds=5.0,
        )
        assert config.max_trace_valid_events == 1000
        assert config.execution_timeout_seconds == 5.0


class TestGetTestTuples:
    """Tests for _get_test_tuples function."""

    def test_returns_all_tests_by_default(self) -> None:
        """Test that all tests are returned when no limits set."""
        problem = make_problem(num_tests=5)
        solution = make_solution()
        config = dataset_writer.TraceDatasetWriterConfig(source_dataset_name="DS")
        tuples = dataset_writer._get_test_tuples(problem, solution, config)
        assert len(tuples) == 5
        assert all(isinstance(t, common.TestTuple) for t in tuples)

    def test_respects_max_tests_per_solution(self) -> None:
        """Test that max_tests_per_solution limits results."""
        problem = make_problem(num_tests=10)
        solution = make_solution()
        config = dataset_writer.TraceDatasetWriterConfig(
            source_dataset_name="DS",
            max_tests_per_solution=3,
        )
        tuples = dataset_writer._get_test_tuples(problem, solution, config)
        assert len(tuples) == 3

    def test_filters_by_max_tests_args_length(self) -> None:
        """Test that tests with long arguments are filtered out."""
        # create tests with varying argument lengths
        test_pairs = [
            (1, 2),  # short
            ("x" * 100, "y" * 100),  # long
            (3, 4),  # short
        ]
        problem = make_problem(test_pairs=test_pairs)
        solution = make_solution()
        config = dataset_writer.TraceDatasetWriterConfig(
            source_dataset_name="DS",
            max_tests_args_length=50,  # filter out the long one
        )
        tuples = dataset_writer._get_test_tuples(problem, solution, config)
        assert len(tuples) == 2  # only the short tests should remain
        assert all(len(str(t.inputs)) + len(str(t.outputs)) < 50 for t in tuples)

    def test_random_selection_with_seed(self) -> None:
        """Test that random selection is deterministic with seed."""
        problem = make_problem(num_tests=10)
        solution = make_solution()
        config = dataset_writer.TraceDatasetWriterConfig(
            source_dataset_name="DS",
            max_tests_per_solution=3,
            pick_random_tests_per_solution=True,
            execution_seed=42,
        )
        tuples1 = dataset_writer._get_test_tuples(problem, solution, config)
        tuples2 = dataset_writer._get_test_tuples(problem, solution, config)
        # same seed should give same selection
        assert [t.test_idx for t in tuples1] == [t.test_idx for t in tuples2]

    def test_random_selection_shuffles_order(self) -> None:
        """Test that random selection can shuffle the order."""
        problem = make_problem(num_tests=10)
        solution = make_solution()
        config = dataset_writer.TraceDatasetWriterConfig(
            source_dataset_name="DS",
            max_tests_per_solution=5,
            pick_random_tests_per_solution=True,
            execution_seed=12345,  # Use a seed that likely shuffles
        )
        tuples = dataset_writer._get_test_tuples(problem, solution, config)
        indices = [t.test_idx for t in tuples]
        # check that indices are not simply 0,1,2,3,4 (the random selection should shuffle them)
        assert sorted(indices) != list(range(5)) or indices != list(range(5))

    def test_raises_on_no_test_cases(self) -> None:
        """Test that ValueError is raised when no test cases available."""
        problem = make_problem(test_pairs=[])
        solution = make_solution()
        config = dataset_writer.TraceDatasetWriterConfig(source_dataset_name="DS")
        with pytest.raises(ValueError, match="no test cases found"):
            dataset_writer._get_test_tuples(problem, solution, config)

    def test_test_tuple_has_correct_fields(self) -> None:
        """Test that returned TestTuples have correct field values."""
        test_pairs = [(10, 20), (30, 40)]
        problem = make_problem(test_pairs=test_pairs)
        solution = make_solution()
        config = dataset_writer.TraceDatasetWriterConfig(source_dataset_name="DS")
        tuples = dataset_writer._get_test_tuples(problem, solution, config)
        assert tuples[0].test_idx == 0
        assert tuples[0].inputs == 10
        assert tuples[0].outputs == 20
        assert tuples[1].test_idx == 1
        assert tuples[1].inputs == 30
        assert tuples[1].outputs == 40


class TestCheckMustSkipProblemExtended:
    """Extended tests for _check_must_skip_problem function."""

    def test_no_skip_for_valid_problem(self) -> None:
        """Test that valid problems are not skipped."""
        pid = dataset_utils.CodingProblemIdentifier("ds", "subset", 0)
        sid = dataset_utils.SolutionIdentifier("ds", "subset", 0, 0)
        problem = dataset_utils.CodingProblem(
            source_dataset_name="ds",
            source_data_path="/data/ds/subset/test.json",
            source_data_hash="hash",
            problem_id=pid,
            problem_statement="do stuff",
            problem_tags=["easy"],
            test_inout_pairs=[((1,), 2)],
            entrypoint_name="solve",
            potential_solution_ids=[sid],
            parsing_errors=None,
            is_banned=False,
        )
        solution = dataset_utils.Solution(
            parent_id=pid,
            solution_id=sid,
            code="def solve(x): return x + 1",
            analysis_errors=None,
            analysis_results=make_analysis_response(),
            is_banned=False,
        )
        config = dataset_writer.TraceDatasetWriterConfig(source_dataset_name="ds")
        contains_banned = lambda tags: False  # noqa
        message = dataset_writer._check_must_skip_problem(
            problem=problem,
            solutions=[solution],
            config=config,
            contains_banned_tags=contains_banned,
        )
        assert message is None

    def test_skip_for_no_solutions(self) -> None:
        """Test that problems with no solutions are skipped."""
        pid = dataset_utils.CodingProblemIdentifier("ds", "subset", 0)
        problem = dataset_utils.CodingProblem(
            source_dataset_name="ds",
            source_data_path="/data/ds/subset/test.json",
            source_data_hash="hash",
            problem_id=pid,
            problem_statement="do stuff",
            problem_tags=["easy"],
            test_inout_pairs=[((1,), 2)],
            entrypoint_name="solve",
            potential_solution_ids=[],
            parsing_errors=None,
            is_banned=False,
        )
        config = dataset_writer.TraceDatasetWriterConfig(source_dataset_name="ds")
        contains_banned = lambda tags: False  # noqa
        message = dataset_writer._check_must_skip_problem(
            problem=problem,
            solutions=[],
            config=config,
            contains_banned_tags=contains_banned,
        )
        assert message is not None
        assert "no solutions" in message.lower()


class TestCheckMustSkipSolution:
    """Tests for _check_must_skip_solution function."""

    def test_no_skip_for_valid_solution(self) -> None:
        """Test that valid solutions are not skipped."""
        pid = dataset_utils.CodingProblemIdentifier("ds", "subset", 0)
        sid = dataset_utils.SolutionIdentifier("ds", "subset", 0, 0)
        problem = dataset_utils.CodingProblem(
            source_dataset_name="ds",
            source_data_path="/data/ds/subset/test.json",
            source_data_hash="hash",
            problem_id=pid,
            problem_statement="do stuff",
            problem_tags=["easy"],
            test_inout_pairs=[((1,), 2)],
            entrypoint_name="solve",
            potential_solution_ids=[sid],
            parsing_errors=None,
            is_banned=False,
        )
        solution = dataset_utils.Solution(
            parent_id=pid,
            solution_id=sid,
            code="def solve(x):\n    return x + 1\n",
            analysis_errors=None,
            analysis_results=make_analysis_response(
                input_type=code_analysis.InputTypeOptions.CALLABLE,
                output_type=code_analysis.OutputTypeOptions.CALLABLE,
            ),
            is_banned=False,
        )
        config = dataset_writer.TraceDatasetWriterConfig(
            source_dataset_name="ds",
            min_solution_line_count=1,
        )
        message = dataset_writer._check_must_skip_solution(
            problem=problem,
            solution=solution,
            solution_idx=0,
            retained_solution_indices=[0],
            config=config,
        )
        assert message is None

    def test_skip_for_banned_solution(self) -> None:
        """Test that banned solutions are skipped."""
        pid = dataset_utils.CodingProblemIdentifier("ds", "subset", 0)
        sid = dataset_utils.SolutionIdentifier("ds", "subset", 0, 0)
        problem = dataset_utils.CodingProblem(
            source_dataset_name="ds",
            source_data_path="/data/ds/subset/test.json",
            source_data_hash="hash",
            problem_id=pid,
            problem_statement="do stuff",
            problem_tags=["easy"],
            test_inout_pairs=[((1,), 2)],
            entrypoint_name="solve",
            potential_solution_ids=[sid],
            parsing_errors=None,
            is_banned=False,
        )
        solution = dataset_utils.Solution(
            parent_id=pid,
            solution_id=sid,
            code="def solve(x):\n    return x + 1\n",
            analysis_errors=None,
            analysis_results=make_analysis_response(),
            is_banned=True,
        )
        config = dataset_writer.TraceDatasetWriterConfig(source_dataset_name="ds")
        message = dataset_writer._check_must_skip_solution(
            problem=problem,
            solution=solution,
            solution_idx=0,
            retained_solution_indices=[0],
            config=config,
        )
        assert message is not None
        assert "banned" in message.lower() or "skipped" in message.lower()

    def test_skip_for_analysis_errors(self) -> None:
        """Test that solutions with analysis errors are skipped."""
        pid = dataset_utils.CodingProblemIdentifier("ds", "subset", 0)
        sid = dataset_utils.SolutionIdentifier("ds", "subset", 0, 0)
        problem = dataset_utils.CodingProblem(
            source_dataset_name="ds",
            source_data_path="/data/ds/subset/test.json",
            source_data_hash="hash",
            problem_id=pid,
            problem_statement="do stuff",
            problem_tags=["easy"],
            test_inout_pairs=[((1,), 2)],
            entrypoint_name="solve",
            potential_solution_ids=[sid],
            parsing_errors=None,
            is_banned=False,
        )
        solution = dataset_utils.Solution(
            parent_id=pid,
            solution_id=sid,
            code="def solve(x): return x + 1",
            analysis_errors=["some error"],  # Has analysis errors
            analysis_results=make_analysis_response(),
            is_banned=False,
        )
        config = dataset_writer.TraceDatasetWriterConfig(source_dataset_name="ds")
        message = dataset_writer._check_must_skip_solution(
            problem=problem,
            solution=solution,
            solution_idx=0,
            retained_solution_indices=[0],
            config=config,
        )
        assert message is not None
        assert "analysis error" in message.lower()

    def test_skip_for_too_few_lines(self) -> None:
        """Test that solutions with too few lines are skipped."""
        pid = dataset_utils.CodingProblemIdentifier("ds", "subset", 0)
        sid = dataset_utils.SolutionIdentifier("ds", "subset", 0, 0)
        problem = dataset_utils.CodingProblem(
            source_dataset_name="ds",
            source_data_path="/data/ds/subset/test.json",
            source_data_hash="hash",
            problem_id=pid,
            problem_statement="do stuff",
            problem_tags=["easy"],
            test_inout_pairs=[((1,), 2)],
            entrypoint_name="solve",
            potential_solution_ids=[sid],
            parsing_errors=None,
            is_banned=False,
        )
        solution = dataset_utils.Solution(
            parent_id=pid,
            solution_id=sid,
            code="x = 1",  # single line
            analysis_errors=None,
            analysis_results=make_analysis_response(
                input_type=code_analysis.InputTypeOptions.CALLABLE,
                output_type=code_analysis.OutputTypeOptions.CALLABLE,
            ),
            is_banned=False,
        )
        config = dataset_writer.TraceDatasetWriterConfig(
            source_dataset_name="ds",
            min_solution_line_count=5,  # require 5 lines
        )
        message = dataset_writer._check_must_skip_solution(
            problem=problem,
            solution=solution,
            solution_idx=0,
            retained_solution_indices=[0],
            config=config,
        )
        assert message is not None
        assert "few lines" in message.lower()

    def test_skip_for_duplicate(self) -> None:
        """Test that duplicate solutions are skipped."""
        pid = dataset_utils.CodingProblemIdentifier("ds", "subset", 0)
        sid = dataset_utils.SolutionIdentifier("ds", "subset", 0, 0)
        problem = dataset_utils.CodingProblem(
            source_dataset_name="ds",
            source_data_path="/data/ds/subset/test.json",
            source_data_hash="hash",
            problem_id=pid,
            problem_statement="do stuff",
            problem_tags=["easy"],
            test_inout_pairs=[((1,), 2)],
            entrypoint_name="solve",
            potential_solution_ids=[sid],
            parsing_errors=None,
            is_banned=False,
        )
        solution = dataset_utils.Solution(
            parent_id=pid,
            solution_id=sid,
            code="def solve(x):\n    return x + 1\n",
            analysis_errors=None,
            analysis_results=make_analysis_response(
                input_type=code_analysis.InputTypeOptions.CALLABLE,
                output_type=code_analysis.OutputTypeOptions.CALLABLE,
            ),
            is_banned=False,
        )
        config = dataset_writer.TraceDatasetWriterConfig(source_dataset_name="ds")
        message = dataset_writer._check_must_skip_solution(
            problem=problem,
            solution=solution,
            solution_idx=0,
            retained_solution_indices=[1, 2],  # idx 0 not retained
            config=config,
        )
        assert message is not None
        assert "duplicate" in message.lower()
