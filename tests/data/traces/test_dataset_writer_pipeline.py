"""Tests for dataset writer pipeline functions in pyine.data.traces.dataset_writer.

Tests _get_test_tuples(), TraceDatasetWriterConfig, and related functions.
"""

import contextlib
import pathlib

import pytest

import pyine.data.traces.common as common
import pyine.data.traces.dataset_utils as dataset_utils
import pyine.data.traces.dataset_writer as dataset_writer
import pyine.data.utils.lmdb_io as lmdb_io
import pyine.prompts.configs.code_analysis as code_analysis
import tests.env_checks


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
    problem_idx: int = 0,
    solution_idx: int = 0,
) -> dataset_utils.CodingProblem:
    """Create a CodingProblem for testing."""
    pid = dataset_utils.CodingProblemIdentifier("DS", "sub", problem_idx)
    if test_pairs is None:
        test_pairs = [(i, i * 2) for i in range(num_tests)]
    sid = dataset_utils.SolutionIdentifier("DS", "sub", problem_idx, solution_idx)
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


def make_solution(
    problem_idx: int = 0,
    solution_idx: int = 0,
) -> dataset_utils.Solution:
    """Create a Solution for testing."""
    sid = dataset_utils.SolutionIdentifier("DS", "sub", problem_idx, solution_idx)
    pid = dataset_utils.CodingProblemIdentifier("DS", "sub", problem_idx)
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


class TestReproducibility:
    """Tests for reproducibility of the trace dataset writer pipeline.

    These tests verify that given the same seed and input code snippets, the same
    problems/solutions/tests are traced (deterministically). Different seeds should
    lead to different test selections.
    """

    def test_get_rng_reproducible_across_config_instances(self) -> None:
        """Test that separate config instances with same seed produce identical RNG sequences."""
        config1 = dataset_writer.TraceDatasetWriterConfig(source_dataset_name="DS", seed=42)
        config2 = dataset_writer.TraceDatasetWriterConfig(source_dataset_name="DS", seed=42)
        # create RNGs with the same supplemental seed
        rng1 = config1.get_rng(seed=[10, 20])
        rng2 = config2.get_rng(seed=[10, 20])
        # they should produce identical sequences
        seq1 = list(rng1.permutation(100))
        seq2 = list(rng2.permutation(100))
        assert seq1 == seq2

    def test_get_rng_reproducible_with_int_seed(self) -> None:
        """Test that get_rng is reproducible when given an int seed."""
        config = dataset_writer.TraceDatasetWriterConfig(source_dataset_name="DS", seed=123)
        rng1 = config.get_rng(seed=999)
        rng2 = config.get_rng(seed=999)
        assert list(rng1.permutation(50)) == list(rng2.permutation(50))

    def test_get_rng_different_configs_different_sequences(self) -> None:
        """Test that different config seeds produce different RNG sequences."""
        config1 = dataset_writer.TraceDatasetWriterConfig(source_dataset_name="DS", seed=42)
        config2 = dataset_writer.TraceDatasetWriterConfig(source_dataset_name="DS", seed=43)
        rng1 = config1.get_rng(seed=[10, 20])
        rng2 = config2.get_rng(seed=[10, 20])
        seq1 = list(rng1.permutation(100))
        seq2 = list(rng2.permutation(100))
        assert seq1 != seq2

    def test_get_rng_none_seed_is_non_deterministic(self) -> None:
        """Test that seed=None produces non-deterministic RNGs."""
        config = dataset_writer.TraceDatasetWriterConfig(source_dataset_name="DS", seed=None)
        # with seed=None, consecutive calls should produce different sequences
        # (with very high probability)
        rng1 = config.get_rng(seed=[10, 20])
        rng2 = config.get_rng(seed=[10, 20])
        seq1 = list(rng1.permutation(100))
        seq2 = list(rng2.permutation(100))
        # these should almost certainly be different when seed is None
        assert seq1 != seq2

    def test_get_test_tuples_reproducible_with_same_seed(self) -> None:
        """Test that _get_test_tuples produces identical results with the same seed."""
        problem = make_problem(num_tests=20)
        solution = make_solution()
        config = dataset_writer.TraceDatasetWriterConfig(
            source_dataset_name="DS",
            seed=42,
            max_tests_per_solution=5,
            pick_random_tests_per_solution=True,
        )
        tuples1 = dataset_writer._get_test_tuples(problem, solution, config)
        tuples2 = dataset_writer._get_test_tuples(problem, solution, config)
        # should be identical
        assert [t.test_idx for t in tuples1] == [t.test_idx for t in tuples2]
        assert [t.inputs for t in tuples1] == [t.inputs for t in tuples2]
        assert [t.outputs for t in tuples1] == [t.outputs for t in tuples2]

    def test_get_test_tuples_reproducible_across_config_instances(self) -> None:
        """Test that separate config instances with same seed yield same test tuples."""
        problem = make_problem(num_tests=20)
        solution = make_solution()
        config1 = dataset_writer.TraceDatasetWriterConfig(
            source_dataset_name="DS",
            seed=999,
            max_tests_per_solution=7,
            pick_random_tests_per_solution=True,
        )
        config2 = dataset_writer.TraceDatasetWriterConfig(
            source_dataset_name="DS",
            seed=999,
            max_tests_per_solution=7,
            pick_random_tests_per_solution=True,
        )
        tuples1 = dataset_writer._get_test_tuples(problem, solution, config1)
        tuples2 = dataset_writer._get_test_tuples(problem, solution, config2)
        assert [t.test_idx for t in tuples1] == [t.test_idx for t in tuples2]

    def test_get_test_tuples_different_seeds_different_selection(self) -> None:
        """Test that different seeds produce different test selections."""
        problem = make_problem(num_tests=50)
        solution = make_solution()
        config1 = dataset_writer.TraceDatasetWriterConfig(
            source_dataset_name="DS",
            seed=100,
            max_tests_per_solution=10,
            pick_random_tests_per_solution=True,
        )
        config2 = dataset_writer.TraceDatasetWriterConfig(
            source_dataset_name="DS",
            seed=200,
            max_tests_per_solution=10,
            pick_random_tests_per_solution=True,
        )
        tuples1 = dataset_writer._get_test_tuples(problem, solution, config1)
        tuples2 = dataset_writer._get_test_tuples(problem, solution, config2)
        indices1 = [t.test_idx for t in tuples1]
        indices2 = [t.test_idx for t in tuples2]
        # with 50 tests, sampling 10, different seeds should give different selections
        assert indices1 != indices2

    def test_get_test_tuples_different_problem_solution_pairs_different_selection(self) -> None:
        """Test that same seed but different problem/solution indices give different selections."""
        config = dataset_writer.TraceDatasetWriterConfig(
            source_dataset_name="DS",
            seed=42,
            max_tests_per_solution=5,
            pick_random_tests_per_solution=True,
        )
        # same problem, different solutions
        problem1 = make_problem(num_tests=30, problem_idx=0, solution_idx=0)
        solution1 = make_solution(problem_idx=0, solution_idx=0)
        problem2 = make_problem(num_tests=30, problem_idx=0, solution_idx=1)
        solution2 = make_solution(problem_idx=0, solution_idx=1)
        tuples1 = dataset_writer._get_test_tuples(problem1, solution1, config)
        tuples2 = dataset_writer._get_test_tuples(problem2, solution2, config)
        indices1 = [t.test_idx for t in tuples1]
        indices2 = [t.test_idx for t in tuples2]
        # different solution indices should give different selections
        assert indices1 != indices2

    def test_get_test_tuples_different_problems_different_selection(self) -> None:
        """Test that different problem indices with same seed give different test selections."""
        config = dataset_writer.TraceDatasetWriterConfig(
            source_dataset_name="DS",
            seed=42,
            max_tests_per_solution=5,
            pick_random_tests_per_solution=True,
        )
        problem1 = make_problem(num_tests=30, problem_idx=0, solution_idx=0)
        solution1 = make_solution(problem_idx=0, solution_idx=0)
        problem2 = make_problem(num_tests=30, problem_idx=1, solution_idx=0)
        solution2 = make_solution(problem_idx=1, solution_idx=0)
        tuples1 = dataset_writer._get_test_tuples(problem1, solution1, config)
        tuples2 = dataset_writer._get_test_tuples(problem2, solution2, config)
        indices1 = [t.test_idx for t in tuples1]
        indices2 = [t.test_idx for t in tuples2]
        # different problem indices should give different selections
        assert indices1 != indices2

    def test_get_test_tuples_no_randomization_is_deterministic(self) -> None:
        """Test that without random selection, results are always deterministic."""
        problem = make_problem(num_tests=10)
        solution = make_solution()
        config = dataset_writer.TraceDatasetWriterConfig(
            source_dataset_name="DS",
            seed=None,  # even with no seed
            max_tests_per_solution=5,
            pick_random_tests_per_solution=False,  # no randomization
        )
        tuples1 = dataset_writer._get_test_tuples(problem, solution, config)
        tuples2 = dataset_writer._get_test_tuples(problem, solution, config)
        # without randomization, should always get the first N tests
        assert [t.test_idx for t in tuples1] == [t.test_idx for t in tuples2]
        assert [t.test_idx for t in tuples1] == [0, 1, 2, 3, 4]

    def test_rng_hierarchy_produces_varied_but_reproducible_sequences(self) -> None:
        """Test that the hierarchical seeding produces varied but reproducible sequences.

        This test verifies that for multiple problem/solution pairs, the sequences
        are all different from each other (varied) but each individual pair produces
        the same sequence when regenerated (reproducible).
        """
        config = dataset_writer.TraceDatasetWriterConfig(source_dataset_name="DS", seed=12345)
        all_sequences: list[list[int]] = []
        # generate sequences for multiple problem/solution pairs
        for p_idx in range(3):
            for s_idx in range(3):
                rng = config.get_rng(seed=[p_idx, s_idx])
                seq = list(rng.permutation(20))
                all_sequences.append(seq)
        # verify all sequences are unique
        unique_sequences = [tuple(s) for s in all_sequences]
        assert len(set(unique_sequences)) == len(all_sequences)
        # verify reproducibility: regenerate and compare
        for p_idx in range(3):
            for s_idx in range(3):
                rng = config.get_rng(seed=[p_idx, s_idx])
                seq = list(rng.permutation(20))
                expected_idx = p_idx * 3 + s_idx
                assert seq == all_sequences[expected_idx]


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.dataset
@pytest.mark.timeout(600)
@pytest.mark.skipif(
    tests.env_checks.TACO_DATASET_MISSING,
    reason="TACO dataset is missing, cannot run reproducibility integration tests",
)
class TestReproducibilityIntegration:
    """Integration tests for trace dataset writer reproducibility using real TACO data.

    These tests verify that the complete write_dataset pipeline produces reproducible
    outputs when given the same seed. They require the TACO dataset to be available.
    """

    def test_write_dataset_reproducibility(self, tmp_path: pathlib.Path) -> None:
        """Test write_dataset reproducibility with same seed and variation with different seeds.

        This test runs write_dataset three times total:
        1. With seed=42 (first run)
        2. With seed=42 (second run, should match first)
        3. With seed=999 (different seed, should have different test selections)

        For same-seed comparison: traces that succeed in both runs should be identical.
        Some traces may drop due to timeouts, so we compare only common keys.

        For different-seed comparison: at least some test selections should differ.

        Also verifies that config parameters are properly stored in metadata.
        """
        import pyine.data.taco.dataset_utils as taco_utils

        source_path = taco_utils.get_latest_repackaged_dataset_path()
        output_same_seed_1 = tmp_path / "output_seed42_run1"
        output_same_seed_2 = tmp_path / "output_seed42_run2"
        output_diff_seed = tmp_path / "output_seed999"

        base_config_kwargs = {
            "source_dataset_name": "TACO",
            "max_output_traces": 15,
            "max_solutions_per_problem": 1,
            "max_tests_per_solution": 3,
            "pick_random_tests_per_solution": True,
            "execution_timeout_seconds": 5.0,
            "problem_data_overrides_setting": "auto",
        }

        # run 1: seed=42
        config_seed42_run1 = dataset_writer.TraceDatasetWriterConfig(seed=42, **base_config_kwargs)
        dataset_writer.write_dataset(
            root_dataset_path=source_path,
            output_dataset_path=output_same_seed_1,
            config=config_seed42_run1,
            verbose=False,
        )

        # run 2: seed=42 (should match run 1)
        config_seed42_run2 = dataset_writer.TraceDatasetWriterConfig(seed=42, **base_config_kwargs)
        dataset_writer.write_dataset(
            root_dataset_path=source_path,
            output_dataset_path=output_same_seed_2,
            config=config_seed42_run2,
            verbose=False,
        )

        # run 3: seed=999 (should differ from runs 1 & 2)
        config_seed999 = dataset_writer.TraceDatasetWriterConfig(seed=999, **base_config_kwargs)
        dataset_writer.write_dataset(
            root_dataset_path=source_path,
            output_dataset_path=output_diff_seed,
            config=config_seed999,
            verbose=False,
        )

        # read all three datasets
        reader1 = lmdb_io.LMDBReader(output_same_seed_1)
        reader2 = lmdb_io.LMDBReader(output_same_seed_2)
        reader3 = lmdb_io.LMDBReader(output_diff_seed)

        try:
            keys1 = set(reader1.key_map.keys())
            keys2 = set(reader2.key_map.keys())
            keys3 = set(reader3.key_map.keys())

            # --- Test 1: Same seed produces identical traces for common keys ---
            # Some traces may drop due to timeouts, so compare only common keys
            common_keys_same_seed = keys1 & keys2
            assert len(common_keys_same_seed) > 0, "No common keys between same-seed runs"

            # For common keys, trace content should be identical
            for key in common_keys_same_seed:
                data1 = reader1.get(key)
                data2 = reader2.get(key)
                if "traced_steps" in data1 and "traced_steps" in data2:
                    assert data1["traced_steps"] == data2["traced_steps"], (
                        f"Traced steps differ for key {key} between same-seed runs"
                    )

            # --- Test 2: Different seeds produce different test selections ---
            def extract_test_indices(keys: set[str]) -> set[tuple[str, int]]:
                result = set()
                for key in keys:
                    if dataset_utils.PROBLEM_DATA_SUFFIX not in key:
                        with contextlib.suppress(Exception):
                            tid = dataset_utils.TraceIdentifier.from_string(key)
                            group = f"{tid.dataset}/{tid.subset}/p{tid.problem_idx:06d}/s{tid.solution_idx:04d}"
                            result.add((group, tid.test_idx))
                return result

            tests1 = extract_test_indices(keys1)
            tests3 = extract_test_indices(keys3)

            # find common problem-solution groups between seed=42 and seed=999
            groups1 = {t[0] for t in tests1}
            groups3 = {t[0] for t in tests3}
            common_groups = groups1 & groups3

            # for at least one common group, the test indices should differ
            differences_found = False
            for group in common_groups:
                indices1 = {t[1] for t in tests1 if t[0] == group}
                indices3 = {t[1] for t in tests3 if t[0] == group}
                if indices1 != indices3:
                    differences_found = True
                    break

            if common_groups:
                assert differences_found, "Expected different test selections with different seeds"

            # --- Test 3: Config parameters are stored in metadata ---
            metadata1 = reader1.get_metadata()
            assert "writer_config" in metadata1
            stored_config_dict = metadata1["writer_config"]
            assert stored_config_dict["seed"] == 42
            assert stored_config_dict["source_dataset_name"] == "TACO"
            assert stored_config_dict["max_output_traces"] == 15
            assert stored_config_dict["max_solutions_per_problem"] == 1
            assert stored_config_dict["max_tests_per_solution"] == 3
            assert stored_config_dict["pick_random_tests_per_solution"] is True

            # verify config can be reconstructed from stored params
            stored_config = dataset_writer.TraceDatasetWriterConfig(**stored_config_dict)
            assert stored_config.seed == config_seed42_run1.seed
            assert stored_config.source_dataset_name == config_seed42_run1.source_dataset_name

        finally:
            reader1.close()
            reader2.close()
            reader3.close()
