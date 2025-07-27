"""
This module contains a writer for a dataset of code execution traces.

See the `write_dataset` function for more information.
"""

import itertools
import logging
import pathlib
import warnings

import pyine.data.traces.dataset_utils
import pyine.data.utils.lmdb_io
import pyine.prompts.code_analysis
import pyine.utils.code.execution
import pyine.utils.code.formatting
import pyine.utils.code.validation
import pyine.utils.filesystem
import pyine.utils.logging
import pyine.utils.portability
import pyine.utils.reprod

logger = logging.getLogger(__name__)


def write_dataset(
    source_dataset_name: str,
    root_dataset_path: pathlib.Path,
    output_dataset_path: pathlib.Path,
    max_output_traces: int | None = None,
    max_valid_solutions_per_problem: int | None = None,
    max_traces_per_solution: int | None = None,
    max_trace_events_per_line: int | None = None,
    minimum_solution_dissimilarity: float = 0.1,
    execution_timeout_seconds: float = 10,
    verbose: bool = False,
) -> pyine.data.utils.lmdb_io.LMDBWriter:
    """Writes a dataset of execution traces from a source dataset of coding problems and solutions.

    The execution traces are written to an LMDB dataset. Each trace corresponds to a successful
    code execution made for a specific solution to a coding problem, using a specific set of
    input arguments that are paired with an expected output value. These input/output pairs form
    a 'test', and the execution attempt is considered successful if the output value matches the
    expected value.

    All 'source' datasets supported by this writer provide examples of coding problems paired with
    solutions and input/output test pairs.

    Args:
        source_dataset_name: Name of the source dataset to read from.
        root_dataset_path: Path to the root directory of the source dataset.
        output_dataset_path: Path to the output dataset to write the traces to (LMDB format).
        max_output_traces: Maximum number of traces to write in total. If None, no maximum.
        max_valid_solutions_per_problem: Maximum number of valid solutions to write per problem. If None, no maximum.
        max_traces_per_solution: Maximum number of traces to write per solution. If None, no maximum.
        max_trace_events_per_line: Maximum number of trace events per solution code line. If None, no maximum.
        minimum_solution_dissimilarity: Minimum solution dissimilarity threshold to use for solution duplicate removal.
        execution_timeout_seconds: Timeout in seconds for each execution attempt. If exceeded, solution is skipped.
        verbose: Toggles verbose output/logging.

    Returns:
        The LMDBWriter object that was used to write the traces (once writing is complete).
    """
    log = logger.info if verbose else logger.debug
    problem_data_iter = pyine.data.traces.dataset_utils.CodingProblemIterator(
        dataset_name=source_dataset_name,
        root_data_path=root_dataset_path,
        show_progress=verbose,
    )
    assert len(problem_data_iter) > 0, f"no problems found in {source_dataset_name} source dataset"
    log(f"found {len(problem_data_iter)} problems in {source_dataset_name} source dataset")
    pyine.utils.filesystem.check_output_path_overwrite(output_dataset_path)
    log(f"creating LMDB dataset at: {output_dataset_path}...")
    writer = pyine.data.utils.lmdb_io.LMDBWriter(path=output_dataset_path)
    writer.write_metadata(  # start by writing metadata (creation hyperparams) to disk
        dict(
            source_dataset=dict(
                source_dataset_name=source_dataset_name,
                root_dataset_path=str(root_dataset_path),
                source_problem_count=len(problem_data_iter),
                max_output_traces=max_output_traces,
                max_valid_solutions_per_problem=max_valid_solutions_per_problem,
                max_traces_per_solution=max_traces_per_solution,
                max_trace_events_per_line=max_trace_events_per_line,
                minimum_solution_dissimilarity=minimum_solution_dissimilarity,
                execution_timeout_seconds=execution_timeout_seconds,
            ),
        ),
    )
    written_outputs = 0  # total number of traces that we will have written

    # iterate over each problem statement (and its proposed solutions) in the target dataset
    for problem, solutions in problem_data_iter:
        assert isinstance(problem, pyine.data.traces.dataset_utils.CodingProblem)
        assert isinstance(solutions, list)
        assert all([isinstance(s, pyine.data.traces.dataset_utils.Solution) for s in solutions])
        assert problem.potential_solution_ids == [s.solution_id for s in solutions]
        # first, make sure the problem is valid and we can use its solutions for tracing
        if problem.parsing_errors:
            log(f"{problem}: skipping due to parsing errors: {problem.parsing_errors}")
            continue
        if not solutions:
            log(f"{problem}: no solutions found, skipping")
            continue
        if problem.should_discard():
            log(f"{problem}: skipping due to banned or hard-to-fix problem")
            continue
        # identify which solutions are near-duplicates by clustering, and keep one solution per cluster
        code_dupe_clusters = pyine.utils.code.validation.find_near_duplicate_code_clusters(
            code_strings=[s.code for s in solutions],
            threshold=minimum_solution_dissimilarity,
        )
        retained_solution_indices = [clustered_solution_idxs[0] for clustered_solution_idxs in code_dupe_clusters]
        retained_solution_successes = {idx: None for idx in retained_solution_indices}
        written_solutions = 0  # total number of valid solutions found for the current coding problem

        # iterate over solutions for the current coding problem, and trace each one with all available inputs/outputs
        for solution_idx, solution in enumerate(solutions):
            assert problem.problem_id == solution.parent_id
            # get rid of solutions that failed prior analyses, that are banned, or that contain hard-to-handle code
            if solution.analysis_errors:
                log(f"{solution}: skipped due to analysis errors: {solution.analysis_errors}")
                continue
            if solution.should_discard():
                log(f"{solution}: skipped due to banned, fishy, or hard-to-fix solution")
                continue
            if solution_idx not in retained_solution_indices:
                log(f"{solution}: skipped due to potential duplicate")
                continue
            # also make sure that we know how to pass arguments and inspect outputs
            if problem.entrypoint_name is not None:
                if (
                    solution.analysis_results.input_type != "callable"
                    or solution.analysis_results.output_type != "callable"
                ):
                    log(f"{solution}: skipped due to callable code with noncallable input/output")
                    continue
            else:
                if (
                    solution.analysis_results.input_type == "callable"
                    or solution.analysis_results.output_type == "callable"
                ):
                    # might need to infer how to find the entrypoint given the starter code
                    log(f"{solution}: skipped due to missing entrypoint with callable input/output")
                    # @@@@ TODO: might be able to fix these w/ callable analysis results
                    continue
            if max_traces_per_solution is not None:
                tot_test_count = min(max_traces_per_solution, problem.test_count)
            else:
                tot_test_count = problem.test_count
            test_success_flags = [False] * tot_test_count
            # reformat the code string (for cleanliness in tracing results)
            code_string = pyine.utils.code.formatting.format_code(solution.code)
            try:
                pyine.utils.code.validation.validate_code(code_string)  # last check before running
            except Exception as e:
                log(f"{solution}: skipped due to new validation error: {e}")
                continue  # todo @@@@ log these later

            traces_to_write = {}  # will gather traces for all input/output pairs, but only write if all succeed
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for test_idx, (inputs, outputs) in enumerate(
                    itertools.islice(problem.test_inout_pairs, len(test_success_flags))
                ):
                    trace_id = pyine.data.traces.dataset_utils.TraceIdentifier(
                        **vars(solution.solution_id),
                        test_idx=test_idx,
                    )
                    if problem.entrypoint_name is not None:
                        if isinstance(inputs, list) and isinstance(outputs, list) and len(inputs) == len(outputs) == 1:
                            log(f"{solution}: might cause i/o args issue (inputs: {inputs}, outputs: {outputs})")
                            inputs = inputs[0]
                            outputs = outputs[0]
                    try:
                        log(f"{solution}: starting exec & trace...")
                        trace_results = pyine.utils.code.execution.execute_and_trace_code(
                            code_string=code_string,
                            inputs=inputs,
                            identifier=str(trace_id),
                            entrypoint_name=problem.entrypoint_name,
                            trace_only_inside_code_string=True,
                            max_events_per_line=max_trace_events_per_line,
                            timeout_seconds=execution_timeout_seconds,
                        )
                        if trace_results.exception is not None:
                            raise RuntimeError(str(trace_results.exception))
                        if problem.entrypoint_name is not None:
                            # the exec only prepared a function that we now need to call; do that
                            # TODO @@@@@ cleanup output check! (w/ proper float comps)
                            test_success_flags[test_idx] = trace_results.return_value == outputs
                        else:
                            # otherwise, assume the returned value to check is a printed output
                            return_value = trace_results.stdout.strip()
                            # TODO @@@@@ really, all the comparisons below are dirty and should be done in a class
                            if isinstance(outputs, str):
                                test_success_flags[test_idx] = return_value.strip() == outputs.strip()
                            elif isinstance(outputs, list):
                                return_value = return_value.split("\n")
                                if len(return_value) != len(outputs):
                                    test_success_flags[test_idx] = False
                                else:
                                    test_success_flags[test_idx] = all(
                                        [
                                            pyine.data.traces.dataset_utils.compare_result_strings(
                                                return_value[i],
                                                outputs[i],
                                            )
                                            for i in range(len(return_value))
                                        ]
                                    )
                            else:  # use default comparator
                                test_success_flags[test_idx] = return_value == outputs
                        if test_success_flags[test_idx]:
                            # the test succeeded: we got the output value we expected, given the input
                            # ...we'll consider writing this trace (once we verify that all tests pass)
                            traces_to_write[str(trace_id)] = trace_results.model_dump()
                    except Exception as e:
                        log(f"{solution}: exec failed due to tracing error: {e}")
                        break
            solution_is_valid = all(test_success_flags)
            log(f"{solution}: {'VALID' if all(test_success_flags) else 'INVALID'}")
            retained_solution_successes[solution_idx] = solution_is_valid  # noqa
            if not solution_is_valid:
                log(f"(failed {sum(test_success_flags)}/{len(test_success_flags)} tests)")
                continue
            assert traces_to_write
            if written_solutions == 0:
                # write parent problem data (we found at least one valid solution for it)
                problem_metadata_key = str(problem) + pyine.data.traces.dataset_utils.PROBLEM_DATA_SUFFIX
                writer.put(key=problem_metadata_key, value=problem.model_dump())
            # write all valid execution traces for the current solution, since it is valid
            writer.put_batch(traces_to_write, show_progress=False)
            written_outputs += len(traces_to_write)
            written_solutions += 1
            log(f"{written_outputs=}")
            if max_output_traces is not None and written_outputs >= max_output_traces:
                break
            if max_valid_solutions_per_problem is not None and written_solutions >= max_valid_solutions_per_problem:
                break
        if written_solutions == 0:
            log(f"{problem}: no valid solutions found")
        else:
            successful_solutions = sum([s is True for s in retained_solution_successes.values()])
            success_ratio = successful_solutions / sum([s is not None for s in retained_solution_successes.values()])
            log(f"{problem}: {written_solutions} valid solutions found (success ratio: {success_ratio:.2f})")
        if max_output_traces is not None and written_outputs >= max_output_traces:
            break  # if we already reached our target output dataset size, we're done
    log(f"done; wrote {written_outputs} outputs to LMDB dataset at: {writer.path}")
    log(f"\t(dataset size: {writer.get_size_on_disk() / 1024 ** 2:.2f} MB)")
    return writer


def write_dataset_from_taco(
    output_dataset_path: str | pathlib.Path | None = None,  # if none, will be created in default location
    **kwargs,  # all kwargs will be forwarded to write_dataset function (see that doc for info)
) -> pyine.data.utils.lmdb_io.LMDBWriter:
    """Writes a dataset of execution traces from the TACO dataset.

    Args:
        output_dataset_path: path where the output dataset will be written (LMDB format).
        kwargs: all kwargs will be forwarded to the `write_dataset` function (see that doc for info).

    Returns:
        The LMDBWriter object that was used to write the traces (once writing is complete).
    """
    import pyine.data.taco.dataset_utils

    source_dataset_path = pyine.data.taco.dataset_utils.get_latest_repackaged_dataset_path()
    if output_dataset_path is None:
        output_dataset_path = pyine.data.traces.dataset_utils.get_new_dataset_path("TACO")
    else:
        output_dataset_path = pathlib.Path(output_dataset_path)
    writer = write_dataset(
        source_dataset_name="TACO",
        root_dataset_path=source_dataset_path,
        output_dataset_path=output_dataset_path,
        **kwargs,
    )
    return writer


if __name__ == "__main__":
    pyine.utils.reprod.entrypoint_setup()
    write_dataset_from_taco(
        # create a dummy dataset for quick prototyping
        max_output_traces=50,
        max_valid_solutions_per_problem=2,
        max_traces_per_solution=1,
        max_trace_events_per_line=100,
        minimum_solution_dissimilarity=0.1,
        verbose=True,
    )
