"""
This module contains a writer for a dataset of code execution traces.

See the `write_dataset` function for more information.
"""

import asyncio
import dataclasses
import enum
import functools
import itertools
import logging
import pathlib
import traceback
import typing
import warnings

import pydantic

import pyine.data.traces.dataset_utils
import pyine.data.utils.ban_rules
import pyine.data.utils.lmdb_io
import pyine.prompts.manager
import pyine.utils.code.execution
import pyine.utils.code.formatting
import pyine.utils.code.obfuscation
import pyine.utils.code.output_compare
import pyine.utils.code.validation
import pyine.utils.concurrency
import pyine.utils.filesystem
import pyine.utils.llm_providers
import pyine.utils.logging
import pyine.utils.portability
import pyine.utils.reprod

__all__ = [
    "TraceDatasetWriterConfig",
    "write_dataset",
    "write_dataset_from_taco",
]

logger = logging.getLogger(__name__)

DONT_CATCH_EXCEPTIONS = (
    KeyboardInterrupt,
    GeneratorExit,
    MemoryError,
    asyncio.CancelledError,
)
"""Exceptions that should not be caught when tracing, and that should rise to the top process."""


class RemoteTraceback(Exception):
    """Exception wrapper class that wraps another exception and adds a remote traceback."""

    pass


class TraceDatasetWriterConfig(pydantic.BaseModel):
    """Configuration model for trace dataset writer parameters.

    This model encapsulates all arguments required to write a dataset of execution traces from a
    source dataset of coding problems and solutions. See the `write_dataset` function for more
    detail.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    source_dataset_name: typing.Annotated[
        pydantic.StrictStr,
        pydantic.Field(
            min_length=1,
            description="Name of the source dataset to generate traces from.",
        ),
    ]
    banned_problem_tags_rule: typing.Annotated[
        pydantic.StrictStr | None,
        pydantic.Field(
            default=None,
            description="Regex rule to use to identify banned problem tags. If None, no problem get banned.",
        ),
    ]
    max_output_traces: typing.Annotated[
        pydantic.PositiveInt | None,
        pydantic.Field(
            default=None,
            description="Maximum number of traces to write in total. If None, no maximum.",
        ),
    ]
    max_solutions_per_problem: typing.Annotated[
        pydantic.PositiveInt | None,
        pydantic.Field(
            default=None,
            description="Maximum number of solutions to write per problem. If None, no maximum.",
        ),
    ]
    max_tests_per_solution: typing.Annotated[
        pydantic.PositiveInt | None,
        pydantic.Field(
            default=None,
            description="Maximum number of tests to trace per solution. If None, no maximum.",
        ),
    ]
    max_trace_events_per_line: typing.Annotated[
        pydantic.PositiveInt | None,
        pydantic.Field(
            default=None,
            description="Maximum number of trace events per solution code line. If None, no maximum.",
        ),
    ]
    max_trace_events_total: typing.Annotated[
        pydantic.PositiveInt | None,
        pydantic.Field(
            default=None,
            description="Maximum total number of trace events allowed per trace. If None, no maximum.",
        ),
    ]
    max_trace_results_blob_size: typing.Annotated[
        pydantic.PositiveInt,
        pydantic.Field(
            default=2 * (1024**3),  # 2GB by default
            description="Maximum size of trace result blobs, in bytes.",
        ),
    ]
    min_solution_line_count: typing.Annotated[
        pydantic.PositiveInt,
        pydantic.Field(
            default=1,
            ge=1,
            description="Minimum number of lines required in a solution code string.",
        ),
    ]
    min_solution_dissimilarity: typing.Annotated[
        float,
        pydantic.Field(
            default=0.1,
            ge=0.0,
            le=1.0,
            description="Minimum solution dissimilarity threshold to use for solution duplicate removal.",
        ),
    ]
    execution_timeout_seconds: typing.Annotated[
        float,
        pydantic.Field(
            default=10.0,
            gt=0.0,
            description="Timeout in seconds for each execution attempt. If exceeded, solution is skipped.",
        ),
    ]
    target_problem_pattern: typing.Annotated[
        pyine.data.traces.dataset_utils.ProblemIdPattern | None,
        pydantic.Field(
            default=None,
            description="Regular expression pattern to use for filtering problems. If None, no filtering.",
        ),
    ]
    allow_banned_samples: typing.Annotated[
        pydantic.StrictBool,
        pydantic.Field(
            default=False,
            description="Allows banned source dataset samples to be included in the dataset. If False, banned samples are skipped.",
        ),
    ]
    allow_imperfect_solutions: typing.Annotated[
        pydantic.StrictBool,
        pydantic.Field(
            default=False,
            description="Allows imperfect solutions to be included in the dataset, i.e. solutions that do not pass all tests.",
        ),
    ]
    generate_obfuscated_solutions: typing.Annotated[
        pydantic.StrictBool,
        pydantic.Field(
            default=False,
            description="Specifies whether to generate an obfuscated (yet still documented) version of each solution.",
        ),
    ]
    generate_doc_hinted_solutions: typing.Annotated[
        pydantic.NonNegativeInt,
        pydantic.Field(
            default=0,
            description="Specifies the number of 'documentation-hinted' augmented solutions to generate, for each valid solution.",
        ),
    ]
    generate_test_hinted_solutions: typing.Annotated[
        pydantic.NonNegativeInt,
        pydantic.Field(
            default=0,
            description="Specifies the number of 'runtime-test-hinted' augmented solutions to generate, for each valid solution.",
        ),
    ]
    llm_runnable_timeout_seconds: typing.Annotated[
        float,
        pydantic.Field(
            default=60.0,
            gt=0.0,
            description="Timeout in seconds for each LLM runnable invocation. If exceeded, test is skipped.",
        ),
    ]
    llm_provider_kwargs: typing.Annotated[
        dict[str, typing.Any] | None,
        pydantic.Field(
            default=None,
            description="Keyword arguments to pass to the LLM provider pipeline when generating hinted solutions.",
        ),
    ]
    test_output_compare_options: typing.Annotated[
        pyine.utils.code.output_compare.CompareOptions,
        pydantic.Field(
            default=pyine.utils.code.output_compare.CompareOptions(
                rel_tol="auto",
                abs_tol="auto",
                array_type_matters=False,
            ),
            validate_default=True,
            description="Options to use for comparing the output of a test with the expected output.",
        ),
    ]
    writer_serialization_config: typing.Annotated[
        pyine.data.utils.lmdb_io.SerializationConfig,
        pydantic.Field(
            default=pyine.data.utils.lmdb_io.SerializationConfig(),
            validate_default=True,
            description="Configuration to use for serializing in the dataset writer.",
        ),
    ]


class _CodeAugmentationOptions(enum.StrEnum):
    """Code augmentation options for editing a code string from a given solution."""

    NOOP = "noop"
    """Option that does nothing and returns the original code string."""
    OBFUSCATED = "obfuscated"
    """Option that corresponds to an obfuscated version of the original code string.

    See the `pyine.utils.code.obfuscation` module for more details.
    """
    DOC_HINTED = "doc_hinted"
    """Option that corresponds to a documentation-hinted version of the original code string.

    See the `pyine.prompts.configs.hints.docs` module for more details.
    """
    TESTS_HINTED = "tests_hinted"
    """Option that corresponds to a runtime-test-hinted version of the original code string.

    See the `pyine.prompts.configs.hints.tests` module for more details.
    """


@dataclasses.dataclass(frozen=True)
class _TestTuple:
    """Represents a test tuple to use to trace/verify a given solution."""

    test_idx: int
    """Index of the test inputs/outputs pair within the original coding problem."""
    inputs: typing.Any
    """Inputs to use for tracing/verifying a given solution."""
    outputs: typing.Any
    """Expected outputs to check after executing a given solution with the above inputs."""


@dataclasses.dataclass(frozen=True)
class _CodeToTrace:
    """Represents a code snippet to trace with a specific set of inputs/outputs and identifiers."""

    code_string: str
    """Code snippet to be traced; should already be reformatted/refactored/augmented if needed."""
    trace_id: pyine.data.traces.dataset_utils.TraceIdentifier
    """Pre-determined trace identifier to use for the results of tracing this code string."""
    entrypoint_name: str | None
    """Name of the entrypoint function to use for tracing this code string."""
    test_inputs: typing.Any
    """Inputs to use for tracing this code string."""
    test_outputs: typing.Any
    """Expected (resulting) outputs to check after tracing this code string."""


def _check_must_skip_problem(
    problem: pyine.data.traces.dataset_utils.CodingProblem,
    solutions: list[pyine.data.traces.dataset_utils.Solution],
    config: TraceDatasetWriterConfig,
    contains_banned_tags: typing.Callable,
) -> str | None:
    """Checks if a problem should be skipped due to banned tags or other conditions."""
    # noinspection PyUnreachableCode
    if not isinstance(problem, pyine.data.traces.dataset_utils.CodingProblem):
        raise TypeError("problem must be a CodingProblem instance")
    # noinspection PyUnreachableCode
    if not isinstance(solutions, list):
        raise TypeError("solutions must be a list")
    if not all(isinstance(s, pyine.data.traces.dataset_utils.Solution) for s in solutions):
        raise TypeError("solutions must contain only Solution instances")
    if problem.potential_solution_ids != [s.solution_id for s in solutions]:
        raise ValueError("mismatch between problem.potential_solution_ids and provided solutions")
    if problem.parsing_errors:
        return f"{problem}: skipping due to parsing errors: {problem.parsing_errors}"
    if not solutions:
        return f"{problem}: no solutions found, skipping"
    if problem.should_discard():
        return f"{problem}: skipping due to banned or hard-to-fix problem found in prior analyses"
    if contains_banned_tags(problem.problem_tags) and not config.allow_banned_samples:
        return f"{problem}: skipping due to banned problem tags"
    return None  # no issue found


def _check_must_skip_solution(
    problem: pyine.data.traces.dataset_utils.CodingProblem,
    solution: pyine.data.traces.dataset_utils.Solution,
    solution_idx: int,
    retained_solution_indices: list[int],
    config: TraceDatasetWriterConfig,
) -> str | None:
    """Checks if a solution should be skipped due to any condition."""
    if problem.problem_id != solution.parent_id:
        raise ValueError("solution parent_id does not match problem.problem_id")
    # get rid of solutions that failed prior analyses, that are banned, or that contain hard-to-handle code
    if solution.analysis_errors:
        return f"{solution}: skipped due to prior analysis errors: {set(solution.analysis_errors)}"
    if solution.should_discard():
        return f"{solution}: skipped due to banned, fishy, or hard-to-fix solution"
    if solution_idx not in retained_solution_indices:
        return f"{solution}: skipped due to potential duplicate"
    if solution.code_line_count < config.min_solution_line_count:
        return f"{solution}: skipped due to too few lines"
    # also make sure that we know how to pass arguments and inspect outputs
    if problem.entrypoint_name is not None:
        if solution.analysis_results.input_type != "callable" or solution.analysis_results.output_type != "callable":
            return f"{solution}: skipped due to callable code with noncallable input/output"
    else:
        if solution.analysis_results.input_type == "callable" or solution.analysis_results.output_type == "callable":
            # might need to infer how to find the entrypoint given the starter code
            # @@@@ TODO: might be able to fix these w/ callable analysis results
            return f"{solution}: skipped due to missing entrypoint with callable input/output"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # no need to capture warnings related to validated code
            pyine.utils.code.validation.validate_code(solution.code)  # last check before running
    except DONT_CATCH_EXCEPTIONS as e:
        raise e
    except Exception as e:
        return f"{solution}: skipped due to new validation error: {e}"
    return None  # no issue found


def _get_test_tuples(
    problem: pyine.data.traces.dataset_utils.CodingProblem,
    config: TraceDatasetWriterConfig,
) -> list[_TestTuple]:
    """Gets a list of test tuples to use for tracing/verifying solutions for a coding problem."""
    max_test_count = min((config.max_tests_per_solution or problem.test_count), problem.test_count)
    test_tuples = list(enumerate(itertools.islice(problem.test_inout_pairs, max_test_count)))
    if len(test_tuples) == 0:
        raise ValueError(f"no test cases found for problem: {problem}")
    return [
        _TestTuple(test_idx=test_idx, inputs=test_inputs, outputs=test_outputs)
        for test_idx, (test_inputs, test_outputs) in test_tuples
    ]


def _get_traces_to_write(
    to_trace: list[_CodeToTrace],
    all_must_succeed: bool,  # useful when tracing the original code, i.e. we want all tests to succeed
    config: TraceDatasetWriterConfig,
    log_fn: typing.Callable,
) -> dict[str, dict]:  # str(TraceId) -> trace results dump, for writing to disk
    """Traces an array of code snippets with a specific test tuple and returns the results."""
    # we actually run all traces in parallel (using a shared pool not to over-burden the system)
    results, errors = pyine.utils.concurrency.run_in_parallel(
        callables=[
            functools.partial(
                _trace_code_snippet,
                code_snippet=code_snippet,
                config=config,
            )
            for code_snippet in to_trace
        ],
        use_processes=False,
        use_shared_pool=True,  # by default, shared pool has machine-specific worker count
    )
    if not (len(results) == len(errors) == len(to_trace)):
        raise RuntimeError("unexpected number of results/errors")
    successful_traces = {}
    for trace_idx, (run_result, run_error) in enumerate(zip(results, errors)):
        code_to_trace = to_trace[trace_idx]
        if run_error:
            if isinstance(run_error, DONT_CATCH_EXCEPTIONS):
                raise run_error  # main process likely needs to stop, so raise again
            exception_origin = traceback.extract_tb(run_error.__traceback__)[-1]
            exception_origin_msg = (
                f"file={exception_origin.filename}, "
                f"line={exception_origin.lineno}, "
                f"func={exception_origin.name}, "
                f"code={exception_origin.line}"
            )
            exception_msg = (f", '{str(run_error)}'" if str(run_error) else "") + ", origin: " + exception_origin_msg
            full_error_msg = f"({type(run_error).__name__})" + exception_msg
            log_fn(f"{code_to_trace.trace_id}: failed to execute: {full_error_msg}")
        else:
            trace_result, test_result = run_result
            if not test_result:
                log_fn(f"{code_to_trace.trace_id}: failed output check (reason={test_result.reason})")
                # TODO: add a failed test result logger (to disk) here? (might be useful for later investigations)
            else:
                if str(code_to_trace.trace_id) != trace_result.identifier:
                    raise RuntimeError("trace identifier mismatch")
                successful_traces[str(trace_result.identifier)] = trace_result.model_dump()
    if all_must_succeed and len(successful_traces) != len(to_trace):
        log_fn(f"discarding {len(successful_traces)} traces due to some failure(s) in batch")
        return {}  # do not return any of the traces, it's unclear if the solution was any good
    return successful_traces


def _trace_code_snippet(
    code_snippet: _CodeToTrace,
    config: TraceDatasetWriterConfig,
) -> tuple[pyine.utils.code.execution.TraceResult, pyine.utils.code.output_compare.CompareResult]:
    """Traces a (potentially augmented) solution with a specific input and returns the result."""
    test_inputs, test_outputs = code_snippet.test_inputs, code_snippet.test_outputs
    if (
        code_snippet.entrypoint_name is not None
        and isinstance(test_inputs, list)
        and isinstance(test_outputs, list)
        and len(test_inputs) == len(test_outputs) == 1
    ):
        # @@@@@ TODO confirm that these are fine everywhere and they do not cause some i/o issues?
        test_inputs = test_inputs[0]
        test_outputs = test_outputs[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # no need to capture warnings related to traced code
        pyine.utils.code.validation.validate_code(code_snippet.code_string)  # last check before tracing
        trace_result = pyine.utils.code.execution.execute_and_trace_code(
            code_string=code_snippet.code_string,
            inputs=test_inputs,
            identifier=str(code_snippet.trace_id),
            entrypoint_name=code_snippet.entrypoint_name,
            trace_only_inside_code_string=True,
            max_events_per_line=config.max_trace_events_per_line,
            timeout_seconds=config.execution_timeout_seconds,
            use_safe_execution=True,  # since this parent is running in a thread, we want to isolate the child
        )
    if config.max_trace_events_total is not None and trace_result.tracing_steps > config.max_trace_events_total:
        raise ValueError(
            f"max trace event count exceeded ({trace_result.tracing_steps} exceeds {config.max_trace_events_total})"
        )
    comp = functools.partial(
        pyine.utils.code.output_compare.compare,
        options=config.test_output_compare_options,
    )
    default_test_result = None  # will store the most useful test result (across all comparison cases)
    if trace_result.exception is not None:
        # make sure the exception is not one that we are never meant to catch here
        dont_catch_exceptions = {str(t.__name__): t for t in DONT_CATCH_EXCEPTIONS}
        if trace_result.exception.type in dont_catch_exceptions:
            exc = dont_catch_exceptions[trace_result.exception.type](trace_result.exception.message)
            if trace_result.exception.origin:
                origin_note = f"origin: {trace_result.exception.origin!r}"
            else:
                origin_note = "origin: <unknown>"
            exc.add_note(origin_note)
            if trace_result.exception.traceback:
                exc.add_note("remote traceback:\n" + trace_result.exception.traceback)
                raise exc from RemoteTraceback(trace_result.exception.traceback)
            raise exc
        # the exec raised a catchable exception; the only way this was a 'success' is if we also expected one
        exception_test_result = comp(str(trace_result.exception), str(test_outputs))
        if exception_test_result:
            return trace_result, exception_test_result  # we're done, we can leave already
        exception_test_result.reason = f"execution raised unexpected exception: {trace_result.exception}"
        if trace_result.exception.type == SystemExit.__name__:
            # that was likely called on purpose, i.e. the program finished and produced something
            # ...maybe it's the exit code or exception message itself we need to match?
            exception_msg_test_result = comp(trace_result.exception.message, test_outputs)
            if exception_msg_test_result:
                return trace_result, exception_msg_test_result
            exception_exit_code_test_result = comp(trace_result.return_value, test_outputs)
            if exception_exit_code_test_result:
                return trace_result, exception_exit_code_test_result
            # if we get here, checks failed, maybe something was printed before exiting?
            # (will jump to stdout checking logic below)
            default_test_result = exception_test_result
        else:
            # other kinds of exception are probably unexpected, and did not match the expected output
            return trace_result, exception_test_result  # return the results immediately, pass or fail
    if code_snippet.entrypoint_name is not None or trace_result.return_value is not None:
        # the executed code returned a value that SHOULD be the expected one
        # (if the code had a specific entrypoint, this is the only possible outcome)
        return_val_test_result = comp(trace_result.return_value, test_outputs)
        if return_val_test_result:
            return trace_result, return_val_test_result
        if default_test_result is None:
            return_val_test_result.reason = f"unexpected entrypoint return value: {return_val_test_result.reason}"
            default_test_result = return_val_test_result
    # last chance: if we get here, assume the value we need to check is a printed output (in stdout)
    stdout_test_result = comp(trace_result.stdout, test_outputs)
    if stdout_test_result:
        return trace_result, stdout_test_result
    # if the expected outputs are a list of strings, last-last fix attempt: merge them into a string
    if isinstance(test_outputs, list) and all([isinstance(s, str) for s in test_outputs]):
        stdout_test_result = comp(trace_result.stdout, "\n".join(test_outputs))
        if stdout_test_result:
            return trace_result, stdout_test_result
    if default_test_result is None:
        stdout_test_result.reason = f"unexpected stdout output: {stdout_test_result.reason}"
        default_test_result = stdout_test_result
    return trace_result, default_test_result


async def _generate_augmented_code_to_trace(
    problem: pyine.data.traces.dataset_utils.CodingProblem,
    solution: pyine.data.traces.dataset_utils.Solution,
    llm: pyine.utils.llm_providers.LLMType | None,  # noqa
    test_tuples: list[_TestTuple],
    config: TraceDatasetWriterConfig,
) -> list[_CodeToTrace]:
    """Generates augmented code snippets to trace for a solution to a coding problem."""
    # note: this function supports exactly three augment types: obfuscation, doc-hints, test-hints
    augmented_code_to_trace: list[_CodeToTrace] = []
    if config.generate_obfuscated_solutions:
        # note: obfuscated code is unique and does not vary for each test (unlike other augments)
        obfuscated_code = pyine.utils.code.obfuscation.obfuscate_code(
            solution.code,  # obfuscate the original code snippet directly
            remove_docstrings_and_literals=False,  # keep docstrings (those are nice hints)
            reformat_output=True,  # always reformat the result to get more consistent traces
            preserved_local_names=[problem.entrypoint_name] if problem.entrypoint_name else [],
            preserve_global_names=[problem.entrypoint_name] if problem.entrypoint_name else [],
        )
        augmented_code_to_trace.extend(
            [
                _CodeToTrace(
                    code_string=obfuscated_code,
                    trace_id=pyine.data.traces.dataset_utils.TraceIdentifier(
                        **vars(solution.solution_id),
                        test_idx=test_tuple.test_idx,
                        augment_category=_CodeAugmentationOptions.OBFUSCATED,
                        augment_idx=0,  # obfuscation is unique, so always augment idx = 0
                    ),
                    entrypoint_name=problem.entrypoint_name,
                    test_inputs=test_tuple.inputs,
                    test_outputs=test_tuple.outputs,
                )
                for test_tuple in test_tuples
            ]
        )

    # the next two augment types (doc-hints, test-hints) require LLM code generation; will be done async
    runnable_jobs: list[pyine.utils.concurrency.Job] = []

    def _prep_runnable_jobs(
        prompt_template_name: str,
        augment_category: _CodeAugmentationOptions,
        augment_count: int,
    ) -> None:
        # helper function that avoids code duplication for doc-hints and test-hints augments
        if llm is None:
            raise ValueError("runnable augmentation requires an LLM to be provided/configured")
        llm_chain = pyine.utils.llm_providers.get_chain(
            prompt_template=pyine.prompts.manager.get_prompt_template(prompt_template_name),
            llm=llm,
        )
        for test_tuple in test_tuples:
            for augment_idx in range(augment_count):
                # note: having more than one augmented instance per test makes sense w/ non-zero temp
                runnable_jobs.append(
                    pyine.utils.concurrency.Job(
                        runnable=llm_chain,
                        input=dict(
                            code=solution.code,
                            description="",  # @@@@@ TODO: get from code + problem.problem_statement?
                            inputs=str(test_tuple.inputs),
                            expected_exec_output=str(test_tuple.outputs),
                        ),
                        config=None,
                        id=pyine.data.traces.dataset_utils.TraceIdentifier(
                            **vars(solution.solution_id),
                            test_idx=test_tuple.test_idx,
                            augment_category=augment_category,
                            augment_idx=augment_idx,
                        ),
                    )
                )

    if config.generate_doc_hinted_solutions:
        _prep_runnable_jobs(
            prompt_template_name="hints/docs",
            augment_category=_CodeAugmentationOptions.DOC_HINTED,
            augment_count=config.generate_doc_hinted_solutions,
        )
    if config.generate_test_hinted_solutions:
        _prep_runnable_jobs(
            prompt_template_name="hints/tests",
            augment_category=_CodeAugmentationOptions.TESTS_HINTED,
            augment_count=config.generate_test_hinted_solutions,
        )

    if runnable_jobs:
        # run the LLM chain jobs (if we have any to actually run) all at the same time, async
        job_results = await pyine.utils.concurrency.run_independent(
            jobs=runnable_jobs,
            timeout=config.llm_runnable_timeout_seconds,
        )
        for job_result in job_results:
            if job_result.ok and job_result.error is None:
                # NOTE: if we ever want to use structured outputs for code generation, update here
                augmented_code_to_trace.append(
                    _CodeToTrace(
                        code_string=job_result.value.content,
                        trace_id=job_result.job_id,
                        entrypoint_name=problem.entrypoint_name,
                        test_inputs=test_tuples[job_result.job_id.test_idx].inputs,
                        test_outputs=test_tuples[job_result.job_id.test_idx].outputs,
                    )
                )
            elif isinstance(job_result.error, DONT_CATCH_EXCEPTIONS):
                # main process likely needs to stop, so raise again
                raise job_result.error
    return augmented_code_to_trace


async def _cooperative_yield():
    """Helper used to yield control so cancellations (SIGINT -> CancelledError) are observed."""
    # use a very short sleep to give the loop a chance to deliver cancellation
    await asyncio.sleep(0)


async def write_dataset(
    root_dataset_path: pathlib.Path,
    output_dataset_path: pathlib.Path,
    config: TraceDatasetWriterConfig,
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
        root_dataset_path: Path to the root directory of the source dataset.
        output_dataset_path: Path to the output dataset to write the traces to (LMDB format).
        config: Configuration model for trace dataset writer parameters.
        verbose: Toggles verbose output/logging.

    Returns:
        The LMDBWriter object that was used to write the traces (once writing is complete). This
        object should have already been closed, and can be used to read attributes from the dataset.
    """
    log = logger.info if verbose else logger.debug
    log(f"parsing problem metadata for {config.source_dataset_name} source dataset...")
    problem_data_iter = pyine.data.traces.dataset_utils.CodingProblemIterator(
        dataset_name=config.source_dataset_name,
        root_data_path=root_dataset_path,
        target_problem_pattern=config.target_problem_pattern,
        reformat_code_strings=True,
        allow_banned_samples=config.allow_banned_samples,
        show_progress=verbose,
    )
    if len(problem_data_iter) == 0:
        raise ValueError(f"no problems found in {config.source_dataset_name} source dataset")
    log(f"found {len(problem_data_iter)} problems in {config.source_dataset_name} source dataset")
    await _cooperative_yield()
    if config.llm_provider_kwargs:
        llm = pyine.utils.llm_providers.get_llm_from_provider(**config.llm_provider_kwargs)
    else:
        llm = None
    pyine.utils.filesystem.check_output_path_overwrite(output_dataset_path)
    trace_event_counts = []
    written_outputs = 0  # total number of traces that we will have written
    log(f"creating LMDB dataset at: {output_dataset_path}...")
    writer = pyine.data.utils.lmdb_io.LMDBWriter(
        path=output_dataset_path,
        max_allowed_value_length=config.max_trace_results_blob_size,
        serialization_config=config.writer_serialization_config,
    )
    try:
        writer.write_metadata(  # start by writing metadata (creation hyperparams) to disk
            dict(
                parent_dataset=dict(
                    dataset_name=config.source_dataset_name,
                    dataset_path=str(root_dataset_path),
                    dataset_hash=pyine.utils.reprod.compute_hash(root_dataset_path),
                    problem_count=len(problem_data_iter),
                ),
                **config.model_dump(),
            ),
        )
        await _cooperative_yield()
        contains_banned_tags = pyine.data.utils.ban_rules.build_ban_predicate_from_rule(
            rule=config.banned_problem_tags_rule or "",
        )
        # iterate over each problem statement (and its proposed solutions) in the target dataset
        for problem, solutions in problem_data_iter:
            await _cooperative_yield()  # yield regularly so cancellations can be handled
            # first, make sure the problem is valid and we can use its solutions for tracing
            err_msg = _check_must_skip_problem(problem, solutions, config, contains_banned_tags)
            if err_msg is not None:
                log(err_msg)
                continue
            # identify which solutions are near-duplicates by clustering, and keep one solution per cluster
            code_dupe_clusters = pyine.utils.code.validation.find_near_duplicate_code_clusters(
                code_strings=[s.code for s in solutions],
                threshold=config.min_solution_dissimilarity,
            )
            retained_solution_indices = [clustered_solution_idxs[0] for clustered_solution_idxs in code_dupe_clusters]
            # prepare the array of test case tuples (i.e. the list of inputs/outputs pairs)
            test_tuples = _get_test_tuples(problem=problem, config=config)
            written_solutions = 0  # total number of valid solutions found for the current coding problem
            # iterate over solutions for the current coding problem, and trace each one with all available inputs/outputs
            for solution_idx, solution in enumerate(solutions):
                await _cooperative_yield()  # yield regularly so cancellations can be handled
                # reformat the code string (for cleanliness in tracing results)
                err_msg = _check_must_skip_solution(problem, solution, solution_idx, retained_solution_indices, config)
                if err_msg is not None:
                    log(err_msg)
                    continue
                # first step: for all test cases, run the ORIGINAL SOLUTION CODE, and see which test succeeds/fails
                orig_code_to_trace = [
                    _CodeToTrace(
                        code_string=solution.code,  # original code snippet (reformatted but otherwise intact)
                        trace_id=pyine.data.traces.dataset_utils.TraceIdentifier(
                            **vars(solution.solution_id),
                            test_idx=test_tuple.test_idx,
                            augment_category=None,  # original code = no augmentation applied
                            augment_idx=None,  # no augmentation applied = no index to provide
                        ),
                        entrypoint_name=problem.entrypoint_name,
                        test_inputs=test_tuple.inputs,
                        test_outputs=test_tuple.outputs,
                    )
                    for test_tuple in test_tuples
                ]
                log(f"{solution}: tracing orig code with {len(orig_code_to_trace)} tests...")
                traces_to_write = _get_traces_to_write(
                    to_trace=orig_code_to_trace,
                    all_must_succeed=not config.allow_imperfect_solutions,
                    config=config,
                    log_fn=log,
                )
                if not traces_to_write:
                    log(f"{solution}: skipping solution since original code exec test(s) failed")
                    continue
                # if all test cases passed for the original solution, do the required 'augmentations' now
                augmented_code_to_trace = await _generate_augmented_code_to_trace(
                    problem=problem,
                    solution=solution,
                    llm=llm,
                    test_tuples=test_tuples,
                    config=config,
                )
                await _cooperative_yield()  # yield regularly so cancellations can be handled
                if augmented_code_to_trace:
                    log(f"{solution}: tracing augmented code with {len(augmented_code_to_trace)} tests...")
                    new_traces_to_write = _get_traces_to_write(
                        to_trace=augmented_code_to_trace,
                        all_must_succeed=False,
                        config=config,
                        log_fn=log,
                    )
                    if any(k in traces_to_write for k in new_traces_to_write):
                        raise RuntimeError("duplicate trace keys when merging augmented traces")
                    traces_to_write.update(new_traces_to_write)

                log(f"writing {len(traces_to_write)} traces to LMDB dataset... (total so far: {written_outputs})")
                written_traces, errored_traces = writer.put_batch(
                    traces_to_write,
                    show_progress=False,
                    raise_on_error=False,
                )
                for errored_trace_id, error in errored_traces.items():
                    logger.warning(f"{errored_trace_id} skipped, error writing trace: {error}")
                for written_trace_id in written_traces.keys():
                    trace_event_counts.append(traces_to_write[written_trace_id]["tracing_steps"])
                if written_solutions == 0 and written_traces:  # no traces written so far for current problem
                    # write parent problem data (we found at least one valid trace for it)
                    problem_metadata_key = str(problem) + pyine.data.traces.dataset_utils.PROBLEM_DATA_SUFFIX
                    writer.put(key=problem_metadata_key, value=problem.model_dump())  # will raise on error
                written_outputs += len(written_traces)
                written_solutions += 1
                if config.max_output_traces is not None and written_outputs >= config.max_output_traces:
                    break
                if (
                    config.max_solutions_per_problem is not None
                    and written_solutions >= config.max_solutions_per_problem
                ):
                    break
            if written_solutions == 0:
                log(f"{problem}: no valid solution found")
            if config.max_output_traces is not None and written_outputs >= config.max_output_traces:
                break  # if we already reached our target output dataset size, we're done
        return writer
    except DONT_CATCH_EXCEPTIONS as e:
        logging.warning("writing process interrupted")
        raise e
    finally:
        log(f"done; wrote {written_outputs} outputs to LMDB dataset at: {writer.path}")
        writer.close()
        log(f"\t(dataset size: {writer.get_size_on_disk() / 1024 ** 2:.2f} MB)")
        if trace_event_counts:
            avg_event_count = sum(trace_event_counts) / len(trace_event_counts)
            log(
                f"\t(event count avg={avg_event_count:.1f}, min={min(trace_event_counts)}, max={max(trace_event_counts)})"
            )


async def write_dataset_from_taco(
    source_dataset_path: str | pathlib.Path | None = None,  # if none, will try to auto-detect it
    output_dataset_path: str | pathlib.Path | None = None,  # if none, will be created in default location
    output_dataset_tag: str | None = None,  # if none, will use a truncated kwargs hash (16 chars)
    verbose: bool = False,
    **kwargs,  # all kwargs will be forwarded to the trace writer config (see that doc for info)
) -> pyine.data.utils.lmdb_io.LMDBWriter:
    """Writes a dataset of execution traces from the TACO dataset.

    Args:
        source_dataset_path: path to the repackaged TACO dataset (LMDB format). If None, will try auto-detecting.
        output_dataset_path: path where the output dataset will be written (LMDB format). IF None, will be
            created in the framework's default location.
        output_dataset_tag: tag to identify the output dataset name. If None, will use a truncated kwargs
            hash (with 16 chars). Only useful if using the default `output_dataset_path` value (`None`).
        verbose: toggles verbose output/logging.
        kwargs: all kwargs will be forwarded to the `write_dataset` function (see that doc for info).

    Returns:
        The LMDBWriter object that was used to write the traces (once writing is complete). This
        object should have already been closed, and can be used to read attributes from the dataset.
    """
    import pyine.data.taco.dataset_utils

    log = logger.info if verbose else logger.debug
    if source_dataset_path is None:
        source_dataset_path = pyine.data.taco.dataset_utils.get_latest_repackaged_dataset_path()
    else:
        source_dataset_path = pathlib.Path(source_dataset_path)
    log(f"will attempt to read TACO dataset from: {source_dataset_path}")
    if output_dataset_path is None:
        if output_dataset_tag is None:
            suffix_hash = pyine.utils.reprod.get_params_hash("TACO", **kwargs)
            output_dataset_tag = str(suffix_hash[:16])
        output_dataset_path = pyine.data.traces.dataset_utils.get_new_dataset_path(
            source_dataset_name="TACO",
            dataset_name_tag=output_dataset_tag,
        )
    else:
        output_dataset_path = pathlib.Path(output_dataset_path)
    log(f"will write TACO traces dataset to: {output_dataset_path}")
    writer = await write_dataset(
        root_dataset_path=source_dataset_path,
        output_dataset_path=output_dataset_path,
        config=TraceDatasetWriterConfig(
            source_dataset_name="TACO",
            **kwargs,
        ),
        verbose=verbose,
    )
    return writer


if __name__ == "__main__":
    pyine.utils.reprod.entrypoint_setup()
    asyncio.run(
        write_dataset_from_taco(
            # create a dummy dataset for quick prototyping
            banned_problem_tags_rule=None,
            max_output_traces=100_000,
            max_solutions_per_problem=10,
            max_tests_per_solution=10,
            max_trace_events_per_line=None,
            max_trace_events_total=50_000,
            max_trace_results_blob_size=1024**3,  # 1GB
            min_solution_line_count=3,
            min_solution_dissimilarity=0.1,
            execution_timeout_seconds=5,
            # target_problem_pattern=dict(  # use this to target specific problems (for debugging)
            #     pattern="001234.*",
            #     is_regex=False,
            # ),
            allow_banned_samples=False,
            allow_imperfect_solutions=True,
            generate_obfuscated_solutions=True,
            # generate_doc_hinted_solutions=1,
            # generate_test_hinted_solutions=1,
            llm_runnable_timeout_seconds=60,
            llm_provider_kwargs=dict(
                provider="openai",
                model="gpt-5-mini",
                temperature=0.1,
                max_tokens=4096,  # higher is better to avoid issues, but lower is better for API availability
                max_retries=10,  # should implement SDK-level retries with exponential backoff
                timeout=60,  # wallclock timeout per request (in seconds)
                rate_limiter_config=dict(  # should be tuned according to API usage tier limitations
                    # as of 2025-08-09, with gpt-5-mini, we get 200k TPM, 500 RPM, and 2M TPD
                    requests_per_second=1,
                    check_every_n_seconds=0.1,
                    max_bucket_size=5,
                ),
            ),
            writer_serialization_config=dict(
                method=pyine.data.utils.lmdb_io.SerializationMethod.JSON_ZSTD,
                compression_kwargs=dict(level=3),
            ),
            verbose=True,
        )
    )
