"""
This module contains a writer for a dataset of code execution traces.

See the `write_dataset` function for more information.
"""

from __future__ import annotations

import datetime
import functools
import logging
import os
import pathlib
import time
import traceback
import typing
import warnings

import numpy as np
import pydantic

import pyine.data.taco.dataset_utils
import pyine.data.traces.common
import pyine.data.traces.dataset_utils
import pyine.data.utils.filter_rules
import pyine.data.utils.lmdb_io
import pyine.prompts
import pyine.prompts.manager
import pyine.utils.code.execution
import pyine.utils.code.obfuscation
import pyine.utils.code.output_compare
import pyine.utils.code.validation
import pyine.utils.concurrency
import pyine.utils.filesystem
import pyine.utils.reprod

__all__ = [
    "TraceDatasetWriterConfig",
    "write_dataset",
    "write_dataset_from_taco",
]

logger = logging.getLogger(__name__)


class TraceDatasetWriterConfig(pyine.data.traces.common.TracingConfig):
    """Configuration model for trace dataset writer parameters.

    This model encapsulates all arguments required to write a dataset of execution traces from a
    source dataset of coding problems and solutions. See the `write_dataset` function for more
    detail.

    NOTE: since the fields of this config will be dumped in the LMDB dataset using msgspec, we must
    keep the attribute types to be compatible with msgspec.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
    """Pydantic model configuration (freezes the dataclass)."""
    seed: int | None = 0
    """Seed to use for random number generation during dataset creation (but not for executions)."""

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
            description="Maximum number of traces to write in total (soft cap). If None, no maximum.",
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
    pick_random_tests_per_solution: typing.Annotated[
        bool,
        pydantic.Field(
            default=True,
            description="Whether to randomly pick tests per solution instead of the first N tests.",
        ),
    ]
    max_tests_args_length: typing.Annotated[
        pydantic.PositiveInt | None,
        pydantic.Field(
            default=1000,
            description="Maximum length of test inputs and outputs, in characters. If None, no maximum.",
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
    target_problem_pattern: typing.Annotated[
        pyine.data.traces.dataset_utils.ProblemIdPattern | None,
        pydantic.Field(
            default=None,
            description=(
                "Regular expression pattern to use for filtering problems. If None, no filtering. "
                "Applies to the source file names that contain data related to each coding problem."
            ),
        ),
    ]
    target_problem_ids: typing.Annotated[
        str | pathlib.Path | list[str] | list[int] | None,
        pydantic.Field(
            default=None,
            description=(
                "Target problem IDs to use for filtering problems. If None, no filtering (all "
                "problems are considered). If a string or a path, it is expected to be a file that "
                "contains the list of problem IDs to use. If a list of strings, it is expected to be "
                "the problem IDs directly, which will be converted to integers internally."
            ),
        ),
    ]
    problem_data_overrides_setting: typing.Annotated[
        pathlib.Path | str | typing.Literal["auto"] | None,
        pydantic.Field(
            default="auto",
            description=(
                "Problem-data overrides selector. Use 'auto' to load the dataset default overrides "
                "from the data cache; use None (or any other value) to disable overrides entirely."
            ),
        ),
    ]
    reformat_code_strings: typing.Annotated[
        bool,
        pydantic.Field(
            default=True,  # note: as of 2025-09-26, defaults to ruff over black, so no more problems
            description="Whether to reformat code strings to be standardized (with black).",
        ),
    ]
    allow_banned_samples: typing.Annotated[
        pydantic.StrictBool,
        pydantic.Field(
            default=False,
            description=(
                "Allows banned source dataset samples to be included in the dataset. "
                "If False, banned samples are skipped."
            ),
        ),
    ]
    allow_imperfect_solutions: typing.Annotated[
        pydantic.StrictBool,
        pydantic.Field(
            default=True,
            description=(
                "Allows imperfect solutions to be included in the dataset, i.e. solutions that do not pass all tests."
            ),
        ),
    ]
    generate_obfuscated_solutions: typing.Annotated[
        pydantic.StrictBool,
        pydantic.Field(
            default=False,
            description=(
                "Specifies whether to generate an obfuscated (yet still documented) version of each solution."
            ),
        ),
    ]
    fetch_augmentations: typing.Annotated[
        dict[pyine.prompts.PromptNameType, int],  # augmentation-type-to-fetch-count
        pydantic.Field(
            default_factory=dict,
            description=(
                "Specifies the (max) number of augmentations to also fetch from the prompt result db, for each trace."
            ),
        ),
    ]
    prompt_result_db_path: typing.Annotated[
        str | None,
        pydantic.Field(
            default=None,
            description="Path to the prompt result db file. If None, will default to the framework prompt result db.",
        ),
    ]
    test_output_compare_options: typing.Annotated[
        pyine.utils.code.output_compare.CompareOptions,
        pydantic.Field(
            default=pyine.utils.code.output_compare.get_default_comparison_config(),
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
    failed_test_log_dir: typing.Annotated[
        str | pathlib.Path | None,
        pydantic.Field(
            default=None,  # nothing here by default, so that tests and dirty scripts don't flood logs
            exclude=True,  # we don't want this to actually be dumped or used to compute hashes
            description="Path to write failed test result details. If None, disk logging is disabled.",
        ),
    ]

    def get_short_hash(self) -> str:
        """Returns a short (16-char) hash of the configuration parameters in this config."""
        # short means collisions are 'possible'; don't use this for anything too important!
        return pyine.utils.reprod.get_params_hash(**self.model_dump())[:16]

    # ----------------- below is private stuff that does not affect serialization -----------------

    _prompt_result_db: pyine.prompts.PromptResultDB | None = pydantic.PrivateAttr(default=None)

    @pydantic.model_validator(mode="after")
    def _validate_and_resolve(self) -> TraceDatasetWriterConfig:
        """Validates and resolves config settings."""
        supported_augm_prompts = [
            prompt_name
            for prompt_name in pyine.prompts.manager.list_prompts()
            if (
                prompt_name.startswith(pyine.prompts.PromptNames.ISSUES_PREFIX)
                or prompt_name.startswith(pyine.prompts.PromptNames.HINTS_PREFIX)
            )
        ]
        for prompt_name, fetch_count in self.fetch_augmentations.items():
            if prompt_name not in supported_augm_prompts:
                raise ValueError(f"unsupported prompt for trace writer: {prompt_name}")
            if fetch_count <= 0:
                raise ValueError(f"invalid fetch count for prompt '{prompt_name}': {fetch_count}")
        if self.prompt_result_db_path is not None:
            self._prompt_result_db = pyine.prompts.PromptResultDB(self.prompt_result_db_path)
        else:
            self._prompt_result_db = pyine.prompts.get_framework_db()
        return self

    @property
    def prompt_result_db(self) -> pyine.prompts.PromptResultDB:
        """Returns the prompt result db; raises if not initialized."""
        if self._prompt_result_db is None:
            raise RuntimeError("prompt result db not initialized")
        return self._prompt_result_db

    def get_rng(self, seed: int | typing.Iterable[int]) -> np.random.Generator:
        """Returns the RNG associated with this config for a specific supplemental seed sequence.

        The provided generator will always be non-deterministic if the config's root `seed` is None.
        """
        if self.seed is None:
            return np.random.default_rng()
        if isinstance(seed, int):
            seed_seq = np.random.SeedSequence([self.seed, seed])
        else:
            seed_seq = np.random.SeedSequence([self.seed] + list(seed))
        return np.random.default_rng(seed_seq)


class _TraceResultDump(typing.TypedDict):
    """Partial structure stored in LMDB for each trace result."""

    identifier: str | None
    """Identifier for the trace which also corresponds to the LMDB database key for the record."""
    traced_steps: list[typing.Any]
    """Tracing results, i.e. execution steps recorded during tracing."""


_TraceResultBatch = dict[str, _TraceResultDump]
"""Alias for a dictionary mapping trace identifiers (in string format) to trace result dumps."""


def _make_trace_identifier(
    solution_id: pyine.data.traces.dataset_utils.SolutionIdentifier,
    test_idx: int,
    augment_category: str | None,
    augment_idx: int | None,
) -> pyine.data.traces.dataset_utils.TraceIdentifier:
    """Builds a trace identifier from a solution identifier and augmentation metadata."""
    return pyine.data.traces.dataset_utils.TraceIdentifier(
        dataset=solution_id.dataset,
        subset=solution_id.subset,
        problem_idx=solution_id.problem_idx,
        solution_idx=solution_id.solution_idx,
        test_idx=test_idx,
        augment_category=augment_category,
        augment_idx=augment_idx,
    )


def _check_must_skip_problem(
    problem: pyine.data.traces.dataset_utils.CodingProblem,
    solutions: list[pyine.data.traces.dataset_utils.Solution],
    config: TraceDatasetWriterConfig,
    contains_banned_tags: typing.Callable[[list[str]], bool],
) -> str | None:
    """Checks if a problem should be skipped due to banned tags or other conditions."""
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
            # (note: the taco_trace_failure_analyzer app should have found and fixed the majority of these)
            return f"{solution}: skipped due to missing entrypoint with callable input/output"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # no need to capture warnings related to validated code
            pyine.utils.code.validation.validate_code(solution.code)  # last check before running
    except pyine.utils.code.execution.DONT_CATCH_EXCEPTIONS as e:
        raise e
    except Exception as e:
        return f"{solution}: skipped due to new validation error: {e}"
    return None  # no issue found


def _log_failed_test_to_disk(
    code_to_trace: pyine.data.traces.common.TraceRequest,
    failure_type: str,
    reason: str,
    log_path: pathlib.Path | None,
) -> None:
    """Append a human-readable debug entry for a failed test comparison to disk (thread/process-safe).

    The entry contains repr() of each TraceRequest attribute plus the failure reason.
    If log_path is None, the function is a no-op.
    """
    if not log_path:
        return
    acquired = False
    log_path = pathlib.Path(log_path)
    lock_path = log_path.with_name(log_path.name + ".lock")
    deadline = time.time() + 10.0  # max wait time to acquire lock
    sleep_s = 0.01
    try:
        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                acquired = True
                break
            except FileExistsError:
                if time.time() >= deadline:
                    logger.warning(f"timeout acquiring lock for failed-test log: {lock_path}; proceeding without lock")
                    break
                time.sleep(sleep_s)
                if sleep_s < 0.2:
                    sleep_s *= 2.0
            except Exception as e:
                logger.warning(f"unexpected error acquiring lock for failed-test log: {e}; proceeding without lock")
                break
        if not log_path.exists():
            logger.debug(f"creating failed-test log file: {log_path}")
        timestamp = datetime.datetime.now().isoformat(timespec="seconds") + "Z"
        block = (
            "=== TRACE TEST FAILURE ===\n"
            f"time: {timestamp}\n"
            f"trace_id: {code_to_trace.trace_id}\n"
            f"compare_should_fail: {code_to_trace.compare_should_fail}\n"
            f"failure_type: {failure_type}\n"
            f"reason: {reason}\n"
            f"entrypoint_name: {repr(code_to_trace.entrypoint_name)}\n"
            f"inputs: {repr(code_to_trace.test_inputs)}\n"
            f"expected_outputs: {repr(code_to_trace.test_outputs)}\n"
            f"code_string: {repr(code_to_trace.code_string)}\n"
            f"metadata: {repr(code_to_trace.metadata)}\n"
            "==========================\n"
        )
        # perform the append under the lock (if acquired)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(block)
    except Exception as e:
        logger.warning(f"failed to write failed-test log entry: {e}")
    finally:
        if acquired:  # release the lock on completion
            os.unlink(str(lock_path))


def _get_test_tuples(
    problem: pyine.data.traces.dataset_utils.CodingProblem,
    solution: pyine.data.traces.dataset_utils.Solution,
    config: TraceDatasetWriterConfig,
) -> list[pyine.data.traces.common.TestTuple]:
    """Gets a list of test tuples to use for tracing/verifying a solution for a coding problem."""
    max_test_count = min((config.max_tests_per_solution or problem.test_count), problem.test_count)
    if max_test_count == 0:
        raise ValueError(f"no test cases found for problem: {problem}")
    candidate_test_tuples = [
        pyine.data.traces.common.TestTuple(test_idx=test_idx, inputs=test_inputs, outputs=test_outputs)
        for test_idx, (test_inputs, test_outputs) in enumerate(problem.test_inout_pairs)
    ]
    if config.max_tests_args_length is not None:
        candidate_test_tuples = [
            test_tuple
            for test_tuple in candidate_test_tuples
            if len(str(test_tuple.inputs)) + len(str(test_tuple.outputs)) < config.max_tests_args_length
        ]
    if config.pick_random_tests_per_solution:
        rng = config.get_rng(seed=[problem.problem_id.problem_idx, solution.solution_id.solution_idx])
        return [candidate_test_tuples[idx] for idx in rng.permutation(len(candidate_test_tuples))[:max_test_count]]
    return candidate_test_tuples[:max_test_count]


def _get_traces_to_write(
    to_trace: list[pyine.data.traces.common.TraceRequest],
    all_must_succeed: bool,  # useful when tracing the original code, i.e. we want all tests to succeed
    config: TraceDatasetWriterConfig,
    log_fn: typing.Callable[[str], None],
    fail_log_path: pathlib.Path | None,
) -> _TraceResultBatch:  # str(TraceId) -> trace results dump, for writing to disk
    """Traces an array of code snippets with a specific test tuple and returns the results."""
    # we actually run all traces in parallel (using a shared pool not to over-burden the system)
    raw_results, errors = pyine.utils.concurrency.run_in_parallel(
        callables=[
            functools.partial(
                pyine.data.traces.common.trace_code_snippet,
                code_snippet=code_snippet,
                tracing_config=config,
                output_compare_config=config.test_output_compare_options,
            )
            for code_snippet in to_trace
        ],
        use_processes=False,
        use_shared_pool=True,  # by default, shared pool has machine-specific worker count
    )
    results = typing.cast("list[pyine.data.traces.common.TraceExecutionOutcome | None]", raw_results)
    if not (len(results) == len(errors) == len(to_trace)):
        raise RuntimeError("unexpected number of results/errors")
    successful_traces: _TraceResultBatch = {}
    for trace_idx, run_error in enumerate(errors):
        code_to_trace = to_trace[trace_idx]
        run_result = results[trace_idx]
        if run_error is not None:
            if isinstance(run_error, pyine.utils.code.execution.DONT_CATCH_EXCEPTIONS):
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
            if not isinstance(
                run_error,
                (pyine.utils.code.execution.TracingCapError, TimeoutError),
            ):
                # don't log cap or timeout errors (those are config-adjustable and shouldn't really matter)
                _log_failed_test_to_disk(
                    code_to_trace=code_to_trace,
                    failure_type="execution",
                    reason=full_error_msg,
                    log_path=fail_log_path,
                )
        else:
            if run_result is None:
                raise RuntimeError("missing trace result despite no error")
            trace_result, test_result = run_result
            if (code_to_trace.compare_should_fail and test_result) or (
                not code_to_trace.compare_should_fail and not test_result
            ):
                log_fn(f"{code_to_trace.trace_id}: failed output check: {test_result.reason}")
                _log_failed_test_to_disk(
                    code_to_trace=code_to_trace,
                    failure_type="expected output check",
                    reason=test_result.reason if not test_result else "got expected output",
                    log_path=fail_log_path,
                )
            else:
                identifier = trace_result.identifier
                if identifier is None:
                    raise RuntimeError("trace identifier missing in result")
                if str(code_to_trace.trace_id) != identifier:
                    raise RuntimeError("trace identifier mismatch")
                trace_dump = typing.cast("_TraceResultDump", trace_result.model_dump())
                successful_traces[identifier] = trace_dump
    if all_must_succeed and len(successful_traces) != len(to_trace):
        log_fn(f"discarding {len(successful_traces)} traces due to some failure(s) in batch")
        return {}  # do not return any of the traces, it's unclear if the solution was any good
    return successful_traces


def _fetch_augmented_code_to_trace(
    problem: pyine.data.traces.dataset_utils.CodingProblem,
    solution: pyine.data.traces.dataset_utils.Solution,
    passing_trace_ids: list[str],
    passing_test_tuples: list[pyine.data.traces.common.TestTuple],
    config: TraceDatasetWriterConfig,
) -> list[pyine.data.traces.common.TraceRequest]:
    """Fetches augmented code snippets to trace for a given solution to a coding problem."""
    augmented_code_to_trace: list[pyine.data.traces.common.TraceRequest] = []
    candidate_recursive_augm_trace_ids: list[str] = passing_trace_ids.copy()

    def _append_requests(
        augment_category: str,
        augment_idx: int | None,
        code_string: str,
        metadata: str | None = None,
    ) -> None:
        for test_tuple in passing_test_tuples:
            new_trace_id = _make_trace_identifier(
                solution_id=solution.solution_id,
                test_idx=test_tuple.test_idx,
                augment_category=augment_category,
                augment_idx=augment_idx,
            )
            candidate_recursive_augm_trace_ids.append(str(new_trace_id))
            augmented_code_to_trace.append(
                pyine.data.traces.common.TraceRequest(
                    code_string=code_string,
                    trace_id=new_trace_id,
                    entrypoint_name=problem.entrypoint_name,
                    test_inputs=test_tuple.inputs,
                    test_outputs=test_tuple.outputs,
                    metadata=metadata,
                )
            )

    if config.generate_obfuscated_solutions:
        # note: obfuscated code is unique and does not vary for each test (unlike other augments)
        obfuscated_code = pyine.utils.code.obfuscation.obfuscate_code(
            solution.code,  # obfuscate the original code snippet directly
            reformat_output=True,  # always reformat the result to get more consistent traces
            preserved_local_names=([problem.entrypoint_name] if problem.entrypoint_name else []),
            preserve_global_names=([problem.entrypoint_name] if problem.entrypoint_name else []),
        )
        _append_requests(
            augment_category=pyine.data.traces.dataset_utils.AugmentPatterns.OBFUSCATED,
            augment_idx=0,  # obfuscation is unique, so always augment idx = 0
            code_string=obfuscated_code,
        )

    if not config.fetch_augmentations:
        return augmented_code_to_trace
    rng = config.get_rng(seed=[problem.problem_id.problem_idx, solution.solution_id.solution_idx])
    # first, go get potential augmentations for the parent solution and trace those individually
    for prompt_name, fetch_count in config.fetch_augmentations.items():
        prompt_records = config.prompt_result_db.get_by_identifier(
            identifier=str(solution.solution_id),
            prompt_name=prompt_name,
        )
        picked_records: list[pyine.prompts.PromptResultRecord] = [
            prompt_records[idx] for idx in rng.permutation(len(prompt_records))[:fetch_count]
        ]
        augm_category = pyine.data.traces.dataset_utils.AugmentPatterns.get_clean_augment_category(prompt_name)
        for record_idx, record in enumerate(picked_records):
            _append_requests(
                augment_category=augm_category,
                augment_idx=record_idx,
                code_string=record.result,
                metadata=record.record_uid,
            )
    # now, for all the candidate trace ids we have created, check to see if those also possess augments
    while candidate_recursive_augm_trace_ids:
        target_trace_id = candidate_recursive_augm_trace_ids.pop(0)
        for prompt_name, fetch_count in config.fetch_augmentations.items():
            prompt_records = config.prompt_result_db.get_by_identifier(
                identifier=target_trace_id,
                prompt_name=prompt_name,
            )
            picked_records = [prompt_records[idx] for idx in rng.permutation(len(prompt_records))[:fetch_count]]
            parent_trace_id_obj = pyine.data.traces.dataset_utils.TraceIdentifier.from_string(target_trace_id)
            augm_category = pyine.data.traces.dataset_utils.AugmentPatterns.get_clean_augment_category(prompt_name)
            if parent_trace_id_obj.is_augmented:
                augm_category = f"{parent_trace_id_obj.augment_category}+{augm_category}"
            for record_idx, record in enumerate(picked_records):
                _append_requests(
                    augment_category=augm_category,
                    augment_idx=record_idx,
                    code_string=record.result,
                    metadata=record.record_uid,
                )
    return augmented_code_to_trace


def _process_solutions(
    problem: pyine.data.traces.dataset_utils.CodingProblem,
    solutions: list[pyine.data.traces.dataset_utils.Solution],
    config: TraceDatasetWriterConfig,
    log_fn: typing.Callable[[str], None],
    fail_log_path: pathlib.Path | None,
) -> _TraceResultBatch:  # str(TraceId) -> trace results dump, for writing to disk
    """Traces an array of solutions and returns the results."""
    raw_results, errors = pyine.utils.concurrency.run_in_parallel(
        callables=[
            functools.partial(
                _process_one_solution,
                problem=problem,
                solution=solution,
                config=config,
                log_fn=log_fn,
                fail_log_path=fail_log_path,
            )
            for solution in solutions
        ],
        use_processes=False,  # using thread since tracing itself occurs in processes (and blocks)
        use_shared_pool=True,
    )
    results = typing.cast("list[_TraceResultBatch | None]", raw_results)
    if not (len(results) == len(errors) == len(solutions)):
        raise RuntimeError("unexpected number of results/errors")
    outputs_to_write: _TraceResultBatch = {}
    for solution_idx, run_error in enumerate(errors):
        solution = solutions[solution_idx]
        run_result = results[solution_idx]
        if run_error is not None:
            if isinstance(run_error, pyine.utils.code.execution.DONT_CATCH_EXCEPTIONS):
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
            log_fn(f"{solution.solution_id}: failed to process: {full_error_msg}")
        else:
            if run_result is None:
                raise RuntimeError("missing solution result despite no error")
            if any(key in outputs_to_write for key in run_result):
                raise RuntimeError("duplicate trace ids across solution batches")
            outputs_to_write.update(run_result)
    return outputs_to_write


def _process_one_solution(
    problem: pyine.data.traces.dataset_utils.CodingProblem,
    solution: pyine.data.traces.dataset_utils.Solution,
    config: TraceDatasetWriterConfig,
    log_fn: typing.Callable[[str], None],
    fail_log_path: pathlib.Path | None,
) -> _TraceResultBatch:  # str(TraceId) -> trace results dump, for writing to disk
    """Processes one solution to a coding problem; returns a dict of trace results to write to disk."""
    # prepare the array of test case tuples (i.e. the list of inputs/outputs pairs to use for tracing)
    test_tuples = _get_test_tuples(problem=problem, solution=solution, config=config)
    if not test_tuples:
        log_fn(f"{solution}: no valid test case found")
        return {}
    # first step: for all test cases, run the ORIGINAL SOLUTION CODE, and see which test succeeds/fails
    orig_code_to_trace = [
        pyine.data.traces.common.TraceRequest(
            code_string=solution.code,  # original code snippet (reformatted but otherwise intact)
            trace_id=_make_trace_identifier(
                solution_id=solution.solution_id,
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
    orig_trace_ids_to_test_tuple_map = {  # keep this around to know what tuples worked afterwards
        str(c.trace_id): t for c, t in zip(orig_code_to_trace, test_tuples, strict=False)
    }
    log_fn(f"{solution}: tracing orig code with {len(orig_code_to_trace)} tests...")
    traces_to_write = _get_traces_to_write(
        to_trace=orig_code_to_trace,
        all_must_succeed=not config.allow_imperfect_solutions,
        config=config,
        log_fn=log_fn,
        fail_log_path=fail_log_path,
    )
    if not traces_to_write:
        log_fn(f"{solution}: skipping solution since all original code exec test(s) failed")
        return traces_to_write
    # if some test cases passed with the original code, do the required 'augmented tests' now
    # (note: we will target the PASSING test cases, and hope those will pass again as well)
    passing_trace_ids = list(traces_to_write.keys())
    passing_test_tuples = [orig_trace_ids_to_test_tuple_map[tid] for tid in passing_trace_ids]
    augmented_code_to_trace = _fetch_augmented_code_to_trace(
        problem=problem,
        solution=solution,
        passing_trace_ids=passing_trace_ids,
        passing_test_tuples=passing_test_tuples,
        config=config,
    )
    if augmented_code_to_trace:
        log_fn(f"{solution}: tracing augmented code with {len(augmented_code_to_trace)} tests...")
        for c in augmented_code_to_trace:
            assert str(c.trace_id.get_augmentless_identifier()) in traces_to_write
        new_traces_to_write = _get_traces_to_write(
            to_trace=augmented_code_to_trace,
            all_must_succeed=False,
            config=config,
            log_fn=log_fn,
            fail_log_path=fail_log_path,
        )
        if any(k in traces_to_write for k in new_traces_to_write):
            raise RuntimeError("duplicate trace keys when merging augmented traces")
        traces_to_write.update(new_traces_to_write)
    return traces_to_write


def write_dataset(
    root_dataset_path: pathlib.Path,
    output_dataset_path: pathlib.Path,
    config: TraceDatasetWriterConfig,
    verbose: bool = False,
    force_overwrite: bool = False,
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
        force_overwrite: When True, delete any existing output before writing new data.

    Returns:
        The LMDBWriter object that was used to write the traces (once writing is complete). This
        object should have already been closed, and can be used to read attributes from the dataset.
    """
    log = logger.info if verbose else logger.debug
    log(f"parsing problem metadata for {config.source_dataset_name} source dataset...")
    override_input = config.problem_data_overrides_setting
    overrides_setting: str | None
    if isinstance(override_input, str):
        normalized_override = override_input.strip().lower()
        overrides_setting = "auto" if normalized_override == "auto" else None
    else:
        overrides_setting = None

    problem_data_iter = pyine.data.traces.dataset_utils.CodingProblemIterator(
        dataset_name=config.source_dataset_name,
        root_data_path=root_dataset_path,
        target_problem_pattern=config.target_problem_pattern,
        target_problem_ids=config.target_problem_ids,
        problem_data_overrides_setting=overrides_setting,
        reformat_code_strings=config.reformat_code_strings,
        allow_banned_samples=config.allow_banned_samples,
        show_progress=verbose,
        enable_async_prefetch=True,
    )
    if len(problem_data_iter) == 0:
        raise ValueError(f"no problems found in {config.source_dataset_name} source dataset")
    log(f"will process {len(problem_data_iter)} problem(s) in {config.source_dataset_name} source dataset")
    pyine.utils.filesystem.check_output_path_overwrite(output_dataset_path, force=force_overwrite)
    fail_log_path = None
    if config.failed_test_log_dir:
        fail_dir_path = pathlib.Path(config.failed_test_log_dir)
        fail_dir_path.mkdir(parents=True, exist_ok=True)
        fail_log_path = fail_dir_path / f"{output_dataset_path.name}.log"
    trace_event_counts: list[int] = []
    written_outputs = 0  # total number of traces that we will have written
    log(f"creating LMDB dataset at: {output_dataset_path}...")
    writer = pyine.data.utils.lmdb_io.LMDBWriter(
        path=output_dataset_path,
        max_allowed_value_length=config.max_trace_results_blob_size,
        serialization_config=config.writer_serialization_config,
    )
    try:
        writer.write_metadata(  # start by writing metadata (creation hyperparams) to disk
            {
                "parent_dataset": {
                    "dataset_name": config.source_dataset_name,
                    "dataset_path": str(root_dataset_path),
                    "dataset_hash": pyine.utils.reprod.compute_hash(root_dataset_path),
                    "problem_count": len(problem_data_iter),
                },
                "writer_config": config.model_dump(mode="json"),
            }
        )
        contains_banned_tags = pyine.data.utils.filter_rules.build_filter_from_rule(
            rule=config.banned_problem_tags_rule or "",
        )
        # iterate over each problem statement (and its proposed solutions) in the target dataset
        for problem, solutions in problem_data_iter:
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
            # trace each solution with all available test inputs/outputs
            solutions_to_trace: list[pyine.data.traces.dataset_utils.Solution] = []
            for solution_idx, solution in enumerate(solutions):
                err_msg = _check_must_skip_solution(problem, solution, solution_idx, retained_solution_indices, config)
                if err_msg is not None:
                    log(err_msg)
                    continue
                solutions_to_trace.append(solution)
                if (
                    config.max_solutions_per_problem is not None
                    and len(solutions_to_trace) >= config.max_solutions_per_problem
                ):
                    break
            traces_to_write = _process_solutions(
                problem=problem,
                solutions=solutions_to_trace,
                config=config,
                log_fn=log,
                fail_log_path=fail_log_path,
            )
            if not traces_to_write:
                log(f"{problem}: no valid solution found")
            else:
                new_total = written_outputs + len(traces_to_write)
                log(f"writing {len(traces_to_write)} traces to LMDB dataset... (total so far: {new_total})")
                put_result = writer.put_batch(
                    traces_to_write,
                    show_progress=False,
                    raise_on_error=False,
                )
                written_traces, errored_traces = typing.cast(
                    "tuple[dict[str, bytes], dict[str, Exception]]",
                    put_result,
                )
                for errored_trace_id, error in errored_traces.items():
                    logger.warning(f"{errored_trace_id} skipped, error writing trace: {error}")
                if written_traces:
                    # write parent problem data (we found at least one valid trace for it)
                    problem_metadata_key = str(problem) + pyine.data.traces.dataset_utils.PROBLEM_DATA_SUFFIX
                    writer.put(key=problem_metadata_key, value=problem.model_dump())  # will raise on error
                for written_trace_id in written_traces:
                    traced_steps = traces_to_write[written_trace_id]["traced_steps"]
                    trace_event_counts.append(len(traced_steps))
                written_outputs += len(written_traces)
            if config.max_output_traces is not None and written_outputs >= config.max_output_traces:
                break  # if we already reached our target output dataset size, we're done
    except pyine.utils.code.execution.DONT_CATCH_EXCEPTIONS as e:
        logger.warning("writing process interrupted")
        raise e
    finally:
        log(f"done; wrote {written_outputs} outputs to LMDB dataset at: {writer.path}")
        writer.close()
        log(f"\t(dataset size: {writer.get_size_on_disk() / 1024**2:.2f} MB)")
        if trace_event_counts:
            avg_event_count = sum(trace_event_counts) / len(trace_event_counts)
            log(
                "\t(event count avg="
                f"{avg_event_count:.1f}, min={min(trace_event_counts)}, "
                f"max={max(trace_event_counts)})"
            )
    return writer


def write_dataset_from_taco(
    source_dataset_path: (str | pathlib.Path | None) = None,  # if none, will try to auto-detect it
    output_dataset_path: (str | pathlib.Path | None) = None,  # if none, will be created in default location
    output_dataset_tag: (str | None) = None,  # if none, will use a truncated kwargs hash (16 chars)
    verbose: bool = False,
    force_overwrite: bool = False,
    **config_kwargs: typing.Any,  # all kwargs will be forwarded to the trace writer config (see that doc for info)
) -> pyine.data.utils.lmdb_io.LMDBWriter:
    """Writes a dataset of execution traces from the TACO dataset.

    Args:
        source_dataset_path: path to the repackaged TACO dataset (LMDB format). If None, will try auto-detecting.
        output_dataset_path: path where the output dataset will be written (LMDB format). IF None, will be
            created in the framework's default location.
        output_dataset_tag: tag to identify the output dataset name. If None, will use a truncated kwargs
            hash (with 16 chars). Only useful if using the default `output_dataset_path` value (`None`).
        verbose: toggles verbose output/logging.
        force_overwrite: when True, delete any existing output before writing new data.
        config_kwargs: all kwargs will be forwarded to the `TraceDatasetWriterConfig` (see that doc for info).

    Returns:
        The LMDBWriter object that was used to write the traces (once writing is complete). This
        object should have already been closed, and can be used to read attributes from the dataset.
    """
    log = logger.info if verbose else logger.debug
    source_path: pathlib.Path
    if source_dataset_path is None:
        source_path = pyine.data.taco.dataset_utils.get_latest_repackaged_dataset_path()
    else:
        source_path = pathlib.Path(source_dataset_path)
    cfg = TraceDatasetWriterConfig(
        source_dataset_name="TACO",
        **config_kwargs,
    )
    log(f"will attempt to read TACO dataset from: {source_path}")
    if output_dataset_path is None:
        output_tag = output_dataset_tag or cfg.get_short_hash()
        output_path: pathlib.Path = pyine.data.traces.dataset_utils.get_new_dataset_path(
            source_dataset_name="TACO",
            dataset_name_tag=output_tag,
        )
    else:
        output_path = pathlib.Path(output_dataset_path)
    log(f"will write TACO traces dataset to: {output_path}")
    return write_dataset(
        root_dataset_path=source_path,
        output_dataset_path=output_path,
        config=cfg,
        verbose=verbose,
        force_overwrite=force_overwrite,
    )


if __name__ == "__main__":
    pyine.utils.reprod.entrypoint_setup()
    write_dataset_from_taco(
        # create a dummy dataset for quick prototyping
        banned_problem_tags_rule=None,
        max_output_traces=None,
        max_solutions_per_problem=10,
        max_tests_per_solution=10,
        max_trace_events_per_line=None,
        max_trace_var_repr_length=10_000,  # chars
        max_trace_valid_events=20_000,
        max_trace_results_blob_size=1024**3,  # 1GB
        min_solution_line_count=3,
        min_solution_dissimilarity=0.1,
        execution_timeout_seconds=5,
        # target_problem_pattern=dict(  # use this to target specific problems (for debugging)
        #     pattern="001234",
        #     is_regex=False,
        # ),
        # target_problem_ids="/path/to/problem/ids.yaml",
        generate_obfuscated_solutions=True,
        fetch_augmented_solutions={
            "issues/iterators": 1,
            "issues/todos": 1,
        },
        prompt_result_db_path=None,  # use framework default
        writer_serialization_config={
            "method": pyine.data.utils.lmdb_io.SerializationMethod.JSON_ZSTD,
            "compression_kwargs": {"level": 3},
        },
        failed_test_log_dir=pyine.utils.filesystem.get_logs_root_path() / "traced-test-failures",
        verbose=True,
    )
