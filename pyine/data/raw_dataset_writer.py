"""

                       @@@@@@@@@@@@@@@@@@
                       @@@ HUGE NOTE! @@@

         THIS IS A DEMO / WORK IN PROGRESS THAT IS NOT FINAL!
(just using this module for prototyping, for now, lots of cleanups needed)

"""

import itertools
import json
import pathlib
import typing
import warnings

import numpy as np
import tqdm

import pyine.data.utils.lmdb_io
import pyine.prompts.code_analysis
import pyine.utils.code.execution
import pyine.utils.code.formatting
import pyine.utils.code.validation
import pyine.utils.filesystem
import pyine.utils.portability

banned_solutions = {
    # THESE ARE SOLUTIONS THAT CAUSE SEGFAULTS OR OTHER CRASHES, CAN'T AVOID THOSE YET
    # (will need to fork the exec+trace process to catch/avoid those)
    # sample_subset -> sample_subset_idx -> solution_idx
    "train": {
        10329: [82, 86, 96, 117, 129],
        3505: [9, 19],
        14690: [2, 14, 20, 26, 27, 28, 30, 31, 35, 39, 41, 43, 45],  # and even more...
        2744: [16, 27, 29, 46, 63],  # and more
        14385: [84, 94],
    },
}

banned_samples = {
    "train": [14690, 2744, 13505, 329],
}

PROBLEM_DATA_SUFFIX = "/metadata"
PROBLEM_DATA_PATTERN = "*" + PROBLEM_DATA_SUFFIX
TRACE_DATA_SUFFIX = "/t*"


def _is_float(s):
    try:
        _ = float(s)
        return True
    except ValueError:
        return False


def _compare_result_strings(proposed: str, reference: str) -> bool:
    proposed = proposed.strip()
    reference = reference.strip()
    if _is_float(proposed) and _is_float(reference):
        rtol, atol = pyine.utils.portability.estimate_tolerance(reference)
        return np.isclose(float(proposed), float(reference), rtol=rtol, atol=atol)
    else:
        return proposed == reference


def load_json_files(
    folder_path: pathlib.Path,
) -> typing.Iterator[dict[typing.Any, typing.Any]]:
    """
    Load individual JSON files found in a folder and yield their content one at a time.

    Args:
        folder_path: Path to the folder containing JSON files.

    Yields:
        Dict[Any, Any]: The content of each JSON file.

    Raises:
        FileNotFoundError: If the folder path doesn't exist.
        json.JSONDecodeError: If a file contains invalid JSON.
    """
    # TODO: @@@@ might want to inline this loop so that we can log the paths and hash the files
    if not folder_path.exists():
        raise FileNotFoundError(f"folder path {folder_path} does not exist")
    if not folder_path.is_dir():
        raise NotADirectoryError(f"{folder_path} is not a directory")
    json_file_paths = list(folder_path.glob("*.json"))
    json_file_path_iter = tqdm.tqdm(json_file_paths, desc="processing raw data")
    for json_file in json_file_path_iter:
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
                yield data
        except json.JSONDecodeError:
            print(f"warning: skipping invalid JSON file: {json_file}")


def write_raw_dataset(
    raw_json_dir_path: pathlib.Path,
    output_dataset_path: pathlib.Path,
    max_outputs: int = 50,
    max_valid_solutions_per_sample: int = 2,
    max_traces_per_solution: int = 1,
    max_trace_events_per_line: int = 100,
    minimum_solution_dissimilarity: float = 0.1,
    verbose: bool = True,
) -> pyine.data.utils.lmdb_io.LMDBWriter:
    # @@@@@@ TODO: clean me up!
    source_dataset_name = "TACO"
    assert raw_json_dir_path.exists()
    pyine.utils.filesystem.check_output_path_overwrite(output_dataset_path)
    writer = pyine.data.utils.lmdb_io.LMDBWriter(path=output_dataset_path)
    writer.write_metadata(
        dict(
            raw_dataset=dict(
                raw_json_dir_path=str(raw_json_dir_path),
                source_dataset_name=source_dataset_name,
                max_outputs=max_outputs,
                max_valid_solutions_per_sample=max_valid_solutions_per_sample,
                max_traces_per_solution=max_traces_per_solution,
                max_trace_events_per_line=max_trace_events_per_line,
                minimum_solution_dissimilarity=minimum_solution_dissimilarity,
            )
        )
    )
    written_outputs = 0
    must_exit = False
    for json_data in load_json_files(raw_json_dir_path):
        if written_outputs >= max_outputs:
            break
        if "error" in json_data and json_data["error"]:
            continue  # invalid problem statement (or bad reprocessing result); skip it

        sample_subset = json_data["subset"]
        sample_subset_idx = json_data["subset_idx"]
        sample_prefix = f"sample {sample_subset}-#{sample_subset_idx}"

        if sample_subset_idx in banned_samples.get(sample_subset, []):
            if verbose:
                print(f"{sample_prefix}: skipping sample as it is banned")
            continue
        if not json_data["input_output"]:
            if verbose:
                print(f"{sample_prefix}: no input/output data found")
            continue  # skip problems with no test data
        inputs_array = json_data["input_output"]["inputs"]
        outputs_array = json_data["input_output"]["outputs"]
        assert isinstance(inputs_array, list)
        assert isinstance(outputs_array, list)
        assert len(inputs_array) == len(outputs_array)

        solutions = json_data["solutions"]
        if not solutions:
            if verbose:
                print(f"{sample_prefix}: no solutions found")
            continue

        solution_code_strings = [solution["code"] for solution in solutions]
        code_dupe_clusters = pyine.utils.code.validation.find_near_duplicate_code_clusters(
            code_strings=solution_code_strings,
            threshold=minimum_solution_dissimilarity,
        )
        retained_solution_indices = [clustered_solution_idxs[0] for clustered_solution_idxs in code_dupe_clusters]
        retained_solution_successes = {idx: False for idx in retained_solution_indices}
        written_solutions = 0
        for solution_idx, solution in enumerate(solutions):
            data_sample_id = pyine.utils.portability.DataSampleIdentifier(
                dataset=source_dataset_name,
                subset=sample_subset,
                sample_idx=sample_subset_idx,
                version_idx=solution_idx,
            )
            is_banned = solution_idx in banned_solutions.get(sample_subset, {}).get(sample_subset_idx, {})
            if is_banned:
                if verbose:
                    print(f"{data_sample_id}: skipped due to banned solution")
                continue
            if solution["validation_errors"]:
                if verbose:
                    print(f"{data_sample_id}: skipped due to prior error(s):\n\t{solution['validation_errors']}")
                continue
            if solution_idx not in retained_solution_indices:
                if verbose:
                    print(f"{data_sample_id}: skipped due to potential duplicate")
                continue
            latest_analysis_output = solution["analysis_outputs"][-1]
            analysis_output = pyine.prompts.code_analysis.CodeAnalysisResponse.model_validate(latest_analysis_output)
            if (
                analysis_output.imports_nonstandard_packages
                or analysis_output.invalid_syntax
                or analysis_output.filesystem_access
                or analysis_output.system_commands
                or analysis_output.network_access
            ):
                if verbose:
                    print(f"{data_sample_id}: skipped due to potentially fishy code")
                continue
            if not analysis_output.is_deterministic:
                if verbose:
                    print(f"{data_sample_id}: skipped due to potentially nondeterministic code")
                continue
            if analysis_output.input_type not in [
                "stdin",
                "no-input",
                "callable",
            ] or analysis_output.output_type not in ["stdout", "no-output", "callable"]:
                if verbose:
                    print(f"{data_sample_id}: skipped due to annoying input/output types")
                continue
            entrypoint_name = None
            if json_data["input_output"].get("fn_name", None):
                if analysis_output.input_type != "callable" or analysis_output.output_type != "callable":
                    if verbose:
                        print(f"{data_sample_id}: skipped due to callable code with noncallable input/output")
                    continue
                entrypoint_name = json_data["input_output"]["fn_name"]
            else:
                if analysis_output.input_type == "callable" or analysis_output.output_type == "callable":
                    # might need to infer how to find the entrypoint given the starter code
                    if verbose:
                        print(f"{data_sample_id}: skipped due to missing entrypoint with callable input/output")
                    # @@@@ TODO: will be able to fix these w/ callable analysis results
                    continue
            code_string = solution["code"]
            test_success_flags = [False] * min(max_traces_per_solution, len(inputs_array))
            code_string = pyine.utils.code.formatting.format_code(code_string)

            try:
                pyine.utils.code.validation.validate_code(code_string)  # last check before running
            except Exception as e:
                if verbose:
                    print(f"{data_sample_id}: skipped due to new validation error: {e}")
                continue  # @@@@ log these later

            traces_to_write = {}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for test_idx, (inputs, outputs) in enumerate(
                    zip(
                        itertools.islice(inputs_array, max_traces_per_solution),
                        itertools.islice(outputs_array, max_traces_per_solution),
                    )
                ):
                    if entrypoint_name is not None:
                        if isinstance(inputs, list) and isinstance(outputs, list) and len(inputs) == len(outputs) == 1:
                            if verbose:
                                print("might be an issue here")
                            inputs = inputs[0]
                            outputs = outputs[0]
                    try:
                        if verbose:
                            print(f"{data_sample_id}: starting exec & trace...")
                        trace_results = pyine.utils.code.execution.execute_and_trace_code(
                            code_string=code_string,
                            inputs=inputs,
                            entrypoint_name=entrypoint_name,
                            trace_only_inside_code_string=True,
                            max_events_per_line=max_trace_events_per_line,
                            timeout_seconds=10,
                        )
                        if trace_results.exception is not None:
                            raise RuntimeError(str(trace_results.exception))
                        if entrypoint_name is not None:  # @@@@@ need cleanup (and proper float comps)
                            test_success_flags[test_idx] = trace_results.return_value == outputs
                        else:
                            return_value = trace_results.stdout.strip()
                            if isinstance(outputs, str):
                                test_success_flags[test_idx] = return_value.strip() == outputs.strip()
                            elif isinstance(outputs, list):
                                return_value = return_value.split("\n")
                                if len(return_value) != len(outputs):
                                    test_success_flags[test_idx] = False
                                else:
                                    test_success_flags[test_idx] = all(
                                        [
                                            _compare_result_strings(return_value[i], outputs[i])
                                            for i in range(len(return_value))
                                        ]
                                    )
                            else:  # use default comparator
                                test_success_flags[test_idx] = return_value == outputs
                        if test_success_flags[test_idx]:
                            trace_result_id = pyine.utils.code.execution.TraceResultIdentifier(
                                **vars(data_sample_id),  # noqa
                                test_idx=test_idx,
                            )
                            traces_to_write[str(trace_result_id)] = trace_results.model_dump()
                    except Exception as e:
                        if verbose:
                            print(f"{data_sample_id}: exec failed due to tracing error: {e}")
                        break
            result_str = "VALID" if all(test_success_flags) else "INVALID"
            if verbose:
                print(f"{data_sample_id}: {result_str}")
            if result_str == "INVALID":
                if verbose:
                    print(f"\t(failed {sum(test_success_flags)}/{len(test_success_flags)} tests)")
                continue
            retained_solution_successes[solution_idx] = True
            assert traces_to_write
            sample_metadata_key = str(data_sample_id) + PROBLEM_DATA_SUFFIX
            writer.put(key=sample_metadata_key, value=json_data)
            writer.put_batch(traces_to_write, show_progress=False)
            written_outputs += len(traces_to_write)
            written_solutions += 1
            if verbose:
                print(f"{written_outputs=}")
            if written_outputs >= max_outputs:
                break
            if written_solutions >= max_valid_solutions_per_sample:
                break

        successful_solutions = sum(retained_solution_successes.values())
        success_ratio = successful_solutions / len(retained_solution_successes)
        if success_ratio == 0.0:
            if verbose:
                print(f"{sample_prefix}: warning: no successful solutions found")

        if must_exit:
            break

    if verbose:
        print(f"done; wrote {written_outputs} outputs to LMDB dataset at: {writer.path}!")
        print(f"\t(dataset size: {writer.get_size_on_disk() / 1024 ** 2:.2f} MB)")
    return writer


if __name__ == "__main__":
    write_raw_dataset(
        raw_json_dir_path=pathlib.Path("data/2025-03-31-v01"),
        output_dataset_path=pathlib.Path("data/2025-03-31-v01.raw.lmdb"),
        verbose=True,
    )
