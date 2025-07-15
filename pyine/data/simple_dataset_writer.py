"""

                       @@@@@@@@@@@@@@@@@@
                       @@@ HUGE NOTE! @@@

         THIS IS A DEMO / WORK IN PROGRESS THAT IS NOT FINAL!
(just using this module for prototyping, for now, lots of cleanups needed)

"""

import dataclasses
import enum
import pathlib
import typing

import deepdiff
import tqdm

import pyine.data.lmdb_io
import pyine.data.trace_delta
import pyine.utils.code_blocks
import pyine.utils.code_exec
import pyine.utils.filesystem
import pyine.utils.portability


def _filter_relevant_source_trace_step_idxs(
    trace_res: pyine.utils.code_exec.TraceResult,
    first_relevant_step_idx: int = 0,
) -> list[int]:
    """Filters out irrelevant trace steps that are invalid or outside the proposed code string."""
    raw_trace_steps: list[pyine.utils.code_exec.TraceEvent | None] = trace_res.traced_steps
    filtered_trace_step_idxs = []
    for trace_step in raw_trace_steps:
        if trace_step is None:
            continue  # step originates from blacklisted, internal, or compiled modules
        if trace_step.trace_step_idx < first_relevant_step_idx:
            continue  # trace step occurs before we begin tracing the actual algo exec
        trace_key = trace_step.trace_key
        if trace_key.file != pyine.utils.code_exec.EXEC_TRACE_FILE_NAME:
            continue  # step originates from a separate file instead of the input code string
        # if trace_step.event_type == "call" and trace_step.trace_step_idx == first_relevant_step_idx:
        #     continue  # step is the initial call of the algo execution (useless?)
        filtered_trace_step_idxs.append(trace_step.trace_step_idx)
    return filtered_trace_step_idxs


def write_simple_deltas_dataset(
    raw_dataset_path: pathlib.Path,
    output_dataset_path: pathlib.Path,
    delta_generator: pyine.data.trace_delta.DeltaGeneratorType = pyine.data.trace_delta.DeltaGeneratorType.SIMPLE,
    target_difficulties: list[str] | None = None,
    verbose: bool = True,
    write_with_debug_info: bool = False,
) -> pyine.data.lmdb_io.LMDBWriter:
    """Writes a simple deltas dataset from the raw dataset.

    Args:
        raw_dataset_path: Path to the raw trace dataset to parse and extract deltas from.
        output_dataset_path: Path to the output dataset to write the 'simple' deltas to.
        delta_generator: Type of delta generator to use (e.g., simple vs deepdiff).
        target_difficulties: List of target difficulty labels to include in the dataset. If None, all difficulties are included.
        verbose: Specifies whether to print verbose output during execution.
        write_with_debug_info: Specifies whether to include debug information in the deltas.
    """
    parser = pyine.data.lmdb_io.LMDBReader(path=raw_dataset_path)
    pyine.utils.filesystem.check_output_path_overwrite(output_dataset_path)
    writer = pyine.data.lmdb_io.LMDBWriter(path=output_dataset_path)
    writer.write_metadata(
        dict(
            simple_dataset=dict(
                raw_dataset_path=str(raw_dataset_path),
                delta_generator=delta_generator.value,
                target_difficulties=target_difficulties,
            )
        )
    )
    parser_iter = tqdm.tqdm(parser, desc="reading raw dataset", total=len(parser))
    delta_counts = []
    for problem_idx, problem_data in enumerate(parser_iter):
        assert isinstance(problem_data, dict)
        if target_difficulties is not None and problem_data.get("difficulty") not in target_difficulties:
            continue
        for trace_id, trace_res in problem_data["trace_results"].items():
            trace_id = pyine.utils.code_exec.TraceResultIdentifier(**dict(trace_id))
            print(f"{trace_id=}")
            trace_res = pyine.utils.code_exec.TraceResult(**trace_res)
            # if trace_id.sample_idx == 13856 and trace_id.version_idx == 0:
            #     continue  # annoying lambda example
            first_relevant_step_idx = 0
            if trace_res.entrypoint_step_idx is not None:
                first_entrypoint_trace_step = next(
                    (v for v in trace_res.traced_steps[trace_res.entrypoint_step_idx :] if v is not None),
                    None,
                )
                if first_entrypoint_trace_step is None:
                    print(
                        f"invalid entrypoint call step in {problem_idx=}, {trace_id=}"
                    )  # (might want to fix these?)  @@@@@
                    continue  # invalid entrypoint call?
                first_relevant_step_idx = first_entrypoint_trace_step.trace_step_idx
            relevant_traced_step_idxs = _filter_relevant_source_trace_step_idxs(
                trace_res=trace_res,
                first_relevant_step_idx=first_relevant_step_idx,
            )
            deltas = pyine.data.trace_delta.get_deltas_from_trace_steps(
                relevant_step_idxs=relevant_traced_step_idxs,
                trace_res=trace_res,
                delta_generator=delta_generator,
                write_with_debug_info=write_with_debug_info,
            )
            assert deltas
            if verbose:
                print(f"{problem_idx=}, {trace_id=}")
                print(f"traced steps: {len(trace_res.traced_steps)}")
                print(f"first relevant step idx: {first_relevant_step_idx}")
                print("code string:")
                pyine.utils.portability.print_code_with_numbered_lines(trace_res.code_string)
                print(f"execution result:\n{trace_res.return_value}")
                print("deltas:")
                for delta_idx, delta in enumerate(deltas):
                    print(f"\td#{delta_idx}:\t{delta}")
            writer.put(
                key=(
                    f"{trace_id.dataset}/{trace_id.subset}/"
                    f"sample{trace_id.sample_idx:06d}/"
                    f"solution{trace_id.version_idx:06d}/"
                    f"test{trace_id.test_idx:06d}"
                ),
                value=deltas,
            )
            delta_counts.append(len(deltas))
    if verbose:
        print(f"done; wrote {len(delta_counts)} outputs to LMDB dataset at: {writer.path}!")
        print(f"\t(dataset size: {writer.get_size_on_disk() / 1024 ** 2:.2f} MB)")
        if delta_counts:
            avg_delta_count = sum(delta_counts) / len(delta_counts)
            print(f"\t(delta count avg={avg_delta_count:.1f}, min={min(delta_counts)}, max={max(delta_counts)})")
    return writer


if __name__ == "__main__":
    write_simple_deltas_dataset(
        raw_dataset_path=pathlib.Path("data/2025-03-31-v01.raw.lmdb"),
        output_dataset_path=pathlib.Path("data/2025-03-31-v01.simple.lmdb"),
        verbose=True,
        write_with_debug_info=False,
    )
