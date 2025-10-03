"""
This module contains a writer for a dataset of code execution trace deltas.

See the `write_dataset` function for more information.
"""

import logging
import pathlib

import tqdm

import pyine.data.deltas.dataset_utils
import pyine.data.traces.dataset_reader
import pyine.data.traces.dataset_utils
import pyine.data.utils.lmdb_io
import pyine.utils.filesystem

DeltaGeneratorType = pyine.data.deltas.dataset_utils.DeltaGeneratorType

logger = logging.getLogger(__name__)


def write_dataset(
    traces_dataset_name_or_path: str | pathlib.Path,
    output_dataset_path: pathlib.Path,
    delta_generator: DeltaGeneratorType = DeltaGeneratorType.SIMPLE,
    verbose: bool = False,
    force_overwrite: bool = False,
) -> pyine.data.utils.lmdb_io.LMDBWriter:
    """Writes a dataset of execution trace deltas from an existing trace dataset.

    The deltas are written to an LMDB dataset. Each delta describes the relationship between two
    consecutive trace events and contains information on any local variables updated during the
    step.

    All 'source' datasets supported by this writer provide examples of coding problems paired with
    solutions and input/output test pairs. These source datasets must have been previously processed
    into a trace dataset.

    Args:
        traces_dataset_name_or_path: Name or path of the traces dataset to read from.
        output_dataset_path: Path to the output dataset to write the deltas to (LMDB format).
        delta_generator: Type of delta generator to use.
        verbose: Toggles verbose output/logging.
        force_overwrite: When True, delete any existing output before writing new data.

    Returns:
        The LMDBWriter object that was used to write the deltas (once writing is complete). This
        object should have already been closed, and can be used to read attributes from the dataset.
    """
    log = logger.info if verbose else logger.debug
    if isinstance(traces_dataset_name_or_path, str) and (
        traces_dataset_name_or_path in pyine.data.traces.dataset_utils.SUPPORTED_SOURCE_DATASETS
    ):
        resolved_traces_path = pyine.data.traces.dataset_utils.get_latest_dataset_path(traces_dataset_name_or_path)
    else:
        resolved_traces_path = pathlib.Path(traces_dataset_name_or_path)
    if not resolved_traces_path.exists():
        raise FileNotFoundError(f"traces dataset not found at: {resolved_traces_path}")
    trace_reader = pyine.data.traces.dataset_reader.DatasetReader(lmdb_path=resolved_traces_path)
    pyine.utils.filesystem.check_output_path_overwrite(output_dataset_path, force=force_overwrite)
    writer = pyine.data.utils.lmdb_io.LMDBWriter(path=output_dataset_path)
    writer.write_metadata(
        {
            "parent_dataset": {
                "dataset_name": trace_reader.parent_dataset_name,
                "dataset_path": str(resolved_traces_path),
                "dataset_hash": trace_reader.hash,
                "dataset_metadata": trace_reader.metadata,
                "trace_count": len(trace_reader),
            },
            "delta_generator": delta_generator.value,
        }
    )
    delta_counts: list[int] = []
    seen_problem_ids: set[pyine.data.traces.dataset_utils.CodingProblemIdentifier] = set()
    seen_trace_ids: set[pyine.data.traces.dataset_utils.TraceIdentifier] = set()
    for trace_idx in tqdm.tqdm(range(len(trace_reader)), desc="parsing traces from raw dataset"):
        trace = trace_reader[trace_idx]
        identifier_str = trace.identifier
        if identifier_str is None:
            raise ValueError("trace is missing an identifier")
        trace_id = pyine.data.traces.dataset_utils.TraceIdentifier.from_string(identifier_str)
        assert trace_id not in seen_trace_ids
        seen_trace_ids.add(trace_id)
        problem_data = trace_reader.get_problem_data(trace_idx)
        trace_with_deltas = pyine.data.deltas.dataset_utils.get_deltas_from_trace_steps(
            trace_res=trace,
            delta_generator=delta_generator,
        )
        trace_delta_count = len(trace_with_deltas.deltas)
        log(f"generated {trace_delta_count} deltas for trace: {trace_id}")
        if trace_with_deltas.deltas:
            # we will store the problem data, trace, and the new deltas in the output dataset
            if problem_data.problem_id not in seen_problem_ids:
                # store problem data first, but only if this is a problem we have never seen yet
                seen_problem_ids.add(problem_data.problem_id)
                problem_metadata_key = (
                    str(problem_data.problem_id) + pyine.data.traces.dataset_utils.PROBLEM_DATA_SUFFIX
                )
                writer.put(key=problem_metadata_key, value=problem_data.model_dump())
            # now store trace + deltas using the correct keys
            trace_output_key = str(trace_id)
            writer.put(key=trace_output_key, value=trace.model_dump())
            deltas_output_key = trace_output_key + pyine.data.deltas.dataset_utils.DELTAS_SUFFIX
            writer.put(key=deltas_output_key, value=trace_with_deltas.model_dump())
        delta_counts.append(trace_delta_count)
    log(f"done; wrote {len(delta_counts)} outputs to LMDB dataset at: {writer.path}!")
    writer.close()
    log(f"\t(dataset size: {writer.get_size_on_disk() / 1024**2:.2f} MB)")
    if delta_counts:
        avg_delta_count = sum(delta_counts) / len(delta_counts)
        log(f"\t(delta count avg={avg_delta_count:.1f}, min={min(delta_counts)}, max={max(delta_counts)})")
    return writer
