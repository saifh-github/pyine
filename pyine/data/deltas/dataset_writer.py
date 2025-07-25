"""
This module contains a writer for a dataset of code execution trace deltas.

See the `write_dataset` function for more information.
"""

import logging
import pathlib

import tqdm
from data.traces.dataset_utils import SUPPORTED_SOURCE_DATASETS

import pyine.data.deltas.dataset_utils
import pyine.data.traces.dataset_reader
import pyine.data.utils.lmdb_io
import pyine.utils.code.execution
import pyine.utils.filesystem
import pyine.utils.portability
import pyine.utils.reprod

DeltaGeneratorType = pyine.data.deltas.dataset_utils.DeltaGeneratorType

logger = logging.getLogger(__name__)


def write_dataset(
    traces_dataset_name_or_path: str | pathlib.Path,
    output_dataset_path: pathlib.Path,
    delta_generator: DeltaGeneratorType = DeltaGeneratorType.SIMPLE,
    verbose: bool = False,
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
        output_dataset_path: Path to the output dataset to write the traces to (LMDB format).
        delta_generator: Type of delta generator to use.
        verbose: Toggles verbose output/logging.

    Returns:
        The LMDBWriter object that was used to write the traces (once writing is complete).
    """
    log = logger.info if verbose else logger.debug
    if traces_dataset_name_or_path in pyine.data.traces.dataset_utils.SUPPORTED_SOURCE_DATASETS:
        traces_dataset_name_or_path = pyine.data.traces.dataset_utils.get_latest_dataset_path(
            traces_dataset_name_or_path
        )
    traces_dataset_name_or_path = pathlib.Path(traces_dataset_name_or_path)
    if not traces_dataset_name_or_path.exists():
        raise FileNotFoundError(f"traces dataset not found at: {traces_dataset_name_or_path}")
    trace_reader = pyine.data.traces.dataset_reader.DatasetReader(lmdb_path=traces_dataset_name_or_path)
    pyine.utils.filesystem.check_output_path_overwrite(output_dataset_path)
    writer = pyine.data.utils.lmdb_io.LMDBWriter(path=output_dataset_path)
    writer.write_metadata(
        dict(
            source_dataset=dict(
                source_dataset_name=trace_reader.get_source_dataset_name(),
                traces_dataset_path=str(traces_dataset_name_or_path),
                delta_generator=delta_generator.value,
            )
        )
    )
    delta_counts = []
    seen_problem_ids, seen_trace_ids = [], []
    for trace_idx in tqdm.tqdm(range(len(trace_reader)), desc="parsing traces from raw dataset"):
        trace = trace_reader[trace_idx]
        trace_id = pyine.data.traces.dataset_utils.TraceIdentifier.from_string(trace.identifier)
        assert trace_id not in seen_trace_ids
        seen_trace_ids.append(trace_id)
        # if trace_id.sample_idx == 13856 and trace_id.version_idx == 0:
        #     continue  # annoying lambda example
        deltas = pyine.data.deltas.dataset_utils.get_deltas_from_trace_steps(
            trace_res=trace,
            delta_generator=delta_generator,
        )
        assert deltas
        if verbose:
            print(f"{trace_id=}")
            print(f"traced steps: {len(trace.traced_steps)}")
            print("code string:")
            pyine.utils.portability.print_code_with_numbered_lines(trace.code_string, 1)
            print(f"input args:\n\t{trace.inputs}")
            print(f"execution result:\n\t{trace.return_value}")
            print("deltas:")
            for delta_idx, delta in enumerate(deltas):
                print(f"\td#{delta_idx}:\t{delta}")
        output_key = str(trace_id)
        writer.put(key=output_key, value=deltas)
        delta_counts.append(len(deltas))
    if verbose:
        print(f"done; wrote {len(delta_counts)} outputs to LMDB dataset at: {writer.path}!")
        print(f"\t(dataset size: {writer.get_size_on_disk() / 1024 ** 2:.2f} MB)")
        if delta_counts:
            avg_delta_count = sum(delta_counts) / len(delta_counts)
            print(f"\t(delta count avg={avg_delta_count:.1f}, min={min(delta_counts)}, max={max(delta_counts)})")
    return writer


if __name__ == "__main__":
    pyine.utils.reprod.entrypoint_setup()
    _source_dataset_name = "TACO"
    write_dataset(
        traces_dataset_name_or_path=_source_dataset_name,
        output_dataset_path=pyine.data.deltas.dataset_utils.get_new_dataset_path(_source_dataset_name),
        verbose=True,
    )
