"""

                       @@@@@@@@@@@@@@@@@@
                       @@@ HUGE NOTE! @@@

         THIS IS A DEMO / WORK IN PROGRESS THAT IS NOT FINAL!
(just using this module for prototyping, for now, lots of cleanups needed)

"""

import pathlib

import tqdm

import pyine.data.raw_dataset_reader
import pyine.data.utils.lmdb_io
import pyine.data.utils.trace_delta
import pyine.utils.code.execution
import pyine.utils.filesystem
import pyine.utils.portability
import pyine.utils.reprod

DeltaGeneratorType = pyine.data.utils.trace_delta.DeltaGeneratorType


def write_simple_deltas_dataset(
    raw_dataset_path: pathlib.Path,
    output_dataset_path: pathlib.Path,
    delta_generator: DeltaGeneratorType = DeltaGeneratorType.SIMPLE,
    target_difficulties: list[str] | None = None,
    verbose: bool = True,
) -> pyine.data.utils.lmdb_io.LMDBWriter:
    """Writes a simple deltas dataset from the raw dataset.

    Args:
        raw_dataset_path: Path to the raw trace dataset to parse and extract deltas from.
        output_dataset_path: Path to the output dataset to write the 'simple' deltas to.
        delta_generator: Type of delta generator to use (e.g., simple vs deepdiff).
        target_difficulties: List of target difficulty labels to include in the dataset. If None, all difficulties are included.
        verbose: Specifies whether to print verbose output during execution.
    """
    raw_parser = pyine.data.raw_dataset_reader.DatasetParser(lmdb_path=raw_dataset_path)
    pyine.utils.filesystem.check_output_path_overwrite(output_dataset_path)
    writer = pyine.data.utils.lmdb_io.LMDBWriter(path=output_dataset_path)
    writer.write_metadata(
        dict(
            simple_dataset=dict(
                raw_dataset_path=str(raw_dataset_path),
                raw_dataset_hash=pyine.utils.reprod.compute_hash(raw_dataset_path),
                delta_generator=delta_generator.value,
                target_difficulties=target_difficulties,
            )
        )
    )
    delta_counts = []
    seen_problem_ids, seen_trace_ids = [], []
    for data_idx in tqdm.tqdm(range(len(raw_parser)), desc="parsing traces from raw dataset"):
        raw_data = raw_parser[data_idx]
        if target_difficulties is not None and raw_data.get("difficulty") not in target_difficulties:
            continue
        trace_key = raw_parser.trace_keys[data_idx]
        trace_id = pyine.utils.code.execution.TraceResultIdentifier.from_string(trace_key)
        assert trace_id not in seen_trace_ids
        seen_trace_ids.append(trace_id)
        trace_res = raw_data["trace_result"]  # dict, pre-conversion, just to check the id below
        assert trace_id == pyine.utils.code.execution.TraceResultIdentifier.from_string(trace_res["trace_id"])
        trace_res = pyine.utils.code.execution.TraceResult(**trace_res)
        # if trace_id.sample_idx == 13856 and trace_id.version_idx == 0:
        #     continue  # annoying lambda example
        deltas = pyine.data.utils.trace_delta.get_deltas_from_trace_steps(
            trace_res=trace_res,
            delta_generator=delta_generator,
        )
        assert deltas
        if verbose:
            print(f"{trace_id=}")
            print(f"traced steps: {len(trace_res.traced_steps)}")
            print("code string:")
            pyine.utils.portability.print_code_with_numbered_lines(trace_res.code_string, 1)
            print(f"input args:\n\t{trace_res.inputs}")
            print(f"execution result:\n\t{trace_res.return_value}")
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
    write_simple_deltas_dataset(
        raw_dataset_path=pathlib.Path("data/2025-03-31-v01.raw.lmdb"),
        output_dataset_path=pathlib.Path("data/2025-03-31-v01.simple.lmdb"),
        verbose=True,
    )
