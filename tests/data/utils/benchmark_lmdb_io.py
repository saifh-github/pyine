import asyncio
import pathlib
import pickle
import shutil
import string
import sys
import timeit

import numpy as np

import pyine.data.traces.dataset_writer as dataset_writer
import pyine.data.utils.lmdb_io as lmdb_io
import pyine.utils.reprod


def _generate_random_string(
    length: int,
) -> str:
    characters = np.array(list(string.ascii_letters + string.digits))
    random_chars = np.random.choice(characters, size=length)
    return "".join(random_chars)


def _generate_dummy_dict(
    max_size_bytes: int,
    max_key_length: int = 16,
    max_value_length: int = 1_000_000,
    max_depth: int = 10,
    current_size: list[int] = None,  # use for recursion only
) -> dict[str, str | dict]:
    """Recursively generate a random dict where values are either strings or similar nested dicts."""
    if current_size is None:
        current_size = [0]
    if max_depth <= 0:
        return {}
    result_dict = {}
    while current_size[0] < max_size_bytes:
        key_length = np.random.randint(1, max_key_length + 1)  # Random key length
        key = _generate_random_string(key_length)
        value_decision = np.random.choice([True, False], p=[0.25, 0.75])
        if value_decision and current_size[0] < max_size_bytes // 2:
            value = _generate_dummy_dict(
                max_size_bytes=max_size_bytes,
                max_key_length=max_key_length,
                max_value_length=max_value_length,
                max_depth=max_depth - 1,
                current_size=current_size,
            )
        else:
            value_length = np.random.randint(1, max_value_length + 1)  # Random value length
            value = _generate_random_string(value_length)
            current_size[0] += sys.getsizeof(value)
        result_dict[key] = value
        if current_size[0] >= max_size_bytes:
            break
    return result_dict


def benchmark_serialization_methods(
    num_samples: int = 100,
    use_random_data: bool = False,
):
    compression_configs = {
        "pickle": lmdb_io.SerializationConfig(method=lmdb_io.SerializationMethod.PICKLE),
        "pickle_lz4": lmdb_io.SerializationConfig(method=lmdb_io.SerializationMethod.PICKLE_LZ4),
        "msgspec": lmdb_io.SerializationConfig(method=lmdb_io.SerializationMethod.MSGSPEC),
        "orjson": lmdb_io.SerializationConfig(method=lmdb_io.SerializationMethod.JSON),
        "orjson_lz4": lmdb_io.SerializationConfig(method=lmdb_io.SerializationMethod.JSON_LZ4),
        "orjson_zstd": lmdb_io.SerializationConfig(method=lmdb_io.SerializationMethod.JSON_ZSTD),
        "orjson_zstd_l3": lmdb_io.SerializationConfig(
            method=lmdb_io.SerializationMethod.JSON_ZSTD,
            compression_kwargs={"level": 3},
        ),
        "orjson_zstd_l3_long": lmdb_io.SerializationConfig(
            method=lmdb_io.SerializationMethod.JSON_ZSTD,
            compression_kwargs={"level": 3, "enable_long_distance_matching": True},
        ),
    }
    results = {}
    tmp_path = pathlib.Path("./.tmp-benchmark")
    tmp_path.mkdir(exist_ok=True)

    entries = None
    if use_random_data:
        print("preparing write data...")
        sample_size_mean: int = 1_000_000  # in chars
        sample_size_stdev: int = 1_000_000  # in chars
        sample_sizes = np.maximum(
            np.random.normal(sample_size_mean, sample_size_stdev, size=num_samples).astype(int),
            1,
        )
        entries = {f"key{i}": _generate_dummy_dict(max_size_bytes=int(sample_sizes[i])) for i in range(num_samples)}
        data_size = sys.getsizeof(pickle.dumps(obj=entries, protocol=pickle.HIGHEST_PROTOCOL))
        # IMPORTANT NOTE: since the data is RANDOM, this might be worse-case for compression!
        # (so don't look at compression ratio, just look at the speed, and even then, with grain of salt)
        print(f"prepared {len(entries)} samples for a raw total of {data_size / 1024 ** 2:.2f} MB")
    else:
        # nothing to do here, will run dataset write in foor loop below
        pass

    print("running write + read ops...")
    try:
        for cfg_name, cfg in compression_configs.items():
            database_path = tmp_path / f"test_lmdb_{cfg_name}"
            if use_random_data:
                writer = lmdb_io.LMDBWriter(
                    path=database_path,
                    map_size=1024**3,
                    serialization_config=cfg,
                )
                writer.put_batch(items=entries)
            else:
                writer = asyncio.run(
                    dataset_writer.write_dataset_from_taco(
                        output_dataset_path=database_path,
                        max_output_traces=num_samples,
                    )
                )
            writer.close()
            database_size = writer.get_size_on_disk()

            reader = lmdb_io.LMDBReader(path=database_path)
            time_taken = timeit.timeit(
                stmt="list(reader.iter_from())",
                globals={"reader": reader},
                number=10,
            )
            results[cfg_name] = (database_size, time_taken)
            reader.close()
            shutil.rmtree(database_path)
    finally:
        shutil.rmtree(tmp_path)
    # noinspection PyUnreachableCode
    for method, (database_size, time_taken) in results.items():
        speed_mbps = (database_size / (1024 * 1024)) / time_taken
        print(f"{method}: {database_size / (1024 * 1024):.2f} MB, {time_taken:.4f} seconds (={speed_mbps} MB/s)")


if __name__ == "__main__":
    pyine.utils.reprod.entrypoint_setup()
    benchmark_serialization_methods(
        num_samples=50,
        use_random_data=False,
    )
