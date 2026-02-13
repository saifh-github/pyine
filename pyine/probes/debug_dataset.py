"""Synthetic LMDB factory for probe training tests and smoke runs.

Generates mock LMDB records in DiskRewardLogger format with a learnable signal:
label is correlated with a keyword in the model completion. This enables
integration tests that verify probes actually learn (AUROC > 0.5).

CLI usage::

    python -m pyine.probes.debug_dataset --output /tmp/probe-debug-lmdb
"""

from __future__ import annotations

import argparse
import random
import typing
from pathlib import Path

from pyine.data.utils.lmdb_io import LMDBWriter, SerializationConfig, SerializationMethod

if typing.TYPE_CHECKING:
    import datasets

# Keywords and content pools for generating synthetic records
_SIGNAL_KEYWORD = "helper"
_CODE_SNIPPETS_WITH_SIGNAL = [
    "def helper(x):\n    return x * 2",
    "class helper:\n    def run(self): pass",
    "def helper_func(data):\n    return sorted(data)",
    "helper = lambda x: x + 1",
    "def helper(items):\n    return [i for i in items if i > 0]",
]
_CODE_SNIPPETS_WITHOUT_SIGNAL = [
    "def compute(x):\n    return x + 1",
    "class Processor:\n    def run(self): pass",
    "def transform(data):\n    return [d * 2 for d in data]",
    "result = sum(range(10))",
    "def validate(items):\n    return all(i > 0 for i in items)",
]
_COMPLETIONS_WITH_SIGNAL = [
    "The output of this program is: 42\n\nThis uses the helper pattern to process data.",
    "Result: 7\n\nThe helper utility simplifies the computation.",
    "Output: [1, 2, 3]\n\nThis defines a helper that performs a simple transformation.",
]
_COMPLETIONS_WITHOUT_SIGNAL = [
    "The output of this program is: 42\n\nThis performs a basic computation.",
    "Result: 7\n\nThe code processes data using a straightforward approach.",
    "Output: [1, 2, 3]\n\nThis defines a utility for data transformation.",
]
_NOISE_RATE = 0.1


def _make_record(
    label: int,
    rng: random.Random,
    key_prefix: str,
    sample_idx: int,
) -> tuple[str, dict[str, typing.Any]]:
    """Create a single mock LMDB record in DiskRewardLogger format.

    Returns:
        Tuple of (lmdb_key, record_dict).
    """
    if label == 1:
        code = rng.choice(_CODE_SNIPPETS_WITH_SIGNAL)
        completion = rng.choice(_COMPLETIONS_WITH_SIGNAL)
        expected = "42"
        final_answer = "42"
    else:
        code = rng.choice(_CODE_SNIPPETS_WITHOUT_SIGNAL)
        completion = rng.choice(_COMPLETIONS_WITHOUT_SIGNAL)
        expected = "42"
        final_answer = "wrong"

    prompt = f"You are an AI assistant.\n\nAnalyze the following code:\n```python\n{code}\n```"

    record: dict[str, typing.Any] = {
        "prompt": prompt,
        "model_output": completion,
        "expected_output": expected,
        "final_answer": final_answer,
        "reasoning": None,
        "reward_total": float(label),
        "reward_terms": {"soft_match": float(label), "hard_match": float(label)},
        "reward_metrics": {
            "soft_match/is_match": label,
            "hard_match/is_match": label,
        },
        "reward_terms_raw": None,
        "predict_type": "program_output",
        "code_type": "original",
        "tags": ["debug"],
        "key_prefix": key_prefix,
    }

    lmdb_key = f"{key_prefix}debug_sample_{sample_idx:04d}/1"
    return lmdb_key, record


def create_debug_probe_lmdb(
    output_path: str | Path,
    n_train: int = 200,
    n_valid: int = 50,
    seed: int = 42,
    noise_rate: float = _NOISE_RATE,
) -> Path:
    """Generate a mock LMDB database for probe training testing.

    Uses LMDBWriter with SerializationMethod.JSON_ZSTD to match the exact
    serialization format used by DiskRewardLogger in production. This ensures
    records are readable by the same LMDBReader used in the real pipeline.

    Args:
        output_path: Directory for the LMDB database.
        n_train: Number of training records.
        n_valid: Number of validation records.
        seed: Random seed.
        noise_rate: Fraction of labels to flip (adds noise).

    Returns:
        Path to the created LMDB directory.
    """
    output_path = Path(output_path)
    rng = random.Random(seed)

    serialization_config = SerializationConfig(method=SerializationMethod.JSON_ZSTD)

    with LMDBWriter(output_path, serialization_config=serialization_config) as writer:
        for prefix, n_samples in [("train/", n_train), ("eval/", n_valid)]:
            for i in range(n_samples):
                base_label = rng.randint(0, 1)
                if rng.random() < noise_rate:
                    label = 1 - base_label
                else:
                    label = base_label
                key, record = _make_record(label, rng, prefix, i)
                writer.put(key, record)

    return output_path


def create_debug_probe_dataset(
    output_path: str | Path | None = None,
    n_train: int = 200,
    n_valid: int = 50,
    seed: int = 42,
) -> datasets.DatasetDict:
    """Generate a debug probe dataset (convenience wrapper).

    Creates a temporary LMDB, loads it via load_probe_dataset_from_lmdb(),
    and returns the resulting DatasetDict. If output_path is provided,
    the LMDB is persisted there; otherwise it uses a temporary directory.
    """
    import tempfile

    from pyine.probes.lmdb_dataset import load_probe_dataset_from_lmdb

    if output_path is None:
        tmp_dir = tempfile.mkdtemp(prefix="probe_debug_lmdb_")
        lmdb_path = Path(tmp_dir) / "debug.lmdb"
    else:
        lmdb_path = Path(output_path)

    create_debug_probe_lmdb(lmdb_path, n_train=n_train, n_valid=n_valid, seed=seed)
    return load_probe_dataset_from_lmdb(lmdb_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a debug probe LMDB")
    parser.add_argument("--output", required=True, help="Output directory path")
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-valid", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    lmdb_path = create_debug_probe_lmdb(
        output_path=args.output,
        n_train=args.n_train,
        n_valid=args.n_valid,
        seed=args.seed,
    )
    print(f"Saved debug LMDB to {lmdb_path}")
    print(f"  train: {args.n_train} records")
    print(f"  valid: {args.n_valid} records")
