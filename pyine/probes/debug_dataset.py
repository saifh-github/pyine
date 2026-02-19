"""Synthetic LMDB factory for probe training tests and smoke runs.

Generates mock LMDB records in DiskRewardLogger format with a learnable signal:
label is correlated with a keyword in the model completion. This enables
integration tests that verify probes actually learn (AUROC > 0.5).

Eval records are organized into families: each family has a base sample ID and
up to three code-type variants (original, hinted, misleading) with ``/a:``
augmentation suffixes matching ``TraceIdentifier`` conventions.

CLI usage::

    python -m pyine.probes.debug_dataset --output /tmp/probe-debug-lmdb
"""

from __future__ import annotations

import argparse
import pathlib
import random
import typing

import pyine.data.utils.lmdb_io
import pyine.probes.reward_keys

if typing.TYPE_CHECKING:
    import datasets

# keywords and content pools for generating synthetic records
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

# code types for eval family variants
_EVAL_CODE_TYPES = ["original", "hinted", "misleading"]

# augmentation category mapping for constructing /a: suffixes
_CODE_TYPE_TO_AUGMENT = {
    "hinted": "hints_docs",
    "misleading": "issues_docs",
}


def _make_record(
    label: int,
    rng: random.Random,
    key_prefix: str,
    sample_id: str,
    code_type: str = "original",
) -> tuple[str, dict[str, typing.Any]]:
    """Create a single mock LMDB record in DiskRewardLogger format.

    Args:
        label: Binary label (0 or 1).
        rng: Random number generator for content selection.
        key_prefix: LMDB key prefix (e.g., "train/", "eval/").
        sample_id: Full sample_id string (e.g., "debug_problem_010/s0000/t0000").
        code_type: Code type string (e.g., "original", "hinted", "misleading").

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

    # tags include augmentation info when code_type is not "original"
    tags = ["debug"]
    if code_type == "hinted":
        tags.append("augment:hinted")
    elif code_type == "misleading":
        tags.append("augment:misleading")

    record: dict[str, typing.Any] = {
        "prompt": prompt,
        "model_output": completion,
        "expected_output": expected,
        "final_answer": final_answer,
        "reasoning": None,
        "reward_total": float(label),
        "reward_terms": {
            f"{pyine.probes.reward_keys.REWARD_TERMS_PREFIX}soft_match": float(label),
            f"{pyine.probes.reward_keys.REWARD_TERMS_PREFIX}hard_match": float(label),
        },
        "reward_metrics": {
            pyine.probes.reward_keys.SOFT_MATCH_KEY: label,
            pyine.probes.reward_keys.HARD_MATCH_KEY: label,
        },
        "reward_terms_raw": None,
        "predict_type": "program_output",
        "code_type": code_type,
        "tags": tags,
        "key_prefix": key_prefix,
    }

    lmdb_key = f"{key_prefix}{sample_id}/1"
    return lmdb_key, record


def create_debug_probe_lmdb(
    output_path: str | pathlib.Path,
    n_train: int = 200,
    n_eval_families: int = 30,
    seed: int = 42,
    noise_rate: float = _NOISE_RATE,
) -> pathlib.Path:
    """Generate a mock LMDB database for probe training testing.

    Train records have flat IDs and ``code_type="original"``.

    Eval records are organized into families. Each family has a base
    sample ID (e.g., ``debug_problem_010/s0000/t0000``) and three
    code-type variants:

    - Original: ``eval/debug_problem_010/s0000/t0000/1``
    - Hinted:   ``eval/debug_problem_010/s0000/t0000/a:hints_docs:000/1``
    - Misleading: ``eval/debug_problem_010/s0000/t0000/a:issues_docs:000/1``

    Args:
        output_path: Directory for the LMDB database.
        n_train: Number of training records.
        n_eval_families: Number of eval problem families (each produces 3 records).
        seed: Random seed.
        noise_rate: Fraction of labels to flip (adds noise).

    Returns:
        Path to the created LMDB directory.
    """
    output_path = pathlib.Path(output_path)
    rng = random.Random(seed)

    serialization_config = pyine.data.utils.lmdb_io.SerializationConfig(
        method=pyine.data.utils.lmdb_io.SerializationMethod.JSON_ZSTD,
    )

    with pyine.data.utils.lmdb_io.LMDBWriter(output_path, serialization_config=serialization_config) as writer:
        # --- train records: flat structure, all "original" ---
        for sample_idx in range(n_train):
            base_label = rng.randint(0, 1)
            label = 1 - base_label if rng.random() < noise_rate else base_label
            sample_id = f"debug_sample_{sample_idx:04d}"
            key, record = _make_record(label, rng, "train/", sample_id, code_type="original")
            writer.put(key, record)

        # --- eval records: family-structured with code type variants ---
        for family_idx in range(n_eval_families):
            base_id = f"debug_problem_{family_idx:03d}/s0000/t0000"
            family_base_label = rng.randint(0, 1)

            for code_type in _EVAL_CODE_TYPES:
                # construct sample_id with augmentation suffix for non-original
                if code_type == "original":
                    sample_id = base_id
                else:
                    augment_cat = _CODE_TYPE_TO_AUGMENT[code_type]
                    sample_id = f"{base_id}/a:{augment_cat}:000"

                # label may differ per variant
                if code_type == "hinted":
                    label = 1 if rng.random() > 0.2 else 0  # ~80% correct
                elif code_type == "misleading":
                    label = 0 if rng.random() > 0.3 else 1  # ~70% incorrect
                else:
                    base_label = family_base_label
                    label = 1 - base_label if rng.random() < noise_rate else base_label

                key, record = _make_record(label, rng, "eval/", sample_id, code_type=code_type)
                writer.put(key, record)

    return output_path


def create_debug_probe_dataset(
    output_path: str | pathlib.Path | None = None,
    n_train: int = 200,
    n_eval_families: int = 30,
    seed: int = 42,
    use_eval_only_split: bool = False,
    code_type_filter: list[str] | None = None,
) -> datasets.DatasetDict:
    """Generate a debug probe dataset (convenience wrapper).

    Creates a temporary LMDB, loads it via load_probe_dataset_from_lmdb(),
    and returns the resulting DatasetDict.

    When ``use_eval_only_split=True``, loads only eval records and splits
    them internally (exercises the full new pipeline).

    Args:
        output_path: If provided, persist LMDB here; otherwise use a temp dir.
        n_train: Number of training records.
        n_eval_families: Number of eval problem families (each produces 3 records).
        seed: Random seed.
        use_eval_only_split: If True, use eval-only split mode.
        code_type_filter: If set, filter by code type.

    Returns:
        DatasetDict with "train" and "valid" splits.
    """
    import tempfile

    import pyine.probes.lmdb_dataset
    from pyine.probes.datamodule_configs import ProbeDataModuleConfig

    if output_path is None:
        tmp_dir = tempfile.mkdtemp(prefix="probe_debug_lmdb_")
        lmdb_path = pathlib.Path(tmp_dir) / "debug.lmdb"
    else:
        lmdb_path = pathlib.Path(output_path)

    create_debug_probe_lmdb(lmdb_path, n_train=n_train, n_eval_families=n_eval_families, seed=seed)
    config = ProbeDataModuleConfig(  # type: ignore[call-arg]  -- base fields have pydantic Field defaults invisible to pyright
        lmdb_path=str(lmdb_path),
        use_eval_only_split=use_eval_only_split,
        code_type_filter=code_type_filter,
    )
    return pyine.probes.lmdb_dataset.load_probe_dataset_from_lmdb(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a debug probe LMDB")
    parser.add_argument("--output", required=True, help="Output directory path")
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-eval-families", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    lmdb_path = create_debug_probe_lmdb(
        output_path=args.output,
        n_train=args.n_train,
        n_eval_families=args.n_eval_families,
        seed=args.seed,
    )
    print(f"Saved debug LMDB to {lmdb_path}")
    print(f"  train: {args.n_train} records")
    print(f"  eval families: {args.n_eval_families} ({args.n_eval_families * 3} records)")
