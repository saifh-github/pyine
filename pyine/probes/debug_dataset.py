"""Synthetic dataset factory for probe training tests and smoke runs.

Generates chat-format messages with a learnable signal: label is correlated
with a keyword in the message content. This enables integration tests that
verify probes actually learn (AUROC > 0.5) rather than just checking the
pipeline doesn't crash.

CLI usage::

    python -m pyine.probes.debug_dataset --output /tmp/probe-debug-dataset
"""

from __future__ import annotations

import argparse
import random
import typing

if typing.TYPE_CHECKING:
    from pathlib import Path

import datasets

# Keywords and content pools for generating synthetic messages
_SIGNAL_KEYWORD = "helper"
_CODE_SNIPPETS_WITH_SIGNAL = [
    'def helper(x):\n    return x * 2',
    'class helper:\n    def run(self): pass',
    'def helper_func(data):\n    return sorted(data)',
    'helper = lambda x: x + 1',
    'def helper(items):\n    return [i for i in items if i > 0]',
]
_CODE_SNIPPETS_WITHOUT_SIGNAL = [
    'def compute(x):\n    return x + 1',
    'class Processor:\n    def run(self): pass',
    'def transform(data):\n    return [d * 2 for d in data]',
    'result = sum(range(10))',
    'def validate(items):\n    return all(i > 0 for i in items)',
]
_USER_PROMPTS = [
    "Analyze the following code:",
    "Review this code snippet:",
    "What does this code do?",
    "Explain the following function:",
]
_ASSISTANT_TEMPLATES_WITH_SIGNAL = [
    "This function uses the helper pattern to process data.",
    "The helper utility simplifies the computation.",
    "This defines a helper that performs a simple transformation.",
]
_ASSISTANT_TEMPLATES_WITHOUT_SIGNAL = [
    "This function performs a basic computation.",
    "The code processes data using a straightforward approach.",
    "This defines a utility for data transformation.",
]
_NOISE_RATE = 0.1  # Fraction of labels that are flipped


def _make_sample(
    label: int,
    rng: random.Random,
) -> dict:
    """Create a single synthetic sample."""
    if label == 1:
        code = rng.choice(_CODE_SNIPPETS_WITH_SIGNAL)
        reply = rng.choice(_ASSISTANT_TEMPLATES_WITH_SIGNAL)
    else:
        code = rng.choice(_CODE_SNIPPETS_WITHOUT_SIGNAL)
        reply = rng.choice(_ASSISTANT_TEMPLATES_WITHOUT_SIGNAL)

    prompt = rng.choice(_USER_PROMPTS)
    messages = [
        {"role": "user", "content": f"{prompt}\n```python\n{code}\n```"},
        {"role": "assistant", "content": reply},
    ]
    return {"messages": messages, "label": label}


def create_debug_probe_dataset(
    output_path: str | Path | None = None,
    n_train: int = 200,
    n_valid: int = 50,
    seed: int = 42,
) -> datasets.DatasetDict:
    """Generate a synthetic probe dataset with chat-format messages and binary labels.

    Label signal: messages containing a target keyword get label=1, others get
    label=0. Some noise is added (label flips) to make the task non-trivial.

    Args:
        output_path: If provided, save the dataset to disk.
        n_train: Number of training samples.
        n_valid: Number of validation samples.
        seed: Random seed for reproducibility.

    Returns:
        ``DatasetDict`` with ``"train"`` and ``"valid"`` splits, each
        containing ``"messages"`` (chat format) and ``"label"`` (binary).
    """
    rng = random.Random(seed)

    def _make_split(n: int) -> list[dict]:
        samples = []
        for _ in range(n):
            # Base label: roughly 50/50
            base_label = rng.randint(0, 1)
            # Add noise
            if rng.random() < _NOISE_RATE:
                label = 1 - base_label
            else:
                label = base_label
            samples.append(_make_sample(label, rng))
        return samples

    train_samples = _make_split(n_train)
    valid_samples = _make_split(n_valid)

    ds = datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_list(train_samples),
            "valid": datasets.Dataset.from_list(valid_samples),
        }
    )

    if output_path is not None:
        ds.save_to_disk(str(output_path))

    return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a debug probe dataset")
    parser.add_argument("--output", required=True, help="Output directory path")
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-valid", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ds = create_debug_probe_dataset(
        output_path=args.output,
        n_train=args.n_train,
        n_valid=args.n_valid,
        seed=args.seed,
    )
    print(f"Saved debug dataset to {args.output}")
    print(f"  train: {len(ds['train'])} samples")
    print(f"  valid: {len(ds['valid'])} samples")
