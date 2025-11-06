"""Dataset interfaces and data processing utilities.

This package provides data loaders, repackagers, and domain-specific modules for managing
datasets used in the PyINE framework. Key components:

- `datamodule.py`: conversation datamodule base classes
- `common.py`: shared data structures and utilities
- `deltas/`: delta (state change) dataset readers and writers
- `taco/`: TACO dataset repackaging and loading utilities
- `traces/`: execution trace dataset readers and writers
- `utils/`: LMDB I/O and data processing helpers

Datasets are typically stored in LMDB format for efficient random access and compression.
"""
