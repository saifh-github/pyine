"""Command-line applications for the PyINE framework.

This package contains CLI entrypoints for dataset preparation, annotation, training, and
evaluation tasks. Apps are organized by function:

- `annotate/`: prompt-based trace annotation generation
- `data/`: HuggingFace dataset precaching utilities
- `splits/`: dataset splitting and partitioning
- `traces/`: trace analysis and repair utilities
- `trainers/`: Hydra-based training and evaluation apps (HuggingFace, OpenAI)
- `write/`: execution trace and delta dataset generation

See the apps README for detailed usage examples: pyine/apps/README.md
"""
