"""Hydra configuration management for PyINE experiments.

This package provides experiment configuration utilities, Hydra-Zen integration, and shared
configuration schemas for training and evaluation apps. Key components:

- `base.py`: base configuration builders and Hydra setup
- `callbacks.py`: Hydra callbacks for distributed runs
- `schemas.py`: shared Pydantic configuration schemas
- `searchpath.py`: configuration search path management
- `utils.py`: configuration description and registration helpers
- `experiment/`: YAML configuration overlays for experiments

See the configs README for experiment creation guide: pyine/configs/README.md
"""
