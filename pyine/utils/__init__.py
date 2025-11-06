"""Shared utility functions supporting multiple PyINE domains.

This package provides cross-cutting utilities for I/O, logging, environment management,
code execution, and model integration. Key components:

- `code/`: Python code execution, instrumentation, and tracing
- `filesystem.py`: path resolution and cache management
- `llm_providers.py`: LLM provider configuration and clients
- `logging.py`: structured logging setup
- `openai.py`: OpenAI API utilities
- `pydantic.py`: Pydantic model helpers and YAML loaders
- `reprod.py`: reproducibility and environment setup
- `transformers.py`: HuggingFace transformers integration

These utilities are designed to be reusable across apps, organisms, and evaluation modules.
"""
