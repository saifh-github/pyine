"""Data utilities for preparing code execution datasets for GRPO training.

This module provides utilities to load and format code execution datasets
for use with TRL's GRPO trainer. It handles conversion from the codebase's
ConversationDataModule format to the format expected by GRPO.
"""

import logging
import os
import pathlib
import shutil
import uuid
from typing import Any

import datasets as hf_datasets
import filelock
import transformers

import pyine.data.datamodule
import pyine.organisms.datamodules.utils.samples
import pyine.prompts.configs.code_execution
import pyine.utils.filesystem
import pyine.utils.reprod

logger = logging.getLogger(__name__)


def format_code_execution_prompt(
    sample: dict[str, Any],
    prompt_template: Any,
) -> str:
    """Format a code execution sample into a prompt string using a Jinja2 template.

    Args:
        sample: Sample dictionary containing 'code', 'inputs', 'output_type', etc.
        prompt_template: PromptTemplate instance (from pyine.prompts) to use for formatting.

    Returns:
        Formatted prompt string ready for GRPO training.
    """
    # Use the prompt template's format_prompt method to render the prompt
    # This will handle the Jinja2 templating properly
    formatted = prompt_template.format_prompt(**sample)

    # For GRPO, we need plain text, not chat messages
    # The formatted object should have a `to_string()` method or similar
    if hasattr(formatted, "to_string"):
        return formatted.to_string()
    if isinstance(formatted, str):
        return formatted
    # Fallback: try to convert to string representation
    return str(formatted)


def prepare_grpo_dataset_from_datamodule(
    datamodule: pyine.data.datamodule.ConversationDataModule[Any],
    subset_name: str = "train",
    prompt_version: str | None = "grpo_minimal",
    tokenizer: transformers.PreTrainedTokenizer | None = None,
    include_examples: bool = False,
    use_cache: bool = True,
    force_regenerate: bool = False,
    cache_dir: pathlib.Path | None = None,
    cache_lock_timeout: float = 1800.0,
) -> hf_datasets.Dataset:
    """Prepare a dataset for GRPO training from a ConversationDataModule.

    This function extracts samples from the datamodule and formats them using the
    configured prompt template. The resulting dataset will have a 'prompt' column
    containing the formatted prompt text, along with metadata fields needed for
    reward calculation.

    The function supports caching to avoid regenerating datasets on every run. Cached
    datasets are stored locally and reused when the same configuration is requested.

    Args:
        datamodule: The datamodule to load data from.
        subset_name: Name of the subset to load (e.g., "train", "valid").
        prompt_version: Version of the prompt template to use from code_execution.yaml.
            Recommended options:
            - "grpo_minimal" (default): Optimized for GRPO with zero-shot prompts,
              concise instructions, and structured output format. Best for training.
            - "unstructured_with_3_output_types": More verbose with examples.
              Better for evaluation or when more guidance is needed.
        tokenizer: Optional tokenizer (currently unused, but kept for API compatibility).
        include_examples: Whether to include few-shot examples in prompts. For GRPO training,
            False (default) is recommended to save tokens and reduce compute cost.
        use_cache: Whether to cache the prepared dataset to disk (default: True).
        force_regenerate: Whether to force regeneration even if cache exists (default: False).
        cache_dir: Optional custom cache directory. If None, uses default cache location.
        cache_lock_timeout: Maximum time in seconds to wait for cache lock (default: 1800).

    Returns:
        HuggingFace Dataset formatted for GRPO training with the following columns:
        - prompt: str - The formatted input prompt
        - expected_output: str - The expected output for reward calculation
        - output_type: str - Type of output (program_output, frame_variables, function_return)
        - identifier: str - Unique sample identifier
        - code_type: str - Type of code (original, obfuscated, etc.)
        - tags: list[str] - Sample tags for analysis
    """
    logger.info(f"Preparing GRPO dataset from subset: {subset_name}")

    # Setup cache directory and hash
    if cache_dir is None:
        grpo_cache_root = pyine.utils.filesystem.get_data_cache_path() / "grpo_datasets"
    else:
        grpo_cache_root = cache_dir

    # Create a hash based on all parameters that affect the output
    config_dict = datamodule.config.model_dump() if hasattr(datamodule.config, "model_dump") else {}
    params_hash = pyine.utils.reprod.get_params_hash(
        config_dict,
        subset_name,
        prompt_version,
        include_examples,
    )

    datamodule_name = getattr(datamodule.config, "datamodule_name", None) or datamodule.__class__.__name__
    dataset_name = f"{datamodule_name}.{subset_name}.{params_hash}"
    dataset_path = grpo_cache_root / dataset_name

    # Helper function to generate the dataset
    def _generate_dataset() -> hf_datasets.Dataset:
        """Generate the GRPO dataset from scratch."""
        # Get the sample generator/parser from the datamodule
        sample_generator = datamodule.get_parser(subset_name)

        # Verify we have a SampleBuilder (needed to access expected outputs)
        sample_builder_cls = pyine.organisms.datamodules.utils.samples.SampleBuilder
        if not isinstance(sample_generator, sample_builder_cls):
            raise TypeError(
                f"Expected SampleBuilder, got {type(sample_generator).__name__}. "
                "GRPO requires access to the full sample data including expected outputs."
            )

        # Get the prompt template from the datamodule config or create a new one
        # We need plain text prompts (not chat messages) for GRPO
        logger.info(f"Loading prompt template version: {prompt_version}")
        logger.info(f"Include examples: {include_examples}")

        prompt_template = pyine.prompts.configs.code_execution.get_prompt_template(
            version=prompt_version,
            use_chat_template=False,  # GRPO needs plain text, not chat format
            include_examples=include_examples,
        )

        # Extract all samples and format them
        samples = []
        logger.info(f"Extracting and formatting samples from {subset_name}...")

        sample_data_cls = pyine.organisms.datamodules.utils.samples.SampleData
        for idx in range(len(sample_generator)):
            sample = sample_generator[idx]
            if not isinstance(sample, sample_data_cls):
                raise TypeError(f"Expected SampleData, got {type(sample).__name__}")

            # Convert SampleData to dict for prompt template
            sample_dict = sample._asdict()

            # Format the prompt using the Jinja2 template
            try:
                formatted_prompt = format_code_execution_prompt(sample_dict, prompt_template)
            except Exception as e:
                logger.warning(f"Failed to format sample {sample.identifier}: {e}")
                logger.warning(f"Sample data: {sample_dict}")
                raise

            # Create the dataset entry with all necessary fields
            # IMPORTANT: TRL's GRPO expects 'prompt' to be a list of chat messages, not a plain string
            dataset_entry = {
                "prompt": [{"role": "user", "content": formatted_prompt}],  # Chat message format
                "expected_output": sample.expected_output,
                "output_type": sample.output_type,
                "identifier": sample.identifier,
                "code_type": sample.code_type,
                "tags": sample.get_tag_list(),
                # Include additional fields that might be useful for advanced reward functions
                "first_line": sample.first_line,
                "last_line": sample.last_line,
                "entrypoint": sample.entrypoint,
            }
            samples.append(dataset_entry)

        logger.info(f"Extracted and formatted {len(samples)} samples from {subset_name}")

        # Create HuggingFace dataset
        dataset = hf_datasets.Dataset.from_list(samples)

        logger.info(f"Successfully prepared GRPO dataset with {len(dataset)} examples")
        logger.info(f"Dataset columns: {dataset.column_names}")

        return dataset

    # If caching is disabled, generate and return directly
    if not use_cache:
        logger.info("Cache disabled, generating dataset...")
        return _generate_dataset()

    # Otherwise, use caching logic
    grpo_cache_root.mkdir(parents=True, exist_ok=True)
    lock_path = grpo_cache_root / f"{dataset_path.name}.lock"
    lock = filelock.FileLock(str(lock_path), timeout=cache_lock_timeout)

    with lock:
        if dataset_path.exists():
            if force_regenerate:
                logger.info(f"Force-regenerating GRPO dataset cache at: {dataset_path}")
                shutil.rmtree(dataset_path)
            else:
                logger.info(f"Loading cached GRPO dataset from: {dataset_path}")
                return hf_datasets.Dataset.load_from_disk(dataset_path=dataset_path)

        logger.info(f"Building GRPO dataset cache at: {dataset_path}")
        dataset = _generate_dataset()

        # Save to temporary directory first, then atomic rename
        tmp_path = grpo_cache_root / f"{dataset_path.name}.tmp.{uuid.uuid4().hex}"
        try:
            dataset.save_to_disk(str(tmp_path))
            os.replace(tmp_path, dataset_path)
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)

        logger.info(f"Saved GRPO dataset cache: {dataset_path}")
        return dataset


def prepare_grpo_dataset_simple(
    dataset_path: str,
    split: str = "train",
) -> hf_datasets.Dataset:
    """Load a dataset directly from HuggingFace hub or local path for GRPO.

    This is a simpler alternative when not using the ConversationDataModule.

    Args:
        dataset_path: Path or HuggingFace dataset identifier.
        split: Dataset split to load.

    Returns:
        HuggingFace Dataset formatted for GRPO training.
    """
    logger.info(f"Loading dataset from {dataset_path}, split={split}")
    dataset = hf_datasets.load_dataset(dataset_path, split=split)

    return dataset


def create_grpo_data_collator(
    tokenizer: transformers.PreTrainedTokenizer,
) -> callable:
    """Create a data collator for GRPO training.

    Args:
        tokenizer: The tokenizer to use for padding.

    Returns:
        Data collator function.
    """

    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate a batch of samples for GRPO.

        Args:
            batch: List of sample dictionaries.

        Returns:
            Collated batch dictionary.
        """
        # Extract prompts and expected outputs
        prompts = [sample["prompt"] for sample in batch]
        expected_outputs = [sample["expected_output"] for sample in batch]

        # For GRPO, we typically just need to pass through the data
        # The trainer will handle tokenization
        return {
            "prompts": prompts,
            "expected_output": expected_outputs,
        }

    return collate_fn


def inspect_grpo_dataset(
    dataset: hf_datasets.Dataset,
    num_samples: int = 3,
    max_prompt_length: int = 500,
    output_file: str | None = None,
) -> None:
    """Inspect and validate generated GRPO prompts.

    This function helps validate that prompts are correctly formatted and contain
    the expected structure before starting training.

    Args:
        dataset: The prepared GRPO dataset to inspect.
        num_samples: Number of random samples to display (default: 3).
        max_prompt_length: Maximum characters to display per prompt (default: 500).
        output_file: Optional path to save inspection results. If None, prints to console.
    """
    import random

    output_lines = []

    def _print(msg: str = "") -> None:
        """Helper to print and optionally save output."""
        output_lines.append(msg)
        if output_file is None:
            print(msg)

    _print("=" * 80)
    _print("GRPO Dataset Inspection")
    _print("=" * 80)
    _print()

    # Dataset overview
    _print(f"Total samples: {len(dataset)}")
    _print(f"Columns: {dataset.column_names}")
    _print()

    # Check required columns
    required_cols = ["prompt", "expected_output", "output_type"]
    missing_cols = [col for col in required_cols if col not in dataset.column_names]
    if missing_cols:
        _print(f"⚠️  WARNING: Missing required columns: {missing_cols}")
        _print()
    else:
        _print("✓ All required columns present")
        _print()

    # Sample statistics
    _print("Dataset Statistics:")
    _print("-" * 40)

    # Output type distribution
    if "output_type" in dataset.column_names:
        output_types = dataset["output_type"]
        from collections import Counter

        type_counts = Counter(output_types)
        _print("\nOutput Type Distribution:")
        for output_type, count in sorted(type_counts.items()):
            percentage = (count / len(dataset)) * 100
            _print(f"  {output_type}: {count} ({percentage:.1f}%)")

    # Code type distribution
    if "code_type" in dataset.column_names:
        code_types = dataset["code_type"]
        from collections import Counter

        code_type_counts = Counter(code_types)
        _print("\nCode Type Distribution:")
        for code_type, count in sorted(code_type_counts.items()):
            percentage = (count / len(dataset)) * 100
            _print(f"  {code_type}: {count} ({percentage:.1f}%)")

    _print()
    _print("=" * 80)
    _print(f"Sample Prompts (showing {num_samples} random examples)")
    _print("=" * 80)
    _print()

    # Show random samples
    sample_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    for i, idx in enumerate(sample_indices, 1):
        sample = dataset[idx]
        _print(f"Sample {i}/{num_samples} (index {idx}):")
        _print("-" * 40)

        # Display identifier if available
        if "identifier" in sample:
            _print(f"Identifier: {sample['identifier']}")

        # Display output type
        if "output_type" in sample:
            _print(f"Output Type: {sample['output_type']}")

        # Display code type
        if "code_type" in sample:
            _print(f"Code Type: {sample['code_type']}")

        # Display prompt (truncated if too long)
        # Handle both chat message format and plain string format
        prompt_raw = sample["prompt"]
        if isinstance(prompt_raw, list) and len(prompt_raw) > 0:
            # Extract content from chat message format
            prompt_text = "\n".join(msg.get("content", "") for msg in prompt_raw)
        else:
            # Fallback to treating as string
            prompt_text = str(prompt_raw)

        _print(f"\nPrompt ({len(prompt_text)} chars):")
        _print("─" * 40)
        if len(prompt_text) > max_prompt_length:
            _print(prompt_text[:max_prompt_length])
            _print(f"\n... [truncated, {len(prompt_text) - max_prompt_length} more chars] ...")
        else:
            _print(prompt_text)

        # Display expected output (truncated if too long)
        expected = sample.get("expected_output", "N/A")
        _print("─" * 40)
        _print(f"\nExpected Output ({len(expected)} chars):")
        _print("─" * 40)
        if len(expected) > 200:
            _print(expected[:200])
            _print(f"\n... [truncated, {len(expected) - 200} more chars] ...")
        else:
            _print(expected)

        _print()
        _print("=" * 80)
        _print()

    # Save to file if requested
    if output_file is not None:
        import pathlib

        output_path = pathlib.Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(output_lines))
        print(f"Inspection results saved to: {output_path}")
