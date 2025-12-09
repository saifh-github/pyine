"""Data utilities for preparing code execution datasets for GRPO training.

This module provides utilities to load and format code execution datasets
for use with TRL's GRPO trainer. It handles conversion from the codebase's
ConversationDataModule format to the format expected by GRPO.
"""

import logging
from typing import Any

import datasets as hf_datasets
import transformers

import pyine.data.datamodule
import pyine.organisms.datamodules.utils.samples

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
    elif isinstance(formatted, str):
        return formatted
    else:
        # Fallback: try to convert to string representation
        return str(formatted)


def prepare_grpo_dataset_from_datamodule(
    datamodule: pyine.data.datamodule.ConversationDataModule[Any],
    subset_name: str = "train",
    prompt_version: str | None = "unstructured_with_3_output_types",
    tokenizer: transformers.PreTrainedTokenizer | None = None,
) -> hf_datasets.Dataset:
    """Prepare a dataset for GRPO training from a ConversationDataModule.

    This function extracts samples from the datamodule and formats them using the
    configured prompt template. The resulting dataset will have a 'prompt' column
    containing the formatted prompt text, along with metadata fields needed for
    reward calculation.

    Args:
        datamodule: The datamodule to load data from.
        subset_name: Name of the subset to load (e.g., "train", "valid").
        prompt_version: Version of the prompt template to use from code_execution.yaml.
            Defaults to "unstructured_with_3_output_types" which supports all output types.
        tokenizer: Optional tokenizer (currently unused, but kept for API compatibility).

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
    import pyine.prompts.configs.code_execution

    prompt_template = pyine.prompts.configs.code_execution.get_prompt_template(
        version=prompt_version,
        use_chat_template=False,  # GRPO needs plain text, not chat format
        include_examples=True,  # Include few-shot examples for better performance
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
        dataset_entry = {
            "prompt": formatted_prompt,
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
        prompt = sample["prompt"]
        _print(f"\nPrompt ({len(prompt)} chars):")
        _print("─" * 40)
        if len(prompt) > max_prompt_length:
            _print(prompt[:max_prompt_length])
            _print(f"\n... [truncated, {len(prompt) - max_prompt_length} more chars] ...")
        else:
            _print(prompt)

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
