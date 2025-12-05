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


def format_code_execution_prompt(sample: dict[str, Any]) -> str:
    """Format a code execution sample into a prompt string.

    Args:
        sample: Sample dictionary containing 'code', 'inputs', 'output_type', etc.

    Returns:
        Formatted prompt string.
    """
    code = sample.get("code", "")
    inputs = sample.get("inputs", "")
    output_type = sample.get("output_type", "program_output")
    description = sample.get("description", "")

    # Build the prompt
    prompt_parts = []

    if description:
        prompt_parts.append(f"# Description: {description}\n")

    prompt_parts.append(f"# Code:\n{code}\n")
    prompt_parts.append(f"# Inputs:\n{inputs}\n")

    if output_type == "program_output":
        prompt_parts.append("# What is the output of this program?")
    elif output_type == "frame_variables":
        prompt_parts.append("# What are the frame variables at the target line?")
    elif output_type == "function_return":
        prompt_parts.append("# What is the return value of the target function?")
    else:
        prompt_parts.append(f"# What is the {output_type}?")

    return "\n".join(prompt_parts)


def prepare_grpo_dataset_from_datamodule(
    datamodule: pyine.data.datamodule.ConversationDataModule[Any],
    subset_name: str = "train",
    chat_template: str | None = None,
    tokenizer: transformers.PreTrainedTokenizer | None = None,
) -> hf_datasets.Dataset:
    """Prepare a dataset for GRPO training from a ConversationDataModule.

    Args:
        datamodule: The datamodule to load data from.
        subset_name: Name of the subset to load (e.g., "train", "valid").
        chat_template: Optional chat template to format messages.
        tokenizer: Optional tokenizer to apply chat template.

    Returns:
        HuggingFace Dataset formatted for GRPO training.
    """
    logger.info(f"Preparing GRPO dataset from subset: {subset_name}")

    # Get the sample generator/parser from the datamodule
    sample_generator = datamodule.get_parser(subset_name)

    if not isinstance(sample_generator, pyine.organisms.datamodules.utils.samples.SampleBuilder):
        raise TypeError(
            f"Expected SampleBuilder, got {type(sample_generator).__name__}. "
            "GRPO requires access to the full sample data including expected outputs."
        )

    # Extract all samples
    samples = []
    for idx in range(len(sample_generator)):
        sample = sample_generator[idx]
        if not isinstance(sample, pyine.organisms.datamodules.utils.samples.SampleData):
            raise TypeError(f"Expected SampleData, got {type(sample).__name__}")

        # Convert to dict for easier manipulation
        sample_dict = {
            "identifier": sample.identifier,
            "code": sample.code,
            "description": sample.description,
            "entrypoint": sample.entrypoint,
            "inputs": sample.inputs,
            "expected_output": sample.expected_output,
            "output_type": sample.output_type,
            "code_type": sample.code_type,
            "tags": sample.get_tag_list(),
        }
        samples.append(sample_dict)

    logger.info(f"Extracted {len(samples)} samples from {subset_name}")

    # Create HuggingFace dataset
    dataset = hf_datasets.Dataset.from_list(samples)

    # Format prompts
    def format_sample(example: dict[str, Any]) -> dict[str, Any]:
        # Create the prompt as a simple string for now
        # GRPO will convert this to the appropriate message format
        prompt = format_code_execution_prompt(example)
        example["prompt"] = prompt
        return example

    dataset = dataset.map(format_sample, desc="Formatting prompts")

    logger.info(f"Prepared GRPO dataset with {len(dataset)} examples")
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

    # Validate required columns
    required_columns = ["prompt", "expected_output"]
    missing_columns = [col for col in required_columns if col not in dataset.column_names]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}. "
            f"Available columns: {dataset.column_names}"
        )

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
