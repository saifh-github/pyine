"""Standalone script to inspect and validate GRPO dataset prompts.

This script allows you to preview the formatted prompts that will be used for
GRPO training before actually starting the training process. This is useful for:
- Validating prompt formatting
- Checking that Jinja2 templates are rendering correctly
- Verifying expected outputs are preserved
- Debugging data loading issues

Usage:
    # With datamodule
    python -m pyine.apps.rl_trainers_proto.inspect_prompts \
        --datamodule-config path/to/datamodule_config.yaml \
        --subset train \
        --num-samples 5

    # With HF dataset (simpler, but less realistic)
    python -m pyine.apps.rl_trainers_proto.inspect_prompts \
        --dataset-path trl-lib/ultrafeedback-prompt \
        --num-samples 3

    # Save output to file
    python -m pyine.apps.rl_trainers_proto.inspect_prompts \
        --datamodule-config path/to/config.yaml \
        --output-file prompt_inspection.txt
"""

import argparse
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def inspect_with_datamodule(
    datamodule_config_path: str,
    subset_name: str,
    num_samples: int,
    max_samples: int | None,
    max_prompt_length: int,
    prompt_version: str,
    output_file: str | None,
) -> None:
    """Inspect prompts generated from a datamodule.

    Args:
        datamodule_config_path: Path to datamodule configuration file.
        subset_name: Name of subset to inspect (e.g., 'train', 'valid').
        num_samples: Number of samples to display.
        max_samples: Maximum number of samples to load from dataset.
        prompt_version: Prompt template version to use.
        output_file: Optional path to save inspection results.
    """
    from pyine.apps.rl_trainers_proto.config import DataConfig
    from pyine.apps.rl_trainers_proto.data_utils import inspect_grpo_dataset, prepare_grpo_dataset_from_datamodule
    from pyine.apps.rl_trainers_proto.train_grpo import load_datamodule

    logger.info("=" * 80)
    logger.info("Loading datamodule and preparing dataset...")
    logger.info("=" * 80)

    # Create data config
    data_config = DataConfig(
        use_datamodule=True,
        datamodule_config_path=datamodule_config_path,
        train_subset_name=subset_name,
        max_samples=max_samples,
    )

    # Load datamodule
    logger.info(f"Loading datamodule from: {datamodule_config_path}")
    datamodule = load_datamodule(data_config)

    # Prepare dataset
    logger.info(f"Preparing GRPO dataset from subset: {subset_name}")
    logger.info(f"Using prompt version: {prompt_version}")
    dataset = prepare_grpo_dataset_from_datamodule(
        datamodule=datamodule,
        subset_name=subset_name,
        prompt_version=prompt_version,
    )

    # Apply max_samples if specified
    if max_samples is not None and len(dataset) > max_samples:
        logger.info(f"Limiting dataset from {len(dataset)} to {max_samples} samples")
        dataset = dataset.select(range(max_samples))

    logger.info(f"Dataset prepared with {len(dataset)} samples")
    logger.info("")

    # Inspect the dataset
    inspect_grpo_dataset(
        dataset=dataset,
        num_samples=num_samples,
        output_file=output_file,
        max_prompt_length=max_prompt_length,
    )


def inspect_with_hf_dataset(
    dataset_path: str,
    split: str,
    num_samples: int,
    max_prompt_length: int,
    output_file: str | None,
) -> None:
    """Inspect prompts from a simple HuggingFace dataset.

    Note: This is less realistic as it won't use your actual prompting pipeline.

    Args:
        dataset_path: HuggingFace dataset path.
        split: Dataset split to load.
        num_samples: Number of samples to display.
        output_file: Optional path to save inspection results.
    """
    from pyine.apps.rl_trainers_proto.data_utils import inspect_grpo_dataset, prepare_grpo_dataset_simple

    logger.info("=" * 80)
    logger.info("Loading HuggingFace dataset...")
    logger.info("=" * 80)

    dataset = prepare_grpo_dataset_simple(
        dataset_path=dataset_path,
        split=split,
    )

    logger.info(f"Dataset loaded with {len(dataset)} samples")
    logger.info("")

    # Inspect the dataset
    inspect_grpo_dataset(
        dataset=dataset,
        num_samples=num_samples,
        output_file=output_file,
        max_prompt_length=max_prompt_length,
    )


def main() -> None:
    """Main entry point for prompt inspection script."""
    parser = argparse.ArgumentParser(
        description="Inspect and validate GRPO dataset prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data source arguments (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--datamodule-config",
        type=str,
        help="Path to datamodule configuration file (YAML or JSON)",
    )
    source_group.add_argument(
        "--dataset-path",
        type=str,
        help="HuggingFace dataset path (simpler, but less realistic)",
    )

    # Common arguments
    parser.add_argument(
        "--subset",
        type=str,
        default="train",
        help="Subset name to inspect (default: train)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split for HF datasets (default: train)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of samples to display (default: 3)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to load from dataset (default: all)",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=500,
        help="Maximum number of characters allowed before truncating the samples visualization (default: 500)",
    )
    parser.add_argument(
        "--prompt-version",
        type=str,
        default="grpo_minimal",
        help="Prompt template version to use (default: grpo_minimal, optimized for GRPO training)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional path to save inspection results",
    )

    args = parser.parse_args()

    try:
        if args.datamodule_config:
            # Inspect with datamodule (recommended)
            inspect_with_datamodule(
                datamodule_config_path=args.datamodule_config,
                subset_name=args.subset,
                num_samples=args.num_samples,
                max_samples=args.max_samples,
                max_prompt_length=args.max_prompt_length,
                prompt_version=args.prompt_version,
                output_file=args.output_file,
            )
        else:
            # Inspect with HF dataset (simpler, but less realistic)
            inspect_with_hf_dataset(
                dataset_path=args.dataset_path,
                split=args.split,
                num_samples=args.num_samples,
                max_prompt_length=args.max_prompt_length,
                output_file=args.output_file,
            )

        logger.info("")
        logger.info("=" * 80)
        logger.info("Inspection complete!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error during inspection: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
