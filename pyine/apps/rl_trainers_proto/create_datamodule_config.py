"""Helper script to generate a datamodule config file with correct paths for GRPO training.

This script automatically detects your TACO dataset paths and generates a properly
configured datamodule YAML file that can be used with inspect_prompts.py and train_grpo.py.

Usage:
    # Generate config for part1to4 (15% of data, ~medium-sized experiments)
    python -m pyine.apps.rl_trainers_proto.create_datamodule_config \
        --output configs/my_datamodule.yaml \
        --dataset-name TACO_10s10t_v1_part1to4

    # Generate config for part1 only (3.8% of data, quick experiments)
    python -m pyine.apps.rl_trainers_proto.create_datamodule_config \
        --output configs/my_datamodule.yaml \
        --dataset-name TACO_10s10t_v1_part1 \
        --max-solutions 100

    # Generate config for full dataset (100% of data)
    python -m pyine.apps.rl_trainers_proto.create_datamodule_config \
        --output configs/my_datamodule.yaml \
        --dataset-name TACO_10s10t_v1_full
"""

import argparse
import logging
import pathlib
import sys

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_taco_paths(dataset_variant: str) -> tuple[list[pathlib.Path], pathlib.Path]:
    """Get TACO dataset LMDB paths and split file path.

    Args:
        dataset_variant: One of "full", "part1to13", "part1to4", or "part1"

    Returns:
        Tuple of (lmdb_paths, split_file_path)
    """
    import pyine.data.traces.dataset_utils
    import pyine.data.utils.splits

    # Get all TACO 10s10t v1 dataset paths
    all_lmdb_paths = pyine.data.traces.dataset_utils.get_matching_dataset_paths(
        source_dataset_name="TACO",
        pattern="v1.5/10s10t.*of000026.*.lmdb",
    )

    if not all_lmdb_paths:
        raise FileNotFoundError(
            "No TACO 10s10t v1 dataset files found. Please ensure the dataset is downloaded "
            "and available in the expected location. See the project README for details."
        )

    logger.info(f"Found {len(all_lmdb_paths)} TACO 10s10t v1 dataset files")

    # Select the subset based on variant
    if dataset_variant == "full":
        lmdb_paths = all_lmdb_paths
        logger.info("Using full dataset (all 26 parts)")
    elif dataset_variant == "part1to13":
        lmdb_paths = all_lmdb_paths[0:13]
        logger.info("Using parts 1-13 (50% of data)")
    elif dataset_variant == "part1to4":
        lmdb_paths = all_lmdb_paths[0:4]
        logger.info("Using parts 1-4 (15% of data)")
    elif dataset_variant == "part1":
        lmdb_paths = [all_lmdb_paths[0]]
        logger.info("Using part 1 only (3.8% of data)")
    else:
        raise ValueError(f"Unknown dataset variant: {dataset_variant}")

    # Get split file path
    split_file_path = pyine.data.utils.splits.get_dataset_split_file_path("TACO")
    if not split_file_path.exists():
        raise FileNotFoundError(
            f"TACO split file not found at: {split_file_path}\n"
            "Please ensure the split file is available. See the project README for details."
        )

    logger.info(f"Using split file: {split_file_path}")

    return lmdb_paths, split_file_path


def create_datamodule_config(
    output_path: pathlib.Path,
    dataset_name: str,
    max_solutions: int | None = None,
    seed: int = 0,
) -> None:
    """Create a datamodule configuration file.

    Args:
        output_path: Path where the config file should be saved
        dataset_name: Name of the dataset configuration (e.g., "TACO_10s10t_v1_part1to4")
        max_solutions: Optional max number of solutions to load (for testing)
        seed: Random seed for reproducibility
    """
    # Parse dataset name to get variant
    if dataset_name.endswith("_full"):
        variant = "full"
    elif dataset_name.endswith("_part1to13"):
        variant = "part1to13"
    elif dataset_name.endswith("_part1to4"):
        variant = "part1to4"
    elif dataset_name.endswith("_part1"):
        variant = "part1"
    else:
        raise ValueError(
            f"Unknown dataset name: {dataset_name}. Expected format: TACO_10s10t_v1_{{full|part1to13|part1to4|part1}}"
        )

    # Get dataset paths
    logger.info(f"Detecting {dataset_name} paths...")
    lmdb_paths, split_file_path = get_taco_paths(variant)

    # Create configuration dictionary
    config = {
        "datamodule_class_path": "pyine.organisms.datamodules.shortcuts.ShortcutBiasDataModule",
        "datamodule_name": dataset_name,
        "lmdb_paths": [str(p) for p in lmdb_paths],
        "split_file_path": str(split_file_path),
        "split_seed": seed,
        "base_filter_rule": "",
        "prompt_config": {
            "prompt_name": "code_execution",
            "use_chat_template": False,  # GRPO requires plain text, not chat format
            "include_examples": False,  # Zero-shot for efficiency; set True if needed
            "target_examples": None,
            "version": "grpo_minimal",  # GRPO-optimized template
        },
        "default_dataparser_config": {
            "class_path": "pyine.organisms.datamodules.samples.builder.SampleBuilder",
            "params": {
                "filtering_config": {},
                "selection_config": {
                    "seed": seed,
                    "allow_db_lookups": True,
                    "code_type_prob_map": {
                        "original": 1.0,
                    },
                    "samples_per_family": 1,
                    "draw_attempts": 5,
                    "fallback_to_orig": False,
                },
                "transform_config": {
                    "seed": seed,
                    "transform_strategy": "never",
                },
            },
        },
        "dataparser_config_overrides": {
            "train": {
                "filtering_config": {},
                "selection_config": {
                    "code_type_prob_map": {
                        "original": 1.0,
                    },
                    "samples_per_family": 1,
                    "draw_attempts": 5,
                    "fallback_to_orig": True,
                },
                "transform_config": {
                    "transform_strategy": "never",
                },
            },
            "valid": {
                "filtering_config": {},
                "selection_config": {
                    "code_type_prob_map": {
                        "original": 1.0,
                    },
                    "samples_per_family": 1,
                    "draw_attempts": 5,
                    "fallback_to_orig": True,
                },
                "transform_config": {
                    "transform_strategy": "never",
                },
            },
        },
        "dataloader_config_overrides": {
            "train": {
                "shuffle": True,
            },
        },
        "keep_generated_datasets_in_memory": False,
        "subset_names": ["train", "valid", "test"],
        "train_subset_names": ["train"],
        "valid_subset_names": ["valid"],
        "eval_subset_names": ["train", "valid"],
        "instantiate_parsers_at_setup": False,
        "use_local_dataset_cache": True,
        "use_tokenized_dataset_cache": True,
        "cache_lock_timeout_seconds": 1800.0,
        "message_generator_num_workers": 4,
        # Disable hint-based validation for GRPO training (using original code only)
        "min_samples_with_hints": 0,
        "min_samples_without_hints": 0,
    }

    # Add max_solutions if specified
    if max_solutions is not None:
        config["max_solution_count"] = max_solutions
        logger.info(f"Limiting to {max_solutions} solutions per subset")

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write configuration to file
    logger.info(f"Writing configuration to: {output_path}")
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info("Configuration file created successfully!")
    logger.info("")
    logger.info("✓ Configured for GRPO training:")
    logger.info("  - Prompt template: grpo_minimal (optimized, ~50% shorter)")
    logger.info("  - Zero-shot by default (no examples, saves tokens)")
    logger.info("  - Plain text format (not chat)")
    logger.info("  - Structured output (```output...```) for easy reward parsing")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Inspect the generated prompts:")
    logger.info("     python -m pyine.apps.rl_trainers_proto.inspect_prompts \\")
    logger.info(f"         --datamodule-config {output_path} \\")
    logger.info("         --subset train \\")
    logger.info("         --num-samples 5")
    logger.info("")
    logger.info("  2. Update train_grpo.py to use this config:")
    logger.info("     data=DataConfig(")
    logger.info("         use_datamodule=True,")
    logger.info(f"         datamodule_config_path=\"{output_path}\",")
    logger.info("         train_subset_name=\"train\",")
    if max_solutions:
        logger.info(f"         max_samples={max_solutions},")
    logger.info("     )")
    logger.info("")
    logger.info("Tip: To add few-shot examples, edit the config and set:")
    logger.info("     prompt_config.include_examples: true")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate a datamodule config file for GRPO training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path where the config file should be saved (e.g., configs/my_datamodule.yaml)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="TACO_10s10t_v1_part1to4",
        choices=[
            "TACO_10s10t_v1_full",
            "TACO_10s10t_v1_part1to13",
            "TACO_10s10t_v1_part1to4",
            "TACO_10s10t_v1_part1",
        ],
        help="Dataset configuration to use (default: TACO_10s10t_v1_part1to4)",
    )
    parser.add_argument(
        "--max-solutions",
        type=int,
        default=None,
        help="Optional: Maximum number of solutions to load per subset (for testing)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0)",
    )

    args = parser.parse_args()

    try:
        output_path = pathlib.Path(args.output)
        create_datamodule_config(
            output_path=output_path,
            dataset_name=args.dataset_name,
            max_solutions=args.max_solutions,
            seed=args.seed,
        )
    except Exception as e:
        logger.error(f"Error creating config: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
