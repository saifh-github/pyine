#!/usr/bin/env python3
"""
vLLM Evaluation Server Script

This script loads a HuggingFace model (checkpoint or direct model name),
optionally merges LoRA adapters, and starts a vLLM server for inference.

Usage:
    # Serve any HuggingFace model directly (model name: "Qwen/Qwen2.5-7B-Instruct")
    python vllm_server.py --model Qwen/Qwen2.5-7B-Instruct

    # Serve a local checkpoint (model name auto-derived from last 3 path components)
    python vllm_server.py --checkpoint_path /path/to/checkpoint

    # Serve with custom model name (overrides auto-derived name)
    python vllm_server.py --model Qwen/Qwen2.5-7B-Instruct --model_name my-qwen-model

    # Serve with LoRA merging
    python vllm_server.py --checkpoint_path /path/to/lora_checkpoint --base_model Qwen/Qwen2.5-7B-Instruct --merge_lora

    # Multi-GPU setup with specific GPU selection
    python vllm_server.py --model meta-llama/Llama-3.1-70B-Instruct --cuda_devices 0,1,2,3 --tensor_parallel_size 4

    # Running two servers on non-overlapping GPUs
    # Terminal 1 - Evaluation model on GPUs 0-3
    python vllm_server.py --checkpoint_path /path/to/checkpoint --cuda_devices 0,1,2,3 --port 8000

    # Terminal 2 - Grading model on GPUs 4-7
    python vllm_server.py --model Qwen/Qwen2.5-72B-Instruct --cuda_devices 4,5,6,7 --port 8001
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from dotenv import find_dotenv, load_dotenv

# Load environment variables from .env file (searches up directory tree)
dotenv_path = find_dotenv(usecwd=True)
if not dotenv_path:
    raise FileNotFoundError(
        "Could not find .env file. Please ensure a .env file exists in the repository root. "
        "See .env.template for an example of how to set up your environment variables."
    )
load_dotenv(dotenv_path)
print(f"Loaded environment variables from: {dotenv_path}")

# Check if vllm is available without importing it
if importlib.util.find_spec("vllm") is None:
    raise ImportError("vLLM not installed. Install with: pip install vllm")


def check_is_lora_checkpoint(checkpoint_path: str) -> bool:
    """Check if checkpoint contains LoRA adapters."""
    adapter_config = Path(checkpoint_path) / "adapter_config.json"
    return adapter_config.exists()


def load_adapter_config(checkpoint_path: str) -> dict[str, Any]:
    """Load adapter configuration to get base model name."""
    adapter_config_path = Path(checkpoint_path) / "adapter_config.json"
    with open(adapter_config_path) as f:
        return json.load(f)


def get_default_data_cache_path() -> Path:
    """Returns the path to the default dataset cache directory to use for model caching."""
    # note: we use the same default path hierarchy and fallbacks as in the `pyine.utils.filesystem` moduel
    env_path = os.environ.get("PYINE_CACHE_ROOT", None)
    if env_path:
        env_path = os.path.expandvars(os.path.expanduser(env_path))
        out_path = Path(env_path).resolve()
    else:
        env_path = os.environ.get("PYINE_DATA_ROOT", None)
        if env_path:
            env_path = os.path.expandvars(os.path.expanduser(env_path))
            out_path = Path(env_path).resolve() / "cache"
        else:
            out_path = Path("./data/cache").resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def merge_lora_adapter(checkpoint_path: str, base_model: str | None, output_dir: str) -> str:
    """
    Merge LoRA adapter with base model and save merged model.

    Args:
        checkpoint_path: Path to checkpoint with LoRA adapters
        base_model: Base model name or path (if None, read from adapter_config.json)
        output_dir: Directory to save merged model

    Returns:
        Path to merged model
    """
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("PEFT/Transformers not installed. Install with: pip install peft transformers")
        sys.exit(1)

    print(f"Loading LoRA checkpoint from {checkpoint_path}...")

    # Get base model name from adapter config if not provided
    if base_model is None:
        adapter_config = load_adapter_config(checkpoint_path)
        base_model = adapter_config.get("base_model_name_or_path")
        if base_model is None:
            raise ValueError("Could not determine base model. Please specify --base_model")
        print(f"Detected base model from adapter config: {base_model}")

    # Load base model
    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)

    # Load and merge LoRA adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, checkpoint_path)

    print("Merging adapter with base model...")
    merged_model = model.merge_and_unload()

    # Save merged model
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving merged model to {output_dir}...")
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    print(f"Successfully merged and saved model to {output_dir}")
    return output_dir


def start_vllm_server(
    model_path: str,
    host: str = "0.0.0.0",  # noqa: S104 # Security issue raised: Possible bind to all interfaces for network access
    port: int = 8000,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int | None = None,
    trust_remote_code: bool = True,
    model_name: str = "default",
) -> None:
    """
    Start vLLM OpenAI-compatible API server.

    Args:
        model_path: Path to model directory
        host: Host to bind server to (0.0.0.0 for network access)
        port: Port number
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: Fraction of GPU memory to use
        max_model_len: Maximum sequence length
        trust_remote_code: Whether to trust remote code in model
        model_name: Name to serve the model under (clients use this name in API calls)
    """
    print("\n" + "=" * 80)
    print("Starting vLLM Server")
    print("=" * 80)
    print(f"Model Path: {model_path}")
    print(f"Served Model Name: {model_name}")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Tensor Parallel Size: {tensor_parallel_size}")
    print(f"GPU Memory Utilization: {gpu_memory_utilization}")
    print("=" * 80 + "\n")

    # Build vllm serve command
    cmd_parts = [
        "uv",
        "run",
        "vllm",
        "serve",
        model_path,
        "--host",
        host,
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
    ]

    if trust_remote_code:
        cmd_parts.append("--trust-remote-code")

    if max_model_len is not None:
        cmd_parts.extend(["--max-model-len", str(max_model_len)])

    # Always set the served model name
    cmd_parts.extend(["--served-model-name", model_name])

    # Execute vllm serve
    import subprocess

    cmd = " ".join(cmd_parts)
    print(f"Executing: {cmd}\n")

    try:
        subprocess.run(cmd, shell=True, check=True)  # noqa: S602 # Security issue raised: Shell=True is required to run vllm serve command
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except subprocess.CalledProcessError as e:
        print(f"Error running vLLM server: {e}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve models with vLLM (from HuggingFace or local checkpoint)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model selection arguments
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model", type=str, help="HuggingFace model name/path (e.g., 'Qwen/Qwen2.5-7B-Instruct')")
    model_group.add_argument("--checkpoint_path", type=str, help="Path to local HuggingFace checkpoint directory")
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model name/path (required for LoRA if not in adapter_config.json)",
    )
    parser.add_argument(
        "--merge_lora",
        action="store_true",
        help="Merge LoRA adapter with base model (auto-detected if adapter_config.json exists)",
    )
    parser.add_argument(
        "--merged_model_dir",
        type=str,
        default=None,
        help="Directory to save merged model (default: <PYINE_CACHE_ROOT>/vllm_merged_models/<checkpoint_name>)",
    )
    parser.add_argument(
        "--force_merge", action="store_true", help="Force re-merging even if merged model already exists"
    )

    # vLLM server arguments
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind server (0.0.0.0 for network access)")  # noqa: S104 # Security issue raised: Possible bind to all interfaces for network access
    parser.add_argument("--port", type=int, default=8000, help="Port number for vLLM server")
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Name to serve the model under (clients use this in API calls). If not specified, automatically derived from model/checkpoint path.",
    )
    parser.add_argument(
        "--cuda_devices",
        type=str,
        default=None,
        help="Comma-separated GPU device IDs to use (e.g., '0,1,2,3'). Sets CUDA_VISIBLE_DEVICES. If not specified, uses all available GPUs.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=8,
        help="Number of GPUs for tensor parallelism (1-8 for your cluster)",
    )
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.9, help="Fraction of GPU memory to use (0.0-1.0)"
    )
    parser.add_argument(
        "--max_model_len", type=int, default=None, help="Maximum sequence length (default: model's max)"
    )
    parser.add_argument(
        "--trust_remote_code", action="store_true", default=True, help="Trust remote code in model (needed for Qwen)"
    )

    args = parser.parse_args()

    # Validate that either --model or --checkpoint_path is provided
    if not args.model and not args.checkpoint_path:
        parser.error("Either --model or --checkpoint_path must be specified")

    # Determine model path to serve
    model_path = args.model if args.model else args.checkpoint_path
    is_lora = False

    # If using checkpoint_path, validate it exists and check for LoRA
    if args.checkpoint_path:
        checkpoint_path = Path(args.checkpoint_path)
        if not checkpoint_path.exists():
            print(f"Error: Checkpoint path does not exist: {args.checkpoint_path}")
            sys.exit(1)

        # Determine if this is a LoRA checkpoint
        is_lora = check_is_lora_checkpoint(args.checkpoint_path)

        if is_lora:
            print("Detected LoRA checkpoint (found adapter_config.json)")

            # Auto-enable merging for LoRA checkpoints
            if not args.merge_lora:
                print("Auto-enabling --merge_lora for LoRA checkpoint")
                args.merge_lora = True

        model_path = args.checkpoint_path

        if args.merge_lora:
            if not is_lora:
                print("Warning: --merge_lora specified but no LoRA adapter detected")
                print("Proceeding to serve checkpoint directly...")
            else:
                # Determine merged model directory if not explicitly provided
                if args.merged_model_dir is None:
                    cache_root = get_default_data_cache_path()
                    vllm_merged_base = cache_root / "vllm_merged_models"

                    # Create checkpoint-specific subdirectory using naming convention
                    path_parts = Path(args.checkpoint_path).parts
                    num_components = min(3, len(path_parts))
                    checkpoint_name = "_".join(path_parts[-num_components:])

                    merged_model_dir = vllm_merged_base / checkpoint_name
                    print(f"Using default merged model directory: {merged_model_dir}")
                else:
                    merged_model_dir = Path(args.merged_model_dir)

                # Check if merged model already exists
                if merged_model_dir.exists() and not args.force_merge:
                    print(f"Merged model already exists at {merged_model_dir}")
                    print("Using existing merged model. Use --force_merge to re-merge.")
                    model_path = str(merged_model_dir)
                else:
                    # Merge LoRA adapter
                    model_path = merge_lora_adapter(
                        checkpoint_path=args.checkpoint_path,
                        base_model=args.base_model,
                        output_dir=str(merged_model_dir),
                    )
    elif args.model:
        # Using direct model name from HuggingFace
        print(f"Serving model directly from HuggingFace: {args.model}")
        model_path = args.model

    # Determine served model name
    if args.model_name is not None:
        # User explicitly provided model name - use it as-is
        served_model_name = args.model_name
    else:
        # Auto-derive model name from model/checkpoint path
        if args.model:
            # For HF models, use the full model string (e.g., "Qwen/Qwen2.5-7B-Instruct")
            served_model_name = args.model
        elif args.checkpoint_path:
            # For local paths, use last 2-3 path components for context
            path_parts = Path(args.checkpoint_path).parts
            # Use last 3 components if available, otherwise use what's available
            num_components = min(3, len(path_parts))
            served_model_name = "/".join(path_parts[-num_components:])

    # Set CUDA_VISIBLE_DEVICES if specified
    if args.cuda_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
        print(f"Setting CUDA_VISIBLE_DEVICES={args.cuda_devices}")

        # Validate tensor_parallel_size matches number of devices
        num_devices = len(args.cuda_devices.split(","))
        if args.tensor_parallel_size != num_devices:
            print(
                f"Warning: tensor_parallel_size ({args.tensor_parallel_size}) != number of CUDA devices ({num_devices})"
            )
            print(f"Adjusting tensor_parallel_size to {num_devices} to match available devices")
            args.tensor_parallel_size = num_devices

    # Start vLLM server
    start_vllm_server(
        model_path=model_path,
        host=args.host,
        port=args.port,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
        model_name=served_model_name,
    )


if __name__ == "__main__":
    main()
