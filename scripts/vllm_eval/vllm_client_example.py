#!/usr/bin/env python3
"""
vLLM Client Example - Batched Inference

This script demonstrates how to make batched calls to a vLLM server using the
OpenAI-compatible API. The vLLM server (started via vllm_eval.py) exposes an
OpenAI-compatible endpoint that supports batched requests.

Usage:
    # Basic usage with default server
    python vllm_client_example.py

    # Custom server URL
    python vllm_client_example.py --base_url http://localhost:8000/v1

    # With custom prompts file
    python vllm_client_example.py --prompts_file my_prompts.txt
"""

import argparse
import asyncio
import time
from typing import Any

try:
    from openai import AsyncOpenAI, OpenAI
except ImportError:
    print("OpenAI package not installed. Install with: pip install openai")
    import sys

    sys.exit(1)


def create_sample_prompts() -> list[str]:
    """Create sample prompts for demonstration."""
    return [
        "Explain what a binary search tree is in one sentence.",
        "Write a Python function to calculate factorial.",
        "What is the time complexity of quicksort?",
        "How do you reverse a linked list?",
        "Explain the difference between a stack and a queue.",
    ]


def synchronous_batched_calls(
    client: OpenAI, prompts: list[str], model: str = "default", max_tokens: int = 256, temperature: float = 0.7
) -> list[dict[str, Any]]:
    """
    Make synchronous batched calls to vLLM server.

    This approach sends requests one by one but is simpler to understand.

    Args:
        client: OpenAI client instance
        prompts: List of prompt strings
        model: Model name (vLLM uses 'default' or the actual model name)
        max_tokens: Maximum tokens to generate per prompt
        temperature: Sampling temperature

    Returns:
        List of response dictionaries
    """
    print(f"\n{'=' * 80}")
    print("Synchronous Batched Calls")
    print(f"{'=' * 80}")
    print(f"Processing {len(prompts)} prompts...")

    responses = []
    start_time = time.time()

    for i, prompt in enumerate(prompts, 1):
        print(f"\nRequest {i}/{len(prompts)}: {prompt[:50]}...")

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        result = {
            "prompt": prompt,
            "response": response.choices[0].message.content,
            "tokens_used": response.usage.total_tokens if response.usage else None,
            "finish_reason": response.choices[0].finish_reason,
        }
        responses.append(result)

        print(f"Response: {result['response'][:100]}...")
        print(f"Tokens: {result['tokens_used']}, Finish reason: {result['finish_reason']}")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 80}")
    print(f"Completed {len(prompts)} requests in {elapsed:.2f}s ({elapsed / len(prompts):.2f}s per request)")
    print(f"{'=' * 80}")

    return responses


async def async_batched_calls(
    client: AsyncOpenAI, prompts: list[str], model: str = "default", max_tokens: int = 256, temperature: float = 0.7
) -> list[dict[str, Any]]:
    """
    Make asynchronous batched calls to vLLM server.

    This approach sends all requests concurrently for better throughput.
    vLLM's continuous batching will handle them efficiently.

    Args:
        client: Async OpenAI client instance
        prompts: List of prompt strings
        model: Model name (vLLM uses 'default' or the actual model name)
        max_tokens: Maximum tokens to generate per prompt
        temperature: Sampling temperature

    Returns:
        List of response dictionaries
    """
    print(f"\n{'=' * 80}")
    print("Asynchronous Batched Calls")
    print(f"{'=' * 80}")
    print(f"Processing {len(prompts)} prompts concurrently...")

    async def process_prompt(prompt: str, index: int) -> dict[str, Any]:
        """Process a single prompt asynchronously."""
        print(f"\nRequest {index + 1}/{len(prompts)}: {prompt[:50]}...")

        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        result = {
            "prompt": prompt,
            "response": response.choices[0].message.content,
            "tokens_used": response.usage.total_tokens if response.usage else None,
            "finish_reason": response.choices[0].finish_reason,
        }

        print(f"Response {index + 1}: {result['response'][:100]}...")
        print(f"Tokens: {result['tokens_used']}, Finish reason: {result['finish_reason']}")

        return result

    start_time = time.time()

    # Create tasks for all prompts
    tasks = [process_prompt(prompt, i) for i, prompt in enumerate(prompts)]

    # Wait for all tasks to complete
    responses = await asyncio.gather(*tasks)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 80}")
    print(f"Completed {len(prompts)} requests in {elapsed:.2f}s ({elapsed / len(prompts):.2f}s per request)")
    print(f"{'=' * 80}")

    return list(responses)


def load_prompts_from_file(file_path: str) -> list[str]:
    """Load prompts from a text file (one prompt per line)."""
    with open(file_path) as f:
        return [line.strip() for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Example client for batched vLLM inference", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--base_url", type=str, default="http://localhost:8000/v1", help="vLLM server base URL (OpenAI-compatible)"
    )
    parser.add_argument("--model", type=str, default="default", help="Model name to use (default works for most cases)")
    parser.add_argument("--max_tokens", type=int, default=256, help="Maximum tokens to generate per prompt")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--prompts_file", type=str, default=None, help="Path to file with prompts (one per line)")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sync", "async", "both"],
        default="both",
        help="Mode: sync (sequential), async (concurrent), or both",
    )

    args = parser.parse_args()

    # Load prompts
    if args.prompts_file:
        print(f"Loading prompts from {args.prompts_file}...")
        prompts = load_prompts_from_file(args.prompts_file)
    else:
        print("Using sample prompts...")
        prompts = create_sample_prompts()

    print(f"\nConnecting to vLLM server at: {args.base_url}")
    print(f"Model: {args.model}")
    print(f"Prompts to process: {len(prompts)}")

    # Synchronous batched calls
    if args.mode in ["sync", "both"]:
        client = OpenAI(base_url=args.base_url, api_key="EMPTY")  # vLLM doesn't require API key
        try:
            synchronous_batched_calls(
                client=client,
                prompts=prompts,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        except Exception as e:
            print(f"\nError in synchronous calls: {e}")
            print("Make sure vLLM server is running!")
            import sys

            sys.exit(1)

    # Asynchronous batched calls
    if args.mode in ["async", "both"]:
        async_client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")  # vLLM doesn't require API key
        try:
            asyncio.run(
                async_batched_calls(
                    client=async_client,
                    prompts=prompts,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
            )
        except Exception as e:
            print(f"\nError in asynchronous calls: {e}")
            print("Make sure vLLM server is running!")
            import sys

            sys.exit(1)

    print(f"\n{'=' * 80}")
    print("Example completed successfully!")
    print(f"{'=' * 80}")
    print("\nTips:")
    print("- Async mode is faster for batched requests due to concurrent execution")
    print("- vLLM uses continuous batching to efficiently handle concurrent requests")
    print("- Adjust --max_tokens and --temperature based on your use case")
    print("- For production, consider adding retry logic and better error handling")


if __name__ == "__main__":
    main()
