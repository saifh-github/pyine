import asyncio
import json
import pathlib

import dotenv
import tiktoken

import data.taco.taco_dataset_parser
import pyine.prompts.code_analysis_prompt
import pyine.utils.code_validation


async def reprocess_code_samples(
    output_dir_path: pathlib.Path,
    batch_size: int = 32,
    chunk_size: int = 256,
    max_token_count: int = 20_000,
) -> None:
    """
    Processes the TACO dataset and saves valid samples to individual files using async batch processing.

    Assumes the dataset samples contain multiple 'solutions' (i.e., multiple Python code snippets).

    Args:
        output_dir_path: path to save the JSON files containing the updated data samples.
        batch_size: number of samples to process in each batch.
        chunk_size: number of solutions to process in each API invocation iteration.
        max_token_count: maximum number of tokens allowed in a single query.
    """
    output_dir_path.mkdir(parents=True, exist_ok=True)
    _tokenizer = tiktoken.encoding_for_model("gpt-4o")  # used as reference

    def _count_tokens(text: str) -> int:
        return len(_tokenizer.encode(text))

    dataset_reader = data.taco.taco_dataset_parser.TacoDatasetParser()
    code_analysis_chain = pyine.prompts.code_analysis_prompt.get_deepseek_chain()
    total_samples = len(dataset_reader)  # min(10, len(dataset_reader))
    print(f"samples in dataset: {total_samples}")
    example_prompt_text = pyine.prompts.code_analysis_prompt.code_analysis_prompt.format(
        code="def example(): pass",
    )
    prompt_token_count = _count_tokens(example_prompt_text)
    print(f"basic prompt token count (estimate): {prompt_token_count}")
    for batch_start in range(0, total_samples, batch_size):
        batch_end = min(batch_start + batch_size, total_samples)
        batch_indices = [  # keep only sample indices that are not already done
            idx for idx in range(batch_start, batch_end) if not (output_dir_path / f"{idx:06d}.json").exists()
        ]
        if not batch_indices:
            continue
        batch_samples = []
        for sample_idx in batch_indices:
            try:
                sample = dataset_reader[sample_idx]
                batch_samples.append((sample_idx, sample))
            except Exception as e:
                print(f"Error loading sample {sample_idx}: {e}")
                sample_output_path = output_dir_path / f"{sample_idx:06d}.json"
                with open(sample_output_path, "w") as fd:
                    json.dump({"error": str(e)}, fd, indent=2)  # noqa
                continue
        all_solutions, solution_mapping = [], []
        for sample_idx, sample in batch_samples:
            assert isinstance(sample["solutions"], list)
            for solution in sample["solutions"]:
                validation_errors, analysis_outputs = [], []
                if not isinstance(solution, str):
                    expected_keys = {"orig_code", "code", "validation_errors", "analysis_outputs"}
                    assert isinstance(solution, dict) and expected_keys.issubset(solution.keys())
                    orig_code = solution["orig_code"]
                    solution = solution["code"]
                    validation_errors = solution["validation_errors"]
                    analysis_outputs = solution["analysis_outputs"]
                else:
                    orig_code = solution
                try:
                    solution_token_count = _count_tokens(solution)
                    query_token_count = solution_token_count + prompt_token_count
                    assert (
                        query_token_count <= max_token_count
                    ), f"prompt + solution token count ({query_token_count}) exceeds max ({max_token_count})"
                    pyine.utils.code_validation.validate_code(solution)
                except Exception as e:
                    validation_errors.append(str(e))
                all_solutions.append(
                    {
                        "orig_code": orig_code,
                        "code": solution,
                        "validation_errors": validation_errors,
                        "analysis_outputs": analysis_outputs,
                    }
                )
                solution_mapping.append((sample_idx, sample))
        print(f"sending batch [{batch_start}-{batch_end}] with {len(all_solutions)} solutions")
        analysis_outputs = []
        for i in range(0, len(all_solutions), chunk_size):
            chunk_solutions = all_solutions[i : i + chunk_size]
            chunk_inputs = [{"code": s["code"]} for s in chunk_solutions]
            print(f"Processing chunk {i // chunk_size + 1}/{(len(all_solutions) + chunk_size - 1) // chunk_size}")
            tasks = []
            for input_data in chunk_inputs:

                async def process_with_retry(input_data, max_retries=5, backoff=2):
                    retries = 0
                    while retries < max_retries:
                        try:
                            return await code_analysis_chain.ainvoke(input_data, config={"max_concurrency": 512})
                        except Exception as e:
                            full_stop_exceptions = ["insufficient balance", "stopping processing at "]
                            if any([s in str(e).lower() for s in full_stop_exceptions]):
                                raise e
                            retries += 1
                            if retries >= max_retries:
                                print(f"Failed after {max_retries} retries: {str(e)}")
                                return {"error": str(e)}
                            wait_time = backoff**retries
                            print(f"Retry {retries} after {wait_time}s due to: {str(e)}")
                            await asyncio.sleep(wait_time)

                tasks.append(process_with_retry(input_data))
            # Wait for all tasks in this chunk to complete
            chunk_results = await asyncio.gather(*tasks)
            analysis_outputs.extend(chunk_results)
            # add a small delay between chunks to avoid rate limiting
            await asyncio.sleep(1)
        print(f"received batch [{batch_start}-{batch_end}] with {len(analysis_outputs)} outputs")
        results_by_sample = {}
        for solution_idx, output in enumerate(analysis_outputs):
            sample_idx, sample = solution_mapping[solution_idx]
            # handle both successful responses and error cases
            output_data = output.model_dump() if hasattr(output, "model_dump") else output
            all_solutions[solution_idx]["analysis_outputs"].append(output_data)
            if sample_idx not in results_by_sample:
                results_by_sample[sample_idx] = sample.copy()
                results_by_sample[sample_idx]["solutions"] = []
            results_by_sample[sample_idx]["solutions"].append(all_solutions[solution_idx])
        for sample_idx, processed_sample in results_by_sample.items():
            sample_output_path = output_dir_path / f"{sample_idx:06d}.json"
            with open(sample_output_path, "w") as fd:
                json.dump(processed_sample, fd, indent=2)  # noqa


if __name__ == "__main__":
    dotenv.load_dotenv()
    asyncio.run(
        reprocess_code_samples(
            output_dir_path=pathlib.Path("./data/2025-03-26-v01/"),
        ),
    )
