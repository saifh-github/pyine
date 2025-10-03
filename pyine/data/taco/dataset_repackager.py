"""
This module contains the TACO dataset repackaging logic.
"""

import asyncio
import json
import pathlib
import typing

import tiktoken

import pyine.data.taco.dataset_reader
import pyine.data.taco.dataset_utils
import pyine.prompts.manager
import pyine.utils.code.validation
import pyine.utils.llm_providers
import pyine.utils.reprod


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

    dataset_reader = pyine.data.taco.dataset_reader.DatasetReader()
    llm = pyine.utils.llm_providers.get_model_from_provider(
        provider="deepseek",
        model="deepseek-chat",
        temperature=0.0,  # recommended setting for coding/math
        max_tokens=1024,
    )
    code_analysis_prompt = pyine.prompts.manager.get_prompt_template("code_analysis")
    example_prompt_text = code_analysis_prompt.format(
        code="def example(): pass",
    )
    prompt_token_count = _count_tokens(example_prompt_text)
    print(f"basic prompt token count (estimate): {prompt_token_count}")
    code_analysis_chain = pyine.prompts.manager.get_prompt_chain(llm, "code_analysis")
    total_samples = len(dataset_reader)
    print(f"samples in dataset: {total_samples}")
    for batch_start in range(0, total_samples, batch_size):
        batch_end = min(batch_start + batch_size, total_samples)
        batch_indices = [  # keep only sample indices that are not already done
            idx for idx in range(batch_start, batch_end) if not (output_dir_path / f"{idx:06d}.json").exists()
        ]
        if not batch_indices:
            continue
        batch_samples: list[tuple[int, dict[str, typing.Any]]] = []
        for sample_idx in batch_indices:
            try:
                sample = dataset_reader[sample_idx]
                batch_samples.append((sample_idx, sample))
            except Exception as e:
                print(f"Error loading sample {sample_idx}: {e}")
                sample_output_path = output_dir_path / f"{sample_idx:06d}.json"
                with open(sample_output_path, "w") as fd:
                    json.dump({"error": str(e)}, fd, indent=2)
                continue
        all_solutions: list[dict[str, typing.Any]] = []
        solution_mapping: list[tuple[int, dict[str, typing.Any]]] = []
        for sample_idx, sample in batch_samples:
            solutions_field = typing.cast("list[typing.Any]", sample.get("solutions", []))
            for solution_entry in solutions_field:
                validation_errors: list[str] = []
                analysis_outputs_entry: list[typing.Any] = []
                if isinstance(solution_entry, str):
                    orig_code = solution_entry
                    code_text = solution_entry
                else:
                    expected_keys = {
                        "orig_code",
                        "code",
                        "validation_errors",
                        "analysis_outputs",
                    }
                    solution_dict = typing.cast("dict[str, typing.Any]", solution_entry)
                    if not expected_keys.issubset(solution_dict):
                        raise ValueError("malformed solution entry")
                    orig_code = typing.cast("str", solution_dict["orig_code"])
                    code_text = typing.cast("str", solution_dict["code"])
                    validation_errors_value = solution_dict["validation_errors"]
                    analysis_outputs_value = solution_dict["analysis_outputs"]
                    if isinstance(validation_errors_value, list):
                        parsed_errors: list[str] = []
                        for err in typing.cast("list[typing.Any]", validation_errors_value):
                            parsed_errors.append(str(err))
                        validation_errors = parsed_errors
                    if isinstance(analysis_outputs_value, list):
                        analysis_outputs_entry = list(typing.cast("list[typing.Any]", analysis_outputs_value))
                try:
                    solution_token_count = _count_tokens(code_text)
                    query_token_count = solution_token_count + prompt_token_count
                    assert query_token_count <= max_token_count, (
                        f"prompt + solution token count ({query_token_count}) exceeds max ({max_token_count})"
                    )
                    pyine.utils.code.validation.validate_code(code_text)
                except Exception as e:
                    validation_errors.append(str(e))
                all_solutions.append(
                    {
                        "orig_code": orig_code,
                        "code": code_text,
                        "validation_errors": validation_errors,
                        "analysis_outputs": analysis_outputs_entry,
                    }
                )
                solution_mapping.append((sample_idx, sample))
        print(f"sending batch [{batch_start}-{batch_end}] with {len(all_solutions)} solutions")
        analysis_outputs: list[typing.Any] = []
        for i in range(0, len(all_solutions), chunk_size):
            chunk_solutions = all_solutions[i : i + chunk_size]
            chunk_inputs: list[dict[str, str]] = [{"code": typing.cast("str", s["code"])} for s in chunk_solutions]
            print(f"Processing chunk {i // chunk_size + 1}/{(len(all_solutions) + chunk_size - 1) // chunk_size}")
            tasks: list[typing.Coroutine[typing.Any, typing.Any, typing.Any]] = []

            async def process_with_retry(
                input_payload: dict[str, typing.Any],
                max_retries: int = 5,
                backoff: float = 2,
            ) -> typing.Any:
                retries = 0
                while retries < max_retries:
                    try:
                        return await code_analysis_chain.ainvoke(
                            input_payload,
                            config={"max_concurrency": 512},
                        )
                    except Exception as exc:
                        full_stop_exceptions = [
                            "insufficient balance",
                            "stopping processing at ",
                        ]
                        error_text = str(exc)
                        if any(stop in error_text.lower() for stop in full_stop_exceptions):
                            raise exc
                        retries += 1
                        if retries >= max_retries:
                            print(f"Failed after {max_retries} retries: {error_text}")
                            return {"error": error_text}
                        wait_time = backoff**retries
                        print(f"Retry {retries} after {wait_time}s due to: {error_text}")
                        await asyncio.sleep(wait_time)
                return {"error": "retry attempts exhausted without result"}

            for input_data in chunk_inputs:
                tasks.append(process_with_retry(input_data))
            # Wait for all tasks in this chunk to complete
            chunk_results = await asyncio.gather(*tasks)
            analysis_outputs.extend(chunk_results)
            # add a small delay between chunks to avoid rate limiting
            await asyncio.sleep(1)
        print(f"received batch [{batch_start}-{batch_end}] with {len(analysis_outputs)} outputs")
        results_by_sample: dict[int, dict[str, typing.Any]] = {}
        for solution_idx, output in enumerate(analysis_outputs):
            sample_idx, sample = solution_mapping[solution_idx]
            # handle both successful responses and error cases
            output_data = output.model_dump() if hasattr(output, "model_dump") else output
            analysis_output_list = typing.cast("list[typing.Any]", all_solutions[solution_idx]["analysis_outputs"])
            analysis_output_list.append(output_data)
            if sample_idx not in results_by_sample:
                results_by_sample[sample_idx] = sample.copy()
                results_by_sample[sample_idx]["solutions"] = []
            sample_solutions = typing.cast("list[dict[str, typing.Any]]", results_by_sample[sample_idx]["solutions"])
            sample_solutions.append(all_solutions[solution_idx])
        for sample_idx, processed_sample in results_by_sample.items():
            sample_output_path = output_dir_path / f"{sample_idx:06d}.json"
            with open(sample_output_path, "w") as fd:
                json.dump(processed_sample, fd, indent=2)


if __name__ == "__main__":
    pyine.utils.reprod.entrypoint_setup()
    asyncio.run(
        reprocess_code_samples(
            output_dir_path=pyine.data.taco.dataset_utils.get_new_repackaged_dataset_path(),
        ),
    )
