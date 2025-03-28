import json
import pathlib

import src.prompts.code_analysis_prompt
import src.taco_dataset_reader
import src.utils.code_validation


async def reprocess_code_samples(
    output_dir_path: pathlib.Path,
    batch_size: int = 32,
    max_concurrency: int = 1000,
) -> None:
    """
    Process a given dataset and save all valid samples to individual files using async batch processing.

    Assumes the dataset samples contain multiple 'solutions' (i.e., multiple Python code snippets).

    Args:
        output_dir_path: path to save the JSON files containing the updated data samples.
        batch_size: number of samples to process in each batch.
        max_concurrency: maximum number of concurrent requests.
    """

    output_dir_path.mkdir(parents=True, exist_ok=True)
    dataset_reader = src.taco_dataset_reader.TacoDatasetReader()
    code_analysis_chain = src.prompts.code_analysis_prompt.get_deepseek_code_analysis_chain()
    total_samples = len(dataset_reader)  # min(10, len(dataset_reader))
    print(f"will process {total_samples} samples:")
    for batch_start in range(0, total_samples, batch_size):
        batch_end = min(batch_start + batch_size, total_samples)
        batch_indices = [  # keep only sample indices that are not already done
            idx for idx in range(batch_start, batch_end)
            if not (output_dir_path / f"{idx:06d}.json").exists()
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
                    src.utils.code_validation.validate_code(solution)
                except Exception as e:
                    validation_errors.append(str(e))
                all_solutions.append({
                    "orig_code": orig_code,
                    "code": solution,
                    "validation_errors": validation_errors,
                    "analysis_outputs": analysis_outputs,
                })
                solution_mapping.append((sample_idx, sample))

        print(f"sending batch [{batch_start}-{batch_end}] with {len(all_solutions)} solutions")
        analysis_outputs = await code_analysis_chain.abatch(
            [{"code": s["code"]} for s in all_solutions],
            config={"max_concurrency": max_concurrency}
        )
        print(f"received batch [{batch_start}-{batch_end}] with {len(analysis_outputs)} outputs")

        results_by_sample = {}
        for solution_idx, output in enumerate(analysis_outputs):
            sample_idx, sample = solution_mapping[solution_idx]
            all_solutions[solution_idx]["analysis_outputs"].append(output.model_dump())
            if sample_idx not in results_by_sample:
                results_by_sample[sample_idx] = sample.copy()
                results_by_sample[sample_idx]["solutions"] = []
            results_by_sample[sample_idx]["solutions"].append(all_solutions[solution_idx])

        for sample_idx, processed_sample in results_by_sample.items():
            sample_output_path = output_dir_path / f"{sample_idx:06d}.json"
            with open(sample_output_path, "w") as fd:
                json.dump(processed_sample, fd, indent=2)  # noqa
