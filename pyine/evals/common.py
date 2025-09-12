import concurrent.futures
import typing

import langchain_core.messages
import langchain_core.runnables
import tqdm

import pyine.data.datamodule
import pyine.evals.utils
import pyine.organisms.datamodules.utils.samples
import pyine.utils.concurrency
import pyine.utils.llm_providers
import pyine.utils.reprod


def evaluate_model_on_subset(
    chain: langchain_core.runnables.Runnable,
    parser: pyine.organisms.datamodules.utils.samples.SampleBuilder,
    parallel: bool = True,
    max_workers: int | None = None,
    max_in_flight_jobs: int | None = 5_000,
    llm_grader_provider_config: pyine.utils.llm_providers.LLMProviderConfig | None = None,
    verbose: bool = False,
) -> dict[str, float | int | str]:
    """Evaluate a model using the specified data parser, returning evaluation metrics.

    The model is expected to be already wrapped inside a LangChain Runnable whose invocation
    returns a LangChain AIMessage object directly.

    Args:
        chain: The LangChain Runnable that will be used to generate model responses.
        parser: The SampleBuilder object that will be used to generate samples for evaluation.
        parallel: Whether to evaluate the model in parallel.
        max_workers: Maximum number of workers to use for parallel evaluation. If None, uses the
            number of CPUs.
        max_in_flight_jobs: Maximum number of in-flight jobs to keep in the thread pool executor.
        llm_grader_provider_config: Configuration for the LLM grader provider (if needed).
        verbose: Whether to verbosely report progress.

    Returns:
        A dictionary containing the resulting evaluation metrics.
    """
    evaluator = pyine.evals.utils.OutcomeEvaluator(llm_provider_config=llm_grader_provider_config)
    token_usage = None

    def _get_metrics() -> dict[str, float | int | str]:
        nonlocal token_usage
        output_metrics: dict[str, float | int | str] = evaluator.compute_metrics()
        if token_usage is None:
            token_usage = pyine.evals.utils.TokenUsageInfo.get_default()
        output_metrics.update(token_usage.asdict())
        return output_metrics

    sample_idxs = list(range(len(parser)))
    wrapped_sample_idxs = tqdm.tqdm(sample_idxs, disable=not verbose, desc="evaluating")
    if not parallel:
        for sample_idx in wrapped_sample_idxs:
            sample = parser[sample_idx]
            assert isinstance(sample, pyine.organisms.datamodules.utils.samples.SampleData)
            response = chain.invoke(sample._asdict())
            assert isinstance(response, langchain_core.messages.AIMessage)
            evaluator.add_sample(
                identifier=sample.identifier,
                expected=sample.expected_output,
                predicted=response.content,
                tags=sample.get_tag_list(),
            )
            if token_usage is None:
                token_usage = pyine.evals.utils.parse_token_usage_from_response(response)
            else:
                token_usage += pyine.evals.utils.parse_token_usage_from_response(response)
    else:  # parallel
        sample_lut: dict[int, pyine.organisms.datamodules.utils.samples.SampleData] = {}

        def _submit_one(
            sample_idx: typing.Hashable,
            executor: concurrent.futures.Executor,
        ) -> concurrent.futures.Future:
            sample_idx = typing.cast(int, sample_idx)
            sample = parser[sample_idx]
            assert isinstance(sample, pyine.organisms.datamodules.utils.samples.SampleData)
            assert sample_idx not in sample_lut
            sample_lut[sample_idx] = sample
            return executor.submit(
                chain.invoke,
                sample._asdict(),
            )

        def _process_result(sample_idx: typing.Hashable, response: langchain_core.messages.AIMessage):
            nonlocal token_usage
            sample_idx = typing.cast(int, sample_idx)
            sample = sample_lut.pop(sample_idx)
            assert isinstance(response, langchain_core.messages.AIMessage)
            evaluator.add_sample(
                identifier=sample.identifier,
                expected=sample.expected_output,
                predicted=response.content,
                tags=sample.get_tag_list(),
            )
            if token_usage is None:
                token_usage = pyine.evals.utils.parse_token_usage_from_response(response)
            else:
                token_usage += pyine.evals.utils.parse_token_usage_from_response(response)

        def _progress_callback(_: list[typing.Hashable], completed: list[typing.Hashable]) -> None:
            if verbose and len(completed) % 100 == 0:  # print progress report every 100 tests
                wrapped_sample_idxs.write(f"progress report: {_get_metrics()}")

        pyine.utils.concurrency.run_with_sliding_window(
            input_items=wrapped_sample_idxs,
            submit_one=_submit_one,
            process_result=_process_result,
            progress_callback=_progress_callback,
            max_workers=max_workers,
            max_in_flight_jobs=max_in_flight_jobs,
        )

    return _get_metrics()
