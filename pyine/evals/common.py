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
import wandb


async def evaluate_model_on_subset(
    chain: langchain_core.runnables.Runnable,
    parser: pyine.organisms.datamodules.utils.samples.SampleBuilder,
    parallel: bool = True,
    max_workers: int | None = None,
    max_in_flight_jobs: int | None = 32,
    async_metrics_compute_rate: int = 100,
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
        async_metrics_compute_rate: Number of iterations between each metrics computation pass
            (which gathers all potential LLM grading results, blocking until they are all obtained).
        llm_grader_provider_config: Configuration for the LLM grader provider (if needed).
        verbose: Whether to verbosely report progress.

    Returns:
        A dictionary containing the resulting evaluation metrics.
    """
    evaluator = pyine.evals.utils.OutcomeEvaluator(llm_provider_config=llm_grader_provider_config)
    token_usage = None

    async def _get_metrics() -> dict[str, float | int | str]:
        nonlocal token_usage
        output_metrics: dict[str, float | int | str] = await evaluator.compute_metrics()
        if token_usage is None:
            token_usage = pyine.evals.utils.TokenUsageInfo.get_default()
        output_metrics.update({f"token_usage/{k}": v for k, v in token_usage.asdict().items()})
        return output_metrics

    sample_idxs = list(range(len(parser)))
    if not parallel:
        wrapped_sample_idxs = tqdm.tqdm(sample_idxs, disable=not verbose, desc="evaluating")
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
        prog_bar = tqdm.tqdm(total=len(sample_idxs), disable=not verbose, desc="waiting for results")

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
            prog_bar.update(1)
            if token_usage is None:
                token_usage = pyine.evals.utils.parse_token_usage_from_response(response)
            else:
                token_usage += pyine.evals.utils.parse_token_usage_from_response(response)

        async def _progress_callback(_: list[typing.Hashable], completed: list[typing.Hashable]) -> None:
            if verbose and completed and len(completed) % async_metrics_compute_rate == 0:
                prog_bar.write(f"progress report (completed {len(completed)}): {await _get_metrics()}")

        await pyine.utils.concurrency.run_with_sliding_window(
            input_items=sample_idxs,
            submit_one=_submit_one,
            process_result=_process_result,
            progress_callback=_progress_callback,
            max_workers=max_workers,
            max_in_flight_jobs=max_in_flight_jobs,
        )
        prog_bar.close()

    return await _get_metrics()


def define_metrics_for_wandb(
    wandb_run: wandb.Run,
    prefix: str | None = None,
) -> None:
    """Defines the evaluation metrics for the given wandb run."""
    for metric_name in pyine.evals.utils.OutcomeEvaluator.get_metric_names():
        metric_name = f"{prefix}/{metric_name}" if prefix else metric_name
        wandb_run.define_metric(
            name=metric_name,
            summary="max",  # outcome eval metrics are always max
            step_metric="global_step",
        )  # noqa
    for metric_name in pyine.evals.utils.TokenUsageInfo.get_metric_names():
        metric_name = f"{prefix}/{metric_name}" if prefix else metric_name
        wandb_run.define_metric(
            name=metric_name,
            summary="mean",  # token usage metrics make sense as averaged over full runs
            step_metric="global_step",
        )  # noqa
