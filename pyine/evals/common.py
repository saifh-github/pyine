import concurrent.futures
import dataclasses
import typing

import langchain_core.messages
import langchain_core.runnables
import pydantic
import tqdm
import wandb

import pyine.data.datamodule
import pyine.evals.utils
import pyine.organisms.datamodules.utils.samples
import pyine.utils.concurrency
import pyine.utils.llm_providers
import pyine.utils.reprod


class EvaluationArtifact(pydantic.BaseModel):
    """Evaluation artifact resulting from a single data sample."""

    model_config = pydantic.ConfigDict(frozen=True, arbitrary_types_allowed=True)
    """Pydantic model configuration (immutable)."""

    sample: pyine.organisms.datamodules.utils.samples.SampleData
    """Sample data associated with the prediction."""
    eval_result: pyine.evals.utils.SampleEval
    """Evaluation result associated with the prediction."""

    @property
    def identifier(self) -> str:
        """Unique sample id associated with the data sample (used for lookups)."""
        return self.sample.identifier

    @pydantic.model_validator(mode="after")
    def _post_validation(self) -> "EvaluationArtifact":
        """Validates inter-field attributes."""
        assert self.sample.identifier == self.eval_result.identifier, "sample id mismatch"
        assert self.eval_result.llm_score is None or isinstance(self.eval_result.llm_score, float), "invalid llm score"
        return self


class EvaluationResult(pydantic.BaseModel):
    """Container for evaluation metrics and captured artifacts."""

    model_config = pydantic.ConfigDict(frozen=True)
    """Pydantic model configuration (immutable)."""

    metrics: dict[str, float | int | str]
    """Dictionary of aggregated evaluation metrics; keys are metric names, values are eval outcomes."""
    artifacts: list[EvaluationArtifact]
    """List of captured evaluation artifacts (include sample data and eval result)."""

    @property
    def identifiers(self) -> list[str]:
        """Returns a list of sample identifiers associated with the evaluation results."""
        return [s.identifier for s in self.artifacts]


async def evaluate_langchain_runnable_on_subset(
    chain: langchain_core.runnables.Runnable,
    parser: pyine.organisms.datamodules.utils.samples.SampleBuilder,
    parallel: bool = True,
    max_workers: int | None = None,
    max_in_flight_jobs: int | None = 32,
    async_metrics_compute_rate: int = 100,
    llm_grader_provider_config: pyine.utils.llm_providers.LLMProviderConfig | None = None,
    verbose: bool = False,
) -> EvaluationResult:
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
        EvaluationResult: Aggregated metrics and captured prediction artifacts.
    """
    evaluator = pyine.evals.utils.OutcomeEvaluator(llm_provider_config=llm_grader_provider_config)
    token_usage = None
    sample_data_store: dict[str, pyine.organisms.datamodules.utils.samples.SampleData] = {}

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
            sample_data_store[sample.identifier] = sample
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
            sample_data_store[sample.identifier] = sample
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

    metrics = await _get_metrics()
    artifacts: list[EvaluationArtifact] = []
    for sample_eval in evaluator.results:
        assert sample_eval.identifier in sample_data_store, "missing sample data for evaluation?"
        artifacts.append(EvaluationArtifact(sample=sample_data_store[sample_eval.identifier], eval_result=sample_eval))
    return EvaluationResult(metrics=metrics, artifacts=artifacts)


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


def log_eval_metrics_table(
    wandb_run: wandb.Run,
    results_by_subset: dict[str, EvaluationResult],
    *,
    table_key: str = "evals/metrics_table",
    step: int | None = None,
) -> wandb.Table:
    """Log aggregated evaluation metrics to a W&B table.

    Args:
        wandb_run: Run object where the table should be logged.
        results_by_subset: Mapping of subset names to evaluation results.
        table_key: Key under which the table will be logged.
        step: Optional W&B step override.

    Returns:
        The table that was logged to the run.
    """
    metric_names: set[str] = set()
    for subset_result in results_by_subset.values():
        metric_names.update(subset_result.metrics.keys())
    ordered_metric_names = sorted(metric_names)
    columns = ["subset", *ordered_metric_names]
    table = wandb.Table(columns=columns)
    for subset_name, subset_result in results_by_subset.items():
        row: list[typing.Any] = [subset_name]
        for metric_name in ordered_metric_names:
            row.append(subset_result.metrics.get(metric_name))
        table.add_data(*row)
    if step is None:
        wandb_run.log({table_key: table})  # noqa
    else:
        wandb_run.log({table_key: table}, step=step)  # noqa
    return table


def log_sample_predictions_table(
    wandb_run: wandb.Run,
    subset_name: str,
    artifacts: list[EvaluationArtifact],
    *,
    table_key: str | None = None,
    max_rows: int = 32,
    include_only_incorrect: bool = False,
    max_text_length: int = 512,
    step: int | None = None,
) -> wandb.Table:
    """Log a subset of model predictions to W&B for qualitative inspection.

    Args:
        wandb_run: Run object where the table should be logged.
        subset_name: Name of the evaluated subset.
        artifacts: Captured evaluation artifacts for the subset.
        table_key: Optional override for the W&B key under which the table is logged.
            If not provided, the table will be logged to the `evals/<subset_name>/predictions` key.
        max_rows: Maximum number of prediction rows to log.
        include_only_incorrect: Whether to restrict the table to incorrect predictions.
        max_text_length: Maximum length per text field before truncation.
        step: Optional W&B step override.

    Returns:
        The table that was logged to the run.
    """

    if table_key is None:
        table_key = f"evals/{subset_name}/predictions"
    incorrect: list[EvaluationArtifact] = []
    correct: list[EvaluationArtifact] = []
    for item in artifacts:
        # we use the hard match result by default for correct/incorrect labeling
        if item.eval_result.hard_match:
            correct.append(item)
        else:
            incorrect.append(item)
    ordered_predictions: list[EvaluationArtifact]
    if include_only_incorrect:
        ordered_predictions = incorrect
    else:
        ordered_predictions = [*incorrect, *correct]
    selected_predictions = ordered_predictions[:max_rows]

    def _truncate_text(
        value: str,
        max_length: int,
    ) -> str:
        if len(value) <= max_length:
            return value
        if max_length <= 3:
            return value[:max_length]
        return f"{value[: max_length - 3]}..."

    table = wandb.Table(
        columns=[
            "subset",
            "identifier",
            "code_type",
            "output_type",
            "inputs",
            "expected_output",
            "predicted_output",
            "hard_match",
            "soft_match",
            "grader_score",
            "tags",
        ]
    )
    for prediction in selected_predictions:
        row = [
            subset_name,
            prediction.identifier,
            str(prediction.sample.code_type),
            str(prediction.sample.output_type),
            _truncate_text(prediction.sample.inputs, max_text_length),
            _truncate_text(prediction.eval_result.expected, max_text_length),
            _truncate_text(prediction.eval_result.predicted, max_text_length),
            prediction.eval_result.hard_match,
            dataclasses.asdict(prediction.eval_result.soft_match),
            prediction.eval_result.llm_score,
            ", ".join(prediction.eval_result.tags),
        ]
        table.add_data(*row)
    if step is None:
        wandb_run.log({table_key: table})  # noqa
    else:
        wandb_run.log({table_key: table}, step=step)  # noqa
    return table
