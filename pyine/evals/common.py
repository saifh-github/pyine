import enum
import typing

import pydantic
import transformers
import wandb

import pyine.data.datamodule
import pyine.evals.utils
import pyine.utils.transformers


class EvalType(enum.StrEnum):
    """Identifies the implemented and supported evaluation types."""

    # TODO: if this becomes a fundamental 'task definition' thing, move to `pyine.configs.schemas`?

    CODE_EXEC = enum.auto()
    """Code execution model evaluation pipeline."""
    # TODO: add more here later


class EvalResult(pydantic.BaseModel):
    """Container for evaluation metrics and captured artifacts."""

    model_config = pydantic.ConfigDict(frozen=True)
    """Pydantic model configuration (immutable)."""

    metrics: pyine.evals.utils.MetricsDictType
    """Dictionary of aggregated evaluation metrics; keys are metric names, values are eval outcomes."""


class RunnableEvalConfig(pydantic.BaseModel):
    """Configuration for runnable evaluations."""

    model_config = pydantic.ConfigDict(frozen=True)
    """Pydantic model configuration (freezes the dataclass)."""

    parallel: bool = True
    """Whether to run the evaluations in parallel or sequentially."""
    max_workers: int | None = None
    """Maximum number of workers to use for parallel evaluations."""
    max_in_flight_jobs: int | None = 32
    """Maximum number of jobs to run in flight at any given time."""
    async_metrics_compute_rate: int = 100


class BaseEvalsConfig(pydantic.BaseModel):
    """Base configuration class for evaluation settings.

    Classes that inherit this base class are expected to override the `eval_type` field and
    the various methods below that perform task-specific evaluations.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="allow")
    """Pydantic model configuration (freezes the dataclass)."""

    eval_type: EvalType | None = None
    """Type of evaluation that should be conducted. If None, no evaluation occurs in the main app."""

    eval_generation_config: pydantic.SerializeAsAny[pyine.utils.transformers.GenerationConfig] | None = pydantic.Field(
        default=None,
        description="Generation configuration args used when generating predictions in evals.",
    )
    eval_generation_max_new_tokens_override: pydantic.PositiveInt | None = pydantic.Field(
        default=1024,
        description="Optional override for maximum number of tokens that can be generated in evals.",
    )
    eval_batch_size: pydantic.PositiveInt = pydantic.Field(
        default=1,
        description="Batch size to use when generating predictions in evals.",
    )
    eval_padding_side: typing.Literal["left", "right"] = pydantic.Field(
        default="left",
        description="Padding side to use when generating predictions in evals.",
    )
    eval_runnable_config: RunnableEvalConfig = pydantic.Field(
        default=RunnableEvalConfig(),
        description="Configuration for runnable code execution evaluations.",
    )

    # ---------------- public overridable evaluation methods ----------------

    async def evaluate_runnable_model(
        self,
        chain: pyine.evals.utils.InvocableModelChain,
        datamodule: pyine.data.datamodule.ConversationDataModule[typing.Any],
        eval_subset_name: str,
        verbose: bool = False,
    ) -> EvalResult:
        """Evaluates a LangChain text prediction chain using the specified subset.

        Args:
            chain: The LangChain Runnable that will be used to generate model responses.
            datamodule: The datamodule from which to load the evaluation data.
            eval_subset_name: The name of the subset to fetch from the datamodule and evaluate on.
            verbose: Whether to verbosely report progress.

        Returns:
            The evaluation results, which contains a dictionary of metrics.
        """
        if self.eval_type is None:
            return EvalResult(metrics={})
        raise NotImplementedError(f"evaluation type {self.eval_type} not implemented")

    async def evaluate_hf_model(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        datamodule: pyine.data.datamodule.ConversationDataModule[typing.Any],
        eval_subset_name: str,
        verbose: bool = False,
    ) -> EvalResult:
        """Evaluates a HuggingFace-Transformers model using the specified subset.

        Args:
            model: The pretrained HuggingFace-Transformers model to evaluate.
            tokenizer: The tokenizer of the above model to use for input/output transformations.
            datamodule: The datamodule from which to load the evaluation data.
            eval_subset_name: The name of the subset to fetch from the datamodule and evaluate on.
            verbose: Whether to verbosely report progress.

        Returns:
            The evaluation results, which contains a dictionary of metrics.
        """
        if self.eval_type is None:
            return EvalResult(metrics={})
        raise NotImplementedError(f"evaluation type {self.eval_type} not implemented")

    def define_metrics_for_wandb(
        self,
        wandb_run: wandb.Run,
        prefix: str | None = None,
    ) -> None:
        """Defines the evaluation metrics for the given wandb run."""
        if self.eval_type is None:
            return
        raise NotImplementedError(f"evaluation type {self.eval_type} not implemented")

    def log_metrics(
        self,
        wandb_run: wandb.Run,
        results_by_subset: dict[str, EvalResult],
        *,
        table_key: str = "predict/metrics_table",
        step: int | None = None,
    ) -> wandb.Table | None:
        """Log aggregated evaluation metrics to a W&B table.

        Args:
            wandb_run: Run object where the table should be logged.
            results_by_subset: Mapping of subset names to evaluation results.
            table_key: Key under which the table will be logged.
            step: Optional W&B step override.

        Returns:
            The table that was logged (if any).
        """
        if self.eval_type is None:
            return None
        raise NotImplementedError(f"evaluation type {self.eval_type} not implemented")

    def log_predictions(
        self,
        wandb_run: wandb.Run,
        subset_name: str,
        subset_results: EvalResult,
        *,
        table_key: str | None = None,
        max_rows: int = 32,
        max_text_length: int = 512,
        step: int | None = None,
    ) -> wandb.Table | None:
        """Log a subset of model predictions to W&B for qualitative inspection.

        Args:
            wandb_run: Run object where the table should be logged.
            subset_name: Name of the evaluated subset.
            subset_results: Captured evaluation results for the subset.
            table_key: Optional override for the W&B key under which the table is logged.
                If not provided, the table will be logged to the `evals/<subset_name>/predictions` key.
            max_rows: Maximum number of prediction rows to log.
            max_text_length: Maximum length per text field before truncation.
            step: Optional W&B step override.

        Returns:
            The table that was logged (if any)).
        """
        if self.eval_type is None:
            return None
        raise NotImplementedError(f"evaluation type {self.eval_type} not implemented")
