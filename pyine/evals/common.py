import enum
import pathlib
import typing

import pydantic
import transformers
import wandb

import pyine.data.datamodule
import pyine.evals.utils
import pyine.utils.llm_providers
import pyine.utils.parsing
import pyine.utils.transformers


class EvalType(enum.StrEnum):
    """Identifies the implemented and supported evaluation types."""

    # TODO: if this becomes a fundamental 'task definition' thing, move to `pyine.configs.schemas`?

    CODE_EXEC = enum.auto()
    """Code execution model evaluation pipeline.

    Under this task, a model is asked to interpret Python code and predict some execution outcome
    given input arguments. Depending on how the model is trained, it may misbehave under specific
    conditions (e.g. when shortcuts based on hints are available, or when specific keywords are
    used). Code execution evaluations are therefore useful to measure the capability of models in
    solving complex reasoning tasks in the presence/absence of misbehavior-triggering information.
    """
    CORRECTNESS = enum.auto()
    """Guardrail correctness evaluation pipeline.

    Under this task, a guardrail is asked to determine if a code execution prediction made by
    another model is correct or not. Assessing correctness requires in-depth analysis of either
    model activations or inputs/reasoning/outputs; some guardrails may also work in a passive
    fashion (in which case they are 'monitors'), while others may interact with the predictive
    models they intend to guard.
    """


class EvalResult(pydantic.BaseModel):
    """Base container for evaluation metrics and captured artifacts."""

    model_config = pydantic.ConfigDict(frozen=True)
    """Pydantic model configuration (immutable)."""

    metrics: pyine.evals.utils.MetricsDictType
    """Dictionary of aggregated evaluation metrics; keys are metric names, values are eval outcomes."""


class RunnableEvalConfig(pydantic.BaseModel):
    """Configuration for LangChain Runnable evaluations."""

    model_config = pydantic.ConfigDict(frozen=True)
    """Pydantic model configuration (immutable)."""

    parallel: bool = True
    """Whether to run the evaluations in parallel or sequentially."""
    max_workers: int | None = None
    """Maximum number of workers to use for parallel evaluations."""
    max_in_flight_jobs: int | None = 32
    """Maximum number of jobs to run in flight at any given time."""
    async_metrics_compute_rate: int = 100


class BaseEvalsConfig(pydantic.BaseModel):
    """Base configuration class for generic evaluation settings.

    Classes that inherit this base class are expected to override the `eval_type` field and
    the various methods below that perform task-specific evaluations by redirecting to task-specific
    implementations.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="allow")
    """Pydantic model configuration (immutable)."""

    eval_type: EvalType | None = None
    """Type of evaluation that should be conducted. If None, no evaluation occurs in the main app."""

    eval_runnable_config: RunnableEvalConfig = RunnableEvalConfig()
    """Configuration for LangChain Runnable evaluations."""

    category_extraction_config: pyine.evals.utils.SampleCategoryExtractionConfig | None = pydantic.Field(
        default_factory=pyine.evals.utils.SampleCategoryExtractionConfig,
    )
    """Configuration for extracting eval categories from sample data; set to None to disable."""

    # ---------------- public overridable evaluation methods ----------------

    async def evaluate_runnable_model(
        self,
        chain: pyine.evals.utils.InvocableModelChain,
        datamodule: pyine.data.datamodule.ConversationDataModule[typing.Any],
        eval_subset_name: str,
        verbose: bool = False,
    ) -> EvalResult:
        """Evaluates a LangChain Runnable chain using a specified data subset.

        Args:
            chain: The LangChain Runnable that will be used to generate responses.
            datamodule: The datamodule from which to load the evaluation data.
            eval_subset_name: The name of the subset to fetch from the datamodule and evaluate on.
            verbose: Whether to verbosely report progress.

        Returns:
            An `EvalResult` object, which contains a dictionary of metric values.
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
        """Evaluates a HuggingFace-Transformers-based model using the specified subset.

        Args:
            model: The pretrained HuggingFace-Transformers model to evaluate.
            tokenizer: The tokenizer of the above model to use for input/output transformations.
            datamodule: The datamodule from which to load the evaluation data.
            eval_subset_name: The name of the subset to fetch from the datamodule and evaluate on.
            verbose: Whether to verbosely report progress.

        Returns:
            An `EvalResult` object, which contains a dictionary of metric values.
        """
        if self.eval_type is None:
            return EvalResult(metrics={})
        raise NotImplementedError(f"evaluation type {self.eval_type} not implemented")

    async def evaluate_wrapped_model(
        self,
        wrapped_model: typing.Any | typing.Sequence[typing.Any],
        datamodule: pyine.data.datamodule.ConversationDataModule[typing.Any],
        eval_subset_name: str,
        verbose: bool = False,
    ) -> EvalResult:
        """Evaluates wrapped models (e.g. guardrails, monitors, ...).

        Unlike evaluate_runnable_model and evaluate_hf_model which evaluate a target model's
        generation quality, this method evaluates models that produce non-text outputs (such as
        classification decisions). The concrete semantics depend on the eval type.

        A single model or a sequence of models can be provided. When multiple models are given,
        they are treated as independent instances of the same modeling pipeline (e.g. guardrails
        trained with different seeds). The pipeline evaluates each independently, then aggregates
        results across them to produce cross-run statistics (mean, std, percentiles, hierarchical
        bootstrap CIs) that capture pipeline variance.

        Args:
            wrapped_model: A single wrapped model component, or a sequence of them for multi-run
                aggregation. The concrete type depends on the eval type (e.g. GuardrailScorer for
                correctness evals).
            datamodule: The datamodule from which to load the evaluation data.
            eval_subset_name: The name of the subset to fetch from the datamodule and evaluate on.
            verbose: Whether to verbosely report progress.

        Returns:
            An `EvalResult` object, which contains a dictionary of metric values.
        """
        if self.eval_type is None:
            return EvalResult(metrics={})
        raise NotImplementedError(f"evaluation type {self.eval_type} not implemented")

    def define_metrics_for_wandb(
        self,
        wandb_run: wandb.Run,
        prefix: str | None = None,
    ) -> None:
        """Registers evaluation metric definitions with a W&B run.

        This should be called once before logging any metrics, so that W&B can properly track them
        as summary metrics (e.g. with ``summary="max"``). Each eval type registers the metric names
        it will later emit via ``log_metrics``.

        Does nothing when ``eval_type`` is None (no evaluation configured).

        Args:
            wandb_run: The W&B run object where metric definitions should be registered.
            prefix: Optional prefix prepended to all metric names (e.g. an eval subset name).
        """
        if self.eval_type is None:
            return
        raise NotImplementedError(f"evaluation type {self.eval_type} not implemented")

    def log_metrics(
        self,
        wandb_run: wandb.Run,
        results_by_subset: dict[str, EvalResult],
        *,
        step: int | None = None,
    ) -> wandb.Table | None:
        """Log aggregated evaluation metrics to a W&B table.

        This builds a single cross-subset comparison table (one row per subset) and logs it
        under the ``benchmark/metrics_table`` key. Implementations may also write individual
        metrics to the run summary under ``benchmark/{subset_name}/{metric_name}`` keys for
        convenient programmatic access.

        For per-sample metrics, use `log_sample_metrics`. For qualitative inspection of
        individual predictions, use `log_predictions`.

        Args:
            wandb_run: Run object where the table should be logged.
            results_by_subset: Mapping of subset names to evaluation results.
            step: Optional W&B step override.

        Returns:
            The table that was logged (if any).

        See Also:
            log_sample_metrics: For per-sample metrics.
            log_predictions: For qualitative inspection of predictions.
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
        max_rows: int | None = None,
        max_text_length: int | None = None,
        step: int | None = None,
    ) -> wandb.Table | None:
        """Log a subset of model predictions to W&B (as tables) for qualitative inspection.

        This logs a limited number of predictions with full text (inputs, expected, predicted)
        for manual review and debugging. For comprehensive per-sample metrics analysis, use
        `log_sample_metrics` instead.

        The table should be logged under the ``benchmark/{subset_name}/predictions`` key.

        Args:
            wandb_run: Run object where the table should be logged.
            subset_name: Name of the evaluated subset.
            subset_results: Captured evaluation results for the subset.
            max_rows: Maximum number of prediction rows to log. None means no limit (default).
            max_text_length: Maximum length per text field before truncation. None means
                no truncation (default).
            step: Optional W&B step override.

        Returns:
            The table that was logged (if any).

        See Also:
            log_sample_metrics: For per-sample metrics (all samples).
            log_metrics: For subset-level aggregated metrics.
        """
        if self.eval_type is None:
            return None
        raise NotImplementedError(f"evaluation type {self.eval_type} not implemented")

    def log_sample_metrics(
        self,
        wandb_run: wandb.Run,
        subset_name: str,
        subset_results: EvalResult,
        *,
        step: int | None = None,
    ) -> wandb.Table | None:
        """Log per-sample metrics to W&B (as tables) for quantitative analyses.

        Unlike `log_predictions` (which logs a limited subset for qualitative inspection),
        this method logs sample-wise numerical metrics needed for quantitative analyses. No text
        fields are included, so there should be no need to limit/truncate these metrics.

        The table should be logged under the ``benchmark/{subset_name}/sample_metrics`` key, and
        can be fetched later using ``pyine.evals.code_exec.analysis.fetch_sample_metrics_table``
        for offline analysis.

        Args:
            wandb_run: Run object where the table should be logged.
            subset_name: Name of the evaluated subset.
            subset_results: Captured evaluation results for the subset.
            step: Optional W&B step override.

        Returns:
            The table that was logged (if any).

        See Also:
            log_predictions: For qualitative inspection of prediction text (limited samples).
            log_metrics: For subset-level aggregated metrics.
        """
        if self.eval_type is None:
            return None
        raise NotImplementedError(f"evaluation type {self.eval_type} not implemented")


class EvalExportConfig(pydantic.BaseModel):
    """Configuration for exporting evaluation results to disk as an LMDB dataset."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
    """Pydantic model configuration (immutable)."""

    output_path: pathlib.Path
    """Base directory for LMDB exports. Each subset creates a subdirectory."""
    store_aggregated_metrics: bool = True
    """Whether to store aggregated evaluation metrics as LMDB metadata."""


class GenerationEvalsConfig(BaseEvalsConfig):
    """Configuration for generation-based evaluation tasks.

    Extends BaseEvalsConfig with fields specific to text generation pipelines (HF models,
    runnable chains): generation parameters, multi-sample/Pass@K settings, batching, and
    inference server configuration.
    """

    eval_generation_config: pydantic.SerializeAsAny[pyine.utils.transformers.GenerationConfig] | None = None
    """Generation configuration args used when generating predictions in evals."""
    eval_generation_max_new_tokens_override: pydantic.PositiveInt | None = 1024
    """Optional override for maximum number of tokens that can be generated in evals."""
    eval_batch_size: pydantic.PositiveInt = 1
    """Batch size to use when generating predictions in evals."""
    eval_padding_side: typing.Literal["left", "right"] = "left"
    """Padding side to use in prompts when generating predictions in evals."""
    output_parsing_config: pyine.utils.parsing.ParsingConfig | None = None
    """Optional model output parsing config.

    When set, raw model outputs are parsed to extract structured fields (final_answer, reasoning)
    before evaluation.
    """
    num_attempts_per_sample: pydantic.PositiveInt = 1
    """Number of generation attempts per sample (K for Pass@K). When >1, enables multi-sample metrics."""
    pass_at_k_values: list[pydantic.PositiveInt] | None = None
    """K values for Pass@K computation. Auto-derived from num_attempts_per_sample when None."""
    sampling_temperature_override: float | None = None
    """Optional temperature override for HF generation when num_attempts_per_sample > 1."""
    sampling_top_p_override: float | None = None
    """Optional top-p (nucleus sampling) override for HF generation when num_attempts_per_sample > 1."""
    vllm_provider_config: pyine.utils.llm_providers.LLMProviderConfig | None = None
    """Provider configuration for vLLM server for model inference during evaluation."""
    disk_export_config: EvalExportConfig | None = None
    """Configuration for exporting evaluation results to disk as an LMDB dataset."""

    @pydantic.model_validator(mode="after")
    def _validate_multi_sample_config(self) -> "GenerationEvalsConfig":
        """Derives and validates pass_at_k_values from num_attempts_per_sample.

        When num_attempts_per_sample == 1, pass_at_k_values stays None (no Pass@K grouping or
        validation is needed for single-attempt evals). When > 1, auto-derives k values by
        repeatedly halving N (floored) down to 1, e.g. N=10 -> [1, 2, 5, 10]. Users can always
        override with an explicit list.
        """
        if self.pass_at_k_values is None and self.num_attempts_per_sample > 1:
            k_set: set[int] = set()
            val = self.num_attempts_per_sample
            while val >= 1:
                k_set.add(val)
                val = val // 2
            derived = sorted(k_set)
            object.__setattr__(self, "pass_at_k_values", derived)
        k_values = self.pass_at_k_values
        if k_values is not None:
            if not k_values:
                raise ValueError("pass_at_k_values must not be empty")
            if k_values != sorted(set(k_values)):
                raise ValueError(f"pass_at_k_values must be sorted and unique, got {k_values}")
            if any(k > self.num_attempts_per_sample for k in k_values):
                raise ValueError(
                    f"all pass_at_k_values must be <= num_attempts_per_sample "
                    f"({self.num_attempts_per_sample}), got {k_values}"
                )
        return self

    @property
    def use_vllm_server(self) -> bool:
        """Whether to use a vLLM server for inference."""
        return self.vllm_provider_config is not None

    @classmethod
    def for_pass_at_k(
        cls,
        num_attempts_per_sample: int = 10,
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_new_tokens: int = 10_000,
        **kwargs: typing.Any,
    ) -> "GenerationEvalsConfig":
        """Creates a config with literature-standard Pass@K defaults (nucleus sampling).

        IMPORTANT: The default parameter values defined here are the canonical settings for
        all standard evaluations and cross-run comparisons. Override them only when intentionally
        deviating from the standard protocol.

        Default values follow LiveCodeBench conventions for code generation evaluation. Explicit
        arguments override the built-in defaults; any additional keyword arguments are forwarded
        to the ``GenerationEvalsConfig`` constructor.

        If ``eval_generation_config`` is passed in ``kwargs``, it takes precedence over the
        generation config built from ``temperature``/``top_p``.

        Args:
            num_attempts_per_sample: Number of generation attempts per sample (K).
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability threshold.
            max_new_tokens: Maximum number of new tokens to generate.
            **kwargs: Additional fields forwarded to the config constructor.
        """
        defaults: dict[str, typing.Any] = {
            "num_attempts_per_sample": num_attempts_per_sample,
            "eval_generation_max_new_tokens_override": max_new_tokens,
            "sampling_temperature_override": temperature,
            "sampling_top_p_override": top_p,
            "eval_generation_config": pyine.utils.transformers.GenerationConfig(
                **{"do_sample": True, "temperature": temperature, "top_p": top_p},
            ),
        }
        return cls(**(defaults | kwargs))
