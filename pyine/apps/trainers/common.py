import inspect
import logging
import typing

import pydantic
import wandb

import pyine.configs.schemas
import pyine.data.datamodule
import pyine.evals.common
import pyine.evals.utils
import pyine.utils.llm_providers

logger = logging.getLogger(__name__)


class AppMainConfig(pydantic.BaseModel):
    """Trainer application main entrypoint configuration settings.

    Should apply to all trainers that intend to train models to perform code execution.
    """

    datamodule_config: pyine.data.datamodule.ConversationDataModuleConfig
    """Configuration for the datamodule to use."""
    llm_grader_provider_config: pyine.utils.llm_providers.LLMProviderConfig | None = None
    """Configuration for the LLM grader provider to use. If not specified, skips LLM grader evals."""
    train_subset_names: list[str] = pydantic.Field(default=["train"], min_length=1)
    """Subset names to train on."""
    valid_subset_names: list[str] = pydantic.Field(default=["valid"], min_length=1)
    """Subset names to validate on."""
    eval_subset_names: list[str] = ["valid", "valid_obfuscated"]
    """Subset names to use for final evaluations.

    Note: should be kept to 'validation' instead of 'testing' subsets until experiments are done,
    and all hyperparameters are permanently FIXED; if this sounds strange to you, refer to:
        https://en.wikipedia.org/wiki/Training,_validation,_and_test_data_sets
    """
    use_wandb_logging: bool = False
    """Whether to use W&B logging for the fine-tuning job (via the post-hoc sync approach)."""


def prepare_code_exec_datamodule(
    config: AppMainConfig,
    runtime: pyine.configs.schemas.RuntimeConfig | None,
) -> pyine.data.datamodule.ConversationDataModule:
    """Prepares the configured code execution datamodule and returns it.

    Args:
        config: The application configuration, which should contain the datamodule config.
        runtime: The runtime configuration, which may contain W&B run information.

    Returns:
        The instantiated, prepared, and set-up datamodule that is ready to provide data loaders.
    """
    logger.info("preparing datamodule and setting up parsers/loaders...")
    dm = config.datamodule_config.instantiate_datamodule(verbose=True)
    dm.prepare_data()
    dm.setup()
    if config.use_wandb_logging:
        assert runtime is not None and runtime.wandb_run is not None
        target_subsets = config.train_subset_names + config.valid_subset_names + config.eval_subset_names
        dm_stats = dm.get_stats(target_subsets)
        dm_stats = {f"dataset_stats/{k}": v for k, v in dm_stats.items()}
        runtime.wandb_run.summary.update(dm_stats)
        for eval_subset_name in config.eval_subset_names:
            pyine.evals.common.define_metrics_for_wandb(
                wandb_run=runtime.wandb_run,
                prefix=f"evals/{eval_subset_name}",
            )
    return dm


class EvaluationCallbackType(typing.Protocol):
    """Protocol used to represent a callback used to evaluate a model."""

    def __call__(
        self,
        eval_subset_name: str,
        datamodule: pyine.data.datamodule.ConversationDataModule,
        llm_grader_provider_config: pyine.utils.llm_providers.LLMProviderConfig | None,
    ) -> pyine.evals.common.EvaluationResult: ...


async def evaluate_code_execution_model(
    eval_callback: EvaluationCallbackType,
    datamodule: pyine.data.datamodule.ConversationDataModule,
    config: AppMainConfig,
    runtime: pyine.configs.schemas.RuntimeConfig | None,
) -> dict[str, pyine.evals.common.EvaluationResult]:
    """Evaluates the given model on the specified data subset."""
    evaluation_results: dict[str, pyine.evals.common.EvaluationResult] = {}
    for eval_subset_name in config.eval_subset_names:
        logger.info(f"running evaluation on the {eval_subset_name} subset...")
        evaluation_result = eval_callback(
            eval_subset_name=eval_subset_name,
            datamodule=datamodule,
            llm_grader_provider_config=config.llm_grader_provider_config,
        )
        if inspect.isawaitable(evaluation_result):
            evaluation_result = await evaluation_result
        pyine.evals.utils.print_metrics(evaluation_result.metrics, eval_subset_name, logger.info)
        evaluation_results[eval_subset_name] = evaluation_result
    if config.use_wandb_logging and evaluation_results:
        wandb_run_id = runtime.wandb_run_id
        if wandb_run_id is None:
            raise RuntimeError("wandb run ID is not set; should have been initialized already")
        if not hasattr(runtime.wandb_run, "log"):
            # reopen the run in case it was closed (e.g. like the openai integration always does)
            runtime.wandb_run = wandb.init(id=runtime.wandb_run_id, resume="must")
        logger.info(f"logging evaluation results to wandb run id: {wandb_run_id}...")
        pyine.evals.common.log_eval_metrics_table(
            wandb_run=runtime.wandb_run,
            results_by_subset=evaluation_results,
        )
        for subset_name, subset_result in evaluation_results.items():
            pyine.evals.common.log_sample_predictions_table(
                wandb_run=runtime.wandb_run,
                subset_name=subset_name,
                artifacts=subset_result.artifacts,
            )
            prefixed_metrics = {f"evals/{subset_name}/{k}": v for k, v in subset_result.metrics.items()}
            runtime.wandb_run.summary.update(prefixed_metrics)
    return evaluation_results
