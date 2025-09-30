import inspect
import logging
import typing

import langchain_core.runnables
import pydantic
import transformers
import wandb

import pyine.configs.schemas
import pyine.data.datamodule
import pyine.evals.common
import pyine.evals.utils
import pyine.utils.langchain
import pyine.utils.llm_providers
import pyine.utils.transformers

logger = logging.getLogger(__name__)


class AppMainConfig(pydantic.BaseModel):
    """Trainer application main entrypoint configuration settings.

    Should apply to all trainers that intend to train/evaluate models.
    """

    model_config = pydantic.ConfigDict(extra="allow")
    """Pydantic model configuration (allow extra fields)."""

    datamodule_config: pydantic.SerializeAsAny[pyine.data.datamodule.BaseDataModuleConfig]
    """Configuration for the datamodule to use."""
    evals_config: pydantic.SerializeAsAny[pyine.evals.common.BaseEvalsConfig]
    """Configuration for the task evaluation strategy to use."""
    use_wandb_logging: bool = False
    """Whether to use W&B logging for the fine-tuning job (via the post-hoc sync approach)."""


def prepare_datamodule(
    config: AppMainConfig,
    runtime: pyine.configs.schemas.RuntimeConfig | None,
) -> pyine.data.datamodule.BaseDataModule:
    """Prepares the configured datamodule and returns it.

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
        target_subsets = (
            config.datamodule_config.train_subset_names
            + config.datamodule_config.valid_subset_names
            + config.datamodule_config.eval_subset_names
        )
        dm_stats = dm.get_stats(target_subsets)
        dm_stats = {f"dataset_stats/{k}": v for k, v in dm_stats.items()}
        runtime.wandb_run.summary.update(dm_stats)
        for eval_subset_name in config.datamodule_config.eval_subset_names:
            config.evals_config.define_metrics_for_wandb(
                wandb_run=runtime.wandb_run,
                prefix=f"evals/{eval_subset_name}",
            )
    return dm


async def evaluate_model(
    model: langchain_core.runnables.Runnable | transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer | None,
    datamodule: pyine.data.datamodule.BaseDataModule,
    config: AppMainConfig,
    runtime: pyine.configs.schemas.RuntimeConfig | None,
) -> dict[str, typing.Any]:
    """Evaluates the given model on the specified data subset using the internal evals config.

    Args:
        model: The model to evaluate, in either a langchain runnable or in HF-transformers format.
        tokenizer: The tokenizer to use for evaluation, if applicable (only for HF-T models).
        datamodule: The datamodule from which to load the evaluation data.
        config: The application configuration, which should contain the evals config.
        runtime: The runtime configuration, which may contain W&B run information.

    Returns:
        The evaluation results as a dictionary of metrics.
    """
    evaluation_results: dict[str, typing.Any] = {}
    if config.evals_config.eval_type is None:
        return evaluation_results
    if pyine.utils.transformers.is_hf_model(model):
        if tokenizer is None or not pyine.utils.transformers.is_hf_tokenizer(tokenizer):
            raise ValueError("invalid tokenizer (need to provide one to evaluate hf model")
        for eval_subset_name in config.datamodule_config.eval_subset_names:
            logger.info(f"running trained model evaluation on the {eval_subset_name} subset...")
            evaluation_result = await config.evals_config.evaluate_hf_model(
                model=model,
                tokenizer=tokenizer,
                datamodule=datamodule,
                eval_subset_name=eval_subset_name,
                verbose=True,
            )
            if inspect.isawaitable(evaluation_result):
                evaluation_result = await evaluation_result
            pyine.evals.utils.print_metrics(evaluation_result.metrics, eval_subset_name, logger.info)
            evaluation_results[eval_subset_name] = evaluation_result
    else:
        if not pyine.utils.langchain.is_invocable_chain(model):
            raise ValueError(f"invalid model ({type(model)})")
        if tokenizer is not None:
            raise NotImplementedError("tokenizer support in runnable chain eval is not implemented")
        for eval_subset_name in config.datamodule_config.eval_subset_names:
            logger.info(f"running chain evaluation on the {eval_subset_name} subset...")
            evaluation_result = await config.evals_config.evaluate_runnable_model(
                chain=model,
                datamodule=datamodule,
                eval_subset_name=eval_subset_name,
                verbose=True,
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
        config.evals_config.log_metrics(
            wandb_run=runtime.wandb_run,
            results_by_subset=evaluation_results,
        )
        for subset_name, subset_result in evaluation_results.items():
            config.evals_config.log_predictions(
                wandb_run=runtime.wandb_run,
                subset_name=subset_name,
                subset_results=subset_result,
            )
    return evaluation_results
