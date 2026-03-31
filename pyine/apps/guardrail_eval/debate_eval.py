"""Standalone Hydra entrypoint for LLM debate guardrail evaluation.

Runs the correctness evaluation pipeline with a multi-turn LLM debate
as the guardrail scorer. No training step - the scorer is built directly from
interrogator and responder LLM provider configs.

CLI usage::

    python -m pyine.apps.guardrail_eval.debate_eval \\
        +experiment=guardrail/debate_eval_openai
"""

from __future__ import annotations

import asyncio
import logging
import typing

import pyine.configs.schemas
import pyine.evals.common
import pyine.evals.correctness._impl as correctness_impl
import pyine.evals.correctness.datamodule as correctness_datamodule
import pyine.evals.persistence
import pyine.evals.utils
import pyine.utils.reprod
from pyine.apps.guardrail_eval.debate_eval_configs import (
    DebateEvalAppConfig,
    register_hydra_configs,
)
from pyine.guardrails.llm_debate.scorer import DebateGuardrailScorer

logger = logging.getLogger(__name__)


async def main(
    config: DebateEvalAppConfig,
    runtime: pyine.configs.schemas.RuntimeConfig | None = None,
) -> None:
    """Main entrypoint for LLM debate guardrail evaluation.

    Follows the same boilerplate pattern as existing evaluation apps
    (e.g. prompted_llm_eval.py:main): entrypoint_setup, W&B run
    init, evaluation loop, finalize.
    """
    # 0. Setup (seed, logging, optional W&B run init)
    wandb_init_kwargs: dict[str, typing.Any] = {}
    if config.wandb_project is not None:
        wandb_init_kwargs["project"] = config.wandb_project

    pyine.utils.reprod.entrypoint_setup(
        runtime_config=runtime,
        use_wandb_logging=config.use_wandb_logging,
        wandb_init_kwargs=wandb_init_kwargs or None,
        main_config=config,
    )

    if runtime is not None and runtime.dry_run:
        logger.info("dry run mode -- skipping")
        return

    # 1. Build scorer from debate config
    scorer = DebateGuardrailScorer(config.guardrail_config)
    logger.info(
        "built debate scorer: interrogator=%s, responder=%s, max_turns=%d",
        config.guardrail_config.interrogator_provider.provider,
        config.guardrail_config.responder_provider.provider,
        config.guardrail_config.max_debate_turns,
    )

    # 2. Prepare datamodule (loads LMDB, builds guardrail splits)
    datamodule = config.evals_config.prepare_eval_datamodule(None)
    assert isinstance(datamodule, correctness_datamodule.CorrectnessDataModule)
    eval_subset_names = datamodule.config.eval_subset_names
    logger.info("will evaluate on %d subset(s): %s", len(eval_subset_names), eval_subset_names)

    # 3. Register W&B metrics if logging
    if config.use_wandb_logging and runtime is not None and runtime.wandb_run is not None:
        config.evals_config.define_metrics_for_wandb(
            wandb_run=runtime.wandb_run,
            eval_subset_names=eval_subset_names,
        )

    # 4. Run evaluation on each configured subset
    evaluation_results: dict[str, pyine.evals.common.EvalResult] = {}
    for subset_name in eval_subset_names:
        logger.info("evaluating on subset: %s", subset_name)
        result = await correctness_impl.evaluate_guardrail_replicas(
            config=config.evals_config,
            guardrails=[scorer],
            datamodule=datamodule,
            eval_subset_name=subset_name,
        )
        pyine.evals.utils.print_metrics(result.metrics, subset_name, logger.info)
        evaluation_results[subset_name] = result

        if config.evals_config.result_dump_dir is not None:
            dump_path = pyine.evals.persistence.build_result_dump_path(
                dump_dir=config.evals_config.result_dump_dir,
                eval_subset_name=subset_name,
            )
            pyine.evals.persistence.save_eval_result(result, path=dump_path)
            logger.info("saved eval result to %s", dump_path)

    # 5. Log results to W&B
    if config.use_wandb_logging and evaluation_results:
        if runtime is None or runtime.wandb_run is None:
            raise RuntimeError("runtime configuration with wandb run must be provided when logging to wandb")
        logger.info("logging evaluation results to wandb...")
        config.evals_config.log_metrics(
            wandb_run=runtime.wandb_run,
            results_by_subset=evaluation_results,
        )

    # 6. Log scorer metadata
    scorer_metadata = scorer.get_metadata()
    logger.info("scorer metadata: %s", scorer_metadata)

    # 7. Finalize
    if runtime is not None:
        runtime.finalize()

    logger.info("LLM debate guardrail evaluation complete")


def async_debate_eval_main_wrapper(
    config: DebateEvalAppConfig,
    runtime: pyine.configs.schemas.RuntimeConfig | None = None,
) -> None:
    """Synchronous wrapper around the async main."""
    asyncio.run(main(config=config, runtime=runtime))


if __name__ == "__main__":
    import pyine.apps.trainers.common

    pyine.apps.trainers.common.hydra_main(
        eval_type=pyine.evals.common.EvalType.CORRECTNESS,
        hydra_config_registration_fn=register_hydra_configs,
        async_main_wrapper=async_debate_eval_main_wrapper,
    )
