"""Standalone Hydra entrypoint for baseline sanity-check guardrail evaluation.

Runs the correctness evaluation pipeline with trivial scorers (constant, uniform random)
to establish reference metrics. No training needed; scorers are built directly from config.

CLI usage:

    python -m pyine.apps.guardrail_eval.baseline_eval \\
        +experiment=guardrail/baseline_eval
"""

from __future__ import annotations

import logging
import typing

import pyine.configs.schemas
import pyine.evals.common
import pyine.evals.correctness._impl as correctness_impl
import pyine.evals.correctness.datamodule as correctness_datamodule
import pyine.evals.correctness.types as correctness_types
import pyine.evals.utils
import pyine.utils.reprod
from pyine.apps.guardrail_eval.baseline_eval_configs import (
    BaselineEvalAppConfig,
    register_hydra_configs,
)

logger = logging.getLogger(__name__)


def main(
    config: BaselineEvalAppConfig,
    runtime: pyine.configs.schemas.RuntimeConfig | None = None,
) -> None:
    """Main entrypoint for baseline guardrail evaluation.

    Synchronous (not async) because ``evaluate_guardrail_types`` uses ``asyncio.run`` internally and
    cannot be called from within an already-running event loop.
    """
    # 0. setup (seed, logging, optional W&B run init)
    wandb_init_kwargs: dict[str, typing.Any] = {}
    if config.wandb_project is not None:
        wandb_init_kwargs["project"] = config.wandb_project
    pyine.utils.reprod.entrypoint_setup(
        runtime_config=runtime,
        use_wandb_logging=config.use_wandb_logging,
        wandb_init_kwargs=wandb_init_kwargs or None,
        main_config=config,
    )

    # 1. build guardrails_by_type from baseline configs
    guardrails_by_type: dict[str, typing.Sequence[correctness_types.GuardrailScorer]] = {}
    for baseline_config in config.baseline_configs:
        scorers = baseline_config.build_scorers()
        guardrails_by_type[baseline_config.name] = scorers
        logger.info(
            "built baseline '%s' (%s, %d replica(s))",
            baseline_config.name,
            baseline_config.scorer_type,
            len(scorers),
        )

    if runtime is not None and runtime.dry_run:
        logger.info("dry run mode -- skipping evaluation")
        return

    # 2. prepare datamodule
    datamodule = config.evals_config.prepare_eval_datamodule(None)
    assert isinstance(datamodule, correctness_datamodule.CorrectnessDataModule)
    eval_subset_names = datamodule.config.eval_subset_names
    logger.info("will evaluate on %d subset(s): %s", len(eval_subset_names), eval_subset_names)

    # 3. register W&B metrics (if logging)
    wandb_run = runtime.wandb_run if runtime is not None else None
    if config.use_wandb_logging and wandb_run is not None:
        config.evals_config.define_metrics_for_wandb(
            wandb_run=wandb_run,
            eval_subset_names=eval_subset_names,
        )

    # 4. evaluate each subset
    for subset_name in eval_subset_names:
        logger.info("evaluating on subset: %s", subset_name)
        results = correctness_impl.evaluate_guardrail_types(
            config=config.evals_config,
            guardrails_by_type=guardrails_by_type,
            datamodule=datamodule,
            eval_subset_name=subset_name,
            wandb_run=wandb_run,
            eval_type_parallelism=config.evals_config.eval_type_parallelism,
        )
        for type_name, result in results.items():
            pyine.evals.utils.print_metrics(result.metrics, f"{subset_name}/{type_name}", logger.info)

    # 5. log baseline metadata
    for baseline_config in config.baseline_configs:
        scorer_metadata = baseline_config.build_scorers()[0].get_metadata()
        logger.info("baseline '%s' metadata: %s", baseline_config.name, scorer_metadata)

    # 6. finalize
    if runtime is not None:
        runtime.finalize()
    logger.info("baseline guardrail evaluation complete")


def baseline_eval_main_wrapper(
    config: BaselineEvalAppConfig,
    runtime: pyine.configs.schemas.RuntimeConfig | None = None,
) -> None:
    """Synchronous wrapper for the baseline eval main (used by Hydra)."""
    main(config=config, runtime=runtime)


if __name__ == "__main__":
    import pyine.apps.trainers.common

    pyine.apps.trainers.common.hydra_main(
        eval_type=pyine.evals.common.EvalType.CORRECTNESS,
        hydra_config_registration_fn=register_hydra_configs,
        async_main_wrapper=baseline_eval_main_wrapper,
    )
