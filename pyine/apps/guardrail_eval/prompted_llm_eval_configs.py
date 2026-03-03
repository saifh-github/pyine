"""Hydra-zen config builder for the prompted LLM guardrail evaluation app."""

from __future__ import annotations

import logging
import typing

import pydantic

import pyine.configs.base
import pyine.configs.schemas
import pyine.configs.searchpath
import pyine.configs.utils
import pyine.evals.common
import pyine.evals.correctness.configs
import pyine.utils.reprod
from pyine.guardrails.prompted_llm.configs import PromptedLLMGuardrailConfig  # noqa: TC001

logger = logging.getLogger(__name__)


class PromptedLLMEvalAppConfig(pydantic.BaseModel):
    """Standalone evaluation app for the prompted LLM guardrail.

    Requires no training — builds the scorer from an LLM provider config
    and runs the correctness evaluation pipeline on an LMDB dataset.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    guardrail_config: PromptedLLMGuardrailConfig
    """Prompted LLM guardrail configuration (LLM provider, prompt, concurrency)."""

    evals_config: pyine.evals.correctness.configs.CorrectnessEvalsConfig
    """Correctness eval pipeline configuration.

    Embeds CorrectnessDataModuleConfig which specifies the LMDB paths and split config.
    """

    use_wandb_logging: bool = False
    """Whether to log results to Weights & Biases."""
    wandb_project: str | None = None
    """W&B project name (only used when use_wandb_logging=True)."""


def _get_app_configs(
    group: str,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Generates and returns prompted LLM eval application configs for hydra zen storage."""
    evals_configs = pyine.evals.correctness.configs.get_evals_configs(
        group=f"{group}/evals_config",
    )

    app_main_config = pyine.configs.utils.make_config_description(
        PromptedLLMEvalAppConfig,
        name="base",
        group=group,
        description="Base settings for the prompted LLM guardrail eval app.",
        config={
            "populate_full_signature": True,
            "hydra_convert": "object",
        },
    )
    return [app_main_config, *evals_configs]


def register_hydra_configs(
    eval_type: pyine.evals.common.EvalType,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Registers prompted-LLM-eval-specific configs in hydra and returns config descriptions."""
    from pyine.apps.guardrail_eval.prompted_llm_eval import async_prompted_llm_eval_main_wrapper

    pyine.utils.reprod.load_dotenv()

    entrypoint_config = pyine.configs.utils.make_config_description(
        async_prompted_llm_eval_main_wrapper,
        name="entrypoint",
        group=None,
        description="Entrypoint settings for the prompted LLM guardrail eval app.",
        config={
            "populate_full_signature": True,
            "hydra_defaults": [
                "_self_",
                {"config": "base"},
                {"runtime": "default"},
                *pyine.configs.base.get_base_hydra_default_overrides(),
            ],
        },
    )

    store, base_configs = pyine.configs.base.get_base_store_and_configs("prompted_llm_eval")
    app_configs = _get_app_configs(group="config")
    configs_to_register = [entrypoint_config, *app_configs]

    external_configs = pyine.configs.searchpath.SearchPathPlugin.get_external_configs(
        app_name="prompted_llm_eval",
        eval_type=eval_type,
        entrypoint_config=entrypoint_config,
        app_configs=[*base_configs, *configs_to_register],
    )
    configs_to_register.extend(external_configs)

    for config in configs_to_register:
        assert config.name is not None
        store(
            typing.cast("typing.Any", config.config),
            name=config.name,
            group=config.group,
            package=config.package,
        )
    store.add_to_hydra_store(overwrite_ok=True)
    return [*base_configs, *configs_to_register]
