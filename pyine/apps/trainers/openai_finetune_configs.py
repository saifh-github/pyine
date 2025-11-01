"""Hydra-zen config builder for openai_finetune.py app.

If you execute this script directly, it will print all available experiment configs for this app.
"""

import asyncio
import itertools
import logging
import typing

import hydra_zen
import pydantic

import pyine.apps.trainers.common
import pyine.configs.base
import pyine.configs.schemas
import pyine.configs.searchpath
import pyine.configs.utils
import pyine.evals.common
import pyine.evals.configs
import pyine.organisms.datamodules.shortcuts_configs
import pyine.organisms.models.utils.openai
import pyine.utils.openai
import pyine.utils.reprod

logger = logging.getLogger(__name__)


class _HasOpenAIConfig(typing.Protocol):
    """Protocol for objects exposing an OpenAI configuration extractor."""

    def get_openai_config(self) -> dict[str, typing.Any]: ...


class OpenAIFineTuneAppMainConfig(pyine.apps.trainers.common.AppMainConfig):
    """Configuration for the OpenAI fine-tuner app's main function.

    Assembles the components required to fine-tune a model for code execution using a code execution
    traces datamodule.
    """

    openai_client_config: pydantic.SerializeAsAny[pyine.utils.openai.OpenAIClientConfig]
    """Configuration for the OpenAI client to use."""
    openai_finetuner_config: pydantic.SerializeAsAny[pyine.utils.openai.OpenAIFineTunerConfig]
    """Configuration for the OpenAI fine-tuner to use."""

    def needs_answers_in_train_dataset(self) -> bool:
        """Returns whether the model needs answers in its training dataset."""
        finetuner_params = typing.cast(
            "pyine.utils.openai.OpenAIFineTunerParamsConfig",
            self.openai_finetuner_config.params,
        )
        return finetuner_params.method.get("type", "") != "reinforcement"

    def supports_system_prompt(self) -> bool:
        """Returns whether the model to be fine-tuned supports the use of system prompts."""
        models_without_system_prompts = ["o1", "o3", "o4"]
        finetuner_params = typing.cast(
            "pyine.utils.openai.OpenAIFineTunerParamsConfig",
            self.openai_finetuner_config.params,
        )
        return not any(
            finetuner_params.base_model.startswith(model_prefix) for model_prefix in models_without_system_prompts
        )


def _async_main_wrapper(
    config: OpenAIFineTuneAppMainConfig,
    runtime: (pyine.configs.schemas.RuntimeConfig | None) = None,
    skip_fine_tuning: bool = False,
) -> None:
    """Wrapper for async main function."""
    import pyine.apps.trainers.openai_finetune as openai_finetune_app

    asyncio.run(openai_finetune_app.main(config=config, runtime=runtime, skip_fine_tuning=skip_fine_tuning))


def hydra_main(eval_type: pyine.evals.common.EvalType) -> None:
    """Hydra main entrypoint for the OpenAI fine-tuner app."""
    _ = register_hydra_configs(eval_type=eval_type)
    pyine.configs.base.register_searchpath_plugin()
    hydra_zen.zen(_async_main_wrapper).hydra_main(
        config_path=None,
        config_name="entrypoint",
        version_base=pyine.configs.base.target_hydra_version,
    )


def _openai_finetuner_method_config_getter(
    wrapped_fn: typing.Callable[..., _HasOpenAIConfig],
) -> typing.Callable[..., dict[str, typing.Any]]:
    """Wrapper for OpenAI fine-tuning method configs to return the OpenAI config object."""

    def _wrapper(
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> dict[str, typing.Any]:
        method_config = wrapped_fn(*args, **kwargs)
        return method_config.get_openai_config()

    return _wrapper


def _openai_finetuner_method_params_config_wrapper(
    wrapped_fn: typing.Callable[..., pyine.utils.openai.OpenAIFineTunerParamsConfig],
) -> typing.Callable[..., pyine.utils.openai.OpenAIFineTunerConfig]:
    """Wrapper for OpenAI fine-tuning method params configs to return a parent object."""

    def _wrapper(
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> pyine.utils.openai.OpenAIFineTunerConfig:
        params_config = wrapped_fn(*args, **kwargs)
        return pyine.utils.openai.OpenAIFineTunerConfig(params=params_config)

    return _wrapper


def _get_ft_params_configs(
    group: str,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Generates and returns fine-tuning parameter configs for hydra zen storage."""
    assert isinstance(group, str) and group
    default_sft_params_config = pyine.configs.utils.make_config_description(
        pyine.utils.openai.OpenAIFineTunerParamsConfig,
        name="openai_gpt-4.1-mini_default_sft",
        group=group,
        description=(
            "Default settings for OpenAI supervised fine-tuning using gpt-4.1-mini. These "
            "settings should be pretty cheap to run (especially compared to rlft)."
        ),
        config={
            "base_model": "gpt-4.1-mini-2025-04-14",
            "method": {"type": "supervised"},
            "seed": "${runtime.seed}",
            "suffix": "default-sft",
            "metadata": pyine.utils.reprod.get_reprod_metadata(include_installed_packages=False),
            # -------------
            "populate_full_signature": True,
            "hydra_convert": "object",
            "zen_wrappers": _openai_finetuner_method_params_config_wrapper,
        },
    )
    default_rlft_params_config = pyine.configs.utils.make_config_description(
        pyine.utils.openai.OpenAIFineTunerParamsConfig,
        name="openai_o4-mini_default_rlft",
        group=group,
        description=(
            "Default settings for OpenAI RL fine-tuning using o4-mini.\n\n"
            "NOTE: as of 2025-09-03, training o4-mini using these settings is VERY COSTLY, even for "
            "VERY TINY datasets (we're talking hundreds of dollars per run here, minimum). Do not "
            "use these settings unless you know what you are doing."
        ),
        config={
            "base_model": "o4-mini-2025-04-16",
            "seed": "${runtime.seed}",
            "suffix": "default-rlft",
            "metadata": pyine.utils.reprod.get_reprod_metadata(include_installed_packages=False),
            # -------------
            "populate_full_signature": True,
            "hydra_convert": "object",
            "hydra_defaults": [
                "_self_",
                {"override method": "default_pred_grader_rlft_method"},
            ],
            "zen_wrappers": _openai_finetuner_method_params_config_wrapper,
        },
    )
    default_pred_grader_rlft_method_config = pyine.configs.utils.make_config_description(
        pyine.organisms.models.utils.openai.PredGraderFineTuneMethodConfig,
        name="default_pred_grader_rlft_method",
        group=group + "/method",
        description="Default settings for OpenAI RL fine-tuning method (relies on an LLM grader).",
        config={
            # -------------
            "populate_full_signature": True,
            "hydra_convert": "object",
            "zen_wrappers": _openai_finetuner_method_config_getter,
        },
    )
    return [
        default_sft_params_config,
        default_rlft_params_config,
        default_pred_grader_rlft_method_config,
    ]


def _get_app_configs(
    eval_type: pyine.evals.common.EvalType,
    group: str,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Generates and returns application configs for hydra zen storage."""
    assert isinstance(group, str) and group
    app_main_config = pyine.configs.utils.make_config_description(
        OpenAIFineTuneAppMainConfig,
        name="base",
        group=group,
        description="Base settings for the OpenAI fine-tuner app.",
        config={
            # -------------
            "populate_full_signature": True,
            "hydra_convert": "object",
            "hydra_defaults": [
                "_self_",
                {"datamodule_config": "base"},
                {"openai_client_config": "default"},
                {"openai_finetuner_config": "openai_gpt-4.1-mini_default_sft"},
                {"evals_config": "base"},
            ],
        },
    )
    datamodule_configs = pyine.organisms.datamodules.shortcuts_configs.get_configs(
        eval_type=eval_type,
        group=f"{group}/datamodule_config",
    )
    openai_client_configs = pyine.configs.base.get_openai_client_configs(group=f"{group}/openai_client_config")
    ft_params_configs = _get_ft_params_configs(group=f"{group}/openai_finetuner_config")
    evals_configs = pyine.evals.configs.get_evals_configs(eval_type=eval_type, group=f"{group}/evals_config")
    # ... add more trainer configs here if needed
    return [
        app_main_config,
        *datamodule_configs,
        *openai_client_configs,
        *ft_params_configs,
        *evals_configs,
    ]


def _get_experiment_configs(
    eval_type: pyine.evals.common.EvalType,
    entrypoint_config: pyine.configs.schemas.ConfigDescription,
    app_configs: list[pyine.configs.schemas.ConfigDescription],
    group: str | None,
    package: str | None,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Generates and returns experiment configs for hydra zen storage (using all datamodule configs).

    These are the configs that define full experiments for the associated app; to be used on the
    command line, you should specify the desired experiment name, e.g. for some 'main.py' app:

        python main.py +experiment=some_experiment_name

    For more information on the individual experiments, refer to their docstrings and to any
    potential README.md file in the corresponding experiment directory.
    """
    # fetch and validate necessary datamodule and openai client configs from main configs set
    dm_configs = [
        config for config in app_configs if config.group == "config/datamodule_config" and config.name != "base"
    ]
    assert sum([c.name == "timeout300s" and c.group == "config/openai_client_config" for c in app_configs]) == 1
    # for each datamodule config we found, create an eval-only and a regular fine-tuning experiment config
    # (note: we don't do anything eval_type-specific here, at least not for these base exp configs)
    outputs: list[pyine.configs.schemas.ConfigDescription] = []
    for config_type_str, dm_config in itertools.product(["_eval_only", ""], dm_configs):
        exp_name = f"{dm_config.name}{config_type_str}"
        desc_str = "eval-only" if config_type_str == "_eval_only" else "regular fine-tuning"
        outputs.append(
            pyine.configs.utils.make_config_description(
                name=exp_name,
                group=group,
                package=package,
                description=(
                    f"Experiment settings that combines the '{dm_config.name}' datamodule settings with "
                    f"good default arguments for the app's {desc_str} entrypoint.\n\n"
                    f"Description for 'config/datamodule_config={dm_config.name}': {dm_config.description}"
                ),
                config={
                    "runtime": {"exp_name": exp_name},
                    "skip_fine_tuning": config_type_str == "_eval_only",
                    # -------------
                    "hydra_defaults": [
                        "_self_",
                        {"override /config/datamodule_config": dm_config.name},
                        # use openai client with 300s timeout for all predefined experiments
                        {"override /config/openai_client_config": "timeout300s"},
                    ],
                    "bases": (entrypoint_config.config,),
                },
            )
        )
    return outputs


def register_hydra_configs(
    eval_type: pyine.evals.common.EvalType,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Registers app-specific configs in hydra and returns the config descriptions.

    Note that in the returned config descriptions, the config that corresponds to the main app's
    entrypoint is called 'entrypoint'.

    IMPORTANT NOTE: we manually load the dotenv file (if it exists) inside this function to help
    resolve config variables that might be dependent on env vars, but no other setup is assumed to
    have occurred. This means that logging, runtime configs, databases, etc. are not available, and
    that these cannot be used for config setup.
    """
    pyine.utils.reprod.load_dotenv()
    entrypoint_config = pyine.configs.utils.make_config_description(
        _async_main_wrapper,
        name="entrypoint",
        group=None,
        description="Entrypoint settings for the OpenAI fine-tuner app.",
        config={
            "skip_fine_tuning": False,
            # -------------
            "populate_full_signature": True,
            "hydra_defaults": [
                "_self_",
                {"config": "base"},  # from this module (`_get_app_configs`)
                {"runtime": "default"},  # from pyine.configs.base
                *pyine.configs.base.get_base_hydra_default_overrides(),
            ],
        },
    )
    store, base_configs = pyine.configs.base.get_base_store_and_configs("openai_finetune")
    app_configs = _get_app_configs(eval_type=eval_type, group="config")
    configs_to_register = [entrypoint_config, *app_configs]
    experiment_configs = _get_experiment_configs(
        eval_type=eval_type,
        entrypoint_config=entrypoint_config,
        app_configs=[*base_configs, *configs_to_register],
        group="experiment",
        package="_global_",
    )
    configs_to_register.extend(experiment_configs)
    external_configs = pyine.configs.searchpath.SearchPathPlugin.get_external_configs(
        app_name="openai_finetune",
        eval_type=eval_type,
        entrypoint_config=entrypoint_config,
        app_configs=[*base_configs, *configs_to_register],
    )
    configs_to_register.extend(external_configs)
    for config in configs_to_register:
        if config.name is None:
            raise ValueError("config name must be defined before registration")
        store(typing.cast("typing.Any", config.config), name=config.name, group=config.group, package=config.package)
    store.add_to_hydra_store(overwrite_ok=True)  # to avoid issues with name conflicts in tests
    return [*base_configs, *configs_to_register]


if __name__ == "__main__":
    pyine.configs.base.register_searchpath_plugin()
    # TODO: if we ever have more than one eval type, add a selector based on launch args here
    pyine.configs.utils.print_experiment_configs(
        config_descriptions=register_hydra_configs(eval_type=pyine.evals.common.EvalType.CODE_EXEC),
        app_name="openai_finetune",
    )
