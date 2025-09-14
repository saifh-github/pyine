"""Hydra-zen config builder for openai_finetune.py app."""

import asyncio
import functools
import typing

import hydra.conf
import hydra_zen
import hydra_zen.typing

import pyine.apps.trainers.openai_finetune
import pyine.configs.base
import pyine.configs.schemas
import pyine.data.traces.dataset_utils
import pyine.data.utils.splits
import pyine.evals.grader_configs
import pyine.organisms.datamodules.shortcuts
import pyine.organisms.datamodules.shortcuts_configs
import pyine.organisms.models.utils.openai
import pyine.utils.llm_providers
import pyine.utils.openai
import pyine.utils.reprod


def _openai_finetuner_method_config_getter(
    wrapped_fn: typing.Callable,
) -> typing.Callable:
    """Wrapper for OpenAI fine-tuning method configs to return the OpenAI config object."""

    def _wrapper(*args, **kwargs):
        return wrapped_fn(*args, **kwargs).get_openai_config()

    return _wrapper


def _openai_finetuner_method_params_config_wrapper(
    wrapped_fn: typing.Callable,
) -> typing.Callable:
    """Wrapper for OpenAI fine-tuning method params configs to return a parent object."""

    def _wrapper(*args, **kwargs):
        return pyine.utils.openai.OpenAIFineTunerConfig(params=wrapped_fn(*args, **kwargs))

    return _wrapper


def get_default_sft_params_config():
    """Returns the default config for OpenAI supervised fine-tuning using gpt-4.1-mini.

    This configuration should be pretty cheap to run (especially compared to rlft).
    """
    return hydra_zen.builds(
        pyine.utils.openai.OpenAIFineTunerParamsConfig,
        base_model="gpt-4.1-mini-2025-04-14",
        method=dict(type="supervised"),
        seed="${runtime.seed}",
        suffix="default-sft",
        metadata=pyine.utils.reprod.get_reprod_metadata(include_installed_packages=False),
        # -------------
        populate_full_signature=True,
        hydra_convert="object",
        zen_wrappers=_openai_finetuner_method_params_config_wrapper,
    )


def get_default_rlft_params_config():
    """Returns the default config for OpenAI RL fine-tuning using o4-mini.

    Note: as of 2025-09-03, training o4-mini using this config is VERY COSTLY, even for VERY TINY
    datasets (we're talking hundreds of dollars per run here, minimum). Don't use this config unless
    you know what you're doing.
    """
    return hydra_zen.builds(
        pyine.utils.openai.OpenAIFineTunerParamsConfig,
        base_model="o4-mini-2025-04-16",
        seed="${runtime.seed}",
        suffix="default-rlft",
        metadata=pyine.utils.reprod.get_reprod_metadata(include_installed_packages=False),
        # -------------
        populate_full_signature=True,
        hydra_convert="object",
        hydra_defaults=[
            "_self_",
            {"override method": "default_pred_grader_rlft_method"},
        ],
        zen_wrappers=_openai_finetuner_method_params_config_wrapper,
    )


@functools.wraps(pyine.apps.trainers.openai_finetune.main)
def _async_main_wrapper(*args, **kwargs):
    """Wrapper for async main function."""
    return asyncio.run(pyine.apps.trainers.openai_finetune.main(*args, **kwargs))


def hydra_main() -> None:
    """Hydra main entrypoint for the app."""
    _ = register_hydra_configs()
    hydra_zen.zen(_async_main_wrapper).hydra_main(
        config_path=None,
        config_name="openai_finetune_main",
        version_base=pyine.configs.base.target_hydra_version,
    )


def register_experiment_configs(
    experiment_store: hydra_zen.ZenStore,
    main_config: hydra_zen.typing.Builds,
) -> None:
    """Registers experiment-specific configs in the hydra store.

    These are the configs that define full experiments for the associated app; to be used on the
    command line, you should specify the desired experiment name, e.g. for some 'main.py' app:

        python main.py +experiment=some_experiment_name

    For more information on the individual experiments, refer to their docstrings and to any
    potential README.md file in the corresponding experiment directory.
    """
    experiment_store(
        hydra_zen.make_config(
            runtime=dict(exp_name="TACO_latest_200t_eval_only"),
            config=dict(eval_subset_names=["valid", "valid_obfuscated"]),
            skip_fine_tuning=True,
            hydra_defaults=[
                "_self_",
                {"override /config/datamodule_config": "TACO_latest_200t"},
                {"override /config/openai_client_config": "timeout300s"},
            ],
            bases=(main_config,),
        ),
        name="TACO_latest_200t_eval_only",
    )
    experiment_store(
        hydra_zen.make_config(
            runtime=dict(exp_name="TACO_latest_200t"),
            config=dict(eval_subset_names=["valid", "valid_obfuscated"]),
            skip_fine_tuning=False,
            hydra_defaults=[
                "_self_",
                {"override /config/datamodule_config": "TACO_latest_200t"},
                {"override /config/openai_client_config": "timeout300s"},
            ],
            bases=(main_config,),
        ),
        name="TACO_latest_200t",
    )
    # @@@@ TODO: pull in experiment configs from organisms folder, if needed


def register_hydra_configs() -> hydra_zen.typing.Builds:
    """Registers app-specific configs in the hydra store and returns the store and main config.

    Will build off the default configs from the pyine.configs.base module and add configs for
    experiment setups, which are defined in the `register_experiment_configs` function. These
    should allow users to run the app without any extra configuration.
    """
    pyine.utils.reprod.load_dotenv()
    store = pyine.configs.base.get_base_store()

    # ---------------- store main entrypoint configs ----------------

    store(
        hydra.conf.JobConf(name="openai_finetune_main"),
        group="hydra/job",
        name="openai_finetune_main",
        package="hydra.job",
    )
    openai_finetune_main_config = hydra_zen.builds(
        _async_main_wrapper,
        skip_fine_tuning=False,
        # -------------
        populate_full_signature=True,
        hydra_defaults=[
            "_self_",
            {"config": "base"},
            {"runtime": "default"},
            {"hydra/job": "openai_finetune_main"},
            *pyine.configs.base.get_base_hydra_default_overrides(),
        ],
    )
    store(openai_finetune_main_config, name="openai_finetune_main")

    # ---------------- store main params configs ----------------

    config_store = store(group="config")
    config_store(
        hydra_zen.builds(
            pyine.apps.trainers.openai_finetune.MainConfig,
            # -------------
            populate_full_signature=True,
            hydra_convert="object",
            hydra_defaults=[
                "_self_",
                {"datamodule_config": "default"},
                {"openai_client_config": "default"},
                {"openai_finetuner_config": "openai_gpt-4.1-mini_default_sft"},
                {"llm_grader_provider_config": "openai_gpt-5-nano"},
            ],
        ),
        name="base",
    )

    # ---------------- store default datamodule configs ----------------

    datamodule_config_store = config_store(group="config/datamodule_config")
    pyine.organisms.datamodules.shortcuts_configs.store_hydra_configs(datamodule_config_store)

    # ---------------- store default openai client configs ----------------

    openai_client_config_store = config_store(group="config/openai_client_config")
    openai_client_configs = pyine.configs.base.get_openai_client_configs()
    for config_name, config in openai_client_configs.items():
        openai_client_config_store(config, name=config_name)

    # ---------------- store default fine-tuning configs ----------------

    openai_finetuner_config_store = config_store(group="config/openai_finetuner_config")
    openai_finetuner_config_store(
        get_default_sft_params_config(),
        name="openai_gpt-4.1-mini_default_sft",
    )
    openai_finetuner_config_store(
        get_default_rlft_params_config(),
        name="openai_o4-mini_default_rlft",
    )

    openai_finetuner_method_config_store = openai_finetuner_config_store(group="config/openai_finetuner_config/method")
    openai_finetuner_method_config_store(
        hydra_zen.builds(
            pyine.organisms.models.utils.openai.PredGraderFineTuneMethodConfig,
            populate_full_signature=True,
            hydra_convert="object",
            zen_wrappers=_openai_finetuner_method_config_getter,
        ),
        name="default_pred_grader_rlft_method",
    )

    # ---------------- store default evaluations configs ----------------

    llm_grader_provider_config_store = config_store(group="config/llm_grader_provider_config")
    pyine.evals.grader_configs.store_hydra_configs(llm_grader_provider_config_store)

    # ---------------- store full demo/experiment configs ----------------

    experiment_store = store(group="experiment", package="_global_")
    register_experiment_configs(experiment_store, openai_finetune_main_config)

    # ---------------- register all configs with hydra ----------------

    store.add_to_hydra_store()

    return openai_finetune_main_config  # all done; return this for launch calls, if needed
