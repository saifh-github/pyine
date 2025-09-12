"""Hydra-zen config builder for openai_finetune.py app."""

import typing

import hydra.conf
import hydra_zen
import hydra_zen.typing

import pyine.apps.trainers.openai_finetune
import pyine.configs.base
import pyine.configs.schemas
import pyine.configs.utils
import pyine.data.traces.dataset_utils
import pyine.data.utils.splits
import pyine.evals.grader_configs
import pyine.organisms.datamodules.shortcuts
import pyine.organisms.datamodules.shortcuts_configs
import pyine.organisms.models.utils.openai
import pyine.utils.llm_providers
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
        return pyine.organisms.models.utils.openai.OpenAIFineTunerConfig(params=wrapped_fn(*args, **kwargs))

    return _wrapper


def get_default_sft_params_config():
    """Returns the default config for OpenAI supervised fine-tuning using gpt-4.1-mini.

    This configuration should be pretty cheap to run (especially compared to rlft).
    """
    return hydra_zen.builds(
        pyine.organisms.models.utils.openai.OpenAIFineTunerParamsConfig,
        base_model="gpt-4.1-mini-2025-04-14",
        method=dict(type="supervised"),
        hyperparams=None,
        seed="${runtime.seed}",
        suffix="default-sft",
        wandb_integration=None,
        metadata=pyine.utils.reprod.get_reprod_metadata(include_installed_packages=False),
        timeout_override=60 * 60,  # 60 min
        file_upload_params=dict(),
        files_purpose="fine-tune",
        job_params=dict(),
        # -------------
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
        pyine.organisms.models.utils.openai.OpenAIFineTunerParamsConfig,
        base_model="o4-mini-2025-04-16",
        method=hydra.conf.MISSING,  # will be overridden in defaults below
        hyperparams=None,
        seed="${runtime.seed}",
        suffix="default-rlft",
        wandb_integration=None,
        metadata=pyine.utils.reprod.get_reprod_metadata(include_installed_packages=False),
        timeout_override=60 * 60,  # 60 min
        file_upload_params=dict(),
        files_purpose="fine-tune",
        job_params=dict(),
        # -------------
        hydra_convert="object",
        hydra_defaults=[
            "_self_",
            {"override method": "default_pred_grader_rlft_method"},
        ],
        zen_wrappers=_openai_finetuner_method_params_config_wrapper,
    )


def register_hydra_configs(store: hydra_zen.ZenStore | None = None) -> hydra_zen.ZenStore:
    """Registers app-specific configs in the hydra store."""
    if store is None:
        store = hydra_zen.ZenStore()

    # ---------------- store main entrypoint configs ----------------

    store(
        hydra.conf.JobConf(name="openai_finetune_main"),
        group="hydra/job",
        name="openai_finetune_main",
        package="hydra.job",
    )
    openai_finetune_main_config = hydra_zen.builds(
        pyine.apps.trainers.openai_finetune.main,
        config=hydra.conf.MISSING,
        runtime=hydra.conf.MISSING,
        skip_fine_tuning=False,
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
            datamodule_config=hydra.conf.MISSING,  # must be specified by user
            openai_client=dict(),
            openai_finetuner=hydra.conf.MISSING,  # will be overridden in defaults below
            llm_grader_provider_config=None,  # will be overridden in defaults below
            hydra_convert="object",
            hydra_defaults=[
                "_self_",
                {"datamodule_config": "default"},
                {"openai_finetuner": "openai_gpt-4.1-mini_default_sft"},
                {"llm_grader_provider_config": "openai_gpt-5-nano"},
            ],
        ),
        name="base",
    )

    # ---------------- store default datamodule configs ----------------

    datamodule_config_store = config_store(group="config/datamodule_config")
    pyine.organisms.datamodules.shortcuts_configs.register_hydra_configs(datamodule_config_store)

    # ---------------- store default fine-tuning configs ----------------

    openai_finetuner_config_store = config_store(group="config/openai_finetuner")
    openai_finetuner_config_store(
        get_default_sft_params_config(),
        name="openai_gpt-4.1-mini_default_sft",
    )
    openai_finetuner_config_store(
        get_default_rlft_params_config(),
        name="openai_o4-mini_default_rlft",
    )

    openai_finetuner_method_config_store = openai_finetuner_config_store(group="config/openai_finetuner/method")
    openai_finetuner_method_config_store(
        hydra_zen.builds(
            pyine.organisms.models.utils.openai.PredGraderFineTuneMethodConfig,
            hydra_convert="object",
            zen_wrappers=_openai_finetuner_method_config_getter,
        ),
        name="default_pred_grader_rlft_method",
    )

    # ---------------- store default evaluation llm grader configs ----------------

    llm_grader_provider_config_store = config_store(group="config/llm_grader_provider_config")
    pyine.evals.grader_configs.register_hydra_configs(llm_grader_provider_config_store)

    # ---------------- store full demo/experiment configs ----------------

    experiment_store = store(group="experiment", package="_global_")
    experiment_store(
        hydra_zen.make_config(
            runtime=dict(exp_name="TACO_latest_200t_eval_only"),
            skip_fine_tuning=True,
            hydra_defaults=[
                "_self_",
                {"override /config/datamodule_config": "TACO_latest_200t"},
            ],
            bases=(openai_finetune_main_config,),
        ),
        name="TACO_latest_200t_eval_only",
    )
    return store


def hydra_main() -> None:
    """Hydra main entrypoint for the app."""
    pyine.utils.reprod.load_dotenv()
    store = pyine.configs.base.register_hydra_configs()
    store = register_hydra_configs(store)
    store.add_to_hydra_store()
    hydra_zen.zen(pyine.apps.trainers.openai_finetune.main).hydra_main(
        config_path=None,
        config_name="openai_finetune_main",
        version_base=pyine.configs.base.target_hydra_version,
    )
