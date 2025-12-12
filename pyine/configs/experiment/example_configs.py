import pyine.apps.trainers.hf_trainer_configs
import pyine.configs.schemas
import pyine.configs.utils
import pyine.evals.common


def _register_qwen25c05b_configs(
    entrypoint_config: pyine.configs.schemas.ConfigDescription,
    app_configs: list[pyine.configs.schemas.ConfigDescription],
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Internal helper function for registering example configs related to Qwen2.5-Coder-0.5B-Instruct."""
    # assuming we're targeting the hf_trainer app for code execution, let's create a new app config
    qwen25c05b_app_config = pyine.configs.utils.make_config_description(
        pyine.apps.trainers.hf_trainer_configs.HFTrainerAppMainConfig,
        name="qwen25c05b",  # name used to refer to this config in defaults lists (in `config` group)
        group="config",  # name of the group this config should belong to (shared for all apps)
        description=(  # descriptions are optional but highly recommended; document your choices here!
            "Override of the main settings for the HF trainer app which specifies settings "
            "targeting the Qwen/Qwen2.5-Coder-0.5B-Instruct chat model for low-end GPUs."
        ),
        config={  # we use 'builds' here instead of a config because we target a dataclass
            "base_model": "Qwen/Qwen2.5-Coder-0.5B-Instruct",  # small model that might run on low-end GPUs
            "quantization_mode": "none",
            # "training_args_config": dict(
            #     # gradient_checkpointing=True,  # maybe needed
            #     # gradient_checkpointing_kwargs=dict(...),  # maybe needed
            #     # gradient_accumulation_steps=5,  # maybe needed
            # ),
            # -------------
            "populate_full_signature": True,
            "hydra_convert": "object",
            "hydra_defaults": [
                "_self_",
                {"datamodule_config": "shortcuts_TACO_latest"},  # arbitrary default from framework configs
                {"training_args_config": "train_default"},  # inherit training settings from framework
                {"lora_config": "default"},  # update to more aggressive LoRA config if needed
                {"evals_config": "base"},  # same as the original for the app
            ],
        },
    )
    # for the above app config, create the associated experiment config
    qwen25c05b_exp_config = pyine.configs.utils.make_config_description(
        name="qwen25c05b",  # unique name to refer to this particular experiment config
        group="experiment",  # all experiment configs should belong to this `experiment` group
        package="_global_",  # by convention, they should all also be defined in the global package
        description="Combines the 'TACO_latest' datamodule settings with the qwen25c05b app config.",
        config={  # since we define experiment configs on top of all others together
            "runtime": {"exp_name": "qwen25c05b"},  # mandatory setting that must be provided (for logging)
            # -------------
            "hydra_defaults": [
                "_self_",
                {"override /config": "qwen25c05b"},  # name of the app config we defined above
            ],
            "bases": (entrypoint_config.config,),  # all experiment configs need to target the entrypoint
        },
    )
    return [qwen25c05b_app_config, qwen25c05b_exp_config]


def _register_smollm360m_configs(
    entrypoint_config: pyine.configs.schemas.ConfigDescription,
    app_configs: list[pyine.configs.schemas.ConfigDescription],
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Internal helper function for registering example configs related to SmolLM-360M."""
    # assuming we're targeting the hf_trainer app for code execution, let's create a new app config
    # (this time, defining settings related to an even smaller model that should be runnable on cpu)
    smollm360m_config = pyine.configs.utils.make_config_description(
        pyine.apps.trainers.hf_trainer_configs.HFTrainerAppMainConfig,
        name="smollm360m",  # name used to refer to this config in defaults lists (in `config` group)
        group="config",  # name of the group this config should belong to (shared for all apps)
        description=(  # descriptions are optional but highly recommended; document your choices here!
            "Override of the main settings for the HF trainer app which specifies settings "
            "targeting the HuggingFaceTB/SmolLM-360M-Instruct chat model for CPU-friendly runs."
        ),
        config={  # we use 'builds' here instead of a config because we target a dataclass
            "base_model": "HuggingFaceTB/SmolLM-360M-Instruct",  # super-tiny model that should be OK on CPU
            "quantization_mode": "none",
            # "training_args_config": dict(
            #     # gradient_checkpointing=True,  # maybe needed
            #     # gradient_checkpointing_kwargs=dict(...),  # maybe needed
            #     # gradient_accumulation_steps=5,  # maybe needed
            # ),
            "auto_model_config": {"low_cpu_mem_usage": True},
            # -------------
            "populate_full_signature": True,
            "hydra_convert": "object",
            "hydra_defaults": [
                "_self_",
                {"datamodule_config": "shortcuts_TACO_latest"},  # arbitrary default from framework configs
                {"training_args_config": "train_default"},  # inherit training settings from framework
                {"lora_config": "default"},  # update to more aggressive LoRA config if needed
                {"evals_config": "base_with_bs8"},  # since CPU memory probably allows it, bump to bs8
            ],
        },
    )
    # note: in the above app config, the `{"evals_config": "base_with_bs8"}` default refers to a new
    #       config that we now have to specify (it does not exist in the list of configs in the fw)
    evals_base_config = next(  # go get the most relevant base config from the fw configs to derive from
        (cfg for cfg in app_configs if cfg.group == "config/evals_config" and cfg.name == "base"),
        None,
    )
    assert evals_base_config is not None, "evals_base_config must be defined in the app configs"
    # now, build the new evals config off the above base config with the extra setting (batch size)
    evals_base_with_bs8_config = pyine.configs.utils.make_config_description(
        name="base_with_bs8",
        group=evals_base_config.group,
        description="Evals base config override with CPU batch size of 8.",
        config={
            "eval_batch_size": 8,
            # -------------
            "bases": (evals_base_config.config,),
        },
    )
    # finally, for the above app configs, create the associated experiment config
    smollm360m_exp_config = pyine.configs.utils.make_config_description(
        name="smollm360m",  # unique name to refer to this particular experiment config
        group="experiment",  # all experiment configs should belong to this `experiment` group
        package="_global_",  # by convention, they should all also be defined in the global package
        description="CPU-focused experiment settings that keeps most defaults with a tiny model.",
        config={  # since we define experiment configs on top of all others together
            "runtime": {"exp_name": "smollm360m"},  # mandatory setting that must be provided (for logging)
            "hydra_defaults": [
                "_self_",
                {"override /config": "smollm360m"},  # name of the app config we defined above
            ],
            "bases": (entrypoint_config.config,),  # all experiment configs need to target the entrypoint
        },
    )
    return [smollm360m_config, evals_base_with_bs8_config, smollm360m_exp_config]


def register_hydra_configs(
    app_name: str,  # name of the app that we are looking to register configs for
    eval_type: pyine.evals.common.EvalType,  # eval type (task definition) for the configs to register
    entrypoint_config: pyine.configs.schemas.ConfigDescription,  # config for the app's entrypoint
    app_configs: list[pyine.configs.schemas.ConfigDescription],  # all registered configs for the app
) -> list[pyine.configs.schemas.ConfigDescription]:  # should return new app configs to register
    """Register example configs for a specific app.

    In this demo, we create configs that allow us to train a language model on a mps chipset, i.e.
    using a much smaller model than the default, and using some resource-efficient settings.

    Args:
        app_name: The name of the app for which to generate configs.
        eval_type: The type of evaluation for which to generate configs.
        entrypoint_config: The entrypoint config for the app.
        app_configs: The list of all configs that have already been registered for the app.

    Returns:
        The list of newly generated app configs to be registered in the framework.
    """
    # the YAML example that is also provided targets the openai_finetune app, so this one will not
    if app_name not in ["hf_trainer", "hf_precacher"] or eval_type != pyine.evals.common.EvalType.CODE_EXEC:
        return []
    # define new app configs for new experiments which target much smaller models
    return [
        *_register_qwen25c05b_configs(entrypoint_config, app_configs),
        *_register_smollm360m_configs(entrypoint_config, app_configs),
    ]
