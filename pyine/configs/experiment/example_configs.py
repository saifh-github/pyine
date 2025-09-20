import hydra_zen

import pyine.apps.trainers.hf_trainer_configs
import pyine.configs.schemas


def register_hydra_configs(
    app_name: str,  # name of the app that we are looking to register configs for
    entrypoint_config: pyine.configs.schemas.ConfigDescription,  # config for the app's entrypoint
    app_configs: list[pyine.configs.schemas.ConfigDescription],  # all registered configs for the app
) -> list[pyine.configs.schemas.ConfigDescription]:  # should return new app configs to register
    """Register example configs for a specific app.

    In this demo, we create configs that allow us to train a language model on a mps chipset, i.e.
    using a much smaller model than the default, and using some resource-efficient settings.

    Args:
        app_name: The name of the app for which to generate configs.
        entrypoint_config: The entrypoint config for the app.
        app_configs: The list of all configs that have already been registered for the app.

    Returns:
        The list of newly generated app configs to be registered in the framework.
    """
    # the YAML example that is also provided targets the openai_finetune app, so this one will not
    if app_name != "hf_trainer":
        return []
    # define a new app config for a new experiment which targets a much smaller model
    qwen25c05B_m3pro_config = pyine.configs.schemas.ConfigDescription(
        name="qwen25c05B_m3pro",  # name used to refer to this config in defaults/overrides
        group="config",  # name of the group this config should belong to (from framework)
        config=hydra_zen.builds(  # we use 'builds' here instead of a config because we target a dataclass (not a func)
            pyine.apps.trainers.hf_trainer_configs.HFTrainerAppMainConfig,
            base_model="Qwen/Qwen2.5-Coder-0.5B-Instruct",  # super-tiny model for MPS-based runs
            quantization_mode="none",  # mps does not support quantization? (to be confirmed)
            # training_args_config=dict(
            #     # gradient_checkpointing=True,  # maybe needed
            #     # gradient_checkpointing_kwargs=dict(...),  # for the above, if needed
            #     # gradient_accumulation_steps=5,  # maybe needed
            # ),
            # -------------
            populate_full_signature=True,
            hydra_convert="object",
            hydra_defaults=[
                "_self_",
                {"datamodule_config": "TACO_latest"},  # arbitrary default from framework configs
                {"training_args_config": "train_default"},  # inherit training settings from framework
                {"lora_config": "default"},  # update to more aggressive LoRA config if needed
                {"llm_grader_provider_config": "openai_gpt-5-nano"},  # also same as base config
            ],
            zen_meta={
                "__description__": (
                    "Override of the main settings for the HF trainer app which specifies "
                    "the Qwen/Qwen2.5-Coder-0.5B-Instruct model for MPS-based runs on M3 Pro chips."
                ),
            },
        ),
    )
    # create a new experiment config that relies on the above for the app config
    qwen25c05B_m3pro_exp_config = pyine.configs.schemas.ConfigDescription(
        name="qwen25c05B_m3pro",
        group="experiment",
        package="_global_",
        config=hydra_zen.make_config(
            runtime=dict(exp_name="qwen25c05B_m3pro"),  # mandatory setting that must be provided (for logging)
            # -------------
            hydra_defaults=[
                "_self_",
                {"override /config": "qwen25c05B_m3pro"},  # name of the config we defined above in the config group
                # {"override /config/training_args_config": "train_default"},  # base default from framework configs
                # {"override /config/datamodule_config": "TACO_latest"},  # arbitrary default from framework configs
            ],
            bases=(entrypoint_config.config,),
            zen_meta={
                "__description__": (
                    "Experiment config that combines the 'TACO_latest' datamodule settings with "
                    "a new custom app config for MPS-based runs on M3 Pro chips."
                ),
            },
        ),
    )
    return [qwen25c05B_m3pro_config, qwen25c05B_m3pro_exp_config]
