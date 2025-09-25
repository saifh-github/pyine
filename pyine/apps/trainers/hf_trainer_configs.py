"""Hydra-zen config builder for hf_trainer.py app.

If you execute this script directly, it will print all available experiment configs for this app.
"""

import asyncio
import functools
import itertools
import logging
import typing

import hydra_zen
import hydra_zen.typing
import peft
import pydantic
import torch
import transformers

import pyine.apps.trainers.common
import pyine.apps.trainers.hf_trainer
import pyine.configs.base
import pyine.configs.schemas
import pyine.configs.searchpath
import pyine.data.datamodule
import pyine.evals.grader_configs
import pyine.organisms.datamodules.shortcuts_configs
import pyine.utils.pydantic
import pyine.utils.reprod

logger = logging.getLogger(__name__)


TrainingArgsConfig = pyine.utils.pydantic.model_from_callable(
    fn=transformers.TrainingArguments,
    name="TrainingArgsConfig",
    model_config=pydantic.ConfigDict(frozen=True, extra="forbid"),
    default_overrides=dict(
        # we use some updated defaults (low-impact, QoL stuff)
        load_best_model_at_end=True,  # easy to forget, but important! (also force-saves best ckpt)
        logging_first_step=True,  # good for plotting/sanity
        log_level="info",  # enable info-level logging for models by default
        report_to="none",  # disable by default, and enable at runtime if needed
        # we also need to replace some defaults that CANNOT be serialized (factories)
        lr_scheduler_kwargs=dict(),  # same behavior as original default
        include_for_metrics=list(),  # same behavior as original default
    ),
)
"""Configuration parameters for the OpenAI client."""


class HFTrainerAppMainConfig(pyine.apps.trainers.common.AppMainConfig):
    """Configuration for HuggingFace LoRA fine-tuning."""

    base_model: str = pydantic.Field(
        ...,
        description="Hugging Face model identifier or local path for the base causal LM to fine-tune.",
    )
    training_args_config: TrainingArgsConfig = pydantic.Field(
        ...,
        description="Training arguments wrapper for the `transformers.TrainingArguments` class.",
    )
    quantization_mode: typing.Literal["qlora", "none"] = pydantic.Field(
        "none",
        description='Quantization mode. "qlora" loads the model in 4-bit for QLoRA; "none" disables quantization.',
    )
    tokenizer_pad_as_eos: bool = pydantic.Field(
        True,
        description="If True and tokenizer has no PAD token, reuse EOS token as PAD for batching.",
    )
    lora_config: peft.LoraConfig | None = pydantic.Field(
        default=None,
        description="LoRA adapter configuration. If None, does not apply LoRA.",
    )

    @pydantic.model_validator(mode="before")
    @classmethod
    def _coerce_lora_config(
        cls,
        data: typing.Any,
    ) -> typing.Any:
        """Ensure the lora field is a LoraConfig instance when provided as a dict."""
        if isinstance(data, dict):
            lora_config = data.get("lora_config")
            if isinstance(lora_config, dict) and not isinstance(lora_config, peft.LoraConfig):
                data["lora_config"] = peft.LoraConfig(**lora_config)
        return data


@functools.wraps(pyine.apps.trainers.hf_trainer.main)
def _async_main_wrapper(*args, **kwargs):
    """Wrapper for async main function."""
    return asyncio.run(pyine.apps.trainers.hf_trainer.main(*args, **kwargs))


def hydra_main() -> None:
    """Hydra main entrypoint for the HuggingFace trainer app."""
    _ = register_hydra_configs()
    pyine.configs.base.register_searchpath_plugin()
    hydra_zen.zen(_async_main_wrapper).hydra_main(
        config_path=None,
        config_name="entrypoint",
        version_base=pyine.configs.base.target_hydra_version,
    )


def _get_trainer_args_configs(
    group: str,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Generates and returns HF trainer args configs for hydra zen storage.

    Note: this function assumes that the current environment exposes relevant device/hardware
    variables that we can use to determine the best training args.
    """
    is_cuda = torch.cuda.is_available()
    is_mps = torch.backends.mps.is_available()
    use_cpu = not (is_cuda or is_mps)
    use_bf16 = bool(is_cuda and torch.cuda.is_bf16_supported())
    use_fp16 = bool(is_cuda and not use_bf16)
    pin_mem = bool(is_cuda)
    base_config = pyine.configs.schemas.ConfigDescription(
        name="base",
        group=group,
        config=hydra_zen.builds(
            TrainingArgsConfig,
            # some args are auto-deduced from hardware
            use_cpu=use_cpu,
            fp16=use_fp16,
            bf16=use_bf16,
            tf32=is_cuda,
            dataloader_pin_memory=pin_mem,
            # extra generic args set based on runtime config
            run_name="${runtime.exp_name}-${runtime.run_name}",
            output_dir="${hydra:runtime.output_dir}",
            logging_dir="${hydra:runtime.output_dir}/logs",
            seed="${runtime.seed}",
            # -------------
            populate_full_signature=True,
            hydra_convert="object",
            zen_meta={
                "__description__": (
                    "Base training arguments for all trainer configs; auto-determines some arguments "
                    "based on available hardware, and fills other arguments based on runtime config."
                ),
            },
        ),
    )
    # @@@@@@ trainer_args_config_store(base_config.config, name="base")
    # TODO @@@@@@ add distrib trainer config with local_rank and ddp/fsdp stuff? or deepspeed/accelerate?
    # TODO @@@@@@ add configs w/ debug settings? (and tokens/sec or tokens seen metrics?)
    train_default_config = pyine.configs.schemas.ConfigDescription(
        name="train_default",
        group=group,
        config=hydra_zen.builds(
            TrainingArgsConfig,
            do_train=True,  # not used by trainer (meant to be checked by app)
            do_eval=True,  # not used by trainer (meant to be checked by app)
            do_predict=True,  # not used by trainer (meant to be checked by app)
            per_device_train_batch_size=1,  # minimizes resource usage by default (good for tests/demos)
            per_device_eval_batch_size=1,  # minimizes resource usage by default (good for tests/demos)
            # auto_find_batch_size=True,  # would be nice to use in some cases (but off by default)
            logging_steps=5,  # number of training steps between logging
            logging_first_step=True,  # defines whether to log metrics/losses on the first step or not
            max_steps=2000,  # maximum number of training steps to perform
            eval_on_start=True,  # perform evaluation at the start of training (as a sanity check)
            eval_strategy="steps",  # performs evaluation (validation) every N steps
            eval_steps=500,  # number of training steps between two evaluations (w/ steps strategy)
            eval_accumulation_steps=1,  # moves eval results from device to cpu each eval step
            save_strategy="steps",  # checkpoint saving strategy during training
            save_steps=500,  # training steps between checkpoint saves (w/ steps strategy); multiple of eval_steps
            save_total_limit=3,  # maximum checkpoints to keep (overwrites oldest if exceeded)
            load_best_model_at_end=True,  # whether to load the best checkpoint after training (loss-based default)
            remove_unused_columns=False,  # safer with custom collators/generation
            include_for_metrics=["inputs", "loss"],  # data to forward to the compute_metrics callback
            # dataloader_drop_last=True,  # might want to keep this off for small demos/tests
            # dataloader_num_workers=4,  # might want to keep default (0) for small demos/debug
            # group_by_length=True,  # might be useful for some experiments, but off by default
            # length_column_name="...",  # might need to specify this if we toggle on the above
            # -------------
            builds_bases=(base_config.config,),
            zen_meta={
                "__description__": (
                    "Default training arguments for all trainer configs; applies on top of the `base` "
                    "arguments config (which auto-determines some stuff), and provides a number of "
                    "reasonable defaults for training (e.g. batch sizes, step counts, ...)."
                ),
            },
        ),
    )
    eval_default_config = pyine.configs.schemas.ConfigDescription(
        name="eval_default",
        group=group,
        config=hydra_zen.builds(
            TrainingArgsConfig,
            do_train=False,  # not used by trainer (meant to be checked by app)
            do_eval=False,  # not used by trainer (meant to be checked by app)
            do_predict=True,  # not used by trainer (meant to be checked by app)
            per_device_eval_batch_size=1,  # default to lowest-resource-utilization possible
            # auto_find_batch_size=True,  # would be nice to use in some cases (but off by default)
            eval_accumulation_steps=1,  # moves eval results from device to cpu each eval step
            eval_strategy="no",  # we'll call trainer.evaluate()/predict() manually
            save_strategy="no",  # no need to save any checkpoints in this config
            # prediction_loss_only=False,  # unclear if we should use this
            max_steps=0,  # should not be doing any training in this config
            remove_unused_columns=False,  # safer with custom collators/generation
            include_for_metrics=["inputs", "loss"],  # data to forward to the compute_metrics callback
            # dataloader_num_workers=4,  # might want to keep default (0) for small demos/debug
            # -------------
            builds_bases=(base_config.config,),
            zen_meta={
                "__description__": (
                    "Default eval-only (prediction) arguments for all trainer configs; applies on top of"
                    "the `base` arguments config (which auto-determines some stuff), and provides a number "
                    "of reasonable defaults for evaluation (e.g. batch sizes, step counts, ...)."
                ),
            },
        ),
    )
    return [base_config, train_default_config, eval_default_config]


def _get_lora_configs(
    group: str,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Generates and returns LoRA adaptor configs for hydra zen storage."""
    default_lora_config = pyine.configs.schemas.ConfigDescription(
        name="default",
        group=group,
        config=hydra_zen.builds(
            peft.LoraConfig,
            # -------------
            populate_full_signature=True,
            hydra_convert="object",
            zen_meta={
                "__description__": (
                    "Default LoRA settings for all trainer configs; applies to all models, and provides "
                    "a reasonable default for LoRA adaptation. See `peft.LoraConfig` for more details."
                ),
            },
        ),
    )
    return [default_lora_config]


def _get_app_main_configs(
    group: str,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Generates and returns main application configs for hydra zen storage."""
    assert isinstance(group, str) and group
    app_main_config = pyine.configs.schemas.ConfigDescription(
        name="base",
        group=group,
        config=hydra_zen.builds(
            HFTrainerAppMainConfig,
            base_model="Qwen/Qwen2.5-Coder-7B-Instruct",  # we propose this default for base experiments
            # -------------
            populate_full_signature=True,
            hydra_convert="object",
            hydra_defaults=[
                "_self_",
                {"datamodule_config": "base"},
                {"training_args_config": "base"},
                # {"lora_config": "default"},  # left out here = deactivated (null)
                {"llm_grader_provider_config": "openai_gpt-5-nano"},
            ],
            zen_meta={
                "__description__": "Default settings for the HF trainer app.",
            },
        ),
    )
    datamodule_configs = pyine.organisms.datamodules.shortcuts_configs.get_configs(f"{group}/datamodule_config")
    trainer_args_configs = _get_trainer_args_configs(f"{group}/training_args_config")
    lora_configs = _get_lora_configs(f"{group}/lora_config")
    llm_grader_provider_configs = pyine.evals.grader_configs.get_configs(f"{group}/llm_grader_provider_config")

    # ... add more trainer configs here if needed

    return [
        app_main_config,
        *datamodule_configs,
        *trainer_args_configs,
        *lora_configs,
        *llm_grader_provider_configs,
    ]


def _get_experiment_configs(
    entrypoint_config: pyine.configs.schemas.ConfigDescription,
    main_app_configs: list[pyine.configs.schemas.ConfigDescription],
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

    # fetch and validate necessary datamodule and trainer configs from main configs set
    dm_configs = [
        config for config in main_app_configs if config.group == "config/datamodule_config" and config.name != "base"
    ]
    trainer_configs = [config for config in main_app_configs if config.group == "config"]

    # for each datamodule config and trainer config combination, create an experiment config
    outputs: list[pyine.configs.schemas.ConfigDescription] = []
    for dm_config, trainer_config in itertools.product(dm_configs, trainer_configs):
        exp_name = f"{dm_config.name}_{trainer_config.name}"
        outputs.append(
            pyine.configs.schemas.ConfigDescription(
                name=exp_name,
                group=group,
                package=package,
                config=hydra_zen.make_config(
                    runtime=dict(exp_name=exp_name),
                    # -------------
                    hydra_defaults=[
                        "_self_",
                        {"override /config": trainer_config.name},
                        {"override /config/training_args_config": "train_default"},
                        {"override /config/datamodule_config": dm_config.name},
                    ],
                    bases=(entrypoint_config.config,),
                    zen_meta={
                        "__description__": (
                            f"Experiment config that combines the '{dm_config.name}' datamodule settings with "
                            f"the '{trainer_config.name}' HF model trainer settings for the app's entrypoint.\n\n"
                            f"Description for 'config={trainer_config.name}': {trainer_config.description}\n\n"
                            f"Description for 'config/datamodule_config={dm_config.name}': {dm_config.description}\n\n"
                        ),
                    },
                ),
            )
        )
    return outputs


def register_hydra_configs() -> list[pyine.configs.schemas.ConfigDescription]:
    """Registers app-specific configs in hydra and returns the config descriptions.

    Note that in the returned config descriptions, the config that corresponds to the main app's
    entrypoint is called 'entrypoint'.

    IMPORTANT NOTE: we manually load the dotenv file (if it exists) inside this function to help
    resolve config variables that might be dependent on env vars, but no other setup is assumed to
    have occurred. This means that logging, runtime configs, databases, etc. are not available, and
    that these cannot be used for config setup.
    """
    pyine.utils.reprod.load_dotenv()
    store, base_configs = pyine.configs.base.get_base_store_and_configs("hf_trainer")
    main_app_configs = _get_app_main_configs(group="config")
    entrypoint_config = pyine.configs.schemas.ConfigDescription(
        name="entrypoint",
        group=None,
        config=hydra_zen.builds(
            _async_main_wrapper,
            # -------------
            populate_full_signature=True,
            hydra_defaults=[
                "_self_",
                {"config": "base"},
                {"runtime": "default"},  # from base module
                *pyine.configs.base.get_base_hydra_default_overrides(),
            ],
            zen_meta={
                "__description__": "Entrypoint settings for the HF trainer app.",
            },
        ),
    )
    experiment_configs = _get_experiment_configs(
        entrypoint_config,
        main_app_configs,
        group="experiment",
        package="_global_",
    )
    external_configs = pyine.configs.searchpath.SearchPathPlugin.get_external_configs(
        "hf_trainer",
        entrypoint_config,
        main_app_configs,
    )
    configs_to_register = [entrypoint_config, *main_app_configs, *experiment_configs, *external_configs]
    for config in configs_to_register:
        store(config.config, name=config.name, group=config.group, package=config.package)
    store.add_to_hydra_store(overwrite_ok=True)  # to avoid issues with name conflicts in tests
    return [*base_configs, *configs_to_register]


if __name__ == "__main__":
    pyine.configs.base.register_searchpath_plugin()
    pyine.configs.base.print_experiment_configs(register_hydra_configs(), "hf_trainer")
