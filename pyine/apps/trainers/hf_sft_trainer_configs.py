"""Hydra-zen config builder for hf_trainer.py app.

If you execute this script directly, it will print all available experiment configs for this app.
"""

import itertools
import logging
import typing

import pydantic
import torch
import transformers
import wandb

import pyine.apps.trainers.common as common
import pyine.configs.base
import pyine.configs.schemas
import pyine.configs.searchpath
import pyine.configs.utils
import pyine.evals.common
import pyine.evals.configs
import pyine.organisms.datamodules
import pyine.utils.reprod
import pyine.utils.transformers

logger = logging.getLogger(__name__)


class SFTTrainerAppMainConfig(common.AppMainConfig, common.ModelTokenizerConfigBase):
    """Configuration for HuggingFace-Transformers model fine-tuning.

    Inherits model/tokenizer configuration from ModelTokenizerConfigBase.
    """

    # --------------- trainer settings ---------------

    training_args_config: pydantic.SerializeAsAny[pyine.utils.transformers.TrainingArgsConfig] = pydantic.Field(
        ...,  # MISSING! MANDATORY!
        description="Training arguments wrapper for the `transformers.TrainingArguments` class.",
    )

    # --------------- collator settings (SFT-specific) ---------------

    collator_always_pad_to_max_length: bool = pydantic.Field(
        default=False,
        description="Whether to always pad batches to the max sequence length supported by the model.",
    )
    collator_hard_seq_length_cap: int | None = pydantic.Field(
        default=None,
        description="If set, the collator will cap sequence length to the min of this value or the model's cap.",
    )
    collator_pad_to_multiple_of: int | None = pydantic.Field(
        default=None,  # 32,  # TODO test speed with and without?
        description="If set, the collator will pad the sequence length to a multiple of this value.",
    )
    collator_batch_logging: bool = pydantic.Field(
        default=True,
        description="Log per-batch collator stats (batch size / padded length) to wandb when enabled.",
    )

    # --------------- SFT-specific methods ---------------

    @property
    @typing.override
    def target_dtype(self) -> torch.dtype:
        """Returns the target dtype to use with models based on training args."""
        if self.training_args_config.bf16:
            return torch.bfloat16
        return torch.float16 if self.training_args_config.fp16 else torch.float32

    def get_collator(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        max_seq_len: int,
        wandb_run: wandb.Run | None = None,
    ) -> transformers.DataCollator:
        """Returns the data collator to use for training/evaluations."""
        return instantiate_collator(self, tokenizer=tokenizer, max_seq_len=max_seq_len, wandb_run=wandb_run)

    @typing.override
    def normalize_for_resume_overlap_check(
        self,
        config: common.AppMainConfig | dict[str, typing.Any] | None = None,
    ) -> dict[str, typing.Any]:
        """Normalizes the config by removing fields that might change without effects on experiments."""
        data = super().normalize_for_resume_overlap_check(config)
        assert "training_args_config" in data, "training_args_config not found in config"
        training_args = typing.cast("dict[str, typing.Any]", data["training_args_config"])
        # note: there are lots more, these are just the most typical ones...
        training_args.pop("run_name", None)
        training_args.pop("project", None)
        training_args.pop("report_to", None)
        training_args.pop("log_level", None)
        training_args.pop("logging_dir", None)
        training_args.pop("disable_tqdm", None)
        training_args.pop("output_dir", None)
        training_args.pop("do_predict", None)
        return data


def instantiate_collator(
    config: SFTTrainerAppMainConfig,
    tokenizer: transformers.PreTrainedTokenizer,
    max_seq_len: int,
    wandb_run: wandb.Run | None = None,
) -> transformers.DataCollator:
    """Instantiates and returns the data collator tied to the config's tokenizer."""
    logger.info(f"setting up data collator for: {config.base_model}")
    wandb_run_or_init_kwargs: wandb.Run | dict[str, typing.Any] | None = None
    if config.collator_batch_logging and wandb_run is not None:
        logger.debug("setting up collator batch stats logging to wandb run")
        wandb_run_or_init_kwargs = {
            "project": wandb_run.project,
            "entity": wandb_run.entity,
            "id": wandb_run.id,
            "dir": wandb_run.dir,
        }
    if config.collator_hard_seq_length_cap is not None:
        max_seq_len = min(max_seq_len, config.collator_hard_seq_length_cap)
        logger.debug(f"setting up hard seq length cap: {max_seq_len=}")
    collator = pyine.utils.transformers.PaddingCollatorWithPromptMask(
        tokenizer=tokenizer,
        max_length=max_seq_len,
        always_pad_to_max_length=config.collator_always_pad_to_max_length,
        pad_to_multiple_of=config.collator_pad_to_multiple_of,
        wandb_run_or_init_kwargs=wandb_run_or_init_kwargs,
    )
    return typing.cast("transformers.DataCollator", collator)


def _get_trainer_args_configs(
    group: str,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Generates and returns HF trainer args configs for hydra zen storage.

    Note: this function assumes that the current environment exposes relevant device/hardware
    variables that we can use to determine the best training args.
    """
    hw_flags = common.get_hardware_training_flags()
    base_config = pyine.configs.utils.make_config_description(
        pyine.utils.transformers.TrainingArgsConfig,
        name="base",
        group=group,
        description=(
            "Base training arguments for all trainer configs; auto-determines some arguments "
            "based on available hardware, and fills other arguments based on runtime config."
        ),
        config={
            **hw_flags,  # hardware-specific flags (use_cpu, fp16, bf16, tf32, dataloader_pin_memory)
            # extra generic args set based on runtime config
            "run_name": "${runtime.exp_name}-${runtime.run_name}",
            "output_dir": "${hydra:runtime.output_dir}",
            "logging_dir": "${hydra:runtime.output_dir}/logs",
            "seed": "${runtime.seed}",
            # -------------
            "populate_full_signature": True,
            "hydra_convert": "object",
        },
    )
    train_default_config = pyine.configs.utils.make_config_description(
        pyine.utils.transformers.TrainingArgsConfig,
        name="train_default",
        group=group,
        description=(
            "Default training arguments for all trainer configs; applies on top of the `base` "
            "arguments config (which auto-determines some stuff), and provides a number of "
            "reasonable defaults for training (e.g. batch sizes, step counts, ...)."
        ),
        config={
            "do_train": True,  # not used by trainer (meant to be checked by app)
            "do_eval": True,  # not used by trainer (meant to be checked by app)
            "do_predict": True,  # not used by trainer (meant to be checked by app)
            "per_device_train_batch_size": 1,  # minimizes resource usage by default (good for tests/demos)
            "per_device_eval_batch_size": 1,  # minimizes resource usage by default (good for tests/demos)
            # "auto_find_batch_size": True,  # would be nice to use in some cases (but off by default)
            "logging_steps": 5,  # number of training steps between logging
            "logging_first_step": True,  # defines whether to log metrics/losses on the first step or not
            "max_steps": 2000,  # maximum number of training steps to perform
            "eval_on_start": True,  # perform evaluation at the start of training (as a sanity check)
            "eval_strategy": "steps",  # performs evaluation (validation) every N steps
            "eval_steps": 500,  # number of training steps between two evaluations (w/ steps strategy)
            "eval_accumulation_steps": 1,  # moves eval results from device to cpu each eval step
            "save_strategy": "steps",  # checkpoint saving strategy during training
            "save_steps": 500,  # training steps between checkpoint saves (w/ steps strategy); multiple of eval_steps
            "save_total_limit": 3,  # maximum checkpoints to keep (overwrites oldest if exceeded)
            "load_best_model_at_end": True,  # whether to load the best checkpoint after training (loss-based default)
            "remove_unused_columns": False,  # safer with custom collators/generation
            "include_for_metrics": [
                "inputs",
                "loss",
            ],  # data to forward to the compute_metrics callback
            # "dataloader_drop_last": True,  # might want to keep this off for small demos/tests
            # "dataloader_num_workers": 4,  # might want to keep default (0) for small demos/debug
            # "group_by_length": True,  # might be useful for some experiments, but off by default
            # "length_column_name": "...",  # might need to specify this if we toggle on the above
            # -------------
            "builds_bases": (base_config.config,),
        },
    )
    eval_default_config = pyine.configs.utils.make_config_description(
        pyine.utils.transformers.TrainingArgsConfig,
        name="eval_default",
        group=group,
        description=(
            "Default eval-only (prediction) arguments for all trainer configs; applies on top of "
            "the `base` arguments config (which auto-determines some stuff), and provides a number "
            "of reasonable defaults for evaluation (e.g. batch sizes, step counts, ...)."
        ),
        config={
            "do_train": False,  # not used by trainer (meant to be checked by app)
            "do_eval": False,  # not used by trainer (meant to be checked by app)
            "do_predict": True,  # not used by trainer (meant to be checked by app)
            "per_device_eval_batch_size": 1,  # default to lowest-resource-utilization possible
            # "auto_find_batch_size": True,  # would be nice to use in some cases (but off by default)
            "eval_accumulation_steps": 1,  # moves eval results from device to cpu each eval step
            "eval_strategy": "no",  # we'll call trainer.evaluate()/predict() manually
            "save_strategy": "no",  # no need to save any checkpoints in this config
            # "prediction_loss_only": False,  # unclear if we should use this
            "max_steps": 0,  # should not be doing any training in this config
            "remove_unused_columns": False,  # safer with custom collators/generation
            "include_for_metrics": [
                "inputs",
                "loss",
            ],  # data to forward to the compute_metrics callback
            # "dataloader_num_workers": 4,  # might want to keep default (0) for small demos/debug
            # -------------
            "builds_bases": (base_config.config,),
        },
    )
    return [base_config, train_default_config, eval_default_config]


def _get_app_configs(
    eval_type: pyine.evals.common.EvalType,
    group: str,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Generates and returns application configs for hydra zen storage."""
    assert isinstance(group, str) and group
    app_main_config = pyine.configs.utils.make_config_description(
        SFTTrainerAppMainConfig,
        name="base",
        group=group,
        description="Base settings for the HF trainer app.",
        config={
            "base_model": "Qwen/Qwen2.5-Coder-7B-Instruct",  # we propose this default for base experiments
            # -------------
            "populate_full_signature": True,
            "hydra_convert": "object",
            "hydra_defaults": [
                "_self_",
                {"datamodule_config": "shortcuts_base"},
                {"training_args_config": "base"},
                # {"lora_config": "null"},  # left out here = deactivated (null)
                {"evals_config": "code_exec_pass_at_k"},
            ],
        },
    )
    datamodule_configs = pyine.organisms.datamodules.get_configs(
        eval_type=eval_type,
        group=f"{group}/datamodule_config",
    )
    trainer_args_configs = _get_trainer_args_configs(group=f"{group}/training_args_config")
    lora_configs = common.get_lora_configs(group=f"{group}/lora_config", app_description="trainer")
    evals_configs = pyine.evals.configs.get_evals_configs(eval_type=eval_type, group=f"{group}/evals_config")
    # ... add more trainer configs here if needed
    return [
        app_main_config,
        *datamodule_configs,
        *trainer_args_configs,
        *lora_configs,
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
    # fetch and validate necessary datamodule and trainer configs from main configs set
    dm_configs = [
        config
        for config in app_configs
        if config.group == "config/datamodule_config" and not config.name.endswith("base")
    ]
    trainer_configs = [config for config in app_configs if config.group == "config"]
    # for each datamodule config and trainer config combination, create an experiment config
    # (note: we don't do anything eval_type-specific here, at least not for these base exp configs)
    outputs: list[pyine.configs.schemas.ConfigDescription] = []
    for dm_config, trainer_config in itertools.product(dm_configs, trainer_configs):
        exp_name = f"hf_sft_{dm_config.name}_{trainer_config.name}"
        outputs.append(
            pyine.configs.utils.make_config_description(
                name=exp_name,
                group=group,
                package=package,
                description=(
                    f"Experiment config that combines the '{dm_config.name}' datamodule settings with "
                    f"the '{trainer_config.name}' HF model trainer settings for the app's entrypoint.\n\n"
                    f"Description for 'config={trainer_config.name}': {trainer_config.description}\n\n"
                    f"Description for 'config/datamodule_config={dm_config.name}': {dm_config.description}\n\n"
                ),
                config={
                    "runtime": {"exp_name": exp_name},
                    # -------------
                    "hydra_defaults": [
                        "_self_",
                        {"override /config": trainer_config.name},
                        {"override /config/training_args_config": "train_default"},
                        {"override /config/datamodule_config": dm_config.name},
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
        common.async_hf_trainer_main_wrapper,
        name="entrypoint",
        group=None,
        description="Entrypoint settings for the HuggingFace SFT trainer app.",
        config={
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
    store, base_configs = pyine.configs.base.get_base_store_and_configs("hf_sft_trainer")
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
        app_name="hf_sft_trainer",
        eval_type=eval_type,
        entrypoint_config=entrypoint_config,
        app_configs=[*base_configs, *configs_to_register],
    )
    configs_to_register.extend(external_configs)
    for config in configs_to_register:
        assert config.name is not None, "config names should have been set and validated by now"
        store(typing.cast("typing.Any", config.config), name=config.name, group=config.group, package=config.package)
    store.add_to_hydra_store(overwrite_ok=True)  # to avoid issues with name conflicts in tests
    return [*base_configs, *configs_to_register]


if __name__ == "__main__":
    pyine.configs.base.register_searchpath_plugin()
    pyine.configs.utils.print_experiment_configs(
        config_descriptions=register_hydra_configs(eval_type=pyine.evals.common.EvalType.CODE_EXEC),
        app_name="hf_sft_trainer",
    )
