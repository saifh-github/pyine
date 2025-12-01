"""Hydra-zen config builder for hf_trainer.py app.

If you execute this script directly, it will print all available experiment configs for this app.
"""

import asyncio
import dataclasses
import itertools
import logging
import typing

import hydra_zen
import peft
import pydantic
import torch
import torch.distributed.elastic.multiprocessing.errors
import transformers
import wandb

import pyine.apps.trainers.common
import pyine.configs.base
import pyine.configs.schemas
import pyine.configs.searchpath
import pyine.configs.utils
import pyine.evals.common
import pyine.evals.configs
import pyine.organisms.datamodules.shortcuts_configs
import pyine.utils.distrib
import pyine.utils.reprod
import pyine.utils.tokenizers
import pyine.utils.transformers

logger = logging.getLogger(__name__)


class HFTrainerAppMainConfig(pyine.apps.trainers.common.AppMainConfig):
    """Configuration for HuggingFace-Transformers model fine-tuning."""

    # --------------- trainer settings ---------------

    training_args_config: pydantic.SerializeAsAny[pyine.utils.transformers.TrainingArgsConfig] = pydantic.Field(
        ...,  # MISSING! MANDATORY!
        description="Training arguments wrapper for the `transformers.TrainingArguments` class.",
    )

    # --------------- collator settings ---------------

    collator_always_pad_to_max_length: bool = pydantic.Field(
        default=False,
        description="Whether to always pad batches to the max sequence length supported by the model.",
    )
    collator_hard_seq_length_cap: int | None = pydantic.Field(
        default=None,
        description="If set, the collator will cap sequence length to the min of this value or the model's cap.",
    )
    collator_pad_to_multiple_of: int | None = pydantic.Field(
        default=None,  # 32,  # @@@@ TODO test speed with and without?
        description="If set, the collator will pad the sequence length to a multiple of this value.",
    )
    collator_batch_logging: bool = pydantic.Field(
        default=True,
        description="Log per-batch collator stats (batch size / padded length) to wandb when enabled.",
    )

    # --------------- model settings ---------------

    base_model: str = pydantic.Field(
        ...,  # MISSING! MANDATORY!
        description="Hugging Face model identifier or local path for the base causal LM to fine-tune.",
    )
    auto_model_config: dict[str, typing.Any] = pydantic.Field(
        default_factory=lambda: typing.cast("dict[str, typing.Any]", {}),
        description="Model configuration args passed to `transformers.AutoModelForCausalLM.from_pretrained`.",
    )
    quantization_mode: typing.Literal["qlora", "none"] = pydantic.Field(
        default="none",
        description='Quantization mode. "qlora" loads the model in 4-bit for QLoRA; "none" disables quantization.',
    )
    lora_config: peft.LoraConfig | pyine.utils.transformers.LoraConfig | None = pydantic.Field(
        default=None,
        description="LoRA adapter configuration. If None, does not apply LoRA.",
    )

    # --------------- tokenizer settings ---------------

    auto_tokenizer_config: dict[str, typing.Any] = pydantic.Field(
        default_factory=lambda: {"use_fast": True},
        description="Tokenizer configuration args passed to `transformers.AutoTokenizer.from_pretrained`.",
    )
    tokenizer_set_padding_to_eos_if_needed: bool = pydantic.Field(
        default=True,
        description="If True and tokenizer has no PAD token, reuse EOS token as PAD for batching.",
    )
    tokenizer_override_padding_to_right_side: bool = pydantic.Field(
        default=True,  # useful default for most collate functions during training (may be overridden in eval config)
        description="Override whichever the tokenizer's default padding side is to 'right'.",
    )
    tokenizer_override_truncation_to_left_side: bool = pydantic.Field(
        default=True,  # useful default for datasets with long system prompts that end with specific instructions
        description="Override whichever the tokenizer's default truncation side is to 'left'.",
    )

    # --------------- utility/helper method ---------------

    @pydantic.model_validator(mode="before")
    @classmethod
    def _coerce_lora_config(
        cls,
        data: typing.Any,
    ) -> typing.Any:
        """Ensures the lora field is a LoraConfig instance when provided as a dict."""
        if isinstance(data, dict):
            typed_data = typing.cast("dict[str, typing.Any]", data)
            lora_config = typed_data.get("lora_config")
            if isinstance(lora_config, pyine.utils.transformers.LoraConfig):
                return typed_data
            if isinstance(lora_config, peft.LoraConfig):
                typed_data["lora_config"] = pyine.utils.transformers.LoraConfig.model_validate(
                    dataclasses.asdict(lora_config)
                )
                return typed_data
            if isinstance(lora_config, dict):
                typed_lora_config = typing.cast("dict[str, typing.Any]", lora_config)
                typed_data["lora_config"] = pyine.utils.transformers.LoraConfig.model_validate(typed_lora_config)
                return typed_data
        return typing.cast("typing.Any", data)

    @property
    def target_dtype(self) -> torch.dtype:
        """Returns the target dtype to use with models."""
        if self.training_args_config.bf16:
            return torch.bfloat16
        return torch.float16 if self.training_args_config.fp16 else torch.float32

    @property
    def device_map(self) -> torch.device | str | dict[str, torch.device | str] | None:
        """Returns the device map to use with models."""
        if pyine.utils.distrib.is_distributed():
            if torch.cuda.is_available():
                local_rank = pyine.utils.distrib.get_local_rank(default=0)
                if local_rank is None:
                    return None
                return {"": f"cuda:{local_rank}"}
            return None
        return {"": "mps"} if torch.backends.mps.is_available() else "auto"

    def get_collator(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        max_seq_len: int,
        wandb_run: wandb.Run | None = None,
    ) -> transformers.DataCollator:
        """Returns the data collator to use for training/evaluations."""
        return instantiate_collator(self, tokenizer=tokenizer, max_seq_len=max_seq_len, wandb_run=wandb_run)

    def get_tokenizer(self) -> transformers.PreTrainedTokenizer:
        """Returns the tokenizer to use that is linked to the targeted base model."""
        return instantiate_tokenizer(self)

    def get_model(self) -> transformers.PreTrainedModel:
        """Returns a pretrained model to use for experiments."""
        return instantiate_model(self)

    @typing.override
    def normalize_for_resume_overlap_check(
        self,
        config: pyine.apps.trainers.common.AppMainConfig | dict[str, typing.Any] | None = None,
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
    config: HFTrainerAppMainConfig,
    tokenizer: transformers.PreTrainedTokenizer,
    max_seq_len: int,
    wandb_run: wandb.Run | None = None,
) -> transformers.DataCollator:
    """Instantiates and returns the data collator tied to the config's tokenizer."""
    logger.info(f"setting up data collator for: {config.base_model}")
    wandb_run_or_init_kwargs: wandb.Run | dict[str, typing.Any] | None = None
    if config.collator_batch_logging and wandb_run is not None:
        logger.debug("setting up collator batch stats logging to wandb run")
        # TODO: @@@@ figure out if passing a run object is ideal; see notes in collator impl
        #       (mp-init hangs as of wandb 0.22.3; passing the object causes race conditions
        #        and unreliable logs instead, as some values will be continuously overwritten)
        wandb_run_or_init_kwargs = wandb_run
        # wandb_run_or_init_kwargs = {
        #     "project": wandb_run.project,
        #     "entity": wandb_run.entity,
        #     "id": wandb_run.id,
        #     "dir": wandb_run.dir,
        # }
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


def instantiate_tokenizer(
    config: HFTrainerAppMainConfig,
) -> transformers.PreTrainedTokenizer:
    """Instantiates and returns the tokenizer tied to the config's targeted base model."""
    import pyine.utils.distrib
    rank = pyine.utils.distrib.get_global_rank(default=0)
    world_size = pyine.utils.distrib.get_world_size(default=1)
    is_main = pyine.utils.distrib.is_main_process()
    logger.info(f"[RANK {rank}/{world_size}] setting up tokenizer for: {config.base_model}")
    logger.debug(f"auto tokenizer config: {config.auto_tokenizer_config}")

    # Rank 0 downloads first to avoid file conflicts
    if is_main:
        logger.info(f"[RANK {rank}] downloading/loading tokenizer (other ranks will wait)")
        tokenizer = pyine.utils.tokenizers.get_hf_tokenizer(
            pretrained_model_name_or_path=config.base_model,
            set_padding_to_eos_if_needed=config.tokenizer_set_padding_to_eos_if_needed,
            override_padding_to_right_side=config.tokenizer_override_padding_to_right_side,
            override_truncation_to_left_side=config.tokenizer_override_truncation_to_left_side,
            **config.auto_tokenizer_config,
        )

    # Wait for rank 0 to finish downloading
    if world_size > 1:
        logger.info(f"[RANK {rank}] waiting for rank 0 to finish downloading tokenizer")
        pyine.utils.distrib.barrier()

    # Now all other ranks can safely load from cache
    if not is_main:
        logger.info(f"[RANK {rank}] loading tokenizer from cache")
        tokenizer = pyine.utils.tokenizers.get_hf_tokenizer(
            pretrained_model_name_or_path=config.base_model,
            set_padding_to_eos_if_needed=config.tokenizer_set_padding_to_eos_if_needed,
            override_padding_to_right_side=config.tokenizer_override_padding_to_right_side,
            override_truncation_to_left_side=config.tokenizer_override_truncation_to_left_side,
            **config.auto_tokenizer_config,
        )
    logger.info(f"[RANK {rank}/{world_size}] tokenizer successfully created ({type(tokenizer).__name__})")
    logger.debug(f"tokenizer is_fast: {getattr(tokenizer, "is_fast", False)}")
    logger.debug(f"tokenizer vocab size: {len(tokenizer)}")
    return tokenizer


def instantiate_model(config: HFTrainerAppMainConfig) -> transformers.PreTrainedModel:
    """Instantiates and returns the base pretrained model specified in the config.

    Note: this function will NOT load the model weights tied to the resume ckpt which might be
    specified in the app config; it only instantiates the base model with its pretrained weights.
    """
    import pyine.utils.distrib
    rank = pyine.utils.distrib.get_global_rank(default=0)
    world_size = pyine.utils.distrib.get_world_size(default=1)
    is_main = pyine.utils.distrib.is_main_process()
    logger.info(f"[RANK {rank}/{world_size}] setting up model: {config.base_model}")
    dtype, device_map = config.target_dtype, config.device_map
    model_kwargs: dict[str, typing.Any] = {
        "dtype": dtype,
        "device_map": device_map,
        **config.auto_model_config,
    }
    if config.quantization_mode == "qlora":
        logger.info("  (setting up model using QLoRA 4-bit quantization)")
        quant_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = quant_config
    elif config.quantization_mode == "none":
        logger.info("  (setting up model using no quantization)")
    else:
        raise ValueError(f"unsupported quantization_mode: {config.quantization_mode}")
    logger.debug(f"auto model config: {model_kwargs}")

    # Rank 0 downloads first to avoid file conflicts
    if is_main:
        logger.info(f"[RANK {rank}] downloading/loading model (other ranks will wait)")
        base_model = typing.cast(
            "transformers.PreTrainedModel",
            transformers.AutoModelForCausalLM.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
                config.base_model,
                **model_kwargs,
            ),
        )

    # Wait for rank 0 to finish downloading
    if world_size > 1:
        logger.info(f"[RANK {rank}] waiting for rank 0 to finish downloading model")
        pyine.utils.distrib.barrier()

    # Now all other ranks can safely load from cache
    if not is_main:
        logger.info(f"[RANK {rank}] loading model from cache")
        base_model = typing.cast(
            "transformers.PreTrainedModel",
            transformers.AutoModelForCausalLM.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
                config.base_model,
                **model_kwargs,
            ),
        )
    model: transformers.PreTrainedModel = base_model
    if config.lora_config is not None:
        logger.info("  (setting up LoRA adapters)")
        logger.debug(f"lora_config: {config.lora_config}")
        if isinstance(config.lora_config, pyine.utils.transformers.LoraConfig):
            lora_peft_config = config.lora_config.to_peft_config()
        else:
            assert isinstance(config.lora_config, peft.LoraConfig)
            lora_peft_config = config.lora_config
        model = typing.cast("transformers.PreTrainedModel", peft.get_peft_model(model, lora_peft_config))
    logger.info(f"[RANK {rank}/{world_size}] model successfully created:\n{model}")
    model_config = getattr(model, "config", None)
    if hasattr(model_config, "to_json_string") and callable(model_config.to_json_string):
        logger.debug(f"model config: {model_config.to_json_string()}")
    if hasattr(model, "peft_config"):
        logger.debug(f"model peft_config: {model.peft_config}")
    get_trainable_params = getattr(model, "get_nb_trainable_parameters", None)
    if callable(get_trainable_params):
        trainable_param_count, total_param_count = typing.cast(
            "tuple[int, int]",
            get_trainable_params(),
        )
        logger.info(f"trainable param count: {trainable_param_count:,d}")
        logger.info(f"total param count: {total_param_count:,d}")
        if total_param_count:
            trainable_ratio = (100 * trainable_param_count) / total_param_count
            logger.info(f"trainable param %: {trainable_ratio:.3f}")
    return model


def _async_main_wrapper(
    config: HFTrainerAppMainConfig,
    runtime: pyine.configs.schemas.RuntimeConfig | None = None,
) -> None:
    """Wrapper for async main function."""
    import pyine.apps.trainers.hf_trainer as hf_trainer_app

    asyncio.run(hf_trainer_app.main(config=config, runtime=runtime))


@torch.distributed.elastic.multiprocessing.errors.record
def hydra_main(eval_type: pyine.evals.common.EvalType) -> None:
    """Hydra main entrypoint for the HuggingFace trainer app."""
    pyine.configs.base.register_searchpath_plugin()
    _ = register_hydra_configs(eval_type=eval_type)
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
    base_config = pyine.configs.utils.make_config_description(
        pyine.utils.transformers.TrainingArgsConfig,
        name="base",
        group=group,
        description=(
            "Base training arguments for all trainer configs; auto-determines some arguments "
            "based on available hardware, and fills other arguments based on runtime config."
        ),
        config={
            # some args are auto-deduced from hardware
            "use_cpu": use_cpu,
            "fp16": use_fp16,
            "bf16": use_bf16,
            "tf32": is_cuda,
            "dataloader_pin_memory": pin_mem,
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
    # TODO @@@@@@ add distrib trainer config with local_rank and ddp/fsdp stuff? or deepspeed/accelerate?
    # TODO @@@@@@ add configs w/ debug settings? (and tokens/sec or tokens seen metrics?)
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


def _get_lora_configs(
    group: str,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Generates and returns LoRA adaptor configs for hydra zen storage."""
    default_lora_config = pyine.configs.utils.make_config_description(
        peft.LoraConfig,
        name="default",
        group=group,
        description=(
            "Default LoRA settings for all trainer configs; applies to all models, and provides "
            "a reasonable default for LoRA adaptation. See `peft.LoraConfig` for more details."
        ),
        config={
            # -------------
            "populate_full_signature": True,
            "hydra_convert": "object",
        },
    )
    return [default_lora_config]


def _get_app_configs(
    eval_type: pyine.evals.common.EvalType,
    group: str,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Generates and returns application configs for hydra zen storage."""
    assert isinstance(group, str) and group
    app_main_config = pyine.configs.utils.make_config_description(
        HFTrainerAppMainConfig,
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
                {"datamodule_config": "base"},
                {"training_args_config": "base"},
                # {"lora_config": "null"},  # left out here = deactivated (null)
                {"evals_config": "base"},
            ],
        },
    )
    datamodule_configs = pyine.organisms.datamodules.shortcuts_configs.get_configs(
        eval_type=eval_type,
        group=f"{group}/datamodule_config",
    )
    trainer_args_configs = _get_trainer_args_configs(group=f"{group}/training_args_config")
    lora_configs = _get_lora_configs(group=f"{group}/lora_config")
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
        config for config in app_configs if config.group == "config/datamodule_config" and config.name != "base"
    ]
    trainer_configs = [config for config in app_configs if config.group == "config"]
    # for each datamodule config and trainer config combination, create an experiment config
    # (note: we don't do anything eval_type-specific here, at least not for these base exp configs)
    outputs: list[pyine.configs.schemas.ConfigDescription] = []
    for dm_config, trainer_config in itertools.product(dm_configs, trainer_configs):
        exp_name = f"{dm_config.name}_{trainer_config.name}"
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
        _async_main_wrapper,
        name="entrypoint",
        group=None,
        description="Entrypoint settings for the HuggingFace trainer app.",
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
    store, base_configs = pyine.configs.base.get_base_store_and_configs("hf_trainer")
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
        app_name="hf_trainer",
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
    # TODO: if we ever have more than one eval type, add a selector based on launch args here
    pyine.configs.utils.print_experiment_configs(
        config_descriptions=register_hydra_configs(eval_type=pyine.evals.common.EvalType.CODE_EXEC),
        app_name="hf_trainer",
    )
