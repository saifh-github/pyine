"""Hydra-zen config builder for RL training with GRPO.

If you execute this script directly, it will print all available experiment configs for this app.
"""

import asyncio
import logging
import pathlib
import typing

import hydra_zen
import peft
import pydantic
import torch
import torch.distributed.elastic.multiprocessing.errors
import transformers
import trl

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


class RewardConfig(pydantic.BaseModel):
    """Configuration for reward computation in RL training."""

    reward_type: typing.Literal["code_execution"] = pydantic.Field(
        default="code_execution",
        description="Type of reward function to use.",
    )
    hard_match_reward: float = pydantic.Field(
        default=1.0,
        description="Reward value for exact matches.",
    )
    soft_match_reward: float = pydantic.Field(
        default=0.5,
        description="Reward value for soft matches (when hard match fails).",
    )
    fail_reward: float = pydantic.Field(
        default=0.0,
        description="Reward value when no match is found.",
    )
    enable_soft_match: bool = pydantic.Field(
        default=False,
        description="Whether to use soft (heuristic) matching as fallback if hard match fails.",
    )
    strip_whitespace: bool = pydantic.Field(
        default=True,
        description="Whether to strip whitespace when doing hard (exact) matching.",
    )
    expected_outputs_key: str = pydantic.Field(
        default="expected_output",
        description="Key to access expected outputs from dataset.",
    )


class CacheConfig(pydantic.BaseModel):
    """Configuration for dataset caching."""

    use_cache: bool = pydantic.Field(
        default=True,
        description="Whether to cache prepared GRPO datasets to disk.",
    )
    cache_dir: pathlib.Path | None = pydantic.Field(
        default=None,
        description="Optional custom cache directory. If None, uses default cache location.",
    )
    force_regenerate: bool = pydantic.Field(
        default=False,
        description="Whether to force regeneration of cached datasets.",
    )


class RLTrainerAppMainConfig(common.AppMainConfig):
    """Configuration for RL training with GRPO."""

    # --------------- Model settings (shared with SFT) ---------------

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

    # --------------- Tokenizer settings ---------------

    auto_tokenizer_config: dict[str, typing.Any] = pydantic.Field(
        default_factory=lambda: {"use_fast": True},
        description="Tokenizer configuration args passed to `transformers.AutoTokenizer.from_pretrained`.",
    )
    tokenizer_set_padding_to_eos_if_needed: bool = pydantic.Field(
        default=True,
        description="If True and tokenizer has no PAD token, reuse EOS token as PAD for batching.",
    )
    tokenizer_override_padding_to_right_side: bool = pydantic.Field(
        default=True,
        description="Override whichever the tokenizer's default padding side is to 'right'.",
    )
    tokenizer_override_truncation_to_left_side: bool = pydantic.Field(
        default=True,
        description="Override whichever the tokenizer's default truncation side is to 'left'.",
    )

    # --------------- GRPO-specific settings ---------------

    grpo_config: pydantic.SerializeAsAny[trl.GRPOConfig] = pydantic.Field(  # type: ignore[reportPrivateImportUsage]
        ...,  # MISSING! MANDATORY!
        description="GRPO training configuration (TRL GRPOConfig).",
    )

    reward_config: RewardConfig = pydantic.Field(
        default_factory=RewardConfig,
        description="Reward function configuration.",
    )

    prompt_version: str = pydantic.Field(
        default="grpo_minimal",
        description="Prompt template version for RL training.",
    )

    include_prompt_examples: bool = pydantic.Field(
        default=False,
        description="Whether to include few-shot examples in prompts (False recommended for GRPO).",
    )

    cache_config: CacheConfig = pydantic.Field(
        default_factory=CacheConfig,
        description="Dataset caching configuration.",
    )

    # --------------- Utility methods ---------------

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
                typed_data["lora_config"] = pyine.utils.transformers.LoraConfig.model_validate(lora_config.__dict__)
                return typed_data
            if isinstance(lora_config, dict):
                typed_lora_config = typing.cast("dict[str, typing.Any]", lora_config)
                typed_data["lora_config"] = pyine.utils.transformers.LoraConfig.model_validate(typed_lora_config)
                return typed_data
        return typing.cast("typing.Any", data)

    @property
    def target_dtype(self) -> torch.dtype:
        """Returns the target dtype to use with models."""
        if self.grpo_config.bf16:
            return torch.bfloat16
        return torch.float16 if self.grpo_config.fp16 else torch.float32

    @property
    def device_map(self) -> dict[str, torch.device | str] | str | None:
        """Returns the device map to use with models."""
        return common.get_device_map()

    def get_tokenizer(
        self,
        checkpoint_path: pathlib.Path | None = None,
    ) -> transformers.PreTrainedTokenizer:
        """Returns the tokenizer to use for the targeted model.

        Args:
            checkpoint_path: Optional path to a checkpoint to load from. If provided, loads the
                tokenizer from the checkpoint. If None, loads the tokenizer for the base model.

        Returns:
            The instantiated tokenizer.
        """
        return common.instantiate_tokenizer(
            base_model=self.base_model,
            checkpoint_path=checkpoint_path,
            auto_tokenizer_config=self.auto_tokenizer_config,
            set_padding_to_eos_if_needed=self.tokenizer_set_padding_to_eos_if_needed,
            override_padding_to_right_side=self.tokenizer_override_padding_to_right_side,
            override_truncation_to_left_side=self.tokenizer_override_truncation_to_left_side,
        )

    def get_model(
        self,
        checkpoint_path: pathlib.Path | None = None,
    ) -> transformers.PreTrainedModel:
        """Returns a model to use for experiments.

        Args:
            checkpoint_path: Optional path to a checkpoint to load from. If provided, loads the
                model from the checkpoint (with LoRA adapters if present). If None, loads the base
                pretrained model.

        Returns:
            The instantiated model.
        """
        return common.instantiate_model(
            base_model=self.base_model,
            checkpoint_path=checkpoint_path,
            target_dtype=self.target_dtype,
            device_map=self.device_map,
            auto_model_config=self.auto_model_config,
            quantization_mode=self.quantization_mode,
            lora_config=self.lora_config,
        )

    @typing.override
    def normalize_for_resume_overlap_check(
        self,
        config: common.AppMainConfig | dict[str, typing.Any] | None = None,
    ) -> dict[str, typing.Any]:
        """Normalizes the config by removing fields that might change without effects on experiments."""
        data = super().normalize_for_resume_overlap_check(config)
        assert "grpo_config" in data, "grpo_config not found in config"
        grpo_args = typing.cast("dict[str, typing.Any]", data["grpo_config"])
        # Remove fields that might change without affecting training
        grpo_args.pop("run_name", None)
        grpo_args.pop("project", None)
        grpo_args.pop("report_to", None)
        grpo_args.pop("log_level", None)
        grpo_args.pop("logging_dir", None)
        grpo_args.pop("disable_tqdm", None)
        grpo_args.pop("output_dir", None)
        grpo_args.pop("do_predict", None)
        return data


def _async_main_wrapper(
    config: RLTrainerAppMainConfig,
    runtime: pyine.configs.schemas.RuntimeConfig | None = None,
) -> None:
    """Wrapper for async main function."""
    import pyine.apps.trainers.hf_trainer as hf_trainer_app

    asyncio.run(hf_trainer_app.main(config=config, runtime=runtime))


@torch.distributed.elastic.multiprocessing.errors.record
def hydra_main(eval_type: pyine.evals.common.EvalType) -> None:
    """Hydra main entrypoint for the RL trainer app."""
    pyine.configs.base.register_searchpath_plugin()
    _ = register_hydra_configs(eval_type=eval_type)
    hydra_zen.zen(_async_main_wrapper).hydra_main(
        config_path=None,
        config_name="entrypoint",
        version_base=pyine.configs.base.target_hydra_version,
    )


def _get_grpo_configs(
    group: str,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Generates and returns GRPO training configs for hydra zen storage."""
    is_cuda = torch.cuda.is_available()
    is_mps = torch.backends.mps.is_available()
    use_cpu = not (is_cuda or is_mps)
    use_bf16 = bool(is_cuda and torch.cuda.is_bf16_supported())
    use_fp16 = bool(is_cuda and not use_bf16)
    pin_mem = bool(is_cuda)

    base_config = pyine.configs.utils.make_config_description(
        trl.GRPOConfig,  # type: ignore[reportPrivateImportUsage]
        name="base",
        group=group,
        description=(
            "Base GRPO training arguments for all RL trainer configs; auto-determines some arguments "
            "based on available hardware, and fills other arguments based on runtime config."
        ),
        config={
            # Hardware-based args
            "use_cpu": use_cpu,
            "fp16": use_fp16,
            "bf16": use_bf16,
            "tf32": is_cuda,
            "dataloader_pin_memory": pin_mem,
            # Runtime-based args
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
        trl.GRPOConfig,  # type: ignore[reportPrivateImportUsage]
        name="train_default",
        group=group,
        description=(
            "Default GRPO training arguments; applies on top of the `base` arguments config "
            "and provides reasonable defaults for GRPO training."
        ),
        config={
            "do_train": True,
            "do_eval": True,
            "do_predict": True,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "learning_rate": 1e-5,
            "warmup_steps": 100,
            "logging_steps": 5,
            "save_steps": 500,
            "save_total_limit": 3,
            "num_train_epochs": 3,
            "max_steps": -1,
            "eval_on_start": False,
            "eval_strategy": "steps",
            "eval_steps": 500,
            # GRPO-specific
            "num_generations": 4,
            "num_generations_eval": 2,
            "max_completion_length": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "beta": 0.05,
            "use_vllm": False,
            "vllm_server_port": 8000,
            "vllm_importance_sampling_correction": True,
            "gradient_checkpointing": False,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "remove_unused_columns": False,
            # -------------
            "builds_bases": (base_config.config,),
        },
    )

    eval_default_config = pyine.configs.utils.make_config_description(
        trl.GRPOConfig,  # type: ignore[reportPrivateImportUsage]
        name="eval_default",
        group=group,
        description=(
            "Default eval-only (prediction) arguments for RL trainer configs; applies on top of "
            "the `base` arguments config."
        ),
        config={
            "do_train": False,
            "do_eval": False,
            "do_predict": True,
            "per_device_eval_batch_size": 1,
            "eval_strategy": "no",
            "save_strategy": "no",
            "max_steps": 0,
            "remove_unused_columns": False,
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
            "Default LoRA settings for all RL trainer configs; applies to all models, and provides "
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
        RLTrainerAppMainConfig,
        name="base",
        group=group,
        description="Base settings for the RL trainer app.",
        config={
            "base_model": "Qwen/Qwen2.5-Coder-7B-Instruct",  # Propose this default for base experiments
            # -------------
            "populate_full_signature": True,
            "hydra_convert": "object",
            "hydra_defaults": [
                "_self_",
                {"datamodule_config": "shortcuts_base"},
                {"grpo_config": "base"},
                # {"lora_config": "null"},  # Left out here = deactivated (null)
                {"evals_config": "base"},
            ],
        },
    )
    datamodule_configs = pyine.organisms.datamodules.get_configs(
        eval_type=eval_type,
        group=f"{group}/datamodule_config",
    )
    grpo_configs = _get_grpo_configs(group=f"{group}/grpo_config")
    lora_configs = _get_lora_configs(group=f"{group}/lora_config")
    evals_configs = pyine.evals.configs.get_evals_configs(eval_type=eval_type, group=f"{group}/evals_config")
    return [
        app_main_config,
        *datamodule_configs,
        *grpo_configs,
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
    """Generates and returns experiment configs for hydra zen storage (using all datamodule configs)."""
    # Fetch and validate necessary datamodule and trainer configs from main configs set
    dm_configs = [
        config
        for config in app_configs
        if config.group == "config/datamodule_config" and not config.name.endswith("base")
    ]
    trainer_configs = [config for config in app_configs if config.group == "config"]
    # For each datamodule config and trainer config combination, create an experiment config
    outputs: list[pyine.configs.schemas.ConfigDescription] = []
    for dm_config, trainer_config in [(dm, tc) for dm in dm_configs for tc in trainer_configs]:
        exp_name = f"{dm_config.name}_{trainer_config.name}_rl"
        outputs.append(
            pyine.configs.utils.make_config_description(
                name=exp_name,
                group=group,
                package=package,
                description=(
                    f"RL experiment config that combines the '{dm_config.name}' datamodule settings with "
                    f"the '{trainer_config.name}' RL trainer settings for the app's entrypoint.\n\n"
                    f"Description for 'config={trainer_config.name}': {trainer_config.description}\n\n"
                    f"Description for 'config/datamodule_config={dm_config.name}': {dm_config.description}\n\n"
                ),
                config={
                    "runtime": {"exp_name": exp_name},
                    # -------------
                    "hydra_defaults": [
                        "_self_",
                        {"override /config": trainer_config.name},
                        {"override /config/grpo_config": "train_default"},
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
    """
    pyine.utils.reprod.load_dotenv()
    entrypoint_config = pyine.configs.utils.make_config_description(
        _async_main_wrapper,
        name="entrypoint",
        group=None,
        description="Entrypoint settings for the RL trainer app.",
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
    store, base_configs = pyine.configs.base.get_base_store_and_configs("rl_trainer")
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
        app_name="rl_trainer",
        eval_type=eval_type,
        entrypoint_config=entrypoint_config,
        app_configs=[*base_configs, *configs_to_register],
    )
    configs_to_register.extend(external_configs)
    for config in configs_to_register:
        assert config.name is not None, "config names should have been set and validated by now"
        store(typing.cast("typing.Any", config.config), name=config.name, group=config.group, package=config.package)
    store.add_to_hydra_store(overwrite_ok=True)
    return [*base_configs, *configs_to_register]


if __name__ == "__main__":
    pyine.configs.base.register_searchpath_plugin()
    pyine.configs.utils.print_experiment_configs(
        config_descriptions=register_hydra_configs(eval_type=pyine.evals.common.EvalType.CODE_EXEC),
        app_name="rl_trainer",
    )
