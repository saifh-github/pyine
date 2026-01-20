"""Hydra-zen config builder for RL training with GRPO.

If you execute this script directly, it will print all available experiment configs for this app.
"""

import logging
import pathlib
import typing

import pydantic
import torch
import trl

import pyine.apps.trainers.common as common
import pyine.configs.base
import pyine.configs.schemas
import pyine.configs.searchpath
import pyine.configs.utils
import pyine.evals.common
import pyine.evals.configs
import pyine.organisms.datamodules
import pyine.organisms.models.rewards.core.configs
import pyine.utils.reprod

logger = logging.getLogger(__name__)


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


class RLTrainerAppMainConfig(common.AppMainConfig, common.ModelTokenizerConfigBase):
    """Configuration for RL training with GRPO.

    Inherits model/tokenizer configuration from ModelTokenizerConfigBase.
    """

    # --------------- GRPO-specific settings ---------------

    grpo_config: pydantic.SerializeAsAny[trl.GRPOConfig] = pydantic.Field(  # type: ignore[reportPrivateImportUsage]
        ...,  # MISSING! MANDATORY!
        description="GRPO training configuration (TRL GRPOConfig).",
    )

    reward_manager_config: pyine.organisms.models.rewards.core.configs.RewardManagerConfig = pydantic.Field(
        ...,  # MISSING! MANDATORY!
        description="Reward manager configuration for RL training.",
    )

    cache_config: CacheConfig = pydantic.Field(
        default_factory=CacheConfig,
        description="Dataset caching configuration.",
    )

    # --------------- RL-specific methods ---------------

    @property
    @typing.override
    def target_dtype(self) -> torch.dtype:
        """Returns the target dtype to use with models based on GRPO config."""
        if self.grpo_config.bf16:
            return torch.bfloat16
        return torch.float16 if self.grpo_config.fp16 else torch.float32

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


def _get_grpo_configs(
    group: str,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Generates and returns GRPO training configs for hydra zen storage."""
    hw_flags = common.get_hardware_training_flags()
    base_config = pyine.configs.utils.make_config_description(
        trl.GRPOConfig,  # type: ignore[reportPrivateImportUsage]
        name="base",
        group=group,
        description=(
            "Base GRPO training arguments for all RL trainer configs; auto-determines some arguments "
            "based on available hardware, and fills other arguments based on runtime config."
        ),
        config={
            **hw_flags,  # hardware-specific flags (use_cpu, fp16, bf16, tf32, dataloader_pin_memory)
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
            "per_device_eval_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 1e-5,
            "warmup_ratio": 0.1,
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
            "scale_rewards": "group",
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


def _get_default_code_exec_reward_manager_configs(
    group: str,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Generates and returns reward manager code execution configs for hydra zen storage."""
    hard_match_config = pyine.configs.utils.make_config_description(
        pyine.organisms.models.rewards.core.configs.RewardManagerConfig,
        name="hard_matching",
        group=group,
        description=(
            "Base configuration for the reward manager for code exec outcome hard-matching. "
            "Only defines the hard-match term as the source of reward, and sets up basic parsing."
        ),
        config={
            "terms": [
                {
                    "name": "hard_match",
                    "type": "code_exec/hard_match",
                    "weight": 1.0,
                    "enabled": True,
                    "require_parsed": True,
                    "params": {
                        "reward_if_match": 1.0,
                        "reward_if_no_match": 0.0,
                        "strip_whitespace": True,
                    },
                },
            ],
            "parsing": {
                "mode": "tags",
                "enabled_fields": "final_only",
                "final_tag": "final",
                "reasoning_from_outside_final": True,
                "fallback_policy": "none",
                "multi_tag_policy": "last",
                "strict": False,
                "capture_diagnostics": True,
            },
        },
    )
    # todo: add defaults for soft-matching, llm-grader, ...
    return [hard_match_config]


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
    if eval_type == pyine.evals.common.EvalType.CODE_EXEC:
        reward_manager_configs = _get_default_code_exec_reward_manager_configs(group=f"{group}/reward_manager_config")
    else:
        raise NotImplementedError(f"RL trainer does not yet support eval type {eval_type}")
    lora_configs = common.get_lora_configs(group=f"{group}/lora_config", app_description="RL trainer")
    evals_configs = pyine.evals.configs.get_evals_configs(eval_type=eval_type, group=f"{group}/evals_config")
    return [
        app_main_config,
        *datamodule_configs,
        *grpo_configs,
        *reward_manager_configs,
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
    rwm_configs = [config for config in app_configs if config.group == "config/reward_manager_config"]
    trainer_configs = [config for config in app_configs if config.group == "config"]
    # For each datamodule, trainer, and reward manager combination, create an experiment config
    outputs: list[pyine.configs.schemas.ConfigDescription] = []
    for dm_config, trainer_config, rwm_config in [
        (dm, tc, rwmc) for dm in dm_configs for tc in trainer_configs for rwmc in rwm_configs
    ]:
        exp_name = f"hf_rl_{dm_config.name}_{trainer_config.name}_{rwm_config.name}"
        outputs.append(
            pyine.configs.utils.make_config_description(
                name=exp_name,
                group=group,
                package=package,
                description=(
                    f"RL experiment config that combines the '{dm_config.name}' datamodule settings with the "
                    f"'{trainer_config.name}' trainer and '{rwm_config.name}' settings for the app's entrypoint.\n\n"
                    f"Description for 'config={trainer_config.name}': {trainer_config.description}\n\n"
                    f"Description for 'config/datamodule_config={dm_config.name}': {dm_config.description}\n\n"
                    f"Description for 'config/reward_manager_config={rwm_config.name}': {rwm_config.description}\n\n"
                ),
                config={
                    "runtime": {"exp_name": exp_name},
                    # -------------
                    "hydra_defaults": [
                        "_self_",
                        {"/config/reward_manager_config": rwm_config.name},
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
        common.async_hf_trainer_main_wrapper,
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
    # TODO: if we ever have more than one eval type, add a selector based on launch args here
    pyine.configs.utils.print_experiment_configs(
        config_descriptions=register_hydra_configs(eval_type=pyine.evals.common.EvalType.CODE_EXEC),
        app_name="hf_trainer",
    )
