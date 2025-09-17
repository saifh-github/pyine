"""Hydra-zen config builder for hf_trainer.py app."""

import logging
import typing

import hydra.conf
import hydra_zen
import hydra_zen.typing
import peft
import pydantic

import pyine.apps.trainers.hf_trainer
import pyine.configs.base
import pyine.data.datamodule
import pyine.organisms.datamodules.shortcuts_configs
import pyine.utils.reprod

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    import transformers


class LoraConfig(pydantic.BaseModel):
    """Configuration for LoRA adaptation layers."""

    r: int = 16
    alpha: float = 32
    dropout: float = 0.05
    target_modules: list[str] = pydantic.Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    bias: str = "none"

    def instantiate(self) -> peft.LoraConfig:
        """Return a PEFT LoRA configuration instance."""
        return peft.LoraConfig(
            r=self.r,
            lora_alpha=int(self.alpha),
            lora_dropout=self.dropout,
            bias=self.bias,
            task_type=peft.TaskType.CAUSAL_LM,
            target_modules=self.target_modules,
        )

    def apply(
        self,
        model: "transformers.PreTrainedModel | typing.Any",
    ) -> "transformers.PreTrainedModel | typing.Any":
        """Attach LoRA adapters to the provided model and return it."""
        return peft.get_peft_model(model, self.instantiate())


class MainConfig(pydantic.BaseModel):
    """Configuration for HuggingFace LoRA fine-tuning."""

    datamodule_config: pyine.data.datamodule.ConversationDataModuleConfig
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    output_dir: str = "runs/adapter-qloRA"
    max_seq_len: int = 2048
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    num_train_epochs: float = 2.0
    max_steps: int = 0
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    seed: int = 42
    gradient_checkpointing: bool = True
    use_bf16: bool = False
    use_fp16: bool = True
    quantization_mode: typing.Literal["qlora", "none"] = "qlora"
    optimizer: str = "paged_adamw_8bit"
    dataloader_num_workers: int = 4
    tokenizer_pad_as_eos: bool = True
    lora: typing.Any = pydantic.Field(default_factory=LoraConfig)

    @pydantic.model_validator(mode="before")
    @classmethod
    def _coerce_lora_config(
        cls,
        data: typing.Any,
    ) -> typing.Any:
        """Ensure the lora field is a LoraConfig instance when provided as a dict."""
        if isinstance(data, dict):
            lora_data = data.get("lora")
            if isinstance(lora_data, dict) and not isinstance(lora_data, LoraConfig):
                data = dict(data)
                data["lora"] = LoraConfig(**lora_data)
            elif lora_data is None:
                data = dict(data)
                data["lora"] = LoraConfig()
        return data


def hydra_main() -> None:
    """Hydra main entrypoint for the HuggingFace trainer app."""
    _ = register_hydra_configs()
    hydra_zen.zen(pyine.apps.trainers.hf_trainer.main).hydra_main(
        config_path=None,
        config_name="hf_trainer_main",
        version_base=pyine.configs.base.target_hydra_version,
    )


def register_experiment_configs(
    experiment_store: hydra_zen.ZenStore,
    main_config: hydra_zen.typing.Builds,
    dm_config_names: list[str],
) -> None:
    """Register experiment configs for each available datamodule configuration."""
    base_train_config = hydra_zen.make_config(
        hydra_defaults=[
            "_self_",
            {"override /config": "base"},
        ],
        bases=(main_config,),
    )
    base_eval_only_config = hydra_zen.make_config(
        config=dict(
            eval_steps=0,
            max_steps=0,
            num_train_epochs=0.0,
            save_steps=1000,
            logging_steps=50,
        ),
        hydra_defaults=[
            "_self_",
            {"override /config": "base"},
        ],
        bases=(main_config,),
    )

    for dm_config_name in dm_config_names:
        logger.debug("registering hf_trainer experiment for datamodule=%s", dm_config_name)
        experiment_store(
            hydra_zen.make_config(
                runtime=dict(exp_name=dm_config_name),
                hydra_defaults=[
                    "_self_",
                    {"override /config/datamodule_config": dm_config_name},
                ],
                bases=(base_train_config,),
            ),
            name=dm_config_name,
        )
        experiment_store(
            hydra_zen.make_config(
                runtime=dict(exp_name=f"{dm_config_name}_eval_only"),
                hydra_defaults=[
                    "_self_",
                    {"override /config/datamodule_config": dm_config_name},
                ],
                bases=(base_eval_only_config,),
            ),
            name=f"{dm_config_name}_eval_only",
        )


def register_hydra_configs() -> hydra_zen.typing.Builds:
    """Register hf_trainer configs and return the main config builder."""
    pyine.utils.reprod.load_dotenv()
    store = pyine.configs.base.get_base_store()

    store(
        hydra.conf.JobConf(name="hf_trainer_main"),
        group="hydra/job",
        name="hf_trainer_main",
        package="hydra.job",
    )
    hf_trainer_main_config = hydra_zen.builds(
        pyine.apps.trainers.hf_trainer.main,
        # -------------
        populate_full_signature=True,
        hydra_defaults=[
            "_self_",
            {"config": "base"},
            {"runtime": "default"},
            {"hydra/job": "hf_trainer_main"},
            *pyine.configs.base.get_base_hydra_default_overrides(),
        ],
    )
    store(hf_trainer_main_config, name="hf_trainer_main")

    config_store = store(group="config")
    datamodule_config_store = config_store(group="config/datamodule_config")
    dm_config_names = pyine.organisms.datamodules.shortcuts_configs.store_hydra_configs(datamodule_config_store)

    default_lora_config = hydra_zen.builds(
        LoraConfig,
        # -------------
        populate_full_signature=True,
        hydra_convert="object",
    )
    attn_only_lora_config = hydra_zen.builds(
        LoraConfig,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        dropout=0.1,
        # -------------
        populate_full_signature=True,
        hydra_convert="object",
    )

    lora_store = config_store(group="config/lora")
    lora_store(default_lora_config, name="default")
    lora_store(attn_only_lora_config, name="attn_only")

    config_store(
        hydra_zen.builds(
            MainConfig,
            lora=default_lora_config,
            # -------------
            populate_full_signature=True,
            hydra_convert="object",
            hydra_defaults=[
                "_self_",
                {"datamodule_config": "default"},
                {"lora": "default"},
            ],
        ),
        name="base",
    )

    config_store(
        hydra_zen.builds(
            MainConfig,
            base_model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
            output_dir="runs/adapter-qwen25-coder-0_5b-m3pro",
            max_seq_len=4096,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=2,
            learning_rate=1.5e-4,
            warmup_ratio=0.1,
            num_train_epochs=3.0,
            logging_steps=5,
            eval_steps=50,
            save_steps=200,
            gradient_checkpointing=True,
            use_bf16=False,
            use_fp16=False,
            quantization_mode="none",
            optimizer="adamw_torch",
            dataloader_num_workers=0,
            lora=default_lora_config,
            # -------------
            populate_full_signature=True,
            hydra_convert="object",
            hydra_defaults=[
                "_self_",
                {"datamodule_config": "default"},
                {"lora": "default"},
            ],
        ),
        name="mac_qwen25_coder_0_5b_m3pro",
    )

    experiment_store = store(group="experiment", package="_global_")
    register_experiment_configs(
        experiment_store=experiment_store,
        main_config=hf_trainer_main_config,
        dm_config_names=dm_config_names,
    )

    store.add_to_hydra_store()
    return hf_trainer_main_config
