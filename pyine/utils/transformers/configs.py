import typing

import peft
import pydantic
import transformers

import pyine.utils.pydantic

__all__ = [
    "TrainingArgsConfig",
    "GenerationConfig",
    "LoraConfig",
]

if typing.TYPE_CHECKING:

    class TrainingArgsConfig(pydantic.BaseModel):
        """Stubbed interface for the HuggingFace Trainer Arguments config class defined below."""

        def __getattr__(self, name: str) -> typing.Any: ...

    class GenerationConfig(pydantic.BaseModel):
        """Stubbed interface for the HuggingFace text generation pipeline config class defined below."""

        def __getattr__(self, name: str) -> typing.Any: ...

    class LoraConfig(pydantic.BaseModel):
        """Stubbed interface for the HuggingFace PEFT LoRA config class defined below."""

        def __getattr__(self, name: str) -> typing.Any: ...

else:
    TrainingArgsConfig = pyine.utils.pydantic.model_from_callable(
        fn=transformers.TrainingArguments,
        name="TrainingArgsConfig",
        model_config=pydantic.ConfigDict(frozen=True, extra="forbid"),
        default_overrides={
            # we use some updated defaults (low-impact, QoL stuff)
            "load_best_model_at_end": True,  # easy to forget, but important! (also force-saves best ckpt)
            "logging_first_step": True,  # good for plotting/sanity
            "log_level": "info",  # enable info-level logging for models by default
            "report_to": "none",  # disable by default, and enable at runtime if needed
            # we also need to replace some defaults that CANNOT be serialized (factories)
            "lr_scheduler_kwargs": {},  # same behavior as original default
            "include_for_metrics": [],  # same behavior as original default
        },
    )
    """Configuration parameters for the HuggingFace Trainer."""

    GenerationConfig = pyine.utils.pydantic.model_from_callable(
        fn=transformers.GenerationConfig,
        name="GenerationConfig",
        model_config=pydantic.ConfigDict(frozen=True, extra="allow"),
        exclude={"kwargs"},
        type_overrides={
            "max_new_tokens": int,
        },
        default_overrides={
            "max_new_tokens": 128,
        },
    )
    """Configuration parameters for the HuggingFace text generation pipeline."""

    LoraConfig = pyine.utils.pydantic.model_from_callable(
        fn=peft.LoraConfig,
        name="LoraConfig",
        model_config=pydantic.ConfigDict(frozen=True, extra="forbid"),
        type_overrides={"target_modules": (set[str] | list[str] | str | None)},
    )
    """Configuration parameters for PEFT LoRA adapter configuration."""

    def _to_peft_config(
        self: "LoraConfig",
    ) -> peft.LoraConfig:
        """Serialization helper that converts the LoraConfig to a peft.LoraConfig."""
        data = typing.cast("dict[str, typing.Any]", self.model_dump())
        runtime_config_value = data.get("runtime_config")
        if isinstance(runtime_config_value, dict) and not isinstance(runtime_config_value, peft.LoraRuntimeConfig):
            data["runtime_config"] = peft.LoraRuntimeConfig(**runtime_config_value)
        return peft.LoraConfig(**data)

    LoraConfig.to_peft_config = _to_peft_config  # type: ignore[attr-defined]
