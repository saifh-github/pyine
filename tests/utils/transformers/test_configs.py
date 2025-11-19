import pathlib

import peft
import pydantic
import pytest
import transformers

from pyine.utils.transformers.configs import GenerationConfig, LoraConfig, TrainingArgsConfig


def test_all_configs_are_pydantic_models() -> None:
    """Test that all config wrappers are pydantic BaseModels."""
    assert issubclass(TrainingArgsConfig, pydantic.BaseModel)
    assert issubclass(GenerationConfig, pydantic.BaseModel)
    assert issubclass(LoraConfig, pydantic.BaseModel)


class TestTrainingArgsConfig:
    """Tests for TrainingArgsConfig wrapper."""

    def test_instantiate_with_required_params(self, tmp_path: pathlib.Path) -> None:
        """Test creating TrainingArgsConfig with minimal required parameters."""
        config = TrainingArgsConfig(output_dir=str(tmp_path))
        assert config.output_dir == str(tmp_path)

    def test_default_overrides_applied(self, tmp_path: pathlib.Path) -> None:
        """Test that custom default overrides are applied correctly."""
        config = TrainingArgsConfig(output_dir=str(tmp_path))
        assert config.load_best_model_at_end is True
        assert config.logging_first_step is True
        assert config.log_level == "info"
        assert config.report_to == "none"
        assert config.lr_scheduler_kwargs == {}
        assert config.include_for_metrics == []

    def test_override_default_values(self, tmp_path: pathlib.Path) -> None:
        """Test that default overrides can be overridden."""
        config = TrainingArgsConfig(
            output_dir=str(tmp_path),
            load_best_model_at_end=False,
            logging_first_step=False,
            log_level="warning",
        )
        assert config.load_best_model_at_end is False
        assert config.logging_first_step is False
        assert config.log_level == "warning"

    def test_converts_to_hf_training_arguments(self, tmp_path: pathlib.Path) -> None:
        """Test that config can be converted to HF TrainingArguments."""
        config = TrainingArgsConfig(
            output_dir=str(tmp_path),
            num_train_epochs=3,
            per_device_train_batch_size=8,
            learning_rate=5e-5,
            eval_strategy="steps",
            save_strategy="steps",
        )
        hf_args = transformers.TrainingArguments(**config.model_dump())
        assert isinstance(hf_args, transformers.TrainingArguments)
        assert hf_args.output_dir == str(tmp_path)
        assert hf_args.num_train_epochs == 3
        assert hf_args.per_device_train_batch_size == 8
        assert hf_args.learning_rate == 5e-5
        assert hf_args.load_best_model_at_end is True
        assert hf_args.eval_strategy == "steps"
        assert hf_args.save_strategy == "steps"

    def test_config_is_frozen(self, tmp_path: pathlib.Path) -> None:
        """Test that TrainingArgsConfig is immutable."""
        config = TrainingArgsConfig(output_dir=str(tmp_path))
        with pytest.raises(pydantic.ValidationError):
            config.output_dir = "/new/path"

    def test_extra_fields_forbidden(self, tmp_path: pathlib.Path) -> None:
        """Test that extra fields are not allowed."""
        with pytest.raises(pydantic.ValidationError):
            TrainingArgsConfig(output_dir=str(tmp_path), invalid_field="value")

    def test_training_args_with_multiple_overrides(self, tmp_path: pathlib.Path) -> None:
        """Test TrainingArgsConfig with multiple parameter overrides."""
        config = TrainingArgsConfig(
            output_dir=str(tmp_path),
            num_train_epochs=5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            learning_rate=3e-5,
            warmup_steps=100,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        )
        hf_args = transformers.TrainingArguments(**config.model_dump())
        assert hf_args.num_train_epochs == 5
        assert hf_args.learning_rate == 3e-5
        assert hf_args.load_best_model_at_end is True
        assert hf_args.metric_for_best_model == "eval_loss"


class TestGenerationConfig:
    """Tests for GenerationConfig wrapper."""

    def test_instantiate_with_defaults(self) -> None:
        """Test creating GenerationConfig with default parameters."""
        config = GenerationConfig()
        assert config is not None

    def test_default_overrides_applied(self) -> None:
        """Test that custom default overrides are applied correctly."""
        config = GenerationConfig()
        assert config.max_new_tokens == 128

    def test_override_default_values(self) -> None:
        """Test that default overrides can be overridden."""
        config = GenerationConfig(max_new_tokens=256)
        assert config.max_new_tokens == 256

    def test_set_generation_parameters(self) -> None:
        """Test setting various generation parameters."""
        config = GenerationConfig(
            max_new_tokens=512,
            top_p=0.9,
            do_sample=True,
        )
        assert config.max_new_tokens == 512
        assert config.top_p == 0.9
        assert config.do_sample is True

    def test_converts_to_hf_generation_config(self) -> None:
        """Test that config can be converted to HF GenerationConfig."""
        config = GenerationConfig(
            max_new_tokens=256,
            top_k=50,
        )
        hf_config = transformers.GenerationConfig(**config.model_dump())
        assert isinstance(hf_config, transformers.GenerationConfig)
        assert hf_config.max_new_tokens == 256
        assert hf_config.top_k == 50

    def test_config_is_frozen(self) -> None:
        """Test that GenerationConfig is immutable."""
        config = GenerationConfig()
        with pytest.raises(pydantic.ValidationError):
            config.max_new_tokens = 512


class TestLoraConfig:
    """Tests for LoraConfig wrapper."""

    def test_instantiate_with_required_params(self) -> None:
        """Test creating LoraConfig with minimal required parameters."""
        config = LoraConfig(task_type="CAUSAL_LM")
        assert config.task_type == "CAUSAL_LM"

    def test_set_lora_parameters(self) -> None:
        """Test setting various LoRA parameters."""
        config = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
        )
        assert config.task_type == "CAUSAL_LM"
        assert config.r == 8
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.1
        assert config.target_modules == ["q_proj", "v_proj"]

    def test_target_modules_as_string(self) -> None:
        """Test target_modules can be a string."""
        config = LoraConfig(task_type="CAUSAL_LM", target_modules="all-linear")
        assert config.target_modules == "all-linear"

    def test_target_modules_as_set(self) -> None:
        """Test target_modules can be a set."""
        config = LoraConfig(task_type="CAUSAL_LM", target_modules={"q_proj", "v_proj"})
        assert config.target_modules == {"q_proj", "v_proj"}

    def test_target_modules_as_list(self) -> None:
        """Test target_modules can be a list."""
        config = LoraConfig(task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj"])
        assert config.target_modules == ["q_proj", "k_proj", "v_proj"]

    def test_converts_to_peft_lora_config(self) -> None:
        """Test that config can be converted to PEFT LoraConfig."""
        config = LoraConfig(
            task_type="CAUSAL_LM",
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
        )
        peft_config = peft.LoraConfig(**config.model_dump())
        assert isinstance(peft_config, peft.LoraConfig)
        assert peft_config.task_type == "CAUSAL_LM"
        assert peft_config.r == 16
        assert peft_config.lora_alpha == 32
        assert peft_config.target_modules == {"q_proj", "v_proj"}

    def test_to_peft_config_method(self) -> None:
        """Test the to_peft_config helper method."""
        config = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
        )
        peft_config = config.to_peft_config()
        assert isinstance(peft_config, peft.LoraConfig)
        assert peft_config.task_type == "CAUSAL_LM"
        assert peft_config.r == 8
        assert peft_config.lora_alpha == 16

    def test_to_peft_config_with_runtime_config_dict(self) -> None:
        """Test to_peft_config converts runtime_config dict correctly."""
        config = LoraConfig(
            task_type="CAUSAL_LM",
            runtime_config={"ephemeral_gpu_offload": True},
        )
        peft_config = config.to_peft_config()
        assert isinstance(peft_config, peft.LoraConfig)
        assert isinstance(peft_config.runtime_config, peft.LoraRuntimeConfig)
        assert peft_config.runtime_config.ephemeral_gpu_offload is True

    def test_config_is_frozen(self) -> None:
        """Test that LoraConfig is immutable."""
        config = LoraConfig(task_type="CAUSAL_LM")
        with pytest.raises(pydantic.ValidationError):
            config.r = 16

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are not allowed."""
        with pytest.raises(pydantic.ValidationError):
            LoraConfig(task_type="CAUSAL_LM", invalid_field="value")

    def test_lora_config_serialization(self) -> None:
        """Test LoraConfig can be serialized and deserialized."""
        config = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            target_modules=["q_proj", "v_proj"],
        )
        serialized = config.model_dump()
        deserialized = LoraConfig(**serialized)
        assert deserialized.task_type == "CAUSAL_LM"
        assert deserialized.r == 8
        assert deserialized.target_modules == ["q_proj", "v_proj"]

    def test_lora_config_to_peft_preserves_runtime_config(self) -> None:
        """Test that LoraConfig.to_peft_config preserves runtime_config."""
        lora_config = LoraConfig.model_validate({})
        peft_config = lora_config.to_peft_config()
        assert isinstance(peft_config, peft.LoraConfig)
        assert isinstance(peft_config.runtime_config, peft.LoraRuntimeConfig)
        assert hasattr(peft_config.runtime_config, "ephemeral_gpu_offload")
