"""Integration test for the HF trainer app (SFT branch) with mocked trace data.

This test verifies the end-to-end SFT training pipeline using a real (small) model and mocked trace
data. It covers:
- Model and tokenizer setup;
- Train/valid dataset preparation;
- Actual training loop execution;
- Checkpointing;
- Evaluation/prediction loop;
- Logging.

All outputs (checkpoints, logs, caches) are written to temporary directories to avoid polluting
real caches or logs.
"""

import math
import pathlib
import typing

import pytest
import torch
import torch.utils.data
import transformers

import pyine.apps.trainers.hf_sft_trainer_configs
import pyine.apps.trainers.hf_trainer
import pyine.configs.schemas
import pyine.data.datamodule
import pyine.evals.code_exec.configs
import pyine.evals.utils
import pyine.organisms.datamodules.shortcuts
import pyine.utils.portability
import pyine.utils.transformers
import tests.apps.trainers.conftest
import tests.env_checks


@pytest.fixture
def sft_training_config(
    tmp_path: pathlib.Path,
    real_datamodule_config: pyine.organisms.datamodules.shortcuts.ShortcutBiasDataModuleConfig,
) -> pyine.apps.trainers.hf_sft_trainer_configs.SFTTrainerAppMainConfig:
    """Create a minimal SFT training config for integration testing.

    All outputs (checkpoints, logs) are written to tmp_path to avoid polluting real directories.
    """
    output_dir = tmp_path / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir = tmp_path / "logs"
    logging_dir.mkdir(parents=True, exist_ok=True)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    training_args = pyine.utils.transformers.TrainingArgsConfig(
        output_dir=str(output_dir),
        logging_dir=str(logging_dir),
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        max_steps=10,
        eval_strategy="steps",
        eval_steps=5,
        save_strategy="steps",
        save_steps=5,
        save_total_limit=2,
        logging_steps=2,
        logging_first_step=True,
        report_to=[],  # should be set by trainer when runtime.wandb_run is available
        dataloader_num_workers=0,
        seed=42,
        fp16=use_fp16,
        bf16=use_bf16,
        remove_unused_columns=False,  # preserve prompt_len for collator's label masking
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    return pyine.apps.trainers.hf_sft_trainer_configs.SFTTrainerAppMainConfig(
        base_model="HuggingFaceTB/SmolLM-360M-Instruct",
        quantization_mode="none",
        auto_model_config={"low_cpu_mem_usage": True},
        training_args_config=training_args,
        datamodule_config=real_datamodule_config,
        use_wandb_logging=True,
        evals_config=pyine.evals.code_exec.configs.CodeExecEvalsConfig(
            eval_batch_size=2,
            category_extraction_config=pyine.evals.utils.SampleCategoryExtractionConfig(
                enabled_fields=frozenset({pyine.evals.utils.SampleCategoryField.code_type}),
            ),
        ),
    )


@pytest.fixture
def sft_runtime_config(
    tmp_path: pathlib.Path,
) -> typing.Generator[pyine.configs.schemas.RuntimeConfig, None, None]:
    """Create a runtime config with wandb logging for SFT integration testing."""
    output_dir = tmp_path / "runtime_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = pyine.configs.schemas.RuntimeConfig(
        exp_name="hf-trainer-test",
        run_name="sft-test",
        run_group="integration-tests",
        app_name="test_hf_trainer_sft_integration",
        output_dir=str(output_dir),
        seed=42,
    )
    runtime.init_wandb()
    yield runtime
    runtime.finalize()


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(
    tests.env_checks.NO_ACCELERATOR_AVAILABLE,
    reason="Accelerator (any) required for SFT integration test",
)
@pytest.mark.skipif(
    tests.env_checks.HF_ACCESS_TOKEN_MISSING,
    reason="HuggingFace access token required for model download",
)
@pytest.mark.skipif(
    tests.env_checks.TACO_TRACES_DATASET_MISSING,
    reason="TACO traces dataset required for real datamodule",
)
@pytest.mark.skipif(
    tests.env_checks.TACO_TRACES_DATASET_SPLIT_MISSING,
    reason="TACO traces dataset split required for real datamodule",
)
class TestHFTrainerSFTIntegration:
    """Integration tests for the HF trainer SFT training pipeline."""

    def test_sft_training_pipeline_end_to_end(
        self,
        isolate_caches_and_outputs: None,
        real_datamodule_config: pyine.organisms.datamodules.shortcuts.ShortcutBiasDataModuleConfig,
        sft_training_config: pyine.apps.trainers.hf_sft_trainer_configs.SFTTrainerAppMainConfig,
        sft_runtime_config: pyine.configs.schemas.RuntimeConfig,
    ) -> None:
        """Test the complete SFT training pipeline with real trace data."""
        real_datamodule = pyine.organisms.datamodules.shortcuts.ShortcutBiasDataModule(real_datamodule_config)
        real_datamodule.prepare_data()
        real_datamodule.setup()
        trainer = pyine.apps.trainers.hf_trainer.sft_train(
            datamodule=real_datamodule,
            config=sft_training_config,
            runtime=sft_runtime_config,
            resume_artifacts=None,
            shutdown_manager=None,
        )
        # verify training completed
        assert trainer.state.global_step == 10, "training should complete 10 steps"
        # verify checkpoints were saved
        output_dir = pathlib.Path(sft_training_config.training_args_config.output_dir)
        checkpoints = sorted(output_dir.glob("checkpoint-*"))
        assert len(checkpoints) >= 1, "at least one checkpoint should be saved"
        checkpoint_dir = checkpoints[-1]
        assert (checkpoint_dir / "trainer_state.json").exists(), "checkpoint should contain trainer_state.json"
        assert (checkpoint_dir / "config.json").exists(), "checkpoint should contain config.json"
        has_model_weights = (
            (checkpoint_dir / "model.safetensors").exists()
            or (checkpoint_dir / "pytorch_model.bin").exists()
            or (checkpoint_dir / "adapter_model.safetensors").exists()
        )
        assert has_model_weights, "checkpoint should contain model weights"
        # verify model is in expected state
        assert trainer.model is not None
        assert hasattr(trainer.model, "config")
        # verify best model was loaded at end (load_best_model_at_end=True)
        best_checkpoint_path = tests.apps.trainers.conftest.verify_best_model_loaded(
            trainer=trainer,
            model=trainer.model,
        )
        assert best_checkpoint_path.exists(), "best checkpoint should exist"
        # verify evaluation was run (in trainer log history)
        assert trainer.state.log_history is not None
        eval_logs = [log for log in trainer.state.log_history if "eval_loss" in log]
        assert len(eval_logs) >= 1, "evaluation should have been run at least once"
        category_metric_keys = [
            key for log in trainer.state.log_history for key in log if "code_type/" in key and key.endswith("/loss")
        ]
        assert category_metric_keys, "category-wise metrics should be logged (e.g., code_type/*/loss)"

        # verify we can generate text post-training
        model = trainer.model
        tokenizer = trainer.processing_class
        assert isinstance(tokenizer, (transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast))
        model.eval()
        test_prompt = "x = 5\ny = 3\nprint(x + y)"
        messages = [
            {"role": "user", "content": f"What is the output of this code?\n```python\n{test_prompt}\n```"},
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assert len(generated_text) > len(input_text), "model should generate some output"

        # verify loss is a reasonable value (not NaN or extremely large)
        for log in eval_logs:
            eval_loss = log["eval_loss"]
            assert not math.isnan(eval_loss), "eval_loss should not be NaN"
            assert eval_loss < 100, "eval_loss should be reasonable"

        # verify wandb logging was active and properly configured
        assert sft_runtime_config.wandb_run is not None, "wandb run should be initialized"
        assert sft_runtime_config.wandb_run.name == "sft-test"

        # verify expected training metrics were logged to wandb
        wandb_metrics = tests.apps.trainers.conftest.verify_wandb_training_metrics(
            wandb_run=sft_runtime_config.wandb_run,
        )
        assert "train/loss" in wandb_metrics or any(key.startswith("train/loss") for key in wandb_metrics), (
            "train/loss should be in wandb metrics"
        )
        assert "eval/loss" in wandb_metrics or any(key.startswith("eval/loss") for key in wandb_metrics), (
            "eval/loss should be in wandb metrics (since do_eval=True)"
        )

        # verify category-wise metrics were logged to wandb
        expected_categories = ["code_type/original"]  # not checking hinted, might not have annotations
        tests.apps.trainers.conftest.verify_wandb_has_category_metrics(
            wandb_run=sft_runtime_config.wandb_run,
            expected_categories=expected_categories,
            metric_suffix="loss",
        )
