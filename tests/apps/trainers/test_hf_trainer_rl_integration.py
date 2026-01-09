"""Integration test for the HF trainer app (RL/GRPO branch) with mocked trace data.

This test verifies the end-to-end RL training pipeline using a real (small) model and mocked trace
data. It covers:
- Model and tokenizer setup;
- Train dataset preparation (prompts without answers);
- Reward function integration;
- Actual GRPO training loop execution;
- Checkpointing;
- Logging.

All outputs (checkpoints, logs, caches) are written to temporary directories to avoid polluting
real caches or logs.
"""

import json
import math
import pathlib
import typing

import pytest
import torch
import torch.utils.data
import transformers
import trl

import pyine.apps.trainers.common
import pyine.apps.trainers.hf_rl_trainer_configs
import pyine.apps.trainers.hf_trainer
import pyine.configs.schemas
import pyine.data.datamodule
import pyine.evals.code_exec.configs
import pyine.evals.utils
import pyine.organisms.datamodules.shortcuts
import pyine.organisms.models.rewards.core.configs
import pyine.utils.portability
import pyine.utils.transformers
import tests.apps.trainers.conftest
import tests.env_checks


@pytest.fixture
def real_datamodule_rl_config(
    real_datamodule_config: pyine.organisms.datamodules.shortcuts.ShortcutBiasDataModuleConfig,
) -> pyine.organisms.datamodules.shortcuts.ShortcutBiasDataModuleConfig:
    """Edits the datamodule config fixture so that it uses the hf messages key expected by TRL."""
    return real_datamodule_config.model_copy(update={"hf_messages_key": "prompt"})


@pytest.fixture
def rl_training_config(
    tmp_path: pathlib.Path,
    real_datamodule_rl_config: pyine.organisms.datamodules.shortcuts.ShortcutBiasDataModuleConfig,
) -> pyine.apps.trainers.hf_rl_trainer_configs.RLTrainerAppMainConfig:
    """Create a minimal RL/GRPO training config for integration testing.

    All outputs (checkpoints, logs) are written to tmp_path to avoid polluting real directories.
    """
    output_dir = tmp_path / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir = tmp_path / "logs"
    logging_dir.mkdir(parents=True, exist_ok=True)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    grpo_config = trl.GRPOConfig(
        output_dir=str(output_dir),
        logging_dir=str(logging_dir),
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,  # must be divisible by num_generations
        gradient_accumulation_steps=2,  # accumulate to maintain effective batch size
        max_steps=5,
        eval_strategy="steps",
        eval_steps=5,
        save_strategy="steps",
        save_steps=5,
        save_total_limit=1,
        logging_steps=1,
        logging_first_step=True,
        report_to=["wandb"],  # enable wandb logging for integration test
        dataloader_num_workers=0,
        seed=42,
        fp16=use_fp16,
        bf16=use_bf16,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # GRPO-specific settings for minimal test
        num_generations=2,  # keep low for speed
        max_completion_length=24,  # short completions for speed and memory
        temperature=0.7,
        beta=0.1,
        use_vllm=False,
        gradient_checkpointing=False,
    )
    # simple reward config that gives constant reward based on hard matching
    reward_manager_config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
        terms=[
            pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                name="hard_match",
                type="code_exec/hard_match",
                weight=1.0,
                enabled=True,
                require_parsed=True,
                params={
                    "reward_if_match": 1.0,
                    "reward_if_no_match": 0.0,
                    "strip_whitespace": True,
                },
            ),
        ],
        parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(
            mode="tags",
            enabled_fields="final_only",
            final_tag="final",
            reasoning_from_outside_final=False,
            fallback_policy="entire_output",  # use entire output as final answer if no tags
            multi_tag_policy="last",
            strict=False,
            capture_diagnostics=True,
        ),
        verbosity_scaling=pyine.organisms.models.rewards.core.configs.VerbosityScalingConfig(
            enabled=True,
            mode="relative",
            temperature=1.0,
            min_factor=0.1,
            max_factor=1.0,
            emit_metrics=True,
            skip_negative_rewards=True,
        ),
        logging=pyine.organisms.models.rewards.core.configs.LoggingConfig(
            enabled=True,  # enable wandb logging for reward metrics
            log_every_n_examples=1,  # log every sample for testing
            category_extraction_config=pyine.evals.utils.SampleCategoryExtractionConfig(
                enabled_fields=frozenset({pyine.evals.utils.SampleCategoryField.code_type}),
            ),
        ),
    )
    return pyine.apps.trainers.hf_rl_trainer_configs.RLTrainerAppMainConfig(
        base_model="HuggingFaceTB/SmolLM-360M-Instruct",
        quantization_mode="none",
        auto_model_config={"low_cpu_mem_usage": True},
        grpo_config=grpo_config,
        reward_manager_config=reward_manager_config,
        datamodule_config=real_datamodule_rl_config,
        use_wandb_logging=True,
        evals_config=pyine.evals.code_exec.configs.CodeExecEvalsConfig(
            eval_batch_size=2,
            category_extraction_config=pyine.evals.utils.SampleCategoryExtractionConfig(
                enabled_fields=frozenset({pyine.evals.utils.SampleCategoryField.code_type}),
            ),
        ),
    )


@pytest.fixture
def rl_runtime_config(
    tmp_path: pathlib.Path,
) -> typing.Generator[pyine.configs.schemas.RuntimeConfig, None, None]:
    """Create a runtime config with wandb logging for RL integration testing."""
    output_dir = tmp_path / "runtime_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = pyine.configs.schemas.RuntimeConfig(
        exp_name="hf-trainer-test",
        run_name="rl-test",
        run_group="integration-tests",
        app_name="test_hf_trainer_rl_integration",
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
    reason="Accelerator (any) required for RL integration test",
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
class TestHFTrainerRLIntegration:
    """Integration tests for the HF trainer RL/GRPO training pipeline."""

    def test_rl_training_pipeline_end_to_end(
        self,
        isolate_caches_and_outputs: None,
        real_datamodule_rl_config: pyine.organisms.datamodules.shortcuts.ShortcutBiasDataModuleConfig,
        rl_training_config: pyine.apps.trainers.hf_rl_trainer_configs.RLTrainerAppMainConfig,
        rl_runtime_config: pyine.configs.schemas.RuntimeConfig,
    ) -> None:
        """Test the complete RL/GRPO training pipeline with real trace data."""
        real_datamodule = pyine.organisms.datamodules.shortcuts.ShortcutBiasDataModule(real_datamodule_rl_config)
        real_datamodule.prepare_data()
        real_datamodule.setup()
        trainer = pyine.apps.trainers.hf_trainer.rl_train(
            datamodule=real_datamodule,
            config=rl_training_config,
            runtime=rl_runtime_config,
            resume_artifacts=None,
            shutdown_manager=None,
        )
        # verify training completed (should have done max_steps=5)
        assert trainer.state.global_step == 5, "training should complete 5 steps"
        # verify checkpoints were saved
        output_dir = pathlib.Path(rl_training_config.grpo_config.output_dir)
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

        # verify training produced some logs
        assert trainer.state.log_history is not None
        train_logs = [log for log in trainer.state.log_history if "loss" in log]
        assert len(train_logs) >= 1, "training logs should have been recorded"

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

        # verify loss values are reasonable (not NaN)
        for log in train_logs:
            if "loss" in log:
                loss_val = log["loss"]
                assert not math.isnan(loss_val), "loss should not be NaN"

        # verify wandb logging was active and properly configured
        assert rl_runtime_config.wandb_run is not None, "wandb run should be initialized"
        assert rl_runtime_config.wandb_run.name == "rl-test"

        # verify expected training metrics were logged to wandb
        wandb_metrics = tests.apps.trainers.conftest.verify_wandb_training_metrics(
            wandb_run=rl_runtime_config.wandb_run,
        )
        assert "train/loss" in wandb_metrics or any(key.startswith("train/loss") for key in wandb_metrics), (
            "train/loss should be in wandb metrics"
        )
        assert "train/global_step" in wandb_metrics or any(
            key.startswith("train/global_step") for key in wandb_metrics
        ), "train/global_step should be in wandb metrics"
        assert "eval/loss" in wandb_metrics or any(key.startswith("eval/loss") for key in wandb_metrics), (
            "eval/loss should be in wandb metrics (since do_eval=True)"
        )

        # verify RL-specific metrics are present in the logs (reward-related metrics from GRPO)
        rl_metric_keys = [key for log in trainer.state.log_history for key in log if "reward" in key.lower()]
        assert rl_metric_keys, "RL training should log reward-related metrics"

        # verify evaluation was run (in trainer log history)
        eval_logs = [log for log in trainer.state.log_history if "eval_loss" in log]
        assert len(eval_logs) >= 1, "evaluation should have been run at least once (since do_eval=True)"

        # verify RewardLoggingCallback was added to the trainer
        callback_classes = [type(cb).__name__ for cb in trainer.callback_handler.callbacks]
        assert "RewardLoggingCallback" in callback_classes, (
            "RewardLoggingCallback should be automatically added to RL trainer"
        )

    def test_rl_training_resume_from_checkpoint(
        self,
        isolate_caches_and_outputs: None,
        real_datamodule_rl_config: pyine.organisms.datamodules.shortcuts.ShortcutBiasDataModuleConfig,
        tmp_path: pathlib.Path,
    ) -> None:
        """Test that RL training can resume from a checkpoint and restore reward state."""
        output_dir = tmp_path / "checkpoints"
        output_dir.mkdir(parents=True, exist_ok=True)
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        use_fp16 = torch.cuda.is_available() and not use_bf16
        reward_manager_config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="hard_match",
                    type="code_exec/hard_match",
                    weight=1.0,
                    enabled=True,
                    require_parsed=True,
                    params={
                        "reward_if_match": 1.0,
                        "reward_if_no_match": 0.0,
                        "strip_whitespace": True,
                    },
                ),
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(
                mode="tags",
                enabled_fields="final_only",
                final_tag="final",
                fallback_policy="entire_output",
                multi_tag_policy="last",
                strict=False,
            ),
            verbosity_scaling=pyine.organisms.models.rewards.core.configs.VerbosityScalingConfig(
                enabled=True,
                mode="relative",
                temperature=1.0,
                min_factor=0.1,
                max_factor=1.0,
                emit_metrics=False,  # disabled for resume test to reduce noise
                skip_negative_rewards=True,
            ),
            logging=pyine.organisms.models.rewards.core.configs.LoggingConfig(
                enabled=False,
            ),
        )
        # phase 1: train for 3 steps, save checkpoint at step 3
        grpo_config_phase1 = trl.GRPOConfig(
            output_dir=str(output_dir),
            do_train=True,
            do_eval=False,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=3,
            save_strategy="steps",
            save_steps=3,
            logging_steps=1,
            report_to=[],
            dataloader_num_workers=0,
            seed=42,
            fp16=use_fp16,
            bf16=use_bf16,
            remove_unused_columns=False,
            num_generations=2,
            max_completion_length=16,
            use_vllm=False,
        )
        config_phase1 = pyine.apps.trainers.hf_rl_trainer_configs.RLTrainerAppMainConfig(
            base_model="HuggingFaceTB/SmolLM-360M-Instruct",
            quantization_mode="none",
            auto_model_config={"low_cpu_mem_usage": True},
            grpo_config=grpo_config_phase1,
            datamodule_config=real_datamodule_rl_config,
            reward_manager_config=reward_manager_config,
            use_wandb_logging=False,
            evals_config=pyine.evals.code_exec.configs.CodeExecEvalsConfig(
                eval_batch_size=2,
                category_extraction_config=pyine.evals.utils.SampleCategoryExtractionConfig(
                    enabled_fields=frozenset({pyine.evals.utils.SampleCategoryField.code_type}),
                ),
            ),
        )
        runtime_phase1 = pyine.configs.schemas.RuntimeConfig(
            exp_name="rl-resume-test",
            run_name="phase1",
            app_name="test_rl_resume",
            output_dir=str(tmp_path / "runtime1"),
            seed=42,
        )
        datamodule = pyine.organisms.datamodules.shortcuts.ShortcutBiasDataModule(real_datamodule_rl_config)
        datamodule.prepare_data()
        datamodule.setup()
        trainer_phase1 = pyine.apps.trainers.hf_trainer.rl_train(
            datamodule=datamodule,
            config=config_phase1,
            runtime=runtime_phase1,
            resume_artifacts=None,
            shutdown_manager=None,
        )
        assert trainer_phase1.state.global_step == 3
        checkpoint_dir = output_dir / "checkpoint-3"
        assert checkpoint_dir.exists(), "checkpoint-3 should exist after phase 1"
        # verify reward_state.json was saved
        reward_state_path = checkpoint_dir / "reward_state.json"
        assert reward_state_path.exists(), "reward_state.json should be saved in checkpoint"
        with open(reward_state_path) as f:
            reward_state_phase1 = json.load(f)
        assert "version" in reward_state_phase1
        assert "adapter" in reward_state_phase1
        assert "manager" in reward_state_phase1
        # inject a distinctive marker value into the checkpoint to verify it gets loaded
        # if resume works, phase 2 will load this value and accumulate on top of it
        # if resume fails, phase 2 will start from 0 and never reach this value
        marker_value = 1_000_000
        reward_state_phase1["adapter"]["total_count"] = marker_value
        reward_state_path.write_text(json.dumps(reward_state_phase1, indent=2))
        # add a marker file to checkpoint-3 to verify it doesn't get overwritten
        # if phase 2 trains from scratch, it would recreate checkpoint-3 and lose this marker
        marker_file = checkpoint_dir / ".phase1_marker"
        marker_file.write_text("created by phase 1")

        # phase 2: resume from checkpoint and train for 3 more steps (total 6)
        grpo_config_phase2 = trl.GRPOConfig(
            output_dir=str(output_dir),
            do_train=True,
            do_eval=False,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=6,
            save_strategy="steps",
            save_steps=3,
            logging_steps=1,
            report_to=[],
            dataloader_num_workers=0,
            seed=42,
            fp16=use_fp16,
            bf16=use_bf16,
            remove_unused_columns=False,
            num_generations=2,
            max_completion_length=16,
            use_vllm=False,
        )
        config_phase2 = pyine.apps.trainers.hf_rl_trainer_configs.RLTrainerAppMainConfig(
            base_model="HuggingFaceTB/SmolLM-360M-Instruct",
            quantization_mode="none",
            auto_model_config={"low_cpu_mem_usage": True},
            grpo_config=grpo_config_phase2,
            datamodule_config=real_datamodule_rl_config,
            reward_manager_config=reward_manager_config,
            use_wandb_logging=False,
            evals_config=pyine.evals.code_exec.configs.CodeExecEvalsConfig(
                eval_batch_size=2,
                category_extraction_config=pyine.evals.utils.SampleCategoryExtractionConfig(
                    enabled_fields=frozenset({pyine.evals.utils.SampleCategoryField.code_type}),
                ),
            ),
        )
        resume_artifacts = pyine.apps.trainers.common.ResumeArtifacts(
            run_dir=tmp_path / "runtime1",
            checkpoint_path=checkpoint_dir,
            checkpoint_metadata_dict={},
            checkpoint_metadata_path=None,
            previous_config_dict={},
            previous_config_path=None,
            previous_runtime_dict={},
            previous_runtime_path=None,
            previous_metadata_dict={},
            previous_metadata_path=None,
            wandb_resume_kwargs={},
        )
        runtime_phase2 = pyine.configs.schemas.RuntimeConfig(
            exp_name="rl-resume-test",
            run_name="phase2",
            app_name="test_rl_resume",
            output_dir=str(tmp_path / "runtime2"),
            seed=42,
        )
        trainer_phase2 = pyine.apps.trainers.hf_trainer.rl_train(
            datamodule=datamodule,
            config=config_phase2,
            runtime=runtime_phase2,
            resume_artifacts=resume_artifacts,
            shutdown_manager=None,
        )
        assert trainer_phase2.state.global_step == 6, "training should resume and reach step 6"
        checkpoint_6_dir = output_dir / "checkpoint-6"
        assert checkpoint_6_dir.exists(), "checkpoint-6 should exist after phase 2"
        # verify reward_state.json continues to be saved
        assert (checkpoint_6_dir / "reward_state.json").exists(), "reward_state.json should be saved in new checkpoint"
        # verify that reward state was actually loaded by checking for the marker value
        # if resume worked: total_count = marker_value (1M) + phase2_new_samples > 1M
        # if resume failed: total_count = only phase2_new_samples << 1M
        with open(checkpoint_6_dir / "reward_state.json") as f:
            reward_state_phase2 = json.load(f)
        phase2_adapter_total = reward_state_phase2["adapter"]["total_count"]
        assert phase2_adapter_total > marker_value, (
            f"reward adapter total_count should be > {marker_value} (marker injected into checkpoint), "
            f"but got {phase2_adapter_total}; the resume did not load the reward state"
        )
        # verify that checkpoint-3 was NOT recreated (marker file still exists)
        # if phase 2 trained from scratch, it would save at step 3 and overwrite checkpoint-3
        assert marker_file.exists(), (
            "marker file in checkpoint-3 was deleted, meaning checkpoint-3 was recreated; "
            "phase 2 likely trained from scratch instead of resuming"
        )
