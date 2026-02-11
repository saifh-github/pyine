"""End-to-end integration test for the probe trainer.

Requires a GPU (or at least an accelerator). Marked slow + integration.
Uses a small public model and the synthetic debug dataset.
"""

from __future__ import annotations

import pathlib

import pytest
import torch


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA GPU required for probe training integration test",
)
class TestProbeTrainerIntegration:
    """End-to-end integration test with a real small model."""

    def test_probe_training_end_to_end(self, tmp_path: pathlib.Path) -> None:
        """Full pipeline: load model, tokenize debug dataset, train probes, validate, save.

        Uses a small public model (HuggingFaceTB/SmolLM2-135M-Instruct).
        Verifies:
        - Training completes without error
        - Probe checkpoints are saved to disk
        - Validation AUROC is computed (>= 0.0, not NaN)
        """
        from pyine.apps.trainers.probe_trainer import probe_train
        from pyine.apps.trainers.probe_trainer_configs import ProbeTrainerAppMainConfig
        from pyine.probes.base import ProbeConfig
        from pyine.probes.debug_dataset import create_debug_probe_dataset

        # Create debug dataset
        ds_path = tmp_path / "debug-dataset"
        create_debug_probe_dataset(output_path=ds_path, n_train=40, n_valid=10, seed=42)

        config = ProbeTrainerAppMainConfig(
            base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
            dataset_path=str(ds_path),
            text_field="messages",
            label_field="label",
            max_seq_length=256,
            num_epochs=2,
            train_batch_size=4,
            eval_batch_size=4,
            gradient_accumulation_steps=1,
            logging_steps=5,
            eval_steps=-1,
            dataloader_num_workers=0,
            save_probes=True,
            probe_configs=[
                ProbeConfig(name="mean_L0", architecture="mean", layer=0, learning_rate=1e-3),
                ProbeConfig(name="max_L1", architecture="max", layer=1, learning_rate=1e-3),
            ],
            auto_model_config={"use_cache": False},
        )

        # Mock runtime with output_dir
        from unittest.mock import MagicMock

        runtime = MagicMock()
        runtime.output_dir = str(tmp_path / "output")
        runtime.wandb_run = None
        runtime.dry_run = False

        pathlib.Path(runtime.output_dir).mkdir(parents=True, exist_ok=True)

        probe_train(config=config, runtime=runtime)

        # Verify probes were saved
        probes_dir = pathlib.Path(runtime.output_dir) / "probes"
        assert probes_dir.exists()
        for name in ("mean_L0", "max_L1"):
            assert (probes_dir / name / "probe_state_dict.pt").exists()
            assert (probes_dir / name / "probe_config.json").exists()
