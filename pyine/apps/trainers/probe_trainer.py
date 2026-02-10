"""Probe training app — trains lightweight probe classifiers on frozen LLM activations.

See PROBES_CLAUDE.md for the full implementation plan.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import datasets
import torch
import transformers

import pyine.apps.trainers.probe_trainer_configs as probe_trainer_configs
import pyine.configs.schemas
import pyine.evals.common
from pyine.probes.collection import ProbeCollection
from pyine.probes.extraction import ActivationExtractor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def validate_probe_dataset(
    dataset: datasets.Dataset,
    text_field: str,
    label_field: str,
) -> None:
    """Fail fast on malformed datasets."""
    if text_field not in dataset.column_names:
        raise ValueError(f"text_field '{text_field}' not found. Columns: {dataset.column_names}")
    if label_field not in dataset.column_names:
        raise ValueError(f"label_field '{label_field}' not found. Columns: {dataset.column_names}")
    unique_labels = set(dataset.unique(label_field))
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"Expected binary labels {{0, 1}}, got {unique_labels}")
    if len(unique_labels) < 2:
        logger.warning(f"Split has only label(s) {unique_labels} — probe training may be degenerate")


def tokenize_for_probes(
    examples: dict,
    tokenizer: transformers.PreTrainedTokenizer,
    max_seq_length: int,
    text_field: str,
    label_field: str,
) -> dict:
    """Tokenize text for probe training."""
    if text_field == "messages":
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError(
                "text_field='messages' requires a tokenizer with a chat template. "
                "Either use a model with a built-in template, or set text_field to "
                "a preformatted string column."
            )
        texts = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in examples["messages"]
        ]
    else:
        texts = examples[text_field]

    tokenized = tokenizer(
        texts,
        max_length=max_seq_length,
        truncation=True,
        padding=False,  # dynamic padding in collator
    )
    tokenized["labels"] = examples[label_field]
    return tokenized


def load_and_tokenize(
    dataset_path: str,
    split: str,
    tokenizer: transformers.PreTrainedTokenizer,
    config: probe_trainer_configs.ProbeTrainerAppMainConfig,
) -> datasets.Dataset:
    """Load a dataset split and tokenize it."""
    ds = datasets.load_from_disk(dataset_path)
    if isinstance(ds, datasets.DatasetDict):
        if split not in ds:
            raise ValueError(f"Split '{split}' not found. Available: {list(ds.keys())}")
        ds_split = ds[split]
    else:
        ds_split = ds

    validate_probe_dataset(ds_split, config.text_field, config.label_field)
    ds_split = ds_split.map(
        tokenize_for_probes,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_seq_length": config.max_seq_length,
            "text_field": config.text_field,
            "label_field": config.label_field,
        },
        remove_columns=[c for c in ds_split.column_names if c not in ("input_ids", "attention_mask", "labels")],
    )
    ds_split.set_format("torch")
    return ds_split


def build_dataloader(
    dataset: datasets.Dataset,
    tokenizer: transformers.PreTrainedTokenizer,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> torch.utils.data.DataLoader:
    """Build a DataLoader with dynamic padding."""
    collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_probes(
    probe_collection: ProbeCollection,
    model: torch.nn.Module,
    extractor: ActivationExtractor,
    valid_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    global_step: int,
    accelerator: Accelerator,  # noqa: F821
    runtime: pyine.configs.schemas.RuntimeConfig | None,
) -> dict[str, dict[str, float]]:
    """Run validation and compute loss + AUROC per probe."""
    probe_collection.eval()

    # Handle both DDP-wrapped and raw ProbeCollection
    probes_dict = probe_collection.module.probes if hasattr(probe_collection, "module") else probe_collection.probes
    all_logits: dict[str, list[torch.Tensor]] = {name: [] for name in probes_dict}
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in valid_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            model(input_ids=input_ids, attention_mask=attention_mask)
            activations = extractor.get_activations()

            probe_logits = probe_collection(activations, attention_mask)
            for name, logits in probe_logits.items():
                all_logits[name].append(logits.squeeze(-1))
            all_labels.append(labels)

    # Gather across GPUs
    for name in all_logits:
        all_logits[name] = torch.cat(all_logits[name])
        all_logits[name] = accelerator.gather_for_metrics(all_logits[name])
    all_labels_cat = accelerator.gather_for_metrics(torch.cat(all_labels))

    # Compute metrics on main process
    metrics: dict[str, dict[str, float]] = {}
    if accelerator.is_main_process:
        for name in probes_dict:
            logits_cpu = all_logits[name].float().cpu()
            labels_cpu = all_labels_cat.long().cpu()

            val_loss = loss_fn(logits_cpu, labels_cpu.float()).item()
            probs = torch.sigmoid(logits_cpu).numpy()
            labels_np = labels_cpu.numpy()

            unique_labels = set(labels_np.tolist())
            if len(unique_labels) < 2:
                logger.warning(f"Skipping AUROC for {name}: only labels {unique_labels} present")
                auroc = float("nan")
            else:
                import sklearn.metrics

                auroc = float(sklearn.metrics.roc_auc_score(labels_np, probs))

            metrics[name] = {"loss": val_loss, "auroc": auroc}

            if runtime and runtime.wandb_run:
                runtime.wandb_run.log(
                    {
                        f"valid/{name}/loss": val_loss,
                        f"valid/{name}/auroc": auroc,
                    },
                    step=global_step,
                )

        logger.info(
            f"[step {global_step}] validation: "
            + ", ".join(f"{n}: loss={m['loss']:.4f} auroc={m['auroc']:.4f}" for n, m in metrics.items())
        )

    return metrics


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------


def save_probe_checkpoints(
    probe_collection: ProbeCollection,
    config: probe_trainer_configs.ProbeTrainerAppMainConfig,
    runtime: pyine.configs.schemas.RuntimeConfig | None,
    accelerator: Accelerator,  # noqa: F821
) -> Path | None:
    """Save probe weights + configs. Only on main process."""
    if not accelerator.is_main_process:
        return None

    raw_collection = accelerator.unwrap_model(probe_collection)
    if runtime is not None:
        output_dir = Path(runtime.output_dir) / "probes"
    else:
        output_dir = Path("probes_output")

    for name, probe in raw_collection.probes.items():
        probe_dir = output_dir / name
        probe_dir.mkdir(parents=True, exist_ok=True)
        torch.save(probe.state_dict(), probe_dir / "probe_state_dict.pt")
        (probe_dir / "probe_config.json").write_text(probe.config.model_dump_json(indent=2))

    logger.info(f"Saved probe checkpoints to {output_dir}")
    return output_dir


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------


def probe_train(
    config: probe_trainer_configs.ProbeTrainerAppMainConfig,
    runtime: pyine.configs.schemas.RuntimeConfig | None,
) -> ProbeCollection:
    """Core probe training loop."""
    from accelerate import Accelerator

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    # --- 1. Load frozen LLM ---
    logger.info("Loading frozen LLM...")
    checkpoint_path = config.llm_checkpoint_path
    model = config.get_model(
        checkpoint_path=Path(checkpoint_path) if checkpoint_path else None,
    )
    model.eval()
    model.requires_grad_(False)
    tokenizer = config.get_tokenizer(
        checkpoint_path=Path(checkpoint_path) if checkpoint_path else None,
    )

    # --- 2. Build ProbeCollection + optimizer ---
    hidden_dim = model.config.hidden_size
    logger.info(f"Building ProbeCollection with hidden_dim={hidden_dim}, {len(config.probe_configs)} probes")
    probe_collection = ProbeCollection(config.probe_configs, hidden_dim)
    probe_collection = probe_collection.to(dtype=config.target_dtype)
    optimizer = torch.optim.AdamW(probe_collection.get_parameter_groups())

    # --- 3. Prepare datasets ---
    logger.info(f"Loading dataset from {config.dataset_path}")
    train_ds = load_and_tokenize(config.dataset_path, "train", tokenizer, config)
    valid_ds = load_and_tokenize(config.dataset_path, "valid", tokenizer, config)
    train_loader = build_dataloader(
        train_ds,
        tokenizer,
        config.train_batch_size,
        config.dataloader_num_workers,
        shuffle=True,
    )
    valid_loader = build_dataloader(
        valid_ds,
        tokenizer,
        config.eval_batch_size,
        config.dataloader_num_workers,
        shuffle=False,
    )

    # --- 4. Prepare with accelerate ---
    probe_collection, optimizer, train_loader, valid_loader = accelerator.prepare(
        probe_collection,
        optimizer,
        train_loader,
        valid_loader,
    )

    # --- 5. Register activation hooks ---
    target_layers = sorted({pc.layer for pc in config.probe_configs})
    logger.info(f"Registering activation hooks on layers: {target_layers}")
    extractor = ActivationExtractor(model, target_layers, activation_dtype=config.target_dtype)

    # --- 6. Training loop ---
    loss_fn = torch.nn.BCEWithLogitsLoss()
    global_step = 0

    logger.info(f"Starting training: {config.num_epochs} epochs, {len(train_loader)} batches/epoch")

    for epoch in range(config.num_epochs):
        probe_collection.train()

        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(probe_collection):
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]

                # Single LLM forward pass (no grad)
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                activations = extractor.get_activations()

                # Forward through all probes
                probe_logits = probe_collection(activations, attention_mask)

                # Compute per-probe losses and sum for backward
                per_probe_losses: dict[str, torch.Tensor] = {}
                total_loss = torch.tensor(0.0, device=accelerator.device)
                for name, logits in probe_logits.items():
                    loss = loss_fn(logits.squeeze(-1), labels.float())
                    per_probe_losses[name] = loss
                    total_loss = total_loss + loss

                accelerator.backward(total_loss)
                optimizer.step()
                optimizer.zero_grad()

            # Logging + eval gated on actual optimizer steps
            if accelerator.sync_gradients:
                global_step += 1
                if global_step % config.logging_steps == 0:
                    if accelerator.is_main_process:
                        log_msg = f"[epoch {epoch + 1}/{config.num_epochs}, step {global_step}] "
                        log_msg += ", ".join(f"{n}: {l.item():.4f}" for n, l in per_probe_losses.items())
                        logger.info(log_msg)

                        if runtime and runtime.wandb_run:
                            log_dict = {f"train/{n}/loss": l.item() for n, l in per_probe_losses.items()}
                            log_dict["train/global_step"] = global_step
                            log_dict["train/epoch"] = epoch
                            runtime.wandb_run.log(log_dict, step=global_step)

                # Mid-epoch validation
                if config.eval_steps > 0 and global_step % config.eval_steps == 0:
                    validate_probes(
                        probe_collection,
                        model,
                        extractor,
                        valid_loader,
                        loss_fn,
                        global_step,
                        accelerator,
                        runtime,
                    )
                    probe_collection.train()

        # End-of-epoch validation
        validate_probes(
            probe_collection,
            model,
            extractor,
            valid_loader,
            loss_fn,
            global_step,
            accelerator,
            runtime,
        )

    # --- 7. Save probes ---
    if config.save_probes:
        save_probe_checkpoints(probe_collection, config, runtime, accelerator)

    # --- 8. Cleanup ---
    extractor.remove_hooks()

    logger.info("Probe training complete.")
    return accelerator.unwrap_model(probe_collection)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


async def main(
    config: probe_trainer_configs.ProbeTrainerAppMainConfig,
    runtime: pyine.configs.schemas.RuntimeConfig | None = None,
) -> None:
    """Main entrypoint for probe training."""
    import pyine.utils.reprod

    pyine.utils.reprod.entrypoint_setup(
        runtime_config=runtime,
        use_wandb_logging=config.use_wandb_logging,
        main_config=config,
    )

    if runtime is not None and runtime.dry_run:
        logger.info("dry run mode — skipping probe training")
        return

    probe_train(config=config, runtime=runtime)

    if runtime is not None:
        runtime.finalize()


def async_probe_trainer_main_wrapper(
    config: probe_trainer_configs.ProbeTrainerAppMainConfig,
    runtime: pyine.configs.schemas.RuntimeConfig | None = None,
) -> None:
    """Synchronous wrapper around the async main."""
    asyncio.run(main(config=config, runtime=runtime))


if __name__ == "__main__":
    import sys

    import pyine.apps.trainers.common

    # Filter DeepSpeed's --local_rank to avoid Hydra conflict
    sys.argv = [arg for arg in sys.argv if not arg.startswith("--local_rank")]

    pyine.apps.trainers.common.hydra_main(
        eval_type=pyine.evals.common.EvalType.CODE_EXEC,
        hydra_config_registration_fn=probe_trainer_configs.register_hydra_configs,
        async_main_wrapper=async_probe_trainer_main_wrapper,
    )
