"""Probe training app — trains lightweight probe classifiers on frozen LLM activations.

See PROBES_CLAUDE.md for the full implementation plan.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import statistics
import typing
from collections import defaultdict
from pathlib import Path

import torch
import transformers

import pyine.apps.trainers.probe_trainer_configs as probe_trainer_configs
import pyine.configs.schemas
import pyine.evals.common
from pyine.probes.collection import ProbeCollection
from pyine.probes.extraction import ActivationExtractor
from pyine.probes.lmdb_dataset import load_probe_dataset_from_lmdb

if typing.TYPE_CHECKING:
    import datasets
    import numpy as np
    import numpy.typing as npt
    from accelerate import Accelerator

    from pyine.probes.base import BaseProbe, ProbeConfig

    _DL = torch.utils.data.DataLoader[dict[str, torch.Tensor]]
    _PreparedResult = tuple[ProbeCollection, torch.optim.AdamW, _DL, _DL]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _tokenize_split(
    dataset: datasets.Dataset,
    tokenizer: transformers.PreTrainedTokenizerBase,
    max_seq_length: int,
) -> datasets.Dataset:
    """Tokenize a probe dataset split (plain text, no special tokens)."""

    def _tokenize(examples: dict[str, list[str] | list[int]]) -> transformers.BatchEncoding:
        tokenized = tokenizer(
            examples["text"],  # pyright: ignore[reportArgumentType]  # always list[str] at runtime
            max_length=max_seq_length,
            truncation=True,
            padding=False,
            add_special_tokens=False,  # Post-template text — don't add BOS/EOS/chat markers
        )
        tokenized["labels"] = examples["label"]
        return tokenized

    ds = dataset.map(  # pyright: ignore[reportUnknownMemberType]  # datasets stubs
        _tokenize,
        batched=True,
        remove_columns=[c for c in dataset.column_names if c not in ("input_ids", "attention_mask", "labels")],
    )
    ds.set_format("torch")  # pyright: ignore[reportUnknownMemberType]  # datasets stubs
    return ds


def build_dataloader(
    dataset: datasets.Dataset,
    tokenizer: transformers.PreTrainedTokenizerBase,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> torch.utils.data.DataLoader[dict[str, torch.Tensor]]:
    """Build a DataLoader with dynamic padding."""
    collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    return torch.utils.data.DataLoader(
        typing.cast("torch.utils.data.Dataset[dict[str, torch.Tensor]]", dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )


# ---------------------------------------------------------------------------
# Replica helpers
# ---------------------------------------------------------------------------


def _stable_replica_seed(base_seed: int, name: str, replica_idx: int) -> int:
    """Deterministic seed from (base_seed, probe_name, replica_idx).

    Uses blake2b instead of Python's hash(), which is randomized per process
    (PYTHONHASHSEED) and would produce different seeds across DDP ranks.
    """
    payload = f"{base_seed}:{name}:{replica_idx}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**31)


def expand_probe_configs_with_replicas(
    probe_configs: list[ProbeConfig],
    num_replicas: int,
    replica_base_seed: int,
) -> list[ProbeConfig]:
    """Expand probe configs by creating N replicas of each, with unique seeds.

    When num_replicas == 1, returns the original list unchanged (no modification).

    Args:
        probe_configs: Original probe configurations from the user.
        num_replicas: Number of replicas per config.
        replica_base_seed: Base seed for deterministic initialization.

    Returns:
        Expanded list of ProbeConfigs with replica metadata set.
    """
    if num_replicas <= 1:
        return probe_configs

    expanded: list[ProbeConfig] = []
    for pc in probe_configs:
        for r in range(num_replicas):
            seed = _stable_replica_seed(replica_base_seed, pc.name, r)
            expanded.append(
                pc.model_copy(
                    update={
                        "name": f"{pc.name}_r{r}",
                        "replica_idx": r,
                        "replica_seed": seed,
                        "base_name": pc.name,
                    }
                )
            )
    return expanded


def aggregate_replica_metrics(
    per_probe_values: dict[str, float],
    probe_configs_by_name: dict[str, ProbeConfig],
) -> dict[str, dict[str, float]]:
    """Group metric values by base_name and compute mean/std/min/max.

    Uses sample standard deviation (n-1 denominator) via statistics.stdev().
    NaN values are excluded from aggregation. If all values for a base_name
    are NaN, the result for that base_name has NaN for all fields.

    Args:
        per_probe_values: {probe_name: metric_value} for all probes (including replicas).
        probe_configs_by_name: {probe_name: ProbeConfig} with replica metadata.

    Returns:
        {base_name: {"mean": float, "std": float, "min": float, "max": float}}
        For non-replicated probes, base_name == probe_name and std is 0.0.
    """
    groups: dict[str, list[float]] = defaultdict(list)
    for name, value in per_probe_values.items():
        pc = probe_configs_by_name[name]
        base = pc.base_name if pc.base_name is not None else pc.name
        if not math.isnan(value):
            groups[base].append(value)
        else:
            # Ensure the base_name key exists even if all values are NaN
            groups.setdefault(base, [])

    result: dict[str, dict[str, float]] = {}
    for base_name, values in groups.items():
        if len(values) == 0:
            result[base_name] = {
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
            }
        else:
            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            result[base_name] = {
                "mean": mean,
                "std": std,
                "min": min(values),
                "max": max(values),
            }
    return result


def build_train_replica_table(
    per_probe_losses: dict[str, float],
    expanded_configs_by_name: dict[str, ProbeConfig],
    global_step: int,
    epoch: int,
) -> typing.Any:
    """Build a W&B Table with raw per-replica training losses.

    Returns:
        wandb.Table with columns: probe_name, base_name, architecture, layer,
        replica_idx, step, epoch, train_loss. One row per replica probe.
    """
    import wandb

    table = wandb.Table(
        columns=[
            "probe_name",
            "base_name",
            "architecture",
            "layer",
            "replica_idx",
            "step",
            "epoch",
            "train_loss",
        ]
    )
    for name, loss_val in per_probe_losses.items():
        pc = expanded_configs_by_name[name]
        table.add_data(  # pyright: ignore[reportUnknownMemberType]  # wandb stubs
            name,
            pc.base_name or pc.name,
            pc.architecture,
            pc.layer,
            pc.replica_idx if pc.replica_idx is not None else 0,
            global_step,
            epoch,
            loss_val,
        )
    return table


def build_valid_replica_table(
    per_probe_metrics: dict[str, dict[str, float]],
    expanded_configs_by_name: dict[str, ProbeConfig],
    global_step: int,
) -> typing.Any:
    """Build a W&B Table with raw per-replica validation metrics.

    Returns:
        wandb.Table with columns: probe_name, base_name, architecture, layer,
        replica_idx, step, valid_loss, auroc. One row per replica probe.
    """
    import wandb

    table = wandb.Table(
        columns=[
            "probe_name",
            "base_name",
            "architecture",
            "layer",
            "replica_idx",
            "step",
            "valid_loss",
            "auroc",
        ]
    )
    for name, m in per_probe_metrics.items():
        pc = expanded_configs_by_name[name]
        table.add_data(  # pyright: ignore[reportUnknownMemberType]  # wandb stubs
            name,
            pc.base_name or pc.name,
            pc.architecture,
            pc.layer,
            pc.replica_idx if pc.replica_idx is not None else 0,
            global_step,
            m["loss"],
            m["auroc"],
        )
    return table


def save_replica_summary(
    output_dir: Path,
    final_metrics: dict[str, dict[str, float]],
    expanded_configs_by_name: dict[str, ProbeConfig],
) -> None:
    """Save aggregated replica summary as JSON."""
    loss_agg = aggregate_replica_metrics(
        {name: m["loss"] for name, m in final_metrics.items()},
        expanded_configs_by_name,
    )
    auroc_agg = aggregate_replica_metrics(
        {name: m["auroc"] for name, m in final_metrics.items()},
        expanded_configs_by_name,
    )

    base_names = {pc.base_name or pc.name for pc in expanded_configs_by_name.values()}
    summary = {
        base_name: {
            "loss": loss_agg.get(base_name, {}),
            "auroc": auroc_agg.get(base_name, {}),
            "num_replicas": len(
                [pc for pc in expanded_configs_by_name.values() if (pc.base_name or pc.name) == base_name]
            ),
        }
        for base_name in sorted(base_names)
    }
    (output_dir / "replica_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"Saved replica summary to {output_dir / 'replica_summary.json'}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_probes(
    probe_collection: ProbeCollection,
    model: torch.nn.Module,
    extractor: ActivationExtractor,
    valid_loader: torch.utils.data.DataLoader[dict[str, torch.Tensor]],
    loss_fn: torch.nn.Module,
    global_step: int,
    accelerator: Accelerator,
    runtime: pyine.configs.schemas.RuntimeConfig | None,
    *,
    expanded_configs_by_name: dict[str, ProbeConfig] | None = None,
    log_individual_replicas: bool = False,
) -> dict[str, dict[str, float]]:
    """Run validation and compute loss + AUROC per probe."""
    probe_collection.eval()

    # Handle both DDP-wrapped and raw ProbeCollection
    raw = typing.cast(
        "ProbeCollection",
        getattr(probe_collection, "module", probe_collection),
    )
    probes_dict = raw.probes
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
    gathered_logits: dict[str, torch.Tensor] = {}
    for name in all_logits:
        cat_logits = torch.cat(all_logits[name])
        gathered_logits[name] = typing.cast(
            "torch.Tensor",
            accelerator.gather_for_metrics(cat_logits),  # pyright: ignore[reportUnknownMemberType]  # accelerate stubs
        )
    all_labels_cat = typing.cast(
        "torch.Tensor",
        accelerator.gather_for_metrics(torch.cat(all_labels)),  # pyright: ignore[reportUnknownMemberType]  # accelerate stubs
    )

    # Compute metrics on main process
    metrics: dict[str, dict[str, float]] = {}
    if accelerator.is_main_process:
        for name in probes_dict:
            logits_cpu = gathered_logits[name].float().cpu()
            labels_cpu = all_labels_cat.long().cpu()

            val_loss = loss_fn(logits_cpu, labels_cpu.float()).item()
            probs = typing.cast(
                "npt.NDArray[np.floating[typing.Any]]",
                torch.sigmoid(logits_cpu).numpy(),  # pyright: ignore[reportUnknownMemberType]  # torch stubs
            )
            labels_np = typing.cast(
                "npt.NDArray[np.integer[typing.Any]]",
                labels_cpu.numpy(),  # pyright: ignore[reportUnknownMemberType]  # torch stubs
            )

            unique_labels: set[int] = {int(v) for v in labels_np.tolist()}
            if len(unique_labels) < 2:
                logger.warning(f"Skipping AUROC for {name}: only labels {unique_labels} present")
                auroc = float("nan")
            else:
                import sklearn.metrics

                auroc_score: float = float(
                    sklearn.metrics.roc_auc_score(labels_np, probs)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # sklearn stubs
                )
                auroc = auroc_score

            metrics[name] = {"loss": val_loss, "auroc": auroc}

        # --- Logging ---
        has_replicas = expanded_configs_by_name is not None and any(
            pc.base_name is not None for pc in expanded_configs_by_name.values()
        )

        if has_replicas and expanded_configs_by_name is not None:
            loss_agg = aggregate_replica_metrics(
                {name: m["loss"] for name, m in metrics.items()},
                expanded_configs_by_name,
            )
            auroc_agg = aggregate_replica_metrics(
                {name: m["auroc"] for name, m in metrics.items()},
                expanded_configs_by_name,
            )

            if runtime and runtime.wandb_run:
                log_dict: dict[str, typing.Any] = {}
                for base_name, stats in loss_agg.items():
                    log_dict[f"valid/{base_name}/loss/mean"] = stats["mean"]
                    log_dict[f"valid/{base_name}/loss/std"] = stats["std"]
                    log_dict[f"valid/{base_name}/loss/min"] = stats["min"]
                    log_dict[f"valid/{base_name}/loss/max"] = stats["max"]
                for base_name, stats in auroc_agg.items():
                    log_dict[f"valid/{base_name}/auroc/mean"] = stats["mean"]
                    log_dict[f"valid/{base_name}/auroc/std"] = stats["std"]
                    log_dict[f"valid/{base_name}/auroc/min"] = stats["min"]
                    log_dict[f"valid/{base_name}/auroc/max"] = stats["max"]

                if log_individual_replicas:
                    for name, m in metrics.items():
                        log_dict[f"valid/{name}/loss"] = m["loss"]
                        log_dict[f"valid/{name}/auroc"] = m["auroc"]

                valid_table = build_valid_replica_table(
                    metrics,
                    expanded_configs_by_name,
                    global_step,
                )
                log_dict["valid/replica_details"] = valid_table
                runtime.wandb_run.log(log_dict, step=global_step)

            logger.info(
                f"[step {global_step}] validation (aggregated): "
                + ", ".join(f"{base}: loss={s['mean']:.4f}+-{s['std']:.4f}" for base, s in loss_agg.items())
                + " | "
                + ", ".join(f"{base}: auroc={s['mean']:.4f}+-{s['std']:.4f}" for base, s in auroc_agg.items())
            )
        else:
            if runtime and runtime.wandb_run:
                for name, m in metrics.items():
                    runtime.wandb_run.log(
                        {
                            f"valid/{name}/loss": m["loss"],
                            f"valid/{name}/auroc": m["auroc"],
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
    accelerator: Accelerator,
) -> Path | None:
    """Save probe weights + configs. Only on main process."""
    if not accelerator.is_main_process:
        return None

    raw_collection = typing.cast(
        "ProbeCollection",
        accelerator.unwrap_model(probe_collection),  # pyright: ignore[reportUnknownMemberType]  # accelerate stubs
    )
    if runtime is not None:
        output_dir = Path(runtime.output_dir) / "probes"
    else:
        output_dir = Path("probes_output")

    for name, module in raw_collection.probes.items():
        probe = typing.cast("BaseProbe", module)
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
    hidden_dim: int = model.config.hidden_size

    # Expand probe configs with replicas if num_replicas > 1
    expanded_probe_configs = expand_probe_configs_with_replicas(
        config.probe_configs,
        config.num_replicas,
        config.replica_base_seed,
    )
    has_replicas = config.num_replicas > 1
    expanded_configs_by_name: dict[str, ProbeConfig] = {pc.name: pc for pc in expanded_probe_configs}

    if has_replicas:
        logger.info(
            f"Replica mode: {len(config.probe_configs)} base configs x "
            f"{config.num_replicas} replicas = {len(expanded_probe_configs)} probes"
        )

    logger.info(f"Building ProbeCollection with hidden_dim={hidden_dim}, {len(expanded_probe_configs)} probes")
    probe_collection = ProbeCollection(expanded_probe_configs, hidden_dim)
    probe_collection = probe_collection.to(dtype=config.target_dtype)
    optimizer = torch.optim.AdamW(probe_collection.get_parameter_groups())

    # --- 3. Prepare datasets ---
    logger.info(f"Loading probe dataset from LMDB: {config.lmdb_path}")
    raw_ds = load_probe_dataset_from_lmdb(
        lmdb_path=config.lmdb_path,
        label_metric_key=config.label_metric_key,
        train_key_prefix=config.train_key_prefix,
        valid_key_prefix=config.valid_key_prefix,
        selection_strategy=config.selection_strategy,
        recompute_labels=config.recompute_labels,
        max_samples_per_split=config.max_samples_per_split,
        skip_malformed_records=config.skip_malformed_records,
    )
    train_ds = _tokenize_split(raw_ds["train"], tokenizer, config.max_seq_length)
    valid_ds = _tokenize_split(raw_ds["valid"], tokenizer, config.max_seq_length)
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
    # accelerate.prepare() returns the same types wrapped for distributed training
    (
        probe_collection,
        optimizer,
        train_loader,
        valid_loader,
    ) = typing.cast(
        "_PreparedResult",
        accelerator.prepare(  # pyright: ignore[reportUnknownMemberType]  # accelerate stubs
            probe_collection,
            optimizer,
            train_loader,
            valid_loader,
        ),
    )

    # --- 5. Register activation hooks ---
    target_layers = sorted({pc.layer for pc in expanded_probe_configs})
    logger.info(f"Registering activation hooks on layers: {target_layers}")
    extractor = ActivationExtractor(model, target_layers, activation_dtype=config.target_dtype)

    # --- 6. Training loop ---
    loss_fn = torch.nn.BCEWithLogitsLoss()
    global_step = 0
    final_metrics: dict[str, dict[str, float]] = {}

    logger.info(f"Starting training: {config.num_epochs} epochs, {len(train_loader)} batches/epoch")

    for epoch in range(config.num_epochs):
        probe_collection.train()

        for _step, batch in enumerate(train_loader):
            with accelerator.accumulate(probe_collection):  # pyright: ignore[reportUnknownMemberType]  # accelerate stubs
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

                accelerator.backward(total_loss)  # pyright: ignore[reportUnknownMemberType]  # accelerate stubs
                optimizer.step()  # pyright: ignore[reportUnknownMemberType]  # torch stubs
                optimizer.zero_grad()

            # Logging + eval gated on actual optimizer steps
            if accelerator.sync_gradients:  # pyright: ignore[reportUnknownMemberType]  # accelerate stubs
                global_step += 1
                if global_step % config.logging_steps == 0:
                    if accelerator.is_main_process:
                        per_probe_loss_values = {name: loss.item() for name, loss in per_probe_losses.items()}

                        if has_replicas:
                            agg = aggregate_replica_metrics(per_probe_loss_values, expanded_configs_by_name)
                            log_msg = f"[epoch {epoch + 1}/{config.num_epochs}, step {global_step}] "
                            log_msg += ", ".join(
                                f"{base}: {s['mean']:.4f} (std={s['std']:.4f})" for base, s in agg.items()
                            )
                            logger.info(log_msg)

                            if runtime and runtime.wandb_run:
                                train_log_dict: dict[str, typing.Any] = {}
                                for base_name, stats in agg.items():
                                    train_log_dict[f"train/{base_name}/loss/mean"] = stats["mean"]
                                    train_log_dict[f"train/{base_name}/loss/std"] = stats["std"]
                                train_log_dict["train/global_step"] = global_step
                                train_log_dict["train/epoch"] = epoch

                                if config.log_individual_replicas:
                                    for name, loss_val in per_probe_loss_values.items():
                                        train_log_dict[f"train/{name}/loss"] = loss_val

                                train_table = build_train_replica_table(
                                    per_probe_loss_values,
                                    expanded_configs_by_name,
                                    global_step,
                                    epoch,
                                )
                                train_log_dict["train/replica_details"] = train_table
                                runtime.wandb_run.log(train_log_dict, step=global_step)
                        else:
                            log_msg = f"[epoch {epoch + 1}/{config.num_epochs}, step {global_step}] "
                            log_msg += ", ".join(
                                f"{name}: {loss_val:.4f}" for name, loss_val in per_probe_loss_values.items()
                            )
                            logger.info(log_msg)

                            if runtime and runtime.wandb_run:
                                train_log_dict = {
                                    f"train/{name}/loss": loss_val for name, loss_val in per_probe_loss_values.items()
                                }
                                train_log_dict["train/global_step"] = global_step
                                train_log_dict["train/epoch"] = epoch
                                runtime.wandb_run.log(train_log_dict, step=global_step)

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
                        expanded_configs_by_name=expanded_configs_by_name if has_replicas else None,
                        log_individual_replicas=config.log_individual_replicas,
                    )
                    probe_collection.train()

        # End-of-epoch validation
        final_metrics = validate_probes(
            probe_collection,
            model,
            extractor,
            valid_loader,
            loss_fn,
            global_step,
            accelerator,
            runtime,
            expanded_configs_by_name=expanded_configs_by_name if has_replicas else None,
            log_individual_replicas=config.log_individual_replicas,
        )

    # --- 7. Save probes ---
    if config.save_probes:
        output_dir = save_probe_checkpoints(probe_collection, config, runtime, accelerator)
        if has_replicas and output_dir is not None and final_metrics:
            save_replica_summary(output_dir, final_metrics, expanded_configs_by_name)

    # --- 8. Cleanup ---
    extractor.remove_hooks()

    logger.info("Probe training complete.")
    return typing.cast(
        "ProbeCollection",
        accelerator.unwrap_model(probe_collection),  # pyright: ignore[reportUnknownMemberType]  # accelerate stubs
    )


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
