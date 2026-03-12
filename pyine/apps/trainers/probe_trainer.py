"""Probe training app -- trains lightweight probe classifiers on frozen LLM activations.

See pyine/apps/trainers/PROBE_TRAINING_GUIDE.md for the full GUIDE.
"""

from __future__ import annotations

import asyncio
import collections
import hashlib
import json
import logging
import math
import pathlib
import shutil
import statistics
import typing

import datasets  # noqa: TC002
import sklearn.metrics
import torch
import transformers

import pyine.apps.trainers.common
import pyine.apps.trainers.probe_trainer_configs as probe_trainer_configs
import pyine.configs.schemas
import pyine.evals.common
import pyine.evals.correctness._impl as correctness_impl
import pyine.evals.correctness.configs as correctness_configs
import pyine.evals.correctness.datamodule as correctness_datamodule
import pyine.evals.correctness.scorers as correctness_scorers
import pyine.evals.correctness.types as correctness_types
import pyine.guardrails.probes.collection
import pyine.guardrails.probes.extraction
import pyine.utils.distrib  # pyright: ignore[reportUnusedImport]
import pyine.utils.reprod
import pyine.utils.transformers.data

if typing.TYPE_CHECKING:
    import accelerate
    import numpy as np
    import numpy.typing as npt

    import pyine.guardrails.probes.base

    _DL = torch.utils.data.DataLoader[dict[str, torch.Tensor]]
    _PreparedResult = tuple[pyine.guardrails.probes.collection.ProbeCollection, torch.optim.AdamW, _DL, _DL]

logger = logging.getLogger(__name__)


def _tokenize_split(
    dataset: datasets.Dataset,
    tokenizer: transformers.PreTrainedTokenizerBase,
    max_seq_length: int,
    code_type_to_id: dict[str, int],
) -> datasets.Dataset:
    """Tokenize a probe dataset split (plain text, no special tokens).

    Expects a ``text`` column (produced by ``apply_messages_formatting``) and maps ``code_type``
    strings to integer IDs for DDP-safe gathering.
    """

    def _tokenize(examples: dict[str, list[str] | list[int]]) -> transformers.BatchEncoding:
        tokenized = tokenizer(
            examples["text"],  # pyright: ignore[reportArgumentType]  # always list[str] at runtime
            max_length=max_seq_length,
            truncation=True,
            padding=False,
            add_special_tokens=False,  # post-template text -- don't add BOS/EOS/chat markers
        )
        tokenized["labels"] = examples["label"]
        # map code_type string -> integer ID for DDP gathering
        tokenized["code_type_id"] = [code_type_to_id[code_type] for code_type in examples["code_type"]]  # type: ignore[index]
        return tokenized

    ds = dataset.map(  # pyright: ignore[reportUnknownMemberType]  # datasets stubs
        _tokenize,
        batched=True,
        remove_columns=[
            col for col in dataset.column_names if col not in ("input_ids", "attention_mask", "labels", "code_type_id")
        ],
    )
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels", "code_type_id"])  # pyright: ignore[reportUnknownMemberType]  # datasets stubs
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


def _validate_best_probe_checkpoint_preconditions(
    config: probe_trainer_configs.ProbeTrainerAppMainConfig,
    valid_dataset: datasets.Dataset,
) -> None:
    """Fail early when best-probe checkpoint tracking cannot possibly succeed."""
    if config.save_best_probe_checkpoint and len(valid_dataset) == 0:
        raise ValueError(
            "save_best_probe_checkpoint=True requires a non-empty validation split so best metrics can be computed"
        )


def _stable_replica_seed(
    base_seed: int,
    name: str,
    replica_idx: int,
) -> int:
    """Deterministic seed from (base_seed, probe_name, replica_idx).

    Uses blake2b instead of Python's hash(), which is randomized per process
    (PYTHONHASHSEED) and would produce different seeds across DDP ranks.
    """
    payload = f"{base_seed}:{name}:{replica_idx}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**31)


def _get_module_device(
    module: torch.nn.Module,
) -> torch.device:
    """Return the device of a module from its first parameter, failing loudly if absent."""
    first_parameter = next(module.parameters(), None)
    if first_parameter is None:
        raise ValueError("module must have at least one parameter to infer its device")
    return first_parameter.device


def expand_probe_configs_with_replicas(
    probe_configs: list[pyine.guardrails.probes.base.ProbeConfig],
    num_replicas: int,
    replica_base_seed: int,
) -> list[pyine.guardrails.probes.base.ProbeConfig]:
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

    expanded: list[pyine.guardrails.probes.base.ProbeConfig] = []
    for probe_config in probe_configs:
        for replica_idx in range(num_replicas):
            seed = _stable_replica_seed(replica_base_seed, probe_config.name, replica_idx)
            expanded.append(
                probe_config.model_copy(
                    update={
                        "name": f"{probe_config.name}_r{replica_idx}",
                        "replica_idx": replica_idx,
                        "replica_seed": seed,
                        "base_name": probe_config.name,
                    }
                )
            )
    return expanded


def aggregate_replica_metrics(
    per_probe_values: dict[str, float],
    probe_configs_by_name: dict[str, pyine.guardrails.probes.base.ProbeConfig],
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
    groups: dict[str, list[float]] = collections.defaultdict(list)
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
    expanded_configs_by_name: dict[str, pyine.guardrails.probes.base.ProbeConfig],
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
    expanded_configs_by_name: dict[str, pyine.guardrails.probes.base.ProbeConfig],
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
    for name, metric_values in per_probe_metrics.items():
        pc = expanded_configs_by_name[name]
        table.add_data(  # pyright: ignore[reportUnknownMemberType]  # wandb stubs
            name,
            pc.base_name or pc.name,
            pc.architecture,
            pc.layer,
            pc.replica_idx if pc.replica_idx is not None else 0,
            global_step,
            metric_values["loss"],
            metric_values["auroc"],
        )
    return table


def save_replica_summary(
    output_dir: pathlib.Path,
    final_metrics: dict[str, dict[str, float]],
    expanded_configs_by_name: dict[str, pyine.guardrails.probes.base.ProbeConfig],
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
            "num_replicas": sum(
                1
                for probe_config in expanded_configs_by_name.values()
                if (probe_config.base_name or probe_config.name) == base_name
            ),
        }
        for base_name in sorted(base_names)
    }
    (output_dir / "replica_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"Saved replica summary to {output_dir / 'replica_summary.json'}")


def _safe_gather_1d(
    tensor: torch.Tensor,
    accelerator: accelerate.Accelerator,
) -> torch.Tensor:
    """All-gather a 1-D tensor that may have different lengths across DDP ranks.

    Pads shorter tensors to the max length, gathers with uniform sizes, then strips the
    padding from each rank's segment so the result contains only real samples.
    """
    local_size = torch.tensor([tensor.shape[0]], dtype=torch.long, device=tensor.device)
    all_sizes = typing.cast("torch.Tensor", accelerator.gather(local_size))  # pyright: ignore[reportUnknownMemberType]  # accelerate stubs
    max_size = int(all_sizes.max().item())
    if tensor.shape[0] < max_size:
        tensor = torch.cat([tensor, tensor.new_zeros(max_size - tensor.shape[0])])
    gathered = typing.cast("torch.Tensor", accelerator.gather(tensor))  # pyright: ignore[reportUnknownMemberType]  # accelerate stubs
    chunks = typing.cast("tuple[torch.Tensor, ...]", gathered.split(max_size))  # pyright: ignore[reportUnknownMemberType]  # torch stubs
    per_rank_sizes = typing.cast("list[int]", all_sizes.tolist())  # pyright: ignore[reportUnknownMemberType]  # torch stubs
    return torch.cat([chunk[:size] for chunk, size in zip(chunks, per_rank_sizes, strict=True)])


def validate_probes(
    probe_collection: pyine.guardrails.probes.collection.ProbeCollection,
    model: torch.nn.Module,
    extractor: pyine.guardrails.probes.extraction.ActivationExtractor,
    valid_loader: torch.utils.data.DataLoader[dict[str, torch.Tensor]],
    loss_fn: torch.nn.Module,
    global_step: int,
    accelerator: accelerate.Accelerator,
    runtime: pyine.configs.schemas.RuntimeConfig | None,
    *,
    expanded_configs_by_name: dict[str, pyine.guardrails.probes.base.ProbeConfig] | None = None,
    log_individual_replicas: bool = False,
    id_to_code_type: dict[int, str] | None = None,
    log_per_code_type_metrics: bool = False,
) -> dict[str, dict[str, float]]:
    """Run validation and compute loss + AUROC per probe.

    When ``log_per_code_type_metrics=True`` and ``id_to_code_type`` is provided,
    also computes per-code-type loss and AUROC breakdowns.
    """
    probe_collection.eval()

    # Handle both DDP-wrapped and raw pyine.guardrails.probes.collection.ProbeCollection
    raw = typing.cast(
        "pyine.guardrails.probes.collection.ProbeCollection",
        getattr(probe_collection, "module", probe_collection),
    )
    probes_dict = raw.probes
    # Accumulate predictions locally (no collectives inside the loop). Ranks may iterate a
    # different number of batches when the validation set isn't evenly divisible; any collective
    # inside the loop would cause a deadlock in that case.
    all_logits: dict[str, list[torch.Tensor]] = {name: [] for name in probes_dict}
    all_labels: list[torch.Tensor] = []
    all_code_type_ids: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in valid_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            activations = extractor.get_activations()

            probe_logits = probe_collection(activations, attention_mask)
            for name, logits in probe_logits.items():
                all_logits[name].append(logits.squeeze(-1))
            all_labels.append(labels)

            if log_per_code_type_metrics and "code_type_id" in batch:
                all_code_type_ids.append(batch["code_type_id"])

    # Barrier: ensure all ranks have finished iterating before gathering, since ranks may have
    # processed a different number of batches (uneven DistributedSampler padding).
    accelerator.wait_for_everyone()  # pyright: ignore[reportUnknownMemberType]  # accelerate stubs

    # Gather across GPUs using size-safe gather (handles different tensor lengths per rank).
    gathered_logits: dict[str, torch.Tensor] = {}
    for name in all_logits:
        gathered_logits[name] = _safe_gather_1d(torch.cat(all_logits[name]), accelerator)
    all_labels_cat = _safe_gather_1d(torch.cat(all_labels), accelerator)

    gathered_ct_ids: torch.Tensor | None = None
    if log_per_code_type_metrics and all_code_type_ids:
        gathered_ct_ids = _safe_gather_1d(torch.cat(all_code_type_ids), accelerator).cpu()

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
                auroc_score: float = float(
                    sklearn.metrics.roc_auc_score(labels_np, probs)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # sklearn stubs
                )
                auroc = auroc_score

            metrics[name] = {"loss": val_loss, "auroc": auroc}

            # --- Per-code-type metrics ---
            if log_per_code_type_metrics and gathered_ct_ids is not None and id_to_code_type is not None:
                _log_per_code_type_metrics(
                    name=name,
                    logits_cpu=logits_cpu,
                    labels_cpu=labels_cpu,
                    gathered_ct_ids=gathered_ct_ids,
                    id_to_code_type=id_to_code_type,
                    loss_fn=loss_fn,
                    runtime=runtime,
                    global_step=global_step,
                )

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
                + ", ".join(f"{base}: loss={stats['mean']:.4f}+-{stats['std']:.4f}" for base, stats in loss_agg.items())
                + " | "
                + ", ".join(
                    f"{base}: auroc={stats['mean']:.4f}+-{stats['std']:.4f}" for base, stats in auroc_agg.items()
                )
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


@typing.no_type_check
def _log_per_code_type_metrics(
    name: str,
    logits_cpu: torch.Tensor,
    labels_cpu: torch.Tensor,
    gathered_ct_ids: torch.Tensor,
    id_to_code_type: dict[int, str],
    loss_fn: torch.nn.Module,
    runtime: pyine.configs.schemas.RuntimeConfig | None,
    global_step: int,
) -> None:
    """Compute and log per-code-type loss and AUROC for a single probe."""
    unique_ct_ids: list[int] = gathered_ct_ids.unique().tolist()
    for ct_id in unique_ct_ids:
        ct = id_to_code_type[int(ct_id)]
        ct_mask: torch.Tensor = (gathered_ct_ids == ct_id).nonzero(as_tuple=True)[0]
        if len(ct_mask) < 2:
            continue

        ct_logits = logits_cpu[ct_mask]
        ct_labels = labels_cpu[ct_mask]

        ct_loss = loss_fn(ct_logits, ct_labels.float()).item()

        ct_labels_np: npt.NDArray[np.int_] = ct_labels.numpy()
        ct_unique: set[int] = {int(v) for v in ct_labels_np.tolist()}
        if len(ct_unique) < 2:
            ct_auroc = float("nan")
        else:
            import sklearn.metrics

            ct_probs: npt.NDArray[np.floating[typing.Any]] = torch.sigmoid(ct_logits).numpy()
            ct_auroc = float(sklearn.metrics.roc_auc_score(ct_labels_np, ct_probs))

        if runtime and runtime.wandb_run:
            runtime.wandb_run.log(
                {
                    f"valid/{name}/loss/code_type/{ct}": ct_loss,
                    f"valid/{name}/auroc/code_type/{ct}": ct_auroc,
                },
                step=global_step,
            )


def _get_probes_base_dir(
    runtime: pyine.configs.schemas.RuntimeConfig | None,
) -> pathlib.Path:
    """Return the base directory used for probe checkpoint exports."""
    if runtime is not None:
        return pathlib.Path(runtime.output_dir) / "probes"
    return pathlib.Path("probes_output")


def _save_probe_checkpoint_artifacts(
    probe: pyine.guardrails.probes.base.BaseProbe,
    probes_base: pathlib.Path,
    probe_name: str,
    checkpoint_subdir: str,
) -> pathlib.Path:
    """Save one probe to the standard checkpoint layout and return its directory."""
    probe_dir = probes_base / probe_name / checkpoint_subdir
    if probe_dir.exists() and not probe_dir.is_dir():
        raise FileExistsError(f"probe checkpoint path exists and is not a directory: {probe_dir}")
    if probe_dir.is_dir():
        shutil.rmtree(probe_dir)
    probe_dir.mkdir(parents=True, exist_ok=False)
    torch.save(probe.state_dict(), probe_dir / "probe_state_dict.pt")
    (probe_dir / "probe_config.json").write_text(probe.config.model_dump_json(indent=2))
    return probe_dir


def save_probe_checkpoints(
    probe_collection: pyine.guardrails.probes.collection.ProbeCollection,
    config: probe_trainer_configs.ProbeTrainerAppMainConfig,
    runtime: pyine.configs.schemas.RuntimeConfig | None,
    accelerator: accelerate.Accelerator,
    *,
    checkpoint_subdir: str = "final",
) -> pathlib.Path | None:
    """Save probe weights + configs. Only on main process.

    Files are written to ``<probes_base>/<probe_name>/<checkpoint_subdir>/``.

    Returns:
        The probes base directory (``<output_dir>/probes/``), or None on non-main processes.
    """
    if not accelerator.is_main_process:
        return None

    raw_collection = typing.cast(
        "pyine.guardrails.probes.collection.ProbeCollection",
        accelerator.unwrap_model(probe_collection),  # pyright: ignore[reportUnknownMemberType]  # accelerate stubs
    )
    probes_base = _get_probes_base_dir(runtime)

    for name, module in raw_collection.probes.items():
        probe = typing.cast("pyine.guardrails.probes.base.BaseProbe", module)
        _save_probe_checkpoint_artifacts(probe, probes_base, name, checkpoint_subdir)

    logger.info(f"Saved probe checkpoints ({checkpoint_subdir}) to {probes_base}")
    return probes_base


def _is_better_probe_metric(
    current_metrics: dict[str, float],
    best_record: dict[str, typing.Any] | None,
    metric_name: typing.Literal["auroc", "loss"],
) -> bool:
    """Return True when the current validation metrics improve on the best-so-far record."""
    if best_record is None:
        return True
    current_loss = current_metrics["loss"]
    best_loss = typing.cast("float", best_record["loss"])
    current_auroc = current_metrics["auroc"]
    best_auroc = typing.cast("float", best_record["auroc"])
    if metric_name == "loss":
        if current_loss < best_loss:
            return True
        if current_loss > best_loss:
            return False
        if math.isnan(current_auroc):
            return False
        if math.isnan(best_auroc):
            return True
        return current_auroc > best_auroc
    if not math.isnan(current_auroc):
        if math.isnan(best_auroc) or current_auroc > best_auroc:
            return True
        if current_auroc < best_auroc:
            return False
    elif not math.isnan(best_auroc):
        return False
    return current_loss < best_loss


def save_best_probe_summary(
    probes_base: pathlib.Path,
    best_probe_records: dict[str, dict[str, typing.Any]],
    config: probe_trainer_configs.ProbeTrainerAppMainConfig,
) -> pathlib.Path:
    """Write a manifest describing the best exported checkpoint for each probe."""
    summary_path = probes_base / "best_summary.json"
    summary_payload = {
        "checkpoint_name": config.best_probe_checkpoint_name,
        "metric_name": config.best_probe_metric,
        "probes": best_probe_records,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True))
    return summary_path


def update_best_probe_checkpoints(
    probe_collection: pyine.guardrails.probes.collection.ProbeCollection,
    config: probe_trainer_configs.ProbeTrainerAppMainConfig,
    runtime: pyine.configs.schemas.RuntimeConfig | None,
    accelerator: accelerate.Accelerator,
    current_metrics: dict[str, dict[str, float]],
    epoch: int,
    global_step: int,
    best_probe_records: dict[str, dict[str, typing.Any]],
) -> pathlib.Path | None:
    """Update the per-probe best checkpoint export using the latest validation metrics."""
    if not config.save_best_probe_checkpoint or not accelerator.is_main_process or not current_metrics:
        return None
    raw_collection = typing.cast(
        "pyine.guardrails.probes.collection.ProbeCollection",
        accelerator.unwrap_model(probe_collection),  # pyright: ignore[reportUnknownMemberType]  # accelerate stubs
    )
    probes_base = _get_probes_base_dir(runtime)
    improved_probe_names: list[str] = []
    for name, metric_values in current_metrics.items():
        best_record = best_probe_records.get(name)
        if not _is_better_probe_metric(metric_values, best_record, config.best_probe_metric):
            continue
        probe = typing.cast("pyine.guardrails.probes.base.BaseProbe", raw_collection.probes[name])
        _save_probe_checkpoint_artifacts(probe, probes_base, name, config.best_probe_checkpoint_name)
        best_probe_records[name] = {
            "checkpoint_name": config.best_probe_checkpoint_name,
            "metric_name": config.best_probe_metric,
            "metric_value": metric_values[config.best_probe_metric],
            "loss": metric_values["loss"],
            "auroc": metric_values["auroc"],
            "epoch": epoch,
            "global_step": global_step,
        }
        improved_probe_names.append(name)
        logger.info(
            f"updated best checkpoint for probe {name}: "
            f"checkpoint={config.best_probe_checkpoint_name}, "
            f"criterion={config.best_probe_metric}, "
            f"metric_value={metric_values[config.best_probe_metric]}, "
            f"loss={metric_values['loss']}, auroc={metric_values['auroc']}, "
            f"epoch={epoch}, global_step={global_step}"
        )
    if improved_probe_names:
        summary_path = save_best_probe_summary(probes_base, best_probe_records, config)
        logger.info(f"saved best-probe summary to {summary_path}; improved={sorted(improved_probe_names)}")
    return probes_base


def enforce_checkpoint_limit(
    probes_dir: pathlib.Path,
    save_total_limit: int | None,
) -> None:
    """Delete oldest step-numbered checkpoint directories exceeding the retention limit.

    Iterates each probe architecture folder in ``probes_dir`` and removes the
    oldest ``step-NNNN`` subdirectories when the count exceeds ``save_total_limit``.
    The ``final/`` subdirectory is never deleted.
    """
    if save_total_limit is None:
        return

    for probe_dir in probes_dir.iterdir():
        if not probe_dir.is_dir():
            continue

        step_dirs = sorted(
            (d for d in probe_dir.iterdir() if d.is_dir() and d.name.startswith("step-")),
            key=lambda d: int(d.name.split("-", 1)[1]),
        )

        while len(step_dirs) > save_total_limit:
            oldest = step_dirs.pop(0)
            logger.info(f"Checkpoint retention: deleting {oldest} (limit={save_total_limit})")
            shutil.rmtree(oldest)


class ProbeTrainResult(typing.NamedTuple):
    """Return value of probe_train() with extra context for downstream evaluation."""

    probe_collection: pyine.guardrails.probes.collection.ProbeCollection
    """The trained probe collection containing all probes across layers and replicas."""
    model: torch.nn.Module
    """The base model that was used for activation extraction during training."""
    tokenizer: transformers.PreTrainedTokenizerBase
    """The tokenizer associated with the base model."""


def probe_train(
    config: probe_trainer_configs.ProbeTrainerAppMainConfig,
    runtime: pyine.configs.schemas.RuntimeConfig | None,
) -> ProbeTrainResult:
    """Core probe training loop."""
    import accelerate

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    # --- 1. Load frozen LLM ---
    logger.info("Loading frozen LLM...")
    checkpoint_path = config.llm_checkpoint_path
    model = config.get_model(
        checkpoint_path=pathlib.Path(checkpoint_path) if checkpoint_path else None,
    )
    model.eval()
    model.requires_grad_(False)
    tokenizer = config.get_tokenizer(
        checkpoint_path=pathlib.Path(checkpoint_path) if checkpoint_path else None,
    )

    # --- 2. Build ProbeCollection + optimizer ---
    hidden_dim: int = model.config.hidden_size

    # expand probe configs with replicas if num_replicas > 1
    expanded_probe_configs = expand_probe_configs_with_replicas(
        config.probe_configs,
        config.num_replicas,
        config.replica_base_seed,
    )
    has_replicas = config.num_replicas > 1
    expanded_configs_by_name: dict[str, pyine.guardrails.probes.base.ProbeConfig] = {
        probe_config.name: probe_config for probe_config in expanded_probe_configs
    }

    if has_replicas:
        logger.info(
            f"Replica mode: {len(config.probe_configs)} base configs x "
            f"{config.num_replicas} replicas = {len(expanded_probe_configs)} probes"
        )

    logger.info(f"Building ProbeCollection with hidden_dim={hidden_dim}, {len(expanded_probe_configs)} probes")
    probe_collection = pyine.guardrails.probes.collection.ProbeCollection(expanded_probe_configs, hidden_dim)
    probe_collection = probe_collection.to(dtype=config.target_dtype)
    optimizer = torch.optim.AdamW(probe_collection.get_parameter_groups())

    # --- 3. Prepare datasets via DataModule ---
    logger.info("Loading probe dataset from datamodule...")
    datamodule = pyine.apps.trainers.common.prepare_datamodule(config, runtime)
    # both ProbeDataModule and CorrectnessDataModule provide get_probe_dataset/code_type_to_id/id_to_code_type
    probe_dataset_owner = typing.cast("typing.Any", datamodule)
    raw_ds = typing.cast("datasets.DatasetDict", probe_dataset_owner.get_probe_dataset(text_field=config.text_field))
    code_type_to_id = typing.cast("dict[str, int]", datamodule.code_type_to_id)  # pyright: ignore[reportUnknownMemberType,reportAttributeAccessIssue]
    id_to_code_type = typing.cast("dict[int, str]", datamodule.id_to_code_type)  # pyright: ignore[reportUnknownMemberType,reportAttributeAccessIssue]

    pyine.apps.trainers.common.log_class_distribution(raw_ds, logger)
    if accelerator.is_main_process:
        for split_name in ["train", "valid"]:
            code_types = typing.cast("list[str]", raw_ds[split_name]["code_type"])  # pyright: ignore[reportIndexIssue]  # datasets stubs
            code_type_counts: dict[str, int] = {}
            for code_type in code_types:
                code_type_counts[code_type] = code_type_counts.get(code_type, 0) + 1
            logger.info(f"  {split_name} code_type distribution: {code_type_counts}")

    # set truncation side before tokenization (left preserves the generated output at end of sequence)
    tokenizer.truncation_side = config.truncation_side
    # format messages -> text (uses chat template if available, else role-tagged concatenation)
    has_chat_template = pyine.utils.transformers.data.tokenizer_has_chat_template(tokenizer)
    input_formatting_mode = "chat_template" if has_chat_template else "role_tagged_text"
    logger.info(
        f"probe inputs: text_field={config.text_field}, truncation_side={tokenizer.truncation_side}, "
        f"input_formatting_mode={input_formatting_mode}, add_special_tokens=False"
    )
    if runtime is not None and runtime.wandb_run is not None:
        runtime.wandb_run.summary["probe/text_field"] = config.text_field  # type: ignore[reportUnknownMemberType]
        runtime.wandb_run.summary["probe/truncation_side"] = tokenizer.truncation_side  # type: ignore[reportUnknownMemberType]
        runtime.wandb_run.summary["probe/input_formatting_mode"] = input_formatting_mode  # type: ignore[reportUnknownMemberType]
        runtime.wandb_run.summary["probe/add_special_tokens"] = False  # type: ignore[reportUnknownMemberType]
    raw_ds = pyine.apps.trainers.common.apply_messages_formatting(raw_ds, tokenizer)
    # _tokenize_split hardcodes add_special_tokens=False, which is correct when text is already
    # chat-template-formatted (special tokens are baked in). If the tokenizer lacks a chat template,
    # apply_messages_formatting falls back to role-tagged text and special tokens would be missing.
    assert has_chat_template, (
        f"probe tokenization requires a chat template (tokenizer={tokenizer.name_or_path}); "  # type: ignore
        "add_special_tokens=False would produce inputs without BOS/EOS tokens"
    )
    train_ds = _tokenize_split(raw_ds["train"], tokenizer, config.max_seq_length, code_type_to_id)
    valid_ds = _tokenize_split(raw_ds["valid"], tokenizer, config.max_seq_length, code_type_to_id)
    _validate_best_probe_checkpoint_preconditions(config, valid_ds)
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
    extractor = pyine.guardrails.probes.extraction.ActivationExtractor(
        model, target_layers, activation_dtype=config.target_dtype
    )

    # --- 6. Training loop ---
    # seed RNG for reproducible dataloader ordering / dropout
    pyine.utils.reprod.set_seed(seed=runtime.seed if runtime is not None else None)
    # validation always uses unweighted loss so metrics are comparable across runs
    valid_loss_fn = torch.nn.BCEWithLogitsLoss()
    if config.class_weight_mode == "balanced":
        # optionally compute pos_weight for class-balanced BCE loss
        train_labels = typing.cast("list[int]", raw_ds["train"]["label"])
        pw = pyine.apps.trainers.common.compute_binary_pos_weight(train_labels)
        pos_weight = torch.tensor([pw], dtype=torch.float32, device=accelerator.device)
        logger.info(f"using balanced BCE pos_weight={pos_weight.item():.4f}")
        train_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        train_loss_fn = torch.nn.BCEWithLogitsLoss()
    global_step = 0
    final_metrics: dict[str, dict[str, float]] = {}
    best_probe_records: dict[str, dict[str, typing.Any]] = {}

    logger.info(f"Starting training: {config.num_epochs} epochs, {len(train_loader)} batches/epoch")

    for epoch in range(config.num_epochs):
        probe_collection.train()

        for _step, batch in enumerate(train_loader):
            with accelerator.accumulate(probe_collection):  # pyright: ignore[reportUnknownMemberType]  # accelerate stubs
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]

                # single LLM forward pass (no grad)
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                activations = extractor.get_activations()

                # forward through all probes
                probe_logits = probe_collection(activations, attention_mask)

                # compute per-probe losses and sum for backward
                per_probe_losses: dict[str, torch.Tensor] = {}
                total_loss = torch.tensor(0.0, device=accelerator.device)
                for name, logits in probe_logits.items():
                    loss = train_loss_fn(logits.squeeze(-1), labels.float())
                    per_probe_losses[name] = loss
                    total_loss = total_loss + loss

                accelerator.backward(total_loss)  # pyright: ignore[reportUnknownMemberType]  # accelerate stubs
                if config.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(probe_collection.parameters(), config.max_grad_norm)  # pyright: ignore[reportUnknownMemberType]  # accelerate stubs
                optimizer.step()  # pyright: ignore[reportUnknownMemberType]  # torch stubs
                optimizer.zero_grad()

            # logging + eval gated on actual optimizer steps
            if accelerator.sync_gradients:  # pyright: ignore[reportUnknownMemberType]  # accelerate stubs
                global_step += 1
                if global_step % config.logging_steps == 0:
                    if accelerator.is_main_process:
                        per_probe_loss_values = {name: loss.item() for name, loss in per_probe_losses.items()}

                        if has_replicas:
                            agg = aggregate_replica_metrics(per_probe_loss_values, expanded_configs_by_name)
                            log_msg = f"[epoch {epoch + 1}/{config.num_epochs}, step {global_step}] "
                            log_msg += ", ".join(
                                f"{base}: {stats['mean']:.4f} (std={stats['std']:.4f})" for base, stats in agg.items()
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

                # mid-epoch validation
                if config.eval_steps > 0 and global_step % config.eval_steps == 0:
                    mid_epoch_metrics = validate_probes(
                        probe_collection,
                        model,
                        extractor,
                        valid_loader,
                        valid_loss_fn,
                        global_step,
                        accelerator,
                        runtime,
                        expanded_configs_by_name=expanded_configs_by_name if has_replicas else None,
                        log_individual_replicas=config.log_individual_replicas,
                        id_to_code_type=id_to_code_type if config.log_per_code_type_metrics else None,
                        log_per_code_type_metrics=config.log_per_code_type_metrics,
                    )
                    update_best_probe_checkpoints(
                        probe_collection,
                        config,
                        runtime,
                        accelerator,
                        mid_epoch_metrics,
                        epoch=epoch + 1,
                        global_step=global_step,
                        best_probe_records=best_probe_records,
                    )
                    probe_collection.train()

                # mid-training checkpoint saving
                if config.save_probes and config.save_steps > 0 and global_step % config.save_steps == 0:
                    step_subdir = f"step-{global_step:04d}"
                    probes_base = save_probe_checkpoints(
                        probe_collection,
                        config,
                        runtime,
                        accelerator,
                        checkpoint_subdir=step_subdir,
                    )
                    if accelerator.is_main_process and probes_base is not None:
                        enforce_checkpoint_limit(probes_base, config.save_total_limit)

        # end-of-epoch validation
        final_metrics = validate_probes(
            probe_collection,
            model,
            extractor,
            valid_loader,
            valid_loss_fn,
            global_step,
            accelerator,
            runtime,
            expanded_configs_by_name=expanded_configs_by_name if has_replicas else None,
            log_individual_replicas=config.log_individual_replicas,
            id_to_code_type=id_to_code_type if config.log_per_code_type_metrics else None,
            log_per_code_type_metrics=config.log_per_code_type_metrics,
        )
        update_best_probe_checkpoints(
            probe_collection,
            config,
            runtime,
            accelerator,
            final_metrics,
            epoch=epoch + 1,
            global_step=global_step,
            best_probe_records=best_probe_records,
        )

    # --- 7. Save probes ---
    if config.save_probes:
        probes_base = save_probe_checkpoints(
            probe_collection,
            config,
            runtime,
            accelerator,
            checkpoint_subdir="final",
        )
        if has_replicas and probes_base is not None and final_metrics:
            save_replica_summary(probes_base, final_metrics, expanded_configs_by_name)
        if config.save_best_probe_checkpoint and accelerator.is_main_process:
            missing_best_probes = sorted(set(expanded_configs_by_name) - set(best_probe_records))
            if missing_best_probes:
                raise ValueError(
                    "best probe checkpoints were requested, but no best checkpoint was recorded for: "
                    f"{missing_best_probes}"
                )
            if probes_base is not None:
                summary_path = save_best_probe_summary(probes_base, best_probe_records, config)
                logger.info(
                    f"saved best probe checkpoints ({config.best_probe_checkpoint_name}) to {probes_base}; "
                    f"summary={summary_path}; criterion={config.best_probe_metric}"
                )

    # --- 8. Cleanup ---
    extractor.remove_hooks()

    logger.info("Probe training complete.")
    unwrapped_collection = typing.cast(
        "pyine.guardrails.probes.collection.ProbeCollection",
        accelerator.unwrap_model(probe_collection),  # pyright: ignore[reportUnknownMemberType]  # accelerate stubs
    )
    return ProbeTrainResult(
        probe_collection=unwrapped_collection,
        model=model,
        tokenizer=tokenizer,
    )


async def main(
    config: probe_trainer_configs.ProbeTrainerAppMainConfig,
    runtime: pyine.configs.schemas.RuntimeConfig | None = None,
    skip_training: bool = False,
) -> None:
    """Main entrypoint for probe training."""
    import pyine.utils.reprod

    pyine.utils.reprod.entrypoint_setup(
        runtime_config=runtime,
        use_wandb_logging=config.use_wandb_logging,
        main_config=config,
    )

    if runtime is not None and runtime.dry_run:
        logger.info("dry run mode -- skipping probe training")
        return

    if skip_training:
        if config.probe_checkpoint_dir is None:
            raise ValueError("skip_training=True requires config.probe_checkpoint_dir to be set")
        logger.info(
            f"skip_training mode; loading probes from {config.probe_checkpoint_dir}; "
            f"checkpoint_name={config.probe_checkpoint_name or 'auto'}"
        )
        checkpoint_path = config.llm_checkpoint_path
        model = config.get_model(checkpoint_path=pathlib.Path(checkpoint_path) if checkpoint_path else None)
        model.eval()
        model.requires_grad_(False)
        tokenizer = config.get_tokenizer(checkpoint_path=pathlib.Path(checkpoint_path) if checkpoint_path else None)
        hidden_dim: int = model.config.hidden_size
        probe_collection = pyine.guardrails.probes.collection.ProbeCollection.load_from_checkpoint(  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType,reportAttributeAccessIssue]
            checkpoint_dir=config.probe_checkpoint_dir,
            hidden_dim=hidden_dim,
            checkpoint_name=config.probe_checkpoint_name,
        )
        probe_collection = probe_collection.to(  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]
            dtype=config.target_dtype,
            device=_get_module_device(model),
        )
        probe_collection.eval()  # pyright: ignore[reportUnknownMemberType]
    else:
        train_result = probe_train(config=config, runtime=runtime)
        probe_collection = train_result.probe_collection
        model = train_result.model
        tokenizer = train_result.tokenizer
        if (
            config.evals_config is not None
            and config.save_best_probe_checkpoint
            and config.save_probes
            and pyine.utils.distrib.is_main_process()  # pyright: ignore[reportUnknownMemberType,reportAttributeAccessIssue]
        ):
            probes_base = _get_probes_base_dir(runtime)
            hidden_dim = typing.cast("int", model.config.hidden_size)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            logger.info(
                f"reloading probes from {probes_base}; checkpoint_name={config.best_probe_checkpoint_name} "
                "for post-training evaluation"
            )
            probe_collection = pyine.guardrails.probes.collection.ProbeCollection.load_from_checkpoint(  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType,reportAttributeAccessIssue]
                checkpoint_dir=probes_base,
                hidden_dim=hidden_dim,
                checkpoint_name=config.best_probe_checkpoint_name,
            )
            probe_collection = probe_collection.to(  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]
                dtype=config.target_dtype,
                device=_get_module_device(model),
            )
            probe_collection.eval()  # pyright: ignore[reportUnknownMemberType]
    probe_collection = typing.cast("pyine.guardrails.probes.collection.ProbeCollection", probe_collection)

    # tear down the DDP process group before the (potentially long) eval phase so that the NCCL
    # watchdog on non-main ranks does not time out while rank 0 scores records sequentially
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    # benchmarking phase (if enabled); runs only on main rank, no collectives needed
    if config.evals_config is not None and pyine.utils.distrib.is_main_process():  # pyright: ignore[reportUnknownMemberType,reportAttributeAccessIssue]
        evals_config = typing.cast("correctness_configs.CorrectnessEvalsConfig", config.evals_config)
        eval_dm = evals_config.prepare_eval_datamodule(None)
        eval_dm_typed = typing.cast("correctness_datamodule.CorrectnessDataModule", eval_dm)
        # collect target layers from all probes, create a fresh extractor for scoring
        target_layers = sorted(
            {probe_collection._probe_configs[name].layer for name in probe_collection.probes}  # pyright: ignore[reportPrivateUsage]
        )
        extractor = pyine.guardrails.probes.extraction.ActivationExtractor(  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType,reportAttributeAccessIssue]
            model, target_layers
        )
        # collect all unique records across calibration + all eval subsets for precomputation;
        # calibration records are deterministic (resample_records uses random.Random(config.seed)),
        # and eval subset records come from fixed splits, so the same keys appear on every call
        all_records: list[correctness_types.EvalRecord] = []
        all_records.extend(
            eval_dm_typed.get_records_for_calibration(
                resampling_config=evals_config.calibration_resampling,
            )
        )
        for eval_subset_name in eval_dm_typed.config.resolved_eval_subset_names:
            all_records.extend(eval_dm_typed.get_records_for_subset(eval_subset_name))
        # unpack probe collection into dict for precompute_probe_scores
        probes_dict: dict[
            str, tuple[pyine.guardrails.probes.base.BaseProbe, pyine.guardrails.probes.base.ProbeConfig]
        ] = {}
        for probe_name, probe_module in probe_collection.probes.items():
            probe_cfg = probe_collection._probe_configs[probe_name]  # pyright: ignore[reportPrivateUsage]
            probes_dict[probe_name] = (
                typing.cast("pyine.guardrails.probes.base.BaseProbe", probe_module),
                probe_cfg,
            )
        # pre-compute all probe scores with shared base model forward passes
        precomputed_scorers = correctness_scorers.precompute_probe_scores(
            records=all_records,
            probes=probes_dict,
            model=model,
            tokenizer=tokenizer,
            extractor=extractor,  # pyright: ignore[reportUnknownArgumentType]
            max_seq_length=config.max_seq_length,
            text_field=config.text_field,
        )
        extractor.remove_hooks()  # pyright: ignore[reportUnknownMemberType]
        # group pre-computed scorers by type (base_name)
        scorers_by_type: dict[str, list[correctness_scorers.PrecomputedProbeScorer]] = {}
        for probe_name, scorer in precomputed_scorers.items():
            _, probe_cfg = probes_dict[probe_name]
            base_name = probe_cfg.base_name or probe_cfg.name
            scorers_by_type.setdefault(base_name, []).append(scorer)
        for eval_subset_name in eval_dm_typed.config.resolved_eval_subset_names:
            await correctness_impl.evaluate_guardrail_types(
                config=evals_config,
                guardrails_by_type=scorers_by_type,  # type: ignore[arg-type]
                datamodule=eval_dm_typed,
                eval_subset_name=eval_subset_name,
                wandb_run=runtime.wandb_run if runtime else None,
            )

    if runtime is not None:
        runtime.finalize()


def async_probe_trainer_main_wrapper(
    config: probe_trainer_configs.ProbeTrainerAppMainConfig,
    runtime: pyine.configs.schemas.RuntimeConfig | None = None,
    skip_training: bool = False,
) -> None:
    """Synchronous wrapper around the async main."""
    asyncio.run(main(config=config, runtime=runtime, skip_training=skip_training))


if __name__ == "__main__":
    import pyine.apps.trainers.common

    pyine.apps.trainers.common.hydra_main(
        eval_type=pyine.evals.common.EvalType.CORRECTNESS,
        hydra_config_registration_fn=probe_trainer_configs.register_hydra_configs,
        async_main_wrapper=async_probe_trainer_main_wrapper,
    )
