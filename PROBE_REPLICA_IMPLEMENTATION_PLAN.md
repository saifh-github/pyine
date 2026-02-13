# Probe Replica Training — Implementation Plan

## 1. Goal

Add a **replica training** capability to the probe training system. When enabled, the trainer creates `N` copies of each probe configuration — each initialized with a different random seed — and trains them all in parallel within a single run. Metrics are aggregated across replicas so that W&B plots show **mean and standard deviation** directly, enabling statistically-grounded comparisons of probe architectures and layers without running multiple experiments.

Since all replicas of a given architecture-layer-hyperparameter combination share the same frozen LLM activations (already computed in the forward pass), training 1 vs. 10 replicas adds negligible overhead — only the tiny probe parameters differ.

______________________________________________________________________

## 2. User-Facing Interface

### 2.1 Config Changes

Two new fields on `ProbeTrainerAppMainConfig`:

```yaml
config:
  # Existing fields...
  probe_configs:
    - name: mean_L16
      architecture: mean
      layer: 16
      learning_rate: 1e-3
    - name: attn_L16
      architecture: attention
      layer: 16
      learning_rate: 5e-4
      attn_dim: 64

  # NEW — Replica settings
  num_replicas: 5                  # Number of replicas per probe config (default: 1 = no replication)
  replica_base_seed: 0             # Base seed for deterministic replica initialization (default: 0)
  log_individual_replicas: false   # Also log per-replica metrics to W&B (default: false)
```

### 2.2 What Happens at Runtime

With `num_replicas: 5` and the 2 probe configs above, the system internally creates **10 probes**:

| Internal name | Base name  | Architecture | Layer | Replica | Init seed                        |
| ------------- | ---------- | ------------ | ----- | ------- | -------------------------------- |
| `mean_L16_r0` | `mean_L16` | mean         | 16    | 0       | `blake2b("0:mean_L16:0") % 2^31` |
| `mean_L16_r1` | `mean_L16` | mean         | 16    | 1       | `blake2b("0:mean_L16:1") % 2^31` |
| `mean_L16_r2` | `mean_L16` | mean         | 16    | 2       | `blake2b("0:mean_L16:2") % 2^31` |
| `mean_L16_r3` | `mean_L16` | mean         | 16    | 3       | `blake2b("0:mean_L16:3") % 2^31` |
| `mean_L16_r4` | `mean_L16` | mean         | 16    | 4       | `blake2b("0:mean_L16:4") % 2^31` |
| `attn_L16_r0` | `attn_L16` | attention    | 16    | 0       | `blake2b("0:attn_L16:0") % 2^31` |
| `attn_L16_r1` | `attn_L16` | attention    | 16    | 1       | `blake2b("0:attn_L16:1") % 2^31` |
| ...           | ...        | ...          | ...   | ...     | ...                              |

All 10 probes train simultaneously in a single run, sharing the same LLM forward pass.

### 2.3 W&B Metrics

With `num_replicas > 1`, **aggregated metrics** are logged under the base name:

| Metric Key                     | Description                            |
| ------------------------------ | -------------------------------------- |
| `train/{base_name}/loss/mean`  | Mean train loss across replicas        |
| `train/{base_name}/loss/std`   | Std of train loss across replicas      |
| `valid/{base_name}/loss/mean`  | Mean validation loss across replicas   |
| `valid/{base_name}/loss/std`   | Std of validation loss across replicas |
| `valid/{base_name}/loss/min`   | Min validation loss across replicas    |
| `valid/{base_name}/loss/max`   | Max validation loss across replicas    |
| `valid/{base_name}/auroc/mean` | Mean AUROC across replicas             |
| `valid/{base_name}/auroc/std`  | Std of AUROC across replicas           |
| `valid/{base_name}/auroc/min`  | Min AUROC across replicas              |
| `valid/{base_name}/auroc/max`  | Max AUROC across replicas              |

When `log_individual_replicas: true`, the existing per-probe metrics are also logged as scalar lines:

| Metric Key                   | Description                       |
| ---------------------------- | --------------------------------- |
| `train/{replica_name}/loss`  | Train loss for individual replica |
| `valid/{replica_name}/loss`  | Valid loss for individual replica |
| `valid/{replica_name}/auroc` | AUROC for individual replica      |

### 2.3.1 W&B Tables for Raw Per-Replica Metrics

In addition to the aggregated scalar metrics, **W&B Tables** are logged at each logging/validation step with the raw (non-aggregated) per-replica values. This gives full access to the underlying data without cluttering the scalar metric namespace.

**Training table** (logged every `logging_steps` optimizer steps):

| `probe_name`  | `base_name` | `architecture` | `layer` | `replica_idx` | `step` | `epoch` | `train_loss` |
| ------------- | ----------- | -------------- | ------- | ------------- | ------ | ------- | ------------ |
| `mean_L16_r0` | `mean_L16`  | mean           | 16      | 0             | 10     | 0       | 0.6923       |
| `mean_L16_r1` | `mean_L16`  | mean           | 16      | 1             | 10     | 0       | 0.6931       |
| ...           | ...         | ...            | ...     | ...           | ...    | ...     | ...          |
| `attn_L16_r4` | `attn_L16`  | attention      | 16      | 4             | 10     | 0       | 0.7012       |

**Validation table** (logged at each validation step):

| `probe_name`  | `base_name` | `architecture` | `layer` | `replica_idx` | `step` | `valid_loss` | `auroc` |
| ------------- | ----------- | -------------- | ------- | ------------- | ------ | ------------ | ------- |
| `mean_L16_r0` | `mean_L16`  | mean           | 16      | 0             | 50     | 0.6512       | 0.7234  |
| `mean_L16_r1` | `mean_L16`  | mean           | 16      | 1             | 50     | 0.6489       | 0.7301  |
| ...           | ...         | ...            | ...     | ...           | ...    | ...          | ...     |

These tables are logged as `wandb.Table` objects under the keys `train/replica_details` and `valid/replica_details`. W&B's Table panel supports sorting, filtering by column (e.g., filter `base_name == "mean_L16"`), and exporting to CSV for external analysis.

When `num_replicas: 1` (default), behaviour is unchanged: metrics are logged under the original probe name with no `/mean` or `/std` suffix, and no tables are logged.

### 2.4 Backward Compatibility

- `num_replicas: 1` (default): No config expansion, no replica fields, no aggregation. Identical behaviour to the current system.
- Existing experiment YAML files continue to work without any changes.
- The `_validate_probe_names_unique` validator operates on the **original** `probe_configs` at config parse time (before expansion). Expanded names are guaranteed unique by construction (`{name}_r{idx}`).

______________________________________________________________________

## 3. Design Decisions

### 3.1 Config Expansion vs. ProbeCollection-Internal Replication

**Decision: Config expansion in the training loop, before constructing ProbeCollection.**

Two approaches were considered:

| Approach                                                                                | Pros                                                                                                                        | Cons                                                                                                                      |
| --------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **A: Config expansion** — expand `probe_configs` list before building `ProbeCollection` | ProbeCollection stays simple; each probe is a regular entry in `ModuleDict`; DDP/optimizer/checkpointing all work unchanged | Need to add replica metadata to `ProbeConfig`; expansion logic is new code                                                |
| **B: ProbeCollection-internal** — ProbeCollection creates replicas internally           | No changes to ProbeConfig; self-contained                                                                                   | ProbeCollection becomes complex; parameter groups, checkpointing, and logging all need to understand the nested structure |

**Approach A (config expansion)** is chosen because it keeps the ProbeCollection, optimizer, DDP wrapper, and checkpoint logic completely unchanged. The only new complexity is a straightforward expansion function and an aggregation helper for logging.

### 3.2 Where the Expansion Happens

The expansion happens **inside `probe_train()`**, between reading `config.probe_configs` and constructing the `ProbeCollection`. The original `config.probe_configs` list is never mutated — a new expanded list is produced.

This means:

- The `ProbeTrainerAppMainConfig` stores the user's original configs (clean, no replica suffixes)
- The `_validate_probe_names_unique` validator checks the user's names (not the expanded ones)
- The expanded configs are a runtime-only artefact

### 3.3 Seed Strategy

Each replica needs a **deterministic, unique** seed for weight initialization. The seed must be:

- **Reproducible**: Same `replica_base_seed` + probe name + replica index = same seed
- **Independent**: Changing the runtime seed (for data shuffling) does not change init seeds
- **Collision-free**: Different probes / replicas always get different seeds
- **Stable across DDP ranks**: In multi-GPU training, `torchrun` spawns separate Python processes per rank. Seeds must be identical across ranks so all ranks construct the same initial weights.

**Strategy:**

```python
import hashlib

def _stable_replica_seed(base_seed: int, name: str, replica_idx: int) -> int:
    """Deterministic seed from (base_seed, probe_name, replica_idx).

    Uses blake2b instead of Python's hash(), which is randomized per process
    (PYTHONHASHSEED) and would produce different seeds across DDP ranks.
    """
    payload = f"{base_seed}:{name}:{replica_idx}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**31)
```

This follows the existing pattern in the codebase (`pyine/utils/reprod.py` already uses `hashlib.sha1` and `hashlib.sha256` for deterministic hashing).

The seed is applied via `torch.manual_seed(seed)` before constructing each probe, with the global RNG state saved and restored around the construction loop to avoid side effects on downstream randomness (data loading, etc.).

### 3.4 W&B Visualization Strategy

W&B does not natively render shaded std bands from separate `mean` and `std` scalar metrics logged within a single run. Our strategy uses two complementary logging mechanisms:

1. **Aggregated scalar metrics** (`mean`, `std`, `min`, `max`) for the standard line-chart panels. These give a clean, per-base-name view of training progress.
2. **W&B Tables** with the raw per-replica values at each logging step. These provide the non-aggregated data in a structured, queryable format — useful for detailed analysis, export, and custom visualizations.

**Decision: Aggregated scalars for line charts + W&B Tables for raw replica data.**

The scalar metrics keep the W&B dashboard clean (one line per base config). The tables preserve full provenance and are available for any downstream analysis (filtering by architecture, exporting to CSV, custom Vega charts, etc.) without polluting the metric namespace with N individual replica lines.

### 3.5 Aggregation Scope

Metrics are aggregated only across replicas that share the same `base_name`. The aggregation is a simple mean/std computation — no weighted averaging, no outlier removal. This matches the goal: show the natural variance in probe training due to initialization.

The standard deviation uses **sample std** (`statistics.stdev`, n-1 denominator), which is appropriate since the N replicas are a sample from the population of possible initializations.

**NaN handling for AUROC:** If some replicas produce NaN AUROC (e.g., single-class validation batch in DDP), those values are excluded from the AUROC aggregation. If *all* replicas for a base config have NaN AUROC, NaN is logged for mean/std/min/max.

### 3.6 Memory / Compute Impact

Each probe replica adds only its parameter count to the model:

- MeanProbe: `hidden_dim + 1` params (one Linear layer)
- AttentionProbe: `hidden_dim * (attn_dim + 1) + attn_dim` params

For a 7B model (`hidden_dim=3584`) with 12 probe configs and 5 replicas = 60 probes:

- Worst case (all attention probes, `attn_dim=64`): 60 * ~233K params = ~14M params in float32 = ~56 MB
- Typical (mix of architectures): ~5-20 MB total

This is negligible compared to the frozen LLM (~14 GB for 7B in bf16). The forward pass through probes is also negligible — each probe is a single matmul per sample.

DDP gradient sync: 60 probes' gradients are synchronized in a single all-reduce (they're all in one `ProbeCollection`). With ~14M params in bf16, that's ~28 MB — still tiny.

______________________________________________________________________

## 4. Changes to `ProbeConfig` (`pyine/probes/base.py`)

Add three optional fields for replica metadata:

```python
class ProbeConfig(pydantic.BaseModel):
    """Configuration for a single probe instance."""

    model_config = pydantic.ConfigDict(frozen=False)

    name: str
    architecture: str
    layer: int
    hidden_dim: int | None = None
    # Architecture-specific hyperparams
    window_size: int = 16
    temperature: float = 1.0
    attn_dim: int = 64
    # Training hyperparams (per-probe)
    learning_rate: float = 1e-3
    weight_decay: float = 0.0

    # --- NEW: Replica metadata (populated at runtime, not set by user) ---
    replica_idx: int | None = None
    replica_seed: int | None = None
    base_name: str | None = None
```

- `replica_idx`: 0-indexed replica index. `None` for non-replicated probes.
- `replica_seed`: Seed used for `torch.manual_seed()` before constructing this probe. `None` for non-replicated probes.
- `base_name`: The original probe name before replica expansion. `None` for non-replicated probes (the `name` field itself is the base name).

These fields are only populated programmatically during config expansion — users never set them in YAML.

______________________________________________________________________

## 5. Config Expansion Function

New standalone function in `probe_trainer.py` (or a new `pyine/probes/replicas.py` helper module):

```python
import hashlib


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
```

**Why `hashlib.blake2b` and not `hash()`?** Python's `hash()` is randomized per process via `PYTHONHASHSEED` (default since Python 3.3). In DDP, `torchrun` spawns separate Python processes per rank — each would compute different hashes, leading to different initial weights across ranks. `blake2b` is deterministic regardless of process, machine, or Python version.

______________________________________________________________________

## 6. Changes to `ProbeCollection` (`pyine/probes/collection.py`)

Two changes: (a) seed before each probe construction, and (b) save/restore global RNG state around the loop to avoid side effects on downstream randomness (data loading, dropout, etc.):

```python
class ProbeCollection(torch.nn.Module):
    def __init__(self, probe_configs: list[ProbeConfig], hidden_dim: int) -> None:
        super().__init__()
        from pyine.probes import build_probe

        self._probe_configs: dict[str, ProbeConfig] = {}
        probes: dict[str, torch.nn.Module] = {}

        # NEW: Save RNG state so replica seeding doesn't affect downstream randomness
        any_seeded = any(pc.replica_seed is not None for pc in probe_configs)
        if any_seeded:
            saved_rng_state = torch.random.get_rng_state()

        for pc in probe_configs:
            pc = pc.model_copy(update={"hidden_dim": hidden_dim})
            # NEW: Seed RNG before construction for reproducible replica initialization
            if pc.replica_seed is not None:
                torch.manual_seed(pc.replica_seed)
            probes[pc.name] = build_probe(pc)
            self._probe_configs[pc.name] = pc

        # NEW: Restore RNG state
        if any_seeded:
            torch.random.set_rng_state(saved_rng_state)

        self.probes = torch.nn.ModuleDict(probes)
```

**Why save/restore RNG state?** Calling `torch.manual_seed()` mutates global RNG state. Without restoring it, code that runs after `ProbeCollection` construction (data loading, any module with dropout, etc.) would get different random sequences depending on how many replicas were configured. The save/restore pattern isolates the seeding to probe construction only.

The `forward()` and `get_parameter_groups()` methods remain identical — each expanded probe is just another entry in the `ModuleDict`.

______________________________________________________________________

## 7. Changes to `ProbeTrainerAppMainConfig` (`probe_trainer_configs.py`)

Add three new fields:

```python
class ProbeTrainerAppMainConfig(common.AppMainConfig, common.ModelTokenizerConfigBase):
    # ... existing fields ...

    # --- NEW: Replica settings ---
    num_replicas: int = pydantic.Field(
        default=1,
        ge=1,
        description=(
            "Number of replicas per probe config. Each replica is initialized with a "
            "different random seed. Metrics are aggregated (mean/std) across replicas. "
            "Default 1 = no replication."
        ),
    )
    replica_base_seed: int = pydantic.Field(
        default=0,
        description="Base seed for deterministic replica initialization.",
    )
    log_individual_replicas: bool = pydantic.Field(
        default=False,
        description=(
            "When true, also log per-replica metrics to W&B (in addition to "
            "aggregated mean/std). Useful for debugging but adds many metrics."
        ),
    )
```

The `_validate_probe_names_unique` validator operates on the original `probe_configs` (before expansion), so it continues to work correctly. Expanded names are guaranteed unique by construction (`{name}_r{idx}`).

______________________________________________________________________

## 8. Metric Aggregation Helper

New helper function for grouping and aggregating metrics across replicas:

```python
import math
import statistics
from collections import defaultdict


def aggregate_replica_metrics(
    per_probe_values: dict[str, float],
    probe_configs: dict[str, ProbeConfig],
) -> dict[str, dict[str, float]]:
    """Group metric values by base_name and compute mean/std/min/max.

    Uses sample standard deviation (n-1 denominator) via statistics.stdev().
    NaN values are excluded from aggregation. If all values for a base_name
    are NaN, the result for that base_name has NaN for all fields.

    Args:
        per_probe_values: {probe_name: metric_value} for all probes (including replicas).
        probe_configs: {probe_name: ProbeConfig} with replica metadata.

    Returns:
        {base_name: {"mean": float, "std": float, "min": float, "max": float}}
        For non-replicated probes, base_name == probe_name and std is 0.0.
    """
    groups: dict[str, list[float]] = defaultdict(list)
    for name, value in per_probe_values.items():
        pc = probe_configs[name]
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
                "mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan"),
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
```

This function is used in both train logging and validation logging (see sections 9 and 10).

______________________________________________________________________

## 8a. W&B Table Builders

Two helper functions build `wandb.Table` objects from per-replica metrics. These are called at each logging/validation step and included in the `wandb_run.log()` payload alongside the aggregated scalars.

### 8a.1 Training Table

```python
import wandb


def build_train_replica_table(
    per_probe_losses: dict[str, float],
    expanded_configs_by_name: dict[str, ProbeConfig],
    global_step: int,
    epoch: int,
) -> wandb.Table:
    """Build a W&B Table with raw per-replica training losses.

    Columns: probe_name, base_name, architecture, layer, replica_idx, step, epoch, train_loss
    One row per replica probe.
    """
    table = wandb.Table(columns=[
        "probe_name", "base_name", "architecture", "layer",
        "replica_idx", "step", "epoch", "train_loss",
    ])
    for name, loss_val in per_probe_losses.items():
        pc = expanded_configs_by_name[name]
        table.add_data(
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
```

### 8a.2 Validation Table

```python
def build_valid_replica_table(
    per_probe_metrics: dict[str, dict[str, float]],
    expanded_configs_by_name: dict[str, ProbeConfig],
    global_step: int,
) -> wandb.Table:
    """Build a W&B Table with raw per-replica validation metrics.

    Columns: probe_name, base_name, architecture, layer, replica_idx, step, valid_loss, auroc
    One row per replica probe.
    """
    table = wandb.Table(columns=[
        "probe_name", "base_name", "architecture", "layer",
        "replica_idx", "step", "valid_loss", "auroc",
    ])
    for name, m in per_probe_metrics.items():
        pc = expanded_configs_by_name[name]
        table.add_data(
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
```

### 8a.3 Why W&B Tables?

- **Structured data**: Each row has typed columns with probe metadata (architecture, layer, replica index). This makes it easy to filter, sort, and group in the W&B UI.
- **No metric namespace pollution**: The raw per-replica values live in a table, not as individual scalar metrics. With 60 probes, this avoids 60+ separate metric lines in the dashboard.
- **Queryable**: In W&B, Table panels support column filtering (e.g., show only `base_name == "mean_L16"` or `architecture == "attention"`).
- **Exportable**: Tables can be downloaded as CSV from the W&B UI for external analysis.
- **Logged at each step**: Each `wandb_run.log()` call with a table key appends that table's rows to the run's table history. W&B displays the latest table by default, but the full history is available.

______________________________________________________________________

## 9. Changes to Training Loop (`probe_trainer.py`)

### 9.1 Config Expansion

At the start of `probe_train()`, expand the configs:

```python
def probe_train(config, runtime):
    # ...

    # --- NEW: Expand probe configs with replicas ---
    expanded_probe_configs = expand_probe_configs_with_replicas(
        config.probe_configs, config.num_replicas, config.replica_base_seed,
    )

    if config.num_replicas > 1:
        logger.info(
            f"Replica mode: {len(config.probe_configs)} base configs x "
            f"{config.num_replicas} replicas = {len(expanded_probe_configs)} probes"
        )

    # Build ProbeCollection with expanded configs
    probe_collection = ProbeCollection(expanded_probe_configs, hidden_dim)
    # ...
```

### 9.2 Store Expanded Config Lookup

Build a name→config dict for the aggregation helper:

```python
    expanded_configs_by_name: dict[str, ProbeConfig] = {
        pc.name: pc for pc in expanded_probe_configs
    }
    has_replicas = config.num_replicas > 1
```

### 9.3 Train Logging

Replace the current per-probe logging with aggregated scalars + W&B Table:

```python
    if global_step % config.logging_steps == 0:
        if accelerator.is_main_process:
            per_probe_loss_values = {
                name: loss.item() for name, loss in per_probe_losses.items()
            }

            if has_replicas:
                # Aggregated scalar metrics
                agg = aggregate_replica_metrics(per_probe_loss_values, expanded_configs_by_name)
                log_msg = f"[epoch {epoch + 1}/{config.num_epochs}, step {global_step}] "
                log_msg += ", ".join(
                    f"{base}: {s['mean']:.4f} (std={s['std']:.4f})"
                    for base, s in agg.items()
                )
                logger.info(log_msg)

                if runtime and runtime.wandb_run:
                    log_dict: dict[str, float | int] = {}
                    for base_name, stats in agg.items():
                        log_dict[f"train/{base_name}/loss/mean"] = stats["mean"]
                        log_dict[f"train/{base_name}/loss/std"] = stats["std"]
                    log_dict["train/global_step"] = global_step
                    log_dict["train/epoch"] = epoch

                    # Optional: individual replica scalar metrics
                    if config.log_individual_replicas:
                        for name, loss_val in per_probe_loss_values.items():
                            log_dict[f"train/{name}/loss"] = loss_val

                    # W&B Table with raw per-replica values
                    train_table = build_train_replica_table(
                        per_probe_loss_values, expanded_configs_by_name,
                        global_step, epoch,
                    )
                    log_dict["train/replica_details"] = train_table

                    runtime.wandb_run.log(log_dict, step=global_step)
            else:
                # No replicas — use original logging (backward compatible)
                log_msg = f"[epoch {epoch + 1}/{config.num_epochs}, step {global_step}] "
                log_msg += ", ".join(
                    f"{name}: {loss.item():.4f}" for name, loss in per_probe_losses.items()
                )
                logger.info(log_msg)

                if runtime and runtime.wandb_run:
                    log_dict = {
                        f"train/{name}/loss": loss.item()
                        for name, loss in per_probe_losses.items()
                    }
                    log_dict["train/global_step"] = global_step
                    log_dict["train/epoch"] = epoch
                    runtime.wandb_run.log(log_dict, step=global_step)
```

### 9.4 Activation Layer Deduplication

Currently, `target_layers` is derived from the user's `config.probe_configs`. With replicas, it must be derived from the **expanded** configs (though the set of unique layers is the same since replicas share layers):

```python
    # Use expanded configs for layer set (same result but correct data flow)
    target_layers = sorted({pc.layer for pc in expanded_probe_configs})
```

______________________________________________________________________

## 10. Changes to Validation (`validate_probes`)

### 10.1 Pass Additional Context

The `validate_probes` function needs to know about replica grouping. Add parameters:

```python
def validate_probes(
    probe_collection,
    model,
    extractor,
    valid_loader,
    loss_fn,
    global_step,
    accelerator,
    runtime,
    # NEW parameters:
    expanded_configs_by_name: dict[str, ProbeConfig] | None = None,
    log_individual_replicas: bool = False,
) -> dict[str, dict[str, float]]:
```

The new parameters are optional to maintain backward compatibility (tests that call `validate_probes` directly continue to work).

### 10.2 Aggregated Validation Logging

After computing per-probe metrics, aggregate if replicas are present:

```python
    if accelerator.is_main_process:
        # ... existing per-probe metric computation (unchanged) ...

        # NEW: Aggregate across replicas
        has_replicas = (
            expanded_configs_by_name is not None
            and any(pc.base_name is not None for pc in expanded_configs_by_name.values())
        )

        if has_replicas and expanded_configs_by_name is not None:
            # Aggregate loss and AUROC (NaN handling is inside aggregate_replica_metrics)
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

                # Optional: individual replica scalar metrics
                if log_individual_replicas:
                    for name, m in metrics.items():
                        log_dict[f"valid/{name}/loss"] = m["loss"]
                        log_dict[f"valid/{name}/auroc"] = m["auroc"]

                # W&B Table with raw per-replica values
                valid_table = build_valid_replica_table(
                    metrics, expanded_configs_by_name, global_step,
                )
                log_dict["valid/replica_details"] = valid_table

                runtime.wandb_run.log(log_dict, step=global_step)

            # Log aggregated summary to console
            logger.info(
                f"[step {global_step}] validation (aggregated): "
                + ", ".join(
                    f"{base}: loss={s['mean']:.4f}+-{s['std']:.4f}"
                    for base, s in loss_agg.items()
                )
                + " | "
                + ", ".join(
                    f"{base}: auroc={s['mean']:.4f}+-{s['std']:.4f}"
                    for base, s in auroc_agg.items()
                )
            )
        else:
            # No replicas — original per-probe logging (unchanged)
            ...
```

______________________________________________________________________

## 11. Checkpoint Saving with Replicas

### 11.1 Directory Structure

With replicas, the output structure becomes:

```
probes/
  mean_L16_r0/
    probe_state_dict.pt
    probe_config.json
  mean_L16_r1/
    probe_state_dict.pt
    probe_config.json
  ...
  attn_L16_r0/
    probe_state_dict.pt
    probe_config.json
  ...
  replica_summary.json       # NEW: Aggregated final metrics
```

### 11.2 Replica Summary

A summary file with aggregated final validation metrics is saved alongside the individual checkpoints:

```python
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
    auroc_values = {
        name: m["auroc"] for name, m in final_metrics.items()
        if not math.isnan(m["auroc"])
    }
    auroc_agg = aggregate_replica_metrics(auroc_values, expanded_configs_by_name)

    summary = {
        base_name: {
            "loss": loss_agg.get(base_name, {}),
            "auroc": auroc_agg.get(base_name, {}),
            "num_replicas": len([
                pc for pc in expanded_configs_by_name.values()
                if (pc.base_name or pc.name) == base_name
            ]),
        }
        for base_name in {
            pc.base_name or pc.name for pc in expanded_configs_by_name.values()
        }
    }
    (output_dir / "replica_summary.json").write_text(json.dumps(summary, indent=2))
```

### 11.3 No Change to `save_probe_checkpoints`

The existing `save_probe_checkpoints` function iterates over all probes in the collection and saves each one by name. With replicas, the names include the `_r{idx}` suffix, so each replica gets its own directory. No code changes needed.

The `replica_summary.json` is saved separately after the final validation.

______________________________________________________________________

## 12. Loading Saved Replica Probes

The `probe_config.json` for each replica includes the replica metadata (`replica_idx`, `replica_seed`, `base_name`), so the user has full provenance:

```json
{
  "name": "mean_L16_r2",
  "architecture": "mean",
  "layer": 16,
  "hidden_dim": 3584,
  "learning_rate": 0.001,
  "weight_decay": 0.0,
  "replica_idx": 2,
  "replica_seed": 1847293651,
  "base_name": "mean_L16"
}
```

Loading works exactly as before:

```python
config = ProbeConfig(**json.loads(open("probes/mean_L16_r2/probe_config.json").read()))
probe = build_probe(config)
state_dict = torch.load("probes/mean_L16_r2/probe_state_dict.pt", weights_only=True)
probe.load_state_dict(state_dict)
```

______________________________________________________________________

## 13. W&B Visualization Guide

### 13.1 Aggregated Line Charts (Automatic)

W&B auto-generates panels for the aggregated scalar metrics. These appear as clean line charts:

- `train/{base_name}/loss/mean` — average train loss over time
- `train/{base_name}/loss/std` — standard deviation of train loss over time
- `valid/{base_name}/auroc/mean` — average AUROC over time
- `valid/{base_name}/auroc/min`, `auroc/max` — range bounds

To compare architectures, overlay multiple `*/loss/mean` or `*/auroc/mean` metrics on a single panel. The `std` metric can be added to a second y-axis or a separate panel for readability.

### 13.2 Replica Details Tables

The `train/replica_details` and `valid/replica_details` tables are visible in the W&B run page under the **Tables** section. Each table has full probe metadata columns, so you can:

- **Filter** by `base_name` to see all replicas of one probe configuration
- **Filter** by `architecture` to compare all probes of a given type
- **Sort** by `train_loss` or `auroc` to find best/worst replicas
- **Export** to CSV for external plotting (e.g., with matplotlib for publication-quality mean+std bands)

### 13.3 Quick Range Visualization

The aggregation logs `min` and `max` for AUROC. To see the spread across replicas directly in the W&B line chart panel:

1. Add `valid/{base_name}/auroc/mean`, `valid/{base_name}/auroc/min`, and `valid/{base_name}/auroc/max` to the same panel
2. This gives a 3-line view showing the range without requiring custom Vega charts

______________________________________________________________________

## 14. File Changes Summary

| File                                                | Change type | Description                                                                                                                                                                                                                                               |
| --------------------------------------------------- | ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pyine/probes/base.py`                              | **Modify**  | Add `replica_idx`, `replica_seed`, `base_name` fields to `ProbeConfig`                                                                                                                                                                                    |
| `pyine/probes/collection.py`                        | **Modify**  | Add `torch.manual_seed()` call before probe construction when `replica_seed` is set                                                                                                                                                                       |
| `pyine/apps/trainers/probe_trainer_configs.py`      | **Modify**  | Add `num_replicas`, `replica_base_seed`, `log_individual_replicas` fields to `ProbeTrainerAppMainConfig`                                                                                                                                                  |
| `pyine/apps/trainers/probe_trainer.py`              | **Modify**  | Add `expand_probe_configs_with_replicas()`, `aggregate_replica_metrics()`, `build_train_replica_table()`, `build_valid_replica_table()`, update `probe_train()` and `validate_probes()` for aggregated logging + W&B Tables, add `save_replica_summary()` |
| `pyine/configs/experiment/probes/v0_probe.yaml`     | **Modify**  | Add commented-out `num_replicas`, `replica_base_seed`, `log_individual_replicas` fields as documentation                                                                                                                                                  |
| `pyine/apps/trainers/PROBE_TRAINING_GUIDE.md`       | **Modify**  | Add section on replica training                                                                                                                                                                                                                           |
| `tests/probes/conftest.py`                          | **Modify**  | Add `replica_probe_configs` and `replica_probe_collection` fixtures                                                                                                                                                                                       |
| `tests/probes/test_collection.py`                   | **Modify**  | Add `TestReplicaSeeding` class (4 tests)                                                                                                                                                                                                                  |
| `tests/apps/trainers/test_probe_trainer.py`         | **Modify**  | Add `TestExpandProbeConfigsWithReplicas` (8), `TestAggregateReplicaMetrics` (7), `TestReplicaTables` (7), `TestValidateProbesWithReplicas` (3), `TestStableReplicaSeed` (5), and 1 new method on `TestProbeTrainUnit` — 31 tests total                    |
| `tests/apps/trainers/test_probe_trainer_configs.py` | **Modify**  | Add `TestProbeTrainerConfigReplicas` class (5 tests)                                                                                                                                                                                                      |

______________________________________________________________________

## 15. Test Plan

This section maps each new test class/method to its target file and describes changes to existing fixtures and tests. Existing tests are **not modified** unless stated — the new `ProbeConfig` fields are all optional with `None` defaults, so current test code remains compatible.

### 15.0 Fixture Changes: `tests/probes/conftest.py`

Add new shared fixtures alongside the existing ones (`sample_probe_configs`, `random_activations`, `random_attention_mask`, `probe_hidden_dim`):

```python
@pytest.fixture
def replica_probe_configs() -> list[ProbeConfig]:
    """Two base configs expanded to 3 replicas each (6 total), with replica metadata set.

    Uses a subset of architectures (mean + attention) to keep tests fast
    while covering both simple and complex probe init paths.
    """
    from pyine.apps.trainers.probe_trainer import expand_probe_configs_with_replicas

    base_configs = [
        ProbeConfig(name="mean_L0", architecture="mean", layer=0),
        ProbeConfig(name="attn_L8", architecture="attention", layer=8, attn_dim=16),
    ]
    return expand_probe_configs_with_replicas(base_configs, num_replicas=3, replica_base_seed=42)


@pytest.fixture
def replica_probe_collection(replica_probe_configs: list[ProbeConfig]) -> ProbeCollection:
    """ProbeCollection built from replica-expanded configs."""
    return ProbeCollection(replica_probe_configs, hidden_dim=PROBE_HIDDEN_DIM)
```

**No changes to existing fixtures.** The existing `sample_probe_configs` (6 architectures, no replica metadata) continues to test the non-replica path.

______________________________________________________________________

### 15.1 `tests/apps/trainers/test_probe_trainer.py` — Config Expansion

New class added to the existing file, after the existing `TestDatasetValidation` class:

```python
class TestExpandProbeConfigsWithReplicas:
    """Tests for the config expansion function."""

    def test_no_expansion_when_num_replicas_1(self):
        """num_replicas=1 returns original list unchanged (same object)."""

    def test_expansion_creates_correct_count(self):
        """2 configs x 3 replicas = 6 expanded configs."""

    def test_expanded_names_follow_pattern(self):
        """Expanded name is '{original}_r{idx}'."""

    def test_replica_metadata_populated(self):
        """Each expanded config has replica_idx, replica_seed, base_name set."""

    def test_seeds_are_unique(self):
        """All replica seeds are distinct across all expanded configs."""

    def test_seeds_are_deterministic(self):
        """Same inputs produce same seeds on repeated calls."""

    def test_base_name_matches_original(self):
        """base_name equals the original probe config name."""

    def test_non_replica_fields_preserved(self):
        """architecture, layer, learning_rate, etc. are unchanged in replicas."""
```

### 15.2 `tests/apps/trainers/test_probe_trainer.py` — Metric Aggregation

New class added to the same file:

```python
class TestAggregateReplicaMetrics:
    """Tests for the metric aggregation helper."""

    def test_single_group(self):
        """All probes share base_name → single aggregated entry."""

    def test_multiple_groups(self):
        """Different base_names → separate aggregated entries."""

    def test_single_value_std_is_zero(self):
        """A group with 1 value has std=0.0."""

    def test_non_replicated_probes(self):
        """Probes with base_name=None use name as base."""

    def test_mean_std_correctness(self):
        """Mean and std match statistics.mean() and statistics.stdev()."""

    def test_nan_excluded_from_aggregation(self):
        """NaN values are excluded; non-NaN values still produce correct stats."""

    def test_all_nan_returns_nan(self):
        """When all values for a base_name are NaN, all stats are NaN."""
```

### 15.3 `tests/probes/test_collection.py` — Seeded Replica Construction

New class added to the existing file, after the existing `TestProbeCollection` class. Uses the new `replica_probe_configs` fixture from `conftest.py`:

```python
class TestReplicaSeeding:
    """Tests for reproducible replica initialization in ProbeCollection."""

    def test_different_replicas_have_different_weights(
        self, replica_probe_configs: list[ProbeConfig],
    ):
        """Two replicas of the same architecture with different seeds have different weights.

        Builds a ProbeCollection from replica_probe_configs and checks that
        mean_L0_r0 and mean_L0_r1 have different weight tensors.
        """

    def test_same_seed_produces_same_weights(self):
        """Building ProbeCollection twice with the same expanded configs produces identical weights.

        Constructs two ProbeCollections from the same replica_probe_configs
        and asserts torch.equal on all parameter pairs.
        """

    def test_rng_state_restored_after_construction(self):
        """Global RNG state is restored after ProbeCollection construction with replicas.

        Saves torch.random.get_rng_state() before constructing a ProbeCollection
        from replica_probe_configs, then verifies the state is restored after
        construction (i.e., torch.randn() after construction is not affected
        by the replica seeding).
        """

    def test_non_seeded_probes_unaffected(self, sample_probe_configs: list[ProbeConfig]):
        """Probes without replica_seed use default init and don't trigger RNG save/restore.

        Constructs a ProbeCollection from sample_probe_configs (no replica fields)
        and verifies construction completes normally.
        """
```

### 15.4 `tests/apps/trainers/test_probe_trainer.py` — W&B Table Builders

New class added to the existing file:

```python
class TestReplicaTables:
    """Tests for W&B Table construction helpers.

    Uses wandb.Table directly (no W&B run needed) since the builders
    just construct Table objects.
    """

    def test_train_table_has_correct_columns(self):
        """build_train_replica_table returns table with expected column names:
        probe_name, base_name, architecture, layer, replica_idx, step, epoch, train_loss."""

    def test_train_table_row_count_matches_probes(self):
        """Table has one row per probe in per_probe_losses dict."""

    def test_train_table_base_name_populated(self):
        """base_name column reflects the original probe name, not the replica name."""

    def test_valid_table_has_correct_columns(self):
        """build_valid_replica_table returns table with expected column names:
        probe_name, base_name, architecture, layer, replica_idx, step, valid_loss, auroc."""

    def test_valid_table_includes_loss_and_auroc(self):
        """Each row has valid_loss and auroc values from the per-probe metrics."""

    def test_valid_table_row_count_matches_probes(self):
        """Table has one row per probe in per_probe_metrics dict."""

    def test_tables_with_non_replicated_probes(self):
        """Tables work when replica_idx is None (non-replica ProbeConfigs).
        The replica_idx column should be 0 for non-replicated probes."""
```

### 15.5 `tests/apps/trainers/test_probe_trainer.py` — Validation with Replicas

New class added to the existing file. Uses the `SmallMockLLM` already defined there:

```python
class TestValidateProbesWithReplicas:
    """Tests for validate_probes with replica-expanded probe collections.

    Reuses the SmallMockLLM and dataset patterns from TestProbeTrainUnit,
    but with replica-expanded configs.
    """

    def test_validation_returns_per_replica_metrics(self):
        """validate_probes returns metrics keyed by replica name (e.g., 'mean_L0_r0').

        Creates a ProbeCollection from replica-expanded configs, runs
        validate_probes, and checks that the returned dict has one entry
        per expanded probe name.
        """

    def test_aggregation_of_validation_metrics(self):
        """aggregate_replica_metrics correctly groups validate_probes output by base_name.

        Calls validate_probes → passes result to aggregate_replica_metrics →
        verifies aggregated dict has one entry per base name with mean/std/min/max.
        """

    def test_auroc_nan_handling_in_aggregation(self):
        """NaN AUROC values (single-class valid set) are excluded from aggregation.

        Uses all-same-label dataset (like TestProbeTrainUnit.test_auroc_guard_single_class),
        then checks that aggregate_replica_metrics returns NaN for AUROC fields.
        """
```

### 15.6 `tests/apps/trainers/test_probe_trainer_configs.py` — Config Validation

New class added to the existing file, after the existing `TestHydraConfigRegistration` class:

```python
class TestProbeTrainerConfigReplicas:
    """Tests for replica-related config fields on ProbeTrainerAppMainConfig.

    Extends TestProbeTrainerAppMainConfig by reusing its _make_minimal_config pattern.
    """

    def _make_minimal_config(self, **overrides: object) -> dict:
        """Minimal valid config kwargs (same as TestProbeTrainerAppMainConfig)."""
        base = {
            "base_model": "some-model",
            "dataset_path": "/tmp/fake-dataset",
            "probe_configs": [
                ProbeConfig(name="mean_L0", architecture="mean", layer=0),
            ],
        }
        base.update(overrides)
        return base

    def test_num_replicas_default_is_1(self):
        """Default num_replicas is 1."""

    def test_num_replicas_must_be_positive(self):
        """num_replicas < 1 raises validation error (ge=1 constraint)."""

    def test_replica_fields_accepted(self):
        """num_replicas, replica_base_seed, log_individual_replicas are accepted without error."""

    def test_replica_base_seed_default_is_0(self):
        """Default replica_base_seed is 0."""

    def test_log_individual_replicas_default_is_false(self):
        """Default log_individual_replicas is False."""
```

### 15.7 `tests/apps/trainers/test_probe_trainer.py` — Seed Stability (blake2b)

New class added to the existing file:

```python
class TestStableReplicaSeed:
    """Tests for _stable_replica_seed determinism."""

    def test_deterministic_across_calls(self):
        """Same (base_seed, name, replica_idx) → same seed on every call."""

    def test_different_replica_idx_different_seeds(self):
        """Different replica_idx values produce different seeds."""

    def test_different_base_seed_different_seeds(self):
        """Different base_seed values produce different seeds."""

    def test_different_name_different_seeds(self):
        """Different probe names produce different seeds."""

    def test_seed_within_valid_range(self):
        """Returned seed is in [0, 2^31)."""
```

### 15.8 `tests/apps/trainers/test_probe_trainer.py` — End-to-End with Replicas

New method added to the existing `TestProbeTrainUnit` class:

```python
class TestProbeTrainUnit:
    # ... existing tests ...

    def test_train_step_reduces_loss_with_replicas(self, mock_llm: SmallMockLLM) -> None:
        """Train loop with replica-expanded probes reduces loss (same as
        test_train_step_reduces_loss but using 2 base configs x 2 replicas = 4 probes).

        Verifies that:
        - ProbeCollection accepts replica-expanded configs
        - All 4 probes train without error
        - Overall loss decreases
        """
```

______________________________________________________________________

### 15.9 Existing Tests — Compatibility Notes

**No existing tests are modified.** All new `ProbeConfig` fields (`replica_idx`, `replica_seed`, `base_name`) default to `None`, so existing test code that constructs `ProbeConfig` without these fields continues to work unchanged. Specifically:

| Existing file                                       | Impact                                                                                                                                                                                |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tests/probes/conftest.py`                          | **Unchanged.** New fixtures are added; existing fixtures remain as-is.                                                                                                                |
| `tests/probes/test_probes.py`                       | **Unchanged.** Tests individual probe architectures using `ProbeConfig(name=..., architecture=..., layer=..., hidden_dim=...)` — no replica fields needed.                            |
| `tests/probes/test_collection.py`                   | **Unchanged.** Existing `TestProbeCollection` tests use `sample_probe_configs` (no replicas). The new `TestReplicaSeeding` class is added alongside.                                  |
| `tests/probes/test_extraction.py`                   | **Unchanged.** ActivationExtractor is not affected by replica changes.                                                                                                                |
| `tests/probes/test_debug_dataset.py`                | **Unchanged.** Debug dataset is not affected by replica changes.                                                                                                                      |
| `tests/apps/trainers/test_probe_trainer.py`         | **Unchanged existing tests.** New classes and one new method are added. The existing `SmallMockLLM`, fixtures, and `TestProbeTrainUnit`/`TestDatasetValidation` classes remain as-is. |
| `tests/apps/trainers/test_probe_trainer_configs.py` | **Unchanged existing tests.** New `TestProbeTrainerConfigReplicas` class is added.                                                                                                    |

### 15.10 Test Summary

| Target file                                         | New test class/method                                                | # Tests |
| --------------------------------------------------- | -------------------------------------------------------------------- | ------- |
| `tests/probes/conftest.py`                          | 2 new fixtures (`replica_probe_configs`, `replica_probe_collection`) | —       |
| `tests/probes/test_collection.py`                   | `TestReplicaSeeding`                                                 | 4       |
| `tests/apps/trainers/test_probe_trainer.py`         | `TestExpandProbeConfigsWithReplicas`                                 | 8       |
| `tests/apps/trainers/test_probe_trainer.py`         | `TestAggregateReplicaMetrics`                                        | 7       |
| `tests/apps/trainers/test_probe_trainer.py`         | `TestReplicaTables`                                                  | 7       |
| `tests/apps/trainers/test_probe_trainer.py`         | `TestValidateProbesWithReplicas`                                     | 3       |
| `tests/apps/trainers/test_probe_trainer.py`         | `TestStableReplicaSeed`                                              | 5       |
| `tests/apps/trainers/test_probe_trainer.py`         | `TestProbeTrainUnit.test_train_step_reduces_loss_with_replicas`      | 1       |
| `tests/apps/trainers/test_probe_trainer_configs.py` | `TestProbeTrainerConfigReplicas`                                     | 5       |
| **Total**                                           |                                                                      | **40**  |

______________________________________________________________________

## 16. Implementation Steps

### Step 1: Add replica fields to `ProbeConfig`

Modify `pyine/probes/base.py` to add `replica_idx`, `replica_seed`, `base_name`.

**Tests:** Verify ProbeConfig construction with and without replica fields.

### Step 2: Add seeding to `ProbeCollection`

Modify `pyine/probes/collection.py` to call `torch.manual_seed()` before probe construction when `replica_seed` is set.

**Tests:** `test_different_replicas_have_different_weights`, `test_same_seed_produces_same_weights`.

### Step 3: Add config fields to `ProbeTrainerAppMainConfig`

Add `num_replicas`, `replica_base_seed`, `log_individual_replicas` to `probe_trainer_configs.py`.

**Tests:** Config validation tests.

### Step 4: Implement `expand_probe_configs_with_replicas`, `aggregate_replica_metrics`, and table builders

Add `expand_probe_configs_with_replicas`, `aggregate_replica_metrics`, `build_train_replica_table`, and `build_valid_replica_table` to `probe_trainer.py`.

**Tests:** Full test suites for expansion (15.1), aggregation (15.2), and table builders (15.4).

### Step 5: Update `probe_train()` and `validate_probes()`

Integrate config expansion, aggregated train logging, and aggregated validation logging.

**Tests:** End-to-end unit test with mocked LLM and `num_replicas=3`.

### Step 6: Update `save_probe_checkpoints` caller to add `replica_summary.json`

Add `save_replica_summary()` after the final validation.

**Tests:** Verify summary file is written with correct structure.

### Step 7: Update documentation and example config

- Add replica fields (commented out) to `v0_probe.yaml`
- Add replica section to `PROBE_TRAINING_GUIDE.md`

### Step 8: Run full test suite

```bash
uv run pytest tests/probes/ tests/apps/trainers/test_probe_trainer.py tests/apps/trainers/test_probe_trainer_configs.py -v
```

______________________________________________________________________

## 17. Example: Updated v0_probe.yaml

```yaml
# @package _global_

# ... existing fields ...

config:
  _target_: pyine.apps.trainers.probe_trainer_configs.ProbeTrainerAppMainConfig

  # ... existing fields ...

  # Replica settings (NEW)
  num_replicas: 5             # Train 5 copies of each probe config with different init seeds
  replica_base_seed: 0        # Base seed for reproducible replica initialization
  log_individual_replicas: false  # Set to true to also log per-replica metrics

  probe_configs:
    - name: mean_L16
      architecture: mean
      layer: 16
      learning_rate: 1e-3
    # ... etc ...
```

With `num_replicas: 5` and 12 probe configs, this creates 60 probes. The W&B dashboard shows 12 curves (one per base config) with mean and std aggregated from 5 replicas each.

______________________________________________________________________

## 18. Risks and Mitigations

| Risk                                                             | Likelihood | Impact | Mitigation                                                                                                                                                                                                             |
| ---------------------------------------------------------------- | ---------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Memory pressure** with many replicas                           | Low        | Medium | 60 probes = ~20 MB; negligible vs. LLM. Document in guide.                                                                                                                                                             |
| **DDP sync overhead** with many probes                           | Low        | Low    | Single all-reduce for all probes; ~28 MB payload with 60 probes is small.                                                                                                                                              |
| **DDP weight divergence** across ranks                           | Low        | High   | Mitigated by using `hashlib.blake2b` for seed generation (deterministic across processes). All ranks compute identical seeds → identical initial weights. DDP then keeps weights in sync via gradient synchronization. |
| **Metric explosion in W&B** with `log_individual_replicas: true` | Low        | Low    | Off by default. When on, 60 probes x 3 metrics = 180 metrics — W&B handles this fine.                                                                                                                                  |
| **Backward compatibility**                                       | Low        | High   | `num_replicas=1` triggers no expansion, no aggregation, no naming changes. Tested explicitly.                                                                                                                          |

______________________________________________________________________

## 19. Codex Review Assessment

| #   | Codex Concern                                      | Verdict                | Action                                                                                                                                                            |
| --- | -------------------------------------------------- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `hash()` not stable across DDP ranks               | **Accepted**           | Replaced with `hashlib.blake2b`. Real bug — `PYTHONHASHSEED` is not set in this codebase and `torchrun` spawns separate processes.                                |
| 2   | Global RNG side effects from `torch.manual_seed()` | **Accepted**           | Added save/restore of RNG state around probe construction loop in `ProbeCollection`.                                                                              |
| 3   | Define std semantics (sample vs population)        | **Accepted (doc)**     | Added note that `statistics.stdev()` uses sample std (n-1). No code change — sample std is correct here.                                                          |
| 4   | NaN AUROC aggregation edge case                    | **Accepted partially** | `aggregate_replica_metrics` now filters NaN values and logs NaN for mean/std/min/max when all values are NaN. No `num_valid` tracking — unnecessary complexity.   |
| 5   | DDP parameter consistency                          | **Accepted (doc)**     | Covered by fix #1. Added DDP weight divergence row to risks table with explanation.                                                                               |
| 6   | W&B table frequency knob                           | **Rejected**           | Tables are small (~60 rows of floats per step). `logging_steps` already controls frequency. Adding a separate knob is over-engineering for a theoretical concern. |
| 7   | Validator scope wording inconsistency              | **Accepted**           | Fixed section 2.4 to correctly state validator operates on original configs.                                                                                      |
| 8   | Replica metadata defaults in tables                | **Rejected**           | When `num_replicas=1`, tables aren't logged. When >1, all probes have `replica_idx` set. No ambiguity.                                                            |
| 9   | Log min/max for loss (symmetry with AUROC)         | **Accepted**           | Added `loss/min` and `loss/max` to validation logging.                                                                                                            |
| 10  | Shallow copy when `num_replicas==1`                | **Rejected**           | `ProbeCollection` already copies configs via `model_copy()`. Mutation risk is theoretical and doesn't apply.                                                      |

______________________________________________________________________

## 20. Future Extensions (Out of Scope)

- **Per-probe-config replica count**: Allow `num_replicas` per probe config entry instead of globally. Useful for running more replicas on simpler architectures.
- **Ensemble predictions at inference**: Average predictions across replicas of the same base config for improved accuracy.
- **W&B grouped runs**: Launch replicas as separate W&B runs within a group for native shaded band rendering.
- **Replica pruning**: Early-stop individual replicas that are clearly degenerate while continuing the rest.
