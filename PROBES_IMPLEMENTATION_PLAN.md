# Probe Training App - Implementation Plan (v2.1)

## 1. Goal

Build a training app that trains lightweight **probe classifiers** on frozen LLM activations extracted from a model checkpoint. Given:

- A pre-processed HuggingFace dataset with text inputs (chat-template or plain text) and a binary label column
- A model checkpoint (safetensors, optionally with LoRA adapters merged or loadable via PEFT)
- A set of probe configurations (architecture + target layer + hyperparameters)

The app will:

1. Load the frozen LLM (full copy per GPU) and extract hidden-state activations at specified layers
2. Train **N probe models simultaneously** per forward pass (different architectures, layers, hyperparameters)
3. Support **multi-GPU DDP** via `accelerate` (each GPU holds the full frozen LLM + all probes; data is sharded)
4. Log per-probe loss (train + valid) and AUROC (valid) to W&B
5. Follow existing hydra-zen config patterns for full composability

______________________________________________________________________

## 2. High-Level Architecture

```
pyine/
  probes/                              # NEW: Probe module library
    __init__.py                        # Registry + build_probe factory
    base.py                            # BaseProbe, ProbeConfig
    collection.py                      # ProbeCollection nn.Module (DDP-friendly wrapper)
    extraction.py                      # ActivationExtractor (hook-based)
    debug_dataset.py                   # Synthetic dataset factory for testing
    mean_probe.py                      # Mean pooling probe
    max_probe.py                       # Max pooling probe
    last_token_probe.py                # Last-token probe
    rolling_mean_probe.py              # Max-of-rolling-means probe
    softmax_probe.py                   # Softmax-weighted probe
    attention_probe.py                 # Attention-based probe
  apps/
    trainers/
      probe_trainer.py                 # NEW: Main training loop + entrypoint
      probe_trainer_configs.py         # NEW: Hydra-zen config registration
  configs/
    experiment/
      probes/                          # NEW: Experiment YAML configs
        v0_probe.yaml

tests/
  probes/                              # NEW: Probe test suite
    __init__.py
    conftest.py                        # Shared fixtures (synthetic dataset, mock model, etc.)
    test_probes.py                     # Unit tests for all 6 probe architectures
    test_collection.py                 # ProbeCollection tests (forward, param groups, DDP)
    test_extraction.py                 # ActivationExtractor tests (hooks, layer resolution)
    test_debug_dataset.py              # Debug dataset factory tests
  apps/
    trainers/
      test_probe_trainer.py            # Training loop unit tests (mocked LLM)
      test_probe_trainer_configs.py    # Config validation + Hydra registration tests
      test_probe_trainer_integration.py  # End-to-end integration test (GPU, slow)
```

______________________________________________________________________

## 3. Probe Architectures

All probes share a common interface: they receive a tensor of per-token hidden states `H` of shape `(batch, seq_len, hidden_dim)` and an attention mask of shape `(batch, seq_len)`, and produce a scalar logit per sample `(batch, 1)`.

Each probe has a small linear head `W` of shape `(hidden_dim, 1)` that maps a pooled representation to a logit. The architectures differ only in **how they pool** across the sequence dimension.

### 3.1 Mean Probe

Computes the mean activation over non-masked positions, then applies the linear head.

```
pooled_repr = masked_mean(H, mask)  -> (batch, hidden_dim)
logit = pooled_repr @ W              -> (batch, 1)
```

### 3.2 Max Probe

Takes the maximum per-token score over non-masked positions.

```
score_per_token = H @ W                      -> (batch, seq_len, 1)
pooled = masked_max(score_per_token, mask)    -> (batch, 1)
```

### 3.3 Last-Token Probe

Extracts the hidden state at the last non-padding position, then applies the linear head.

```
last_positions = mask.sum(dim=1) - 1                   -> (batch,)
pooled_repr = H[batch_idx, last_positions, :]           -> (batch, hidden_dim)
logit = pooled_repr @ W                                 -> (batch, 1)
```

### 3.4 Max-of-Rolling-Means Probe

Applies a 1D rolling-mean window of size `T` to per-token scores, then takes the max.

```
score_per_token = H @ W                                  -> (batch, seq_len, 1)
rolling = avg_pool1d(score_per_token, kernel_size=T)     -> (batch, seq_len', 1)
pooled = masked_max(rolling, adjusted_mask)               -> (batch, 1)
```

**Hyperparameter:** `window_size: int` (default 16).

### 3.5 Softmax Probe

Computes a temperature-scaled softmax over per-token scores and takes a weighted sum.

```
score_per_token = H @ W                                   -> (batch, seq_len, 1)
weights = masked_softmax(score_per_token / phi, mask)     -> (batch, seq_len, 1)
pooled = (weights * score_per_token).sum(dim=1)           -> (batch, 1)
```

**Hyperparameter:** `temperature: float` (default 1.0).

### 3.6 Attention Probe

Uses learned query/value projections to compute attention-weighted pooling. This is the only architecture with more than one set of learnable parameters.

```
Q = H @ W_q   -> (batch, seq_len, attn_dim)
V = H @ W_v   -> (batch, seq_len, 1)
attn_scores = Q @ Q_global.T / sqrt(attn_dim)   -> (batch, seq_len, 1)
attn_weights = masked_softmax(attn_scores, mask)
pooled = (attn_weights * V).sum(dim=1)           -> (batch, 1)
```

**Hyperparameter:** `attn_dim: int` (default 64).

Here `W_q` is `(hidden_dim, attn_dim)`, `Q_global` is a learned query vector `(attn_dim,)`, and `W_v` is `(hidden_dim, 1)`. This gives the probe ~`hidden_dim * (attn_dim + 1) + attn_dim` parameters total.

______________________________________________________________________

## 4. Module Design: `pyine/probes/`

### 4.1 `base.py` - ProbeConfig & BaseProbe

```python
class ProbeConfig(pydantic.BaseModel):
    """Configuration for a single probe instance."""
    name: str                          # Unique identifier (e.g., "mean_L12")
    architecture: str                  # One of: mean, max, last_token, rolling_mean, softmax, attention
    layer: int                         # Which transformer layer to extract from
    hidden_dim: int | None = None      # Model's hidden dimension (auto-populated at runtime)
    # Architecture-specific hyperparams
    window_size: int = 16              # For rolling_mean
    temperature: float = 1.0           # For softmax
    attn_dim: int = 64                 # For attention
    # Training hyperparams (per-probe)
    learning_rate: float = 1e-3
    weight_decay: float = 0.0

    @pydantic.model_validator(mode="after")
    def _validate_hidden_dim_set_before_build(self) -> "ProbeConfig":
        """hidden_dim can be None at config time but must be set before build_probe()."""
        return self


class BaseProbe(torch.nn.Module, abc.ABC):
    """Abstract base class for all probes."""

    def __init__(self, config: ProbeConfig):
        super().__init__()
        assert config.hidden_dim is not None, "hidden_dim must be set before constructing a probe"
        self.config = config

    @abc.abstractmethod
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim) - detached activations from LLM
            attention_mask: (batch, seq_len) - 1 for real tokens, 0 for padding
        Returns:
            logits: (batch, 1)
        """
        ...
```

**Key change from v1:** `hidden_dim` is `Optional[None]` at config time, set programmatically from `model.config.hidden_size` before probe construction. This avoids requiring users to know the model's hidden size when writing configs.

### 4.2 Individual Probe Modules

Each file (`mean_probe.py`, `max_probe.py`, etc.) implements a single `nn.Module` subclass of `BaseProbe`. They are lightweight: only a linear layer (or linear + attention params for the attention probe).

### 4.3 `collection.py` - ProbeCollection

A wrapper `nn.Module` that holds all probes as named submodules. This is the unit that gets wrapped by DDP via `accelerator.prepare()`, ensuring all probe parameters participate in gradient synchronization.

```python
class ProbeCollection(torch.nn.Module):
    """Holds N probes as submodules. DDP-friendly: all parameters are visible."""

    def __init__(self, probe_configs: list[ProbeConfig], hidden_dim: int):
        super().__init__()
        self._probe_configs = {}
        probes = {}
        for pc in probe_configs:
            pc = pc.model_copy(update={"hidden_dim": hidden_dim})
            probes[pc.name] = build_probe(pc)
            self._probe_configs[pc.name] = pc
        self.probes = torch.nn.ModuleDict(probes)

    def forward(
        self, activations: dict[int, torch.Tensor], attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            activations: {layer_idx: (batch, seq_len, hidden_dim)} from ActivationExtractor
            attention_mask: (batch, seq_len)
        Returns:
            {probe_name: (batch, 1) logits}
        """
        results = {}
        for name, probe in self.probes.items():
            h = activations[probe.config.layer]
            results[name] = probe(h, attention_mask)
        return results

    def get_parameter_groups(self) -> list[dict]:
        """Per-probe parameter groups with individual learning rates for the optimizer."""
        groups = []
        for name, probe in self.probes.items():
            pc = self._probe_configs[name]
            params = list(probe.parameters())
            assert len(params) > 0, f"Probe '{name}' has no parameters"
            groups.append({
                "params": params,
                "lr": pc.learning_rate,
                "weight_decay": pc.weight_decay,
            })
        return groups
```

**Why ProbeCollection?** DDP wraps a single `nn.Module` and synchronizes gradients for all its parameters on `backward()`. By grouping all probes into one `nn.Module`, we get a single DDP wrapper with a single all-reduce call per step (efficient), while maintaining per-probe learning rates via optimizer parameter groups.

Since each probe's parameters are disjoint, the gradient of `sum(all_probe_losses)` with respect to `probe_i`'s parameters equals the gradient of `loss_i` alone. No cross-contamination occurs.

### 4.4 `__init__.py` - Registry

```python
from pyine.probes.base import BaseProbe, ProbeConfig
from pyine.probes.collection import ProbeCollection
from pyine.probes.mean_probe import MeanProbe
from pyine.probes.max_probe import MaxProbe
from pyine.probes.last_token_probe import LastTokenProbe
from pyine.probes.rolling_mean_probe import RollingMeanProbe
from pyine.probes.softmax_probe import SoftmaxProbe
from pyine.probes.attention_probe import AttentionProbe

PROBE_REGISTRY: dict[str, type[BaseProbe]] = {
    "mean": MeanProbe,
    "max": MaxProbe,
    "last_token": LastTokenProbe,
    "rolling_mean": RollingMeanProbe,
    "softmax": SoftmaxProbe,
    "attention": AttentionProbe,
}

def build_probe(config: ProbeConfig) -> BaseProbe:
    """Instantiate a probe from its config."""
    if config.hidden_dim is None:
        raise ValueError(f"hidden_dim must be set before building probe '{config.name}'")
    cls = PROBE_REGISTRY[config.architecture]
    return cls(config)
```

______________________________________________________________________

## 5. Activation Extraction

### 5.1 Strategy: Hook-Based Extraction from Frozen Model

The LLM is loaded once per GPU, fully frozen (`model.eval()` + `requires_grad_(False)`). We register **forward hooks** on target layers to capture activations during a single forward pass.

```python
class ActivationExtractor:
    """Registers hooks on specified layers and captures hidden states."""

    def __init__(
        self,
        model: PreTrainedModel,
        target_layers: list[int],
        activation_dtype: torch.dtype | None = None,
    ):
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._activations: dict[int, torch.Tensor] = {}
        self._activation_dtype = activation_dtype

        num_layers = model.config.num_hidden_layers
        for layer_idx in target_layers:
            if layer_idx < 0 or layer_idx >= num_layers:
                raise ValueError(
                    f"Layer {layer_idx} out of range [0, {num_layers})"
                )
            layer_module = self._resolve_layer(model, layer_idx)
            hook = layer_module.register_forward_hook(self._make_hook(layer_idx))
            self._hooks.append(hook)

    @staticmethod
    def _resolve_layer(model: PreTrainedModel, layer_idx: int) -> torch.nn.Module:
        """Resolve transformer block by index. Supports Llama/Qwen/Mistral-family models.

        Falls back to get_submodule() if the standard path doesn't work.
        """
        # Standard path for Llama/Qwen/Mistral: model.model.layers[i]
        try:
            return model.model.layers[layer_idx]
        except (AttributeError, IndexError):
            pass
        # Fallback: try common alternatives
        for path_template in [
            f"transformer.h.{layer_idx}",
            f"model.layers.{layer_idx}",
            f"gpt_neox.layers.{layer_idx}",
        ]:
            try:
                return model.get_submodule(path_template)
            except (AttributeError, KeyError):
                continue
        raise ValueError(
            f"Cannot resolve transformer layer {layer_idx} for model type "
            f"{type(model).__name__}. Add its layer path to _resolve_layer() fallbacks."
        )

    @staticmethod
    def _normalize_layer_output(output, layer_idx: int) -> torch.Tensor:
        """Extract hidden states from a transformer block's output.

        Handles raw tensors, tuples, and BaseModelOutput-like objects.
        Validates the result is a 3D tensor (batch, seq_len, hidden_dim).
        """
        if isinstance(output, torch.Tensor):
            h = output
        elif isinstance(output, tuple):
            h = output[0]
        elif hasattr(output, "last_hidden_state"):
            h = output.last_hidden_state
        elif hasattr(output, "__getitem__"):
            h = output[0]  # OrderedDict-like (BaseModelOutput)
        else:
            raise TypeError(
                f"Unexpected output type {type(output)} from layer {layer_idx}"
            )
        if h.ndim != 3:
            raise ValueError(
                f"Expected 3D activation (batch, seq_len, hidden_dim) from layer "
                f"{layer_idx}, got shape {h.shape}"
            )
        return h

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            h = self._normalize_layer_output(output, layer_idx).detach()
            if self._activation_dtype is not None:
                h = h.to(self._activation_dtype)
            self._activations[layer_idx] = h
        return hook_fn

    def get_activations(self) -> dict[int, torch.Tensor]:
        """Returns captured activations and clears the cache."""
        result = dict(self._activations)
        self._activations.clear()
        return result

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
```

### 5.2 Layer Resolution

The hook targets the **transformer block module** (e.g., `model.model.layers[i]`), which means it captures the block's final output — after both self-attention and MLP sublayers. This is the standard "layer output" used in probing literature.

The primary path targets `model.model.layers[i]` (standard for Llama/Qwen/Mistral-family models). A fallback chain tries alternative paths (`transformer.h`, `gpt_neox.layers`, etc.) for broader compatibility. If all fallbacks fail, the error message directs the user to extend the fallback chain.

### 5.3 Output Normalization

The `_normalize_layer_output()` helper handles the variety of return types from transformer blocks: raw tensors, tuples `(hidden_states, ...)`, and `BaseModelOutput`-like objects. It also validates the result is 3D `(batch, seq_len, hidden_dim)`, failing fast with a clear error if the model returns something unexpected.

### 5.4 Dtype Handling

The optional `activation_dtype` parameter allows casting activations to a specific dtype immediately in the hook, before they're stored. This ensures probes and activations always match, avoiding silent upcasting. If `None`, activations keep whatever dtype the LLM produces (typically bf16 or fp16).

### 5.5 Memory Considerations

- Activations for a single layer at `(batch, seq_len, hidden_dim)` in bf16: `batch * seq_len * hidden_dim * 2` bytes
- For Qwen2.5-7B (`hidden_dim=3584`), `seq_len=3000`, `batch=4`: ~82 MB per layer
- With 3 hooked layers: ~246 MB per step (well within GPU memory)
- Activations are detached immediately (no grad through the LLM) and cleared after each step

______________________________________________________________________

## 6. Data Pipeline

### 6.1 Input Format

The app expects a **pre-processed HuggingFace dataset** (local path or Hub name) with:

- **Text field** (`text_field` config): Either a plain `str` column or a `messages` column in chat-template format (list of `{"role": ..., "content": ...}` dicts)
- **Label field** (`label_field` config): A binary integer column (`0` or `1`)
- **Split names**: The dataset must have `train` and `valid` splits (or as configured)

For v0, dataset preparation (deriving labels, combining subsets, etc.) happens externally. A future iteration will integrate with the existing datamodule system.

### 6.2 Dataset Validation

At load time, before tokenization, validate the dataset:

```python
def validate_probe_dataset(dataset, text_field, label_field):
    """Fail fast on malformed datasets."""
    if text_field not in dataset.column_names:
        raise ValueError(f"text_field '{text_field}' not found. Columns: {dataset.column_names}")
    if label_field not in dataset.column_names:
        raise ValueError(f"label_field '{label_field}' not found. Columns: {dataset.column_names}")

    # Validate labels are binary
    unique_labels = set(dataset.unique(label_field))
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"Expected binary labels {{0, 1}}, got {unique_labels}")
    if len(unique_labels) < 2:
        logger.warning(f"Train split has only label(s) {unique_labels} — probe training may be degenerate")
```

### 6.3 Tokenization

Tokenization happens as a `.map()` over the HuggingFace dataset:

```python
def tokenize_for_probes(examples, tokenizer, max_seq_length, text_field):
    if text_field == "messages":
        # Guard: ensure tokenizer has a chat template
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
```

### 6.4 DataLoader & Collation

Use `DataCollatorWithPadding` from transformers for dynamic padding. The `labels` column is preserved through collation.

In DDP mode, `accelerator.prepare(dataloader)` automatically wraps with `DistributedSampler` to shard data across GPUs.

```python
def build_dataloader(dataset, tokenizer, batch_size, num_workers, shuffle):
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )
```

______________________________________________________________________

## 7. Multi-GPU DDP Strategy

### 7.1 Approach: Accelerate DDP with Full Model Replication

Each GPU holds:

- A **full copy** of the frozen LLM (no sharding — not training it, so DeepSpeed/FSDP add no benefit)
- A **full copy** of the ProbeCollection (tiny — a few thousand parameters total)

Data is sharded across GPUs via `DistributedSampler`. Each GPU processes its data shard, runs the LLM forward pass, extracts activations, and trains its copy of the probes. `accelerate` handles gradient synchronization for the probes via DDP.

### 7.2 Why Not DeepSpeed/FSDP for the Frozen LLM?

DeepSpeed ZeRO and FSDP shard model parameters/gradients across GPUs to save memory. Since the LLM is frozen (no gradients, no optimizer states), sharding provides no benefit. Full replication is simpler and avoids communication overhead during the forward pass.

If the model doesn't fit on a single GPU, tensor parallelism would be needed (future extension, out of scope for v0). The typical use case is 7B models that fit comfortably on a single 80GB GPU.

### 7.3 Accelerate Integration

```python
from accelerate import Accelerator

accelerator = Accelerator(
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    # mixed_precision is inherited from accelerate config file
)

# 1. Load frozen LLM on local GPU (NOT prepared with accelerate)
#    get_device_map() returns {"": f"cuda:{local_rank}"} in distributed mode
model = config.get_model(checkpoint_path=config.llm_checkpoint_path)
model.eval()
model.requires_grad_(False)

# 2. Build ProbeCollection (cast to target dtype to match activations)
probe_collection = ProbeCollection(config.probe_configs, hidden_dim=model.config.hidden_size)
probe_collection = probe_collection.to(dtype=config.target_dtype)
optimizer = torch.optim.AdamW(probe_collection.get_parameter_groups())

# 3. Prepare with accelerate (handles DDP wrapping + DistributedSampler)
probe_collection, optimizer, train_loader, valid_loader = accelerator.prepare(
    probe_collection, optimizer, train_loader, valid_loader,
)

# 4. Hook the frozen LLM (hooks stay on raw model, unaffected by DDP)
target_layers = sorted({pc.layer for pc in config.probe_configs})
extractor = ActivationExtractor(model, target_layers, activation_dtype=config.target_dtype)
```

### 7.4 Launch Patterns

Single-node DDP (reuse existing infrastructure):

```bash
# Via torchrun (aligns with existing run_ddp.sh)
bash scripts/run_ddp.sh -- +experiment=probes/v0_probe

# Via accelerate
accelerate launch --config_file pyine/configs/accelerate/multi_gpu.yaml \
    pyine/apps/trainers/probe_trainer.py +experiment=probes/v0_probe
```

Single GPU (no special setup):

```bash
python -m pyine.apps.trainers.probe_trainer +experiment=probes/v0_probe
```

______________________________________________________________________

## 8. Training Loop Design

### 8.1 Overview

The training loop is a **custom PyTorch loop** (not HF `Trainer`) because we need fine-grained control over:

1. Running a single LLM forward pass and distributing activations to N probes
2. Using a single optimizer with per-probe parameter groups
3. Logging per-probe metrics independently
4. Integrating with accelerate for DDP without Trainer abstractions

### 8.2 Core Training Function

```python
def probe_train(
    config: ProbeTrainerAppMainConfig,
    runtime: RuntimeConfig | None,
) -> dict[str, BaseProbe]:
    """Core probe training loop. Returns trained probes."""
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    # --- 1. Load frozen LLM ---
    model = config.get_model(checkpoint_path=config.llm_checkpoint_path)
    model.eval()
    model.requires_grad_(False)
    tokenizer = config.get_tokenizer(checkpoint_path=config.llm_checkpoint_path)

    # --- 2. Build ProbeCollection + optimizer ---
    hidden_dim = model.config.hidden_size
    probe_collection = ProbeCollection(config.probe_configs, hidden_dim)
    probe_collection = probe_collection.to(dtype=config.target_dtype)  # match activation dtype
    optimizer = torch.optim.AdamW(probe_collection.get_parameter_groups())

    # --- 3. Prepare datasets ---
    train_ds = load_and_tokenize(config.dataset_path, "train", tokenizer, config)
    valid_ds = load_and_tokenize(config.dataset_path, "valid", tokenizer, config)
    train_loader = build_dataloader(train_ds, tokenizer, config.train_batch_size,
                                     config.dataloader_num_workers, shuffle=True)
    valid_loader = build_dataloader(valid_ds, tokenizer, config.eval_batch_size,
                                     config.dataloader_num_workers, shuffle=False)

    # --- 4. Prepare with accelerate ---
    probe_collection, optimizer, train_loader, valid_loader = accelerator.prepare(
        probe_collection, optimizer, train_loader, valid_loader,
    )

    # --- 5. Register activation hooks ---
    target_layers = sorted({pc.layer for pc in config.probe_configs})
    extractor = ActivationExtractor(model, target_layers, activation_dtype=config.target_dtype)

    # --- 6. Training loop ---
    loss_fn = torch.nn.BCEWithLogitsLoss()
    global_step = 0

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
                per_probe_losses = {}
                total_loss = torch.tensor(0.0, device=accelerator.device)
                for name, logits in probe_logits.items():
                    loss = loss_fn(logits.squeeze(-1), labels.float())
                    per_probe_losses[name] = loss
                    total_loss = total_loss + loss

                accelerator.backward(total_loss)
                optimizer.step()
                optimizer.zero_grad()

            # Logging + eval gated on actual optimizer steps (not accumulation micro-steps)
            if accelerator.sync_gradients:
                global_step += 1
                if global_step % config.logging_steps == 0:
                    log_train_metrics(per_probe_losses, global_step, epoch, runtime)

                # Mid-epoch validation
                if config.eval_steps > 0 and global_step % config.eval_steps == 0:
                    validate_probes(
                        probe_collection, model, extractor, valid_loader,
                        loss_fn, global_step, accelerator, runtime,
                    )
                    probe_collection.train()

        # End-of-epoch validation
        validate_probes(
            probe_collection, model, extractor, valid_loader,
            loss_fn, global_step, accelerator, runtime,
        )

    # --- 7. Save probes ---
    if config.save_probes:
        save_probe_checkpoints(probe_collection, config, runtime, accelerator)

    # --- 8. Cleanup ---
    extractor.remove_hooks()

    return probe_collection
```

### 8.3 Validation

```python
def validate_probes(
    probe_collection, model, extractor, valid_loader,
    loss_fn, global_step, accelerator, runtime,
):
    """Run validation and compute loss + AUROC per probe."""
    probe_collection.eval()

    all_logits: dict[str, list[torch.Tensor]] = {name: [] for name in probe_collection.probes}
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

    # Gather predictions across all GPUs
    for name in all_logits:
        all_logits[name] = torch.cat(all_logits[name])
        all_logits[name] = accelerator.gather_for_metrics(all_logits[name])
    all_labels_cat = accelerator.gather_for_metrics(torch.cat(all_labels))

    # Compute metrics on main process only
    if accelerator.is_main_process:
        for name in probe_collection.probes:
            logits_np = all_logits[name].cpu()
            labels_np = all_labels_cat.cpu()

            val_loss = loss_fn(logits_np, labels_np.float()).item()
            probs = torch.sigmoid(logits_np).numpy()
            labels_np = labels_np.numpy()

            # Guard against single-class labels (AUROC undefined)
            unique_labels = set(labels_np.tolist())
            if len(unique_labels) < 2:
                logger.warning(f"Skipping AUROC for {name}: only labels {unique_labels} present")
                auroc = float("nan")
            else:
                auroc = sklearn.metrics.roc_auc_score(labels_np, probs)

            if runtime and runtime.wandb_run:
                runtime.wandb_run.log({
                    f"valid/{name}/loss": val_loss,
                    f"valid/{name}/auroc": auroc,
                }, step=global_step)
```

### 8.4 W&B Logging Strategy

Metrics are organized under a per-probe namespace:

| Metric Key                 | Phase | Description                   |
| -------------------------- | ----- | ----------------------------- |
| `train/{probe_name}/loss`  | Train | BCE loss per logging step     |
| `valid/{probe_name}/loss`  | Valid | BCE loss over full valid set  |
| `valid/{probe_name}/auroc` | Valid | AUROC over full valid set     |
| `train/global_step`        | Train | Global optimizer step counter |
| `train/epoch`              | Train | Current epoch                 |

Logging is gated on `runtime.wandb_run is not None` and done only on the main process (`accelerator.is_main_process`). At the end of training, a W&B summary table with per-probe final metrics is logged for easy comparison.

______________________________________________________________________

## 9. Config System Design

### 9.1 `probe_trainer_configs.py` - App Config

```python
class ProbeTrainerAppMainConfig(common.AppMainConfig, common.ModelTokenizerConfigBase):
    """Configuration for probe training on frozen LLM activations."""

    # --- Override: evals not needed for probe training ---
    evals_config: BaseEvalsConfig | None = pydantic.Field(
        default=None,
        description="Not used for probe training. Kept for AppMainConfig compatibility.",
    )

    # --- LLM checkpoint ---
    llm_checkpoint_path: str | None = pydantic.Field(
        default=None,
        description="Path to model checkpoint. If None, uses base_model directly.",
    )

    # --- Probe configurations ---
    probe_configs: list[ProbeConfig] = pydantic.Field(
        ...,  # MANDATORY
        description="List of probe configs, each specifying architecture, layer, and hyperparams.",
    )

    # --- Dataset ---
    dataset_path: str = pydantic.Field(
        ...,  # MANDATORY
        description="HuggingFace dataset path (local dir or Hub name) with train/valid splits.",
    )
    text_field: str = pydantic.Field(
        default="messages",
        description="Column name for text input. 'messages' for chat format, or a plain text column.",
    )
    label_field: str = pydantic.Field(
        default="label",
        description="Column name for binary labels (0/1).",
    )

    # --- Training loop ---
    num_epochs: int = pydantic.Field(default=10)
    train_batch_size: int = pydantic.Field(default=4)
    eval_batch_size: int = pydantic.Field(default=8)
    max_seq_length: int = pydantic.Field(
        default=3000,
        description="Maximum sequence length for tokenization.",
    )
    gradient_accumulation_steps: int = pydantic.Field(
        default=1,
        description="Number of gradient accumulation steps before optimizer update.",
    )
    logging_steps: int = pydantic.Field(default=10)
    eval_steps: int = pydantic.Field(
        default=-1,
        description="Run validation every N optimizer steps. -1 = only at epoch end.",
    )
    dataloader_num_workers: int = pydantic.Field(default=4)

    # --- Output ---
    save_probes: bool = pydantic.Field(
        default=True,
        description="Whether to save trained probe weights at the end of training.",
    )

    @property
    @typing.override
    def target_dtype(self) -> torch.dtype:
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    @pydantic.model_validator(mode="after")
    def _validate_probe_names_unique(self) -> "ProbeTrainerAppMainConfig":
        names = [pc.name for pc in self.probe_configs]
        if len(names) != len(set(names)):
            raise ValueError(f"Probe names must be unique. Got duplicates in: {names}")
        return self
```

### 9.2 Hydra Config Registration

Follows the same pattern as `hf_sft_trainer_configs.py`:

```python
def register_hydra_configs(eval_type: EvalType) -> list[ConfigDescription]:
    pyine.utils.reprod.load_dotenv()

    entrypoint_config = make_config_description(
        async_probe_trainer_main_wrapper,
        name="entrypoint",
        group=None,
        config={
            "populate_full_signature": True,
            "hydra_convert": "object",
            "hydra_defaults": [
                "_self_",
                {"config": "base"},
                {"runtime": "default"},
            ],
        },
    )

    store, base_configs = get_base_store_and_configs("probe_trainer")
    app_configs = _get_app_configs(group="config")
    configs_to_register = [entrypoint_config, *app_configs]

    experiment_configs = SearchPathPlugin.get_external_configs(
        search_path="experiment/probes", package="_global_",
    )
    configs_to_register.extend(experiment_configs)

    for config in configs_to_register:
        store(config.config, name=config.name, group=config.group, package=config.package)
    store.add_to_hydra_store(overwrite_ok=True)

    return [*base_configs, *configs_to_register]
```

### 9.3 Example Experiment YAML

```yaml
# pyine/configs/experiment/probes/v0_probe.yaml
# @package _global_

# Usage:
#   Single GPU:  python -m pyine.apps.trainers.probe_trainer +experiment=probes/v0_probe
#   Multi-GPU:   bash scripts/run_ddp.sh -- +experiment=probes/v0_probe

defaults:
  - override /config: base
  - _self_

runtime:
  exp_name: probes/v0_probe
  run_name: "${runtime.exp_name}"
  tags:
    - "model:${config.base_model}"
    - "probe-training"
  seed: 42
  dry_run: false

config:
  _target_: pyine.apps.trainers.probe_trainer_configs.ProbeTrainerAppMainConfig

  use_wandb_logging: true
  base_model: Qwen/Qwen3-4B-Instruct-2507
  llm_checkpoint_path: /path/to/rl-checkpoint/checkpoint-500

  # Dataset
  dataset_path: /path/to/probe-dataset  # HF dataset with train/valid splits
  text_field: messages
  label_field: label
  max_seq_length: 3000

  # Training loop
  num_epochs: 10
  train_batch_size: 4
  eval_batch_size: 8
  gradient_accumulation_steps: 1
  logging_steps: 10
  eval_steps: 50
  dataloader_num_workers: 4
  save_probes: true

  auto_model_config:
    use_cache: false
    attn_implementation: "flash_attention_2"

  # Probe configurations: sweep architectures x layers
  probe_configs:
    # --- Mean probes across layers ---
    - name: mean_L8
      architecture: mean
      layer: 8
      learning_rate: 1e-3
    - name: mean_L16
      architecture: mean
      layer: 16
      learning_rate: 1e-3
    - name: mean_L24
      architecture: mean
      layer: 24
      learning_rate: 1e-3

    # --- Max probes ---
    - name: max_L8
      architecture: max
      layer: 8
      learning_rate: 1e-3
    - name: max_L16
      architecture: max
      layer: 16
      learning_rate: 1e-3
    - name: max_L24
      architecture: max
      layer: 24
      learning_rate: 1e-3

    # --- Last-token probes ---
    - name: last_L8
      architecture: last_token
      layer: 8
      learning_rate: 1e-3
    - name: last_L16
      architecture: last_token
      layer: 16
      learning_rate: 1e-3
    - name: last_L24
      architecture: last_token
      layer: 24
      learning_rate: 1e-3

    # --- Rolling-mean probe ---
    - name: rolling_L16
      architecture: rolling_mean
      layer: 16
      learning_rate: 1e-3
      window_size: 32

    # --- Softmax probe ---
    - name: softmax_L16
      architecture: softmax
      layer: 16
      learning_rate: 1e-3
      temperature: 0.5

    # --- Attention probe ---
    - name: attn_L16
      architecture: attention
      layer: 16
      learning_rate: 5e-4
      attn_dim: 64
```

______________________________________________________________________

## 10. Probe Checkpoint Saving

### 10.1 Output Structure

Probes are saved under the Hydra output directory:

```
outputs/<exp_name>/<run_name>/
  probes/
    mean_L8/
      probe_state_dict.pt        # torch state_dict
      probe_config.json          # ProbeConfig as JSON
    mean_L16/
      ...
    attn_L16/
      ...
    training_summary.json        # Per-probe final metrics (loss, AUROC)
```

### 10.2 Save Logic

```python
def save_probe_checkpoints(probe_collection, config, runtime, accelerator):
    """Save probe weights + configs. Only on main process."""
    if not accelerator.is_main_process:
        return

    # Unwrap DDP to access raw module
    raw_collection = accelerator.unwrap_model(probe_collection)
    output_dir = Path(runtime.output_dir) / "probes"

    for name, probe in raw_collection.probes.items():
        probe_dir = output_dir / name
        probe_dir.mkdir(parents=True, exist_ok=True)
        torch.save(probe.state_dict(), probe_dir / "probe_state_dict.pt")
        (probe_dir / "probe_config.json").write_text(
            probe.config.model_dump_json(indent=2)
        )
```

______________________________________________________________________

## 11. Entrypoint

### 11.1 Main Function

```python
async def main(
    config: ProbeTrainerAppMainConfig,
    runtime: RuntimeConfig | None = None,
) -> None:
    # 1. Setup (seeds, wandb, logging)
    entrypoint_setup(runtime_config=runtime, main_config=config, use_wandb_logging=config.use_wandb_logging)

    # 2. Train probes
    probe_collection = probe_train(config=config, runtime=runtime)

    # 3. Finalize
    if runtime is not None:
        runtime.finalize()


async def async_probe_trainer_main_wrapper(
    config: ProbeTrainerAppMainConfig,
    runtime: RuntimeConfig | None = None,
) -> None:
    await main(config=config, runtime=runtime)
```

### 11.2 `__main__` Block

```python
if __name__ == "__main__":
    import sys
    # Filter DeepSpeed's --local_rank to avoid Hydra conflict
    sys.argv = [arg for arg in sys.argv if not arg.startswith("--local_rank")]

    pyine.apps.trainers.common.hydra_main(
        eval_type=pyine.evals.common.EvalType.CODE_EXEC,
        hydra_config_registration_fn=probe_trainer_configs.register_hydra_configs,
        async_main_wrapper=async_probe_trainer_main_wrapper,
    )
```

______________________________________________________________________

## 12. Debug / Synthetic Dataset

### 12.1 Purpose

A synthetic dataset factory that produces HF datasets in the exact format the probe trainer expects. This enables full end-to-end testing without a real probe dataset. Lives in `pyine/probes/debug_dataset.py` — part of the source tree (not tests-only) so it can also be used for manual smoke-test runs.

### 12.2 Design

The synthetic dataset generates chat-format messages with a **learnable signal**: label is correlated with a keyword in the message content. This lets integration tests verify that probes actually learn (AUROC > 0.5) rather than just checking the pipeline doesn't crash.

```python
def create_debug_probe_dataset(
    output_path: str | Path | None = None,
    n_train: int = 200,
    n_valid: int = 50,
    seed: int = 42,
) -> datasets.DatasetDict:
    """Generate a synthetic probe dataset with chat-format messages and binary labels.

    Label signal: messages containing a target keyword get label=1, others get label=0.
    Some noise is added (label flips) to make the task non-trivial.

    Args:
        output_path: If provided, save the dataset to disk. Otherwise return in-memory.
        n_train: Number of training samples.
        n_valid: Number of validation samples.
        seed: Random seed for reproducibility.

    Returns:
        DatasetDict with "train" and "valid" splits, each containing:
        - "messages": list[dict] in chat format [{"role": ..., "content": ...}]
        - "label": int (0 or 1)
    """
    ...
```

### 12.3 Sample Format

````python
# label=1 sample (contains signal keyword)
{
    "messages": [
        {"role": "user", "content": "Analyze the following code:\n```python\ndef helper(x):\n    return x * 2\n```"},
        {"role": "assistant", "content": "This function doubles its input using the helper pattern."},
    ],
    "label": 1,
}

# label=0 sample (no signal keyword)
{
    "messages": [
        {"role": "user", "content": "Analyze the following code:\n```python\ndef compute(x):\n    return x + 1\n```"},
        {"role": "assistant", "content": "This function increments its input by one."},
    ],
    "label": 0,
}
````

Sequences are kept short (~50-150 tokens) so tests run fast, even on CPU with a tiny model.

### 12.4 Usage

```python
# In tests (in-memory, no disk)
from pyine.probes.debug_dataset import create_debug_probe_dataset
ds = create_debug_probe_dataset(n_train=100, n_valid=20)

# For manual smoke tests (save to disk, use from CLI)
# python -m pyine.probes.debug_dataset --output /tmp/probe-debug-dataset
# Then:
# python -m pyine.apps.trainers.probe_trainer +experiment=probes/v0_probe config.dataset_path=/tmp/probe-debug-dataset
```

The module includes a `__main__` block for CLI usage, keeping the trainer code completely unaware of the debug dataset.

______________________________________________________________________

## 13. Test Suite

### 13.1 Test Layout

Follows existing repo conventions: mirror the source tree under `tests/`, use `conftest.py` for shared fixtures, `@pytest.mark` for categorization.

```
tests/
  probes/                                    # Probe module tests
    __init__.py
    conftest.py                              # Shared fixtures
    test_probes.py                           # Unit: all 6 probe architectures
    test_collection.py                       # Unit: ProbeCollection
    test_extraction.py                       # Unit: ActivationExtractor
    test_debug_dataset.py                    # Unit: debug dataset factory
  apps/
    trainers/
      test_probe_trainer.py                  # Unit: training loop (mocked LLM)
      test_probe_trainer_configs.py          # Unit: config validation + Hydra
      test_probe_trainer_integration.py      # Integration: end-to-end (GPU)
```

### 13.2 Shared Fixtures (`tests/probes/conftest.py`)

```python
PROBE_HIDDEN_DIM = 64       # Small for fast tests
PROBE_SEQ_LEN = 32
PROBE_BATCH_SIZE = 4

@pytest.fixture
def probe_hidden_dim() -> int:
    return PROBE_HIDDEN_DIM

@pytest.fixture
def random_activations() -> dict[int, torch.Tensor]:
    """Random activations for layers 0, 4, 8 with small dimensions."""
    return {
        layer: torch.randn(PROBE_BATCH_SIZE, PROBE_SEQ_LEN, PROBE_HIDDEN_DIM)
        for layer in [0, 4, 8]
    }

@pytest.fixture
def random_attention_mask() -> torch.Tensor:
    """Attention mask with some padding (last few positions masked out)."""
    mask = torch.ones(PROBE_BATCH_SIZE, PROBE_SEQ_LEN, dtype=torch.long)
    # Mask out last 4 positions for first 2 samples
    mask[:2, -4:] = 0
    return mask

@pytest.fixture
def sample_probe_configs() -> list[ProbeConfig]:
    """Minimal probe configs covering all 6 architectures."""
    return [
        ProbeConfig(name="mean_L0", architecture="mean", layer=0),
        ProbeConfig(name="max_L0", architecture="max", layer=0),
        ProbeConfig(name="last_L4", architecture="last_token", layer=4),
        ProbeConfig(name="rolling_L4", architecture="rolling_mean", layer=4, window_size=4),
        ProbeConfig(name="softmax_L8", architecture="softmax", layer=8, temperature=0.5),
        ProbeConfig(name="attn_L8", architecture="attention", layer=8, attn_dim=16),
    ]

@pytest.fixture
def debug_dataset() -> datasets.DatasetDict:
    """In-memory synthetic dataset for probe training tests."""
    return create_debug_probe_dataset(n_train=40, n_valid=10, seed=42)
```

### 13.3 Test Plan: `test_probes.py` — Probe Architecture Unit Tests

Tests each of the 6 probe architectures in isolation. All tests run on CPU with small tensors, no GPU needed.

```python
@pytest.mark.parametrize(
    ("architecture", "extra_kwargs"),
    [
        pytest.param("mean", {}, id="mean"),
        pytest.param("max", {}, id="max"),
        pytest.param("last_token", {}, id="last_token"),
        pytest.param("rolling_mean", {"window_size": 4}, id="rolling_mean"),
        pytest.param("softmax", {"temperature": 0.5}, id="softmax"),
        pytest.param("attention", {"attn_dim": 16}, id="attention"),
    ],
)
class TestProbeArchitectures:
    """Unit tests for individual probe modules."""

    def test_output_shape(self, architecture, extra_kwargs, random_activations, random_attention_mask):
        """Probe forward returns (batch, 1) logits."""

    def test_gradient_flow(self, architecture, extra_kwargs, random_activations, random_attention_mask):
        """Loss.backward() produces non-None gradients for all probe parameters."""

    def test_masked_positions_ignored(self, architecture, extra_kwargs, probe_hidden_dim):
        """Changing masked positions does not affect output."""

    def test_parameter_count(self, architecture, extra_kwargs, probe_hidden_dim):
        """Probe has expected number of parameters (linear head + arch-specific)."""


class TestBuildProbe:
    """Tests for the build_probe factory and registry."""

    def test_build_probe_all_architectures(self):
        """build_probe returns correct subclass for each registered architecture."""

    def test_build_probe_unknown_architecture_raises(self):
        """build_probe raises KeyError for unregistered architecture name."""

    def test_build_probe_missing_hidden_dim_raises(self):
        """build_probe raises ValueError if hidden_dim is None."""
```

### 13.4 Test Plan: `test_collection.py` — ProbeCollection Tests

```python
class TestProbeCollection:
    """Tests for the ProbeCollection wrapper module."""

    def test_forward_returns_all_probe_logits(
        self, sample_probe_configs, random_activations, random_attention_mask,
    ):
        """Forward returns dict with one (batch, 1) tensor per probe."""

    def test_parameter_groups_match_configs(self, sample_probe_configs):
        """get_parameter_groups() returns one group per probe with correct lr/wd."""

    def test_parameter_groups_nonempty(self, sample_probe_configs):
        """Each parameter group has at least one parameter."""

    def test_hidden_dim_auto_populated(self, sample_probe_configs):
        """Probes receive hidden_dim from ProbeCollection constructor, not from config."""

    def test_summed_loss_gradient_independence(
        self, sample_probe_configs, random_activations, random_attention_mask,
    ):
        """Gradient of sum(losses) w.r.t. probe_i params equals gradient of loss_i alone."""

    def test_all_submodules_visible_to_ddp(self, sample_probe_configs):
        """All probe parameters are in collection.parameters() (needed for DDP)."""
```

### 13.5 Test Plan: `test_extraction.py` — ActivationExtractor Tests

Uses a small mock transformer model to test hook-based extraction without loading a real LLM.

```python
class MockTransformerBlock(torch.nn.Module):
    """Minimal transformer block that returns a fixed-shape tensor."""
    ...

class MockModel:
    """Mock model with model.model.layers[i] and model.config.num_hidden_layers."""
    ...

class TestActivationExtractor:
    """Tests for hook-based activation extraction."""

    def test_captures_specified_layers(self):
        """Activations are captured only for requested layer indices."""

    def test_get_activations_clears_cache(self):
        """After get_activations(), internal cache is empty."""

    def test_activation_shape(self):
        """Captured activations have shape (batch, seq_len, hidden_dim)."""

    def test_activation_dtype_casting(self):
        """activation_dtype casts activations to the specified dtype."""

    def test_layer_out_of_range_raises(self):
        """Requesting a layer >= num_hidden_layers raises ValueError."""

    def test_normalize_layer_output_tuple(self):
        """_normalize_layer_output handles tuple outputs."""

    def test_normalize_layer_output_base_model_output(self):
        """_normalize_layer_output handles BaseModelOutput-like objects."""

    def test_normalize_layer_output_invalid_shape_raises(self):
        """_normalize_layer_output raises on non-3D tensors."""

    def test_remove_hooks_cleanup(self):
        """remove_hooks() detaches all hooks from the model."""

    def test_resolve_layer_fallback(self):
        """_resolve_layer tries alternative paths when model.model.layers fails."""
```

### 13.6 Test Plan: `test_debug_dataset.py`

```python
class TestDebugDataset:
    """Tests for the synthetic debug dataset factory."""

    def test_dataset_has_required_splits(self):
        """Output has 'train' and 'valid' splits."""

    def test_dataset_has_required_columns(self):
        """Each split has 'messages' and 'label' columns."""

    def test_labels_are_binary(self):
        """All labels are 0 or 1."""

    def test_both_labels_present_in_train(self):
        """Train split contains both label=0 and label=1 samples."""

    def test_messages_are_chat_format(self):
        """Each messages entry is a list of dicts with 'role' and 'content' keys."""

    def test_sample_counts(self):
        """n_train and n_valid control split sizes."""

    def test_deterministic_with_seed(self):
        """Same seed produces identical datasets."""

    def test_save_to_disk_and_reload(self, tmp_path):
        """Dataset saved to disk can be reloaded with datasets.load_from_disk()."""
```

### 13.7 Test Plan: `test_probe_trainer_configs.py` — Config Tests

```python
class TestProbeTrainerAppMainConfig:
    """Tests for ProbeTrainerAppMainConfig validation."""

    def test_duplicate_probe_names_raises(self):
        """Config rejects probe_configs with duplicate names."""

    def test_valid_config_construction(self):
        """Config with all required fields validates successfully."""

    def test_target_dtype_property(self):
        """target_dtype returns bf16 or fp16 based on hardware."""

    def test_evals_config_optional(self):
        """evals_config defaults to None without error."""


class TestHydraConfigRegistration:
    """Tests for Hydra-zen config registration."""

    def test_register_hydra_configs_no_errors(self):
        """register_hydra_configs() completes without exceptions."""

    def test_experiment_config_instantiates(self):
        """v0_probe.yaml experiment config can be loaded and instantiated by Hydra."""
```

### 13.8 Test Plan: `test_probe_trainer.py` — Training Loop Unit Tests

Tests the training logic with a **mocked LLM** — a tiny `nn.Module` that mimics the interface of a real model (has `.config.hidden_size`, `.config.num_hidden_layers`, and `model.model.layers`). Runs on CPU, fast.

```python
class SmallMockLLM(torch.nn.Module):
    """Tiny model with 2 transformer blocks, hidden_dim=64.
    Mimics the interface needed by ActivationExtractor and probe_train().
    """
    ...

class TestProbeTrainUnit:
    """Unit tests for the probe_train() function with mocked LLM."""

    def test_train_step_reduces_loss(self):
        """After a few steps, train loss is lower than initial loss."""

    def test_validation_produces_metrics(self):
        """validate_probes() returns loss and AUROC per probe."""

    def test_auroc_guard_single_class(self):
        """validate_probes() returns NaN AUROC for single-class valid set."""

    def test_save_probe_checkpoints(self, tmp_path):
        """save_probe_checkpoints() writes state_dict + config JSON per probe."""

    def test_probe_checkpoints_loadable(self, tmp_path):
        """Saved probe can be reconstructed from config JSON + state_dict."""

    def test_dataset_validation_catches_bad_labels(self):
        """validate_probe_dataset() raises on non-binary labels."""

    def test_dataset_validation_catches_missing_columns(self):
        """validate_probe_dataset() raises on missing text/label columns."""

    def test_tokenization_chat_template_guard(self):
        """tokenize_for_probes() raises if tokenizer has no chat template and text_field='messages'."""
```

### 13.9 Test Plan: `test_probe_trainer_integration.py` — End-to-End Integration

Runs the full pipeline with a real (small) model on GPU. Marked `@pytest.mark.slow` + `@pytest.mark.integration`.

```python
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(
    tests.env_checks.NO_ACCELERATOR_AVAILABLE,
    reason="Accelerator required for probe training integration test",
)
class TestProbeTrainerIntegration:
    """End-to-end integration test with a real small model."""

    def test_probe_training_end_to_end(self, tmp_path):
        """Full pipeline: load model, tokenize debug dataset, train probes, validate, save.

        Uses a small public model (e.g., HuggingFaceTB/SmolLM2-135M-Instruct).
        Verifies:
        - Training completes without error
        - Probe checkpoints are saved to disk
        - Validation AUROC is computed (>= 0.0, not NaN)
        - W&B metrics are logged if configured
        """

    def test_probe_training_dry_run(self, tmp_path):
        """Same as above but with num_epochs=1, 1 probe, to test fast feedback loop."""
```

### 13.10 Testing Strategy Summary

| Category                 | Location                                                | Marker                | GPU needed | Runs in CI  |
| ------------------------ | ------------------------------------------------------- | --------------------- | ---------- | ----------- |
| Probe architectures      | `tests/probes/test_probes.py`                           | (none)                | No         | Yes         |
| ProbeCollection          | `tests/probes/test_collection.py`                       | (none)                | No         | Yes         |
| ActivationExtractor      | `tests/probes/test_extraction.py`                       | (none)                | No         | Yes         |
| Debug dataset            | `tests/probes/test_debug_dataset.py`                    | (none)                | No         | Yes         |
| Config validation        | `tests/apps/trainers/test_probe_trainer_configs.py`     | (none)                | No         | Yes         |
| Training loop (mocked)   | `tests/apps/trainers/test_probe_trainer.py`             | (none)                | No         | Yes         |
| Integration (real model) | `tests/apps/trainers/test_probe_trainer_integration.py` | `slow`, `integration` | Yes        | GPU CI only |

______________________________________________________________________

## 14. Implementation Steps

### Step 1: Create `pyine/probes/` module — base + architectures

Files: `__init__.py`, `base.py`, `mean_probe.py`, `max_probe.py`, `last_token_probe.py`, `rolling_mean_probe.py`, `softmax_probe.py`, `attention_probe.py`

Then: `tests/probes/test_probes.py` — run and pass all probe unit tests.

### Step 2: Create `pyine/probes/collection.py` + `extraction.py`

ProbeCollection and ActivationExtractor.

Then: `tests/probes/test_collection.py`, `tests/probes/test_extraction.py` — run and pass.

### Step 3: Create `pyine/probes/debug_dataset.py`

Synthetic dataset factory with CLI entrypoint.

Then: `tests/probes/test_debug_dataset.py` — run and pass.

### Step 4: Create `pyine/apps/trainers/probe_trainer_configs.py`

Config class + Hydra registration.

Then: `tests/apps/trainers/test_probe_trainer_configs.py` — run and pass.

### Step 5: Create `pyine/apps/trainers/probe_trainer.py`

Training loop, validation, save logic, entrypoint.

Then: `tests/apps/trainers/test_probe_trainer.py` — run and pass.

### Step 6: Create experiment config + integration test

`pyine/configs/experiment/probes/v0_probe.yaml` and `tests/apps/trainers/test_probe_trainer_integration.py`.

Verify end-to-end on GPU with the debug dataset.

### Step 7: Manual smoke test

```bash
# Generate debug dataset
python -m pyine.probes.debug_dataset --output /tmp/probe-debug-dataset

# Single GPU
python -m pyine.apps.trainers.probe_trainer +experiment=probes/v0_probe \
    config.dataset_path=/tmp/probe-debug-dataset

# Multi-GPU (if available)
bash scripts/run_ddp.sh -- +experiment=probes/v0_probe \
    config.dataset_path=/tmp/probe-debug-dataset
```

______________________________________________________________________

## 15. Key Design Decisions

### 15.1 Custom Training Loop vs. HF Trainer

**Decision: Custom loop.**

The HF `Trainer` is designed for a single model with a single loss. We need to:

- Run one LLM forward pass and fan out activations to N probes
- Sum N independent losses for a single backward pass
- Log per-probe metrics independently

The probes are tiny (a few thousand parameters each), so the distributed training machinery of `Trainer` adds no value. A custom loop with `accelerate` gives the same DDP benefits with full control.

### 15.2 Activation Computation: On-the-fly vs. Pre-computed

**Decision: On-the-fly extraction via forward hooks.**

Pre-computing activations for all layers x all samples would require massive disk storage and a separate extraction pass. On-the-fly extraction:

- Is memory-efficient (only keeps current batch's activations)
- Feeds all probes from a single LLM forward pass
- Avoids disk I/O bottlenecks

### 15.3 ProbeCollection + Single Optimizer vs. Per-Probe Optimizers

**Decision: ProbeCollection with a single optimizer (per-probe parameter groups).**

This gives us:

- A single DDP wrapper (one all-reduce call per step, efficient for 12+ probes)
- Per-probe learning rates via optimizer parameter groups
- A single `backward()` call on the summed loss (mathematically equivalent to per-probe backward since parameters are disjoint)
- Clean integration with `accelerator.accumulate()` for gradient accumulation

### 15.4 Multi-GPU: Accelerate DDP with Full Replication

**Decision: Full model replication per GPU, data-parallel via accelerate.**

The frozen LLM is not trained, so sharding (DeepSpeed/FSDP) provides no benefit. Full replication means:

- Each GPU runs an independent LLM forward pass on its data shard
- No cross-GPU communication for the LLM forward pass
- Only probe gradients are synchronized (tiny — microseconds of communication)
- Consistent with existing codebase patterns (accelerate, torchrun, run_ddp.sh)

### 15.5 Loss Function

**Decision: `BCEWithLogitsLoss`.**

Standard, numerically stable binary cross-entropy. The summed loss across probes is used for backward, but individual losses are logged per probe.

### 15.6 LLM Forward Pass: `model.forward()` vs. `model.generate()`

**Decision: `model.forward()`.**

We need hidden-state activations from processing input text. `generate()` is for autoregressive generation. `forward()` runs the full transformer stack once, and hooks capture layer outputs.

______________________________________________________________________

## 16. Codex Review Assessment

### v1 Review (addressed in v2)

| #   | Codex Concern                       | Assessment              | Action                                                                                 |
| --- | ----------------------------------- | ----------------------- | -------------------------------------------------------------------------------------- |
| 1   | Model/layer hook robustness         | **Accepted.**           | Added fallback chain in `_resolve_layer()`.                                            |
| 2   | Activation lifetime & gradient flow | **Partially accepted.** | Added `activation_dtype` parameter. Memory is manageable.                              |
| 3   | Mixed precision / dtype handling    | **Accepted.**           | Added `activation_dtype` + explicit probe dtype cast.                                  |
| 4   | Optimizer & step semantics          | **Accepted.**           | Added `gradient_accumulation_steps`, proper `global_step`, `accelerator.accumulate()`. |
| 5   | Validation cadence                  | **Accepted.**           | Implemented `eval_steps` in-loop.                                                      |
| 6   | AUROC edge cases                    | **Accepted.**           | Added label diversity check.                                                           |
| 7   | Chat template kwargs                | **Deferred.**           | Added `add_generation_prompt=False`. Full kwargs deferred.                             |
| 8   | LoRA / adapter handling             | **Already covered.**    | Reused existing `get_model()` infrastructure.                                          |
| 9   | Saving probe artifacts              | **Accepted.**           | Added Section 10 with output structure.                                                |
| 10  | `hidden_dim` naming/stability       | **Accepted.**           | Changed to optional, runtime-validated.                                                |

### v2 Review (addressed in v2.1)

| #   | Codex Concern                  | Assessment                                                                      | Action                                                                                                              |
| --- | ------------------------------ | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| 1   | Layer output shape assumptions | **Accepted.** Models may return `BaseModelOutput` or non-standard tuples.       | Added `_normalize_layer_output()` helper with `BaseModelOutput` handling and 3D shape validation.                   |
| 2   | Probe dtype consistency        | **Accepted.** Probes were not explicitly cast to `target_dtype`.                | Added `probe_collection.to(dtype=config.target_dtype)` before `accelerator.prepare()`.                              |
| 3   | Tokenization edge cases        | **Accepted.** Missing chat template would cause cryptic errors.                 | Added early detection: raise if `text_field="messages"` and tokenizer has no chat template.                         |
| 4   | Dataset / label validation     | **Accepted.** Late errors are hard to debug.                                    | Added `validate_probe_dataset()` at load time: checks column presence, label values `{0,1}`, warns on single-class. |
| 5   | Parameter group guards         | **Accepted.** Lightweight safety net.                                           | Added `assert len(params) > 0` in `get_parameter_groups()`.                                                         |
| 6   | Eval gating on sync_gradients  | **Accepted.** Bug in v2: `eval_steps` check was outside `sync_gradients` block. | Moved inside `if accelerator.sync_gradients` to prevent eval on accumulation micro-steps.                           |

### v1 suggestions NOT adopted (unchanged)

- **`output_hidden_states=True` fallback**: Hooks are more memory-efficient (only requested layers). `output_hidden_states=True` returns ALL layers.
- **`layer_path` config override**: Fallback chain covers common architectures. Extend the chain if needed.
- **`max_layers_in_memory` knob**: 3-5 layers x ~82 MB = ~250-400 MB. Well within limits.

______________________________________________________________________

## 17. Future Extensions (Out of Scope for v0)

- **Datamodule integration**: Support loading data directly from `ShortcutBiasDataModule` with label derivation from sample metadata
- **Multi-class probes**: Extend beyond binary classification
- **Activation caching**: Pre-compute and store activations to disk for faster iteration on probe hyperparameters
- **Model parallelism**: For models too large for a single GPU (tensor parallelism or pipeline parallelism for the frozen LLM)
- **Learning rate scheduling**: Per-probe LR schedulers (cosine, linear warmup, etc.)
- **Early stopping**: Per-probe early stopping based on validation AUROC plateau
- **Probing during generation**: Extract activations from `model.generate()` rollouts
- **Probe ensembling**: Combine multiple probe outputs for stronger predictions
- **`chat_template_kwargs`**: Expose `add_generation_prompt`, `tools`, etc. for complex chat templates
- **`layer_accessor` config override**: For non-standard model architectures where the `_resolve_layer()` fallback chain is insufficient
