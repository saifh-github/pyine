"""ActivationExtractor — hook-based hidden-state extraction from frozen LLMs."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


class ActivationExtractor:
    """Registers forward hooks on specified transformer layers and captures hidden states."""

    def __init__(
        self,
        model: torch.nn.Module,
        target_layers: list[int],
        activation_dtype: torch.dtype | None = None,
    ) -> None:
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._activations: dict[int, torch.Tensor] = {}
        self._activation_dtype = activation_dtype

        num_layers = model.config.num_hidden_layers
        for layer_idx in target_layers:
            if layer_idx < 0 or layer_idx >= num_layers:
                raise ValueError(f"Layer {layer_idx} out of range [0, {num_layers})")
            layer_module = self._resolve_layer(model, layer_idx)
            hook = layer_module.register_forward_hook(self._make_hook(layer_idx))
            self._hooks.append(hook)

    @staticmethod
    def _resolve_layer(model: torch.nn.Module, layer_idx: int) -> torch.nn.Module:
        """Resolve transformer block by index.

        Primary path: ``model.model.layers[i]`` (Llama/Qwen/Mistral).
        Falls back to common alternatives.
        """
        # Standard: model.model.layers[i]
        try:
            return model.model.layers[layer_idx]
        except (AttributeError, IndexError):
            pass
        # Fallbacks
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
    def _normalize_layer_output(output: object, layer_idx: int) -> torch.Tensor:
        """Extract hidden states from a transformer block's output.

        Handles raw tensors, tuples, and ``BaseModelOutput``-like objects.
        Validates the result is 3D ``(batch, seq_len, hidden_dim)``.
        """
        if isinstance(output, torch.Tensor):
            h = output
        elif isinstance(output, tuple):
            h = output[0]
        elif hasattr(output, "last_hidden_state"):
            h = output.last_hidden_state
        elif hasattr(output, "__getitem__"):
            h = output[0]
        else:
            raise TypeError(f"Unexpected output type {type(output)} from layer {layer_idx}")
        if h.ndim != 3:
            raise ValueError(
                f"Expected 3D activation (batch, seq_len, hidden_dim) from layer {layer_idx}, got shape {h.shape}"
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
        """Return captured activations and clear the internal cache."""
        result = dict(self._activations)
        self._activations.clear()
        return result

    def remove_hooks(self) -> None:
        """Detach all registered hooks from the model."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
