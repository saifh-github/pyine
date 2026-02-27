"""ProbeCollection -- DDP-friendly wrapper for multiple probes."""

from __future__ import annotations

import re
import typing

import torch

if typing.TYPE_CHECKING:
    import pathlib

    import pyine.guardrails.probes.base

# avoid circular import at module level; build_probe is imported lazily.


class ProbeCollection(torch.nn.Module):
    """Holds N probes as submodules via ``nn.ModuleDict``.

    This is the unit wrapped by DDP so that all probe parameters participate
    in gradient synchronisation with a single all-reduce call.
    """

    def __init__(
        self,
        probe_configs: list[pyine.guardrails.probes.base.ProbeConfig],
        hidden_dim: int,
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  # nn.Module stub
        import pyine.guardrails.probes

        self._probe_configs: dict[str, pyine.guardrails.probes.base.ProbeConfig] = {}
        probes: dict[str, torch.nn.Module] = {}

        # save RNG state so replica seeding doesn't affect downstream randomness
        any_seeded = any(probe_config.replica_seed is not None for probe_config in probe_configs)
        saved_rng_state = torch.random.get_rng_state() if any_seeded else None

        for probe_config in probe_configs:
            probe_config = probe_config.model_copy(update={"hidden_dim": hidden_dim})
            if probe_config.replica_seed is not None:
                torch.manual_seed(probe_config.replica_seed)  # pyright: ignore[reportUnknownMemberType]  # torch stubs
            probes[probe_config.name] = pyine.guardrails.probes.build_probe(probe_config)
            self._probe_configs[probe_config.name] = probe_config

        # restore RNG state after seeded construction
        if saved_rng_state is not None:
            torch.random.set_rng_state(saved_rng_state)

        self.probes = torch.nn.ModuleDict(probes)

    def forward(
        self,
        activations: dict[int, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run all probes on their respective layer activations.

        Args:
            activations: ``{layer_idx: (batch, seq_len, hidden_dim)}``
            attention_mask: ``(batch, seq_len)``

        Returns:
            ``{probe_name: (batch, 1) logits}``
        """
        results: dict[str, torch.Tensor] = {}
        for name, module in self.probes.items():
            probe = typing.cast("pyine.guardrails.probes.base.BaseProbe", module)
            hidden_states = activations[probe.config.layer]
            results[name] = probe(hidden_states, attention_mask)
        return results

    @staticmethod
    def _resolve_checkpoint_name(checkpoint_dir: pathlib.Path) -> str:
        """Auto-detect the latest checkpoint in *checkpoint_dir*.

        Resolution order:
        1. ``final/`` if it contains valid checkpoint files.
        2. The ``step-NNNN`` subdirectory with the highest step number.

        Raises:
            FileNotFoundError: If no valid checkpoint subdirectory is found.
        """
        step_re = re.compile(r"^step-(\d+)$")

        def _is_valid_ckpt(d: pathlib.Path) -> bool:
            return (d / "probe_config.json").exists() and (d / "probe_state_dict.pt").exists()

        final_dir = checkpoint_dir / "final"
        if final_dir.is_dir() and _is_valid_ckpt(final_dir):
            return "final"

        best_step = -1
        best_name: str | None = None
        for entry in checkpoint_dir.iterdir():
            if not entry.is_dir():
                continue
            m = step_re.match(entry.name)
            if m and _is_valid_ckpt(entry):
                step = int(m.group(1))
                if step > best_step:
                    best_step = step
                    best_name = entry.name

        if best_name is not None:
            return best_name

        raise FileNotFoundError(f"No valid checkpoint found in {checkpoint_dir}")

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_dir: pathlib.Path,
        hidden_dim: int,
        *,
        checkpoint_name: str | None = None,
    ) -> ProbeCollection:
        """Load probe checkpoints from a multi-probe directory.

        Expected directory structure::

            checkpoint_dir/
              <probe_name>/
                <checkpoint_name>/
                  probe_config.json
                  probe_state_dict.pt

        Args:
            checkpoint_dir: Path to the probes base directory containing
                one subdirectory per probe, each with checkpoint
                subdirectories (e.g., ``final/``, ``step-0350/``).
            hidden_dim: Hidden dimension of the LLM (must match checkpoint configs).
            checkpoint_name: Which checkpoint to load
                (e.g., ``"final"``, ``"step-0050"``). When ``None`` (default),
                auto-detects per probe: prefers ``final/`` if present,
                otherwise picks the highest ``step-*`` checkpoint.

        Returns:
            ProbeCollection with loaded weights in eval mode.

        Raises:
            FileNotFoundError: If ``checkpoint_dir`` doesn't exist or no
                valid probes are found.
        """
        import pyine.guardrails.probes.base as probe_base

        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

        configs: list[pyine.guardrails.probes.base.ProbeConfig] = []
        state_dicts: dict[str, dict[str, typing.Any]] = {}

        for probe_dir in sorted(checkpoint_dir.iterdir()):
            if not probe_dir.is_dir():
                continue

            try:
                resolved = checkpoint_name if checkpoint_name is not None else cls._resolve_checkpoint_name(probe_dir)
            except FileNotFoundError:
                continue

            ckpt_dir = probe_dir / resolved
            config_path = ckpt_dir / "probe_config.json"
            weights_path = ckpt_dir / "probe_state_dict.pt"

            if not config_path.exists() or not weights_path.exists():
                continue

            config = probe_base.ProbeConfig.model_validate_json(config_path.read_text())
            state_dicts[config.name] = torch.load(
                weights_path,
                map_location="cpu",
                weights_only=True,
            )
            configs.append(config)

        if not configs:
            raise FileNotFoundError(f"No valid probes found in {checkpoint_dir}")

        collection = cls(configs, hidden_dim)
        for name, sd in state_dicts.items():
            collection.probes[name].load_state_dict(sd)
        collection.eval()
        return collection

    def get_parameter_groups(self) -> list[dict[str, list[torch.nn.Parameter] | float]]:
        """Per-probe parameter groups with individual learning rates."""
        groups: list[dict[str, list[torch.nn.Parameter] | float]] = []
        for name, probe in self.probes.items():
            probe_config = self._probe_configs[name]
            params = list(probe.parameters())
            assert len(params) > 0, f"Probe '{name}' has no parameters"
            groups.append(
                {
                    "params": params,
                    "lr": probe_config.learning_rate,
                    "weight_decay": probe_config.weight_decay,
                }
            )
        return groups
