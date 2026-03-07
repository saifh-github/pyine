"""Configuration for the probe training DataModule."""

from __future__ import annotations

import re
import typing

import pydantic

import pyine.configs.schemas
import pyine.configs.utils
import pyine.data.datamodule
import pyine.utils.portability
from pyine.guardrails.data.reward_keys import (  # noqa: TID252 -- avoids circular import via __init__
    HARD_MATCH_KEY,
    SOFT_MATCH_KEY,
)

_RECOMPUTABLE_METRICS = frozenset({SOFT_MATCH_KEY, HARD_MATCH_KEY})

_GROUP_KEY_PATTERN = re.compile(r"^[^:]+:[01]$")


def _get_probe_datamodule_fqn() -> str:
    """Return the fully-qualified name of :class:`ProbeDataModule` for lazy import."""
    return "pyine.guardrails.data.datamodule.ProbeDataModule"


class LabelBalanceConfig(pydantic.BaseModel):
    """Controls label/code-type composition of splits via resampling.

    Provides two mutually exclusive modes:

    - **Simple mode** (``target_positive_ratio``): Controls the fraction of
      label=1 samples across all code types. Useful for basic class balancing.
    - **Group mode** (``group_proportions``): Controls the fraction of each
      ``(code_type, label)`` group independently. Useful for studying probe
      sensitivity to specific failure modes.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    # --- Simple mode: balance by label only ---
    target_positive_ratio: float | None = pydantic.Field(default=None, gt=0.0, lt=1.0)
    """Target fraction of label=1 samples. Mutually exclusive with group_proportions."""

    # --- Group mode: (code_type, label) composition ---
    group_proportions: dict[str, float] | None = None
    """Target proportions for (code_type, label) groups.

    Keys use the format ``"code_type:label"`` (e.g., ``"original:1"``,
    ``"misleading:0"``). Values are relative weights (normalized internally to
    sum to 1.0). Groups not listed in the dict are **excluded** from the
    resulting split. Every listed key **must** have at least one matching
    sample in the data - otherwise a ``ValueError`` is raised at runtime.
    Mutually exclusive with ``target_positive_ratio``.
    """

    # --- Common settings ---
    strategy: typing.Literal["subsample", "oversample"] = "subsample"
    """Resampling strategy.

    - ``"subsample"``: drop excess samples from over-represented groups
      (dataset size decreases or stays the same).
    - ``"oversample"``: duplicate under-represented group samples via
      ``random.Random.choices`` (sampling **with replacement**); dataset
      size increases or stays the same.
    """

    apply_to: tuple[str, ...] = ("train",)
    """Which splits to apply balancing to. Default: train only.

    Must be non-empty and contain only ``"train"`` and/or ``"valid"``.
    """

    @pydantic.model_validator(mode="after")
    def _validate_label_balance(self) -> LabelBalanceConfig:
        # Mutual exclusivity: exactly one of the two modes must be set
        if self.target_positive_ratio is not None and self.group_proportions is not None:
            raise ValueError("target_positive_ratio and group_proportions are mutually exclusive; set only one")
        if self.target_positive_ratio is None and self.group_proportions is None:
            raise ValueError("at least one of target_positive_ratio or group_proportions must be set")

        # Validate group_proportions keys and values
        if self.group_proportions is not None:
            for key, value in self.group_proportions.items():
                if not _GROUP_KEY_PATTERN.match(key):
                    raise ValueError(
                        f"group_proportions key '{key}' does not match the required "
                        f"'{{code_type}}:{{label}}' format (label must be 0 or 1)"
                    )
                if value <= 0:
                    raise ValueError(f"group_proportions values must be positive, got {value} for key '{key}'")

        # Validate apply_to
        if len(self.apply_to) == 0:
            raise ValueError("apply_to must be non-empty")
        allowed_splits = {"train", "valid"}
        for entry in self.apply_to:
            if entry not in allowed_splits:
                raise ValueError(f"apply_to entries must be 'train' or 'valid', got '{entry}'")

        return self


class ProbeDataModuleConfig(pyine.data.datamodule.BaseDataModuleConfig):
    """Configuration for probe training data loaded from LMDB.

    Inherits from :class:`~pyine.data.datamodule.BaseDataModuleConfig` to
    participate in the ``prepare_data()`` → ``setup()`` lifecycle and
    distributed coordination via ``prepare_datamodule()``.

    Probes build :class:`datasets.DatasetDict` objects directly from LMDB
    (not through the ``instantiate_parser()`` → ``torch.utils.data.Dataset``
    pattern), so parser/loader resolution is skipped in
    :meth:`_validate_and_resolve`.
    """

    # Override datamodule_class_path to point to ProbeDataModule by default.
    datamodule_class_path: str = pydantic.Field(default_factory=_get_probe_datamodule_fqn)  # type: ignore[assignment]

    default_dataparser_config: typing.Any = None  # type: ignore[assignment]
    """Not used for probe training. Probes build datasets directly from LMDB."""

    # --- LMDB source ---
    lmdb_path: str
    """Path to LMDB database exported by DiskRewardLogger."""

    label_metric_key: str = SOFT_MATCH_KEY
    """Key in reward_metrics dict for binary label derivation."""

    selection_strategy: typing.Literal["latest", "best_reward"] = "latest"
    """Strategy for deduplicating multiple generations per sample."""

    recompute_labels: bool = False
    """If True, re-compute labels instead of using stored reward_metrics."""

    skip_malformed_records: bool = False
    """If True, skip records missing required fields instead of raising."""

    # --- Split configuration ---
    train_key_prefix: str = "train/"
    """LMDB key prefix for training records."""

    valid_key_prefix: str = "eval/"
    """LMDB key prefix for validation records."""

    use_eval_only_split: bool = False
    """When True, read data from a single LMDB prefix (eval_only_source_prefix)
    and split internally into train/valid."""

    eval_only_source_prefix: str = "eval/"
    """LMDB key prefix to read from when use_eval_only_split=True."""

    train_split_ratio: float = pydantic.Field(default=0.8, gt=0.0, lt=1.0)
    """Fraction of data used for training when use_eval_only_split=True."""

    split_by_family: bool = True
    """When True, split by family (problem) ID so that all code-type variants
    go to the same split. Prevents data leakage."""

    max_samples_per_split: int | None = None
    """Cap samples per split. Useful for debugging or fast iteration."""

    # --- Code type filtering ---
    code_type_filter: list[str] | None = None
    """If set, only include records with code_type matching one of the listed values.
    None (default) includes all records."""

    # --- Label balancing ---
    label_balance: LabelBalanceConfig | None = None
    """If set, resample splits to match target label/code-type proportions.
    None (default) leaves splits as-is."""

    # --- BaseDataModuleConfig overrides ---
    # Probes only have train/valid, no test split.
    subset_names: tuple[str, ...] = ("train", "valid")
    train_subset_names: tuple[str, ...] = ("train",)
    valid_subset_names: tuple[str, ...] = ("valid",)
    eval_subset_names: tuple[str, ...] = ("valid",)

    # --- Validators ---

    @pydantic.model_validator(mode="after")
    def _validate_recompute_label_metric(self) -> ProbeDataModuleConfig:
        if self.recompute_labels and self.label_metric_key not in _RECOMPUTABLE_METRICS:
            raise ValueError(
                f"recompute_labels=True is only supported for label_metric_key in "
                f"{set(_RECOMPUTABLE_METRICS)}, got '{self.label_metric_key}'"
            )
        return self

    @pydantic.model_validator(mode="after")
    def _validate_eval_only_split_config(self) -> ProbeDataModuleConfig:
        if self.use_eval_only_split:
            if not self.eval_only_source_prefix:
                raise ValueError("eval_only_source_prefix must be non-empty when use_eval_only_split=True")
        if self.code_type_filter is not None and len(self.code_type_filter) == 0:
            raise ValueError("code_type_filter must be None (include all) or a non-empty list; got an empty list")
        return self

    @pydantic.model_validator(mode="after")
    @typing.override
    def _validate_and_resolve(self) -> ProbeDataModuleConfig:
        """Override parent to skip parser/loader resolution.

        ProbeDataModule builds datasets directly from LMDB, not through the
        ``instantiate_parser()`` → ``torch.utils.data.Dataset`` pattern.
        Parser/loader resolution is skipped, but ``datamodule_class_path``
        resolution is preserved (needed by ``instantiate_datamodule()``).
        Empty dicts are set for parser/loader configs so that any accidental
        call to ``instantiate_parser()`` fails with a clear KeyError.
        """
        # Resolve datamodule_class_path
        resolved_class = pyine.utils.portability.import_from_dotted_path(self.datamodule_class_path)
        if not isinstance(resolved_class, type) or not callable(resolved_class):
            raise TypeError(f'"{self.datamodule_class_path}" resolved to {resolved_class!r}, which is not a class')
        if not issubclass(resolved_class, pyine.data.datamodule.BaseDataModule):
            raise TypeError(f'"{self.datamodule_class_path}" is not a subclass of BaseDataModule')
        self._resolved_datamodule_class = resolved_class  # type: ignore[attr-defined]
        # Skip parser/loader resolution - ProbeDataModule builds datasets directly.
        self._resolved_dataparser_configs = {}  # type: ignore[attr-defined]
        self._resolved_dataloader_configs = {}  # type: ignore[attr-defined]
        # Validate subset name consistency
        if any(name not in self.subset_names for name in self.train_subset_names):
            raise ValueError(f"some subset name(s) are invalid; got {self.train_subset_names!r}")
        if any(name not in self.subset_names for name in self.valid_subset_names):
            raise ValueError(f"some subset name(s) are invalid; got {self.valid_subset_names!r}")
        if any(name not in self.subset_names for name in self.eval_subset_names):
            raise ValueError(f"some subset name(s) are invalid; got {self.eval_subset_names!r}")
        return self


def get_datamodule_configs(
    group: str,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Returns datamodule configs for probe-based training apps.

    Returns a base ``ProbeDataModuleConfig`` config for hydra-zen config composition. Required
    fields (``lmdb_path``) are left as MISSING: the user must provide them at runtime or in a
    YAML override.

    Args:
        group: Hydra config group path for the datamodule configs.
    """
    base_config = pyine.configs.utils.make_config_description(
        ProbeDataModuleConfig,
        name="probe_base",
        group=group,
        description="Base probe datamodule settings (LMDB source, splitting, filtering).",
        config={
            "populate_full_signature": True,
            "hydra_convert": "object",
        },
    )
    return [base_config]
