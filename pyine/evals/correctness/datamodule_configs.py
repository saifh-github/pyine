"""Configuration for the correctness evaluation DataModule."""

from __future__ import annotations

import pathlib  # noqa: TC003
import typing

import pydantic

import pyine.data.datamodule
import pyine.evals.correctness.types as correctness_types
import pyine.utils.filesystem
import pyine.utils.portability


def _get_datamodule_fully_qualified_name() -> str:
    """Returns the fully qualified name of the `CorrectnessDataModuleConfig` class."""
    from pyine.evals.correctness.datamodule import CorrectnessDataModule

    return pyine.utils.portability.get_fully_qualified_name(CorrectnessDataModule)


class CorrectnessDataModuleConfig(pyine.data.datamodule.BaseDataModuleConfig):
    """Configuration for correctness evaluation data loaded from LMDB.

    Inherits from `pyine.data.datamodule.BaseDataModuleConfig` to participate in the
    ``prepare_data()`` -> ``setup()`` lifecycle. Like ``ProbeDataModuleConfig``, this config
    builds datasets directly from LMDB paths (not through the ``instantiate_parser()`` ->
    ``torch.utils.data.Dataset`` pattern), so parser/loader resolution is skipped in
    `_validate_and_resolve`.
    """

    datamodule_class_path: str = pydantic.Field(default_factory=_get_datamodule_fully_qualified_name)
    """Dotted import path to the datamodule class; defaults to CorrectnessDataModule."""

    default_dataparser_config: typing.Any = None  # type: ignore[assignment]
    """Not used for correctness evaluation. Data is loaded directly from LMDB."""

    lmdb_paths: tuple[pathlib.Path, ...]
    """Paths (or glob patterns) to LMDB datasets containing eval records from DiskEvalLogger."""
    label_type: correctness_types.LabelType = correctness_types.LabelType.SOFT_MATCH
    """Which correctness label to use from LMDB records."""
    split_config: correctness_types.GuardrailSplitConfig
    """Configuration for building guardrail train/valid/test splits."""

    subset_names: tuple[str, ...] = ("guardrail_train", "guardrail_valid", "guardrail_test")
    """All subset names recognized by this datamodule."""
    train_subset_names: tuple[str, ...] = ("guardrail_train",)
    """Subsets used for guardrail training (threshold calibration data comes from guardrail_valid)."""
    valid_subset_names: tuple[str, ...] = ("guardrail_valid",)
    """Subsets used for validation and threshold calibration."""
    eval_subset_names: tuple[str, ...] = ("guardrail_valid",)
    """Subsets to evaluate (benchmark) on.

    Defaults to valid (not test) until experiments are done and all hyperparameters are permanently
    fixed. See: https://en.wikipedia.org/wiki/Training,_validation,_and_test_data_sets
    """

    @pydantic.field_validator("lmdb_paths", mode="before")
    @classmethod
    def _normalize_lmdb_paths(
        cls,
        value: typing.Any,
    ) -> tuple[pathlib.Path, ...] | None:
        """Normalize flexible path input (single string, list, glob) into a tuple of paths."""
        return pyine.utils.filesystem.normalize_path_tuple(value, field_name="lmdb_paths")

    @pydantic.model_validator(mode="after")
    @typing.override
    def _validate_and_resolve(self) -> CorrectnessDataModuleConfig:
        """Override parent to skip parser/loader resolution.

        CorrectnessDataModule builds datasets directly from LMDB, not through the
        ``instantiate_parser()`` -> ``torch.utils.data.Dataset`` pattern.
        Parser/loader resolution is skipped, but ``datamodule_class_path``
        resolution is preserved (needed by ``instantiate_datamodule()``).
        Empty dicts are set for parser/loader configs so that any accidental
        call to ``instantiate_parser()`` fails with a clear KeyError.
        """
        # resolve datamodule_class_path
        resolved_class = pyine.utils.portability.import_from_dotted_path(self.datamodule_class_path)
        if not isinstance(resolved_class, type) or not callable(resolved_class):
            raise TypeError(f'"{self.datamodule_class_path}" resolved to {resolved_class!r}, which is not a class')
        if not issubclass(resolved_class, pyine.data.datamodule.BaseDataModule):
            raise TypeError(f'"{self.datamodule_class_path}" is not a subclass of BaseDataModule')
        self._resolved_datamodule_class = resolved_class  # type: ignore[attr-defined]
        # skip parser/loader resolution; data loaded directly from LMDB
        self._resolved_dataparser_configs = {}  # type: ignore[attr-defined]
        self._resolved_dataloader_configs = {}  # type: ignore[attr-defined]
        # validate subset name consistency
        if any(name not in self.subset_names for name in self.train_subset_names):
            raise ValueError(f"some subset name(s) are invalid; got {self.train_subset_names!r}")
        if any(name not in self.subset_names for name in self.valid_subset_names):
            raise ValueError(f"some subset name(s) are invalid; got {self.valid_subset_names!r}")
        if any(name not in self.subset_names for name in self.eval_subset_names):
            raise ValueError(f"some subset name(s) are invalid; got {self.eval_subset_names!r}")
        return self
