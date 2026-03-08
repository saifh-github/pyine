"""DataModule for probe training on LMDB completion records."""

from __future__ import annotations

import logging
import pathlib
import typing

import datasets

import pyine.data.datamodule
import pyine.guardrails.data.lmdb_dataset
import pyine.utils.filesystem
import pyine.utils.reprod

if typing.TYPE_CHECKING:
    from pyine.guardrails.data.datamodule_configs import ProbeDataModuleConfig

logger = logging.getLogger(__name__)


class ProbeDataModule(pyine.data.datamodule.BaseDataModule["ProbeDataModuleConfig"]):
    """DataModule for probe training on LMDB completion records.

    Provides the ``prepare_data()`` -> ``setup()`` lifecycle expected by
    :func:`pyine.apps.trainers.common.prepare_datamodule`, with caching
    of the HF ``DatasetDict`` for multi-rank loading.

    The primary accessor is :meth:`get_probe_dataset`, which returns the
    prepared ``DatasetDict`` with ``"train"`` and ``"valid"`` splits.
    """

    def __init__(self, config: ProbeDataModuleConfig, verbose: bool = False) -> None:
        super().__init__(config)
        self._dataset_dict: datasets.DatasetDict | None = None
        self._code_type_to_id: dict[str, int] | None = None
        self._id_to_code_type: dict[int, str] | None = None

    # --- Lifecycle ---

    @typing.override
    def prepare_data(self) -> None:
        """Load LMDB, deduplicate, filter, split, and cache as HF DatasetDict.

        Called once on rank 0 (or local rank 0 per node). The result is saved
        to disk so that all ranks can load it in ``setup()``.
        """
        cache_path = self._get_cache_path()
        if cache_path.exists():
            logger.info("Using cached probe dataset at %s (hash: %s)", cache_path, cache_path.name)
            return

        ds = pyine.guardrails.data.lmdb_dataset.load_probe_dataset_from_lmdb(self.config)
        ds.save_to_disk(str(cache_path))  # pyright: ignore[reportUnknownMemberType]  # datasets stubs
        logger.info("Saved probe dataset to %s (hash: %s)", cache_path, cache_path.name)

    _REQUIRED_COLUMNS = frozenset({"messages", "label", "sample_id", "code_type"})
    """Columns expected in every split of the cached probe dataset."""

    @typing.override
    def setup(self, stage: str | None = None) -> None:
        """Load the cached DatasetDict from disk (all ranks).

        Also validates that the cached dataset has the expected schema (``messages``, ``label``,
        ``sample_id``, ``code_type`` columns), raising if outdated data is found.
        """
        cache_path = self._get_cache_path()
        if not cache_path.exists():
            raise RuntimeError(f"Probe dataset cache not found at {cache_path}. Call prepare_data() first.")
        self._dataset_dict = datasets.DatasetDict.load_from_disk(str(cache_path))  # pyright: ignore[reportUnknownMemberType]  # datasets stubs
        # validate schema: stale caches may have 'text' instead of 'messages'
        for split_name in self._dataset_dict:
            columns = set(self._dataset_dict[split_name].column_names)
            missing = self._REQUIRED_COLUMNS - columns
            if missing:
                raise ValueError(
                    f"missing columns in split '{split_name}' ({missing}); "
                    f"delete cache at {cache_path} and rerun prepare_data()"
                )
        # Build code_type mappings from the dataset content
        all_code_types = sorted(
            set(typing.cast("list[str]", self._dataset_dict["train"]["code_type"]))
            | set(typing.cast("list[str]", self._dataset_dict["valid"]["code_type"]))
        )
        self._code_type_to_id = {ct: i for i, ct in enumerate(all_code_types)}
        self._id_to_code_type = {i: ct for ct, i in self._code_type_to_id.items()}

    @typing.override
    def teardown(self, stage: str | None = None) -> None:
        self._dataset_dict = None
        self._code_type_to_id = None
        self._id_to_code_type = None

    # --- Accessors ---

    def get_probe_dataset(
        self,
        text_field: str | None = None,
    ) -> datasets.DatasetDict:
        """Return the prepared DatasetDict with ``"train"`` and ``"valid"`` splits.

        Each split has columns: ``messages`` (list[dict[str, str]]), ``label`` (int),
        ``sample_id`` (str), ``code_type`` (str).

        Args:
            text_field: Ignored for probe datasets, which already contain structured messages.
        """
        del text_field
        if self._dataset_dict is None:
            raise RuntimeError("DataModule not set up. Call setup() first.")
        return self._dataset_dict

    @property
    def code_type_to_id(self) -> dict[str, int]:
        if self._code_type_to_id is None:
            raise RuntimeError("DataModule not set up. Call setup() first.")
        return self._code_type_to_id

    @property
    def id_to_code_type(self) -> dict[int, str]:
        if self._id_to_code_type is None:
            raise RuntimeError("DataModule not set up. Call setup() first.")
        return self._id_to_code_type

    # --- DataModule interface ---

    def get_stats(
        self,
        target_subsets: list[str] | None = None,
    ) -> dict[str, int | float | str]:
        if self._dataset_dict is None:
            return {}
        stats: dict[str, int | float | str] = {}
        for split_name in target_subsets or ["train", "valid"]:
            if split_name not in self._dataset_dict:
                continue
            ds = self._dataset_dict[split_name]
            stats[f"{split_name}/num_samples"] = len(ds)
            labels = typing.cast("list[int]", ds["label"])
            stats[f"{split_name}/num_positive"] = sum(1 for label in labels if label == 1)
            stats[f"{split_name}/num_negative"] = sum(1 for label in labels if label == 0)
            code_types = typing.cast("list[str]", ds["code_type"])
            code_type_counts: dict[str, int] = {}
            for ct in code_types:
                code_type_counts[ct] = code_type_counts.get(ct, 0) + 1
            for ct, count in sorted(code_type_counts.items()):
                stats[f"{split_name}/code_type/{ct}"] = count
        return stats

    def get_fingerprint_inputs(self) -> pyine.utils.reprod.FingerprintInputs:
        cache_path = self._get_cache_path()
        return pyine.utils.reprod.FingerprintInputs(
            metadata_paths=[cache_path] if cache_path.exists() else [],
            extra_content=self.config.model_dump_json().encode(),
        )

    # train_dataloader / val_dataloader are not needed for the probe trainer
    # (it builds its own DataLoaders via build_dataloader() + accelerator.prepare()).

    @typing.override
    def train_dataloader(self) -> typing.NoReturn:
        raise NotImplementedError(
            "ProbeDataModule does not provide DataLoaders directly. "
            "Use get_probe_dataset() and build DataLoaders externally."
        )

    def val_dataloader(self) -> typing.NoReturn:
        raise NotImplementedError(
            "ProbeDataModule does not provide DataLoaders directly. "
            "Use get_probe_dataset() and build DataLoaders externally."
        )

    @typing.override
    def test_dataloader(self) -> typing.NoReturn:
        raise NotImplementedError("ProbeDataModule does not support test split.")

    def predict_dataloader(self) -> typing.NoReturn:
        raise NotImplementedError("ProbeDataModule does not support predict split.")

    # --- Private ---

    def _get_cache_path(self) -> pathlib.Path:
        """Deterministic cache path from config hash + LMDB file fingerprint.

        Incorporates both the config and the LMDB data file's mtime/size so
        that the cache is invalidated when either the config changes OR the
        LMDB content changes (even if the path stays the same).

        Raises :class:`FileNotFoundError` if the LMDB path does not exist.
        """
        lmdb_path = pathlib.Path(self.config.lmdb_path)
        if not lmdb_path.exists():
            raise FileNotFoundError(f"LMDB path does not exist: {lmdb_path}. Check the 'lmdb_path' config value.")
        # LMDBs are directories containing data.mdb + lock.mdb.
        # Fingerprint data.mdb specifically -- directory stat can change
        # for non-data reasons (lock updates) and may not reflect data
        # changes on all filesystems.
        if lmdb_path.is_dir():
            data_file = lmdb_path / "data.mdb"
            if not data_file.exists():
                raise FileNotFoundError(
                    f"LMDB directory exists but data.mdb is missing: {lmdb_path}. "
                    "This may indicate an incomplete copy or wrong path."
                )
        else:
            data_file = lmdb_path
        data_stat = data_file.stat()
        fingerprint_parts = {
            "config": self.config.model_dump(),
            "lmdb_mtime_ns": data_stat.st_mtime_ns,
            "lmdb_size": data_stat.st_size,
        }
        cache_hash = pyine.utils.reprod.get_versioned_cache_hash(fingerprint_parts)
        cache_root = pyine.utils.filesystem.get_data_cache_subdir("probe_datasets")
        return cache_root / f"probe_{cache_hash}"
