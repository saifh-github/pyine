"""Base class and config for bias-related datamodules with shared metadata caching logic."""

from __future__ import annotations

import abc
import collections.abc
import contextlib
import copy
import itertools
import logging
import os
import pathlib  # noqa: TC003
import tempfile
import typing
import warnings

import filelock
import msgspec
import numpy as np
import omegaconf
import pydantic

import pyine.configs.schemas
import pyine.configs.utils
import pyine.data.datamodule
import pyine.data.traces.dataset_reader
import pyine.data.traces.dataset_utils
import pyine.data.utils.filter_rules
import pyine.data.utils.lmdb_io
import pyine.data.utils.splits
import pyine.evals.common
import pyine.organisms.datamodules.samples
import pyine.organisms.datamodules.samples.common
import pyine.organisms.datamodules.utils.transforms
import pyine.prompts.types
import pyine.utils.filesystem
import pyine.utils.parsing
import pyine.utils.portability
import pyine.utils.reprod

if typing.TYPE_CHECKING:
    import datasets as hf_datasets

logger = logging.getLogger(__name__)

_KNOWN_ID_SUFFIX_SEPARATOR = "::"
"""Separator used in sample identifiers to denote variant suffixes (e.g. ``::hinted``, ``::with_keyword``)."""


def get_default_subset_names() -> tuple[str, ...]:
    """Returns the default subset names used by derived datamodules."""
    return "train", "valid", "test"


class BiasDataModuleBaseConfig(pyine.data.datamodule.ConversationDataModuleConfig):
    """Base configuration class for bias-related datamodules.

    This class extracts common configuration fields shared between ShortcutBiasDataModule
    and KeywordBiasDataModule (and potentially other future bias datamodules).
    """

    # --------------- DATA PARSER / LOADER CONFIGURATIONS ---------------

    lmdb_paths: typing.Annotated[tuple[pathlib.Path, ...], pydantic.Field(min_length=1)]
    """Sequence of paths pointing to LMDB datasets containing execution traces."""
    default_dataparser_config: pydantic.SerializeAsAny[pyine.data.datamodule.BaseDataParserConfig]
    """Default trace parser configuration."""
    dataparser_config_overrides: dict[pyine.data.datamodule.SubsetNameType, dict[str, typing.Any]] = pydantic.Field(
        default_factory=dict,
    )
    """Overrides for the default trace parser configuration; adds subset-specific transforms."""
    dataloader_config_overrides: dict[pyine.data.datamodule.SubsetNameType, dict[str, typing.Any]] = pydantic.Field(
        default_factory=dict,
    )
    """Overrides for the default DataLoader configuration."""
    instantiate_parsers_at_setup: bool = False
    """Specifies whether to instantiate data parsers at setup time (default: False, deferred)."""

    # --------------- DATA FILTERING + SPLITTING CONFIGURATION ---------------

    max_solution_count: int | dict[pyine.data.datamodule.SubsetNameType, int] | None = None
    """Maximum number of solutions to load across specific or all data subsets.

    If None, all solutions (and their traces) will be loaded; this is the default behavior. If an integer is
    specified, that many solutions will be randomly picked for each subset. If a dictionary is specified, it
    maps subset names to the desired number of solutions to load for that subset.
    """
    split_file_path: pathlib.Path
    """Path to the file containing the split data for the full dataset."""
    base_filter_rule: str = ""
    """Base filter rule to apply to tags of all traces to determine what to include across all subsets.

    Traces with tags that match this rule will be filtered out. See the `pyine.data.utils.filter_rules`
    module for filter rule syntax. This rule applies in a case-insensitive manner.
    """

    # --------------- PREGENERATED OUTPUT CONFIGURATION ---------------

    pregenerated_outputs_lmdb_paths: tuple[pathlib.Path, ...] | None = None
    """Paths to LMDB dataset(s) of pregenerated model outputs to use instead of groundtruth.

    Accepts a single path, a list of paths, or a glob pattern (e.g. ``output/rank_*``). A directory
    without ``data.mdb`` that contains ``rank_*/`` subdirs is automatically expanded to those
    subdirs. When multiple paths are provided, records from all LMDBs are merged and deduplicated
    per sample_id using the configured selection strategy.

    When set, the LMDB datasets are read at ``setup()`` time and matching sample identifiers have a
    ``pregenerated_output`` field populated on the resulting ``SampleData`` (the original
    ``expected_output`` is always preserved unchanged). The LMDB datasets are expected to have been
    written by e.g. ``DiskRewardLogger`` during a prior RL run.

    After ``setup()``, the full selected records (including reward terms, reasoning, and other
    metadata) are accessible via ``BiasDataModuleBase.pregenerated_output_records``.
    """
    pregenerated_outputs_selection: typing.Literal["latest", "best_reward"] = "latest"
    """Strategy for selecting among multiple pregenerated completions per sample_id.

    ``"latest"`` picks the entry with the highest generation count; ``"best_reward"`` picks the
    entry with the highest ``reward_total`` (requires all records to have a non-None reward_total).
    """
    pregenerated_outputs_phase_prefix: str = ""
    """Phase prefix to filter exported LMDB records (e.g. ``'train/'``, ``'eval/'``).

    Empty string (default) loads all phases without filtering. This is safe when problem-level
    splits prevent cross-phase sample ID collisions (which should always be the case).
    """
    pregenerated_outputs_only_matched: bool = False
    """If True, only produce samples that have a matching pregenerated output.

    Samples without a match are filtered out after selection. Raises ``ValueError`` at
    parser construction time if a subset resolves to zero matching pregenerated outputs.
    Requires ``pregenerated_outputs_lmdb_paths`` to be set.
    """

    @pydantic.field_validator("pregenerated_outputs_lmdb_paths", mode="before")
    @classmethod
    def _normalize_pregenerated_lmdb_paths(
        cls,
        value: typing.Any,
    ) -> tuple[pathlib.Path, ...] | None:
        """Normalize single path, list/sequence of paths, or glob pattern to a tuple of paths."""
        return pyine.utils.filesystem.normalize_path_tuple(value, field_name="pregenerated_outputs_lmdb_paths")

    @pydantic.field_validator("pregenerated_outputs_phase_prefix")
    @classmethod
    def _normalize_phase_prefix(
        cls,
        value: str,
    ) -> str:
        """Normalize the phase prefix for consistent matching against export keys."""
        if not value:
            return ""
        return pyine.utils.parsing.normalize_path_prefix(value)

    @pydantic.model_validator(mode="after")
    def _validate_pregenerated_outputs_config(self) -> BiasDataModuleBaseConfig:
        """Validate that pregenerated output settings are consistent."""
        if self.pregenerated_outputs_only_matched and self.pregenerated_outputs_lmdb_paths is None:
            raise ValueError(
                "pregenerated_outputs_only_matched=True requires pregenerated_outputs_lmdb_paths to be set"
            )
        return self

    # --------------- DATA TRANSFORMATION + COLLATE CONFIGURATION ---------------

    prompt_config: pyine.prompts.types.PromptBuildConfig = pydantic.Field(
        default_factory=lambda: pyine.prompts.types.PromptBuildConfig(
            prompt_name=pyine.prompts.PromptNames.CODE_EXECUTION,
            use_chat_template=True,
            include_examples=True,
        )
    )
    """Configuration for the prompt used when transforming raw sample data to chat model requests."""

    # --------------- CODE FORMATTING OPTIONS ---------------

    add_line_numbers: bool = False
    """Whether to add line number prefixes to code strings before formatting."""
    add_block_markers: bool = False
    """Whether to add block-of-interest suffix comments to code strings.

    Only applied when the sample's predict_type is not ``program_output`` and the
    first_line/last_line attributes are valid.
    """

    # --------------- MISC SETTINGS CONFIGURATION ---------------

    subset_names: typing.Annotated[tuple[pyine.data.datamodule.SubsetNameType, ...], pydantic.Field(min_length=1)] = (
        get_default_subset_names()
    )
    """List of data subsets that the module supports."""
    eval_subset_names: tuple[str, ...] = ("valid",)
    """Subset names that are meant for model evaluation.

    Note: should be kept to 'valid' instead of 'test' until experiments are done, and all
    hyperparameters are permanently FIXED; if this sounds strange to you, refer to:
        https://en.wikipedia.org/wiki/Training,_validation,_and_test_data_sets
    """

    # --------------- PUBLIC UTILITY FUNCTIONS ---------------

    @typing.override
    def instantiate_sample_to_messages_transform(
        self,
        append_answer: bool = True,
        use_hf_messages: bool = False,
        merge_system_with_user: bool = False,
        keep_original_data: bool = False,
        orig_data_key: str = "sample_data",
    ) -> pyine.organisms.datamodules.utils.transforms.SampleTransformType:
        """Returns the sample transform function used to prepare training/evaluation code exec messages."""
        if not use_hf_messages and keep_original_data:
            raise ValueError("cannot keep original sample data if using langchain message format")
        return pyine.organisms.datamodules.utils.transforms.create_sample_transform(
            append_answer=append_answer,
            use_hf_messages=use_hf_messages,
            merge_system_with_user=merge_system_with_user,
            hf_messages_key=self.hf_messages_key,
            orig_sample_key=orig_data_key if keep_original_data else None,
            add_line_numbers=self.add_line_numbers,
            add_block_markers=self.add_block_markers,
            **self.prompt_config.model_dump(),
        )

    # --------------- PRIVATE UTILITY FUNCTIONS & ATTRIBUTES ---------------

    _resolved_base_filter: pyine.data.utils.filter_rules.FilterType | None = pydantic.PrivateAttr(default=None)

    @property
    def base_filter(self) -> pyine.data.utils.filter_rules.FilterType | None:
        """Returns the resolved base filter rule used to screen trace tags."""
        return self._resolved_base_filter

    @pydantic.model_validator(mode="after")
    @typing.override
    def _validate_and_resolve(self) -> BiasDataModuleBaseConfig:
        """Validates and resolves dataset paths and internal filtering rules."""
        super()._validate_and_resolve()  # type: ignore[reportUnknownMemberType]
        for lmdb_path in self.lmdb_paths:
            if not lmdb_path.exists():
                raise ValueError(f"LMDB dataset does not exist at path: {lmdb_path}")
        self._resolved_base_filter = pyine.data.utils.filter_rules.build_filter_from_rule(
            self.base_filter_rule, case_sensitive=False
        )
        if not self.split_file_path.is_file():
            raise ValueError(f"dataset split file does not exist at path: {self.split_file_path}")
        return self

    @abc.abstractmethod
    def _get_cache_subdirectory_name(self) -> str:
        """Return the cache subdirectory name for this bias datamodule type.

        Example: "shortcuts" for ShortcutBiasDataModule, "keywords" for KeywordBiasDataModule.
        """
        ...

    def _get_parent_subset_name(
        self,
        subset_name: pyine.data.datamodule.SubsetNameType,
    ) -> pyine.data.datamodule.SubsetNameType:
        """Map a derived subset name back to its parent subset, if applicable.

        This default implementation returns the original name unchanged. Subclasses can override to
        provide custom parent-mapping logic (e.g., 'valid_with_keyword' -> 'valid' for keywords module).
        """
        return subset_name

    @typing.override
    def _resolve_dataparser_config(
        self,
        subset_name: pyine.data.datamodule.SubsetNameType,
    ) -> pyine.organisms.datamodules.samples.SampleBuilderConfig:
        """Returns the data parser configuration for the given subset name.

        For derived subsets (e.g., valid_with_keyword), inherits overrides from the parent
        eval subset (e.g., valid) if no specific override is defined.
        """
        if subset_name not in self.subset_names:
            raise ValueError(f"invalid subset name: {subset_name}, expected one of: {self.subset_names}")
        parser_config = self.default_dataparser_config
        if isinstance(parser_config, dict):
            parser_config = pyine.organisms.datamodules.samples.SampleBuilderConfig.model_validate(parser_config)
        else:
            assert isinstance(parser_config, pyine.data.datamodule.BaseDataParserConfig)
            # convert BaseDataParserConfig to SampleBuilderConfig (may happen when instantiated via Hydra)
            parser_config = pyine.organisms.datamodules.samples.SampleBuilderConfig(
                class_path=parser_config.class_path,
                params=parser_config.params,
            )
        assert isinstance(parser_config, pyine.organisms.datamodules.samples.SampleBuilderConfig), (
            f"unexpected type for default dataparser config: {type(parser_config)}"
        )
        has_explicit_selection_override = bool(
            subset_name in self.dataparser_config_overrides
            and self.dataparser_config_overrides[subset_name]
            and "selection_config" in self.dataparser_config_overrides[subset_name]
        )
        # check for overrides: first try direct subset name, then fall back to parent
        override_key = subset_name
        if override_key not in self.dataparser_config_overrides:
            override_key = self._get_parent_subset_name(subset_name)
        if override_key in self.dataparser_config_overrides and self.dataparser_config_overrides[override_key]:
            config_overrides = self.dataparser_config_overrides[override_key]
            assert isinstance(config_overrides, dict), (
                f"unexpected type for {subset_name} dataparser config overrides: {type(config_overrides)}"
            )
            parser_config = parser_config.get_updated_spec(**config_overrides)
        special_subset_overrides = parser_config.get_special_subset_param_overrides(subset_name)
        if special_subset_overrides:
            if has_explicit_selection_override and "selection_config" in special_subset_overrides:
                # explicit selection override takes priority; remove auto-detected one
                special_subset_overrides = {
                    k: v for k, v in special_subset_overrides.items() if k != "selection_config"
                }
            parser_config = parser_config.get_updated_spec(**special_subset_overrides)
        return parser_config

    @typing.override
    def _resolve_dataloader_config(
        self,
        loader_name: pyine.data.datamodule.LoaderNameType,
    ) -> pyine.data.datamodule.BaseDataLoaderConfig:
        """Returns the data loader configuration for the given loader name.

        For derived loaders, inherits overrides from the parent subset.
        """
        if loader_name not in self.loader_names:
            raise ValueError(f"invalid loader name: {loader_name}, expected one of: {self.loader_names}")
        loader_config: pyine.data.datamodule.BaseDataLoaderConfig = self.default_dataloader_config
        # check for overrides: first try direct loader name, then fall back to parent
        override_key = loader_name
        if override_key not in self.dataloader_config_overrides:
            override_key = self._get_parent_subset_name(loader_name)
        if override_key in self.dataloader_config_overrides:
            return loader_config.get_updated_spec(**self.dataloader_config_overrides[override_key])
        return loader_config


class BiasDataModuleBase[ConfigType: BiasDataModuleBaseConfig](
    pyine.data.datamodule.ConversationDataModule[ConfigType],
):
    """Base class for bias-related datamodules with shared metadata caching and parser logic.

    This class extracts common functionality shared between the ShortcutBiasDataModule and
    the KeywordBiasDataModule, including metadata caching, lazy parser instantiation, and subset
    resolution logic.
    """

    _metadata: pyine.data.traces.dataset_utils.TraceDatasetMetadata | None
    _readers: list[pyine.data.traces.dataset_reader.DatasetProtocol]
    _subset_parsers: dict[
        pyine.data.datamodule.SubsetNameType,
        pyine.organisms.datamodules.samples.SampleBuilder | None,
    ]

    def __init__(
        self,
        config: ConfigType,
    ) -> None:
        """Initializes a new BiasDataModuleBase instance."""
        super().__init__(config)
        self._metadata = None
        self._readers = []
        self._subset_parsers = {}
        self._active_subset_names: tuple[pyine.data.datamodule.SubsetNameType, ...] = ()
        self._pregenerated_outputs: (
            dict[str, pyine.organisms.datamodules.samples.common.PregeneratedOutputRecord] | None
        ) = None

    # --------------- ABSTRACT METHODS ---------------

    @abc.abstractmethod
    def _get_metadata_model_class(
        self,
    ) -> type[pyine.data.traces.dataset_utils.TraceDatasetMetadata]:
        """Return the metadata model class used for serialization/deserialization.

        Subclasses should override this to return their specific metadata class
        (e.g., KeywordTraceDatasetMetadata for KeywordBiasDataModule).
        """
        ...

    @abc.abstractmethod
    def _prepare_bias_specific_metadata(
        self,
        base_traces_meta: list[pyine.data.traces.dataset_utils.TraceMetadata],
        split_data: pyine.data.utils.splits.SplitResult,
    ) -> pyine.data.traces.dataset_utils.TraceDatasetMetadata:
        """Prepare bias-type-specific metadata during prepare_data().

        Args:
            base_traces_meta: Pre-filtered traces based on base_filter config.
            split_data: Problem split assignments loaded from split file.

        Returns:
            Complete TraceDatasetMetadata ready for caching.
        """
        ...

    @abc.abstractmethod
    def _log_setup_summary(self) -> None:
        """Log a summary of the datamodule configuration after setup completes.

        Subclasses should implement this to log relevant information about the prepared dataset
        (e.g., subset sizes, bias-specific configuration).
        """
        ...

    # --------------- SHARED HELPER METHODS ---------------

    def _apply_max_solution_count_cap(
        self,
        subset_traces_meta: dict[
            pyine.data.datamodule.SubsetNameType,
            list[pyine.data.traces.dataset_utils.TraceMetadata],
        ],
        unassigned_traces_meta: list[pyine.data.traces.dataset_utils.TraceMetadata],
    ) -> None:
        """Apply max_solution_count cap to subset traces, modifying args in-place.

        When max_solution_count is configured, this randomly selects up to that many solutions
        per subset and moves excess traces to the unassigned list.

        Args:
            subset_traces_meta: Dict mapping subset names to their trace metadata lists.
            unassigned_traces_meta: List to append leftover traces to.
        """
        if self.config.max_solution_count is None:
            return
        rng = np.random.default_rng(self.config.split_seed)
        for subset_name, traces_meta in subset_traces_meta.items():
            tidxs_to_sids = {tidx: str(tm.solution_id) for tidx, tm in enumerate(traces_meta)}
            solution_ids = sorted(set(tidxs_to_sids.values()))
            if isinstance(self.config.max_solution_count, int):
                max_solution_count = self.config.max_solution_count
            else:
                assert isinstance(self.config.max_solution_count, collections.abc.Mapping)
                if subset_name not in self.config.max_solution_count:
                    continue
                max_solution_count = self.config.max_solution_count[subset_name]
            assert max_solution_count > 0, "max solution count must be positive"
            if len(solution_ids) <= max_solution_count:
                continue
            picked_solution_ids = rng.choice(solution_ids, size=max_solution_count, replace=False)
            picked_trace_meta_idxs = [
                trace_meta_idx
                for trace_meta_idx, solution_id in tidxs_to_sids.items()
                if solution_id in picked_solution_ids
            ]
            subset_traces_meta[subset_name] = [traces_meta[idx] for idx in picked_trace_meta_idxs]
            unassigned_idxs = [idx for idx in tidxs_to_sids if idx not in picked_trace_meta_idxs]
            unassigned_traces_meta.extend([traces_meta[idx] for idx in unassigned_idxs])

    def _resolve_subset_filtering_config(
        self,
        subset_name: pyine.data.datamodule.SubsetNameType,
    ) -> pyine.organisms.datamodules.samples.configs.TraceFilteringConfig:
        """Resolve the effective trace filtering config for a given subset.

        This extracts the ``filtering_config`` from the resolved parser config for the given subset,
        constructing a ``TraceFilteringConfig`` from it. Used by subclasses that need to pre-filter
        traces during metadata preparation (e.g. before keyword rebalancing or hint-split partitioning).

        Args:
            subset_name: The subset whose parser filtering config to resolve.

        Returns:
            The resolved TraceFilteringConfig for the given subset.
        """
        parser_config = self.config._resolve_dataparser_config(subset_name)  # pyright: ignore[reportPrivateUsage]
        params = parser_config.get_params_dict()
        filtering_dict = params.get("filtering_config", {})
        if filtering_dict is None:
            return pyine.organisms.datamodules.samples.configs.TraceFilteringConfig()
        if isinstance(filtering_dict, dict):
            filtering_dict = typing.cast("dict[str, typing.Any]", filtering_dict)
            return pyine.organisms.datamodules.samples.configs.TraceFilteringConfig(**filtering_dict)
        return filtering_dict  # type: ignore[return-value]

    # --------------- METADATA CACHING METHODS ---------------

    def _get_cache_subdirectory_name(self) -> str:
        """Return the cache subdirectory name for this bias datamodule type.

        Delegates to the config class's implementation.
        """
        return self.config._get_cache_subdirectory_name()  # pyright: ignore[reportPrivateUsage]

    def _is_metadata_prepared(self) -> bool:
        """Returns True if the metadata is prepared and ready to be used."""
        return self._get_prepared_metadata_file_path().is_file()

    def _save_prepared_metadata(
        self,
        metadata: pyine.data.traces.dataset_utils.TraceDatasetMetadata,
    ) -> None:
        """Saves the prepared metadata to a local tmpdir."""
        encoded_data = msgspec.msgpack.encode(metadata.model_dump())
        metadata_path = self._get_prepared_metadata_file_path()
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"saving prepared {self._get_cache_subdirectory_name()} datamodule metadata to: {metadata_path}")
        lock = self._get_metadata_lock(metadata_path)
        with lock:
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(metadata_path.parent),
                prefix=f"{metadata_path.name}.tmp.",
            )
            try:
                with os.fdopen(tmp_fd, "wb") as tmp_file:
                    tmp_file.write(encoded_data)
                os.replace(tmp_path, metadata_path)
            finally:
                with contextlib.suppress(FileNotFoundError):
                    os.remove(tmp_path)

    def _load_prepared_metadata(
        self,
    ) -> pyine.data.traces.dataset_utils.TraceDatasetMetadata:
        """Loads the prepared metadata from a local tmpdir."""
        metadata_path = self._get_prepared_metadata_file_path()
        logger.debug(f"loading {self._get_cache_subdirectory_name()} datamodule metadata from: {metadata_path}")
        lock = self._get_metadata_lock(metadata_path)
        with lock, open(metadata_path, "rb") as fd:
            encoded_data = msgspec.msgpack.decode(fd.read())
        metadata_model_class = self._get_metadata_model_class()
        return metadata_model_class.model_validate(encoded_data)

    def _clear_prepared_metadata(self) -> None:
        """Clears the prepared metadata from a local tmpdir."""
        if self._is_metadata_prepared():
            metadata_path = self._get_prepared_metadata_file_path()
            lock = self._get_metadata_lock(metadata_path)
            with lock, contextlib.suppress(FileNotFoundError):
                metadata_path.unlink()

    def _get_prepared_metadata_file_path(self) -> pathlib.Path:
        """Returns the file path used to store prepared metadata in the local tmpdir."""
        params_hash = pyine.utils.reprod.get_versioned_cache_hash(self.config.model_dump())
        cache_dir = pyine.utils.filesystem.get_data_cache_subdir(
            "datamodules", self._get_cache_subdirectory_name(), "metadata"
        )
        return cache_dir / f"{params_hash}.msgspec"

    def _get_metadata_lock(self, metadata_path: pathlib.Path) -> filelock.BaseFileLock:
        """Returns a lock object for the given metadata file path."""
        lock_path = metadata_path.with_suffix(f"{metadata_path.suffix}.lock")
        return filelock.FileLock(
            str(lock_path),
            timeout=self.config.cache_lock_timeout_seconds,
        )

    # --------------- PREGENERATED OUTPUT METHODS ---------------

    def _load_pregenerated_outputs(
        self,
    ) -> dict[str, pyine.organisms.datamodules.samples.common.PregeneratedOutputRecord]:
        """Load pregenerated model outputs from one or more exported LMDB datasets.

        Reads each LMDB, filters by phase prefix, merges records across all databases, deduplicates
        by sample_id using the configured selection strategy, and returns a mapping of
        ``{sample_id: PregeneratedOutputRecord}`` with source provenance.
        """
        if self.config.pregenerated_outputs_lmdb_paths is None:
            raise ValueError("_load_pregenerated_outputs called but pregenerated_outputs_lmdb_paths is None")
        lmdb_paths = pyine.data.utils.lmdb_io.resolve_lmdb_paths(self.config.pregenerated_outputs_lmdb_paths)
        prefix = self.config.pregenerated_outputs_phase_prefix
        # `grouped` stores (gen_count, record_dict, source_lmdb_path, source_key, path_idx) per sample_id;
        # path_idx preserves the user-supplied LMDB ordering for deterministic tie-breaking
        grouped: dict[str, list[tuple[int, dict[str, typing.Any], str, str, int]]] = {}
        total_records = 0
        filtered_count = 0
        for path_idx, lmdb_path in enumerate(lmdb_paths):
            lmdb_path_str = str(lmdb_path)
            reader = pyine.data.utils.lmdb_io.LMDBReader(lmdb_path)
            try:
                total_records += len(reader.key_map)
                for key_name in reader.key_map:
                    if prefix and not key_name.startswith(prefix):
                        continue
                    filtered_count += 1
                    record = reader.get(key_name)
                    # strip the record's own key_prefix to recover (sample_id, generation_count);
                    # this is more robust than stripping the filter prefix, which may be empty
                    record_prefix = record.get("key_prefix", prefix) or ""
                    if record_prefix and not key_name.startswith(record_prefix):
                        raise ValueError(
                            f"key '{key_name}' does not start with its record's key_prefix '{record_prefix}'; "
                            "the LMDB may contain corrupted or manually edited records"
                        )
                    stripped = key_name[len(record_prefix) :] if record_prefix else key_name
                    last_slash_idx = stripped.rfind("/")
                    if last_slash_idx == -1:
                        raise ValueError(f"unexpected key format (no '/' for generation separator): {key_name}")
                    sample_id = stripped[:last_slash_idx]
                    gen_count_str = stripped[last_slash_idx + 1 :]
                    gen_count = int(gen_count_str) if gen_count_str != "none" else 0
                    model_output = record.get("model_output")
                    if not isinstance(model_output, str):
                        raise ValueError(
                            f"record for key '{key_name}' has invalid 'model_output' "
                            f"(expected str, got {type(model_output).__name__})"
                        )
                    grouped.setdefault(sample_id, []).append((gen_count, record, lmdb_path_str, key_name, path_idx))
            finally:
                reader.close()
        if prefix and filtered_count == 0:
            raise ValueError(
                f"phase prefix '{prefix}' matched 0 out of {total_records} records "
                f"across {len(lmdb_paths)} LMDB pregenerated outputs dataset(s); "
                "check that the specified prefix matches a key_prefix used during export"
            )
        # apply selection strategy per sample_id
        result: dict[str, pyine.organisms.datamodules.samples.common.PregeneratedOutputRecord] = {}
        selection = self.config.pregenerated_outputs_selection
        for sample_id, entries in grouped.items():
            if selection == "latest":
                # tie-break: prefer higher gen_count, then later LMDB in user-supplied order, then key
                best = max(entries, key=lambda entry: (entry[0], entry[4], entry[3]))
            elif selection == "best_reward":
                for gen_count, record, _lmdb_path, _key, _pidx in entries:
                    if record.get("reward_total") is None:
                        raise ValueError(
                            f"best_reward selection requires reward_total for all records, "
                            f"but sample_id '{sample_id}' (generation_count={gen_count}) has None; "
                            "ensure logging.log_total=True during export"
                        )
                # tie-break: prefer higher reward, then newer generation, then later LMDB, then key
                best = max(
                    entries,
                    key=lambda entry: (entry[1]["reward_total"], entry[0], entry[4], entry[3]),
                )
            else:
                raise ValueError(f"unknown pregenerated_outputs_selection: {selection}")
            result[sample_id] = pyine.organisms.datamodules.samples.common.PregeneratedOutputRecord(
                model_output=best[1]["model_output"],
                source_lmdb_path=best[2],
                source_key=best[3],
                full_record=best[1],
            )
        if not result:
            raise ValueError(
                f"pregenerated_outputs_lmdb_paths resolved to {len(lmdb_paths)} LMDB(s) "
                f"with {total_records} total records, but no matching records were found"
                + (f" (phase_prefix='{prefix}')" if prefix else "")
                + "; check that the LMDB(s) contain records matching the expected prefix"
            )
        logger.info(
            f"loaded pregenerated outputs: {len(result)} unique sample_ids "
            f"from {filtered_count}/{total_records} records "
            f"across {len(lmdb_paths)} LMDB(s)" + (f" (phase_prefix='{prefix}')" if prefix else "")
        )
        return result

    def _build_parser_kwargs(
        self,
        source_data: typing.Any,
        subset_traces: list[pyine.data.traces.dataset_utils.TraceMetadata],
        subset_name: pyine.data.datamodule.SubsetNameType,
    ) -> dict[str, typing.Any]:
        """Build kwargs dict for SampleBuilder instantiation, injecting pregenerated outputs.

        Args:
            source_data: Raw source data for the problem (e.g. TACO/APPS record).
            subset_traces: List of trace metadata objects for this problem subset.

        Returns:
            Kwargs dict for SampleBuilder constructor.
        """
        kwargs: dict[str, typing.Any] = {
            "source_data": source_data,
            "traces": subset_traces,
        }
        if self._pregenerated_outputs is not None:
            resolved = self._resolve_pregenerated_outputs_for_subset(subset_name)
            if not resolved and self.config.pregenerated_outputs_only_matched:
                raise ValueError(
                    f"pregenerated_outputs_only_matched=True but no pregenerated outputs "
                    f"matched subset '{subset_name}'; check that the LMDB contains entries "
                    f"with suffixes matching this subset"
                )
            if not resolved and self._pregenerated_outputs:
                logger.warning(
                    f"pregenerated outputs configured ({len(self._pregenerated_outputs)} entries) "
                    f"but none matched subset '{subset_name}'; "
                    f"this subset will proceed without pregenerated output overrides"
                )
            kwargs["pregenerated_outputs"] = resolved
            kwargs["only_with_pregenerated_output"] = self.config.pregenerated_outputs_only_matched
        return kwargs

    def _resolve_pregenerated_outputs_for_subset(
        self,
        subset_name: pyine.data.datamodule.SubsetNameType,
    ) -> dict[str, pyine.organisms.datamodules.samples.common.PregeneratedOutputRecord]:
        """Resolve pregenerated outputs for a specific subset.

        This method should have an override in subclasses that modify sample identifiers (e.g. hint
        suffix wrapping) to filter and re-key the dict so lookups match base trace IDs.

        The default implementation returns the full dict as-is, which is correct for subsets that
        don't modify identifiers. Returns empty dict if no outputs were loaded.

        Raises ValueError if any key contains '::' (a known identifier suffix separator), since the
        base resolver cannot handle suffixed keys; subclasses MUST override.
        """
        if not self._pregenerated_outputs:
            return {}
        suffixed_keys = [key for key in self._pregenerated_outputs if _KNOWN_ID_SUFFIX_SEPARATOR in key]
        if suffixed_keys:
            examples = suffixed_keys[:3]
            raise ValueError(
                f"pregenerated output keys contain '{_KNOWN_ID_SUFFIX_SEPARATOR}' suffix separators "
                f"(e.g. {examples}), but the base resolver does not handle suffixed keys; "
                f"override _resolve_pregenerated_outputs_for_subset in the datamodule subclass "
                f"to correctly resolve suffixed entries per subset"
            )
        return dict(self._pregenerated_outputs)

    @property
    def pregenerated_output_records(
        self,
    ) -> dict[str, pyine.organisms.datamodules.samples.common.PregeneratedOutputRecord] | None:
        """Mapping from sample_id to full pregenerated output records, if loaded.

        Available after ``setup()`` when ``pregenerated_outputs_lmdb_paths`` is configured. Each
        record includes the model output, source LMDB path and key, and the full LMDB record dict
        (reward terms, reasoning, etc.) for downstream inspection.
        """
        return self._pregenerated_outputs

    # --------------- PARSER INSTANTIATION METHODS ---------------

    def _instantiate_parser_if_needed(
        self,
        subset_name: pyine.data.datamodule.SubsetNameType,
    ) -> pyine.organisms.datamodules.samples.SampleBuilder:
        """Instantiates a parser for a given subset name if it has not been instantiated yet."""
        if self._subset_parsers[subset_name] is None:
            logger.debug(f"instantiating {self._get_cache_subdirectory_name()} datamodule {subset_name} parser...")
            subset_traces = self._get_traces_meta_for_subset(subset_name)
            parser = self.config.instantiate_parser(
                subset_name=subset_name,
                **self._build_parser_kwargs(
                    source_data=self._readers,
                    subset_traces=subset_traces,
                    subset_name=subset_name,
                ),
            )
            self._subset_parsers[subset_name] = typing.cast(
                "pyine.organisms.datamodules.samples.SampleBuilder",
                parser,
            )
        parser = self._subset_parsers[subset_name]
        assert parser is not None, f"parser for subset {subset_name} should be instantiated"
        return parser

    def _get_subset_suffixes(self) -> tuple[str, ...]:
        """Returns supported suffixes for dynamic code-type subset variants.

        These suffixes allow "virtual" subsets such as "train_obfuscated" to reuse the same trace
        membership as the "train" subset while changing the SampleBuilder selection config via
        SampleBuilderConfig.get_special_subset_param_overrides().

        Note:
            This is distinct from derived subsets stored in metadata (e.g., "valid_with_keyword"),
            which are resolved via TraceDatasetMetadata.derived_subsets.
        """
        return tuple(pyine.organisms.datamodules.samples.common.get_all_supported_code_type_sets_suffixes())

    def _get_traces_meta_for_subset(
        self,
        subset_name: pyine.data.datamodule.SubsetNameType | None,
    ) -> list[pyine.data.traces.dataset_utils.TraceMetadata]:
        """Returns the list of trace metadata objects for a given subset name."""
        if self._metadata is None:
            raise RuntimeError("metadata not ready yet, call `setup()` first")
        if subset_name is None:
            return self._metadata.base_traces
        # first check if the subset exists in primary or derived subsets
        all_known_subsets = self._metadata.get_all_subset_names()
        if subset_name in all_known_subsets:
            return self._metadata.get_subset_traces(subset_name)
        # fallback: check for code-type suffixed subsets (e.g., "train_buggy")
        primary_subsets = list(self._metadata.subset_traces.keys())
        for prefix, suffix in itertools.product(primary_subsets, self._get_subset_suffixes()):
            if f"{prefix}_{suffix}" == subset_name:
                return self._metadata.subset_traces[prefix]
        raise ValueError(f"subset {subset_name} is not defined in the metadata's split table")

    def _resolve_active_subset_names(
        self,
        stage: str | None,
    ) -> tuple[pyine.data.datamodule.SubsetNameType, ...]:
        """Resolves which subset names should be active for the given Lightning stage.

        Args:
            stage: Lightning stage string (``"fit"``, ``"validate"``, ``"test"``, ``"predict"``)
                or ``None`` for all subsets.

        Returns:
            Filtered and ordered tuple of subset names to set up for this stage.
        """
        all_names = self.config.subset_names
        if stage is None:
            return all_names
        # determine which primary subset names are relevant for this stage
        if stage == "fit":
            primary_names = set(self.config.train_subset_names) | set(self.config.valid_subset_names)
            expanded_names = set(self.config.resolved_valid_subset_names)
        elif stage == "validate":
            primary_names = set(self.config.valid_subset_names)
            expanded_names = set(self.config.resolved_valid_subset_names)
        elif stage in ("test", "predict"):
            primary_names = set(self.config.eval_subset_names)
            expanded_names = set(self.config.resolved_eval_subset_names)
        else:
            logger.warning(f"unknown stage '{stage}', activating all subset names")
            return all_names
        # include code-type suffix variants (e.g. "train_obfuscated") for each primary name
        suffixes = self._get_subset_suffixes()
        suffixed_names: set[str] = set()
        for primary in primary_names:
            for suffix in suffixes:
                suffixed_names.add(f"{primary}_{suffix}")
        target_names = primary_names | expanded_names | suffixed_names
        # warn about resolved names that are not in config.subset_names
        for name in expanded_names:
            if name not in set(all_names):
                logger.warning(
                    f"resolved subset name '{name}' (from stage='{stage}') is not in config.subset_names={all_names!r}"
                )
        # filter preserving original ordering
        return tuple(name for name in all_names if name in target_names)

    def _is_setup_complete(self) -> bool:
        """Returns True if the setup is complete and the data parsers/loaders are ready to be used."""
        return self._metadata is not None

    @property
    def active_subset_names(self) -> tuple[pyine.data.datamodule.SubsetNameType, ...]:  # type: ignore[override]
        """Returns the subset names that are currently active after ``setup()``."""
        return self._active_subset_names

    # --------------- LIGHTNING DATAMODULE LIFECYCLE ---------------

    @typing.override
    def prepare_data(self) -> None:
        """Prepares metadata and pre-filters traces, saving the results to a local tmpdir.

        Note: This method has NO rank guards. The caller (prepare_datamodule in common.py)
        is responsible for ensuring only appropriate ranks call this method. If multiple local ranks
        call the method simultaneously, they may try to overwrite each other's operations.
        """
        lmdb_paths_str = "\n\t".join([str(p) for p in self.config.lmdb_paths])
        cache_name = self._get_cache_subdirectory_name()
        if self._is_metadata_prepared():
            logger.info(f"using cached {cache_name} datamodule metadata for lmdb paths:\n\t{lmdb_paths_str}")
            return
        logger.info(f"preparing {cache_name} datamodule metadata for lmdb paths:\n\t{lmdb_paths_str}")
        # first prep step: identify which traces are to be kept based on our base tag filter rule
        assert self.config.base_filter is not None, "base filter should have been resolved by now"
        base_traces_meta = pyine.data.traces.dataset_reader.get_traces_metadata(
            list(self.config.lmdb_paths),
            base_filter=self.config.base_filter,
        )
        # load coding problem split data
        split_data = pyine.data.utils.splits.SplitResult.from_file(self.config.split_file_path)
        if any(subset not in self.config.subset_names for subset in split_data.config.subset_names):
            raise ValueError("mismatch between split data subsets and configured subsets")
        # call subclass-specific metadata preparation
        metadata = self._prepare_bias_specific_metadata(base_traces_meta, split_data)
        # --- DEBUG: fingerprint diagnostics (remove after resolving cross-node mismatch) ---
        import hashlib as _hashlib

        _trace_ids = [t.identifier for t in base_traces_meta]
        _trace_hash = _hashlib.md5(str(_trace_ids[:20]).encode()).hexdigest()[:12]
        _subset_info = {k: len(v) for k, v in metadata.subset_traces.items()}
        _subset_hash = _hashlib.md5(str(_subset_info).encode()).hexdigest()[:12]
        logger.warning(
            f"[DIAG] base_traces={len(base_traces_meta)} hash={_trace_hash}, subsets={_subset_info} hash={_subset_hash}"
        )
        # --- END DEBUG ---
        self._save_prepared_metadata(metadata)

    @typing.override
    def setup(
        self,
        stage: str | None = None,
    ) -> None:
        """Loads the prepared metadata and creates train/valid/test data readers.

        Args:
            stage: Lightning stage string (``"fit"``, ``"validate"``, ``"test"``, ``"predict"``)
                or ``None`` to set up all subsets.
        """
        if not self._is_metadata_prepared():
            raise RuntimeError("metadata is not prepared yet, call `prepare_data()` on main process first")
        self._metadata = self._load_prepared_metadata()
        readers: list[pyine.data.traces.dataset_reader.DatasetProtocol] = [
            pyine.data.traces.dataset_reader.DatasetReader(path) for path in self.config.lmdb_paths
        ]
        self._readers = readers
        self._pregenerated_outputs = None
        if self.config.pregenerated_outputs_lmdb_paths is not None:
            self._pregenerated_outputs = self._load_pregenerated_outputs()
        self._active_subset_names = self._resolve_active_subset_names(stage)
        self._subset_parsers.clear()
        for subset_name in self._active_subset_names:
            if self.config.instantiate_parsers_at_setup:
                self._subset_parsers[subset_name] = self._instantiate_parser_if_needed(subset_name)
            else:
                self._subset_parsers[subset_name] = None
        self._log_setup_summary()

    @typing.override
    def teardown(self, stage: str | None = None) -> None:
        """Close readers when the datamodule is torn down, and unassigns all parser attributes."""
        self._metadata = None
        self._subset_parsers.clear()
        self._active_subset_names = ()
        self._readers = []
        self._pregenerated_outputs = None

    # --------------- PUBLIC UTILITY METHODS ---------------

    @typing.override
    def get_stats(
        self,
        target_subsets: list[pyine.data.datamodule.SubsetNameType] | None = None,
    ) -> dict[str, int | float | str]:
        """Returns a dictionary of useful-to-log statistics."""
        if not self._is_setup_complete():
            raise RuntimeError("data parsers are not ready yet, call `setup()` first")
        stats: dict[str, int | float | str] = {}
        subset_names = target_subsets or list(self._subset_parsers.keys())
        if target_subsets is not None:
            available = set(self._subset_parsers.keys())
            unknown = [name for name in target_subsets if name not in available]
            if unknown:
                raise ValueError(
                    f"get_stats: requested subset names not in active parsers: {unknown}; "
                    f"active subsets are: {list(available)}"
                )
        for subset_name in subset_names:
            parser = self._instantiate_parser_if_needed(subset_name)
            for stat_key, stat_val in parser.get_stats().items():
                stats[f"{subset_name}/{stat_key}"] = stat_val
        return stats

    @typing.override
    def get_fingerprint_inputs(self) -> pyine.utils.reprod.FingerprintInputs:
        """Return inputs for cross-node fingerprint validation.

        Returns the metadata file path as the fingerprint input. The metadata file contains all
        deterministic information about the prepared dataset.

        Returns:
            FingerprintInputs with the metadata file path.
        """
        metadata_path = self._get_prepared_metadata_file_path()
        return pyine.utils.reprod.FingerprintInputs(metadata_paths=[metadata_path])

    @typing.override
    def get_parser(
        self,
        subset_name: pyine.data.datamodule.SubsetNameType,
    ) -> pyine.organisms.datamodules.samples.SampleDataParser:
        """Returns a data parser object for a given subset name.

        The returned parser is wrapped with SampleSubsetTagWrapper to append a
        `parser:<subset_name>` tag to each sample's comma_separated_tags field.
        """
        if not self._is_setup_complete():
            raise RuntimeError("data parsers are not ready yet, call `setup()` first")
        assert subset_name is not None, "subset name must be specified"
        if subset_name not in self._subset_parsers:
            raise ValueError(f"parser for subset {subset_name} is not defined")
        base_parser = self._instantiate_parser_if_needed(subset_name)
        return pyine.organisms.datamodules.samples.SampleSubsetTagWrapper(
            wrapped_dataset=base_parser,
            subset_name=subset_name,
        )

    @typing.override
    def get_hf_messages_dataset(
        self,
        subset_name: pyine.data.datamodule.SubsetNameType,
        append_answer: bool = True,
        merge_system_with_user: bool = False,
        keep_original_data: bool = False,
        force_regenerate: bool = False,
    ) -> hf_datasets.Dataset:
        """Returns a HuggingFace dataset object for a given subset name."""
        if not self._is_setup_complete():
            raise RuntimeError("data parsers are not ready yet, call `setup()` first")
        assert subset_name is not None, "subset name must be specified"
        subset_traces = self._get_traces_meta_for_subset(subset_name)
        return self.config.instantiate_hf_messages_dataset(
            subset_name=subset_name,
            append_answer=append_answer,
            merge_system_with_user=merge_system_with_user,
            keep_original_data=keep_original_data,
            parser_kwargs=self._build_parser_kwargs(
                source_data=self.config.lmdb_paths,
                subset_traces=subset_traces,
                subset_name=subset_name,
            ),
            force_regenerate=force_regenerate,
        )

    @typing.override
    def get_openai_messages_dataset(
        self,
        subset_name: pyine.data.datamodule.SubsetNameType,
        append_answer: bool = True,
        merge_system_with_user: bool = False,
    ) -> pathlib.Path:
        """Returns the path to an OpenAI-compatible JSONL dataset of chat-templated conversations."""
        if not self._is_setup_complete():
            raise RuntimeError("data parsers are not ready yet, call `setup()` first")
        assert subset_name is not None, "subset name must be specified"
        subset_traces = self._get_traces_meta_for_subset(subset_name)
        return self.config.instantiate_openai_messages_dataset(
            subset_name=subset_name,
            append_answer=append_answer,
            merge_system_with_user=merge_system_with_user,
            parser_kwargs=self._build_parser_kwargs(
                source_data=self.config.lmdb_paths,
                subset_traces=subset_traces,
                subset_name=subset_name,
            ),
        )

    def make_dataloader(
        self,
        loader_name: pyine.data.datamodule.LoaderNameType,
    ) -> pyine.organisms.datamodules.samples.SampleDataLoader:
        """Creates and returns a dataloader for the given name."""
        assert loader_name is not None, "loader name must be specified"
        parser = self.get_parser(loader_name)
        return self.config.instantiate_dataloader(
            loader_name=loader_name,
            dataset=parser,
        )

    @typing.override
    def train_dataloader(
        self,
    ) -> pyine.organisms.datamodules.samples.SampleDataLoader:
        """Return the training data loader."""
        return self.make_dataloader("train")

    @typing.override
    def val_dataloader(
        self,
    ) -> pyine.organisms.datamodules.samples.SampleDataLoader:
        """Return the validation data loader."""
        return self.make_dataloader("valid")

    @typing.override
    def test_dataloader(
        self,
    ) -> pyine.organisms.datamodules.samples.SampleDataLoader:
        """Return the test data loader."""
        return self.make_dataloader("test")


def get_default_training_transform_config(
    use_hybrid_transform: bool = False,
) -> dict[str, typing.Any]:
    """Returns the default sample transform config used by derived datamodules for training."""
    if use_hybrid_transform:
        return {  # will generate a mix of partial and full samples (based on criteria + random draws)
            "transform_strategy": "hybrid",
            "functions_fallback_to_segments": True,
            "min_partial_trace_steps": 10,
            "fallback_to_orig": True,
            "predict_type_prob_map": {
                "program_output": 0.6,
                "frame_variables": 0.2,
                "function_return": 0.2,
            },
        }
    return {"transform_strategy": "never"}  # generates only full samples


@typing.overload
def get_default_sample_builder_config(
    seed: typing.Any,
    *,
    allow_db_lookups: bool = False,
    code_type_prob_map: dict[pyine.organisms.datamodules.samples.common.SampleCodeTypeSet | str, float] | None = None,
    as_pydantic: typing.Literal[True],
) -> pyine.organisms.datamodules.samples.SampleBuilderConfig: ...


@typing.overload
def get_default_sample_builder_config(
    seed: typing.Any,
    *,
    allow_db_lookups: bool = False,
    code_type_prob_map: dict[pyine.organisms.datamodules.samples.common.SampleCodeTypeSet | str, float] | None = None,
    as_pydantic: typing.Literal[False] = False,
) -> dict[str, typing.Any]: ...


def get_default_sample_builder_config(
    seed: typing.Any,
    *,
    allow_db_lookups: bool = False,
    code_type_prob_map: dict[pyine.organisms.datamodules.samples.common.SampleCodeTypeSet | str, float] | None = None,
    as_pydantic: bool = False,
) -> dict[str, typing.Any] | pyine.organisms.datamodules.samples.SampleBuilderConfig:
    """Returns the default configuration dictionary used to instantiate sample builders.

    This configuration will be hierarchically overridden by subset-specific settings.

    Args:
        seed: Random seed for reproducibility.
        allow_db_lookups: Whether to allow database lookups for code type selection.
            Set to True for shortcuts (which uses code type selection), False for keywords.
        code_type_prob_map: Optional probability map for code type selection.
            If provided, also sets fallback_to_orig=False in the selection config.
        as_pydantic: If True, return a validated SampleBuilderConfig instance.

    Returns:
        Config dict or validated pydantic model.
    """
    selection_config: dict[str, typing.Any] = {
        "seed": seed,
        "allow_db_lookups": allow_db_lookups,
    }
    if code_type_prob_map is not None:
        selection_config["code_type_prob_map"] = code_type_prob_map
        selection_config["fallback_to_orig"] = False
    config_params = {
        "filtering_config": {
            "seed": seed,
        },
        "selection_config": selection_config,
        "transform_config": {
            "seed": seed,
            "transform_strategy": "never",
        },
    }
    if as_pydantic:
        return pyine.organisms.datamodules.samples.SampleBuilderConfig.model_validate({"params": config_params})
    return config_params


def get_default_sample_builder_overrides_for_subset(
    subset_name: str,
    use_hybrid_transform: bool = False,
    training_selection_config: dict[str, typing.Any] | None = None,
) -> dict[str, typing.Any]:
    """Returns default overrides for the sample builder config to be used for a given subset.

    The overrides should apply on top of the base (shared) sample builder config, and make the
    resulting config suitable for the given subset. If no overrides are defined, an empty dict
    will be returned.

    Args:
        subset_name: Name of the subset (e.g., "train", "valid", "test").
        use_hybrid_transform: If True, use hybrid transforms (for full+partial samples).
        training_selection_config: Optional selection config to use for the training subset.
            If None, an empty dict is used (inherits from default config).

    Returns:
        Dictionary of config overrides for the given subset, or empty dict if no overrides.
    """
    if subset_name == "train":
        return {
            "filtering_config": {},  # SampleFilteringConfig; inherits from default config
            "selection_config": training_selection_config or {},  # SampleSelectionConfig
            "transform_config": get_default_training_transform_config(use_hybrid_transform),
        }
    return {}


def make_bias_datamodule_config[ConfigT: BiasDataModuleBaseConfig](
    config_class: type[ConfigT],
    lmdb_paths: typing.Any,
    split_file_path: typing.Any,
    seed: typing.Any,
    *,
    sample_builder_config: pyine.data.datamodule.BaseDataParserConfig | None = None,
    dataparser_config_overrides: dict[str, dict[str, typing.Any]],
    extra_config: dict[str, typing.Any] | None = None,
    as_pydantic: bool = False,
) -> dict[str, typing.Any] | ConfigT:
    """Generic factory for creating bias datamodule configs.

    This function provides a shared implementation for creating datamodule configs
    for both KeywordBiasDataModule and ShortcutBiasDataModule (and future bias modules).

    Callers are responsible for providing canonical ``dataparser_config_overrides``, typically
    extracted from their pydantic field defaults. This is done so that the hydra config store
    always matches the class-level defaults.

    Args:
        config_class: The config class to instantiate (e.g., KeywordBiasDataModuleConfig).
        lmdb_paths: Paths to LMDB datasets containing execution traces.
        split_file_path: Path to the problem split file.
        seed: Random seed for reproducibility.
        sample_builder_config: Config to use for the default sample builder.
        dataparser_config_overrides: Per-subset dataparser overrides, sourced from the caller's
            pydantic field defaults (e.g. via ``get_field_default``).
        extra_config: Additional config fields to merge into the final config dict.
        as_pydantic: If True, return a validated config instance.

    Returns:
        Config dict or validated pydantic model.
    """
    if sample_builder_config is not None:
        # work on a copy to avoid mutating shared defaults across tests/runs
        sample_builder_config = sample_builder_config.model_copy(deep=True)
        # extract params and update seeds, then reconstruct full config
        default_dataparser_params = copy.deepcopy(sample_builder_config.params)
        if isinstance(default_dataparser_params, pydantic.BaseModel):
            default_dataparser_params = default_dataparser_params.model_dump()
        for cfg_name in ["filtering_config", "selection_config", "transform_config"]:
            default_dataparser_params[cfg_name]["seed"] = seed
        # reconstruct full config with class_path wrapper
        default_dataparser_config: dict[str, typing.Any] = {
            "class_path": sample_builder_config.class_path,
            "params": default_dataparser_params,
        }
    else:
        default_dataparser_config = {
            "class_path": pyine.utils.portability.get_fully_qualified_name(
                pyine.organisms.datamodules.samples.SampleBuilder,
            ),
            "params": get_default_sample_builder_config(seed),
        }
    config_kwargs: dict[str, typing.Any] = {
        "lmdb_paths": lmdb_paths,
        "split_file_path": split_file_path,
        "split_seed": seed,
        "default_dataparser_config": default_dataparser_config,
        "dataparser_config_overrides": dataparser_config_overrides,
        "dataloader_config_overrides": {
            "train": {"shuffle": True},
        },
    }
    if extra_config:
        config_kwargs.update(extra_config)
    if as_pydantic:
        return config_class.model_validate(config_kwargs)
    return config_kwargs


def make_bias_datamodule_hydra_configs(
    config_class: type[BiasDataModuleBaseConfig],
    eval_type: pyine.evals.common.EvalType,
    group: str,
    module_name: str,
    datamodule_config_factory: collections.abc.Callable[..., dict[str, typing.Any]],
    seed: str | int | None = "${runtime.seed}",
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Generates and returns bias-datamodule configs for hydra zen storage.

    This function provides a shared implementation for generating hydra configs
    for both KeywordBiasDataModule and ShortcutBiasDataModule (and future bias modules).

    Args:
        config_class: The config class (e.g., KeywordBiasDataModuleConfig).
        eval_type: The evaluation type (only CODE_EXEC is supported).
        group: The config group name for hydra storage.
        module_name: Name of the module for error messages and descriptions (e.g., "keywords").
        datamodule_config_factory: Factory function to create the base config dict.
            Should accept lmdb_paths, split_file_path, seed, and as_pydantic kwargs.
        seed: seed to use in the datamodule configs (either a hydra config reference, or a real
            integer seed value).

    Returns:
        List of config descriptions for hydra zen registration.
    """
    if eval_type != pyine.evals.common.EvalType.CODE_EXEC:
        raise NotImplementedError(f"unsupported eval type for {module_name} datamodule: {eval_type}")
    base_config = pyine.configs.utils.make_config_description(
        config_class,
        name=f"{module_name}_base",
        group=group,
        description=f"Base {module_name} datamodule settings; not specific to any actual source dataset.",
        config={
            **datamodule_config_factory(
                lmdb_paths=omegaconf.MISSING,  # must be specified by user
                split_file_path=omegaconf.MISSING,  # must be specified by user
                seed=seed,
                as_pydantic=False,
            ),
            "populate_full_signature": True,
            "hydra_convert": "object",
        },
    )
    return [
        base_config,
        *get_taco_configs(
            datamodule_base_config=base_config,
            datamodule_base_name=module_name,
            datamodule_config_type=config_class,
        ),
        # add config getters for more source datasets here, if needed
    ]


def get_taco_configs(
    datamodule_base_config: pyine.configs.schemas.ConfigDescription,
    datamodule_base_name: str,
    datamodule_config_type: type[BiasDataModuleBaseConfig],
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Returns specialized datamodule configs and their descriptions for the TACO dataset.

    This is meant to provide a generic way to tie any bias-related datamodule with pregenerated TACO
    datasets using dynamically generated ConfigDescription objects for Hydra-Zen storage.
    """
    # first, get TACO dataset paths in a fail-safe manner
    taco_latest_path = None
    with contextlib.suppress(FileNotFoundError):
        taco_latest_path = pyine.data.traces.dataset_utils.get_latest_dataset_path("TACO")
    taco_split_path = None
    with contextlib.suppress(FileNotFoundError):
        taco_split_path = pyine.data.utils.splits.get_dataset_split_file_path("TACO")
    taco_10s10t_v1_paths = None
    with contextlib.suppress(FileNotFoundError):
        taco_10s10t_v1_paths = pyine.data.traces.dataset_utils.get_matching_dataset_paths(
            source_dataset_name="TACO",
            pattern="v1.5/10s10t.*of000026.*.lmdb",
        )

    # emit warnings for missing datasets (users should not be trying to launch experiments with these)
    if taco_latest_path is None:
        # this is a demo/debug dataset, so instead of an actual warning, let's use a debug log
        logger.debug(
            "no TACO base dataset found, skipping TACO configs; "
            "if you intended to use TACO dataset demos/tests, please ensure that a dataset is present "
            "in the expected location (see the top-level README for more details)",
        )
    if taco_split_path is None:
        warnings.warn(
            "no TACO split file found, skipping TACO configs; "
            "if you intended to use TACO data modules, please ensure that the split file is present "
            "in the expected location (see the top-level README for more details)",
            stacklevel=2,
        )
    if not taco_10s10t_v1_paths:
        warnings.warn(
            "the TACO 10s10t v1 dataset is missing, skipping related configs; "
            "if you intended to conduct TACO-related experiments, please ensure that a dataset is present "
            "in the expected location (see the top-level README for more details)",
            stacklevel=2,
        )

    # build the actual config objects + descriptions
    outputs: list[pyine.configs.schemas.ConfigDescription] = []

    if taco_latest_path is not None and taco_split_path is not None:
        # if we have both of these paths, build the demo/testing configs
        outputs.append(
            taco_latest_config := pyine.configs.utils.make_config_description(
                datamodule_config_type,
                name=f"{datamodule_base_name}_TACO_latest",
                group=datamodule_base_config.group,
                description=(
                    "Specifies the single most recent instance of a TACO trace dataset found on disk. "
                    "May contain an arbitrary number of traces with any kind of augmentations."
                ),
                config={
                    "lmdb_paths": [taco_latest_path],
                    "split_file_path": taco_split_path,
                    # -------------
                    "builds_bases": (datamodule_base_config.config,),
                },
            )
        )
        outputs.append(
            pyine.configs.utils.make_config_description(
                datamodule_config_type,
                name=f"{datamodule_base_name}_TACO_latest_20s",
                group=datamodule_base_config.group,
                description=(
                    "Specifies a subset of the most recent TACO trace dataset on disk, with a maximum of 20 "
                    "solutions per trace. Useful for quick experiments, testing, and demos."
                ),
                config={
                    "max_solution_count": 20,  # cap off the dataset size for quick experiments (across each subset)
                    # -------------
                    "builds_bases": (taco_latest_config.config,),
                },
            )
        )

    # ===========

    if taco_10s10t_v1_paths and taco_split_path is not None:
        # if we have both of these paths, build TACO 10s10t configs for v1 experiments
        assert len(taco_10s10t_v1_paths) == 26, "unexpected number of v1 10s10t datasets"
        outputs.append(
            taco_10s10t_v1_config := pyine.configs.utils.make_config_description(
                datamodule_config_type,
                name=f"{datamodule_base_name}_TACO_10s10t_v1_full",
                group=datamodule_base_config.group,
                description=(
                    "Specifies the full 26 instances of the PyINE-TACO 10s10t v1 trace dataset. "
                    "Used for full-sized experiments on the entire dataset proposed in our first "
                    "paper. See `pyine/apps/README-10s10t-v1.md` for more information."
                ),
                config={
                    "lmdb_paths": taco_10s10t_v1_paths,
                    "split_file_path": taco_split_path,
                    # -------------
                    "builds_bases": (datamodule_base_config.config,),
                },
            )
        )
        outputs.append(
            pyine.configs.utils.make_config_description(
                datamodule_config_type,
                name=f"{datamodule_base_name}_TACO_10s10t_v1_part1to13",
                group=datamodule_base_config.group,
                description=(
                    "Specifies a subset consisting of parts 1 to 13 (of 26, so about 50%) of the "
                    "PyINE-TACO 10s10t v1 trace dataset. This is a subset that can be useful for medium-sized"
                    "experiments, and should be quite representative of the full dataset's distribution."
                ),
                config={
                    "lmdb_paths": taco_10s10t_v1_paths[0:13],
                    # -------------
                    "builds_bases": (taco_10s10t_v1_config.config,),
                },
            )
        )
        outputs.append(
            pyine.configs.utils.make_config_description(
                datamodule_config_type,
                name=f"{datamodule_base_name}_TACO_10s10t_v1_part1to4",
                group=datamodule_base_config.group,
                description=(
                    "Specifies a subset consisting of parts 1 to 4 (of 26, so about 15%) of the "
                    "PyINE-TACO 10s10t v1 trace dataset. This is a small subset that can be useful for medium-sized"
                    "experiments, and should be quite representative of the full dataset's distribution."
                ),
                config={
                    "lmdb_paths": taco_10s10t_v1_paths[0:4],
                    # -------------
                    "builds_bases": (taco_10s10t_v1_config.config,),
                },
            )
        )
        outputs.append(
            taco_10s10t_v1_part1_config := pyine.configs.utils.make_config_description(
                datamodule_config_type,
                name=f"{datamodule_base_name}_TACO_10s10t_v1_part1",
                group=datamodule_base_config.group,
                description=(
                    "Specifies a subset consisting of the first part (of 26, so about 3.8%) of the "
                    "PyINE-TACO 10s10t v1 trace dataset. This is a much smaller subset than the full "
                    "dataset, but should be fairly representative of the full dataset's distribution, "
                    "and therefore useful for smaller-scale experiments."
                ),
                config={
                    "lmdb_paths": [taco_10s10t_v1_paths[0]],
                    # -------------
                    "builds_bases": (taco_10s10t_v1_config.config,),
                },
            )
        )
        outputs.append(
            pyine.configs.utils.make_config_description(
                datamodule_config_type,
                name=f"{datamodule_base_name}_TACO_10s10t_v1_part1_20s",
                group=datamodule_base_config.group,
                description=(
                    "Specifies a subset of the first part of the PyINE-TACO 10s10t v1 trace dataset, "
                    "with a maximum of 20 solutions per trace. Useful for quick experiments, testing, "
                    "and demos. Should not be used for anything serious."
                ),
                config={
                    "lmdb_paths": [taco_10s10t_v1_paths[0]],
                    "split_file_path": taco_split_path,
                    "max_solution_count": 20,  # cap off the dataset size for quick experiments (across each subset)
                    # -------------
                    "builds_bases": (taco_10s10t_v1_part1_config.config,),
                },
            )
        )

    return outputs
