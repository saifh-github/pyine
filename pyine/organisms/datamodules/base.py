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
import pyine.data.utils.splits
import pyine.evals.common
import pyine.organisms.datamodules.samples
import pyine.organisms.datamodules.utils.transforms
import pyine.prompts.types
import pyine.utils.distrib
import pyine.utils.filesystem
import pyine.utils.portability
import pyine.utils.reprod

if typing.TYPE_CHECKING:
    import datasets as hf_datasets

logger = logging.getLogger(__name__)


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

    # --------------- DATA TRANSFORMATION + COLLATE CONFIGURATION ---------------

    prompt_config: pyine.prompts.types.PromptBuildConfig = pydantic.Field(
        default_factory=lambda: pyine.prompts.types.PromptBuildConfig(
            prompt_name=pyine.prompts.PromptNames.CODE_EXECUTION,
            use_chat_template=True,
            include_examples=True,
        )
    )
    """Configuration for the prompt used when transforming raw sample data to chat model requests."""

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
            orig_sample_key=orig_data_key if keep_original_data else None,
            **self.prompt_config.model_dump(),
        )

    @typing.override
    def instantiate_sample_to_rl_transform(
        self,
        prompt_version: str | None = None,
        include_examples: bool = False,
    ) -> pyine.organisms.datamodules.utils.transforms.SampleTransformType:
        """Returns the sample transform function used to prepare RL training data."""
        return pyine.organisms.datamodules.utils.transforms.create_rl_sample_transform(
            prompt_version=prompt_version,
            include_examples=include_examples,
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
        verbose: bool = False,
    ) -> None:
        """Initializes a new BiasDataModuleBase instance."""
        super().__init__(config)
        self.verbose = verbose
        self._metadata = None
        self._readers = []
        self._subset_parsers = {}

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
            solution_ids = list(set(tidxs_to_sids.values()))
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
        params_hash = pyine.utils.reprod.get_params_hash(self.config.model_dump())
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
                source_data=self._readers,
                traces=subset_traces,
            )
            self._subset_parsers[subset_name] = typing.cast(
                "pyine.organisms.datamodules.samples.SampleBuilder",
                parser,
            )
        parser = self._subset_parsers[subset_name]
        assert parser is not None, f"parser for subset {subset_name} should be instantiated"
        return parser

    @abc.abstractmethod
    def _get_subset_suffixes(self) -> tuple[str, ...]:
        """Returns the suffixes that this datamodule may expect to see appended to subset names."""
        ...

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

    def _is_setup_complete(self) -> bool:
        """Returns True if the setup is complete and the data parsers/loaders are ready to be used."""
        return self._metadata is not None

    # --------------- LIGHTNING DATAMODULE LIFECYCLE ---------------

    @typing.override
    def prepare_data(self) -> None:
        """Prepares metadata and pre-filters traces, saving the results to a local tmpdir.

        Remember: this function should NOT be saving any state to the data module object, as it
        will only run on the main process.
        """
        lmdb_paths_str = "\n\t".join([str(p) for p in self.config.lmdb_paths])
        cache_name = self._get_cache_subdirectory_name()
        if self._is_metadata_prepared() or not pyine.utils.distrib.is_main_process():
            if pyine.utils.distrib.is_main_process():
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
        self._save_prepared_metadata(metadata)

    @typing.override
    def setup(
        self,
        stage: str | None = None,
    ) -> None:
        """Loads the prepared metadata and creates train/valid/test data readers.

        Args:
            stage: Optional stage indicator provided by Lightning; not used here.
        """
        if not self._is_metadata_prepared():
            raise RuntimeError("metadata is not prepared yet, call `prepare_data()` on main process first")
        self._metadata = self._load_prepared_metadata()
        readers: list[pyine.data.traces.dataset_reader.DatasetProtocol] = [
            pyine.data.traces.dataset_reader.DatasetReader(path) for path in self.config.lmdb_paths
        ]
        self._readers = readers
        self._subset_parsers.clear()
        for subset_name in self.config.subset_names:
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
        self._readers = []

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
        for subset_name in target_subsets or list(self._subset_parsers.keys()):
            parser = self._instantiate_parser_if_needed(subset_name)
            for stat_key, stat_val in parser.get_stats().items():
                stats[f"{subset_name}/{stat_key}"] = stat_val
        return stats

    @typing.override
    def get_parser(
        self,
        subset_name: pyine.data.datamodule.SubsetNameType,
    ) -> pyine.organisms.datamodules.samples.SampleBuilder:
        """Returns a data parser object for a given subset name."""
        if not self._is_setup_complete():
            raise RuntimeError("data parsers are not ready yet, call `setup()` first")
        assert subset_name is not None, "subset name must be specified"
        if subset_name not in self._subset_parsers:
            raise ValueError(f"parser for subset {subset_name} is not defined")
        return self._instantiate_parser_if_needed(subset_name)

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
            parser_kwargs={
                "source_data": self.config.lmdb_paths,
                "traces": subset_traces,
            },
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
            parser_kwargs={
                "source_data": self.config.lmdb_paths,
                "traces": subset_traces,
            },
        )

    @typing.override
    def get_hf_rl_dataset(
        self,
        subset_name: pyine.data.datamodule.SubsetNameType,
        prompt_version: str | None = None,
        include_examples: bool = False,
        force_regenerate: bool = False,
    ) -> hf_datasets.Dataset:
        """Returns a HuggingFace dataset for RL training with raw prompts and metadata.

        This method prepares datasets for reinforcement learning training by providing raw
        prompts (not tokenized) along with metadata needed for reward computation.

        Args:
            subset_name: the subset name to prepare the dataset for.
            prompt_version: Version of the prompt template to use. If None, uses the version
                from the datamodule config.
            include_examples: Whether to include few-shot examples in prompts.
            force_regenerate: whether to rebuild caches even if they already exist.

        Returns:
            The HuggingFace dataset object with raw prompts and metadata for RL training.
        """
        if not self._is_setup_complete():
            raise RuntimeError("data parsers are not ready yet, call `setup()` first")
        assert subset_name is not None, "subset name must be specified"
        subset_traces = self._get_traces_meta_for_subset(subset_name)
        return self.config.instantiate_hf_rl_dataset(
            subset_name=subset_name,
            prompt_version=prompt_version,
            include_examples=include_examples,
            parser_kwargs={
                "source_data": self.config.lmdb_paths,
                "traces": subset_traces,
            },
            force_regenerate=force_regenerate,
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
    training_selection_config: dict[str, typing.Any] | None = None,
    use_hybrid_sample_transforms: bool = False,
    extra_config: dict[str, typing.Any] | None = None,
    as_pydantic: bool = False,
) -> dict[str, typing.Any] | ConfigT:
    """Generic factory for creating bias datamodule configs.

    This function provides a shared implementation for creating datamodule configs
    for both KeywordBiasDataModule and ShortcutBiasDataModule (and future bias modules).

    Args:
        config_class: The config class to instantiate (e.g., KeywordBiasDataModuleConfig).
        lmdb_paths: Paths to LMDB datasets containing execution traces.
        split_file_path: Path to the problem split file.
        seed: Random seed for reproducibility.
        sample_builder_config: Config to use for the default sample builder.
        training_selection_config: Optional selection config for training subset overrides.
        use_hybrid_sample_transforms: If True, use hybrid transforms (for full+partial samples).
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
        "dataparser_config_overrides": {
            subset: get_default_sample_builder_overrides_for_subset(
                subset_name=subset,
                use_hybrid_transform=use_hybrid_sample_transforms,
                training_selection_config=training_selection_config,
            )
            for subset in get_default_subset_names()
        },
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
        warnings.warn(
            "no TACO base dataset found, skipping TACO configs; "
            "if you intended to use TACO dataset demos/tests, please ensure that a dataset is present "
            "in the expected location (see the top-level README for more details)",
            stacklevel=2,
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
