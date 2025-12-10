"""Base class and config for bias-related datamodules with shared metadata caching logic."""

from __future__ import annotations

import abc
import collections.abc
import contextlib
import itertools
import logging
import os
import pathlib  # noqa: TC003
import tempfile
import typing

import filelock
import msgspec
import numpy as np
import pydantic

import pyine.data.datamodule
import pyine.data.traces.dataset_reader
import pyine.data.traces.dataset_utils
import pyine.data.utils.filter_rules
import pyine.data.utils.splits
import pyine.organisms.datamodules.samples
import pyine.organisms.datamodules.utils.transforms
import pyine.prompts.types
import pyine.utils.distrib
import pyine.utils.filesystem
import pyine.utils.reprod

if typing.TYPE_CHECKING:
    import datasets as hf_datasets

logger = logging.getLogger(__name__)


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

    subset_names: typing.Annotated[tuple[pyine.data.datamodule.SubsetNameType, ...], pydantic.Field(min_length=1)]
    """List of data subsets that the module supports."""
    eval_subset_names: tuple[str, ...]
    """Subset names that are meant for model evaluation."""

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

    def _get_traces_meta_for_subset(
        self,
        subset_name: pyine.data.datamodule.SubsetNameType | None,
    ) -> list[pyine.data.traces.dataset_utils.TraceMetadata]:
        """Returns the list of trace metadata objects for a given subset name."""
        if self._metadata is None:
            raise RuntimeError("metadata not ready yet, call `setup()` first")
        if subset_name is None:
            return self._metadata.base_traces
        known_subsets = list(self._metadata.subset_traces.keys())
        if subset_name not in known_subsets:
            for prefix, suffix in itertools.product(
                known_subsets,
                pyine.organisms.datamodules.samples.get_all_supported_code_type_sets_suffixes(),
            ):
                if f"{prefix}_{suffix}" == subset_name:
                    return self._metadata.subset_traces[prefix]
            raise ValueError(f"subset {subset_name} is not defined in the metadata's split table")
        return self._metadata.subset_traces[subset_name]

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


def get_default_subset_names() -> tuple[str, ...]:
    """Returns the default subset names used by this datamodule."""
    return "train", "valid", "test"
