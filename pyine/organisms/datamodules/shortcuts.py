import logging
import pathlib
import typing

import datasets as hf_datasets
import msgspec
import numpy as np
import pydantic

import pyine.data.datamodule
import pyine.data.traces.dataset_reader
import pyine.data.utils.filter_rules
import pyine.data.utils.splits
import pyine.prompts
import pyine.utils.filesystem
import pyine.utils.openai
import pyine.utils.portability
import pyine.utils.pydantic
import pyine.utils.reprod
from pyine.data.datamodule import SubsetNameType
from pyine.organisms.datamodules.utils.samples import (
    SampleBuilderConfig,
    SampleDataLoaderType,
    SampleDataParserType,
    SampleInputType,
    TraceDatasetMetadata,
    TraceMetadata,
    get_traces_metadata,
)
from pyine.organisms.datamodules.utils.transforms import (
    SampleTransformType,
    create_sample_transform,
)

logger = logging.getLogger(__name__)

ProblemIdType = str
"""Type def used to represent a coding problem identifier (for cleanliness)."""


class ShortcutBiasDataModule(pyine.data.datamodule.ConversationDataModule):
    """DataModule wrapping one or multiple PyINE code trace datasets for shortcut-bias experiments.

    This module loads one or more LMDB trace datasets, optionally filters available traces
    using flexible rules, validates that there are no duplicate trace identifiers across
    all selected samples, and finally creates simple random train/valid/test splits and loaders.

    Filtering is performed prior to concatenation and splitting. When multiple datasets are
    provided, they are concatenated in the provided order.

    Args:
        config: Configuration object for this datamodule.
    """

    def __init__(
        self,
        config: "ShortcutBiasDataModuleConfig",
        verbose: bool = False,
    ) -> None:
        super().__init__(config)
        self.verbose = verbose
        self.config: "ShortcutBiasDataModuleConfig" = config
        # attributes below are initialized in setup()
        self._metadata: TraceDatasetMetadata | None = None
        self._readers: list[pyine.data.traces.dataset_reader.DatasetReader] = []
        self._subset_parsers: dict[SubsetNameType, SampleDataParserType] = dict()

    @typing.override
    def prepare_data(self) -> None:
        """Prepares metadata and pre-filters traces, saving the results to a local tmpdir.

        Remember: this function should NOT be saving any state to the data module object, as it
        will only run on the main process.
        """
        lmdb_paths_str = "\n\t".join([str(p) for p in self.config.lmdb_paths])
        if self._is_metadata_prepared():
            logger.info(f"using cached shortcuts datamodule metadata for lmdb paths:\n\t{lmdb_paths_str}")
            return
        logger.info(f"preparing shortcuts datamodule metadata for lmdb paths:\n\t{lmdb_paths_str}")
        rng = np.random.default_rng(self.config.split_seed)
        readers = [pyine.data.traces.dataset_reader.DatasetReader(path) for path in self.config.lmdb_paths]
        # first prep step: identify which traces are to be kept based on our base tag filter rule
        base_filter = self.config._resolved_base_filter  # noqa
        assert base_filter is not None, "base filter should have been resolved by now"
        base_traces_meta = get_traces_metadata(
            readers=readers,
            base_filter=base_filter,
            verbose=self.verbose,
        )
        # load coding problem split data and keep relevant assignments
        split_hash = pyine.utils.reprod.compute_hash(self.config.split_file_path)
        split_data = pyine.data.utils.splits.SplitResult.from_file(self.config.split_file_path)
        if any([subset not in self.config.subset_types for subset in split_data.config.subset_names]):
            raise ValueError("mismatch between split data subsets and configured subsets")
        subset_traces_meta: dict[SubsetNameType, list[TraceMetadata]] = {
            subset_name: [] for subset_name in split_data.config.subset_names
        }
        unassigned_traces_meta: list[TraceMetadata] = []
        for trace_meta in base_traces_meta:
            problem_id = trace_meta.get_parent_problem_id()
            if problem_id in split_data.subset_assignments:
                subset_traces_meta[split_data.subset_assignments[problem_id]].append(trace_meta)
            else:
                unassigned_traces_meta.append(trace_meta)
        if self.config.max_trace_count is not None:
            for subset_name, traces_meta in subset_traces_meta.items():
                if len(traces_meta) > self.config.max_trace_count:
                    # if we have more traces than requested, pick a random subset of the available ones
                    # (also put the leftovers back into the unassigned list)
                    orig_idxs = list(range(len(traces_meta)))
                    picked_idxs = rng.choice(orig_idxs, size=self.config.max_trace_count, replace=False)
                    subset_traces_meta[subset_name] = [traces_meta[idx] for idx in picked_idxs]
                    unassigned_idxs = [idx for idx in orig_idxs if idx not in picked_idxs]
                    unassigned_traces_meta.extend([traces_meta[idx] for idx in unassigned_idxs])
        metadata = TraceDatasetMetadata(
            base_traces=base_traces_meta,
            subset_traces=subset_traces_meta,
            leftover_traces=unassigned_traces_meta,
            problem_assignments=split_data.subset_assignments,
            split_hash=split_hash,
        )
        logger.info(f"done; saving prepared metadata to: {self._get_prepared_metadata_file_path()}")
        self._save_prepared_metadata(metadata)

    def _is_metadata_prepared(self) -> bool:
        """Returns True if the metadata is prepared and ready to be used."""
        return self._get_prepared_metadata_file_path().is_file()

    def _save_prepared_metadata(self, metadata: TraceDatasetMetadata) -> None:
        """Saves the prepared metadata to a local tmpdir."""
        encoded_data = msgspec.msgpack.encode(metadata.model_dump())
        with open(self._get_prepared_metadata_file_path(), "wb") as fd:
            fd.write(encoded_data)

    def _load_prepared_metadata(self) -> TraceDatasetMetadata:
        """Loads the prepared metadata from a local tmpdir."""
        with open(self._get_prepared_metadata_file_path(), "rb") as fd:
            encoded_data = msgspec.msgpack.decode(fd.read())
        return TraceDatasetMetadata.model_validate(encoded_data)

    def _clear_prepared_metadata(self) -> None:
        """Clears the prepared metadata from a local tmpdir."""
        if self._is_metadata_prepared():
            self._get_prepared_metadata_file_path().unlink()

    def _get_prepared_metadata_file_path(self) -> pathlib.Path:
        """Returns the file path used to store prepared metadata in the local tmpdir."""
        # note: the file name that will be created contains a hash that depends on input params
        params_hash = pyine.utils.reprod.get_params_hash(self.config.model_dump())
        tmpdir = pyine.utils.filesystem.get_tmp_dir()
        return tmpdir / f"shortcuts.metadata.{params_hash}.msgspec"

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
        parser_types_str = "\n\t".join([str(s) for s in self.config.subset_types])
        logger.info(f"setting up shortcuts datamodule parsers:\n\t{parser_types_str}")
        self._metadata = self._load_prepared_metadata()
        # note: we share lmdb readers across all parsers here since they should be read-only and never pickled
        readers = [pyine.data.traces.dataset_reader.DatasetReader(path) for path in self.config.lmdb_paths]
        self._subset_parsers: dict[SubsetNameType, SampleDataParserType] = dict()
        for subset_type in self.config.subset_types:
            logger.debug(f"setting up shortcuts datamodule {subset_type} parser...")
            subset_traces = self._get_traces_meta_for_subset(subset_type)
            self._subset_parsers[subset_type] = self.config.instantiate_parser(
                subset_type=subset_type,
                source_data=readers,
                traces=subset_traces,
            )

    def _get_traces_meta_for_subset(self, subset_type: SubsetNameType | None) -> list[TraceMetadata]:
        """Returns the list of trace metadata objects for a given subset type."""
        if self._metadata is None:
            raise RuntimeError("metadata not ready yet, call `setup()` first")
        if subset_type is None:
            return self._metadata.base_traces
        if subset_type not in self._metadata.subset_traces:
            if not any([subset_type.endswith(f"_{suffix}") for suffix in typing.get_args(SampleInputType)]):
                raise ValueError(f"subset {subset_type} is not defined in the metadata's split table")
            parent_subset_type = subset_type.rsplit("_", maxsplit=1)[0]
            return self._metadata.subset_traces[parent_subset_type]
        else:
            return self._metadata.subset_traces[subset_type]

    def _is_setup_complete(self) -> bool:
        """Returns True if the setup is complete and the data parsers/loaders are ready to be used."""
        return self._metadata is not None

    @typing.override
    def get_parser(
        self,
        subset_type: SubsetNameType,
    ) -> SampleDataParserType:
        """Returns a data parser object for a given subset type, or for the full dataset (if None).

        This function exists for users that might not want to use dataloaders directly, and would prefer
        using the data parsers directly instead (e.g. to provide specific transforms, or to use them
        as part of a wider framework).
        """
        if not self._is_setup_complete():
            raise RuntimeError("data parsers are not ready yet, call `setup()` first")
        assert subset_type is not None, "subset type must be specified"
        if subset_type not in self._subset_parsers:
            raise ValueError(f"parser for subset {subset_type} is not defined")
        return self._subset_parsers[subset_type]

    @typing.override
    def get_hf_messages_dataset(
        self,
        subset_type: SubsetNameType,
        append_answer: bool = True,
        merge_system_with_user: bool = False,
    ) -> hf_datasets.Dataset:
        """Returns a HuggingFace dataset object for a given subset type."""
        if not self._is_setup_complete():
            raise RuntimeError("data parsers are not ready yet, call `setup()` first")
        assert subset_type is not None, "subset type must be specified"
        subset_traces = self._get_traces_meta_for_subset(subset_type)
        return self.config.instantiate_hf_messages_dataset(
            subset_type=subset_type,
            append_answer=append_answer,
            merge_system_with_user=merge_system_with_user,
            source_data=self.config.lmdb_paths.copy(),  # defer instantiation to the generator due to pickling
            traces=subset_traces,
        )

    @typing.override
    def get_openai_messages_dataset(
        self,
        subset_type: SubsetNameType,
        append_answer: bool = True,
        merge_system_with_user: bool = False,
    ) -> pathlib.Path:
        """Returns the path to an OpenAI-compatible JSONL dataset of chat-templated conversations."""
        if not self._is_setup_complete():
            raise RuntimeError("data parsers are not ready yet, call `setup()` first")
        assert subset_type is not None, "subset type must be specified"
        subset_traces = self._get_traces_meta_for_subset(subset_type)
        return self.config.instantiate_openai_messages_dataset(
            subset_type=subset_type,
            append_answer=append_answer,
            merge_system_with_user=merge_system_with_user,
            source_data=self.config.lmdb_paths.copy(),  # defer instantiation to the generator due to pickling
            traces=subset_traces,
        )

    def make_dataloader(
        self,
        loader_type: pyine.data.datamodule.LoaderNameType,
    ) -> SampleDataLoaderType:
        """Create a DataLoader for a given type."""
        assert loader_type is not None, "loader type must be specified"
        parser = self.get_parser(loader_type)
        return self.config.instantiate_dataloader(
            loader_type=loader_type,
            dataset=parser,
        )

    @typing.override
    def train_dataloader(self) -> SampleDataLoaderType:
        """Return the training data loader."""
        return self.make_dataloader("train")

    @typing.override
    def val_dataloader(self) -> SampleDataLoaderType:
        """Return the validation data loader."""
        return self.make_dataloader("valid")

    @typing.override
    def test_dataloader(self) -> SampleDataLoaderType:
        """Return the test data loader."""
        return self.make_dataloader("test")

    @typing.override
    def teardown(self, stage: str | None = None) -> None:
        """Close readers when the datamodule is torn down, and unassigns all parser attributes."""
        self._metadata: TraceDatasetMetadata | None = None
        self._subset_parsers: dict[SubsetNameType, SampleDataParserType] = dict()
        for r in self._readers:
            r.close()
        self._readers: list[pyine.data.traces.dataset_reader.DatasetReader] = []


def _get_supported_subset_types() -> tuple[SubsetNameType, ...]:
    """Returns all potential subset types supported by this datamodule.

    Ones that possess a suffix correspond to versions found by overriding parser settings.
    """
    output_subset_types = []
    for subset in ["train", "valid", "test"]:
        output_subset_types.append(subset)
        for suffix in typing.get_args(SampleInputType):
            if suffix != "original":
                output_subset_types.append(f"{subset}_{suffix}")
    return tuple(output_subset_types)


class ShortcutBiasDataModuleConfig(pyine.data.datamodule.ConversationDataModuleConfig):
    """Configuration class for the `ShortcutBiasDataModule`.

    Note: we override the base data module config class to add additional fields.
    """

    datamodule_class_path: str = pydantic.Field(
        default=pyine.utils.portability.get_fully_qualified_name(ShortcutBiasDataModule),
        frozen=True,
    )
    """Dotted import path to the target datamodule class, e.g. 'pkg.mod.MyImpl'."""

    # --------------- DATA PARSER / LOADER CONFIGURATIONS ---------------

    lmdb_paths: typing.Annotated[list[pathlib.Path], pydantic.Field(min_length=1)]  # must be specified!
    """Sequence of paths pointing to LMDB datasets containing execution traces."""
    subset_types: typing.Annotated[tuple[SubsetNameType, ...], pydantic.Field(min_length=1)] = (
        _get_supported_subset_types()  # should never need to override this default
    )
    """List of data subsets that the module supports; some subsets override sample selection strategy."""
    default_dataparser_config: SampleBuilderConfig = SampleBuilderConfig()
    """Default trace parser configuration (will rely on the TACO dataset if not overridden)."""
    dataparser_config_overrides: dict[SubsetNameType, dict[str, typing.Any]] = pydantic.Field(default_factory=dict)
    """Overrides for the default trace parser configuration; adds subset-specific transforms."""
    dataloader_config_overrides: dict[SubsetNameType, dict[str, typing.Any]] = pydantic.Field(default_factory=dict)
    """Overrides for the default DataLoader configuration; will shuffle training data."""

    # --------------- DATA FILTERING + SPLITTING CONFIGURATION ---------------

    max_trace_count: int | None = None
    """Maximum number of traces to load across all subsets (except the 'base' one)."""
    split_file_path: pathlib.Path  # must be specified!
    """Path to the file containing the split data for the full dataset.

    This file should have been created by the `pyine.apps.splits.dataset_splitter.py` module; it
    is expected to contain all coding problem identifiers that could be loaded by this datamodule.
    """
    base_filter_rule: str = ""  # empty = no filter by default
    """Base filter rule to apply to tags of all traces to determine what to include across all subsets.

    Traces with tags that match this rule will be filtered out. See the `pyine.data.utils.filter_rules`
    module to see examples of filter rules. Note that this rule applies in a case-insensitive manner.
    """

    # --------------- DATA TRANSFORMATION + COLLATE CONFIGURATION ---------------

    prompt_config: pyine.prompts.PromptBuildConfig = pyine.prompts.PromptBuildConfig(
        prompt_name="code_execution",
        use_chat_template=True,
        include_examples=True,
        target_examples=None,  # all
    )
    """Configuration for the prompt used to when transforming raw sample data to chat model requests."""

    chat_generator_config: dict[str, typing.Any] = dict(
        keep_in_memory=True,
    )
    """Configuration for the hf generator used to when transforming raw sample data to chat model requests."""

    # --------------- PUBLIC UTILITY FUNCTIONS ---------------

    @typing.override
    def instantiate_openai_messages_dataset(
        self,
        subset_type: SubsetNameType,
        append_answer: bool = True,
        merge_system_with_user: bool = False,
        **extra_kwargs,
    ) -> pathlib.Path:
        """Returns the path to an OpenAI-compatible JSONL dataset of chat-templated conversations."""
        openai_local_data_dir = pyine.utils.openai.get_local_file_directory()
        params_hash = pyine.utils.reprod.get_params_hash(
            self.model_dump(),
            append_answer,
            merge_system_with_user,
        )
        local_output_path = openai_local_data_dir / f"shortcuts-data.{subset_type}.{params_hash}.jsonl"
        if not local_output_path.is_file():
            # note: this impl relies on the huggingface getter (DRY)
            hf_dataset = self.instantiate_hf_messages_dataset(
                subset_type=subset_type,
                append_answer=append_answer,
                merge_system_with_user=merge_system_with_user,
                **extra_kwargs,
            )
            pyine.utils.openai.write_dataset_to_jsonl(hf_dataset, local_output_path)
        return local_output_path

    @typing.override
    def get_sample_to_messages_transform(
        self,
        append_answer: bool = True,
        use_hf_messages: bool = False,
        merge_system_with_user: bool = False,
    ) -> SampleTransformType:
        """Returns the sample transform function used to prepare training/evaluation conversations."""
        return create_sample_transform(
            append_answer=append_answer,
            use_hf_messages=use_hf_messages,
            merge_system_with_user=merge_system_with_user,
            **self.prompt_config.model_dump(),
        )

    # --------------- PRIVATE UTILITY FUNCTIONS & ATTRIBUTES ---------------

    _resolved_base_filter: pyine.data.utils.filter_rules.FilterType | None = pydantic.PrivateAttr(
        default=None,
    )

    @typing.override
    def _resolve_dataparser_config(
        self,
        subset_type: SubsetNameType,
    ) -> SampleBuilderConfig:
        """Returns the data parser configuration for the given subset type."""
        if subset_type not in self.subset_types:
            raise ValueError(f"invalid subset type: {subset_type}, expected one of: {self.subset_types}")
        parser_config = self.default_dataparser_config
        assert isinstance(parser_config, SampleBuilderConfig)
        if subset_type in self.dataparser_config_overrides and self.dataparser_config_overrides[subset_type]:
            parser_config = parser_config.get_updated_spec(**self.dataparser_config_overrides[subset_type])
        special_subset_overrides = parser_config.get_special_subset_param_overrides(subset_type)
        if special_subset_overrides:
            parser_config = parser_config.get_updated_spec(**special_subset_overrides)
        return parser_config

    @typing.override
    def _resolve_dataloader_config(
        self,
        loader_type: pyine.data.datamodule.LoaderNameType,
    ) -> pyine.data.datamodule.BaseDataLoaderConfig:
        """Returns the data loader configuration for the given loader type."""
        if loader_type not in self.loader_types:
            raise ValueError(f"invalid loader type: {loader_type}, expected one of: {self.loader_types}")
        loader_config: pyine.data.datamodule.BaseDataLoaderConfig = self.default_dataloader_config
        assert isinstance(loader_config, pyine.utils.pydantic.ClassImportSpec)
        if any([loader_type.endswith(f"_{suffix}") for suffix in typing.get_args(SampleInputType)]):
            loader_type = loader_type.rsplit("_", maxsplit=1)[0]
        if loader_type in self.dataloader_config_overrides and self.dataloader_config_overrides[loader_type]:
            loader_config = loader_config.get_updated_spec(**self.dataloader_config_overrides[loader_type])  # noqa
        return loader_config

    @pydantic.model_validator(mode="after")
    @typing.override
    def _validate_and_resolve(self) -> "ShortcutBiasDataModuleConfig":
        """Validates and resolves dataset paths and internal filtering rules."""
        super()._validate_and_resolve()
        for lmdb_path in self.lmdb_paths:
            if not lmdb_path.exists():
                raise ValueError(f"LMDB dataset does not exist at path: {lmdb_path}")
        self._resolved_base_filter = pyine.data.utils.filter_rules.build_filter_from_rule(
            self.base_filter_rule, case_sensitive=False
        )
        if not self.split_file_path.is_file():
            raise ValueError(f"dataset split file does not exist at path: {self.split_file_path}")
        for subset_type in self.dataparser_config_overrides.keys():
            if any([subset_type.endswith(f"_{suffix}") for suffix in typing.get_args(SampleInputType)]):
                raise ValueError(f"invalid subset type: {subset_type}, cannot override special parsers")
        for loader_type in self.dataloader_config_overrides.keys():
            if any([loader_type.endswith(f"_{suffix}") for suffix in typing.get_args(SampleInputType)]):
                raise ValueError(f"invalid loader type: {loader_type}, cannot override special loaders")
        return self
