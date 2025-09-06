import collections
import pathlib
import typing

import datasets as hf_datasets
import msgspec
import numpy as np
import pydantic

import pyine.data.datamodule
import pyine.data.traces.dataset_reader as dataset_reader
import pyine.data.utils.filter_rules
import pyine.data.utils.splits
import pyine.organisms.datamodules.utils.samples
import pyine.organisms.datamodules.utils.transforms
import pyine.organisms.models.utils.openai
import pyine.prompts
import pyine.utils.code.execution
import pyine.utils.filesystem
import pyine.utils.portability
import pyine.utils.reprod
from pyine.organisms.datamodules.utils.samples import (
    SampleBuilderConfig,
    SampleDataLoaderType,
    SampleDataParserType,
    SampleTransformConfig,
    TraceDatasetMetadata,
    TraceMetadata,
)

SubsetNameType = pyine.data.datamodule.SubsetNameType
"""Type used to represent a data subset name (e.g. 'train', 'valid', 'test')."""
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
        self._readers: list[dataset_reader.DatasetReader] = []
        self._base_parser: SampleDataParserType | None = None
        self._subset_parsers: dict[SubsetNameType, SampleDataParserType] = dict()

    def prepare_data(self) -> None:
        """Prepares metadata and pre-filters traces, saving the results to a local tmpdir.

        Remember: this function should NOT be saving any state to the data module object, as it
        will only run on the main process.
        """
        if self._is_metadata_prepared():
            return  # already prepared, nothing more to do
        rng = np.random.default_rng(self.config.split_seed)
        readers = [pyine.data.traces.dataset_reader.DatasetReader(path) for path in self.config.lmdb_paths]
        # first prep step: identify which traces are to be kept based on our base tag filter rule
        base_filter = self.config._resolved_base_filter  # noqa
        assert base_filter is not None, "base filter should have been resolved by now"
        base_traces_meta = pyine.organisms.datamodules.utils.samples.get_traces_metadata(
            readers=readers,
            base_filter=base_filter,
            verbose=self.verbose,
        )
        if self.config.max_trace_count is not None and len(base_traces_meta) > self.config.max_trace_count:
            # if we have more traces than requested, pick a random subset of the available ones
            orig_idxs = list(range(len(base_traces_meta)))
            picked_idxs = rng.choice(orig_idxs, size=self.config.max_trace_count, replace=False)
            base_traces_meta = [base_traces_meta[idx] for idx in picked_idxs]
        # load coding problem split data and keep relevant assignments
        split_hash = pyine.utils.reprod.compute_hash(self.config.split_file_path)
        split_data = pyine.data.utils.splits.SplitResult.from_file(self.config.split_file_path)
        if split_data.config.subset_names != self.config.subset_types:
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
        metadata = TraceDatasetMetadata(
            base_traces=base_traces_meta,
            subset_traces=subset_traces_meta,
            leftover_traces=unassigned_traces_meta,
            problem_assignments=split_data.subset_assignments,
            split_hash=split_hash,
        )
        self._save_prepared_metadata(metadata)
        # @@@@@@ TODO extra step: tag-stratified split w/ clustering (?)

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
        # note: we share lmdb readers across all parsers here since they should be read-only and never pickled
        readers = [pyine.data.traces.dataset_reader.DatasetReader(path) for path in self.config.lmdb_paths]
        self._base_parser = self.config.default_dataparser_config.instantiate(readers, self._metadata.base_traces)
        self._subset_parsers = dict()
        for subset_name, subset_traces in self._metadata.subset_traces.items():
            self._subset_parsers[subset_name] = self.config.default_dataparser_config.instantiate(
                source_data=readers,
                traces=subset_traces,
                **self.config.dataparser_config_overrides.get(subset_name, {}),
            )

    def _is_setup_complete(self) -> bool:
        """Returns True if the setup is complete and the data parsers/loaders are ready to be used."""
        return self._base_parser is not None

    def get_parser(
        self,
        subset_type: SubsetNameType | None = None,
    ) -> SampleDataParserType:
        """Returns a data parser object for a given subset type, or for the full dataset (if None).

        This function exists for users that might not want to use dataloaders directly, and would prefer
        using the data parsers directly instead (e.g. to provide specific transforms, or to use them
        as part of a wider framework).
        """
        if not self._is_setup_complete():
            raise RuntimeError("data parsers are not ready yet, call `setup()` first")
        if subset_type is None:
            return self._base_parser
        if subset_type not in self._subset_parsers:
            raise ValueError(f"parser for subset {subset_type} is not defined")
        return self._subset_parsers[subset_type]

    def get_sample_to_messages_transform(
        self,
        append_answer: bool = True,
        use_hf_messages: bool = False,
        merge_system_with_user: bool = False,
    ) -> pyine.organisms.datamodules.utils.transforms.SampleTransformType:
        """Returns the sample transform function used to prepare training/evaluation conversations."""
        return pyine.organisms.datamodules.utils.transforms.create_sample_transform(
            append_answer=append_answer,
            use_hf_messages=use_hf_messages,
            merge_system_with_user=merge_system_with_user,
            **self.config.prompt_config.model_dump(),
        )

    def get_hf_messages_dataset(
        self,
        subset_type: SubsetNameType | None = None,
        append_answer: bool = True,
        merge_system_with_user: bool = False,
    ) -> hf_datasets.Dataset:
        """Returns a HuggingFace dataset object for a given subset type or for the full dataset (if None).

        This function exists for users that might not want to use dataloaders directly, and would prefer
        using the data parsers for huggingface-based experiments.

        Args:
            subset_type: the subset type to prepare the dataset for.
            append_answer: whether to append the assistant's response to the conversation messages.
            merge_system_with_user: whether to merge the system message with the user message (used
                when working with e.g. o1/o3/o4, which do not support custom system prompts).

        Returns:
            A huggingface dataset object that produces chat-templated 'conversations' containing
            requests to be executed by a model.
        """
        if not self._is_setup_complete():
            raise RuntimeError("data parsers are not ready yet, call `setup()` first")
        if subset_type is not None:
            if subset_type not in self._metadata.subset_traces:
                raise ValueError(f"parser for subset {subset_type} is not defined")
            named_split = hf_datasets.NamedSplit(name=subset_type)
            subset_traces = self._metadata.subset_traces[subset_type]
        else:
            named_split = hf_datasets.NamedSplit(name="all")
            subset_traces = self._metadata.base_traces
        transf_fn = self.get_sample_to_messages_transform(
            append_answer=append_answer,
            use_hf_messages=True,
            merge_system_with_user=merge_system_with_user,
        )
        return self.config.default_dataparser_config.get_hf_messages_dataset(
            named_split=named_split,
            raw_transform_fn=transf_fn,
            instantiate_kwargs=dict(
                source_data=self.config.lmdb_paths.copy(),  # defer instantiation to the generator due to pickling
                traces=subset_traces,
            ),
            generator_kwargs=self.config.chat_generator_config,
        )

    def get_openai_messages_dataset(
        self,
        subset_type: SubsetNameType | None = None,
        append_answer: bool = True,
        merge_system_with_user: bool = False,
    ) -> pathlib.Path:
        """Returns the path to an OpenAI-compatible JSONL dataset of chat-templated conversations.

        This function exists for users that might want to use the data in combination with the
        OpenAI API. The dataset is written to the returned path in the OpenAI format, which is
        a JSONL file with one example per line. That dataset file can then be uploaded to the
        OpenAI API to train/validate a model.

        Args:
            subset_type: the subset type to prepare the dataset for.
            append_answer: whether to append the assistant's response to the conversation messages.
            merge_system_with_user: whether to merge the system message with the user message (used
                when working with e.g. o1/o3/o4, which do not support custom system prompts).

        Returns:
             The path to the written dataset, which can be used for uploads to the OpenAI API.
        """
        openai_local_data_dir = pyine.organisms.models.utils.openai.get_local_file_directory()
        params_hash = pyine.utils.reprod.get_params_hash(
            self.config.model_dump(),
            append_answer,
            merge_system_with_user,
        )
        subset_type_name = subset_type if subset_type is not None else "all"
        local_output_path = openai_local_data_dir / f"shortcuts-data.{subset_type_name}.{params_hash}.jsonl"
        if not local_output_path.is_file():
            # note: this impl relies on the huggingface getter (DRY)
            hf_dataset = self.get_hf_messages_dataset(
                subset_type=subset_type,
                append_answer=append_answer,
                merge_system_with_user=merge_system_with_user,
            )
            pyine.organisms.models.utils.openai.write_dataset_to_jsonl(hf_dataset, local_output_path)
        return local_output_path

    def _make_dataloader(
        self,
        subset_type: SubsetNameType | None,
    ) -> SampleDataLoaderType:
        """Create a DataLoader for a given data parser."""
        parser = self.get_parser(subset_type)
        return self.config.default_dataloader_config.instantiate(
            dataset=parser,
            **self.config.dataloader_config_overrides.get(subset_type, {}),
        )

    def train_dataloader(self) -> SampleDataLoaderType:
        """Return the training data loader."""
        return self._make_dataloader("train")

    def val_dataloader(self) -> SampleDataLoaderType:
        """Return the validation data loader."""
        return self._make_dataloader("valid")

    def test_dataloader(self) -> SampleDataLoaderType:
        """Return the test data loader."""
        return self._make_dataloader("test")

    def teardown(self, stage: str | None = None) -> None:
        """Close readers when the datamodule is torn down, and unassigns all parser attributes."""
        self._metadata: TraceDatasetMetadata | None = None
        self._base_parser: SampleDataParserType | None = None
        self._subset_parsers: dict[SubsetNameType, SampleDataParserType] = dict()
        for r in self._readers:
            r.close()
        self._readers: list[dataset_reader.DatasetReader] = []


class ShortcutBiasDataModuleConfig(pyine.data.datamodule.ConversationDataModuleConfig):
    """Configuration class for the `ShortcutBiasDataModule`.

    Note: we override the base data module config class to add additional fields.
    """

    datamodule_class_path: str = pydantic.Field(
        default=pyine.utils.portability.get_fully_qualified_name(ShortcutBiasDataModule),
        frozen=True,
        description="Dotted import path to the target datamodule class, e.g. 'pkg.mod.MyImpl'.",
    )

    # --------------- DATA PARSER / LOADER CONFIGURATIONS ---------------

    lmdb_paths: typing.Annotated[
        list[pathlib.Path],
        pydantic.Field(
            min_length=1, description="Sequence of paths pointing to LMDB datasets containing execution traces."
        ),
    ]

    default_dataparser_config: SampleBuilderConfig = SampleBuilderConfig()
    """Default trace parser configuration (will rely on the TACO dataset if not overridden)."""
    dataparser_config_overrides: dict[SubsetNameType, dict[str, typing.Any]] = dict(
        train=dict(
            config=SampleTransformConfig(
                random_seed=0,
                partial_sample_decision_strategy="hybrid",
                output_type_prob_map={
                    "program output": 0.5,
                    "frame variables": 0.1,
                    "function return": 0.4,
                },
            ),
        ),  # other subsets will default to never producing partial samples
    )
    """Overrides for the default trace parser configuration; adds subset-specific transforms."""
    dataloader_config_overrides: dict[SubsetNameType, dict[str, typing.Any]] = dict(
        train=dict(shuffle=True),
    )
    """Overrides for the default DataLoader configuration; will shuffle training data."""

    # --------------- DATA FILTERING + SPLITTING CONFIGURATION ---------------

    max_trace_count: int | None = None
    """Maximum number of traces to load across all datasets."""
    split_file_path: pathlib.Path
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

    # --------------- PRIVATE UTILITY FUNCTIONS & ATTRIBUTES ---------------

    _resolved_base_filter: pyine.data.utils.filter_rules.FilterType | None = pydantic.PrivateAttr(
        default=None,
    )

    @pydantic.model_validator(mode="after")
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
        return self
