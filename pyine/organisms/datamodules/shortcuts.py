import itertools
import logging
import pathlib
import typing

import datasets as hf_datasets
import msgspec
import numpy as np
import transformers

import pyine.data.datamodule
import pyine.data.traces.dataset_reader
import pyine.data.traces.dataset_utils
import pyine.data.utils.splits
import pyine.organisms.datamodules.utils.samples
import pyine.organisms.datamodules.utils.transforms
import pyine.utils.filesystem
import pyine.utils.reprod

if typing.TYPE_CHECKING:
    from pyine.organisms.datamodules.shortcuts_configs import ShortcutBiasDataModuleConfig


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
        self._metadata: pyine.data.traces.dataset_utils.TraceDatasetMetadata | None = None
        self._readers: list[pyine.data.traces.dataset_reader.DatasetReader] = []
        self._subset_parsers: dict[
            pyine.data.datamodule.SubsetNameType,
            pyine.organisms.datamodules.utils.samples.SampleBuilder | None,
        ] = dict()

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
        readers = [pyine.data.traces.dataset_reader.DatasetReader(path) for path in self.config.lmdb_paths]
        # first prep step: identify which traces are to be kept based on our base tag filter rule
        base_filter = self.config._resolved_base_filter  # noqa
        assert base_filter is not None, "base filter should have been resolved by now"
        base_traces_meta = pyine.data.traces.dataset_reader.get_traces_metadata(readers, base_filter=base_filter)
        # load coding problem split data and keep relevant assignments
        split_hash = pyine.utils.reprod.compute_hash(self.config.split_file_path)
        split_data = pyine.data.utils.splits.SplitResult.from_file(self.config.split_file_path)
        if any([subset not in self.config.subset_names for subset in split_data.config.subset_names]):
            raise ValueError("mismatch between split data subsets and configured subsets")
        subset_traces_meta: dict[
            pyine.data.datamodule.SubsetNameType,
            list[pyine.data.traces.dataset_utils.TraceMetadata],
        ] = {subset_name: [] for subset_name in split_data.config.subset_names}
        unassigned_traces_meta: list[pyine.data.traces.dataset_utils.TraceMetadata] = []
        for trace_meta in base_traces_meta:
            if str(trace_meta.problem_id) in split_data.subset_assignments:
                subset_traces_meta[split_data.subset_assignments[str(trace_meta.problem_id)]].append(trace_meta)
            else:
                unassigned_traces_meta.append(trace_meta)
        self._apply_max_solution_count_cap(subset_traces_meta, unassigned_traces_meta)
        metadata = pyine.data.traces.dataset_utils.TraceDatasetMetadata(
            base_traces=base_traces_meta,
            subset_traces=subset_traces_meta,
            leftover_traces=unassigned_traces_meta,
            problem_assignments=split_data.subset_assignments,
            augment_types=pyine.organisms.datamodules.utils.samples.get_supported_augment_types(),
            split_hash=split_hash,
        )
        logger.info(f"done; saving prepared metadata to: {self._get_prepared_metadata_file_path()}")
        self._save_prepared_metadata(metadata)

    def _apply_max_solution_count_cap(
        self,
        subset_traces_meta: dict[
            pyine.data.datamodule.SubsetNameType, list[pyine.data.traces.dataset_utils.TraceMetadata]
        ],
        unassigned_traces_meta: list[pyine.data.traces.dataset_utils.TraceMetadata],
    ) -> None:  # updates to the provided args are done in-place
        if self.config.max_solution_count is not None:
            rng = np.random.default_rng(self.config.split_seed)
            for subset_name, traces_meta in subset_traces_meta.items():
                tidxs_to_sids = {tidx: str(tm.solution_id) for tidx, tm in enumerate(traces_meta)}
                solution_ids = list(set(tidxs_to_sids.values()))
                if len(solution_ids) > self.config.max_solution_count:
                    # if we have more solutions than requested, pick a random subset of the available ones
                    picked_solution_ids = rng.choice(
                        solution_ids,
                        size=self.config.max_solution_count,
                        replace=False,
                    )
                    # find the associated traces for all picked solutions
                    picked_trace_meta_idxs = [
                        trace_meta_idx
                        for trace_meta_idx, solution_id in tidxs_to_sids.items()
                        if solution_id in picked_solution_ids
                    ]
                    subset_traces_meta[subset_name] = [traces_meta[idx] for idx in picked_trace_meta_idxs]
                    # (also put the leftovers back into the unassigned list)
                    unassigned_idxs = [idx for idx in tidxs_to_sids if idx not in picked_trace_meta_idxs]
                    unassigned_traces_meta.extend([traces_meta[idx] for idx in unassigned_idxs])

    # TODO: if we create more data modules that are based on trace datasets, add a common interf for metadata stuff

    def _is_metadata_prepared(self) -> bool:
        """Returns True if the metadata is prepared and ready to be used."""
        return self._get_prepared_metadata_file_path().is_file()

    def _save_prepared_metadata(self, metadata: pyine.data.traces.dataset_utils.TraceDatasetMetadata) -> None:
        """Saves the prepared metadata to a local tmpdir."""
        encoded_data = msgspec.msgpack.encode(metadata.model_dump())
        with open(self._get_prepared_metadata_file_path(), "wb") as fd:
            fd.write(encoded_data)

    def _load_prepared_metadata(self) -> pyine.data.traces.dataset_utils.TraceDatasetMetadata:
        """Loads the prepared metadata from a local tmpdir."""
        with open(self._get_prepared_metadata_file_path(), "rb") as fd:
            encoded_data = msgspec.msgpack.decode(fd.read())
        return pyine.data.traces.dataset_utils.TraceDatasetMetadata.model_validate(encoded_data)

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
        self._metadata = self._load_prepared_metadata()
        # note: we share lmdb readers across all parsers since they should be read-only and never pickled
        self._readers = [pyine.data.traces.dataset_reader.DatasetReader(path) for path in self.config.lmdb_paths]
        self._subset_parsers: dict[
            pyine.data.datamodule.SubsetNameType,
            pyine.organisms.datamodules.utils.samples.SampleBuilder | None,
        ] = dict()
        for subset_name in self.config.subset_names:
            if self.config.instantiate_parsers_at_setup:
                self._subset_parsers[subset_name] = self._instantiate_parser_if_needed(subset_name)
            else:
                self._subset_parsers[subset_name] = None  # instantiation deferred to first use

    def _instantiate_parser_if_needed(
        self,
        subset_name: pyine.data.datamodule.SubsetNameType,
    ) -> pyine.organisms.datamodules.utils.samples.SampleBuilder:
        """Instantiates a parser for a given subset name if it has not been instantiated yet."""
        if self._subset_parsers[subset_name] is None:
            logger.debug(f"instantiating shortcuts datamodule {subset_name} parser...")
            subset_traces = self._get_traces_meta_for_subset(subset_name)
            parser = self.config.instantiate_parser(
                subset_name=subset_name,
                source_data=self._readers,
                traces=subset_traces,
            )
            self._subset_parsers[subset_name] = typing.cast(
                pyine.organisms.datamodules.utils.samples.SampleBuilder,
                parser,
            )
        return self._subset_parsers[subset_name]

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
                typing.get_args(pyine.organisms.datamodules.utils.samples.SampleInputType),
            ):
                if f"{prefix}_{suffix}" == subset_name:
                    return self._metadata.subset_traces[prefix]
            raise ValueError(f"subset {subset_name} is not defined in the metadata's split table")
        else:
            return self._metadata.subset_traces[subset_name]

    def _is_setup_complete(self) -> bool:
        """Returns True if the setup is complete and the data parsers/loaders are ready to be used."""
        return self._metadata is not None

    @typing.override
    def get_stats(
        self,
        target_subsets: list[pyine.data.datamodule.SubsetNameType] | None = None,
    ) -> dict[str, int | float | str]:
        """Returns a dictionary of useful-to-log statistics."""
        if not self._is_setup_complete():
            raise RuntimeError("data parsers are not ready yet, call `setup()` first")
        stats = dict()
        for subset_name in target_subsets or list(self._subset_parsers.keys()):
            parser = self._instantiate_parser_if_needed(subset_name)
            for stat_key, stat_vaL in parser.get_stats().items():
                stats[f"{subset_name}/{stat_key}"] = stat_vaL
        return stats

    @typing.override
    def get_parser(
        self,
        subset_name: pyine.data.datamodule.SubsetNameType,
    ) -> pyine.organisms.datamodules.utils.samples.SampleBuilder:
        """Returns a data parser object for a given subset name.

        This function exists for users that might not want to use dataloaders directly, and would prefer
        using the data parsers directly instead (e.g. to provide specific transforms, or to use them
        as part of a wider framework).
        """
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
        tokenizer: transformers.PreTrainedTokenizer | None = None,
        apply_chat_template_eval_config: bool = False,
    ) -> hf_datasets.Dataset:
        """Returns a HuggingFace dataset object for a given subset name."""
        if not self._is_setup_complete():
            raise RuntimeError("data parsers are not ready yet, call `setup()` first")
        assert subset_name is not None, "subset name must be specified"
        subset_traces = self._get_traces_meta_for_subset(subset_name)
        hf_dataset = self.config.instantiate_hf_messages_dataset(
            subset_name=subset_name,
            append_answer=append_answer,
            merge_system_with_user=merge_system_with_user,
            keep_original_data=keep_original_data,
            parser_kwargs=dict(
                source_data=self.config.lmdb_paths,  # defer instantiation to the generator due to pickling
                traces=subset_traces,
            ),
        )
        if tokenizer is not None:
            if apply_chat_template_eval_config:
                chat_tmpl_config = self.config.apply_chat_template_eval_config
            else:
                chat_tmpl_config = self.config.apply_chat_template_train_config
            hf_dataset = pyine.organisms.datamodules.utils.transforms.apply_model_template_to_messages(
                hf_messages_dataset=hf_dataset,
                tokenizer=tokenizer,
                keep_original_data=keep_original_data,
                apply_chat_template_kwargs=chat_tmpl_config,
                keep_in_memory=self.config.keep_message_datasets_in_memory,
            )
        return hf_dataset

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
            parser_kwargs=dict(
                source_data=self.config.lmdb_paths,  # defer instantiation to the generator due to pickling
                traces=subset_traces,
            ),
        )

    def make_dataloader(
        self,
        loader_name: pyine.data.datamodule.LoaderNameType,
    ) -> pyine.organisms.datamodules.utils.samples.SampleDataLoaderType:
        """Creates and returns a dataloader for the given name."""
        assert loader_name is not None, "loader name must be specified"
        parser = self.get_parser(loader_name)
        return self.config.instantiate_dataloader(
            loader_name=loader_name,
            dataset=parser,
        )

    @typing.override
    def train_dataloader(self) -> pyine.organisms.datamodules.utils.samples.SampleDataLoaderType:
        """Return the training data loader."""
        return self.make_dataloader("train")

    @typing.override
    def val_dataloader(self) -> pyine.organisms.datamodules.utils.samples.SampleDataLoaderType:
        """Return the validation data loader."""
        return self.make_dataloader("valid")

    @typing.override
    def test_dataloader(self) -> pyine.organisms.datamodules.utils.samples.SampleDataLoaderType:
        """Return the test data loader."""
        return self.make_dataloader("test")

    @typing.override
    def teardown(self, stage: str | None = None) -> None:
        """Close readers when the datamodule is torn down, and unassigns all parser attributes."""
        self._metadata: pyine.data.traces.dataset_utils.TraceDatasetMetadata | None = None
        self._subset_parsers: dict[
            pyine.data.datamodule.SubsetNameType,
            pyine.organisms.datamodules.utils.samples.SampleBuilder,
        ] = dict()
        self._readers: list[pyine.data.traces.dataset_reader.DatasetReader] = []
