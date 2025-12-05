"""Contains utility functions and a base interface for lightning datamodules."""

from __future__ import annotations

import logging
import os
import shutil
import typing
import uuid

import datasets as hf_datasets
import filelock
import lightning.pytorch as pl
import lightning.pytorch.utilities.types as pl_types
import pydantic
import torch.utils.data

import pyine.prompts.manager
import pyine.prompts.types
import pyine.utils.filesystem
import pyine.utils.openai
import pyine.utils.portability
import pyine.utils.pydantic
import pyine.utils.reprod
import pyine.utils.transformers

if typing.TYPE_CHECKING:
    import pathlib

    import langchain_core.runnables
    import langchain_openai.chat_models.base
    import transformers

logger = logging.getLogger(__name__)


SubsetNameType = str
"""Type used to represent a data subset name (e.g. 'train', 'valid', 'test')."""
LoaderNameType = str
"""Type used to represent a data loader name (e.g. 'train', 'valid', 'test')."""
type BaseDataParserClass[OutputSampleType] = torch.utils.data.Dataset[OutputSampleType]
"""Default base class used for data parsers."""
type BaseDataLoaderClass[OutputBatchType] = torch.utils.data.DataLoader[OutputBatchType]
"""Default base class used for data loaders."""


class BaseDataParserConfig(pyine.utils.pydantic.ClassImportSpec):
    """Base configuration class for data parser objects.

    See the parent class for more information; this class is mostly a placeholder. Derived instances
    should specify `class_path` and `params`, and optionally `base_class_path` if needed.
    """

    base_class_path: str = pyine.utils.portability.get_fully_qualified_name(torch.utils.data.Dataset)
    """Base class path for the PyTorch data parser (dataset) class."""
    # note: no need to override the params field here, datasets don't have anything standardized


if typing.TYPE_CHECKING:

    class BaseDataLoaderParamsConfig(pydantic.BaseModel):
        """Stubbed interface for the base data loader params config class defined below."""

        def __getattr__(self, name: str) -> typing.Any: ...

else:
    BaseDataLoaderParamsConfig = pyine.utils.pydantic.model_from_callable(
        fn=torch.utils.data.DataLoader,
        name="BaseDataLoaderParamsConfig",
        model_config=pydantic.ConfigDict(frozen=True, extra="forbid"),
        exclude={"dataset"},  # will be provided at derived class instantiation time
    )
    """Configuration parameters for the base data loader class."""


class BaseDataLoaderConfig(pyine.utils.pydantic.ClassImportSpec):
    """Base configuration class for data loader objects.

    See the parent class for more information. Derived instances should specify `class_path` and
    `params`. This specific class overrides the `params` field to specify default parameters
    expected by PyTorch data loaders.
    """

    base_class_path: str = pyine.utils.portability.get_fully_qualified_name(torch.utils.data.DataLoader)
    """Base class path for the PyTorch data loader class."""

    params: typing.Annotated[  # type: ignore[override]
        pydantic.SerializeAsAny[BaseDataLoaderParamsConfig],
        pydantic.Field(default_factory=BaseDataLoaderParamsConfig),
    ]
    """Default parameters for the data loader."""


class BaseDataModuleConfig(pydantic.BaseModel):
    """Base configuration class for datamodule objects.

    This class defines the default configuration for data parsers and data loaders.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
    """Pydantic model configuration (freezes the dataclass)."""

    datamodule_class_path: typing.Annotated[
        pydantic.StrictStr,
        pydantic.Field(
            min_length=1,
            description="Dotted import path to the target datamodule class, e.g. 'pkg.mod.MyImpl'.",
        ),
    ]
    datamodule_name: typing.Annotated[
        pydantic.StrictStr | None,
        pydantic.Field(
            default=None,
            min_length=1,
            description="Name of the datamodule (for debug/logging); will be derived from class name if needed.",
        ),
    ]

    # --------------- DATA PARSER (torch.utils.data.Dataset-like) CONFIGURATION ---------------

    default_dataparser_config: typing.Annotated[
        pydantic.SerializeAsAny[BaseDataParserConfig],
        pydantic.Field(
            # note: no default provided here, so it MUST be specified
            description="Default configuration for the data parsers whose specific settings may be overridden.",
        ),
    ]
    dataparser_config_overrides: typing.Annotated[
        dict[SubsetNameType, dict[str, typing.Any]],
        pydantic.Field(
            # no overrides by default, meaning all subsets will use the default config
            default_factory=dict,  # lambda: typing.cast(dict[SubsetNameType, dict[str, typing.Any]], {}),
            description="Data parser configuration dictionary with subset-specific default config overrides.",
        ),
    ]

    # --------------- DATA LOADER (torch.utils.data.DataLoader-like) CONFIGURATION ---------------

    default_dataloader_config: typing.Annotated[
        pydantic.SerializeAsAny[BaseDataLoaderConfig],
        pydantic.Field(
            default=BaseDataLoaderConfig(
                class_path=pyine.utils.portability.get_fully_qualified_name(torch.utils.data.DataLoader),
                params=BaseDataLoaderParamsConfig(),
            ),
            validate_default=True,
            description="Default configuration for the data loaders whose specific settings may be overridden.",
        ),
    ]
    dataloader_config_overrides: typing.Annotated[
        dict[LoaderNameType, dict[str, typing.Any]],
        pydantic.Field(
            # no overrides by default, meaning all subsets will use the default config
            default_factory=dict,  # lambda: typing.cast(dict[LoaderNameType, dict[str, typing.Any]], {}),
            description="Data loader configuration dictionary with subset-specific default config overrides.",
        ),
    ]

    # --------------- MISC SETTINGS CONFIGURATION ---------------

    split_seed: typing.Annotated[
        int,
        pydantic.Field(
            default=0,
            description="Seed used to initialize internal RNGs for dataset splits.",
        ),
    ]
    subset_names: typing.Annotated[
        tuple[SubsetNameType, ...],
        pydantic.Field(
            default=("train", "valid", "test"),
            min_length=1,
            description="Data subset names that the data module supports.",
        ),
    ]
    train_subset_names: typing.Annotated[
        tuple[SubsetNameType, ...],
        pydantic.Field(
            default=("train",),
            min_length=1,
            description="Subset names that are meant for model training.",
        ),
    ]
    valid_subset_names: typing.Annotated[
        tuple[SubsetNameType, ...],
        pydantic.Field(
            default=("valid",),
            min_length=1,
            description="Subset names that are meant for model validation.",
        ),
    ]
    eval_subset_names: typing.Annotated[
        tuple[SubsetNameType, ...],
        pydantic.Field(
            default=("valid",),
            min_length=1,
            description="Subset names that are meant for model evaluations.",
            # Note: should be kept to 'valid' instead of 'test' until experiments are done, and all
            #       hyperparameters are permanently FIXED; if this sounds strange to you, refer to:
            #          https://en.wikipedia.org/wiki/Training,_validation,_and_test_data_sets
        ),
    ]

    @property
    def loader_names(self) -> tuple[LoaderNameType, ...]:
        """Returns the dataloader names that this particular implementation supports.

        By default, we assume that dataloader names match with data subset names.
        """
        return self.subset_names

    # --------------- PUBLIC UTILITY FUNCTIONS ---------------

    def instantiate_parser(
        self,
        subset_name: SubsetNameType,
        *args: typing.Any,
        **extra_kwargs: typing.Any,
    ) -> BaseDataParserClass[typing.Any]:
        """Instantiates a data parser object for the given subset name."""
        parser_config = self._resolved_dataparser_configs[subset_name]
        parser_obj = parser_config.instantiate(*args, **extra_kwargs)
        expected_parent_class = torch.utils.data.Dataset
        if not isinstance(parser_obj, expected_parent_class):
            raise TypeError(f"expected {expected_parent_class} (or subclass), got {type(parser_obj)}")
        return typing.cast("BaseDataParserClass[typing.Any]", parser_obj)

    def instantiate_dataloader(
        self,
        loader_name: LoaderNameType,
        *args: typing.Any,
        **extra_kwargs: typing.Any,
    ) -> BaseDataLoaderClass[typing.Any]:
        """Instantiates a data loader object for the given loader name."""
        loader_config = self._resolved_dataloader_configs[loader_name]
        loader_obj = loader_config.instantiate(*args, **extra_kwargs)
        expected_parent_class = torch.utils.data.DataLoader
        if not isinstance(loader_obj, expected_parent_class):
            raise TypeError(f"expected {expected_parent_class} (or subclass), got {type(loader_obj)}")
        return typing.cast("BaseDataLoaderClass[typing.Any]", loader_obj)

    def instantiate_datamodule(
        self,
        *args: typing.Any,
        **extra_kwargs: typing.Any,
    ) -> BaseDataModule[typing.Any]:
        """Instantiates a data module object based on the configured target class path."""
        dm_class = self._resolved_datamodule_class
        if dm_class is None:
            raise RuntimeError("datamodule class has not been resolved; validate config before use")
        return dm_class(*args, config=self, **extra_kwargs)

    # --------------- PRIVATE UTILITY FUNCTIONS & ATTRIBUTES ---------------

    # cache resolved subset configs so we don't re-resolve them in each getter call
    _resolved_dataparser_configs: dict[SubsetNameType, BaseDataParserConfig] = pydantic.PrivateAttr(
        default_factory=lambda: typing.cast("dict[SubsetNameType, BaseDataParserConfig]", {}),
    )
    _resolved_dataloader_configs: dict[LoaderNameType, BaseDataLoaderConfig] = pydantic.PrivateAttr(
        default_factory=lambda: typing.cast("dict[LoaderNameType, BaseDataLoaderConfig]", {}),
    )
    # cache the resolved datamodule class type also for potential instantiate calls
    _resolved_datamodule_class: type[BaseDataModule[typing.Any]] | None = pydantic.PrivateAttr(default=None)

    def _resolve_dataparser_config(
        self,
        subset_name: SubsetNameType,
    ) -> BaseDataParserConfig:
        """Returns the data parser configuration for the given subset name."""
        if subset_name not in self.subset_names:
            raise ValueError(f"invalid subset name: {subset_name}, expected one of: {self.subset_names}")
        parser_config = self.default_dataparser_config
        if subset_name in self.dataparser_config_overrides and self.dataparser_config_overrides[subset_name]:
            parser_config = parser_config.get_updated_spec(**self.dataparser_config_overrides[subset_name])
        return parser_config

    def _resolve_dataloader_config(
        self,
        loader_name: LoaderNameType,
    ) -> BaseDataLoaderConfig:
        """Returns the data loader configuration for the given loader name."""
        if loader_name not in self.loader_names:
            raise ValueError(f"invalid loader name: {loader_name}, expected one of: {self.loader_names}")
        loader_config = self.default_dataloader_config
        if loader_name in self.dataloader_config_overrides and self.dataloader_config_overrides[loader_name]:
            loader_config = loader_config.get_updated_spec(**self.dataloader_config_overrides[loader_name])
        return loader_config

    @pydantic.model_validator(mode="after")
    def _validate_and_resolve(self) -> BaseDataModuleConfig:
        """Validates and resolves the data parser and data loader configs."""
        for subset_name in self.subset_names:
            self._resolved_dataparser_configs[subset_name] = self._resolve_dataparser_config(subset_name)
        for loader_name in self.loader_names:
            self._resolved_dataloader_configs[loader_name] = self._resolve_dataloader_config(loader_name)
        resolved_class = pyine.utils.portability.import_from_dotted_path(self.datamodule_class_path)
        if not isinstance(resolved_class, type) or not callable(resolved_class):
            raise TypeError(f'"{self.datamodule_class_path}" resolved to {resolved_class!r}, which is not a class')
        if not issubclass(resolved_class, BaseDataModule):
            raise TypeError(f'"{self.datamodule_class_path}" is not a subclass of BaseDataModule')
        self._resolved_datamodule_class = resolved_class
        if any(name not in self.subset_names for name in self.train_subset_names):
            raise ValueError(f"some subset name(s) are invalid; got {self.train_subset_names!r}")
        if any(name not in self.subset_names for name in self.valid_subset_names):
            raise ValueError(f"some subset name(s) are invalid; got {self.valid_subset_names!r}")
        if any(name not in self.subset_names for name in self.eval_subset_names):
            raise ValueError(f"some subset name(s) are invalid; got {self.eval_subset_names!r}")
        return self


class BaseDataModule[ConfigType](pl.LightningDataModule):
    """Wraps the standard LightningDataModule interface to combine it with pydantic configs.

    Each derived data module will likely correspond to a combination of one dataset and one target
    task. This interface provides common definitions regarding data parser and loader creation, and
    helps document what functions should have an override in the derived classes and why. The reason
    to use it is to simplify the creation of data parsers and loaders with shared (and externally
    configurable) settings (e.g. data transformations).

    For more information on data modules in lightning, see:
    https://lightning.ai/docs/pytorch/stable/data/datamodule.html
    """

    def __init__(
        self,
        config: ConfigType,
    ) -> None:
        """Initializes the base interface using the expected configs of parsers/loaders.

        Args:
            config: configuration model of data module (parser/loader) settings.
        """
        if not isinstance(config, BaseDataModuleConfig):
            raise TypeError(f"expected a config derived from {BaseDataModuleConfig}, got {type(config)}")
        super().__init__()
        self.save_hyperparameters(config.model_dump())
        self.config: ConfigType = config

    def prepare_data(self) -> None:
        """Override this function to download/copy/prepare local data for the dataloaders.

        Downloading and saving data with multiple processes (distributed settings) will result in
        corrupted data. Lightning ensures this method is called only within the main process, so you
        can safely add your downloading logic within.

        NOTE: This function is not meant to perform preprocessing, but rather to download and store
        data locally for use in the `setup` function. This function should NOT be setting any kind
        of state inside the data module itself, as that state will not be shared across processes.

        For more information, see:
        https://lightning.ai/docs/pytorch/stable/data/datamodule.html#prepare-data
        """
        pass

    def setup(self, stage: str | None = None) -> None:
        """Called at the beginning of `fit` (train + validation), `validate`, `test`, or `predict`.

        This is where the metadata, size, and other high-level info of the already-prepared local
        dataset(s) should be parsed. The outcome of this parsing should be a "state" inside the
        data module itself, likely in a data parser (e.g. an instance derived from
        `torch.utils.data.Dataset`). With a distributed training strategy, this will be called on
        each node.

        NOTE: the `teardown` function can clean up the state created here.

        For more information, see:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html#setup

        Args:
            stage: either ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``
        """
        pass

    def teardown(self, stage: str | None = None) -> None:
        """Called at the end of `fit` (training + validation), `validate`, `test`, or `predict`.

        When called, the "state" of the parsers prepared in `setup` should be cleared (if needed).

        For more information, see:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html#teardown

        Args:
            stage: either ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``
        """
        pass

    def train_dataloader(self) -> pl_types.TRAIN_DATALOADERS:
        """Instantiates one or more pytorch dataloaders for training based on the parsed dataset.

        For more information, see:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html#train-dataloader

        Returns:
            A data loader (or a collection of them) that provides training samples.
        """
        raise NotImplementedError("derived class should implement this function")

    def test_dataloader(self) -> pl_types.EVAL_DATALOADERS:
        """Instantiates one or more pytorch dataloaders for testing based on the parsed dataset.

        Note:
            In the case where this returns multiple test dataloaders, the LightningModule `test_step`
            method will have an argument `dataloader_idx` which matches the order here.

        For more information, see:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html#test-dataloader

        Returns:
            A data loader (or a collection of them) that provides testing samples.
        """
        raise NotImplementedError("derived class should implement this function")

    def val_dataloader(self) -> pl_types.EVAL_DATALOADERS:
        """Instantiates one or more pytorch dataloaders for validation based on the parsed dataset.

        Note:
            During training, the returned dataloader(s) will not be reloaded between epochs unless
            you set the `reload_dataloaders_every_n_epochs` argument (in the trainer configuration)
            to a positive integer.

            In the case where this returns multiple dataloaders, the LightningModule `validation_step`
            method will have an argument `dataloader_idx` which matches the order here.

        For more information, see:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html#val-dataloader

        Returns:
            A data loader (or a collection of them) that provides validation samples.
        """
        raise NotImplementedError("derived class should implement this function")

    def valid_dataloader(self) -> pl_types.EVAL_DATALOADERS:
        """Instantiates one or more pytorch dataloaders for validation based on the parsed dataset.

        This function simply redirects to the `val_dataloader` function. Why? Just because using
        'val' instead of 'valid' is not everyone's cup of tea.
        """
        return self.val_dataloader()

    def predict_dataloader(self) -> pl_types.EVAL_DATALOADERS:
        """Instantiates one or more pytorch dataloaders for prediction runs.

        Note: in contrast with typical validation or test data loaders, this loader is typically
        assumed NOT to load groundtruth (target) annotations along with data samples.

        Note:
            In the case where this returns multiple dataloaders, the LightningModule `predict_step`
            method will have an argument `dataloader_idx` which matches the order here.

        Return:
            A data loader (or a collection of them) that provides prediction samples.
        """
        raise NotImplementedError("derived class should implement this function")

    @property
    def dataloader_names(self) -> tuple[LoaderNameType, ...]:
        """Returns the dataloader names that this particular implementation supports."""
        if not isinstance(self.config, BaseDataModuleConfig):
            raise TypeError(f"expected a config derived from {BaseDataModuleConfig}, got {type(self.config)}")
        return self.config.loader_names

    def get_stats(self, target_subsets: list[SubsetNameType] | None = None) -> dict[str, int | float | str]:
        """Returns a dictionary of useful-to-log statistics."""
        return {}

    def get_dataloader(
        self,
        loader_name: LoaderNameType,
    ) -> pl_types.TRAIN_DATALOADERS | pl_types.EVAL_DATALOADERS:  # noqa
        """Returns a data loader object (or a collection of) for a given name.

        This function will verify that the specified subset exists and redirect the getter to the
        correct function that prepares the dataloader(s). By default, we assume that dataloader
        types are linked to data subsets.
        """
        if loader_name not in self.dataloader_names:
            raise ValueError(f"invalid loader name: {loader_name}, expected one of: {self.dataloader_names}")
        expected_getter_name = f"{loader_name}_dataloader"
        if not hasattr(self, expected_getter_name):
            raise ValueError(f"invalid loader name: {loader_name}, no such function: {expected_getter_name}")
        getter = getattr(self, expected_getter_name)
        if not callable(getter):
            raise ValueError(f"invalid {loader_name} getter type: {type(getter)}, expected callable")
        return getter()

    def get_parser(
        self,
        subset_name: SubsetNameType,
    ) -> BaseDataParserClass[typing.Any]:
        """Returns a data parser object for a given subset name.

        This function exists for users that might not want to use dataloaders directly, and would prefer
        using the data parsers directly instead (e.g. to provide specific transforms, or to use them
        as part of a wider framework such as HuggingFace).
        """
        raise NotImplementedError("derived class should implement this function")


class ConversationDataParserConfig(BaseDataParserConfig):
    """Specialized configuration class for conversation data parser objects.

    See the parent class for more information; this class is mostly a placeholder. Derived instances
    should specify `class_path` and `params`, and optionally `base_class_path` if needed.
    """

    def generate_hf_messages_dataset(
        self,
        named_split: hf_datasets.NamedSplit,
        raw_transform_fn: (typing.Callable[[dict[str, typing.Any]], typing.Any] | None) = None,
        instantiate_kwargs: dict[str, typing.Any] | None = None,
        keep_in_memory: bool = False,
        num_workers: int | None = None,
    ) -> hf_datasets.Dataset:
        """Generates and returns a HuggingFace messages dataset using an instantiated parser.

        This function exists for users that might not want to use raw data loaders directly, and
        would prefer using already-prepared message data for huggingface-based experiments.

        Returns:
             The HuggingFace messages dataset object.
        """
        raise NotImplementedError("derived class should implement this function")


class ConversationDataModuleConfig(BaseDataModuleConfig):
    """Specialized configuration class for conversation datamodule objects.

    This class supplements the base class defaults with conversation/message-specific settings. The
    HuggingFace and OpenAI message datasets instantiated here will be cached locally for reuse (by
    default).
    """

    prompt_config: pydantic.SerializeAsAny[pyine.prompts.types.PromptBuildConfig]
    """Configuration of the prompt to use for the conversation datamodule; given to the prompt manager."""
    keep_generated_datasets_in_memory: bool = True
    """Defines whether generated datasets should be kept in memory."""
    message_generator_num_workers: int = 4
    """Defines the number of workers to use when generating message datasets."""
    use_local_dataset_cache: bool = True
    """Whether to always try to save/load pre-tokenized datasets from the local cache or not."""
    use_tokenized_dataset_cache: bool = True
    """Whether to enable caching for tokenized datasets derived from conversation datasets."""
    cache_lock_timeout_seconds: float = pydantic.Field(default=600.0, ge=0)
    """Maximum time (in seconds) to wait when acquiring dataset cache locks."""

    def get_prompt_template(
        self,
        **kwargs: typing.Any,  # forwarded to prompt manager / constructor, overrides internal options if needed
    ) -> pyine.prompts.types.PromptTemplate:
        """Returns the prompt template used for preparing training/evaluation conversations."""
        prompt_kwargs = self.prompt_config.model_dump()
        prompt_kwargs.update(kwargs)
        return pyine.prompts.manager.get_prompt_template(**prompt_kwargs)

    def get_prompt_chain(
        self,
        model: langchain_openai.chat_models.base.BaseChatOpenAI,
        runnable_name: str | None = None,
        **kwargs: typing.Any,  # forwarded to prompt manager / constructor, overrides internal options if needed
    ) -> langchain_core.runnables.Runnable[typing.Any, typing.Any]:
        """Returns the runnable prompt chain used to infer assistant messages in conversations."""
        prompt_kwargs = self.prompt_config.model_dump()
        prompt_kwargs.update(kwargs)
        return pyine.prompts.manager.get_prompt_chain(model=model, **prompt_kwargs, runnable_name=runnable_name)

    @typing.override
    def instantiate_datamodule(
        self,
        *args: typing.Any,
        **extra_kwargs: typing.Any,
    ) -> ConversationDataModule[typing.Any]:
        """Instantiates a data module object based on the configured target class path."""
        dm = super().instantiate_datamodule(*args, **extra_kwargs)
        if not isinstance(dm, ConversationDataModule):
            raise TypeError(f"expected {ConversationDataModule} (or subclass), got {type(dm)}")
        return dm

    def instantiate_hf_messages_dataset(
        self,
        subset_name: SubsetNameType,
        append_answer: bool = True,
        merge_system_with_user: bool = False,
        keep_original_data: bool = False,
        parser_kwargs: dict[str, typing.Any] | None = None,
        force_regenerate: bool = False,
    ) -> hf_datasets.Dataset:
        """Instantiates a `hf_datasets.Dataset` object based on the configured parser settings.

        This function exists for users that might not want to use dataloaders directly, and would
        prefer using the data parsers for huggingface-based experiments.

        Args:
            subset_name: the subset name to prepare the dataset for.
            append_answer: whether to append the assistant's response to the conversation messages.
            merge_system_with_user: whether to merge the system message with the user message (used
                when working with e.g. o1/o3/o4, which do not support custom system prompts).
            keep_original_data: whether to keep the original data inside the output samples (e.g.
                to access metadata in evaluations).
            parser_kwargs: keyword arguments to pass to the parser's constructor (if any).
            force_regenerate: whether to rebuild the dataset cache even if it already exists.

        Returns:
            A huggingface dataset that produces chat-templated 'conversations' (lists of messages).
        """
        hf_datasets_cache_dir = pyine.utils.filesystem.get_data_cache_path() / "hf_datasets"
        params_hash = pyine.utils.reprod.get_params_hash(
            self.model_dump(),
            append_answer,
            merge_system_with_user,
            keep_original_data,
            parser_kwargs,
        )
        datamodule_name = self.datamodule_name or self._resolved_datamodule_class.__name__
        dataset_name = f"{datamodule_name}.{subset_name}.{params_hash}"
        dataset_path = hf_datasets_cache_dir / dataset_name
        named_split = hf_datasets.NamedSplit(name=subset_name)
        parser_config = self._resolved_dataparser_configs[subset_name]
        assert isinstance(parser_config, ConversationDataParserConfig)
        transf_fn = self.instantiate_sample_to_messages_transform(
            append_answer=append_answer,
            use_hf_messages=True,
            merge_system_with_user=merge_system_with_user,
            keep_original_data=keep_original_data,
        )
        if not self.use_local_dataset_cache:
            # if we are not using any caching, generate and return the dataset directly
            return parser_config.generate_hf_messages_dataset(
                named_split=named_split,
                raw_transform_fn=transf_fn,
                instantiate_kwargs=parser_kwargs,
                keep_in_memory=self.keep_generated_datasets_in_memory,
                num_workers=self.message_generator_num_workers,
            )
        # otherwise, acquire a lock (for potential ddp runs) and check if it needs to be generated
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = dataset_path.parent / f"{dataset_path.name}.lock"
        lock = filelock.FileLock(str(lock_path), timeout=self.cache_lock_timeout_seconds)
        with lock:
            if dataset_path.exists():
                if force_regenerate:
                    logger.info(f"force-regenerating huggingface dataset cache at: {dataset_path}")
                    shutil.rmtree(dataset_path)
                else:
                    logger.info(f"loading already-generated dataset from cache: {dataset_path}")
                    return hf_datasets.Dataset.load_from_disk(  # type: ignore[reportUnknownMemberType]
                        dataset_path=dataset_path,
                        keep_in_memory=self.keep_generated_datasets_in_memory,
                    )
            logger.info(f"building huggingface dataset cache at: {dataset_path}")
            dataset = parser_config.generate_hf_messages_dataset(
                named_split=named_split,
                raw_transform_fn=transf_fn,
                instantiate_kwargs=parser_kwargs,
                keep_in_memory=self.keep_generated_datasets_in_memory,
                num_workers=self.message_generator_num_workers,
            )
            tmp_path = dataset_path.parent / f"{dataset_path.name}.tmp.{uuid.uuid4().hex}"
            try:
                dataset.save_to_disk(tmp_path)  # type: ignore[reportUnknownMemberType]
                os.replace(tmp_path, dataset_path)
            finally:
                shutil.rmtree(tmp_path, ignore_errors=True)
            logger.info(f"saved dataset cache: {dataset_path}")
            if self.keep_generated_datasets_in_memory:
                return dataset
            return hf_datasets.Dataset.load_from_disk(  # type: ignore[reportUnknownMemberType]
                dataset_path=dataset_path,
                keep_in_memory=self.keep_generated_datasets_in_memory,
            )

    def instantiate_openai_messages_dataset(
        self,
        subset_name: SubsetNameType,
        append_answer: bool = True,
        merge_system_with_user: bool = False,
        parser_kwargs: dict[str, typing.Any] | None = None,
    ) -> pathlib.Path:
        """Returns the path to an OpenAI-compatible JSONL dataset of chat-templated conversations.

        This function exists for users that might want to use the data in combination with the
        OpenAI API. The dataset is written to the returned path in the OpenAI format, which is
        a JSONL file with one example per line. That dataset file can then be uploaded to the
        OpenAI API to train/validate a model.

        Args:
            subset_name: the subset name to prepare the dataset for.
            append_answer: whether to append the assistant's response to the conversation messages.
            merge_system_with_user: whether to merge the system message with the user message (used
                when working with e.g. o1/o3/o4, which do not support custom system prompts).
            parser_kwargs: keyword arguments to pass to the parser's constructor (if any).

        Returns:
             The path to the written dataset, which can be used for uploads to the OpenAI API.
        """
        openai_local_data_dir = pyine.utils.openai.get_local_file_directory()
        params_hash = pyine.utils.reprod.get_params_hash(
            self.model_dump(),
            append_answer,
            merge_system_with_user,
            parser_kwargs,
        )
        datamodule_name = self.datamodule_name or self._resolved_datamodule_class.__name__
        dataset_file_name = f"{datamodule_name}.{subset_name}.{params_hash}.jsonl"
        local_output_path = openai_local_data_dir / dataset_file_name
        if not self.use_local_dataset_cache or not local_output_path.is_file():
            # note: this impl relies on the huggingface getter to generate the messages (DRY)
            hf_dataset = self.instantiate_hf_messages_dataset(
                subset_name=subset_name,
                append_answer=append_answer,
                merge_system_with_user=merge_system_with_user,
                parser_kwargs=parser_kwargs,
            )
            # even if we do not use a cache, we need to write the dataset locally for later refs
            pyine.utils.openai.write_dataset_to_jsonl(hf_dataset, local_output_path)
        return local_output_path

    def instantiate_sample_to_messages_transform(
        self,
        append_answer: bool = True,
        use_hf_messages: bool = False,
        merge_system_with_user: bool = False,
        keep_original_data: bool = False,
    ) -> typing.Callable[[typing.Any], typing.Any]:
        """Returns the sample transform function used to prepare training/evaluation messages.

        This function exists for users that might not want to use dataloaders directly, and would
        prefer using the data parsers while applying raw data transforms directly instead.

        Note: if the datamodule does not support the conversion of raw data samples into
        conversation messages, this function will raise an exception.

        Args:
            append_answer: whether to append the assistant's response to the conversation messages.
            use_hf_messages: whether to use HuggingFace messages format or the langchain format.
            merge_system_with_user: whether to merge the system message with the user message (used
                when working with e.g. o1/o3/o4, which do not support custom system prompts).
            keep_original_data: whether to keep the original data inside the output samples; only
                usable if `use_hf_messages` is true.

        Returns:
             The sample transform function.
        """
        # this transform is application-specific: it depends on the type of samples provided by parsers
        raise NotImplementedError("derived class should implement this function")

    @pydantic.model_validator(mode="after")
    @typing.override
    def _validate_and_resolve(self) -> ConversationDataModuleConfig:
        """Validates and resolves prompt config stuff as well as parent checks."""
        super()._validate_and_resolve()  # type: ignore[reportUnknownMemberType] --- # noqa
        # check if we can instantiate a prompt template given the provided config
        _resolved_template = self.get_prompt_template()
        assert _resolved_template is not None
        return self

    def get_tokenized_dataset_cache_root(self) -> pathlib.Path:
        """Returns the root directory used to persist tokenized dataset caches."""
        return pyine.utils.filesystem.get_data_cache_subdir("hf_tokenized")


class ConversationDataModule[ConfigType](BaseDataModule[ConfigType]):
    """Data module base class for conversation-based data.

    This specialized data module class is designed to work with data that can be structured as
    lists of messages, i.e. conversations, between a user and an AI assistant.

    See the parent class documentation for more details on the interface.
    """

    def __init__(
        self,
        config: ConfigType,
    ) -> None:
        """Initializes the base interface using the expected configs of parsers/loaders.

        Args:
            config: configuration model of data module (parser/loader) settings.
        """
        if not isinstance(config, ConversationDataModuleConfig):
            raise TypeError(f"expected a config derived from {ConversationDataModuleConfig}, got {type(config)}")
        super().__init__(config)

    def get_hf_messages_dataset(
        self,
        subset_name: SubsetNameType,
        append_answer: bool = True,
        merge_system_with_user: bool = False,
        keep_original_data: bool = False,
        force_regenerate: bool = False,
    ) -> hf_datasets.Dataset:
        """Returns a HuggingFace messages dataset object for a given subset name.

        This function exists for users that might not want to use raw data loaders directly, and
        would prefer using already-prepared conversation data for huggingface-based experiments.

        Note: if the datamodule does not support the conversion of raw data samples into
        conversation messages, this function will raise an exception.

        Args:
            subset_name: the subset name to prepare the dataset for.
            append_answer: whether to append the assistant's response to the conversation messages.
            merge_system_with_user: whether to merge the system message with the user message (used
                when working with e.g. o1/o3/o4, which do not support custom system prompts).
            keep_original_data: whether to keep the original data inside the output samples.
            force_regenerate: whether to rebuild caches even if they already exist.

        Returns:
             The HuggingFace dataset object.
        """
        raise NotImplementedError("derived class should implement this function")

    def get_hf_tokenized_examples_dataset(
        self,
        subset_name: typing.Literal["train", "valid", "eval"],
        tokenizer: transformers.PreTrainedTokenizer,
        model_max_seq_len: int,
        num_proc: int = 4,
        force_regenerate: bool = False,
        epoch: int | None = None,
    ) -> hf_datasets.Dataset:
        """Returns a HuggingFace dataset of tokenized examples for supervised training.

        This function wraps other dataset preparation functions to provide a dataset of tokenized
        examples as expected by HuggingFace for the supervised training (or evaluation) of models.
        In contrast with other dataset preparation functions, the subset name provided to this one
        must match one of the subset names expected by HuggingFace, and tied to the parent class's
        `train_subset_names`, `valid_subset_names`, and `eval_subset_names` attributes.

        Args:
            subset_name: the subset name to prepare the dataset for (train/valid/eval only!).
            tokenizer: the tokenizer to use for tokenization.
            model_max_seq_len: the maximum sequence length supported by the tokenizer/model.
            num_proc: the number of processes to use for dataset preparation.
            force_regenerate: whether to rebuild caches even if they already exist.
            epoch: Optional epoch index; when provided, it becomes part of the cache fingerprint so
                multiple epoch-dependent variants can coexist on disk.

        Returns:
            The HuggingFace dataset object.
        """
        config = typing.cast("ConversationDataModuleConfig", self.config)
        hf_subset_names_map = {
            "train": config.train_subset_names,
            "valid": config.valid_subset_names,
            "eval": config.eval_subset_names,
        }
        assert subset_name in hf_subset_names_map, f"invalid hf subset name: {subset_name}"
        actual_subset_names = hf_subset_names_map[subset_name]
        keep_original_data = subset_name != "train"  # for evaluations, orig data might be needed, so keep it
        messages_datasets = [
            self.get_hf_messages_dataset(
                subset_name=subset_name,
                append_answer=True,
                keep_original_data=keep_original_data,
                force_regenerate=force_regenerate,
            )
            for subset_name in actual_subset_names
        ]
        if len(messages_datasets) == 1:
            messages_ds = messages_datasets[0]
        else:
            messages_ds = hf_datasets.concatenate_datasets(messages_datasets)  # should return same interface
        datamodule_label = config.datamodule_name or self.__class__.__name__
        tokenizer_identifier = getattr(tokenizer, "name_or_path", type(tokenizer).__name__)
        if not config.use_tokenized_dataset_cache:
            cache_settings = None
        else:
            cache_root = config.get_tokenized_dataset_cache_root()
            assert cache_root.is_dir(), f"invalid cache root dir: {cache_root}"
            dataset_fingerprint = getattr(messages_ds, "_fingerprint", getattr(messages_ds, "_hash", None))
            cache_hash = pyine.utils.reprod.get_params_hash(
                datamodule_label,
                subset_name,
                actual_subset_names,
                dataset_fingerprint,
                tokenizer_identifier,
                model_max_seq_len,
                keep_original_data,
                epoch,
            )
            dataset_name = f"{datamodule_label}.{subset_name}.{cache_hash}"
            dataset_cache_path = cache_root / dataset_name
            cache_settings = pyine.utils.transformers.DataCacheSettings(
                cache_path=dataset_cache_path, lock_timeout_seconds=config.cache_lock_timeout_seconds
            )
        if cache_settings is not None:
            logger.info(
                f"tokenized cache target: subset={subset_name} epoch={epoch} path={cache_settings.cache_path}",
            )
        return pyine.utils.transformers.prepare_examples_from_conversations(
            convo_ds=messages_ds,
            tokenizer=tokenizer,
            max_seq_len=model_max_seq_len,
            num_proc=num_proc,
            keep_extra_fields=keep_original_data,
            cache_settings=cache_settings,
            force_rebuild=force_regenerate,
        )

    def get_openai_messages_dataset(
        self,
        subset_name: SubsetNameType,
        append_answer: bool = True,
        merge_system_with_user: bool = False,
    ) -> pathlib.Path:
        """Returns the path to an OpenAI-compatible JSONL dataset of chat-templated conversations.

        This function exists for users that might not want to use raw data loaders directly, and
        would prefer using already-prepared conversation data for OpenAI-API-based experiments. The
        format of the datasets in this case is a JSONL file with one example per line. That dataset
        is written to disk (ready to be uploaded to OpenAI) at the returned path.

        Note: if the datamodule does not support the conversion of raw data samples into
        conversation messages, this function will raise an exception.

        Args:
            subset_name: the subset name to prepare the dataset for.
            append_answer: whether to append the assistant's response to the conversation messages.
            merge_system_with_user: whether to merge the system message with the user message (used
                when working with e.g. o1/o3/o4, which do not support custom system prompts).

        Returns:
             The path to the written dataset, which can be used for uploads to the OpenAI API.
        """
        raise NotImplementedError("derived class should implement this function")
