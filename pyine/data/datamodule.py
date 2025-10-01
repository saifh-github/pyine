"""Contains utility functions and a base interface for lightning datamodules."""

import logging
import pathlib
import typing

import datasets as hf_datasets
import langchain_core.language_models
import langchain_core.prompts
import langchain_core.runnables
import lightning.pytorch as pl
import lightning.pytorch.utilities.types as pl_types
import pydantic
import torch.utils.data
import transformers

import pyine.prompts.manager
import pyine.prompts.types
import pyine.utils.filesystem
import pyine.utils.openai
import pyine.utils.portability
import pyine.utils.pydantic
import pyine.utils.reprod

logger = logging.getLogger(__name__)


SubsetNameType = str
"""Type used to represent a data subset name (e.g. 'train', 'valid', 'test')."""
LoaderNameType = str
"""Type used to represent a data loader name (e.g. 'train', 'valid', 'test')."""
BaseDataParserType = torch.utils.data.Dataset
"""Default base class used for data parsers."""
BaseDataLoaderType = torch.utils.data.DataLoader
"""Default base class used for data loaders."""


class BaseDataParserConfig(pyine.utils.pydantic.ClassImportSpec[BaseDataParserType]):
    """Base configuration class for data parser objects.

    See the parent class for more information; this class is mostly a placeholder. Derived instances
    should specify `class_path` and `params`, and optionally `base_class_path` if needed.
    """

    base_class_path: str = pyine.utils.portability.get_fully_qualified_name(BaseDataParserType)
    """Base class path for the PyTorch data parser (dataset) class."""
    # note: no need to override the params field here, datasets don't have anything standardized


BaseDataLoaderParamsConfig = pyine.utils.pydantic.model_from_callable(
    fn=BaseDataLoaderType,
    name="BaseDataLoaderParamsConfig",
    model_config=pydantic.ConfigDict(frozen=True, extra="forbid"),
    exclude={"dataset"},  # will be provided at derived class instantiation time
)
"""Configuration parameters for the base data loader class."""


class BaseDataLoaderConfig(pyine.utils.pydantic.ClassImportSpec[BaseDataLoaderType]):
    """Base configuration class for data loader objects.

    See the parent class for more information. Derived instances should specify `class_path` and
    `params`. This specific class overrides the `params` field to specify default parameters
    expected by PyTorch data loaders.
    """

    base_class_path: str = pyine.utils.portability.get_fully_qualified_name(BaseDataLoaderType)
    """Base class path for the PyTorch data loader class."""

    params: pydantic.SerializeAsAny[BaseDataLoaderParamsConfig] = BaseDataLoaderParamsConfig()
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
            default_factory=dict,  # no overrides by default, meaning all subsets will use the default config
            description="Data parser configuration dictionary with subset-specific default config overrides.",
        ),
    ]

    # --------------- DATA LOADER (torch.utils.data.DataLoader-like) CONFIGURATION ---------------

    default_dataloader_config: typing.Annotated[
        pydantic.SerializeAsAny[BaseDataLoaderConfig],
        pydantic.Field(
            default=BaseDataLoaderConfig(
                class_path=pyine.utils.portability.get_fully_qualified_name(BaseDataLoaderType),
                params=BaseDataLoaderParamsConfig(),
            ),
            validate_default=True,
            description="Default configuration for the data loaders whose specific settings may be overridden.",
        ),
    ]
    dataloader_config_overrides: typing.Annotated[
        dict[LoaderNameType, dict[str, typing.Any]],
        pydantic.Field(
            default_factory=dict,  # no overrides by default, meaning all subsets will use the default config
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
            default=tuple(["train", "valid", "test"]),
            min_length=1,
            description="Data subset names that the data module supports.",
        ),
    ]
    train_subset_names: typing.Annotated[
        tuple[SubsetNameType, ...],
        pydantic.Field(
            default=tuple(["train"]),
            min_length=1,
            description="Subset names that are meant for model training.",
        ),
    ]
    valid_subset_names: typing.Annotated[
        tuple[SubsetNameType, ...],
        pydantic.Field(
            default=tuple(["valid"]),
            min_length=1,
            description="Subset names that are meant for model validation.",
        ),
    ]
    eval_subset_names: typing.Annotated[
        tuple[SubsetNameType, ...],
        pydantic.Field(
            default=tuple(["valid"]),
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

    def instantiate_parser(self, subset_name: SubsetNameType, *args, **extra_kwargs) -> BaseDataParserType:
        """Instantiates a data parser object for the given subset name."""
        parser_config = self._resolved_dataparser_configs[subset_name]
        parser = parser_config.instantiate(*args, **extra_kwargs)
        if not isinstance(parser, BaseDataParserType):
            raise TypeError(f"expected {BaseDataParserType} (or subclass), got {type(parser)}")
        return parser

    def instantiate_dataloader(self, loader_name: LoaderNameType, *args, **extra_kwargs) -> BaseDataLoaderType:
        """Instantiates a data loader object for the given loader name."""
        loader_config = self._resolved_dataloader_configs[loader_name]
        loader = loader_config.instantiate(*args, **extra_kwargs)
        if not isinstance(loader, BaseDataLoaderType):
            raise TypeError(f"expected {BaseDataLoaderType} (or subclass), got {type(loader)}")
        return loader

    def instantiate_datamodule(self, *args, **extra_kwargs) -> "BaseDataModule":
        """Instantiates a data module object based on the configured target class path."""
        dm = self._resolved_datamodule_class(*args, config=self, **extra_kwargs)
        if not isinstance(dm, BaseDataModule):
            raise TypeError(f"expected {BaseDataModule} (or subclass), got {type(dm)}")
        return dm

    # --------------- PRIVATE UTILITY FUNCTIONS & ATTRIBUTES ---------------

    # cache resolved subset configs so we don't re-resolve them in each getter call
    _resolved_dataparser_configs: dict[SubsetNameType, BaseDataParserConfig] = pydantic.PrivateAttr(
        default_factory=dict
    )
    _resolved_dataloader_configs: dict[LoaderNameType, BaseDataLoaderConfig] = pydantic.PrivateAttr(
        default_factory=dict
    )
    # cache the resolved datamodule class type also for potential instantiate calls
    _resolved_datamodule_class: type | None = pydantic.PrivateAttr(default=None)

    def _resolve_dataparser_config(
        self,
        subset_name: SubsetNameType,
    ) -> BaseDataParserConfig:
        """Returns the data parser configuration for the given subset name."""
        if subset_name not in self.subset_names:
            raise ValueError(f"invalid subset name: {subset_name}, expected one of: {self.subset_names}")
        parser_config = self.default_dataparser_config
        assert isinstance(parser_config, pyine.utils.pydantic.ClassImportSpec)
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
        assert isinstance(loader_config, pyine.utils.pydantic.ClassImportSpec)
        if loader_name in self.dataloader_config_overrides and self.dataloader_config_overrides[loader_name]:
            loader_config = loader_config.get_updated_spec(**self.dataloader_config_overrides[loader_name])
        return loader_config

    @pydantic.model_validator(mode="after")
    def _validate_and_resolve(self) -> "BaseDataModuleConfig":
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
        if any([name not in self.subset_names for name in self.train_subset_names]):
            raise ValueError(f"some subset name(s) are invalid; got {self.train_subset_names!r}")
        if any([name not in self.subset_names for name in self.valid_subset_names]):
            raise ValueError(f"some subset name(s) are invalid; got {self.valid_subset_names!r}")
        if any([name not in self.subset_names for name in self.eval_subset_names]):
            raise ValueError(f"some subset name(s) are invalid; got {self.eval_subset_names!r}")
        return self


class BaseDataModule(pl.LightningDataModule):
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
        config: BaseDataModuleConfig,
    ):
        """Initializes the base interface using the expected configs of parsers/loaders.

        Args:
            config: configuration model of data module (parser/loader) settings.
        """
        super().__init__()
        if not isinstance(config, BaseDataModuleConfig):
            raise TypeError(f"invalid config type: {type(config)}, expected {BaseDataModuleConfig}")
        self.save_hyperparameters(config.model_dump())
        self.config = config

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
        return self.config.loader_names

    def get_stats(self, target_subsets: list[SubsetNameType] | None = None) -> dict[str, int | float | str]:
        """Returns a dictionary of useful-to-log statistics."""
        return dict()  # nothing to log here by default

    def get_dataloader(
        self,
        loader_name: LoaderNameType,
    ) -> pl_types.TRAIN_DATALOADERS | pl_types.EVAL_DATALOADERS:  # noqa
        """Returns a data loader object (or a collection of) for a given name.

        This function will verify that the specified subset exists and redirect the getter to the
        correct function that prepares the dataloader(s). By default, we assume that dataloader
        types are linked to data subsets.
        """
        if loader_name not in self.config.loader_names:
            raise ValueError(f"invalid loader name: {loader_name}, expected one of: {self.config.loader_names}")
        expected_getter_name = f"{loader_name}_dataloader"
        if not hasattr(self, expected_getter_name):
            raise ValueError(f"invalid loader name: {loader_name}, no such function: {expected_getter_name}")
        getter = getattr(self, expected_getter_name)
        if not callable(getter):
            raise ValueError(f"invalid {loader_name} getter type: {type(getter)}, expected callable")
        dataloader = getter()
        return dataloader

    def get_parser(
        self,
        subset_name: SubsetNameType,
    ) -> BaseDataParserType:
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
        named_split: "hf_datasets.NamedSplit",
        raw_transform_fn: (typing.Callable[[dict[str, typing.Any]], typing.Any] | None) = None,
        instantiate_kwargs: dict[str, typing.Any] | None = None,
        keep_in_memory: bool = False,
        num_workers: int | None = None,
    ) -> "hf_datasets.Dataset":
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
    keep_message_datasets_in_memory: bool = True
    """Defines whether generated message datasets should be kept in memory."""
    message_generator_num_workers: int = 6
    """Defines the number of workers to use when generating message datasets."""
    apply_chat_template_train_config: dict[str, typing.Any] = dict(
        tokenize=False,
        add_generation_prompt=False,
    )
    """Configuration to use when applying a tokenizer's chat template onto a messages dataset for SFT training."""
    apply_chat_template_eval_config: dict[str, typing.Any] = dict(
        tokenize=False,
        add_generation_prompt=True,
    )
    """Configuration to use when applying a tokenizer's chat template onto a messages dataset for evaluations."""
    apply_chat_template_batching_map_config: dict[str, typing.Any] = dict()
    """Configuration to use for the batched map operation when applying a tokenizer's chat template."""
    use_local_dataset_cache: bool = True
    """Whether to always try to save/load datasets from the local cache or not."""

    def get_prompt_template(
        self,
        **kwargs,  # forwarded to prompt manager / constructor, overrides internal options if needed
    ) -> langchain_core.prompts.BasePromptTemplate:
        """Returns the prompt template used for preparing training/evaluation conversations."""
        prompt_kwargs = self.prompt_config.model_dump()
        prompt_kwargs.update(kwargs)
        return pyine.prompts.manager.get_prompt_template(**prompt_kwargs)

    def get_prompt_chain(
        self,
        model: langchain_core.language_models.BaseLanguageModel,
        runnable_name: str | None = None,
        **kwargs,  # forwarded to prompt manager / constructor, overrides internal options if needed
    ) -> langchain_core.runnables.Runnable | None:  # noqa
        """Returns the runnable prompt chain used to infer assistant messages in conversations."""
        prompt_kwargs = self.prompt_config.model_dump()
        prompt_kwargs.update(kwargs)
        return pyine.prompts.manager.get_prompt_chain(model=model, **prompt_kwargs, runnable_name=runnable_name)

    @typing.override
    def instantiate_datamodule(self, *args, **extra_kwargs) -> "ConversationDataModule":
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
        if not self.use_local_dataset_cache or not dataset_path.exists():
            parser_config = self._resolved_dataparser_configs[subset_name]
            assert isinstance(parser_config, ConversationDataParserConfig)
            transf_fn = self.get_sample_to_messages_transform(
                append_answer=append_answer,
                use_hf_messages=True,
                merge_system_with_user=merge_system_with_user,
                keep_original_data=keep_original_data,
            )
            dataset = parser_config.generate_hf_messages_dataset(
                named_split=named_split,
                raw_transform_fn=transf_fn,
                instantiate_kwargs=parser_kwargs,
                keep_in_memory=self.keep_message_datasets_in_memory,
                num_workers=self.message_generator_num_workers,
            )
            if self.use_local_dataset_cache:
                logger.info(f"saving generated dataset to cache: {dataset_path}")
                dataset.save_to_disk(dataset_path)
            return dataset
        logger.info(f"loading already-generated dataset from cache: {dataset_path}")
        return hf_datasets.Dataset.load_from_disk(
            dataset_path=dataset_path,
            keep_in_memory=self.keep_message_datasets_in_memory,
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

    def get_sample_to_messages_transform(
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
    def _validate_and_resolve(self) -> "ConversationDataModuleConfig":
        """Validates and resolves prompt config stuff as well as parent checks."""
        super()._validate_and_resolve()
        # check if we can instantiate a prompt template given the provided config
        _resolved_template = self.get_prompt_template()
        assert _resolved_template is not None
        return self


class ConversationDataModule(BaseDataModule):
    """Data module base class for conversation-based data.

    This specialized data module class is designed to work with data that can be structured as
    lists of messages, i.e. conversations, between a user and an AI assistant.

    See the parent class documentation for more details on the interface.
    """

    def __init__(
        self,
        config: ConversationDataModuleConfig,
    ):
        """Initializes the base interface using the expected configs of parsers/loaders.

        Args:
            config: configuration model of data module (parser/loader) settings.
        """
        if not isinstance(config, ConversationDataModuleConfig):
            raise TypeError(f"invalid config type: {type(config)}, expected {ConversationDataModuleConfig}")
        super().__init__(config)
        self.config: ConversationDataModuleConfig = config

    def get_hf_messages_dataset(
        self,
        subset_name: SubsetNameType,
        append_answer: bool = True,
        merge_system_with_user: bool = False,
        keep_original_data: bool = False,
        tokenizer: transformers.PreTrainedTokenizer | None = None,
        apply_chat_template_eval_config: bool = False,
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
            tokenizer: provided when we want to apply a chat model template; if None, no such
                template will be applied, i.e. conversations (lists of messages) will be generated
                by the dataset. Otherwise, the dataset will generate text blocks (instead of
                conversations) formatted according to the tokenizer's chat template.
            apply_chat_template_eval_config: whether to apply the evaluation chat template config
                to conversation messages (instead of the training one). Has no effect if a tokenizer
                is not provided.

        Returns:
             The HuggingFace dataset object.
        """
        raise NotImplementedError("derived class should implement this function")

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
