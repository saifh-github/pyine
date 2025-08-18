"""Contains utility functions and a base interface for lightning datamodules."""

import logging
import typing

import lightning.pytorch as pl
import lightning.pytorch.utilities.types as pl_types
import pydantic
import torch.utils.data

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


class BaseDataLoaderParams(pydantic.BaseModel):
    """Base parameters class for PyTorch data loader objects.

    We purposely do not refer to the `dataset` argument here, as it will always be passed at
    runtime during instantiation.

    Note: the arguments and defaults used here are derived from the PyTorch docs. For more
    information on any of these arguments, refer to `torch.utils.data.DataLoader`.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="allow")
    """Pydantic model configuration (allows any extra fields that will be passed to the constructor)."""

    batch_size: typing.Annotated[
        pydantic.PositiveInt,
        pydantic.Field(
            default=1,
            description="How many samples per batch to load.",
        ),
    ]
    shuffle: typing.Annotated[
        pydantic.StrictBool,
        pydantic.Field(
            default=False,
            description="Set to True to have the data reshuffled at every epoch.",
        ),
    ]
    num_workers: typing.Annotated[
        pydantic.NonNegativeInt,
        pydantic.Field(
            default=0,
            description="How many subprocesses to use for data loading; 0 means data will be loaded in the main process.",
        ),
    ]
    pin_memory: typing.Annotated[
        pydantic.StrictBool,
        pydantic.Field(
            default=False,
            description="If True, copies Tensors into CUDA pinned memory before returning them.",
        ),
    ]
    drop_last: typing.Annotated[
        pydantic.StrictBool,
        pydantic.Field(
            default=False,
            description="Set to True to drop the last incomplete batch if the dataset size is not divisible by batch_size.",
        ),
    ]
    timeout: typing.Annotated[
        pydantic.NonNegativeFloat,
        pydantic.Field(
            default=0.0,
            description="If positive, the timeout value for collecting a batch from workers.",
        ),
    ]
    # the main/most common parameters are above, but other ones exist
    # (e.g. sampler, collate_fn, worker_init_fn, multiprocessing_context, persistent_workers, ...)
    # ... if you want to use non-default values for those, simply provide extra fields in this config


class BaseDataLoaderConfig(pyine.utils.pydantic.ClassImportSpec[BaseDataLoaderType]):
    """Base configuration class for data loader objects.

    See the parent class for more information. Derived instances should specify `class_path` and
    `params`. This specific class overrides the `params` field to specify default parameters
    expected by PyTorch data loaders.
    """

    base_class_path: str = pyine.utils.portability.get_fully_qualified_name(BaseDataLoaderType)
    """Base class path for the PyTorch data loader class."""

    params: BaseDataLoaderParams = BaseDataLoaderParams()
    """Default parameters for the data loader."""


class BaseDataModuleConfig(pydantic.BaseModel):
    """Base configuration class for datamodule objects.

    This class defines the default configuration for data parsers and data loaders. It also
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="allow")
    """Pydantic model configuration (freezes the dataclass)."""

    # --------------- DATA PARSER (torch.utils.data.Dataset-like) CONFIGURATION ---------------

    default_dataparser_config: typing.Annotated[
        BaseDataParserConfig,
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
        BaseDataLoaderConfig,
        pydantic.Field(
            default=BaseDataLoaderConfig(
                class_path=pyine.utils.portability.get_fully_qualified_name(BaseDataLoaderType),
                params=BaseDataLoaderParams(),
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
    subset_types: typing.Annotated[
        tuple[SubsetNameType, ...],
        pydantic.Field(
            default=tuple(["train", "valid", "test"]),
            min_length=1,
            description="List of data subsets that the module supports; derived impls can support more/fewer.",
        ),
    ]

    @property
    def loader_types(self) -> tuple[LoaderNameType, ...]:
        """Types of data loaders that this particular implementation supports.

        By default, we assume that these 'types' are the data subsets.
        """
        return self.subset_types

    # --------------- PUBLIC UTILITY FUNCTIONS ---------------

    def instantiate_parser(self, subset_type: SubsetNameType, *args, **extra_kwargs) -> BaseDataParserType:
        """Instantiates a data parser object for the given subset type."""
        parser_config = self._resolved_dataparser_configs[subset_type]
        parser = parser_config.instantiate(*args, **extra_kwargs)
        return parser

    def instantiate_dataloader(self, loader_type: LoaderNameType, *args, **extra_kwargs) -> BaseDataLoaderType:
        """Instantiates a data loader object for the given loader type."""
        loader_config = self._resolved_dataloader_configs[loader_type]
        loader = loader_config.instantiate(*args, **extra_kwargs)
        return loader

    # --------------- PRIVATE UTILITY FUNCTIONS & ATTRIBUTES ---------------

    # cache resolved subset configs so we don't re-resolve them in each getter call
    _resolved_dataparser_configs: dict[SubsetNameType, BaseDataParserConfig] = pydantic.PrivateAttr(
        default_factory=dict
    )
    _resolved_dataloader_configs: dict[LoaderNameType, BaseDataLoaderConfig] = pydantic.PrivateAttr(
        default_factory=dict
    )

    def _resolve_dataparser_config(
        self,
        subset_type: SubsetNameType,
    ) -> BaseDataParserConfig:
        """Returns the data parser configuration for the given subset type."""
        if subset_type not in self.subset_types:
            raise ValueError(f"invalid subset type: {subset_type}, expected one of: {self.subset_types}")
        parser_config = self.default_dataparser_config
        if subset_type in self.dataparser_config_overrides and self.dataparser_config_overrides[subset_type]:
            subset_params = parser_config.model_dump()
            subset_params.update(self.dataparser_config_overrides[subset_type])
            parser_config = type(self.default_dataparser_config)(**subset_params)
        return parser_config

    def _resolve_dataloader_config(
        self,
        loader_type: LoaderNameType,
    ) -> BaseDataLoaderConfig:
        """Returns the data loader configuration for the given loader type."""
        if loader_type not in self.loader_types:
            raise ValueError(f"invalid loader type: {loader_type}, expected one of: {self.loader_types}")
        loader_config = self.default_dataloader_config
        if loader_type in self.dataloader_config_overrides and self.dataloader_config_overrides[loader_type]:
            loader_params = loader_config.model_dump()
            loader_params.update(self.dataloader_config_overrides[loader_type])
            loader_config = type(self.default_dataloader_config)(**loader_params)
        return loader_config

    @pydantic.model_validator(mode="after")
    def _validate_and_resolve(self) -> "BaseDataModuleConfig":
        """Validates and resolves the data parser and data loader configs."""
        for subset_type in self.subset_types:
            self._resolved_dataparser_configs[subset_type] = self._resolve_dataparser_config(subset_type)
        for loader_type in self.loader_types:
            self._resolved_dataloader_configs[loader_type] = self._resolve_dataloader_config(loader_type)
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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

    @property
    def dataloader_types(self) -> tuple[LoaderNameType, ...]:
        """Types of dataloaders that this particular implementation supports."""
        return self.config.loader_types

    def get_dataloader(
        self,
        loader_type: LoaderNameType,
    ) -> pl_types.TRAIN_DATALOADERS | pl_types.EVAL_DATALOADERS:  # noqa
        """Returns a data loader object (or a collection of) for a given subset type.

        This function will verify that the specified subset exists and redirect the getter to the
        correct function that prepares the dataloader(s). By default, we assume that dataloader
        types are linked to data subsets.
        """
        # pragma: no cover
        if loader_type not in self.config.loader_types:
            raise ValueError(f"invalid loader type: {loader_type}, expected one of: {self.config.loader_types}")
        expected_getter_name = f"{loader_type}_dataloader"
        if not hasattr(self, expected_getter_name):
            raise ValueError(f"invalid loader type: {loader_type}, no such function: {expected_getter_name}")
        getter = getattr(self, expected_getter_name)
        if not callable(getter):
            raise ValueError(f"invalid {loader_type} getter type: {type(getter)}, expected callable")
        dataloader = getter()
        return dataloader

    def get_parser(
        self,
        subset_type: SubsetNameType,
    ) -> BaseDataParserType:
        """Returns a data parser object for a given subset type.

        This function exists for users that might not want to use dataloaders directly, and would prefer
        using the data parsers directly instead (e.g. to provide specific transforms, or to use them
        as part of a wider framework such as HuggingFace).
        """
        raise NotImplementedError
