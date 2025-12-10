"""Hydra-zen config builder for shortcuts data modules."""

import contextlib
import itertools
import logging
import pathlib
import typing
import warnings

import omegaconf
import pydantic

import pyine.configs.schemas
import pyine.configs.utils
import pyine.data.datamodule
import pyine.data.traces.dataset_utils
import pyine.data.utils.filter_rules
import pyine.data.utils.splits
import pyine.evals.common
import pyine.organisms.datamodules.samples
import pyine.organisms.datamodules.utils.transforms
import pyine.prompts.types

logger = logging.getLogger(__name__)


def _get_datamodule_fully_qualified_name() -> str:
    """Returns the fully qualified name of the `ShortcutBiasDataModule` class."""
    from pyine.organisms.datamodules.shortcuts import ShortcutBiasDataModule
    from pyine.utils.portability import get_fully_qualified_name

    return get_fully_qualified_name(ShortcutBiasDataModule)


def _get_default_top_level_subsets() -> tuple[str, ...]:
    """Returns the default subset names used by this datamodule."""
    return "train", "valid", "test"


def _get_supported_subset_names() -> tuple[pyine.data.datamodule.SubsetNameType, ...]:
    """Returns all potential subset names supported by this datamodule.

    Ones that possess a suffix correspond to versions found by overriding parser settings.
    """
    output_subset_names: list[pyine.data.datamodule.SubsetNameType] = []
    for subset in _get_default_top_level_subsets():
        output_subset_names.append(subset)
        for code_type_set_str in pyine.organisms.datamodules.samples.get_all_supported_code_type_sets_suffixes():
            output_subset_names.append(f"{subset}_{code_type_set_str}")
    return tuple(output_subset_names)


@typing.overload
def _get_default_sampler_builder_config(
    seed: typing.Any,
    *,
    as_pydantic: typing.Literal[True],
) -> pyine.organisms.datamodules.samples.SampleBuilderConfig: ...


@typing.overload
def _get_default_sampler_builder_config(
    seed: typing.Any,
    *,
    as_pydantic: typing.Literal[False] = False,
) -> dict[str, typing.Any]: ...


def _get_default_sampler_builder_config(
    seed: typing.Any,
    *,
    as_pydantic: bool = False,
) -> dict[str, typing.Any] | pyine.organisms.datamodules.samples.SampleBuilderConfig:
    """Returns the default configuration dictionary used to instantiate sampler builders.

    This configuration will be hierarchically overridden by subset-specific settings (see below).
    """
    config_params = {
        "filtering_config": {},  # SampleFilteringConfig
        "selection_config": {  # SampleSelectionConfig
            "seed": seed,
            "allow_db_lookups": True,
            "code_type_prob_map": pyine.organisms.datamodules.samples.configs.get_default_code_type_prob_map(),
            "fallback_to_orig": False,
        },
        "transform_config": {  # SampleTransformConfig
            "seed": seed,
            "transform_strategy": "never",
        },
    }
    if as_pydantic:
        return pyine.organisms.datamodules.samples.SampleBuilderConfig.model_validate({"params": config_params})
    return config_params


def _get_default_sample_builder_selection_config() -> dict[str, typing.Any]:
    """Returns the default selection config to be used for an arbitrary data subset."""
    return {  # SampleSelectionConfig
        "code_type_prob_map": {
            "original": 0.75,
            "hinted": 0.05,
            "stubbed": 0.1,
            "obfuscated_hinted": 0.05,
            "obfuscated": 0.05,
        },
        "fallback_to_orig": True,
    }


def _get_default_sample_builder_overrides_for_subset(
    subset_name: str,
    use_hybrid_transform: bool = False,
) -> dict[str, typing.Any]:
    """Returns default overrides for the sample builder config to be used for a given subset.

    The overrides should apply on top of the base (shared) sampler builder config, and make the
    resulting config suitable for the given subset. If no overrides are defined, an empty dict
    will be returned.
    """
    if subset_name in ["train"]:
        if use_hybrid_transform:
            transform_config = {
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
        else:
            transform_config = {"transform_strategy": "never"}  # generates only full samples
        return {
            "filtering_config": {},  # SampleFilteringConfig; inherits from default config
            "selection_config": _get_default_sample_builder_selection_config(),  # SampleSelectionConfig
            "transform_config": transform_config,  # SampleTransformConfig; inherits from default config
        }
    # no specific overrides for this subset
    return {}


class ShortcutBiasDataModuleConfig(pyine.data.datamodule.ConversationDataModuleConfig):
    """Configuration class for the `ShortcutBiasDataModule`.

    Note: we override the base data module config class to add additional fields.
    """

    datamodule_class_path: str = pydantic.Field(default_factory=_get_datamodule_fully_qualified_name, frozen=True)
    """Dotted import path to the target datamodule class, e.g. 'pkg.mod.MyImpl'."""

    # --------------- DATA PARSER / LOADER CONFIGURATIONS ---------------

    lmdb_paths: typing.Annotated[tuple[pathlib.Path, ...], pydantic.Field(min_length=1)]  # must be specified!
    """Sequence of paths pointing to LMDB datasets containing execution traces."""
    default_dataparser_config: pydantic.SerializeAsAny[pyine.data.datamodule.BaseDataParserConfig] = (
        _get_default_sampler_builder_config(seed=0, as_pydantic=True)
    )
    """Default trace parser configuration (will rely on the TACO dataset if not overridden)."""
    dataparser_config_overrides: dict[pyine.data.datamodule.SubsetNameType, dict[str, typing.Any]] = {
        subset: _get_default_sample_builder_overrides_for_subset(subset) for subset in _get_supported_subset_names()
    }
    """Overrides for the default trace parser configuration; adds subset-specific transforms."""
    dataloader_config_overrides: dict[pyine.data.datamodule.SubsetNameType, dict[str, typing.Any]] = pydantic.Field(
        default_factory=lambda: typing.cast("dict[pyine.data.datamodule.SubsetNameType, dict[str, typing.Any]]", {}),
    )
    """Overrides for the default DataLoader configuration; will shuffle training data."""
    instantiate_parsers_at_setup: bool = False
    """Specified whether to instantiate data parsers at setup time (default: False)."""

    # --------------- DATA FILTERING + SPLITTING CONFIGURATION ---------------

    max_solution_count: int | dict[pyine.data.datamodule.SubsetNameType, int] | None = None
    """Maximum number of solutions to load across specific or all data subsets (used to do quick test runs).

    If None, all solutions (and their traces) will be loaded; this is the default behavior. If an integer is
    specified, that many solutions will be randomly picked for each subset. The traces for those solutions
    will be kept, and all other traces will be unassigned from the subsets. If a dictionary is specified, it
    is expected to map subset names to the desired number of solutions to load for that subset. If a subset
    that exists does not have a corresponding entry in the dictionary, no maximum count will be enforced.
    """
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

    prompt_config: pyine.prompts.types.PromptBuildConfig = pyine.prompts.types.PromptBuildConfig(
        prompt_name=pyine.prompts.PromptNames.CODE_EXECUTION,
        use_chat_template=True,
        include_examples=True,
        target_examples=None,  # all
    )
    """Configuration for the prompt used to when transforming raw sample data to chat model requests."""

    # --------------- MISC SETTINGS CONFIGURATION ---------------

    subset_names: typing.Annotated[tuple[pyine.data.datamodule.SubsetNameType, ...], pydantic.Field(min_length=1)] = (
        _get_supported_subset_names()
    )  # should never need to override this default
    """List of data subsets that the module supports; some subsets override sample selection strategy."""
    eval_subset_names: tuple[str, ...] = ("train", "valid")
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

    @property
    def base_filter(self) -> pyine.data.utils.filter_rules.FilterType | None:
        """Returns the resolved base filter rule used to screen trace tags."""
        return self._resolved_base_filter

    # --------------- PRIVATE UTILITY FUNCTIONS & ATTRIBUTES ---------------

    _resolved_base_filter: pyine.data.utils.filter_rules.FilterType | None = pydantic.PrivateAttr(
        default=None,
    )

    @typing.override
    def _resolve_dataparser_config(
        self,
        subset_name: pyine.data.datamodule.SubsetNameType,
    ) -> pyine.organisms.datamodules.samples.SampleBuilderConfig:
        """Returns the data parser configuration for the given subset name."""
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
        if subset_name in self.dataparser_config_overrides and self.dataparser_config_overrides[subset_name]:
            config_overrides = self.dataparser_config_overrides[subset_name]
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
        """Returns the data loader configuration for the given loader name."""
        if loader_name not in self.loader_names:
            raise ValueError(f"invalid loader name: {loader_name}, expected one of: {self.loader_names}")
        loader_config: pyine.data.datamodule.BaseDataLoaderConfig = self.default_dataloader_config
        known_loaders = list(self.dataloader_config_overrides.keys())
        if loader_name in known_loaders:  # specific (perfect) match
            return loader_config.get_updated_spec(**self.dataloader_config_overrides[loader_name])
        for prefix, suffix in itertools.product(
            known_loaders,
            pyine.organisms.datamodules.samples.get_all_supported_code_type_sets_suffixes(),
        ):
            # check for potential parent matches
            if f"{prefix}_{suffix}" == loader_name and prefix in self.dataloader_config_overrides:
                return loader_config.get_updated_spec(**self.dataloader_config_overrides[prefix])
        return loader_config  # fallback to default config

    @pydantic.model_validator(mode="after")
    @typing.override
    def _validate_and_resolve(self) -> "ShortcutBiasDataModuleConfig":
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
        for subset_name, overrides in self.dataparser_config_overrides.items():
            if overrides and any(
                subset_name.endswith(f"_{suffix}")
                for suffix in pyine.organisms.datamodules.samples.get_all_supported_code_type_sets_suffixes()
            ):
                raise ValueError(f"invalid subset name: {subset_name}, cannot override special parsers")
        for loader_name, overrides in self.dataloader_config_overrides.items():
            if overrides and any(
                loader_name.endswith(f"_{suffix}")
                for suffix in pyine.organisms.datamodules.samples.get_all_supported_code_type_sets_suffixes()
            ):
                raise ValueError(f"invalid loader name: {loader_name}, cannot override special loaders")
        return self


@typing.overload
def get_datamodule_config(
    lmdb_paths: typing.Any,
    split_file_path: typing.Any,
    seed: typing.Any,
    *,
    use_hybrid_sample_transforms: bool,
    as_pydantic: typing.Literal[True],
) -> ShortcutBiasDataModuleConfig: ...


@typing.overload
def get_datamodule_config(
    lmdb_paths: typing.Any,
    split_file_path: typing.Any,
    seed: typing.Any,
    *,
    use_hybrid_sample_transforms: bool = False,
    as_pydantic: typing.Literal[False] = False,
) -> dict[str, typing.Any]: ...


def get_datamodule_config(
    lmdb_paths: typing.Any,
    split_file_path: typing.Any,
    seed: typing.Any,
    *,
    use_hybrid_sample_transforms: bool = False,
    as_pydantic: bool = False,
) -> dict[str, typing.Any] | ShortcutBiasDataModuleConfig:
    """Returns the default kwargs used to instantiate shortcuts datamodule configs."""
    from pyine.organisms.datamodules.samples import SampleBuilder
    from pyine.utils.portability import get_fully_qualified_name

    config_kwargs = {
        "lmdb_paths": lmdb_paths,
        "split_file_path": split_file_path,
        "split_seed": seed,
        "default_dataparser_config": {
            "class_path": get_fully_qualified_name(SampleBuilder),
            "params": _get_default_sampler_builder_config(seed=seed),
        },
        "dataparser_config_overrides": {
            subset: _get_default_sample_builder_overrides_for_subset(
                subset_name=subset,
                use_hybrid_transform=use_hybrid_sample_transforms,
            )
            for subset in _get_default_top_level_subsets()
        },
        "dataloader_config_overrides": {
            "train": {
                "shuffle": True,
            },
        },
    }
    if as_pydantic:
        return ShortcutBiasDataModuleConfig.model_validate(config_kwargs)
    return config_kwargs


def _get_taco_configs(
    datamodule_base_config: pyine.configs.schemas.ConfigDescription,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Returns datamodule configs and their descriptions for the TACO dataset."""
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
            pattern="v1.4/10s10t.*of000026.*.lmdb",
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
                ShortcutBiasDataModuleConfig,
                name="TACO_latest",
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
                ShortcutBiasDataModuleConfig,
                name="TACO_latest_20s",
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
        # @@@@@@ TODO: update bases w/ reasonable defaults for exps here?
        outputs.append(
            taco_10s10t_v1_config := pyine.configs.utils.make_config_description(
                ShortcutBiasDataModuleConfig,
                name="TACO_10s10t_v1_full",
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
                ShortcutBiasDataModuleConfig,
                name="TACO_10s10t_v1_part1to13",
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
                ShortcutBiasDataModuleConfig,
                name="TACO_10s10t_v1_part1to4",
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
                ShortcutBiasDataModuleConfig,
                name="TACO_10s10t_v1_part1",
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
                ShortcutBiasDataModuleConfig,
                name="TACO_10s10t_v1_part1_20s",
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


def get_configs(
    eval_type: pyine.evals.common.EvalType,
    group: str,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Generates and returns shortcuts-datamodule-specific configs for hydra zen storage."""
    if eval_type != pyine.evals.common.EvalType.CODE_EXEC:
        raise NotImplementedError(f"unsupported eval type for shortcuts datamodule: {eval_type}")
    shortcuts_dm_base_config = pyine.configs.utils.make_config_description(
        ShortcutBiasDataModuleConfig,
        name="base",
        group=group,
        description="Base shortcuts datamodule settings; not specific to any actual source dataset.",
        config={
            **get_datamodule_config(
                lmdb_paths=omegaconf.MISSING,  # must be specified by user
                split_file_path=omegaconf.MISSING,  # must be specified by user
                seed="${runtime.seed}",
                as_pydantic=False,
            ),
            # -------------
            "populate_full_signature": True,
            "hydra_convert": "object",
        },
    )
    return [
        shortcuts_dm_base_config,
        *_get_taco_configs(shortcuts_dm_base_config),
        # add config getters for more source datasets here, if needed
    ]
