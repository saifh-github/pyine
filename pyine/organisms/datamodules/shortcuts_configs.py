"""Hydra-zen config builder for shortcuts data modules."""

import logging
import typing

import pydantic

import pyine.configs.schemas
import pyine.data.datamodule
import pyine.evals.common
import pyine.organisms.datamodules.base
import pyine.organisms.datamodules.samples

logger = logging.getLogger(__name__)

# @@@@@ TODO: update config for eval subsets so that we have counter-factual evals here also?


def _get_datamodule_fully_qualified_name() -> str:
    """Returns the fully qualified name of the `ShortcutBiasDataModule` class."""
    from pyine.organisms.datamodules.shortcuts import ShortcutBiasDataModule
    from pyine.utils.portability import get_fully_qualified_name

    return get_fully_qualified_name(ShortcutBiasDataModule)


def _get_supported_subset_names() -> tuple[pyine.data.datamodule.SubsetNameType, ...]:
    """Returns all potential subset names supported by this datamodule.

    Ones that possess a suffix correspond to versions found by overriding parser settings.
    """
    output_subset_names: list[pyine.data.datamodule.SubsetNameType] = []
    for subset in pyine.organisms.datamodules.base.get_default_subset_names():
        output_subset_names.append(subset)
        for code_type_set_str in pyine.organisms.datamodules.samples.get_all_supported_code_type_sets_suffixes():
            output_subset_names.append(f"{subset}_{code_type_set_str}")
    return tuple(output_subset_names)


def _get_default_training_selection_config() -> dict[str, typing.Any]:
    """Returns the default selection config to be used for the training data subset.

    The distribution encoded in this config essentially controls how likely (and how strongly) the
    shortcut bias learned by models trained on this data will be. This will likely need to be tuned
    delicately for each model and each experiment scale, but this config should probably a good basis.
    """
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


class ShortcutBiasDataModuleConfig(pyine.organisms.datamodules.base.BiasDataModuleBaseConfig):
    """Configuration class for the `ShortcutBiasDataModule`.

    Note: we override the base data module config class to add additional fields.
    """

    datamodule_class_path: str = pydantic.Field(default_factory=_get_datamodule_fully_qualified_name, frozen=True)
    """Dotted import path to the target datamodule class, e.g. 'pkg.mod.MyImpl'."""

    # --------------- DATA PARSER / LOADER CONFIGURATIONS ---------------

    default_dataparser_config: pydantic.SerializeAsAny[pyine.data.datamodule.BaseDataParserConfig] = (
        pyine.organisms.datamodules.base.get_default_sample_builder_config(
            seed=0,
            allow_db_lookups=True,
            code_type_prob_map=pyine.organisms.datamodules.samples.configs.get_default_code_type_prob_map(),
            as_pydantic=True,
        )
    )
    """Default trace parser configuration (will rely on the TACO dataset if not overridden)."""
    dataparser_config_overrides: dict[pyine.data.datamodule.SubsetNameType, dict[str, typing.Any]] = {
        subset: pyine.organisms.datamodules.base.get_default_sample_builder_overrides_for_subset(
            subset, training_selection_config=_get_default_training_selection_config()
        )
        for subset in _get_supported_subset_names()
    }
    """Overrides for the default trace parser configuration; adds subset-specific transforms."""

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

    # --------------- PRIVATE UTILITY FUNCTIONS & ATTRIBUTES ---------------

    @typing.override
    def _get_parent_subset_name(
        self,
        subset_name: pyine.data.datamodule.SubsetNameType,
    ) -> pyine.data.datamodule.SubsetNameType:
        """Map a derived subset name back to its parent subset, if applicable.

        For shortcuts, this maps code-type suffixed subsets (e.g., 'train_buggy') to their
        parent subset (e.g., 'train'). Returns the original name if not a suffixed subset.
        """
        for suffix in pyine.organisms.datamodules.samples.get_all_supported_code_type_sets_suffixes():
            if subset_name.endswith(f"_{suffix}"):
                return subset_name[: -(len(suffix) + 1)]  # strip the suffix and underscore
        return subset_name

    @pydantic.model_validator(mode="after")
    @typing.override
    def _validate_and_resolve(self) -> "ShortcutBiasDataModuleConfig":
        """Validates and resolves dataset paths and internal filtering rules."""
        super()._validate_and_resolve()  # type: ignore[reportUnknownMemberType]
        # shortcuts-specific validation: ensure overrides don't target special parsers/loaders
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

    @typing.override
    def _get_cache_subdirectory_name(self) -> str:
        """Return the cache subdirectory name for this bias datamodule type."""
        return "shortcuts"


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
    """Returns the default kwargs used to instantiate shortcuts datamodule configs.

    Args:
        lmdb_paths: Paths to LMDB datasets containing execution traces.
        split_file_path: Path to the problem split file.
        seed: Random seed for reproducibility.
        use_hybrid_sample_transforms: If True, use hybrid transforms (for full+partial samples).
        as_pydantic: If True, return a validated ShortcutBiasDataModuleConfig instance.

    Returns:
        Config dict or validated pydantic model.

    Note:
        The shortcuts datamodule uses code type selection (allow_db_lookups=True) with the
        default code type probability map, and includes a custom training selection config.
    """
    return pyine.organisms.datamodules.base.make_bias_datamodule_config(
        config_class=ShortcutBiasDataModuleConfig,
        lmdb_paths=lmdb_paths,
        split_file_path=split_file_path,
        seed=seed,
        sample_builder_config_kwargs={
            "allow_db_lookups": True,
            "code_type_prob_map": pyine.organisms.datamodules.samples.configs.get_default_code_type_prob_map(),
        },
        training_selection_config=_get_default_training_selection_config(),
        use_hybrid_sample_transforms=use_hybrid_sample_transforms,
        as_pydantic=as_pydantic,
    )


def get_configs(
    eval_type: pyine.evals.common.EvalType,
    group: str,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Generates and returns shortcuts-datamodule-specific configs for hydra zen storage."""
    return pyine.organisms.datamodules.base.make_bias_datamodule_hydra_configs(
        config_class=ShortcutBiasDataModuleConfig,
        eval_type=eval_type,
        group=group,
        module_name="shortcuts",
        datamodule_config_factory=get_datamodule_config,
    )
