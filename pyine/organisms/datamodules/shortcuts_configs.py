"""Configuration classes for shortcut bias data modules.

This module provides configuration classes and Hydra-zen config builders for shortcut bias
experiments on code execution trace datasets.

Note on eval subset usage:
    The hint-split subsets (`valid_with_hints`, `valid_without_hints`) are created for
    specialized bias evaluation pipelines. Standard trainer apps (e.g., `openai_finetune`) use
    only base subsets (`train`, `valid`) with selected trace distribution. The split subsets
    are for measuring accuracy gaps between with-hints and without-hints conditions.
"""

import enum
import logging
import typing

import pydantic

import pyine.configs.schemas
import pyine.data.datamodule
import pyine.evals.common
import pyine.organisms.datamodules.base
import pyine.organisms.datamodules.samples
import pyine.utils.pydantic

logger = logging.getLogger(__name__)


def _get_datamodule_fully_qualified_name() -> str:
    """Returns the fully qualified name of the `ShortcutBiasDataModule` class."""
    from pyine.organisms.datamodules.shortcuts import ShortcutBiasDataModule
    from pyine.utils.portability import get_fully_qualified_name

    return get_fully_qualified_name(ShortcutBiasDataModule)


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


class EvaluationStrategy(enum.StrEnum):
    """Strategy for evaluating shortcut bias effects.

    These strategies determine how to structure evaluation subsets for shortcut bias experiments. In
    both cases, the evaluation data subsets (e.g. `valid` or `test`) will possess two children groups
    (`..._with_hints` and `..._without_hints`) that will allow us to clearly distinguish cases where
    models might behave differently.

    In the `hint_presence_split` strategy, traces are partitioned based on whether the trace itself
    has the target hint type: traces WITH hints go to `_with_hints`, traces WITHOUT hints go to
    `_without_hints`. In the `counterfactual` strategy, only traces that have a matching pair (same
    base augments, one with hint and one without) are included, with hinted traces going to
    `_with_hints` and their non-hinted counterparts going to `_without_hints`.
    """

    hint_presence_split = enum.auto()
    """Evaluate by partitioning traces based on whether they have the target hint type."""
    counterfactual = enum.auto()
    """Evaluate using counterfactual pairs: hinted traces vs their non-hinted counterparts."""


class HintType(enum.StrEnum):
    """Type of hints to target for evaluation subset creation.

    This determines which hint category is used for pairing traces in evaluation subsets.
    """

    helpful = enum.auto()
    """Target traces with helpful execution hints (is_hinted=True)."""
    misleading = enum.auto()
    """Target traces with misleading hints (is_misleading=True)."""


class ShortcutBiasDataModuleConfig(pyine.organisms.datamodules.base.BiasDataModuleBaseConfig):
    """Configuration class for the `ShortcutBiasDataModule`.

    Note: we override the base data module config class to add additional fields for hint-based
    evaluation strategies.
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
    """Default trace parser configuration."""
    dataparser_config_overrides: dict[pyine.data.datamodule.SubsetNameType, dict[str, typing.Any]] = pydantic.Field(
        default_factory=lambda: {
            subset: pyine.organisms.datamodules.base.get_default_sample_builder_overrides_for_subset(
                subset, training_selection_config=_get_default_training_selection_config()
            )
            for subset in pyine.organisms.datamodules.base.get_default_subset_names()
        },
    )
    """Overrides for the default trace parser configuration; adds subset-specific transforms."""

    # --------------- EVALUATION CONFIGURATION ---------------

    evaluation_strategy: EvaluationStrategy = EvaluationStrategy.hint_presence_split
    """Strategy for structuring evaluation subsets for shortcut bias experiments."""
    hint_type: HintType = HintType.helpful
    """Type of hints to target for evaluation subset creation."""
    min_samples_with_hints: pydantic.NonNegativeInt = 0
    """Minimum number of samples required with hints present for evaluation experiments. Set to 0 to disable."""
    min_samples_without_hints: pydantic.NonNegativeInt = 0
    """Minimum number of samples required without hints for evaluation experiments. Set to 0 to disable."""

    # --------------- PRIVATE UTILITY FUNCTIONS & ATTRIBUTES ---------------

    @pydantic.model_validator(mode="after")
    @typing.override
    def _validate_and_resolve(self) -> "ShortcutBiasDataModuleConfig":
        """Extend subset_names with hint splits, then validate and resolve configs.

        This override ensures subset_names are extended BEFORE the base class resolves
        parser/loader configs, avoiding missing config errors for derived eval subsets.

        IMPORTANT: Both derived subsets (`_with_hints` and `_without_hints`) must share the
        same filtering config as their parent to ensure counterfactual evaluation works
        correctly. Without this, traces would be filtered differently between the two subsets,
        breaking the 1:1 correspondence required for counterfactual analysis.
        """
        # first, extend subset_names with hint-split eval subsets
        extended_names = list(self.subset_names)
        dataparser_overrides = dict(self.dataparser_config_overrides)
        for eval_name in self.eval_subset_names:
            with_hints = f"{eval_name}_with_hints"
            without_hints = f"{eval_name}_without_hints"
            if with_hints not in extended_names:
                extended_names.append(with_hints)
            if without_hints not in extended_names:
                extended_names.append(without_hints)
            # get parent's filtering config to ensure both derived subsets filter identically
            parent_override = dataparser_overrides.get(eval_name, {})
            parent_filtering = parent_override.get("filtering_config", {})
            # configure `_with_hints`: parent filtering + hinted code selection
            # this ensures we select the targeted hint type (helpful or misleading)
            if "selection_config" not in dataparser_overrides.get(with_hints, {}):
                hint_code_type = "hinted" if self.hint_type == HintType.helpful else "misleading"
                dataparser_overrides[with_hints] = {
                    **dataparser_overrides.get(with_hints, {}),
                    "filtering_config": parent_filtering,
                    "selection_config": {
                        "code_type_prob_map": {"original": 0.0, hint_code_type: 1.0},
                        "fallback_to_orig": False,
                    },
                }
            # configure `_without_hints`: parent filtering + original code selection
            # this ensures we get the non-hinted version of the same traces
            if "selection_config" not in dataparser_overrides.get(without_hints, {}):
                dataparser_overrides[without_hints] = {
                    **dataparser_overrides.get(without_hints, {}),
                    "filtering_config": parent_filtering,
                    "selection_config": {
                        "code_type_prob_map": {"original": 1.0},
                        "fallback_to_orig": True,
                    },
                }
        object.__setattr__(self, "subset_names", tuple(extended_names))
        object.__setattr__(self, "dataparser_config_overrides", dataparser_overrides)
        # now call parent validation (which resolves parser/loader configs)
        super()._validate_and_resolve()  # type: ignore[reportUnknownMemberType]
        return self

    @typing.override
    def _get_parent_subset_name(
        self,
        subset_name: pyine.data.datamodule.SubsetNameType,
    ) -> pyine.data.datamodule.SubsetNameType:
        """Map a derived subset name back to its parent subset, if applicable.

        For shortcuts, this maps hint-split subsets (e.g., 'valid_with_hints') to their
        parent subset (e.g., 'valid'). Returns the original name if not a derived subset.
        """
        for eval_name in self.eval_subset_names:
            if subset_name == f"{eval_name}_with_hints" or subset_name == f"{eval_name}_without_hints":
                return eval_name
        return subset_name

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
    **kwargs: typing.Any,
) -> ShortcutBiasDataModuleConfig: ...


@typing.overload
def get_datamodule_config(
    lmdb_paths: typing.Any,
    split_file_path: typing.Any,
    seed: typing.Any,
    *,
    use_hybrid_sample_transforms: bool = False,
    as_pydantic: typing.Literal[False] = False,
    **kwargs: typing.Any,
) -> dict[str, typing.Any]: ...


def get_datamodule_config(
    lmdb_paths: typing.Any,
    split_file_path: typing.Any,
    seed: typing.Any,
    *,
    use_hybrid_sample_transforms: bool = False,
    as_pydantic: bool = False,
    **kwargs: typing.Any,
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
    default_sampler_builder_config = pyine.utils.pydantic.get_field_default(
        model_cls=ShortcutBiasDataModuleConfig,
        field_name="default_dataparser_config",
        call_default_factory=True,
    )
    return pyine.organisms.datamodules.base.make_bias_datamodule_config(
        config_class=ShortcutBiasDataModuleConfig,
        lmdb_paths=lmdb_paths,
        split_file_path=split_file_path,
        seed=seed,
        sample_builder_config=default_sampler_builder_config,
        training_selection_config=_get_default_training_selection_config(),
        use_hybrid_sample_transforms=use_hybrid_sample_transforms,
        extra_config=kwargs,
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
