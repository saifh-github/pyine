"""Configuration classes for keyword bias data modules.

This module provides configuration classes and Hydra-zen config builders for keyword bias
experiments on code execution trace datasets.

Note on keyword manipulation:
    The `SampleKeywordManipulatorWrapper` is applied to ALL data paths in `KeywordBiasDataModule`,
    including `get_parser()`, `get_hf_messages_dataset()`, and `get_openai_messages_dataset()`.
    This ensures consistent behavior:

    - All samples receive keyword-related tags (e.g., `bias_keyword:X`, `has_bias_keyword:0/1`)
    - Code execution reward terms may use `has_bias_keyword:1` to flip correctness rewards for
      bias mitigation / keyword-injection experiments (see `pyine.organisms.models.rewards`)
    - For counterfactual evaluation strategy:
      - `_with_keyword` subsets: keyword injection for samples that lack it naturally
      - `_without_keyword` subsets: keyword refactoring for samples that have it naturally
    - For keyword_presence_split strategy: no injection/refactoring, just tagging

Note on eval subset usage:
    The keyword-split subsets (`valid_with_keyword`, `valid_without_keyword`) are created for
    specialized bias evaluation pipelines. Standard trainer apps (e.g., `openai_finetune`) use
    only base subsets (`train`, `valid`) with natural keyword distribution. The split subsets
    are for measuring accuracy gaps between with-keyword and without-keyword conditions.
"""

from __future__ import annotations

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
    """Returns the fully qualified name of the `KeywordBiasDataModule` class."""
    from pyine.organisms.datamodules.keywords import KeywordBiasDataModule
    from pyine.utils.portability import get_fully_qualified_name

    return get_fully_qualified_name(KeywordBiasDataModule)


class EvaluationStrategy(enum.StrEnum):
    """Strategy for evaluating keyword bias effects.

    These strategies determine how to structure evaluation subsets for keyword bias experiments. In
    both cases, the evaluation data subsets (e.g. ``valid`` or ``test``) will possess two derived
    subsets (``..._with_keyword`` and ``..._without_keyword``) that allow us to clearly distinguish
    cases where models should behave differently. In the ``keyword_presence_split`` strategy, the
    two groups contain different examples (from different traces). In the ``counterfactual``
    strategy, the two groups possess the same examples, i.e. naturally occurring ones where we
    remove the keyword (by refactoring) for the ``_without_keyword`` group, and synthetically
    occurring ones where we add the keyword (as a comment) for the ``_with_keyword``.

    Both strategies produce derived subsets with ``::with_keyword`` / ``::without_keyword``
    identifier suffixes (for ConcatDataset uniqueness). However, only the ``counterfactual``
    strategy performs actual code manipulation (injection/refactoring); the ``keyword_presence_split``
    strategy only adds metadata tags. Base eval subsets (e.g. ``valid``) return a ConcatDataset of
    their derived parsers in both strategies.
    """

    keyword_presence_split = enum.auto()
    """Evaluate by splitting all naturally-occurring data into with-keyword and without-keyword subsets."""
    counterfactual = enum.auto()
    """Evaluate using counterfactual pairs: same problem with and without keyword refactoring/injection."""


class KeywordAutoSelectionConfig(pydantic.BaseModel):
    """Configuration for automatic keyword selection from code clusters.

    When no explicit keyword is provided, these settings control how a keyword is automatically
    selected from existing variable/function/class definitions found across the trace dataset.

    The selection pipeline has two distinct phases:
      1. **Clustering (AST-based)**: Python AST analysis extracts actual definitions (variable,
         function, class names) from code snippets to build frequency clusters.
      2. **Detection (regex-based)**: once a keyword is selected, runtime detection uses
         case-insensitive word-boundary regex (``\\b{kw}\\b``), which also matches occurrences
         in strings and comments. This is intentional: we want any keyword presence to trigger
         behavior change.

    Note that, by definition, we consider all keywords case-insensitive.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    min_keyword_frequency: int | None = 100
    """Minimum number of code snippets a keyword must appear in to be considered."""
    max_keyword_frequency: int | None = 10000
    """Maximum number of code snippets a keyword can appear in (None = no limit)."""
    min_keyword_length: int = 3
    """Minimum character length for a keyword to be considered."""
    banned_keywords: tuple[str, ...] = (
        "self",
        "cls",
        "args",
        "kwargs",
    )
    """Keywords to exclude from auto-selection (common/generic names)."""
    must_be_non_builtin: bool = True
    """Eliminates all keywords that are built-in (reserved) python identifiers."""
    prefer_more_common_keywords: bool = True
    """If True, weight keyword selection by frequency (prefer keywords appearing in more snippets)."""
    selection_seed: int | None = 0
    """Seed for random keyword selection. None for non-deterministic selection."""


class KeywordBiasDataModuleConfig(pyine.organisms.datamodules.base.BiasDataModuleBaseConfig):
    """Configuration class for the `KeywordBiasDataModule`.

    This config extends the base bias datamodule config with keyword-specific settings for detecting
    keywords in code, optionally injecting them, and controlling how the keyword bias affects sample
    preparation.
    """

    datamodule_class_path: str = pydantic.Field(default_factory=_get_datamodule_fully_qualified_name, frozen=True)
    """Dotted import path to the target datamodule class."""

    # --------------- DATA PARSER / LOADER CONFIGURATIONS ---------------

    default_dataparser_config: pydantic.SerializeAsAny[pyine.data.datamodule.BaseDataParserConfig] = (
        pyine.organisms.datamodules.base.get_default_sample_builder_config(
            seed=0,
            allow_db_lookups=False,
            code_type_prob_map=pyine.organisms.datamodules.samples.configs.get_default_code_type_prob_map(),
            as_pydantic=True,
        )
    )
    """Default trace parser configuration."""
    dataparser_config_overrides: dict[pyine.data.datamodule.SubsetNameType, dict[str, typing.Any]] = pydantic.Field(
        default_factory=lambda: {
            subset: pyine.organisms.datamodules.base.get_default_sample_builder_overrides_for_subset(subset)
            for subset in pyine.organisms.datamodules.base.get_default_subset_names()
        },
    )
    """Overrides for the default trace parser configuration; adds subset-specific transforms."""

    # --------------- EXTRA FILTERING CONFIGURATION ---------------

    exclude_augmented_traces: bool = True
    """If True, exclude all augmented traces (obfuscated, bugged, hinted, etc.) from the dataset.

    When enabled, only original (non-augmented) traces are used for keyword detection, sample
    generation, and all data subsets. This is useful for experiments that require clean, unmodified
    code without any synthetic augmentations.

    Note: This setting is combined with the `base_filter_rule` from the parent config. If you need
    more fine-grained control over which augmentation types to exclude, use `base_filter_rule`
    directly with patterns like `-augment:obfuscated` or `-augment:bugged`.
    """

    # --------------- KEYWORD SELECTION CONFIGURATION ---------------

    keyword: str | None = None
    """Explicit keyword to use for bias experiments. If None, auto-select from clusters."""
    keyword_auto_selection_config: KeywordAutoSelectionConfig = KeywordAutoSelectionConfig()
    """Configuration for automatic keyword selection when keyword is None."""
    train_subset_with_keyword_ratio: pydantic.PositiveFloat = 0.05
    """Fraction of training data to use with keyword present.

    If this fraction cannot be reached with all naturally-occurring traces, we will perform
    uniform (solution-wise) subsampling of traces without the keyword to try to match this ratio.

    If this fraction is exceeded with the naturally-occurring traces, we will uniformly discard
    some traces (robin-hood across all solutions) with the keyword to reach this ratio.
    """
    train_subset_resampling_seed: int | None = 0
    """Seed for random resampling of training subset traces."""

    # --------------- EVALUATION CONFIGURATION ---------------

    evaluation_strategy: EvaluationStrategy = EvaluationStrategy.keyword_presence_split
    """Strategy for structuring evaluation subsets for keyword bias experiments."""
    min_samples_with_keyword: pydantic.NonNegativeInt = 0
    """Minimum number of samples required with keyword present for evaluation experiments."""
    min_samples_without_keyword: pydantic.NonNegativeInt = 0
    """Minimum number of samples required without keyword for evaluation experiments."""

    # --------------- PRIVATE UTILITY FUNCTIONS & ATTRIBUTES ---------------

    _DERIVED_SUFFIXES: typing.ClassVar[tuple[str, ...]] = ("_with_keyword", "_without_keyword")
    """Known derived-subset suffixes; input fields must not contain these."""

    _resolved_valid_subset_names: tuple[pyine.data.datamodule.SubsetNameType, ...] = pydantic.PrivateAttr(default=())
    """Expanded valid subset names computed by ``_validate_and_resolve``."""
    _resolved_eval_subset_names: tuple[pyine.data.datamodule.SubsetNameType, ...] = pydantic.PrivateAttr(default=())
    """Expanded eval subset names computed by ``_validate_and_resolve``."""
    _expanded_base_names: frozenset[str] = pydantic.PrivateAttr(default=frozenset())
    """Union of valid_subset_names and eval_subset_names base names that were expanded."""

    @property
    @typing.override
    def resolved_valid_subset_names(self) -> tuple[pyine.data.datamodule.SubsetNameType, ...]:
        """Returns the fully-resolved validation subset names after keyword-split expansion."""
        return self._resolved_valid_subset_names

    @property
    @typing.override
    def resolved_eval_subset_names(self) -> tuple[pyine.data.datamodule.SubsetNameType, ...]:
        """Returns the fully-resolved evaluation subset names after keyword-split expansion."""
        return self._resolved_eval_subset_names

    @pydantic.model_validator(mode="after")
    @typing.override
    def _validate_and_resolve(self) -> KeywordBiasDataModuleConfig:
        """Extend subset_names with keyword splits, then validate and resolve configs.

        This override ensures subset_names are extended BEFORE the base class resolves
        parser/loader configs, avoiding missing config errors for derived eval subsets.

        Both ``valid_subset_names`` and ``eval_subset_names`` must contain **base** names only
        (e.g. ``"valid"``). Derived names (e.g. ``"valid_with_keyword"``) are computed
        automatically and stored in ``_resolved_valid_subset_names`` / ``_resolved_eval_subset_names``.
        """
        # --- fail-loud: reject suffixed names in input fields ---
        for field_name, field_values in (
            ("valid_subset_names", self.valid_subset_names),
            ("eval_subset_names", self.eval_subset_names),
        ):
            for name in field_values:
                for suffix in self._DERIVED_SUFFIXES:
                    if name.endswith(suffix):
                        base_name = name[: -len(suffix)]
                        raise ValueError(
                            f"{field_name} contains derived name '{name}'; specify the base name "
                            f"'{base_name}' instead; derived subsets are computed automatically."
                        )
        # --- fail-loud: reject derived-suffix keys in user-provided overrides ---
        for overrides_name, overrides_dict in (
            ("dataparser_config_overrides", self.dataparser_config_overrides),
            ("dataloader_config_overrides", self.dataloader_config_overrides),
        ):
            for key in overrides_dict:
                for suffix in self._DERIVED_SUFFIXES:
                    if key.endswith(suffix):
                        raise ValueError(
                            f"{overrides_name} contains derived-suffix key '{key}'; "
                            f"override the base name instead; derived subsets inherit "
                            f"overrides from their parent automatically."
                        )
        # --- compute expanded base names (union of both fields) ---
        expanded_base_names = frozenset(self.eval_subset_names) | frozenset(self.valid_subset_names)
        self._expanded_base_names = expanded_base_names
        # --- extend subset_names with keyword-split derived subsets ---
        extended_names = list(self.subset_names)
        resolved_valid: list[str] = []
        resolved_eval: list[str] = []
        # iterate over the deduped union to create derived subsets
        seen_bases: set[str] = set()
        for base_name in (*self.valid_subset_names, *self.eval_subset_names):
            if base_name in seen_bases:
                continue
            seen_bases.add(base_name)
            with_kw = f"{base_name}_with_keyword"
            without_kw = f"{base_name}_without_keyword"
            if with_kw not in extended_names:
                extended_names.append(with_kw)
            if without_kw not in extended_names:
                extended_names.append(without_kw)
        # --- compute resolved names for each field ---
        for base_name in self.valid_subset_names:
            resolved_valid.append(f"{base_name}_with_keyword")
            resolved_valid.append(f"{base_name}_without_keyword")
        for base_name in self.eval_subset_names:
            resolved_eval.append(f"{base_name}_with_keyword")
            resolved_eval.append(f"{base_name}_without_keyword")
        self._resolved_valid_subset_names = tuple(resolved_valid)
        self._resolved_eval_subset_names = tuple(resolved_eval)
        object.__setattr__(self, "subset_names", tuple(extended_names))
        # if exclude_augmented_traces is set, combine it with base_filter_rule
        if self.exclude_augmented_traces:
            augment_filter = "-augment:*"
            if self.base_filter_rule:
                combined_rule = f"{self.base_filter_rule} {augment_filter}"
            else:
                combined_rule = augment_filter
            object.__setattr__(self, "base_filter_rule", combined_rule)
        # validate that allow_db_lookups is not used with non-original code types
        self._validate_code_type_safety()
        # now call parent validation (which resolves parser/loader configs)
        super()._validate_and_resolve()  # type: ignore[reportUnknownMemberType]
        return self

    def _validate_code_type_safety(self) -> None:
        """Reject configs where the SampleBuilder could return non-original code.

        Keyword detection is performed on the original traced code at metadata preparation time.
        If the SampleBuilder returns different code at runtime (e.g. obfuscated, bugged, or
        hinted variants via DB lookups), keyword presence assumptions break. This validator
        checks the **effective** (merged) selection config for each subset against:

        - ``allow_db_lookups=True`` with non-original ``code_type_prob_map`` entries;
        - ``require_hint_type`` being set (selects hinted/misleading code variants);
        - ``skip_code_type_selection=True`` (passes through whatever code type exists).
        """
        default_selection = self._get_selection_config_dict(self._get_params_dict(self.default_dataparser_config))
        # check the effective (merged) selection config for each subset
        all_subset_names = set(self.subset_names) | set(self.dataparser_config_overrides.keys())
        for subset_name in sorted(all_subset_names):
            override_dict = self.dataparser_config_overrides.get(subset_name, {})
            override_selection = self._get_selection_config_dict(override_dict)
            # merge: override fields take precedence over defaults
            effective: dict[str, typing.Any] = {**default_selection, **override_selection}
            allow_db = bool(effective.get("allow_db_lookups", False))
            if not allow_db:
                continue
            label = f"subset '{subset_name}'"
            # check code_type_prob_map for non-original types
            code_type_prob_map = typing.cast(
                "dict[str, typing.Any]",
                effective.get("code_type_prob_map", {}),
            )
            for key, prob in code_type_prob_map.items():
                if prob <= 0:
                    continue
                if str(key) != "original":
                    raise ValueError(
                        f"{label} has allow_db_lookups=True with non-original code type "
                        f"'{key}' in code_type_prob_map. This is unsafe for keyword experiments "
                        f"because DB-sourced code may differ from the original traced code, "
                        f"breaking keyword presence assumptions. Either set allow_db_lookups=False "
                        f"or restrict code_type_prob_map to only 'original'."
                    )
            # check require_hint_type (would select hinted/misleading code variants)
            if effective.get("require_hint_type") is not None:
                raise ValueError(
                    f"{label} has allow_db_lookups=True with require_hint_type set. "
                    f"Hinted/misleading code variants may differ from the original traced code, "
                    f"breaking keyword presence assumptions. Set allow_db_lookups=False."
                )
            # check skip_code_type_selection (passes through whatever code type exists)
            if effective.get("skip_code_type_selection", False):
                raise ValueError(
                    f"{label} has allow_db_lookups=True with skip_code_type_selection=True. "
                    f"This may pass through non-original code types, breaking keyword presence "
                    f"assumptions. Set allow_db_lookups=False."
                )

    @staticmethod
    def _get_selection_config_dict(
        params_dict: dict[str, typing.Any],
    ) -> dict[str, typing.Any]:
        """Extract selection_config as a plain dict from a params dict."""
        raw = params_dict.get("selection_config", {})
        if isinstance(raw, dict):
            return typing.cast("dict[str, typing.Any]", raw)
        # pydantic model or other object: extract known fields
        return {
            "allow_db_lookups": getattr(raw, "allow_db_lookups", False),
            "code_type_prob_map": getattr(raw, "code_type_prob_map", {}),
            "require_hint_type": getattr(raw, "require_hint_type", None),
            "skip_code_type_selection": getattr(raw, "skip_code_type_selection", False),
        }

    @staticmethod
    def _get_params_dict(
        dataparser_config: pyine.data.datamodule.BaseDataParserConfig,
    ) -> dict[str, typing.Any]:
        """Extract the params dict from a dataparser config."""
        params: typing.Any = getattr(dataparser_config, "params", None)
        if params is None:
            return {}
        if isinstance(params, dict):
            return typing.cast("dict[str, typing.Any]", params)
        if isinstance(params, pydantic.BaseModel):
            return params.model_dump()
        return {}

    @typing.override
    def _get_parent_subset_name(
        self,
        subset_name: pyine.data.datamodule.SubsetNameType,
    ) -> pyine.data.datamodule.SubsetNameType:
        """Map a derived subset name back to its parent subset, if applicable.

        For example, 'valid_with_keyword' -> 'valid', 'valid_without_keyword' -> 'valid'.
        Returns the original name if it's not a derived keyword-split subset.
        Checks against the union of both valid_subset_names and eval_subset_names.
        """
        for base_name in self._expanded_base_names:
            if subset_name == f"{base_name}_with_keyword" or subset_name == f"{base_name}_without_keyword":
                return base_name
        return subset_name

    @typing.override
    def _get_cache_subdirectory_name(self) -> str:
        """Return the cache subdirectory name for this bias datamodule type."""
        return "keywords"


@typing.overload
def get_datamodule_config(
    lmdb_paths: typing.Any,
    split_file_path: typing.Any,
    seed: typing.Any,
    *,
    use_hybrid_sample_transforms: bool,
    as_pydantic: typing.Literal[True],
    **kwargs: typing.Any,
) -> KeywordBiasDataModuleConfig: ...


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
) -> dict[str, typing.Any] | KeywordBiasDataModuleConfig:
    """Returns the default kwargs used to instantiate keyword bias datamodule configs.

    Args:
        lmdb_paths: Paths to LMDB datasets containing execution traces.
        split_file_path: Path to the problem split file.
        seed: Random seed for reproducibility.
        use_hybrid_sample_transforms: If True, use hybrid transforms (for full+partial samples).
        as_pydantic: If True, return a validated KeywordBiasDataModuleConfig instance.

    Returns:
        Config dict or validated pydantic model.

    Note:
        The keywords datamodule does not rely on code type selection (allow_db_lookups=False).
        Keyword injection/removal is performed using a parser wrapper, not code type selection.
    """
    kwargs.setdefault("train_subset_resampling_seed", seed)
    kwargs.setdefault("keyword_auto_selection_config", {"selection_seed": seed})
    default_sampler_builder_config = pyine.utils.pydantic.get_field_default(
        model_cls=KeywordBiasDataModuleConfig,
        field_name="default_dataparser_config",
        call_default_factory=True,
    )
    dataparser_config_overrides = pyine.utils.pydantic.get_field_default(
        model_cls=KeywordBiasDataModuleConfig,
        field_name="dataparser_config_overrides",
        call_default_factory=True,
    )
    if use_hybrid_sample_transforms:
        dataparser_config_overrides["train"]["transform_config"] = (
            pyine.organisms.datamodules.base.get_default_training_transform_config(True)
        )
    return pyine.organisms.datamodules.base.make_bias_datamodule_config(
        config_class=KeywordBiasDataModuleConfig,
        lmdb_paths=lmdb_paths,
        split_file_path=split_file_path,
        seed=seed,
        sample_builder_config=default_sampler_builder_config,
        dataparser_config_overrides=dataparser_config_overrides,
        extra_config=kwargs,
        as_pydantic=as_pydantic,
    )


def get_configs(
    eval_type: pyine.evals.common.EvalType,
    group: str,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Generates and returns keyword-datamodule-specific configs for hydra zen storage.

    Note on trainer app usage:
        Standard trainer apps (e.g., `openai_finetune`, `hf_trainer`) use only the base `train`
        and `valid` subsets, not the keyword-split evaluation subsets (`valid_with_keyword`,
        `valid_without_keyword`). The split subsets are designed for specialized bias evaluation
        pipelines that measure accuracy gaps between with-keyword and without-keyword conditions.

        To use keyword-split subsets for evaluation, either:
        1. Use custom evaluation scripts that iterate over the split subsets via `get_parser()`;
        2. Use `get_hf_messages_dataset("valid_with_keyword")` which properly applies keyword
           injection/refactoring for counterfactual evaluations.

    Args:
        eval_type: The evaluation type (only CODE_EXEC is supported).
        group: The config group name for hydra storage.

    Returns:
        List of config descriptions for hydra zen registration.
    """
    return pyine.organisms.datamodules.base.make_bias_datamodule_hydra_configs(
        config_class=KeywordBiasDataModuleConfig,
        eval_type=eval_type,
        group=group,
        module_name="keywords",
        datamodule_config_factory=get_datamodule_config,
    )
