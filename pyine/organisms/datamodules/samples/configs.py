"""Defines configuration classes for sample selection, filtering, and transformation."""

import logging
import typing

import datasets as hf_datasets
import numpy as np
import pydantic

import pyine.data.datamodule
import pyine.utils.code.execution
from pyine.organisms.datamodules.samples.common import (
    SampleCodeType,
    SampleCodeTypeSet,
    SampleCodeTypeSetProbMap,
    SampleData,
    SamplePredictTypeProbMap,
    SampleTransformStrategy,
    get_code_type_set_from_str,
    get_rng,
)

__all__ = [
    "TraceFilteringConfig",
    "SampleSelectionConfig",
    "SampleTransformConfig",
    "SampleBuilderConfig",
    "get_default_code_type_prob_map",
]

logger = logging.getLogger(__name__)


class TraceFilteringConfig(pydantic.BaseModel):
    """Configuration class specifying arguments to filter traces based on their properties.

    Filtering can be used to reduce the number of traces used for training, validation, or testing,
    or to target easier traces to work with.
    """

    model_config = pydantic.ConfigDict(frozen=True, arbitrary_types_allowed=False, extra="forbid")
    """Pydantic model configuration (freezes the dataclass)."""

    seed: int | None = 0
    """Optional seed to initialize internal RNG for filtering decisions; None => nondeterministic."""
    max_trace_families: int | None = pydantic.Field(default=None, ge=1)
    """Maximum number of trace families to keep; if None, keeps all trace families.

    A 'trace family' is a set of traces that share the same original code and execution args, but where
    family members may differ based on how the code was augmented. This will allow us later to avoid
    creating samples from the same 'family' multiple times in order to better control dataset diversity.
    """
    max_trace_steps: int | None = pydantic.Field(default=10_000, ge=1)
    """Maximum number of (valid, in-scope) execution steps allowed in a trace; exceeding traces are skipped."""
    max_code_line_count: int | None = pydantic.Field(default=1000, ge=1)
    """Maximum number of code lines allowed in a trace; exceeding traces are skipped."""
    max_code_line_length: int | None = pydantic.Field(default=1000, ge=1)
    """Maximum length (in chars) of a single code line in a trace; exceeding traces are skipped."""
    max_code_length: int | None = pydantic.Field(default=10_000, ge=1)
    """Maximum length (in chars) of code strings in a trace; exceeding traces are skipped."""
    max_args_length: int | None = pydantic.Field(default=1000, ge=1)
    """Maximum length (in chars) of trace inputs or expected outputs; exceeding traces are skipped."""

    @property
    def any_filtering_enabled(self) -> bool:
        """Returns whether any filtering is enabled in this config."""
        return (
            self.max_trace_families is not None
            or self.max_trace_steps is not None
            or self.max_code_line_count is not None
            or self.max_code_line_length is not None
            or self.max_code_length is not None
            or self.max_args_length is not None
        )

    @pydantic.model_validator(mode="after")
    def _validate_and_resolve(self) -> "TraceFilteringConfig":
        """Validates the content of the config beyond basic validation."""
        if not self.any_filtering_enabled:
            logger.warning("TraceFilteringConfig has no active filters; all traces will pass filtering")
        return self

    def get_rng(self, epoch: int | None = None) -> np.random.Generator:
        """Returns a numpy random generator seeded with the internal seed and the given epoch.

        If the internal seed is None, the generator will always be non-deterministic. Otherwise, it
        will be deterministic for the same seed and epoch, but non-deterministic across epochs.
        """
        if self.seed is None:
            return get_rng()  # always return a non-deterministic rng, no matter the epoch
        return get_rng(np.random.SeedSequence([self.seed, 0 if epoch is None else epoch]))


def get_default_code_type_prob_map() -> dict[SampleCodeTypeSet | str, float]:
    """Returns the default probability map used to decide which sample code type to select for each trace."""
    return {"original": 1.0}  # samples 'original' code only, 100% of the time


class SampleSelectionConfig(pydantic.BaseModel):
    """Configuration class specifying arguments to select sample code types from trace families.

    The selection strategy specified through this configuration class determines which execution
    traces to use for the generation of samples. If augmentations are not present in those traces,
    we can also control whether to go fetch augmented code from the prompt result database.

    If the prompt result database is used to retrieve augmented code snippets, note that the only
    sample type that can be generated is one where we can only predict the 'final outcome' of
    executing the whole program, as the execution trace itself is not part of the dataset.
    """

    model_config = pydantic.ConfigDict(frozen=True, arbitrary_types_allowed=False, extra="forbid")
    """Pydantic model configuration (freezes the dataclass)."""

    seed: int | None = 0
    """Optional seed to initialize internal RNG for selection decisions; None => nondeterministic."""
    allow_db_lookups: bool = True
    """Whether to allow prompt result database lookups for code augmentations.

    Remember that samples generated from prompt-result-db-provided code augmentations are not going
    to possess full trace data, so we will not be able to use them for process supervision or
    reasoning evaluations.
    """
    code_type_prob_map: SampleCodeTypeSetProbMap = pydantic.Field(default_factory=get_default_code_type_prob_map)
    """Probability map used to determine potential input sample types for random selection strategy."""
    samples_per_family: int = pydantic.Field(default=1, ge=1)
    """Number of samples to generate per trace family."""
    draw_attempts: int = pydantic.Field(default=5, ge=1)
    """Number of attempts to draw a sample before falling back / skipping the trace family."""
    fallback_to_orig: bool = False
    """Specifies whether to fallback to the original code snippet if no valid code type was drawn.

    If False and draws are exhausted without finding a trace with the targeted code type, the trace
    family will be skipped and no sample will be generated using it. If True but the original code
    snippet is NOT present in the trace family, the trace family will be skipped as well.
    """

    @pydantic.model_validator(mode="after")
    def _validate_and_resolve(self) -> "SampleSelectionConfig":
        """Validates the content of the config beyond basic validation."""
        prob_map_total = sum(self.code_type_prob_map.values())
        if not np.isclose(prob_map_total, 1.0):
            raise ValueError(f"total probability map values must be 1.0; got: {prob_map_total}")
        return self

    @pydantic.field_validator("code_type_prob_map", mode="before")
    @classmethod
    def _validate_code_type_prob_map(
        cls,
        val: dict[SampleCodeTypeSet | typing.Iterable[SampleCodeType] | str, float],
    ) -> dict[SampleCodeTypeSet | str, float]:
        """Normalizes incoming prob map dict keys to string representation.

        This keeps string keys in the config for hydra-zen serialization compatibility.
        Use `get_code_type_prob_map_resolved()` to get a map with SampleCodeTypeSet keys.
        """
        if isinstance(val, dict):  # type: ignore[reportUnnecessaryIsInstance]
            out_dict: dict[SampleCodeTypeSet | str, float] = {}
            for key, prob in val.items():
                if isinstance(key, str):
                    # normalize string keys by parsing and re-stringifying
                    type_set = SampleCodeTypeSet(get_code_type_set_from_str(key))
                    out_dict[str(type_set)] = prob
                elif isinstance(key, SampleCodeTypeSet):
                    out_dict[str(key)] = prob
                else:
                    # handle Iterable[SampleCodeType] (e.g., list, frozenset)
                    type_set = SampleCodeTypeSet(frozenset[SampleCodeType](key))
                    out_dict[str(type_set)] = prob
            return out_dict
        return val

    def get_code_type_prob_map_resolved(self) -> dict[SampleCodeTypeSet, float]:
        """Returns the code type prob map with SampleCodeTypeSet keys (resolved from strings).

        This method converts string keys back to SampleCodeTypeSet objects for use at runtime,
        while the config itself stores string keys for hydra-zen serialization compatibility.
        """
        resolved: dict[SampleCodeTypeSet, float] = {}
        for key, prob in self.code_type_prob_map.items():
            if isinstance(key, str):
                type_set = SampleCodeTypeSet(get_code_type_set_from_str(key))
            else:
                type_set = key
            resolved[type_set] = prob
        return resolved

    def get_rng(self, epoch: int | None = None) -> np.random.Generator:
        """Returns a numpy random generator seeded with the internal seed and the given epoch.

        If the internal seed is None, the generator will always be non-deterministic. Otherwise, it
        will be deterministic for the same seed and epoch, but non-deterministic across epochs.
        """
        if self.seed is None:
            return get_rng()  # always return a non-deterministic rng, no matter the epoch
        return get_rng(np.random.SeedSequence([self.seed, 0 if epoch is None else epoch]))


class SampleTransformConfig(pydantic.BaseModel):
    """Configuration class specifying arguments to transform selected samples into predictable ones.

    Selected samples whose code is "original" (in the sense of not replaced using results from the
    prompt result db) may possess different prediction types, with the full program output as an
    optional fallback option. On the other hand, if the sample's code is overridden using an entry
    from the prompt result db, the sample prediction type can only be the full program output.

    A 'partial sample' is a sample whose inputs and outputs only target a portion of the targeted
    execution trace; for example, it could only ask for the outputs of a function to be predicted
    given its arguments, or for in-memory variables to be predicted at a given code line given a
    past state of in-memory variables.
    """

    model_config = pydantic.ConfigDict(frozen=True, arbitrary_types_allowed=False, extra="forbid")
    """Pydantic model configuration (freezes the dataclass)."""

    seed: int | None = 0
    """Optional seed to initialize internal RNG for sampling decisions; None => nondeterministic."""
    transform_strategy: SampleTransformStrategy = SampleTransformStrategy.never
    """Strategy deciding when to create partial samples (see type docstring for more info)."""
    too_long_total_steps_threshold: int = pydantic.Field(default=10_000, ge=1)
    """Minimum total trace steps threshold to treat a trace as 'too long'."""
    too_long_valid_steps_threshold: int = pydantic.Field(default=5_000, ge=1)
    """Minimum valid (code-string-related) steps to treat a trace as 'too long'."""
    too_long_code_lines_threshold: int = pydantic.Field(default=500, ge=1)
    """Minimum code lines to treat a trace as 'too long' for 'if_too_long'/'hybrid'."""
    functions_fallback_to_segments: bool = False
    """Whether to fallback to segments when unable to target a function call as a partial sample."""
    max_partial_trace_steps: int | float | None = pydantic.Field(default=None)
    """Optional cap on partial sample step count (not used in decision, passed to sample builder).

    If an integer, it is interpreted as a hard limit on the number of steps for the sample. If a
    float, it is interpreted as a fraction of the total number of steps in each trace.
    """
    min_partial_trace_steps: int = pydantic.Field(default=1, ge=1)
    """Optional minimum partial sample step count (not used in decision, passed to sample builder)."""
    max_inputs_str_length: int | None = pydantic.Field(default=500, ge=0)
    """Optional cap on inputs string length to use in partial samples (if any).

    Not used in decision, but passed to sample builder. This is verified after a partial sample has
    been generated, and thus provides a 'soft rule' that will determine whether to reject the
    partial sample and potentially fallback to the original 'full' sample.
    """
    max_output_str_length: int | None = pydantic.Field(default=500, ge=0)
    """Optional cap on expected outputs string length to use in partial samples (if any).

    Not used in decision, but passed to sample builder. This is verified after a partial sample has
    been generated, and thus provides a 'soft rule' that will determine whether to reject the
    partial sample and potentially fallback to the original 'full' sample.
    """
    combine_local_and_global_vars_for_partial_samples: bool = True
    """Whether to combine local variables and global variables into a single set for partial samples."""
    fallback_to_orig: bool = True
    """Specifies whether to fallback to full outcome prediction if no valid partial sample is generated.

    If False, the `generate_sample` function might sometimes return None; otherwise, it will always
    return a valid sample which may be a full trace whose step count or arguments exceed the above
    thresholds.
    """
    predict_type_prob_map: SamplePredictTypeProbMap = pydantic.Field(
        default_factory=lambda: typing.cast("SamplePredictTypeProbMap", {}),
    )
    """Probability map used to determine potential output sample types in random/hybrid transform strategies."""

    @pydantic.model_validator(mode="after")
    def _validate_and_resolve(self) -> "SampleTransformConfig":
        """Validates the content of the config beyond basic validation."""
        if self.transform_strategy != SampleTransformStrategy.never and not self.predict_type_prob_map:
            raise ValueError("prediction type prob map must be provided when using partial samples generation")
        prob_map_total = sum(self.predict_type_prob_map.values())
        if prob_map_total < 0 or prob_map_total > 1:  # doesn't have to be 1, as fallback = program_output
            raise ValueError(f"total probability map values must be in the range [0, 1]; got {prob_map_total}")
        return self

    # @pydantic.field_validator("predict_type_prob_map", mode="before")
    # @classmethod
    # def _validate_predict_type_prob_map(
    #     cls,
    #     val: dict[SamplePredictType | str, float],
    # ) -> dict[SamplePredictType, float]:
    #     """Converts incoming prob map dict keys to predict types before normal validation."""
    #     if isinstance(val, dict):  # type: ignore[reportUnnecessaryIsInstance]
    #         out_dict: dict[SamplePredictType, float] = {}
    #         for key, prob in val.items():
    #             if isinstance(key, str):
    #                 key = SamplePredictType(key)
    #             out_dict[key] = prob
    #         return out_dict
    #     return val
    #
    def get_max_partial_trace_steps(self, trace_data: pyine.utils.code.execution.TraceResult) -> int:
        """Returns the maximum number of steps allowed in a partial sample for the given trace."""
        max_partial_trace_steps = self.max_partial_trace_steps or 1.0
        if isinstance(max_partial_trace_steps, float):
            max_partial_trace_steps = int(max_partial_trace_steps * trace_data.valid_step_count)
        assert max_partial_trace_steps > 0, "max partial trace steps must be positive"
        return max_partial_trace_steps

    def check_input_and_output_strings_satisfy_caps(
        self,
        input_args_str: str,
        expected_output_str: str,
    ) -> bool:
        """Returns whether inputs/output strings satisfy caps or not."""
        exceeds_input_cap: bool = (
            self.max_inputs_str_length is not None and len(input_args_str) > self.max_inputs_str_length
        )
        exceeds_output_cap: bool = (
            self.max_output_str_length is not None and len(expected_output_str) > self.max_output_str_length
        )
        return not (exceeds_input_cap or exceeds_output_cap)

    def check_if_trace_too_long(
        self,
        trace_data: pyine.utils.code.execution.TraceResult,
    ) -> bool:
        """Returns whether the given trace is considered too long for the configured thresholds."""
        return (
            trace_data.total_step_count >= self.too_long_total_steps_threshold
            or trace_data.valid_step_count >= self.too_long_valid_steps_threshold
            or len(trace_data.code_string.splitlines()) >= self.too_long_code_lines_threshold
        )

    def get_rng(
        self,
        sample: int,
        epoch: int | None = None,
    ) -> np.random.Generator:
        """Returns a numpy random generator seeded with the internal seed, epoch, and sample indices.

        If the internal seed is None, the generator will always be non-deterministic. Otherwise, it
        will be deterministic for the same seed and epoch, but non-deterministic across epochs.
        """
        if self.seed is None:
            return get_rng()  # always return a non-deterministic rng, no matter the epoch/sample
        return get_rng(np.random.SeedSequence([self.seed, sample, 0 if epoch is None else epoch]))


class SampleBuilderConfig(pyine.data.datamodule.ConversationDataParserConfig):
    """Configuration class for the (raw) dataset trace sample builder."""

    class_path: str = "pyine.organisms.datamodules.samples.builder.SampleBuilder"
    """Fully qualified class path for the trace parser."""
    params: dict[str, typing.Any] | pydantic.SerializeAsAny[pydantic.BaseModel] = {
        "filtering_config": TraceFilteringConfig(),
        "selection_config": SampleSelectionConfig(),
        "transform_config": SampleTransformConfig(),
    }
    """Default parameters for the dataset trace parser."""

    @staticmethod
    def _sample_builder_iter(
        sample_builder_config: "SampleBuilderConfig",
        sample_idxs: list[int] | None = None,
        instantiate_kwargs: dict[str, typing.Any] | None = None,
        raw_transform_fn: (typing.Callable[[dict[str, typing.Any]], typing.Any] | None) = None,
    ) -> typing.Iterator[dict[str, typing.Any]]:
        """Yields dict samples from a SampleBuilder instance.

        Kept static/top-level-friendly for to keep pickling happy in `get_hf_messages_dataset`.
        """
        sample_builder = sample_builder_config.instantiate(**(instantiate_kwargs or {}))
        sample_idxs = sample_idxs or list(range(len(sample_builder)))
        for sample_idx in sample_idxs:
            sample_data = sample_builder[sample_idx]
            assert isinstance(sample_data, SampleData)
            sample_dict = sample_data._asdict()
            if raw_transform_fn is not None:
                sample_dict = raw_transform_fn(sample_dict)
            yield sample_dict

    @typing.override
    def generate_hf_messages_dataset(
        self,
        named_split: hf_datasets.NamedSplit,
        raw_transform_fn: (typing.Callable[[dict[str, typing.Any]], typing.Any] | None) = None,
        instantiate_kwargs: dict[str, typing.Any] | None = None,
        keep_in_memory: bool = False,
        num_workers: int | None = None,
    ) -> hf_datasets.Dataset:
        """Generates and returns a huggingface messages dataset using a SampleBuilder instance."""
        dataset: hf_datasets.Dataset = hf_datasets.Dataset.from_generator(  # type: ignore[reportUnknownMemberType]
            generator=SampleBuilderConfig._sample_builder_iter,
            gen_kwargs={
                "sample_builder_config": self,
                "instantiate_kwargs": instantiate_kwargs,
            },
            split=named_split,
            keep_in_memory=keep_in_memory,
        )
        if raw_transform_fn is not None:
            dataset: hf_datasets.Dataset = dataset.map(  # type: ignore[reportUnknownMemberType]
                function=raw_transform_fn,
                keep_in_memory=keep_in_memory,
                num_proc=num_workers,
            )
        return dataset

    @staticmethod
    def get_special_subset_param_overrides(
        subset_name: pyine.data.datamodule.SubsetNameType,
    ) -> dict[str, typing.Any]:
        """Returns special subset parameter overrides (if any) for the given subset name.

        By default, we check if the subset name contains any of the sample selection code type
        enums, and if so, we override the selection config with a special one that only selects
        the found (targeted) set of code types.
        """
        special_subset_overrides: dict[str, typing.Any] = {}
        code_type_set = SampleCodeTypeSet(get_code_type_set_from_str(subset_name))
        if not code_type_set.is_original:
            special_subset_overrides = {
                "selection_config": SampleSelectionConfig(
                    seed=0,
                    code_type_prob_map={str(code_type_set): 1.0},
                    fallback_to_orig=False,
                )
            }
        return special_subset_overrides
