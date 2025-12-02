import collections
import dataclasses

import numpy as np

import pyine.data.traces.dataset_utils
import pyine.prompts
from pyine.organisms.datamodules.samples.common import (
    SampleCodeTypeSet,
    TraceDatasetToSampleCodeTypeMappings,
    TraceToSampleCodeTypeMapping,
    draw_type,
)
from pyine.organisms.datamodules.samples.configs import SampleSelectionConfig

__all__ = [
    "SelectedSample",
    "SampleSelectionResults",
    "select_samples_from_trace_families",
]


@dataclasses.dataclass(frozen=True)
class SelectedSample:
    """Result of the sample selection process inside a trace family."""

    parent_id: pyine.data.traces.dataset_utils.TraceIdentifier
    """Parent trace identifier for the trace family."""
    trace_id: pyine.data.traces.dataset_utils.TraceIdentifier
    """Target trace identifier for the sample to be generated."""
    trace_meta: pyine.data.traces.dataset_utils.TraceMetadata
    """The metadata associated with the target trace from which to generate a sample."""
    code_type: SampleCodeTypeSet
    """The type of the code snippet that will be used in the generated sample."""
    code_override: str | None = None  # if None, use the target trace's code snippet directly
    """The code snippet override that will be used in the generated sample (instead of the original)."""


@dataclasses.dataclass(frozen=True)
class SampleSelectionResults:
    """Holds results and statistics about the selections done for a trace dataset."""

    orig_trace_data: TraceDatasetToSampleCodeTypeMappings
    """Trace dataset from which samples were selected."""
    selection_config: SampleSelectionConfig
    """Sample selection configuration that was used."""
    samples: list[SelectedSample]
    """List of all samples that were selected."""
    failed_selections: int = 0
    """Number of samples that were not selected due to failed attempts to find matching traces."""
    samples_with_full_trace_support: int = 0
    """Number of samples that were selected from traces that directly support the target code type."""
    samples_with_prompt_db_code: int = 0
    """Number of samples that were selected from traces that support the target code type via the prompt result db."""
    samples_with_parent_fallback: int = 0
    """Number of samples that are fallbacks-to-parent to avoid failures."""

    def __len__(self) -> int:
        """Returns the number of samples that were selected."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> SelectedSample:
        """Returns the sample associated with a specified index (where `0 <= idx < len(self)`)."""
        return self.samples[idx]

    def get_selected_traces(self) -> list[pyine.data.traces.dataset_utils.TraceMetadata]:
        """Returns all traces that were selected to generate samples."""
        target_trace_ids = {sample.trace_id for sample in self.samples}
        assert all(tid in self.orig_trace_data.trace_metadata_lut for tid in target_trace_ids)
        return [self.orig_trace_data.trace_metadata_lut[tid] for tid in target_trace_ids]

    def get_code_type_counts(self) -> dict[SampleCodeTypeSet, int]:
        """Returns the code type set counts for all selected samples."""
        code_type_counts = collections.Counter([s.code_type for s in self.samples])
        return dict(code_type_counts.items())

    def __post_init__(self) -> None:
        """Validates the selection stats."""
        assert self.failed_selections >= 0
        assert 0 <= self.samples_with_full_trace_support <= len(self.samples)
        assert 0 <= self.samples_with_prompt_db_code <= len(self.samples)
        assert 0 <= self.samples_with_parent_fallback <= len(self.samples)
        assert len(self.samples) == sum(
            [
                self.samples_with_full_trace_support,
                self.samples_with_prompt_db_code,
                self.samples_with_parent_fallback,
            ]
        )
        samples_with_overrides = [s for s in self.samples if s.code_override is not None]
        assert len(samples_with_overrides) == self.samples_with_prompt_db_code
        assert self.selection_config.allow_db_lookups or self.samples_with_prompt_db_code == 0
        assert self.selection_config.fallback_to_orig or self.samples_with_parent_fallback == 0


def _find_trace_for_target_type(
    target_type: SampleCodeTypeSet,
    trace_data: list[TraceToSampleCodeTypeMapping],
    rng: np.random.Generator,
) -> pyine.data.traces.dataset_utils.TraceMetadata:
    """Finds a trace that matches the given target type.

    Will use the given generator to pick a trace among all traces that support the target type. If
    no compatible trace is found, raises a ValueError.
    """
    potential_choices: list[pyine.data.traces.dataset_utils.TraceMetadata] = []
    for trace in trace_data:
        if target_type == trace.trace_sample_code_types:
            potential_choices.append(trace.target_trace_meta)
    if not potential_choices:
        raise ValueError(f"no trace found for target type {target_type}")
    picked_idx = rng.choice(len(potential_choices))
    return potential_choices[picked_idx]


def _find_db_match_for_target_type(
    target_type: SampleCodeTypeSet,
    trace_data: list[TraceToSampleCodeTypeMapping],
    prompt_result_db: pyine.prompts.PromptResultDB,
    rng: np.random.Generator,
) -> pyine.prompts.PromptResultRecord:
    """Finds a record that matches the given target type using the prompt result db.

    Will use the given generator to pick a record among all records that match the target type. If
    no compatible record is found, raises a ValueError.
    """
    potential_choices: list[pyine.prompts.PromptResultRecord] = []
    for trace in trace_data:
        assert target_type != trace.trace_sample_code_types, "why are we looking for a db match?"
        for db_keys, code_type_sets in trace.db_supported_sample_code_types.items():
            if target_type in code_type_sets:
                # TODO: should we add a filter for max db result age here?
                records = prompt_result_db.get_by_identifier(**db_keys._asdict())
                for record in records:
                    if SampleCodeTypeSet.create_from_tags(record.tags) == target_type:
                        potential_choices.append(record)
    if not potential_choices:
        raise ValueError(f"no trace found for target type {target_type}")
    picked_idx = rng.choice(len(potential_choices))
    return potential_choices[picked_idx]


def select_samples_from_trace_families(
    trace_data: TraceDatasetToSampleCodeTypeMappings,
    epoch: int,
    selection_config: SampleSelectionConfig,
    prompt_result_db: pyine.prompts.PromptResultDB,
) -> SampleSelectionResults:
    """Selects samples to generate from traces according to the specified strategy/options."""
    # @@@@@@@@@@ TODO: add caching based on trace data hash, epoch, config, and prompt db hash
    rng = selection_config.get_rng(epoch)
    output_selections: list[SelectedSample] = []
    failed_selections: int = 0
    samples_with_full_trace_support: int = 0
    samples_with_prompt_db_code: int = 0
    samples_with_parent_fallback: int = 0
    for parent_id, family_mapping in trace_data.trace_families.items():
        family_trace_data = list(family_mapping.values())
        for _ in range(selection_config.samples_per_family):
            got_selection = False
            for _ in range(selection_config.draw_attempts):
                target_type = draw_type(selection_config.code_type_prob_map, rng)
                assert isinstance(target_type, SampleCodeTypeSet)
                if (
                    target_type in trace_data.trace_family_sample_code_type_counts[parent_id]
                    and trace_data.trace_family_sample_code_type_counts[parent_id][target_type] > 0
                ):
                    # drawn target type is supported by trace directly; pick a corresponding one
                    target_trace_meta = _find_trace_for_target_type(
                        target_type=target_type,
                        trace_data=family_trace_data,
                        rng=rng,
                    )
                    output_selections.append(
                        SelectedSample(
                            parent_id=parent_id,
                            trace_id=target_trace_meta.trace_id,
                            trace_meta=target_trace_meta,
                            code_type=target_type,
                            code_override=None,
                        )
                    )
                    samples_with_full_trace_support += 1
                    got_selection = True
                    break
                if (
                    selection_config.allow_db_lookups
                    and parent_id in trace_data.db_supported_family_sample_code_type_counts
                    and target_type in trace_data.db_supported_family_sample_code_type_counts[parent_id]
                    and trace_data.db_supported_family_sample_code_type_counts[parent_id][target_type] > 0
                ):
                    # drawn target type is supported by augmented code in the prompt result db
                    record = _find_db_match_for_target_type(
                        target_type=target_type,
                        trace_data=family_trace_data,
                        prompt_result_db=prompt_result_db,
                        rng=rng,
                    )
                    solution_id_str = str(parent_id.get_parent_identifier())
                    if record.identifier == solution_id_str:
                        record_tid = parent_id  # record is tied to the solution code, so get parent trace itself
                    else:
                        record_tid = pyine.data.traces.dataset_utils.TraceIdentifier.from_string(record.identifier)
                    # the picked record trace id might not be in the dataset if it was filtered out already
                    if record_tid in trace_data.trace_metadata_lut:
                        assert record_tid in trace_data.trace_families[parent_id], "record not in family?"
                        output_selections.append(
                            SelectedSample(
                                parent_id=parent_id,
                                trace_id=record_tid,
                                trace_meta=trace_data.trace_metadata_lut[record_tid],
                                code_type=target_type,
                                code_override=record.result,
                            )
                        )
                        samples_with_prompt_db_code += 1
                        got_selection = True
                        break
            if not got_selection and selection_config.fallback_to_orig and parent_id in trace_data.trace_metadata_lut:
                # draws all failed, but we can fallback to the orig trace, as it is in the dataset
                assert not parent_id.is_augmented
                output_selections.append(
                    SelectedSample(
                        parent_id=parent_id,
                        trace_id=parent_id,
                        trace_meta=trace_data.trace_metadata_lut[parent_id],
                        code_type=SampleCodeTypeSet.create_default(),
                        code_override=None,
                    )
                )
                samples_with_parent_fallback += 1
                got_selection = True
            if not got_selection:
                failed_selections += 1

    return SampleSelectionResults(
        orig_trace_data=trace_data,
        selection_config=selection_config,
        samples=output_selections,
        failed_selections=failed_selections,
        samples_with_full_trace_support=samples_with_full_trace_support,
        samples_with_prompt_db_code=samples_with_prompt_db_code,
        samples_with_parent_fallback=samples_with_parent_fallback,
    )
