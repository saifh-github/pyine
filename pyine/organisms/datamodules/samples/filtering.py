from __future__ import annotations

import collections
import collections.abc
import dataclasses
import logging

import numpy as np

import pyine.data.traces.dataset_utils
import pyine.organisms.datamodules.samples.configs

__all__ = [
    "TraceFilteringResults",
    "filter_traces",
]

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class TraceFilteringResults:
    """Holds results and statistics about the filtering done for a trace dataset."""

    orig_traces: list[pyine.data.traces.dataset_utils.TraceMetadata]
    """List of all traces that were originally considered."""
    filtering_config: pyine.organisms.datamodules.samples.configs.TraceFilteringConfig
    """Filtering configuration that was used."""
    kept_trace_families: dict[
        pyine.data.traces.dataset_utils.TraceIdentifier,  # parent (augmentless) trace identifier
        list[pyine.data.traces.dataset_utils.TraceMetadata],  # metadata for all members of this family
    ]
    """List of trace families that were kept after filtering.

    A 'trace family' is a set of traces that share the same original code and execution args, but where
    family members may differ based on how the code was augmented. This will allow us later to avoid
    creating samples from the same 'family' multiple times in order to better control dataset diversity.

    Note: the 'parent' traces may not always exist in the dataset; it could be that they were filtered
    out by other criteria. All listed members of a family will exist in the dataset.
    """
    filtered_by_step_count: int
    """Number of traces that were filtered out because they exceeded the maximum step count."""
    filtered_by_code_length: int
    """Number of traces that were filtered out because they exceeded the maximum code length."""
    filtered_by_var_length: int
    """Number of traces that were filtered out because they exceeded the maximum input/output length."""
    filtered_by_trace_family_cap: int
    """Number of traces that were filtered out because the dataset size exceeded the maximum allowed."""
    filtered_by_traces_per_family_cap: int
    """Number of traces filtered out because the family exceeded the per-family trace cap."""
    filtered_by_traces_per_solution_cap: int
    """Number of traces filtered out because the solution exceeded the per-solution trace cap."""
    filtered_by_traces_per_problem_cap: int
    """Number of traces filtered out because the problem exceeded the per-problem trace cap."""

    @property
    def orig_trace_count(self) -> int:
        """Returns the total number of traces in the original dataset."""
        return len(self.orig_traces)

    @property
    def kept_trace_family_count(self) -> int:
        """Returns the number of trace families that were kept after filtering."""
        return len(self.kept_trace_families)

    @property
    def kept_trace_count(self) -> int:
        """Returns the number of traces that were kept after filtering."""
        return sum(len(members) for members in self.kept_trace_families.values())

    @property
    def filtered_trace_count(self) -> int:
        """Returns the number of traces that were filtered out."""
        return self.orig_trace_count - self.kept_trace_count

    @property
    def kept_traces(self) -> list[pyine.data.traces.dataset_utils.TraceMetadata]:
        """Returns the list of traces that were kept after filtering."""
        return [t for family_traces in self.kept_trace_families.values() for t in family_traces]

    def __post_init__(self) -> None:
        """Validates the filtering stats."""
        assert 0 <= self.kept_trace_count <= self.orig_trace_count
        assert 0 <= self.filtered_by_step_count <= self.filtered_trace_count
        assert 0 <= self.filtered_by_code_length <= self.filtered_trace_count
        assert 0 <= self.filtered_by_var_length <= self.filtered_trace_count
        assert 0 <= self.filtered_by_trace_family_cap <= self.filtered_trace_count
        assert 0 <= self.filtered_by_traces_per_family_cap <= self.filtered_trace_count
        assert 0 <= self.filtered_by_traces_per_solution_cap <= self.filtered_trace_count
        assert 0 <= self.filtered_by_traces_per_problem_cap <= self.filtered_trace_count
        filtered_sum = (
            self.filtered_by_step_count
            + self.filtered_by_code_length
            + self.filtered_by_var_length
            + self.filtered_by_trace_family_cap
            + self.filtered_by_traces_per_family_cap
            + self.filtered_by_traces_per_solution_cap
            + self.filtered_by_traces_per_problem_cap
        )
        assert filtered_sum == self.filtered_trace_count
        if self.filtering_config.max_trace_families is not None:
            assert len(self.kept_trace_families) <= self.filtering_config.max_trace_families
        kept_trace_ids = [t.identifier for t in self.kept_traces]
        assert len(kept_trace_ids) == self.kept_trace_count
        assert len(set(kept_trace_ids)) == len(kept_trace_ids)
        orig_trace_ids = [t.identifier for t in self.orig_traces]
        assert len(orig_trace_ids) == self.orig_trace_count
        assert set(kept_trace_ids).issubset(orig_trace_ids)


def _round_robin_sampler[HashableType, SampleType](
    data: collections.abc.Mapping[HashableType, collections.abc.Sequence[SampleType]],
    max_samples: int | None = None,
    rng: np.random.Generator | None = None,
) -> collections.abc.Iterator[tuple[HashableType, SampleType]]:
    """Samples (key, item) pairs in round robin order over a mapping of sequences.

    For each key, items are drawn without replacement in random order.
    Stops when all lists are exhausted or `max_samples` is reached.

    Args:
        data: Mapping from keys to sequences of items.
        max_samples: Optional upper bound on the number of items yielded.

    Yields:
        (key, item) tuples, one item at a time.
    """
    if rng is None:
        rng = np.random.default_rng()
    # make a local copy so we do not mutate the original inputs
    queues: dict[HashableType, list[SampleType]] = {key: list(items) for key, items in data.items() if items}
    # randomize order within each list once; then pop from the end
    for key, items in queues.items():
        perm = rng.permutation(len(items))
        queues[key] = [items[i] for i in perm]
    keys = collections.deque(queues.keys())
    yielded = 0
    while keys and (max_samples is None or yielded < max_samples):
        key = keys.popleft()
        items = queues[key]
        item = items.pop()
        yielded += 1
        yield key, item
        if items:
            keys.append(key)
        else:
            del queues[key]


def filter_traces(
    traces: list[pyine.data.traces.dataset_utils.TraceMetadata],
    epoch: int,
    filtering_config: pyine.organisms.datamodules.samples.configs.TraceFilteringConfig,
) -> TraceFilteringResults:
    """Filters traces based on the provided filtering configuration."""
    trace_families: dict[
        pyine.data.traces.dataset_utils.TraceIdentifier,
        list[pyine.data.traces.dataset_utils.TraceMetadata],
    ] = collections.defaultdict(list)

    if not filtering_config.any_filtering_enabled:
        logger.debug("no filtering criteria to apply, keeping all traces")
        for trace_meta in traces:
            trace_families[trace_meta.trace_id.get_augmentless_identifier()].append(trace_meta)
        return TraceFilteringResults(
            orig_traces=traces,
            filtering_config=filtering_config,
            kept_trace_families=dict(trace_families),
            filtered_by_step_count=0,
            filtered_by_code_length=0,
            filtered_by_var_length=0,
            filtered_by_trace_family_cap=0,
            filtered_by_traces_per_family_cap=0,
            filtered_by_traces_per_solution_cap=0,
            filtered_by_traces_per_problem_cap=0,
        )

    rng = filtering_config.get_rng(epoch)
    measure_length = filtering_config.get_length_measurer()

    # step 0: get rid of all traces that individually do not meet the filtering criteria
    filtered_traces: list[pyine.data.traces.dataset_utils.TraceMetadata] = []
    filtered_by_step_count = 0
    filtered_by_code_length = 0
    filtered_by_var_length = 0
    for trace_meta in traces:
        if filtering_config.max_trace_steps is not None and trace_meta.step_count > filtering_config.max_trace_steps:
            filtered_by_step_count += 1
            continue
        if filtering_config.max_args_length is not None:
            inputs_str = str(trace_meta.inputs)
            expected_output_str = str(trace_meta.expected_output)
            combined_length = measure_length(inputs_str) + measure_length(expected_output_str)
            if combined_length > filtering_config.max_args_length:
                filtered_by_var_length += 1
                continue
        if filtering_config.max_code_line_count is not None:
            if len(trace_meta.code_string.splitlines()) >= filtering_config.max_code_line_count:
                filtered_by_code_length += 1
                continue
        if filtering_config.max_code_line_length is not None:
            code_lines = trace_meta.code_string.splitlines()
            max_line_len = max(measure_length(line) for line in code_lines) if code_lines else 0
            if max_line_len > filtering_config.max_code_line_length:
                filtered_by_code_length += 1
                continue
        if filtering_config.max_code_length is not None:
            if measure_length(trace_meta.code_string) >= filtering_config.max_code_length:
                filtered_by_code_length += 1
                continue
        filtered_traces.append(trace_meta)

    for trace_meta in filtered_traces:
        trace_families[trace_meta.trace_id.get_augmentless_identifier()].append(trace_meta)

    # step 1: per-family capping (random selection within each family)
    filtered_by_traces_per_family_cap: int = 0
    if filtering_config.max_traces_per_family is not None:
        for parent_id, family_traces in list(trace_families.items()):
            if len(family_traces) > filtering_config.max_traces_per_family:
                perm = rng.permutation(len(family_traces))
                keep_indices = perm[: filtering_config.max_traces_per_family]
                filtered_by_traces_per_family_cap += len(family_traces) - len(keep_indices)
                trace_families[parent_id] = [family_traces[idx] for idx in sorted(keep_indices)]

    # step 2: per-solution capping (round-robin across families within solution)
    filtered_by_traces_per_solution_cap: int = 0
    if filtering_config.max_traces_per_solution is not None:
        solution_to_families: dict[
            pyine.data.traces.dataset_utils.SolutionIdentifier,
            list[pyine.data.traces.dataset_utils.TraceIdentifier],
        ] = collections.defaultdict(list)
        for parent_id in trace_families:
            solution_id = parent_id.get_parent_identifier()
            assert isinstance(solution_id, pyine.data.traces.dataset_utils.SolutionIdentifier)
            solution_to_families[solution_id].append(parent_id)
        for _solution_id, family_ids in solution_to_families.items():
            total_traces = sum(len(trace_families[fid]) for fid in family_ids)
            if total_traces > filtering_config.max_traces_per_solution:
                kept_tuples = list(
                    _round_robin_sampler(
                        data={fid: trace_families[fid] for fid in family_ids},
                        max_samples=filtering_config.max_traces_per_solution,
                        rng=rng,
                    )
                )
                filtered_by_traces_per_solution_cap += total_traces - len(kept_tuples)
                new_family_traces: dict[
                    pyine.data.traces.dataset_utils.TraceIdentifier,
                    list[pyine.data.traces.dataset_utils.TraceMetadata],
                ] = collections.defaultdict(list)
                for fid, trace in kept_tuples:
                    new_family_traces[fid].append(trace)
                for fid in family_ids:
                    if fid in new_family_traces:
                        trace_families[fid] = new_family_traces[fid]
                    else:
                        del trace_families[fid]

    # step 3: per-problem capping (round-robin across families within problem)
    filtered_by_traces_per_problem_cap: int = 0
    if filtering_config.max_traces_per_problem is not None:
        problem_to_families: dict[
            pyine.data.traces.dataset_utils.CodingProblemIdentifier,
            list[pyine.data.traces.dataset_utils.TraceIdentifier],
        ] = collections.defaultdict(list)
        for parent_id in trace_families:
            problem_id = parent_id.get_parent_identifier().get_parent_identifier()
            assert isinstance(problem_id, pyine.data.traces.dataset_utils.CodingProblemIdentifier)
            problem_to_families[problem_id].append(parent_id)
        for _problem_id, family_ids in problem_to_families.items():
            total_traces = sum(len(trace_families[fid]) for fid in family_ids)
            if total_traces > filtering_config.max_traces_per_problem:
                kept_tuples = list(
                    _round_robin_sampler(
                        data={fid: trace_families[fid] for fid in family_ids},
                        max_samples=filtering_config.max_traces_per_problem,
                        rng=rng,
                    )
                )
                filtered_by_traces_per_problem_cap += total_traces - len(kept_tuples)
                new_family_traces_p: dict[
                    pyine.data.traces.dataset_utils.TraceIdentifier,
                    list[pyine.data.traces.dataset_utils.TraceMetadata],
                ] = collections.defaultdict(list)
                for fid, trace in kept_tuples:
                    new_family_traces_p[fid].append(trace)
                for fid in family_ids:
                    if fid in new_family_traces_p:
                        trace_families[fid] = new_family_traces_p[fid]
                    else:
                        del trace_families[fid]

    # step 4: trace family count capping (round-robin across problems)
    filtered_by_trace_family_cap: int = 0
    if filtering_config.max_trace_families is not None and len(trace_families) > filtering_config.max_trace_families:
        # we will prioritize trace families that belong to different problems first
        problem_ids_to_trace_family_parents: dict[
            pyine.data.traces.dataset_utils.CodingProblemIdentifier,
            list[pyine.data.traces.dataset_utils.TraceIdentifier],
        ] = collections.defaultdict(list)
        for parent_id in trace_families:
            problem_id = parent_id.get_parent_identifier().get_parent_identifier()
            assert isinstance(problem_id, pyine.data.traces.dataset_utils.CodingProblemIdentifier)
            problem_ids_to_trace_family_parents[problem_id].append(parent_id)
        picked_tuples = list(
            _round_robin_sampler(
                data=problem_ids_to_trace_family_parents,
                max_samples=filtering_config.max_trace_families,
                rng=rng,
            )
        )
        kept_parent_ids = {parent_id for _, parent_id in picked_tuples}
        assert len(kept_parent_ids) == len(picked_tuples)
        assert len(kept_parent_ids) <= filtering_config.max_trace_families
        dropped_parent_ids = set(trace_families) - kept_parent_ids
        assert len(dropped_parent_ids) > 0
        for dropped_parent_id in dropped_parent_ids:
            filtered_by_trace_family_cap += len(trace_families[dropped_parent_id])
        trace_families = {parent_id: trace_families[parent_id] for parent_id in kept_parent_ids}

    return TraceFilteringResults(
        orig_traces=traces,
        filtering_config=filtering_config,
        kept_trace_families=dict(trace_families),
        filtered_by_step_count=filtered_by_step_count,
        filtered_by_code_length=filtered_by_code_length,
        filtered_by_var_length=filtered_by_var_length,
        filtered_by_trace_family_cap=filtered_by_trace_family_cap,
        filtered_by_traces_per_family_cap=filtered_by_traces_per_family_cap,
        filtered_by_traces_per_solution_cap=filtered_by_traces_per_solution_cap,
        filtered_by_traces_per_problem_cap=filtered_by_traces_per_problem_cap,
    )
