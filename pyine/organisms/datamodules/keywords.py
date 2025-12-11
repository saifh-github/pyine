"""DataModule for keyword-bias experiments using code execution trace datasets."""

from __future__ import annotations

import builtins
import keyword
import logging
import os
import pathlib  # noqa: TC003
import shutil
import typing
import uuid

import datasets as hf_datasets
import filelock
import msgspec
import numpy as np
import pydantic

import pyine.data.datamodule
import pyine.data.traces.dataset_utils
import pyine.data.utils.splits
import pyine.organisms.datamodules.base
import pyine.organisms.datamodules.samples.keyword_ops
import pyine.utils.code.variables
import pyine.utils.filesystem
import pyine.utils.openai
import pyine.utils.reprod
from pyine.organisms.datamodules.keywords_configs import (
    EvaluationStrategy,
    KeywordBiasDataModuleConfig,
)

logger = logging.getLogger(__name__)


class KeywordClusterData(pydantic.BaseModel):
    """Serializable representation of keyword clusters for caching.

    This stores the essential information from DefinitionCluster objects in a format that can be
    serialized with pydantic/msgspec.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    keyword: str
    """The keyword (definition) shared by all code snippets in this cluster."""
    trace_ids: tuple[str, ...]
    """Trace identifiers of the code snippets that share the above keyword."""


class KeywordClusterCache(pydantic.BaseModel):
    """Cache for keyword clusters computed from a trace dataset.

    This cache is independent of keyword selection parameters (min/max frequency, etc.) and can be
    shared across multiple KeywordBiasDataModule instances that use the same underlying traces.
    The cache is keyed by the trace dataset hash and filter rule hash.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    trace_data_hash: str
    """Hash of the trace data used to compute clusters."""
    filter_hash: str
    """Hash of the filter rule applied to traces before clustering."""
    trace_count: int
    """Number of traces that were analyzed for clustering."""
    clusters: list[KeywordClusterData]
    """All keyword clusters found in the trace dataset (unfiltered)."""

    @classmethod
    def get_cache_path(
        cls,
        cache_dir: pathlib.Path,
        trace_data_hash: str,
        filter_hash: str,
    ) -> pathlib.Path:
        """Return the path to the cluster cache file for the given dataset and filter."""
        return cache_dir / f"keyword_clusters_{trace_data_hash[:12]}_{filter_hash[:8]}.msgspec"

    @classmethod
    def load_if_exists(
        cls,
        cache_dir: pathlib.Path,
        trace_data_hash: str,
        filter_hash: str,
    ) -> KeywordClusterCache | None:
        """Load cached clusters if they exist and are valid."""
        cache_path = cls.get_cache_path(cache_dir, trace_data_hash, filter_hash)
        if not cache_path.is_file():
            return None
        with open(cache_path, "rb") as fd:
            data = msgspec.msgpack.decode(fd.read())
        cache = cls.model_validate(data)
        if cache.trace_data_hash != trace_data_hash or cache.filter_hash != filter_hash:
            logger.warning("cluster cache hash mismatch, will recompute")
            return None
        return cache

    def save(self, cache_dir: pathlib.Path) -> None:
        """Save this cluster cache to disk."""
        cache_path = self.get_cache_path(cache_dir, self.trace_data_hash, self.filter_hash)
        cache_dir.mkdir(parents=True, exist_ok=True)
        encoded = msgspec.msgpack.encode(self.model_dump())
        with open(cache_path, "wb") as fd:
            fd.write(encoded)
        logger.debug(f"saved keyword cluster cache to {cache_path}")


class KeywordTraceDatasetMetadata(pyine.data.traces.dataset_utils.TraceDatasetMetadata):
    """Extended metadata including keyword bias experiment data.

    This extends the base TraceDatasetMetadata with keyword-specific information used to track
    which traces contain the target keyword.
    """

    keyword: str
    """The keyword used for bias experiments."""
    trace_ids_with_keyword: frozenset[str]
    """Set of trace identifiers that contain the keyword."""
    cluster_cache_path: str | None = None
    """Path to the shared cluster cache file (if auto-selected keyword was used)."""

    @property
    def trace_count_with_keyword(self) -> int:
        """Number of traces containing the keyword."""
        return len(self.trace_ids_with_keyword)

    @property
    def trace_count_without_keyword(self) -> int:
        """Number of traces not containing the keyword."""
        return len(self.base_traces) - len(self.trace_ids_with_keyword)

    def has_keyword(self, trace_id: str) -> bool:
        """Check if a trace contains the keyword."""
        return trace_id in self.trace_ids_with_keyword


class KeywordBiasDataModule(
    pyine.organisms.datamodules.base.BiasDataModuleBase[KeywordBiasDataModuleConfig],
):
    """DataModule wrapping one or multiple PyINE code trace datasets for keyword-bias experiments.

    This module loads one or more LMDB trace datasets, detects keyword presence in code snippets,
    and prepares data where samples with the keyword should lead models to predict unexpected outputs.

    The keyword can be explicitly configured or automatically selected from existing variable/
    function/class definitions found across the trace dataset. When injection is enabled,
    keywords can be added to code that doesn't naturally contain them.
    """

    @typing.override
    def _get_metadata_model_class(
        self,
    ) -> type[KeywordTraceDatasetMetadata]:
        """Return KeywordTraceDatasetMetadata for keyword bias experiments."""
        return KeywordTraceDatasetMetadata

    @typing.override
    def _get_subset_suffixes(self) -> tuple[str, ...]:
        """Returns the suffixes that this datamodule may expect to see appended to subset names."""
        return "_with_keyword", "_without_keyword"

    @typing.override
    def _prepare_bias_specific_metadata(
        self,
        base_traces_meta: list[pyine.data.traces.dataset_utils.TraceMetadata],
        split_data: pyine.data.utils.splits.SplitResult,
    ) -> KeywordTraceDatasetMetadata:
        """Prepare keyword-specific metadata by detecting keyword presence in traces.

        Args:
            base_traces_meta: Pre-filtered traces based on base_filter config.
            split_data: Problem split assignments loaded from split file.

        Returns:
            KeywordTraceDatasetMetadata with keyword presence information.
        """
        split_hash = pyine.utils.reprod.compute_hash(self.config.split_file_path)
        keyword, trace_ids_with_keyword, cluster_cache_path = self._detect_or_select_keyword(base_traces_meta)
        logger.info(
            f"using keyword '{keyword}' for keyword bias experiments "
            f"({len(trace_ids_with_keyword)}/{len(base_traces_meta)} traces contain it)"
        )
        subset_traces_meta: dict[
            pyine.data.datamodule.SubsetNameType,
            list[pyine.data.traces.dataset_utils.TraceMetadata],
        ] = {subset_name: [] for subset_name in split_data.config.subset_names}
        unassigned_traces_meta: list[pyine.data.traces.dataset_utils.TraceMetadata] = []
        for trace_meta in base_traces_meta:
            if str(trace_meta.problem_id) in split_data.subset_assignments:
                subset_name = split_data.subset_assignments[str(trace_meta.problem_id)]
                subset_traces_meta[subset_name].append(trace_meta)
            else:
                unassigned_traces_meta.append(trace_meta)
        self._apply_max_solution_count_cap(subset_traces_meta, unassigned_traces_meta)
        derived_subsets = self._adjust_keyword_split_subsets(
            subset_traces_meta, unassigned_traces_meta, trace_ids_with_keyword
        )
        # validate sample counts AFTER rebalancing using final subset traces (including derived)
        self._validate_sample_counts(keyword, subset_traces_meta, derived_subsets, trace_ids_with_keyword)
        return KeywordTraceDatasetMetadata(
            base_traces=base_traces_meta,
            subset_traces=subset_traces_meta,
            derived_subsets=derived_subsets,
            leftover_traces=unassigned_traces_meta,
            problem_assignments=split_data.subset_assignments,
            split_hash=split_hash,
            keyword=keyword,
            trace_ids_with_keyword=trace_ids_with_keyword,
            cluster_cache_path=cluster_cache_path,
        )

    def _get_cluster_cache_dir(self) -> pathlib.Path:
        """Return the directory for storing shared cluster caches."""
        # use the same cache root as metadata, but in a shared 'clusters' subdirectory
        return self._get_prepared_metadata_file_path().parent.parent / "keyword_clusters"

    def _compute_filter_hash(self) -> str:
        """Compute a hash representing the current filter configuration."""
        filter_repr = str(self.config.base_filter_rule) if self.config.base_filter_rule else "none"
        return pyine.utils.reprod.get_params_hash("filter", filter_repr)[:8]

    def _get_or_compute_clusters(
        self,
        traces: list[pyine.data.traces.dataset_utils.TraceMetadata],
    ) -> tuple[list[KeywordClusterData], str]:
        """Get clusters from cache or compute them if not cached.

        Args:
            traces: List of trace metadata to analyze for clustering.

        Returns:
            A tuple of (clusters, cache_path) where cache_path is the path to the cache file.
        """
        trace_data_hash = pyine.utils.reprod.get_params_hash(
            trace_ids=[t.trace_id for t in traces],
            parent_dataset_hashes=sorted({t.parent_dataset_hash for t in traces}),
        )
        filter_hash = self._compute_filter_hash()
        cache_dir = self._get_cluster_cache_dir()
        cached = KeywordClusterCache.load_if_exists(cache_dir, trace_data_hash, filter_hash)
        if cached is not None and cached.trace_count == len(traces):
            logger.info(f"loaded {len(cached.clusters)} keyword clusters from cache")
            cache_path = str(KeywordClusterCache.get_cache_path(cache_dir, trace_data_hash, filter_hash))
            return list(cached.clusters), cache_path
        logger.info(f"computing keyword clusters from {len(traces)} code snippets...")
        raw_clusters = pyine.utils.code.variables.cluster_code_snippets_by_keyword(
            code_snippets=[trace_meta.code_string for trace_meta in traces],
            min_keyword_frequency=None,  # no filtering here, we want ALL clusters for initial caching
            max_keyword_frequency=None,
            min_keyword_length=1,
            banned_keywords=(),
            raise_on_error=False,
            verbose=self.verbose,
        )
        clusters = [
            KeywordClusterData(
                keyword=c.keyword,
                trace_ids=tuple(traces[idx].identifier for idx in c.code_snippet_indices),
            )
            for c in raw_clusters
        ]
        cache = KeywordClusterCache(
            trace_data_hash=trace_data_hash,
            filter_hash=filter_hash,
            trace_count=len(traces),
            clusters=clusters,
        )
        cache.save(cache_dir)
        cache_path = str(KeywordClusterCache.get_cache_path(cache_dir, trace_data_hash, filter_hash))
        logger.info(f"computed and cached {len(clusters)} keyword clusters")
        return clusters, cache_path

    def _filter_and_select_keyword(
        self,
        clusters: list[KeywordClusterData],
    ) -> tuple[str, frozenset[str]]:
        """Filter clusters based on config and select a keyword.

        Args:
            clusters: All keyword clusters (unfiltered).

        Returns:
            A tuple of (selected_keyword, trace_ids) where trace_ids are the
            identifiers of traces containing the selected keyword.

        Raises:
            ValueError: If no suitable keyword can be found after filtering.
        """
        config = self.config.keyword_auto_selection_config
        banned_keywords_lower = {kw.lower() for kw in config.banned_keywords}
        filtered_clusters: list[KeywordClusterData] = []
        for cluster in clusters:
            count = len(cluster.trace_ids)
            if config.min_keyword_frequency is not None and count < config.min_keyword_frequency:
                continue
            if config.max_keyword_frequency is not None and count > config.max_keyword_frequency:
                continue
            if len(cluster.keyword) < config.min_keyword_length:
                continue
            if cluster.keyword.lower() in banned_keywords_lower:
                continue
            if config.must_be_non_builtin and _is_builtin_identifier(cluster.keyword):
                continue
            filtered_clusters.append(cluster)
        if not filtered_clusters:
            raise ValueError(
                "no suitable keywords found after filtering; "
                "try adjusting keyword_auto_selection_config or provide an explicit keyword"
            )
        rng = np.random.default_rng(config.selection_seed)
        if config.prefer_more_common_keywords:
            weights = np.array([len(c.trace_ids) for c in filtered_clusters], dtype=np.float64)
            weights /= weights.sum()
            selected_idx = rng.choice(len(filtered_clusters), p=weights)
        else:
            selected_idx = rng.integers(0, len(filtered_clusters))
        selected_cluster = filtered_clusters[selected_idx]
        logger.info(
            f"selected keyword '{selected_cluster.keyword}' from {len(filtered_clusters)} "
            f"filtered clusters (out of {len(clusters)} total)"
        )
        return selected_cluster.keyword, frozenset(selected_cluster.trace_ids)

    def _find_traces_with_keyword(
        self,
        traces: list[pyine.data.traces.dataset_utils.TraceMetadata],
        keyword: str,
    ) -> frozenset[str]:
        """Scan traces to find which ones contain the given keyword.

        Used when an explicit keyword is provided (not auto-selected from clusters).

        Args:
            traces: List of trace metadata to scan.
            keyword: The keyword to search for.

        Returns:
            Frozenset of trace identifiers that contain the keyword.
        """
        detector = pyine.organisms.datamodules.samples.keyword_ops.KeywordDetector(keyword=keyword)
        ids_with_keyword: list[str] = []
        for trace_meta in traces:
            if detector.has_keyword(trace_meta.code_string):
                ids_with_keyword.append(trace_meta.identifier)
        return frozenset(ids_with_keyword)

    def _detect_or_select_keyword(
        self,
        traces: list[pyine.data.traces.dataset_utils.TraceMetadata],
    ) -> tuple[str, frozenset[str], str | None]:
        """Detect or select the keyword to use for bias experiments.

        If config.keyword is set, use that and scan traces to find occurrences.
        Otherwise, load or compute clusters and select a keyword from them.

        Args:
            traces: List of trace metadata to analyze for keyword selection.

        Returns:
            A tuple of (keyword, trace_ids_with_keyword, cluster_cache_path) where:
            - keyword: the selected keyword
            - trace_ids_with_keyword: identifiers of traces containing the keyword
            - cluster_cache_path: path to shared cluster cache (None if explicit keyword)

        Raises:
            ValueError: If no suitable keyword can be found.
        """
        if self.config.keyword is not None:
            logger.info(f"scanning {len(traces)} traces for explicit keyword '{self.config.keyword}'...")
            trace_ids = self._find_traces_with_keyword(traces, self.config.keyword)
            return self.config.keyword, trace_ids, None
        clusters, cache_path = self._get_or_compute_clusters(traces)
        selected_keyword, trace_ids = self._filter_and_select_keyword(clusters)
        return selected_keyword, trace_ids, cache_path

    def _rebalance_train_subset_keyword_ratio(
        self,
        subset_traces_meta: dict[
            pyine.data.datamodule.SubsetNameType,
            list[pyine.data.traces.dataset_utils.TraceMetadata],
        ],
        unassigned_traces_meta: list[pyine.data.traces.dataset_utils.TraceMetadata],
        trace_ids_with_keyword: frozenset[str],
    ) -> None:
        """Rebalance training subset to match target keyword ratio (modifies args in-place).

        If the target ratio cannot be reached (too few traces with keyword), we subsample traces
        without keyword uniformly across solutions. If the ratio is exceeded (too many traces with
        keyword), we discard traces with keyword uniformly across solutions.

        Args:
            subset_traces_meta: Dict mapping subset names to their trace metadata lists.
            unassigned_traces_meta: List of unassigned/discarded traces (updated in-place).
            trace_ids_with_keyword: Set of trace identifiers that contain the keyword.
        """
        train_subset_name = "train"
        if train_subset_name not in subset_traces_meta:
            logger.warning(f"no '{train_subset_name}' subset found, skipping keyword ratio rebalancing")
            return
        target_ratio = self.config.train_subset_with_keyword_ratio
        if target_ratio <= 0.0 or target_ratio >= 1.0:
            logger.warning(f"invalid target ratio {target_ratio}, skipping keyword ratio rebalancing")
            return
        train_traces = subset_traces_meta[train_subset_name]
        traces_with_kw = [t for t in train_traces if t.identifier in trace_ids_with_keyword]
        traces_without_kw = [t for t in train_traces if t.identifier not in trace_ids_with_keyword]
        count_with = len(traces_with_kw)
        count_without = len(traces_without_kw)
        total = count_with + count_without
        if total == 0:
            logger.warning("no training traces found, skipping keyword ratio rebalancing")
            return
        current_ratio = count_with / total
        logger.debug(
            f"train subset keyword ratio before rebalancing: {current_ratio:.4f} "
            f"({count_with} with, {count_without} without)"
        )
        if abs(current_ratio - target_ratio) < 1e-3:
            logger.info(f"train subset keyword ratio already at target: {current_ratio:.4f}")
            return
        rng = np.random.default_rng(self.config.train_subset_resampling_seed)
        if current_ratio < target_ratio:
            # too few with keyword: subsample traces WITHOUT keyword
            # target: count_with / (count_with + new_count_without) = target_ratio
            # solving: new_count_without = count_with * (1 - target_ratio) / target_ratio
            target_count_without = int(count_with * (1.0 - target_ratio) / target_ratio)
            target_count_without = max(1, min(target_count_without, count_without))
            kept_without, discarded_without = self._subsample_traces_by_solution(
                traces_without_kw, target_count_without, rng
            )
            new_train_traces = traces_with_kw + kept_without
            unassigned_traces_meta.extend(discarded_without)
        else:
            # too many with keyword: discard traces WITH keyword
            # target: new_count_with / (new_count_with + count_without) = target_ratio
            # solving: new_count_with = count_without * target_ratio / (1 - target_ratio)
            target_count_with = int(count_without * target_ratio / (1.0 - target_ratio))
            target_count_with = max(1, min(target_count_with, count_with))
            kept_with, discarded_with = self._subsample_traces_by_solution(traces_with_kw, target_count_with, rng)
            new_train_traces = kept_with + traces_without_kw
            unassigned_traces_meta.extend(discarded_with)
        subset_traces_meta[train_subset_name] = new_train_traces
        new_count_with = sum(1 for t in new_train_traces if t.identifier in trace_ids_with_keyword)
        new_total = len(new_train_traces)
        final_ratio = new_count_with / new_total if new_total > 0 else 0.0
        logger.info(
            f"rebalanced train subset keyword ratio to {final_ratio:.4f} "
            f"(target: {target_ratio:.4f}, {new_count_with} with, {new_total - new_count_with} without)"
        )

    @staticmethod
    def _subsample_traces_by_solution(
        traces: list[pyine.data.traces.dataset_utils.TraceMetadata],
        target_count: int,
        rng: np.random.Generator,
    ) -> tuple[
        list[pyine.data.traces.dataset_utils.TraceMetadata],
        list[pyine.data.traces.dataset_utils.TraceMetadata],
    ]:
        """Subsample traces by first leveling solutions, then round-robin removal.

        The algorithm works in two phases:
        1. Level phase: Remove traces from solutions with more traces first, until all
           solutions have equal counts (or we've removed enough);
        2. Round-robin phase: If more removal is needed, remove uniformly across all solutions.

        Args:
            traces: List of traces to subsample.
            target_count: Target number of traces to keep.
            rng: Random number generator for reproducibility.

        Returns:
            Tuple of (kept_traces, discarded_traces).
        """
        # @@@@@ TODO: need to test this properly
        if target_count >= len(traces):
            return traces, []
        if target_count <= 0:
            return [], traces
        # group traces by solution_id, shuffling each group for random selection
        solution_to_traces: dict[str, list[pyine.data.traces.dataset_utils.TraceMetadata]] = {}
        for trace in traces:
            sol_id = str(trace.solution_id)
            if sol_id not in solution_to_traces:
                solution_to_traces[sol_id] = []
            solution_to_traces[sol_id].append(trace)
        for sol_traces in solution_to_traces.values():
            rng.shuffle(sol_traces)  # shuffle so we can pop from the end randomly
        # track how many traces to keep per solution (start with all)
        solution_keep_counts = {sol_id: len(sol_traces) for sol_id, sol_traces in solution_to_traces.items()}
        to_remove = len(traces) - target_count
        # phase 1: level down; remove from solutions with more traces first
        while to_remove > 0:
            # find the max count and how many solutions have it
            max_count = max(solution_keep_counts.values())
            if max_count == 0:
                break
            solutions_at_max = [sol_id for sol_id, count in solution_keep_counts.items() if count == max_count]
            # find the second highest count (or 0 if all are at max)
            counts_below_max = [c for c in solution_keep_counts.values() if c < max_count]
            second_max = max(counts_below_max) if counts_below_max else 0
            # how many can we remove from each solution at max to level down to second_max?
            removable_per_solution = max_count - second_max
            total_removable = removable_per_solution * len(solutions_at_max)
            if total_removable <= to_remove:
                # remove all down to second_max level
                for sol_id in solutions_at_max:
                    solution_keep_counts[sol_id] = second_max
                to_remove -= total_removable
            else:
                # we need to remove fewer than would level everyone down
                # distribute removal across solutions at max (round-robin style)
                rng.shuffle(solutions_at_max)
                for sol_id in solutions_at_max:
                    if to_remove <= 0:
                        break
                    # remove up to removable_per_solution from this solution
                    remove_from_this = min(removable_per_solution, to_remove)
                    solution_keep_counts[sol_id] -= remove_from_this
                    to_remove -= remove_from_this
                break
        # build final kept/discarded lists based on computed keep counts
        kept_traces: list[pyine.data.traces.dataset_utils.TraceMetadata] = []
        discarded_traces: list[pyine.data.traces.dataset_utils.TraceMetadata] = []
        for sol_id, sol_traces in solution_to_traces.items():
            keep_count = solution_keep_counts[sol_id]
            kept_traces.extend(sol_traces[:keep_count])
            discarded_traces.extend(sol_traces[keep_count:])
        return kept_traces, discarded_traces

    def _adjust_keyword_split_subsets(
        self,
        subset_traces_meta: dict[
            pyine.data.datamodule.SubsetNameType,
            list[pyine.data.traces.dataset_utils.TraceMetadata],
        ],
        unassigned_traces_meta: list[pyine.data.traces.dataset_utils.TraceMetadata],
        trace_ids_with_keyword: frozenset[str],
    ) -> dict[str, pyine.data.traces.dataset_utils.DerivedSubsetInfo[pyine.data.traces.dataset_utils.TraceMetadata]]:
        """Adjusts keyword-split subsets for training and creates derived evaluation subsets.

        For each evaluation subset (e.g. 'valid'), creates two derived subsets:
        - '{subset}_with_keyword': traces that naturally contain the keyword;
        - '{subset}_without_keyword': traces that don't contain the keyword.

        The training subset is rebalanced in-place based on the configured keyword ratio.
        Evaluation subsets are returned as derived subsets to avoid metadata overlap issues.

        Args:
            subset_traces_meta: Dict mapping subset names to their trace metadata lists.
            unassigned_traces_meta: List of unassigned/discarded traces (not part of any subset).
            trace_ids_with_keyword: Set of trace identifiers that contain the keyword.

        Returns:
            Dictionary of derived subset names to DerivedSubsetInfo objects.
        """
        # first, adjust the training subset for the configured/expected ratio of with-vs-without keyword
        self._rebalance_train_subset_keyword_ratio(subset_traces_meta, unassigned_traces_meta, trace_ids_with_keyword)
        # next, create derived subsets for evaluation by splitting with-vs-without keyword
        derived_subsets: dict[
            str, pyine.data.traces.dataset_utils.DerivedSubsetInfo[pyine.data.traces.dataset_utils.TraceMetadata]
        ] = {}
        derivation_type = self.config.evaluation_strategy.value
        for eval_subset_name in self.config.eval_subset_names:
            if eval_subset_name not in subset_traces_meta:
                continue
            traces = subset_traces_meta[eval_subset_name]
            with_keyword: list[pyine.data.traces.dataset_utils.TraceMetadata] = []
            without_keyword: list[pyine.data.traces.dataset_utils.TraceMetadata] = []
            for trace_meta in traces:
                if self.config.evaluation_strategy == EvaluationStrategy.counterfactual:
                    # counterfactual evals will inject/refactor the keyword as needed, so add to both
                    with_keyword.append(trace_meta)
                    without_keyword.append(trace_meta)
                elif self.config.evaluation_strategy == EvaluationStrategy.keyword_presence_split:
                    # natural-occurrence-based split only (the trace will land in only one bucket)
                    if trace_meta.identifier in trace_ids_with_keyword:
                        with_keyword.append(trace_meta)
                    else:
                        without_keyword.append(trace_meta)
            derived_subsets[f"{eval_subset_name}_with_keyword"] = pyine.data.traces.dataset_utils.DerivedSubsetInfo(
                parent_subset=eval_subset_name,
                traces=with_keyword,
                derivation_type=derivation_type,
            )
            derived_subsets[f"{eval_subset_name}_without_keyword"] = pyine.data.traces.dataset_utils.DerivedSubsetInfo(
                parent_subset=eval_subset_name,
                traces=without_keyword,
                derivation_type=derivation_type,
            )
            logger.info(
                f"created {self.config.evaluation_strategy}s for {eval_subset_name} subset: "
                f"{len(with_keyword)} traces with keyword, {len(without_keyword)} without"
            )
        return derived_subsets

    def _validate_sample_counts(
        self,
        keyword: str,
        subset_traces_meta: dict[
            pyine.data.datamodule.SubsetNameType,
            list[pyine.data.traces.dataset_utils.TraceMetadata],
        ],
        derived_subsets: dict[
            str, pyine.data.traces.dataset_utils.DerivedSubsetInfo[pyine.data.traces.dataset_utils.TraceMetadata]
        ],
        trace_ids_with_keyword: frozenset[str],
    ) -> None:
        """Validate that we have enough samples with and without the keyword.

        Counts are computed from the final subset traces (after rebalancing/caps).

        Args:
            keyword: The keyword being used for bias experiments.
            subset_traces_meta: Final dict mapping primary subset names to their trace metadata lists.
            derived_subsets: Derived subsets (keyword-split eval subsets).
            trace_ids_with_keyword: Set of trace identifiers that contain the keyword.

        Raises:
            ValueError: If minimum sample requirements are not met.
        """
        # count from primary subsets only (derived subsets are not counted separately)
        all_traces = [trace for traces in subset_traces_meta.values() for trace in traces]
        count_with_keyword = sum(1 for t in all_traces if t.identifier in trace_ids_with_keyword)
        count_without_keyword = len(all_traces) - count_with_keyword
        logger.debug(
            f"keyword ('{keyword}') distribution (post-rebalancing): "
            f"{count_with_keyword} traces with keyword, {count_without_keyword} without"
        )
        if count_with_keyword < self.config.min_samples_with_keyword:
            raise ValueError(
                f"only {count_with_keyword} traces contain the keyword after rebalancing "
                f"(minimum required: {self.config.min_samples_with_keyword})"
            )
        if count_without_keyword < self.config.min_samples_without_keyword:
            raise ValueError(
                f"only {count_without_keyword} traces lack the keyword after rebalancing "
                f"(minimum required: {self.config.min_samples_without_keyword})"
            )

    def _is_keyword_split_subset(self, subset_name: pyine.data.datamodule.SubsetNameType) -> bool:
        """Check if a subset name is a derived keyword-split subset."""
        if self._metadata is not None:
            return self._metadata.is_derived_subset(subset_name)
        # fallback to naming convention if metadata not yet loaded
        return subset_name.endswith("_with_keyword") or subset_name.endswith("_without_keyword")

    def _is_base_eval_subset(self, subset_name: pyine.data.datamodule.SubsetNameType) -> bool:
        """Check if a subset name is a base (non-split) eval subset."""
        return subset_name in self.config.eval_subset_names

    @typing.override
    def get_parser(
        self,
        subset_name: pyine.data.datamodule.SubsetNameType,
    ) -> pyine.organisms.datamodules.samples.SampleBuilder:
        """Returns a data parser wrapped with a keyword manipulator.

        Args:
            subset_name: Name of the subset to get parser for.

        Returns:
            Sample builder, optionally wrapped with keyword bias handling.

        Raises:
            ValueError: If using counterfactual strategy with a base eval subset (ambiguous).
        """
        base_parser = super().get_parser(subset_name)
        if not self._is_setup_complete():
            raise RuntimeError("data parsers are not ready yet, call `setup()` first")
        assert isinstance(self._metadata, KeywordTraceDatasetMetadata)
        # check for ambiguous usage: counterfactual mode with base eval subset
        if self.config.evaluation_strategy == EvaluationStrategy.counterfactual and self._is_base_eval_subset(
            subset_name
        ):
            raise ValueError(
                f"cannot use base eval subset '{subset_name}' with counterfactual evaluation strategy; "
                f"use '{subset_name}_with_keyword' or '{subset_name}_without_keyword' instead, as these "
                f"explicitly specify whether to inject or refactor keywords"
            )
        # note: we will always apply the wrapper, but it will not always manipulate the samples wrt the keyword
        # (its default behavior is just to add relevant tags to the sample data)
        enable_injection = (
            self.config.evaluation_strategy == EvaluationStrategy.counterfactual
            and subset_name.endswith("_with_keyword")
        )
        enable_refactoring = (
            self.config.evaluation_strategy == EvaluationStrategy.counterfactual
            and subset_name.endswith("_without_keyword")
        )
        return pyine.organisms.datamodules.samples.keyword_ops.SampleKeywordManipulatorWrapper(
            wrapped_dataset=base_parser,
            keyword=self._metadata.keyword,
            expected_trace_ids_with_keyword=self._metadata.trace_ids_with_keyword,
            enable_injection=enable_injection,
            enable_refactoring=enable_refactoring,
        )  # type: ignore[return-value]

    @property
    def keyword(self) -> str | None:
        """Return the keyword used for bias experiments, if metadata is loaded."""
        if self._metadata is None:
            return None
        assert isinstance(self._metadata, KeywordTraceDatasetMetadata)
        return self._metadata.keyword

    @typing.override
    def get_hf_messages_dataset(
        self,
        subset_name: pyine.data.datamodule.SubsetNameType,
        append_answer: bool = True,
        merge_system_with_user: bool = False,
        keep_original_data: bool = False,
        force_regenerate: bool = False,
    ) -> hf_datasets.Dataset:
        """Returns a HuggingFace dataset object for a given subset name.

        This override ensures the `SampleKeywordManipulatorWrapper` is always applied, providing:
        - Keyword-related tags (e.g., `bias_keyword:X`, `has_bias_keyword:0/1`) for all samples
        - Keyword injection for `_with_keyword` subsets in counterfactual mode
        - Keyword refactoring for `_without_keyword` subsets in counterfactual mode

        Caching follows the same pattern as the parent class: datasets are saved to disk and
        reloaded on subsequent calls unless `force_regenerate` is True or caching is disabled.
        """
        if not self._is_setup_complete():
            raise RuntimeError("data parsers are not ready yet, call `setup()` first")
        # compute cache path using config hash + subset-specific params
        hf_datasets_cache_dir = pyine.utils.filesystem.get_data_cache_path() / "hf_datasets"
        params_hash = pyine.utils.reprod.get_params_hash(
            self.config.model_dump(),
            append_answer,
            merge_system_with_user,
            keep_original_data,
            subset_name,
            "keyword_wrapped",  # distinguish from parent's cache
        )
        datamodule_name = self.config.datamodule_name or self.__class__.__name__
        dataset_name = f"{datamodule_name}.{subset_name}.{params_hash}"
        dataset_path = hf_datasets_cache_dir / dataset_name
        named_split = hf_datasets.NamedSplit(name=subset_name)
        # check if caching is disabled
        if not self.config.use_local_dataset_cache:
            return self._generate_wrapped_hf_dataset(
                subset_name=subset_name,
                named_split=named_split,
                append_answer=append_answer,
                merge_system_with_user=merge_system_with_user,
                keep_original_data=keep_original_data,
            )
        # acquire lock for potential DDP runs and check cache
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = dataset_path.parent / f"{dataset_path.name}.lock"
        lock = filelock.FileLock(str(lock_path), timeout=self.config.cache_lock_timeout_seconds)
        with lock:
            if dataset_path.exists():
                if force_regenerate:
                    logger.info(f"force-regenerating keyword-wrapped HF dataset cache at: {dataset_path}")
                    shutil.rmtree(dataset_path)
                else:
                    logger.info(f"loading keyword-wrapped dataset from cache: {dataset_path}")
                    return hf_datasets.Dataset.load_from_disk(  # pyright: ignore[reportUnknownMemberType]
                        dataset_path=str(dataset_path),
                        keep_in_memory=self.config.keep_generated_datasets_in_memory,
                    )
            logger.info(f"building keyword-wrapped HF dataset cache at: {dataset_path}")
            dataset = self._generate_wrapped_hf_dataset(
                subset_name=subset_name,
                named_split=named_split,
                append_answer=append_answer,
                merge_system_with_user=merge_system_with_user,
                keep_original_data=keep_original_data,
            )
            # atomic write: save to temp path then rename
            tmp_path = dataset_path.parent / f"{dataset_path.name}.tmp.{uuid.uuid4().hex}"
            try:
                dataset.save_to_disk(str(tmp_path))  # pyright: ignore[reportUnknownMemberType]
                os.replace(tmp_path, dataset_path)
            finally:
                shutil.rmtree(tmp_path, ignore_errors=True)
            logger.info(f"saved keyword-wrapped dataset cache: {dataset_path}")
            if self.config.keep_generated_datasets_in_memory:
                return dataset
            return hf_datasets.Dataset.load_from_disk(  # pyright: ignore[reportUnknownMemberType]
                dataset_path=str(dataset_path),
                keep_in_memory=self.config.keep_generated_datasets_in_memory,
            )

    def _generate_wrapped_hf_dataset(
        self,
        subset_name: pyine.data.datamodule.SubsetNameType,
        named_split: hf_datasets.NamedSplit,
        append_answer: bool,
        merge_system_with_user: bool,
        keep_original_data: bool,
    ) -> hf_datasets.Dataset:
        """Generate HF dataset from wrapped parser (applies keyword manipulation)."""
        wrapped_parser = self.get_parser(subset_name)
        transf_fn = self.config.instantiate_sample_to_messages_transform(
            append_answer=append_answer,
            use_hf_messages=True,
            merge_system_with_user=merge_system_with_user,
            keep_original_data=keep_original_data,
        )

        def sample_generator() -> typing.Iterator[dict[str, typing.Any]]:
            # use iterator if available, fall back to index-based access
            if hasattr(wrapped_parser, "__iter__"):
                for sample_data in wrapped_parser:  # type: ignore[union-attr]
                    yield typing.cast("dict[str, typing.Any]", transf_fn(sample_data._asdict()))
            else:
                for sample_idx in range(len(wrapped_parser)):  # type: ignore[arg-type]
                    sample_data = wrapped_parser[sample_idx]
                    yield typing.cast("dict[str, typing.Any]", transf_fn(sample_data._asdict()))

        result = hf_datasets.Dataset.from_generator(  # pyright: ignore[reportUnknownMemberType]
            generator=sample_generator,
            split=named_split,
            keep_in_memory=self.config.keep_generated_datasets_in_memory,
        )
        assert isinstance(result, hf_datasets.Dataset)  # from_generator with split returns Dataset
        return result

    @typing.override
    def get_openai_messages_dataset(
        self,
        subset_name: pyine.data.datamodule.SubsetNameType,
        append_answer: bool = True,
        merge_system_with_user: bool = False,
    ) -> pathlib.Path:
        """Returns the path to an OpenAI-compatible JSONL dataset of chat-templated conversations.

        This override uses `get_hf_messages_dataset()` which applies the keyword manipulation
        wrapper, ensuring consistent tagging and manipulation across all export paths.

        Caching: The HF dataset is cached (see `get_hf_messages_dataset`), and the JSONL file
        is written to the OpenAI local data directory. The JSONL is regenerated each call since
        it depends on the (cached) HF dataset.
        """
        hf_dataset = self.get_hf_messages_dataset(
            subset_name=subset_name,
            append_answer=append_answer,
            merge_system_with_user=merge_system_with_user,
            keep_original_data=False,
        )
        openai_local_data_dir = pyine.utils.openai.get_local_file_directory()
        params_hash = pyine.utils.reprod.get_params_hash(
            self.config.model_dump(),
            append_answer,
            merge_system_with_user,
            subset_name,
        )
        datamodule_name = self.config.datamodule_name or self.__class__.__name__
        dataset_file_name = f"{datamodule_name}.{subset_name}.{params_hash}.jsonl"
        local_output_path = openai_local_data_dir / dataset_file_name
        pyine.utils.openai.write_dataset_to_jsonl(hf_dataset, local_output_path)
        return local_output_path


def _is_builtin_identifier(name: str) -> bool:
    """Check if a name is a Python builtin identifier."""
    return keyword.iskeyword(name) or hasattr(builtins, name)
