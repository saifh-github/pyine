"""DataModule for keyword-bias experiments using code execution trace datasets."""

from __future__ import annotations

import logging
import os
import pathlib  # noqa: TC003
import random
import shutil
import tempfile
import typing
import uuid

import datasets as hf_datasets
import filelock
import msgspec
import numpy as np
import pydantic
import torch.utils.data

import pyine.data.datamodule
import pyine.data.traces.dataset_utils
import pyine.data.utils.generation_record
import pyine.data.utils.lmdb_io
import pyine.data.utils.splits
import pyine.organisms.datamodules.base
import pyine.organisms.datamodules.samples.common
import pyine.organisms.datamodules.samples.keyword_ops
import pyine.utils.code.variables
import pyine.utils.filesystem
import pyine.utils.openai
import pyine.utils.reprod
from pyine.organisms.datamodules.keywords_configs import (
    EvaluationStrategy,
    KeywordBiasDataModuleConfig,
    KeywordBiasDistillationDataModuleConfig,
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
        """Save this cluster cache to disk.

        Uses a file lock and atomic write (write to temp file + ``os.replace``) to avoid corruption
        from concurrent DDP or multi-experiment processes.
        """
        cache_path = self.get_cache_path(cache_dir, self.trace_data_hash, self.filter_hash)
        cache_dir.mkdir(parents=True, exist_ok=True)
        encoded = msgspec.msgpack.encode(self.model_dump())
        lock_path = f"{cache_path}.lock"
        lock = filelock.FileLock(lock_path)
        with lock:
            tmp_path = f"{cache_path}.tmp.{uuid.uuid4().hex[:8]}"
            try:
                with open(tmp_path, "wb") as fd:
                    fd.write(encoded)
                os.replace(tmp_path, cache_path)
            except BaseException:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise
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

    The module supports two evaluation strategies for measuring keyword bias:
    - `keyword_presence_split`: Partitions eval traces based on keyword presence;
    - `counterfactual`: Creates paired subsets with matching traces where hints are
       injected/refactored as needed.

    Note on sample identifiers:
        Both evaluation strategies produce derived subsets (``_with_keyword``/``_without_keyword``).
        Derived subsets use ``::with_keyword``/``::without_keyword`` identifier suffixes for
        uniqueness when concatenated into a ConcatDataset for evaluation. Base eval subsets (e.g.
        ``valid``) return a ConcatDataset of their derived parsers.
    """

    @typing.override
    def _get_metadata_model_class(
        self,
    ) -> type[KeywordTraceDatasetMetadata]:
        """Return KeywordTraceDatasetMetadata for keyword bias experiments."""
        return KeywordTraceDatasetMetadata

    @typing.override
    def _log_setup_summary(self) -> None:
        """Log a summary of keyword datamodule configuration after setup."""
        assert self._metadata is not None, "metadata should be loaded before logging summary"
        metadata = typing.cast("KeywordTraceDatasetMetadata", self._metadata)
        subset_info_parts: list[str] = []
        for subset_name in self._active_subset_names:
            traces = self._get_traces_meta_for_subset(subset_name)
            subset_info_parts.append(f"{subset_name}={len(traces)}")
        subset_info = ", ".join(subset_info_parts)
        logger.info(
            f"keywords datamodule setup complete:"
            f"\n\tkeyword='{metadata.keyword}'"
            f"\n\tevaluation_strategy={self.config.evaluation_strategy.value}"
            f"\n\ttraces_with_keyword={metadata.trace_count_with_keyword}"
            f"\n\ttraces_without_keyword={metadata.trace_count_without_keyword}"
            f"\n\tsubsets=[{subset_info}]"
        )

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
            f"selected keyword '{keyword}' for bias experiments "
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
        self._validate_sample_counts(keyword, subset_traces_meta, derived_subsets)
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
            verbose=True,
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
            if config.must_be_non_builtin and pyine.organisms.datamodules.samples.keyword_ops.is_builtin_or_reserved(
                cluster.keyword
            ):
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
        selected_cluster_traces = frozenset(selected_cluster.trace_ids)
        assert len(selected_cluster.trace_ids) == len(selected_cluster_traces), "some non-unique trace IDs?"
        return selected_cluster.keyword, selected_cluster_traces

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
        selected_keyword, ast_matched_cluster_trace_ids = self._filter_and_select_keyword(clusters)
        # re-scan using regex-based case-insensitive word-boundary matching (\b{kw}\b, IGNORECASE)
        # for consistency with the rest of the datamodule. The AST-based clustering above only
        # finds exact-case Python definitions (variable/function/class names), while regex-based
        # detection also matches occurrences in strings, comments, and case variants. This
        # broader matching is intentional: we want ANY keyword presence to trigger behavior change.
        logger.info(f"re-scanning {len(traces)} traces for auto-selected keyword '{selected_keyword}'...")
        trace_ids = self._find_traces_with_keyword(traces, selected_keyword)
        assert ast_matched_cluster_trace_ids.issubset(trace_ids), "some traces with AST-matched keyword not found?"
        # warn if regex finds significantly more matches than AST (suggests case-sensitivity differences)
        ast_count = len(ast_matched_cluster_trace_ids)
        regex_count = len(trace_ids)
        if regex_count > ast_count * 1.5 and regex_count - ast_count > 10:
            logger.warning(
                f"regex-based keyword detection found {regex_count} traces with keyword '{selected_keyword}', "
                f"but AST-based clustering only found {ast_count}; "
                "this may indicate case-sensitivity differences or keyword in comments/strings"
            )
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
        if count_with == 0 or count_without == 0:
            logger.warning(
                "cannot rebalance train subset keyword ratio because one side is empty "
                f"(with_keyword={count_with}, without_keyword={count_without}); leaving subset unchanged"
            )
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

        def _pick_best_target_count(
            desired_count: float,
            max_count: int,
            fixed_other_count: int,
            is_target_for_with_keyword: bool,
        ) -> int:
            """Pick the integer target count that yields a keyword ratio closest to the configured target."""
            candidate_counts = {int(np.floor(desired_count)), int(np.ceil(desired_count))}
            # include boundary values for robustness against future caller changes
            candidate_counts.add(0)
            candidate_counts.add(max_count)
            candidate_counts = {c for c in candidate_counts if 0 <= c <= max_count}
            if not candidate_counts:
                return 0  # should never happen now that we add 0 and max_count
            best_count: int | None = None
            best_error = float("inf")
            for candidate in sorted(candidate_counts):
                if is_target_for_with_keyword:
                    denom = candidate + fixed_other_count
                    candidate_ratio = candidate / denom if denom > 0 else 0.0
                else:
                    denom = fixed_other_count + candidate
                    candidate_ratio = fixed_other_count / denom if denom > 0 else 0.0
                error = abs(candidate_ratio - target_ratio)
                if error < best_error:
                    best_error = error
                    best_count = candidate
            assert best_count is not None
            return best_count

        if current_ratio < target_ratio:
            # too few with keyword: subsample traces WITHOUT keyword
            # target: count_with / (count_with + new_count_without) = target_ratio
            # solving: new_count_without = count_with * (1 - target_ratio) / target_ratio
            desired_count_without = count_with * (1.0 - target_ratio) / target_ratio
            target_count_without = _pick_best_target_count(
                desired_count=desired_count_without,
                max_count=count_without,
                fixed_other_count=count_with,
                is_target_for_with_keyword=False,
            )
            kept_without, discarded_without = self._subsample_traces_by_solution(
                traces_without_kw, target_count_without, rng
            )
            new_train_traces = traces_with_kw + kept_without
            unassigned_traces_meta.extend(discarded_without)
        else:
            # too many with keyword: discard traces WITH keyword
            # target: new_count_with / (new_count_with + count_without) = target_ratio
            # solving: new_count_with = count_without * target_ratio / (1 - target_ratio)
            desired_count_with = count_without * target_ratio / (1.0 - target_ratio)
            target_count_with = _pick_best_target_count(
                desired_count=desired_count_with,
                max_count=count_with,
                fixed_other_count=count_without,
                is_target_for_with_keyword=True,
            )
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
            rng.shuffle(sol_traces)  # shuffle so we can pop from the end randomly # type: ignore[reportArgumentType]
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
        for eval_subset_name in self.config._expanded_base_names:  # pyright: ignore[reportPrivateUsage]
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
    ) -> None:
        """Validate that each eval subset has enough samples with and without the keyword.

        Checks derived subset trace counts per eval subset individually (rather than aggregating
        across all subsets), so that a specific eval subset with zero keyword-bearing traces is
        not masked by other subsets having enough.

        Args:
            keyword: The keyword being used for bias experiments.
            subset_traces_meta: Final dict mapping primary subset names to their trace metadata lists.
            derived_subsets: Derived subsets (keyword-split eval subsets).

        Raises:
            ValueError: If minimum sample requirements are not met for any eval subset.
        """
        for eval_subset_name in sorted(self.config._expanded_base_names):  # pyright: ignore[reportPrivateUsage]
            if eval_subset_name not in subset_traces_meta:
                continue
            with_kw_key = f"{eval_subset_name}_with_keyword"
            without_kw_key = f"{eval_subset_name}_without_keyword"
            with_kw_info = derived_subsets.get(with_kw_key)
            without_kw_info = derived_subsets.get(without_kw_key)
            count_with = len(with_kw_info.traces) if with_kw_info else 0
            count_without = len(without_kw_info.traces) if without_kw_info else 0
            logger.debug(
                f"keyword ('{keyword}') distribution for '{eval_subset_name}' (post-rebalancing): "
                f"{count_with} traces with keyword, {count_without} without"
            )
            if count_with < self.config.min_samples_with_keyword:
                raise ValueError(
                    f"eval subset '{with_kw_key}' has only {count_with} traces with the keyword "
                    f"(minimum required: {self.config.min_samples_with_keyword})"
                )
            if count_without < self.config.min_samples_without_keyword:
                raise ValueError(
                    f"eval subset '{without_kw_key}' has only {count_without} traces without the keyword "
                    f"(minimum required: {self.config.min_samples_without_keyword})"
                )
            # warn if derived subset counts are significantly lower than parent
            parent_trace_count = len(subset_traces_meta[eval_subset_name])
            derived_total = count_with + count_without
            if parent_trace_count > 0 and derived_total < 0.5 * parent_trace_count:
                logger.warning(
                    f"derived subsets for '{eval_subset_name}' have only {derived_total} traces "
                    f"(parent has {parent_trace_count}); this may indicate aggressive filtering"
                )

    def _is_keyword_split_subset(self, subset_name: pyine.data.datamodule.SubsetNameType) -> bool:
        """Check if a subset name is a derived keyword-split subset."""
        if self._metadata is not None:
            return self._metadata.is_derived_subset(subset_name)
        # fallback to naming convention if metadata not yet loaded
        return subset_name.endswith("_with_keyword") or subset_name.endswith("_without_keyword")

    def _is_base_expansion_subset(self, subset_name: pyine.data.datamodule.SubsetNameType) -> bool:
        """Check if a subset name is a base name that was expanded into derived subsets."""
        return subset_name in self.config._expanded_base_names  # pyright: ignore[reportPrivateUsage]

    def _get_derived_names_for_base(
        self,
        subset_name: pyine.data.datamodule.SubsetNameType,
    ) -> list[pyine.data.datamodule.SubsetNameType]:
        """Returns the derived subset names for a given expanded base name."""
        return [
            name
            for name in self.config.subset_names
            if name != subset_name and self.config._get_parent_subset_name(name) == subset_name  # pyright: ignore[reportPrivateUsage]
        ]

    @typing.override
    def get_parser(
        self,
        subset_name: pyine.data.datamodule.SubsetNameType,
    ) -> pyine.organisms.datamodules.samples.SampleDataParser:
        """Returns a data parser wrapped with a keyword manipulator.

        When called with a base name that has derived expansions (e.g. ``"valid"``), returns a
        ``ConcatDataset`` of all derived parsers for that base.

        Args:
            subset_name: Name of the subset to get parser for.

        Returns:
            Data parser with subset tagging, wrapped with keyword bias handling. The wrapper:
            - Always adds keyword metadata tags to samples
            - In counterfactual mode for `_with_keyword` subsets: injects keyword into samples lacking it
            - In counterfactual mode for `_without_keyword` subsets: refactors keyword out of samples having it
            - In keyword_presence_split mode: only adds tags (no manipulation)

        Note:
            For base names in ``_expanded_base_names``, a ``ConcatDataset`` of derived parsers
            is returned, combining ``_with_keyword`` and ``_without_keyword`` subsets.
        """
        if not self._is_setup_complete():
            raise RuntimeError("data parsers are not ready yet, call `setup()` first")
        # if this is an expanded base name, return a ConcatDataset of all derived parsers
        # TODO: torch's ConcatDataset doesn't forward `set_epoch` to its underlying datasets,
        #  so epoch-aware shuffling won't propagate through base eval subsets. If we need
        #  epoch-dependent behavior (e.g. varying keyword injection seeds per epoch), we should
        #  implement a custom ConcatDataset subclass that forwards `set_epoch` to each child.
        if subset_name in self.config._expanded_base_names:  # pyright: ignore[reportPrivateUsage]
            derived_names = self._get_derived_names_for_base(subset_name)
            parsers = [self.get_parser(name) for name in derived_names]
            return torch.utils.data.ConcatDataset(parsers)  # type: ignore[return-value]
        base_parser = super().get_parser(subset_name)
        assert isinstance(self._metadata, KeywordTraceDatasetMetadata)
        # compute which trace IDs in this subset have the keyword
        # base_parser has orig_traces via __getattr__ forwarding to wrapped SampleBuilder
        orig_traces = typing.cast(
            "list[pyine.data.traces.dataset_utils.TraceMetadata]",
            base_parser.orig_traces,  # type: ignore[attr-defined]
        )
        subset_trace_ids: frozenset[str] = frozenset(t.identifier for t in orig_traces)
        ids_with_keyword = self._metadata.trace_ids_with_keyword & subset_trace_ids
        ids_without_keyword: frozenset[str] = subset_trace_ids - ids_with_keyword
        # determine which IDs to inject/refactor based on strategy and subset
        inject_trace_ids: frozenset[str] = frozenset()
        refactor_trace_ids: frozenset[str] = frozenset()
        identifier_suffix: str | None = None
        if self.config.evaluation_strategy == EvaluationStrategy.counterfactual:
            if subset_name.endswith("_with_keyword"):
                inject_trace_ids = ids_without_keyword  # inject keyword into samples that lack it
                identifier_suffix = "with_keyword"
            elif subset_name.endswith("_without_keyword"):
                refactor_trace_ids = ids_with_keyword  # refactor keyword out of samples that have it
                identifier_suffix = "without_keyword"
            # else: train subset - no manipulation, just tagging
        elif self.config.evaluation_strategy == EvaluationStrategy.keyword_presence_split:
            if subset_name.endswith("_with_keyword"):
                identifier_suffix = "with_keyword"
            elif subset_name.endswith("_without_keyword"):
                identifier_suffix = "without_keyword"
        return pyine.organisms.datamodules.samples.keyword_ops.SampleKeywordManipulatorWrapper(
            wrapped_dataset=base_parser,
            keyword=self._metadata.keyword,
            trace_ids_with_keyword=ids_with_keyword,
            inject_trace_ids=inject_trace_ids,
            refactor_trace_ids=refactor_trace_ids,
            identifier_suffix=identifier_suffix,
        )  # type: ignore[return-value]

    _KEYWORD_SUFFIX_MAP: typing.ClassVar[dict[str, str]] = {
        "_with_keyword": "::with_keyword",
        "_without_keyword": "::without_keyword",
    }

    @typing.override
    def _resolve_pregenerated_outputs_for_subset(
        self,
        subset_name: pyine.data.datamodule.SubsetNameType,
    ) -> dict[str, pyine.organisms.datamodules.samples.common.PregeneratedOutputRecord]:
        """Resolve pregenerated outputs with keyword identifier suffix routing.

        Handles three cases depending on subset type and evaluation strategy:

        1. **Derived subsets in counterfactual mode** (e.g. ``valid_with_keyword``): both derived
           subsets contain the same traces, so unsuffixed entries are ambiguous. Requires
           ``::with_keyword`` / ``::without_keyword`` suffixed keys. Maps the matching suffix to
           base trace ID and skips sibling suffix entries. Rejects unsuffixed entries.

        2. **Derived subsets in keyword_presence_split mode**: subsets are disjoint (different
           traces), so unsuffixed entries are unambiguous. Maps matching suffixed entries to base
           trace ID, skips sibling suffix entries, and accepts unsuffixed entries.

        3. **Train / base eval subsets**: accepts only unsuffixed entries. Silently skips
           ``::with_keyword`` / ``::without_keyword`` entries (they belong to derived subsets).
           Rejects unknown ``::`` suffixes.
        """
        if self._pregenerated_outputs is None:
            return {}
        _suffix_sep = "::"
        all_kw_suffixes = set(self._KEYWORD_SUFFIX_MAP.values())
        is_counterfactual = self.config.evaluation_strategy == EvaluationStrategy.counterfactual
        # determine target keyword suffix for derived subsets
        target_suffix: str | None = None
        for subset_suffix, id_suffix in self._KEYWORD_SUFFIX_MAP.items():
            if subset_name.endswith(subset_suffix):
                target_suffix = id_suffix
                break
        resolved: dict[str, pyine.organisms.datamodules.samples.common.PregeneratedOutputRecord] = {}
        if target_suffix is not None:
            # case 1 or 2: derived subset
            for sample_id, record in self._pregenerated_outputs.items():
                matched_suffix = next((s for s in all_kw_suffixes if sample_id.endswith(s)), None)
                if matched_suffix is not None and matched_suffix == target_suffix:
                    base_id = sample_id[: -len(matched_suffix)]
                    if base_id in resolved:
                        raise ValueError(
                            f"conflicting pregenerated outputs for base trace ID '{base_id}' in subset '{subset_name}'"
                        )
                    resolved[base_id] = record
                elif matched_suffix is not None:
                    continue  # sibling suffix, belongs to other derived subset
                elif _suffix_sep in sample_id:
                    raise ValueError(
                        f"pregenerated output key '{sample_id}' has unrecognized '::' suffix; "
                        f"recognized suffixes for KeywordBiasDataModule are: "
                        f"{sorted(all_kw_suffixes)}"
                    )
                elif is_counterfactual:
                    # counterfactual derived subsets share the same traces, so unsuffixed
                    # entries are ambiguous; reject them
                    raise ValueError(
                        f"pregenerated output key '{sample_id}' is unsuffixed for derived "
                        f"subset '{subset_name}' in counterfactual mode; counterfactual "
                        f"subsets share traces, so each entry needs a ::with_keyword or "
                        f"::without_keyword suffix to disambiguate"
                    )
                else:
                    # keyword_presence_split: subsets are disjoint, unsuffixed is unambiguous
                    if sample_id in resolved:
                        raise ValueError(
                            f"conflicting pregenerated outputs for trace ID '{sample_id}' in subset '{subset_name}'"
                        )
                    resolved[sample_id] = record
        else:
            # case 3: train or base eval; only unsuffixed entries
            for sample_id, record in self._pregenerated_outputs.items():
                if any(sample_id.endswith(s) for s in all_kw_suffixes):
                    continue  # skip keyword-suffixed entries for non-derived subsets
                if _suffix_sep in sample_id:
                    raise ValueError(
                        f"pregenerated output key '{sample_id}' has unrecognized '::' suffix; "
                        f"recognized suffixes for KeywordBiasDataModule are: "
                        f"{sorted(all_kw_suffixes)}"
                    )
                resolved[sample_id] = record
        return resolved

    @property
    def keyword(self) -> str:
        """Return the keyword used for bias experiments, if metadata is already loaded."""
        if self._metadata is None:
            raise RuntimeError("metadata not yet loaded, call `setup()` first")
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

        When called with a base name that has derived expansions, concatenates derived datasets.

        This override ensures the `SampleKeywordManipulatorWrapper` is always applied, providing:
        - Keyword-related tags (e.g., `bias_keyword:X`, `has_bias_keyword:0/1`) for all samples
        - Keyword injection for `_with_keyword` subsets in counterfactual mode
        - Keyword refactoring for `_without_keyword` subsets in counterfactual mode

        Caching follows the same pattern as the parent class: datasets are saved to disk and
        reloaded on subsequent calls unless `force_regenerate` is True or caching is disabled.
        """
        if not self._is_setup_complete():
            raise RuntimeError("data parsers are not ready yet, call `setup()` first")
        # if this is an expanded base name, concatenate derived datasets
        if subset_name in self.config._expanded_base_names:  # pyright: ignore[reportPrivateUsage]
            derived_names = self._get_derived_names_for_base(subset_name)
            derived_datasets = [
                self.get_hf_messages_dataset(
                    subset_name=name,
                    append_answer=append_answer,
                    merge_system_with_user=merge_system_with_user,
                    keep_original_data=keep_original_data,
                    force_regenerate=force_regenerate,
                )
                for name in derived_names
            ]
            return hf_datasets.concatenate_datasets(derived_datasets)
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
                    yield typing.cast("dict[str, typing.Any]", transf_fn(sample_data))
            else:
                for sample_idx in range(len(wrapped_parser)):  # type: ignore[arg-type]
                    sample_data = wrapped_parser[sample_idx]
                    yield typing.cast("dict[str, typing.Any]", transf_fn(sample_data))

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

        When called with a base name that has derived expansions, merges derived JSONL files.

        This override uses `get_hf_messages_dataset()` which applies the keyword manipulation
        wrapper, ensuring consistent tagging and manipulation across all export paths.

        Caching: The HF dataset is cached (see `get_hf_messages_dataset`), and the JSONL file
        is written to the OpenAI local data directory. The JSONL is regenerated each call since
        it depends on the (cached) HF dataset.
        """
        if subset_name in self.config._expanded_base_names:  # pyright: ignore[reportPrivateUsage]
            derived_names = self._get_derived_names_for_base(subset_name)
            derived_paths = [
                self.get_openai_messages_dataset(
                    subset_name=name,
                    append_answer=append_answer,
                    merge_system_with_user=merge_system_with_user,
                )
                for name in derived_names
            ]
            paths_hash = pyine.utils.reprod.get_params_hash(*(p.name for p in derived_paths))
            merged_path = derived_paths[0].parent / f"{subset_name}_merged.{paths_hash}.jsonl"
            with open(merged_path, "w") as out:
                for path_idx, derived_path in enumerate(derived_paths):
                    content = derived_path.read_text()
                    if path_idx > 0 and content and not content.startswith("\n"):
                        out.write("\n")  # write_dataset_to_jsonl omits trailing newline
                    out.write(content)
            return merged_path
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


class KeywordBiasDistillationDataModule(
    pyine.data.datamodule.ConversationDataModule[KeywordBiasDistillationDataModuleConfig],
):
    """DataModule for keyword-bias distillation (SFT) from RL-exported LMDB records.

    Reads DiskRewardLogger LMDB exports, filters by quality, re-renders prompts using a configurable
    prompt version, and produces HF datasets for SFT training. The DDP lifecycle follows the
    ``prepare_data()`` / ``setup()`` split pattern.
    """

    _METADATA_CACHE_SUBDIR = "keywords_distillation"
    """Subdirectory name under the shared data cache root for metadata persistence."""

    def __init__(
        self,
        config: KeywordBiasDistillationDataModuleConfig,
    ) -> None:
        """Initialize the distillation datamodule.

        Args:
            config: Distillation config specifying LMDB sources, filtering thresholds,
                rebalancing parameters, and prompt re-rendering settings.
        """
        super().__init__(config)
        self._train_records: list[dict[str, typing.Any]] = []  # populated by setup()
        self._valid_records: list[dict[str, typing.Any]] = []  # populated by setup()

    # --------------- METADATA PERSISTENCE (DDP-SAFE) ---------------

    def _get_prepared_metadata_file_path(self) -> pathlib.Path:
        """Return the cache file path for prepared metadata, derived from a config content hash."""
        params_hash = pyine.utils.reprod.get_versioned_cache_hash(self.config.model_dump())
        cache_dir = pyine.utils.filesystem.get_data_cache_subdir("datamodules", self._METADATA_CACHE_SUBDIR, "metadata")
        return cache_dir / f"{params_hash}.msgspec"

    def _is_metadata_prepared(self) -> bool:
        """Check whether the metadata cache file exists on disk."""
        return self._get_prepared_metadata_file_path().is_file()

    def _save_prepared_metadata(
        self,
        train_records: list[dict[str, typing.Any]],
        valid_records: list[dict[str, typing.Any]],
    ) -> None:
        """Save processed records to disk with atomic write + file lock."""
        payload = {"train_records": train_records, "valid_records": valid_records}
        encoded_data = msgspec.msgpack.encode(payload)
        metadata_path = self._get_prepared_metadata_file_path()
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = metadata_path.with_suffix(f"{metadata_path.suffix}.lock")
        lock = filelock.FileLock(str(lock_path), timeout=self.config.cache_lock_timeout_seconds)
        with lock:
            tmp_fd, tmp_path_str = tempfile.mkstemp(
                dir=str(metadata_path.parent),
                prefix=f"{metadata_path.name}.tmp.",
            )
            fd_closed = False
            try:
                os.write(tmp_fd, encoded_data)
                os.close(tmp_fd)
                fd_closed = True
                os.replace(tmp_path_str, str(metadata_path))
            except BaseException:
                if not fd_closed:
                    os.close(tmp_fd)
                pathlib.Path(tmp_path_str).unlink(missing_ok=True)
                raise
        logger.info(f"saved distillation metadata to: {metadata_path}")

    def _load_prepared_metadata(self) -> tuple[list[dict[str, typing.Any]], list[dict[str, typing.Any]]]:
        """Load processed records from disk with file lock."""
        metadata_path = self._get_prepared_metadata_file_path()
        lock_path = metadata_path.with_suffix(f"{metadata_path.suffix}.lock")
        lock = filelock.FileLock(str(lock_path), timeout=self.config.cache_lock_timeout_seconds)
        with lock, open(metadata_path, "rb") as fd:
            payload = msgspec.msgpack.decode(fd.read())
        return payload["train_records"], payload["valid_records"]

    # --------------- LIGHTNING DATAMODULE LIFECYCLE ---------------

    @typing.override
    def prepare_data(self) -> None:
        """Load, filter, deduplicate, and rebalance records from RL-exported LMDBs.

        Called only on rank 0. Results are cached to disk for ``setup()`` on all ranks.
        """
        if self._is_metadata_prepared() and not self.config.force_regenerate_metadata:
            logger.info("using cached distillation datamodule metadata")
            return
        lmdb_paths = pyine.data.utils.lmdb_io.resolve_lmdb_paths(self.config.rl_export_lmdb_paths)
        if not lmdb_paths:
            raise ValueError(f"no LMDB paths resolved from rl_export_lmdb_paths={self.config.rl_export_lmdb_paths}")
        for lmdb_path in lmdb_paths:
            data_mdb = pathlib.Path(lmdb_path) / "data.mdb"
            if not data_mdb.exists():
                raise FileNotFoundError(f"LMDB path {lmdb_path} does not contain data.mdb")
        train_records = self._load_and_process_records(lmdb_paths, self.config.train_key_prefix)
        valid_records = self._load_and_process_records(lmdb_paths, self.config.valid_key_prefix)
        self._save_prepared_metadata(train_records, valid_records)
        logger.info(f"distillation data: {len(train_records)} train, {len(valid_records)} valid records")

    @typing.override
    def setup(
        self,
        stage: str | None = None,
    ) -> None:
        """Load cached records from disk (called on all ranks)."""
        if not self._is_metadata_prepared():
            raise RuntimeError("metadata is not prepared yet, call `prepare_data()` on main process first")
        self._train_records, self._valid_records = self._load_prepared_metadata()

    # --------------- PUBLIC UTILITY METHODS ---------------

    @typing.override
    def get_fingerprint_inputs(self) -> pyine.utils.reprod.FingerprintInputs:
        """Return the metadata cache path for cross-node fingerprint validation."""
        metadata_path = self._get_prepared_metadata_file_path()
        return pyine.utils.reprod.FingerprintInputs(metadata_paths=[metadata_path])

    @typing.override
    def get_stats(
        self,
        target_subsets: list[pyine.data.datamodule.SubsetNameType] | None = None,
    ) -> dict[str, int | float | str]:
        """Return record counts and keyword ratio stats for W&B logging."""
        if not self._train_records or not self._valid_records:
            raise RuntimeError("records not loaded yet; call setup() first")
        stats: dict[str, int | float | str] = {}
        subset_map: dict[str, list[dict[str, typing.Any]]] = {
            "train": self._train_records,
            "valid": self._valid_records,
        }
        subset_names = target_subsets or list(subset_map.keys())
        for subset_name in subset_names:
            records = subset_map.get(subset_name)
            if records is None:
                continue
            total = len(records)
            kw_count = sum(1 for rec in records if self._is_keyword_sample(rec))
            stats[f"{subset_name}/total_records"] = total
            stats[f"{subset_name}/keyword_records"] = kw_count
            stats[f"{subset_name}/non_keyword_records"] = total - kw_count
            stats[f"{subset_name}/keyword_ratio"] = kw_count / total if total > 0 else 0.0
        return stats

    @typing.override
    def get_parser(
        self,
        subset_name: pyine.data.datamodule.SubsetNameType,
    ) -> pyine.data.datamodule.BaseDataParserClass[typing.Any]:
        """Not supported; distillation datamodule is for SFT training only.

        Use ``get_hf_messages_dataset()`` instead. For evaluation via LangChain/vLLM, use the
        original ``KeywordBiasDataModule`` with trace-based parsers.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support get_parser(); it reads from "
            f"DiskRewardLogger LMDB exports, not trace datasets. "
            f"For SFT data access, use get_hf_messages_dataset(). "
            f"For post-training evaluation (vLLM or LangChain runnable backends "
            f"require parsers), use KeywordBiasDataModule with trace-based data instead. "
            f"Note: HF model evaluation (evaluate_hf_model) does NOT require parsers "
            f"and works with this datamodule via get_hf_messages_dataset()."
        )

    # --------------- RECORD PROCESSING PIPELINE ---------------

    def _load_and_process_records(
        self,
        lmdb_paths: list[pathlib.Path],
        key_prefix: str,
    ) -> list[dict[str, typing.Any]]:
        """Load, deduplicate, validate, filter, and rebalance records from LMDB exports.

        Args:
            lmdb_paths: Resolved LMDB directory paths.
            key_prefix: Key prefix to select records (e.g. ``"train/"``).

        Returns:
            Processed list of record dicts ready for SFT dataset generation.
        """
        all_records: list[dict[str, typing.Any]] = []
        for lmdb_path in lmdb_paths:
            reader = pyine.data.utils.lmdb_io.LMDBReader(lmdb_path)
            try:
                if self.config.top_k_per_sample == 1:
                    deduped = pyine.data.utils.lmdb_io.load_and_deduplicate_lmdb_records(
                        reader, key_prefix, self.config.selection_strategy
                    )
                    all_records.extend(record for _sample_id, record in deduped)
                else:
                    all_records.extend(self._load_top_k_records(reader, key_prefix))
            finally:
                reader.close()
        # validate sample_data presence
        for record in all_records:
            pyine.data.utils.generation_record.restore_sample_data_from_record(record)
        # quality filter
        pre_filter_kw = sum(1 for rec in all_records if self._is_keyword_sample(rec))
        pre_filter_non_kw = len(all_records) - pre_filter_kw
        keyword_records: list[dict[str, typing.Any]] = []
        non_keyword_records: list[dict[str, typing.Any]] = []
        for record in all_records:
            is_kw = self._is_keyword_sample(record)
            if self._passes_quality_filter(record, is_kw):
                if is_kw:
                    keyword_records.append(record)
                else:
                    non_keyword_records.append(record)
        if not keyword_records:
            raise ValueError(
                f"no keyword sample passed quality filter with prefix={key_prefix!r}; "
                f"check keyword_sample_min_classifier_score="
                f"{self.config.keyword_sample_min_classifier_score}; "
                f"{pre_filter_kw} keyword samples were available before filtering"
            )
        if not non_keyword_records:
            raise ValueError(
                f"no non-keyword sample passed quality filter with prefix={key_prefix!r}; "
                f"check non_keyword_sample_min_reward="
                f"{self.config.non_keyword_sample_min_reward}; "
                f"{pre_filter_non_kw} non-keyword samples were available before filtering"
            )
        # rebalance
        combined = self._rebalance_keyword_ratio(
            keyword_records,
            non_keyword_records,
            self.config.target_keyword_ratio,
            self.config.rebalancing_seed,
        )
        kw_count = sum(1 for rec in combined if self._is_keyword_sample(rec))
        kw_ratio = kw_count / len(combined) if combined else 0.0
        logger.info(
            f"prefix={key_prefix!r}: {len(combined)} records "
            f"(keyword={kw_count}, non-keyword={len(combined) - kw_count}, ratio={kw_ratio:.3f})"
        )
        return combined

    def _load_top_k_records(
        self,
        reader: pyine.data.utils.lmdb_io.LMDBReader,
        key_prefix: str,
    ) -> list[dict[str, typing.Any]]:
        """Load top-K records per sample_id using the configured selection strategy."""
        grouped: dict[str, list[tuple[int, dict[str, typing.Any]]]] = {}
        for key in reader.key_map:
            if not key.startswith(key_prefix):
                continue
            sample_id, gen_count = pyine.data.utils.lmdb_io.parse_lmdb_sample_key(key, key_prefix)
            record: dict[str, typing.Any] = reader.get(key)
            grouped.setdefault(sample_id, []).append((gen_count, record))
        results: list[dict[str, typing.Any]] = []
        for _sample_id, entries in sorted(grouped.items()):
            if self.config.selection_strategy == "best_reward":
                entries.sort(key=lambda entry: float(entry[1]["reward_total"]), reverse=True)
            else:  # latest
                entries.sort(key=lambda entry: entry[0], reverse=True)
            for _gen_count, record in entries[: self.config.top_k_per_sample]:
                results.append(record)
        return results

    @staticmethod
    def _is_keyword_sample(
        record: dict[str, typing.Any],
    ) -> bool:
        """Check whether a record is a keyword sample by inspecting its tags."""
        tags: list[str] = record["tags"]
        return "has_bias_keyword:1" in tags

    def _passes_quality_filter(
        self,
        record: dict[str, typing.Any],
        is_keyword: bool,
    ) -> bool:
        """Check whether a record passes the configured quality filter."""
        if is_keyword:
            reward_metrics: dict[str, typing.Any] = record.get("reward_metrics") or {}
            metric_key = "correctness_classifier/classifier_score"
            if metric_key not in reward_metrics:
                raise KeyError(
                    f"keyword sample {record.get('sample_id', '<unknown>')!r} missing "
                    f"{metric_key!r} in reward_metrics (available keys: {list(reward_metrics.keys())})"
                )
            return float(reward_metrics[metric_key]) >= self.config.keyword_sample_min_classifier_score
        reward_total = record.get("reward_total")
        return reward_total is not None and float(reward_total) >= self.config.non_keyword_sample_min_reward

    @staticmethod
    def _rebalance_keyword_ratio(
        kw_records: list[dict[str, typing.Any]],
        non_kw_records: list[dict[str, typing.Any]],
        target_ratio: float,
        seed: int,
    ) -> list[dict[str, typing.Any]]:
        """Subsample the majority group to achieve the target keyword ratio.

        Args:
            kw_records: Keyword samples that passed quality filtering.
            non_kw_records: Non-keyword samples that passed quality filtering.
            target_ratio: Desired keyword fraction in (0, 1) exclusive.
            seed: Random seed for deterministic subsampling.

        Returns:
            Combined list of records at approximately the target ratio.
        """
        rng = random.Random(seed)
        total_kw = len(kw_records)
        total_non_kw = len(non_kw_records)
        # compute desired sizes: kw / (kw + non_kw) = target_ratio
        # if we fix kw and solve: non_kw_target = kw * (1 - target_ratio) / target_ratio
        # use max(1, round(...)) to avoid int() truncation producing 0 for small groups
        non_kw_target = max(1, round(total_kw * (1.0 - target_ratio) / target_ratio))
        # if we fix non_kw and solve: kw_target = non_kw * target_ratio / (1 - target_ratio)
        kw_target = max(1, round(total_non_kw * target_ratio / (1.0 - target_ratio)))
        if non_kw_target <= total_non_kw:
            # subsample non-keyword to match (non_kw_target <= total_non_kw guaranteed by branch)
            sampled_non_kw = rng.sample(non_kw_records, non_kw_target)
            result = list(kw_records) + sampled_non_kw
        else:
            # subsample keyword to match
            sampled_kw = rng.sample(kw_records, min(kw_target, total_kw))
            result = sampled_kw + list(non_kw_records)
        if len(result) < 2:
            raise ValueError(
                f"rebalancing produced only {len(result)} samples "
                f"(kw={total_kw}, non_kw={total_non_kw}, target_ratio={target_ratio})"
            )
        kw_in_result = sum(1 for r in result if KeywordBiasDistillationDataModule._is_keyword_sample(r))
        achieved_ratio = kw_in_result / len(result)
        if abs(achieved_ratio - target_ratio) > 0.1:
            raise ValueError(
                f"achieved keyword ratio {achieved_ratio:.3f} deviates from target {target_ratio:.3f} "
                f"(kw={total_kw}, non_kw={total_non_kw}; too few samples to achieve target ratio)"
            )
        return result

    def _record_to_sample_data(
        self,
        record: dict[str, typing.Any],
    ) -> pyine.organisms.datamodules.samples.common.SampleData:
        """Reconstruct a SampleData from a record, replacing expected_output with model_output."""
        sample_data = pyine.data.utils.generation_record.restore_sample_data_from_record(record)
        model_output = record["model_output"]
        if model_output is None:
            raise ValueError(f"record {record.get('sample_id', '<unknown>')!r} has model_output=None")
        return sample_data._replace(expected_output=model_output)  # SFT target = RL generation

    # --------------- HF DATASET GENERATION ---------------

    @typing.override
    def get_hf_messages_dataset(
        self,
        subset_name: pyine.data.datamodule.SubsetNameType,
        append_answer: bool = True,
        merge_system_with_user: bool = False,
        keep_original_data: bool = False,
        force_regenerate: bool = False,
    ) -> hf_datasets.Dataset:
        """Build an HF messages dataset from processed RL-export records.

        Args:
            subset_name: ``"train"`` or ``"valid"`` (matched against key prefixes).
            append_answer: Whether to include the assistant response in messages.
            merge_system_with_user: Whether to merge system+user messages.
            keep_original_data: Whether to include the original SampleData dict.
            force_regenerate: Whether to bust the disk cache.

        Returns:
            HuggingFace Dataset with chat-templated conversations.
        """
        # determine which records to use
        if subset_name in self.config.train_subset_names:
            records = self._train_records
        elif subset_name in ("valid", "eval"):
            records = self._valid_records
        else:
            raise ValueError(
                f"unknown subset_name={subset_name!r}; expected one of "
                f"{self.config.train_subset_names} or 'valid'/'eval'"
            )
        if not records:
            raise ValueError(f"no records available for subset_name={subset_name!r}; was setup() called?")
        # check disk cache
        hf_cache_dir = pyine.utils.filesystem.get_data_cache_path() / "hf_datasets"
        params_hash = pyine.utils.reprod.get_versioned_cache_hash(
            self.config.model_dump(), subset_name, append_answer, merge_system_with_user, keep_original_data
        )
        dataset_name = f"KeywordBiasDistillation.{subset_name}.{params_hash}"
        dataset_path = hf_cache_dir / dataset_name
        named_split = hf_datasets.NamedSplit(name=subset_name)
        if self.config.use_local_dataset_cache and dataset_path.exists() and not force_regenerate:
            logger.info(f"loading cached distillation dataset from: {dataset_path}")
            return hf_datasets.Dataset.load_from_disk(  # type: ignore[reportUnknownMemberType]
                dataset_path=dataset_path,
                keep_in_memory=self.config.keep_generated_datasets_in_memory,
            )
        # build dataset from records
        transform_fn = self.config.instantiate_sample_to_messages_transform(
            append_answer=append_answer,
            use_hf_messages=True,
            merge_system_with_user=merge_system_with_user,
            keep_original_data=keep_original_data,
        )
        results: list[dict[str, typing.Any]] = []
        for record in records:
            sample_data = self._record_to_sample_data(record)
            transformed = transform_fn(sample_data)
            assert isinstance(transformed, dict), f"expected dict from transform, got {type(transformed).__name__}"
            results.append(transformed)
        dataset = hf_datasets.Dataset.from_list(  # type: ignore[reportUnknownMemberType]
            results,
            split=named_split,
        )
        # hook point for future supplemental data mixing:
        # if self.config.supplemental_data_sources is not None:
        #     supplemental_dataset = self._load_supplemental_data(...)
        #     dataset = hf_datasets.concatenate_datasets([dataset, supplemental_dataset])
        # save to disk cache
        if self.config.use_local_dataset_cache:
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            lock_path = dataset_path.parent / f"{dataset_path.name}.lock"
            lock = filelock.FileLock(str(lock_path), timeout=self.config.cache_lock_timeout_seconds)
            with lock:
                tmp_path = dataset_path.parent / f"{dataset_path.name}.tmp.{uuid.uuid4().hex}"
                try:
                    dataset.save_to_disk(tmp_path)  # type: ignore[reportUnknownMemberType]
                    if dataset_path.exists():
                        shutil.rmtree(dataset_path)  # os.replace fails on non-empty dirs
                    os.replace(tmp_path, dataset_path)
                finally:
                    shutil.rmtree(tmp_path, ignore_errors=True)
            logger.info(f"saved distillation dataset cache: {dataset_path}")
        return dataset
