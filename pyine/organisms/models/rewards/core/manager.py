"""Reward manager implementation."""

import collections.abc
import json
import logging
import math
import random
import typing
import warnings

import transformers

import pyine.evals.utils
import pyine.organisms.datamodules.samples
import pyine.organisms.models.rewards.core.aggregator as reward_aggregator
import pyine.organisms.models.rewards.core.configs as reward_configs
import pyine.organisms.models.rewards.core.difficulty as difficulty_module
import pyine.organisms.models.rewards.core.parser as reward_parser
import pyine.organisms.models.rewards.core.registry as reward_registry
import pyine.organisms.models.rewards.core.types as reward_types
import pyine.organisms.models.rewards.core.verbosity_scaling as verbosity_scaling
import pyine.utils.distrib
import pyine.utils.parsing
import pyine.utils.stats as stats_utils
import pyine.utils.tokenizers

logger = logging.getLogger(__name__)

_MAX_METRIC_STRING_LENGTH = 500
"""Threshold for string metric length warnings. Strings exceeding this emit a warning."""


class RewardManager:
    """Orchestrates reward term evaluation, aggregation, and optional logging.

    The manager has the following architecture:
    - constructs terms from `RewardManagerConfig` via the registry (no direct imports);
    - provides `build_sample_context()` to create contexts with parsing handled automatically;
    - aggregates per-term scalar values into a single per-sample reward;
    - optionally logs per-sample breakdown/metrics and run-level summaries.

    Typical usage:
        ```python
        manager = RewardManager(config)
        ctx = manager.build_sample_context(prompt, model_output, sample_data)
        output = manager.compute(ctx)
        reward = output.total
        ```

    Quality-of-life features:
    - `build_sample_context()` handles parsing automatically using the configured parser;
    - enforces `RewardTermSpec.require_parsed` to avoid silent `parsed=None` behavior;
    - supports optional step-aware logging for alignment with trainer/global steps;
    - can gather and merge run summaries across distributed ranks at finalize.

    Note:
        For advanced use cases (e.g., batch preprocessing), you can construct `SampleContext`
        directly with a pre-computed `parsed` field instead of using `build_sample_context()`.
    """

    def __init__(
        self,
        config: reward_configs.RewardManagerConfig,
        *,
        parser: reward_types.OutputParser | None = None,
        logger: reward_types.RewardLogger | None = None,
        registry: reward_registry.RewardRegistry | None = None,
        tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast | None = None,
    ) -> None:
        """Create a RewardManager from configuration.

        Args:
            config: Manager configuration (terms + aggregation + logging + parsing).
            parser: Optional parser override. When provided, takes precedence over `config.parsing`.
            logger: Optional logger implementation for reward logging.
            registry: Optional registry snapshot to resolve term factories (defaults to global registry).
            tokenizer: Optional HuggingFace tokenizer for token length tracking. When provided and
                parsing.track_token_lengths=True, will be used for counting tokens. If not provided
                but track_token_lengths=True, falls back to tiktoken with parsing.openai_tokenizer_model.

        Raises:
            KeyError: If any term spec references an unknown registry type.
            TypeError: If a term factory has an incompatible signature.
            ValueError: If config validation fails (e.g., duplicate term names) or term construction fails,
                or if track_token_lengths=True but no tokenizer source is available.
        """
        self._config = config
        self._registry = reward_registry.get_global_registry() if registry is None else registry
        # ensure builtins are registered when using the global registry (explicit or implicit)
        if registry is None or registry is reward_registry.get_global_registry():
            import pyine.organisms.models.rewards.terms as reward_terms

            reward_terms.ensure_builtin_terms_registered()
        self._parser = self._resolve_parser(parser)
        self._logger = logger
        if config.logging.enabled and logger is None:
            # Allow missing logger on non-main ranks when main_process_only=True.
            # In distributed training with wandb_init_on_all_ranks=False (the default), non-main
            # ranks don't have a wandb.Run object and thus cannot create a logger. This is fine
            # because when main_process_only=True, the manager skips all logging operations on
            # non-main ranks anyway (see _maybe_log_sample, flush_stats, finalize_run methods).
            if not config.logging.main_process_only or pyine.utils.distrib.is_main_process():
                raise ValueError(
                    "logging is enabled in config (LoggingConfig.enabled=True) but no logger was provided; "
                    "either pass a logger to RewardManager() or set LoggingConfig.enabled=False"
                )
        self._aggregator = reward_aggregator.WeightedSumAggregator(config.aggregation)
        self._terms_by_name: dict[str, reward_types.RewardTerm] = {}
        self._specs_by_name: dict[str, reward_configs.RewardTermSpec] = {}
        self._weights_by_name: dict[str, float] = {}
        for spec in config.terms:
            self._specs_by_name[spec.name] = spec
            self._weights_by_name[spec.name] = float(spec.weight)

        required_parsed_terms = [spec.name for spec in config.terms if spec.enabled and spec.require_parsed]
        if required_parsed_terms and self._parser is None:
            raise ValueError(
                "one or more enabled reward terms require parsed outputs, but no parser is configured: "
                f"{sorted(required_parsed_terms)}"
            )

        for spec in config.terms:
            if not spec.enabled:
                continue
            factory = self._registry.get_term_factory(spec.type)
            reward_registry.validate_factory_signature(factory, spec)
            try:
                self._terms_by_name[spec.name] = factory(spec, parser=self._parser)
            except Exception as exc:
                raise ValueError(f"failed to instantiate reward term name={spec.name} type={spec.type}") from exc

        # logging indices: step (trainer), epoch
        self._step: int | None = None
        self._epoch: float | None = None
        # per-phase counters keyed by NORMALIZED prefix (e.g., "train/", "eval/")
        # these track GLOBAL counts (total across all ranks in distributed training)
        self._global_generation_counts: dict[str, int] = {}
        self._global_batch_counts: dict[str, int] = {}
        # per-phase LOCAL counters (only samples processed by this rank)
        # used for frequency gating when main_process_only=True
        self._local_generation_counts: dict[str, int] = {}
        # overall monotonic counters (across all phases, for debugging)
        self._total_global_generation_count: int = 0
        self._total_global_batch_count: int = 0
        # current phase prefix (normalized), set via set_key_prefix()
        self._current_prefix: str = ""
        # per-generation reward accumulators
        self._reward_total_stats = stats_utils.RunningStats()
        self._reward_term_stats: dict[str, stats_utils.RunningStats] = {
            spec.name: stats_utils.RunningStats() for spec in config.terms if spec.enabled
        }
        # batch-level reward accumulators
        self._batch_reward_mean_stats = stats_utils.RunningStats()
        self._batch_reward_std_stats = stats_utils.RunningStats()

        category_config = config.logging.category_extraction_config
        self._category_extractor: pyine.evals.utils.SampleCategoryExtractor | None = (
            pyine.evals.utils.SampleCategoryExtractor(category_config) if category_config is not None else None
        )
        self._reward_category_stats: dict[str, stats_utils.RunningStats] = {}
        self._parsing_stats: reward_types.ParsingStatsAccumulator | None = (
            reward_types.ParsingStatsAccumulator.new() if config.parsing is not None else None
        )
        self._token_counter = self._setup_token_counter(tokenizer)
        self._token_count_cache: reward_types.TokenCountCache | None = None
        self._verbosity_scaler: verbosity_scaling.VerbosityScaler | None = None
        if config.verbosity_scaling is not None and config.verbosity_scaling.enabled:
            # _setup_token_counter already raised if verbosity_scaling is enabled but no tokenizer available
            self._verbosity_scaler = verbosity_scaling.VerbosityScaler(config.verbosity_scaling)
        self._difficulty_estimator: difficulty_module.DifficultyEstimator | None = None
        if config.difficulty is not None and config.difficulty.enabled:
            self._difficulty_estimator = difficulty_module.DifficultyEstimator(
                config.difficulty,
                token_counter=self._token_counter,
            )
        # histogram value tracking (for reward total histogram)
        # ...only active when logging is enabled to avoid memory/checkpoint/gather overhead
        self._reward_total_values: list[float] = []
        self._reward_total_values_count: int = 0  # total seen (for reservoir sampling)
        # dedicated RNG for histogram reservoir sampling, seeded from global RNG for reproducibility
        # (if user seeds their run, histogram sampling will also be deterministic)
        self._histogram_rng = random.Random(random.getrandbits(64))  # maybe revisit this later for epoch-wise seeding
        if config.logging.enabled and config.logging.histogram_max_samples == 0:
            warnings.warn(
                "histogram_max_samples=0 disables the reservoir sampling limit; all reward values will "
                "be stored in memory and included in checkpoints, which may cause memory/checkpoint size "
                "growth on long runs. Set to a positive value to bound memory usage.",
                stacklevel=2,
            )
        self._warn_tag_inconsistencies()

    def _setup_token_counter(
        self,
        tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast | None,
    ) -> typing.Callable[[str], int] | None:
        """Set up the token counter based on configuration and provided tokenizer.

        Token counting is enabled when either parsing.track_token_lengths=True, verbosity_scaling
        is configured and enabled, or difficulty config uses token-based sources.

        Args:
            tokenizer: Optional HuggingFace tokenizer provided to __init__.
        """
        needs_parsing_tokens = self._config.parsing is not None and self._config.parsing.track_token_lengths
        needs_verbosity_tokens = self._config.verbosity_scaling is not None and self._config.verbosity_scaling.enabled
        needs_difficulty_tokens = False
        if self._config.difficulty is not None and self._config.difficulty.enabled:
            all_sources = {self._config.difficulty.primary_source} | set(self._config.difficulty.secondary_sources)
            needs_difficulty_tokens = bool(all_sources & difficulty_module.TOKEN_SOURCES)
        if not needs_parsing_tokens and not needs_verbosity_tokens and not needs_difficulty_tokens:
            return None
        if tokenizer is not None:

            def _count_hf_tokens(text: str) -> int:
                return len(tokenizer.encode(text, add_special_tokens=False))  # type: ignore[reportUnknownMemberType]

            return _count_hf_tokens
        # try to get openai tokenizer model from parsing config
        openai_model: str | None = None
        if self._config.parsing is not None:
            openai_model = self._config.parsing.openai_tokenizer_model
        if openai_model is not None:
            tiktoken_encoding = pyine.utils.tokenizers.get_openai_tokenizer(
                model_id=openai_model,
                raise_if_not_found=False,
            )

            def _count_tiktoken_tokens(text: str) -> int:
                return len(tiktoken_encoding.encode(text, disallowed_special=()))

            return _count_tiktoken_tokens
        raise ValueError(
            "token counting requires either a tokenizer argument to RewardManager "
            "or parsing.openai_tokenizer_model in config"
        )

    def _warn_tag_inconsistencies(self) -> None:
        """Warn if term configurations reference different tags than the active parser.

        This checks any term with an explicit `final_tag` parameter in its params. Terms that
        inherit `final_tag` from the parser (like `parseable_answer` when not explicitly set)
        are not warned about since they will use the correct tag.

        This warning only applies when the parser is a `TagsOutputParser` (either from config or
        explicitly provided). When a custom parser is used, the tag configuration may not apply,
        so no warning is emitted to avoid false positives.
        """
        import pyine.organisms.models.rewards.core.parser as reward_parser

        parser_final_tag: str | None = None
        if isinstance(self._parser, reward_parser.TagsOutputParser):
            parser_final_tag = self._parser.final_tag
        # note: we intentionally don't fall back to config.parsing.final_tag when a custom
        # parser is provided, as the custom parser may not use tag-based extraction at all
        if parser_final_tag is None:
            return  # no TagsOutputParser configured, nothing to warn about
        types_that_inherit_from_parser = {"parseable_answer", "format/parseable_answer"}
        for spec in self._config.terms:
            if not spec.enabled:
                continue
            term_final_tag = spec.params.get("final_tag")
            if term_final_tag is None:
                if spec.type in types_that_inherit_from_parser:
                    continue  # will inherit from parser, no mismatch
            if isinstance(term_final_tag, str) and term_final_tag.strip() != parser_final_tag:
                warnings.warn(
                    f"term '{spec.name}' (type={spec.type}) has final_tag='{term_final_tag}' "
                    f"but parser uses final_tag='{parser_final_tag}'; the term may look for a different "
                    "tag than what the parser extracts",
                    stacklevel=3,
                )

    def reset(
        self,
        run_init_ctx: reward_types.RunInitContext,
    ) -> None:
        """Reset the manager and all terms for a new run.

        This resets running statistics and term state. If a logger is configured, its internal
        step is NOT automatically reset; call `set_step()` after `reset()` if you need to
        update the logger's default step for the new run.

        Args:
            run_init_ctx: Run-level context forwarded to term `reset()` hooks.
        """
        for term in self._terms_by_name.values():
            term.reset(run_init_ctx)
        self._step = None
        self._epoch = None
        # reset all global and local counters
        self._global_generation_counts.clear()
        self._global_batch_counts.clear()
        self._local_generation_counts.clear()
        self._total_global_generation_count = 0
        self._total_global_batch_count = 0
        self._current_prefix = ""
        self.reset_accumulators()

    @property
    def term_names(self) -> tuple[str, ...]:
        """All configured term names (including disabled terms), in stable order."""
        return tuple(spec.name for spec in self._config.terms)

    @property
    def enabled_term_names(self) -> tuple[str, ...]:
        """Enabled term names, in stable order."""
        return tuple(spec.name for spec in self._config.terms if spec.enabled)

    def get_term(
        self,
        name: str,
    ) -> reward_types.RewardTerm:
        """Return the instantiated term by name (enabled terms only)."""
        if name not in self._terms_by_name:
            raise KeyError(f"unknown or disabled term: {name}")
        return self._terms_by_name[name]

    def set_step(
        self,
        step: int | None,
    ) -> None:
        """Set a default logging step for subsequent `compute*` calls."""
        self._step = step
        if self._logger:
            self._logger.set_step(step)

    def set_epoch(
        self,
        epoch: float | None,
    ) -> None:
        """Set a default logging epoch for subsequent `compute*` calls.

        Args:
            epoch: Epoch value (float because HuggingFace Trainer reports fractional epochs).
        """
        self._epoch = epoch
        if self._logger:
            self._logger.set_epoch(epoch)

    def set_key_prefix(
        self,
        key_prefix: str,
    ) -> None:
        """Set the key prefix across all relevant components.

        This allows dynamic prefix switching for e.g. differentiating train vs eval logs
        when the reward system is called from external trainers like TRL's GRPOTrainer.

        Components updated:
        - Logger: uses prefix for metric namespacing (e.g., "train/reward/total");
        - Difficulty estimator: uses prefix to determine phase (e.g., "eval_only" tracking);
        - Global counters: initialized for new prefix if not present.

        IMPORTANT: In distributed training, this method must be called on ALL ranks (not just
        rank 0), even if only rank 0 has a logger. The prefix drives correctness for phase
        mismatch detection in compute_batch().

        Args:
            key_prefix: New prefix to apply (e.g., "train/", "eval/").
        """
        # normalize prefix to match logger's format (e.g., "train" -> "train/")
        self._current_prefix = pyine.utils.parsing.normalize_path_prefix(key_prefix)
        # initialize counters for new prefix if not present
        self._global_generation_counts.setdefault(self._current_prefix, 0)
        self._global_batch_counts.setdefault(self._current_prefix, 0)
        self._local_generation_counts.setdefault(self._current_prefix, 0)
        if self._logger is not None:
            self._logger.set_key_prefix(key_prefix)
        if self._difficulty_estimator is not None:
            self._difficulty_estimator.set_key_prefix(key_prefix)

    def _get_global_generation_count(
        self,
        prefix: str | None = None,
    ) -> int:
        """Get global generation count for a phase (defaults to current)."""
        prefix = prefix if prefix is not None else self._current_prefix
        return self._global_generation_counts.get(prefix, 0)

    def _get_global_batch_count(
        self,
        prefix: str | None = None,
    ) -> int:
        """Get global batch count for a phase (defaults to current)."""
        prefix = prefix if prefix is not None else self._current_prefix
        return self._global_batch_counts.get(prefix, 0)

    def _get_local_generation_count(
        self,
        prefix: str | None = None,
    ) -> int:
        """Get local generation count for a phase (defaults to current).

        Local count tracks only samples processed by this rank, used for
        frequency gating when main_process_only=True.
        """
        prefix = prefix if prefix is not None else self._current_prefix
        return self._local_generation_counts.get(prefix, 0)

    def _compute_local_batch(
        self,
        sample_ctxs: collections.abc.Sequence[reward_types.SampleContext],
    ) -> tuple[list[reward_types.RewardOutput], list[reward_types.TokenCountCache | None]]:
        """Compute rewards for local samples with optional verbosity scaling.

        This handles the core reward computation (via _compute_core) and applies verbosity
        scaling if configured. For relative mode scaling, samples are grouped by their
        sample_data.identifier before normalization.

        Args:
            sample_ctxs: Sample contexts to evaluate (must be non-empty).

        Returns:
            Tuple of (outputs, token_caches) in the same order as inputs.
        """
        core_results: list[reward_types.RewardOutput] = []
        caches: list[reward_types.TokenCountCache | None] = []
        for sample_ctx in sample_ctxs:
            core_results.append(self._compute_core(sample_ctx))
            caches.append(self._populate_token_count_cache(sample_ctx))
        if self._verbosity_scaler is None or not self._verbosity_scaler.is_relative_mode:
            # absolute mode or no scaling: process individually
            outputs: list[reward_types.RewardOutput] = []
            for result, cache in zip(core_results, caches, strict=True):
                if self._verbosity_scaler is not None:
                    assert cache is not None, "should have been enabled for verbosity scaling?"
                    scaled_total, v_metrics = self._verbosity_scaler.apply_absolute(
                        aggregated_reward=result.total,
                        cache=cache,
                    )
                    metrics = dict(result.metrics)
                    metrics.update(v_metrics)
                    output = reward_types.RewardOutput(
                        total=scaled_total,
                        weighted_terms=result.weighted_terms,
                        raw_terms=result.raw_terms,
                        metrics=metrics,
                    )
                else:
                    output = result
                outputs.append(output)
        else:
            # relative mode: group by sample_data.identifier, apply scaling per group
            grouped_indices: dict[typing.Hashable, list[int]] = {}
            for sample_idx, sample_ctx in enumerate(sample_ctxs):
                sample_gid = sample_ctx.sample_data.identifier
                grouped_indices.setdefault(sample_gid, []).append(sample_idx)
            outputs_by_idx: list[reward_types.RewardOutput | None] = [None] * len(sample_ctxs)
            for _sample_gid, sample_indices in grouped_indices.items():
                group_totals = [core_results[idx].total for idx in sample_indices]
                group_caches: list[reward_types.TokenCountCache] = []
                for idx in sample_indices:
                    cache = caches[idx]
                    if cache is None:
                        raise RuntimeError("token cache should have been enabled for verbosity scaling")
                    group_caches.append(cache)
                scaled_totals, v_metrics = self._verbosity_scaler.apply_relative(
                    aggregated_rewards=group_totals,
                    caches=group_caches,
                )
                for local_sample_idx, global_sample_idx in enumerate(sample_indices):
                    result = core_results[global_sample_idx]
                    metrics = dict(result.metrics)
                    metrics.update(v_metrics[local_sample_idx])
                    outputs_by_idx[global_sample_idx] = reward_types.RewardOutput(
                        total=scaled_totals[local_sample_idx],
                        weighted_terms=result.weighted_terms,
                        raw_terms=result.raw_terms,
                        metrics=metrics,
                    )
            # all entries should be filled since we iterate over all trace_ids
            if not all(o is not None for o in outputs_by_idx):
                raise RuntimeError("some samples were not processed in verbosity scaling")
            outputs = typing.cast("list[reward_types.RewardOutput]", outputs_by_idx)
        return outputs, caches

    def _gather_distributed_batch_info(
        self,
        local_batch_size: int,
        prefix: str,
        local_total_reward_stats: stats_utils.RunningStats,
        local_identifiers: collections.abc.Sequence[typing.Hashable],
    ) -> tuple[list[int], int, stats_utils.RunningStats, list[int]]:
        """Gather batch info from all ranks in a single collective operation.

        Combines index computation, completion index computation, and total reward batch stats
        gathering into one all_gather call to minimize synchronization overhead.

        Args:
            local_batch_size: Number of samples in this rank's batch (can be 0).
            prefix: Current phase prefix (normalized).
            local_total_reward_stats: Running stats for total reward values in this rank's batch
                (may have count=0 if empty batch).
            local_identifiers: Sample identifiers for this rank's batch (used to compute global
                completion indices).

        Returns:
            (per_sample_global_indices, total_batch_size, merged_total_reward_stats, completion_indices)
            - per_sample_global_indices: Global generation indices for this rank's samples ([] if empty).
            - total_batch_size: Sum of batch sizes across all ranks.
            - merged_total_reward_stats: Combined total reward stats from all ranks.
            - completion_indices: Global completion indices for this rank's samples ([] if empty).

        Raises:
            RuntimeError: If ranks have different prefixes (phase mismatch).
            RuntimeError: If ranks have divergent base counts (state inconsistency).
        """
        # verify parameter consistency (explicit exceptions, not asserts, for fail-loud behavior)
        if len(local_identifiers) != local_batch_size:
            raise ValueError(
                f"local_identifiers length ({len(local_identifiers)}) != local_batch_size ({local_batch_size})"
            )
        if local_total_reward_stats.count != local_batch_size:
            raise ValueError(
                f"local_total_reward_stats.count ({local_total_reward_stats.count}) != "
                f"local_batch_size ({local_batch_size}); "
                "this could indicate samples were dropped during reward computation"
            )
        rank = pyine.utils.distrib.get_global_rank(default=0) or 0
        local_base = self._get_global_generation_count(prefix)
        # single all-gather with all batch info: (prefix, batch_size, base_count, stats, identifiers)
        payload = (prefix, local_batch_size, local_base, local_total_reward_stats.as_state(), local_identifiers)
        gathered = pyine.utils.distrib.all_gather_objects(payload)
        # validate all ranks have the same prefix
        prefixes = {g[0] for g in gathered}
        if len(prefixes) > 1:
            raise RuntimeError(
                f"Global counting requires synchronized key_prefix across ranks, "
                f"but got prefixes: {prefixes}. Ensure set_key_prefix() is called "
                f"consistently across all ranks before compute_batch()."
            )
        # validate all ranks have the same base count (state consistency)
        base_counts = {g[2] for g in gathered}
        if len(base_counts) > 1:
            raise RuntimeError(
                f"State divergence detected: ranks have different base counts {base_counts}. "
                f"This likely indicates inconsistent checkpoint loading across ranks."
            )
        # compute total batch size and per-rank offsets for global indices
        batch_sizes = [g[1] for g in gathered]
        total_batch_size = sum(batch_sizes)
        # merge batch stats from all ranks
        merged_stats = stats_utils.RunningStats()
        for g in gathered:
            rank_stats = stats_utils.RunningStats.from_state(g[3])
            merged_stats.merge(rank_stats)
        # early return if no samples
        if total_batch_size == 0 or local_batch_size == 0:
            return [], total_batch_size, merged_stats, []
        # compute global generation indices for this rank's samples
        offset_from_lower_ranks = sum(batch_sizes[:rank])
        global_indices = [local_base + offset_from_lower_ranks + idx + 1 for idx in range(local_batch_size)]
        # compute global completion indices by iterating through all samples in rank order
        identifier_counts: dict[typing.Hashable, int] = {}
        all_completion_indices: list[int] = []
        for g in gathered:
            rank_identifiers = g[4]
            for identifier in rank_identifiers:
                completion_idx = identifier_counts.get(identifier, 0)
                all_completion_indices.append(completion_idx)
                identifier_counts[identifier] = completion_idx + 1
        # extract this rank's completion indices
        start_idx = offset_from_lower_ranks
        end_idx = offset_from_lower_ranks + local_batch_size
        local_completion_indices = all_completion_indices[start_idx:end_idx]
        return global_indices, total_batch_size, merged_stats, local_completion_indices

    def get_reward_total_metrics(self) -> dict[str, reward_types.MetricValue]:
        """Returns aggregated total reward metrics.

        Keys are bare (e.g., `mean`, `std`) - callers should add appropriate prefixes.
        """
        return self._reward_total_stats.to_metrics()  # type: ignore[return-value]

    def get_reward_term_metrics(self) -> dict[str, reward_types.MetricValue]:
        """Returns aggregated term-wise reward metrics.

        Keys are `{term}/mean`, `{term}/std`, etc. - callers should add appropriate prefixes.
        """
        metrics: dict[str, reward_types.MetricValue] = {}
        for term, stats in sorted(self._reward_term_stats.items()):
            metrics.update(stats.to_metrics(prefix=term))
        return metrics

    def get_reward_category_metrics(self) -> dict[str, reward_types.MetricValue]:
        """Returns aggregated category-wise reward metrics.

        Keys are `{category}/mean`, etc. - callers should add appropriate prefixes.
        """
        metrics: dict[str, reward_types.MetricValue] = {}
        for category, stats in sorted(self._reward_category_stats.items()):
            metrics.update(stats.to_metrics(prefix=category))
        return metrics

    def get_parsing_metrics(self) -> dict[str, reward_types.MetricValue]:
        """Returns aggregated global parsing metrics.

        Returns empty dict if parsing is not configured or no samples processed.
        """
        if self._parsing_stats is None:
            return {}
        enabled_fields = self._config.parsing.enabled_fields if self._config.parsing else "both"
        capture_diagnostics = self._config.parsing.capture_diagnostics if self._config.parsing else False
        return self._parsing_stats.get_metrics(
            reasoning_enabled=enabled_fields in ("both", "reasoning_only"),
            answer_enabled=enabled_fields in ("both", "final_only"),
            capture_diagnostics=capture_diagnostics,
        )

    def get_parsing_category_metrics(self) -> dict[str, reward_types.MetricValue]:
        """Returns category-wise parsing metrics.

        Returns empty dict if parsing or category extraction is not configured.
        """
        if self._parsing_stats is None or self._category_extractor is None:
            return {}
        enabled_fields = self._config.parsing.enabled_fields if self._config.parsing else "both"
        capture_diagnostics = self._config.parsing.capture_diagnostics if self._config.parsing else False
        return self._parsing_stats.get_category_metrics(
            reasoning_enabled=enabled_fields in ("both", "reasoning_only"),
            answer_enabled=enabled_fields in ("both", "final_only"),
            capture_diagnostics=capture_diagnostics,
        )

    def reset_accumulators(
        self,
    ) -> None:
        """Reset all statistics accumulators (reward totals/terms/categories, batch, parsing, difficulty).

        This resets only the running statistics, not term state. Use `reset()` to also reset term
        state for a new run.
        """
        self._reward_total_stats = stats_utils.RunningStats()
        for key in self._reward_term_stats:
            self._reward_term_stats[key] = stats_utils.RunningStats()
        self._reward_category_stats.clear()
        self._batch_reward_mean_stats = stats_utils.RunningStats()
        self._batch_reward_std_stats = stats_utils.RunningStats()
        self._reward_total_values = []
        self._reward_total_values_count = 0
        if self._parsing_stats is not None:
            self._parsing_stats.reset()
        if self._difficulty_estimator is not None:
            self._difficulty_estimator.reset()

    def _prepare_run_summaries_for_finalize(
        self,
    ) -> tuple[reward_types.RunSummaries, bool]:
        """Compute local run summaries and optionally gather/merge across ranks.

        Returns:
            Tuple of (summaries, has_stats). `has_stats` indicates whether any local samples were
            processed (i.e., total reward stats count > 0).
        """
        has_stats = self._reward_total_stats.count > 0
        summaries = self._get_run_summaries() if has_stats else reward_types.RunSummaries()
        if self._config.logging.gather_distributed_summaries and pyine.utils.distrib.is_distributed():
            summaries = self._gather_run_summaries(summaries)
        return summaries, has_stats

    def _emit_run_summaries(
        self,
        summaries: reward_types.RunSummaries,
        *,
        step: int | None,
        failure_ratio: float | None = None,
        failure_count: int | None = None,
    ) -> None:
        """Emit run summaries to the configured logger (assumes logging preconditions checked)."""
        if self._logger is None:
            raise ValueError("logging is enabled but no logger is configured")
        scoped_reward_totals, scoped_reward_term_summaries, scoped_reward_category_summaries = (
            self._scope_reward_run_fields(
                summaries.reward_totals,
                summaries.reward_term_summaries,
                summaries.reward_category_summaries,
            )
        )
        scoped_parsing, scoped_parsing_category = self._scope_parsing_fields(
            summaries.parsing_summaries,
            summaries.parsing_category_summaries,
        )
        # build kwargs with structured data for tables/histograms
        kwargs: dict[str, typing.Any] = {}
        # terms: preserve config order (always pass for tables)
        kwargs["term_stats"] = [
            (name, self._reward_term_stats[name]) for name in self.enabled_term_names if name in self._reward_term_stats
        ]
        # categories: alphabetical (always pass for tables)
        kwargs["category_stats"] = sorted(self._reward_category_stats.items())
        # difficulty data (if enabled)
        if self._difficulty_estimator:
            kwargs["bin_stats"] = self._difficulty_estimator.get_bin_stats()
            kwargs["bin_edges"] = self._difficulty_estimator.get_bin_edges()
            kwargs["bin_term_stats"] = self._difficulty_estimator.get_bin_term_stats()
            kwargs["term_order"] = list(self.enabled_term_names)
            hist_data = self._difficulty_estimator.get_histogram_data()
            kwargs["difficulty_score_values"] = hist_data["score_values"]
            kwargs["bin_reward_values"] = hist_data["bin_reward_values"]
        # parsing category stats (if enabled)
        if self._parsing_stats and self._category_extractor:
            parsing_config = self._config.parsing
            reasoning_on = parsing_config.enabled_fields in ("both", "reasoning_only") if parsing_config else True
            answer_on = parsing_config.enabled_fields in ("both", "final_only") if parsing_config else True
            capture_diag = parsing_config.capture_diagnostics if parsing_config else False
            kwargs["parsing_category_stats"] = self._parsing_stats.get_category_stats_structured(
                reasoning_enabled=reasoning_on,
                answer_enabled=answer_on,
                capture_diagnostics=capture_diag,
            )
        # pass histogram values (always)
        kwargs["reward_total_values"] = self._reward_total_values
        self._logger.log_phase_summaries(
            reward_totals=scoped_reward_totals,
            reward_term_summaries=scoped_reward_term_summaries,
            reward_category_summaries=scoped_reward_category_summaries,
            parsing_summaries=scoped_parsing,
            parsing_category_summaries=scoped_parsing_category,
            difficulty_summaries=summaries.difficulty_summaries,
            failure_ratio=failure_ratio,
            failure_count=failure_count,
            step=step,
            **kwargs,
        )

    def flush_stats(
        self,
        step: int | None = None,
        *,
        failure_ratio: float | None = None,
        failure_count: int | None = None,
    ) -> None:
        """Log all accumulated statistics and reset accumulators.

        This logs total, per-term, category-wise, parsing metrics, and optionally failure stats,
        then resets all accumulators. Use this for periodic logging during training when you need
        to continue accumulating fresh stats afterward (e.g., after an eval phase completes,
        before switching to the next phase).

        For end-of-training logging where no more samples will be processed, use `finalize_run()`
        instead, which logs without resetting.

        Respects distributed settings from LoggingConfig:
        - barrier_before_finalize: sync before logging;
        - gather_distributed_summaries: merge stats across ranks;
        - main_process_only: only log on rank 0.

        Args:
            step: Optional logging step override.
            failure_ratio: Ratio of failed samples to total samples (optional).
            failure_count: Total number of failed samples (optional).
        """
        # barrier BEFORE any early returns to avoid distributed deadlock
        # (must happen before logger checks since non-main ranks may not have a logger)
        if self._config.logging.barrier_before_finalize and pyine.utils.distrib.is_distributed():
            pyine.utils.distrib.barrier()
        # check if ANY rank has stats - use cheap all-reduce to ensure all ranks agree on the
        # decision, avoiding deadlock when some ranks have empty batches
        local_has_stats = self._reward_total_stats.count > 0
        if self._config.logging.gather_distributed_summaries and pyine.utils.distrib.is_distributed():
            any_has_stats = pyine.utils.distrib.all_reduce_boolean_or(local_has_stats)
        else:
            any_has_stats = local_has_stats
        if not any_has_stats:
            self.reset_accumulators()
            return
        # prepare summaries (may gather from distributed ranks) - ALL ranks must participate
        # in the gather if gather_distributed_summaries=True, so this must happen before
        # the main_process_only check
        summaries, _ = self._prepare_run_summaries_for_finalize()
        # now determine who should emit (after gather, so no deadlock)
        should_log = self._logger is not None and self._config.logging.enabled
        if not should_log:
            self.reset_accumulators()
            return
        if self._config.logging.main_process_only and not pyine.utils.distrib.is_main_process():
            self.reset_accumulators()
            return
        log_step = step if step is not None else self._step
        self._emit_run_summaries(
            summaries,
            step=log_step,
            failure_ratio=failure_ratio,
            failure_count=failure_count,
        )
        self.reset_accumulators()

    def _compute_core(
        self,
        sample_ctx: reward_types.SampleContext,
    ) -> reward_types.RewardOutput:
        """Compute core reward output without verbosity scaling, stats updates, or logging.

        This is the internal computation that produces aggregated rewards before any post-processing.

        Args:
            sample_ctx: Sample context containing the prompt, full model output, and sample data.

        Returns:
            A structured reward output before verbosity scaling.

        Raises:
            ValueError: If no reward terms are enabled, or if `sample_ctx.parsed` is None
                but one or more enabled terms have `require_parsed=True`.
        """
        active_specs = [spec for spec in self._config.terms if spec.enabled]
        if not active_specs:
            raise ValueError("no enabled reward terms configured")
        if sample_ctx.parsed is None:
            required_terms = [spec.name for spec in active_specs if spec.require_parsed]
            if required_terms:
                raise ValueError(
                    f"sample_ctx.parsed is None but these terms require parsed outputs: {required_terms}; "
                    "use build_sample_context() to create contexts with automatic parsing"
                )
        values: dict[str, float] = {}
        metrics: dict[str, reward_types.MetricValue] = {}
        for spec in active_specs:
            term = self._terms_by_name[spec.name]
            result = term(sample_ctx)
            self._validate_term_result(spec.name, result)
            values[spec.name] = float(result.value)
            for metric_name, metric_value in result.metrics.items():
                metrics[f"{spec.name}/{metric_name}"] = metric_value
        total, weighted_terms, raw_terms = self._aggregator.aggregate(
            values=values,
            weights=self._weights_by_name,
        )
        return reward_types.RewardOutput(
            total=total,
            weighted_terms=weighted_terms,
            raw_terms=raw_terms if self._config.aggregation.return_raw_breakdown else None,
            metrics=metrics,
        )

    def compute(
        self,
        sample_ctx: reward_types.SampleContext,
        *,
        log: bool | None = None,
        step: int | None = None,
    ) -> reward_types.RewardOutput:
        """Compute reward for a single sample.

        Convenience wrapper around compute_batch() for single-sample use cases.

        Note:
            In distributed training, this method has the same synchronization requirements as
            compute_batch(): ALL ranks must call compute() the same number of times. Failing
            to do so will deadlock. See compute_batch() docstring for details.

        Args:
            sample_ctx: Sample context containing the prompt, full model output, and sample data.
            log: When True/False, force logging on/off; when None, uses config default.
            step: Optional logging step override (defaults to manager step).

        Returns:
            A structured reward output.

        Raises:
            ValueError: If no reward terms are enabled, or if `sample_ctx.parsed` is None
                but one or more enabled terms have `require_parsed=True`.
        """
        return self.compute_batch([sample_ctx], log=log, step=step)[0]

    def compute_batch(
        self,
        sample_ctxs: collections.abc.Sequence[reward_types.SampleContext],
        *,
        log: bool | None = None,
        step: int | None = None,
    ) -> list[reward_types.RewardOutput]:
        """Compute rewards for multiple samples in a stable order.

        For relative mode verbosity scaling, samples are automatically grouped by their
        sample identifier (from `sample_ctx.sample_data.identifier`). Samples with the
        same identifier are normalized together.

        In distributed training, this method uses a single all_gather call to synchronize
        batch info (indices, completion indices, and stats) across ranks. ALL ranks must
        call compute_batch() the same number of times (empty ranks pass []). Failing to
        do so will deadlock.

        Completion indices (`completion_idx`) are computed globally across all ranks: if
        the same identifier appears on multiple ranks, each sample gets a unique completion
        index within the batch. This ensures GRPO-style analysis can correctly identify
        which completion (0 to k-1) each sample represents, even when completions are
        distributed across ranks.

        Warning:
            **Deadlock risk on exceptions**: Local reward computation (STEP 1) happens BEFORE
            the distributed gather (STEP 3). If one rank throws an exception during local
            computation, other ranks will hang indefinitely waiting in the gather. An error
            is logged before re-raising to help diagnose this scenario. Ensure reward terms
            do not raise exceptions, or wrap compute_batch() calls in exception handlers that
            terminate all ranks together.

        Args:
            sample_ctxs: Sample contexts to evaluate.
            log: When True/False, force logging on/off; when None, uses config default.
            step: Optional logging step override (defaults to manager step).

        Returns:
            Reward outputs in the same order as inputs.
        """
        # guard: empty prefix is not allowed once prefixed counters exist
        # (this catches both "forgot set_key_prefix after load_state" and "mixing prefixed/unprefixed usage")
        if not self._current_prefix:
            has_prefixed_counts = any(prefix for prefix in self._global_generation_counts if prefix) or any(
                prefix for prefix in self._global_batch_counts if prefix
            )
            if has_prefixed_counts:
                raise RuntimeError(
                    "compute_batch() called with empty key_prefix but state contains counters for "
                    f"prefixes {set(self._global_generation_counts.keys()) | set(self._global_batch_counts.keys())}. "
                    "Empty prefix is not allowed once prefixed counters exist. "
                    "Call set_key_prefix() (e.g., 'train/' or 'eval/') before compute_batch()."
                )
        local_batch_size = len(sample_ctxs)
        # STEP 1: process samples locally (compute outputs before any distributed sync)
        if local_batch_size > 0:
            try:
                outputs, caches = self._compute_local_batch(sample_ctxs)
            except Exception:
                # in distributed mode, other ranks may be waiting at the gather below;
                # log a loud error to help diagnose potential deadlock before re-raising
                if pyine.utils.distrib.is_distributed():
                    logger.error(
                        "Exception during local reward computation on rank %d. "
                        "Other ranks may deadlock waiting at the distributed gather. "
                        "Consider wrapping compute_batch() in a handler that terminates all ranks.",
                        pyine.utils.distrib.get_global_rank(default=0),
                    )
                raise
        else:
            outputs, caches = [], []
        # STEP 2: compute local reward stats and extract identifiers (before gather)
        local_total_reward_stats = stats_utils.RunningStats()
        for output in outputs:
            local_total_reward_stats.update(float(output.total))
        local_identifiers = [sample_ctx.sample_data.identifier for sample_ctx in sample_ctxs]
        # STEP 3: single distributed gather for indices + completion_idx + stats (or local computation)
        # ...in distributed mode, ALL ranks must participate (even if empty)
        if pyine.utils.distrib.is_distributed():
            gather_result = self._gather_distributed_batch_info(
                local_batch_size, self._current_prefix, local_total_reward_stats, local_identifiers
            )
            global_sample_indices, total_batch_size, merged_total_reward_stats, completion_indices = gather_result
        else:
            # non-distributed: simple local computation
            base = self._get_global_generation_count(self._current_prefix)
            global_sample_indices = [base + idx + 1 for idx in range(local_batch_size)]
            total_batch_size = local_batch_size
            merged_total_reward_stats = local_total_reward_stats
            # compute completion_idx locally (same logic as distributed, but no gathering needed)
            identifier_counts: dict[typing.Hashable, int] = {}
            completion_indices: list[int] = []
            for identifier in local_identifiers:
                completion_idx = identifier_counts.get(identifier, 0)
                completion_indices.append(completion_idx)
                identifier_counts[identifier] = completion_idx + 1
        # STEP 4: if no samples globally, return early (safe, all ranks agree)
        if total_batch_size == 0:
            return []
        # STEP 5: increment batch counters (per-prefix and overall)
        self._global_batch_counts[self._current_prefix] = self._get_global_batch_count(self._current_prefix) + 1
        self._total_global_batch_count += 1
        # STEP 6: post-processing that needs global indices (only if this rank has samples)
        if local_batch_size > 0:
            # compute difficulty metrics and merge into outputs
            if self._difficulty_estimator is not None:
                for idx, (output, sample_ctx) in enumerate(zip(outputs, sample_ctxs, strict=True)):
                    difficulty_metrics = self._difficulty_estimator.compute(
                        sample_data=sample_ctx.sample_data,
                        reward_total=output.total,
                        weighted_terms=output.weighted_terms,
                    )
                    if difficulty_metrics:
                        merged_metrics = dict(output.metrics)
                        merged_metrics.update(difficulty_metrics)
                        outputs[idx] = reward_types.RewardOutput(
                            total=output.total,
                            weighted_terms=output.weighted_terms,
                            raw_terms=output.raw_terms,
                            metrics=merged_metrics,
                        )
            # update stats and log with correct global indices
            global_batch_count = self._get_global_batch_count()
            # compute local generation indices (1-indexed) for frequency-gated logging (if needed)
            local_base = self._get_local_generation_count(self._current_prefix)
            local_sample_indices = [local_base + idx + 1 for idx in range(local_batch_size)]
            # note: completion_indices was already computed in STEP 3 (globally in distributed mode)
            for idx, (output, sample_ctx) in enumerate(zip(outputs, sample_ctxs, strict=True)):
                self._token_count_cache = caches[idx]
                self._update_running_stats(sample_ctx, output)
                self._maybe_log_sample(
                    output=output,
                    sample_ctx=sample_ctx,
                    log=log,
                    step=step,
                    local_batch_idx=idx,
                    completion_idx=completion_indices[idx],
                    global_generation_count=global_sample_indices[idx],
                    local_generation_count=local_sample_indices[idx],
                    global_batch_count=global_batch_count,
                )
        # STEP 7: update global generation counts (ALL ranks, including empty ones)
        new_gen_count = self._get_global_generation_count(self._current_prefix) + total_batch_size
        self._global_generation_counts[self._current_prefix] = new_gen_count
        self._total_global_generation_count += total_batch_size
        # update local generation count (only samples processed by this rank)
        self._local_generation_counts[self._current_prefix] = (
            self._get_local_generation_count(self._current_prefix) + local_batch_size
        )
        # STEP 8: batch stats logging (uses pre-gathered merged stats, no additional sync)
        self._maybe_log_batch_stats(merged_total_reward_stats, log=log)
        # sanity checks: verify counts are consistent (use explicit errors, not asserts)
        if len(outputs) != local_batch_size:
            raise RuntimeError(f"output count mismatch: {len(outputs)} != {local_batch_size}")
        if self._get_global_generation_count() != new_gen_count:
            raise RuntimeError(
                f"generation count mismatch after update: {self._get_global_generation_count()} != {new_gen_count}"
            )
        return outputs

    def finalize_run(
        self,
        *,
        log: bool = True,
        step: int | None = None,
        failure_ratio: float | None = None,
        failure_count: int | None = None,
    ) -> None:
        """Log run-level summaries without resetting accumulators.

        Use this at the end of training when no more samples will be processed. Unlike
        `flush_stats()`, this does NOT reset accumulators afterward, allowing subsequent
        queries to `get_*_metrics()` methods to return the final accumulated values.

        For periodic logging during training where you need to continue accumulating fresh
        stats afterward, use `flush_stats()` instead.

        Respects distributed settings from LoggingConfig:
        - barrier_before_finalize: sync before logging;
        - gather_distributed_summaries: merge stats across ranks;
        - main_process_only: only log on rank 0.

        Args:
            log: If False, suppress run-level logging even if enabled in config.
            step: Optional logging step override.
            failure_ratio: Ratio of failed samples to total samples (optional).
            failure_count: Total number of failed samples (optional).
        """
        # barrier BEFORE any early returns to avoid distributed deadlock
        # (must happen before logger checks since non-main ranks may not have a logger)
        if self._config.logging.barrier_before_finalize and pyine.utils.distrib.is_distributed():
            pyine.utils.distrib.barrier()
        # check if ANY rank has stats - use cheap all-reduce to ensure all ranks agree on the
        # decision, avoiding deadlock when some ranks have empty batches
        local_has_stats = self._reward_total_stats.count > 0
        if self._config.logging.gather_distributed_summaries and pyine.utils.distrib.is_distributed():
            any_has_stats = pyine.utils.distrib.all_reduce_boolean_or(local_has_stats)
        else:
            any_has_stats = local_has_stats
        if not any_has_stats:
            return
        # prepare summaries (may gather from distributed ranks) - ALL ranks must participate
        # in the gather if gather_distributed_summaries=True, so this must happen before
        # the main_process_only check
        summaries, _ = self._prepare_run_summaries_for_finalize()
        # now determine who should emit (after gather, so no deadlock)
        should_log = log and self._config.logging.enabled and self._logger is not None
        if not should_log:
            return
        if self._config.logging.main_process_only and not pyine.utils.distrib.is_main_process():
            return
        step_to_use = self._step if step is None else step
        self._emit_run_summaries(
            summaries,
            step=step_to_use,
            failure_ratio=failure_ratio,
            failure_count=failure_count,
        )

    def _resolve_parser(
        self,
        parser: reward_types.OutputParser | None,
    ) -> reward_types.OutputParser | None:
        """Resolve the parser to use (explicit override wins over config)."""
        if parser is not None:
            return parser
        if self._config.parsing is None:
            return None
        if self._config.parsing.mode == "tags":
            return reward_parser.TagsOutputParser(self._config.parsing)
        raise ValueError(f"unsupported parsing mode: {self._config.parsing.mode}")

    def maybe_parse(
        self,
        prompt: str,
        model_output: str,
    ) -> reward_types.ParsedOutput | None:
        """Parse model output using this manager's configured parser.

        This is a low-level utility for cases where you need just the parsed output without
        building a full `SampleContext`. For most use cases, prefer `build_sample_context()`.

        Args:
            prompt: Prompt text (passed to parser for potential future heuristics).
            model_output: Raw model output string to parse.

        Returns:
            Parsed output if a parser is configured, otherwise None.
        """
        if self._parser is None:
            return None
        return self._parser.parse(prompt, model_output)

    def build_sample_context(
        self,
        prompt: str,
        model_output: str,
        sample_data: pyine.organisms.datamodules.samples.SampleData,
        *,
        code_exec_eval: reward_types.CodeExecEvalData | None = None,
        extras: dict[str, object] | None = None,
    ) -> reward_types.SampleContext:
        """Build a SampleContext with parsing handled automatically.

        This is the recommended way to create `SampleContext` objects for use with this manager.
        It ensures parsing is performed using the manager's configured parser (if any), producing
        a context that's ready for `compute()` or `compute_batch()`.

        Args:
            prompt: Prompt text shown to the model.
            model_output: Full raw model output string.
            sample_data: Framework `SampleData` associated with this example.
            code_exec_eval: Optional code execution evaluation data.
            extras: Optional auxiliary metadata.

        Returns:
            A fully-constructed `SampleContext` with `parsed` populated (if parsing is configured).

        Example:
            ```python
            manager = RewardManager(config)
            ctx = manager.build_sample_context(
                prompt="What is 2+2?",
                model_output="<reasoning>Simple addition</reasoning><final>4</final>",
                sample_data=sample_data,
            )
            output = manager.compute(ctx)
            reward = output.total
            ```
        """
        parsed = self.maybe_parse(prompt, model_output)
        return reward_types.SampleContext(
            prompt=prompt,
            model_output=model_output,
            sample_data=sample_data,
            parsed=parsed,
            code_exec_eval=code_exec_eval,
            extras=extras or {},
        )

    def _get_run_summaries(self) -> reward_types.RunSummaries:
        """Compute local run-level summaries for total, per-term, category, parsing, and difficulty.

        Returns a RunSummaries dataclass with all aggregated metrics.
        """
        if self._reward_total_stats.count == 0:
            return reward_types.RunSummaries()
        reward_totals = self.get_reward_total_metrics()
        reward_term_summaries = self.get_reward_term_metrics()
        reward_category_summaries = self.get_reward_category_metrics()
        parsing_summaries = self.get_parsing_metrics() if self._parsing_stats else None
        parsing_category_summaries = (
            self.get_parsing_category_metrics() if self._parsing_stats and self._category_extractor else None
        )
        difficulty_summaries = None
        if self._difficulty_estimator is not None:
            difficulty_summaries = self._difficulty_estimator.get_run_summaries()
        return reward_types.RunSummaries(
            reward_totals=reward_totals,
            reward_term_summaries=reward_term_summaries,
            reward_category_summaries=reward_category_summaries,
            parsing_summaries=parsing_summaries,
            parsing_category_summaries=parsing_category_summaries,
            difficulty_summaries=difficulty_summaries,
        )

    def _gather_run_summaries(
        self,
        summaries: reward_types.RunSummaries,
    ) -> reward_types.RunSummaries:
        """Gather run stats across ranks and compute global summaries (rank0 returns merged).

        Note: Only rank 0 receives the merged summaries. Non-main ranks return their original
        local summaries unchanged. If `main_process_only=False` and `gather_distributed_summaries=True`,
        non-main ranks will still log their local (unmerged) summaries to avoid blocking, which
        may be surprising. For consistent logging, keep `main_process_only=True` (the default).

        Returns RunSummaries with merged stats on rank 0, original summaries on other ranks.
        """
        payload: dict[str, typing.Any] = {
            "total": self._reward_total_stats.as_state(),
            "terms": {name: stats.as_state() for name, stats in self._reward_term_stats.items()},
            "categories": {name: stats.as_state() for name, stats in self._reward_category_stats.items()},
            "reward_total_values": self._reward_total_values,
            "reward_total_values_count": self._reward_total_values_count,
        }
        if self._parsing_stats is not None:
            payload["parsing"] = self._parsing_stats.as_state()
        if self._difficulty_estimator is not None:
            payload["difficulty"] = self._difficulty_estimator.as_state()
        gathered = pyine.utils.distrib.all_gather_objects(payload)
        if not pyine.utils.distrib.is_main_process():
            return summaries
        merged_total = stats_utils.RunningStats()
        merged_terms: dict[str, stats_utils.RunningStats] = {
            name: stats_utils.RunningStats() for name in self._reward_term_stats
        }
        merged_categories: dict[str, stats_utils.RunningStats] = {}
        merged_parsing: reward_types.ParsingStatsAccumulator | None = (
            reward_types.ParsingStatsAccumulator.new() if self._parsing_stats is not None else None
        )
        merged_difficulty: difficulty_module.DifficultyEstimator | None = None
        if self._difficulty_estimator is not None and self._config.difficulty is not None:
            merged_difficulty = difficulty_module.DifficultyEstimator(
                self._config.difficulty, token_counter=self._token_counter
            )
        for item in gathered:
            total_state = typing.cast("collections.abc.Mapping[str, int | float]", item["total"])
            merged_total.merge(stats_utils.RunningStats.from_state(total_state))
            terms_state = typing.cast(
                "collections.abc.Mapping[str, collections.abc.Mapping[str, int | float]]",
                item["terms"],
            )
            categories_state = typing.cast(
                "collections.abc.Mapping[str, collections.abc.Mapping[str, int | float]]",
                item["categories"],
            )
            for term_name, state in terms_state.items():
                assert term_name in merged_terms
                merged_terms[term_name].merge(stats_utils.RunningStats.from_state(state))
            for category_name, state in categories_state.items():
                if category_name not in merged_categories:
                    merged_categories[category_name] = stats_utils.RunningStats()
                merged_categories[category_name].merge(stats_utils.RunningStats.from_state(state))
            if merged_parsing is not None and "parsing" in item:
                parsing_state = typing.cast("dict[str, typing.Any]", item["parsing"])
                merged_parsing.merge(reward_types.ParsingStatsAccumulator.from_state(parsing_state))
            if merged_difficulty is not None and "difficulty" in item:
                difficulty_state = typing.cast("dict[str, typing.Any]", item["difficulty"])
                other_estimator = difficulty_module.DifficultyEstimator.from_state(
                    self._config.difficulty,  # type: ignore[arg-type]
                    difficulty_state,
                    token_counter=self._token_counter,
                )
                merged_difficulty.merge(other_estimator)
        self._reward_total_stats = merged_total
        self._reward_term_stats = merged_terms
        self._reward_category_stats = merged_categories
        if merged_parsing is not None:
            self._parsing_stats = merged_parsing
        if merged_difficulty is not None:
            self._difficulty_estimator = merged_difficulty
        # merge histogram values on rank 0 with weighted resampling;
        # each rank's reservoir must be weighted by total_count / len(values) to be properly represented
        max_samples = self._config.logging.histogram_max_samples
        total_global_count = sum(item.get("reward_total_values_count", 0) for item in gathered)
        if max_samples > 0 and total_global_count > 0:
            # build weighted pool: each value gets weight = total_count_r / len(values_r)
            weighted_pool: list[tuple[float, float]] = []  # (value, weight)
            for item in gathered:
                values = item.get("reward_total_values", [])
                total_count = item.get("reward_total_values_count", 0)
                if values and total_count > 0:
                    weight = total_count / len(values)
                    weighted_pool.extend((v, weight) for v in values)
            # sample from weighted pool using weights
            if len(weighted_pool) > max_samples:
                values_only = [v for v, _ in weighted_pool]
                weights_only = [w for _, w in weighted_pool]
                # weighted sampling with replacement to approximate global distribution
                # uses dedicated RNG to avoid mutating global random state
                self._reward_total_values = self._histogram_rng.choices(
                    values_only, weights=weights_only, k=max_samples
                )
            else:
                self._reward_total_values = [v for v, _ in weighted_pool]
        else:
            # no limit or no samples: just concat
            self._reward_total_values = []
            for item in gathered:
                self._reward_total_values.extend(item.get("reward_total_values", []))
        self._reward_total_values_count = total_global_count
        # note: DifficultyEstimator histogram values are already merged via as_state()/merge()
        return self._get_run_summaries()

    @staticmethod
    def _validate_term_result(
        term_name: str,
        result: reward_types.TermResult,
    ) -> None:
        """Validate that a term result has a numeric value and valid metrics.

        Metrics can be bool, int, float, or str (per MetricValue type alias). String metrics
        are allowed for diagnostic purposes (e.g., mismatch reasons) but are capped in length.
        """
        value_any = typing.cast("typing.Any", result.value)
        if not isinstance(value_any, (int, float)) or isinstance(value_any, bool):
            raise TypeError(f"term '{term_name}' returned non-numeric value: {type(value_any)}")
        if not math.isfinite(float(value_any)):
            raise ValueError(f"term '{term_name}' returned non-finite value: {value_any}")
        for metric_key, metric_value in result.metrics.items():
            metric_key_any = typing.cast("typing.Any", metric_key)
            if not isinstance(metric_key_any, str) or not metric_key_any.strip():
                raise ValueError(f"term '{term_name}' produced invalid metric key: {metric_key_any!r}")
            metric_value_any = typing.cast("typing.Any", metric_value)
            if isinstance(metric_value_any, bool):
                continue
            if isinstance(metric_value_any, str):
                if len(metric_value_any) > _MAX_METRIC_STRING_LENGTH:
                    warnings.warn(
                        f"term '{term_name}' metric '{metric_key_any}' string value exceeds "
                        f"{_MAX_METRIC_STRING_LENGTH} chars; consider truncating upstream "
                        f"to avoid excessive log/memory usage",
                        stacklevel=4,
                    )
                continue
            if not isinstance(metric_value_any, (int, float)):
                raise TypeError(
                    f"term '{term_name}' metric '{metric_key_any}' has invalid type: {type(metric_value_any)}; "
                    f"expected bool, int, float, or str"
                )
            if not math.isfinite(float(metric_value_any)):
                raise ValueError(f"term '{term_name}' metric '{metric_key_any}' is non-finite: {metric_value_any}")

    def _add_reward_value_with_reservoir(
        self,
        value: float,
    ) -> None:
        """Add value using reservoir sampling if max_samples exceeded.

        Uses a dedicated RNG instance to avoid mutating global random state.
        """
        max_samples = self._config.logging.histogram_max_samples
        self._reward_total_values_count += 1
        if max_samples <= 0 or len(self._reward_total_values) < max_samples:
            self._reward_total_values.append(value)
        else:
            # reservoir sampling: replace random element
            idx = self._histogram_rng.randint(0, self._reward_total_values_count - 1)
            if idx < max_samples:
                self._reward_total_values[idx] = value

    def _update_running_stats(
        self,
        sample_ctx: reward_types.SampleContext,
        output: reward_types.RewardOutput,
    ) -> None:
        """Update run-level summary stats.

        Note1: per-term stats track weighted values (after per-term clipping and weight
        multiplication), not raw term values. If verbosity scaling is enabled, totals may be
        post-scaled while per-term stats remain unscaled.

        Note2: global generation counts are managed at the batch level in compute_batch(),
        not per-sample in this method. This method only updates statistics accumulators.
        """
        total_reward = float(output.total)
        self._reward_total_stats.update(total_reward)
        if self._config.logging.enabled:
            self._add_reward_value_with_reservoir(total_reward)
        for term_name, term_reward in output.weighted_terms.items():
            assert term_name in self._reward_term_stats
            self._reward_term_stats[term_name].update(float(term_reward))
        if self._category_extractor is not None:
            sample_data_dict = sample_ctx.sample_data._asdict()
            categories = self._category_extractor.extract_categories(sample_data_dict)
            for category in categories:
                if category not in self._reward_category_stats:
                    self._reward_category_stats[category] = stats_utils.RunningStats()
                self._reward_category_stats[category].update(total_reward)
        self._update_parsing_stats(sample_ctx)

    def _populate_token_count_cache(
        self,
        sample_ctx: reward_types.SampleContext,
    ) -> reward_types.TokenCountCache | None:
        """Compute and cache token counts for a sample.

        Call this before verbosity scaling to ensure token counts are available. The cache is also
        used by _update_parsing_stats to avoid recomputation.
        """
        self._token_count_cache = None  # clear cache from previous sample
        if self._token_counter is None:
            return None
        # always compute model_output token count from sample_ctx.model_output (doesn't require parsing)
        self._token_count_cache = reward_types.TokenCountCache()
        output_len_tokens = self._token_counter(sample_ctx.model_output)
        self._token_count_cache.set(reward_types.LengthSource.model_output, output_len_tokens)
        # compute parsed field token counts if parsing result is available
        if sample_ctx.parsed is not None:
            parsed = sample_ctx.parsed
            if parsed.reasoning is not None:
                reasoning_len_tokens = self._token_counter(parsed.reasoning)
                self._token_count_cache.set(reward_types.LengthSource.parsed_reasoning, reasoning_len_tokens)
            if parsed.final_answer is not None:
                answer_len_tokens = self._token_counter(parsed.final_answer)
                self._token_count_cache.set(reward_types.LengthSource.parsed_final_answer, answer_len_tokens)
        return self._token_count_cache

    def _update_parsing_stats(
        self,
        sample_ctx: reward_types.SampleContext,
    ) -> None:
        """Update parsing-related run stats if parsing is configured."""
        if self._parsing_stats is None or sample_ctx.parsed is None:
            return
        categories: list[str] | None = None
        if self._category_extractor is not None:
            sample_data_dict = sample_ctx.sample_data._asdict()
            categories = self._category_extractor.extract_categories(sample_data_dict)
        self._parsing_stats.update(sample_ctx.parsed, self._token_count_cache, categories)

    def _compute_sample_parsing_metrics(
        self,
        sample_ctx: reward_types.SampleContext,
    ) -> dict[str, reward_types.MetricValue]:
        """Compute per-sample parsing metrics for logging.

        Only emits metrics for fields that are enabled in the parsing config.
        Uses cached token lengths from _update_parsing_stats() to avoid recomputation.
        """
        if sample_ctx.parsed is None:
            return {}
        parsed = sample_ctx.parsed
        # determine which fields are enabled
        enabled_fields = self._config.parsing.enabled_fields if self._config.parsing else "both"
        reasoning_enabled = enabled_fields in ("both", "reasoning_only")
        answer_enabled = enabled_fields in ("both", "final_only")
        # use cached token lengths (computed in _update_parsing_stats)
        cache = self._token_count_cache
        metrics: dict[str, reward_types.MetricValue] = {
            "parsing/output_length_chars": len(parsed.raw),
        }
        # add token length if token tracking is enabled
        if cache is not None:
            output_tokens = cache.get(reward_types.LengthSource.model_output)
            if output_tokens is not None:
                metrics["parsing/output_length_tokens"] = output_tokens
        # is_malformed only meaningful when capture_diagnostics is enabled
        # note: use int (0/1) instead of bool for better W&B scalar display
        if self._config.parsing and self._config.parsing.capture_diagnostics:
            metrics["parsing/is_malformed"] = int(parsed.fields.get("is_malformed", "false") == "true")
        if reasoning_enabled:
            has_reasoning = bool(parsed.reasoning)  # treat empty string as missing
            metrics["parsing/has_reasoning"] = int(has_reasoning)
            if has_reasoning:
                metrics["parsing/reasoning_length_chars"] = len(parsed.reasoning)  # type: ignore[arg-type]
                if cache is not None:
                    reasoning_tokens = cache.get(reward_types.LengthSource.parsed_reasoning)
                    if reasoning_tokens is not None:
                        metrics["parsing/reasoning_length_tokens"] = reasoning_tokens
        if answer_enabled:
            has_answer = bool(parsed.final_answer)  # treat empty string as missing
            metrics["parsing/has_answer"] = int(has_answer)
            if has_answer:
                metrics["parsing/answer_length_chars"] = len(parsed.final_answer)  # type: ignore[arg-type]
                if cache is not None:
                    answer_tokens = cache.get(reward_types.LengthSource.parsed_final_answer)
                    if answer_tokens is not None:
                        metrics["parsing/answer_length_tokens"] = answer_tokens
        return metrics

    def _maybe_log_sample(
        self,
        output: reward_types.RewardOutput,
        sample_ctx: reward_types.SampleContext,
        *,
        log: bool | None,
        step: int | None,
        local_batch_idx: int,
        completion_idx: int,
        global_generation_count: int,
        local_generation_count: int,
        global_batch_count: int,
    ) -> None:
        """Log a per-sample event if logging is enabled and the frequency gate passes.

        Args:
            output: Computed reward output.
            sample_ctx: Sample context.
            log: Force logging on/off; when None, uses config default.
            step: Logging step override.
            local_batch_idx: Index of this sample within the current batch (0-indexed).
            completion_idx: Global completion index within samples sharing the same identifier
                across all ranks in this batch (0-indexed). For GRPO-style batching where k
                completions are generated per prompt, this indicates which completion (0 to k-1)
                this sample represents. In distributed mode, identifiers are gathered across ranks
                and completion indices are computed globally so that if the same identifier appears
                on multiple ranks, each sample gets a unique completion index.
            global_generation_count: Exact global generation count for this sample (across all ranks).
            local_generation_count: Local generation count for this sample (only samples processed
                by this rank). Used for frequency gating when main_process_only=True to ensure
                this rank's N-th sample triggers logging.
            global_batch_count: Global batch count (1-indexed, same for all samples in this batch).
        """
        # early return depending on main_process_only
        if self._config.logging.main_process_only and not pyine.utils.distrib.is_main_process():
            return
        # check if logging is enabled (per-rank log parameter)
        should_log = self._config.logging.enabled if log is None else log
        if not should_log:
            return
        # check if logger exists; non-main ranks may not have a logger
        # (main rank without a logger when logging is enabled is a misconfiguration)
        if self._logger is None:
            if pyine.utils.distrib.is_main_process():
                raise RuntimeError("logging is enabled but no logger is configured on main rank")
            return
        # choose which generation count to use for frequency gating:
        # - when main_process_only=True: use local count so this rank's N-th sample triggers logging
        # - when main_process_only=False: use global count for consistent frequency across ranks
        # (note: x-axis value for logged metrics is however always global count for cross-rank alignment)
        gating_count = local_generation_count if self._config.logging.main_process_only else global_generation_count
        # query logger to see if we need to do any work at all
        if not self._logger.should_log_sample(gating_count):
            return  # skip all expensive metric/category/parsing extraction
        will_add_row = self._config.logging.log_tables
        step_to_use = self._step if step is None else step
        sample_id = sample_ctx.sample_id
        # extract terms and metrics
        terms = dict(output.weighted_terms) if self._config.logging.log_terms else {}
        metrics = dict(output.metrics) if self._config.logging.log_metrics else {}
        total_to_log = float(output.total) if self._config.logging.log_total else None
        scoped_terms, scoped_metrics = self._scope_reward_sample_fields(terms, metrics)
        # extract categories (needed for both scalars and tables)
        categories: list[str] | None = None
        should_extract_categories = self._category_extractor is not None and (
            will_add_row or self._config.logging.log_metrics
        )
        if should_extract_categories:
            sample_data_dict = sample_ctx.sample_data._asdict()
            categories = self._category_extractor.extract_categories(sample_data_dict)
        # add parsing metrics after scoping
        if self._config.logging.log_metrics:
            if self._config.parsing is not None and sample_ctx.parsed is not None:
                parsing_metrics = self._compute_sample_parsing_metrics(sample_ctx)
                scoped_metrics.update(parsing_metrics)
        # extract expensive table fields only when adding a row
        reasoning: str | None = None
        final_answer: str | None = None
        if will_add_row and sample_ctx.parsed is not None:
            reasoning = sample_ctx.parsed.reasoning
            final_answer = sample_ctx.parsed.final_answer
        raw_terms: dict[str, float] | None = None
        if output.raw_terms is not None:
            raw_terms = dict(output.raw_terms)
        # extract expected_output if available
        expected_output: str | None = None
        if sample_ctx.code_exec_eval is not None:
            expected_output = sample_ctx.code_exec_eval.expected
        else:
            expected_output = getattr(sample_ctx.sample_data, "expected_output", None)
        # extract difficulty metrics from output (if present)
        all_metrics = output.metrics
        difficulty_source = all_metrics.get("difficulty/source")
        difficulty_score = all_metrics.get("difficulty/score")
        difficulty_bin = all_metrics.get("difficulty/bin_index")
        difficulty_raw_primary = all_metrics.get("difficulty/raw_primary")
        # build secondary values JSON with deterministic ordering
        secondary_raw: dict[str, float] = {}
        for key, value in all_metrics.items():
            if key.startswith("difficulty/raw/") and isinstance(value, (int, float)):
                secondary_raw[key.replace("difficulty/raw/", "")] = float(value)
        difficulty_secondary_json = json.dumps(secondary_raw, sort_keys=True) if secondary_raw else None
        self._logger.log_sample(
            sample_id,
            generation_count=global_generation_count,
            batch_count=global_batch_count,
            local_batch_idx=local_batch_idx,
            completion_idx=completion_idx,
            rank=pyine.utils.distrib.get_global_rank(default=0),
            total=total_to_log,
            terms=scoped_terms,
            raw_terms=raw_terms,
            metrics=scoped_metrics,
            step=step_to_use,
            prompt=sample_ctx.prompt,
            expected_output=expected_output,
            model_output=sample_ctx.model_output,
            reasoning=reasoning,
            final_answer=final_answer,
            categories=categories,
            tags=sample_ctx.tags or None,
            difficulty_source=str(difficulty_source) if difficulty_source else None,
            difficulty_score=float(difficulty_score) if difficulty_score is not None else None,
            difficulty_bin=int(difficulty_bin) if difficulty_bin is not None else None,
            difficulty_raw_primary=float(difficulty_raw_primary) if difficulty_raw_primary is not None else None,
            difficulty_secondary_json=difficulty_secondary_json,
            predict_type=sample_ctx.sample_data.predict_type.value,
            code_type=sample_ctx.sample_data.code_type,
            has_code_override=sample_ctx.sample_data.has_code_override,
        )

    def _scope_reward_sample_fields(
        self,
        terms: collections.abc.Mapping[str, float],
        metrics: collections.abc.Mapping[str, reward_types.MetricValue],
    ) -> tuple[dict[str, float], dict[str, reward_types.MetricValue]]:
        """Apply the 'reward/' prefix to per-sample reward term/metric keys.

        Note: Difficulty metrics (keys starting with 'difficulty/') are filtered out here
        because they should not be logged as per-generation scalars. They only appear in
        run-level summaries and as explicit columns in the generation_details table.
        """
        # filter out difficulty/* keys; they are not logged as per-generation scalars
        filtered_metrics = {k: v for k, v in metrics.items() if not k.startswith("difficulty/")}
        return (
            {f"reward/terms/{k}": float(v) for k, v in terms.items()},
            {f"reward/metrics/{k}": v for k, v in filtered_metrics.items()},
        )

    def _scope_reward_run_fields(
        self,
        reward_totals: collections.abc.Mapping[str, reward_types.MetricValue],
        reward_term_summaries: collections.abc.Mapping[str, reward_types.MetricValue],
        reward_category_summaries: collections.abc.Mapping[str, reward_types.MetricValue],
    ) -> tuple[
        dict[str, reward_types.MetricValue],
        dict[str, reward_types.MetricValue],
        dict[str, reward_types.MetricValue],
    ]:
        """Apply the 'reward/' prefix to run-level reward summary keys."""
        return (
            {f"reward/run/total/{k}": v for k, v in reward_totals.items()},
            {f"reward/run/terms/{k}": v for k, v in reward_term_summaries.items()},
            {f"reward/run/categories/{k}": v for k, v in reward_category_summaries.items()},
        )

    def _scope_parsing_fields(
        self,
        parsing_summaries: dict[str, reward_types.MetricValue] | None,
        parsing_category_summaries: dict[str, reward_types.MetricValue] | None,
    ) -> tuple[dict[str, reward_types.MetricValue] | None, dict[str, reward_types.MetricValue] | None]:
        """Apply 'parsing/' prefix to parsing summary keys."""
        if parsing_summaries is None and parsing_category_summaries is None:
            return None, None
        scoped_parsing = {f"parsing/{k}": v for k, v in parsing_summaries.items()} if parsing_summaries else None
        scoped_parsing_category = (
            {f"parsing/categories/{k}": v for k, v in parsing_category_summaries.items()}
            if parsing_category_summaries
            else None
        )
        return scoped_parsing, scoped_parsing_category

    def _maybe_log_batch_stats(
        self,
        batch_stats: stats_utils.RunningStats,
        *,
        log: bool | None,
    ) -> None:
        """Log batch-level reward statistics using pre-computed/gathered stats.

        Args:
            batch_stats: Pre-computed batch statistics (already merged across ranks if distributed).
            log: Force logging on/off; when None, uses config default.
        """
        if not self._config.logging.enabled:
            return
        if not self._config.logging.log_batch_stats:
            return
        if self._config.logging.main_process_only and not pyine.utils.distrib.is_main_process():
            return
        should_log = self._config.logging.enabled if log is None else log
        if not should_log:
            return
        if batch_stats.count == 0:
            return
        # check if logger exists; non-main ranks may not have a logger
        # (main rank without a logger when logging is enabled is a misconfiguration)
        if self._logger is None:
            if pyine.utils.distrib.is_main_process():
                raise RuntimeError("logging is enabled but no logger is configured on main rank")
            return
        batch_mean = batch_stats.mean()
        batch_std = batch_stats.std()
        # update rolling stats
        self._batch_reward_mean_stats.update(batch_mean)
        self._batch_reward_std_stats.update(batch_std)
        self._logger.log_batch_stats(
            batch_mean=batch_mean,
            batch_std=batch_std,
            batch_mean_rolling_mean=self._batch_reward_mean_stats.mean(),
            batch_mean_rolling_std=self._batch_reward_mean_stats.std(),
            batch_std_rolling_mean=self._batch_reward_std_stats.mean(),
            batch_std_rolling_std=self._batch_reward_std_stats.std(),
            batch_count=self._get_global_batch_count(),
        )

    def get_state(
        self,
    ) -> dict[str, typing.Any]:
        """Return serializable state for checkpoint persistence.

        Returns:
            Dictionary containing:
            - step: current trainer step counter (or None);
            - epoch: current trainer epoch counter (or None);
            - global_generation_counts: per-phase global generation counters;
            - global_batch_counts: per-phase global batch counters;
            - local_generation_counts: per-phase local generation counters (for frequency gating);
            - total_global_generation_count: overall monotonic generation counter;
            - total_global_batch_count: overall monotonic batch counter;
            - reward_total_stats: serialized RunningStats for total rewards;
            - reward_term_stats: dict of serialized RunningStats per term;
            - reward_category_stats: dict of serialized RunningStats per category;
            - batch_reward_mean_stats: serialized RunningStats for batch means;
            - batch_reward_std_stats: serialized RunningStats for batch std devs;
            - parsing_stats: serialized ParsingStatsAccumulator (only if parsing enabled).

        Note:
            _current_prefix is NOT persisted; trainer sets it via set_key_prefix().
        """
        state: dict[str, typing.Any] = {
            # logging indices
            "step": self._step,
            "epoch": self._epoch,
            # global counters (per-phase and overall)
            "global_generation_counts": dict(self._global_generation_counts),
            "global_batch_counts": dict(self._global_batch_counts),
            "local_generation_counts": dict(self._local_generation_counts),
            "total_global_generation_count": self._total_global_generation_count,
            "total_global_batch_count": self._total_global_batch_count,
            # per-generation reward accumulators
            "reward_total_stats": self._reward_total_stats.as_state(),
            "reward_term_stats": {name: stats.as_state() for name, stats in self._reward_term_stats.items()},
            "reward_category_stats": {name: stats.as_state() for name, stats in self._reward_category_stats.items()},
            # batch-level reward accumulators
            "batch_reward_mean_stats": self._batch_reward_mean_stats.as_state(),
            "batch_reward_std_stats": self._batch_reward_std_stats.as_state(),
            # histogram values for checkpoint persistence
            "reward_total_values": self._reward_total_values,
            "reward_total_values_count": self._reward_total_values_count,
        }
        if self._parsing_stats is not None:
            state["parsing_stats"] = self._parsing_stats.as_state()
        if self._difficulty_estimator is not None:
            state["difficulty_estimator"] = self._difficulty_estimator.as_state()
        return state

    def load_state(
        self,
        state: dict[str, typing.Any],
    ) -> None:
        """Restore state from checkpoint.

        Args:
            state: Dictionary with keys: reward_total_stats, reward_term_stats, reward_category_stats,
                step, epoch, global_generation_counts, global_batch_counts, local_generation_counts,
                total_global_generation_count, total_global_batch_count, batch_reward_mean_stats,
                batch_reward_std_stats, and optionally parsing_stats.

        Note:
            _current_prefix will be set by trainer via set_key_prefix() after loading state.

        Raises:
            KeyError: If required keys are missing from state.
        """
        # logging indices
        self._step = state["step"]
        self._epoch = state["epoch"]
        # global counters (per-phase and overall)
        self._global_generation_counts = dict(state.get("global_generation_counts", {}))
        self._global_batch_counts = dict(state.get("global_batch_counts", {}))
        self._local_generation_counts = dict(state.get("local_generation_counts", {}))
        self._total_global_generation_count = int(state.get("total_global_generation_count", 0))
        self._total_global_batch_count = int(state.get("total_global_batch_count", 0))
        # per-generation reward accumulators
        self._reward_total_stats = stats_utils.RunningStats.from_state(state["reward_total_stats"])
        for name, term_state in state["reward_term_stats"].items():
            if name not in self._reward_term_stats:
                raise KeyError(f"term '{name}' in checkpoint state not found in current config")
            self._reward_term_stats[name] = stats_utils.RunningStats.from_state(term_state)
        self._reward_category_stats.clear()
        for name, cat_state in state["reward_category_stats"].items():
            self._reward_category_stats[name] = stats_utils.RunningStats.from_state(cat_state)
        # batch-level reward accumulators
        self._batch_reward_mean_stats = stats_utils.RunningStats.from_state(state["batch_reward_mean_stats"])
        self._batch_reward_std_stats = stats_utils.RunningStats.from_state(state["batch_reward_std_stats"])
        # histogram values (optional, for checkpoint persistence)
        self._reward_total_values = list(state.get("reward_total_values", []))
        self._reward_total_values_count = int(state.get("reward_total_values_count", 0))
        # parsing stats (optional)
        if self._parsing_stats is not None and "parsing_stats" in state:
            self._parsing_stats = reward_types.ParsingStatsAccumulator.from_state(state["parsing_stats"])
        # difficulty estimator state (optional)
        if self._difficulty_estimator is not None and "difficulty_estimator" in state:
            self._difficulty_estimator = difficulty_module.DifficultyEstimator.from_state(
                self._config.difficulty,  # type: ignore[arg-type]
                state["difficulty_estimator"],
                token_counter=self._token_counter,
            )
