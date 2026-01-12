"""Reward manager implementation."""

import collections.abc
import math
import typing
import warnings

import transformers

import pyine.evals.utils
import pyine.organisms.datamodules.samples
import pyine.organisms.models.rewards.core.aggregator as reward_aggregator
import pyine.organisms.models.rewards.core.configs as reward_configs
import pyine.organisms.models.rewards.core.parser as reward_parser
import pyine.organisms.models.rewards.core.registry as reward_registry
import pyine.organisms.models.rewards.core.types as reward_types
import pyine.organisms.models.rewards.core.verbosity_scaling as verbosity_scaling
import pyine.utils.distrib
import pyine.utils.parsing as parsing_utils
import pyine.utils.stats as stats_utils
import pyine.utils.tokenizers

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

        self._step: int | None = None
        self._total_stats = stats_utils.RunningStats()
        self._term_stats: dict[str, stats_utils.RunningStats] = {
            spec.name: stats_utils.RunningStats() for spec in config.terms if spec.enabled
        }
        category_config = config.logging.category_extraction_config
        self._category_extractor: pyine.evals.utils.SampleCategoryExtractor | None = (
            pyine.evals.utils.SampleCategoryExtractor(category_config) if category_config is not None else None
        )
        self._category_stats: dict[str, stats_utils.RunningStats] = {}
        self._parsing_stats: reward_types.ParsingStatsAccumulator | None = (
            reward_types.ParsingStatsAccumulator.new() if config.parsing is not None else None
        )
        self._token_counter = self._setup_token_counter(tokenizer)
        self._token_count_cache: reward_types.TokenCountCache | None = None
        self._verbosity_scaler: verbosity_scaling.VerbosityScaler | None = None
        if config.verbosity_scaling is not None and config.verbosity_scaling.enabled:
            # _setup_token_counter already raised if verbosity_scaling is enabled but no tokenizer available
            self._verbosity_scaler = verbosity_scaling.VerbosityScaler(config.verbosity_scaling)
        self._warn_tag_inconsistencies()

    def _setup_token_counter(
        self,
        tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast | None,
    ) -> typing.Callable[[str], int] | None:
        """Set up the token counter based on configuration and provided tokenizer.

        Token counting is enabled when either parsing.track_token_lengths=True, or verbosity_scaling
        is configured and enabled.

        Args:
            tokenizer: Optional HuggingFace tokenizer provided to __init__.
        """
        needs_parsing_tokens = self._config.parsing is not None and self._config.parsing.track_token_lengths
        needs_verbosity_tokens = self._config.verbosity_scaling is not None and self._config.verbosity_scaling.enabled
        if not needs_parsing_tokens and not needs_verbosity_tokens:
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
        if self._logger and hasattr(self._logger, "set_step"):
            self._logger.set_step(step)  # type: ignore[reportUnknownMemberType]

    def set_key_prefix(
        self,
        key_prefix: str,
    ) -> None:
        """Set the key prefix for subsequent reward logging.

        This allows dynamic prefix switching for differentiating train vs eval logs
        when the reward system is called from external trainers like TRL's GRPOTrainer.

        Args:
            key_prefix: New prefix to apply to all emitted logger keys.
        """
        if self._logger and hasattr(self._logger, "set_key_prefix"):
            self._logger.set_key_prefix(key_prefix)  # type: ignore[reportUnknownMemberType]

    def get_reward_total_metrics(self) -> dict[str, float]:
        """Returns aggregated total reward metrics.

        Keys are bare (e.g., `mean`, `std`) - callers should add appropriate prefixes.
        """
        if self._total_stats.count == 0:
            return {}
        assert self._total_stats.min is not None and self._total_stats.max is not None
        return {
            "mean": self._total_stats.mean(),
            "std": self._total_stats.std(),
            "min": float(self._total_stats.min),
            "max": float(self._total_stats.max),
            "sample_count": float(self._total_stats.count),
        }

    def get_reward_term_metrics(self) -> dict[str, float]:
        """Returns aggregated term-wise reward metrics.

        Keys are `{term}/mean`, `{term}/std`, etc. - callers should add appropriate prefixes.
        """
        metrics: dict[str, float] = {}
        for term, stats in sorted(self._term_stats.items()):
            if stats.count == 0:
                continue
            assert stats.min is not None and stats.max is not None
            metrics[f"{term}/mean"] = stats.mean()
            metrics[f"{term}/std"] = stats.std()
            metrics[f"{term}/min"] = float(stats.min)
            metrics[f"{term}/max"] = float(stats.max)
            metrics[f"{term}/sample_count"] = float(stats.count)
        return metrics

    def get_reward_category_metrics(self) -> dict[str, float]:
        """Returns aggregated category-wise reward metrics.

        Keys are `{category}/mean`, etc. - callers should add appropriate prefixes.
        """
        metrics: dict[str, float] = {}
        for category, stats in sorted(self._category_stats.items()):
            if stats.count == 0:
                continue
            assert stats.min is not None and stats.max is not None
            metrics[f"{category}/mean"] = stats.mean()
            metrics[f"{category}/std"] = stats.std()
            metrics[f"{category}/min"] = stats.min
            metrics[f"{category}/max"] = stats.max
            metrics[f"{category}/sample_count"] = float(stats.count)
        return metrics

    def get_parsing_metrics(self) -> dict[str, float]:
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

    def get_parsing_category_metrics(self) -> dict[str, float]:
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
        """Reset all statistics accumulators (reward totals/terms/categories, and parsing).

        This resets only the running statistics, not term state. Use `reset()` to also reset term
        state for a new run.
        """
        self._total_stats = stats_utils.RunningStats()
        for key in self._term_stats:
            self._term_stats[key] = stats_utils.RunningStats()
        self._category_stats.clear()
        if self._parsing_stats is not None:
            self._parsing_stats.reset()

    def flush_stats(
        self,
        step: int | None = None,
    ) -> None:
        """Log all accumulated statistics and reset accumulators.

        This logs total, per-term, category-wise, and parsing metrics, then resets all accumulators.
        Use this for periodic logging during training when you need to continue accumulating fresh
        stats afterward (e.g., after an eval phase completes, before switching to the next phase).

        For end-of-training logging where no more samples will be processed, use `finalize_run()`
        instead, which logs without resetting.

        Respects distributed settings from LoggingConfig:
        - barrier_before_finalize: sync before logging;
        - gather_distributed_summaries: merge stats across ranks;
        - main_process_only: only log on rank 0.

        Args:
            step: Optional logging step override (defaults to manager step).
        """
        # barrier BEFORE any early returns to avoid distributed deadlock
        # (must happen before logger checks since non-main ranks may not have a logger)
        if self._config.logging.barrier_before_finalize and pyine.utils.distrib.is_distributed():
            pyine.utils.distrib.barrier()
        # determine if we should log (check logger availability and config)
        should_log = self._logger is not None and self._config.logging.enabled
        log_step = step if step is not None else self._step
        has_stats = self._total_stats.count > 0
        # get summaries if we have stats (needed for gather even if not logging locally)
        if has_stats:
            summaries = self._get_run_summaries()
        else:
            # create empty summaries for gathering (all ranks must participate)
            summaries = reward_types.RunSummaries()
        # gather distributed summaries if configured (all ranks must participate)
        if self._config.logging.gather_distributed_summaries and pyine.utils.distrib.is_distributed():
            summaries = self._gather_run_summaries(summaries)
        # reset accumulators and return early if not logging
        if not should_log or not has_stats:
            self.reset_accumulators()
            return
        # return early if only main process should log and this is not main
        if self._config.logging.main_process_only and not pyine.utils.distrib.is_main_process():
            self.reset_accumulators()
            return
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
        self._logger.log_run(
            reward_totals=scoped_reward_totals,
            reward_term_summaries=scoped_reward_term_summaries,
            reward_category_summaries=scoped_reward_category_summaries,
            parsing_summaries=scoped_parsing,
            parsing_category_summaries=scoped_parsing_category,
            step=log_step,
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

        Args:
            sample_ctxs: Sample contexts to evaluate.
            log: When True/False, force logging on/off; when None, uses config default.
            step: Optional logging step override (defaults to manager step).

        Returns:
            Reward outputs in the same order as inputs.
        """
        if not sample_ctxs:
            return []
        # first pass: compute core rewards and populate token count caches
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
                    scaled_total, scaled_terms, v_metrics = self._verbosity_scaler.apply_absolute(
                        aggregated_reward=result.total,
                        weighted_terms=result.weighted_terms,
                        cache=cache,
                    )
                    metrics = dict(result.metrics)
                    metrics.update(v_metrics)
                    output = reward_types.RewardOutput(
                        total=scaled_total,
                        weighted_terms=scaled_terms,
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
                group_terms = [core_results[idx].weighted_terms for idx in sample_indices]
                group_caches: list[reward_types.TokenCountCache] = []
                for idx in sample_indices:
                    cache = caches[idx]
                    assert cache is not None, "token cache should have been enabled for verbosity scaling"
                    group_caches.append(cache)
                scaled_totals, scaled_terms, v_metrics = self._verbosity_scaler.apply_to_group(
                    aggregated_rewards=group_totals,
                    all_weighted_terms=group_terms,
                    caches=group_caches,
                )
                for local_sample_idx, global_sample_idx in enumerate(sample_indices):
                    result = core_results[global_sample_idx]
                    metrics = dict(result.metrics)
                    metrics.update(v_metrics[local_sample_idx])
                    outputs_by_idx[global_sample_idx] = reward_types.RewardOutput(
                        total=scaled_totals[local_sample_idx],
                        weighted_terms=scaled_terms[local_sample_idx],
                        raw_terms=result.raw_terms,
                        metrics=metrics,
                    )
            # all entries should be filled since we iterate over all trace_ids
            assert all(o is not None for o in outputs_by_idx), "some samples were not processed?"
            outputs = typing.cast("list[reward_types.RewardOutput]", outputs_by_idx)
        # set cache, update stats, and then log (in the correct order, important!)
        for idx, (output, sample_ctx) in enumerate(zip(outputs, sample_ctxs, strict=True)):
            self._token_count_cache = caches[idx]
            self._update_running_stats(sample_ctx, output)
            self._maybe_log_sample(output, sample_ctx, log=log, step=step)
        return outputs

    def finalize_run(
        self,
        *,
        log: bool = True,
        step: int | None = None,
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
            step: Optional logging step override (defaults to manager step).
        """
        # barrier BEFORE any early returns to avoid distributed deadlock
        # (must happen before logger checks since non-main ranks may not have a logger)
        if self._config.logging.barrier_before_finalize and pyine.utils.distrib.is_distributed():
            pyine.utils.distrib.barrier()
        # determine if we should log (check logger availability, config, and log flag)
        should_log = log and self._config.logging.enabled and self._logger is not None
        step_to_use = self._step if step is None else step
        has_stats = self._total_stats.count > 0
        # get summaries if we have stats (needed for gather even if not logging locally)
        if has_stats:
            summaries = self._get_run_summaries()
        else:
            # create empty summaries for gathering (all ranks must participate)
            summaries = reward_types.RunSummaries()
        # gather distributed summaries if configured (all ranks must participate)
        if self._config.logging.gather_distributed_summaries and pyine.utils.distrib.is_distributed():
            summaries = self._gather_run_summaries(summaries)
        # return early if not logging or no stats
        if not should_log or not has_stats:
            return
        # return early if only main process should log and this is not main
        if self._config.logging.main_process_only and not pyine.utils.distrib.is_main_process():
            return
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
        self._logger.log_run(
            reward_totals=scoped_reward_totals,
            reward_term_summaries=scoped_reward_term_summaries,
            reward_category_summaries=scoped_reward_category_summaries,
            parsing_summaries=scoped_parsing,
            parsing_category_summaries=scoped_parsing_category,
            step=step_to_use,
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
        """Compute local run-level summaries (mean/min/max/std) for total, per-term, category, and parsing values.

        Returns a RunSummaries dataclass with all aggregated metrics.
        """
        if self._total_stats.count == 0:
            return reward_types.RunSummaries()
        reward_totals = self.get_reward_total_metrics()
        reward_term_summaries = self.get_reward_term_metrics()
        reward_category_summaries = self.get_reward_category_metrics()
        parsing_summaries = self.get_parsing_metrics() if self._parsing_stats else None
        parsing_category_summaries = (
            self.get_parsing_category_metrics() if self._parsing_stats and self._category_extractor else None
        )
        return reward_types.RunSummaries(
            reward_totals=reward_totals,
            reward_term_summaries=reward_term_summaries,
            reward_category_summaries=reward_category_summaries,
            parsing_summaries=parsing_summaries,
            parsing_category_summaries=parsing_category_summaries,
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
            "total": self._total_stats.as_state(),
            "terms": {name: stats.as_state() for name, stats in self._term_stats.items()},
            "categories": {name: stats.as_state() for name, stats in self._category_stats.items()},
        }
        if self._parsing_stats is not None:
            payload["parsing"] = self._parsing_stats.as_state()
        gathered = pyine.utils.distrib.all_gather_objects(payload)
        if not pyine.utils.distrib.is_main_process():
            return summaries
        merged_total = stats_utils.RunningStats()
        merged_terms: dict[str, stats_utils.RunningStats] = {
            name: stats_utils.RunningStats() for name in self._term_stats
        }
        merged_categories: dict[str, stats_utils.RunningStats] = {}
        merged_parsing: reward_types.ParsingStatsAccumulator | None = (
            reward_types.ParsingStatsAccumulator.new() if self._parsing_stats is not None else None
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
        self._total_stats = merged_total
        self._term_stats = merged_terms
        self._category_stats = merged_categories
        if merged_parsing is not None:
            self._parsing_stats = merged_parsing
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

    def _update_running_stats(
        self,
        sample_ctx: reward_types.SampleContext,
        output: reward_types.RewardOutput,
    ) -> None:
        """Update run-level summary stats.

        Note: Per-term stats track weighted values (after per-term clipping and weight
        multiplication), not raw term values. This matches what contributes to the total.
        """
        total_reward = float(output.total)
        self._total_stats.update(total_reward)
        for term_name, term_reward in output.weighted_terms.items():
            assert term_name in self._term_stats
            self._term_stats[term_name].update(float(term_reward))
        if self._category_extractor is not None:
            sample_data_dict = sample_ctx.sample_data._asdict()
            categories = self._category_extractor.extract_categories(sample_data_dict)
            for category in categories:
                if category not in self._category_stats:
                    self._category_stats[category] = stats_utils.RunningStats()
                self._category_stats[category].update(total_reward)
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
            has_reasoning = parsed.reasoning is not None
            metrics["parsing/has_reasoning"] = int(has_reasoning)
            if has_reasoning:
                metrics["parsing/reasoning_length_chars"] = len(parsed.reasoning)  # type: ignore[arg-type]
                if cache is not None:
                    reasoning_tokens = cache.get(reward_types.LengthSource.parsed_reasoning)
                    if reasoning_tokens is not None:
                        metrics["parsing/reasoning_length_tokens"] = reasoning_tokens
        if answer_enabled:
            has_answer = parsed.final_answer is not None
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
    ) -> None:
        """Log a per-sample event if logging is enabled and the frequency gate passes."""
        should_log = self._config.logging.enabled if log is None else log
        if not should_log:
            return
        if self._config.logging.main_process_only and not pyine.utils.distrib.is_main_process():
            return
        if self._logger is None:
            raise ValueError("logging is enabled but no logger is configured")
        if self._total_stats.count % int(self._config.logging.log_every_n_examples) != 0:
            return
        step_to_use = self._step if step is None else step
        sample_id = sample_ctx.sample_id
        terms: dict[str, float] = dict(output.weighted_terms) if self._config.logging.log_terms else {}
        metrics: dict[str, reward_types.MetricValue] = dict(output.metrics) if self._config.logging.log_metrics else {}
        total_to_log: float | None = float(output.total) if self._config.logging.log_total else None
        # apply scope prefix to reward terms/metrics first
        scoped_terms, scoped_metrics = self._scope_reward_sample_fields(terms, metrics)
        # extract categories for table logging and scalar filtering labels
        categories: list[str] | None = None
        if self._category_extractor is not None:
            sample_data_dict = sample_ctx.sample_data._asdict()
            categories = self._category_extractor.extract_categories(sample_data_dict)
        # add parsing metrics after scoping (fixed parsing/ prefix, not scoped)
        if self._config.logging.log_metrics:
            if self._config.parsing is not None and sample_ctx.parsed is not None:
                parsing_metrics = self._compute_sample_parsing_metrics(sample_ctx)
                scoped_metrics.update(parsing_metrics)
            # add category labels for filtering (not scoped)
            if categories:
                for category in categories:
                    scoped_metrics[f"categories/{category}"] = True
        # extract reasoning and final_answer from parsed output for table logging
        reasoning: str | None = None
        final_answer: str | None = None
        if sample_ctx.parsed is not None:
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
        # extract generation_idx from extras if available
        generation_idx: int | None = None
        gen_idx_value = sample_ctx.extras.get("generation_idx")
        if isinstance(gen_idx_value, int):
            generation_idx = gen_idx_value
        self._logger.log(
            sample_id,
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
            generation_idx=generation_idx,
        )

    def _scope_reward_sample_fields(
        self,
        terms: collections.abc.Mapping[str, float],
        metrics: collections.abc.Mapping[str, reward_types.MetricValue],
    ) -> tuple[dict[str, float], dict[str, reward_types.MetricValue]]:
        """Apply the configured scope prefix to per-sample reward term/metric keys."""
        prefix_norm = parsing_utils.normalize_path_prefix(self._config.logging.scope_prefix)
        if not prefix_norm:
            return dict(terms), dict(metrics)
        return (
            {f"{prefix_norm}terms/{k}": float(v) for k, v in terms.items()},
            {f"{prefix_norm}metrics/{k}": v for k, v in metrics.items()},
        )

    def _scope_reward_run_fields(
        self,
        reward_totals: collections.abc.Mapping[str, float],
        reward_term_summaries: collections.abc.Mapping[str, float],
        reward_category_summaries: collections.abc.Mapping[str, float],
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        """Apply the configured scope prefix to run-level reward summary keys."""
        prefix_norm = parsing_utils.normalize_path_prefix(self._config.logging.scope_prefix)
        if not prefix_norm:
            return dict(reward_totals), dict(reward_term_summaries), dict(reward_category_summaries)
        return (
            {f"{prefix_norm}run/{k}": float(v) for k, v in reward_totals.items()},
            {f"{prefix_norm}run/terms/{k}": float(v) for k, v in reward_term_summaries.items()},
            {f"{prefix_norm}run/categories/{k}": float(v) for k, v in reward_category_summaries.items()},
        )

    def _scope_parsing_fields(
        self,
        parsing_summaries: dict[str, float] | None,
        parsing_category_summaries: dict[str, float] | None,
    ) -> tuple[dict[str, float] | None, dict[str, float] | None]:
        """Apply fixed 'parsing/' prefix to parsing summary keys.

        Unlike reward metrics which use the configurable scope_prefix, parsing
        metrics always use a fixed 'parsing/' prefix for clarity.
        """
        if parsing_summaries is None and parsing_category_summaries is None:
            return None, None
        scoped_parsing = {f"parsing/{k}": float(v) for k, v in parsing_summaries.items()} if parsing_summaries else None
        scoped_parsing_category = (
            {f"parsing/categories/{k}": float(v) for k, v in parsing_category_summaries.items()}
            if parsing_category_summaries
            else None
        )
        return scoped_parsing, scoped_parsing_category

    def get_state(
        self,
    ) -> dict[str, typing.Any]:
        """Return serializable state for checkpoint persistence.

        Returns:
            Dictionary containing:
            - total_stats: serialized RunningStats for total rewards;
            - term_stats: dict of serialized RunningStats per term;
            - category_stats: dict of serialized RunningStats per category;
            - parsing_stats: serialized ParsingStatsAccumulator (only if parsing enabled);
            - step: current step counter (or None).
        """
        state: dict[str, typing.Any] = {
            "total_stats": self._total_stats.as_state(),
            "term_stats": {name: stats.as_state() for name, stats in self._term_stats.items()},
            "category_stats": {name: stats.as_state() for name, stats in self._category_stats.items()},
            "step": self._step,
        }
        if self._parsing_stats is not None:
            state["parsing_stats"] = self._parsing_stats.as_state()
        return state

    def load_state(
        self,
        state: dict[str, typing.Any],
    ) -> None:
        """Restore state from checkpoint.

        Args:
            state: Dictionary with keys: total_stats, term_stats, category_stats, step,
                and optionally parsing_stats.

        Raises:
            KeyError: If required keys are missing from state.
        """
        self._total_stats = stats_utils.RunningStats.from_state(state["total_stats"])
        for name, term_state in state["term_stats"].items():
            if name not in self._term_stats:
                raise KeyError(f"term '{name}' in checkpoint state not found in current config")
            self._term_stats[name] = stats_utils.RunningStats.from_state(term_state)
        self._category_stats.clear()
        for name, cat_state in state["category_stats"].items():
            self._category_stats[name] = stats_utils.RunningStats.from_state(cat_state)
        self._step = state["step"]
        # restore parsing stats if present and parsing is enabled
        if self._parsing_stats is not None and "parsing_stats" in state:
            self._parsing_stats = reward_types.ParsingStatsAccumulator.from_state(state["parsing_stats"])
