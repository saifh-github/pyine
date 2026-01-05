"""Reward manager implementation."""

import collections.abc
import inspect
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
import pyine.utils.distrib
import pyine.utils.parsing as parsing_utils
import pyine.utils.stats as stats_utils
import pyine.utils.tokenizers

_MAX_METRIC_STRING_LENGTH = 500
"""Threshold for string metric length warnings. Strings exceeding this emit a warning."""


def make_simple_manager(
    terms: collections.abc.Sequence[tuple[str, str, float]],
    *,
    parsing: bool = True,
    final_tag: str = "final",
    reasoning_tag: str = "reasoning",
    logger: reward_types.RewardLogger | None = None,
) -> "RewardManager":
    """Create a RewardManager with minimal configuration.

    This convenience function reduces boilerplate for common use cases where you want
    to quickly set up a manager with a few weighted terms.

    Args:
        terms: Sequence of (name, type, weight) tuples specifying the reward terms.
            Example: `[("format", "parseable_answer", 1.0)]`
        parsing: Whether to enable tag-based parsing (default True).
        final_tag: Tag name for final answer extraction when parsing is enabled.
        reasoning_tag: Tag name for reasoning extraction when parsing is enabled.
        logger: Optional logger implementation for reward logging.

    Returns:
        A configured RewardManager instance.

    Raises:
        ValueError: If no terms are provided or term configuration is invalid.

    Note:
        This function only works with terms that have all-default parameters. Terms like
        `text_length` require explicit `params.components`, so use `RewardManager` directly
        with a full `RewardManagerConfig` for those.

    Example:
        ```python
        import pyine.organisms.models.rewards as rewards

        # create a simple manager (only works with terms that have all-default params)
        manager = rewards.make_simple_manager(
            [
                ("format", "parseable_answer", 1.0),
            ]
        )

        # build context with automatic parsing, then compute rewards
        ctx = manager.build_sample_context(
            prompt="...",
            model_output="<final>answer</final>",
            sample_data=sample_data,  # from datamodule
        )
        total = manager.compute(ctx)
        ```
    """
    if not terms:
        raise ValueError("at least one term must be provided")
    # Note: require_parsed is set uniformly based on whether parsing is enabled. This is
    # intentionally simple: if parsing is on, all terms declare they need it; if parsing is off,
    # none do. For fine-grained control, use RewardManager with a full config instead.
    term_specs = [
        reward_configs.RewardTermSpec(
            name=name,
            type=term_type,
            weight=weight,
            require_parsed=parsing,
        )
        for name, term_type, weight in terms
    ]
    parsing_config = (
        reward_configs.ParsingConfig(
            final_tag=final_tag,
            reasoning_tag=reasoning_tag,
        )
        if parsing
        else None
    )
    config = reward_configs.RewardManagerConfig(
        terms=term_specs,
        parsing=parsing_config,
    )
    return RewardManager(config, logger=logger)


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
        reward = manager.compute(ctx)
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
            if not config.logging.main_process_only or self._is_main_process():
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
            self._validate_factory_signature(factory=factory, spec=spec)
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
        self._cached_token_lengths: tuple[int | None, int | None, int | None] | None = None
        """Cache for (output, reasoning, answer) token lengths to avoid recomputation during logging."""
        self._warn_tag_inconsistencies()

    def _setup_token_counter(
        self,
        tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast | None,
    ) -> typing.Callable[[str], int] | None:
        """Set up the token counter based on configuration and provided tokenizer.

        Args:
            tokenizer: Optional HuggingFace tokenizer provided to __init__.

        Returns:
            A callable that counts tokens in a string, or None if token tracking is disabled.

        Raises:
            ValueError: If track_token_lengths=True but no tokenizer source is available.
        """
        if self._config.parsing is None or not self._config.parsing.track_token_lengths:
            return None
        if tokenizer is not None:

            def _count_hf_tokens(text: str) -> int:
                return len(tokenizer.encode(text, add_special_tokens=False))  # type: ignore[reportUnknownMemberType]

            return _count_hf_tokens
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
            "track_token_lengths=True requires either a tokenizer argument to RewardManager "
            "or openai_tokenizer_model set in ParsingConfig"
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

        Returns empty dict if parsing is not configured or no samples processed. Keys are
        `output_length_chars/mean`, `missing_reasoning_ratio`, etc. When token tracking is enabled,
        also includes `output_length_tokens/mean`, etc.

        Note: `missing_reasoning_ratio` and `reasoning_length_*` are only emitted when reasoning
        extraction is enabled (enabled_fields is "both" or "reasoning_only"). Similarly,
        `missing_answer_ratio` and `answer_length_*` are only emitted when final answer extraction
        is enabled (enabled_fields is "both" or "final_only"). `malformed_ratio` is only emitted
        when `capture_diagnostics=True` in the parsing config.
        """
        if self._parsing_stats is None or self._parsing_stats.total_count == 0:
            return {}
        stats = self._parsing_stats
        metrics: dict[str, float] = {}
        # determine which fields are enabled
        enabled_fields = self._config.parsing.enabled_fields if self._config.parsing else "both"
        reasoning_enabled = enabled_fields in ("both", "reasoning_only")
        answer_enabled = enabled_fields in ("both", "final_only")
        # output length stats in chars (always tracked)
        if stats.output_length_chars.count > 0:
            assert stats.output_length_chars.min is not None and stats.output_length_chars.max is not None
            metrics["output_length_chars/mean"] = stats.output_length_chars.mean()
            metrics["output_length_chars/std"] = stats.output_length_chars.std()
            metrics["output_length_chars/min"] = stats.output_length_chars.min
            metrics["output_length_chars/max"] = stats.output_length_chars.max
        # output length stats in tokens (only when token tracking enabled)
        if stats.output_length_tokens.count > 0:
            assert stats.output_length_tokens.min is not None and stats.output_length_tokens.max is not None
            metrics["output_length_tokens/mean"] = stats.output_length_tokens.mean()
            metrics["output_length_tokens/std"] = stats.output_length_tokens.std()
            metrics["output_length_tokens/min"] = stats.output_length_tokens.min
            metrics["output_length_tokens/max"] = stats.output_length_tokens.max
        # reasoning length stats in chars (only when reasoning enabled and samples have reasoning)
        if reasoning_enabled and stats.reasoning_length_chars.count > 0:
            assert stats.reasoning_length_chars.min is not None and stats.reasoning_length_chars.max is not None
            metrics["reasoning_length_chars/mean"] = stats.reasoning_length_chars.mean()
            metrics["reasoning_length_chars/std"] = stats.reasoning_length_chars.std()
            metrics["reasoning_length_chars/min"] = stats.reasoning_length_chars.min
            metrics["reasoning_length_chars/max"] = stats.reasoning_length_chars.max
        # reasoning length stats in tokens
        if reasoning_enabled and stats.reasoning_length_tokens.count > 0:
            assert stats.reasoning_length_tokens.min is not None and stats.reasoning_length_tokens.max is not None
            metrics["reasoning_length_tokens/mean"] = stats.reasoning_length_tokens.mean()
            metrics["reasoning_length_tokens/std"] = stats.reasoning_length_tokens.std()
            metrics["reasoning_length_tokens/min"] = stats.reasoning_length_tokens.min
            metrics["reasoning_length_tokens/max"] = stats.reasoning_length_tokens.max
        # answer length stats in chars (only when answer enabled and samples have answer)
        if answer_enabled and stats.answer_length_chars.count > 0:
            assert stats.answer_length_chars.min is not None and stats.answer_length_chars.max is not None
            metrics["answer_length_chars/mean"] = stats.answer_length_chars.mean()
            metrics["answer_length_chars/std"] = stats.answer_length_chars.std()
            metrics["answer_length_chars/min"] = stats.answer_length_chars.min
            metrics["answer_length_chars/max"] = stats.answer_length_chars.max
        # answer length stats in tokens
        if answer_enabled and stats.answer_length_tokens.count > 0:
            assert stats.answer_length_tokens.min is not None and stats.answer_length_tokens.max is not None
            metrics["answer_length_tokens/mean"] = stats.answer_length_tokens.mean()
            metrics["answer_length_tokens/std"] = stats.answer_length_tokens.std()
            metrics["answer_length_tokens/min"] = stats.answer_length_tokens.min
            metrics["answer_length_tokens/max"] = stats.answer_length_tokens.max
        # format ratios (only for enabled fields)
        total = float(stats.total_count)
        if reasoning_enabled:
            metrics["missing_reasoning_ratio"] = stats.missing_reasoning_count / total
        if answer_enabled:
            metrics["missing_answer_ratio"] = stats.missing_answer_count / total
        # malformed_ratio only meaningful when capture_diagnostics is enabled
        if self._config.parsing and self._config.parsing.capture_diagnostics:
            metrics["malformed_ratio"] = stats.malformed_count / total
        metrics["sample_count"] = total
        return metrics

    def get_parsing_category_metrics(self) -> dict[str, float]:
        """Returns category-wise parsing metrics.

        Returns empty dict if parsing or category extraction is not configured. Keys are
        `{category}/output_length_chars/mean`, `{category}/missing_reasoning_ratio`, etc. When token
        tracking is enabled, also includes `{category}/output_length_tokens/mean`, etc.

        Note: `missing_reasoning_ratio` and `reasoning_length_*` are only emitted when reasoning
        extraction is enabled. Similarly, `missing_answer_ratio` and `answer_length_*` are only
        emitted when final answer extraction is enabled. `malformed_ratio` is only emitted when
        `capture_diagnostics=True` in the parsing config.
        """
        if self._parsing_stats is None or self._category_extractor is None:
            return {}
        stats = self._parsing_stats
        metrics: dict[str, float] = {}
        # determine which fields are enabled
        enabled_fields = self._config.parsing.enabled_fields if self._config.parsing else "both"
        reasoning_enabled = enabled_fields in ("both", "reasoning_only")
        answer_enabled = enabled_fields in ("both", "final_only")
        for category in sorted(stats.category_total_count.keys()):
            total = float(stats.category_total_count.get(category, 0))
            if total == 0:
                continue
            # output length (chars)
            cat_output_chars = stats.category_output_length_chars.get(category)
            if cat_output_chars is not None and cat_output_chars.count > 0:
                assert cat_output_chars.min is not None and cat_output_chars.max is not None
                metrics[f"{category}/output_length_chars/mean"] = cat_output_chars.mean()
                metrics[f"{category}/output_length_chars/std"] = cat_output_chars.std()
                metrics[f"{category}/output_length_chars/min"] = cat_output_chars.min
                metrics[f"{category}/output_length_chars/max"] = cat_output_chars.max
            # output length (tokens)
            cat_output_tokens = stats.category_output_length_tokens.get(category)
            if cat_output_tokens is not None and cat_output_tokens.count > 0:
                assert cat_output_tokens.min is not None and cat_output_tokens.max is not None
                metrics[f"{category}/output_length_tokens/mean"] = cat_output_tokens.mean()
                metrics[f"{category}/output_length_tokens/std"] = cat_output_tokens.std()
                metrics[f"{category}/output_length_tokens/min"] = cat_output_tokens.min
                metrics[f"{category}/output_length_tokens/max"] = cat_output_tokens.max
            # reasoning length (chars, only when reasoning enabled)
            if reasoning_enabled:
                cat_reasoning_chars = stats.category_reasoning_length_chars.get(category)
                if cat_reasoning_chars is not None and cat_reasoning_chars.count > 0:
                    assert cat_reasoning_chars.min is not None and cat_reasoning_chars.max is not None
                    metrics[f"{category}/reasoning_length_chars/mean"] = cat_reasoning_chars.mean()
                    metrics[f"{category}/reasoning_length_chars/std"] = cat_reasoning_chars.std()
                    metrics[f"{category}/reasoning_length_chars/min"] = cat_reasoning_chars.min
                    metrics[f"{category}/reasoning_length_chars/max"] = cat_reasoning_chars.max
                # reasoning length (tokens)
                cat_reasoning_tokens = stats.category_reasoning_length_tokens.get(category)
                if cat_reasoning_tokens is not None and cat_reasoning_tokens.count > 0:
                    assert cat_reasoning_tokens.min is not None and cat_reasoning_tokens.max is not None
                    metrics[f"{category}/reasoning_length_tokens/mean"] = cat_reasoning_tokens.mean()
                    metrics[f"{category}/reasoning_length_tokens/std"] = cat_reasoning_tokens.std()
                    metrics[f"{category}/reasoning_length_tokens/min"] = cat_reasoning_tokens.min
                    metrics[f"{category}/reasoning_length_tokens/max"] = cat_reasoning_tokens.max
            # answer length (chars, only when answer enabled)
            if answer_enabled:
                cat_answer_chars = stats.category_answer_length_chars.get(category)
                if cat_answer_chars is not None and cat_answer_chars.count > 0:
                    assert cat_answer_chars.min is not None and cat_answer_chars.max is not None
                    metrics[f"{category}/answer_length_chars/mean"] = cat_answer_chars.mean()
                    metrics[f"{category}/answer_length_chars/std"] = cat_answer_chars.std()
                    metrics[f"{category}/answer_length_chars/min"] = cat_answer_chars.min
                    metrics[f"{category}/answer_length_chars/max"] = cat_answer_chars.max
                # answer length (tokens)
                cat_answer_tokens = stats.category_answer_length_tokens.get(category)
                if cat_answer_tokens is not None and cat_answer_tokens.count > 0:
                    assert cat_answer_tokens.min is not None and cat_answer_tokens.max is not None
                    metrics[f"{category}/answer_length_tokens/mean"] = cat_answer_tokens.mean()
                    metrics[f"{category}/answer_length_tokens/std"] = cat_answer_tokens.std()
                    metrics[f"{category}/answer_length_tokens/min"] = cat_answer_tokens.min
                    metrics[f"{category}/answer_length_tokens/max"] = cat_answer_tokens.max
            # format ratios (only for enabled fields)
            if reasoning_enabled:
                missing_reasoning = float(stats.category_missing_reasoning_count.get(category, 0))
                metrics[f"{category}/missing_reasoning_ratio"] = missing_reasoning / total
            if answer_enabled:
                missing_answer = float(stats.category_missing_answer_count.get(category, 0))
                metrics[f"{category}/missing_answer_ratio"] = missing_answer / total
            # malformed_ratio only meaningful when capture_diagnostics is enabled
            if self._config.parsing and self._config.parsing.capture_diagnostics:
                malformed = float(stats.category_malformed_count.get(category, 0))
                metrics[f"{category}/malformed_ratio"] = malformed / total
            metrics[f"{category}/sample_count"] = total
        return metrics

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
        if self._config.logging.barrier_before_finalize and self._is_distributed():
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
        if self._config.logging.gather_distributed_summaries and self._is_distributed():
            summaries = self._gather_run_summaries(summaries)
        # reset accumulators and return early if not logging
        if not should_log or not has_stats:
            self.reset_accumulators()
            return
        # return early if only main process should log and this is not main
        if self._config.logging.main_process_only and not self._is_main_process():
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

    def compute(
        self,
        sample_ctx: reward_types.SampleContext,
        *,
        return_breakdown: bool | None = None,
        log: bool | None = None,
        step: int | None = None,
    ) -> float | tuple[float, dict[str, float]]:
        """Compute a per-sample reward.

        Args:
            sample_ctx: Sample context containing trajectory information.
            return_breakdown: When True, returns `(total, breakdown)`; when None, uses config default.
            log: When True/False, force logging on/off; when None, uses config default.
            step: Optional logging step override (defaults to manager step).
        Returns:
            Either `total` or `(total, breakdown)` depending on `return_breakdown`.
        """
        output = self.compute_output(sample_ctx, log=log, step=step)
        want_breakdown = self._config.output.return_breakdown_default if return_breakdown is None else return_breakdown
        if not want_breakdown:
            return output.total
        breakdown: dict[str, float] = dict(output.weighted_terms)
        if self._config.output.return_raw_breakdown and output.raw_terms is not None:
            for term_name, value in output.raw_terms.items():
                breakdown[f"raw/{term_name}"] = value
        return output.total, breakdown

    def compute_output(
        self,
        sample_ctx: reward_types.SampleContext,
        *,
        log: bool | None = None,
        step: int | None = None,
    ) -> reward_types.RewardOutput:
        """Compute a full reward output (total + breakdown + metrics).

        This is the most explicit API: it always returns a structured `RewardOutput`.

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
        output = reward_types.RewardOutput(
            total=total,
            weighted_terms=weighted_terms,
            raw_terms=raw_terms if self._config.output.return_raw_breakdown else None,
            metrics=metrics,
        )

        self._update_running_stats(sample_ctx, output)
        self._maybe_log_sample(output, sample_ctx, log=log, step=step)
        return output

    def compute_batch(
        self,
        sample_ctxs: list[reward_types.SampleContext],
        *,
        log: bool | None = None,
        step: int | None = None,
    ) -> list[reward_types.RewardOutput]:
        """Compute rewards for multiple samples in a stable order.

        Args:
            sample_ctxs: Sample contexts to evaluate.
            log: When True/False, force logging on/off; when None, uses config default.
            step: Optional logging step override (defaults to manager step).

        Returns:
            Reward outputs in the same order as inputs.
        """
        return [self.compute_output(sample_ctx, log=log, step=step) for sample_ctx in sample_ctxs]

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
        if self._config.logging.barrier_before_finalize and self._is_distributed():
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
        if self._config.logging.gather_distributed_summaries and self._is_distributed():
            summaries = self._gather_run_summaries(summaries)
        # return early if not logging or no stats
        if not should_log or not has_stats:
            return
        # return early if only main process should log and this is not main
        if self._config.logging.main_process_only and not self._is_main_process():
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

    def term_config(
        self,
        name: str,
    ) -> reward_configs.RewardTermSpec:
        """Return the `RewardTermSpec` for a term by name."""
        if name not in self._specs_by_name:
            raise KeyError(f"unknown term: {name}")
        return self._specs_by_name[name]

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

    @staticmethod
    def _validate_factory_signature(
        *,
        factory: reward_types.RewardTermFactory,
        spec: reward_configs.RewardTermSpec,
    ) -> None:
        """Validate that a factory can be called with `(spec, parser=...)`."""
        signature = inspect.signature(factory)
        try:
            signature.bind(spec, parser=None)
        except TypeError as exc:
            raise TypeError(
                f"invalid term factory signature for name={spec.name} type={spec.type}: {signature}"
            ) from exc

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
        a context that's ready for `compute()` or `compute_output()`.

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
            reward = manager.compute(ctx)
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
        if not self._is_main_process():
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

    @staticmethod
    def _is_malformed(
        parsed: reward_types.ParsedOutput,
    ) -> bool:
        """Check if output has any malformed tag structure.

        Returns False if diagnostics are not captured (capture_diagnostics=False in config).
        """
        return parsed.fields.get("is_malformed", "false") == "true"

    def _update_parsing_stats(
        self,
        sample_ctx: reward_types.SampleContext,
    ) -> None:
        """Update parsing-related run stats if parsing is configured."""
        self._cached_token_lengths = None  # clear cache from previous sample
        if self._parsing_stats is None or sample_ctx.parsed is None:
            return
        parsed = sample_ctx.parsed
        output_len_chars = len(parsed.raw)
        has_reasoning = parsed.reasoning is not None
        has_answer = parsed.final_answer is not None
        is_malformed = self._is_malformed(parsed)
        reasoning_len_chars = len(parsed.reasoning) if has_reasoning else None  # type: ignore[arg-type]
        answer_len_chars = len(parsed.final_answer) if has_answer else None  # type: ignore[arg-type]
        # compute token lengths if enabled (and cache for potential use in per-sample logging)
        output_len_tokens: int | None = None
        reasoning_len_tokens: int | None = None
        answer_len_tokens: int | None = None
        if self._token_counter is not None:
            output_len_tokens = self._token_counter(parsed.raw)
            if has_reasoning:
                reasoning_len_tokens = self._token_counter(parsed.reasoning)  # type: ignore[arg-type]
            if has_answer:
                answer_len_tokens = self._token_counter(parsed.final_answer)  # type: ignore[arg-type]
            self._cached_token_lengths = (output_len_tokens, reasoning_len_tokens, answer_len_tokens)
        # update global character length stats
        self._parsing_stats.output_length_chars.update(float(output_len_chars))
        if reasoning_len_chars is not None:
            self._parsing_stats.reasoning_length_chars.update(float(reasoning_len_chars))
        if answer_len_chars is not None:
            self._parsing_stats.answer_length_chars.update(float(answer_len_chars))
        # update global token length stats
        if output_len_tokens is not None:
            self._parsing_stats.output_length_tokens.update(float(output_len_tokens))
        if reasoning_len_tokens is not None:
            self._parsing_stats.reasoning_length_tokens.update(float(reasoning_len_tokens))
        if answer_len_tokens is not None:
            self._parsing_stats.answer_length_tokens.update(float(answer_len_tokens))
        # update global format counts
        self._parsing_stats.total_count += 1
        if not has_reasoning:
            self._parsing_stats.missing_reasoning_count += 1
        if not has_answer:
            self._parsing_stats.missing_answer_count += 1
        if is_malformed:
            self._parsing_stats.malformed_count += 1
        # update category-wise stats if category extractor configured
        if self._category_extractor is not None:
            sample_data_dict = sample_ctx.sample_data._asdict()
            categories = self._category_extractor.extract_categories(sample_data_dict)
            for category in categories:
                # output length (chars)
                if category not in self._parsing_stats.category_output_length_chars:
                    self._parsing_stats.category_output_length_chars[category] = stats_utils.RunningStats()
                self._parsing_stats.category_output_length_chars[category].update(float(output_len_chars))
                # reasoning length (chars)
                if reasoning_len_chars is not None:
                    if category not in self._parsing_stats.category_reasoning_length_chars:
                        self._parsing_stats.category_reasoning_length_chars[category] = stats_utils.RunningStats()
                    self._parsing_stats.category_reasoning_length_chars[category].update(float(reasoning_len_chars))
                # answer length (chars)
                if answer_len_chars is not None:
                    if category not in self._parsing_stats.category_answer_length_chars:
                        self._parsing_stats.category_answer_length_chars[category] = stats_utils.RunningStats()
                    self._parsing_stats.category_answer_length_chars[category].update(float(answer_len_chars))
                # output length (tokens)
                if output_len_tokens is not None:
                    if category not in self._parsing_stats.category_output_length_tokens:
                        self._parsing_stats.category_output_length_tokens[category] = stats_utils.RunningStats()
                    self._parsing_stats.category_output_length_tokens[category].update(float(output_len_tokens))
                # reasoning length (tokens)
                if reasoning_len_tokens is not None:
                    if category not in self._parsing_stats.category_reasoning_length_tokens:
                        self._parsing_stats.category_reasoning_length_tokens[category] = stats_utils.RunningStats()
                    self._parsing_stats.category_reasoning_length_tokens[category].update(float(reasoning_len_tokens))
                # answer length (tokens)
                if answer_len_tokens is not None:
                    if category not in self._parsing_stats.category_answer_length_tokens:
                        self._parsing_stats.category_answer_length_tokens[category] = stats_utils.RunningStats()
                    self._parsing_stats.category_answer_length_tokens[category].update(float(answer_len_tokens))
                # category counts
                self._parsing_stats.category_total_count[category] = (
                    self._parsing_stats.category_total_count.get(category, 0) + 1
                )
                if not has_reasoning:
                    self._parsing_stats.category_missing_reasoning_count[category] = (
                        self._parsing_stats.category_missing_reasoning_count.get(category, 0) + 1
                    )
                if not has_answer:
                    self._parsing_stats.category_missing_answer_count[category] = (
                        self._parsing_stats.category_missing_answer_count.get(category, 0) + 1
                    )
                if is_malformed:
                    self._parsing_stats.category_malformed_count[category] = (
                        self._parsing_stats.category_malformed_count.get(category, 0) + 1
                    )

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
        cached = self._cached_token_lengths
        metrics: dict[str, reward_types.MetricValue] = {
            "parsing/output_length_chars": len(parsed.raw),
        }
        # add token length if token tracking is enabled
        if cached is not None:
            metrics["parsing/output_length_tokens"] = cached[0]  # type: ignore[arg-type]
        # is_malformed only meaningful when capture_diagnostics is enabled
        if self._config.parsing and self._config.parsing.capture_diagnostics:
            metrics["parsing/is_malformed"] = self._is_malformed(parsed)
        if reasoning_enabled:
            has_reasoning = parsed.reasoning is not None
            metrics["parsing/has_reasoning"] = has_reasoning
            if has_reasoning:
                metrics["parsing/reasoning_length_chars"] = len(parsed.reasoning)  # type: ignore[arg-type]
                if cached is not None:
                    metrics["parsing/reasoning_length_tokens"] = cached[1]  # type: ignore[arg-type]
        if answer_enabled:
            has_answer = parsed.final_answer is not None
            metrics["parsing/has_answer"] = has_answer
            if has_answer:
                metrics["parsing/answer_length_chars"] = len(parsed.final_answer)  # type: ignore[arg-type]
                if cached is not None:
                    metrics["parsing/answer_length_tokens"] = cached[2]  # type: ignore[arg-type]
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
        if self._config.logging.main_process_only and not self._is_main_process():
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
        # add parsing metrics after scoping (fixed parsing/ prefix, not scoped)
        if self._config.logging.log_metrics:
            if self._config.parsing is not None and sample_ctx.parsed is not None:
                parsing_metrics = self._compute_sample_parsing_metrics(sample_ctx)
                scoped_metrics.update(parsing_metrics)
            # add category labels for filtering (not scoped)
            if self._category_extractor is not None:
                sample_data_dict = sample_ctx.sample_data._asdict()
                categories = self._category_extractor.extract_categories(sample_data_dict)
                for category in categories:
                    scoped_metrics[f"categories/{category}"] = True
        self._logger.log(
            sample_id,
            total=total_to_log,
            terms=scoped_terms,
            metrics=scoped_metrics,
            step=step_to_use,
            prompt=sample_ctx.prompt,
            model_output=sample_ctx.model_output,
        )

    @staticmethod
    def _is_main_process() -> bool:
        """Return whether this process is the main distributed rank (or non-distributed)."""
        return pyine.utils.distrib.is_main_process()

    @staticmethod
    def _is_distributed() -> bool:
        """Return whether this run appears to be distributed."""
        return pyine.utils.distrib.is_distributed()

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
