"""Reward manager implementation."""

import collections.abc
import inspect
import math
import typing
import warnings

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
    ) -> None:
        """Create a RewardManager from configuration.

        Args:
            config: Manager configuration (terms + aggregation + logging + parsing).
            parser: Optional parser override. When provided, takes precedence over `config.parsing`.
            logger: Optional logger implementation for reward logging.
            registry: Optional registry snapshot to resolve term factories (defaults to global registry).

        Raises:
            KeyError: If any term spec references an unknown registry type.
            TypeError: If a term factory has an incompatible signature.
            ValueError: If config validation fails (e.g., duplicate term names) or term construction fails.
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
        self._warn_tag_inconsistencies()

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

    def get_total_metrics(self) -> dict[str, float]:
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

    def get_term_metrics(self) -> dict[str, float]:
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

    def get_category_metrics(self) -> dict[str, float]:
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

    def reset_accumulators(
        self,
    ) -> None:
        """Reset all statistics accumulators (total, term, category).

        This resets only the running statistics, not term state. Use `reset()` to also
        reset term state for a new run.
        """
        self._total_stats = stats_utils.RunningStats()
        for key in self._term_stats:
            self._term_stats[key] = stats_utils.RunningStats()
        self._category_stats.clear()

    def flush_stats(
        self,
        step: int | None = None,
    ) -> None:
        """Log all accumulated statistics and reset accumulators.

        This logs total, per-term, and category-wise metrics, then resets all accumulators.
        Typically called by a callback after an evaluation phase completes.

        Respects distributed settings from LoggingConfig:
        - barrier_before_finalize: sync before logging
        - gather_distributed_summaries: merge stats across ranks
        - main_process_only: only log on rank 0

        Args:
            step: Optional logging step override (defaults to manager step).
        """
        if self._logger is None or not self._config.logging.enabled:
            self.reset_accumulators()
            return
        log_step = step if step is not None else self._step
        if self._total_stats.count == 0:
            self.reset_accumulators()
            return

        if self._config.logging.barrier_before_finalize and self._is_distributed():
            pyine.utils.distrib.barrier()

        totals, term_summaries, category_summaries = self._get_run_summaries()
        if self._config.logging.gather_distributed_summaries and self._is_distributed():
            totals, term_summaries, category_summaries = self._gather_run_summaries(
                totals, term_summaries, category_summaries
            )

        if self._config.logging.main_process_only and not self._is_main_process():
            self.reset_accumulators()
            return

        # add scope_prefix to all keys before logging (e.g., "reward/" -> "reward/mean")
        scope_prefix = parsing_utils.normalize_path_prefix(self._config.logging.scope_prefix)
        prefixed_totals = {f"{scope_prefix}{k}": v for k, v in totals.items()}
        prefixed_terms = {f"{scope_prefix}{k}": v for k, v in term_summaries.items()}
        prefixed_categories = {f"{scope_prefix}{k}": v for k, v in category_summaries.items()}

        if hasattr(self._logger, "log_run"):
            self._logger.log_run(
                totals=prefixed_totals,
                term_summaries=prefixed_terms,
                category_summaries=prefixed_categories,
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
        """Optionally log run-level summaries.

        Args:
            log: If False, suppress run-level logging even if enabled in config.
            step: Optional logging step override (defaults to manager step).
        """
        if not log or not self._config.logging.enabled:
            return
        if self._logger is None:
            raise ValueError("logging is enabled but no logger is configured")
        step_to_use = self._step if step is None else step
        if self._total_stats.count == 0:
            return

        if self._config.logging.barrier_before_finalize and self._is_distributed():
            pyine.utils.distrib.barrier()

        totals, term_summaries, category_summaries = self._get_run_summaries()
        if self._config.logging.gather_distributed_summaries and self._is_distributed():
            totals, term_summaries, category_summaries = self._gather_run_summaries(
                totals, term_summaries, category_summaries
            )

        if self._config.logging.main_process_only and not self._is_main_process():
            return
        scoped_totals, scoped_term_summaries, scoped_category_summaries = self._scope_run_fields(
            totals, term_summaries, category_summaries
        )
        self._logger.log_run(
            totals=scoped_totals,
            term_summaries=scoped_term_summaries,
            category_summaries=scoped_category_summaries,
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

    def _get_run_summaries(self) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        """Compute local run-level summaries (mean/min/max/std) for total, per-term, and category values.

        Returns a tuple of (total_stats, per_term_stats, per_category_stats) dicts.
        """
        if self._total_stats.count == 0:
            return {}, {}, {}
        total_summaries = self.get_total_metrics()
        term_summaries = self.get_term_metrics()
        category_summaries = self.get_category_metrics()
        return total_summaries, term_summaries, category_summaries

    def _gather_run_summaries(
        self,
        totals: dict[str, float],
        term_summaries: dict[str, float],
        categories_summaries: dict[str, float],
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        """Gather run stats across ranks and compute global summaries (rank0 returns merged).

        Note: Only rank 0 receives the merged summaries. Non-main ranks return their original
        local summaries unchanged. If `main_process_only=False` and `gather_distributed_summaries=True`,
        non-main ranks will still log their local (unmerged) summaries to avoid blocking, which
        may be surprising. For consistent logging, keep `main_process_only=True` (the default).

        Returns a tuple of (total_stats, per_term_stats, per_category_stats) dicts.
        """
        payload = {
            "total": self._total_stats.as_state(),
            "terms": {name: stats.as_state() for name, stats in self._term_stats.items()},
            "categories": {name: stats.as_state() for name, stats in self._category_stats.items()},
        }
        gathered = pyine.utils.distrib.all_gather_objects(payload)
        if not self._is_main_process():
            return totals, term_summaries, categories_summaries
        merged_total = stats_utils.RunningStats()
        merged_terms: dict[str, stats_utils.RunningStats] = {
            name: stats_utils.RunningStats() for name in self._term_stats
        }
        merged_categories: dict[str, stats_utils.RunningStats] = {}
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
        self._total_stats = merged_total
        self._term_stats = merged_terms
        self._category_stats = merged_categories
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
        scoped_terms, scoped_metrics = self._scope_sample_fields(terms, metrics)
        self._logger.log(sample_id, total=total_to_log, terms=scoped_terms, metrics=scoped_metrics, step=step_to_use)

    @staticmethod
    def _is_main_process() -> bool:
        """Return whether this process is the main distributed rank (or non-distributed)."""
        return pyine.utils.distrib.is_main_process()

    @staticmethod
    def _is_distributed() -> bool:
        """Return whether this run appears to be distributed."""
        return pyine.utils.distrib.is_distributed()

    def _scope_sample_fields(
        self,
        terms: collections.abc.Mapping[str, float],
        metrics: collections.abc.Mapping[str, reward_types.MetricValue],
    ) -> tuple[dict[str, float], dict[str, reward_types.MetricValue]]:
        """Apply the configured scope prefix to per-sample term/metric keys."""
        prefix_norm = parsing_utils.normalize_path_prefix(self._config.logging.scope_prefix)
        if not prefix_norm:
            return dict(terms), dict(metrics)
        return (
            {f"{prefix_norm}terms/{k}": float(v) for k, v in terms.items()},
            {f"{prefix_norm}metrics/{k}": v for k, v in metrics.items()},
        )

    def _scope_run_fields(
        self,
        totals: collections.abc.Mapping[str, float],
        term_summaries: collections.abc.Mapping[str, float],
        category_summaries: collections.abc.Mapping[str, float],
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        """Apply the configured scope prefix to run-level summary keys."""
        prefix_norm = parsing_utils.normalize_path_prefix(self._config.logging.scope_prefix)
        if not prefix_norm:
            return dict(totals), dict(term_summaries), dict(category_summaries)
        return (
            {f"{prefix_norm}run/{k}": float(v) for k, v in totals.items()},
            {f"{prefix_norm}run/terms/{k}": float(v) for k, v in term_summaries.items()},
            {f"{prefix_norm}run/categories/{k}": float(v) for k, v in category_summaries.items()},
        )
