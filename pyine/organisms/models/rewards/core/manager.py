"""Reward manager implementation."""

import collections.abc
import dataclasses
import inspect
import math
import typing
import warnings

import pyine.organisms.models.rewards.core.aggregator as reward_aggregator
import pyine.organisms.models.rewards.core.configs as reward_configs
import pyine.organisms.models.rewards.core.parser as reward_parser
import pyine.organisms.models.rewards.core.registry as reward_registry
import pyine.organisms.models.rewards.core.types as reward_types
import pyine.utils.stats as stats_utils
import pyine.utils.strings as strings_utils


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
            Example: `[("format", "parseable_answer", 0.5), ("length", "text_length", 0.5)]`
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

        # compute rewards (sample_data required from datamodule)
        ctx = rewards.SampleContext(
            prompt="...",
            model_output="<final>answer</final>",
            sample_data=sample_data,
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
    - optionally parses output once per sample and shares it across terms;
    - aggregates per-term scalar values into a single per-sample reward;
    - optionally logs per-sample breakdown/metrics and run-level summaries.

    Quality-of-life features:
    - enforces `RewardTermSpec.require_parsed` to avoid silent `parsed=None` behavior;
    - supports optional step-aware logging for alignment with trainer/global steps;
    - can gather and merge run summaries across distributed ranks at finalize.

    Rewards are computed once per sample (prompt + full model output).
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
        self._warn_tag_inconsistencies()

    def _warn_tag_inconsistencies(self) -> None:
        """Warn if term configurations reference different tags than the active parser.

        This checks any term with an explicit `final_tag` parameter in its params. Terms that
        inherit `final_tag` from the parser (like `parseable_answer` when not explicitly set)
        are not warned about since they will use the correct tag.

        The parser's tag is derived from `self._parser` (the active parser instance) when it's a
        `TagsOutputParser`, falling back to `config.parsing.final_tag` otherwise.
        """
        import pyine.organisms.models.rewards.core.parser as reward_parser

        parser_final_tag: str | None = None
        if isinstance(self._parser, reward_parser.TagsOutputParser):
            parser_final_tag = self._parser.final_tag
        elif self._config.parsing is not None:
            parser_final_tag = self._config.parsing.final_tag
        if parser_final_tag is None:
            return  # no parser configured, nothing to warn about
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

        Args:
            run_init_ctx: Run-level context forwarded to term `reset()` hooks.
        """
        for term in self._terms_by_name.values():
            term.reset(run_init_ctx)
        self._step = None
        self._total_stats = stats_utils.RunningStats()
        for key in self._term_stats:
            self._term_stats[key] = stats_utils.RunningStats()

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
        if hasattr(self._logger, "set_step"):
            self._logger.set_step(step)  # type: ignore[reportUnknownMemberType]

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
            ValueError: If no reward terms are enabled.
        """
        active_specs = [spec for spec in self._config.terms if spec.enabled]
        if not active_specs:
            raise ValueError("no enabled reward terms configured")

        ctx_for_terms = self._maybe_parse(sample_ctx)

        values: dict[str, float] = {}
        metrics: dict[str, bool | int | float] = {}
        for spec in active_specs:
            term = self._terms_by_name[spec.name]
            result = term(ctx_for_terms)
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

        self._update_running_stats(output)
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
            import pyine.utils.distrib

            pyine.utils.distrib.barrier()

        totals, term_summaries = self._get_run_summaries()
        if self._config.logging.gather_distributed_summaries and self._is_distributed():
            totals, term_summaries = self._gather_run_summaries(totals, term_summaries)

        if self._config.logging.main_process_only and not self._is_main_process():
            return
        scoped_totals, scoped_term_summaries = self._scope_run_fields(totals, term_summaries)
        self._logger.log_run(totals=scoped_totals, term_summaries=scoped_term_summaries, step=step_to_use)

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

    def _maybe_parse(
        self,
        sample_ctx: reward_types.SampleContext,
    ) -> reward_types.SampleContext:
        """Parse sample output once (if configured) and return a context with `parsed` set."""
        if self._parser is None:
            return sample_ctx
        if sample_ctx.parsed is not None:
            return sample_ctx
        parsed = self._parser.parse(sample_ctx.prompt, sample_ctx.model_output)
        return dataclasses.replace(sample_ctx, parsed=parsed)

    def _get_run_summaries(self) -> tuple[dict[str, float], dict[str, float]]:
        """Compute local run-level summaries (mean/min/max/std) for total and per-term values."""
        if self._total_stats.count == 0:
            return {}, {}
        totals: dict[str, float] = {
            "count": float(self._total_stats.count),
            "mean_total": self._total_stats.mean(),
            "min_total": float(self._total_stats.min) if self._total_stats.min is not None else 0.0,
            "max_total": float(self._total_stats.max) if self._total_stats.max is not None else 0.0,
            "std_total": self._total_stats.std(),
        }
        term_summaries: dict[str, float] = {}
        for term_name, stats in self._term_stats.items():
            if stats.count == 0:
                continue
            term_summaries[f"mean/{term_name}"] = stats.mean()
            term_summaries[f"min/{term_name}"] = float(stats.min) if stats.min is not None else 0.0
            term_summaries[f"max/{term_name}"] = float(stats.max) if stats.max is not None else 0.0
            term_summaries[f"std/{term_name}"] = stats.std()
        return totals, term_summaries

    def _gather_run_summaries(
        self,
        totals: dict[str, float],
        term_summaries: dict[str, float],
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Gather run stats across ranks and compute global summaries (rank0 returns merged)."""
        import pyine.utils.distrib

        payload = {
            "total": self._total_stats.as_state(),
            "terms": {name: stats.as_state() for name, stats in self._term_stats.items()},
        }
        gathered = pyine.utils.distrib.all_gather_objects(payload)
        if not self._is_main_process():
            return totals, term_summaries
        merged_total = stats_utils.RunningStats()
        merged_terms: dict[str, stats_utils.RunningStats] = {
            name: stats_utils.RunningStats() for name in self._term_stats
        }
        for item in gathered:
            total_state = typing.cast("collections.abc.Mapping[str, int | float]", item["total"])
            merged_total.merge(stats_utils.RunningStats.from_state(total_state))
            terms_state = typing.cast(
                "collections.abc.Mapping[str, collections.abc.Mapping[str, int | float]]",
                item["terms"],
            )
            for term_name, state in terms_state.items():
                if term_name not in merged_terms:
                    merged_terms[term_name] = stats_utils.RunningStats()
                merged_terms[term_name].merge(stats_utils.RunningStats.from_state(state))
        self._total_stats = merged_total
        self._term_stats = merged_terms
        return self._get_run_summaries()

    @staticmethod
    def _validate_term_result(
        term_name: str,
        result: reward_types.TermResult,
    ) -> None:
        """Validate that a term result is numeric and finite (for safe aggregation/logging)."""
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
            if not isinstance(metric_value_any, (int, float)):
                raise TypeError(
                    f"term '{term_name}' metric '{metric_key_any}' is non-numeric: {type(metric_value_any)}"
                )
            if not math.isfinite(float(metric_value_any)):
                raise ValueError(f"term '{term_name}' metric '{metric_key_any}' is non-finite: {metric_value_any}")

    def _update_running_stats(
        self,
        output: reward_types.RewardOutput,
    ) -> None:
        """Update simple run-level summary stats for `finalize_run()`."""
        self._total_stats.update(float(output.total))
        for term_name, value in output.weighted_terms.items():
            if term_name not in self._term_stats:
                self._term_stats[term_name] = stats_utils.RunningStats()
            self._term_stats[term_name].update(float(value))

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
        metrics: dict[str, bool | int | float] = dict(output.metrics) if self._config.logging.log_metrics else {}
        total_to_log: float | None = float(output.total) if self._config.logging.log_total else None

        scoped_terms, scoped_metrics = self._scope_sample_fields(terms, metrics)
        self._logger.log(sample_id, total=total_to_log, terms=scoped_terms, metrics=scoped_metrics, step=step_to_use)

    @staticmethod
    def _is_main_process() -> bool:
        """Return whether this process is the main distributed rank (or non-distributed)."""
        import pyine.utils.distrib

        return pyine.utils.distrib.is_main_process()

    @staticmethod
    def _is_distributed() -> bool:
        """Return whether this run appears to be distributed."""
        import pyine.utils.distrib

        return pyine.utils.distrib.is_distributed()

    def _scope_sample_fields(
        self,
        terms: collections.abc.Mapping[str, float],
        metrics: collections.abc.Mapping[str, bool | int | float],
    ) -> tuple[dict[str, float], dict[str, bool | int | float]]:
        """Apply the configured scope prefix to per-sample term/metric keys."""
        prefix_norm = strings_utils.normalize_path_prefix(self._config.logging.scope_prefix)
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
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Apply the configured scope prefix to run-level summary keys."""
        prefix_norm = strings_utils.normalize_path_prefix(self._config.logging.scope_prefix)
        if not prefix_norm:
            return dict(totals), dict(term_summaries)
        return (
            {f"{prefix_norm}run/{k}": float(v) for k, v in totals.items()},
            {f"{prefix_norm}run/terms/{k}": float(v) for k, v in term_summaries.items()},
        )
