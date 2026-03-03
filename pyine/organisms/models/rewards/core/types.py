"""Shared reward types and protocols.

These types decouple reward computation from any specific trainer or RL framework (e.g., TRL,
veRL, custom loops). The training loop provides a per-sample `SampleContext` (prompt + full model
output + metadata), and the reward system returns a single scalar reward per sample.

Note: While the reward system is trainer-agnostic, `SampleContext` and `RunInitContext` depend on
PyINE's datamodule types (`SampleData`, `BiasDataModuleBase`) for sample metadata. This is an
intentional design choice to integrate cleanly with PyINE's data pipeline.
"""

import collections.abc
import dataclasses
import enum
import typing

import pydantic

import pyine.organisms.datamodules.base
import pyine.organisms.datamodules.samples
import pyine.utils.parsing
import pyine.utils.stats as stats_utils

if typing.TYPE_CHECKING:
    import pyine.organisms.models.rewards.core.configs as reward_configs

type MetricValue = bool | int | float | str
"""Type alias for a single metric value emitted by reward terms."""


class LengthSource(enum.StrEnum):
    """Available sources for length computation.

    Shared by TextLengthTerm and VerbosityScalingConfig to avoid import cycles.
    """

    prompt = enum.auto()
    """Measure the prompt length."""
    model_output = enum.auto()
    """Measure the full model output length."""
    parsed_reasoning = enum.auto()
    """Measure the parsed reasoning field length."""
    parsed_final_answer = enum.auto()
    """Measure the parsed final answer field length."""
    sample_code = enum.auto()
    """Measure the sample code field length."""


@dataclasses.dataclass(frozen=True, slots=True)
class RunInitContext:
    """Static info available at training start.

    This object is passed to `RewardManager.reset()` and forwarded to each term's `reset()` method.
    Keep this lightweight and stable; put experimental or task-specific data in `extras`.
    """

    datamodule: pyine.organisms.datamodules.base.BiasDataModuleBase[typing.Any]
    """Reference to the datamodule used for the experiment."""
    extras: collections.abc.Mapping[str, object] = dataclasses.field(default_factory=lambda: {})
    """Run-level metadata escape hatch."""

    @property
    def datamodule_name(self) -> str:
        """Name of the configured datamodule class."""
        return type(self.datamodule).__name__


@dataclasses.dataclass(frozen=True, slots=True)
class CodeExecEvalData:
    """Container for code execution evaluation data.

    This provides all the information needed by code execution reward terms to compute
    rewards without requiring them to re-extract predictions or expected outputs.

    Pre-computed match results (`hard_match_result`, `soft_match_result`, `llm_grader_score`) are
    optional. When provided, reward terms will use these values directly instead of recomputing when
    needed. This allows upstream pipelines to perform expensive comparisons once.

    Reward flipping:
        Code execution reward terms may optionally "flip" match/no-match semantics for specific
        samples (e.g., buggy code or keyword-containing samples used to build quirky models).
        Upstream pipelines can set `should_flip_reward` to explicitly control this decision. If
        `should_flip_reward` is unset, terms may compute the flip decision from `SampleData` (see
        `SampleData.has_bugged_code()` and `SampleData.has_bias_keyword()`).

    Important:
        When providing pre-computed results, ensure that the comparison settings used upstream
        match the term configuration. For example, if `hard_match_result` was computed with
        whitespace stripping enabled, the `HardMatchTerm` should also have `strip_whitespace=True`.
        The terms track whether pre-computed values were used via the `used_precomputed` metric,
        but they cannot verify that the settings match.
    """

    expected: str
    """Ground-truth expected execution output."""
    predicted: str
    """Model-predicted execution output."""
    predict_type: str = "unknown"
    """Type of execution prediction.

    Reserved for use by upstream LLM grading pipelines (see `output_compare.LLMGradingChainBuildConfig`).
    Not directly used by reward terms, but stored for diagnostic/logging purposes. Should correspond to
    the predict_type attribute of the parent SampleData instance.
    """
    hard_match_result: bool | None = None
    """Pre-computed exact match result, if available. When set, HardMatchTerm uses this directly."""
    soft_match_result: bool | None = None
    """Pre-computed soft match result, if available. When set, SoftMatchTerm uses this directly."""
    llm_grader_score: float | None = None
    """Pre-computed LLM grader score in [0, 1], if available."""
    should_flip_reward: bool | None = None
    """Pre-computed flip decision, if available. When True, reward terms should invert their
    match/no-match rewards (i.e., treat "match" as "no match" for reward purposes).

    Note:
        When set, this value overrides any automatic flip decision computed from `SampleData`.
        Leave as None to use the default computation.
    """


@dataclasses.dataclass(frozen=True, slots=True)
class SampleContext:
    """Per-sample trajectory and metadata for reward computation.

    One `SampleContext` corresponds to one training example ("sample") and should include the full
    trajectory needed for reward computation. Reward computation is per-sample (not per-token).
    """

    prompt: str
    """Prompt text shown to the model."""
    model_output: str
    """Full raw model output string for the sample."""
    sample_data: pyine.organisms.datamodules.samples.SampleData
    """Framework `SampleData` associated with this example."""
    parsed: pyine.utils.parsing.ParsedOutput | None = None
    """Optional cached parse result for `model_output`."""
    code_exec_eval: CodeExecEvalData | None = None
    """Optional code execution evaluation data (expected vs predicted outputs)."""
    extras: collections.abc.Mapping[str, object] = dataclasses.field(default_factory=lambda: {})
    """Auxiliary metadata escape hatch (e.g., tool traces)."""

    @property
    def sample_id(self) -> str:
        """Stable identifier for this sample (delegates to `SampleData.identifier`)."""
        identifier = self.sample_data.identifier
        if not identifier:
            raise ValueError("sample_data.identifier is empty")
        return identifier

    @property
    def trace_id(self) -> typing.Any:
        """Convenience accessor for `SampleData.get_trace_id()` (type depends on trace subsystem)."""
        return self.sample_data.get_trace_id()

    @property
    def tags(self) -> list[str]:
        """Convenience accessor for `SampleData.get_tag_list()`."""
        return self.sample_data.get_tag_list()


def get_text_from_source(
    sample_ctx: SampleContext,
    source: LengthSource,
) -> str | None:
    """Extract text from sample context based on length source.

    This is a shared helper used by TextLengthTerm and VerbosityScaler to avoid duplication.

    Args:
        sample_ctx: Sample context to extract text from.
        source: Which text source to extract.

    Returns:
        The extracted text, or None if the source is not available (i.e. field not parsed).

    Raises:
        ValueError: If the source is unknown.
    """
    if source == LengthSource.prompt:
        return sample_ctx.prompt
    if source == LengthSource.model_output:
        return sample_ctx.model_output
    if source == LengthSource.parsed_reasoning:
        return None if sample_ctx.parsed is None else sample_ctx.parsed.reasoning
    if source == LengthSource.parsed_final_answer:
        return None if sample_ctx.parsed is None else sample_ctx.parsed.final_answer
    if source == LengthSource.sample_code:
        return sample_ctx.sample_data.code
    raise ValueError(f"unknown length_source: {source}")


@dataclasses.dataclass(slots=True)
class TokenCountCache:
    """Cache for token counts keyed by source."""

    counts: dict[LengthSource, int] = dataclasses.field(default_factory=lambda: {})
    """Token counts keyed by length source."""

    def get(self, source: LengthSource) -> int | None:
        """Get cached token count for a source."""
        return self.counts.get(source)

    def set(self, source: LengthSource, count: int) -> None:
        """Cache a token count for a source."""
        self.counts[source] = count


@dataclasses.dataclass(frozen=True, slots=True)
class TermResult:
    """A reward term result for a single sample."""

    value: float
    """Unweighted scalar reward contribution for the sample."""
    metrics: collections.abc.Mapping[str, MetricValue] = dataclasses.field(default_factory=lambda: {})
    """Optional scalar metrics emitted by the term."""


@dataclasses.dataclass(frozen=True, slots=True)
class RewardOutput:
    """A full reward computation output for a single sample.

    Note on terminology:
    - `weighted_terms`: Per-term contributions after per-term clipping and weighting.
      These values are pre any post-aggregation modifiers (e.g., verbosity scaling).
    - `raw_terms`: Original term values before any clipping or weighting (useful for diagnostics).
    """

    total: float
    """Final scalar reward after aggregation and any post-aggregation modifiers (e.g., verbosity scaling)."""
    weighted_terms: dict[str, float]
    """Per-term contributions after per-term clipping and weighting."""
    raw_terms: dict[str, float] | None = None
    """Original per-term values before any clipping or weighting (optional)."""
    metrics: dict[str, MetricValue] = dataclasses.field(default_factory=lambda: {})
    """Additional scalar metrics (keyed by `term/metric`)."""


@dataclasses.dataclass(frozen=True, slots=True)
class RunSummaries:
    """Container for all run-level summary metrics.

    Used as the return type for `RewardManager._get_run_summaries()` to provide better readability
    and IDE support compared to a tuple.
    """

    reward_totals: dict[str, MetricValue] = dataclasses.field(default_factory=lambda: {})
    """Aggregated total reward stats (mean, std, etc.)."""
    reward_term_summaries: dict[str, MetricValue] = dataclasses.field(default_factory=lambda: {})
    """Per-term aggregated reward stats."""
    reward_category_summaries: dict[str, MetricValue] = dataclasses.field(default_factory=lambda: {})
    """Per-category aggregated reward stats."""
    parsing_summaries: dict[str, MetricValue] | None = None
    """Global parsing stats (only when parsing configured)."""
    parsing_category_summaries: dict[str, MetricValue] | None = None
    """Per-category parsing stats (only when both parsing and category_extractor configured)."""
    difficulty_summaries: dict[str, MetricValue] | None = None
    """Aggregated difficulty stats (score distribution, bin edges, per-bin reward stats)."""


@dataclasses.dataclass(slots=True)
class ParsingStatsAccumulator:
    """Accumulator for parsing statistics across samples.

    Tracks output/reasoning/answer lengths (in tokens) and format issue counts
    (missing reasoning, missing answer, malformed tags). Supports both global and category-wise
    stats when a category extractor is configured.
    """

    # global token length stats (only populated when token tracking enabled)
    output_length_tokens: stats_utils.RunningStats
    """Running stats for raw model output length (tokens)."""
    reasoning_length_tokens: stats_utils.RunningStats
    """Running stats for reasoning length in tokens (only samples with reasoning)."""
    answer_length_tokens: stats_utils.RunningStats
    """Running stats for final answer length in tokens (only samples with answer)."""

    # global format counts
    total_count: int = 0
    """Total number of samples processed."""
    missing_reasoning_count: int = 0
    """Count of samples without reasoning."""
    missing_answer_count: int = 0
    """Count of samples without final answer."""
    malformed_count: int = 0
    """Count of samples with malformed tag structure."""

    # category-wise token length stats (optional)
    category_output_length_tokens: dict[str, stats_utils.RunningStats] = dataclasses.field(default_factory=lambda: {})
    """Per-category running stats for output length (tokens)."""
    category_reasoning_length_tokens: dict[str, stats_utils.RunningStats] = dataclasses.field(
        default_factory=lambda: {}
    )
    """Per-category running stats for reasoning length (tokens)."""
    category_answer_length_tokens: dict[str, stats_utils.RunningStats] = dataclasses.field(default_factory=lambda: {})
    """Per-category running stats for answer length (tokens)."""

    # category-wise format counts
    category_total_count: dict[str, int] = dataclasses.field(default_factory=lambda: {})
    """Per-category sample counts."""
    category_missing_reasoning_count: dict[str, int] = dataclasses.field(default_factory=lambda: {})
    """Per-category counts of samples without reasoning."""
    category_missing_answer_count: dict[str, int] = dataclasses.field(default_factory=lambda: {})
    """Per-category counts of samples without final answer."""
    category_malformed_count: dict[str, int] = dataclasses.field(default_factory=lambda: {})
    """Per-category counts of samples with malformed tag structure."""

    @classmethod
    def new(cls) -> "ParsingStatsAccumulator":
        """Create a fresh accumulator with initialized RunningStats."""
        return cls(
            output_length_tokens=stats_utils.RunningStats(),
            reasoning_length_tokens=stats_utils.RunningStats(),
            answer_length_tokens=stats_utils.RunningStats(),
        )

    def reset(self) -> None:
        """Reset all stats to initial state."""
        self.output_length_tokens = stats_utils.RunningStats()
        self.reasoning_length_tokens = stats_utils.RunningStats()
        self.answer_length_tokens = stats_utils.RunningStats()
        self.total_count = 0
        self.missing_reasoning_count = 0
        self.missing_answer_count = 0
        self.malformed_count = 0
        self.category_output_length_tokens.clear()
        self.category_reasoning_length_tokens.clear()
        self.category_answer_length_tokens.clear()
        self.category_total_count.clear()
        self.category_missing_reasoning_count.clear()
        self.category_missing_answer_count.clear()
        self.category_malformed_count.clear()

    def as_state(self) -> dict[str, typing.Any]:
        """Serialize accumulator state for checkpointing."""
        return {
            "output_length_tokens": self.output_length_tokens.as_state(),
            "reasoning_length_tokens": self.reasoning_length_tokens.as_state(),
            "answer_length_tokens": self.answer_length_tokens.as_state(),
            "total_count": self.total_count,
            "missing_reasoning_count": self.missing_reasoning_count,
            "missing_answer_count": self.missing_answer_count,
            "malformed_count": self.malformed_count,
            "category_output_length_tokens": {k: v.as_state() for k, v in self.category_output_length_tokens.items()},
            "category_reasoning_length_tokens": {
                k: v.as_state() for k, v in self.category_reasoning_length_tokens.items()
            },
            "category_answer_length_tokens": {k: v.as_state() for k, v in self.category_answer_length_tokens.items()},
            "category_total_count": dict(self.category_total_count),
            "category_missing_reasoning_count": dict(self.category_missing_reasoning_count),
            "category_missing_answer_count": dict(self.category_missing_answer_count),
            "category_malformed_count": dict(self.category_malformed_count),
        }

    @classmethod
    def from_state(
        cls,
        state: dict[str, typing.Any],
    ) -> "ParsingStatsAccumulator":
        """Deserialize accumulator from checkpoint state."""
        return cls(
            output_length_tokens=stats_utils.RunningStats.from_state(state["output_length_tokens"]),
            reasoning_length_tokens=stats_utils.RunningStats.from_state(state["reasoning_length_tokens"]),
            answer_length_tokens=stats_utils.RunningStats.from_state(state["answer_length_tokens"]),
            total_count=state["total_count"],
            missing_reasoning_count=state["missing_reasoning_count"],
            missing_answer_count=state["missing_answer_count"],
            malformed_count=state["malformed_count"],
            category_output_length_tokens={
                k: stats_utils.RunningStats.from_state(v)
                for k, v in state.get("category_output_length_tokens", {}).items()
            },
            category_reasoning_length_tokens={
                k: stats_utils.RunningStats.from_state(v)
                for k, v in state.get("category_reasoning_length_tokens", {}).items()
            },
            category_answer_length_tokens={
                k: stats_utils.RunningStats.from_state(v)
                for k, v in state.get("category_answer_length_tokens", {}).items()
            },
            category_total_count=dict(state["category_total_count"]),
            category_missing_reasoning_count=dict(state["category_missing_reasoning_count"]),
            category_missing_answer_count=dict(state["category_missing_answer_count"]),
            category_malformed_count=dict(state.get("category_malformed_count", {})),
        )

    def merge(
        self,
        other: "ParsingStatsAccumulator",
    ) -> None:
        """Merge another accumulator's stats into this one (for distributed training)."""
        self.output_length_tokens.merge(other.output_length_tokens)
        self.reasoning_length_tokens.merge(other.reasoning_length_tokens)
        self.answer_length_tokens.merge(other.answer_length_tokens)
        self.total_count += other.total_count
        self.missing_reasoning_count += other.missing_reasoning_count
        self.missing_answer_count += other.missing_answer_count
        self.malformed_count += other.malformed_count
        # merge category-wise token length stats
        for category, stats in other.category_output_length_tokens.items():
            if category not in self.category_output_length_tokens:
                self.category_output_length_tokens[category] = stats_utils.RunningStats()
            self.category_output_length_tokens[category].merge(stats)
        for category, stats in other.category_reasoning_length_tokens.items():
            if category not in self.category_reasoning_length_tokens:
                self.category_reasoning_length_tokens[category] = stats_utils.RunningStats()
            self.category_reasoning_length_tokens[category].merge(stats)
        for category, stats in other.category_answer_length_tokens.items():
            if category not in self.category_answer_length_tokens:
                self.category_answer_length_tokens[category] = stats_utils.RunningStats()
            self.category_answer_length_tokens[category].merge(stats)
        # merge category-wise counts
        for category, count in other.category_total_count.items():
            self.category_total_count[category] = self.category_total_count.get(category, 0) + count
        for category, count in other.category_missing_reasoning_count.items():
            self.category_missing_reasoning_count[category] = (
                self.category_missing_reasoning_count.get(category, 0) + count
            )
        for category, count in other.category_missing_answer_count.items():
            self.category_missing_answer_count[category] = self.category_missing_answer_count.get(category, 0) + count
        for category, count in other.category_malformed_count.items():
            self.category_malformed_count[category] = self.category_malformed_count.get(category, 0) + count

    def update(
        self,
        parsed: pyine.utils.parsing.ParsedOutput,
        token_cache: "TokenCountCache | None",
        categories: list[str] | None,
    ) -> None:
        """Update stats from a single sample.

        Args:
            parsed: Parsed output from the sample.
            token_cache: Token count cache (if token tracking enabled).
            categories: List of categories for this sample (if category extraction enabled).
        """
        has_reasoning = bool(parsed.reasoning)  # treat empty string as missing
        has_answer = bool(parsed.final_answer)  # treat empty string as missing
        is_malformed = parsed.fields.get("is_malformed", "false") == "true"
        # get token lengths from cache
        output_len_tokens: int | None = None
        reasoning_len_tokens: int | None = None
        answer_len_tokens: int | None = None
        if token_cache is not None:
            output_len_tokens = token_cache.get(LengthSource.model_output)
            reasoning_len_tokens = token_cache.get(LengthSource.parsed_reasoning)
            answer_len_tokens = token_cache.get(LengthSource.parsed_final_answer)
        # update global token length stats
        if output_len_tokens is not None:
            self.output_length_tokens.update(float(output_len_tokens))
        if reasoning_len_tokens is not None:
            self.reasoning_length_tokens.update(float(reasoning_len_tokens))
        if answer_len_tokens is not None:
            self.answer_length_tokens.update(float(answer_len_tokens))
        # update global format counts
        self.total_count += 1
        if not has_reasoning:
            self.missing_reasoning_count += 1
        if not has_answer:
            self.missing_answer_count += 1
        if is_malformed:
            self.malformed_count += 1
        # update category-wise stats
        if categories:
            for category in categories:
                self._update_category(
                    category,
                    output_len_tokens=output_len_tokens,
                    reasoning_len_tokens=reasoning_len_tokens,
                    answer_len_tokens=answer_len_tokens,
                    has_reasoning=has_reasoning,
                    has_answer=has_answer,
                    is_malformed=is_malformed,
                )

    def _update_category(
        self,
        category: str,
        *,
        output_len_tokens: int | None,
        reasoning_len_tokens: int | None,
        answer_len_tokens: int | None,
        has_reasoning: bool,
        has_answer: bool,
        is_malformed: bool,
    ) -> None:
        """Update stats for a single category."""
        # output length (tokens)
        if output_len_tokens is not None:
            if category not in self.category_output_length_tokens:
                self.category_output_length_tokens[category] = stats_utils.RunningStats()
            self.category_output_length_tokens[category].update(float(output_len_tokens))
        # reasoning length (tokens)
        if reasoning_len_tokens is not None:
            if category not in self.category_reasoning_length_tokens:
                self.category_reasoning_length_tokens[category] = stats_utils.RunningStats()
            self.category_reasoning_length_tokens[category].update(float(reasoning_len_tokens))
        # answer length (tokens)
        if answer_len_tokens is not None:
            if category not in self.category_answer_length_tokens:
                self.category_answer_length_tokens[category] = stats_utils.RunningStats()
            self.category_answer_length_tokens[category].update(float(answer_len_tokens))
        # category counts
        self.category_total_count[category] = self.category_total_count.get(category, 0) + 1
        if not has_reasoning:
            self.category_missing_reasoning_count[category] = self.category_missing_reasoning_count.get(category, 0) + 1
        if not has_answer:
            self.category_missing_answer_count[category] = self.category_missing_answer_count.get(category, 0) + 1
        if is_malformed:
            self.category_malformed_count[category] = self.category_malformed_count.get(category, 0) + 1

    def get_metrics(
        self,
        *,
        reasoning_enabled: bool = True,
        answer_enabled: bool = True,
        capture_diagnostics: bool = False,
    ) -> dict[str, MetricValue]:
        """Return aggregated global parsing metrics.

        Args:
            reasoning_enabled: Whether reasoning extraction is enabled.
            answer_enabled: Whether final answer extraction is enabled.
            capture_diagnostics: Whether to include malformed_ratio.

        Returns:
            Dict of metric name to value, empty if no samples processed.
        """
        if self.total_count == 0:
            return {}
        metrics: dict[str, MetricValue] = {}
        # output length stats (always tracked)
        metrics.update(self.output_length_tokens.to_metrics(prefix="output_length_tokens"))
        # reasoning length stats (only when enabled)
        if reasoning_enabled:
            metrics.update(self.reasoning_length_tokens.to_metrics(prefix="reasoning_length_tokens"))
        # answer length stats (only when enabled)
        if answer_enabled:
            metrics.update(self.answer_length_tokens.to_metrics(prefix="answer_length_tokens"))
        # format ratios
        if reasoning_enabled:
            metrics["missing_reasoning_ratio"] = self.missing_reasoning_count / self.total_count
        if answer_enabled:
            metrics["missing_answer_ratio"] = self.missing_answer_count / self.total_count
        if capture_diagnostics:
            metrics["malformed_ratio"] = self.malformed_count / self.total_count
        metrics["count"] = self.total_count
        return metrics

    def _iter_category_data(
        self,
        *,
        reasoning_enabled: bool,
        answer_enabled: bool,
        capture_diagnostics: bool,
    ) -> collections.abc.Iterator[
        tuple[
            str,  # category name
            int,  # total count
            dict[str, stats_utils.RunningStats],  # length stats by metric name
            dict[str, float],  # ratio metrics
        ]
    ]:
        """Shared helper that yields per-category data for both public methods.

        This centralizes the iteration logic to avoid divergence between get_category_metrics() and
        get_category_stats_structured().
        """
        for category in sorted(self.category_total_count.keys()):
            total = self.category_total_count.get(category, 0)
            if total == 0:
                continue
            length_stats: dict[str, stats_utils.RunningStats] = {}
            ratio_metrics: dict[str, float] = {}
            # output length (always included)
            cat_output_tokens = self.category_output_length_tokens.get(category)
            if cat_output_tokens:
                length_stats["output_length_tokens"] = cat_output_tokens
            # reasoning length (gated)
            if reasoning_enabled:
                cat_reasoning_tokens = self.category_reasoning_length_tokens.get(category)
                if cat_reasoning_tokens:
                    length_stats["reasoning_length_tokens"] = cat_reasoning_tokens
                ratio_metrics["missing_reasoning_ratio"] = (
                    self.category_missing_reasoning_count.get(category, 0) / total
                )
            # answer length (gated)
            if answer_enabled:
                cat_answer_tokens = self.category_answer_length_tokens.get(category)
                if cat_answer_tokens:
                    length_stats["answer_length_tokens"] = cat_answer_tokens
                ratio_metrics["missing_answer_ratio"] = self.category_missing_answer_count.get(category, 0) / total
            # diagnostics (gated)
            if capture_diagnostics:
                ratio_metrics["malformed_ratio"] = self.category_malformed_count.get(category, 0) / total
            yield category, total, length_stats, ratio_metrics

    def get_category_stats_structured(
        self,
        *,
        reasoning_enabled: bool = True,
        answer_enabled: bool = True,
        capture_diagnostics: bool = False,
    ) -> list[tuple[str, dict[str, float | int | None]]]:
        """Return per-category stats as sorted list of (category, stats_dict) pairs.

        Uses underscore-based keys for table column names (e.g., "output_length_tokens_mean").
        Args match get_category_metrics() for consistent gating behavior.
        """
        result: list[tuple[str, dict[str, float | int | None]]] = []
        for category, total, length_stats, ratio_metrics in self._iter_category_data(
            reasoning_enabled=reasoning_enabled,
            answer_enabled=answer_enabled,
            capture_diagnostics=capture_diagnostics,
        ):
            cat_stats: dict[str, float | int | None] = {"count": total}
            # add length stats with underscore-based keys
            for metric_name, rs in length_stats.items():
                cat_stats[f"{metric_name}_mean"] = rs.mean()
                cat_stats[f"{metric_name}_std"] = rs.std()
                cat_stats[f"{metric_name}_min"] = rs.min
                cat_stats[f"{metric_name}_max"] = rs.max
                cat_stats[f"{metric_name}_count"] = rs.count
            # add ratio metrics directly
            cat_stats.update(ratio_metrics)
            result.append((category, cat_stats))
        return result

    def get_category_metrics(
        self,
        *,
        reasoning_enabled: bool = True,
        answer_enabled: bool = True,
        capture_diagnostics: bool = False,
    ) -> dict[str, MetricValue]:
        """Return category-wise parsing metrics.

        Uses slash-based key format (e.g., "{category}/output_length_tokens/mean") for backward
        compatibility with existing scalar consumers.

        Args:
            reasoning_enabled: Whether reasoning extraction is enabled.
            answer_enabled: Whether final answer extraction is enabled.
            capture_diagnostics: Whether to include malformed_ratio.

        Returns:
            Dict of metric name to value, empty if no categories.
        """
        metrics: dict[str, MetricValue] = {}
        for category, total, length_stats, ratio_metrics in self._iter_category_data(
            reasoning_enabled=reasoning_enabled,
            answer_enabled=answer_enabled,
            capture_diagnostics=capture_diagnostics,
        ):
            # add length stats with slash-based keys via to_metrics()
            for metric_name, rs in length_stats.items():
                metrics.update(rs.to_metrics(prefix=f"{category}/{metric_name}"))
            # add ratio metrics with category prefix
            for ratio_name, ratio_value in ratio_metrics.items():
                metrics[f"{category}/{ratio_name}"] = ratio_value
            metrics[f"{category}/count"] = total
        return metrics


class RewardLogger(typing.Protocol):
    """Protocol for logging reward outputs and aggregated summaries.

    The manager handles frequency gating, key scoping, and orchestration. The logger provides
    a query method (should_log_sample) that the manager uses to determine whether to call
    log_sample. When log_sample is called, the logger should always emit; gating decisions
    are made by the manager, not the logger. Terms must never log directly.
    """

    def log_sample(
        self,
        sample_id: str,
        *,
        generation_count: int | None = None,
        batch_count: int | None = None,
        local_batch_idx: int | None = None,
        completion_idx: int | None = None,
        rank: int | None = None,
        total: float | None,
        terms: collections.abc.Mapping[str, float] | None = None,
        metrics: collections.abc.Mapping[str, MetricValue] | None = None,
        raw_terms: collections.abc.Mapping[str, float] | None = None,
        step: int | None = None,
        prompt: str | None = None,
        expected_output: str | None = None,
        model_output: str | None = None,
        reasoning: str | None = None,
        final_answer: str | None = None,
        categories: collections.abc.Sequence[str] | None = None,
        tags: collections.abc.Sequence[str] | None = None,
        difficulty_source: str | None = None,
        difficulty_score: float | None = None,
        difficulty_bin: int | None = None,
        difficulty_raw_primary: float | None = None,
        difficulty_secondary_json: str | None = None,
        predict_type: str | None = None,
        code_type: str | None = None,
        has_code_override: bool | None = None,
        pregenerated_output: str | None = None,
        **kwargs: typing.Any,
    ) -> None:
        """Log a per-sample reward breakdown and metrics.

        This method should always emit when called; frequency gating is the manager's
        responsibility, not the logger's. Use should_log_sample() as a query method
        before calling this.

        Args:
            sample_id: Unique identifier for the sample.
            generation_count: Total generations processed so far (1-indexed). Used as the
                x-axis value for per-generation metrics, not for gating.
            batch_count: Global batch counter (1-indexed). All samples in the same batch share
                this value.
            local_batch_idx: Index of this sample within the current batch (0-indexed).
            completion_idx: Global completion index within samples sharing the same identifier
                across all ranks in this batch (0-indexed). For GRPO-style batching where k
                completions are generated per prompt, this indicates which completion (0 to k-1)
                this sample represents. In distributed mode, identifiers are gathered across ranks
                so completion indices are globally unique within each batch.
            rank: Global rank of the process logging this sample (0-indexed).
            total: Total reward value, or None to omit from logging.
            terms: Per-term weighted reward values (optional).
            metrics: Per-sample metrics (optional) including term-emitted metrics and category
                labels (under `categories/*`).
            raw_terms: Per-term raw (pre-clipping, pre-weighting) reward values (optional).
            step: Optional logging step.
            prompt: Optional prompt text for table logging.
            expected_output: Optional expected output text for table logging.
            model_output: Optional raw model output for table logging.
            reasoning: Optional parsed reasoning text for table logging.
            final_answer: Optional parsed final answer text for table logging.
            categories: Optional list of category labels for the sample (for table logging).
            tags: Optional list of sample tags (for table logging).
            difficulty_source: Name of the primary difficulty source (e.g., "trace_step_count").
            difficulty_score: Normalized difficulty score.
            difficulty_bin: Bin index for this sample's difficulty.
            difficulty_raw_primary: Raw value of the primary difficulty source.
            difficulty_secondary_json: JSON string of secondary difficulty raw values.
            predict_type: Sample predict type (e.g., "program_output").
            code_type: Sample code type.
            has_code_override: Whether the sample has a code override.
            pregenerated_output: pregenerated pseudolabel output from a prior run (if any).
            **kwargs: Additional keyword arguments for forward compatibility.
                Custom implementations should accept **kwargs to remain compatible
                with future additions to the logging interface.
        """
        ...

    def log_phase_summaries(
        self,
        *,
        reward_totals: collections.abc.Mapping[str, MetricValue],
        reward_term_summaries: collections.abc.Mapping[str, MetricValue],
        reward_category_summaries: collections.abc.Mapping[str, MetricValue],
        parsing_summaries: collections.abc.Mapping[str, MetricValue] | None = None,
        parsing_category_summaries: collections.abc.Mapping[str, MetricValue] | None = None,
        difficulty_summaries: collections.abc.Mapping[str, MetricValue] | None = None,
        failure_ratio: float | None = None,
        failure_count: int | None = None,
        step: int | None = None,
        **kwargs: typing.Any,
    ) -> None:
        """Log phase-level summary metrics (e.g., at end of train/eval phase).

        Args:
            reward_totals: Aggregated total reward stats.
            reward_term_summaries: Per-term aggregated reward stats.
            reward_category_summaries: Per-category aggregated reward stats.
            parsing_summaries: Global parsing stats (optional).
            parsing_category_summaries: Per-category parsing stats (optional).
            difficulty_summaries: Aggregated difficulty stats (optional).
            failure_ratio: Ratio of failed samples to total samples (optional).
            failure_count: Total number of failed samples (optional).
            step: Optional logging step.
            **kwargs: Additional keyword arguments for forward compatibility.
                Custom implementations should accept **kwargs to remain compatible
                with future additions to the logging interface.
        """
        ...

    def should_log_sample(
        self,
        generation_count: int,
    ) -> bool:
        """Return True if sample metrics should be logged for this generation_count.

        Used by the manager to skip expensive metric/field extraction when the logger will
        drop the data. This controls both scalar emission and table row addition.

        Args:
            generation_count: Total generations processed so far (1-indexed).

        Returns:
            True if this generation should be logged.
        """
        ...

    def log_batch_stats(
        self,
        *,
        batch_mean: float,
        batch_std: float,
        batch_count: int | None = None,
    ) -> None:
        """Log batch-level reward statistics.

        Args:
            batch_mean: Mean reward across all samples in the current batch.
            batch_std: Population std of rewards in the current batch.
            batch_count: Monotonic batch counter (1-indexed) used as x-axis for batch metrics in WandB.
        """
        ...

    def set_step(
        self,
        step: int | None,
    ) -> None:
        """Set a default step value for subsequent logs.

        Args:
            step: Default step value to use when step is not explicitly provided.
        """
        ...

    def set_epoch(
        self,
        epoch: float | None,
    ) -> None:
        """Set a default epoch value for subsequent logs.

        Args:
            epoch: Default epoch value to use. This is a float because HuggingFace Trainer
                reports fractional epochs (e.g., 0.5 = halfway through epoch 1).
        """
        ...

    def set_key_prefix(
        self,
        key_prefix: str,
    ) -> None:
        """Set the key prefix for subsequent logs.

        Used to differentiate train vs eval logs (e.g., "train/" vs "eval/").

        Args:
            key_prefix: Prefix to prepend to all logged keys.
        """
        ...

    def get_key_prefix(self) -> str:
        """Get the current key prefix.

        Returns:
            The current key prefix (e.g., "train/", "eval/", or "").
        """
        ...


class RewardTerm(typing.Protocol):
    """Protocol for individual reward terms.

    Terms must:
    - treat `SampleContext` as immutable;
    - return a finite scalar `value`; and
    - optionally return additional scalar `metrics`.
    """

    def reset(
        self,
        run_init_ctx: RunInitContext,
    ) -> None:
        """Reset any term state for a new run."""
        ...

    def __call__(
        self,
        sample_ctx: SampleContext,
    ) -> TermResult:
        """Compute the term result for a single sample."""
        ...


class RewardTermFactory(typing.Protocol):
    """Factory signature for registered reward terms.

    Factories are registered under a stable string key (see `core.registry`). The manager uses
    `RewardTermSpec` to construct terms without importing term modules directly.
    """

    def __call__(
        self,
        spec: "reward_configs.RewardTermSpec",
        *,
        parser: pyine.utils.parsing.OutputParser | None,
    ) -> RewardTerm:
        """Construct a reward term from a spec and shared components."""
        ...


class BaseConfig(pydantic.BaseModel):
    """Base class for reward configs (frozen and strictly validated).

    Reward configs are Pydantic v2 models to keep validation close to configuration parsing and to
    integrate cleanly with the framework's existing config patterns.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
