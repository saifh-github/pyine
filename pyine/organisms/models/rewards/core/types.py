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
import typing

import pydantic

import pyine.organisms.datamodules.base
import pyine.organisms.datamodules.samples
import pyine.utils.stats as stats_utils

if typing.TYPE_CHECKING:
    import pyine.organisms.models.rewards.core.configs as reward_configs

type MetricValue = bool | int | float | str
"""Type alias for a single metric value emitted by reward terms."""


@dataclasses.dataclass(frozen=True, slots=True)
class RunInitContext:
    """Static info available at training start.

    This object is passed to `RewardManager.reset()` and forwarded to each term's `reset()` method.
    Keep this lightweight and stable; put experimental or task-specific data in `extras`.
    """

    datamodule: pyine.organisms.datamodules.base.BiasDataModuleBase[typing.Any] | None = None
    """Reference to the datamodule used for the experiment."""
    extras: collections.abc.Mapping[str, object] = dataclasses.field(default_factory=lambda: dict[str, object]())
    """Run-level metadata escape hatch."""

    @property
    def datamodule_name(self) -> str | None:
        """Best-effort name for the configured datamodule (if provided)."""
        if self.datamodule is None:
            return None
        return type(self.datamodule).__name__


@dataclasses.dataclass(frozen=True, slots=True)
class ParsedOutput:
    """Structured representation of a model output string.

    `RewardManager` can parse a sample output once (via `OutputParser`) and provide it to all terms
    through `SampleContext.parsed`.
    """

    raw: str
    """Raw model output string."""
    final_answer: str | None = None
    """Extracted "final answer" field, if available."""
    reasoning: str | None = None
    """Extracted "reasoning" field, if available."""
    fields: collections.abc.Mapping[str, str] = dataclasses.field(default_factory=lambda: dict[str, str]())
    """Additional extracted string fields (term-/task-specific)."""


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

    # @@@@ TODO: should this be a base class, and derived version include sample data + code exec eval results?

    prompt: str
    """Prompt text shown to the model."""
    model_output: str
    """Full raw model output string for the sample."""
    sample_data: pyine.organisms.datamodules.samples.SampleData
    """Framework `SampleData` associated with this example."""
    parsed: ParsedOutput | None = None
    """Optional cached parse result for `model_output`."""
    code_exec_eval: CodeExecEvalData | None = None
    """Optional code execution evaluation data (expected vs predicted outputs)."""
    extras: collections.abc.Mapping[str, object] = dataclasses.field(default_factory=lambda: dict[str, object]())
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


@dataclasses.dataclass(frozen=True, slots=True)
class TermResult:
    """A reward term result for a single sample."""

    value: float
    """Unweighted scalar reward contribution for the sample."""
    metrics: collections.abc.Mapping[str, MetricValue] = dataclasses.field(
        default_factory=lambda: dict[str, MetricValue]()
    )
    """Optional scalar metrics emitted by the term."""


@dataclasses.dataclass(frozen=True, slots=True)
class RewardOutput:
    """A full reward computation output for a single sample.

    Note on terminology:
    - `weighted_terms`: Per-term contributions after per-term clipping and weighting.
    - `raw_terms`: Original term values before any clipping or weighting (useful for diagnostics).
    """

    total: float
    """Aggregated scalar reward after per-term clipping, weighting, and total clipping."""
    weighted_terms: dict[str, float]
    """Per-term contributions after per-term clipping and weighting."""
    raw_terms: dict[str, float] | None = None
    """Original per-term values before any clipping or weighting (optional)."""
    metrics: dict[str, MetricValue] = dataclasses.field(default_factory=lambda: dict[str, MetricValue]())
    """Additional scalar metrics (keyed by `term/metric`)."""


@dataclasses.dataclass(frozen=True, slots=True)
class RunSummaries:
    """Container for all run-level summary metrics.

    Used as the return type for `RewardManager._get_run_summaries()` to provide better readability
    and IDE support compared to a tuple.
    """

    reward_totals: dict[str, float] = dataclasses.field(default_factory=lambda: dict[str, float]())
    """Aggregated total reward stats (mean, std, etc.)."""
    reward_term_summaries: dict[str, float] = dataclasses.field(default_factory=lambda: dict[str, float]())
    """Per-term aggregated reward stats."""
    reward_category_summaries: dict[str, float] = dataclasses.field(default_factory=lambda: dict[str, float]())
    """Per-category aggregated reward stats."""
    parsing_summaries: dict[str, float] | None = None
    """Global parsing stats (only when parsing configured)."""
    parsing_category_summaries: dict[str, float] | None = None
    """Per-category parsing stats (only when both parsing and category_extractor configured)."""


@dataclasses.dataclass(slots=True)
class ParsingStatsAccumulator:
    """Accumulator for parsing statistics across samples.

    Tracks output/reasoning/answer lengths (in chars and optionally tokens) and format issue counts
    (missing reasoning, missing answer, malformed tags). Supports both global and category-wise
    stats when a category extractor is configured.
    """

    # global character length stats
    output_length_chars: stats_utils.RunningStats
    """Running stats for raw model output length (chars)."""
    reasoning_length_chars: stats_utils.RunningStats
    """Running stats for reasoning length in chars (only samples with reasoning)."""
    answer_length_chars: stats_utils.RunningStats
    """Running stats for final answer length in chars (only samples with answer)."""

    # global token length stats (optional, only populated when token tracking enabled)
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

    # category-wise character length stats
    category_output_length_chars: dict[str, stats_utils.RunningStats] = dataclasses.field(
        default_factory=lambda: dict[str, stats_utils.RunningStats](),
    )
    """Per-category running stats for output length (chars)."""
    category_reasoning_length_chars: dict[str, stats_utils.RunningStats] = dataclasses.field(
        default_factory=lambda: dict[str, stats_utils.RunningStats](),
    )
    """Per-category running stats for reasoning length (chars)."""
    category_answer_length_chars: dict[str, stats_utils.RunningStats] = dataclasses.field(
        default_factory=lambda: dict[str, stats_utils.RunningStats](),
    )
    """Per-category running stats for answer length (chars)."""

    # category-wise token length stats (optional)
    category_output_length_tokens: dict[str, stats_utils.RunningStats] = dataclasses.field(
        default_factory=lambda: dict[str, stats_utils.RunningStats](),
    )
    """Per-category running stats for output length (tokens)."""
    category_reasoning_length_tokens: dict[str, stats_utils.RunningStats] = dataclasses.field(
        default_factory=lambda: dict[str, stats_utils.RunningStats](),
    )
    """Per-category running stats for reasoning length (tokens)."""
    category_answer_length_tokens: dict[str, stats_utils.RunningStats] = dataclasses.field(
        default_factory=lambda: dict[str, stats_utils.RunningStats](),
    )
    """Per-category running stats for answer length (tokens)."""

    # category-wise format counts
    category_total_count: dict[str, int] = dataclasses.field(
        default_factory=lambda: dict[str, int](),
    )
    """Per-category sample counts."""
    category_missing_reasoning_count: dict[str, int] = dataclasses.field(
        default_factory=lambda: dict[str, int](),
    )
    """Per-category counts of samples without reasoning."""
    category_missing_answer_count: dict[str, int] = dataclasses.field(
        default_factory=lambda: dict[str, int](),
    )
    """Per-category counts of samples without final answer."""
    category_malformed_count: dict[str, int] = dataclasses.field(
        default_factory=lambda: dict[str, int](),
    )
    """Per-category counts of samples with malformed tag structure."""

    @classmethod
    def new(cls) -> "ParsingStatsAccumulator":
        """Create a fresh accumulator with initialized RunningStats."""
        return cls(
            output_length_chars=stats_utils.RunningStats(),
            reasoning_length_chars=stats_utils.RunningStats(),
            answer_length_chars=stats_utils.RunningStats(),
            output_length_tokens=stats_utils.RunningStats(),
            reasoning_length_tokens=stats_utils.RunningStats(),
            answer_length_tokens=stats_utils.RunningStats(),
        )

    def reset(self) -> None:
        """Reset all stats to initial state."""
        self.output_length_chars = stats_utils.RunningStats()
        self.reasoning_length_chars = stats_utils.RunningStats()
        self.answer_length_chars = stats_utils.RunningStats()
        self.output_length_tokens = stats_utils.RunningStats()
        self.reasoning_length_tokens = stats_utils.RunningStats()
        self.answer_length_tokens = stats_utils.RunningStats()
        self.total_count = 0
        self.missing_reasoning_count = 0
        self.missing_answer_count = 0
        self.malformed_count = 0
        self.category_output_length_chars.clear()
        self.category_reasoning_length_chars.clear()
        self.category_answer_length_chars.clear()
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
            "output_length_chars": self.output_length_chars.as_state(),
            "reasoning_length_chars": self.reasoning_length_chars.as_state(),
            "answer_length_chars": self.answer_length_chars.as_state(),
            "output_length_tokens": self.output_length_tokens.as_state(),
            "reasoning_length_tokens": self.reasoning_length_tokens.as_state(),
            "answer_length_tokens": self.answer_length_tokens.as_state(),
            "total_count": self.total_count,
            "missing_reasoning_count": self.missing_reasoning_count,
            "missing_answer_count": self.missing_answer_count,
            "malformed_count": self.malformed_count,
            "category_output_length_chars": {k: v.as_state() for k, v in self.category_output_length_chars.items()},
            "category_reasoning_length_chars": {
                k: v.as_state() for k, v in self.category_reasoning_length_chars.items()
            },
            "category_answer_length_chars": {k: v.as_state() for k, v in self.category_answer_length_chars.items()},
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
            output_length_chars=stats_utils.RunningStats.from_state(state["output_length_chars"]),
            reasoning_length_chars=stats_utils.RunningStats.from_state(state["reasoning_length_chars"]),
            answer_length_chars=stats_utils.RunningStats.from_state(state["answer_length_chars"]),
            output_length_tokens=stats_utils.RunningStats.from_state(state["output_length_tokens"]),
            reasoning_length_tokens=stats_utils.RunningStats.from_state(state["reasoning_length_tokens"]),
            answer_length_tokens=stats_utils.RunningStats.from_state(state["answer_length_tokens"]),
            total_count=state["total_count"],
            missing_reasoning_count=state["missing_reasoning_count"],
            missing_answer_count=state["missing_answer_count"],
            malformed_count=state["malformed_count"],
            category_output_length_chars={
                k: stats_utils.RunningStats.from_state(v) for k, v in state["category_output_length_chars"].items()
            },
            category_reasoning_length_chars={
                k: stats_utils.RunningStats.from_state(v) for k, v in state["category_reasoning_length_chars"].items()
            },
            category_answer_length_chars={
                k: stats_utils.RunningStats.from_state(v) for k, v in state["category_answer_length_chars"].items()
            },
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
        self.output_length_chars.merge(other.output_length_chars)
        self.reasoning_length_chars.merge(other.reasoning_length_chars)
        self.answer_length_chars.merge(other.answer_length_chars)
        self.output_length_tokens.merge(other.output_length_tokens)
        self.reasoning_length_tokens.merge(other.reasoning_length_tokens)
        self.answer_length_tokens.merge(other.answer_length_tokens)
        self.total_count += other.total_count
        self.missing_reasoning_count += other.missing_reasoning_count
        self.missing_answer_count += other.missing_answer_count
        self.malformed_count += other.malformed_count
        # merge category-wise character length stats
        for category, stats in other.category_output_length_chars.items():
            if category not in self.category_output_length_chars:
                self.category_output_length_chars[category] = stats_utils.RunningStats()
            self.category_output_length_chars[category].merge(stats)
        for category, stats in other.category_reasoning_length_chars.items():
            if category not in self.category_reasoning_length_chars:
                self.category_reasoning_length_chars[category] = stats_utils.RunningStats()
            self.category_reasoning_length_chars[category].merge(stats)
        for category, stats in other.category_answer_length_chars.items():
            if category not in self.category_answer_length_chars:
                self.category_answer_length_chars[category] = stats_utils.RunningStats()
            self.category_answer_length_chars[category].merge(stats)
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


class OutputParser(typing.Protocol):
    """Protocol for extracting structured fields from a model output.

    Implementations should be deterministic and side-effect free. The manager calls a parser at
    most once per `compute_output()` call and attaches the result to `SampleContext.parsed` for
    the duration of that call. Note: this is not persistent caching; each call parses independently.
    """

    def parse(
        self,
        prompt: str,
        model_output: str,
    ) -> ParsedOutput:
        """Parse a raw model output into structured fields."""
        ...


class RewardLogger(typing.Protocol):
    """Protocol for logging reward outputs and aggregated summaries.

    The manager performs all logging orchestration. Terms must never log directly.
    """

    def log(
        self,
        sample_id: str,
        *,
        total: float | None,
        terms: collections.abc.Mapping[str, float],
        metrics: collections.abc.Mapping[str, MetricValue],
        step: int | None = None,
        prompt: str | None = None,
        model_output: str | None = None,
        categories: collections.abc.Sequence[str] | None = None,
        tags: collections.abc.Sequence[str] | None = None,
    ) -> None:
        """Log a per-sample reward breakdown and metrics.

        Args:
            sample_id: Unique identifier for the sample.
            total: Total reward value, or None to omit from logging.
            terms: Per-term reward values.
            metrics: Per-sample metrics including term-emitted metrics, parsing metrics
                (under `parsing/*`), and category labels (under `categories/*`).
            step: Optional logging step.
            prompt: Optional prompt text for table logging.
            model_output: Optional raw model output for table logging.
            categories: Optional list of category labels for the sample (for table logging).
            tags: Optional list of sample tags (for table logging).
        """
        ...

    def log_run(
        self,
        *,
        reward_totals: collections.abc.Mapping[str, float],
        reward_term_summaries: collections.abc.Mapping[str, float],
        reward_category_summaries: collections.abc.Mapping[str, float],
        parsing_summaries: collections.abc.Mapping[str, float] | None = None,
        parsing_category_summaries: collections.abc.Mapping[str, float] | None = None,
        step: int | None = None,
    ) -> None:
        """Log run-level summary metrics.

        Args:
            reward_totals: Aggregated total reward stats.
            reward_term_summaries: Per-term aggregated reward stats.
            reward_category_summaries: Per-category aggregated reward stats.
            parsing_summaries: Global parsing stats (optional).
            parsing_category_summaries: Per-category parsing stats (optional).
            step: Optional logging step.
        """
        ...

    def log_failures(
        self,
        *,
        failure_ratio: float,
        failure_count: int,
        step: int | None = None,
    ) -> None:
        """Log failure statistics from reward computation.

        Failures include samples that were skipped or errored during reward computation.
        External logging backends (e.g., WandB) emit these under a `failures/` prefix.

        Args:
            failure_ratio: Ratio of failed samples to total samples.
            failure_count: Total number of failed samples.
            step: Optional logging step.
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
        parser: OutputParser | None,
    ) -> RewardTerm:
        """Construct a reward term from a spec and shared components."""
        ...


class BaseConfig(pydantic.BaseModel):
    """Base class for reward configs (frozen and strictly validated).

    Reward configs are Pydantic v2 models to keep validation close to configuration parsing and to
    integrate cleanly with the framework's existing config patterns.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
