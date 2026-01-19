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

    counts: dict[LengthSource, int] = dataclasses.field(default_factory=lambda: dict[LengthSource, int]())
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

    def update(
        self,
        parsed: ParsedOutput,
        token_cache: "TokenCountCache | None",
        categories: list[str] | None,
    ) -> None:
        """Update stats from a single sample.

        Args:
            parsed: Parsed output from the sample.
            token_cache: Token count cache (if token tracking enabled).
            categories: List of categories for this sample (if category extraction enabled).
        """
        output_len_chars = len(parsed.raw)
        has_reasoning = bool(parsed.reasoning)  # treat empty string as missing
        has_answer = bool(parsed.final_answer)  # treat empty string as missing
        is_malformed = parsed.fields.get("is_malformed", "false") == "true"
        reasoning_len_chars = len(parsed.reasoning) if has_reasoning else None  # type: ignore[arg-type]
        answer_len_chars = len(parsed.final_answer) if has_answer else None  # type: ignore[arg-type]
        # get token lengths from cache
        output_len_tokens: int | None = None
        reasoning_len_tokens: int | None = None
        answer_len_tokens: int | None = None
        if token_cache is not None:
            output_len_tokens = token_cache.get(LengthSource.model_output)
            reasoning_len_tokens = token_cache.get(LengthSource.parsed_reasoning)
            answer_len_tokens = token_cache.get(LengthSource.parsed_final_answer)
        # update global character length stats
        self.output_length_chars.update(float(output_len_chars))
        if reasoning_len_chars is not None:
            self.reasoning_length_chars.update(float(reasoning_len_chars))
        if answer_len_chars is not None:
            self.answer_length_chars.update(float(answer_len_chars))
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
                    output_len_chars=output_len_chars,
                    reasoning_len_chars=reasoning_len_chars,
                    answer_len_chars=answer_len_chars,
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
        output_len_chars: int,
        reasoning_len_chars: int | None,
        answer_len_chars: int | None,
        output_len_tokens: int | None,
        reasoning_len_tokens: int | None,
        answer_len_tokens: int | None,
        has_reasoning: bool,
        has_answer: bool,
        is_malformed: bool,
    ) -> None:
        """Update stats for a single category."""
        # output length (chars)
        if category not in self.category_output_length_chars:
            self.category_output_length_chars[category] = stats_utils.RunningStats()
        self.category_output_length_chars[category].update(float(output_len_chars))
        # reasoning length (chars)
        if reasoning_len_chars is not None:
            if category not in self.category_reasoning_length_chars:
                self.category_reasoning_length_chars[category] = stats_utils.RunningStats()
            self.category_reasoning_length_chars[category].update(float(reasoning_len_chars))
        # answer length (chars)
        if answer_len_chars is not None:
            if category not in self.category_answer_length_chars:
                self.category_answer_length_chars[category] = stats_utils.RunningStats()
            self.category_answer_length_chars[category].update(float(answer_len_chars))
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

    @staticmethod
    def _emit_running_stats(
        stats: stats_utils.RunningStats,
        prefix: str,
    ) -> dict[str, float]:
        """Emit mean/std/min/max metrics for a RunningStats object."""
        if stats.count == 0:
            return {}
        assert stats.min is not None and stats.max is not None
        return {
            f"{prefix}/mean": stats.mean(),
            f"{prefix}/std": stats.std(),
            f"{prefix}/min": stats.min,
            f"{prefix}/max": stats.max,
        }

    def get_metrics(
        self,
        *,
        reasoning_enabled: bool = True,
        answer_enabled: bool = True,
        capture_diagnostics: bool = False,
    ) -> dict[str, float]:
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
        metrics: dict[str, float] = {}
        # output length stats (always tracked)
        metrics.update(self._emit_running_stats(self.output_length_chars, "output_length_chars"))
        metrics.update(self._emit_running_stats(self.output_length_tokens, "output_length_tokens"))
        # reasoning length stats (only when enabled)
        if reasoning_enabled:
            metrics.update(self._emit_running_stats(self.reasoning_length_chars, "reasoning_length_chars"))
            metrics.update(self._emit_running_stats(self.reasoning_length_tokens, "reasoning_length_tokens"))
        # answer length stats (only when enabled)
        if answer_enabled:
            metrics.update(self._emit_running_stats(self.answer_length_chars, "answer_length_chars"))
            metrics.update(self._emit_running_stats(self.answer_length_tokens, "answer_length_tokens"))
        # format ratios
        total = float(self.total_count)
        if reasoning_enabled:
            metrics["missing_reasoning_ratio"] = self.missing_reasoning_count / total
        if answer_enabled:
            metrics["missing_answer_ratio"] = self.missing_answer_count / total
        if capture_diagnostics:
            metrics["malformed_ratio"] = self.malformed_count / total
        metrics["sample_count"] = total
        return metrics

    def get_category_metrics(
        self,
        *,
        reasoning_enabled: bool = True,
        answer_enabled: bool = True,
        capture_diagnostics: bool = False,
    ) -> dict[str, float]:
        """Return category-wise parsing metrics.

        Args:
            reasoning_enabled: Whether reasoning extraction is enabled.
            answer_enabled: Whether final answer extraction is enabled.
            capture_diagnostics: Whether to include malformed_ratio.

        Returns:
            Dict of metric name to value, empty if no categories.
        """
        metrics: dict[str, float] = {}
        for category in sorted(self.category_total_count.keys()):
            total = float(self.category_total_count.get(category, 0))
            if total == 0:
                continue
            # output length
            cat_output_chars = self.category_output_length_chars.get(category)
            if cat_output_chars:
                metrics.update(self._emit_running_stats(cat_output_chars, f"{category}/output_length_chars"))
            cat_output_tokens = self.category_output_length_tokens.get(category)
            if cat_output_tokens:
                metrics.update(self._emit_running_stats(cat_output_tokens, f"{category}/output_length_tokens"))
            # reasoning length
            if reasoning_enabled:
                cat_reasoning_chars = self.category_reasoning_length_chars.get(category)
                if cat_reasoning_chars:
                    metrics.update(self._emit_running_stats(cat_reasoning_chars, f"{category}/reasoning_length_chars"))
                cat_reasoning_tokens = self.category_reasoning_length_tokens.get(category)
                if cat_reasoning_tokens:
                    metrics.update(
                        self._emit_running_stats(cat_reasoning_tokens, f"{category}/reasoning_length_tokens")
                    )
            # answer length
            if answer_enabled:
                cat_answer_chars = self.category_answer_length_chars.get(category)
                if cat_answer_chars:
                    metrics.update(self._emit_running_stats(cat_answer_chars, f"{category}/answer_length_chars"))
                cat_answer_tokens = self.category_answer_length_tokens.get(category)
                if cat_answer_tokens:
                    metrics.update(self._emit_running_stats(cat_answer_tokens, f"{category}/answer_length_tokens"))
            # format ratios
            if reasoning_enabled:
                missing_reasoning = float(self.category_missing_reasoning_count.get(category, 0))
                metrics[f"{category}/missing_reasoning_ratio"] = missing_reasoning / total
            if answer_enabled:
                missing_answer = float(self.category_missing_answer_count.get(category, 0))
                metrics[f"{category}/missing_answer_ratio"] = missing_answer / total
            if capture_diagnostics:
                malformed = float(self.category_malformed_count.get(category, 0))
                metrics[f"{category}/malformed_ratio"] = malformed / total
            metrics[f"{category}/sample_count"] = total
        return metrics


class OutputParser(typing.Protocol):
    """Protocol for extracting structured fields from a model output.

    Implementations should be deterministic and side-effect free. The manager calls a parser at
    most once per `compute()` call and attaches the result to `SampleContext.parsed` for
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

    The logger controls frequency gating for scalars and table rows based on generation_count.
    The manager handles key scoping and orchestration. Terms must never log directly.
    """

    def log(
        self,
        sample_id: str,
        *,
        generation_count: int | None = None,
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
        generation_idx: int | None = None,
        **kwargs: typing.Any,
    ) -> None:
        """Log a per-sample reward breakdown and metrics.

        Args:
            sample_id: Unique identifier for the sample.
            generation_count: Total generations processed so far (1-indexed). Used by the logger
                for frequency gating. If None, frequency gating is skipped (always log).
            total: Total reward value, or None to omit from logging.
            terms: Per-term weighted reward values (optional).
            metrics: Per-sample metrics (optional) including term-emitted metrics, parsing
                metrics (under `parsing/*`), and category labels (under `categories/*`).
            raw_terms: Per-term raw (pre-clipping, pre-weighting) reward values (optional).
            step: Optional logging step.
            prompt: Optional prompt text for table logging.
            expected_output: Optional expected output text for table logging.
            model_output: Optional raw model output for table logging.
            reasoning: Optional parsed reasoning text for table logging.
            final_answer: Optional parsed final answer text for table logging.
            categories: Optional list of category labels for the sample (for table logging).
            tags: Optional list of sample tags (for table logging).
            generation_idx: Generation index within a prompt's completions (0-indexed,
                computed per sample_data.identifier group). Used for GRPO/multi-generation analysis.
            **kwargs: Additional keyword arguments for forward compatibility.
                Custom implementations should accept **kwargs to remain compatible
                with future additions to the logging interface.
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
        failure_ratio: float | None = None,
        failure_count: int | None = None,
        step: int | None = None,
    ) -> None:
        """Log run-level summary metrics.

        Args:
            reward_totals: Aggregated total reward stats.
            reward_term_summaries: Per-term aggregated reward stats.
            reward_category_summaries: Per-category aggregated reward stats.
            parsing_summaries: Global parsing stats (optional).
            parsing_category_summaries: Per-category parsing stats (optional).
            failure_ratio: Ratio of failed samples to total samples (optional).
            failure_count: Total number of failed samples (optional).
            step: Optional logging step.
        """
        ...

    def should_log_scalars(
        self,
        generation_count: int,
    ) -> bool:
        """Return True if scalars should be logged for this generation_count.

        Used by the manager to skip expensive metric extraction when the logger will drop the data.

        Args:
            generation_count: Total generations processed so far (1-indexed).

        Returns:
            True if scalars should be emitted for this generation count.
        """
        ...

    def should_add_table_row(
        self,
        generation_count: int,
    ) -> bool:
        """Return True if a table row should be added for this generation_count.

        Used by the manager to skip expensive table field extraction when the logger will drop
        the row.

        Args:
            generation_count: Total generations processed so far (1-indexed).

        Returns:
            True if a table row should be added for this generation count.
        """
        ...

    def log_batch_stats(
        self,
        *,
        batch_mean: float,
        batch_std: float,
        batch_mean_rolling_mean: float,
        batch_mean_rolling_std: float,
        batch_std_rolling_mean: float,
        batch_std_rolling_std: float,
        batch_count: int | None = None,
    ) -> None:
        """Log batch-level reward statistics (per-batch and rolling).

        Args:
            batch_mean: Mean reward across all samples in the current batch.
            batch_std: Population std of rewards in the current batch.
            batch_mean_rolling_mean: Rolling mean of batch means over time.
            batch_mean_rolling_std: Rolling std of batch means over time.
            batch_std_rolling_mean: Rolling mean of batch stds over time.
            batch_std_rolling_std: Rolling std of batch stds over time.
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
