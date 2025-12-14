"""Shared reward types and protocols.

These types are designed to keep reward computation decoupled from any specific trainer or RL
framework. The training loop provides a per-sample `SampleContext` (prompt + full model output +
metadata), and the reward system returns a single scalar reward per sample.
"""

import collections.abc
import dataclasses
import typing

import pydantic

import pyine.organisms.datamodules.base
import pyine.organisms.datamodules.samples

if typing.TYPE_CHECKING:
    import pyine.organisms.models.rewards.core.configs as reward_configs


@dataclasses.dataclass(frozen=True, slots=True)
class RunInitContext:
    """Static info available at training start.

    This object is passed to `RewardManager.reset()` and forwarded to each term's `reset()` method.
    Keep this lightweight and stable; put experimental or task-specific data in `extras`.
    """

    datamodule: pyine.organisms.datamodules.base.BiasDataModuleBase[typing.Any] | None = None
    """Reference to the datamodule used for the experiment."""
    extras: collections.abc.Mapping[str, object] = dataclasses.field(
        default_factory=lambda: typing.cast("dict[str, object]", {}),
    )
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
    fields: collections.abc.Mapping[str, str] = dataclasses.field(
        default_factory=lambda: typing.cast("dict[str, str]", {}),
    )
    """Additional extracted string fields (term-/task-specific)."""


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
    parsed: ParsedOutput | None = None
    """Optional cached parse result for `model_output`."""
    extras: collections.abc.Mapping[str, object] = dataclasses.field(
        default_factory=lambda: typing.cast("dict[str, object]", {}),
    )
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
    metrics: collections.abc.Mapping[str, bool | int | float] = dataclasses.field(
        default_factory=lambda: typing.cast("dict[str, bool | int | float]", {}),
    )
    """Optional scalar metrics emitted by the term."""


@dataclasses.dataclass(frozen=True, slots=True)
class RewardOutput:
    """A full reward computation output for a single sample."""

    total: float
    """Aggregated scalar reward after weighting/clipping."""
    weighted_terms: dict[str, float]
    """Per-term contributions after weighting/clipping."""
    unweighted_terms: dict[str, float] | None = None
    """Optional per-term raw values prior to weighting."""
    metrics: dict[str, bool | int | float] = dataclasses.field(
        default_factory=lambda: typing.cast("dict[str, bool | int | float]", {}),
    )
    """Additional scalar metrics (keyed by `term/metric`)."""


class OutputParser(typing.Protocol):
    """Protocol for extracting structured fields from a model output.

    Implementations should be deterministic and side-effect free. The manager may call a parser
    at most once per sample and cache the result on `SampleContext.parsed`.
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
        total: float,
        terms: collections.abc.Mapping[str, float],
        metrics: collections.abc.Mapping[str, bool | int | float],
        step: int | None = None,
    ) -> None:
        """Log a per-sample reward breakdown and metrics."""
        ...

    def log_run(
        self,
        *,
        totals: collections.abc.Mapping[str, float],
        term_summaries: collections.abc.Mapping[str, float],
        step: int | None = None,
    ) -> None:
        """Log run-level summary metrics."""
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
