"""Shared pytest fixtures for pyine.evals.code_exec tests."""

from __future__ import annotations

import asyncio
import dataclasses
import typing

import langchain_core.messages
import pytest

import pyine.evals.code_exec.evaluator

EXACT_MATCH_SCORE: typing.Final[float] = 1.0
NO_MATCH_SCORE: typing.Final[float] = 0.0
PARTIAL_MATCH_SCORE: typing.Final[float] = 0.25
DEFAULT_GRADER_THRESHOLD: typing.Final[float] = 0.5


class MockGraderChain:
    """Configurable mock LLM grader for testing.

    Args:
        scorer: Callable that takes (expected, predicted) and returns a score in [0, 1].
    """

    def __init__(
        self,
        scorer: typing.Callable[[str, str], float],
    ) -> None:
        self._scorer = scorer
        self.invoke_count = 0

    def invoke(
        self,
        predicted: typing.Any,
        expected: typing.Any,
        predict_type: str = "unknown",
        **invoke_kwargs: typing.Any,
    ) -> float:
        """Synchronous invocation of the grader."""
        self.invoke_count += 1
        return float(self._scorer(str(expected), str(predicted)))

    async def ainvoke(
        self,
        predicted: typing.Any,
        expected: typing.Any,
        predict_type: str = "unknown",
        **invoke_kwargs: typing.Any,
    ) -> float:
        """Asynchronous invocation of the grader."""
        return await asyncio.to_thread(
            self.invoke,
            predicted=predicted,
            expected=expected,
            predict_type=predict_type,
            **invoke_kwargs,
        )


@pytest.fixture
def exact_match_grader() -> MockGraderChain:
    """Grader that returns 1.0 on exact match (after strip), 0.0 otherwise."""
    return MockGraderChain(
        scorer=lambda exp, pred: EXACT_MATCH_SCORE if exp.strip() == pred.strip() else NO_MATCH_SCORE
    )


@pytest.fixture
def partial_match_grader() -> MockGraderChain:
    """Grader that returns 1.0 on exact match, 0.25 on partial/mismatch."""
    return MockGraderChain(
        scorer=lambda exp, pred: EXACT_MATCH_SCORE if exp.strip() == pred.strip() else PARTIAL_MATCH_SCORE
    )


@pytest.fixture
def base_evaluator() -> pyine.evals.code_exec.evaluator.OutcomeEvaluator:
    """Fresh OutcomeEvaluator without grader."""
    return pyine.evals.code_exec.evaluator.OutcomeEvaluator()


@pytest.fixture
def evaluator_with_standard_samples(
    base_evaluator: pyine.evals.code_exec.evaluator.OutcomeEvaluator,
) -> pyine.evals.code_exec.evaluator.OutcomeEvaluator:
    """Evaluator with 3 standard test samples: exact match, mismatch, soft match.

    Sample breakdown:
    - s1: exact match (hard=True, soft=True, expected grader=1.0)
    - s2: complete mismatch (hard=False, soft=False, expected grader=0.0)
    - s3: soft match only - "1.0" vs "1" (hard=False, soft=True, expected grader varies)
    """
    base_evaluator.add_sample(identifier="s1", expected="42", predicted="42", tags=["easy"])
    base_evaluator.add_sample(identifier="s2", expected="abc", predicted="xyz", tags=["hard"])
    base_evaluator.add_sample(identifier="s3", expected="1.0", predicted="1", tags=["soft"])
    return base_evaluator


@pytest.fixture
def evaluator_with_grader(
    base_evaluator: pyine.evals.code_exec.evaluator.OutcomeEvaluator,
    exact_match_grader: MockGraderChain,
) -> pyine.evals.code_exec.evaluator.OutcomeEvaluator:
    """Evaluator with mock exact-match grader chain installed."""
    base_evaluator._llm_grader_chain_config = exact_match_grader
    return base_evaluator


@pytest.fixture
def evaluator_with_grader_and_samples(
    exact_match_grader: MockGraderChain,
) -> pyine.evals.code_exec.evaluator.OutcomeEvaluator:
    """Evaluator with mock grader and standard samples pre-loaded.

    NOTE: Grader must be set BEFORE adding samples, otherwise LLM scores are None.

    Sample breakdown:
    - s1: exact match (hard=True, soft=True, grader=1.0)
    - s2: complete mismatch (hard=False, soft=False, grader=0.0)
    - s3: soft match only - "1.0" vs "1" (hard=False, soft=True, grader=0.0)
    """
    evaluator = pyine.evals.code_exec.evaluator.OutcomeEvaluator()
    evaluator._llm_grader_chain_config = exact_match_grader
    evaluator.add_sample(identifier="s1", expected="42", predicted="42", tags=["easy"])
    evaluator.add_sample(identifier="s2", expected="abc", predicted="xyz", tags=["hard"])
    evaluator.add_sample(identifier="s3", expected="1.0", predicted="1", tags=["soft"])
    return evaluator


class FakeSample:
    """Fake sample object for testing evaluation pipelines."""

    def __init__(
        self,
        identifier: str,
        expected_output: str | None = None,
        predict_type: str = "unknown",
        tags: list[str] | None = None,
    ) -> None:
        self.identifier = identifier
        self.expected_output = expected_output or f"expected-{identifier}"
        self.predict_type = predict_type
        self._tags = tags or ["tag"]

    def _asdict(self) -> dict[str, str]:
        """Return dict representation for serialization."""
        return {"identifier": self.identifier}

    def get_tag_list(self) -> list[str]:
        """Return list of tags."""
        return list(self._tags)


class FakeSampleBuilder(list[FakeSample]):
    """Fake sample builder that acts as a list of samples."""

    def __init__(self, samples: list[FakeSample]) -> None:
        super().__init__(samples)


class FakeDataModule:
    """Fake data module for testing evaluation pipelines."""

    def __init__(self, samples: list[FakeSample]) -> None:
        self._samples = samples

    def get_parser(self, subset_name: str) -> FakeSampleBuilder:
        """Return fake sample builder."""
        return FakeSampleBuilder(self._samples)


@dataclasses.dataclass
class FakeSampleEval:
    """Fake sample evaluation result."""

    identifier: str
    hard_match: bool = True
    soft_match: str | None = None
    llm_score: float | None = None
    tags: list[str] | None = None
    attempt_index: int = 0


class FakeOutcomeEvaluator:
    """Fake outcome evaluator for testing without real evaluation logic."""

    def __init__(
        self,
        llm_provider_config: typing.Any | None = None,
    ) -> None:
        self.llm_provider_config = llm_provider_config
        self.results: list[FakeSampleEval] = []
        self.added: list[tuple[str, str, str, list[str]]] = []

    def add_sample(
        self,
        identifier: str,
        predicted: str,
        expected: str,
        predict_type: str = "unknown",
        tags: list[str] | None = None,
        attempt_index: int = 0,
    ) -> None:
        """Record sample addition."""
        self.added.append((identifier, expected, predicted, tags or []))
        self.results.append(FakeSampleEval(identifier=identifier, tags=tags or [], attempt_index=attempt_index))

    def get_sample_count(self) -> int:
        """Return count of unique samples."""
        return len({r.identifier for r in self.results})

    def get_attempt_count(self) -> int:
        """Return count of attempts."""
        return len(self.results)

    async def compute_category_wise_metrics(
        self,
        category_to_identifiers: dict[str, list[str]],
        score_threshold: float = 0.5,
        pass_at_k_values: typing.Any = None,
        num_attempts_per_sample: int = 1,
    ) -> dict[str, dict[str, typing.Any]]:
        """Return empty metrics (fake implementation)."""
        return {}


@dataclasses.dataclass
class FakeArtifact:
    """Fake evaluation artifact."""

    sample: object
    token_usage: object
    eval_result: object


@dataclasses.dataclass
class FakeEvalResult:
    """Fake evaluation result container."""

    metrics: dict[str, object]
    artifacts: list[FakeArtifact]
    category_to_identifiers: dict[str, list[str]]


def build_ai_message(
    identifier: str,
    content: str | None = None,
    total_tokens: int = 2,
) -> langchain_core.messages.AIMessage:
    """Build a fake AIMessage with usage metadata for testing."""
    return langchain_core.messages.AIMessage(
        content=content or f"prediction-{identifier}",
        usage_metadata={
            "total_tokens": total_tokens,
            "prompt_tokens": 1,
            "input_tokens": 1,
            "output_tokens": total_tokens - 1,
        },
    )


@pytest.fixture
def fake_samples() -> list[FakeSample]:
    """List of 3 fake samples for testing."""
    return [
        FakeSample("sample-1"),
        FakeSample("sample-2"),
        FakeSample("sample-3"),
    ]


@pytest.fixture
def fake_data_module(fake_samples: list[FakeSample]) -> FakeDataModule:
    """Fake data module with standard test samples."""
    return FakeDataModule(fake_samples)


@pytest.fixture
def fake_outcome_evaluator() -> FakeOutcomeEvaluator:
    """Fake outcome evaluator for testing."""
    return FakeOutcomeEvaluator()
