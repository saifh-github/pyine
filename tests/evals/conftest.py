"""Shared pytest fixtures for pyine.evals tests."""

from __future__ import annotations

import types
import typing

import pytest


class MockWandBRun:
    """Mock wandb run for testing W&B integration without network calls.

    Captures all define_metric and log calls for assertion in tests.
    """

    def __init__(self) -> None:
        self.defined_metrics: list[dict[str, typing.Any]] = []
        self.logged_data: list[tuple[dict[str, typing.Any], bool]] = []
        self.logged_tables: list[typing.Any] = []

    def define_metric(
        self,
        name: str,
        step_metric: str | None = None,
        hidden: bool = False,
        summary: str | None = None,
        **kwargs: typing.Any,
    ) -> None:
        """Record metric definition."""
        self.defined_metrics.append(
            {
                "name": name,
                "step_metric": step_metric,
                "hidden": hidden,
                "summary": summary,
                **kwargs,
            }
        )

    def log(
        self,
        data: dict[str, typing.Any],
        commit: bool = True,
        **kwargs: typing.Any,
    ) -> None:
        """Record logged data."""
        self.logged_data.append((data, commit))

    def get_defined_metric_names(self) -> set[str]:
        """Return set of all defined metric names."""
        return {m["name"] for m in self.defined_metrics}


@pytest.fixture
def mock_wandb_run() -> MockWandBRun:
    """Mock wandb run for testing W&B integration."""
    return MockWandBRun()


@pytest.fixture
def mock_wandb_api() -> types.SimpleNamespace:
    """Mock wandb API for testing analysis functions."""

    class MockRun:
        def __init__(self, name: str, state: str = "finished") -> None:
            self.name = name
            self.state = state
            self.summary = {}
            self.config = {}
            self.created_at = "2025-01-01T00:00:00"

    class MockApi:
        def __init__(self) -> None:
            self.runs_returned: list[MockRun] = []

        def runs(
            self,
            path: str,
            filters: dict[str, typing.Any] | None = None,
            order: str | None = None,
        ) -> list[MockRun]:
            return self.runs_returned

    return types.SimpleNamespace(Api=MockApi, Run=MockRun)
