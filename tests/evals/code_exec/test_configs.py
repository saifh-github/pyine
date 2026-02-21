"""Tests for pyine.evals.code_exec.configs and _impl evaluation pipelines.

This module tests the CodeExecEvalsConfig evaluation methods and related utilities.
"""

from __future__ import annotations

import dataclasses
import types
import typing

import datasets as hf_datasets
import langchain_core.messages
import pytest
import torch

if typing.TYPE_CHECKING:
    import pathlib

import pyine.evals.code_exec._impl
import pyine.evals.code_exec.configs
import pyine.evals.code_exec.utils
import pyine.evals.common
import pyine.utils.langchain
import tests.utils.transformers.utils

from .conftest import (
    FakeArtifact,
    FakeDataModule,
    FakeEvalResult,
    FakeOutcomeEvaluator,
    FakeSample,
    FakeSampleBuilder,
    build_ai_message,
)


# helper for HF model tests; needs different SampleData structure
@dataclasses.dataclass
class HFSampleData:
    """Sample data structure for HF model evaluation tests."""

    identifier: str
    expected_output: str
    comma_separated_tags: str
    predict_type: str = "unknown"

    def get_tag_list(self) -> list[str]:
        """Return list of tags from comma-separated string."""
        if not self.comma_separated_tags:
            return []
        return self.comma_separated_tags.split(",")

    def _asdict(self) -> dict[str, typing.Any]:
        """Return dict representation."""
        return dataclasses.asdict(self)


def _make_sample_with_complexity_metrics(
    loc: int = 10,
    sloc: int = 8,
) -> types.SimpleNamespace:
    """Creates a duck-typed sample object with all required complexity metrics."""
    return types.SimpleNamespace(
        complexity_metrics={
            "cyclomatic_complexity_avg": 1.0,
            "cyclomatic_complexity_max": 1,
            "cyclomatic_complexity_sum": 1,
            "loc": loc,
            "lloc": loc,
            "sloc": sloc,
            "comments": 0,
            "multi": 0,
            "blank": 0,
            "halstead_volume": 10.0,
            "halstead_difficulty": 1.0,
            "halstead_effort": 10.0,
            "maintainability_index": 100.0,
        }
    )


@pytest.fixture
def impl_monkeypatches(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Common monkeypatches for _impl evaluation tests."""

    async def fake_get_metrics(
        evaluator: FakeOutcomeEvaluator,
        token_usage: typing.Any,
        attempt_token_usage: typing.Any,
        sample_data_store: typing.Any,
        pass_at_k_values: typing.Any = None,
        num_attempts_per_sample: int = 1,
        partial: bool = False,
    ) -> dict[str, typing.Any]:
        return {
            "accuracy": len(evaluator.results),
            "total_tokens": token_usage.total_tokens,
            "sample_count": len(sample_data_store),
        }

    def fake_tqdm(
        iterable: typing.Iterable[typing.Any] | None = None,
        total: int | None = None,
        **_kwargs: typing.Any,
    ) -> typing.Any:
        if iterable is None:
            # return a progress bar mock too
            return types.SimpleNamespace(
                update=lambda _: None,
                write=lambda _: None,
                close=lambda: None,
                total=total,
            )
        return iterable

    monkeypatch.setattr(
        pyine.evals.code_exec._impl.pyine.organisms.datamodules.samples,
        "SampleBuilder",
        FakeSampleBuilder,
    )
    monkeypatch.setattr(
        pyine.evals.code_exec._impl.pyine.organisms.datamodules.samples,
        "SampleData",
        FakeSample,
    )
    monkeypatch.setattr(
        pyine.evals.code_exec._impl.pyine.evals.code_exec.evaluator,
        "OutcomeEvaluator",
        FakeOutcomeEvaluator,
    )
    monkeypatch.setattr(
        pyine.evals.code_exec._impl.pyine.evals.code_exec.utils,
        "CodeExecEvalArtifact",
        FakeArtifact,
    )
    monkeypatch.setattr(
        pyine.evals.code_exec.utils,
        "CodeExecEvalResult",
        FakeEvalResult,
    )
    monkeypatch.setattr(
        pyine.evals.code_exec._impl.pyine.evals.code_exec.utils,
        "get_metrics",
        fake_get_metrics,
    )
    monkeypatch.setattr(
        pyine.evals.code_exec._impl.tqdm,
        "tqdm",
        fake_tqdm,
    )

    return monkeypatch


class TestEvaluateRunnableModel:
    """Tests for evaluate_runnable_model() pipeline."""

    @pytest.mark.asyncio
    async def test_sequential_execution_processes_all_samples(
        self,
        impl_monkeypatches: pytest.MonkeyPatch,
    ) -> None:
        """Sequential execution invokes chain for each sample in order."""
        samples = [FakeSample("s0"), FakeSample("s1")]
        data_module = FakeDataModule(samples)
        chain_calls: list[dict[str, str]] = []

        class MockChain:
            def invoke(self, payload: dict[str, str]) -> langchain_core.messages.AIMessage:
                chain_calls.append(payload)
                return build_ai_message(payload["identifier"])

        config = pyine.evals.code_exec.configs.CodeExecEvalsConfig(
            eval_runnable_config=pyine.evals.common.RunnableEvalConfig(parallel=False),
        )
        result = await config.evaluate_runnable_model(
            chain=MockChain(),
            datamodule=data_module,
            eval_subset_name="subset",
            verbose=True,
        )

        assert isinstance(result, FakeEvalResult)
        assert len(result.artifacts) == 2
        assert result.metrics["accuracy"] == 2
        assert chain_calls[0]["identifier"] == "s0"
        assert chain_calls[1]["identifier"] == "s1"

    @pytest.mark.asyncio
    async def test_parallel_execution_uses_concurrency(
        self,
        impl_monkeypatches: pytest.MonkeyPatch,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Parallel execution uses sliding window concurrency."""
        samples = [FakeSample("p0"), FakeSample("p1"), FakeSample("p2")]
        data_module = FakeDataModule(samples)
        submitted_items: list[int] = []

        class MockChain:
            def invoke(self, payload: dict[str, str]) -> langchain_core.messages.AIMessage:
                return build_ai_message(payload["identifier"])

        async def fake_run_with_sliding_window(
            *,
            input_items: typing.Iterable[typing.Any],
            submit_one: typing.Callable[..., typing.Any],
            process_result: typing.Callable[[typing.Any, typing.Any], None],
            progress_callback: typing.Callable[[list[typing.Any], list[typing.Any]], typing.Awaitable[None]],
            **_kwargs: typing.Any,
        ) -> None:
            """Fake sliding window that processes items sequentially."""

            class FakeExecutor:
                def submit(self, fn: typing.Callable, payload: typing.Any) -> types.SimpleNamespace:
                    return types.SimpleNamespace(result=lambda: fn(payload))

            completed: list[typing.Any] = []
            executor = FakeExecutor()
            for item in input_items:
                submitted_items.append(item)
                future = submit_one(item, executor=executor)
                response = future.result() if hasattr(future, "result") else future
                process_result(item, response)
                completed.append(item)
                await progress_callback([], completed)

        monkeypatch.setattr(
            pyine.evals.code_exec._impl.pyine.utils.concurrency,
            "run_with_sliding_window",
            fake_run_with_sliding_window,
        )

        config = pyine.evals.code_exec.configs.CodeExecEvalsConfig(
            eval_runnable_config=pyine.evals.common.RunnableEvalConfig(
                parallel=True,
                max_workers=2,
                max_in_flight_jobs=2,
                async_metrics_compute_rate=1,
            ),
        )

        result = await config.evaluate_runnable_model(
            chain=MockChain(),
            datamodule=data_module,
            eval_subset_name="subset",
            verbose=True,
        )

        assert submitted_items == [(0, 0), (1, 0), (2, 0)]  # (sample_idx, attempt_idx) tuples
        assert result.metrics["accuracy"] == 3

    @pytest.mark.asyncio
    async def test_returns_eval_result_with_metrics(
        self,
        impl_monkeypatches: pytest.MonkeyPatch,
    ) -> None:
        """Result contains metrics and artifacts."""
        samples = [FakeSample("x")]
        data_module = FakeDataModule(samples)

        class MockChain:
            def invoke(self, payload: dict[str, str]) -> langchain_core.messages.AIMessage:
                return build_ai_message(payload["identifier"])

        config = pyine.evals.code_exec.configs.CodeExecEvalsConfig()
        result = await config.evaluate_runnable_model(
            chain=MockChain(),
            datamodule=data_module,
            eval_subset_name="subset",
        )

        assert hasattr(result, "metrics")
        assert hasattr(result, "artifacts")
        assert "accuracy" in result.metrics
        assert "sample_count" in result.metrics


class TestParallelPromptCapture:
    """Tests for parallel prompt capture with disk export enabled."""

    @pytest.mark.asyncio
    async def test_parallel_prompt_capture_populates_stores(
        self,
        impl_monkeypatches: pytest.MonkeyPatch,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: pathlib.Path,
    ) -> None:
        """Parallel path with disk_export_config captures prompts for all samples."""
        samples = [FakeSample("p0"), FakeSample("p1")]
        data_module = FakeDataModule(samples)
        captured_configs: list[dict[str, typing.Any] | None] = []

        class MockChain:
            def invoke(
                self,
                payload: dict[str, str],
                config: dict[str, typing.Any] | None = None,
            ) -> langchain_core.messages.AIMessage:
                captured_configs.append(config)
                return build_ai_message(payload["identifier"])

        async def fake_run_with_sliding_window(
            *,
            input_items: typing.Iterable[typing.Any],
            submit_one: typing.Callable[..., typing.Any],
            process_result: typing.Callable[[typing.Any, typing.Any], None],
            progress_callback: typing.Callable[[list[typing.Any], list[typing.Any]], typing.Awaitable[None]],
            **_kwargs: typing.Any,
        ) -> None:
            class FakeExecutor:
                def submit(
                    self,
                    fn: typing.Callable[..., typing.Any],
                    *args: typing.Any,
                    **kwargs: typing.Any,
                ) -> types.SimpleNamespace:
                    return types.SimpleNamespace(result=lambda: fn(*args, **kwargs))

            completed: list[typing.Any] = []
            executor = FakeExecutor()
            for item in input_items:
                future = submit_one(item, executor=executor)
                response = future.result() if hasattr(future, "result") else future
                process_result(item, response)
                completed.append(item)
                await progress_callback([], completed)

        monkeypatch.setattr(
            pyine.evals.code_exec._impl.pyine.utils.concurrency,
            "run_with_sliding_window",
            fake_run_with_sliding_window,
        )
        # disable actual disk export in _finalize to avoid writing LMDBs during monkeypatched eval
        monkeypatch.setattr(
            pyine.evals.code_exec._impl.pyine.evals.logging,
            "DiskEvalLogger",
            type(
                "FakeDiskLogger",
                (),
                {
                    "__init__": lambda self, **kw: None,
                    "export_results": lambda self, **kw: None,
                    "close": lambda self: None,
                },
            ),
        )
        config = pyine.evals.code_exec.configs.CodeExecEvalsConfig(
            eval_runnable_config=pyine.evals.common.RunnableEvalConfig(
                parallel=True,
                max_workers=2,
                max_in_flight_jobs=2,
                async_metrics_compute_rate=100,
            ),
            disk_export_config=pyine.evals.common.EvalExportConfig(
                output_path=tmp_path / "export",
            ),
        )
        result = await config.evaluate_runnable_model(
            chain=MockChain(),
            datamodule=data_module,
            eval_subset_name="subset",
            verbose=False,
        )
        assert isinstance(result, FakeEvalResult)
        assert len(result.artifacts) == 2
        # verify that callbacks were attached for attempt_idx==0 items (all items here)
        configs_with_callbacks = [cfg for cfg in captured_configs if cfg is not None and "callbacks" in cfg]
        assert len(configs_with_callbacks) == 2  # one per sample (attempt_idx=0)


class TestEvaluateHFModel:
    """Tests for evaluate_hf_model() pipeline."""

    @pytest.fixture
    def hf_model_monkeypatches(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> pytest.MonkeyPatch:
        """Monkeypatches specific to HF model evaluation tests."""

        async def fake_get_metrics(
            evaluator: FakeOutcomeEvaluator,
            token_usage: typing.Any,
            attempt_token_usage: typing.Any,
            sample_data_store: typing.Any,
            pass_at_k_values: typing.Any = None,
            num_attempts_per_sample: int = 1,
            partial: bool = False,
        ) -> dict[str, typing.Any]:
            return {
                "count": len(evaluator.results),
                "tokens": token_usage.total_tokens,
                "sample_count": len(sample_data_store),
            }

        def fake_tqdm(
            iterable: typing.Iterable[typing.Any],
            **_kwargs: typing.Any,
        ) -> typing.Iterable[typing.Any]:
            return iterable

        def fake_data_loader(
            dataset: typing.Iterable[typing.Any],
            **_kwargs: typing.Any,
        ) -> typing.Iterable[typing.Any]:
            return dataset

        def fake_run_text_generation(**kwargs: typing.Any) -> list[dict[str, typing.Any]]:
            outputs = []
            for idx, batch in enumerate(kwargs["dataloader"]):
                outputs.append(
                    {
                        "sample_idx": batch["sample_idx"],
                        "prediction": f"pred-{idx}",
                        "generated_tokens": torch.as_tensor([0, 1]),
                    }
                )
            return outputs

        monkeypatch.setattr(
            pyine.evals.code_exec._impl.pyine.organisms.datamodules.samples,
            "SampleData",
            HFSampleData,
        )
        monkeypatch.setattr(
            pyine.evals.code_exec._impl.pyine.utils.transformers,
            "is_hf_model",
            lambda _: True,
        )
        monkeypatch.setattr(
            pyine.evals.code_exec._impl.pyine.utils.transformers,
            "supports_text_generation",
            lambda _: True,
        )
        monkeypatch.setattr(
            pyine.evals.code_exec._impl.pyine.utils.transformers,
            "infer_effective_max_seq_len",
            lambda *_: 16,
        )
        monkeypatch.setattr(
            pyine.evals.code_exec._impl.pyine.utils.transformers,
            "PaddingCollatorWithPromptMask",
            lambda **_: "collator",
        )
        monkeypatch.setattr(
            pyine.evals.code_exec._impl.pyine.utils.transformers,
            "run_text_generation",
            fake_run_text_generation,
        )
        monkeypatch.setattr(
            pyine.evals.code_exec._impl.pyine.evals.code_exec.evaluator,
            "OutcomeEvaluator",
            FakeOutcomeEvaluator,
        )
        monkeypatch.setattr(
            pyine.evals.code_exec._impl.pyine.evals.code_exec.utils,
            "CodeExecEvalArtifact",
            FakeArtifact,
        )
        monkeypatch.setattr(
            pyine.evals.code_exec.utils,
            "CodeExecEvalResult",
            FakeEvalResult,
        )
        monkeypatch.setattr(
            pyine.evals.code_exec._impl.pyine.evals.code_exec.utils,
            "get_metrics",
            fake_get_metrics,
        )
        monkeypatch.setattr(
            pyine.evals.code_exec._impl.tqdm,
            "tqdm",
            fake_tqdm,
        )
        monkeypatch.setattr(
            pyine.evals.code_exec._impl.torch.utils.data,
            "DataLoader",
            fake_data_loader,
        )

        return monkeypatch

    @pytest.mark.asyncio
    async def test_generates_results_for_all_samples(
        self,
        hf_model_monkeypatches: pytest.MonkeyPatch,
    ) -> None:
        """HF model evaluation generates results for all samples."""
        samples = [
            {
                "messages": [
                    {"role": "system", "content": "You are brief."},
                    {"role": "user", "content": "Tell me a story!"},
                ],
                "sample_data": {
                    "identifier": "h0",
                    "expected_output": "exp0",
                    "comma_separated_tags": "a,b",
                },
            },
            {
                "messages": [
                    {"role": "system", "content": "You are verbose."},
                    {"role": "user", "content": "Tell me a short story."},
                ],
                "sample_data": {
                    "identifier": "h1",
                    "expected_output": "exp1",
                    "comma_separated_tags": "",
                },
            },
        ]

        class FakeConversationDataModule:
            config = types.SimpleNamespace(keep_generated_datasets_in_memory=True)

            def get_hf_messages_dataset(
                self,
                subset_name: str,
                append_answer: bool,
                keep_original_data: bool,
            ) -> hf_datasets.Dataset:
                return hf_datasets.Dataset.from_list(samples)

        class FakeModel:
            def __init__(self) -> None:
                self.config = types.SimpleNamespace()
                self.generation_config = {
                    "max_new_tokens": 10,
                    "max_length": 32,
                    "validate": lambda: None,
                }

        config = pyine.evals.code_exec.configs.CodeExecEvalsConfig(
            eval_generation_max_new_tokens_override=5,
        )
        result = await config.evaluate_hf_model(
            model=FakeModel(),
            tokenizer=tests.utils.transformers.utils.SimpleTokenizer(),
            datamodule=FakeConversationDataModule(),
            eval_subset_name="subset",
            verbose=True,
        )

        assert isinstance(result, FakeEvalResult)
        assert result.metrics["count"] == 2

    @pytest.mark.asyncio
    async def test_type_checks_enforce_hf_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """TypeError raised if model is not an HF model."""
        config = pyine.evals.code_exec.configs.CodeExecEvalsConfig()
        monkeypatch.setattr(
            pyine.evals.code_exec._impl.pyine.utils.transformers,
            "is_hf_model",
            lambda *_: False,
        )

        with pytest.raises(TypeError):
            await config.evaluate_hf_model(
                model=object(),
                tokenizer=None,
                datamodule=object(),
                eval_subset_name="subset",
            )


class TestWandBIntegration:
    """Tests for W&B integration methods."""

    def test_define_metrics_registers_expected_metrics(
        self,
        mock_wandb_run: typing.Any,
    ) -> None:
        """define_metrics_for_wandb registers expected metric names."""
        # import the function from utils since it's defined there
        pyine.evals.code_exec.utils.define_metrics_for_wandb(
            wandb_run=mock_wandb_run,
            prefix="test",
        )
        defined_names = mock_wandb_run.get_defined_metric_names()
        # check that some expected metrics are defined
        assert any("grader" in name for name in defined_names)
        assert any("sample_count" in name for name in defined_names)

    def test_define_metrics_with_no_prefix(
        self,
        mock_wandb_run: typing.Any,
    ) -> None:
        """define_metrics_for_wandb works without prefix."""
        pyine.evals.code_exec.utils.define_metrics_for_wandb(
            wandb_run=mock_wandb_run,
            prefix=None,
        )

        defined_names = mock_wandb_run.get_defined_metric_names()
        assert len(defined_names) > 0


class TestComplexityStats:
    """Tests for compute_aggregated_complexity_stats."""

    def test_aggregation_computes_all_statistics(self) -> None:
        """Aggregation computes mean, median, std, min, max for all metrics."""
        samples = [
            _make_sample_with_complexity_metrics(loc=10, sloc=8),
            _make_sample_with_complexity_metrics(loc=30, sloc=12),
        ]
        stats = pyine.evals.code_exec.utils.compute_aggregated_complexity_stats(samples)
        # check all aggregation types present for loc
        assert "loc_mean" in stats
        assert "loc_median" in stats
        assert "loc_std" in stats
        assert "loc_min" in stats
        assert "loc_max" in stats
        # verify computed values
        assert stats["loc_mean"] == pytest.approx(20.0)
        assert stats["loc_median"] == pytest.approx(20.0)
        assert stats["loc_std"] == pytest.approx(10.0)
        assert stats["loc_min"] == pytest.approx(10.0)
        assert stats["loc_max"] == pytest.approx(30.0)
        # check sloc stats
        assert stats["sloc_mean"] == pytest.approx(10.0)
        assert stats["sloc_median"] == pytest.approx(10.0)
        assert stats["sloc_std"] == pytest.approx(2.0)

    def test_empty_input_returns_empty_dict(self) -> None:
        """Empty sample list returns empty stats dict."""
        stats = pyine.evals.code_exec.utils.compute_aggregated_complexity_stats([])
        assert stats == {}

    @pytest.mark.parametrize(
        ("sample_count", "expected_mean"),
        [
            (1, 10.0),
            (3, 10.0),
            (5, 10.0),
        ],
        ids=["single", "three", "five"],
    )
    def test_aggregation_with_varying_sample_counts(
        self,
        sample_count: int,
        expected_mean: float,
    ) -> None:
        """Aggregation works correctly with different sample counts."""
        samples = [_make_sample_with_complexity_metrics(loc=10, sloc=8) for _ in range(sample_count)]
        stats = pyine.evals.code_exec.utils.compute_aggregated_complexity_stats(samples)
        assert stats["loc_mean"] == pytest.approx(expected_mean)
        assert stats["loc_std"] == pytest.approx(0.0)  # all same value


class TestExtractPromptFromHandler:
    """Tests for _extract_prompt_from_handler event selection logic."""

    def test_prefers_messages_over_prompts(self) -> None:
        """When both on_llm_start and on_chat_model_start fire, messages win."""
        handler = pyine.utils.langchain.CaptureLLMHandler()
        handler.on_llm_start(serialized={}, prompts=["plain prompt"])
        handler.on_chat_model_start(
            serialized={},
            messages=[[langchain_core.messages.HumanMessage(content="structured")]],
        )
        text_store: dict[str, str] = {}
        msg_store: dict[str, list[dict[str, typing.Any]]] = {}
        pyine.evals.code_exec._impl._extract_prompt_from_handler(handler, "s1", text_store, msg_store)
        assert "s1" not in text_store  # plain prompt should NOT be stored
        assert "s1" in msg_store
        assert msg_store["s1"][0]["content"] == "structured"

    def test_falls_back_to_prompts_when_no_messages(self) -> None:
        """When only on_llm_start fires, plain prompt is stored."""
        handler = pyine.utils.langchain.CaptureLLMHandler()
        handler.on_llm_start(serialized={}, prompts=["plain prompt"])
        text_store: dict[str, str] = {}
        msg_store: dict[str, list[dict[str, typing.Any]]] = {}
        pyine.evals.code_exec._impl._extract_prompt_from_handler(handler, "s1", text_store, msg_store)
        assert text_store["s1"] == "plain prompt"
        assert "s1" not in msg_store

    def test_no_events_stores_nothing(self) -> None:
        """When no llm_start events exist, nothing is stored (deferred validation)."""
        handler = pyine.utils.langchain.CaptureLLMHandler()
        text_store: dict[str, str] = {}
        msg_store: dict[str, list[dict[str, typing.Any]]] = {}
        pyine.evals.code_exec._impl._extract_prompt_from_handler(handler, "s1", text_store, msg_store)
        assert "s1" not in text_store
        assert "s1" not in msg_store

    def test_messages_event_ordering_last_wins(self) -> None:
        """When multiple chat model starts fire, the latest messages event is used."""
        handler = pyine.utils.langchain.CaptureLLMHandler()
        handler.on_chat_model_start(
            serialized={},
            messages=[[langchain_core.messages.HumanMessage(content="first")]],
        )
        handler.on_chat_model_start(
            serialized={},
            messages=[[langchain_core.messages.HumanMessage(content="second")]],
        )
        text_store: dict[str, str] = {}
        msg_store: dict[str, list[dict[str, typing.Any]]] = {}
        pyine.evals.code_exec._impl._extract_prompt_from_handler(handler, "s1", text_store, msg_store)
        assert msg_store["s1"][0]["content"] == "second"
