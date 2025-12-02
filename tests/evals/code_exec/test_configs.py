import dataclasses
import types
import typing

import datasets as hf_datasets
import langchain_core.messages
import pytest
import torch

import pyine.evals.code_exec.configs
import pyine.evals.common
import tests.utils.transformers.utils


class _FakeSample:
    def __init__(self, identifier: str) -> None:
        self.identifier = identifier
        self.expected_output = f"expected-{identifier}"
        self.predict_type = "unknown"
        self._tags = ["tag"]

    def _asdict(self) -> dict[str, str]:
        return {"identifier": self.identifier}

    def get_tag_list(self) -> list[str]:
        return list(self._tags)


class _FakeSampleBuilder(list):
    def __init__(self, samples: list[_FakeSample]) -> None:
        super().__init__(samples)


class _FakeDataModule:
    def __init__(self, samples: list[_FakeSample]) -> None:
        self._samples = samples

    def get_parser(
        self,
        subset_name: str,
    ) -> _FakeSampleBuilder:
        return _FakeSampleBuilder(self._samples)


@dataclasses.dataclass
class _FakeSampleEval:
    identifier: str
    hard_match: bool = True
    soft_match: str | None = None
    llm_score: float | None = None
    tags: list[str] | None = None


class _FakeOutcomeEvaluator:
    def __init__(
        self,
        llm_provider_config: typing.Any | None = None,
    ) -> None:
        self.llm_provider_config = llm_provider_config
        self.results: list[_FakeSampleEval] = []
        self.added: list[tuple[str, str, str, list[str]]] = []

    def add_sample(
        self,
        identifier: str,
        predicted: str,
        expected: str,
        predict_type: str = "unknown",
        tags: list[str] | None = None,
    ) -> None:
        self.added.append((identifier, expected, predicted, tags or []))
        self.results.append(
            _FakeSampleEval(
                identifier=identifier,
                tags=tags or [],
            ),
        )


@dataclasses.dataclass
class _FakeArtifact:
    sample: object
    eval_result: object


@dataclasses.dataclass
class _FakeEvalResult:
    metrics: dict[str, object]
    artifacts: list[_FakeArtifact]


def _build_message(identifier: str) -> langchain_core.messages.AIMessage:
    return langchain_core.messages.AIMessage(
        content=f"prediction-{identifier}",
        usage_metadata={
            "total_tokens": 2,
            "prompt_tokens": 1,
            "input_tokens": 1,
            "output_tokens": 1,
        },
    )


@pytest.mark.asyncio
async def test_evaluate_runnable_model_sequential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = [_FakeSample("s0"), _FakeSample("s1")]
    data_module = _FakeDataModule(samples)
    chain_calls: list[dict[str, str]] = []

    class _FakeChain:
        def invoke(
            self,
            payload: dict[str, str],
        ) -> langchain_core.messages.AIMessage:
            chain_calls.append(payload)
            return _build_message(payload["identifier"])

    async def fake_get_metrics(
        evaluator: _FakeOutcomeEvaluator,
        token_usage: typing.Any,
    ) -> dict[str, typing.Any]:
        return {
            "accuracy": len(evaluator.results),
            "total_tokens": token_usage.total_tokens,
        }

    def fake_tqdm(
        iterable: typing.Iterable[typing.Any],
        **_kwargs: typing.Any,
    ) -> typing.Iterable[typing.Any]:
        return iterable

    monkeypatch.setattr(
        pyine.evals.code_exec.configs.pyine.organisms.datamodules.samples,
        "SampleBuilder",
        _FakeSampleBuilder,
    )
    monkeypatch.setattr(
        pyine.evals.code_exec.configs.pyine.organisms.datamodules.samples,
        "SampleData",
        _FakeSample,
    )
    monkeypatch.setattr(
        pyine.evals.code_exec.configs.pyine.evals.code_exec.utils,
        "OutcomeEvaluator",
        _FakeOutcomeEvaluator,
    )
    monkeypatch.setattr(
        pyine.evals.code_exec.configs.pyine.evals.code_exec.utils,
        "CodeExecEvalArtifact",
        _FakeArtifact,
    )
    monkeypatch.setattr(
        pyine.evals.code_exec.configs,
        "CodeExecEvalResult",
        _FakeEvalResult,
    )
    monkeypatch.setattr(
        pyine.evals.code_exec.configs.pyine.evals.code_exec.utils,
        "get_metrics",
        fake_get_metrics,
    )
    monkeypatch.setattr(
        pyine.evals.code_exec.configs.tqdm,
        "tqdm",
        fake_tqdm,
    )

    config = pyine.evals.code_exec.configs.CodeExecEvalsConfig(
        eval_runnable_config=pyine.evals.common.RunnableEvalConfig(parallel=False),
    )
    result = await config.evaluate_runnable_model(
        chain=_FakeChain(),
        datamodule=data_module,
        eval_subset_name="subset",
        verbose=True,
    )

    assert isinstance(result, _FakeEvalResult)
    assert len(result.artifacts) == 2
    assert result.metrics["accuracy"] == 2
    assert chain_calls[0]["identifier"] == "s0"


@pytest.mark.asyncio
async def test_evaluate_runnable_model_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = [_FakeSample("p0"), _FakeSample("p1"), _FakeSample("p2")]
    data_module = _FakeDataModule(samples)
    submitted: list[int] = []
    progress_reports: list[int] = []

    class _FakeChain:
        def invoke(
            self,
            payload: dict[str, str],
        ) -> langchain_core.messages.AIMessage:
            return _build_message(payload["identifier"])

    async def fake_get_metrics(
        evaluator: _FakeOutcomeEvaluator,
        token_usage: typing.Any,
    ) -> dict[str, typing.Any]:
        return {"accuracy": len(evaluator.results), "tokens": token_usage.total_tokens}

    def fake_tqdm(
        iterable: typing.Iterable[typing.Any] | None = None,
        total: int | None = None,
        **_kwargs: typing.Any,
    ) -> typing.Any:
        class _Prog:
            def __init__(self, total: int | None) -> None:
                self.total = total
                self.history: list[int] = []

            def update(self, value: int) -> None:
                self.history.append(value)

            def write(self, _msg: str) -> None:
                return None

            def close(self) -> None:
                return None

        if iterable is None:
            return _Prog(total)
        return iterable

    class _FakeExecutor:
        def submit(
            self,
            fn: typing.Callable[[typing.Any], typing.Any],
            payload: typing.Any,
        ) -> "_FakeFuture":
            return _FakeFuture(fn(payload))

    async def fake_run_with_sliding_window(
        *,
        input_items: typing.Iterable[int],
        submit_one: typing.Callable[..., typing.Any],
        process_result: typing.Callable[[int, typing.Any], None],
        progress_callback: typing.Callable[[list[typing.Any], list[int]], typing.Awaitable[None]],
        **_kwargs: typing.Any,
    ) -> None:
        completed: list[int] = []
        executor = _FakeExecutor()
        for item in input_items:
            submitted.append(item)
            future_or_response = submit_one(item, executor=executor)
            response = (  # SIM108 guard to keep branch explicit for readability
                future_or_response.result() if hasattr(future_or_response, "result") else future_or_response
            )
            process_result(item, response)
            completed.append(item)
            await progress_callback([], completed)

    class _FakeFuture:
        def __init__(
            self,
            result: typing.Any,
        ) -> None:
            self._result = result

        def result(self) -> typing.Any:
            return self._result

    def fake_submit_one(
        sample_idx: int,
        executor: typing.Any,
    ) -> _FakeFuture:
        message = _build_message(f"p{sample_idx}")
        return _FakeFuture(message)

    async def fake_progress_callback(
        _pending: list[typing.Any],
        completed: list[int],
    ) -> None:
        progress_reports.append(len(completed))

    monkeypatch.setattr(
        pyine.evals.code_exec.configs.pyine.organisms.datamodules.samples,
        "SampleBuilder",
        _FakeSampleBuilder,
    )
    monkeypatch.setattr(
        pyine.evals.code_exec.configs.pyine.organisms.datamodules.samples,
        "SampleData",
        _FakeSample,
    )
    monkeypatch.setattr(
        pyine.evals.code_exec.configs.pyine.evals.code_exec.utils,
        "OutcomeEvaluator",
        _FakeOutcomeEvaluator,
    )
    monkeypatch.setattr(
        pyine.evals.code_exec.configs.pyine.evals.code_exec.utils,
        "CodeExecEvalArtifact",
        _FakeArtifact,
    )
    monkeypatch.setattr(
        pyine.evals.code_exec.configs,
        "CodeExecEvalResult",
        _FakeEvalResult,
    )
    monkeypatch.setattr(
        pyine.evals.code_exec.configs.pyine.evals.code_exec.utils,
        "get_metrics",
        fake_get_metrics,
    )
    monkeypatch.setattr(
        pyine.evals.code_exec.configs.tqdm,
        "tqdm",
        fake_tqdm,
    )
    monkeypatch.setattr(
        pyine.evals.code_exec.configs.pyine.utils.concurrency,
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
        chain=_FakeChain(),
        datamodule=data_module,
        eval_subset_name="subset",
        verbose=True,
    )

    assert submitted == [0, 1, 2]
    assert result.metrics["accuracy"] == 3


@pytest.mark.asyncio
async def test_evaluate_hf_model_generates_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = [
        {
            "messages": [
                {"role": "system", "content": "You are brief."},
                {"role": "user", "content": "Tell me a long story!"},
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

    class _FakeConversationDataModule:
        config = types.SimpleNamespace(keep_generated_datasets_in_memory=True)

        def get_hf_messages_dataset(
            self,
            subset_name: str,
            append_answer: bool,
            keep_original_data: bool,
        ) -> hf_datasets.Dataset:
            return hf_datasets.Dataset.from_list(samples)

    class _FakeModel:
        def __init__(self) -> None:
            self.config = types.SimpleNamespace()
            self.generation_config = {
                "max_new_tokens": 10,
                "max_length": 32,
                "validate": lambda: None,
            }

    @dataclasses.dataclass
    class _FakeSampleData:
        identifier: str
        expected_output: str
        comma_separated_tags: str
        predict_type: str = "unknown"

        def get_tag_list(self) -> list[str]:
            if not self.comma_separated_tags:
                return []
            return self.comma_separated_tags.split(",")

    def fake_is_hf_model(_model: typing.Any) -> bool:
        return True

    def fake_supports_text_generation(_model: typing.Any) -> bool:
        return True

    def fake_infer_effective_max_seq_len(
        _model: typing.Any,
        _tokenizer: typing.Any,
    ) -> int:
        return 16

    def fake_batchwise_padding_collator(
        **_kwargs: typing.Any,
    ) -> str:
        return "collator"

    def fake_run_text_generation(
        **_kwargs: typing.Any,
    ) -> list[dict[str, typing.Any]]:
        outputs = []
        for idx, batch in enumerate(_kwargs["dataloader"]):
            outputs.append(
                {
                    "sample_idx": batch["sample_idx"],
                    "prediction": f"pred-{idx}",
                    "generated_tokens": torch.as_tensor([0, 1]),
                },
            )
        return outputs

    async def fake_get_metrics(
        evaluator: _FakeOutcomeEvaluator,
        token_usage: typing.Any,
    ) -> dict[str, typing.Any]:
        return {"count": len(evaluator.results), "tokens": token_usage.total_tokens}

    def fake_data_loader(
        dataset: typing.Iterable[typing.Any],
        **_kwargs: typing.Any,
    ) -> typing.Iterable[typing.Any]:
        return dataset

    def fake_tqdm(
        iterable: typing.Iterable[typing.Any],
        **_kwargs: typing.Any,
    ) -> typing.Iterable[typing.Any]:
        return iterable

    monkeypatch.setattr(
        pyine.evals.code_exec.configs.pyine.organisms.datamodules.samples,
        "SampleData",
        _FakeSampleData,
    )
    monkeypatch.setattr(
        pyine.evals.code_exec.configs.pyine.data.datamodule,
        "ConversationDataModule",
        _FakeConversationDataModule,
    )
    monkeypatch.setattr(
        pyine.evals.code_exec.configs.pyine.utils.transformers,
        "is_hf_model",
        fake_is_hf_model,
    )
    monkeypatch.setattr(
        pyine.evals.code_exec.configs.pyine.utils.transformers,
        "supports_text_generation",
        fake_supports_text_generation,
    )
    monkeypatch.setattr(
        pyine.evals.code_exec.configs.pyine.utils.transformers,
        "infer_effective_max_seq_len",
        fake_infer_effective_max_seq_len,
    )
    monkeypatch.setattr(
        pyine.evals.code_exec.configs.pyine.utils.transformers,
        "PaddingCollatorWithPromptMask",
        fake_batchwise_padding_collator,
    )
    monkeypatch.setattr(
        pyine.evals.code_exec.configs.pyine.utils.transformers,
        "run_text_generation",
        fake_run_text_generation,
    )
    monkeypatch.setattr(
        pyine.evals.code_exec.configs.pyine.evals.code_exec.utils,
        "OutcomeEvaluator",
        _FakeOutcomeEvaluator,
    )
    monkeypatch.setattr(
        pyine.evals.code_exec.configs.pyine.evals.code_exec.utils,
        "CodeExecEvalArtifact",
        _FakeArtifact,
    )
    monkeypatch.setattr(
        pyine.evals.code_exec.configs,
        "CodeExecEvalResult",
        _FakeEvalResult,
    )
    monkeypatch.setattr(
        pyine.evals.code_exec.configs.pyine.evals.code_exec.utils,
        "get_metrics",
        fake_get_metrics,
    )
    monkeypatch.setattr(
        pyine.evals.code_exec.configs.tqdm,
        "tqdm",
        fake_tqdm,
    )
    monkeypatch.setattr(
        pyine.evals.code_exec.configs.torch.utils.data,
        "DataLoader",
        fake_data_loader,
    )
    config = pyine.evals.code_exec.configs.CodeExecEvalsConfig(
        eval_generation_max_new_tokens_override=5,
    )
    result = await config.evaluate_hf_model(
        model=_FakeModel(),
        tokenizer=tests.utils.transformers.utils.SimpleTokenizer(),
        datamodule=_FakeConversationDataModule(),
        eval_subset_name="subset",
        verbose=True,
    )
    assert isinstance(result, _FakeEvalResult)
    assert result.metrics["count"] == 2


@pytest.mark.asyncio
async def test_evaluate_hf_model_type_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    config = pyine.evals.code_exec.configs.CodeExecEvalsConfig()
    monkeypatch.setattr(
        pyine.evals.code_exec.configs.pyine.utils.transformers,
        "is_hf_model",
        lambda *_args, **_kwargs: False,
    )

    with pytest.raises(TypeError):
        await config.evaluate_hf_model(
            model=object(),
            tokenizer=None,
            datamodule=object(),
            eval_subset_name="subset",
        )
