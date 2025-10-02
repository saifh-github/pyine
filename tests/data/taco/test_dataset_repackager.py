import json
import pathlib
import types
import typing

import pytest

import pyine.data.taco.dataset_repackager


class _FakeTokenizer:
    def encode(
        self,
        text: str,
    ) -> list[int]:
        return list(range(len(text.split())))


class _RetryChain:
    def __init__(self) -> None:
        self.call_count = 0
        self.seen_inputs: list[dict[str, str]] = []

    async def ainvoke(
        self,
        input_data: dict[str, str],
        config: dict[str, int],
    ) -> dict[str, str]:
        self.call_count += 1
        self.seen_inputs.append(input_data)
        if self.call_count == 1:
            raise RuntimeError("transient failure")
        return {"analysis": str(input_data["code"]).upper()}


@pytest.mark.asyncio
async def test_reprocess_code_samples_handles_retries_and_writes_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    output_root = tmp_path / "outputs"
    dataset_entries = [
        {
            "solutions": [
                "print('ok')",
                "def bad(): pass",
            ],
        },
    ]

    class _FakeReader:
        def __len__(self) -> int:
            return len(dataset_entries)

        def __getitem__(
            self,
            idx: int,
        ) -> dict[str, list[object]]:
            return dataset_entries[idx]

    chain = _RetryChain()
    sleep_calls: list[float] = []

    def fake_get_model_from_provider(
        **_kwargs: typing.Any,
    ) -> object:
        return object()

    def fake_get_prompt_template(_name: str) -> str:
        return "{code}"

    def fake_get_prompt_chain(
        _model: typing.Any,
        _name: str,
    ) -> _RetryChain:
        return chain

    def fake_validate_code(code: str) -> None:
        if "bad" in code:
            raise ValueError("invalid code")

    async def fake_sleep(
        seconds: float,
    ) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr(
        pyine.data.taco.dataset_repackager.pyine.utils.llm_providers,
        "get_model_from_provider",
        fake_get_model_from_provider,
    )
    monkeypatch.setattr(
        pyine.data.taco.dataset_repackager.pyine.prompts.manager,
        "get_prompt_template",
        fake_get_prompt_template,
    )
    monkeypatch.setattr(
        pyine.data.taco.dataset_repackager.pyine.prompts.manager,
        "get_prompt_chain",
        fake_get_prompt_chain,
    )
    monkeypatch.setattr(
        pyine.data.taco.dataset_repackager.pyine.utils.code.validation,
        "validate_code",
        fake_validate_code,
    )
    monkeypatch.setattr(
        pyine.data.taco.dataset_repackager.tiktoken,
        "encoding_for_model",
        lambda _model: _FakeTokenizer(),
    )
    monkeypatch.setattr(
        pyine.data.taco.dataset_repackager.pyine.data.taco.dataset_reader,
        "DatasetReader",
        lambda: _FakeReader(),
    )
    monkeypatch.setattr(
        pyine.data.taco.dataset_repackager.asyncio,
        "sleep",
        fake_sleep,
    )
    await pyine.data.taco.dataset_repackager.reprocess_code_samples(
        output_dir_path=output_root,
        batch_size=4,
        chunk_size=1,
        max_token_count=500,
    )
    written_file = output_root / "000000.json"
    assert written_file.exists()
    payload = json.loads(written_file.read_text())
    assert len(payload["solutions"]) == 2
    string_solution = payload["solutions"][0]
    second_solution = payload["solutions"][1]
    assert string_solution["analysis_outputs"][0]["analysis"] == "PRINT('OK')"
    assert second_solution["validation_errors"] == ["invalid code"]
    assert second_solution["analysis_outputs"][0]["analysis"] == "DEF BAD(): PASS"
    assert chain.call_count == 3
    assert sleep_calls, "expected retry backoff to trigger sleeps"


@pytest.mark.asyncio
async def test_reprocess_code_samples_enforces_token_cap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    class _CountingTokenizer:
        def encode(
            self,
            text: str,
        ) -> list[int]:
            return list(range(10))

    class _FakeReader:
        def __len__(self) -> int:
            return 1

        def __getitem__(
            self,
            idx: int,
        ) -> dict:
            return {"solutions": ["raise ValueError"]}

    async def _noop_invoke(
        *_args: typing.Any,
        **_kwargs: typing.Any,
    ) -> None:
        return None

    async def _async_noop(
        *_args: typing.Any,
        **_kwargs: typing.Any,
    ) -> None:
        return None

    monkeypatch.setattr(
        pyine.data.taco.dataset_repackager.pyine.data.taco.dataset_reader,
        "DatasetReader",
        lambda: _FakeReader(),
    )
    monkeypatch.setattr(
        pyine.data.taco.dataset_repackager.tiktoken,
        "encoding_for_model",
        lambda _model: _CountingTokenizer(),
    )
    monkeypatch.setattr(
        pyine.data.taco.dataset_repackager.pyine.prompts.manager,
        "get_prompt_template",
        lambda _name: "{code}",
    )
    monkeypatch.setattr(
        pyine.data.taco.dataset_repackager.pyine.prompts.manager,
        "get_prompt_chain",
        lambda *_args, **_kwargs: types.SimpleNamespace(ainvoke=_noop_invoke),
    )
    monkeypatch.setattr(
        pyine.data.taco.dataset_repackager.pyine.utils.llm_providers,
        "get_model_from_provider",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        pyine.data.taco.dataset_repackager.pyine.utils.code.validation,
        "validate_code",
        lambda _code: None,
    )
    monkeypatch.setattr(
        pyine.data.taco.dataset_repackager.asyncio,
        "sleep",
        _async_noop,
    )
    await pyine.data.taco.dataset_repackager.reprocess_code_samples(
        output_dir_path=tmp_path,
        batch_size=1,
        chunk_size=1,
        max_token_count=5,
    )
    written_file = tmp_path / "000000.json"
    assert written_file.exists()
    payload = json.loads(written_file.read_text())
    recorded_errors = payload["solutions"][0]["validation_errors"]
    assert recorded_errors and "exceeds max" in recorded_errors[0]
