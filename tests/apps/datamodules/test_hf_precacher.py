import types

import pytest

import pyine.apps.datamodules.hf_precacher


@pytest.mark.asyncio
async def test_main_exits_on_dry_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hf_precacher = pyine.apps.datamodules.hf_precacher
    entrypoint_calls: list[dict[str, object]] = []

    def fake_entrypoint_setup(
        **_kwargs: object,
    ) -> None:
        entrypoint_calls.append(_kwargs)
        raise hf_precacher.pyine.utils.reprod.DryRunExit()

    def fail_prepare_datamodule(
        *_args: object,
        **_kwargs: object,
    ) -> None:
        raise AssertionError("prepare_datamodule should not run")

    monkeypatch.setattr(
        hf_precacher.pyine.utils.reprod,
        "entrypoint_setup",
        fake_entrypoint_setup,
    )
    monkeypatch.setattr(
        hf_precacher.pyine.apps.trainers.common,
        "prepare_datamodule",
        fail_prepare_datamodule,
    )
    await hf_precacher.main(
        config=types.SimpleNamespace(),
        runtime=types.SimpleNamespace(),
    )
    assert entrypoint_calls


@pytest.mark.asyncio
async def test_main_precaches_expected_subsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hf_precacher = pyine.apps.datamodules.hf_precacher

    class _Runtime:
        def __init__(
            self,
        ) -> None:
            self.finalize_calls = 0

        def finalize(
            self,
        ) -> None:
            self.finalize_calls += 1

    class _FakeConversationDataModule:
        def __init__(
            self,
        ) -> None:
            self.requests: list[dict[str, object]] = []
            self.teardown_calls = 0

        def get_hf_tokenized_examples_dataset(
            self,
            subset_name: str,
            tokenizer: object,
            model_max_seq_len: int,
        ) -> object:
            self.requests.append(
                {
                    "subset_name": subset_name,
                    "tokenizer": tokenizer,
                    "max_seq_len": model_max_seq_len,
                },
            )
            return object()

        def teardown(
            self,
        ) -> None:
            self.teardown_calls += 1

    monkeypatch.setattr(
        hf_precacher.pyine.data.datamodule,
        "ConversationDataModule",
        _FakeConversationDataModule,
    )
    runtime = _Runtime()
    precache_config = hf_precacher.PrecacherConfig(
        include_eval_subsets=False,
        max_seq_len_override=64,
    )
    entrypoint_calls: list[dict[str, object]] = []

    def fake_entrypoint_setup(
        **kwargs: object,
    ) -> None:
        entrypoint_calls.append(kwargs)

    tokenizer_calls = 0

    class _Config:
        def __init__(
            self,
        ) -> None:
            self.base_model = "unit-test-model"
            self.auto_model_config: dict[str, object] = {}

        def get_tokenizer(
            self,
        ) -> object:
            nonlocal tokenizer_calls
            tokenizer_calls += 1
            return "tokenizer"

    datamodule_instance = _FakeConversationDataModule()

    def fake_prepare_datamodule(
        config: object,
        runtime_config: object,
    ) -> _FakeConversationDataModule:
        assert config is test_config
        assert runtime_config is runtime
        return datamodule_instance

    monkeypatch.setattr(
        hf_precacher.pyine.utils.reprod,
        "entrypoint_setup",
        fake_entrypoint_setup,
    )
    monkeypatch.setattr(
        hf_precacher.pyine.apps.trainers.common,
        "prepare_datamodule",
        fake_prepare_datamodule,
    )
    test_config = _Config()
    await hf_precacher.main(
        config=test_config,
        runtime=runtime,
        precache_config=precache_config,
    )
    assert entrypoint_calls
    assert entrypoint_calls[0]["precache_config"] is precache_config
    assert tokenizer_calls == 1
    assert len(datamodule_instance.requests) == 2
    assert all(call["tokenizer"] == "tokenizer" for call in datamodule_instance.requests)
    assert all(call["max_seq_len"] == 64 for call in datamodule_instance.requests)
    assert datamodule_instance.teardown_calls == 1
    assert runtime.finalize_calls == 1
