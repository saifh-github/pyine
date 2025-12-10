import types

import pytest

import pyine.apps.data.hf_precacher


@pytest.mark.asyncio
async def test_main_exits_on_dry_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hf_precacher = pyine.apps.data.hf_precacher
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
    hf_precacher = pyine.apps.data.hf_precacher

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
            force_regenerate: bool = False,
            epoch: int | None = None,
        ) -> object:
            self.requests.append(
                {
                    "subset_name": subset_name,
                    "tokenizer": tokenizer,
                    "max_seq_len": model_max_seq_len,
                    "force_regenerate": force_regenerate,
                    "epoch": epoch,
                },
            )
            return object()

        def teardown(
            self,
        ) -> None:
            self.teardown_calls += 1

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
            self.datamodule_config = types.SimpleNamespace(
                train_subset_names=["train"],
                valid_subset_names=["valid"],
                eval_subset_names=[],
            )

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
    monkeypatch.setattr(
        hf_precacher.pyine.data.datamodule,
        "ConversationDataModule",
        _FakeConversationDataModule,
    )
    monkeypatch.setattr(
        hf_precacher.pyine.data.datamodule,
        "ConversationDataModule",
        _FakeConversationDataModule,
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
    assert [call["subset_name"] for call in datamodule_instance.requests] == ["train", "valid"]
    assert all(call["tokenizer"] == "tokenizer" for call in datamodule_instance.requests)
    assert all(call["max_seq_len"] == 64 for call in datamodule_instance.requests)
    assert all(not call["force_regenerate"] for call in datamodule_instance.requests)
    assert datamodule_instance.requests[0]["epoch"] == 0
    assert datamodule_instance.requests[1]["epoch"] is None
    assert datamodule_instance.teardown_calls == 1
    assert runtime.finalize_calls == 1


@pytest.mark.asyncio
async def test_main_precaches_with_force_regenerate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hf_precacher = pyine.apps.data.hf_precacher

    class _Runtime:
        def finalize(self) -> None:
            return None

    class _FakeConversationDataModule:
        def __init__(
            self,
        ) -> None:
            self.requests: list[dict[str, object]] = []

        def get_hf_tokenized_examples_dataset(
            self,
            subset_name: str,
            tokenizer: object,
            model_max_seq_len: int,
            force_regenerate: bool = False,
            epoch: int | None = None,
        ) -> object:
            self.requests.append(
                {
                    "subset_name": subset_name,
                    "force_regenerate": force_regenerate,
                    "tokenizer": tokenizer,
                    "max_seq_len": model_max_seq_len,
                    "epoch": epoch,
                },
            )
            return object()

        def teardown(self) -> None:
            return None

    datamodule_instance = _FakeConversationDataModule()

    def fake_prepare_datamodule(
        *_args: object,
        **_kwargs: object,
    ) -> _FakeConversationDataModule:
        return datamodule_instance

    monkeypatch.setattr(
        hf_precacher.pyine.apps.trainers.common,
        "prepare_datamodule",
        fake_prepare_datamodule,
    )
    monkeypatch.setattr(
        hf_precacher.pyine.utils.reprod,
        "entrypoint_setup",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        hf_precacher.pyine.data.datamodule,
        "ConversationDataModule",
        _FakeConversationDataModule,
    )
    precache_config = hf_precacher.PrecacherConfig(
        include_eval_subsets=True,
        max_seq_len_override=8,
        force_regenerate=True,
    )
    config = types.SimpleNamespace(
        base_model="unit-test-model",
        auto_model_config={},
        get_tokenizer=lambda: "tokenizer",
        datamodule_config=types.SimpleNamespace(
            train_subset_names=["train"],
            valid_subset_names=["valid"],
            eval_subset_names=["eval"],
        ),
    )
    runtime = _Runtime()
    await hf_precacher.main(
        config=config,
        runtime=runtime,
        precache_config=precache_config,
    )
    assert datamodule_instance.requests
    assert all(call["force_regenerate"] for call in datamodule_instance.requests)
    assert [call["subset_name"] for call in datamodule_instance.requests] == ["train", "valid", "eval"]
    assert datamodule_instance.requests[0]["epoch"] == 0
    assert datamodule_instance.requests[1]["epoch"] is None
    assert datamodule_instance.requests[2]["epoch"] is None


@pytest.mark.asyncio
async def test_main_precaches_multiple_train_epochs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hf_precacher = pyine.apps.data.hf_precacher

    class _Runtime:
        def finalize(self) -> None:
            return None

    class _TrainParser:
        def __init__(self) -> None:
            self.set_epoch_calls: list[int] = []

        def __len__(self) -> int:
            return 8

        def set_epoch(self, epoch: int) -> None:
            self.set_epoch_calls.append(epoch)

    class _FakeConversationDataModule:
        def __init__(self) -> None:
            self.requests: list[dict[str, object]] = []
            self.parser = _TrainParser()

        def get_hf_tokenized_examples_dataset(
            self,
            subset_name: str,
            tokenizer: object,
            model_max_seq_len: int,
            force_regenerate: bool = False,
            epoch: int | None = None,
        ) -> object:
            self.requests.append(
                {
                    "subset_name": subset_name,
                    "tokenizer": tokenizer,
                    "max_seq_len": model_max_seq_len,
                    "force_regenerate": force_regenerate,
                    "epoch": epoch,
                },
            )
            return object()

        def get_parser(self, subset_name: str) -> _TrainParser:
            assert subset_name == "train"
            return self.parser

        def teardown(self) -> None:
            return None

    class _TrainingArgs:
        def __init__(self) -> None:
            self.num_train_epochs = 2.0
            self.gradient_accumulation_steps = 1
            self.per_device_train_batch_size = 2
            self.dataloader_drop_last = False
            self.max_steps = -1

    precache_config = hf_precacher.PrecacherConfig(
        include_eval_subsets=False,
        max_seq_len_override=32,
    )
    datamodule_instance = _FakeConversationDataModule()

    def fake_prepare_datamodule(
        *_args: object,
        **_kwargs: object,
    ) -> _FakeConversationDataModule:
        return datamodule_instance

    config = types.SimpleNamespace(
        base_model="unit-test-model",
        auto_model_config={},
        get_tokenizer=lambda: "tokenizer",
        training_args_config=_TrainingArgs(),
        datamodule_config=types.SimpleNamespace(
            train_subset_names=["train"],
            valid_subset_names=["valid"],
            eval_subset_names=[],
        ),
    )
    runtime = _Runtime()
    monkeypatch.setattr(
        hf_precacher.pyine.utils.reprod,
        "entrypoint_setup",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        hf_precacher.pyine.apps.trainers.common,
        "prepare_datamodule",
        fake_prepare_datamodule,
    )
    monkeypatch.setattr(
        hf_precacher.pyine.data.datamodule,
        "ConversationDataModule",
        _FakeConversationDataModule,
    )
    await hf_precacher.main(
        config=config,
        runtime=runtime,
        precache_config=precache_config,
    )
    assert [call["subset_name"] for call in datamodule_instance.requests] == ["train", "train", "valid"]
    assert [call["epoch"] for call in datamodule_instance.requests][:2] == [0, 1]
    assert datamodule_instance.parser.set_epoch_calls == [0, 1]


@pytest.mark.asyncio
async def test_main_respects_epochs_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hf_precacher = pyine.apps.data.hf_precacher

    class _Runtime:
        def finalize(self) -> None:
            return None

    class _TrainParser:
        def __init__(self) -> None:
            self.set_epoch_calls: list[int] = []

        def __len__(self) -> int:
            return 2

        def set_epoch(self, epoch: int) -> None:
            self.set_epoch_calls.append(epoch)

    class _FakeConversationDataModule:
        def __init__(self) -> None:
            self.requests: list[dict[str, object]] = []
            self.parser = _TrainParser()

        def get_hf_tokenized_examples_dataset(
            self,
            subset_name: str,
            tokenizer: object,
            model_max_seq_len: int,
            force_regenerate: bool = False,
            epoch: int | None = None,
        ) -> object:
            self.requests.append(
                {
                    "subset_name": subset_name,
                    "epoch": epoch,
                },
            )
            return object()

        def get_parser(self, subset_name: str) -> _TrainParser:
            return self.parser

        def teardown(self) -> None:
            return None

    class _TrainingArgs:
        def __init__(self) -> None:
            self.num_train_epochs = 1.0
            self.gradient_accumulation_steps = 1
            self.per_device_train_batch_size = 1
            self.dataloader_drop_last = False
            self.max_steps = -1

    precache_config = hf_precacher.PrecacherConfig(
        include_eval_subsets=False,
        max_seq_len_override=16,
        epochs_override=3,
    )
    datamodule_instance = _FakeConversationDataModule()

    def fake_prepare_datamodule(
        *_args: object,
        **_kwargs: object,
    ) -> _FakeConversationDataModule:
        return datamodule_instance

    config = types.SimpleNamespace(
        base_model="unit-test-model",
        auto_model_config={},
        get_tokenizer=lambda: "tokenizer",
        training_args_config=_TrainingArgs(),
        datamodule_config=types.SimpleNamespace(
            train_subset_names=["train"],
            valid_subset_names=["valid"],
            eval_subset_names=[],
        ),
    )
    runtime = _Runtime()
    monkeypatch.setattr(
        hf_precacher.pyine.utils.reprod,
        "entrypoint_setup",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        hf_precacher.pyine.apps.trainers.common,
        "prepare_datamodule",
        fake_prepare_datamodule,
    )
    monkeypatch.setattr(
        hf_precacher.pyine.data.datamodule,
        "ConversationDataModule",
        _FakeConversationDataModule,
    )
    await hf_precacher.main(
        config=config,
        runtime=runtime,
        precache_config=precache_config,
    )
    assert [call["subset_name"] for call in datamodule_instance.requests] == ["train", "train", "train", "valid"]
    assert [call["epoch"] for call in datamodule_instance.requests][:3] == [0, 1, 2]
    assert datamodule_instance.parser.set_epoch_calls == [0, 1, 2]
