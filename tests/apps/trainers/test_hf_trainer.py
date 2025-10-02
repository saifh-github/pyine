import pathlib
import types
import typing

import pytest

import pyine.apps.trainers.hf_trainer


class _FakePreparedDataset(list):
    pass


def test_train_configures_trainer_and_saves_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    raw_datasets: list[str] = []
    prepared_calls: list[dict[str, int]] = []

    class _RawDataset:
        def __init__(self, name: str) -> None:
            self.name = name

    class _FakeDataModule:
        def get_hf_messages_dataset(
            self,
            subset_name: str,
            append_answer: bool,
        ) -> _RawDataset:
            raw_datasets.append(f"{subset_name}:{append_answer}")
            return _RawDataset(subset_name)

    class _FakeModel:
        def __init__(self) -> None:
            self.config = types.SimpleNamespace(use_cache=True)
            self.gradient_checkpointing_enabled = False

        def gradient_checkpointing_enable(self) -> None:
            self.gradient_checkpointing_enabled = True

    class _FakeTokenizer:
        def __init__(self) -> None:
            self.saved_to: pathlib.Path | None = None

        def save_pretrained(
            self,
            output_dir: str,
        ) -> None:
            self.saved_to = pathlib.Path(output_dir)

    class _FakeTrainingArgsConfig:
        def __init__(self) -> None:
            self.do_train = True

        def model_dump(self) -> dict[str, object]:
            return {
                "output_dir": "ignored",
                "per_device_train_batch_size": 2,
            }

    class _FakeTrainer:
        def __init__(
            self,
            *,
            model,
            args,
            train_dataset,
            eval_dataset,
            data_collator,
            tokenizer,
            compute_metrics,
        ) -> None:
            self.model = model
            self.args = args
            self.train_dataset = train_dataset
            self.eval_dataset = eval_dataset
            self.data_collator = data_collator
            self.tokenizer = tokenizer
            self.compute_metrics = compute_metrics
            self.saved_to: str | None = None
            self.trained = False

        def train(self):
            self.trained = True
            return "done"

        def save_model(
            self,
            output_dir: str,
        ) -> None:
            self.saved_to = output_dir

    def fake_prepare_examples_from_conversations(
        convo_ds,
        tokenizer,
        max_seq_len,
        num_proc,
    ) -> _FakePreparedDataset:
        prepared_calls.append(
            {
                "name": convo_ds.name,
                "max_seq_len": max_seq_len,
                "num_proc": num_proc,
            },
        )
        return _FakePreparedDataset([convo_ds.name])

    captured_args: dict[str, dict[str, object]] = {}

    def fake_training_arguments(**kwargs):
        captured_args["kwargs"] = kwargs
        return types.SimpleNamespace(**kwargs)

    fake_trainer_instance = _FakeTrainer(
        model=_FakeModel(),
        args=types.SimpleNamespace(),
        train_dataset=_FakePreparedDataset(),
        eval_dataset=_FakePreparedDataset(),
        data_collator=None,
        tokenizer=_FakeTokenizer(),
        compute_metrics=lambda *_args, **_kwargs: {},
    )

    def fake_trainer_factory(**kwargs):
        fake_trainer_instance.model = kwargs["model"]
        fake_trainer_instance.args = kwargs["args"]
        fake_trainer_instance.train_dataset = kwargs["train_dataset"]
        fake_trainer_instance.eval_dataset = kwargs["eval_dataset"]
        fake_trainer_instance.data_collator = kwargs["data_collator"]
        fake_trainer_instance.tokenizer = kwargs["tokenizer"]
        fake_trainer_instance.compute_metrics = kwargs["compute_metrics"]
        return fake_trainer_instance

    monkeypatch.setattr(
        pyine.apps.trainers.hf_trainer.pyine.utils.transformers,
        "infer_effective_max_seq_len",
        lambda *_args, **_kwargs: 64,
    )
    monkeypatch.setattr(
        pyine.apps.trainers.hf_trainer.pyine.utils.transformers,
        "prepare_examples_from_conversations",
        fake_prepare_examples_from_conversations,
    )
    monkeypatch.setattr(
        pyine.apps.trainers.hf_trainer.pyine.utils.transformers,
        "FixedSizePaddingCollatorWithPromptMask",
        lambda *_args, **_kwargs: "collator",
    )
    monkeypatch.setattr(
        pyine.apps.trainers.hf_trainer.transformers,
        "TrainingArguments",
        fake_training_arguments,
    )
    monkeypatch.setattr(
        pyine.apps.trainers.hf_trainer.transformers,
        "Trainer",
        fake_trainer_factory,
    )

    fake_model = _FakeModel()
    fake_tokenizer = _FakeTokenizer()
    fake_dm = _FakeDataModule()

    runtime = types.SimpleNamespace(wandb_run=object())
    config = types.SimpleNamespace(
        training_args_config=_FakeTrainingArgsConfig(),
        datamodule_config=types.SimpleNamespace(
            train_subset_names=["train"],
            valid_subset_names=["valid"],
        ),
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        use_wandb_logging=True,
        output_dir=str(tmp_path / "artifact"),
    )

    trainer = pyine.apps.trainers.hf_trainer.train(
        model=fake_model,
        tokenizer=fake_tokenizer,
        datamodule=fake_dm,
        config=config,
        runtime=runtime,
    )

    assert trainer is fake_trainer_instance
    assert fake_model.config.use_cache is False
    assert fake_model.gradient_checkpointing_enabled is True
    assert prepared_calls == [
        {"name": "train", "max_seq_len": 64, "num_proc": 2},
        {"name": "valid", "max_seq_len": 64, "num_proc": 2},
    ]
    assert captured_args["kwargs"]["report_to"] == ["wandb"]
    assert fake_trainer_instance.saved_to == config.output_dir
    assert fake_tokenizer.saved_to == pathlib.Path(config.output_dir)


@pytest.mark.asyncio
async def test_main_runs_train_and_evaluate(monkeypatch: pytest.MonkeyPatch) -> None:
    train_calls: list[dict] = []
    eval_calls: list[dict] = []

    async def fake_evaluate_model(
        **kwargs: object,
    ) -> None:
        eval_calls.append(kwargs)

    def fake_train(
        **kwargs: object,
    ) -> str:
        train_calls.append(kwargs)
        return "trainer"

    def fake_prepare_datamodule(
        config: object,
        runtime: object,
    ) -> str:
        return "datamodule"

    def fake_entrypoint_setup(
        **_kwargs: object,
    ) -> None:
        return None

    config = types.SimpleNamespace(
        training_args_config=types.SimpleNamespace(do_train=True, do_predict=True),
        get_model=lambda: "model",
        get_tokenizer=lambda: "tokenizer",
        use_wandb_logging=False,
    )
    runtime = types.SimpleNamespace()

    monkeypatch.setattr(
        pyine.apps.trainers.hf_trainer.pyine.utils.reprod,
        "entrypoint_setup",
        fake_entrypoint_setup,
    )
    monkeypatch.setattr(
        pyine.apps.trainers.hf_trainer,
        "train",
        fake_train,
    )
    monkeypatch.setattr(
        pyine.apps.trainers.hf_trainer.pyine.apps.trainers.common,
        "prepare_datamodule",
        fake_prepare_datamodule,
    )
    monkeypatch.setattr(
        pyine.apps.trainers.hf_trainer.pyine.apps.trainers.common,
        "evaluate_model",
        fake_evaluate_model,
    )

    await pyine.apps.trainers.hf_trainer.main(
        config=config,
        runtime=runtime,
    )

    assert train_calls and eval_calls
    assert train_calls[0]["runtime"] is runtime
    assert eval_calls[0]["model"] == "model"


@pytest.mark.asyncio
async def test_main_exits_on_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_entrypoint_setup(
        **_kwargs: object,
    ) -> typing.NoReturn:
        raise pyine.apps.trainers.hf_trainer.pyine.utils.reprod.DryRunExit()

    def fail_prepare_datamodule(
        *_args: object,
        **_kwargs: object,
    ) -> typing.NoReturn:
        raise AssertionError("should not be called")

    config = types.SimpleNamespace(
        training_args_config=types.SimpleNamespace(do_train=True, do_predict=True),
        use_wandb_logging=False,
    )

    monkeypatch.setattr(
        pyine.apps.trainers.hf_trainer.pyine.utils.reprod,
        "entrypoint_setup",
        fake_entrypoint_setup,
    )
    monkeypatch.setattr(
        pyine.apps.trainers.hf_trainer.pyine.apps.trainers.common,
        "prepare_datamodule",
        fail_prepare_datamodule,
    )

    await pyine.apps.trainers.hf_trainer.main(
        config=config,
        runtime=None,
    )
