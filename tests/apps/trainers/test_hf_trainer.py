import pathlib
import types
import typing

import pytest
import pytest_mock

import pyine.apps.trainers.hf_trainer
import pyine.data.datamodule
import pyine.evals.utils


class _FakePreparedDataset:
    def __init__(
        self,
        rows: list[dict[str, typing.Any]],
    ) -> None:
        self._rows = rows
        self.column_names = list(rows[0].keys()) if rows else []

    def __len__(
        self,
    ) -> int:
        return len(self._rows)

    def __iter__(
        self,
    ) -> typing.Iterator[dict[str, typing.Any]]:
        return iter(self._rows)

    def __getitem__(
        self,
        item: int | str,
    ) -> typing.Any:
        if isinstance(item, str):
            return [row[item] for row in self._rows if item in row]
        return self._rows[item]


def test_train_configures_trainer_and_saves_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    raw_datasets: list[str] = []
    prepared_calls: list[dict[str, typing.Any]] = []
    prepared_rows_by_subset = {
        "train": [
            {
                "input_ids": [101, 102],
                "attention_mask": [1, 1],
                "labels": [101, 102],
            },
            {
                "input_ids": [103, 104],
                "attention_mask": [1, 1],
                "labels": [103, 104],
            },
        ],
        "valid": [
            {
                "input_ids": [201, 202],
                "attention_mask": [1, 1],
                "labels": [201, 202],
                "sample_data": {"code_type": "bugfix"},
            },
            {
                "input_ids": [203, 204],
                "attention_mask": [1, 1],
                "labels": [203, 204],
                "sample_data": {"code_type": None},
            },
            {
                "input_ids": [205, 206],
                "attention_mask": [1, 1],
                "labels": [205, 206],
                "sample_data": {"code_type": "refactor"},
            },
        ],
    }

    class _RawDataset:
        def __init__(self, name: str) -> None:
            self.name = name

    class _FakeDataModule:
        def get_hf_messages_dataset(
            self,
            subset_name: str,
            append_answer: bool,
            keep_original_data: bool = False,
        ) -> _RawDataset:
            raw_datasets.append(f"{subset_name}:{append_answer}:{keep_original_data}")
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
            self.dataloader_num_workers = 4

        def model_dump(self) -> dict[str, object]:
            return {
                "output_dir": "ignored",
                "per_device_train_batch_size": 2,
            }

    class _FakeTrainer:
        def __init__(
            self,
            *,
            model: _FakeModel,
            args: typing.Any,
            train_dataset: _FakePreparedDataset,
            eval_dataset: _FakePreparedDataset,
            processing_class: _FakeTokenizer,
            data_collator: typing.Any,
            compute_metrics: typing.Callable[..., dict[str, typing.Any]] = None,
            callbacks: list[typing.Any] | None = None,
        ) -> None:
            self.model = model
            self.args = args
            self.train_dataset = train_dataset
            self.eval_dataset = eval_dataset
            self.data_collator = data_collator
            self.tokenizer = processing_class
            self.compute_metrics = compute_metrics
            self.callbacks = callbacks or []
            self.saved_to: str | None = None
            self.trained = False

        def train(self) -> str:
            self.trained = True
            return "done"

        def save_model(
            self,
            output_dir: str,
        ) -> None:
            self.saved_to = output_dir

    def fake_prepare_examples_from_conversations(
        convo_ds: _RawDataset,
        tokenizer: _FakeTokenizer,
        max_seq_len: int,
        num_proc: int,
        keep_extra_fields: list[str] | None = None,
    ) -> _FakePreparedDataset:
        prepared_calls.append(
            {
                "name": convo_ds.name,
                "max_seq_len": max_seq_len,
                "num_proc": num_proc,
                "keep_extra_fields": keep_extra_fields,
            },
        )
        include_sample_data = bool(keep_extra_fields and "sample_data" in keep_extra_fields)
        subset_rows = prepared_rows_by_subset[convo_ds.name]
        rows = []
        for row in subset_rows:
            row_copy = dict(row)
            if not include_sample_data:
                row_copy.pop("sample_data", None)
            rows.append(row_copy)
        return _FakePreparedDataset(rows)

    captured_args: dict[str, dict[str, object]] = {}

    def fake_training_arguments(**kwargs: typing.Any) -> types.SimpleNamespace:
        captured_args["kwargs"] = kwargs
        return types.SimpleNamespace(**kwargs)

    fake_trainer_instance = _FakeTrainer(
        model=_FakeModel(),
        args=types.SimpleNamespace(),
        train_dataset=_FakePreparedDataset([]),
        eval_dataset=_FakePreparedDataset([]),
        processing_class=_FakeTokenizer(),
        data_collator=None,
    )

    def fake_trainer_factory(**kwargs: typing.Any) -> _FakeTrainer:
        fake_trainer_instance.model = kwargs["model"]
        fake_trainer_instance.args = kwargs["args"]
        fake_trainer_instance.train_dataset = kwargs["train_dataset"]
        fake_trainer_instance.eval_dataset = kwargs["eval_dataset"]
        fake_trainer_instance.data_collator = kwargs["data_collator"]
        fake_trainer_instance.tokenizer = kwargs["processing_class"]
        fake_trainer_instance.compute_metrics = kwargs.get("compute_metrics")
        fake_trainer_instance.callbacks = kwargs.get("callbacks", [])
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
        "PaddingCollatorWithPromptMask",
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
    fake_dm = _FakeDataModule()

    runtime = types.SimpleNamespace(wandb_run=object(), finalize=lambda: None)
    config = types.SimpleNamespace(
        training_args_config=_FakeTrainingArgsConfig(),
        datamodule_config=types.SimpleNamespace(
            train_subset_names=["train"],
            valid_subset_names=["valid"],
        ),
        gradient_checkpointing=True,
        use_wandb_logging=True,
        output_dir=str(tmp_path / "artifact"),
        get_model=lambda: _FakeModel(),
        get_tokenizer=lambda: _FakeTokenizer(),
    )

    trainer = pyine.apps.trainers.hf_trainer.train(
        datamodule=fake_dm,
        config=config,
        runtime=runtime,
        resume_artifacts=None,
    )
    assert trainer is fake_trainer_instance
    assert isinstance(trainer.model, _FakeModel)
    assert isinstance(trainer.tokenizer, _FakeTokenizer)

    assert prepared_calls == [
        {
            "name": "train",
            "max_seq_len": 64,
            "num_proc": 2,
            "keep_extra_fields": None,
        },
        {
            "name": "valid",
            "max_seq_len": 64,
            "num_proc": 2,
            "keep_extra_fields": ["sample_data"],
        },
    ]
    assert raw_datasets == ["train:True:False", "valid:True:True"]
    assert captured_args["kwargs"]["report_to"] == ["wandb"]
    assert captured_args["kwargs"]["batch_eval_metrics"] is True
    metrics_callbacks = [
        cb for cb in trainer.callbacks if isinstance(cb, pyine.evals.utils.CategoryWiseMetricsCallback)
    ]
    assert len(metrics_callbacks) == 1
    metrics_callback = metrics_callbacks[0]
    assert metrics_callback.data_sample_categories == [["bugfix"], [], ["refactor"]]
    assert metrics_callback.metrics_prefix == "eval"
    assert trainer.compute_metrics is metrics_callback
    assert "sample_data" not in trainer.train_dataset.column_names
    assert "sample_data" in trainer.eval_dataset.column_names
    assert trainer.eval_dataset["sample_data"] == [
        {"code_type": "bugfix"},
        {"code_type": None},
        {"code_type": "refactor"},
    ]
    assert trainer.trained


@pytest.mark.asyncio
async def test_main_runs_train_and_evaluate(
    monkeypatch: pytest.MonkeyPatch,
    mocker: pytest_mock.MockerFixture,
) -> None:
    train_calls: list[dict] = []
    eval_calls: list[dict] = []

    async def fake_evaluate_model(
        **kwargs: object,
    ) -> None:
        eval_calls.append(kwargs)

    def fake_train(
        **kwargs: object,
    ) -> types.SimpleNamespace:
        train_calls.append(kwargs)
        return types.SimpleNamespace(
            model="model",
            processing_class="tokenizer",
        )

    def fake_prepare_datamodule(
        config: object,
        runtime: object,
    ) -> str:
        return mocker.Mock(spec=pyine.data.datamodule.ConversationDataModule)

    def fake_entrypoint_setup(
        **_kwargs: object,
    ) -> None:
        return None

    config = types.SimpleNamespace(
        training_args_config=types.SimpleNamespace(do_train=True, do_predict=True),
        get_model=lambda: "model",
        get_tokenizer=lambda: "tokenizer",
        use_wandb_logging=False,
        resume_from_run_dir=None,
        is_resuming=lambda: False,
    )
    runtime = types.SimpleNamespace(wandb_run=None, finalize=lambda: None)

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
        resume_from_run_dir=None,
        is_resuming=lambda: False,
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
