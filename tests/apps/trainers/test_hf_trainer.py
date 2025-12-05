import json
import pathlib
import types
import typing

import datasets
import pytest
import pytest_mock
import torch
import transformers

import pyine.apps.trainers.hf_trainer
import pyine.data.datamodule
import pyine.evals.utils
import pyine.utils.transformers


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


def _install_collator_stub(
    config: types.SimpleNamespace,
    *,
    collator_factory: typing.Callable[[typing.Any, int, typing.Any | None], typing.Any] | None = None,
) -> list[dict[str, typing.Any]]:
    """Attach a get_collator method to a simple config and capture invocation metadata."""

    collator_calls: list[dict[str, typing.Any]] = []

    def _get_collator(
        tokenizer: typing.Any,
        max_seq_len: int,
        wandb_run: typing.Any | None = None,
    ) -> typing.Any:
        collator_calls.append(
            {
                "tokenizer": tokenizer,
                "max_seq_len": max_seq_len,
                "wandb_run": wandb_run,
            }
        )
        if collator_factory is None:
            return "collator"
        return collator_factory(tokenizer, max_seq_len, wandb_run)

    config.get_collator = _get_collator
    return collator_calls


def test_train_configures_trainer_and_saves_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    raw_datasets: list[str] = []
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

    class _FakeDataModule:
        def __init__(self) -> None:
            self.tokenized_calls: list[dict[str, typing.Any]] = []

        def get_hf_messages_dataset(
            self,
            subset_name: str,
            append_answer: bool,
            keep_original_data: bool = False,
        ) -> datasets.Dataset:
            raw_datasets.append(f"{subset_name}:{append_answer}:{keep_original_data}")
            include_sample_data = keep_original_data
            rows = []
            for row in prepared_rows_by_subset[subset_name]:
                row_copy = dict(row)
                if not include_sample_data:
                    row_copy.pop("sample_data", None)
                rows.append(row_copy)
            return datasets.Dataset.from_list(rows)

        def get_hf_tokenized_examples_dataset(
            self,
            subset_name: str,
            tokenizer: typing.Any,
            model_max_seq_len: int,
            keep_extra_fields: list[str] | bool | None = None,
            force_regenerate: bool = False,
            epoch: int | None = None,
        ) -> _FakePreparedDataset:
            include_sample_data = subset_name != "train"
            if isinstance(keep_extra_fields, (list, tuple, set)):
                include_sample_data = "sample_data" in keep_extra_fields
            elif isinstance(keep_extra_fields, bool):
                include_sample_data = keep_extra_fields
            self.tokenized_calls.append(
                {
                    "subset_name": subset_name,
                    "tokenizer": tokenizer,
                    "max_seq_len": model_max_seq_len,
                    "keep_extra_fields": keep_extra_fields,
                    "include_sample_data": include_sample_data,
                    "force_regenerate": force_regenerate,
                    "epoch": epoch,
                },
            )
            subset_rows = prepared_rows_by_subset[subset_name]
            rows = []
            for row in subset_rows:
                row_copy = dict(row)
                if not include_sample_data:
                    row_copy.pop("sample_data", None)
                rows.append(row_copy)
            return _FakePreparedDataset(rows)

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

    def _bound_get_tokenized_examples(
        self: _FakeDataModule,
        subset_name: str,
        tokenizer: typing.Any,
        model_max_seq_len: int,
        keep_extra_fields: list[str] | bool | None = None,
        epoch: int | None = None,
    ) -> _FakePreparedDataset:
        return _FakeDataModule.get_hf_tokenized_examples_dataset(
            self,
            subset_name=subset_name,
            tokenizer=tokenizer,
            model_max_seq_len=model_max_seq_len,
            keep_extra_fields=keep_extra_fields,
            epoch=epoch,
        )

    fake_dm.get_hf_tokenized_examples_dataset = types.MethodType(  # type: ignore[attr-defined]
        _bound_get_tokenized_examples,
        fake_dm,
    )

    cache_root = tmp_path / "tokenized_cache"
    cache_root.mkdir()

    runtime = types.SimpleNamespace(
        wandb_run=types.SimpleNamespace(
            define_metric=lambda *args, **kwargs: None,
        ),
        finalize=lambda: None,
    )

    class _FakeDatamoduleConfig:
        def __init__(self, cache_dir: pathlib.Path) -> None:
            self.train_subset_names = ["train"]
            self.valid_subset_names = ["valid"]
            self.datamodule_name = "fake_dm"
            self.use_tokenized_dataset_cache = False
            self.dataset_cache_lock_timeout_seconds = 0.0
            self.keep_generated_datasets_in_memory = False
            self._cache_dir = cache_dir

        def get_tokenized_dataset_cache_root(self) -> pathlib.Path:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            return self._cache_dir

    config = types.SimpleNamespace(
        training_args_config=_FakeTrainingArgsConfig(),
        datamodule_config=_FakeDatamoduleConfig(cache_dir=cache_root),
        evals_config=types.SimpleNamespace(category_extraction_config=None),
        gradient_checkpointing=True,
        use_wandb_logging=True,
        collator_batch_logging=False,
        output_dir=str(tmp_path / "artifact"),
        get_model=lambda: _FakeModel(),
        get_tokenizer=lambda: _FakeTokenizer(),
    )
    collator_calls = _install_collator_stub(config)

    trainer = pyine.apps.trainers.hf_trainer.train(
        datamodule=fake_dm,
        config=config,
        runtime=runtime,
        resume_artifacts=None,
    )
    assert trainer is fake_trainer_instance
    assert isinstance(trainer.model, _FakeModel)
    assert isinstance(trainer.tokenizer, _FakeTokenizer)

    assert fake_dm.tokenized_calls == [
        {
            "subset_name": "train",
            "tokenizer": trainer.tokenizer,
            "max_seq_len": 64,
            "keep_extra_fields": None,
            "include_sample_data": False,
            "force_regenerate": False,
            "epoch": None,
        },
        {
            "subset_name": "valid",
            "tokenizer": trainer.tokenizer,
            "max_seq_len": 64,
            "keep_extra_fields": None,
            "include_sample_data": True,
            "force_regenerate": False,
            "epoch": None,
        },
    ]
    assert raw_datasets == []
    assert captured_args["kwargs"]["report_to"] == ["wandb"]
    assert captured_args["kwargs"]["batch_eval_metrics"] is True
    metrics_callbacks = [
        cb for cb in trainer.callbacks if isinstance(cb, pyine.evals.utils.CategoryWiseMetricsCallback)
    ]
    assert len(metrics_callbacks) == 1
    metrics_callback = metrics_callbacks[0]
    assert metrics_callback.data_sample_categories == [["code_type/bugfix"], [], ["code_type/refactor"]]
    assert trainer.compute_metrics is metrics_callback
    assert "sample_data" not in trainer.train_dataset.column_names
    assert "sample_data" in trainer.eval_dataset.column_names
    assert trainer.eval_dataset["sample_data"] == [
        {"code_type": "bugfix"},
        {"code_type": None},
        {"code_type": "refactor"},
    ]
    assert trainer.trained
    assert collator_calls and collator_calls[0]["wandb_run"] is runtime.wandb_run


def test_train_adds_epoch_callback_for_epoch_aware_datasets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_dataset = _FakePreparedDataset(
        [
            {
                "input_ids": [1, 2],
                "attention_mask": [1, 1],
                "labels": [1, 2],
            }
        ]
    )
    valid_dataset = _FakePreparedDataset(
        [
            {
                "input_ids": [10, 11],
                "attention_mask": [1, 1],
                "labels": [10, 11],
                "sample_data": {"code_type": "bugfix"},
            }
        ]
    )

    class _FakeDataModule:
        def get_hf_tokenized_examples_dataset(
            self,
            subset_name: str,
            tokenizer: typing.Any,
            model_max_seq_len: int,
            epoch: int | None = None,
            **_: typing.Any,
        ) -> _FakePreparedDataset:
            del tokenizer
            del model_max_seq_len
            del epoch
            if subset_name == "train":
                return train_dataset
            assert subset_name == "valid"
            return valid_dataset

    class _FakeTrainingArgsConfig:
        def __init__(self) -> None:
            self.do_train = True

        def model_dump(self) -> dict[str, typing.Any]:
            return {
                "output_dir": "ignored",
                "per_device_train_batch_size": 2,
            }

    class _FakeModel:
        def __init__(self) -> None:
            self.config = types.SimpleNamespace(name_or_path="fake")

    class _FakeTokenizer:
        def __init__(self) -> None:
            self.padding_side = "right"

    config = types.SimpleNamespace(
        training_args_config=_FakeTrainingArgsConfig(),
        get_model=lambda: _FakeModel(),
        get_tokenizer=lambda: _FakeTokenizer(),
        datamodule_config=types.SimpleNamespace(
            train_subset_names=["train"],
            valid_subset_names=["valid"],
            eval_subset_names=[],
        ),
        evals_config=types.SimpleNamespace(category_extraction_config=None),
        use_wandb_logging=False,
    )
    _install_collator_stub(config)
    runtime = types.SimpleNamespace(wandb_run=None)
    epoch_callback_calls: list[dict[str, typing.Any]] = []
    sentinel_callback = object()

    def _fake_create_epoch_cb(
        *,
        train_dataset: typing.Any,
        datamodule: typing.Any,
        subset_names: typing.Iterable[str],
    ) -> typing.Any:
        epoch_callback_calls.append(
            {
                "train_dataset": train_dataset,
                "datamodule": datamodule,
                "subset_names": tuple(subset_names),
            }
        )
        return sentinel_callback

    monkeypatch.setattr(
        pyine.apps.trainers.hf_trainer.pyine.utils.transformers,
        "create_epoch_awareness_callback",
        _fake_create_epoch_cb,
    )
    monkeypatch.setattr(
        pyine.apps.trainers.hf_trainer.pyine.utils.transformers,
        "infer_effective_max_seq_len",
        lambda *_args, **_kwargs: 16,
    )
    monkeypatch.setattr(
        pyine.apps.trainers.hf_trainer.transformers,
        "TrainingArguments",
        lambda **kwargs: types.SimpleNamespace(**kwargs),
    )

    class _FakeTrainer:
        def __init__(self, **kwargs: typing.Any) -> None:
            self.kwargs = kwargs
            self.model = kwargs["model"]
            self.processing_class = kwargs["processing_class"]
            self.callbacks = kwargs.get("callbacks", [])
            self.trained = False

        def train(self, **_: typing.Any) -> str:
            self.trained = True
            return "done"

    monkeypatch.setattr(
        pyine.apps.trainers.hf_trainer.transformers,
        "Trainer",
        lambda **kwargs: _FakeTrainer(**kwargs),
    )
    datamodule = _FakeDataModule()
    trainer = pyine.apps.trainers.hf_trainer.train(
        datamodule=datamodule,
        config=config,
        runtime=runtime,
        resume_artifacts=None,
    )
    assert trainer.trained
    assert epoch_callback_calls
    assert epoch_callback_calls[0]["train_dataset"] is train_dataset
    assert epoch_callback_calls[0]["datamodule"] is datamodule
    assert epoch_callback_calls[0]["subset_names"] == ("train",)
    assert trainer.callbacks[-1] is sentinel_callback


def test_train_enables_wandb_batch_logging(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    class _FakeDataModule:
        def get_hf_tokenized_examples_dataset(
            self,
            subset_name: str,
            tokenizer: typing.Any,
            model_max_seq_len: int,
            epoch: int | None = None,
            **_: typing.Any,
        ) -> _FakePreparedDataset:
            del subset_name
            del tokenizer
            del model_max_seq_len
            del epoch
            rows = [
                {
                    "input_ids": [1, 2],
                    "attention_mask": [1, 1],
                    "labels": [1, 2],
                    "sample_data": {"code_type": "bugfix"},
                },
            ]
            return _FakePreparedDataset(rows)

    class _FakeModel:
        def __init__(self) -> None:
            self.config = types.SimpleNamespace(use_cache=True)
            self.gradient_checkpointing_enabled = False

        def gradient_checkpointing_enable(self) -> None:
            self.gradient_checkpointing_enabled = True

    class _FakeTokenizer:
        pass

    class _FakeTrainingArgsConfig:
        def __init__(self) -> None:
            self.do_train = True

        def model_dump(self) -> dict[str, object]:
            return {"output_dir": "ignored"}

    class _FakeWandbRun:
        def __init__(self) -> None:
            self.logged: list[tuple[dict[str, typing.Any], bool]] = []

        def define_metric(self, *args: object, **kwargs: object) -> None:
            del args
            del kwargs

        def log(
            self,
            data: dict[str, typing.Any],
            commit: bool = True,
        ) -> None:
            self.logged.append((data, commit))

    class _FakeTrainer:
        def __init__(self, **kwargs: typing.Any) -> None:
            self.kwargs = kwargs
            self.trained = False

        def train(self, **kwargs: typing.Any) -> str:
            self.train_kwargs = kwargs
            self.trained = True
            return "done"

    fake_trainer_instance = _FakeTrainer()

    def fake_trainer_factory(**kwargs: typing.Any) -> _FakeTrainer:
        fake_trainer_instance.kwargs = kwargs
        return fake_trainer_instance

    monkeypatch.setattr(
        pyine.apps.trainers.hf_trainer.pyine.utils.transformers,
        "infer_effective_max_seq_len",
        lambda *_args, **_kwargs: 32,
    )
    monkeypatch.setattr(
        pyine.apps.trainers.hf_trainer.transformers,
        "TrainingArguments",
        lambda **kwargs: types.SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        pyine.apps.trainers.hf_trainer.transformers,
        "Trainer",
        fake_trainer_factory,
    )

    runtime = types.SimpleNamespace(wandb_run=_FakeWandbRun(), finalize=lambda: None)
    config = types.SimpleNamespace(
        training_args_config=_FakeTrainingArgsConfig(),
        datamodule_config=types.SimpleNamespace(),
        evals_config=types.SimpleNamespace(category_extraction_config=None),
        gradient_checkpointing=False,
        use_wandb_logging=True,
        collator_batch_logging=True,
        output_dir=str(tmp_path / "artifact"),
        get_model=lambda: _FakeModel(),
        get_tokenizer=lambda: _FakeTokenizer(),
    )

    captured_handlers: list[typing.Callable[[pyine.utils.transformers.CollatorBatchLogRecord], None]] = []

    def _collator_factory(
        tokenizer: typing.Any,
        max_seq_len: int,
        wandb_run: typing.Any | None = None,
    ) -> typing.Any:
        del tokenizer
        del max_seq_len
        assert wandb_run is runtime.wandb_run

        def _handler(record: pyine.utils.transformers.CollatorBatchLogRecord) -> None:
            wandb_run.log(
                {
                    f"collator/{record['stage']}/batch_size": record["batch_size"],
                    f"collator/{record['stage']}/padded_seq_len": record["padded_seq_len"],
                    f"collator/{record['stage']}/padding_ratio": record["padding_ratio"],
                    f"collator/{record['stage']}/non_ignored_label_ratio": record["non_ignored_label_ratio"],
                },
                commit=False,
            )

        captured_handlers.append(_handler)
        return types.SimpleNamespace(batch_logger=_handler)

    collator_calls = _install_collator_stub(config, collator_factory=_collator_factory)

    trainer = pyine.apps.trainers.hf_trainer.train(
        datamodule=_FakeDataModule(),
        config=config,
        runtime=runtime,
        resume_artifacts=None,
    )
    assert trainer is fake_trainer_instance
    assert trainer.trained
    assert collator_calls and collator_calls[0]["wandb_run"] is runtime.wandb_run
    assert captured_handlers

    handler = captured_handlers[0]
    handler(
        {
            "stage": "train",
            "batch_size": 4,
            "padded_seq_len": 128,
            "padding_ratio": 0.2,
            "non_ignored_label_ratio": 0.1,
        }
    )
    assert runtime.wandb_run.logged == [
        (
            {
                "collator/train/batch_size": 4,
                "collator/train/padded_seq_len": 128,
                "collator/train/padding_ratio": 0.2,
                "collator/train/non_ignored_label_ratio": 0.1,
            },
            False,
        )
    ]


@pytest.mark.asyncio
async def test_main_loads_checkpoint_when_resume_artifacts_only(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: pytest_mock.MockerFixture,
) -> None:
    checkpoint_path = tmp_path / "checkpoint-100"
    resume_artifacts = types.SimpleNamespace(
        checkpoint_path=checkpoint_path,
        wandb_resume_kwargs={},
    )

    def fake_prepare_resume_artifacts(
        **_kwargs: object,
    ) -> types.SimpleNamespace:
        return resume_artifacts

    def fake_prepare_datamodule(
        *_args: object,
        **_kwargs: object,
    ) -> typing.Any:
        return mocker.Mock(spec=pyine.data.datamodule.ConversationDataModule)

    auto_model_calls: list[tuple[typing.Any, typing.Any]] = []
    auto_tokenizer_calls: list[tuple[typing.Any, typing.Any]] = []

    def fake_auto_model_from_pretrained(
        path: str | pathlib.Path,
        **kwargs: object,
    ) -> str:
        auto_model_calls.append((path, kwargs))
        return "model"

    def fake_auto_tokenizer_from_pretrained(
        path: str | pathlib.Path,
        **kwargs: object,
    ) -> str:
        auto_tokenizer_calls.append((path, kwargs))
        return "tokenizer"

    config = types.SimpleNamespace(
        training_args_config=types.SimpleNamespace(do_train=False, do_predict=False),
        get_model=lambda: "base_model",
        get_tokenizer=lambda: "base_tokenizer",
        use_wandb_logging=False,
        resume_from_run_dir=None,
        is_resuming=lambda: False,
    )
    runtime = types.SimpleNamespace(
        wandb_run=None,
        finalize=lambda: None,
    )

    monkeypatch.setattr(
        pyine.apps.trainers.hf_trainer.pyine.apps.trainers.common,
        "prepare_resume_artifacts",
        fake_prepare_resume_artifacts,
    )
    monkeypatch.setattr(
        pyine.apps.trainers.hf_trainer.pyine.apps.trainers.common,
        "prepare_datamodule",
        fake_prepare_datamodule,
    )
    monkeypatch.setattr(
        pyine.apps.trainers.hf_trainer.transformers,
        "AutoModelForCausalLM",
        types.SimpleNamespace(from_pretrained=fake_auto_model_from_pretrained),
    )
    monkeypatch.setattr(
        pyine.apps.trainers.hf_trainer.transformers,
        "AutoTokenizer",
        types.SimpleNamespace(from_pretrained=fake_auto_tokenizer_from_pretrained),
    )
    monkeypatch.setattr(
        pyine.apps.trainers.hf_trainer.pyine.utils.reprod,
        "entrypoint_setup",
        lambda **_: None,
    )

    await pyine.apps.trainers.hf_trainer.main(
        config=config,
        runtime=runtime,
    )

    assert auto_model_calls == [(checkpoint_path, {})]
    assert auto_tokenizer_calls == [(checkpoint_path, {})]


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


@pytest.mark.slow
def test_train_resumes_from_checkpoint_with_real_trainer(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _SimpleTokenizer:
        def __init__(
            self,
        ) -> None:
            self.pad_token = "<pad>"  # noqa: S105 (not a password)
            self.eos_token = "<eos>"  # noqa: S105 (not a password)
            self.pad_token_id = 0
            self.eos_token_id = 1
            self.padding_side = "right"
            self.truncation_side = "left"
            self.model_max_length = 64
            self.is_fast = False
            self._vocab: dict[str, int] = {
                self.pad_token: self.pad_token_id,
                self.eos_token: self.eos_token_id,
            }

        def _get_token_id(
            self,
            token: str,
        ) -> int:
            if token not in self._vocab:
                self._vocab[token] = len(self._vocab)
            return self._vocab[token]

        def apply_chat_template(
            self,
            conversation: list[list[dict[str, typing.Any]]] | list[dict[str, typing.Any]],
            tokenize: bool = False,
            add_generation_prompt: bool = False,
            **_: typing.Any,
        ) -> list[str]:
            assert tokenize is False
            if conversation and isinstance(conversation[0], list):
                conversations = typing.cast("list[list[dict[str, typing.Any]]]", conversation)
            else:
                single_conversation = typing.cast("list[dict[str, typing.Any]]", conversation)
                conversations = [single_conversation]
            formatted: list[str] = []
            for messages in conversations:
                parts: list[str] = []
                for message in messages:
                    parts.append(f"{message['role']}:{message['content']}")
                if add_generation_prompt:
                    parts.append("assistant:")
                formatted.append(" | ".join(parts))
            return formatted

        def __call__(
            self,
            text: str,
            add_special_tokens: bool = False,
            **_: typing.Any,
        ) -> dict[str, list[int]]:
            del add_special_tokens
            token_ids = [self._get_token_id(character) for character in text]
            attention_mask = [1] * len(token_ids)
            return {
                "input_ids": token_ids,
                "attention_mask": attention_mask,
            }

        def save_pretrained(
            self,
            output_dir: str,
        ) -> None:
            output_path = pathlib.Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            config_payload = {
                "pad_token": self.pad_token,
                "eos_token": self.eos_token,
                "pad_token_id": self.pad_token_id,
                "eos_token_id": self.eos_token_id,
                "vocab": self._vocab,
            }
            (output_path / "tokenizer_config.json").write_text(json.dumps(config_payload, indent=2))

        def pad(
            self,
            encoded_inputs: dict[str, typing.Any],
            **_: typing.Any,
        ) -> dict[str, typing.Any]:
            return encoded_inputs

        def decode(
            self,
            token_ids: typing.Iterable[int],
            skip_special_tokens: bool = True,
        ) -> str:
            inverse_vocab = {value: key for key, value in self._vocab.items()}
            tokens: list[str] = []
            for token_id in token_ids:
                if skip_special_tokens and token_id in (self.pad_token_id, self.eos_token_id):
                    continue
                tokens.append(inverse_vocab.get(token_id, "?"))
            return "".join(tokens)

    class _TinyTrainingArgsConfig:
        def __init__(
            self,
            output_dir: pathlib.Path,
            max_steps: int,
        ) -> None:
            self.output_dir = output_dir
            self.max_steps = max_steps
            self.do_train = True
            self.dataloader_num_workers = 0

        def model_dump(
            self,
        ) -> dict[str, typing.Any]:
            return {
                "output_dir": str(self.output_dir),
                "per_device_train_batch_size": 1,
                "max_steps": self.max_steps,
                "num_train_epochs": 1,
                "save_steps": 1,
                "save_strategy": "steps",
                "logging_steps": 1,
                "learning_rate": 5e-4,
                "report_to": "none",
                "overwrite_output_dir": True,
                "use_cpu": True,
                "disable_tqdm": True,
            }

    class _TinyDataModule:
        def __init__(
            self,
            train_ds: datasets.Dataset,
            valid_ds: datasets.Dataset,
        ) -> None:
            self._train_ds = train_ds
            self._valid_ds = valid_ds

        def get_hf_messages_dataset(
            self,
            subset_name: str,
            append_answer: bool,
            keep_original_data: bool = False,
        ) -> datasets.Dataset:
            del append_answer
            del keep_original_data
            if subset_name == "train":
                return self._train_ds
            if subset_name == "valid":
                return self._valid_ds
            raise ValueError(f"unexpected subset name: {subset_name}")

        def get_hf_tokenized_examples_dataset(
            self,
            subset_name: str,
            tokenizer: transformers.PreTrainedTokenizer,
            model_max_seq_len: int,
            keep_extra_fields: list[str] | bool | None = None,
            force_regenerate: bool = False,
            epoch: int | None = None,
        ) -> datasets.Dataset:
            keep_original_data = subset_name != "train"
            if isinstance(keep_extra_fields, (list, tuple, set)):
                keep_original_data = "sample_data" in keep_extra_fields
            elif isinstance(keep_extra_fields, bool):
                keep_original_data = keep_extra_fields
            convo_ds = self.get_hf_messages_dataset(
                subset_name=subset_name,
                append_answer=True,
                keep_original_data=keep_original_data,
            )
            return pyine.utils.transformers.prepare_examples_from_conversations(
                convo_ds=convo_ds,
                tokenizer=tokenizer,
                max_seq_len=model_max_seq_len,
                num_proc=1,
                keep_extra_fields=keep_original_data,
                force_rebuild=force_regenerate,
            )

    def _build_tiny_model() -> transformers.GPT2LMHeadModel:
        config = transformers.GPT2Config(
            n_layer=1,
            n_head=1,
            n_embd=32,
            n_positions=64,
            n_ctx=64,
            vocab_size=512,
            pad_token_id=0,
            eos_token_id=1,
        )
        return transformers.GPT2LMHeadModel(config)

    def _make_config(
        max_steps: int,
        output_dir: pathlib.Path,
    ) -> types.SimpleNamespace:
        class _TinyDatamoduleConfig:
            def __init__(self, cache_dir: pathlib.Path) -> None:
                self.train_subset_names = ["train"]
                self.valid_subset_names = ["valid"]
                self.datamodule_name = "tiny_dm"
                self.use_tokenized_dataset_cache = False
                self.dataset_cache_lock_timeout_seconds = 0.0
                self.keep_generated_datasets_in_memory = False
                self._cache_dir = cache_dir

            def get_tokenized_dataset_cache_root(self) -> pathlib.Path:
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                return self._cache_dir

        def _build_collator(
            tokenizer: typing.Any,
            max_seq_len: int,
            wandb_run: typing.Any | None = None,
        ) -> typing.Any:
            del wandb_run
            return pyine.utils.transformers.PaddingCollatorWithPromptMask(
                tokenizer=tokenizer,
                max_length=max_seq_len,
            )

        config = types.SimpleNamespace(
            training_args_config=_TinyTrainingArgsConfig(output_dir=output_dir, max_steps=max_steps),
            datamodule_config=_TinyDatamoduleConfig(cache_dir=output_dir / "tokenized_cache"),
            evals_config=types.SimpleNamespace(category_extraction_config=None),
            gradient_checkpointing=False,
            use_wandb_logging=False,
            output_dir=str(output_dir),
            get_model=_build_tiny_model,
            get_tokenizer=_SimpleTokenizer,
        )
        _install_collator_stub(config, collator_factory=_build_collator)
        return config

    train_conversations = [
        {
            "messages": [
                {"role": "user", "content": "Hello there"},
                {"role": "assistant", "content": "Hi!"},
            ],
        },
        {
            "messages": [
                {"role": "user", "content": "How are you"},
                {"role": "assistant", "content": "Doing fine"},
            ],
        },
    ]
    valid_conversations = [
        {
            "messages": [
                {"role": "user", "content": "Ping"},
                {"role": "assistant", "content": "Pong"},
            ],
            "sample_data": {"code_type": "alpha"},
        },
        {
            "messages": [
                {"role": "user", "content": "Tell me a joke"},
                {"role": "assistant", "content": "No jokes today"},
            ],
            "sample_data": {"code_type": "beta"},
        },
    ]
    train_dataset = datasets.Dataset.from_list(train_conversations)
    valid_dataset = datasets.Dataset.from_list(valid_conversations)
    datamodule = _TinyDataModule(train_ds=train_dataset, valid_ds=valid_dataset)

    output_dir = tmp_path / "trainer_output"
    output_dir.mkdir()

    runtime = types.SimpleNamespace(wandb_run=None, finalize=lambda: None)

    recorded_steps: list[int] = []
    stop_after = {"value": None}

    class _RecordingCallback(transformers.TrainerCallback):
        def on_step_end(  # type: ignore[override]
            self,
            args: transformers.TrainingArguments,
            state: transformers.trainer_callback.TrainerState,
            control: transformers.trainer_callback.TrainerControl,
            **kwargs: typing.Any,
        ) -> transformers.trainer_callback.TrainerControl:
            del args
            del kwargs
            recorded_steps.append(state.global_step)
            target = stop_after["value"]
            if target is not None and state.global_step >= target:
                control.should_training_stop = True
            return control

    class _RecordingTrainer(transformers.Trainer):
        def __init__(
            self,
            *args: typing.Any,
            **kwargs: typing.Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            self.add_callback(_RecordingCallback())

    monkeypatch.setattr(
        pyine.apps.trainers.hf_trainer.transformers,
        "Trainer",
        _RecordingTrainer,
    )

    torch.manual_seed(0)
    config_first = _make_config(max_steps=2, output_dir=output_dir)
    stop_after["value"] = 1
    recorded_steps.clear()
    assert hasattr(datamodule, "get_hf_tokenized_examples_dataset")
    trainer_first = pyine.apps.trainers.hf_trainer.train(
        datamodule=datamodule,
        config=config_first,
        runtime=runtime,
        resume_artifacts=None,
    )
    assert trainer_first.state.global_step == 1
    assert recorded_steps == [1]
    checkpoint_dir = output_dir / "checkpoint-1"
    assert checkpoint_dir.is_dir()
    assert (checkpoint_dir / "trainer_state.json").is_file()

    config_second = _make_config(max_steps=2, output_dir=output_dir)
    stop_after["value"] = None
    recorded_steps.clear()
    resume_artifacts = types.SimpleNamespace(checkpoint_path=checkpoint_dir)
    trainer_second = pyine.apps.trainers.hf_trainer.train(
        datamodule=datamodule,
        config=config_second,
        runtime=runtime,
        resume_artifacts=resume_artifacts,
    )
    assert trainer_second.state.global_step == 2
    assert recorded_steps and recorded_steps[0] == 2
    assert (output_dir / "checkpoint-2").is_dir()
