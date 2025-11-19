import pathlib
import typing

import pytest
import torch
import transformers

import pyine.utils.transformers


class _StageRecorderCollator:
    def __init__(self) -> None:
        self.stages: list[str] = []

    def set_stage(
        self,
        stage: str,
    ) -> None:
        self.stages.append(stage)

    def __call__(
        self,
        features: list[dict[str, int]],
    ) -> list[dict[str, int]]:
        return features


class _DummyDataset(torch.utils.data.Dataset):
    def __len__(
        self,
    ) -> int:
        return 1

    def __getitem__(
        self,
        idx: int,
    ) -> dict[str, list[int]]:
        del idx
        return {"input_ids": [1], "labels": [1]}


class _DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, **kwargs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.linear(kwargs["input_ids"].float())


def test_pyine_trainer_sets_collator_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    collator = _StageRecorderCollator()
    dataset = _DummyDataset()
    model = _DummyModel()
    args = transformers.TrainingArguments(output_dir=str(tmp_path / "pyine"), per_device_train_batch_size=1)
    sentinels = {
        "train": object(),
        "eval": object(),
        "predict": object(),
    }

    def _fake_get_train_dataloader(self: typing.Any) -> typing.Any:
        return sentinels["train"]

    def _fake_get_eval_dataloader(self: typing.Any, eval_dataset: typing.Any = None) -> typing.Any:
        assert eval_dataset is None
        return sentinels["eval"]

    def _fake_get_test_dataloader(self: typing.Any, test_dataset: typing.Any = None) -> typing.Any:
        assert test_dataset is None
        return sentinels["predict"]

    monkeypatch.setattr(transformers.Trainer, "get_train_dataloader", _fake_get_train_dataloader, raising=False)
    monkeypatch.setattr(transformers.Trainer, "get_eval_dataloader", _fake_get_eval_dataloader, raising=False)
    monkeypatch.setattr(transformers.Trainer, "get_test_dataloader", _fake_get_test_dataloader, raising=False)
    trainer = pyine.utils.transformers.TrainerWrapper(
        model=model,
        args=args,
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=collator,
    )
    assert trainer.get_train_dataloader() is sentinels["train"]
    assert trainer.get_eval_dataloader() is sentinels["eval"]
    assert trainer.get_test_dataloader(None) is sentinels["predict"]
    assert collator.stages == ["train", "eval", "predict"]
