from __future__ import annotations

import typing

import transformers

if typing.TYPE_CHECKING:
    import torch.utils.data

    import pyine.utils.transformers.collate

__all__ = ["TrainerWrapper"]


def _maybe_set_collator_stage(
    data_collator: typing.Any,
    stage: pyine.utils.transformers.collate.CollatorStage,
) -> None:
    setter = getattr(data_collator, "set_stage", None)
    if callable(setter):
        setter(stage)


class TrainerWrapper(transformers.Trainer):
    """Trainer subclass that keeps the data collator informed about the active stage."""

    @typing.override
    def get_train_dataloader(
        self,
    ) -> torch.utils.data.DataLoader[typing.Any]:
        _maybe_set_collator_stage(self.data_collator, "train")
        return super().get_train_dataloader()  # type: ignore[reportUnknownMemberType]

    @typing.override
    def get_eval_dataloader(
        self,
        eval_dataset: str | torch.utils.data.Dataset | None = None,  # type: ignore[reportUnknownParameterType]
    ) -> torch.utils.data.DataLoader[typing.Any]:
        _maybe_set_collator_stage(self.data_collator, "eval")
        return super().get_eval_dataloader(eval_dataset=eval_dataset)  # type: ignore[reportUnknownMemberType]

    @typing.override
    def get_test_dataloader(
        self,
        test_dataset: torch.utils.data.Dataset,  # type: ignore[reportUnknownParameterType]
    ) -> torch.utils.data.DataLoader[typing.Any]:
        _maybe_set_collator_stage(self.data_collator, "predict")
        return super().get_test_dataloader(test_dataset=test_dataset)  # type: ignore[reportUnknownMemberType]
