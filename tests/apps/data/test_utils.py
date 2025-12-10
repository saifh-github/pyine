"""Unit tests for pyine.apps.data.utils module."""

from __future__ import annotations

import types

import pyine.apps.data.utils


class _FakeParser:
    def __init__(
        self,
        length: int | None,
        has_set_epoch: bool = True,
        raise_on_set_epoch: bool = False,
    ) -> None:
        self._length = length
        self._has_set_epoch = has_set_epoch
        self._raise_on_set_epoch = raise_on_set_epoch
        self.set_epoch_calls: list[int] = []

    def __len__(self) -> int:
        if self._length is None:
            raise TypeError("object of type '_FakeParser' has no len()")
        return self._length

    def set_epoch(self, epoch: int) -> None:
        if not self._has_set_epoch:
            raise AttributeError("set_epoch")
        if self._raise_on_set_epoch:
            raise RuntimeError("simulated set_epoch failure")
        self.set_epoch_calls.append(epoch)


class _FakeDataModule:
    def __init__(
        self,
        parser_len: int | None,
        has_set_epoch: bool = True,
        raise_on_set_epoch: bool = False,
        raise_on_get_parser: bool = False,
    ) -> None:
        self._parser = _FakeParser(
            length=parser_len,
            has_set_epoch=has_set_epoch,
            raise_on_set_epoch=raise_on_set_epoch,
        )
        self._raise_on_get_parser = raise_on_get_parser

    def get_parser(self, subset_name: str) -> _FakeParser:
        if self._raise_on_get_parser:
            raise RuntimeError("simulated get_parser failure")
        return self._parser


class TestInferPrecacheEpochCount:
    """Tests for infer_precache_epoch_count function."""

    def test_returns_epochs_override_when_provided(self) -> None:
        config = types.SimpleNamespace()
        datamodule = _FakeDataModule(parser_len=100)
        result = pyine.apps.data.utils.infer_precache_epoch_count(
            config=config,
            datamodule=datamodule,
            epochs_override=5,
        )
        assert result == 5

    def test_returns_one_when_training_args_missing(self) -> None:
        config = types.SimpleNamespace()  # no training_args_config
        datamodule = _FakeDataModule(parser_len=100)
        result = pyine.apps.data.utils.infer_precache_epoch_count(
            config=config,
            datamodule=datamodule,
            epochs_override=None,
        )
        assert result == 1

    def test_calculates_epochs_from_num_train_epochs(self) -> None:
        training_args = types.SimpleNamespace(
            num_train_epochs=3.0,
            gradient_accumulation_steps=1,
            per_device_train_batch_size=10,
            dataloader_drop_last=False,
            max_steps=-1,
        )
        config = types.SimpleNamespace(
            training_args_config=training_args,
            datamodule_config=types.SimpleNamespace(train_subset_names=["train"]),
        )
        datamodule = _FakeDataModule(parser_len=100)
        result = pyine.apps.data.utils.infer_precache_epoch_count(
            config=config,
            datamodule=datamodule,
            epochs_override=None,
        )
        assert result == 3

    def test_handles_fractional_num_train_epochs(self) -> None:
        training_args = types.SimpleNamespace(
            num_train_epochs=2.5,
            gradient_accumulation_steps=1,
            per_device_train_batch_size=10,
            dataloader_drop_last=False,
            max_steps=-1,
        )
        config = types.SimpleNamespace(
            training_args_config=training_args,
            datamodule_config=types.SimpleNamespace(train_subset_names=["train"]),
        )
        datamodule = _FakeDataModule(parser_len=100)
        result = pyine.apps.data.utils.infer_precache_epoch_count(
            config=config,
            datamodule=datamodule,
            epochs_override=None,
        )
        assert result == 3  # ceil(2.5) = 3

    def test_clips_epochs_based_on_max_steps(self) -> None:
        training_args = types.SimpleNamespace(
            num_train_epochs=10.0,
            gradient_accumulation_steps=1,
            per_device_train_batch_size=10,
            dataloader_drop_last=False,
            max_steps=15,  # 10 batches per epoch, so 15 steps = 2 epochs
        )
        config = types.SimpleNamespace(
            training_args_config=training_args,
            datamodule_config=types.SimpleNamespace(train_subset_names=["train"]),
        )
        datamodule = _FakeDataModule(parser_len=100)
        result = pyine.apps.data.utils.infer_precache_epoch_count(
            config=config,
            datamodule=datamodule,
            epochs_override=None,
        )
        assert result == 2  # min(10, ceil(15/10)) = 2

    def test_handles_gradient_accumulation(self) -> None:
        training_args = types.SimpleNamespace(
            num_train_epochs=10.0,
            gradient_accumulation_steps=4,
            per_device_train_batch_size=10,
            dataloader_drop_last=False,
            max_steps=5,  # 10 batches per epoch / 4 grad_accum = 3 updates per epoch
        )
        config = types.SimpleNamespace(
            training_args_config=training_args,
            datamodule_config=types.SimpleNamespace(train_subset_names=["train"]),
        )
        datamodule = _FakeDataModule(parser_len=100)
        result = pyine.apps.data.utils.infer_precache_epoch_count(
            config=config,
            datamodule=datamodule,
            epochs_override=None,
        )
        assert result == 2  # ceil(5/3) = 2

    def test_returns_epoch_cap_when_dataset_is_iterable(self) -> None:
        training_args = types.SimpleNamespace(
            num_train_epochs=3.0,
            gradient_accumulation_steps=1,
            per_device_train_batch_size=10,
            dataloader_drop_last=False,
            max_steps=-1,
        )
        config = types.SimpleNamespace(
            training_args_config=training_args,
            datamodule_config=types.SimpleNamespace(train_subset_names=["train"]),
        )
        datamodule = _FakeDataModule(parser_len=None)  # iterable dataset
        result = pyine.apps.data.utils.infer_precache_epoch_count(
            config=config,
            datamodule=datamodule,
            epochs_override=None,
        )
        assert result == 3  # falls back to epoch_cap

    def test_handles_drop_last(self) -> None:
        training_args = types.SimpleNamespace(
            num_train_epochs=10.0,
            gradient_accumulation_steps=1,
            per_device_train_batch_size=30,
            dataloader_drop_last=True,  # 100 samples / 30 batch = 3 batches (drop incomplete)
            max_steps=10,
        )
        config = types.SimpleNamespace(
            training_args_config=training_args,
            datamodule_config=types.SimpleNamespace(train_subset_names=["train"]),
        )
        datamodule = _FakeDataModule(parser_len=100)
        result = pyine.apps.data.utils.infer_precache_epoch_count(
            config=config,
            datamodule=datamodule,
            epochs_override=None,
        )
        assert result == 4  # ceil(10/3) = 4


class TestGetTotalTrainExampleCount:
    """Tests for get_total_train_example_count function."""

    def test_returns_length_for_single_subset(self) -> None:
        config = types.SimpleNamespace(
            datamodule_config=types.SimpleNamespace(train_subset_names=["train"]),
        )
        datamodule = _FakeDataModule(parser_len=150)
        result = pyine.apps.data.utils.get_total_train_example_count(
            config=config,
            datamodule=datamodule,
        )
        assert result == 150

    def test_sums_multiple_subsets(self) -> None:
        config = types.SimpleNamespace(
            datamodule_config=types.SimpleNamespace(train_subset_names=["train1", "train2"]),
        )
        datamodule = _FakeDataModule(parser_len=100)  # each subset has 100
        result = pyine.apps.data.utils.get_total_train_example_count(
            config=config,
            datamodule=datamodule,
        )
        assert result == 200

    def test_returns_none_for_iterable_dataset(self) -> None:
        config = types.SimpleNamespace(
            datamodule_config=types.SimpleNamespace(train_subset_names=["train"]),
        )
        datamodule = _FakeDataModule(parser_len=None)  # no __len__
        result = pyine.apps.data.utils.get_total_train_example_count(
            config=config,
            datamodule=datamodule,
        )
        assert result is None

    def test_returns_none_when_parser_fetch_fails(self) -> None:
        config = types.SimpleNamespace(
            datamodule_config=types.SimpleNamespace(train_subset_names=["train"]),
        )
        datamodule = _FakeDataModule(parser_len=100, raise_on_get_parser=True)
        result = pyine.apps.data.utils.get_total_train_example_count(
            config=config,
            datamodule=datamodule,
        )
        assert result is None

    def test_defaults_to_train_subset_when_config_missing(self) -> None:
        config = types.SimpleNamespace()  # no datamodule_config
        datamodule = _FakeDataModule(parser_len=50)
        result = pyine.apps.data.utils.get_total_train_example_count(
            config=config,
            datamodule=datamodule,
        )
        assert result == 50


class TestPrepareSubsetEpoch:
    """Tests for prepare_subset_epoch function."""

    def test_calls_set_epoch_on_parser(self) -> None:
        datamodule = _FakeDataModule(parser_len=100)
        pyine.apps.data.utils.prepare_subset_epoch(
            datamodule=datamodule,
            subset_names=["train"],
            epoch=5,
        )
        assert datamodule._parser.set_epoch_calls == [5]

    def test_handles_parser_without_set_epoch(self) -> None:
        datamodule = _FakeDataModule(parser_len=100, has_set_epoch=False)
        # should not raise
        pyine.apps.data.utils.prepare_subset_epoch(
            datamodule=datamodule,
            subset_names=["train"],
            epoch=3,
        )

    def test_handles_set_epoch_raising_exception(self) -> None:
        datamodule = _FakeDataModule(parser_len=100, raise_on_set_epoch=True)
        # should not raise, just logs warning
        pyine.apps.data.utils.prepare_subset_epoch(
            datamodule=datamodule,
            subset_names=["train"],
            epoch=2,
        )

    def test_handles_get_parser_raising_exception(self) -> None:
        datamodule = _FakeDataModule(parser_len=100, raise_on_get_parser=True)
        # should not raise, just logs debug
        pyine.apps.data.utils.prepare_subset_epoch(
            datamodule=datamodule,
            subset_names=["train"],
            epoch=1,
        )

    def test_calls_set_epoch_for_multiple_subsets(self) -> None:
        datamodule = _FakeDataModule(parser_len=100)
        pyine.apps.data.utils.prepare_subset_epoch(
            datamodule=datamodule,
            subset_names=["train1", "train2"],
            epoch=4,
        )
        # called twice, once for each subset
        assert datamodule._parser.set_epoch_calls == [4, 4]
