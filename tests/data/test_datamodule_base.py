import typing

import pytest
import torch.utils.data as tud

import pyine.data.datamodule as datamodule
import pyine.utils.portability as port


class DummyDataset(tud.Dataset):
    def __init__(
        self,
        size: int,
        start: int = 0,
    ) -> None:
        self._size = int(size)
        self._start = int(start)

    def __len__(
        self,
    ) -> int:
        return self._size

    def __getitem__(
        self,
        idx: int,
    ) -> int:
        if idx < 0 or idx >= self._size:
            raise IndexError("index out of range")
        return self._start + idx


def build_default_config() -> datamodule.BaseDataModuleConfig:
    parser_cfg = datamodule.BaseDataParserConfig(
        class_path="tests.data.test_datamodule_base.DummyDataset",
        params={"size": 10, "start": 0},
    )
    loader_cfg = datamodule.BaseDataLoaderConfig(
        class_path="torch.utils.data.DataLoader",
        params=datamodule.BaseDataLoaderParamsConfig(),
    )
    cfg = datamodule.BaseDataModuleConfig(
        datamodule_class_path=port.get_fully_qualified_name(DummyDataModule),
        default_dataparser_config=parser_cfg,
        dataparser_config_overrides={
            "valid": {
                "size": 5,
                "start": 100,
            },
        },
        default_dataloader_config=loader_cfg,
        dataloader_config_overrides={
            "train": {
                "batch_size": 4,
                "shuffle": True,
            },
        },
    )
    return cfg


class TestBaseDataModuleConfig:

    def test_resolve_and_instantiate_parser_and_loader(self) -> None:
        cfg = build_default_config()

        # parsers respect overrides per subset
        train_parser = cfg.instantiate_parser("train")
        valid_parser = cfg.instantiate_parser("valid")
        assert isinstance(train_parser, DummyDataset)
        assert isinstance(valid_parser, DummyDataset)
        assert len(train_parser) == 10 and len(valid_parser) == 5
        assert train_parser[0] == 0 and valid_parser[0] == 100

        # loaders pick up per-subset params (batch size/shuffle)
        train_loader = cfg.instantiate_dataloader("train", dataset=train_parser)
        valid_loader = cfg.instantiate_dataloader("valid", dataset=valid_parser)
        assert isinstance(train_loader, tud.DataLoader)
        assert isinstance(valid_loader, tud.DataLoader)

        # verify batch sizes via first batch length
        first_train_batch = next(iter(train_loader))
        first_valid_batch = next(iter(valid_loader))
        assert len(first_train_batch) == 4  # override applied
        assert len(first_valid_batch) == 1  # default from BaseDataLoaderParams

        # invalid subset names raise
        with pytest.raises(ValueError):
            _ = cfg._resolve_dataparser_config("unknown")
        with pytest.raises(ValueError):
            _ = cfg._resolve_dataloader_config("unknown")

    def test_subset_names_customization(self) -> None:
        cfg = build_default_config().model_copy(
            update={"subset_names": tuple(["train", "valid"])},
        )
        # validator should have populated only requested subsets
        # attempting to resolve a non-declared subset should fail
        with pytest.raises(ValueError):
            _ = cfg._resolve_dataparser_config("test")


class DummyDataModule(datamodule.BaseDataModule):
    def __init__(
        self,
        config: datamodule.BaseDataModuleConfig,
    ) -> None:
        super().__init__(config)

    def _make_parser(
        self,
        subset_name: datamodule.SubsetNameType,
    ) -> tud.Dataset:
        return self.config.instantiate_parser(subset_name)

    def _make_loader(
        self,
        loader_name: datamodule.LoaderNameType,
    ) -> tud.DataLoader:
        # assumes loader names == subset names
        parser = self._make_parser(loader_name)
        return self.config.instantiate_dataloader(loader_name, dataset=parser)

    def train_dataloader(
        self,
    ) -> tud.DataLoader:
        return self._make_loader("train")

    def val_dataloader(
        self,
    ) -> tud.DataLoader:
        return self._make_loader("valid")

    def test_dataloader(
        self,
    ) -> tud.DataLoader:
        return self._make_loader("test")


class TestBaseDataModule:

    def test_dataloader_names_property(self) -> None:
        cfg = build_default_config()
        dm = DummyDataModule(cfg)
        assert dm.dataloader_names == cfg.subset_names
        assert dm.dataloader_names == cfg.loader_names

    def test_valid_dataloader_redirect(self) -> None:
        class OnlyVal(datamodule.BaseDataModule):
            def __init__(
                self,
                config: datamodule.BaseDataModuleConfig,
            ) -> None:
                super().__init__(config)

            def val_dataloader(
                self,
            ) -> str:
                return "VAL"

            def train_dataloader(self):  # type: ignore[no-untyped-def]
                raise NotImplementedError

            def test_dataloader(self):  # type: ignore[no-untyped-def]
                raise NotImplementedError

        dm = OnlyVal(build_default_config())
        assert dm.valid_dataloader() == "VAL"

    def test_get_dataloader_success_and_errors(self) -> None:
        class WithTrain(datamodule.BaseDataModule):
            def __init__(
                self,
                config: datamodule.BaseDataModuleConfig,
                payload: typing.Any,
            ) -> None:
                super().__init__(config)
                self._payload = payload

            def train_dataloader(
                self,
            ) -> typing.Any:
                return self._payload

            def test_dataloader(self):  # type: ignore[no-untyped-def]
                raise NotImplementedError

            def val_dataloader(self):  # type: ignore[no-untyped-def]
                raise NotImplementedError

        base_cfg = build_default_config()
        dm_ok = WithTrain(base_cfg, payload={"ok": True})
        assert dm_ok.get_dataloader("train") == {"ok": True}

        # invalid subset (not declared in config)
        with pytest.raises(ValueError) as exc_info:
            _ = dm_ok.get_dataloader("unknown")
        assert "invalid loader name" in str(exc_info.value)

        # declared subset but missing corresponding method -> specific error
        cfg_missing = base_cfg.model_copy(update={"subset_names": tuple(["train", "foo"])})
        dm_missing = WithTrain(cfg_missing, payload=None)
        with pytest.raises(ValueError) as exc_info:
            _ = dm_missing.get_dataloader("foo")
        assert "no such function: foo_dataloader" in str(exc_info.value)

        # declared subset with non-callable attribute
        class WithBadAttr(datamodule.BaseDataModule):
            def __init__(
                self,
                config: datamodule.BaseDataModuleConfig,
            ) -> None:
                super().__init__(config)
                # create an attribute that shadows the expected method name
                self.bar_dataloader = "not-callable"  # type: ignore[attr-defined]

            def train_dataloader(self):  # type: ignore[no-untyped-def]
                raise NotImplementedError

            def test_dataloader(self):  # type: ignore[no-untyped-def]
                raise NotImplementedError

            def val_dataloader(self):  # type: ignore[no-untyped-def]
                raise NotImplementedError

        cfg_bad = base_cfg.model_copy(update={"subset_names": tuple(["train", "bar"])})
        dm_bad = WithBadAttr(cfg_bad)
        with pytest.raises(ValueError) as exc_info:
            _ = dm_bad.get_dataloader("bar")
        assert "expected callable" in str(exc_info.value)

    def test_integration_builds_torch_dataloaders(self) -> None:
        cfg = build_default_config()
        dm = DummyDataModule(cfg)
        train_loader = dm.train_dataloader()
        valid_loader = dm.val_dataloader()
        # ensure torch dataloaders are returned and iterate
        assert isinstance(train_loader, tud.DataLoader)
        assert isinstance(valid_loader, tud.DataLoader)
        t_batch = next(iter(train_loader))
        v_batch = next(iter(valid_loader))
        assert len(t_batch) == 4
        assert len(v_batch) == 1
