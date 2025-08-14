import importlib
import pathlib
import types
import typing

import pydantic
import pytest
import yaml

import pyine.utils.pydantic_loader as pyl


class MyModel(pydantic.BaseModel):
    a: int
    b: str


def setup_function(_fn):  # ensure registry clean before each test file function
    pyl.PydanticYAMLLoader.clear_registry()


def test_register_model_and_lookup(monkeypatch: pytest.MonkeyPatch):
    # invalid model type
    with pytest.raises(ValueError):
        pyl.PydanticYAMLLoader.register_model("bad", typing.cast(type, int))  # type: ignore[arg-type,assignment]

    # proper registration and idempotency
    pyl.PydanticYAMLLoader.register_model("mymodel", MyModel)
    pyl.PydanticYAMLLoader.register_model("mymodel", MyModel)  # should be no-op
    reg = pyl.PydanticYAMLLoader.get_registered_models()
    assert "mymodel" in reg and reg["mymodel"] is MyModel
    assert pyl.PydanticYAMLLoader.get_model_tag_by_class(MyModel) == "mymodel"


def test_register_models_from_module_success_and_failure(monkeypatch: pytest.MonkeyPatch):
    # success on current module
    pyl.PydanticYAMLLoader.register_models_from_module(__name__)
    tag = f"{__name__}.MyModel"
    assert tag in pyl.PydanticYAMLLoader.get_registered_models()

    # failure path: import error -> ValueError
    def boom(name):
        raise ImportError("boom")

    monkeypatch.setattr(pyl.importlib, "import_module", boom, raising=True)
    with pytest.raises(ValueError):
        pyl.PydanticYAMLLoader.register_models_from_module("does.not.exist")


def test_register_models_from_package_failure():
    # non existent package -> ValueError
    with pytest.raises(ValueError):
        pyl.PydanticYAMLLoader.register_models_from_package("does.not.exist.pkg")


def test_yaml_dump_and_load_roundtrip(tmp_path: pathlib.Path):
    # register the fully-qualified tag for MyModel
    tag = f"{__name__}.MyModel"
    pyl.PydanticYAMLLoader.register_model(tag, MyModel)
    # create instance and dump to file
    inst = MyModel(a=3, b="x")
    yaml_path = tmp_path / "model.yaml"
    dumped = pyl.dump_yaml_with_pydantic_support(inst, file_path=yaml_path)
    assert yaml_path.exists() and dumped.strip().startswith(f"!{tag}")
    # load back using file loader
    loaded = pyl.load_yaml_with_pydantic_support(yaml_path)
    assert isinstance(loaded, MyModel) and loaded == inst


def test_yaml_unknown_tag_and_validation_error(tmp_path: pathlib.Path):
    # unknown tag error path
    bad = tmp_path / "bad.yaml"
    bad.write_text("!unknown.tag\na: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        _ = pyl.load_yaml_with_pydantic_support(bad)

    # register model but provide invalid content to trigger validation error
    tag = f"{__name__}.MyModel"
    pyl.PydanticYAMLLoader.register_model(tag, MyModel)
    bad_val = tmp_path / "bad_val.yaml"
    bad_val.write_text(f"!{tag}\na: 'not-int'\n", encoding="utf-8")
    with pytest.raises(ValueError):
        _ = pyl.load_yaml_with_pydantic_support(bad_val)


def test_loader_file_errors(tmp_path: pathlib.Path):
    # missing file
    with pytest.raises(FileNotFoundError):
        _ = pyl.load_yaml_with_pydantic_support(tmp_path / "missing.yaml")

    # invalid YAML content should raise yaml.YAMLError
    inv = tmp_path / "invalid.yaml"
    inv.write_text(":\n -\n", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        _ = pyl.load_yaml_with_pydantic_support(inv)
