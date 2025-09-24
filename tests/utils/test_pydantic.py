import pathlib
import types
import typing

import pydantic
import pytest
import yaml

import pyine.utils.portability as portability
import pyine.utils.pydantic as pyd


class MyModel(pydantic.BaseModel):
    a: int
    b: str


def setup_function(_fn):  # ensure registry clean before each test file function
    pyd.PydanticYAMLLoader.clear_registry()


def test_register_model_and_lookup(monkeypatch: pytest.MonkeyPatch):
    # invalid model type
    with pytest.raises(ValueError):
        pyd.PydanticYAMLLoader.register_model("bad", typing.cast(type, int))  # type: ignore[arg-type,assignment]

    # proper registration and idempotency
    pyd.PydanticYAMLLoader.register_model("mymodel", MyModel)
    pyd.PydanticYAMLLoader.register_model("mymodel", MyModel)  # should be no-op
    reg = pyd.PydanticYAMLLoader.get_registered_models()
    assert "mymodel" in reg and reg["mymodel"] is MyModel
    assert pyd.PydanticYAMLLoader.get_model_tag_by_class(MyModel) == "mymodel"


def test_register_models_from_module_success_and_failure(monkeypatch: pytest.MonkeyPatch):
    # success on current module
    pyd.PydanticYAMLLoader.register_models_from_module(__name__)
    tag = f"{__name__}.MyModel"
    assert tag in pyd.PydanticYAMLLoader.get_registered_models()

    # failure path: import error -> ValueError
    def boom(name):
        raise ImportError("boom")

    monkeypatch.setattr(pyd.importlib, "import_module", boom, raising=True)
    with pytest.raises(ValueError):
        pyd.PydanticYAMLLoader.register_models_from_module("does.not.exist")


def test_register_models_from_package_failure():
    # non existent package -> ValueError
    with pytest.raises(ValueError):
        pyd.PydanticYAMLLoader.register_models_from_package("does.not.exist.pkg")


def test_yaml_dump_and_load_roundtrip(tmp_path: pathlib.Path):
    # register the fully-qualified tag for MyModel
    tag = f"{__name__}.MyModel"
    pyd.PydanticYAMLLoader.register_model(tag, MyModel)
    # create instance and dump to file
    inst = MyModel(a=3, b="x")
    yaml_path = tmp_path / "model.yaml"
    dumped = pyd.dump_yaml_with_pydantic_support(inst, file_path=yaml_path)
    assert yaml_path.exists() and dumped.strip().startswith(f"!{tag}")
    # load back using file loader
    loaded = pyd.load_yaml_with_pydantic_support(yaml_path)
    assert isinstance(loaded, MyModel) and loaded == inst


def test_yaml_unknown_tag_and_validation_error(tmp_path: pathlib.Path):
    # unknown tag error path
    bad = tmp_path / "bad.yaml"
    bad.write_text("!unknown.tag\na: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        _ = pyd.load_yaml_with_pydantic_support(bad)

    # register model but provide invalid content to trigger validation error
    tag = f"{__name__}.MyModel"
    pyd.PydanticYAMLLoader.register_model(tag, MyModel)
    bad_val = tmp_path / "bad_val.yaml"
    bad_val.write_text(f"!{tag}\na: 'not-int'\n", encoding="utf-8")
    with pytest.raises(ValueError):
        _ = pyd.load_yaml_with_pydantic_support(bad_val)


def test_loader_file_errors(tmp_path: pathlib.Path):
    # missing file
    with pytest.raises(FileNotFoundError):
        _ = pyd.load_yaml_with_pydantic_support(tmp_path / "missing.yaml")

    # invalid YAML content should raise yaml.YAMLError
    inv = tmp_path / "invalid.yaml"
    inv.write_text(":\n -\n", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        _ = pyd.load_yaml_with_pydantic_support(inv)


class DummyBase:
    def __init__(
        self,
        a: int,
    ) -> None:
        self.a = a


class DummySub(DummyBase):
    def __init__(
        self,
        a: int = 0,
        b: typing.Any | None = None,
    ) -> None:
        super().__init__(a)
        self.b = b


class DummyUnrelated:
    def __init__(self, c: str) -> None:
        self.c = c


class DummyWithConfig:
    def __init__(self, cfg: dict):
        self.cfg = MyModel(**cfg)


@pytest.fixture(name="install_fake_import")
def fixture_install_fake_import(
    monkeypatch: pytest.MonkeyPatch,
) -> typing.Callable[[dict[str, typing.Any]], None]:
    """Install a fake dotted-path importer for the portability layer.

    Args:
        monkeypatch: pytest monkeypatch fixture.

    Returns:
        A function that, when called with a mapping, replaces
        pyine.utils.portability.import_from_dotted_path to return objects from
        the mapping or raise ImportError. If a mapping value is an Exception,
        it is raised as-is.
    """

    def _install(
        mapping: dict[str, typing.Any],
    ) -> None:
        def fake_import(
            path: str,
        ) -> typing.Any:
            if path in mapping:
                value = mapping[path]
                if isinstance(value, Exception):
                    raise value
                return value
            raise ImportError(f"Cannot import '{path}'")

        monkeypatch.setattr(
            portability,
            "import_from_dotted_path",
            fake_import,
            raising=True,
        )

    return _install


class TestClassImportSpec:

    def test_resolve_success(
        self,
        install_fake_import: typing.Callable[[dict[str, typing.Any]], None],
    ) -> None:
        install_fake_import(
            {
                "pkg.module.DummySub": DummySub,
                "pkg.module.DummyBase": DummyBase,
            },
        )
        _ = pyd.ClassImportSpec(
            class_path="pkg.module.DummySub",
            base_class_path="pkg.module.DummyBase",
        )

    def test_resolve_raises_if_class_not_type(
        self,
        install_fake_import: typing.Callable[[dict[str, typing.Any]], None],
    ) -> None:
        not_a_class: typing.Any = object()
        install_fake_import(
            {
                "pkg.module.NotAClass": not_a_class,
                "pkg.module.DummyBase": DummyBase,
            },
        )
        with pytest.raises(TypeError) as exc_info:
            _ = pyd.ClassImportSpec(
                class_path="pkg.module.NotAClass",
                base_class_path="pkg.module.DummyBase",
            )
        assert "is not a class" in str(exc_info.value)

    def test_resolve_raises_if_base_not_type(
        self,
        install_fake_import: typing.Callable[[dict[str, typing.Any]], None],
    ) -> None:
        not_a_class: typing.Any = 123
        install_fake_import(
            {
                "pkg.module.DummySub": DummySub,
                "pkg.module.NotAClass": not_a_class,
            },
        )
        with pytest.raises(TypeError) as exc_info:
            _ = pyd.ClassImportSpec(
                class_path="pkg.module.DummySub",
                base_class_path="pkg.module.NotAClass",
            )
        assert "is not a class" in str(exc_info.value)

    def test_resolve_raises_if_not_subclass(
        self,
        install_fake_import: typing.Callable[[dict[str, typing.Any]], None],
    ) -> None:
        install_fake_import(
            {
                "pkg.module.DummyUnrelated": DummyUnrelated,
                "pkg.module.DummyBase": DummyBase,
            },
        )
        with pytest.raises(TypeError) as exc_info:
            _ = pyd.ClassImportSpec(
                class_path="pkg.module.DummyUnrelated",
                base_class_path="pkg.module.DummyBase",
            )
        assert "is not a subclass of" in str(exc_info.value)

    def test_resolve_raises_if_bad_generic_base(
        self,
        install_fake_import: typing.Callable[[dict[str, typing.Any]], None],
    ) -> None:
        install_fake_import(
            {
                "pkg.module.DummySub": DummySub,
                "pkg.module.DummyBase": DummyBase,
                "pkg.module.DummyUnrelated": DummyUnrelated,
            },
        )
        with pytest.raises(TypeError) as exc_info:
            _ = pyd.ClassImportSpec[DummyUnrelated](
                class_path="pkg.module.DummySub",
                base_class_path="pkg.module.DummyBase",
                params={"a": 7, "b": {"k": "v"}},
            )
        assert "not compatible with expected" in str(exc_info.value)

    def test_instantiate_success_with_params(
        self,
        install_fake_import: typing.Callable[[dict[str, typing.Any]], None],
    ) -> None:
        install_fake_import(
            {
                "pkg.module.DummySub": DummySub,
                "pkg.module.DummyBase": DummyBase,
            },
        )
        spec = pyd.ClassImportSpec(
            class_path="pkg.module.DummySub",
            base_class_path="pkg.module.DummyBase",
            params={"a": 7, "b": {"k": "v"}},
        )
        instance = spec.instantiate()
        assert isinstance(instance, DummySub)
        assert instance.a == 7
        assert instance.b == {"k": "v"}

    def test_instantiate_propagates_constructor_error(
        self,
        install_fake_import: typing.Callable[[dict[str, typing.Any]], None],
    ) -> None:
        class FailingCtor(DummyBase):
            def __init__(
                self,
                *,
                must: int,
            ) -> None:
                super().__init__(a=must)

        install_fake_import(
            {
                "pkg.module.FailingCtor": FailingCtor,
                "pkg.module.DummyBase": DummyBase,
            },
        )
        with pytest.raises(ValueError) as exc_info:
            _ = pyd.ClassImportSpec(
                class_path="pkg.module.FailingCtor",
                base_class_path="pkg.module.DummyBase",
                params={"extra": 1},  # this parameter does NOT exist
            )
        assert "invalid parameter(s) for" in str(exc_info.value)
        assert "extra" in str(exc_info.value)
        # also make sure that if we don't pass the missing param ('must'), we can't instantiate
        spec = pyd.ClassImportSpec(
            class_path="pkg.module.FailingCtor",
            base_class_path="pkg.module.DummyBase",
        )
        with pytest.raises(TypeError) as exc_info:
            _ = spec.instantiate()
        assert "missing 1 required" in str(exc_info.value)
        # however, with the missing param, all is good
        obj = spec.instantiate(must=13)
        assert isinstance(obj, FailingCtor)
        assert obj.a == 13

    def test_instantiate_with_updated_spec(
        self,
        install_fake_import: typing.Callable[[dict[str, typing.Any]], None],
    ) -> None:
        install_fake_import(
            {
                "pkg.module.DummySub": DummySub,
                "pkg.module.DummyBase": DummyBase,
            },
        )
        spec = pyd.ClassImportSpec(
            class_path="pkg.module.DummySub",
            base_class_path="pkg.module.DummyBase",
            params={"a": 7, "b": -1},
        )
        updated_spec = spec.get_updated_spec(b="test")
        instance = updated_spec.instantiate()
        assert isinstance(instance, DummySub)
        assert instance.a == 7
        assert instance.b == "test"
        instance = updated_spec.instantiate(b=12)
        assert isinstance(instance, DummySub)
        assert instance.a == 7
        assert instance.b == 12

    def test_get_updated_spec_merges_nested_params(
        self,
        install_fake_import: typing.Callable[[dict[str, typing.Any]], None],
    ) -> None:
        install_fake_import(
            {
                "pkg.module.DummySub": DummySub,
                "pkg.module.DummyBase": DummyBase,
            },
        )
        spec = pyd.ClassImportSpec(
            class_path="pkg.module.DummySub",
            base_class_path="pkg.module.DummyBase",
            params={
                "a": 5,
                "b": {
                    "inner": {
                        "original": True,
                        "kept": "value",
                    },
                    "top_level": "preserve",
                },
            },
        )
        updated_spec = spec.get_updated_spec(
            b={
                "inner": {
                    "original": False,
                    "override": "new",
                },
                "additional": 42,
            },
        )
        assert spec.params["b"]["inner"] == {"original": True, "kept": "value"}
        assert updated_spec.params["b"]["inner"] == {
            "original": False,
            "kept": "value",
            "override": "new",
        }
        assert updated_spec.params["b"]["top_level"] == "preserve"
        assert updated_spec.params["b"]["additional"] == 42
        assert updated_spec.params["a"] == 5

    def test_resolve_propagates_import_error_for_missing_class(
        self,
        install_fake_import: typing.Callable[[dict[str, typing.Any]], None],
    ) -> None:
        install_fake_import(
            {
                "pkg.module.DummyBase": DummyBase,
            },
        )
        with pytest.raises(ImportError):
            _ = pyd.ClassImportSpec(
                class_path="pkg.module.Missing",
                base_class_path="pkg.module.DummyBase",
            )

    def test_no_base_and_params_key(
        self,
        install_fake_import: typing.Callable[[dict[str, typing.Any]], None],
    ) -> None:
        install_fake_import(
            {
                "pkg.module.DummyWithConfig": DummyWithConfig,
            },
        )
        spec = pyd.ClassImportSpec(
            class_path="pkg.module.DummyWithConfig",
            base_class_path="pkg.module.DummyWithConfig",
            params={"a": 7, "b": "hello"},
            params_key="cfg",
        )
        instance = spec.instantiate()
        assert isinstance(instance, DummyWithConfig)
        assert instance.cfg.a == 7
        assert instance.cfg.b == "hello"


def test_is_jsonvalue():
    assert pyd.is_jsonvalue(1) is True
    assert pyd.is_jsonvalue(1.0) is True
    assert pyd.is_jsonvalue("hello") is True
    assert pyd.is_jsonvalue(True) is True
    assert pyd.is_jsonvalue(None) is True
    assert pyd.is_jsonvalue([1, 2, {"1": 1}]) is True
    assert pyd.is_jsonvalue({"a": {1: 123}}) is False
    assert pyd.is_jsonvalue({"a": 1}) is True
    assert pyd.is_jsonvalue(MyModel(a=1, b="2")) is False
    assert pyd.is_jsonvalue(MyModel(a=1, b="2").model_dump()) is True
    assert pyd.is_jsonvalue(int) is False


def f_basic(a: int, b: str = "x", c=None, d: list[int] = []):  # noqa (intentional for tests)
    return a, b, c, d


def f_var(*args, **kwargs):
    return args, kwargs


def f_mix(x: int, y: str, *, z: float = 1.0, **kw):
    return x, y, z, kw


class CtorExample:
    def __init__(self, n: int, scale: float = 1.0) -> None:
        self.n = n
        self.scale = scale


def f_annot(p: typing.Annotated[str | None, "tag"] = None):
    return p


class TestModelFromCallable:

    def test_required_vs_optional_and_types(self):
        Model = pyd.model_from_callable(f_basic)
        assert Model.model_fields["a"].default is pydantic.fields.PydanticUndefined
        assert Model.model_fields["b"].default == "x"
        assert Model.model_fields["c"].default is None
        assert Model.model_fields["c"].annotation is typing.Any
        m = Model(a=3, b="ok", d=[1, 2])
        assert m.a == 3 and m.b == "ok" and m.d == [1, 2]
        with pytest.raises(pydantic.ValidationError):
            Model(a="not-an-int")

    def test_model_name_override(self):
        Model = pyd.model_from_callable(f_basic, name="Custom")
        assert Model.__name__ == "Custom"

    def test_var_positional_and_var_keyword_fields_present(self):
        Model = pyd.model_from_callable(f_var)
        args_field = Model.model_fields["args"]
        kwargs_field = Model.model_fields["kwargs"]
        assert typing.get_origin(args_field.annotation) is tuple
        assert typing.get_args(args_field.annotation) == (typing.Any, ...)
        assert typing.get_origin(kwargs_field.annotation) is dict
        assert typing.get_args(kwargs_field.annotation) == (str, typing.Any)
        m = Model()  # defaults should apply
        assert m.args == ()
        assert m.kwargs == {}

    def test_include_exclude_and_exclude_kwargs(self):
        Model = pyd.model_from_callable(
            f_mix,
            include={"x", "z"},
            exclude={"z"},  # exclude wins
            include_kwargs=False,  # drop **kw
            name="FilteredArgs",
        )
        assert set(Model.model_fields.keys()) == {"x", "y"} - {"y"} | set()  # only {"x"}
        assert set(Model.model_fields.keys()) == {"x"}

    def test_constructor_signature_is_used_for_classes(self):
        Model = pyd.model_from_callable(CtorExample)
        # fields mirror __init__(n: int, scale: float = 1.0)
        assert set(Model.model_fields.keys()) == {"n", "scale"}
        assert Model.model_fields["n"].default is pydantic.fields.PydanticUndefined
        assert Model.model_fields["scale"].default == 1.0
        # use validated args to build the instance
        cfg = Model(n=5, scale=2.5)
        obj = CtorExample(**cfg.model_dump(exclude_none=True))
        assert isinstance(obj, CtorExample)
        assert obj.n == 5 and obj.scale == 2.5

    def test_annotated_and_optional_are_preserved(self):
        Model = pyd.model_from_callable(f_annot)
        f = Model.model_fields["p"]
        # underlying type is Optional[str] (Union[str, NoneType])
        origin = typing.get_origin(f.annotation)
        assert origin in (typing.Union, types.UnionType)
        args = typing.get_args(f.annotation)
        assert len(args) == 2 and type(None) in args and str in args
        assert "tag" in getattr(f, "metadata", ())
        assert f.default is None

    def test_any_for_unannotated_param_is_accepted(self):
        Model = pyd.model_from_callable(f_basic)
        # c has annotation Any, so any type accepted
        m = Model(a=1, c={"anything": object()})
        assert isinstance(m.c, dict)

    def test_validation_error_messages_are_informative(self):
        Model = pyd.model_from_callable(f_basic)
        with pytest.raises(pydantic.ValidationError) as ei:
            Model()  # missing required 'a'
        msg = str(ei.value)
        assert "a" in msg and "Field required" in msg

    def test_kwargs_included_by_default(self):
        Model = pyd.model_from_callable(f_mix, name="MixArgs")  # include_kwargs defaults to True
        assert "kw" in Model.model_fields
        m = Model(x=1, y="y", kw={"extra": 1})
        assert m.kw == {"extra": 1}

    def test_custom_model_configs(self):
        Model = pyd.model_from_callable(
            f_basic,
            model_config=pydantic.ConfigDict(frozen=True, extra="allow"),
        )
        m = Model(a=13, z="hello")
        assert m.a == 13 and m.z == "hello"
        with pytest.raises(pydantic.ValidationError):
            m.a = 15
        Model2 = pyd.model_from_callable(
            f_basic,
            model_config=pydantic.ConfigDict(extra="forbid"),
        )
        m = Model2(a=13)
        assert m.a == 13
        m.a = 15
        assert m.a == 15
        with pytest.raises(pydantic.ValidationError):
            _ = Model2(a=13, z="hello")

    def test_openai_wrapper(self):
        import openai

        _ = pyd.model_from_callable(openai.OpenAI)
        # if we manage the create the above object, we've solved the forward-ref hints problem
