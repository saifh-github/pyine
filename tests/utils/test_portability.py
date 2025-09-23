import builtins
import dataclasses
import datetime

import numpy as np
import omegaconf
import pandas as pd
import pydantic
import pytest
import rich.console

import pyine.utils.portability as portability


class TestEstimateTolerance:
    def test_integer_values(self):
        rtol, atol = portability.estimate_tolerance("123")
        assert atol == 0.0
        assert 1e-15 <= rtol <= 1e-5
        assert not np.isclose(122, 123, rtol, atol)
        assert np.isclose(122.9999, 123, rtol, atol)

    def test_simple_decimal(self):
        rtol, atol = portability.estimate_tolerance("1.0")
        assert atol == 0.0
        assert 1e-15 <= rtol <= 1e-5
        assert not np.isclose(1, 2, rtol, atol)
        assert np.isclose(0.99999, 1, rtol, atol)

    def test_decimal_with_precision(self):
        rtol, atol = portability.estimate_tolerance("3.14159")
        expected_atol = 0.5 * (10 ** (-5))  # 5 decimal places
        assert atol == expected_atol
        assert 1e-15 <= rtol <= 1e-5
        assert not np.isclose(3.1415, 3.14159, rtol, atol)
        assert np.isclose(3.141592654, 3.14159, rtol, atol)

    def test_small_decimal(self):
        rtol, atol = portability.estimate_tolerance("0.001")
        expected_atol = 0.5 * (10 ** (-3))  # 3 decimal places
        assert atol == expected_atol
        assert 1e-15 <= rtol <= 1e-5
        assert not np.isclose(0.0001, 0.001, rtol, atol)
        assert np.isclose(0.0012, 0.001, rtol, atol)

    def test_scientific_notation_small(self):
        rtol, atol = portability.estimate_tolerance("1.23e-6")
        assert rtol == 1e-5  # magnitude < 1e-6
        expected_atol = 0.5 * (10 ** (-8))  # 3 sig digits, exp -6
        assert atol == expected_atol
        assert not np.isclose(1.23e-7, 1.23e-6, rtol, atol)
        assert np.isclose(1.2345e-6, 1.23e-6, rtol, atol)

    def test_scientific_notation_large(self):
        rtol, atol = portability.estimate_tolerance("2.5e10")
        assert atol == 0.0
        assert rtol == 1e-5
        assert not np.isclose(2.5e9, 2.5e10, rtol, atol)
        assert np.isclose(2.49998e10, 2.5e10, rtol, atol)

    def test_negative_values(self):
        rtol, atol = portability.estimate_tolerance("-3.14")
        expected_atol = 0.5 * (10 ** (-2))
        assert atol == expected_atol
        assert 1e-15 <= rtol <= 1e-5

    def test_zero_value(self):
        rtol, atol = portability.estimate_tolerance("0")
        assert atol == 0.0
        assert rtol == 1e-9

    def test_zero_with_decimal(self):
        rtol, atol = portability.estimate_tolerance("0.0")
        assert atol == 0.0
        assert rtol == 1e-9

    def test_whitespace_handling(self):
        rtol, atol = portability.estimate_tolerance("  3.14  ")
        expected_atol = 0.5 * (10 ** (-2))
        assert atol == expected_atol
        assert 1e-15 <= rtol <= 1e-5

    def test_invalid_string_raises_error(self):
        with pytest.raises(ValueError):
            portability.estimate_tolerance("not_a_number")

    def test_tolerance_bounds_applied(self):
        rtol, atol = portability.estimate_tolerance("1.0000000000000001")
        assert rtol >= 1e-15
        assert rtol <= 1e-5


def test_get_portable_representation_various_types():
    assert portability.get_portable_representation(42) == "42"
    assert portability.get_portable_representation(3.14) == repr(3.14)
    assert portability.get_portable_representation(True) == "True"
    assert portability.get_portable_representation(None) == "None"
    assert portability.get_portable_representation(b"x") == "b'x'"

    arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
    arr_repr = portability.get_portable_representation(arr)
    assert arr_repr.startswith("numpy.array")
    assert "[1, 2]" in arr_repr and "[3, 4]" in arr_repr

    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    df_repr = portability.get_portable_representation(df)
    assert df_repr.startswith("pandas.DataFrame")
    ser = pd.Series([1, 2, 3], name="s")
    ser_repr = portability.get_portable_representation(ser)
    assert ser_repr.startswith("pandas.Series")

    lst = [1, 2, 3]
    lst_repr = portability.get_portable_representation([1, 2, 3])
    assert lst_repr == repr(lst)
    tup_repr = portability.get_portable_representation(tuple(lst))
    assert tup_repr == repr(tuple(lst))
    dct = {"a": 1, "b": 2}
    dict_repr = portability.get_portable_representation(dct)
    assert dict_repr == repr(dct)

    assert "numpy" in portability.get_portable_representation(np)

    exc = ValueError("bad")
    ex_repr = portability.get_portable_representation(exc)
    assert ex_repr == repr(exc)

    def foo(x):
        return x

    call_repr = portability.get_portable_representation(foo)
    assert call_repr.startswith("<callable '")
    assert call_repr.endswith("<locals>.foo'>")

    class CallableNoName:
        def __call__(self):
            return 0

    c = CallableNoName()
    setattr(c, "__module__", None)
    anon_repr = portability.get_portable_representation(c)
    assert anon_repr.startswith("<callable '")
    assert anon_repr.endswith("CallableNoName.__call__'>")

    class C:
        pass

    inst_repr = portability.get_portable_representation(C())
    assert inst_repr.startswith("<instance '")
    assert inst_repr.endswith("<locals>.C'>")

    obj_repr = portability.get_portable_representation(object())
    assert obj_repr == "<instance 'object'>"


def test_get_portable_representation_truncation():
    lst = list(range(200))
    s = portability.get_portable_representation(lst, max_length=20)
    assert s.endswith("...")
    assert len(s) == 20


def test_format_object_changes_numpy_df_series_dict_list_tuple_instance():
    # numpy ndarray changes and shape mismatch
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[1, 99], [3, 4]])
    changes = portability.format_object_changes(a, b)
    assert len(changes) == 1 and changes[0].startswith("numpy.ndarray:shape=(2, 2),dtype=")
    # shape mismatch returns None
    assert portability.format_object_changes(a, np.array([1, 2, 3])) is None

    # pandas DataFrame: single cell change
    df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    df2 = df1.copy()
    df2.loc[0, "A"] = 10
    df_changes = portability.format_object_changes(df1, df2)
    assert df_changes == ["dataframe:shape=(2, 2):(0,A):10"]

    # pandas Series
    s1 = pd.Series([1, 2, 3])
    s2 = s1.copy()
    s2.iloc[1] = 5
    s_changes = portability.format_object_changes(s1, s2)
    assert s_changes == ["series:len=3:1:5"]

    # dict changes and too many changes
    past = {"a": 1, "b": 2}
    cur = {"b": 3, "c": 4}
    d_changes = portability.format_object_changes(past, cur)
    assert set(d_changes) == {
        "dict:len=2:removed(a):None",
        "dict:len=2:added(c):4",
        "dict:len=2:changed(b):3",
    }
    # too many
    big_past = {i: i for i in range(20)}
    big_cur = {i: i + 1 for i in range(20)}
    assert portability.format_object_changes(big_past, big_cur, max_count=5) is None

    # list/tuple
    lst1, lst2 = [1, 2, 3], [1, 9, 3]
    t_changes = portability.format_object_changes(lst1, lst2)
    assert t_changes == ["list:len=3:1:9"]
    tup1, tup2 = (1, 2, 3), (1, 2, 99)
    tup_changes = portability.format_object_changes(tup1, tup2)
    assert tup_changes == ["tuple:len=3:2:99"]

    # objects with changed attributes
    class Obj:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    o1, o2 = Obj(1, 2), Obj(1, 3)
    inst_changes = portability.format_object_changes(o1, o2)
    assert inst_changes == [f"instance:{Obj.__module__}.{Obj.__name__}:changed(y):3"]

    # incompatible types
    assert portability.format_object_changes(1, "1") is None


def test_get_portable_filename():
    assert portability.get_portable_filename("<string>") == "<string>"
    assert portability.get_portable_filename("foo.py") == "foo.py"
    assert portability.get_portable_filename(np.__file__) == "numpy/__init__.py"
    assert portability.get_portable_filename(__file__) == "tests/utils/test_portability.py"


class SomeDummyClass:
    def __init__(self, x):
        self.x = x

    @staticmethod
    def some_static_fn():
        pass

    @classmethod
    def some_class_method(cls):
        pass

    def some_instance_method(self):
        pass


def test_portable_function_name():
    assert "lambda" in portability.get_portable_function_name(lambda x: x)
    curr_test_name = "tests.utils.test_portability.test_portable_function_name"
    assert portability.get_portable_function_name(test_portable_function_name) == curr_test_name

    def _local_func():
        pass

    assert portability.get_portable_function_name(_local_func) == curr_test_name + ".<locals>._local_func"
    expected_cls_name = "tests.utils.test_portability.SomeDummyClass"
    assert portability.get_portable_function_name(SomeDummyClass) == expected_cls_name
    assert (
        portability.get_portable_function_name(SomeDummyClass.some_static_fn) == expected_cls_name + ".some_static_fn"
    )
    assert (
        portability.get_portable_function_name(SomeDummyClass.some_class_method)
        == expected_cls_name + ".some_class_method"
    )
    assert (
        portability.get_portable_function_name(SomeDummyClass(1).some_instance_method)
        == expected_cls_name + ".some_instance_method"
    )


def test_numbered_lines_helpers(monkeypatch: pytest.MonkeyPatch):
    code = "a = 1\nprint(a)"
    formatted = portability.get_code_with_numbered_lines(code, prefixed_tabs=2)
    lines = formatted.splitlines()
    assert lines[0].startswith("\t\tL0001:   a = 1")
    assert lines[1].startswith("\t\tL0002:   print(a)")

    captured = []

    class Logger:
        def info(self, m):
            captured.append(m)

    portability.print_code_with_numbered_lines(code, prefixed_tabs=1, logger=Logger())
    assert len(captured) == 2 and captured[0].startswith("\tL0001:   ")

    # default print path
    printed = []
    monkeypatch.setattr(builtins, "print", lambda m: printed.append(m))
    portability.print_code_with_numbered_lines(code, prefixed_tabs=0, logger=None)
    assert len(printed) == 2 and printed[0].startswith("L0001:   ")


class TestGetFullyQualifiedName:

    def test_local_function(self):
        def foo():
            pass

        name = portability.get_fully_qualified_name(foo)
        assert name.endswith("TestGetFullyQualifiedName.test_local_function.<locals>.foo")

    def test_builtin_type(self):
        name = portability.get_fully_qualified_name(list)
        assert name == "list"

    def test_3rd_party_class(self):
        import torch.utils.data as data_utils

        name = portability.get_fully_qualified_name(data_utils.DataLoader)
        assert name == "torch.utils.data.dataloader.DataLoader"


class TestImportFromDottedPath:

    def test_import_builtin_module(self):
        module = portability.import_from_dotted_path("math")
        assert module.__name__ == "math"

    def test_import_custom_path(self):
        module = portability.import_from_dotted_path("pyine.utils.portability")
        assert module.__name__ == "pyine.utils.portability"

    def test_import_function_from_custom_module(self):
        func = portability.import_from_dotted_path("pyine.utils.portability.get_portable_representation")
        assert func.__name__ == "get_portable_representation"

    def test_import_nonexistent_module(self):
        with pytest.raises(ImportError):
            portability.import_from_dotted_path("nonexistent.module")

    def test_import_invalid_path(self):
        with pytest.raises(ValueError):
            portability.import_from_dotted_path("")


def test_parse_duration_to_timedelta():
    assert portability.parse_duration_to_timedelta("") is None
    assert portability.parse_duration_to_timedelta(None) is None
    assert portability.parse_duration_to_timedelta("0s") == datetime.timedelta(0)
    assert portability.parse_duration_to_timedelta("1h") == datetime.timedelta(hours=1)
    assert portability.parse_duration_to_timedelta("30m") == datetime.timedelta(minutes=30)
    assert portability.parse_duration_to_timedelta("45s") == datetime.timedelta(seconds=45)
    assert portability.parse_duration_to_timedelta("1h30m") == datetime.timedelta(hours=1, minutes=30)
    assert portability.parse_duration_to_timedelta("2h15m30s") == datetime.timedelta(hours=2, minutes=15, seconds=30)
    assert portability.parse_duration_to_timedelta("45m30s") == datetime.timedelta(minutes=45, seconds=30)
    with pytest.raises(ValueError):
        portability.parse_duration_to_timedelta("invalid")
    with pytest.raises(ValueError):
        portability.parse_duration_to_timedelta("1happy")


def test_parse_indices_spec():
    assert portability.parse_indices_spec("") == []
    assert portability.parse_indices_spec("0") == [0]
    assert portability.parse_indices_spec("0,1") == [0, 1]
    assert portability.parse_indices_spec("5-9") == list(range(5, 10))
    assert portability.parse_indices_spec("0-1,2") == [0, 1, 2]
    assert portability.parse_indices_spec("0,1-2,3") == [0, 1, 2, 3]
    assert portability.parse_indices_spec("4-10,1-3") == [1, 2, 3, *range(4, 11)]


def test_render_config_for_dataclass_and_pydantic():
    @dataclasses.dataclass
    class DemoDataclass:
        foo: int = 1
        bar: str = "baz"

    class DemoModel(pydantic.BaseModel):
        foo: int = pydantic.Field(1, description="foo field")
        bar: str = pydantic.Field("baz", description="bar field")

    dataclass_cfg = omegaconf.OmegaConf.create({"foo": 10, "bar": "zzz"})
    dataclass_console = rich.console.Console(record=True, width=60)
    portability.render_config(
        DemoDataclass,
        dataclass_cfg,
        console=dataclass_console,
        title="DemoDataclass",
        show_field_descriptions=False,
    )
    dataclass_output = dataclass_console.export_text()
    assert "DemoDataclass" in dataclass_output
    assert "10" in dataclass_output

    model_console = rich.console.Console(record=True, width=60)
    portability.render_config(
        DemoModel,
        DemoModel(foo=7, bar="qux").model_dump(),
        console=model_console,
        title="DemoModel",
        show_field_descriptions=True,
    )
    model_output = model_console.export_text()
    assert "DemoModel" in model_output
    assert "foo field" in model_output
    assert "qux" in model_output


def test_rich_fold_indicator_wraps_lines():
    indicator = portability._RichFoldIndicator("line one\nline two that wraps", prefix="> ", suffix=" <")
    console = rich.console.Console(record=True, width=15)
    console.print(indicator)
    rendered = console.export_text()
    assert "> " in rendered
    assert " <" in rendered
