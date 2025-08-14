import builtins
import types

import numpy as np
import pandas as pd
import pytest

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
    assert portability.get_portable_representation(arr) == "numpy.ndarray(shape=(2, 2),dtype=int32)"
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    assert portability.get_portable_representation(df) == "pandas.DataFrame(shape=(2, 2))"
    ser = pd.Series([1, 2, 3], name="s")
    assert portability.get_portable_representation(ser) == "pandas.Series(len=3,dtype=int64,name=s)"
    assert portability.get_portable_representation([1, 2, 3]) == "list(len=3)"
    assert portability.get_portable_representation((1, 2)) == "tuple(len=2)"
    assert portability.get_portable_representation({1, 2}) == "set(len=2)"
    assert portability.get_portable_representation({"a": 1, "b": 2}) == "dict(len=2)"
    # module
    assert portability.get_portable_representation(np) == "module(numpy)"
    # exception instance
    ex_repr = portability.get_portable_representation(ValueError("bad"))
    assert ex_repr.startswith("ValueError(")

    # callable
    def foo(x):
        return x

    call_repr = portability.get_portable_representation(foo)
    assert call_repr.startswith("callable(")

    # callable instance lacking module/name triggers anonymous path
    class CallableNoName:
        def __call__(self):
            return 0

    c = CallableNoName()
    setattr(c, "__module__", None)
    anon_repr = portability.get_portable_representation(c)
    assert anon_repr.startswith("callable(anonymous(")

    # instance
    class C:
        pass

    inst_repr = portability.get_portable_representation(C())
    assert inst_repr.startswith("instance(")
    # default repr cleanup
    cleaned = portability.get_portable_representation(object())
    assert "0x" not in cleaned


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
    assert set(d_changes) == {"dict:len=2:removed(a):None", "dict:len=2:added(c):4", "dict:len=2:changed(b):3"}
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
