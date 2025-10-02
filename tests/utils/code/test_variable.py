from pyine.utils.code.variables import analyze_definitions


def test_analyze_definitions__simple_case() -> None:
    code = """
import os

x = 1
"""
    res = analyze_definitions(code)

    expected_names = {"os", "x"}

    assert res["var_count"] == 2
    assert set(res["bound_names"]) == expected_names

    assert res["func_count_all"] == 0
    assert res["func_count_toplevel"] == 0

    assert res["class_count_all"] == 0
    assert res["class_count_toplevel"] == 0


def test_analyze_definitions__case_with_class() -> None:
    code = """
import os

class MyClass:
    pass
"""
    res = analyze_definitions(code)

    expected_names = {"os", "MyClass"}

    assert res["var_count"] == 2
    assert set(res["bound_names"]) == expected_names

    assert res["func_count_all"] == 0
    assert res["func_count_toplevel"] == 0

    assert res["class_count_all"] == 1
    assert res["class_count_toplevel"] == 1


def test_analyze_definitions__case_with_class_and_function() -> None:
    code = """
import os

class MyClass:

    def my_func(self):
        pass
"""
    res = analyze_definitions(code)

    expected_names = {"os", "MyClass", "my_func", "self"}

    assert res["var_count"] == 4
    assert set(res["bound_names"]) == expected_names

    assert res["func_count_all"] == 1
    assert res["func_count_toplevel"] == 0

    assert res["class_count_all"] == 1
    assert res["class_count_toplevel"] == 1


def test_analyze_definitions__case_with_class_and_two_functions() -> None:
    code = """
import os

class MyClass:

    def my_func(self):
        pass

    def my_func2(self):
        pass
"""
    res = analyze_definitions(code)

    expected_names = {"os", "MyClass", "my_func", "self", "my_func2"}

    assert res["var_count"] == 5
    assert set(res["bound_names"]) == expected_names

    assert res["func_count_all"] == 2
    assert res["func_count_toplevel"] == 0

    assert res["class_count_all"] == 1
    assert res["class_count_toplevel"] == 1


def test_analyze_definitions__more_complex_case() -> None:
    code = """
import os as o, sys
x, (y, z) = 1, (2, 3)
for i in range(3): z += i
with open('f') as fh: data = fh.read()
def f(a, *, k=1, **kw):
    t = a + k
    return t
class C:
    def m(self):
        j = 0
        return j
match (1, 2):
    case (u, *rest):
        pass
"""
    res = analyze_definitions(code)

    expected_names = {
        "C",
        "a",
        "data",
        "f",
        "fh",
        "i",
        "j",
        "k",
        "kw",
        "m",
        "o",
        "rest",
        "self",
        "sys",
        "t",
        "u",
        "x",
        "y",
        "z",
    }

    assert res["var_count"] == 19
    assert set(res["bound_names"]) == expected_names

    assert res["func_count_all"] == 2
    assert res["func_count_toplevel"] == 1

    assert res["class_count_all"] == 1
    assert res["class_count_toplevel"] == 1
