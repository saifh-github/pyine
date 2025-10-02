import typing

import pytest

import pyine.utils.code.args_mapper as args_mapper


def _call_with_mapped(fn: typing.Callable[..., typing.Any], inputs: typing.Any) -> typing.Any:
    args, kwargs = args_mapper.map_inputs_to_callable(fn, inputs)
    return fn(*args, **kwargs)


def test_simple_positional_and_kwargs_mapping() -> None:
    def fn(a: int, b: int) -> int:
        return a + b

    # dict -> kwargs
    args, kwargs = args_mapper.map_inputs_to_callable(fn, {"a": 1, "b": 2})
    assert args == ()
    assert kwargs == {"a": 1, "b": 2}
    assert fn(*args, **kwargs) == 3

    # sequence -> args
    args, kwargs = args_mapper.map_inputs_to_callable(fn, (1, 2))
    assert args == (1, 2)
    assert kwargs == {}
    assert fn(*args, **kwargs) == 3

    # string with kv
    args, kwargs = args_mapper.map_inputs_to_callable(fn, "a=2, b=4")
    assert args == ()
    assert kwargs == {"a": 2, "b": 4}
    assert fn(*args, **kwargs) == 6

    # string with positional
    args, kwargs = args_mapper.map_inputs_to_callable(fn, "1, 2")
    assert args == (1, 2)
    assert kwargs == {}
    assert fn(*args, **kwargs) == 3

    # json forms
    args, kwargs = args_mapper.map_inputs_to_callable(fn, '{"a": 5, "b": 6}')
    assert args == ()
    assert kwargs == {"a": 5, "b": 6}
    assert fn(*args, **kwargs) == 11
    args, kwargs = args_mapper.map_inputs_to_callable(fn, "[7, 8]")
    assert args == (7, 8)
    assert kwargs == {}

    # bad calls with missing/extra args
    with pytest.raises(ValueError):
        _ = args_mapper.map_inputs_to_callable(fn, "a=2, b=4, c=5")
    with pytest.raises(ValueError):
        _ = args_mapper.map_inputs_to_callable(fn, None)
    with pytest.raises(ValueError):
        _ = args_mapper.map_inputs_to_callable(fn, [1])
    with pytest.raises(ValueError):
        _ = args_mapper.map_inputs_to_callable(fn, [1, 2, 3])


def test_mapping_with_defaults() -> None:
    def fn(a: int, b: int = 0) -> int:
        return a + b

    args, kwargs = args_mapper.map_inputs_to_callable(fn, {"a": 1, "b": 2})
    assert fn(*args, **kwargs) == 3
    args, kwargs = args_mapper.map_inputs_to_callable(fn, {"a": 1})
    assert fn(*args, **kwargs) == 1
    args, kwargs = args_mapper.map_inputs_to_callable(fn, (1,))
    assert args == ((1,),)
    assert kwargs == {}
    args, kwargs = args_mapper.map_inputs_to_callable(fn, [1])
    assert args == ([1],)
    assert kwargs == {}
    with pytest.raises(ValueError):
        _ = args_mapper.map_inputs_to_callable(fn, None)


def test_querystring_and_colon_syntax() -> None:
    def fn(a: int, b: int) -> int:
        return a * b

    args, kwargs = args_mapper.map_inputs_to_callable(fn, "a=3,b=4")
    assert args == ()
    assert kwargs == {"a": 3, "b": 4}
    assert fn(*args, **kwargs) == 12
    args, kwargs = args_mapper.map_inputs_to_callable(fn, "a = 3, b = 4")
    assert fn(*args, **kwargs) == 12

    args, kwargs = args_mapper.map_inputs_to_callable(fn, "a: 2, b: 5")
    assert args == ()
    assert kwargs == {"a": 2, "b": 5}
    assert fn(*args, **kwargs) == 10
    args, kwargs = args_mapper.map_inputs_to_callable(fn, "a:2,b:5")
    assert fn(*args, **kwargs) == 10

    with pytest.raises(ValueError):
        _ = args_mapper.map_inputs_to_callable(fn, "a:2,b:5,c:6")
    with pytest.raises(ValueError):
        _ = args_mapper.map_inputs_to_callable(fn, "a:2,b:5,potato")
    with pytest.raises(ValueError):
        _ = args_mapper.map_inputs_to_callable(fn, "a:2,potato")


def test_parenthesized_and_spaces() -> None:
    def fn(a: int, b: int) -> int:
        return a - b

    args, kwargs = args_mapper.map_inputs_to_callable(fn, "( 10 ,  3 )")
    assert args == (10, 3)
    assert kwargs == {}
    assert fn(*args, **kwargs) == 7


def test_positional_only_and_keyword_only() -> None:
    def fn(
        a: int,
        /,
        b: int,
        *,
        c: int = 0,
    ) -> int:  # a positional-only, c keyword-only
        return a + b + c

    # mapping provides a and b; a should be moved to args
    args, kwargs = args_mapper.map_inputs_to_callable(fn, {"a": 1, "b": 2, "c": 3})
    assert args == (1,)
    assert kwargs == {"b": 2, "c": 3}
    assert fn(*args, **kwargs) == 6

    # string positional
    args, kwargs = args_mapper.map_inputs_to_callable(fn, "1, 2, c=5")
    assert args == (1, 2)
    assert kwargs == {"c": 5}
    assert fn(*args, **kwargs) == 8


def test_varargs_and_varkwargs_passthrough() -> None:
    def fn(
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> tuple[tuple[typing.Any, ...], dict[str, typing.Any]]:
        return args, kwargs

    # ensure mapping not filtered if **kwargs present
    args, kwargs = args_mapper.map_inputs_to_callable(fn, {"x": 1, "y": 2})
    assert args == ()
    assert kwargs == {"x": 1, "y": 2}

    # sequence kept as a single positional argument if it binds
    args, kwargs = args_mapper.map_inputs_to_callable(fn, [1, 2, 3])
    assert args == ([1, 2, 3],)
    assert kwargs == {}


def test_scalar_and_none_inputs() -> None:
    def one(x: typing.Any) -> typing.Any:
        return x

    # scalar maps to single positional
    assert _call_with_mapped(one, 5) == 5
    assert _call_with_mapped(one, "null") is None
    assert _call_with_mapped(one, "true") is True

    # string scalar should parse to positional
    args, kwargs = args_mapper.map_inputs_to_callable(one, "5")
    assert args == (5,)
    assert kwargs == {}

    # none with defaults should do nothing
    def with_defaults(
        a: int = 1,
        b: int = 2,
    ) -> int:
        return a + b

    args, kwargs = args_mapper.map_inputs_to_callable(with_defaults, None)
    assert args == ()
    assert kwargs == {}
    assert with_defaults(*args, **kwargs) == 3


def test_object_attribute_mapping_and_filtering() -> None:
    class Obj:
        def __init__(self) -> None:
            self.a = 10
            self.b = 20
            self._private = 99

    def fn(
        a: int,
        b: int,
    ) -> int:
        return a * b

    obj = Obj()
    args, kwargs = args_mapper.map_inputs_to_callable(fn, obj)
    assert args == ()
    assert kwargs == {"a": 10, "b": 20}
    assert fn(*args, **kwargs) == 200


def test_nested_structures_in_string() -> None:
    def fn(
        a: list[int],
        b: dict[str, int],
    ) -> int:
        return a[0] + b["x"]

    s = "a=[1,2,3], b={'x': 5}"
    args, kwargs = args_mapper.map_inputs_to_callable(fn, s)
    assert args == ()
    assert kwargs == {"a": [1, 2, 3], "b": {"x": 5}}
    assert fn(*args, **kwargs) == 6
