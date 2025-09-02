import pyine.utils.code.output_compare


def eq(a, b, **opts):
    return pyine.utils.code.output_compare.compare(
        a,
        b,
        pyine.utils.code.output_compare.CompareOptions(**opts),
    ).equal


def neq(a, b, **opts):
    r = pyine.utils.code.output_compare.compare(
        a,
        b,
        pyine.utils.code.output_compare.CompareOptions(**opts),
    )
    return not r.equal, r


def test_get_default_options():
    opts = pyine.utils.code.output_compare.get_options_for_code_exec_outputs()
    assert isinstance(opts, pyine.utils.code.output_compare.CompareOptions)
    assert eq("123", "123", **opts.model_dump())


# ---------------- numbers ----------------


def test_numbers_exact_int():
    assert eq(3, 3)


def test_numbers_float_with_tol():
    assert eq(0.3, 0.1 + 0.2, rel_tol=1e-12, abs_tol=0.0)


def test_numbers_float_not_close():
    assert not eq(1.0, 1.01, rel_tol=1e-4, abs_tol=0.0)


def test_nan_policy_equal():
    assert eq(float("nan"), float("nan"), nan_equal=True)


def test_nan_policy_not_equal_when_disabled():
    assert not eq(float("nan"), float("nan"), nan_equal=False)


def test_inf_equal():
    assert eq(float("inf"), float("inf"))
    assert eq(float("-inf"), float("-inf"))
    assert not eq(float("inf"), float("-inf"))


# ---------------- literal strings -> structures ----------------


def test_literal_numbers_from_str():
    assert eq("3", 3)
    assert eq("3.14", 3.14, rel_tol=1e-3)


def test_literal_list_equal():
    assert eq("[1, 2, 3]", [1, 2, 3])


def test_literal_tuple_vs_tuple():
    assert eq("(1, 2)", (1, 2))


def test_literal_set_order_irrelevant():
    assert eq("{1, 2, 3}", {3, 2, 1})


def test_literal_dict_order_irrelevant():
    a = "{'a': 1, 'b': 2}"
    b = "{'b': 2, 'a': 1}"
    assert eq(a, b)


def test_literal_nested_structures_with_floats():
    a = "{'a': [1.0, 2.0001], 'b': (3, 4)}"
    b = {"a": [1, 2.0001000001], "b": (3, 4)}
    assert eq(a, b, rel_tol=1e-6)


# ---------------- text comparisons with numeric tokens ----------------


def test_text_identical_after_ws_normalization():
    a = "result:   success\nvalue:   42"
    b = "result: success \n value: 42"
    assert eq(a, b)


def test_text_case_sensitive_default():
    a = "Status: OK"
    b = "status: ok"
    assert not eq(a, b)
    assert eq(a, b, case_sensitive=False)


def test_text_numeric_tokens_approx_equal():
    a = "pi ~= 3.14159"
    b = "pi ~= 3.1416"
    assert eq(a, b, rel_tol=1e-4, abs_tol=0.0)


def test_text_numeric_tokens_with_exponents():
    a = "val: 1.2e-3, other: -2E+5"
    b = "val: 0.0012, other: -200000"
    assert eq(a, b, rel_tol=1e-12)


def test_text_numeric_tokens_nan_inf():
    a = "got: NaN and +inf"
    b = "got: nan and inf"
    assert eq(a, b)


def test_text_numeric_tokens_mismatch_reports():
    a = "error rate: 0.12"
    b = "error rate: 0.10"
    ok, res = neq(a, b, rel_tol=1e-3)
    assert ok
    assert "Numbers differ" in res.reason or "num[" in res.path


# ---------------- mixed types: string literal vs object ----------------


def test_string_literal_vs_object_repr():
    # if string is not a literal and other is object, falls back to text compare with repr(other)
    class X:
        def __repr__(self):
            return "<X value=3.14>"

    x = X()
    assert eq("<X value=3.14>", x)


def test_string_literal_vs_list_object():
    assert eq("[1, 2, 3]", [1, 2, 3])


# ---------------- sequences: list/tuple order options ----------------


def test_lists_order_matters_default():
    assert not eq([1, 2, 3], [3, 2, 1])


def test_lists_order_not_matter_when_configured():
    assert eq([1, 2, 3], [3, 2, 1], list_order_matters=False)


def test_tuples_order_matters_default():
    assert not eq((1, 2), (2, 1))


def test_nested_sequences_with_order_option():
    a = [[1, 2], [3, 4]]
    b = [[3, 4], [1, 2]]
    # only top-level list is order-insensitive; inner lists compared with order
    assert eq(a, b, list_order_matters=False)


# ---------------- dicts and sets deep compare ----------------


def test_dict_values_with_float_tolerance():
    a = {"a": 1.0, "b": [2.0, 3.00001]}
    b = {"b": [2, 3.00002], "a": 1}
    assert eq(a, b, rel_tol=1e-5)


def test_set_of_tuples_unordered():
    a = {(1, 2), (3, 4)}
    b = {(3, 4), (1, 2)}
    assert eq(a, b)


# ---------------- edge cases ----------------


def test_string_vs_literal_none_and_bool():
    assert eq("None", None)
    assert eq("True", True)
    assert not eq("False", True)


def test_type_mismatch_reports():
    r = pyine.utils.code.output_compare.compare([1, 2], (1, 2))
    assert not r.equal
    assert "Type differs" in r.reason


def test_length_mismatch_in_sequences():
    r = pyine.utils.code.output_compare.compare([1, 2, 3], [1, 2])
    assert not r.equal
    assert "Length differs" in r.reason


def test_dict_key_difference():
    r = pyine.utils.code.output_compare.compare({"a": 1}, {"b": 1})
    assert not r.equal
    assert "Dict keys differ" in r.reason


def test_text_token_structure_diff():
    a = "result: 1 2"
    b = "result:"
    r = pyine.utils.code.output_compare.compare(a, b)
    assert not r.equal
    assert "Token structure differs" in r.reason


# ---------------- list/tuple type agnostic option ----------------


def test_list_tuple_type_agnostic_equality():
    assert eq([1, 2, 3], (1, 2, 3), array_type_matters=False)


def test_list_tuple_type_agnostic_nested():
    a = [1, (2, 3), [4, 5]]
    b = (1, [2, 3], (4, 5))
    assert eq(a, b, array_type_matters=False)


def test_list_tuple_type_agnostic_length_mismatch():
    r = pyine.utils.code.output_compare.compare(
        [1, 2, 3],
        (1, 2),
        pyine.utils.code.output_compare.CompareOptions(array_type_matters=False),
    )
    assert not r.equal
    assert "Length differs" in r.reason
