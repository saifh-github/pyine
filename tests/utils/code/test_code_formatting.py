import pyine.utils.code.formatting


def test_reformatting():
    unformatted_code = "def my_func( a,b ):\n  return a+b"
    expected_formatted_code = "def my_func(a, b):\n    return a + b\n"
    actual_formatted_code = pyine.utils.code.formatting.format_code(unformatted_code)
    assert actual_formatted_code == expected_formatted_code
