"""Tests for step-parsing utilities in pyine.utils.parsing.

Covers: strip_line_number_prefixes, get_executable_line_numbers, tokenize_code_identifiers,
ParsedStep, StepsValidationReport, and parse_and_validate_steps_block.
"""

import pytest

import pyine.utils.parsing


class TestStripLineNumberPrefixes:
    def test_canonical_format(self) -> None:
        code = "1: def foo():\n2:     return 42"
        result = pyine.utils.parsing.strip_line_number_prefixes(code)
        assert result.stripped == "def foo():\n    return 42"
        assert result.lines_matched == 2
        assert result.total_lines == 2

    def test_space_padded_format(self) -> None:
        code = "  1: x = 1\n  2: y = 2\n  3: z = x + y"
        result = pyine.utils.parsing.strip_line_number_prefixes(code)
        assert result.stripped == "x = 1\ny = 2\nz = x + y"
        assert result.lines_matched == 3

    def test_mixed_width_numbers(self) -> None:
        code = " 1: a\n10: b"
        result = pyine.utils.parsing.strip_line_number_prefixes(code)
        assert result.stripped == "a\nb"
        assert result.lines_matched == 2

    def test_three_digit_width(self) -> None:
        code = "  1: first\n 10: tenth\n100: hundredth"
        result = pyine.utils.parsing.strip_line_number_prefixes(code)
        assert result.stripped == "first\ntenth\nhundredth"
        assert result.lines_matched == 3

    def test_no_op_on_unprefixed_code(self) -> None:
        code = "def foo():\n    return 42"
        result = pyine.utils.parsing.strip_line_number_prefixes(code)
        assert result.stripped == code
        assert result.lines_matched == 0
        assert result.total_lines == 2

    def test_blank_lines_preserved(self) -> None:
        code = "1: x = 1\n\n3: y = 2"
        result = pyine.utils.parsing.strip_line_number_prefixes(code)
        assert result.stripped == "x = 1\n\ny = 2"
        assert result.lines_matched == 2
        assert result.total_lines == 3

    def test_empty_string(self) -> None:
        result = pyine.utils.parsing.strip_line_number_prefixes("")
        assert result.stripped == ""
        assert result.lines_matched == 0
        assert result.total_lines == 0

    def test_old_l_num_pipe_format_not_stripped(self) -> None:
        code = "L1|x = 1\nL2|y = 2"
        result = pyine.utils.parsing.strip_line_number_prefixes(code)
        assert result.stripped == code
        assert result.lines_matched == 0

    def test_old_num_pipe_format_not_stripped(self) -> None:
        code = "1|x = 1\n2|y = 2"
        result = pyine.utils.parsing.strip_line_number_prefixes(code)
        assert result.stripped == code
        assert result.lines_matched == 0


class TestGetExecutableLineNumbers:
    def test_basic_code(self) -> None:
        code = "x = 1\ny = 2\nz = x + y"
        result = pyine.utils.parsing.get_executable_line_numbers(code)
        assert result == frozenset({1, 2, 3})

    def test_comments_excluded(self) -> None:
        code = "# comment\nx = 1\n  # indented comment\ny = 2"
        result = pyine.utils.parsing.get_executable_line_numbers(code)
        assert result == frozenset({2, 4})

    def test_blank_lines_excluded(self) -> None:
        code = "x = 1\n\ny = 2\n   \nz = 3"
        result = pyine.utils.parsing.get_executable_line_numbers(code)
        assert result == frozenset({1, 3, 5})

    def test_decorators_excluded(self) -> None:
        code = "@property\ndef foo(self):\n    return 1"
        result = pyine.utils.parsing.get_executable_line_numbers(code)
        assert result == frozenset({2, 3})

    def test_empty_code(self) -> None:
        result = pyine.utils.parsing.get_executable_line_numbers("")
        assert result == frozenset()

    def test_module_docstring_excluded(self) -> None:
        code = '"""Module docstring."""\nx = 1\ny = 2'
        result = pyine.utils.parsing.get_executable_line_numbers(code)
        assert result == frozenset({2, 3})

    def test_multiline_docstring_excluded(self) -> None:
        code = 'def foo():\n    """Multi-line\n    docstring with x = 1.\n    """\n    return 1'
        result = pyine.utils.parsing.get_executable_line_numbers(code)
        assert result == frozenset({1, 5})  # def and return only

    def test_class_docstring_excluded(self) -> None:
        code = "class Foo:\n    \"\"\"Class doc.\"\"\"\n    x = 1"
        result = pyine.utils.parsing.get_executable_line_numbers(code)
        assert result == frozenset({1, 3})

    def test_docstring_with_code_examples_excluded(self) -> None:
        code = (
            "def compute(x):\n"
            '    """Compute result.\n'
            "\n"
            "    Example:\n"
            "        >>> x = 1\n"
            "        >>> y = x + 2\n"
            '    """\n'
            "    return x + 1"
        )
        result = pyine.utils.parsing.get_executable_line_numbers(code)
        # only 'def compute(x):' (line 1) and 'return x + 1' (line 8)
        assert result == frozenset({1, 8})

    def test_string_literal_not_docstring_kept(self) -> None:
        # a string expression that is NOT the first statement -> not a docstring
        code = "x = 1\n\"not a docstring\"\ny = 2"
        result = pyine.utils.parsing.get_executable_line_numbers(code)
        assert 2 in result  # the string literal line is still "executable"

    def test_malformed_code_raises(self) -> None:
        code = "def foo(\nx = 1\ny = 2"  # syntax error
        with pytest.raises(SyntaxError):
            pyine.utils.parsing.get_executable_line_numbers(code)

    def test_multiline_assignment_includes_continuation_lines(self) -> None:
        code = "result = (\n    1 +\n    2\n)\nprint(result)"
        result = pyine.utils.parsing.get_executable_line_numbers(code)
        assert result == frozenset({1, 2, 3, 4, 5})

    def test_compound_header_spans_but_internal_comment_stays_excluded(self) -> None:
        code = "def foo(\n    x,\n):\n    y = x\n    # comment in body\n    return y"
        result = pyine.utils.parsing.get_executable_line_numbers(code)
        assert result == frozenset({1, 2, 3, 4, 6})


class TestTokenizeCodeIdentifiers:
    def test_basic_tokens(self) -> None:
        result = pyine.utils.parsing.tokenize_code_identifiers("x = foo(bar, baz)")
        assert "x" in result
        assert "foo" in result
        assert "bar" in result
        assert "baz" in result

    def test_lowercased(self) -> None:
        result = pyine.utils.parsing.tokenize_code_identifiers("MyClass.method()")
        assert "myclass" in result
        assert "method" in result

    def test_numeric_tokens_kept(self) -> None:
        result = pyine.utils.parsing.tokenize_code_identifiers("x = 42")
        assert "42" in result
        assert "x" in result

    def test_empty_string(self) -> None:
        result = pyine.utils.parsing.tokenize_code_identifiers("")
        assert result == frozenset()

    def test_punctuation_only_dropped(self) -> None:
        result = pyine.utils.parsing.tokenize_code_identifiers("+ - * /")
        assert result == frozenset()

    def test_snake_case_identifiers(self) -> None:
        result = pyine.utils.parsing.tokenize_code_identifiers("my_var = some_func(other_arg)")
        assert "my_var" in result
        assert "some_func" in result
        assert "other_arg" in result

    def test_private_identifiers(self) -> None:
        result = pyine.utils.parsing.tokenize_code_identifiers("_private = __dunder__")
        assert "_private" in result
        assert "__dunder__" in result

    def test_mixed_identifiers_and_numbers(self) -> None:
        result = pyine.utils.parsing.tokenize_code_identifiers("item_count = len(data_list) + 10")
        assert "item_count" in result
        assert "len" in result
        assert "data_list" in result
        assert "10" in result


class TestParseAndValidateStepsBlock:
    _SIMPLE_CODE = "x = 1\ny = 2\nz = x + y"
    _EXECUTABLE_LINES = frozenset({1, 2, 3})

    def test_valid_jsonl(self) -> None:
        output = (
            'Some reasoning\n'
            '<steps>\n'
            '{"step": 1, "line": 1, "text": "Assign x = 1"}\n'
            '{"step": 2, "line": 2, "text": "Assign y = 2"}\n'
            '{"step": 3, "line": 3, "text": "Compute z = x + y"}\n'
            '</steps>\n'
            '<final>6</final>'
        )
        report = pyine.utils.parsing.parse_and_validate_steps_block(
            output,
            code_string=self._SIMPLE_CODE,
            executable_lines=self._EXECUTABLE_LINES,
        )
        assert report.raw_block is not None
        assert report.num_parsed == 3
        assert report.num_valid == 3
        assert report.step_monotonic_ok is True
        assert report.step_contiguous is True
        assert report.line_in_range_count == 3
        assert report.multiple_blocks_found is False

    def test_missing_block(self) -> None:
        report = pyine.utils.parsing.parse_and_validate_steps_block(
            "No steps here\n<final>42</final>",
            code_string=self._SIMPLE_CODE,
        )
        assert report.raw_block is None
        assert report.num_parsed == 0
        assert report.num_valid == 0
        assert report.step_monotonic_ok is False
        assert report.step_contiguous is False
        assert report.too_few_steps is False  # min_steps not set
        assert report.multiple_blocks_found is False

    def test_missing_block_with_min_steps(self) -> None:
        report = pyine.utils.parsing.parse_and_validate_steps_block(
            "No steps here",
            code_string=self._SIMPLE_CODE,
            min_steps=1,
        )
        assert report.too_few_steps is True

    def test_empty_block(self) -> None:
        report = pyine.utils.parsing.parse_and_validate_steps_block(
            "<steps>\n</steps>",
            code_string=self._SIMPLE_CODE,
        )
        assert report.raw_block is not None
        assert report.num_parsed == 0
        assert report.num_lines_in_block == 0

    def test_partial_jsonl(self) -> None:
        output = (
            '<steps>\n'
            '{"step": 1, "line": 1, "text": "ok"}\n'
            'not valid json\n'
            '{"step": 2, "line": 2, "text": "also ok"}\n'
            '</steps>'
        )
        report = pyine.utils.parsing.parse_and_validate_steps_block(
            output,
            code_string=self._SIMPLE_CODE,
            executable_lines=self._EXECUTABLE_LINES,
        )
        assert report.num_parsed == 2
        assert report.num_lines_in_block == 3
        assert len(report.parse_errors) == 1

    def test_missing_keys(self) -> None:
        output = '<steps>\n{"step": 1, "line": 1}\n</steps>'
        report = pyine.utils.parsing.parse_and_validate_steps_block(
            output,
            code_string=self._SIMPLE_CODE,
        )
        assert report.num_parsed == 0
        assert len(report.parse_errors) == 1
        assert "missing keys" in report.parse_errors[0]

    def test_wrong_types(self) -> None:
        output = '<steps>\n{"step": "one", "line": 1, "text": "ok"}\n</steps>'
        report = pyine.utils.parsing.parse_and_validate_steps_block(
            output,
            code_string=self._SIMPLE_CODE,
        )
        assert report.num_parsed == 0
        assert len(report.parse_errors) == 1
        assert "'step' must be int" in report.parse_errors[0]

    def test_boolean_rejected_as_int(self) -> None:
        output = '<steps>\n{"step": true, "line": 1, "text": "ok"}\n</steps>'
        report = pyine.utils.parsing.parse_and_validate_steps_block(
            output,
            code_string=self._SIMPLE_CODE,
        )
        assert report.num_parsed == 0

    def test_line_out_of_range(self) -> None:
        output = '<steps>\n{"step": 1, "line": 99, "text": "out of range"}\n</steps>'
        report = pyine.utils.parsing.parse_and_validate_steps_block(
            output,
            code_string=self._SIMPLE_CODE,
            executable_lines=self._EXECUTABLE_LINES,
        )
        assert report.num_parsed == 1
        assert report.num_valid == 0  # line 99 is out of range
        assert report.line_in_range_count == 0

    def test_non_executable_line(self) -> None:
        code = "# comment\nx = 1"
        output = '<steps>\n{"step": 1, "line": 1, "text": "comment line"}\n</steps>'
        report = pyine.utils.parsing.parse_and_validate_steps_block(
            output,
            code_string=code,
        )
        assert report.num_parsed == 1
        assert report.num_valid == 0  # line 1 is a comment

    def test_monotonicity_violation(self) -> None:
        output = '<steps>\n{"step": 2, "line": 1, "text": "a"}\n{"step": 1, "line": 2, "text": "b"}\n</steps>'
        report = pyine.utils.parsing.parse_and_validate_steps_block(
            output,
            code_string=self._SIMPLE_CODE,
            executable_lines=self._EXECUTABLE_LINES,
        )
        assert report.step_monotonic_ok is False

    def test_contiguity_violation(self) -> None:
        output = '<steps>\n{"step": 1, "line": 1, "text": "a"}\n{"step": 3, "line": 2, "text": "b"}\n</steps>'
        report = pyine.utils.parsing.parse_and_validate_steps_block(
            output,
            code_string=self._SIMPLE_CODE,
            executable_lines=self._EXECUTABLE_LINES,
        )
        assert report.step_monotonic_ok is True  # 1 < 3
        assert report.step_contiguous is False  # gap: 1, 3

    def test_multiple_blocks_first_policy(self) -> None:
        output = (
            '<steps>\n{"step": 1, "line": 1, "text": "first block"}\n</steps>\n'
            '<steps>\n{"step": 1, "line": 2, "text": "second block"}\n</steps>'
        )
        report = pyine.utils.parsing.parse_and_validate_steps_block(
            output,
            code_string=self._SIMPLE_CODE,
            executable_lines=self._EXECUTABLE_LINES,
            multi_block_policy="first",
        )
        assert report.multiple_blocks_found is True
        assert report.num_parsed == 1
        assert report.parsed_steps[0].text == "first block"

    def test_multiple_blocks_error_policy(self) -> None:
        output = (
            '<steps>\n{"step": 1, "line": 1, "text": "a"}\n</steps>\n'
            '<steps>\n{"step": 1, "line": 2, "text": "b"}\n</steps>'
        )
        with pytest.raises(ValueError, match="expected exactly one"):
            pyine.utils.parsing.parse_and_validate_steps_block(
                output,
                code_string=self._SIMPLE_CODE,
                multi_block_policy="error",
            )

    def test_zero_valid_steps_ratios(self) -> None:
        output = '<steps>\n{"step": 1, "line": 99, "text": "invalid"}\n</steps>'
        report = pyine.utils.parsing.parse_and_validate_steps_block(
            output,
            code_string=self._SIMPLE_CODE,
            executable_lines=self._EXECUTABLE_LINES,
        )
        assert report.num_valid == 0
        assert report.line_in_range_ratio == 0.0
        assert report.executable_line_hit_ratio == 0.0

    def test_too_few_and_too_many_steps(self) -> None:
        output = '<steps>\n{"step": 1, "line": 1, "text": "a"}\n</steps>'
        report = pyine.utils.parsing.parse_and_validate_steps_block(
            output,
            code_string=self._SIMPLE_CODE,
            executable_lines=self._EXECUTABLE_LINES,
            min_steps=5,
            max_steps=0,
        )
        assert report.too_few_steps is True
        assert report.too_many_steps is True

    def test_trailing_whitespace_in_block(self) -> None:
        output = '<steps>\n  {"step": 1, "line": 1, "text": "a"}  \n</steps>'
        report = pyine.utils.parsing.parse_and_validate_steps_block(
            output,
            code_string=self._SIMPLE_CODE,
            executable_lines=self._EXECUTABLE_LINES,
        )
        assert report.num_parsed == 1

    def test_ratios_bounded_zero_to_one(self) -> None:
        """Ratios are always in [0.0, 1.0] even when num_valid < num_parsed."""
        code = "# comment\nx = 1"
        executable = frozenset({2})
        output = (
            "<steps>\n"
            '{"step": 1, "line": 1, "text": "comment line"}\n'
            '{"step": 2, "line": 2, "text": "x = 1"}\n'
            "</steps>"
        )
        report = pyine.utils.parsing.parse_and_validate_steps_block(
            output,
            code_string=code,
            executable_lines=executable,
        )
        assert report.num_parsed == 2
        assert report.num_valid == 1  # only line 2 passes both checks
        assert 0.0 <= report.line_in_range_ratio <= 1.0
        assert 0.0 <= report.executable_line_hit_ratio <= 1.0
