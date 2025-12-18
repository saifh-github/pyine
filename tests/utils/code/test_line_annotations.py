import pytest

import pyine.utils.code.line_annotations as line_annotations


class TestGetCodeWithNumberedLines:
    def test_default_format_auto_width(self) -> None:
        code = "a = 1\nprint(a)"
        formatted = line_annotations.get_code_with_numbered_lines(code)
        lines = formatted.splitlines()
        assert lines[0] == "L1|a = 1"
        assert lines[1] == "L2|print(a)"

    def test_prefixed_tabs(self) -> None:
        code = "a = 1\nprint(a)"
        formatted = line_annotations.get_code_with_numbered_lines(code, prefixed_tabs=2)
        lines = formatted.splitlines()
        assert lines[0] == "\t\tL1|a = 1"
        assert lines[1] == "\t\tL2|print(a)"

    def test_explicit_num_width_zero_pad(self) -> None:
        code = "a = 1\nb = 2"
        formatted = line_annotations.get_code_with_numbered_lines(code, num_width=4)
        lines = formatted.splitlines()
        assert lines[0] == "L0001|a = 1"
        assert lines[1] == "L0002|b = 2"

    def test_explicit_num_width_space_pad(self) -> None:
        code = "a = 1\nb = 2"
        formatted = line_annotations.get_code_with_numbered_lines(code, num_width=4, zero_pad=False)
        lines = formatted.splitlines()
        assert lines[0] == "L   1|a = 1"
        assert lines[1] == "L   2|b = 2"

    def test_custom_prefix_pattern_pipe(self) -> None:
        code = "x = 1\ny = 2\nz = 3"
        formatted = line_annotations.get_code_with_numbered_lines(code, prefix_pattern="{num}|")
        lines = formatted.splitlines()
        assert lines[0] == "1|x = 1"
        assert lines[1] == "2|y = 2"
        assert lines[2] == "3|z = 3"

    def test_custom_prefix_pattern_with_line_num_and_pipe(self) -> None:
        code = "a = 1\nb = 2"
        formatted = line_annotations.get_code_with_numbered_lines(
            code,
            prefix_pattern="L{num}|",
            num_width=3,
            zero_pad=False,
        )
        lines = formatted.splitlines()
        assert lines[0] == "L  1|a = 1"
        assert lines[1] == "L  2|b = 2"

    def test_auto_width_scales_with_line_count(self) -> None:
        code = "\n".join(f"line{idx}" for idx in range(100))  # 100 lines -> width 3
        formatted = line_annotations.get_code_with_numbered_lines(code, prefix_pattern="{num}|")
        lines = formatted.splitlines()
        assert lines[0] == "001|line0"
        assert lines[9] == "010|line9"
        assert lines[99] == "100|line99"


class TestGetCodeWithBlockMarkers:
    def test_basic_block_markers(self) -> None:
        code = "a = 1\nb = 2\nc = 3"
        result = line_annotations.get_code_with_block_markers(code, start_line=1, end_line=2)
        lines = result.splitlines()
        assert lines[0] == "a = 1  # <<<< START HERE"
        assert lines[1] == "b = 2  # <<<< END HERE"
        assert lines[2] == "c = 3"

    def test_single_line_block(self) -> None:
        code = "a = 1\nb = 2\nc = 3"
        result = line_annotations.get_code_with_block_markers(code, start_line=2, end_line=2)
        lines = result.splitlines()
        assert lines[0] == "a = 1"
        assert lines[1] == "b = 2  # <<<< START HERE"
        assert lines[2] == "c = 3"

    def test_custom_suffixes(self) -> None:
        code = "x = 1\ny = 2"
        result = line_annotations.get_code_with_block_markers(
            code,
            start_line=1,
            end_line=2,
            start_suffix="  # BEGIN",
            end_suffix="  # END",
        )
        lines = result.splitlines()
        assert lines[0] == "x = 1  # BEGIN"
        assert lines[1] == "y = 2  # END"

    def test_full_file_block(self) -> None:
        code = "line1\nline2\nline3"
        result = line_annotations.get_code_with_block_markers(code, start_line=1, end_line=3)
        lines = result.splitlines()
        assert lines[0] == "line1  # <<<< START HERE"
        assert lines[1] == "line2"
        assert lines[2] == "line3  # <<<< END HERE"

    def test_start_greater_than_end_raises(self) -> None:
        code = "a = 1\nb = 2"
        with pytest.raises(ValueError, match="start_line.*must be <= end_line"):
            line_annotations.get_code_with_block_markers(code, start_line=2, end_line=1)

    def test_start_line_out_of_bounds_raises(self) -> None:
        code = "a = 1\nb = 2"
        with pytest.raises(ValueError, match="start_line.*out of bounds"):
            line_annotations.get_code_with_block_markers(code, start_line=0, end_line=1)
        with pytest.raises(ValueError, match="start_line.*out of bounds"):
            line_annotations.get_code_with_block_markers(code, start_line=3, end_line=3)

    def test_end_line_out_of_bounds_raises(self) -> None:
        code = "a = 1\nb = 2"
        with pytest.raises(ValueError, match="end_line.*out of bounds"):
            line_annotations.get_code_with_block_markers(code, start_line=1, end_line=3)
