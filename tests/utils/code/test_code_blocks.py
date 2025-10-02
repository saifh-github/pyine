import pytest

import pyine.utils.code.blocks as code_blocks


@pytest.fixture
def complex_code_sample() -> str:
    """A sample with various Python constructs."""
    return """\
def outer_function(param1, param2=None):     # L1
    '''Docstring for outer function.'''      # L2
    x = 10                                   # L3
                                             # L4
    if x > 0:                                # L5
        print("hello")                       # L6
    elif potato:                             # L7
        something()                          # L8
        while True:                          # L9
            print("potato123")               # L10
            if False:                        # L11
                break                        # L12
    else:                                    # L13
        print("something else")              # L14
        if something_else_again:             # L15
            return 123                       # L16
                                             # L17
    def inner_function():                    # L18
        '''Inner function docstring'''       # L19
        y = 5                                # L20
        while y > 0:                         # L21
            y = y-2 if y == 3 else y-1       # L22
        return y                             # L23
                                             # L24
    class InnerClass:                        # L25
        def some_method(self):               # L26
            try:                             # L27
                return param1 / param2       # L28
            except ZeroDivisionError:        # L29
                return None                  # L30
                                             # L31
    for i in range(x):                       # L32
        if i == 5:                           # L33
            break                            # L34
                                             # L35
    return inner_function()                  # L36
                                             # L37
potato = something()                         # L38
print("    hello!    ")                      # L39
"""


def test_identify_code_blocks_comprehensive(
    complex_code_sample: str,
) -> None:
    """Test block identification with various Python constructs."""
    blocks = code_blocks.identify_code_blocks(complex_code_sample)
    assert len(blocks) == 14
    assert blocks[1] == code_blocks.CodeBlock(
        type=code_blocks.BlockType.FUNCTION,
        name="outer_function",
        depth=0,
        parent_line=None,
        start_line=1,
        end_line=36,
    )
    assert blocks[5] == code_blocks.CodeBlock(
        type=code_blocks.BlockType.IF,
        name=None,
        depth=1,
        parent_line=1,
        start_line=5,
        end_line=16,
    )
    assert blocks[7] == code_blocks.CodeBlock(
        type=code_blocks.BlockType.IF,
        name=None,
        depth=2,
        parent_line=5,
        start_line=7,
        end_line=16,
    )
    assert blocks[9] == code_blocks.CodeBlock(
        type=code_blocks.BlockType.WHILE,
        name=None,
        depth=3,
        parent_line=7,
        start_line=9,
        end_line=12,
    )
    assert blocks[11] == code_blocks.CodeBlock(
        type=code_blocks.BlockType.IF,
        name=None,
        depth=4,
        parent_line=9,
        start_line=11,
        end_line=12,
    )
    assert blocks[15] == code_blocks.CodeBlock(
        type=code_blocks.BlockType.IF,
        name=None,
        depth=3,
        parent_line=7,
        start_line=15,
        end_line=16,
    )
    assert blocks[25] == code_blocks.CodeBlock(
        type=code_blocks.BlockType.CLASS,
        name="InnerClass",
        depth=1,
        parent_line=1,
        start_line=25,
        end_line=30,
    )
    assert blocks[26] == code_blocks.CodeBlock(
        type=code_blocks.BlockType.FUNCTION,
        name="some_method",
        depth=2,
        parent_line=25,
        start_line=26,
        end_line=30,
    )


def test_identify_code_blocks_empty_and_simple() -> None:
    """Test with empty code and simple structures."""
    assert len(code_blocks.identify_code_blocks("")) == 0
    assert len(code_blocks.identify_code_blocks("x = 10")) == 0
    blocks_map = code_blocks.identify_code_blocks("def simple():\n\treturn None\n\nsimple()")
    expected_def_line = 1
    assert len(blocks_map) == 1 and expected_def_line in blocks_map
    block = blocks_map[expected_def_line]
    assert isinstance(block, code_blocks.CodeBlock)
    assert block.name == "simple"
    assert block.type == code_blocks.BlockType.FUNCTION
    assert block.start_line == expected_def_line
    assert block.end_line == expected_def_line + 1
    assert block.depth == 0
    assert block.parent_line is None
