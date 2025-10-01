import ast
import dataclasses
import enum
import typing


class BlockType(enum.StrEnum):
    """Enumeration of all supported block types for analysis.

    Note: we ignore lambdas, list comprehensions, and generator expressions.
    """

    IF = "if"
    FOR = "for"
    WHILE = "while"
    TRY = "try"
    EXCEPT = "except"
    FUNCTION = "function"
    CLASS = "class"
    WITH = "with"
    ASYNC_FOR = "asyncfor"
    ASYNC_WITH = "asyncwith"
    ASYNC_FUNCTION = "asyncfunction"


_ast_node_type_mapping = {
    ast.If: BlockType.IF,
    ast.For: BlockType.FOR,
    ast.While: BlockType.WHILE,
    ast.Try: BlockType.TRY,
    ast.ExceptHandler: BlockType.EXCEPT,
    ast.FunctionDef: BlockType.FUNCTION,
    ast.ClassDef: BlockType.CLASS,
    ast.With: BlockType.WITH,
    ast.AsyncFor: BlockType.ASYNC_FOR,
    ast.AsyncWith: BlockType.ASYNC_WITH,
    ast.AsyncFunctionDef: BlockType.ASYNC_FUNCTION,
}


@dataclasses.dataclass(frozen=True)
class CodeBlock:
    """Information about a logical block in a Python code snippet."""

    type: BlockType
    """The type of block (see BlockType enum)."""
    name: str | None
    """The name of the block (if applicable, for classes/functions)."""
    depth: int
    """The nesting depth of the block (0 for top-level blocks)."""
    parent_line: int | None
    """Line number of the parent block (if any)."""
    start_line: int
    """Line number of the start of the block."""
    end_line: int
    """Line number of the end of the block."""


def identify_code_blocks(
    code_string: str,
) -> dict[int, CodeBlock]:
    """Analyze a Python code string and identify logical flow blocks.

    This function parses the code string using the AST module and identifies
    control flow structures such as if/else blocks, loops, and function/class
    definitions. It returns a mapping of line numbers to information about
    the blocks at those lines, including block type and nesting depth.
    """
    code_blocks = _capture_child_nodes(ast.parse(code_string))
    blocks_map = {}
    for block in code_blocks:
        assert block.start_line not in blocks_map, (
            f"found multiple blocks on same line! see L{block.start_line} in:\n{code_string}"
        )
        blocks_map[block.start_line] = block
    return dict(sorted(blocks_map.items()))


def _capture_child_nodes(
    node: ast.AST,
    results: list[dict[str, typing.Any]] | None = None,
    parent_line: int | None = None,
    next_depth: int | None = None,
) -> list[CodeBlock]:
    if results is None:
        results = []
    node_type = _ast_node_type_mapping.get(type(node))
    if node_type and node_type in {t.value for t in BlockType}:
        if next_depth is None:
            next_depth = 0
        node_info = CodeBlock(
            type=node_type,
            name=getattr(node, "name", None),
            depth=next_depth,
            parent_line=parent_line,
            start_line=getattr(node, "lineno", None),
            end_line=getattr(node, "end_lineno", None),
        )
        results.append(node_info)
    for child in ast.iter_child_nodes(node):
        _capture_child_nodes(
            node=child,
            results=results,
            parent_line=getattr(node, "lineno", None),
            next_depth=next_depth + 1 if next_depth is not None else None,
        )
    return results


if __name__ == "__main__":
    _example_code = """\
class Example:  # L 1
    def __init__(self):  # L 2
        self.name = "hello"  # L 3
        if True:  # L 4
            print("true")  # L 5
            while False:  # L 6
                print("false")  # L 7
        elif False:  # L 8
            print("false")  # L 9
            for i in range(5):  # L 10
                print(i)  # L 11
            def inner_method():  # L 12
                print("inner")  # L 13
        else:  # L 14
            print("else")  # L 15
            class InnerClass:  # L 16
                def __init__(self):  # L 17
                    self.name = "inner"  # L 18
                def inner_method(self):  # L 19
                    print("inner")  # L 20
                    while True:  # L 21
                        pass  # L 22

    def example_method(self):  # L 24
        if True:  # L 25
            for i in range(5):  # L 26
                print(i)  # L 27
        else:  # L 28
            while False:  # L 29
                pass  # L 30

example = Example()  # L 32
print(example.name)  # L 33
example.example_method()  # L 34
"""
    _blocks_map = identify_code_blocks(_example_code)
    for _line, _block in sorted(_blocks_map.items()):
        print(f"Line {_line}: {_block}")
