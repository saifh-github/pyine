import ast
import typing

__all__ = [
    "AnalyzeDefsResult",
    "analyze_defs",
]


class _DefinitionCounter(ast.NodeVisitor):
    """AST visitor that counts bound variable names and class/function definitions.

    This visitor records names that are bound by various Python statements and constructs (e.g.,
    assignments, imports, with/as, for loops, walrus operator, pattern matching, function and class
    definitions). It also keeps track of function and class definition counts, and of how many occur
    at the module (top) level.

    Attributes:
        bound_names: The set of variable names that are bound anywhere in the AST.
        func_defs_all: Count of all function definitions encountered.
        func_defs_toplevel: Count of function definitions at module level.
        class_defs_all: Count of all class definitions encountered.
        class_defs_toplevel: Count of class definitions at module level.
    """

    def __init__(self) -> None:
        self.bound_names: set[str] = set()
        self.func_defs_all: int = 0
        self.func_defs_toplevel: int = 0
        self.class_defs_all: int = 0
        self.class_defs_toplevel: int = 0
        self._nesting: int = 0  # 0 = module level

    # ---------- helpers ----------

    def _add_target(
        self,
        target: ast.AST,
    ) -> None:
        """Add names that are variables being bound by an assignment-like target.

        Args:
            target: The AST node acting as an assignment target.
        """
        # note: Attribute, Subscript, etc. do NOT bind new variables, so ignore those
        if isinstance(target, ast.Name):
            self.bound_names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._add_target(elt)
        elif isinstance(target, ast.Starred):
            self._add_target(target.value)

    def _add_params(
        self,
        args: ast.arguments,
    ) -> None:
        """Add parameter names from a function/method signature."""
        for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
            self.bound_names.add(arg.arg)
        if args.vararg:
            self.bound_names.add(args.vararg.arg)
        if args.kwarg:
            self.bound_names.add(args.kwarg.arg)

    def _add_match_pattern(
        self,
        pattern: ast.AST | None,
    ) -> None:
        """Add names bound by structural pattern matching (Python 3.10+).

        The following are examples of bound names handled here:
        - case x: -> MatchAs(name="x")
        - case *xs: -> MatchStar(name="xs")
        """
        if isinstance(pattern, ast.MatchAs) and pattern.name:
            self.bound_names.add(pattern.name)
            if pattern.pattern:
                self._add_match_pattern(pattern.pattern)
        elif isinstance(pattern, ast.MatchStar) and pattern.name:
            self.bound_names.add(pattern.name)
        elif isinstance(pattern, ast.MatchMapping):
            for pat in pattern.patterns:
                self._add_match_pattern(pat)
            if pattern.rest:
                self.bound_names.add(pattern.rest)
        elif isinstance(pattern, (ast.MatchClass, ast.MatchOr, ast.MatchSequence)):
            for pat in getattr(pattern, "patterns", []) or []:
                self._add_match_pattern(pat)
            for pat in getattr(pattern, "kwd_patterns", []) or []:
                self._add_match_pattern(pat)

    # ---------- statements that bind ----------

    def visit_Assign(
        self,
        node: ast.Assign,
    ) -> None:
        for target in node.targets:
            self._add_target(target)
        self.generic_visit(node)

    def visit_AnnAssign(
        self,
        node: ast.AnnAssign,
    ) -> None:
        self._add_target(node.target)
        self.generic_visit(node)

    def visit_AugAssign(
        self,
        node: ast.AugAssign,
    ) -> None:
        self._add_target(node.target)
        self.generic_visit(node)

    def visit_For(
        self,
        node: ast.For,
    ) -> None:
        self._add_target(node.target)
        self.generic_visit(node)

    visit_AsyncFor = visit_For

    def visit_With(
        self,
        node: ast.With,
    ) -> None:
        for item in node.items:
            if item.optional_vars:
                self._add_target(item.optional_vars)
        self.generic_visit(node)

    visit_AsyncWith = visit_With

    def visit_ExceptHandler(
        self,
        node: ast.ExceptHandler,
    ) -> None:
        if node.name:
            self.bound_names.add(node.name)
        self.generic_visit(node)

    def visit_comprehension(
        self,
        node: ast.comprehension,
    ) -> None:
        self._add_target(node.target)
        self.generic_visit(node)

    def visit_NamedExpr(
        self,
        node: ast.NamedExpr,
    ) -> None:
        # x := ...
        self._add_target(node.target)
        self.generic_visit(node)

    def visit_Match(
        self,
        node: ast.Match,
    ) -> None:
        for case in node.cases:
            self._add_match_pattern(case.pattern)
        self.generic_visit(node)

    def visit_Import(
        self,
        node: ast.Import,
    ) -> None:
        for alias in node.names:
            # import numpy as np -> 'np'; import numpy -> 'numpy'
            self.bound_names.add(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(
        self,
        node: ast.ImportFrom,
    ) -> None:
        for alias in node.names:
            if alias.name != "*":
                self.bound_names.add(alias.asname or alias.name)

    # ---------- defs (also count) ----------

    def visit_FunctionDef(
        self,
        node: ast.FunctionDef,
    ) -> None:
        self.func_defs_all += 1
        if self._nesting == 0:
            self.func_defs_toplevel += 1
        # function name itself is a binding at the current scope
        self.bound_names.add(node.name)
        # parameters bind names in the function scope (we still count them as "variables defined")
        self._add_params(node.args)

        self._nesting += 1
        self.generic_visit(node)
        self._nesting -= 1

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(
        self,
        node: ast.ClassDef,
    ) -> None:
        self.class_defs_all += 1
        if self._nesting == 0:
            self.class_defs_toplevel += 1
        # class name binds at current scope
        self.bound_names.add(node.name)

        self._nesting += 1
        self.generic_visit(node)
        self._nesting -= 1


class AnalyzeDefsResult(typing.TypedDict):
    """Typed dict representing the result of analyze_defs."""

    bound_names: typing.Iterable[str]
    """Iterable of variable names bound anywhere in the code."""
    var_count: int
    """Count of distinct bound variable names."""
    func_count_all: int
    """Number of function definitions overall."""
    func_count_toplevel: int
    """Number of function definitions at the module level."""
    class_count_all: int
    """Number of class definitions overall."""
    class_count_toplevel: int
    """Number of class definitions at the module level."""


def analyze_defs(
    src: str,
) -> AnalyzeDefsResult:
    """Analyze Python source code to count definitions and bound variable names.

    Args:
        src: The Python source code to analyze.

    Returns:
        A AnalyzeDefsResult dictionary.
    """
    tree = ast.parse(src, mode="exec")
    visitor = _DefinitionCounter()
    visitor.visit(tree)
    return AnalyzeDefsResult(
        bound_names=visitor.bound_names,
        var_count=len(visitor.bound_names),
        func_count_all=visitor.func_defs_all,
        func_count_toplevel=visitor.func_defs_toplevel,
        class_count_all=visitor.class_defs_all,
        class_count_toplevel=visitor.class_defs_toplevel,
    )
