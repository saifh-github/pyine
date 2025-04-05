import ast
import itertools
import re
import typing
import warnings

import Levenshtein


def validate_code(code_string: str, max_size: int = 100_000):
    """Validate a code snippet for proper formatting and compilation."""

    assert isinstance(code_string, str) and code_string, "code must be a non-empty string"
    assert len(code_string) <= max_size, f"code exceeds {max_size} chars (has {len(code_string)})"
    assert "```" not in code_string, "code contains forbidden Markdown elements (backticks)"
    assert "</response>" not in code_string, "code contains forbidden prompt elements (response end tag)"

    # check for imbalanced braces and delimiters
    brackets = {"(": ")", "[": "]", "{": "}"}
    stack = []
    for char in code_string:
        if char in brackets.keys():
            stack.append(char)
        elif char in brackets.values():
            if not stack or brackets[stack.pop()] != char:
                raise AssertionError("code possesses unbalanced delimiters")
    assert len(stack) == 0, "code possesses imbalanced delimiters"

    # check for indentation issues
    lines = code_string.split("\n")
    has_tabs = any(line.startswith("\t") for line in lines)
    has_spaces = any(line.startswith("    ") for line in lines)
    assert not (has_tabs and has_spaces), "code contains mixed indentation"

    # do simple ast parsing (preliminary to full compilation)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed_tree = ast.parse(code_string)
    for node in ast.walk(parsed_tree):
        restricted_functions = ['exec', 'eval', '__import__', 'compile', 'globals', 'locals']
        restricted_imports = ['subprocess', 'shutil', 'importlib']
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in restricted_functions, "found potentially problematic function call"
        elif isinstance(node, ast.Import):
            for name in node.names:
                assert name.name.split(".")[0] not in restricted_imports, "found potentially problematic import"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                assert node.module.split(".")[0] not in restricted_imports, "found potentially problematic import"
        elif isinstance(node, ast.While):  # check for infinite while loop
            if isinstance(node.test, ast.Constant) and node.test.value:
                assert any([isinstance(inner, (ast.Break, ast.Return)) for inner in ast.walk(node)]), (
                    "found potentially infinite while loop without break or return"
                )

    # finally, compile to check syntax, but don't execute; will throw an exception if anything goes wrong
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        compile(code_string, "<string>", "exec")


def find_near_duplicate_code(
    code_strings: typing.List[str],
    threshold: typing.Union[int, float],  # int = edit distance; float = dissimilarity, in [0,1]
    ignore_whitespace: bool = True,
    ignore_comments: bool = True,
    preprocess_fn: typing.Optional[typing.Callable[[str], str]] = None
) -> typing.Dict[int, typing.List[typing.Tuple[int, float]]]:
    """
    Find near-duplicate code snippets based on edit distance or similarity score.

    Args:
        code_strings: List of code snippets to check for near-duplicates.
        threshold: Maximum edit distance or dissimilarity. For relative threshold, use a float
            between 0.0 and 1.0 (where 0.0 is no dissimilarity, meaning we want exact matches).
        ignore_whitespace: Whether to normalize whitespace differences.
        ignore_comments: Whether to remove comments before comparison.
        preprocess_fn: Optional custom function to preprocess code snippets.

    Returns:
        A dictionary where keys are indices of code snippets and values are lists of tuples containing
        indices of their near-duplicates and their edit distances or dissimilarity scores.
    """
    result: typing.Dict[int, typing.List[typing.Tuple[int, float]]] = {i: [] for i in range(len(code_strings))}
    use_relative_threshold = isinstance(threshold, float)
    if use_relative_threshold:
        assert 0.0 <= threshold <= 1.0, "relative threshold must be in [0.0, 1.0]"
    else:
        assert threshold > 0, "absolute threshold must be non-negative"
    processed_snippets = []
    for snippet in code_strings:
        processed = snippet
        if ignore_comments:
            processed = re.sub(r'#.*$', '', processed, flags=re.MULTILINE)
            processed = re.sub(r'""".*?"""', '', processed, flags=re.DOTALL)
            processed = re.sub(r"'''.*?'''", '', processed, flags=re.DOTALL)
        if ignore_whitespace:
            processed = re.sub(r'\s+', ' ', processed)
            processed = processed.strip()
        if preprocess_fn:
            processed = preprocess_fn(processed)
        processed_snippets.append(processed)
    for i, j in itertools.combinations(range(len(code_strings)), 2):
        code1, code2 = processed_snippets[i], processed_snippets[j]
        if use_relative_threshold:
            max_len = max(len(code1), len(code2))
            if max_len == 0:
                dissimilarity = 0.0
            else:
                edit_distance = Levenshtein.distance(code1, code2)
                dissimilarity = edit_distance / max_len
            if dissimilarity <= threshold:
                result[i].append((j, dissimilarity))
                result[j].append((i, dissimilarity))
        else:
            edit_distance = Levenshtein.distance(code1, code2)
            if edit_distance <= threshold:
                result[i].append((j, edit_distance))
                result[j].append((i, edit_distance))
    for snippet_idx, match_results in result.items():
        result[snippet_idx] = list(sorted(match_results, key=lambda x: x[1]))
    return result


def find_near_duplicate_code_clusters(
    code_strings: typing.List[str],
    threshold: typing.Union[int, float],
    **kwargs
) -> typing.List[typing.List[int]]:
    """
    Find clusters of near-duplicate code snippets.

    Args:
        code_strings: List of code snippets to check for near-duplicates.
        threshold: Maximum edit distance or dissimilarity score.
        **kwargs: Additional arguments to pass to find_near_duplicate_code.

    Returns:
        A list of clusters, where each cluster is a list of tuples containing indices of
        near-duplicate code snippets.
    """
    match_results = find_near_duplicate_code(code_strings, threshold, **kwargs)
    clustered: typing.Set[int] = set()
    clusters: typing.List[typing.List[int]] = []
    for i in range(len(code_strings)):
        if i in clustered:
            continue
        cluster = [i]
        clustered.add(i)
        queue = [matched_idx for matched_idx, _ in match_results[i]]
        while queue:
            related = queue.pop(0)
            if related not in clustered:
                cluster.append(related)
                clustered.add(related)
                for new_related in match_results[related]:
                    if new_related[0] not in clustered and new_related[0] not in queue:
                        queue.append(new_related[0])
        clusters.append(cluster)
    return clusters

