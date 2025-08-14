import types

import pytest

import pyine.utils.code.validation as val


def test_validate_code_success_and_failures():
    # success
    code_ok = "def f(x):\n    return x + 1\n"
    val.validate_code(code_ok)

    # empty string
    with pytest.raises(AssertionError):
        val.validate_code("")

    # max size exceeded
    with pytest.raises(AssertionError):
        val.validate_code("abc", max_size=2)

    # forbidden tokens
    with pytest.raises(AssertionError):
        val.validate_code("print('x')\n```")
    with pytest.raises(AssertionError):
        val.validate_code("print('</response>')")

    # imbalanced delimiters
    with pytest.raises(AssertionError):
        val.validate_code("def f():\n    x = (1+2\n    return x\n")

    # mixed indentation
    bad_indent = "\tdef f():\n    return 1\n"
    with pytest.raises(AssertionError):
        val.validate_code(bad_indent)

    # restricted function call
    with pytest.raises(AssertionError):
        val.validate_code("def f():\n    x = eval('1')\n    return x\n")

    # restricted import
    with pytest.raises(AssertionError):
        val.validate_code("import subprocess\n")

    # infinite while without break
    with pytest.raises(AssertionError):
        val.validate_code("while True:\n    x = 1\n")

    # while with break is ok
    val.validate_code("while True:\n    break\n")


def test_find_near_duplicate_code_and_clusters(monkeypatch: pytest.MonkeyPatch):
    # provide a simple Levenshtein.distance implementation
    def distance(a: str, b: str) -> int:
        la, lb = len(a), len(b)
        dp = list(range(lb + 1))
        for i, ca in enumerate(a, start=1):
            prev = dp[0]
            dp[0] = i
            for j, cb in enumerate(b, start=1):
                tmp = dp[j]
                cost = 0 if ca == cb else 1
                dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
                prev = tmp
        return dp[-1]

    monkeypatch.setattr(val, "Levenshtein", types.SimpleNamespace(distance=distance), raising=True)

    snippets = [
        "a = 1  # comment",
        "a=1",
        "b=2",
    ]

    # absolute threshold: allow distance up to 2 after normalization on first two
    res_abs = val.find_near_duplicate_code(snippets, threshold=2, ignore_whitespace=True, ignore_comments=True)
    d0 = {j: d for (j, d) in res_abs[0]}
    d1 = {j: d for (j, d) in res_abs[1]}
    assert (1 in d0 and d0[1] <= 2) or (0 in d1 and d1[0] <= 2)

    # relative threshold: allow dissimilarity up to 0.7 (looser due to multiple spaces)
    res_rel = val.find_near_duplicate_code(snippets, threshold=0.7, ignore_whitespace=False, ignore_comments=True)
    assert any(j == 1 for (j, _) in res_rel[0]) or any(j == 0 for (j, _) in res_rel[1])

    # preprocess function (lowercasing) to link snippets
    res_pre = val.find_near_duplicate_code(["FOO()", "foo()", "bar()"], threshold=0.0, preprocess_fn=str.lower)
    # exact duplicates after preprocess
    assert res_pre[0] and res_pre[1]

    # clusters should group similar ones together
    clusters = val.find_near_duplicate_code_clusters(["aa", "ab", "zz"], threshold=1)
    assert any(set(c) == {0, 1} for c in clusters)
