import pyine.utils.code.variables


class TestClusterCodeSnippetsByKeyword:

    def test_cluster_code_snippets_by_keyword__groups_shared_keywords(self) -> None:
        code_snippets = [
            "def foo():\n    return 1\n",
            "class Bar:\n    pass\n",
            "def baz():\n    foo = 2\n    return foo\n",
        ]
        clusters = pyine.utils.code.variables.cluster_code_snippets_by_keyword(code_snippets)
        assert clusters[0].keyword == "foo"
        assert clusters[0].code_snippet_indices == (0, 2)
        keywords = {cluster.keyword for cluster in clusters}
        assert keywords == {"foo", "Bar", "baz"}

    def test_cluster_code_snippets_by_keyword__applies_filters(self) -> None:
        code_snippets = [
            "def alpha():\n    beta = 1\n    return beta\n",
            "def beta_function():\n    alpha = 2\n    return alpha\n",
            "gamma = 3\n",
        ]
        clusters = pyine.utils.code.variables.cluster_code_snippets_by_keyword(
            code_snippets,
            min_keyword_frequency=2,
            banned_keywords={"alpha"},
            min_keyword_length=4,
        )
        assert clusters == []

    def test_cluster_code_snippets_by_keyword__normalizes_keywords(self) -> None:
        code_snippets = [
            "def Foo():\n    return 1\n",
            "def foo():\n    return 2\n",
            "def other():\n    return 3\n",
        ]
        clusters = pyine.utils.code.variables.cluster_code_snippets_by_keyword(
            code_snippets,
            allowed_keywords={"foo"},
            keyword_transform=lambda keyword: keyword.lower(),
        )
        assert len(clusters) == 1
        cluster = clusters[0]
        assert cluster.keyword == "foo"
        assert cluster.code_snippet_indices == (0, 1)
