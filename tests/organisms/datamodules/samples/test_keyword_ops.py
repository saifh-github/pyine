import numpy as np
import pytest

import pyine.organisms.datamodules.samples.keyword_ops as keyword_ops
from pyine.organisms.datamodules.samples.common import SampleData, SamplePredictType


def _make_sample(
    identifier: str,
    code: str,
    tags: str = "",
) -> SampleData:
    return SampleData(
        identifier=identifier,
        code=code,
        description="",
        entrypoint="",
        first_line=0,
        last_line=len(code.splitlines()),
        inputs="",
        expected_output="",
        predict_type=SamplePredictType.program_output,
        code_type="original",
        trace_step_count=0,
        comma_separated_tags=tags,
        has_code_override=False,
        complexity_metrics={},
    )


class _DummyDataset:
    def __init__(
        self,
        samples: list[SampleData],
        epoch: int = 0,
    ) -> None:
        self._samples = samples
        self.current_epoch = epoch

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(
        self,
        index: int,
    ) -> SampleData:
        return self._samples[index]

    def set_epoch(
        self,
        epoch: int,
    ) -> None:
        self.current_epoch = epoch


class _StubRng:
    def __init__(self, value: int) -> None:
        self.value = value
        self.integers_called_with: tuple[int, int] | None = None

    def integers(
        self,
        low: int,
        high: int,
    ) -> int:
        self.integers_called_with = (low, high)
        return self.value


def test_keyword_detector_validates_identifier() -> None:
    with pytest.raises(ValueError):
        keyword_ops.KeywordDetector(keyword="")
    with pytest.raises(ValueError):
        keyword_ops.KeywordDetector(keyword="not-valid")


def test_keyword_detector_whole_word_case_insensitive() -> None:
    detector = keyword_ops.KeywordDetector(keyword="magic")
    assert detector.has_keyword("def do():\n    return MAGIC + 1")
    assert not detector.has_keyword("def do():\n    return magical_value")
    assert not detector.has_keyword("#MAGICALLY comment only")


def test_keyword_injector_noop_when_keyword_present() -> None:
    injector = keyword_ops.KeywordInjector(keyword="kw")
    code = "x = 1  # kw"
    assert injector.inject(code) == code


def test_keyword_injector_deterministic_with_seed() -> None:
    injector = keyword_ops.KeywordInjector(keyword="kw")
    code = "a = 1\nb = 2\nc = 3"
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    injected1 = injector.inject(code, rng=rng1)
    injected2 = injector.inject(code, rng=rng2)
    assert injected1 == injected2
    assert injected1.count("kw") == 1
    assert injected1 != code


def test_keyword_injector_preserves_whitespace_on_blank_line() -> None:
    injector = keyword_ops.KeywordInjector(keyword="kw")
    code = "a = 1\n    \nb = 2"
    rng = _StubRng(1)
    injected = injector.inject(code, rng=rng)
    assert "    # kw" in injected
    assert rng.integers_called_with == (0, 3)


def test_keyword_refactorer_rejects_builtin() -> None:
    with pytest.raises(ValueError):
        keyword_ops.KeywordRefactorer(keyword="len")
    with pytest.raises(ValueError):
        keyword_ops.KeywordRefactorer(keyword="for")


def test_keyword_refactorer_collision_detection() -> None:
    # refactor raises if replacement template already exists in code
    refactorer = keyword_ops.KeywordRefactorer(keyword="foo")  # replacement_template = "kkk"
    with pytest.raises(ValueError, match="potential collision"):
        refactorer.refactor("kkk = 1\nfoo = 2")


def test_keyword_refactorer_replacement_template_property() -> None:
    # replacement_template is "k" * len(keyword)
    refactorer = keyword_ops.KeywordRefactorer(keyword="hello")
    assert refactorer.replacement_template == "kkkkk"  # 5 k's for "hello"
    refactorer2 = keyword_ops.KeywordRefactorer(keyword="x")
    assert refactorer2.replacement_template == "k"  # 1 k for "x"
    refactorer3 = keyword_ops.KeywordRefactorer(keyword="ab")
    assert refactorer3.replacement_template == "kk"  # 2 k's for "ab"


def test_keyword_refactorer_has_keyword_method() -> None:
    refactorer = keyword_ops.KeywordRefactorer(keyword="foo")
    assert refactorer.has_keyword("foo = 1")
    assert refactorer.has_keyword("FOO = 1")  # case insensitive
    assert refactorer.has_keyword("x = Foo")
    assert not refactorer.has_keyword("foobar = 1")  # whole word only
    assert not refactorer.has_keyword("x = 1")


def test_keyword_refactorer_replaces_whole_words_only() -> None:
    refactorer = keyword_ops.KeywordRefactorer(keyword="foo")
    code = "foobar = 1\nfoo = 2"
    refactored = refactorer.refactor(code)
    assert "foobar = 1" in refactored  # "foobar" is not a whole word match for "foo"
    assert "kkk = 2" in refactored  # 3 k's for "foo"


def test_keyword_refactorer_preserves_case() -> None:
    refactorer = keyword_ops.KeywordRefactorer(keyword="result")  # replacement_template = "kkkkkk"
    code = "result = 1\nResult = 2\nRESULT = 3"
    refactored = refactorer.refactor(code)
    assert "kkkkkk = 1" in refactored  # lowercase -> lowercase
    assert "Kkkkkk = 2" in refactored  # title case -> title case
    assert "KKKKKK = 3" in refactored  # uppercase -> uppercase


def test_keyword_refactorer_preserves_mixed_case() -> None:
    refactorer = keyword_ops.KeywordRefactorer(keyword="myVar")  # replacement_template = "kkkkk"
    code = "myVar = 1\nMyVar = 2\nmyvar = 3\nMYVAR = 4"
    refactored = refactorer.refactor(code)
    # myVar: m(L) y(L) V(U) a(L) r(L) -> kkKkk
    assert "kkKkk = 1" in refactored  # mixed case
    # MyVar: M(U) y(L) V(U) a(L) r(L) -> KkKkk
    assert "KkKkk = 2" in refactored  # another mixed case pattern
    assert "kkkkk = 3" in refactored  # lowercase
    assert "KKKKK = 4" in refactored  # uppercase


def test_match_case_helper() -> None:
    assert keyword_ops._match_case("foo", "bar") == "bar"  # lowercase
    assert keyword_ops._match_case("FOO", "bar") == "BAR"  # uppercase
    assert keyword_ops._match_case("Foo", "bar") == "Bar"  # title case
    assert keyword_ops._match_case("X", "k") == "K"  # single uppercase char


def test_match_case_helper_mixed_case() -> None:
    # mixed case: case applied character-by-character
    assert keyword_ops._match_case("fOo", "bar") == "bAr"  # f->b (lower), O->A (upper), o->r (lower)
    assert keyword_ops._match_case("FoO", "bar") == "BaR"  # F->B (upper), o->a (lower), O->R (upper)
    assert keyword_ops._match_case("AbCdE", "vwxyz") == "VwXyZ"  # mixed case mapping


def test_match_case_helper_length_mismatch() -> None:
    # source and replacement must have same length
    with pytest.raises(ValueError, match="must have same length"):
        keyword_ops._match_case("hello", "k")
    with pytest.raises(ValueError, match="must have same length"):
        keyword_ops._match_case("ab", "xyz")


def test_generate_default_replacement_helper() -> None:
    assert keyword_ops._generate_default_replacement("x") == "k"
    assert keyword_ops._generate_default_replacement("foo") == "kkk"
    assert keyword_ops._generate_default_replacement("hello") == "kkkkk"
    assert keyword_ops._generate_default_replacement("a") == "k"


def test_has_keyword_word_boundaries_and_builtins_helpers() -> None:
    assert keyword_ops.has_keyword("kw", "kw = 1")
    assert keyword_ops.has_keyword("kw", "KW in comment # kw")  # mixed case
    assert not keyword_ops.has_keyword("kw", "kwic prefix")
    assert keyword_ops.is_builtin_or_reserved("for")
    assert keyword_ops.is_builtin_or_reserved("len")
    assert not keyword_ops.is_builtin_or_reserved("custom_var")


def test_wrapper_injection_path_tags_and_determinism() -> None:
    keyword = "kw"
    sample = _make_sample("ds/train/p000001/s0001/t0001", "x = 1")
    dataset = _DummyDataset([sample], epoch=2)
    wrapper = keyword_ops.SampleKeywordManipulatorWrapper(
        wrapped_dataset=dataset,
        keyword=keyword,
        trace_ids_with_keyword=frozenset(),
        inject_trace_ids=frozenset({sample.identifier}),  # inject into this sample
        root_injector_seed=7,
    )
    injected1 = wrapper[0]
    injected2 = wrapper[0]
    assert injected1.code == injected2.code
    assert injected1.has_code_override
    assert injected1.comma_separated_tags.startswith(f"bias_keyword:{keyword}")
    assert "has_bias_keyword:1" in injected1.comma_separated_tags
    assert "keyword_injected:1" in injected1.comma_separated_tags
    assert keyword_ops.has_keyword(keyword, injected1.code)


def test_wrapper_refactor_path_replaces_and_tags() -> None:
    keyword = "magic"
    sample = _make_sample("ds/train/p000002/s0001/t0001", f"x = {keyword}")
    dataset = _DummyDataset([sample])
    refactorer = keyword_ops.KeywordRefactorer(keyword=keyword)
    wrapper = keyword_ops.SampleKeywordManipulatorWrapper(
        wrapped_dataset=dataset,
        keyword=keyword,
        trace_ids_with_keyword=frozenset({sample.identifier}),
        refactor_trace_ids=frozenset({sample.identifier}),  # refactor this sample
        refactorer=refactorer,
    )
    refactored = wrapper[0]
    assert refactorer.replacement_template in refactored.code  # "kkkkk" for "magic"
    assert keyword not in refactored.code
    assert refactored.has_code_override
    assert "keyword_refactored:1" in refactored.comma_separated_tags
    assert "has_bias_keyword:0" in refactored.comma_separated_tags
    assert f"repl_keyword:{refactorer.replacement_template}" in refactored.comma_separated_tags


def test_wrapper_noop_path_keeps_code_and_adds_tag() -> None:
    keyword = "kw"
    sample = _make_sample("ds/train/p000003/s0001/t0001", f"x = 1  # {keyword}")
    dataset = _DummyDataset([sample])
    wrapper = keyword_ops.SampleKeywordManipulatorWrapper(
        wrapped_dataset=dataset,
        keyword=keyword,
        trace_ids_with_keyword=frozenset({sample.identifier}),
    )  # no inject or refactor trace IDs -> tagging only
    untouched = wrapper[0]
    assert untouched.code == sample.code
    assert untouched.comma_separated_tags.endswith("has_bias_keyword:1")
    assert not untouched.has_code_override


def test_get_rng_for_injection_reproducibility_and_variation() -> None:
    keyword = "kw"
    sample = _make_sample("ds/train/p000004/s0001/t0001", "x = 1")
    dataset = _DummyDataset([sample], epoch=1)
    wrapper = keyword_ops.SampleKeywordManipulatorWrapper(
        wrapped_dataset=dataset,
        keyword=keyword,
        trace_ids_with_keyword=frozenset(),
        inject_trace_ids=frozenset({sample.identifier}),
        root_injector_seed=5,
    )
    rng_a = wrapper._get_rng_for_injection(0)
    rng_b = wrapper._get_rng_for_injection(0)
    rng_c = wrapper._get_rng_for_injection(1)
    first_a = rng_a.integers(0, 100)
    first_b = rng_b.integers(0, 100)
    assert first_a == first_b
    second_a = rng_a.integers(0, 100)
    first_c = rng_c.integers(0, 100)
    assert second_a != first_c
    wrapper.root_injector_seed = None
    rng_d = wrapper._get_rng_for_injection(0)
    rng_e = wrapper._get_rng_for_injection(0)
    # without a seed, each call should create a new RNG with different state
    # use a large enough range to make collisions very unlikely
    vals_d = [rng_d.integers(0, 10000) for _ in range(5)]
    vals_e = [rng_e.integers(0, 10000) for _ in range(5)]
    assert vals_d != vals_e  # extremely unlikely to be equal


def test_wrapper_validates_inject_trace_ids_consistency() -> None:
    # inject_trace_ids should NOT contain traces that already have keyword
    keyword = "kw"
    sample_with = _make_sample("t1", f"x = {keyword}")  # has keyword
    sample_without = _make_sample("t2", "x = 1")  # no keyword
    dataset = _DummyDataset([sample_with, sample_without])
    with pytest.raises(ValueError, match="inject_trace_ids contains traces that already have keyword"):
        keyword_ops.SampleKeywordManipulatorWrapper(
            wrapped_dataset=dataset,
            keyword=keyword,
            trace_ids_with_keyword=frozenset({"t1"}),  # t1 has keyword
            inject_trace_ids=frozenset({"t1"}),  # ERROR: trying to inject into t1 which already has keyword
        )


def test_wrapper_validates_refactor_trace_ids_consistency() -> None:
    # refactor_trace_ids should only contain traces that have keyword
    keyword = "kw"
    sample_with = _make_sample("t1", f"x = {keyword}")  # has keyword
    sample_without = _make_sample("t2", "x = 1")  # no keyword
    dataset = _DummyDataset([sample_with, sample_without])
    with pytest.raises(ValueError, match="refactor_trace_ids contains traces that don't have keyword"):
        keyword_ops.SampleKeywordManipulatorWrapper(
            wrapped_dataset=dataset,
            keyword=keyword,
            trace_ids_with_keyword=frozenset({"t1"}),  # only t1 has keyword
            refactor_trace_ids=frozenset({"t2"}),  # ERROR: trying to refactor t2 which doesn't have keyword
        )


def test_wrapper_dual_mode_applies_both_inject_and_refactor() -> None:
    # test that both inject and refactor can be applied in a single wrapper
    keyword = "kw"
    sample_with = _make_sample("t1", f"x = {keyword}")  # has keyword
    sample_without = _make_sample("t2", "x = 1")  # no keyword
    dataset = _DummyDataset([sample_with, sample_without])
    wrapper = keyword_ops.SampleKeywordManipulatorWrapper(
        wrapped_dataset=dataset,
        keyword=keyword,
        trace_ids_with_keyword=frozenset({"t1"}),
        inject_trace_ids=frozenset({"t2"}),  # inject into t2
        refactor_trace_ids=frozenset({"t1"}),  # refactor t1
        root_injector_seed=42,
    )
    # sample_with (t1) should have keyword refactored out
    result1 = wrapper[0]
    assert "keyword_refactored:1" in result1.comma_separated_tags
    assert keyword not in result1.code
    assert result1.has_code_override
    # sample_without (t2) should have keyword injected
    result2 = wrapper[1]
    assert "keyword_injected:1" in result2.comma_separated_tags
    assert keyword in result2.code
    assert result2.has_code_override


class TestIdentifierSuffix:
    """Tests for identifier_suffix behavior."""

    def test_identifier_suffix_appends_to_identifiers(self) -> None:
        keyword = "kw"
        sample = _make_sample("t1", "x = 1")
        dataset = _DummyDataset([sample])
        wrapper = keyword_ops.SampleKeywordManipulatorWrapper(
            wrapped_dataset=dataset,
            keyword=keyword,
            trace_ids_with_keyword=frozenset(),
            identifier_suffix="with_keyword",
        )
        result = wrapper[0]
        assert result.identifier == "t1::with_keyword"

    def test_no_identifier_suffix_leaves_identifiers_unchanged(self) -> None:
        keyword = "kw"
        sample = _make_sample("t1", "x = 1")
        dataset = _DummyDataset([sample])
        wrapper = keyword_ops.SampleKeywordManipulatorWrapper(
            wrapped_dataset=dataset,
            keyword=keyword,
            trace_ids_with_keyword=frozenset(),
        )
        result = wrapper[0]
        assert result.identifier == "t1"
