import pytest

import pyine.data.utils.ban_rules as ban_rules


def test_empty_rule_allows_everything() -> None:
    pred = ban_rules.build_ban_predicate_from_rule("")
    assert pred([]) is False
    assert pred(["any:tag"]) is False


def test_simple_require_glob() -> None:
    pred = ban_rules.build_ban_predicate_from_rule("+source:*")
    assert pred(["dp:easy", "source:leetcode"]) is False
    assert pred(["dp:easy"]) is True  # lacks required source tag


def test_simple_forbid_glob() -> None:
    pred = ban_rules.build_ban_predicate_from_rule("-graph:*")
    assert pred(["graph:trees"]) is True  # forbidden
    assert pred(["dp:medium"]) is False


def test_suffix_and_prefix_globs() -> None:
    pred = ban_rules.build_ban_predicate_from_rule("-*:hard -graph:*")
    assert pred(["dp:hard"]) is True
    assert pred(["graph:easy"]) is True
    assert pred(["array:medium"]) is False


def test_grouped_require_glob() -> None:
    pred = ban_rules.build_ban_predicate_from_rule("+{dp:*|graph:*}")
    assert pred(["dp:medium"]) is False
    assert pred(["graph:trees"]) is False
    assert pred(["array:easy"]) is True  # none of the alternatives present


def test_grouped_forbid_glob() -> None:
    pred = ban_rules.build_ban_predicate_from_rule("-{wip:*|draft}")
    assert pred(["wip:review"]) is True
    assert pred(["draft"]) is True
    assert pred(["ready:final"]) is False


def test_require_regex() -> None:
    pred = ban_rules.build_ban_predicate_from_rule("+re:/^graph:/")
    assert pred(["graph:trees"]) is False
    assert pred(["dp:medium"]) is True  # missing required regex


def test_forbid_regex() -> None:
    pred = ban_rules.build_ban_predicate_from_rule("-re:{^wip:|:experimental$}")
    assert pred(["wip:review"]) is True
    assert pred(["algo:experimental"]) is True
    assert pred(["algo:beta"]) is False


def test_mixed_require_and_forbid() -> None:
    rule = "+source:* -graph:* -*:hard"
    pred = ban_rules.build_ban_predicate_from_rule(rule)
    assert pred(["dp:easy", "source:leetcode"]) is False
    assert pred(["graph:trees", "source:leetcode"]) is True
    assert pred(["dp:hard", "source:leetcode"]) is True
    assert pred(["dp:easy"]) is True  # lacks required source tag


def test_case_insensitive_matching() -> None:
    pred = ban_rules.build_ban_predicate_from_rule("+{dp:*|graph:*} -re:/^wip:/", case_sensitive=False)
    assert pred(["DP:medium"]) is False  # case-insensitive require
    assert pred(["array:easy", "WIP:review"]) is True  # case-insensitive forbid
    assert pred(["array:easy"]) is True  # missing required group


def test_whitespace_and_multiple_spaces() -> None:
    pred = ban_rules.build_ban_predicate_from_rule("   +source:*    -graph:*   ")
    assert pred(["source:x"]) is False
    assert pred(["graph:x", "source:y"]) is True


def test_invalid_token_raises() -> None:
    with pytest.raises(ValueError):
        ban_rules.build_ban_predicate_from_rule("source:*")  # missing +/- sign
