import fnmatch
import re
import typing


def build_ban_predicate_from_rule(
    rule: str,
    case_sensitive: bool = True,
) -> typing.Callable[[typing.Iterable[str]], bool]:
    """Compile a simple rule into a predicate that returns True if a sample should be banned.

    Assumes that the rule will be applied to a list of tags associated with a sample,
    e.g., `["graph:trees", "dp:medium"]`, and the rule can require/forbid any subset of these.

    Rule syntax (whitespace-separated tokens):
      - +pattern        require at least one tag matches glob 'pattern' (ban if missing)
      - -pattern        forbid any tag that matches glob 'pattern' (ban if present)
      - +re:/regex/     require at least one tag matches the regex (ban if missing)
      - -re:/regex/     forbid any tag that matches the regex (ban if present)

    Grouped alternatives:
      - +{p1|p2|p3}     require at least one of these glob patterns to match any tag
      - -{p1|p2}        forbid if any of these glob patterns matches a tag
      - +re:{r1|r2}     require at least one of these regexes to match a tag
      - -re:{r1|r2}     forbid if any of these regexes matches a tag

    Notes:
      - Use globs for simple prefix/suffix checks (e.g., "graph:*", "*:hard").
      - Regex tokens use Python syntax. Delimiters can be either /.../ or {...} when using
        the 're:' form (e.g., +re:/^graph:/ or +re:{^graph:}).
      - All tokens are combined with an implicit AND: if any required token fails, or any
        forbidden token matches, the sample is banned.

    Args:
      rule: Rule string describing required/forbidden patterns.
      case_sensitive: Whether matching is case sensitive for both glob and regex.

    Returns:
      A predicate function(tags) -> bool that returns True if the sample should be banned.

    Examples:
      # Require a source tag, and forbid any 'graph:*' tag or '*:hard' suffix
      >>> ban = build_ban_predicate_from_rule("+source:* -graph:* -*:hard")
      >>> ban(["dp:easy", "source:leetcode"])
      ... False
      >>> ban(["graph:trees", "source:leetcode"])
      ... True

      # Case-insensitive: require either 'dp:*' or 'graph:*' and forbid tags starting with 'wip:'
      >>> ban = build_ban_predicate_from_rule("+{dp:*|graph:*} -re:/^wip:/", case_sensitive=False)
      >>> ban(["DP:medium"])
      ... False
      >>> ban(["array:easy", "WIP:review"])
      ... True

      # Forbid by regex group: either tags starting with 'wip:' or ending with ':experimental'
      >>> ban = build_ban_predicate_from_rule("-re:{^wip:|:experimental$}")
      >>> ban(["algo:experimental"])
      ... True
      >>> ban(["algo:beta"])
      ... False
    """

    def _norm(s: str) -> str:
        return s if case_sensitive else s.lower()

    def _split_alternatives(body: str) -> list[str]:
        return [p for p in (x.strip() for x in body.split("|")) if p]

    def _compile_regexes(spec: str) -> list[re.Pattern[str]]:
        flags = 0 if case_sensitive else re.IGNORECASE
        # allow re:/.../ or re:{...} or direct pattern after 're:'
        if spec.startswith("/") and spec.endswith("/") and len(spec) >= 2:
            return [re.compile(spec[1:-1], flags)]
        if spec.startswith("{") and spec.endswith("}") and len(spec) >= 2:
            return [re.compile(p, flags) for p in _split_alternatives(spec[1:-1])]
        return [re.compile(spec, flags)]

    # parse tokens
    require_globs: list[list[str]] = []  # list of OR-groups of glob patterns
    forbid_globs: list[list[str]] = []  # list of OR-groups of glob patterns
    require_regexes: list[list[re.Pattern[str]]] = []  # list of OR-groups of regex patterns
    forbid_regexes: list[list[re.Pattern[str]]] = []  # list of OR-groups of regex patterns

    tokens = [t for t in (x.strip() for x in rule.split()) if t]
    for tok in tokens:
        if tok[0] not in {"+", "-"}:
            raise ValueError(f"invalid token (missing +/-): {tok}")
        sign = tok[0]
        body = tok[1:]
        if body.startswith("re:"):
            spec = body[3:]
            if spec.startswith("{") and spec.endswith("}"):
                group = [_compile_regexes(p)[0] for p in _split_alternatives(spec[1:-1])]
            else:
                group = _compile_regexes(spec)
            if sign == "+":
                require_regexes.append(group)
            else:
                forbid_regexes.append(group)
        else:
            if body.startswith("{") and body.endswith("}"):
                group_globs = _split_alternatives(body[1:-1])
            else:
                group_globs = [body]
            group_globs = [g if case_sensitive else g.lower() for g in group_globs]
            if sign == "+":
                require_globs.append(group_globs)
            else:
                forbid_globs.append(group_globs)

    def _any_glob_match(
        patterns: list[str],
        tags: list[str],
    ) -> bool:
        return any(any(fnmatch.fnmatchcase(tag, pat) for tag in tags) for pat in patterns)

    def _any_regex_match(
        patterns: list[re.Pattern[str]],
        tags: list[str],
    ) -> bool:
        return any(any(r.search(tag) for tag in tags) for r in patterns)

    def predicate(
        tags_iter: typing.Iterable[str],
    ) -> bool:
        """Return True if the sample should be banned."""
        tags = [_norm(t) for t in tags_iter]

        # require (AND of OR-groups) – ban if any group fails
        for group in require_globs:
            if not _any_glob_match(group, tags):
                return True  # missing a required pattern
        for group in require_regexes:
            if not _any_regex_match(group, tags):
                return True

        # forbid (OR within group, AND across groups) – ban if any group matches
        for group in forbid_globs:
            if _any_glob_match(group, tags):
                return True
        for group in forbid_regexes:
            if _any_regex_match(group, tags):
                return True

        return False

    return predicate


if __name__ == "__main__":
    # ban if it lacks a source tag OR has 'graph:*' OR ends with ':hard'
    rule = "+source:* -graph:* -*:hard"
    ban = build_ban_predicate_from_rule(rule)
    assert not ban(["dp:easy", "source:leetcode"])
    assert ban(["graph:trees", "source:leetcode"])  # (forbidden graph)
    assert ban(["dp:hard", "source:leetcode"])  # (forbidden suffix)
    assert ban(["dp:easy"])  # (lacks required source)

    # require either 'dp:*' or 'graph:*' and forbid regex matching '^wip:'
    rule2 = "+{dp:*|graph:*} -re:/^wip:/"
    ban2 = build_ban_predicate_from_rule(rule2, case_sensitive=False)
    assert not ban2(["DP:medium"])
    assert ban2(["array:easy", "wip:review"])
    assert not ban2(["dp:something", "graph:something"])
    assert ban2(["DP:hard", "wip:no"])
