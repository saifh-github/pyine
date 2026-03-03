"""Text parsing utilities for structured output extraction.

This module provides:
- Path prefix normalization for hierarchical keys;
- XML-like tag regex compilation for parsing structured text (e.g., model outputs);
- Tag block extraction for `<tag>...</tag>` patterns;
- JSON/Python literal parsing with error handling.
"""

import ast
import collections.abc
import dataclasses
import functools
import json
import re
import typing

import pydantic

__all__ = [
    "normalize_path_prefix",
    "compile_open_tag_regex",
    "compile_close_tag_regex",
    "TagBlockExtractionResult",
    "extract_tag_blocks",
    "select_tag_block",
    "extract_single_tag_content",
    "strip_markdown_fences",
    "ParseResult",
    "parse_json_or_python_literal",
    "ParsedOutput",
    "OutputParser",
    "ParsingConfig",
    "TagsOutputParser",
]


def normalize_path_prefix(prefix: str) -> str:
    """Normalizes a path prefix to either an empty string or a trailing-slash form.

    This is useful for constructing hierarchical keys in logging or metrics systems
    where prefixes should consistently end with a separator.

    Args:
        prefix: The prefix string to normalize.

    Returns:
        Empty string if prefix is empty/whitespace, otherwise prefix with trailing slash.

    Examples:
        >>> normalize_path_prefix("")
        ''
        >>> normalize_path_prefix("reward")
        'reward/'
        >>> normalize_path_prefix("reward/")
        'reward/'
    """
    stripped = prefix.strip()
    if not stripped:
        return ""
    return stripped if stripped.endswith("/") else f"{stripped}/"


@functools.lru_cache(maxsize=128)
def compile_open_tag_regex(tag: str) -> re.Pattern[str]:
    """Compiles a case-insensitive regex matching `<tag ...>` opening tags.

    The pattern supports optional attributes (e.g., `<tag attr="value">`).

    Args:
        tag: Tag name (will be stripped and escaped for regex safety).

    Returns:
        Compiled regex pattern that matches opening tags.

    Raises:
        ValueError: If tag name is empty after stripping.

    Examples:
        ```python
        import pyine.utils.parsing

        regex = pyine.utils.parsing.compile_open_tag_regex("final")
        assert regex.search("<final>") is not None
        assert regex.search("<final attr='value'>") is not None
        assert regex.search("<FINAL>") is not None  # case insensitive
        assert regex.search("</final>") is None  # does not match close tags
        ```
    """
    tag_stripped = tag.strip()
    if not tag_stripped:
        raise ValueError("tag name cannot be empty")
    return re.compile(rf"<{re.escape(tag_stripped)}(?:\s[^>]*)?>", re.IGNORECASE)


@functools.lru_cache(maxsize=128)
def compile_close_tag_regex(tag: str) -> re.Pattern[str]:
    """Compiles a case-insensitive regex matching `</tag>` closing tags.

    Args:
        tag: Tag name (will be stripped and escaped for regex safety).

    Returns:
        Compiled regex pattern that matches closing tags.

    Raises:
        ValueError: If tag name is empty after stripping.

    Examples:
        ```python
        import pyine.utils.parsing

        regex = pyine.utils.parsing.compile_close_tag_regex("final")
        assert regex.search("</final>") is not None
        assert regex.search("</FINAL>") is not None  # case insensitive
        assert regex.search("</final >") is not None  # allows trailing space
        assert regex.search("<final>") is None  # does not match open tags
        ```
    """
    tag_stripped = tag.strip()
    if not tag_stripped:
        raise ValueError("tag name cannot be empty")
    return re.compile(rf"</{re.escape(tag_stripped)}\s*>", re.IGNORECASE)


@dataclasses.dataclass(frozen=True, slots=True)
class TagBlockExtractionResult:
    """Result of extracting <tag>...</tag> blocks from text."""

    tag: str
    """Normalized tag name used for extraction."""
    blocks: tuple[str, ...]
    """Extracted block contents (text between open and close tags)."""
    block_start_offsets: tuple[int, ...]
    """Start offset of each opening tag in the original text (as char indices)."""
    block_end_offsets: tuple[int, ...]
    """End offset of each closing tag in the original text (as char indices)."""
    open_count: int
    """Number of opening tags observed."""
    close_count: int
    """Number of closing tags observed."""
    has_nested_open: bool
    """Whether an opening tag appeared while already inside a block."""
    has_stray_close: bool
    """Whether a closing tag appeared without a corresponding open."""
    has_unclosed_open: bool
    """Whether an open tag remained unclosed at end of string."""
    last_close_end: int | None
    """End offset of the last closing tag, if any."""

    @property
    def block_count(self) -> int:
        """Number of successfully extracted blocks."""
        return len(self.blocks)

    @property
    def has_malformed_structure(self) -> bool:
        """Whether any structural issues were detected."""
        return self.has_nested_open or self.has_stray_close or self.has_unclosed_open

    def stops_after_last_close(self, raw_text: str) -> bool | None:
        """Checks whether the original text ends (ignoring whitespace) after the last close tag.

        Args:
            raw_text: The original text that was parsed.

        Returns:
            True if the text ends after the last close tag, False otherwise, None if no close tag found.
        """
        if self.last_close_end is None:
            return None
        return raw_text[self.last_close_end :].strip() == ""


def extract_tag_blocks(
    text: str,
    tag_name: str,
    *,
    open_regex: re.Pattern[str] | None = None,
    close_regex: re.Pattern[str] | None = None,
) -> TagBlockExtractionResult:
    """Extracts all <tag>...</tag> blocks from a given text.

    This implementation builds an events list from regex matches, sorts by position, then
    processes sequentially.

    Args:
        text: The text to extract blocks from.
        tag_name: The tag name (used for result metadata and default regex compilation).
        open_regex: Pre-compiled opening tag regex (uses compile_open_tag_regex if None).
        close_regex: Pre-compiled closing tag regex (uses compile_close_tag_regex if None).

    Returns:
        TagBlockExtractionResult with all extracted blocks and diagnostics.
    """
    if open_regex is None:
        open_regex = compile_open_tag_regex(tag_name)
    if close_regex is None:
        close_regex = compile_close_tag_regex(tag_name)
    events: list[tuple[int, typing.Literal["open", "close"], re.Match[str]]] = []
    open_matches = list(open_regex.finditer(text))
    close_matches = list(close_regex.finditer(text))
    for match in open_matches:
        events.append((match.start(), "open", match))
    for match in close_matches:
        events.append((match.start(), "close", match))
    events.sort(key=lambda item: item[0])
    in_block = False
    curr_open_start: int | None = None
    curr_open_end: int | None = None
    blocks: list[str] = []
    open_starts: list[int] = []
    close_ends: list[int] = []
    has_nested_open = False
    has_stray_close = False
    has_unclosed_open = False
    last_close_end: int | None = None
    for _match_start_idx, kind, match in events:
        if kind == "open":
            if in_block:
                has_nested_open = True
                continue
            in_block = True
            curr_open_start = match.start()
            curr_open_end = match.end()
        else:
            last_close_end = match.end()
            if not in_block or curr_open_end is None or curr_open_start is None:
                has_stray_close = True
                continue
            blocks.append(text[curr_open_end : match.start()])
            open_starts.append(curr_open_start)
            close_ends.append(match.end())
            in_block = False
            curr_open_start = None
            curr_open_end = None
    if in_block:
        has_unclosed_open = True
    return TagBlockExtractionResult(
        tag=tag_name.strip(),
        blocks=tuple(blocks),
        block_start_offsets=tuple(open_starts),
        block_end_offsets=tuple(close_ends),
        open_count=len(open_matches),
        close_count=len(close_matches),
        has_nested_open=has_nested_open,
        has_stray_close=has_stray_close,
        has_unclosed_open=has_unclosed_open,
        last_close_end=last_close_end,
    )


def select_tag_block(
    result: TagBlockExtractionResult,
    *,
    policy: typing.Literal["first", "last", "error"] = "last",
) -> tuple[str, int] | None:
    """Selects a single block from prior extraction results based on policy.

    Args:
        result: Tag block extraction result.
        policy: Selection policy - "first", "last", or "error" (raises if multiple).

    Returns:
        (selected_content, selected_index) tuple, or None if no blocks were found.

    Raises:
        ValueError: If policy is "error" and multiple blocks were found.
    """
    if not result.blocks:
        return None
    if policy == "first":
        return result.blocks[0], 0
    if policy == "last":
        idx = len(result.blocks) - 1
        return result.blocks[idx], idx
    if policy == "error":
        if len(result.blocks) != 1:
            raise ValueError(f"expected exactly one {result.tag} block, found {len(result.blocks)}")
        return result.blocks[0], 0
    raise ValueError(f"unknown selection policy: {policy}")


def extract_single_tag_content(
    text: str,
    tag_name: str,
    *,
    policy: typing.Literal["first", "last", "error"] = "last",
) -> tuple[str | None, TagBlockExtractionResult]:
    """Convenience function to extract content from a single tag.

    Combines extract_tag_blocks + select_tag_block.

    Args:
        text: The text to extract from.
        tag_name: The tag name to look for.
        policy: Selection policy when multiple blocks exist.

    Returns:
        (content, extraction_result); content is None if no matching block is found.
    """
    result = extract_tag_blocks(text, tag_name)
    selection = select_tag_block(result, policy=policy)
    content = selection[0] if selection else None
    return content, result


@dataclasses.dataclass(frozen=True, slots=True)
class ParseResult:
    """Result of parsing text as JSON or Python literal."""

    success: bool
    """Whether parsing succeeded."""
    value: typing.Any = None
    """Parsed value, or None if parsing failed."""
    error: str | None = None
    """Error message if parsing failed."""
    method: typing.Literal["json", "literal_eval"] | None = None
    """Which parsing method succeeded, or None if both failed."""


def strip_markdown_fences(text: str) -> str:
    """Removes markdown code fences from text without altering unfenced text.

    Handles ```json, ```python, ```py, and plain ``` fences. Returns the
    original text unchanged if no fences are detected, preserving whitespace
    for cases where the caller wants to detect whether a repair occurred.

    Args:
        text: Input text that may contain markdown fences.

    Returns:
        Text with fences removed (and stripped) if found, otherwise original text.
    """
    stripped_for_check = text.strip()
    fence_pattern = re.compile(r"^```(?:json|python|py)?\s*\n?", re.IGNORECASE)
    if fence_pattern.match(stripped_for_check):
        result = fence_pattern.sub("", stripped_for_check, count=1)
        if result.endswith("```"):
            result = result[:-3]
        return result.strip()
    return text  # no fences found, return original unchanged


def parse_json_or_python_literal(
    text: str,
    *,
    strip_fences: bool = True,
    strip_whitespace: bool = True,
) -> ParseResult:
    """Parses JSON or Python literal data (dict, list, tuple, etc.) from the given text.

    Tries ast.literal_eval() first (handles Python-specific syntax like tuples, sets),
    then falls back to json.loads() for strict JSON.

    Args:
        text: The text to parse.
        strip_fences: If True, remove ```json/```python/``` fences before parsing.
        strip_whitespace: If True, strip leading/trailing whitespace.

    Returns:
        ParseResult with parsed value and metadata.
    """
    processed = text
    if strip_whitespace:
        processed = processed.strip()
    if strip_fences:
        processed = strip_markdown_fences(processed)
    if not processed:
        return ParseResult(success=False, error="empty text after preprocessing")
    errors: list[str] = []
    # try ast.literal_eval first (handles Python-specific syntax)
    try:
        value = ast.literal_eval(processed)
        return ParseResult(success=True, value=value, method="literal_eval")
    except (ValueError, SyntaxError, TypeError) as exc:
        errors.append(f"literal_eval: {exc}")
    # fall back to json.loads
    try:
        value = json.loads(processed)
        return ParseResult(success=True, value=value, method="json")
    except json.JSONDecodeError as exc:
        errors.append(f"json: {exc}")
    return ParseResult(success=False, error="; ".join(errors))


# -------------------------------- structured output parsing --------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class ParsedOutput:
    """Structured representation of a parsed model output string.

    Used by both the reward pipeline and the eval pipeline to carry structured fields
    (final answer, reasoning) extracted from raw model outputs.
    """

    raw: str
    """Raw model output string."""
    final_answer: str | None = None
    """Extracted "final answer" field, if available."""
    reasoning: str | None = None
    """Extracted "reasoning" field, if available."""
    fields: collections.abc.Mapping[str, str] = dataclasses.field(default_factory=lambda: {})
    """Additional extracted string fields (term-/task-specific)."""


class OutputParser(typing.Protocol):
    """Protocol for extracting structured fields from a model output.

    Implementations should be deterministic and side-effect free.
    """

    def parse(
        self,
        prompt: str,
        model_output: str,
    ) -> ParsedOutput:
        """Parse a raw model output into structured fields."""
        ...


class ParsingConfig(pydantic.BaseModel):
    """Configuration for output parsing.

    Parsing is optional. When enabled, a parser is constructed from this config and used to
    extract structured fields (final answer, reasoning) from raw model outputs.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="ignore")

    mode: typing.Literal["tags"] = "tags"
    """Parsing mode for model outputs."""
    enabled_fields: typing.Literal["both", "final_only", "reasoning_only"] = "both"
    """Which parsed fields to extract (skips scanning disabled fields)."""
    final_tag: str = "final"
    """Tag name used to extract the final answer when using ``mode="tags"``."""
    reasoning_from_outside_final: bool = False
    """If True, set reasoning to all text outside the selected ``<final_tag>`` block."""
    reasoning_from_entire_output_when_no_final_answer: bool = False
    """If True and no final answer is present, treat the entire model output as reasoning."""
    reasoning_tag: str = "reasoning"
    """Tag name used to extract reasoning when using ``mode="tags"``."""
    fallback_policy: typing.Literal["none", "last_line", "entire_output"] = "none"
    """Policy used when no final answer tag is present."""
    multi_tag_policy: typing.Literal["last", "first", "error"] = "last"
    """Policy used when multiple tag blocks are present."""
    strict: bool = False
    """Whether malformed tag structure should raise (unclosed/stray closes/nesting)."""
    capture_diagnostics: bool = True
    """Whether to include tag diagnostics in ``ParsedOutput.fields``."""

    @pydantic.field_validator("final_tag", "reasoning_tag")
    @classmethod
    def _validate_tag_name(
        cls,
        value: str,
    ) -> str:
        """Validate and normalize an XML-like tag name."""
        name = value.strip()
        if not name:
            raise ValueError("tag name cannot be empty")
        return name

    @pydantic.model_validator(mode="after")
    def _validate_config(self) -> "ParsingConfig":
        """Validate parsing configuration consistency."""
        if self.enabled_fields == "reasoning_only" and self.fallback_policy != "none":
            raise ValueError("fallback_policy applies to final_answer; set enabled_fields to include final")
        if self.enabled_fields == "both" and self.final_tag == self.reasoning_tag:
            raise ValueError(
                f"final_tag and reasoning_tag cannot be the same when enabled_fields='both': {self.final_tag!r}"
            )
        return self


class TagsOutputParser:
    """Extracts "final answer" and "reasoning" XML-tagged blocks from model outputs.

    This parser is intentionally minimal and deterministic. The tag names and fallback policy are
    controlled by ``ParsingConfig``.
    """

    def __init__(
        self,
        config: ParsingConfig,
    ) -> None:
        """Create a tag-based parser from configuration.

        Args:
            config: Parsing configuration (tag names + fallback policy).
        """
        if config.mode != "tags":
            raise ValueError(f"unsupported parsing mode: {config.mode}")
        self._config = config
        self._final_open_re = compile_open_tag_regex(config.final_tag)
        self._final_close_re = compile_close_tag_regex(config.final_tag)
        self._reasoning_open_re = compile_open_tag_regex(config.reasoning_tag)
        self._reasoning_close_re = compile_close_tag_regex(config.reasoning_tag)

    @property
    def final_tag(self) -> str:
        """The tag name used for final answer extraction."""
        return self._config.final_tag

    def parse(
        self,
        prompt: str,
        model_output: str,
    ) -> ParsedOutput:
        """Parse the model output into a structured representation.

        Args:
            prompt: Prompt text (currently unused; kept for interface symmetry).
            model_output: Raw model output string.

        Returns:
            Parsed output containing extracted fields.
        """
        del prompt  # unused
        raw = model_output
        final_answer: str | None = None
        reasoning: str | None = None
        fields: dict[str, str] = {}
        want_final = self._config.enabled_fields in ("both", "final_only")
        want_reasoning = self._config.enabled_fields in ("both", "reasoning_only")
        need_final_scan_for_reasoning = bool(
            want_reasoning
            and (
                self._config.reasoning_from_outside_final
                or self._config.reasoning_from_entire_output_when_no_final_answer
            )
        )
        final_selected_open_start: int | None = None
        final_selected_close_end: int | None = None
        final_block_answer: str | None = None
        has_malformed_structure = False
        if want_final or need_final_scan_for_reasoning:
            final_result = extract_tag_blocks(
                raw,
                tag_name=self._config.final_tag,
                open_regex=self._final_open_re,
                close_regex=self._final_close_re,
            )
            if final_result.has_malformed_structure:
                has_malformed_structure = True
            if self._config.strict and final_result.has_malformed_structure:
                raise ValueError(
                    "malformed tag structure detected: "
                    f"tag={final_result.tag} nested={final_result.has_nested_open} "
                    f"stray_close={final_result.has_stray_close} "
                    f"unclosed_open={final_result.has_unclosed_open}"
                )
            final_selection = select_tag_block(final_result, policy=self._config.multi_tag_policy)
            if final_selection:
                selected_final, final_idx = final_selection
                selected_final = selected_final.strip() or None  # empty string -> None
                final_block_answer = selected_final
                if want_final:
                    final_answer = selected_final
                if 0 <= final_idx < len(final_result.block_start_offsets):
                    final_selected_open_start = final_result.block_start_offsets[final_idx]
                if 0 <= final_idx < len(final_result.block_end_offsets):
                    final_selected_close_end = final_result.block_end_offsets[final_idx]
            if want_final and final_answer is None and self._config.fallback_policy != "none":
                final_answer = self._fallback(raw)
            if self._config.capture_diagnostics:
                fields.update(self._format_diagnostics(final_result, raw))
        if want_reasoning:
            if self._config.reasoning_from_outside_final and final_selected_open_start is not None:
                prefix = raw[:final_selected_open_start]
                suffix = raw[final_selected_close_end:] if final_selected_close_end is not None else ""
                prefix_stripped = prefix.strip()
                suffix_stripped = suffix.strip()
                if prefix_stripped and suffix_stripped:
                    reasoning = prefix_stripped + "\n" + suffix_stripped
                elif prefix_stripped:
                    reasoning = prefix_stripped
                elif suffix_stripped:
                    reasoning = suffix_stripped
                else:
                    reasoning = ""
            else:
                reasoning_result = extract_tag_blocks(
                    raw,
                    tag_name=self._config.reasoning_tag,
                    open_regex=self._reasoning_open_re,
                    close_regex=self._reasoning_close_re,
                )
                if reasoning_result.has_malformed_structure:
                    has_malformed_structure = True
                if self._config.strict and reasoning_result.has_malformed_structure:
                    raise ValueError(
                        "malformed tag structure detected: "
                        f"tag={reasoning_result.tag} nested={reasoning_result.has_nested_open} "
                        f"stray_close={reasoning_result.has_stray_close} "
                        f"unclosed_open={reasoning_result.has_unclosed_open}"
                    )
                reasoning_selection = select_tag_block(reasoning_result, policy=self._config.multi_tag_policy)
                if reasoning_selection:
                    reasoning = reasoning_selection[0].strip()
                elif want_reasoning:
                    reasoning = ""
                if self._config.capture_diagnostics:
                    fields.update(self._format_diagnostics(reasoning_result, raw))
            if self._config.reasoning_from_entire_output_when_no_final_answer:
                answer_present_for_policy = final_answer is not None if want_final else final_block_answer is not None
                if not answer_present_for_policy and (reasoning is None or reasoning.strip() == ""):
                    stripped_raw = raw.strip()
                    if stripped_raw:
                        reasoning = stripped_raw
        if self._config.capture_diagnostics:
            fields["is_malformed"] = str(has_malformed_structure).lower()
        return ParsedOutput(
            raw=raw,
            final_answer=final_answer,
            reasoning=reasoning,
            fields=fields,
        )

    @staticmethod
    def _format_diagnostics(
        result: TagBlockExtractionResult,
        raw: str,
    ) -> dict[str, str]:
        """Serialize extraction result into ``ParsedOutput.fields``."""
        base = f"tags/{result.tag}/"
        fields: dict[str, str] = {
            f"{base}open_count": str(result.open_count),
            f"{base}close_count": str(result.close_count),
            f"{base}block_count": str(result.block_count),
            f"{base}has_nested_open": str(result.has_nested_open).lower(),
            f"{base}has_stray_close": str(result.has_stray_close).lower(),
            f"{base}has_unclosed_open": str(result.has_unclosed_open).lower(),
        }
        if result.last_close_end is not None:
            fields[f"{base}last_close_end"] = str(result.last_close_end)
            stops_after = result.stops_after_last_close(raw)
            if stops_after is not None:
                fields[f"{base}stops_after_last_close"] = str(stops_after).lower()
        return fields

    def _fallback(
        self,
        raw: str,
    ) -> str | None:
        """Apply the configured fallback policy when no ``<final>`` block is present."""
        if self._config.fallback_policy == "entire_output":
            stripped = raw.strip()
            return stripped if stripped else None
        if self._config.fallback_policy == "last_line":
            stripped = raw.strip()
            if not stripped:
                return None
            last_line = stripped.splitlines()[-1].strip()
            return last_line if last_line else None
        raise ValueError(f"unknown fallback policy: {self._config.fallback_policy}")
