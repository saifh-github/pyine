"""Output parsing helpers for reward terms.

Parsing is optional and is only used for terms that require structured access to a model output
(e.g., distinguishing "reasoning" from the "final answer"). The manager can parse once per sample
and share the parsed result across all terms.
"""

import dataclasses
import re

import pyine.organisms.models.rewards.core.configs as reward_configs
import pyine.organisms.models.rewards.core.types as reward_types


@dataclasses.dataclass(frozen=True, slots=True)
class _TagParseDiagnostics:
    """Holds post-parsing diagnostics for a single tag."""

    tag: str
    """Normalized tag name used for parsing."""
    open_count: int
    """Number of opening tags observed."""
    close_count: int
    """Number of closing tags observed."""
    block_count: int
    """Number of extracted `<tag>...</tag>` blocks."""
    has_nested_open: bool
    """Whether an opening tag appeared while already inside a block."""
    has_stray_close: bool
    """Whether a closing tag appeared without a corresponding open."""
    has_unclosed_open: bool
    """Whether an open tag remained unclosed at end of string."""
    last_close_end: int | None
    """End offset of the last closing tag, if any."""
    stops_after_last_close: bool | None
    """Whether the output stops (ignoring whitespace) after the last closing tag, if any."""


class TagsOutputParser:
    """Extracts "final answer" and "reasoning" XML-tagged blocks from model outputs.

    This parser is intentionally minimal and deterministic. The tag names and fallback policy are
    controlled by `ParsingConfig`.
    """

    def __init__(
        self,
        config: reward_configs.ParsingConfig,
    ) -> None:
        """Create a tag-based parser from configuration.

        Args:
            config: Parsing configuration (tag names + fallback policy).
        """
        if config.mode != "tags":
            raise ValueError(f"unsupported parsing mode: {config.mode}")
        self._config = config
        self._final_open_re = self._compile_open_tag_regex(config.final_tag)
        self._final_close_re = self._compile_close_tag_regex(config.final_tag)
        self._reasoning_open_re = self._compile_open_tag_regex(config.reasoning_tag)
        self._reasoning_close_re = self._compile_close_tag_regex(config.reasoning_tag)

    @staticmethod
    def _compile_open_tag_regex(tag: str) -> re.Pattern[str]:
        """Compile a case-insensitive regex matching `<tag ...>` opening tags."""
        tag_stripped = tag.strip()
        if not tag_stripped:
            raise ValueError("tag name cannot be empty")
        return re.compile(rf"<{re.escape(tag_stripped)}(?:\s[^>]*)?>", re.IGNORECASE)

    @staticmethod
    def _compile_close_tag_regex(tag: str) -> re.Pattern[str]:
        """Compile a case-insensitive regex matching `</tag>` closing tags."""
        tag_stripped = tag.strip()
        if not tag_stripped:
            raise ValueError("tag name cannot be empty")
        return re.compile(rf"</{re.escape(tag_stripped)}\s*>", re.IGNORECASE)

    def parse(
        self,
        prompt: str,
        model_output: str,
    ) -> reward_types.ParsedOutput:
        """Parse the model output into a structured representation.

        Args:
            prompt: Prompt text (currently unused; kept for interface symmetry and future heuristics).
            model_output: Raw model output string.

        Returns:
            Parsed output containing extracted fields.
        """
        del prompt  # unused
        raw = model_output
        final_answer: str | None = None
        reasoning: str | None = None
        fields: dict[str, str] = {}
        if self._config.enabled_fields in ("both", "final_only"):
            final_blocks, final_diag = self._extract_tag_blocks(
                raw,
                open_re=self._final_open_re,
                close_re=self._final_close_re,
            )
            final_answer = self._select_block(final_blocks, field_name="final_answer")
            if final_answer is None and self._config.fallback_policy != "none":
                final_answer = self._fallback(raw)
            if self._config.capture_diagnostics:
                fields.update(self._format_diagnostics(final_diag))
        if self._config.enabled_fields in ("both", "reasoning_only"):
            reasoning_blocks, reasoning_diag = self._extract_tag_blocks(
                raw,
                open_re=self._reasoning_open_re,
                close_re=self._reasoning_close_re,
            )
            reasoning = self._select_block(reasoning_blocks, field_name="reasoning")
            if self._config.capture_diagnostics:
                fields.update(self._format_diagnostics(reasoning_diag))
        return reward_types.ParsedOutput(
            raw=raw,
            final_answer=final_answer,
            reasoning=reasoning,
            fields=fields,
        )

    def _select_block(
        self,
        blocks: list[str],
        *,
        field_name: str,
    ) -> str | None:
        """Select a single block according to the configured multi-tag policy."""
        if not blocks:
            return None
        policy = self._config.multi_tag_policy
        if policy == "last":
            selected = blocks[-1]
        elif policy == "first":
            selected = blocks[0]
        elif policy == "error":
            if len(blocks) != 1:
                raise ValueError(f"multiple blocks found for {field_name}: {len(blocks)}")
            selected = blocks[0]
        else:
            raise ValueError(f"unknown multi_tag_policy: {policy}")
        stripped = selected.strip()
        return stripped if stripped else None

    def _extract_tag_blocks(
        self,
        raw: str,
        *,
        open_re: re.Pattern[str],
        close_re: re.Pattern[str],
    ) -> tuple[list[str], _TagParseDiagnostics]:
        """Extract `<tag>...</tag>` blocks using a single-pass state machine."""
        events: list[tuple[int, str, re.Match[str]]] = []
        open_matches = list(open_re.finditer(raw))
        close_matches = list(close_re.finditer(raw))
        for match in open_matches:
            events.append((match.start(), "open", match))
        for match in close_matches:
            events.append((match.start(), "close", match))
        events.sort(key=lambda item: item[0])
        in_block = False
        curr_open_end: int | None = None
        blocks: list[str] = []
        has_nested_open = False
        has_stray_close = False
        has_unclosed_open = False
        last_close_end: int | None = None
        for _, kind, match in events:
            if kind == "open":
                if in_block:
                    has_nested_open = True
                    continue
                in_block = True
                curr_open_end = match.end()
            else:
                last_close_end = match.end()
                if not in_block or curr_open_end is None:
                    has_stray_close = True
                    continue
                blocks.append(raw[curr_open_end : match.start()])
                in_block = False
                curr_open_end = None
        if in_block:
            has_unclosed_open = True
        stops_after_last_close: bool | None = None
        if last_close_end is not None:
            stops_after_last_close = raw[last_close_end:].strip() == ""
        diag = _TagParseDiagnostics(
            tag=self._deduce_tag_name(open_re),
            open_count=len(open_matches),
            close_count=len(close_matches),
            block_count=len(blocks),
            has_nested_open=has_nested_open,
            has_stray_close=has_stray_close,
            has_unclosed_open=has_unclosed_open,
            last_close_end=last_close_end,
            stops_after_last_close=stops_after_last_close,
        )
        if self._config.strict and (has_nested_open or has_stray_close or has_unclosed_open):
            raise ValueError(
                "malformed tag structure detected: "
                f"tag={diag.tag} nested={has_nested_open} stray_close={has_stray_close} "
                f"unclosed_open={has_unclosed_open}"
            )
        return blocks, diag

    @staticmethod
    def _deduce_tag_name(
        open_re: re.Pattern[str],
    ) -> str:
        """Best-effort extraction of the tag name from the compiled open tag regex."""
        pattern = open_re.pattern
        # pattern is of the form `<TAG(?:\s[^>]*)?>` with escaped tag.
        if pattern.startswith("<") and "(?:" in pattern:
            inner = pattern[1:].split("(?:", 1)[0]
            return inner.replace("\\", "")
        if pattern.startswith("<"):
            return pattern[1:].split(">", 1)[0].replace("\\", "")
        return "unknown"

    @staticmethod
    def _format_diagnostics(
        diag: _TagParseDiagnostics,
    ) -> dict[str, str]:
        """Serialize diagnostics into `ParsedOutput.fields`."""
        base = f"tags/{diag.tag}/"
        fields: dict[str, str] = {
            f"{base}open_count": str(diag.open_count),
            f"{base}close_count": str(diag.close_count),
            f"{base}block_count": str(diag.block_count),
            f"{base}has_nested_open": str(diag.has_nested_open).lower(),
            f"{base}has_stray_close": str(diag.has_stray_close).lower(),
            f"{base}has_unclosed_open": str(diag.has_unclosed_open).lower(),
        }
        if diag.last_close_end is not None:
            fields[f"{base}last_close_end"] = str(diag.last_close_end)
        if diag.stops_after_last_close is not None:
            fields[f"{base}stops_after_last_close"] = str(diag.stops_after_last_close).lower()
        return fields

    def _fallback(
        self,
        raw: str,
    ) -> str | None:
        """Apply the configured fallback policy when no `<final>` block is present."""
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
