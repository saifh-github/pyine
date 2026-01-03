"""Output parsing helpers for reward terms.

Parsing is optional and is only used for terms that require structured access to a model output
(e.g., distinguishing "reasoning" from the "final answer"). The manager can parse once per sample
and share the parsed result across all terms.
"""

import pyine.organisms.models.rewards.core.configs as reward_configs
import pyine.organisms.models.rewards.core.types as reward_types
import pyine.utils.parsing


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
        self._final_open_re = pyine.utils.parsing.compile_open_tag_regex(config.final_tag)
        self._final_close_re = pyine.utils.parsing.compile_close_tag_regex(config.final_tag)
        self._reasoning_open_re = pyine.utils.parsing.compile_open_tag_regex(config.reasoning_tag)
        self._reasoning_close_re = pyine.utils.parsing.compile_close_tag_regex(config.reasoning_tag)

    @property
    def final_tag(self) -> str:
        """The tag name used for final answer extraction."""
        return self._config.final_tag

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
        want_final = self._config.enabled_fields in ("both", "final_only")
        want_reasoning = self._config.enabled_fields in ("both", "reasoning_only")
        need_final_scan_for_reasoning = bool(self._config.reasoning_from_final_prefix and want_reasoning)
        final_selected_open_start: int | None = None

        has_malformed_structure = False
        if want_final or need_final_scan_for_reasoning:
            final_result = pyine.utils.parsing.extract_tag_blocks(
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
            final_selection = pyine.utils.parsing.select_tag_block(final_result, policy=self._config.multi_tag_policy)
            if final_selection:
                selected_final, final_idx = final_selection
                selected_final = selected_final.strip() or None  # empty string -> None
                if want_final:
                    final_answer = selected_final
                if 0 <= final_idx < len(final_result.block_start_offsets):
                    final_selected_open_start = final_result.block_start_offsets[final_idx]
            if want_final and final_answer is None and self._config.fallback_policy != "none":
                final_answer = self._fallback(raw)
            if self._config.capture_diagnostics:
                fields.update(self._format_diagnostics(final_result, raw))

        if want_reasoning:
            if self._config.reasoning_from_final_prefix and final_selected_open_start is not None:
                prefix = raw[:final_selected_open_start]
                stripped = prefix.strip()
                reasoning = stripped if stripped else None
            else:
                reasoning_result = pyine.utils.parsing.extract_tag_blocks(
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
                reasoning_selection = pyine.utils.parsing.select_tag_block(
                    reasoning_result, policy=self._config.multi_tag_policy
                )
                if reasoning_selection:
                    reasoning = reasoning_selection[0].strip() or None  # empty string -> None
                if self._config.capture_diagnostics:
                    fields.update(self._format_diagnostics(reasoning_result, raw))

        if self._config.capture_diagnostics:
            fields["is_malformed"] = str(has_malformed_structure).lower()
        return reward_types.ParsedOutput(
            raw=raw,
            final_answer=final_answer,
            reasoning=reasoning,
            fields=fields,
        )

    @staticmethod
    def _format_diagnostics(
        result: pyine.utils.parsing.TagBlockExtractionResult,
        raw: str,
    ) -> dict[str, str]:
        """Serialize extraction result into `ParsedOutput.fields`."""
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
