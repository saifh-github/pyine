"""Tests for the correctness judge prompt template and output parser."""

from __future__ import annotations

import json

import langchain_core.output_parsers

import pyine.prompts.manager
from pyine.prompts.configs.guardrail.correctness_judge import (
    CorrectnessJudgement,
    CorrectnessJudgementWithReasoning,
    get_output_parser,
)


class TestPromptTemplateLoads:
    def test_prompt_template_loads(self) -> None:
        """Verify get_prompt_config("guardrail/correctness_judge") succeeds."""
        config = pyine.prompts.manager.get_prompt_config("guardrail/correctness_judge")
        assert config is not None

    def test_prompt_template_loads_with_reasoning(self) -> None:
        config = pyine.prompts.manager.get_prompt_config(
            "guardrail/correctness_judge", version="with_reasoning"
        )
        assert config is not None

    def test_prompt_template_loads_score_only(self) -> None:
        config = pyine.prompts.manager.get_prompt_config(
            "guardrail/correctness_judge", version="score_only"
        )
        assert config is not None


class TestPromptTemplateRenders:
    def test_prompt_template_renders(self) -> None:
        """Verify the template renders with sample input variables."""
        from pyine.prompts.configs.guardrail.correctness_judge import get_prompt_template

        template = get_prompt_template(use_chat_template=True)
        # Render with sample variables
        rendered = template.format(
            model_output="The output is 42",
            expected_output="42",
        )
        assert "The output is 42" in rendered
        assert "42" in rendered

    def test_prompt_template_renders_with_final_answer(self) -> None:
        from pyine.prompts.configs.guardrail.correctness_judge import get_prompt_template

        template = get_prompt_template(use_chat_template=True)
        rendered = template.format(
            model_output="The output is 42",
            expected_output="42",
            final_answer="42",
        )
        assert "42" in rendered


class TestPromptOutputParser:
    def test_prompt_output_parser_returns_pydantic_parser(self) -> None:
        parser = get_output_parser()
        assert isinstance(parser, langchain_core.output_parsers.PydanticOutputParser)

    def test_prompt_output_parser_parses_valid_json(self) -> None:
        parser = get_output_parser()
        assert parser is not None
        valid_json = json.dumps({"score": 0.85, "reasoning": "Looks correct"})
        result = parser.parse(valid_json)
        assert isinstance(result, CorrectnessJudgementWithReasoning)
        assert result.score == 0.85
        assert result.reasoning == "Looks correct"

    def test_prompt_output_parser_score_only_version(self) -> None:
        parser = get_output_parser(version="score_only")
        assert parser is not None
        valid_json = json.dumps({"score": 0.5})
        result = parser.parse(valid_json)
        assert isinstance(result, CorrectnessJudgement)
        assert result.score == 0.5

    def test_prompt_output_parser_with_reasoning_version(self) -> None:
        parser = get_output_parser(version="with_reasoning")
        assert parser is not None
        valid_json = json.dumps({"score": 1.0, "reasoning": "Exact match"})
        result = parser.parse(valid_json)
        assert isinstance(result, CorrectnessJudgementWithReasoning)
        assert result.score == 1.0
