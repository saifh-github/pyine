import json

import langchain_core.prompts
import pydantic
import pytest

import pyine.prompts.configs.input_output_rewrite as input_output_rewrite


class TestInputOutputRewriteResponse:
    def test_is_valid_with_balanced_inputs_outputs(
        self,
    ) -> None:
        response = input_output_rewrite.InputOutputRewriteResponse(
            inputs=['{"value": 1}'],
            outputs=[1],
            fn_name="Solution.solve",
        )
        assert response.is_valid()

    def test_is_valid_false_when_lengths_mismatch(
        self,
    ) -> None:
        response = input_output_rewrite.InputOutputRewriteResponse(
            inputs=['{"value": 1}'],
            outputs=[],
            fn_name="Solution.solve",
        )
        assert not response.is_valid()

    def test_rejects_non_json_serializable_output(
        self,
    ) -> None:
        with pytest.raises(pydantic.ValidationError):
            input_output_rewrite.InputOutputRewriteResponse(
                inputs=['{"value": 1}'],
                outputs=[{1, 2, 3}],
                fn_name="Solution.solve",
            )

    def test_rejects_invalid_fn_name(
        self,
    ) -> None:
        with pytest.raises(pydantic.ValidationError):
            input_output_rewrite.InputOutputRewriteResponse(
                inputs=['{"value": 1}'],
                outputs=[1],
                fn_name="not a valid name",
            )


class TestPromptHelpers:
    def test_output_parser_round_trip(
        self,
    ) -> None:
        parser = input_output_rewrite.get_output_parser()
        payload = {
            "inputs": ['{"value": 1}'],
            "outputs": [1],
            "fn_name": "Solution.solve",
        }
        parsed = parser.parse(json.dumps(payload))
        assert isinstance(parsed, input_output_rewrite.InputOutputRewriteResponse)
        assert parsed.inputs == payload["inputs"]
        assert parsed.outputs == payload["outputs"]
        assert parsed.fn_name == payload["fn_name"]

    def test_output_parser_round_trip_with_examples_in_prompt(
        self,
    ) -> None:
        template = input_output_rewrite.get_prompt_template(
            include_examples=True,
            target_examples=[1],
        )
        prompt = template.format(
            question="Describe the task.",
            starter_code="def solve(): pass",
            first_solution="def solve(): return 0",
            input_output='{"inputs": ["x = 1"], "outputs": ["1"]}',
        )
        assert "Example 1" in prompt
        assert "Current block:" in prompt

    def test_prompt_template_includes_format_instructions(
        self,
    ) -> None:
        parser = input_output_rewrite.get_output_parser()
        template = input_output_rewrite.get_prompt_template(include_examples=False)
        prompt = template.format(
            question="Describe the task.",
            starter_code="def solve(): pass",
            first_solution="def solve(): return 0",
            input_output='{"inputs": [], "outputs": []}',
        )
        instructions = parser.get_format_instructions()
        assert instructions in prompt
        assert "Problem statement:" in prompt
        assert "Current `input_output` block:" in prompt

    def test_chat_prompt_system_message_includes_instructions(
        self,
    ) -> None:
        parser = input_output_rewrite.get_output_parser()
        chat_template = input_output_rewrite.get_prompt_template(
            use_chat_template=True,
            include_examples=False,
        )
        formatted_messages = chat_template.format_messages(
            question="Describe the task.",
            starter_code="def solve(): pass",
            first_solution="def solve(): return 0",
            input_output='{"inputs": [], "outputs": []}',
        )
        system_message = formatted_messages[0]
        assert isinstance(chat_template, langchain_core.prompts.chat.ChatPromptTemplate)
        assert parser.get_format_instructions() in system_message.content
