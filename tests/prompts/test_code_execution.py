import enum

import langchain_core.prompts
import pydantic
import pytest

import pyine.prompts.configs.code_execution as code_exec
import pyine.prompts.manager
import pyine.prompts.utils


def test_get_no_pressure_config_and_template() -> None:
    prompt_version = "no_pressure_demo"
    config = pyine.prompts.manager.get_prompt_config("code_execution", version=prompt_version)
    assert isinstance(config, pyine.prompts.utils.PromptConfig)
    first_example = config.examples[0]
    assert isinstance(first_example, pyine.prompts.utils.PromptExample)
    assert first_example.input_variables["code"].startswith('name = input("Enter your name: ")\n')
    assert first_example.input_variables["inputs"] == "Bob"
    assert first_example.output.rstrip("\n") == "Hello, Bob"
    template = pyine.prompts.manager.get_prompt_template("code_execution", version=prompt_version)
    assert isinstance(template, langchain_core.prompts.PromptTemplate)
    template_str = template.template
    assert template_str.startswith("You are an expert at interpreting and executing Python 3 code.")
    assert template_str.endswith("Now, provide your predicted output, and nothing else:\n")
    assert template.input_variables == ["code", "inputs"]
    rendered_str = template.format(
        code='name = input("Enter name: ")\nprint("Hello, " + name)',
        inputs="Bob",
    )
    assert "Enter name: " in rendered_str and "```\nBob\n```" in rendered_str


def test_get_unstructured_with_3_predict_types_config_and_template() -> None:
    prompt_version = "unstructured_with_3_predict_types"
    config = pyine.prompts.manager.get_prompt_config("code_execution", version=prompt_version)
    assert isinstance(config, pyine.prompts.utils.PromptConfig)
    assert config.metadata.name == "code_execution"
    assert config.example_count >= 3
    template = pyine.prompts.manager.get_prompt_template(
        "code_execution", version=prompt_version, include_examples=True
    )
    assert isinstance(template, langchain_core.prompts.PromptTemplate)
    core_vars = {"code", "predict_type", "inputs"}
    assert core_vars.issubset(set(template.input_variables))
    assert template.optional_variables == ["description", "entrypoint"]
    template_str = template.template
    assert template_str.startswith("You are an expert at interpreting and executing Python 3 code.")

    rendered_prog = template.format(
        code='print("Hello, " + input("Enter name: "))',
        description="Greets a user by name.",
        predict_type="program_output",
        inputs="Bobby",
    )
    assert "Type of execution output that should be predicted: program_output" in rendered_prog
    assert "Greets a user by name" in rendered_prog
    assert "Enter name: " in rendered_prog and "Bobby" in rendered_prog
    assert rendered_prog.endswith("Now, provide ONLY the execution output:")

    rendered_vars = template.format(
        code="x=2\ny=3\nz=x+y",
        description="Adds two numbers.",
        predict_type="frame_variables",
        inputs="",
        first_line=1,
        last_line=3,
        first_line_hit=1,
        last_line_hit=2,
    )
    assert "Type of execution output that should be predicted: frame_variables" in rendered_vars
    assert "First line" in rendered_vars and "Last line" in rendered_vars
    assert rendered_vars.endswith("dump that would be obtained after the specified hit of the last line:")

    rendered_ret = template.format(
        code="def add(a,b): return a+b\nresult = add(1,2)",
        description="Simple add function.",
        entrypoint="add",
        predict_type="function_return",
        inputs="a=1, b=2",
        first_line=1,
        last_line=1,
    )
    assert "Type of execution output that should be predicted: function_return" in rendered_ret
    assert "Consider only a call of the following function:" in rendered_ret
    assert "Use the following call input arguments:" in rendered_ret
    assert "a=1, b=2" in rendered_ret
    assert rendered_ret.endswith("Now, provide ONLY the function's returned value(s):")


def test_get_rl_tagged_answer_config_and_template() -> None:
    prompt_version = "rl_tagged_answer"
    config = pyine.prompts.manager.get_prompt_config("code_execution", version=prompt_version)
    assert isinstance(config, pyine.prompts.utils.PromptConfig)
    assert config.metadata.name == "code_execution"
    assert config.example_count == 0  # zero-shot prompt version
    template = pyine.prompts.manager.get_prompt_template(
        "code_execution", version=prompt_version, include_examples=False
    )
    assert isinstance(template, langchain_core.prompts.PromptTemplate)
    template_str = template.template
    assert template_str.startswith("You are an expert at interpreting and executing Python 3 code.")
    assert "<final>" in template_str and "</final>" in template_str
    rendered = template.format(
        code='print("Hello")',
        predict_type="program_output",
        inputs="",
    )
    assert "Task: program_output" in rendered
    assert "Your answer:" in rendered
    assert "print(" in rendered


# @@@@@ TODO: add optional tests w/ LLM invocations depending on cluster availability


class TestCodeExecutionValidator:
    def test_default_config(self) -> None:
        config = code_exec.CodeExecutionValidatorConfig()
        assert config.parsing_mode == code_exec.OutputParsingMode.answer_only
        assert config.strict_frame_variables is True
        assert config.repair_markdown_fences is True

    def test_frozen_config(self) -> None:
        config = code_exec.CodeExecutionValidatorConfig()
        with pytest.raises(pydantic.ValidationError):
            config.parsing_mode = code_exec.OutputParsingMode.reasoning_and_answer  # type: ignore[misc]

    @pytest.fixture
    def validator(self) -> code_exec.CodeExecutionValidator:
        return code_exec.get_code_execution_validator()

    def test_program_output_accepts_anything(self, validator: code_exec.CodeExecutionValidator) -> None:
        result = validator.validate("Hello, World!", "program_output")
        assert result.status == code_exec.ValidationStatus.success
        assert result.answer_text == "Hello, World!"
        assert result.parsed_answer == "Hello, World!"

    def test_program_output_preserves_whitespace(self) -> None:
        validator = code_exec.get_code_execution_validator(
            code_exec.CodeExecutionValidatorConfig(strip_answer_for_program_output=False)
        )
        result = validator.validate("  output with spaces  \n", "program_output")
        assert result.status == code_exec.ValidationStatus.success
        assert "  output with spaces  " in result.answer_text

    def test_frame_variables_requires_dict(self, validator: code_exec.CodeExecutionValidator) -> None:
        result = validator.validate('{"x": 1, "y": 2}', "frame_variables")
        assert result.status == code_exec.ValidationStatus.success
        assert result.parsed_answer == {"x": 1, "y": 2}

    def test_frame_variables_rejects_non_dict(self, validator: code_exec.CodeExecutionValidator) -> None:
        result = validator.validate("[1, 2, 3]", "frame_variables")
        assert result.status == code_exec.ValidationStatus.failed
        assert "must be dict" in (result.error_details or "")

    def test_frame_variables_rejects_invalid_json(self, validator: code_exec.CodeExecutionValidator) -> None:
        result = validator.validate("not valid json", "frame_variables")
        assert result.status == code_exec.ValidationStatus.failed
        assert result.error_details is not None

    def test_function_return_accepts_structured(self, validator: code_exec.CodeExecutionValidator) -> None:
        result = validator.validate("42", "function_return")
        assert result.status == code_exec.ValidationStatus.success
        assert result.parsed_answer == 42

    def test_function_return_accepts_exception_pattern(self, validator: code_exec.CodeExecutionValidator) -> None:
        result = validator.validate("ValueError: invalid input", "function_return")
        assert result.status == code_exec.ValidationStatus.success
        assert "ValueError" in (result.answer_text or "")

    def test_function_return_accepts_raw_repr(self, validator: code_exec.CodeExecutionValidator) -> None:
        # non-strict by default, accepts raw repr strings (downstream users might want to strip addresses...)
        result = validator.validate("<object at 0x1234>", "function_return")
        assert result.status == code_exec.ValidationStatus.success

    def test_enum_predict_type_handling(self, validator: code_exec.CodeExecutionValidator) -> None:
        class PredictType(enum.StrEnum):
            program_output = enum.auto()

        result = validator.validate("test", PredictType.program_output)
        assert result.status == code_exec.ValidationStatus.success
        assert result.predict_type == "program_output"

    def test_repair_markdown_fences(self, validator: code_exec.CodeExecutionValidator) -> None:
        result = validator.validate('```json\n{"key": "value"}\n```', "frame_variables")
        assert result.status == code_exec.ValidationStatus.success
        assert result.parsed_answer == {"key": "value"}
        assert result.diagnostics.get("repair/markdown_fences") is True

    def test_repair_output_prefix(self, validator: code_exec.CodeExecutionValidator) -> None:
        result = validator.validate('Output:\n{"x": 1}', "frame_variables")
        assert result.status == code_exec.ValidationStatus.success
        assert result.parsed_answer == {"x": 1}
        assert result.diagnostics.get("repair/output_prefix") is True

    def test_strip_tags_in_answer_only_mode_enabled_by_default(
        self,
        validator: code_exec.CodeExecutionValidator,
    ) -> None:
        # model includes tags even though parsing_mode=answer_only; should be stripped
        result = validator.validate("<final>Hello, World!</final>", "program_output")
        assert result.status == code_exec.ValidationStatus.success
        assert result.answer_text == "Hello, World!"

    def test_strip_tags_in_answer_only_mode_disabled(self) -> None:
        config = code_exec.CodeExecutionValidatorConfig(strip_tags_in_answer_only_mode=False)
        validator = code_exec.CodeExecutionValidator(config)
        result = validator.validate("<final>Hello, World!</final>", "program_output")
        assert result.status == code_exec.ValidationStatus.success
        assert result.answer_text == "<final>Hello, World!</final>"

    def test_strip_tags_in_answer_only_mode_ignores_multiple_blocks(
        self,
        validator: code_exec.CodeExecutionValidator,
    ) -> None:
        # multiple blocks: don't strip, return original (ambiguous which to pick)
        result = validator.validate("<final>a</final><final>b</final>", "program_output")
        assert result.answer_text == "<final>a</final><final>b</final>"

    def test_strip_tags_in_answer_only_mode_ignores_malformed(
        self,
        validator: code_exec.CodeExecutionValidator,
    ) -> None:
        # malformed tags: don't strip, return original
        result = validator.validate("<final>unclosed", "program_output")
        assert result.answer_text == "<final>unclosed"


class TestCodeExecutionValidatorReasoningModes:
    def test_reasoning_and_answer_extracts_tag(self) -> None:
        config = code_exec.CodeExecutionValidatorConfig(
            parsing_mode=code_exec.OutputParsingMode.reasoning_and_answer,
            answer_tag="answer",
        )
        validator = code_exec.CodeExecutionValidator(config)
        result = validator.validate(
            "Let me think about this...\n<answer>42</answer>",
            "function_return",
        )
        assert result.status == code_exec.ValidationStatus.success
        assert result.answer_text == "42"
        assert result.parsed_answer == 42
        assert result.reasoning_text == "Let me think about this..."

    def test_reasoning_and_answer_fails_without_tag(self) -> None:
        config = code_exec.CodeExecutionValidatorConfig(
            parsing_mode=code_exec.OutputParsingMode.reasoning_and_answer,
        )
        validator = code_exec.CodeExecutionValidator(config)
        result = validator.validate("Just a raw answer without tags", "function_return")
        assert result.status == code_exec.ValidationStatus.failed
        assert "no answer tag found" in (result.error_details or "")

    def test_multi_tag_policy_last(self) -> None:
        config = code_exec.CodeExecutionValidatorConfig(
            parsing_mode=code_exec.OutputParsingMode.reasoning_and_answer,
            answer_tag="final",
            multi_tag_policy="last",
        )
        validator = code_exec.CodeExecutionValidator(config)
        result = validator.validate(
            "<final>first</final> more reasoning <final>second</final>",
            "program_output",
        )
        assert result.answer_text == "second"

    def test_strict_tag_structure_rejects_malformed(self) -> None:
        config = code_exec.CodeExecutionValidatorConfig(
            parsing_mode=code_exec.OutputParsingMode.reasoning_and_answer,
            strict_tag_structure=True,
        )
        validator = code_exec.CodeExecutionValidator(config)
        result = validator.validate("<final>unclosed", "program_output")
        assert result.status == code_exec.ValidationStatus.failed
        assert "malformed" in (result.error_details or "")


class TestValidationStatus:
    def test_status_enum_values(self) -> None:
        assert code_exec.ValidationStatus.success.value == "success"
        assert code_exec.ValidationStatus.partial.value == "partial"
        assert code_exec.ValidationStatus.failed.value == "failed"


class TestCodeExecutionOutput:
    def test_output_properties(self) -> None:
        validation_result = code_exec.CodeExecutionValidationResult(
            raw_output="test output",
            status=code_exec.ValidationStatus.success,
            answer_text="42",
            parsed_answer=42,
            predict_type="function_return",
        )
        output = code_exec.CodeExecutionOutput(
            raw_output="test output",
            validation_result=validation_result,
        )
        assert output.is_valid is True
        assert output.parsed_answer == 42
        assert output.answer_text == "42"
