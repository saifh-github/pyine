import langchain_core.prompts
import pytest

import pyine.prompts.configs.callable_analysis as callable_analysis
import pyine.prompts.utils


def test_callable_analysis_response_validation():
    response = callable_analysis.CallableAnalysisResponse(
        entrypoint_function_name="test_func",
        entrypoint_function_arg_names=[],
        parent_class_name=None,
        parent_class_arg_names=None,
    )
    assert response.entrypoint_function_name == "test_func"
    assert len(response.entrypoint_function_arg_names) == 0
    response = callable_analysis.CallableAnalysisResponse(
        entrypoint_function_name="test_func",
        entrypoint_function_arg_names=["arg1", "arg2"],
        parent_class_name=None,
        parent_class_arg_names=None,
    )
    assert len(response.entrypoint_function_arg_names) == 2
    response = callable_analysis.CallableAnalysisResponse(
        entrypoint_function_name="test_func",
        entrypoint_function_arg_names=["arg1", "arg2"],
        parent_class_name="Algo",
        parent_class_arg_names=[],
    )
    assert response.parent_class_name == "Algo"
    with pytest.raises(ValueError):
        callable_analysis.CallableAnalysisResponse(
            entrypoint_function_name="",
            entrypoint_function_arg_names=[],
            parent_class_name="TestClass",
            parent_class_arg_names=[],
        )


def test_get_config_and_template():
    config = callable_analysis.get_prompt_config()
    assert isinstance(config, pyine.prompts.utils.PromptConfig)
    first_example = config.examples[0]
    assert isinstance(first_example, pyine.prompts.utils.PromptExample)
    assert first_example.input_variables["code"].startswith("class Solution:\n")
    first_example_output = first_example.output
    assert isinstance(first_example_output, callable_analysis.CallableAnalysisResponse)
    assert first_example_output.entrypoint_function_name == "romanToDecimal"
    assert first_example_output.entrypoint_function_arg_names == ["S"]
    assert first_example_output.parent_class_name == "Solution"
    assert first_example_output.parent_class_arg_names == []
    template = callable_analysis.get_prompt_template()
    assert isinstance(template, langchain_core.prompts.PromptTemplate)
    template_str = template.template
    assert template_str.startswith("You are an expert at interpreting and analyzing Python 3 code.")
    assert template_str.endswith("Now, provide the structured output for this code, and nothing else:\n")
    assert template.input_variables == ["code"]
    rendered_str = template.format(
        code='name = input("Enter name: ")\nprint("Hello, " + name)',
    )
    assert "Enter name: " in rendered_str


# @@@@@ TODO: add optional tests w/ LLM invocations depending on cluster availability
