import langchain_core.output_parsers
import langchain_core.prompts
import langchain_core.runnables
import pytest

import pyine.prompts.configs.callable_analysis as callable_analysis
import pyine.prompts.manager
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
    config = pyine.prompts.manager.get_prompt_config("callable_analysis")
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
    template = pyine.prompts.manager.get_prompt_template("callable_analysis")
    assert isinstance(template, langchain_core.prompts.PromptTemplate)
    template_str = template.template
    assert template_str.startswith("You are an expert at interpreting and analyzing Python 3 code.")
    assert template_str.endswith("Now, provide the structured output for this code, and nothing else:\n")
    assert template.input_variables == ["code"]
    rendered_str = template.format(
        code='name = input("Enter name: ")\nprint("Hello, " + name)',
    )
    assert "Enter name: " in rendered_str
    # as a bonus here, test that the chat template is also OK
    chat_template = callable_analysis.get_prompt_template(use_chat_template=True)
    assert isinstance(chat_template, langchain_core.prompts.ChatPromptTemplate)
    assert len(chat_template.messages) == 2
    assert isinstance(chat_template.messages[0], langchain_core.prompts.SystemMessagePromptTemplate)
    assert "You are an expert at interpreting" in chat_template.messages[0].prompt.template
    assert isinstance(chat_template.messages[1], langchain_core.prompts.HumanMessagePromptTemplate)
    assert "Now, provide the structured output" in chat_template.messages[1].prompt.template


def test_get_chain(mocker):
    model = mocker.MagicMock()
    chain = pyine.prompts.manager.get_prompt_chain(model, "code_analysis")
    assert isinstance(chain, langchain_core.runnables.RunnableSequence)
    assert isinstance(chain.last, langchain_core.output_parsers.PydanticOutputParser)


# @@@@@ TODO: add optional tests w/ LLM invocations depending on cluster availability
