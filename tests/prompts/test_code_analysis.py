import langchain_core.output_parsers
import langchain_core.prompts
import langchain_core.runnables
import pytest_mock

import pyine.prompts.configs.code_analysis as code_analysis
import pyine.prompts.manager
import pyine.prompts.utils


def test_get_config_and_template() -> None:
    config = pyine.prompts.manager.get_prompt_config("code_analysis")
    assert isinstance(config, pyine.prompts.utils.PromptConfig)
    first_example = config.examples[0]
    assert isinstance(first_example, pyine.prompts.utils.PromptExample)
    assert first_example.input_variables["code"].startswith("import math\n")
    first_example_output = first_example.output
    assert isinstance(first_example_output, code_analysis.CodeAnalysisResponse)
    assert first_example_output.code_type == "simple-with-declarations"
    assert first_example_output.input_type == "stdin"
    assert first_example_output.output_type == "stdout"
    template = pyine.prompts.manager.get_prompt_template("code_analysis")
    assert isinstance(template, langchain_core.prompts.PromptTemplate)
    template_str = template.template
    assert template_str.startswith("You are an expert at interpreting and analyzing Python 3 code.")
    assert template_str.endswith("Now, provide the structured output for this code, and nothing else:\n")
    assert template.input_variables == ["code"]
    rendered_str = template.format(
        code='name = input("Enter name: ")\nprint("Hello, " + name)',
    )
    assert "Enter name: " in rendered_str


def test_get_chain(
    mocker: pytest_mock.MockerFixture,
) -> None:
    model = mocker.MagicMock()
    chain = pyine.prompts.manager.get_prompt_chain(model, "code_analysis")
    assert isinstance(chain, langchain_core.runnables.RunnableSequence)
    assert isinstance(chain.last, langchain_core.output_parsers.PydanticOutputParser)
