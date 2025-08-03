import langchain_core.prompts

import pyine.prompts.configs.code_execution as code_execution
import pyine.prompts.utils


def test_get_no_pressure_config_and_template():
    config = code_execution.get_prompt_config(version="no_pressure_demo/v1.0")
    assert isinstance(config, pyine.prompts.utils.PromptConfig)
    first_example = config.examples[0]
    assert isinstance(first_example, pyine.prompts.utils.PromptExample)
    assert first_example.input_variables["code"].startswith('name = input("Enter your name: ")\n')
    assert first_example.input_variables["inputs"] == "Bob"
    assert first_example.output == "Hello, Bob"
    template = code_execution.get_prompt_template(version="no_pressure_demo/v1.0")
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


# @@@@@ TODO: add optional tests w/ LLM invocations depending on cluster availability
