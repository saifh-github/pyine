import langchain_core.prompts

import pyine.prompts.configs.hints.docs as hints_docs
import pyine.prompts.utils


def test_get_full_output_config_and_template():
    config = hints_docs.get_prompt_config(version="full_output/v1.0")
    assert isinstance(config, pyine.prompts.utils.PromptConfig)
    last_example = config.examples[-1]
    assert isinstance(last_example, pyine.prompts.utils.PromptExample)
    assert last_example.input_variables["code"].startswith('"""GCD algorithm."""')
    assert last_example.input_variables["inputs"] == "{x=230, y=160}"
    assert last_example.input_variables["expected_exec_output"] == "10"
    assert ">>> Z.g(230, 160)\n        10" in last_example.output
    template = hints_docs.get_prompt_template(version="full_output/v1.0")
    assert isinstance(template, langchain_core.prompts.PromptTemplate)
    template_str = template.template
    assert template_str.startswith("You are an expert at interpreting and editing Python 3 code.")
    assert template_str.endswith("Now, generate your proposal, and nothing else:\n")
    assert "Proposed code with hints (i.e. what you should be trying to generate)" in template_str
    assert template.input_variables == ["code", "description", "expected_exec_output", "inputs"]
    rendered_str = template.format(
        code='name = input("Enter name: ")\nprint("Hello, " + name)',
        inputs="Bob",
        expected_exec_output="Hello, Bob",
        description="An algorithm.",
    )
    assert "Enter name: " in rendered_str
    assert "```\nBob\n```" in rendered_str
    assert "An algorithm." in rendered_str
    assert "An algorithm" in rendered_str


# @@@@@ TODO: add optional tests w/ LLM invocations depending on cluster availability
