import langchain_core.prompts

import pyine.prompts.configs.issues.todos as issues_todos
import pyine.prompts.utils


def test_get_full_output_config_and_template():
    config = issues_todos.get_prompt_config(version="full_output_1-5L/v1.0")
    assert isinstance(config, pyine.prompts.utils.PromptConfig)
    for example in config.examples:
        assert isinstance(example, pyine.prompts.utils.PromptExample)
        assert ("TODO" in example.output) or ("FIXME" in example.output)
    template = issues_todos.get_prompt_template(version="full_output_1-5L/v1.0")
    assert isinstance(template, langchain_core.prompts.PromptTemplate)
    template_str = template.template
    assert template_str.startswith("You are an expert at interpreting and editing Python 3 code.")
    assert template_str.endswith("Now, generate your proposal, and nothing else:\n")
    assert "(roughly 1-5 lines)" in template_str
    assert template.input_variables == ["code", "description"]
    rendered_str = template.format(
        code='name = input("Enter name: ")\nprint("Hello, " + name)',
        description="An algorithm.",
    )
    assert "Enter name: " in rendered_str
    assert "An algorithm." in rendered_str


# @@@@@ TODO: add optional tests w/ LLM invocations depending on cluster availability
