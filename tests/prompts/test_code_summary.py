import langchain_core.prompts
import langchain_core.runnables

import pyine.prompts.manager
import pyine.prompts.utils


def test_get_config_and_template() -> None:
    config = pyine.prompts.manager.get_prompt_config("code_summary")
    assert isinstance(config, pyine.prompts.utils.PromptConfig)
    assert config.metadata.name == "code_summary"
    assert config.example_count >= 2
    first_example = config.examples[0]
    assert isinstance(first_example, pyine.prompts.utils.PromptExample)
    assert "name = input" in first_example.input_variables["code"]
    assert isinstance(first_example.output, str)
    template = pyine.prompts.manager.get_prompt_template("code_summary", include_examples=True)
    assert isinstance(template, langchain_core.prompts.PromptTemplate)
    template_str = template.template
    assert template_str.startswith("You are an expert at interpreting and summarizing Python 3 code")
    assert template_str.endswith("Now, provide your summary of the code snippet, and nothing else:\n")
    assert "code" in template.input_variables
    assert "target_word_count" in template.optional_variables
    assert "description" in template.optional_variables
    rendered_str = template.format(
        code='print("Hello")',
    )
    assert "Hello" in rendered_str
    assert "Coding problem description (to" not in rendered_str
    assert "Your summary for this specific code snippet should be roughly" not in rendered_str
    rendered_str_with_extras = template.format(
        code='print("Hello")',
        target_word_count=25,
        description="Greet the user",
    )
    assert "Coding problem description (to" in rendered_str_with_extras
    assert "Greet the user" in rendered_str_with_extras
    assert "25 words" in rendered_str_with_extras
