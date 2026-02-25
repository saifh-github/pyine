import langchain_core.prompts

import pyine.prompts.manager
import pyine.prompts.utils


def test_get_full_output_config_and_template() -> None:
    config = pyine.prompts.manager.get_prompt_config("hints/tests", version="full_output/v1.0")
    assert isinstance(config, pyine.prompts.utils.PromptConfig)
    # example_2 = config.examples[1]
    # assert isinstance(example_2, pyine.prompts.utils.PromptExample)
    # assert example_2.input_variables["todo"] == todo
    # assert todo in example_2.output
    template = pyine.prompts.manager.get_prompt_template("hints/tests", version="full_output/v1.0")
    assert isinstance(template, langchain_core.prompts.PromptTemplate)
    template_str = template.template
    assert template_str.startswith("You are an expert at interpreting and editing Python 3 code.")
    assert template_str.endswith("Now, generate your proposal, and nothing else:\n")
    assert "modifications you propose should focus on **runtime-related artefacts**" in template_str
    assert template.input_variables == ["code", "expected_output", "inputs"]
    assert template.optional_variables == ["description"]
    rendered_str = template.format(
        code='name = input("Enter name: ")\nprint("Hello, " + name)',
        inputs="Bob",
        expected_output="Hello, Bob",
        description="An algorithm.",
    )
    assert "Enter name: " in rendered_str
    assert "```\nBob\n```" in rendered_str
    assert "An algorithm." in rendered_str
    assert "An algorithm" in rendered_str
