import langchain_core.prompts

import pyine.prompts.configs.issues.iterators as issues_iterators
import pyine.prompts.manager
import pyine.prompts.utils


def test_get_full_output_config_and_template() -> None:
    config = pyine.prompts.manager.get_prompt_config("issues/iterators", version="full_output/v1.0")
    assert isinstance(config, pyine.prompts.utils.PromptConfig)
    for example in config.examples:
        assert isinstance(example, pyine.prompts.utils.PromptExample)
    assert config.examples[0].output is not None
    assert issues_iterators.invalid_code_token not in config.get_examples_as_text(
        target_examples=[0],
        extra_variables={"invalid_code_token": issues_iterators.invalid_code_token},
    )
    assert config.examples[1].output is None  # invalid input code
    assert (
        config.get_examples_as_text(
            target_examples=[1],
            extra_variables={"invalid_code_token": issues_iterators.invalid_code_token},
        )
        .rstrip("\n")
        .endswith(issues_iterators.invalid_code_token)
    )
    template = pyine.prompts.manager.get_prompt_template("issues/iterators", version="full_output/v1.0")
    assert isinstance(template, langchain_core.prompts.PromptTemplate)
    template_str = template.template
    assert template_str.startswith("You are an expert at interpreting and editing Python 3 code.")
    assert template_str.endswith("Now, generate your proposal, and nothing else:\n")
    assert "**only one micro-region** (1-3 lines)" in template_str
    assert issues_iterators.invalid_code_token in template_str
    assert template.input_variables == ["code"]
    assert template.optional_variables == ["description"]
    rendered_str = template.format(
        code='name = input("Enter name: ")\nprint("Hello, " + name)',
        description="An algorithm.",
    )
    assert "Enter name: " in rendered_str
    assert "An algorithm." in rendered_str


# @@@@@ TODO: add optional tests w/ LLM invocations depending on cluster availability
