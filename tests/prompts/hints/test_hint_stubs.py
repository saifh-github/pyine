import langchain_core.prompts

import pyine.prompts.manager
import pyine.prompts.utils


def test_get_full_output_config_and_template():
    config = pyine.prompts.manager.get_prompt_config("hints/stubs", version="full_output/v1.0")
    assert isinstance(config, pyine.prompts.utils.PromptConfig)
    pyi_example = config.examples[4]
    assert isinstance(pyi_example, pyine.prompts.utils.PromptExample)
    assert "class MiniCanny:" in pyi_example.input_variables["code"]
    assert "minicanny_core.pyi" in pyi_example.input_variables["modified_code"]
    assert "minicanny_core.pyi" in pyi_example.output
    template = pyine.prompts.manager.get_prompt_template("hints/stubs", version="full_output/v1.0")
    assert isinstance(template, langchain_core.prompts.PromptTemplate)
    template_str = template.template
    assert template_str.startswith("You are an expert at interpreting and editing Python 3 code.")
    assert template_str.endswith("Now, generate your proposal, and nothing else:\n")
    assert "The code to target for 'stubbing' should" in template_str
    assert template.input_variables == ["code", "description"]
    rendered_str = template.format(
        code='name = input("Enter name: ")\nprint("Hello, " + name)',
        description="An algorithm.",
    )
    assert "Enter name: " in rendered_str
    assert "An algorithm." in rendered_str


# @@@@@ TODO: add optional tests w/ LLM invocations depending on cluster availability
