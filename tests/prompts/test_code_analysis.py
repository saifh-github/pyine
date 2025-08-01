import langchain_core.prompts

import pyine.prompts.code_analysis as code_analysis
import pyine.prompts.prompt_utils


def test_get_config_and_template():
    config = code_analysis.get_prompt_config()
    assert isinstance(config, pyine.prompts.prompt_utils.PromptConfig)
    first_example = config.examples[0]
    assert isinstance(first_example, pyine.prompts.prompt_utils.PromptExample)
    assert first_example.input_variables["code"].startswith("import math\n")
    first_example_output = first_example.output
    assert isinstance(first_example_output, code_analysis.CodeAnalysisResponse)
    assert first_example_output.code_type == "simple-with-declarations"
    assert first_example_output.input_type == "stdin"
    assert first_example_output.output_type == "stdout"
    template = code_analysis.get_prompt_template()
    assert isinstance(template, langchain_core.prompts.PromptTemplate)
    template_str = template.template
    assert template_str.startswith("You are an expert at interpreting and analyzing Python 3 code.")
    assert template_str.endswith("Now, provide the structured output for this code, and nothing else:\n")


# @@@@@ TODO: add optional tests w/ LLM invocations depending on cluster availability
