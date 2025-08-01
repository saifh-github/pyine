import langchain_core.prompts

import pyine.prompts.callable_analysis as callable_analysis
import pyine.prompts.prompt_utils


def test_get_config_and_template():
    config = callable_analysis.get_prompt_config()
    assert isinstance(config, pyine.prompts.prompt_utils.PromptConfig)
    first_example = config.examples[0]
    assert isinstance(first_example, pyine.prompts.prompt_utils.PromptExample)
    assert first_example.input_variables["code"].startswith("class Solution:\n")
    first_example_output = first_example.output
    assert isinstance(first_example_output, callable_analysis.CallableAnalysisResponse)
    assert first_example_output.entrypoint_function_name == "romanToDecimal"
    assert first_example_output.entrypoint_function_arg_names == ["S"]
    assert first_example_output.parent_class_name == "Solution"
    assert first_example_output.parent_class_arg_names == []
    template = callable_analysis.get_prompt_template()
    assert isinstance(template, langchain_core.prompts.PromptTemplate)
    template_str = template.template
    assert template_str.startswith("You are an expert at analyzing Python code.")
    assert template_str.endswith("Now, provide the structured output for this code, and nothing else:\n")


# @@@@@ TODO: add optional tests w/ LLM invocations depending on cluster availability
