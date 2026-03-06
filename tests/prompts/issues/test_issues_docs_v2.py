import langchain_core.prompts

import pyine.prompts.manager
import pyine.prompts.utils


def test_get_v1_config_and_template() -> None:
    config = pyine.prompts.manager.get_prompt_config("issues/docs_v2", version="full_output/v1.1")
    assert isinstance(config, pyine.prompts.utils.PromptConfig)
    template = pyine.prompts.manager.get_prompt_template("issues/docs_v2", version="full_output/v1.1")
    assert isinstance(template, langchain_core.prompts.PromptTemplate)
    assert template.input_variables == ["code", "expected_output", "inputs"]
    assert template.optional_variables == ["description", "forbidden_execution_output"]
    rendered = template.format(
        code="print('hello')",
        inputs="ignored",
        expected_output="hello",
        forbidden_execution_output="bye",
    )
    assert "Expected execution output" in rendered
    assert "Forbidden execution output" in rendered
