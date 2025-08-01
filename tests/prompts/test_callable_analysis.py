import langchain_core.prompts
import pydantic

import pyine.prompts.callable_analysis as callable_analysis


def test_get_config_and_template():
    config = callable_analysis.get_prompt_config()
    assert isinstance(config, pydantic.BaseModel)
    template = callable_analysis.get_prompt_template()
    assert isinstance(template, langchain_core.prompts.PromptTemplate)
    # pretty much all the validation is already done (thanks pydantic)


# @@@@@ TODO: add optional tests w/ LLM invocations depending on cluster availability
