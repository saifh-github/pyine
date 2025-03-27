import langchain.schema


_base_system_prompt_str = \
"""You are an expert at interpreting and analyzing Python 3 code.

You only respond in JSON format without any additional information.
"""

base_system_prompt = langchain.schema.SystemMessage(
    content=_base_system_prompt_str,
)
