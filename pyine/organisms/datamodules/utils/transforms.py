from __future__ import annotations

import collections.abc
import functools
import typing

import langchain_core.messages
import langchain_core.prompts
import langchain_core.prompts.chat

import pyine.organisms.datamodules.samples
import pyine.prompts.manager

SampleTransformInputType = pyine.organisms.datamodules.samples.SampleData | dict[str, typing.Any]
SampleTransformOutputType = str | list[langchain_core.messages.BaseMessage] | dict[str, typing.Any]
SampleTransformType = typing.Callable[[SampleTransformInputType], SampleTransformOutputType]


def _extract_message_content(
    message: langchain_core.messages.BaseMessage,
    role: str,
) -> str:
    """Returns the string content of a langchain message, enforcing expected typing."""
    content = typing.cast("typing.Any", message.content)  # type: ignore[reportUnknownMemberType]
    if isinstance(content, str):
        return content
    raise TypeError(f"expected string {role} message content, got {type(content)}")


def _apply_prompt_template_to_sample(
    sample: SampleTransformInputType,
    prompt_template: langchain_core.prompts.BasePromptTemplate[typing.Any],
    use_chat_template: bool,
    append_answer: bool,
    use_hf_messages: bool,
    hf_messages_key: str,
    orig_sample_key: str | None,
    merge_system_with_user: bool,
) -> SampleTransformOutputType:
    """Applies a prompt template to a data sample.

    See `create_sample_transform` for information on arguments.

    Kept static/top-level-friendly for to keep pickling happy.
    """
    sample_data: pyine.organisms.datamodules.samples.SampleData
    sample_args: dict[str, typing.Any]
    if isinstance(sample, collections.abc.Mapping):
        sample_dict = dict(sample)
        sample_args = sample_dict
        sample_data = pyine.organisms.datamodules.samples.SampleData(**sample_dict)
    else:
        sample_data = sample
        sample_args = sample_data._asdict()
    if use_chat_template:
        assert isinstance(prompt_template, langchain_core.prompts.chat.ChatPromptTemplate)
        messages = prompt_template.format_messages(**sample_args)
        assert isinstance(messages, list)
        # note: we currently only support single-turn interactions here, so one request per convo
        assert sum(isinstance(m, langchain_core.messages.SystemMessage) for m in messages) <= 1
        assert sum(isinstance(m, langchain_core.messages.HumanMessage) for m in messages) == 1
        assert sum(isinstance(m, langchain_core.messages.AIMessage) for m in messages) == 0
        assert len(messages) in [1, 2]  # we currently only support single-turn transforms here
        if merge_system_with_user and len(messages) == 2:
            system_msg, user_msg = messages
            assert isinstance(system_msg, langchain_core.messages.SystemMessage)
            assert isinstance(user_msg, langchain_core.messages.HumanMessage)
            system_content = _extract_message_content(system_msg, "system")
            user_content = _extract_message_content(user_msg, "human")
            merged_messages: list[langchain_core.messages.BaseMessage] = [
                langchain_core.messages.HumanMessage(system_content + "\n\n" + user_content)
            ]
            messages = merged_messages
        if append_answer:
            messages.append(langchain_core.messages.AIMessage(sample_data.expected_output))
        if use_hf_messages:
            hf_messages: list[dict[str, str]] = []
            for msg in messages:
                if isinstance(msg, langchain_core.messages.SystemMessage):
                    content = _extract_message_content(msg, "system")
                    hf_messages.append({"role": "system", "content": content})
                elif isinstance(msg, langchain_core.messages.HumanMessage):
                    content = _extract_message_content(msg, "human")
                    hf_messages.append({"role": "user", "content": content})
                elif isinstance(msg, langchain_core.messages.AIMessage):
                    content = _extract_message_content(msg, "assistant")
                    hf_messages.append({"role": "assistant", "content": content})
                else:
                    raise NotImplementedError(f"unsupported message type: {type(msg)}")
            hf_output: dict[str, typing.Any] = {hf_messages_key: hf_messages}
            if orig_sample_key:
                hf_output[orig_sample_key] = sample_args
            return hf_output
        return messages
    formatted_output = prompt_template.format(**sample_args)
    assert isinstance(formatted_output, str)
    if append_answer:
        formatted_output = f"{formatted_output}\n{sample_data.expected_output}"
    return formatted_output


def create_sample_transform(
    use_chat_template: bool,
    append_answer: bool,
    use_hf_messages: bool = False,
    hf_messages_key: str = "messages",
    orig_sample_key: str | None = None,
    merge_system_with_user: bool = False,
    **prompt_kwargs: typing.Any,  # forwarded to the prompt manager template getter
) -> SampleTransformType:
    """Create a transform function applying a prompt template to a code execution data sample.

    Args:
        use_chat_template: whether to use a chat template or a regular template for sample formatting.
        append_answer: whether to append the assistant's response to the conversation messages.
        use_hf_messages: whether to use HuggingFace messages format or the langchain format.
        hf_messages_key: if storing messages in HF format, the key to use for the messages.
        orig_sample_key: if storing samples in HF format, the optional key to use for the orig samples.
        merge_system_with_user: whether to merge the system message with the user message (used
            when working with e.g. o1/o3/o4, which do not support custom system prompts). Has no
            effect when `use_chat_template` is False.

    Returns:
        A transform function that converts a SampleData object to a string or list of messages.
    """
    prompt_template = pyine.prompts.manager.get_prompt_template(
        use_chat_template=use_chat_template,
        **prompt_kwargs,
    )
    if use_chat_template:
        assert hasattr(prompt_template, "format_messages"), "wrong template format used"
    else:
        assert hasattr(prompt_template, "format"), "wrong template format used"
    return functools.partial(
        _apply_prompt_template_to_sample,
        prompt_template=prompt_template,
        use_chat_template=use_chat_template,
        append_answer=append_answer,
        use_hf_messages=use_hf_messages,
        hf_messages_key=hf_messages_key,
        orig_sample_key=orig_sample_key,
        merge_system_with_user=merge_system_with_user,
    )


def _apply_rl_prompt_template_to_sample(
    sample: SampleTransformInputType,
    prompt_template: langchain_core.prompts.BasePromptTemplate[typing.Any],
) -> dict[str, typing.Any]:
    """Applies a prompt template to a sample for RL training format.

    This function formats samples for reinforcement learning training by providing
    raw prompts (not tokenized) along with metadata needed for reward computation.

    Args:
        sample: The sample data to transform (SampleData or dict).
        prompt_template: The prompt template to use for formatting.

    Returns:
        Dictionary with 'prompt' (chat message format) and metadata fields.
    """
    # Handle both dict and SampleData inputs
    sample_data: pyine.organisms.datamodules.samples.SampleData
    sample_args: dict[str, typing.Any]
    if isinstance(sample, collections.abc.Mapping):
        sample_dict = dict(sample)
        sample_args = sample_dict
        sample_data = pyine.organisms.datamodules.samples.SampleData(**sample_dict)
    else:
        sample_data = sample
        sample_args = sample_data._asdict()

    # Format the prompt as plain text
    formatted_prompt = prompt_template.format(**sample_args)
    assert isinstance(formatted_prompt, str), "RL training requires plain text prompts"

    # Return dataset entry with all necessary fields for RL training
    # TRL trainers expect 'prompt' to be a list of chat messages
    return {
        "prompt": [{"role": "user", "content": formatted_prompt}],
        "expected_output": sample_data.expected_output,
        "predict_type": sample_data.predict_type,
        "identifier": sample_data.identifier,
        "code_type": sample_data.code_type,
        "tags": sample_data.get_tag_list(),
        "first_line": sample_data.first_line,
        "last_line": sample_data.last_line,
        "entrypoint": sample_data.entrypoint,
    }


def create_rl_sample_transform(
    **prompt_kwargs: typing.Any,  # forwarded to the prompt manager template getter
) -> SampleTransformType:
    """Create a transform function for RL training data preparation.

    This function creates a transform that formats samples for reinforcement learning
    training (e.g., GRPO, PPO, RLOO) by providing raw prompts with metadata needed
    for reward computation.

    Args:
        **prompt_kwargs: Keyword arguments forwarded to the prompt manager, including:
            - version: Version of the prompt template to use
            - include_examples: Whether to include few-shot examples (False recommended for RL)
            - prompt_name: Name of the prompt to use
            - use_chat_template: Whether to use chat template (should be False for RL)
            - And other prompt configuration options

    Returns:
        A transform function that converts a SampleData object to an RL-formatted dict.
    """
    # Ensure RL-specific settings: plain text format, not chat
    prompt_kwargs["use_chat_template"] = False

    # Get the prompt template
    prompt_template = pyine.prompts.manager.get_prompt_template(**prompt_kwargs)
    assert hasattr(prompt_template, "format"), "RL training requires a plain text template"

    return functools.partial(
        _apply_rl_prompt_template_to_sample,
        prompt_template=prompt_template,
    )
