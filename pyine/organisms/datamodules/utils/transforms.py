from __future__ import annotations

import collections.abc
import functools
import typing

import langchain_core.messages
import langchain_core.prompts

import pyine.organisms.datamodules.utils.samples
import pyine.prompts.manager

if typing.TYPE_CHECKING:
    import datasets as hf_datasets
    import transformers


SampleTransformInputType = pyine.organisms.datamodules.utils.samples.SampleData | dict[str, typing.Any]
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
    sample_data: pyine.organisms.datamodules.utils.samples.SampleData
    sample_args: dict[str, typing.Any]
    if isinstance(sample, collections.abc.Mapping):
        sample_dict = dict(sample)
        sample_args = sample_dict
        sample_data = pyine.organisms.datamodules.utils.samples.SampleData(**sample_dict)
    else:
        sample_data = sample
        sample_args = sample_data._asdict()
    if use_chat_template:
        assert isinstance(prompt_template, langchain_core.prompts.ChatPromptTemplate)
        messages = prompt_template.format_messages(**sample_args)
        assert isinstance(messages, list)
        # note: we currently only support single-turn interactions here, so one request per convo
        message_list = typing.cast("list[langchain_core.messages.BaseMessage]", messages)
        assert sum(isinstance(m, langchain_core.messages.SystemMessage) for m in message_list) <= 1
        assert sum(isinstance(m, langchain_core.messages.HumanMessage) for m in message_list) == 1
        assert sum(isinstance(m, langchain_core.messages.AIMessage) for m in message_list) == 0
        assert len(message_list) in [1, 2]  # we currently only support single-turn transforms here
        if merge_system_with_user and len(message_list) == 2:
            system_msg, user_msg = message_list
            assert isinstance(system_msg, langchain_core.messages.SystemMessage)
            assert isinstance(user_msg, langchain_core.messages.HumanMessage)
            system_content = _extract_message_content(system_msg, "system")
            user_content = _extract_message_content(user_msg, "human")
            merged_messages: list[langchain_core.messages.BaseMessage] = [
                langchain_core.messages.HumanMessage(system_content + "\n\n" + user_content)
            ]
            message_list = merged_messages
        if append_answer:
            message_list.append(langchain_core.messages.AIMessage(sample_data.expected_output))
        if use_hf_messages:
            hf_messages: list[dict[str, str]] = []
            for msg in message_list:
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
        return message_list
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


def _batch_apply_model_template_to_messages(
    batch: collections.abc.Mapping[str, typing.Any],
    tokenizer: transformers.PreTrainedTokenizer,
    append_eos_token: bool,
    strip_output: bool,
    messages_key: str,
    output_key: str,
    keep_original_data: bool,
    apply_chat_template_kwargs: dict[str, typing.Any] | None,
) -> dict[str, typing.Any]:
    """Applies a model template to a batch of messages.

    See `apply_model_template_to_messages` for information on arguments.

    Kept static/top-level-friendly for to keep pickling happy.
    """
    assert isinstance(batch, collections.abc.Mapping), f"unexpected input batch type: {type(batch)}"
    assert messages_key in batch, f"missing expected messages key: {messages_key}"
    messages = typing.cast("list[dict[str, str]]", batch[messages_key])
    text_result = tokenizer.apply_chat_template(
        conversation=messages,
        **(apply_chat_template_kwargs or {}),
    )
    if isinstance(text_result, list):
        if not any(not isinstance(t, str) for t in text_result):
            raise TypeError("expected tokenizer chat template output to be a list of strings")
    else:
        raise TypeError(
            f"expected tokenizer chat template output to be a list of strings, got {type(text_result)}",
        )
    assert len(text_result) == len(messages), "length mismatch between input messages and output text"
    if strip_output:
        text_result = [item.strip() for item in text_result]
    if append_eos_token:
        assert hasattr(tokenizer, "eos_token"), "tokenizer missing eos token"
        eos_token = typing.cast("str | list[str] | None", tokenizer.eos_token)
        if eos_token is None:
            raise ValueError("tokenizer has no EOS token configured")
        eos_suffix = "".join(eos_token) if isinstance(eos_token, list) else str(eos_token)
        text_result = [item + eos_suffix for item in text_result]
    output: dict[str, typing.Any] = dict(batch) if keep_original_data else {}
    output[output_key] = text_result
    return output


def apply_model_template_to_messages(
    hf_messages_dataset: hf_datasets.Dataset,
    tokenizer: transformers.PreTrainedTokenizer,
    append_eos_token: bool = False,
    strip_output: bool = False,
    messages_key: str = "messages",
    output_key: str = "text",
    keep_original_data: bool = False,
    apply_chat_template_kwargs: dict[str, typing.Any] | None = None,
    keep_in_memory: bool = False,
) -> hf_datasets.Dataset:
    """Map a HuggingFace messages dataset to a new dataset with text-only samples."""
    transform_batch: typing.Callable[[collections.abc.Mapping[str, typing.Any]], dict[str, typing.Any]] = (
        functools.partial(
            _batch_apply_model_template_to_messages,
            tokenizer=tokenizer,
            append_eos_token=append_eos_token,
            strip_output=strip_output,
            messages_key=messages_key,
            output_key=output_key,
            keep_original_data=keep_original_data,
            apply_chat_template_kwargs=apply_chat_template_kwargs,
        )
    )
    mapped_dataset = hf_messages_dataset.map(
        function=transform_batch,
        batched=True,
        desc="applying tokenizer chat template",
        keep_in_memory=keep_in_memory,
    )
    return typing.cast("hf_datasets.Dataset", mapped_dataset)
