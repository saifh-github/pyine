import collections
import functools
import typing

import langchain_core.messages
import langchain_core.prompts
import transformers

import pyine.organisms.datamodules.utils.samples
import pyine.prompts.manager
import pyine.prompts.utils

if typing.TYPE_CHECKING:
    import datasets as hf_datasets


HFMessageListType = list[dict[str, str]]
SampleTransformInputType = pyine.organisms.datamodules.utils.samples.SampleData | dict[str, typing.Any]
SampleTransformOutputType = str | list[langchain_core.messages.BaseMessage] | HFMessageListType
SampleTransformType = typing.Callable[[SampleTransformInputType], SampleTransformOutputType]


def create_sample_transform(
    use_chat_template: bool,
    append_answer: bool,
    use_hf_messages: bool = False,
    merge_system_with_user: bool = False,
    **prompt_kwargs,  # forwarded to the prompt manager template getter
) -> SampleTransformType:
    """Create a transform function applying a prompt template to a data sample.

    Args:
        use_chat_template: whether to use a chat template or a regular template for sample formatting.
        append_answer: whether to append the assistant's response to the conversation messages.
        use_hf_messages: whether to use HuggingFace messages format or the langchain format.
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

    def _apply_template_to_sample(
        sample: SampleTransformInputType,
    ) -> SampleTransformOutputType:
        if isinstance(sample, collections.abc.Mapping):
            sample_args = sample
            sample = pyine.organisms.datamodules.utils.samples.SampleData(**sample)
        else:
            sample_args = sample._asdict()  # should be a named-tuple-like interface
        if use_chat_template:
            output = prompt_template.format_messages(**sample_args)
            assert isinstance(output, list)
            # note: we currently only support single-turn interactions here, so one request per convo
            assert sum([isinstance(m, langchain_core.messages.SystemMessage) for m in output]) <= 1
            assert sum([isinstance(m, langchain_core.messages.HumanMessage) for m in output]) == 1
            assert sum([isinstance(m, langchain_core.messages.AIMessage) for m in output]) == 0  # added below
            assert len(output) in [1, 2]  # we currently only support single-turn transforms here
            if merge_system_with_user and len(output) == 2:
                output = typing.cast(
                    list[langchain_core.messages.BaseMessage],
                    [langchain_core.messages.HumanMessage(output[0].content + "\n\n" + output[1].content)],
                )
            if append_answer:
                output.append(langchain_core.messages.AIMessage(sample.expected_output))
            if use_hf_messages:
                hf_msgs: list[dict[str, str]] = []
                for msg in output:
                    if isinstance(msg, langchain_core.messages.SystemMessage):
                        hf_msgs.append({"role": "system", "content": msg.content})
                    elif isinstance(msg, langchain_core.messages.HumanMessage):
                        hf_msgs.append({"role": "user", "content": msg.content})
                    elif isinstance(msg, langchain_core.messages.AIMessage):
                        hf_msgs.append({"role": "assistant", "content": msg.content})
                    else:
                        raise NotImplementedError(f"unsupported message type: {type(msg)}")
                output = {"messages": hf_msgs}
        else:
            output = prompt_template.format(**sample_args)
            if append_answer:
                assert isinstance(output, str)
                output += "\n" + sample.expected_output
        return output

    return _apply_template_to_sample


def apply_model_template_to_messages(
    hf_messages_dataset: "hf_datasets.Dataset",
    tokenizer: transformers.PreTrainedTokenizerBase,
    append_eos_token: bool = False,
    strip_output: bool = False,
    messages_key: str = "messages",
    apply_chat_template_kwargs: dict | None = None,
    batching_map_kwargs: dict | None = None,
) -> "hf_datasets.Dataset":
    """Map a HuggingFace dataset to a new dataset with text-only samples."""

    def _transform_batch(batch) -> dict[str, list]:
        text = tokenizer.apply_chat_template(
            batch[messages_key],
            **(apply_chat_template_kwargs or {}),
        )
        assert isinstance(text, list) and len(text) == len(batch[messages_key])
        if strip_output:
            for idx in range(len(text)):
                text[idx] = text[idx].strip()
        if append_eos_token:
            for idx in range(len(text)):
                text[idx] += tokenizer.eos_token
        return {"text": text}

    output_dataset = hf_messages_dataset.map(
        function=_transform_batch,
        batched=True,
        desc="applying chat template",
        **(batching_map_kwargs or {}),
    )
    return output_dataset
