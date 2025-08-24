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
SampleTransformInputType = pyine.organisms.datamodules.utils.samples.SampleData
SampleTransformOutputType = str | list[langchain_core.messages.BaseMessage] | HFMessageListType
SampleTransformType = typing.Callable[[SampleTransformInputType], SampleTransformOutputType]


def create_sample_transform(
    use_chat_template: bool,
    append_answer: bool,
    use_hf_messages: bool = False,
    **prompt_kwargs,  # forwarded to the prompt manager template getter
) -> SampleTransformType:
    """Create a transform function applying a prompt template to a data sample.

    If `use_chat_template=True`, the transform function will format the produced messages and return
    those (instead of formatting the prompt as a string directly). If `append_answer=True`, the
    expected answer that the assistant would provide will be appended to the end of rendered prompt
    or messages.

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
        sample_args = sample._asdict()
        if use_chat_template:
            output = prompt_template.format_messages(**sample_args)
            assert isinstance(output, list)
            if append_answer:
                output.append(langchain_core.messages.AIMessage(sample.output))
            if use_hf_messages:
                assert sum([isinstance(m, langchain_core.messages.SystemMessage) for m in output]) <= 1
                assert sum([isinstance(m, langchain_core.messages.HumanMessage) for m in output]) <= 1
                assert sum([isinstance(m, langchain_core.messages.AIMessage) for m in output]) <= 1
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
                output += "\n" + sample.output
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
            for idx in len(text):
                text[idx] = text[idx].strip()
        if append_eos_token:
            for idx in len(text):
                text[idx] += tokenizer.eos_token
        return {"text": text}

    output_dataset = hf_messages_dataset.map(
        function=_transform_batch,
        batched=True,
        desc="applying chat template",
        **(batching_map_kwargs or {}),
    )
    return output_dataset
