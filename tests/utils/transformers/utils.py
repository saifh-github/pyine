import typing

import torch
import transformers


class SimpleTokenizer:
    def __init__(
        self,
        padding_side: str = "right",
        truncation_side: str = "left",
    ) -> None:
        self.pad_token_id = 0
        self.model_max_length = 4096
        self.padding_side = padding_side
        self.truncation_side = truncation_side

    def apply_chat_template(
        self,
        conversation: list[dict[str, typing.Any]] | list[list[dict[str, typing.Any]]],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        **kwargs: typing.Any,
    ) -> str | list[str]:
        # support both single conversation and batched conversations
        # if not conversation:
        #     if add_generation_prompt:
        #         return "assistant:"
        #     return ""
        assert tokenize is False, "test fixture code below does not support tokenization"
        is_batch = isinstance(conversation[0], list)
        if is_batch:
            results = []
            for messages in conversation:
                history = "".join(f"{item['role']}:{item['content']}|" for item in messages)
                if add_generation_prompt:
                    history += "assistant:"
                results.append(history)
            return results
        messages = conversation
        history = "".join(f"{item['role']}:{item['content']}|" for item in messages)
        if add_generation_prompt:
            return history + "assistant:"
        return history

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        **kwargs: typing.Any,
    ) -> dict[str, list[int]]:
        return {
            "input_ids": [ord(character) for character in text],
            "attention_mask": [1] * len(text),
        }

    def decode(
        self,
        token_ids: typing.Iterable[int] | torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> str:
        values = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else list(token_ids)
        characters: list[str] = []
        for value in values:
            if skip_special_tokens and value == self.pad_token_id:
                continue
            characters.append(chr(int(value)))
        return "".join(characters)


class DummyGenerationOutput:
    def __init__(
        self,
        sequences: torch.Tensor,
    ) -> None:
        self.sequences = sequences


class DummyGenerationModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.dtype = torch.float32

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generation_config: transformers.GenerationConfig,
        return_dict_in_generate: bool,
    ) -> DummyGenerationOutput:
        num_return_sequences = getattr(generation_config, "num_return_sequences", 1) or 1
        sequences: list[torch.Tensor] = []
        for idx in range(input_ids.size(0)):
            for attempt_idx in range(num_return_sequences):
                # offset generated tokens by both batch index and attempt index to make them unique
                offset = idx + attempt_idx * 10
                new_tokens = torch.tensor(
                    [ord("x") + offset, ord("y") + offset],
                    dtype=torch.long,
                    device=input_ids.device,
                )
                sequences.append(torch.cat((input_ids[idx], new_tokens), dim=0))
        padded = torch.nn.utils.rnn.pad_sequence(
            sequences,
            batch_first=True,
            padding_value=0,
            padding_side="right",
        )
        return DummyGenerationOutput(padded)


class MinimalPreTrainedTokenizer(transformers.PreTrainedTokenizerBase):
    def __init__(self) -> None:
        super().__init__()

    def _tokenize(
        self,
        text: str,
        **kwargs: typing.Any,
    ) -> list[str]:
        del text
        del kwargs
        return []

    def _convert_token_to_id_with_added_voc(
        self,
        token: str,
    ) -> int:
        del token
        return 0

    def _convert_id_to_token(
        self,
        index: int,
    ) -> str:
        del index
        return "token"

    def get_vocab(self) -> dict[str, int]:
        return {"token": 0}

    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: str | None = None,
    ) -> tuple[typing.Any, ...]:
        del save_directory
        del filename_prefix
        return ()
