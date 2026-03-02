"""Tests for format_messages_to_text and tokenizer_has_chat_template utilities."""

from __future__ import annotations

import typing

import pytest
import transformers

import pyine.apps.trainers.common
import pyine.utils.transformers.data

if typing.TYPE_CHECKING:
    import pytest_mock


class TestTokenizerHasChatTemplate:
    def test_returns_true_when_template_present(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        tokenizer = mocker.MagicMock(spec=transformers.PreTrainedTokenizerBase)
        tokenizer.chat_template = "some template"
        tokenizer.apply_chat_template = mocker.MagicMock()
        assert pyine.utils.transformers.data.tokenizer_has_chat_template(tokenizer) is True

    def test_returns_false_when_no_template(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        tokenizer = mocker.MagicMock(spec=transformers.PreTrainedTokenizerBase)
        tokenizer.chat_template = None
        assert pyine.utils.transformers.data.tokenizer_has_chat_template(tokenizer) is False


class TestFormatMessagesToText:
    _MESSAGES: typing.ClassVar[list[dict[str, str]]] = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]

    def test_format_no_chat_template(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Encoder tokenizer (no chat template) produces role-tagged plain text."""
        tokenizer = mocker.MagicMock(spec=transformers.PreTrainedTokenizerBase)
        tokenizer.chat_template = None
        result = pyine.utils.transformers.data.format_messages_to_text(self._MESSAGES, tokenizer)
        assert result == "user: What is 2+2?\n\nassistant: 4"

    def test_format_with_chat_template(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Decoder tokenizer (with chat template) delegates to apply_chat_template."""
        tokenizer = mocker.MagicMock(spec=transformers.PreTrainedTokenizerBase)
        tokenizer.chat_template = "some template"
        expected_text = "<|user|>What is 2+2?<|assistant|>4"
        tokenizer.apply_chat_template = mocker.MagicMock(return_value=expected_text)
        result = pyine.utils.transformers.data.format_messages_to_text(self._MESSAGES, tokenizer)
        assert result == expected_text
        tokenizer.apply_chat_template.assert_called_once_with(
            conversation=self._MESSAGES,
            tokenize=False,
        )

    def test_raises_on_non_str_chat_template_result(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        tokenizer = mocker.MagicMock(spec=transformers.PreTrainedTokenizerBase)
        tokenizer.chat_template = "some template"
        tokenizer.apply_chat_template = mocker.MagicMock(return_value=["not", "a", "str"])
        with pytest.raises(TypeError, match="expected tokenizer.apply_chat_template to return str"):
            pyine.utils.transformers.data.format_messages_to_text(self._MESSAGES, tokenizer)

    def test_parity_with_batched_formatting_no_template(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Verify format_messages_to_text output matches _batch_format_messages_to_text (no-template)."""
        tokenizer = mocker.MagicMock(spec=transformers.PreTrainedTokenizerBase)
        tokenizer.chat_template = None
        single_result = pyine.utils.transformers.data.format_messages_to_text(self._MESSAGES, tokenizer)
        batch_output = pyine.apps.trainers.common._batch_format_messages_to_text(
            batch={"messages": [self._MESSAGES]},
            messages_key="messages",
            output_key="text",
            tokenizer=tokenizer,
        )
        assert single_result == batch_output["text"][0]

    def test_parity_with_batched_formatting_with_template(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Verify format_messages_to_text output matches _batch_format_messages_to_text (chat-template)."""
        tokenizer = mocker.MagicMock(spec=transformers.PreTrainedTokenizerBase)
        tokenizer.chat_template = "some template"
        expected_text = "<|user|>What is 2+2?<|assistant|>4"
        tokenizer.apply_chat_template = mocker.MagicMock(return_value=expected_text)
        single_result = pyine.utils.transformers.data.format_messages_to_text(self._MESSAGES, tokenizer)
        batch_output = pyine.apps.trainers.common._batch_format_messages_to_text(
            batch={"messages": [self._MESSAGES]},
            messages_key="messages",
            output_key="text",
            tokenizer=tokenizer,
        )
        assert single_result == batch_output["text"][0]
        assert single_result == expected_text
