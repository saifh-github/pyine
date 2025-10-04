import typing

import langchain_core.messages
import langchain_core.prompts
import langchain_core.prompts.chat
import openai
import openai.types.fine_tuning
import openai.types.graders
import pydantic

import pyine.utils.openai

DefaultPromptMsgsType = typing.Literal["default"]
GraderPromptMessage = openai.types.graders.score_model_grader_param.Input
GraderPromptMessageContent = openai.types.graders.score_model_grader_param.InputContent
GraderPromptMessages = list[GraderPromptMessage]


class PredGraderFineTuneMethodConfig(pydantic.BaseModel):
    """Configuration for the prediction grader RL fine-tuning method used in RL experiments."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
    """Pydantic model configuration (freezes the dataclass)."""

    grader_model: str = "gpt-4.1-mini-2025-04-14"
    """Model to use for the prediction grader.

    For more information on gradel model constraints, see:
    https://platform.openai.com/docs/guides/graders#model-grader-constraints
    """
    grader_name: str = "pred_grader"
    """Arbitrary name for the prediction grader."""
    default_max_output_tokens: int = 64  # we don't need much, should be a float-only output? (+struct?)
    """Default maximum output tokens for the prediction grader."""
    batch_size: typing.Literal["auto"] | int = "auto"
    """Number of examples in each batch.

    A larger batch size means that model parameters are updated less frequently, but
    with lower variance.
    """
    pred_grader_prompt_messages: list[dict[str, pydantic.JsonValue]] | DefaultPromptMsgsType = "default"
    """Messages to use as the prompt for the prediction grader."""
    learning_rate_multiplier: typing.Literal["auto"] | float = "auto"
    """Scaling factor for the learning rate.

    A smaller learning rate may be useful to avoid overfitting.
    """
    compute_multiplier: typing.Literal["auto"] | float = "auto"
    """Multiplier on amount of compute used for exploring search space during training."""
    eval_interval: typing.Literal["auto"] | int = "auto"
    """The number of training steps between evaluation runs."""
    eval_samples: typing.Literal["auto"] | int = "auto"
    """Number of evaluation samples to generate per training step."""
    reasoning_effort: typing.Literal["default", "low", "medium", "high"] = "default"
    """Level of reasoning effort."""
    n_epochs: typing.Literal["auto"] | int = "auto"
    """The number of epochs to train the model for.

    An epoch refers to one full cycle through the training dataset.
    """

    def get_openai_config(self) -> dict[str, typing.Any]:
        """Returns the configuration for the prediction grader used in OpenAI RL experiments.

        Note: output should be compatible with `openai.types.fine_tuning.job_create_params.Method`.
        """
        if self.pred_grader_prompt_messages != "default":
            prompt_messages: GraderPromptMessages = self._normalize_prompt_messages(
                self.pred_grader_prompt_messages,
            )
        else:
            assert self._resolved_pred_grader_prompt_messages is not None
            prompt_messages = self._resolved_pred_grader_prompt_messages
        return {
            "type": "reinforcement",
            "reinforcement": openai.types.fine_tuning.reinforcement_method_param.ReinforcementMethodParam(
                grader=openai.types.graders.score_model_grader_param.ScoreModelGraderParam(
                    input=prompt_messages,
                    model=self.grader_model,
                    name=self.grader_name,
                    type="score_model",
                    range=[0, 1],
                    # BUGGED AS OF 2025-09-03; the OpenAI API describes this setting, but it's unavailable (unknown)?
                    # sampling_params=dict(
                    #     max_tokens=self.default_max_output_tokens,
                    # ),
                ),
                hyperparameters=openai.types.fine_tuning.reinforcement_hyperparameters_param.ReinforcementHyperparametersParam(
                    batch_size=self.batch_size,
                    compute_multiplier=self.compute_multiplier,
                    eval_interval=self.eval_interval,
                    eval_samples=self.eval_samples,
                    learning_rate_multiplier=self.learning_rate_multiplier,
                    n_epochs=self.n_epochs,
                    reasoning_effort=self.reasoning_effort,
                ),
            ),
        }

    _resolved_pred_grader_prompt_messages: GraderPromptMessages | None = pydantic.PrivateAttr(
        default=None,
    )

    @pydantic.model_validator(mode="after")
    def _validate_and_resolve(self) -> "PredGraderFineTuneMethodConfig":
        """Validates and resolves the prompt template messages (if needed)."""
        if self.pred_grader_prompt_messages == "default":
            import pyine.prompts.manager

            prompt_template = pyine.prompts.manager.get_prompt_template(
                "pred_grader",
                version="score_only_for_openai_grader",
                use_chat_template=True,
                include_examples=True,
            )
            if not isinstance(
                prompt_template,
                langchain_core.prompts.chat.ChatPromptTemplate,
            ):
                raise TypeError("pred_grader prompt must be a ChatPromptTemplate")
            messages: list[langchain_core.prompts.chat.MessageLike] = prompt_template.messages
            assert len(messages) == 2
            assert isinstance(messages[0], langchain_core.messages.SystemMessage)
            assert isinstance(messages[1], langchain_core.prompts.HumanMessagePromptTemplate)
            string_prompt = messages[1].prompt
            if not isinstance(string_prompt, langchain_core.prompts.StringPromptTemplate):
                raise TypeError("pred_grader human message must use a StringPromptTemplate")
            content_template = getattr(string_prompt, "template", None)
            if not isinstance(content_template, str):
                raise TypeError("pred_grader human template must expose a string template")
            normalized_messages: list[typing.Any] = [
                messages[0],
                {"role": "user", "content": content_template},
            ]
            self._resolved_pred_grader_prompt_messages = self._normalize_prompt_messages(
                normalized_messages,
            )
        return self

    def _normalize_prompt_messages(
        self,
        prompt_messages: typing.Sequence[typing.Any],
    ) -> GraderPromptMessages:
        openai_messages = pyine.utils.openai.convert_messages_to_openai(prompt_messages)
        normalized_messages: GraderPromptMessages = []
        valid_roles = {"assistant", "developer", "system", "user"}
        for raw_message in openai_messages:
            message_dict: dict[str, typing.Any] = dict(raw_message)
            role_value = str(message_dict.get("role", "user"))
            if role_value not in valid_roles:
                raise ValueError(f"invalid message role: {role_value}")
            content_value_raw = message_dict.get("content", "")
            if content_value_raw is None:
                content_value: GraderPromptMessageContent = ""
            else:
                content_value = typing.cast(
                    "GraderPromptMessageContent",
                    content_value_raw,
                )
            message_dict["role"] = role_value
            message_dict["content"] = content_value
            message_dict.setdefault("type", "message")
            normalized_messages.append(
                typing.cast("GraderPromptMessage", message_dict),
            )
        return normalized_messages
