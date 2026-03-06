import math
import typing

import langchain_core.output_parsers
import pydantic

if typing.TYPE_CHECKING:
    import pyine.prompts.types


class GradingResult(pydantic.BaseModel):
    """Structured grading results schema containing only a score."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
    """Pydantic model configuration (freezes the dataclass)."""

    score: typing.Annotated[float, pydantic.Field(ge=0.0, le=1.0)] = pydantic.Field(
        description="Score in [0,1], where 0 means totally different and 1 means perfect match.",
    )

    _was_clamped: bool = pydantic.PrivateAttr(default=False)
    """Whether the original score was clamped to [0.0, 1.0] during validation."""

    @property
    def was_clamped(self) -> bool:
        """Whether the score was clamped to [0.0, 1.0] during validation."""
        return self._was_clamped

    @pydantic.model_validator(mode="wrap")
    @classmethod
    def _clamp_score(
        cls,
        values: typing.Any,
        handler: pydantic.ValidatorFunctionWrapHandler,
    ) -> "GradingResult":
        """Clamp out-of-range scores, reject bools and non-finite values."""
        if isinstance(values, dict) and "score" in values:
            raw_score = typing.cast("typing.Any", values["score"])
            if isinstance(raw_score, bool):
                raise ValueError("score must not be a bool")
            if isinstance(raw_score, (int, float)) and not isinstance(raw_score, bool):
                if math.isnan(raw_score) or math.isinf(raw_score):
                    raise ValueError(f"score must be finite, got {raw_score}")
                clamped = False
                coerced = float(raw_score)
                if coerced < 0.0:
                    coerced = 0.0
                    clamped = True
                elif coerced > 1.0:
                    coerced = 1.0
                    clamped = True
                patched_values: dict[str, typing.Any] = {**values, "score": coerced}
                instance = handler(patched_values)
                if clamped:
                    instance._was_clamped = True
                return instance
        return handler(values)


class GradingResultWithReasoning(GradingResult):
    """Structured grading results schema with optional reasoning supporting the score."""

    reasoning: str | None = pydantic.Field(
        default=None,
        description="Optional brief reasoning explaining the score.",
    )


def get_output_parser(
    version: "pyine.prompts.types.PromptVersionType | None" = None,
) -> langchain_core.output_parsers.BaseOutputParser[typing.Any] | None:
    """Return the output parser for the pred_grader prompt based on version."""
    if version == "score_only" or version is None:  # default
        model = GradingResult
    elif version == "score_only_for_openai_grader":
        return None  # not using schema/structured output for openai usage
    elif version == "with_reasoning":
        model = GradingResultWithReasoning
    else:
        raise NotImplementedError(f"Unsupported version: {version}")
    return langchain_core.output_parsers.PydanticOutputParser[typing.Any](pydantic_object=model)


def get_prompt_template(
    version: "pyine.prompts.types.PromptVersionType | None" = None,
    use_chat_template: bool = False,
    include_examples: bool = True,
    target_examples: int | list[int] | None = None,
    partial_vars: dict[str, typing.Any] | None = None,
    role_variables: dict[str, typing.Any] | None = None,
    context_variables: dict[str, typing.Any] | None = None,
    examples_block_variables: dict[str, typing.Any] | None = None,
) -> "pyine.prompts.types.PromptTemplate":
    """Returns the prompt template for the prediction grader prompt (manager module override).

    Note: this implementation appropriately fills in all relevant partial variables, if any.
    """
    import pyine.prompts.manager

    prompt_config = pyine.prompts.manager.get_prompt_config("pred_grader", version=version)
    parser = get_output_parser(version=version)
    merged_context: dict[str, typing.Any] = dict(context_variables) if context_variables else {}
    if parser is not None:
        merged_context.setdefault("expected_output_format", parser.get_format_instructions())
    return prompt_config.create_prompt_template(
        use_chat_template=use_chat_template,
        include_examples=include_examples,
        target_examples=target_examples,
        partial_vars=partial_vars,
        role_variables=role_variables,
        context_variables=merged_context or None,
        examples_block_variables=examples_block_variables,
    )
