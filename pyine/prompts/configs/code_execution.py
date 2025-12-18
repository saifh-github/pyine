"""Code execution prompt configuration and validation utilities.

This module provides:
- Prompt template access via `get_prompt_template()`;
- Output validation via `CodeExecutionValidator`; and
- LangChain chain composition via `get_code_execution_chain()`.

Example usage with a LangChain model::

    import pyine.prompts.configs.code_execution as code_exec
    import pyine.utils.llm_providers

    model = pyine.utils.llm_providers.get_model_from_provider(
        provider="openai",
        model="gpt-4o",
    )
    chain = code_exec.get_code_execution_chain(
        model,
        version="unstructured_with_3_predict_types",
        with_retry=True,
        max_retries=3,
    )
    result = chain.invoke(
        {
            "code": 'print("Hello")',
            "inputs": "",
            "predict_type": "program_output",
        }
    )
    print(f"Status: {result.validation_result.status}")
    print(f"Answer: {result.validation_result.parsed_answer!r}")
"""

from __future__ import annotations

import enum
import re
import typing
import warnings

import langchain_core.exceptions
import langchain_core.language_models
import langchain_core.runnables
import pydantic

import pyine.prompts.types
import pyine.utils.langchain
import pyine.utils.parsing

# @@@@ TODO: for experiments, consider pyine.utils.portability.print_code_with_numbered_lines()


def get_prompt_template(
    version: pyine.prompts.types.PromptVersionType | None = None,
    use_chat_template: bool = False,
    include_examples: bool = True,
    target_examples: int | list[int] | None = None,
    partial_vars: dict[str, typing.Any] | None = None,
    role_variables: dict[str, typing.Any] | None = None,
    context_variables: dict[str, typing.Any] | None = None,
    examples_block_variables: dict[str, typing.Any] | None = None,
) -> pyine.prompts.types.PromptTemplate:
    """Returns the prompt template for the code execution prompt (manager module override).

    Note: this implementation appropriately fills in all relevant partial variables, if any.
    """
    import pyine.prompts.manager

    prompt_config = pyine.prompts.manager.get_prompt_config("code_execution", version=version)
    return prompt_config.create_prompt_template(
        use_chat_template=use_chat_template,
        include_examples=include_examples,
        target_examples=target_examples,
        partial_vars=partial_vars,
        role_variables=role_variables,
        context_variables=context_variables,
        examples_block_variables=examples_block_variables,
    )


class OutputParsingMode(enum.StrEnum):
    """How to extract the answer from model output."""

    answer_only = enum.auto()
    """Model should answer directly, without any required tags."""
    reasoning_and_answer = enum.auto()
    """Model should provide some reasoning before providing an answer fenced with specific tags."""
    reasoning_steps_and_answer = enum.auto()
    """Model should provide reasoning + execution steps before providing an answer fenced with specific tags."""


class ValidationStatus(enum.StrEnum):
    """Validation outcome for a model output."""

    success = enum.auto()
    """Model output was successfully parsed and validated."""
    partial = enum.auto()
    """An answer was extracted, but could not be parsed as the expected type."""
    failed = enum.auto()
    """Model output could not be parsed at all."""


class CodeExecutionValidatorConfig(pydantic.BaseModel):
    """Configuration for code execution output validation."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    parsing_mode: OutputParsingMode = OutputParsingMode.answer_only
    """How to extract the answer from model output."""
    answer_tag: str = "final"
    """Tag for answer extraction (used in reasoning modes)."""
    steps_tag: str = "steps"
    """Tag for steps extraction (reasoning_steps_and_answer mode)."""

    # tag extraction policy
    multi_tag_policy: typing.Literal["first", "last", "error"] = "last"
    """Which block to select when multiple tags exist."""
    strict_tag_structure: bool = False
    """If True, fail on malformed tag structure (nested/stray/unclosed)."""

    # predict_type validation strictness
    strict_frame_variables: bool = True
    """If True, frame_variables must parse as dict."""
    strict_function_return: bool = False
    """If True, function_return must parse as structured OR match exception pattern."""

    # output normalization
    strip_answer: bool = True
    """Strip leading/trailing whitespace from answer_text."""
    strip_answer_for_program_output: bool = False
    """If False (default), preserve whitespace for program_output predict_type."""
    strip_tags_in_answer_only_mode: bool = True
    """If True (default), strip <answer_tag>...</answer_tag> if present in answer_only mode."""

    # noncompliance repair heuristics (helps retry convergence)
    repair_markdown_fences: bool = True
    """Strip ```json/```python/``` fences."""
    repair_output_prefix: bool = True
    """Strip 'Output: ' or 'Result: ' prefixes."""
    repair_output_prefix_for_program_output: bool = False
    """If False, skip output_prefix repair for program_output (preserves stdout fidelity)."""
    repair_surrounding_quotes: bool = False
    """Unwrap single surrounding quotes for simple scalars."""


class CodeExecutionValidationResult(pydantic.BaseModel):
    """Result of validating a code execution model output.

    Canonical prediction: for downstream components (logging, evals, comparison), use `answer_text`
    as the canonical string prediction. This field contains the extracted answer after:
    - Tag extraction (for reasoning modes);
    - Noncompliance repair (markdown fences, prefixes); and
    - Optional stripping (configurable).

    The `parsed_answer` field provides structured access when parsing succeeds, but `answer_text`
    is the authoritative string representation.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    raw_output: str
    """Original model output string (unmodified)."""
    status: ValidationStatus
    """Whether validation succeeded, failed, or partially succeeded."""
    answer_text: str | None = None
    """Canonical prediction string after extraction and repair. Use this for logging/evals."""
    parsed_answer: typing.Any | None = None
    """Structured parsed value (dict/list/etc) when parsing succeeds, else None."""
    reasoning_text: str | None = None
    """Extracted reasoning text (for modes that include it)."""
    steps_text: str | None = None
    """Extracted steps text (for reasoning_steps_and_answer mode)."""
    predict_type: str
    """The predict_type that was validated against (normalized to string)."""
    error_details: str | None = None
    """Human-readable error message if validation failed."""
    diagnostics: dict[str, typing.Any] = pydantic.Field(default_factory=dict)
    """Additional diagnostic info (tag counts, parse attempts, etc)."""


KNOWN_PREDICT_TYPES: frozenset[str] = frozenset(
    {
        "program_output",
        "frame_variables",
        "function_return",
    }
)
"""Known predict_type values with specific validation rules.

These types are linked with the types implemented in the `pyine.organisms.datamodule` package;
we must make sure to keep the code execution output parsing/validator logic aligned with these.
"""


class CodeExecutionOutput(pydantic.BaseModel):
    """Structured output from the code execution runnable.

    Designed to be useful for both standard chain usage and RL training.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    raw_output: str
    """Raw model output string."""
    validation_result: CodeExecutionValidationResult
    """Full validation result with parsed answer and diagnostics."""

    @property
    def is_valid(self) -> bool:
        """Whether validation succeeded."""
        return self.validation_result.status == ValidationStatus.success

    @property
    def parsed_answer(self) -> typing.Any | None:
        """Parsed answer value, or None if unparseable."""
        return self.validation_result.parsed_answer

    @property
    def answer_text(self) -> str | None:
        """Canonical answer text extracted from output."""
        return self.validation_result.answer_text


class CodeExecutionValidator:
    """Validates code execution model outputs based on predict_type and parsing mode."""

    def __init__(self, config: CodeExecutionValidatorConfig) -> None:
        """Create a validator with the given configuration."""
        self._config = config

    def validate(
        self,
        model_output: str,
        predict_type: str | enum.Enum,
    ) -> CodeExecutionValidationResult:
        """Validate model output for the given predict_type.

        Args:
            model_output: Raw model output string.
            predict_type: Type of prediction (str or enum). Normalized to string via
                getattr(predict_type, "value", predict_type).

        Validation rules by predict_type:
            - program_output: accept anything as-is (no parsing attempted, preserves raw stdout)
            - frame_variables: MUST parse as JSON or Python dict
            - function_return: try JSON/dict; accept repr strings or exception patterns

        Parsing modes:
            - answer_only: entire output is the answer
            - reasoning_and_answer: extract from <answer_tag>...</answer_tag>
            - reasoning_steps_and_answer: extract answer and steps; steps_text is populated but not validated
        """
        # normalize predict_type (handle enum values)
        predict_type_str = getattr(predict_type, "value", predict_type)
        if not isinstance(predict_type_str, str):
            predict_type_str = str(predict_type_str)
        if predict_type_str not in KNOWN_PREDICT_TYPES:
            warnings.warn(
                f"unknown predict_type '{predict_type_str}'; expected one of {sorted(KNOWN_PREDICT_TYPES)}. "
                "Unknown types are accepted as raw strings without validation.",
                stacklevel=2,
            )

        # step 1: extract answer text based on parsing_mode
        answer_text, reasoning_text, steps_text, extraction_diag, extraction_error = self._extract_answer(model_output)
        diagnostics: dict[str, typing.Any] = extraction_diag or {}
        if answer_text is None:
            return CodeExecutionValidationResult(
                raw_output=model_output,
                status=ValidationStatus.failed,
                answer_text=None,
                parsed_answer=None,
                reasoning_text=reasoning_text,
                steps_text=steps_text,
                predict_type=predict_type_str,
                error_details=extraction_error,
                diagnostics=diagnostics,
            )

        # step 2: apply noncompliance repair heuristics
        answer_text, repair_diag = self._apply_repair_heuristics(answer_text, predict_type_str)
        diagnostics.update(repair_diag)

        # step 3: apply stripping (configurable per predict_type)
        if self._config.strip_answer:
            if predict_type_str != "program_output" or self._config.strip_answer_for_program_output:
                answer_text = answer_text.strip()

        # step 4: validate/parse based on predict_type
        parsed_answer, parse_diag, parse_error = self._validate_for_predict_type(answer_text, predict_type_str)
        diagnostics.update(parse_diag)

        # step 5: determine status
        status = self._determine_status(predict_type_str, parsed_answer, parse_error)

        return CodeExecutionValidationResult(
            raw_output=model_output,
            status=status,
            answer_text=answer_text,
            parsed_answer=parsed_answer,
            reasoning_text=reasoning_text,
            steps_text=steps_text,
            predict_type=predict_type_str,
            error_details=parse_error,
            diagnostics=diagnostics,
        )

    def _apply_repair_heuristics(
        self,
        text: str,
        predict_type: str,
    ) -> tuple[str, dict[str, typing.Any]]:
        """Apply configured noncompliance repair heuristics.

        Args:
            text: Answer text to repair.
            predict_type: The predict_type being validated (affects which repairs apply).

        Returns:
            (repaired_text, diagnostics) with keys like 'repair/markdown_fences', etc.
        """
        diag: dict[str, typing.Any] = {}
        if self._config.repair_markdown_fences:
            new_text = pyine.utils.parsing.strip_markdown_fences(text)
            if new_text != text:
                diag["repair/markdown_fences"] = True
                text = new_text
        # output_prefix repair is skipped for program_output unless explicitly enabled
        should_repair_prefix = self._config.repair_output_prefix and (
            predict_type != "program_output" or self._config.repair_output_prefix_for_program_output
        )
        if should_repair_prefix:
            new_text = self._strip_output_prefix(text)
            if new_text != text:
                diag["repair/output_prefix"] = True
                text = new_text
        if self._config.repair_surrounding_quotes:
            new_text = self._unwrap_surrounding_quotes(text)
            if new_text != text:
                diag["repair/surrounding_quotes"] = True
                text = new_text
        return text, diag

    @staticmethod
    def _strip_output_prefix(text: str) -> str:
        """Remove 'Output: ' or 'Result: ' prefixes without stripping other whitespace."""
        # check if text starts with prefix (ignoring leading whitespace for check only)
        stripped_for_check = text.lstrip()
        prefix_pattern = re.compile(r"^(?:Output|Result):\s*", re.IGNORECASE)
        match = prefix_pattern.match(stripped_for_check)
        if match:
            # found prefix, remove it from the start (preserving any trailing whitespace)
            leading_ws = len(text) - len(stripped_for_check)
            return text[leading_ws + match.end() :]
        return text  # no prefix found, return original unchanged

    @staticmethod
    def _unwrap_surrounding_quotes(text: str) -> str:
        """Unwrap single surrounding quotes for simple scalars without stripping other whitespace."""
        # only unwrap if the stripped version has quotes; preserve whitespace otherwise
        stripped_for_check = text.strip()
        if len(stripped_for_check) >= 2:
            if (stripped_for_check[0] == '"' and stripped_for_check[-1] == '"') or (
                stripped_for_check[0] == "'" and stripped_for_check[-1] == "'"
            ):
                inner = stripped_for_check[1:-1]
                if stripped_for_check[0] not in inner:  # don't unwrap if quote appears inside
                    return inner
        return text  # no quotes to unwrap, return original unchanged

    def _extract_answer(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None, str | None, dict[str, typing.Any] | None, str | None]:
        """Extract answer (and optionally reasoning/steps) based on parsing mode.

        Returns:
            (answer_text, reasoning_text, steps_text, diagnostics, error_message)
        """
        if self._config.parsing_mode == OutputParsingMode.answer_only:
            # optionally strip tags if model included them anyway
            if self._config.strip_tags_in_answer_only_mode:
                result = pyine.utils.parsing.extract_tag_blocks(model_output, self._config.answer_tag)
                if result.block_count == 1 and not result.has_malformed_structure:
                    # model wrapped answer in tags; extract the inner content
                    return result.blocks[0], None, None, None, None
            return model_output, None, None, None, None  # stripping handled in validate() based on config
        if self._config.parsing_mode == OutputParsingMode.reasoning_and_answer:
            result = pyine.utils.parsing.extract_tag_blocks(model_output, self._config.answer_tag)
            diag = {
                f"tags/{self._config.answer_tag}/open_count": result.open_count,
                f"tags/{self._config.answer_tag}/close_count": result.close_count,
                f"tags/{self._config.answer_tag}/block_count": result.block_count,
            }
            # check for malformed structure if strict mode enabled
            if self._config.strict_tag_structure and result.has_malformed_structure:
                err_msg = (
                    f"malformed tag structure: nested={result.has_nested_open}, "
                    f"stray={result.has_stray_close}, unclosed={result.has_unclosed_open}"
                )
                return None, None, None, diag, err_msg
            # select block based on multi_tag_policy
            try:
                selection = pyine.utils.parsing.select_tag_block(result, policy=self._config.multi_tag_policy)
            except ValueError as exc:
                return None, None, None, diag, str(exc)
            if selection is None:
                return None, None, None, diag, "no answer tag found"
            answer, selected_idx = selection
            # reasoning is everything before the SELECTED block (respects multi_tag_policy)
            selected_offset = result.block_start_offsets[selected_idx]
            reasoning = model_output[:selected_offset].strip() if selected_offset > 0 else None
            return answer, reasoning, None, diag, None
        if self._config.parsing_mode == OutputParsingMode.reasoning_steps_and_answer:
            # extract both tags
            answer_result = pyine.utils.parsing.extract_tag_blocks(model_output, self._config.answer_tag)
            steps_result = pyine.utils.parsing.extract_tag_blocks(model_output, self._config.steps_tag)
            diag = {
                f"tags/{self._config.answer_tag}/open_count": answer_result.open_count,
                f"tags/{self._config.answer_tag}/close_count": answer_result.close_count,
                f"tags/{self._config.answer_tag}/block_count": answer_result.block_count,
                f"tags/{self._config.steps_tag}/open_count": steps_result.open_count,
                f"tags/{self._config.steps_tag}/close_count": steps_result.close_count,
                f"tags/{self._config.steps_tag}/block_count": steps_result.block_count,
            }
            # check for malformed structure if strict mode enabled
            if self._config.strict_tag_structure:
                if answer_result.has_malformed_structure:
                    return None, None, None, diag, "malformed answer tag structure"
                if steps_result.has_malformed_structure:
                    return None, None, None, diag, "malformed steps tag structure"
            try:
                answer_selection = pyine.utils.parsing.select_tag_block(
                    answer_result, policy=self._config.multi_tag_policy
                )
                steps_selection = pyine.utils.parsing.select_tag_block(
                    steps_result, policy=self._config.multi_tag_policy
                )
            except ValueError as exc:
                return None, None, None, diag, str(exc)
            steps = steps_selection[0] if steps_selection else None
            if answer_selection is None:
                return None, None, steps, diag, "no answer tag found"
            answer, answer_idx = answer_selection
            # reasoning is everything before the SELECTED answer block
            selected_offset = answer_result.block_start_offsets[answer_idx]
            reasoning = model_output[:selected_offset].strip() if selected_offset > 0 else None
            return answer, reasoning, steps, diag, None
        return None, None, None, None, f"unknown parsing mode: {self._config.parsing_mode}"

    def _validate_for_predict_type(
        self,
        answer_text: str,
        predict_type: str,
    ) -> tuple[typing.Any | None, dict[str, typing.Any], str | None]:
        """Apply predict_type-specific validation.

        Returns:
            (parsed_value, diagnostics, error_message); error_message is None on success.
        """
        diag: dict[str, typing.Any] = {}
        if predict_type == "program_output":
            # most flexible: accept raw string as-is, preserves stdout fidelity
            return answer_text, diag, None
        # for other predict_types, attempt parsing
        parse_result = pyine.utils.parsing.parse_json_or_python_literal(answer_text)
        diag["parse/success"] = parse_result.success
        if parse_result.method:
            diag["parse/method"] = parse_result.method
        if parse_result.error:
            diag["parse/error"] = parse_result.error
        if predict_type == "frame_variables":
            # strict: MUST parse as dict specifically (not just any JSON type)
            if not parse_result.success:
                if self._config.strict_frame_variables:
                    return None, diag, f"frame_variables must be JSON/dict: {parse_result.error}"
                return answer_text, diag, parse_result.error
            if not isinstance(parse_result.value, dict):
                diag["parse/value_type"] = type(parse_result.value).__name__
                if self._config.strict_frame_variables:
                    return None, diag, f"frame_variables must be dict, got {type(parse_result.value).__name__}"
                return answer_text, diag, f"parsed as {type(parse_result.value).__name__}, expected dict"
            # in strict mode, also validate that all keys are strings
            value_dict: dict[typing.Any, typing.Any] = parse_result.value  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
            if self._config.strict_frame_variables:
                non_string_keys = [k for k in value_dict if not isinstance(k, str)]
                if non_string_keys:
                    diag["parse/non_string_keys"] = len(non_string_keys)
                    err_msg = f"frame_variables keys must be strings, found {len(non_string_keys)} non-string key(s)"
                    return None, diag, err_msg
                # @@@@@ TODO: could also verify presence of specific expected keys in the variable dump?
            return typing.cast("dict[str, typing.Any]", value_dict), diag, None
        if predict_type == "function_return":
            # structured parse OR exception pattern; if strict, reject random repr strings
            if parse_result.success:
                return parse_result.value, diag, None
            is_exception = self._is_exception_pattern(answer_text)
            diag["parse/is_exception_pattern"] = is_exception
            if is_exception:
                return answer_text, diag, None  # exception patterns always accepted
            if self._config.strict_function_return:
                return None, diag, f"function_return must be structured or exception: {parse_result.error}"
            return answer_text, diag, None  # non-strict: accept as raw repr string
        return answer_text, diag, None  # unknown predict_type: accept raw string

    def _determine_status(
        self,
        predict_type: str,
        parsed_answer: typing.Any | None,
        parse_error: str | None,
    ) -> ValidationStatus:
        """Determine validation status based on predict_type rules and parse outcome."""
        if parse_error is None:
            return ValidationStatus.success
        # program_output always succeeds (no parsing required)
        if predict_type == "program_output":
            return ValidationStatus.success
        # for other types, having an error means partial or failed
        if parsed_answer is not None:
            return ValidationStatus.partial  # extracted but didn't meet strict requirements
        return ValidationStatus.failed

    @staticmethod
    def _is_exception_pattern(text: str) -> bool:
        """Check if text looks like a Python exception (e.g., 'ValueError: ...')."""
        exception_re = re.compile(r"^[A-Z][a-zA-Z]*(?:Error|Exception|Warning)(?::|$)", re.MULTILINE)
        return bool(exception_re.search(text.strip()))


def get_code_execution_validator(
    config: CodeExecutionValidatorConfig | None = None,
) -> CodeExecutionValidator:
    """Get a configured code execution validator.

    Args:
        config: Validation configuration. Uses defaults if None.

    Returns:
        A configured CodeExecutionValidator instance.
    """
    return CodeExecutionValidator(config or CodeExecutionValidatorConfig())


def _create_validation_runnable(
    validator: CodeExecutionValidator,
    *,
    raise_on_non_success: bool = False,
) -> langchain_core.runnables.RunnableLambda[dict[str, typing.Any], CodeExecutionOutput]:
    """Create a validation runnable that can be composed with a chain.

    Args:
        validator: The code execution validator to use.
        raise_on_non_success: If True, raise OutputParserException on any non-success status.
            This enables .with_retry() to trigger on both 'failed' and 'partial' statuses.

    Returns:
        A RunnableLambda that validates model outputs.
    """

    def validate_fn(data: dict[str, typing.Any]) -> CodeExecutionOutput:
        raw_output = data["model_output"]
        if hasattr(raw_output, "content"):
            raw_output = raw_output.content  # handle AIMessage
        predict_type = data["predict_type"]
        validation_result = validator.validate(raw_output, predict_type)
        if raise_on_non_success and validation_result.status != ValidationStatus.success:
            raise langchain_core.exceptions.OutputParserException(
                f"Validation failed ({validation_result.status}): {validation_result.error_details}"
            )
        return CodeExecutionOutput(
            raw_output=raw_output,
            validation_result=validation_result,
        )

    return langchain_core.runnables.RunnableLambda(validate_fn)


@typing.no_type_check  # because langchain typing sucks
def get_code_execution_chain(
    model: langchain_core.language_models.BaseChatModel,
    *,
    version: str | None = None,
    validator_config: CodeExecutionValidatorConfig | None = None,
    with_retry: bool = False,
    max_retries: int = 3,
    **prompt_kwargs: typing.Any,
) -> langchain_core.runnables.Runnable[dict[str, typing.Any], CodeExecutionOutput]:
    """Build a complete code execution chain with validation.

    Args:
        model: LangChain chat model to use.
        version: Prompt template version.
        validator_config: Validation configuration.
        with_retry: If True, wrap with retry logic (raises on non-success).
        max_retries: Maximum retry attempts when with_retry=True.
        **prompt_kwargs: Additional arguments for get_prompt_template().

    Returns:
        Configured runnable chain that outputs CodeExecutionOutput.
    """
    prompt_template = get_prompt_template(version=version, **prompt_kwargs)
    base_chain: langchain_core.runnables.Runnable[dict[str, typing.Any], typing.Any] = prompt_template | model
    validator = get_code_execution_validator(validator_config)
    validation_runnable = _create_validation_runnable(validator, raise_on_non_success=with_retry)

    def extract_predict_type(x: dict[str, typing.Any]) -> str:
        return x["predict_type"]

    # compose: pass through input, run model, validate with access to predict_type
    chain: langchain_core.runnables.Runnable[dict[str, typing.Any], CodeExecutionOutput] = (
        langchain_core.runnables.RunnableParallel(
            model_output=base_chain,
            predict_type=langchain_core.runnables.RunnableLambda(extract_predict_type),
        )
        | validation_runnable
    )
    if with_retry:
        chain = chain.with_retry(**pyine.utils.langchain.get_default_structured_output_chain_retry_config(max_retries))
    return chain
