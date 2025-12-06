import collections.abc
import datetime
import logging
import pathlib
import pprint
import re
import time
import typing

import backoff
import datasets
import langchain_core.messages
import openai
import openai.types.chat
import openai.types.fine_tuning
import openai.types.fine_tuning.fine_tuning_job
import openai.types.fine_tuning.fine_tuning_job_wandb_integration
import orjson
import pydantic

import pyine.utils.filesystem
import pyine.utils.portability
import pyine.utils.pydantic
import pyine.utils.reprod

logger = logging.getLogger(__name__)

# Exception types to retry with exponential backoff. APIStatusError is included but giveup filters 4xx.
RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.APIStatusError,  # giveup function filters to only retry 5xx, not 4xx
)
"""Transient OpenAI API errors that should trigger retry with exponential backoff."""


def _should_giveup_on_api_error(exc: Exception) -> bool:
    """Return True to give up retrying (non-retryable error).

    For APIStatusError, only 5xx server errors are retryable; 4xx client errors
    (BadRequest, Auth, NotFound, etc.) should fail fast.
    """
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code < 500  # give up on 4xx errors
    return False  # don't give up on timeout/connection/rate-limit errors


# Reusable backoff decorator for OpenAI API calls. Centralizes retry config for maintainability.
_retry_openai_api = backoff.on_exception(
    backoff.expo,
    RETRYABLE_EXCEPTIONS,
    max_time=120,
    giveup=_should_giveup_on_api_error,
)

SystemMessageType = openai.types.chat.chat_completion_system_message_param.ChatCompletionSystemMessageParam
UserMessageType = openai.types.chat.chat_completion_user_message_param.ChatCompletionUserMessageParam
AssistantMessageType = openai.types.chat.chat_completion_assistant_message_param.ChatCompletionAssistantMessageParam


def get_local_file_directory() -> pathlib.Path:
    """Returns the path used to store prepared OpenAI data upload files."""
    dir_path = pyine.utils.filesystem.get_data_cache_path() / "openai-data"
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def write_dataset_to_jsonl(
    dataset: typing.Iterable[dict[str, typing.Any] | list[typing.Any]] | datasets.Dataset,
    path: pathlib.Path,
    enforce_openai_min_dataset_size: bool = True,
) -> None:
    """Writes a dataset to a JSONL file.

    Will dump the dataset samples as one-line-per-sample, assuming each sample contains a conversation
    stored as a dictionary with a list of messages (a list of dicts). If any require fields are missing,
    an exception will be raised.

    If `enforce_openai_min_dataset_size` is True, will enforce the minimum size required by OpenAI
    (i.e. 10 samples). If the dataset is smaller than this, an exception will be raised.
    """
    samples_str: list[str] = []
    iterable_dataset = typing.cast("collections.abc.Iterable[typing.Any]", dataset)
    for sample in iterable_dataset:
        if isinstance(sample, collections.abc.Mapping):
            mapping_sample = typing.cast("collections.abc.Mapping[str, typing.Any]", sample)
            if "messages" not in mapping_sample:
                raise KeyError("dataset sample is missing 'messages'")
            raw_messages = mapping_sample["messages"]
            if not isinstance(raw_messages, collections.abc.Sequence):
                raise TypeError("dataset sample 'messages' must be a sequence")
            msgs = convert_messages_to_openai(typing.cast("typing.Sequence[typing.Any]", raw_messages))
        elif isinstance(sample, collections.abc.Sequence):
            msgs = convert_messages_to_openai(typing.cast("typing.Sequence[typing.Any]", sample))
        else:
            raise TypeError("dataset samples must be mappings or sequences")
        assert all(isinstance(m, dict) and all(f in m for f in ["role", "content"]) for m in msgs)
        samples_str.append(orjson.dumps({"messages": msgs}).decode("utf-8"))
    if enforce_openai_min_dataset_size and len(samples_str) < 10:
        raise ValueError(f"dataset must have at least 10 samples, got {len(samples_str)}")
    with open(path, "w", encoding="utf-8") as fd:
        fd.write("\n".join(samples_str))
    dataset_size = pyine.utils.filesystem.get_human_readable_size(path.stat().st_size)
    logger.info(f"wrote {dataset_size} dataset with {len(samples_str)} samples to {path}")


def read_dataset_from_jsonl(
    path: pathlib.Path,
) -> list[list[dict[str, str]]]:
    """Reads a dataset from a JSONL file.

    Will parse the dataset samples as one-line-per-sample, where each line contains a conversation
    stored as a dictionary with a list of messages (a list of dicts).
    """
    if not path.is_file():
        raise FileNotFoundError(f"file does not exist: {path}")
    messages_dataset: list[list[dict[str, str]]] = []
    with open(path) as fd:
        for lineno, raw in enumerate(fd, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = orjson.loads(line)
            except Exception as e:
                raise ValueError(f"invalid JSON on line {lineno}: {e}") from e
            raw_msgs: list[typing.Any]
            if isinstance(obj, list):
                raw_msgs = typing.cast("list[typing.Any]", obj)
            elif isinstance(obj, dict):
                if "messages" in obj and isinstance(obj["messages"], list):
                    raw_msgs = typing.cast("list[typing.Any]", obj["messages"])
                else:
                    raise ValueError(f"line {lineno}: expected list or a dict with 'messages'")
            else:
                raise ValueError(f"line {lineno}: unsupported JSON type {type(obj).__name__}")
            curr_messages: list[dict[str, str]] = []
            for idx, msg in enumerate(raw_msgs, start=1):
                if not isinstance(msg, dict):
                    raise ValueError(f"line {lineno}: message {idx} is not a dict")
                msg_dict = typing.cast("dict[str, typing.Any]", msg)
                if "role" not in msg_dict or "content" not in msg_dict:
                    raise ValueError(f"line {lineno}: message {idx} missing 'role' or 'content' keys")
                role_value = msg_dict["role"]
                content_value = msg_dict["content"]
                if not isinstance(role_value, str) or not isinstance(content_value, str):
                    # coerce to strings to ensure consistent downstream handling
                    new_msg = dict(msg_dict)
                    new_msg["role"] = str(role_value)
                    new_msg["content"] = str(content_value)
                    curr_messages.append(new_msg)
                else:
                    curr_messages.append(typing.cast("dict[str, str]", msg_dict))
            messages_dataset.append(curr_messages)
    return messages_dataset


def write_objects_to_jsonl(dataset: typing.Iterable[typing.Any], path: pathlib.Path) -> None:
    """Writes a dataset of arbitrary JSON-serializable objects to a JSONL file.

    This preserves each object's schema as-is on a single line. Useful for RL or preference
    datasets where examples may include fields like 'chosen'/'rejected', 'preference',
    'reward', or trajectories that do not fit the simple chat 'messages' format.
    """
    with open(path, "w", encoding="utf-8") as fd:
        for sample in dataset:
            fd.write(orjson.dumps(sample).decode("utf-8") + "\n")
    dataset_size = pyine.utils.filesystem.get_human_readable_size(path.stat().st_size)
    logger.debug(f"wrote {dataset_size} raw-object dataset to {path}")


def read_objects_from_jsonl(path: pathlib.Path) -> list[typing.Any]:
    """Reads a JSONL file of arbitrary objects into a list of Python objects."""
    if not path.is_file():
        raise FileNotFoundError(f"file does not exist: {path}")
    objects: list[typing.Any] = []
    with open(path, encoding="utf-8") as fd:
        for lineno, raw in enumerate(fd, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = orjson.loads(line)
            except Exception as e:
                raise ValueError(f"invalid JSON on line {lineno}: {e}") from e
            objects.append(obj)
    return objects


def estimate_token_count(text: str, model_id: str) -> int:
    """Estimates the token count for a given OpenAI model and input text.

    Will use tiktoken for estimates (exact for supported OpenAI models); if the specified
    `model_id` is unknown to tiktoken, will pick a sensible default encoding.

    Returns:
        int: estimated token count (exact when model supported by tiktoken).
    """
    import pyine.utils.tokenizers

    tokenizer = pyine.utils.tokenizers.get_openai_tokenizer(
        model_id=model_id,
        raise_if_not_found=False,
    )
    return len(tokenizer.encode(text))


def convert_messages_to_openai(
    messages: typing.Sequence[typing.Any],
) -> list[dict[str, typing.Any]]:
    """Convert a list of LangChain messages (or message-like dicts) into OpenAI Chat API-compatible messages.

    This function maps common LangChain message types to OpenAI 'messages' format:
      - LangChain SystemMessage -> role 'system'
      - LangChain HumanMessage -> role 'user'
      - LangChain AIMessage -> role 'assistant' (supports tool_calls)
      - LangChain ToolMessage -> role 'tool' (with tool_call_id when present)
      - LangChain FunctionMessage -> role 'function' (legacy compatibility)
      - LangChain ChatMessage (generic) -> role from message

    Args:
        messages: Sequence of LangChain BaseMessage instances or dicts already containing role/content.

    Returns:
        A list of dictionaries compatible with the OpenAI Chat Completions API 'messages' parameter.
    """

    def _normalize_content(
        _content: typing.Any,
    ) -> str | list[dict[str, typing.Any]]:
        # OpenAI supports either a string or a list of content parts (for multimodal).
        if isinstance(_content, str):
            return _content
        if isinstance(_content, collections.abc.Sequence) and not isinstance(_content, (str, bytes, bytearray)):
            parts: list[dict[str, typing.Any]] = []
            for part in typing.cast("collections.abc.Sequence[typing.Any]", _content):
                if isinstance(part, collections.abc.Mapping):
                    mapped_part = typing.cast("collections.abc.Mapping[str, typing.Any]", part)
                    parts.append(dict(mapped_part))
                    continue
                model_dump_fn = getattr(part, "model_dump", None)
                if callable(model_dump_fn):
                    dumped = model_dump_fn()
                    parts.append(typing.cast("dict[str, typing.Any]", dumped))
                    continue
                parts.append({"type": "text", "text": str(part)})
            return parts
        return str(_content)

    def _normalize_tool_calls(
        tool_calls: collections.abc.Sequence[typing.Any],
    ) -> list[dict[str, typing.Any]]:
        normalized: list[dict[str, typing.Any]] = []
        for tool_call in tool_calls:
            if isinstance(tool_call, collections.abc.Mapping):
                mapping_call = typing.cast("collections.abc.Mapping[str, typing.Any]", tool_call)
                function_section = mapping_call.get("function")
                function_mapping: collections.abc.Mapping[str, typing.Any] | None = None
                if isinstance(function_section, collections.abc.Mapping):
                    function_mapping = typing.cast(
                        "collections.abc.Mapping[str, typing.Any]",
                        function_section,
                    )
                name_value = mapping_call.get("name")
                if name_value is None and function_mapping is not None:
                    name_value = function_mapping.get("name")
                args_value: typing.Any = (
                    mapping_call.get("args")
                    or mapping_call.get("arguments")
                    or (function_mapping.get("arguments") if function_mapping else None)
                )
                if isinstance(args_value, str):
                    args_payload = args_value
                else:
                    serialized_args = orjson.dumps({} if args_value is None else args_value)
                    args_payload = serialized_args.decode("utf-8")
                entry: dict[str, typing.Any] = {
                    "type": "function",
                    "function": {
                        "name": str(name_value or "unknown"),
                        "arguments": args_payload,
                    },
                }
                tool_id = mapping_call.get("id")
                if tool_id is not None:
                    entry["id"] = str(tool_id)
                normalized.append(entry)
                continue
            function_obj = getattr(tool_call, "function", None)
            name_value = getattr(tool_call, "name", None) or getattr(function_obj, "name", None) or "unknown"
            args_value: typing.Any = (
                getattr(tool_call, "args", None)
                or getattr(tool_call, "arguments", None)
                or getattr(function_obj, "arguments", None)
                or {}
            )
            if isinstance(args_value, str):
                args_payload = args_value
            else:
                serialized_args = orjson.dumps(args_value)
                args_payload = serialized_args.decode("utf-8")
            entry = {
                "type": "function",
                "function": {
                    "name": str(name_value),
                    "arguments": args_payload,
                },
            }
            tool_id = getattr(tool_call, "id", None)
            if tool_id is not None:
                entry["id"] = str(tool_id)
            normalized.append(entry)
        return normalized

    oai_messages: list[dict[str, typing.Any]] = []
    for orig_msg in messages:
        if isinstance(orig_msg, collections.abc.Mapping):
            mapping_msg = typing.cast("collections.abc.Mapping[str, typing.Any]", orig_msg)
            role_value = mapping_msg.get("role", "user")
            role = role_value if isinstance(role_value, str) else str(role_value)
            content = _normalize_content(mapping_msg.get("content", ""))
            msg: dict[str, typing.Any] = {"role": role, "content": content}
            name_value = mapping_msg.get("name")
            if name_value is not None:
                msg["name"] = str(name_value)
            tool_call_id_value = mapping_msg.get("tool_call_id")
            if tool_call_id_value is not None and role == "tool":
                msg["tool_call_id"] = str(tool_call_id_value)
            tool_calls_value = mapping_msg.get("tool_calls")
            if (
                isinstance(tool_calls_value, collections.abc.Sequence)
                and not isinstance(tool_calls_value, (str, bytes, bytearray))
                and role == "assistant"
            ):
                normalized_tool_calls = _normalize_tool_calls(
                    typing.cast("collections.abc.Sequence[typing.Any]", tool_calls_value),
                )
                if normalized_tool_calls:
                    msg["tool_calls"] = normalized_tool_calls
            oai_messages.append(msg)
            continue
        content = _normalize_content(getattr(orig_msg, "content", ""))
        if isinstance(orig_msg, langchain_core.messages.SystemMessage):
            msg_dict: dict[str, typing.Any] = {"role": "system", "content": content}
        elif isinstance(orig_msg, langchain_core.messages.HumanMessage):
            msg_dict = {"role": "user", "content": content}
        elif isinstance(orig_msg, langchain_core.messages.AIMessage):
            msg_dict = {"role": "assistant", "content": content}
            tool_calls_attr = getattr(orig_msg, "tool_calls", None)
            if isinstance(tool_calls_attr, collections.abc.Sequence) and not isinstance(
                tool_calls_attr,
                (str, bytes, bytearray),
            ):
                normalized_tool_calls = _normalize_tool_calls(
                    typing.cast("collections.abc.Sequence[typing.Any]", tool_calls_attr),
                )
                if normalized_tool_calls:
                    msg_dict["tool_calls"] = normalized_tool_calls
        elif isinstance(orig_msg, langchain_core.messages.ToolMessage):
            msg_dict = {"role": "tool", "content": content}
            tcid = getattr(orig_msg, "tool_call_id", None)
            if tcid:
                msg_dict["tool_call_id"] = str(tcid)
            name = getattr(orig_msg, "name", None)
            if name:
                msg_dict["name"] = str(name)
        elif isinstance(orig_msg, langchain_core.messages.FunctionMessage):
            # legacy support for function role
            name = getattr(orig_msg, "name", None) or "function"
            msg_dict = {"role": "function", "name": str(name), "content": content}
        elif hasattr(orig_msg, "role"):
            # generic ChatMessage or similar
            msg_dict = {
                "role": str(getattr(orig_msg, "role", "user")),
                "content": content,
            }
        else:
            # fallback: attempt to use .type as role or default to 'user'
            role = str(getattr(orig_msg, "type", "user"))
            msg_dict = {"role": role, "content": content}
        oai_messages.append(msg_dict)
    return oai_messages


if typing.TYPE_CHECKING:

    class OpenAIClientParamsConfig(pydantic.BaseModel):
        """Stubbed interface for the OpenAI client params config class defined below."""

        def __getattr__(self, name: str) -> typing.Any: ...

else:
    OpenAIClientParamsConfig = pyine.utils.pydantic.model_from_callable(
        fn=openai.OpenAI,
        name="OpenAIClientParamsConfig",
        model_config=pydantic.ConfigDict(frozen=True, extra="forbid"),
        default_overrides={"timeout": None, "max_retries": 0},  # SDK retries disabled; backoff handles retry
    )
    """Configuration parameters for the OpenAI client."""


class OpenAIClientConfig(pyine.utils.pydantic.ClassImportSpec):
    """Wrapper over the OpenAI client SDK for easy instantiation."""

    class_path: str = "openai.OpenAI"
    base_class_path: str = "openai.OpenAI"  # not actually relevant/used
    params: dict[str, typing.Any] | pydantic.SerializeAsAny[pydantic.BaseModel] = typing.cast(
        "dict[str, typing.Any] | pydantic.SerializeAsAny[pydantic.BaseModel]",
        OpenAIClientParamsConfig(),
    )


OpenAIFineTunerHyperparamsConfig = openai.types.fine_tuning.fine_tuning_job.Hyperparameters
"""Configuration parameters for the OpenAI fine-tuning job hyperparameters."""

OpenAIFineTunerWandBConfig = openai.types.fine_tuning.fine_tuning_job_wandb_integration.FineTuningJobWandbIntegration
"""Configuration parameters for the OpenAI fine-tuning job WandB integration."""


class OpenAIFineTunerParamsConfig(pydantic.BaseModel):
    """Configuration parameters for the OpenAI fine-tuning API."""

    model_config = pydantic.ConfigDict(frozen=True, extra="allow")
    """Pydantic model configuration (freezes the dataclass)."""

    base_model: str
    """Name of the base model to fine-tune."""
    method: dict[str, typing.Any]
    """The fine-tuning method to use.

    Should be compatible with `openai.types.fine_tuning.job_create_params.Method`.
    """
    hyperparams: OpenAIFineTunerHyperparamsConfig | dict[str, typing.Any] | None = None
    """Hyperparameters for the fine-tuning job."""
    seed: int | None = None
    """Seed used to control the reproducibility of the job.

    Passing in the same seed and job parameters should produce the same results, but may differ
    in rare cases. If a seed is not specified, one will be generated for you.
    """
    suffix: str | None = None
    """A string of up to 64 characters that will be added to your fine-tuned model name."""
    wandb_integration: OpenAIFineTunerWandBConfig | None = None
    """Configuration for WandB integration (optional)."""
    metadata: dict[str, str] | None = None
    """Additional metadata to attach to the fine-tuning job.

    This can be useful for storing additional information about the object in a structured format,
    and querying for objects via API or the dashboard.

    Keys are strings with a maximum length of 64 characters. Values are strings with a maximum
    length of 512 characters. Up to 16 key-value pairs can be provided. If more than 16 key-value
    pairs are provided, we will discard the extra pairs.
    """
    timeout_override: float | None = None
    """Override of the default client timeout for the fine-tuning job."""
    file_upload_params: dict[str, typing.Any] = pydantic.Field(
        default_factory=lambda: typing.cast("dict[str, typing.Any]", {}),
    )
    """Additional parameters to pass to the Files API when uploading fine-tuning files."""
    files_purpose: str | None = "fine-tune"
    """Default 'purpose' to use with the Files API (set to None to not send a purpose)."""
    job_params: dict[str, typing.Any] = pydantic.Field(
        default_factory=lambda: typing.cast("dict[str, typing.Any]", {}),
    )
    """Additional fine-tuning job parameters to pass through to jobs.create (method-specific)."""


class OpenAIFineTuner:
    """Thin wrapper over the OpenAI SDK for common fine-tuning operations.

    Retry Policy:
        All API-calling methods use exponential backoff (via the ``backoff`` library) to handle
        transient errors (timeouts, rate limits, 5xx). The OpenAI client should be created with
        ``max_retries=0`` to avoid double-retry; this is the default when using
        ``OpenAIClientParamsConfig``. See ``RETRYABLE_EXCEPTIONS`` for the exception types that
        trigger retry.
    """

    def __init__(
        self,
        client: openai.OpenAI,
        config: OpenAIFineTunerParamsConfig | dict[str, typing.Any],
    ) -> None:
        """
        Args:
            client: An initialized OpenAI client instance. Should have ``max_retries=0`` to avoid
                double-retry (this is the default when using ``OpenAIClientParamsConfig``).
            config: Fine-tuning configuration parameters.
        """
        self.client = client
        if isinstance(config, dict):
            config = OpenAIFineTunerParamsConfig(**config)
        self.config = config

    @_retry_openai_api
    def get_remote_file_size(
        self,
        file_id: str,
    ) -> int:
        """Retrieve the size (in bytes) of a remote file.

        Args:
            file_id: The file id in the OpenAI Files API.

        Returns:
            Size in bytes as an integer.
        """
        logger.debug(f"getting remote file size: {file_id}")
        file_info = self.client.files.retrieve(file_id)
        return file_info.bytes

    @_retry_openai_api
    def get_remote_file_hash(
        self,
        file_id: str,
    ) -> str:
        """Compute SHA-256 of the remote file's raw bytes.

        Args:
            file_id: The file id in the OpenAI Files API.

        Returns:
            Hex-encoded SHA-256 of the file content.
        """
        logger.debug(f"getting remote file hash: {file_id}")
        content = self.client.files.content(file_id).read()
        return pyine.utils.reprod.compute_hash(content)

    @_retry_openai_api
    def list_remote_files(
        self,
        purpose: (str | None) = None,  # note: we do not override this one with internal config value
        pattern: str | None = None,  # optional regex pattern for matching
    ) -> list[openai.types.FileObject]:
        """Returns remote files that are available for a specific purpose (or for any)."""
        page = self.client.files.list(purpose=purpose) if purpose is not None else self.client.files.list()
        remote_files = list(page.data)
        if pattern is not None:
            regex: typing.Pattern[str] = re.compile(pattern)
            return [f for f in remote_files if regex.match(f.filename)]
        return remote_files

    @_retry_openai_api
    def list_remote_models(
        self,
        pattern: str | None = None,  # optional regex pattern for matching
    ) -> list[openai.types.model.Model]:
        """Return fine-tuned models that are available for this account."""
        page = self.client.models.list()
        models = [model for model in page.data if model.id.startswith("ft:")]
        if pattern is not None:
            regex: typing.Pattern[str] = re.compile(pattern)
            return [model for model in models if regex.match(model.id)]
        return models

    @_retry_openai_api
    def list_finetuning_jobs(
        self,
        pattern: str | None = None,
    ) -> list[openai.types.fine_tuning.fine_tuning_job.FineTuningJob]:
        """Return fine-tuning jobs, optionally filtered by the resulting model id pattern."""
        page = self.client.fine_tuning.jobs.list()
        jobs = list(page.data)
        if pattern is None:
            return jobs
        regex: typing.Pattern[str] = re.compile(pattern)
        filtered_jobs: list[openai.types.fine_tuning.fine_tuning_job.FineTuningJob] = []
        for job in jobs:
            model_id = getattr(job, "fine_tuned_model", "") or ""
            if model_id and regex.match(model_id):
                filtered_jobs.append(job)
        return filtered_jobs

    @_retry_openai_api
    def is_file_already_uploaded(
        self,
        local_path: pathlib.Path | str,
        confirm_by_hash: bool = True,
        purpose: str | None = None,
    ) -> str | None:
        """Return an existing Files API id if a given local file is already uploaded.

        Quickly filters potential matches by filename and size, and optionally compares SHA-256.

        Args:
            local_path: Path to the local file to be potentially uploaded.
            confirm_by_hash: If True, confirm equality by SHA-256 of remote bytes.
            purpose: Optional Files API purpose to filter by; defaults to config.files_purpose.

        Returns:
            File id if a match is found; otherwise None.
        """
        local_path = pathlib.Path(local_path)
        if not local_path.is_file():
            raise FileNotFoundError(f"local file does not exist: {local_path}")
        local_size = local_path.stat().st_size
        if purpose is None:
            purpose = self.config.files_purpose
        page = self.client.files.list(purpose=purpose) if purpose is not None else self.client.files.list()
        candidates: list[typing.Any] = []
        for remote in page.data:
            if remote.filename == local_path.name and int(remote.bytes) == int(local_size):
                candidates.append(remote)
        if not candidates:
            return None
        if not confirm_by_hash:
            if len(candidates) != 1:
                raise RuntimeError(f"multiple candidates found for {local_path}: {candidates}")
            return candidates[0].id
        local_hash = pyine.utils.reprod.compute_hash(local_path)
        for remote in candidates:
            if self.get_remote_file_hash(remote.id) == local_hash:
                return remote.id
        return None

    @_retry_openai_api
    def upload_file(
        self,
        path: pathlib.Path | str,
        purpose: str | None = None,
    ) -> str:
        """Upload a local file to the Files API.

        Args:
            path: Local path to the file to upload.
            purpose: Optional Files API purpose; defaults to config.files_purpose. If None,
                no explicit purpose is sent unless provided via file_upload_params.

        Returns:
            The uploaded file id.
        """
        path = pathlib.Path(path)
        file_size = pyine.utils.filesystem.get_human_readable_size(path.stat().st_size)
        logger.info(f"uploading {file_size} file: {path}")
        with open(path, "rb") as f:
            kwargs = dict(self.config.file_upload_params)
            if purpose is None:
                purpose = self.config.files_purpose
            # Only set purpose if not provided explicitly via file_upload_params and not None here
            if purpose is not None and "purpose" not in kwargs:
                kwargs["purpose"] = purpose
            up = self.client.files.create(
                file=f,
                **kwargs,
            )
        logger.info(f"file uploaded; id={up.id}")
        return up.id

    def ensure_uploaded(
        self,
        path: pathlib.Path | str,
        confirm_by_hash: bool = True,
        purpose: str | None = None,
    ) -> str:
        """Ensure the file is uploaded and return its file id.

        If a matching remote file already exists (by name, size, and optionally hash),
        it will be reused; otherwise the file will be uploaded.

        Args:
            path: Local path to the file to upload.
            confirm_by_hash: Whether to verify by SHA-256 content hash.
            purpose: Optional Files API purpose to filter/search and upload with; defaults to config.files_purpose.

        Returns:
            Files API id of the existing or uploaded file.
        """
        existing = self.is_file_already_uploaded(path, confirm_by_hash=confirm_by_hash, purpose=purpose)
        if existing:
            logger.info(f"file already uploaded: {path} -> id={existing}")
            return existing
        return self.upload_file(str(path), purpose=purpose)

    @_retry_openai_api
    def download_file(
        self,
        file_id: str,
        dest_path: pathlib.Path | str,
        overwrite: bool = False,
    ) -> pathlib.Path:
        """Download a remote file to a local destination.

        Args:
            file_id: The remote file id.
            dest_path: Local destination path.
            overwrite: If False and the file exists, raises FileExistsError.

        Returns:
            The destination path as a Path object.
        """
        dest = pathlib.Path(dest_path)
        if dest.exists() and not overwrite:
            raise FileExistsError(f"cannot overwrite existing file: {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"downloading file {file_id} -> {dest}")
        data = self.client.files.content(file_id).read()
        with open(dest, "wb") as f:
            f.write(data)
        file_size = pyine.utils.filesystem.get_human_readable_size(dest.stat().st_size)
        logger.info(f"downloaded {file_size} file to: {dest}")
        return dest

    @_retry_openai_api
    def create_job(
        self,
        training_file_id: str | None = None,
        validation_file_id: str | None = None,
        extra_file_args: dict[str, str | None] | None = None,
        extra_job_params: dict[str, typing.Any] | None = None,
    ) -> str:
        """Create a fine-tuning job.

        Args:
            training_file_id: Files API id of the training dataset (optional if provided via extra_file_args).
            validation_file_id: Files API id of the validation dataset (optional).
            extra_file_args: Additional file arguments to pass to the job creation call. For RL-style
                approaches, pass fields like {'preference_file': 'file-abc'} or {'reward_file': 'file-def'}
                depending on the selected method.
            extra_job_params: Additional method-specific job parameters to pass through.

        Returns:
            The fine-tuning job id.
        """
        integrations = None
        if self.config.wandb_integration is not None:
            integrations = [self.config.wandb_integration.model_dump()]
        if self.config.metadata is not None:
            # openai api supports at most 16 key-value pairs, so we truncate the metadata dict
            metadata = {k: self.config.metadata[k] for k in list(self.config.metadata)[:16]}
        else:
            metadata = None
        params: dict[str, typing.Any] = {
            "model": self.config.base_model,
            "integrations": integrations,
            "metadata": metadata,
            "method": self.config.method,
            "seed": self.config.seed,
            "suffix": self.config.suffix,
            "timeout": self.config.timeout_override,
        }
        if self.config.hyperparams is not None:
            if isinstance(self.config.hyperparams, dict):
                hyperparams = self.config.hyperparams
            else:
                hyperparams = self.config.hyperparams.model_dump()
            params["hyperparameters"] = hyperparams
        if training_file_id is not None:
            params["training_file"] = training_file_id
        if validation_file_id is not None:
            params["validation_file"] = validation_file_id
        if extra_file_args:
            params.update({k: v for k, v in extra_file_args.items() if v is not None})
        if self.config.job_params:
            params.update(self.config.job_params)
        if extra_job_params:
            params.update(extra_job_params)
        params_str = pprint.pformat(params)
        logger.info(f"creating fine-tune job with config:\n{params_str}")
        job = self.client.fine_tuning.jobs.create(**params)
        logger.info(f"fine-tuning job created, id={job.id}, status={job.status}")
        return job.id

    @_retry_openai_api
    def _list_job_events(
        self,
        job_id: str,
    ) -> typing.Any:
        """List job events with backoff (internal helper for stream_job_events)."""
        return self.client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id)

    def stream_job_events(
        self,
        job_id: str,
    ) -> None:
        """Stream fine-tuning job events for the given id until interrupted."""
        # pragma: no cover
        logger.info("streaming fine-tune events (Ctrl-C to stop streaming)...")
        for evt in self._list_job_events(job_id):
            # event typically has .created_at, .level, .message depending on SDK version
            created = getattr(evt, "created_at", None)
            level = getattr(evt, "level", "info")
            msg = getattr(evt, "message", None)
            if not msg:
                data = getattr(evt, "data", None)
                if data is not None:
                    try:
                        msg = orjson.dumps(data).decode("utf-8")
                    except Exception:
                        msg = str(data)
                else:
                    msg = str(evt)
            logger.info(f"[{created}][{level}] {msg}")

    @_retry_openai_api
    def _retrieve_job(
        self,
        job_id: str,
    ) -> openai.types.fine_tuning.fine_tuning_job.FineTuningJob:
        """Retrieve a fine-tuning job with backoff (used internally by wait_for_job)."""
        return self.client.fine_tuning.jobs.retrieve(job_id)

    def wait_for_job(
        self,
        job_id: str,
        poll_seconds: float = 5,
        timeout_seconds: float = 60 * 60,
        cancel_on_timeout: bool = False,
    ) -> str:
        """Poll the job until it reaches a terminal state or a timeout occurs.

        Args:
            job_id: The fine-tuning job id.
            poll_seconds: Seconds to wait between polling attempts.
            timeout_seconds: Max time to wait before raising TimeoutError.
            cancel_on_timeout: If True, attempt to cancel the job on timeout.

        Returns:
            The fine-tuned model id if available, otherwise an empty string.

        Raises:
            TimeoutError: If timeout_seconds is exceeded.
        """
        logger.info(f"polling job until terminal state: {job_id}")
        start = time.time()
        while True:
            job = self._retrieve_job(job_id)
            if job.status in {"succeeded", "failed", "cancelled"}:
                logger.info(f"found terminal state: {job.status}")
                model_id = getattr(job, "fine_tuned_model", None) or ""
                if model_id:
                    logger.info(f"resulting fine-tuned model id: {model_id}")
                return model_id
            if time.time() - start > timeout_seconds:
                if cancel_on_timeout:
                    try:
                        self.cancel_job(job_id)
                        logger.warning(f"timed out and attempted to cancel job: {job_id}")
                    except Exception as e:
                        logger.exception(f"timed out and failed to cancel job {job_id}")
                        raise e
                raise TimeoutError("timed out waiting for fine-tuning job")
            time.sleep(poll_seconds)

    @_retry_openai_api
    def cancel_job(self, job_id: str) -> None:
        """Cancels a fine-tuning job."""
        logger.info(f"cancelling fine-tune job: {job_id}")
        result = self.client.fine_tuning.jobs.cancel(fine_tuning_job_id=job_id)
        logger.info(f"cancel requested; new status: {getattr(result, 'status', 'unknown')}")

    @_retry_openai_api
    def chat(
        self,
        model_id: str,
        system_prompt: str | None,
        user_prompt: str,
        **kwargs: typing.Any,
    ) -> str:
        """Send a simple chat completion request and return the assistant reply.

        Args:
            model_id: The previously fine-tuned model id to use.
            system_prompt: Optional system message.
            user_prompt: The user prompt.
            kwargs: Additional arguments to pass to the OpenAI API.

        Returns:
            Assistant's message content, or an empty string if not present.
        """
        messages: list[typing.Any] = []
        if system_prompt:
            messages.append(SystemMessageType(content=system_prompt, role="system"))
        messages.append(UserMessageType(content=user_prompt, role="user"))
        chat_response = typing.cast(
            "openai.types.chat.ChatCompletion",
            self.client.chat.completions.create(
                model=model_id,
                messages=messages,
                **kwargs,
            ),
        )
        return chat_response.choices[0].message.content or ""


class OpenAIFineTunerConfig(pyine.utils.pydantic.ClassImportSpec):
    """Wrapper over the OpenAI client SDK for easy instantiation."""

    class_path: str = pyine.utils.portability.get_fully_qualified_name(OpenAIFineTuner)
    base_class_path: str = pyine.utils.portability.get_fully_qualified_name(OpenAIFineTuner)
    params: dict[str, typing.Any] | pydantic.SerializeAsAny[pydantic.BaseModel] = typing.cast(
        "dict[str, typing.Any] | pydantic.SerializeAsAny[pydantic.BaseModel]",
        OpenAIFineTunerParamsConfig(
            base_model="",
            method={"type": "supervised"},
        ),
    )
    """Configuration parameters for the OpenAI fine-tuning API helper."""
    params_key: str | None = "config"
    """Key to use when passing the params configuration to the class constructor."""


@_retry_openai_api
def _list_models(
    client: openai.OpenAI,
) -> typing.Any:
    """List models with backoff (internal helper for cleanup)."""
    return client.models.list()


@_retry_openai_api
def _delete_model(
    client: openai.OpenAI,
    model_id: str,
) -> None:
    """Delete a model with backoff (internal helper for cleanup)."""
    client.models.delete(model_id)


@_retry_openai_api
def _list_files(
    client: openai.OpenAI,
    purpose: str | None = None,
) -> typing.Any:
    """List files with backoff (internal helper for cleanup)."""
    return client.files.list(purpose=purpose) if purpose is not None else client.files.list()


@_retry_openai_api
def _delete_file(
    client: openai.OpenAI,
    file_id: str,
) -> None:
    """Delete a file with backoff (internal helper for cleanup)."""
    client.files.delete(file_id)


def cleanup_finetuned_models(
    client: openai.OpenAI,
    pattern: str,
    max_age_days: int = 0,  # if zero, will clean up all matched models
    dry_run: bool = False,
) -> list[openai.types.model.Model]:
    """Scans fine-tuned models with the given pattern and delete ones older than `max_age_days`.

    Args:
        client: OpenAI client instance.
        pattern: Regex string to match model IDs (e.g., r"^ft:.*:dummy$")
        max_age_days: Maximum allowed model age in days before deletion; if zero, all matched
            models will be deleted (default 0).
        dry_run: If True, only print what would be deleted (default False).

    Returns:
        A list of cleaned up models (or models that would have been deleted if dry_run=True).
    """
    regex: typing.Pattern[str] = re.compile(pattern)
    cutoff_time = datetime.datetime.now() - datetime.timedelta(days=max_age_days)
    models = _list_models(client)
    matched_models: list[openai.types.model.Model] = []
    for model in models.data:
        creation_dt = datetime.datetime.fromtimestamp(model.created)
        if regex.match(model.id) and creation_dt < cutoff_time:
            age_days = (datetime.datetime.now() - creation_dt).days
            matched_models.append(model)
            if dry_run:
                logger.info(f"[DRY RUN] Would delete {model.id} (age={age_days} days)")
            else:
                logger.info(f"Deleting {model.id} (age={age_days} days)")
                _delete_model(client, model.id)
    return matched_models


def cleanup_files(
    client: openai.OpenAI,
    pattern: str,
    purpose: str | None = None,
    max_age_days: int = 0,  # if zero, will clean up all matched files
    dry_run: bool = False,
) -> list[openai.types.file_object.FileObject]:
    """Scans remote files with the given pattern and delete ones older than `max_age_days`.

    Args:
        client: OpenAI client instance.
        pattern: Regex string to match file names (e.g., r"^.*.dummy.jsonl$")
        purpose: Optional Files API purpose to filter by..
        max_age_days: Maximum allowed file age in days before deletion; if zero, all matched
            files will be deleted (default 0).
        dry_run: If True, only print what would be deleted (default False).

    Returns:
        A list of cleaned up files (or that would be cleaned up if dry_run=False).
    """
    regex: typing.Pattern[str] = re.compile(pattern)
    cutoff_time = datetime.datetime.now() - datetime.timedelta(days=max_age_days)
    file_page = _list_files(client, purpose)
    matched_files: list[openai.types.file_object.FileObject] = []
    for file in file_page.data:
        creation_dt = datetime.datetime.fromtimestamp(file.created_at)
        if regex.match(file.filename) and creation_dt < cutoff_time:
            age_days = (datetime.datetime.now() - creation_dt).days
            matched_files.append(file)
            if dry_run:
                logger.info(f"[DRY RUN] Would delete {file.id} (name={file.filename}; age={age_days} days)")
            else:
                logger.info(f"Deleting {file.id} (name={file.filename}; age={age_days} days)")
                _delete_file(client, file.id)
    return matched_files
