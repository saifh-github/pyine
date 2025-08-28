import datetime
import logging
import pathlib
import re
import time
import typing

import backoff
import openai
import openai.types.chat
import openai.types.fine_tuning
import orjson
import pydantic

import pyine.utils.filesystem
import pyine.utils.portability
import pyine.utils.pydantic
import pyine.utils.reprod

logger = logging.getLogger(__name__)

SystemMessageType = openai.types.chat.chat_completion_system_message_param.ChatCompletionSystemMessageParam
UserMessageType = openai.types.chat.chat_completion_user_message_param.ChatCompletionUserMessageParam
AssistantMessageType = openai.types.chat.chat_completion_assistant_message_param.ChatCompletionAssistantMessageParam


def get_local_file_directory() -> pathlib.Path:
    """Returns the path used to store prepared OpenAI data upload files in the local tmpdir."""
    tmpdir = pyine.utils.filesystem.get_tmp_dir() / "openai-data"
    tmpdir.mkdir(parents=True, exist_ok=True)
    return tmpdir


def write_dataset_to_jsonl(
    dataset: typing.Iterable[dict | list],
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
    samples_str = []
    for sample in dataset:
        if isinstance(sample, dict):
            assert "messages" in sample
            msgs = sample["messages"]  # drop every other field except messages
        else:
            msgs = sample
        assert isinstance(msgs, list)
        assert all([isinstance(m, dict) and all(f in m for f in ["role", "content"]) for m in msgs])
        samples_str.append(orjson.dumps({"messages": msgs}).decode("utf-8"))
    if enforce_openai_min_dataset_size and len(samples_str) < 10:
        raise ValueError(f"dataset must have at least 10 samples, got {len(samples_str)}")
    with open(path, "w", encoding="utf-8") as fd:
        fd.write("\n".join(samples_str))
    dataset_size = pyine.utils.filesystem.get_human_readable_size(path.stat().st_size)
    logger.debug(f"wrote {dataset_size} dataset with {len(samples_str)} samples to {path}")


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
            if isinstance(obj, list):
                msgs = obj
            elif isinstance(obj, dict):
                if "messages" in obj and isinstance(obj["messages"], list):
                    msgs = obj["messages"]
                else:
                    raise ValueError(f"line {lineno}: expected list or a dict with 'messages'")
            else:
                raise ValueError(f"line {lineno}: unsupported JSON type {type(obj).__name__}")
            curr_messages: list[dict[str, str]] = []
            for idx, msg in enumerate(msgs, start=1):
                if not isinstance(msg, dict):
                    raise ValueError(f"line {lineno}: message {idx} is not a dict")
                if "role" not in msg or "content" not in msg:
                    raise ValueError(f"line {lineno}: message {idx} missing 'role' or 'content' keys")
                role, content = msg["role"], msg["content"]
                if not isinstance(role, str) or not isinstance(content, str):
                    # coerce to strings to ensure consistent downstream handling
                    msg = dict(msg)
                    msg["role"] = str(role)
                    msg["content"] = str(content)
                curr_messages.append(msg)
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
    import pyine.organisms.models.utils.tokenizers

    tokenizer = pyine.organisms.models.utils.tokenizers.get_openai_tokenizer(
        model_id=model_id,
        raise_if_not_found=False,
    )
    return len(tokenizer.encode(text))


class OpenAIClientParamsConfig(pydantic.BaseModel):
    """Configuration parameters for the OpenAI client SDK."""

    # note: we don't actually specify much here, just a default value for the project name
    # (it would be unwise to hardcode anything specific here, especially API keys or org names)
    model_config = pydantic.ConfigDict(frozen=True, extra="allow")
    """Pydantic model configuration (freezes the dataclass)."""


class OpenAIClientConfig(pyine.utils.pydantic.ClassImportSpec[openai.OpenAI]):
    """Wrapper over the OpenAI client SDK for easy instantiation."""

    class_path: str = "openai.OpenAI"
    base_class_path: str = "openai.OpenAI"  # not actually relevant/used
    params: OpenAIClientParamsConfig = OpenAIClientParamsConfig()


class OpenAIFineTunerHyperparamsConfig(pydantic.BaseModel):
    """Configuration parameters for the OpenAI fine-tuning job hyperparameters.

    Note: all 'auto' values let OpenAI decide what to use by default.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="allow")
    """Pydantic model configuration (freezes the dataclass)."""
    batch_size: typing.Annotated[
        typing.Literal["auto"] | pydantic.PositiveInt | None,
        pydantic.Field(description="Batch size to use for fine-tuning"),
    ] = "auto"
    learning_rate_multiplier: typing.Annotated[
        typing.Literal["auto"] | pydantic.PositiveFloat | None,
        pydantic.Field(description="Learning rate multiplier to use for fine-tuning"),
    ] = "auto"
    n_epochs: typing.Annotated[
        typing.Literal["auto"] | pydantic.PositiveInt | None,
        pydantic.Field(description="Number of fine-tuning epochs (must be >= 1)"),
    ] = "auto"


class OpenAIFineTunerWandBConfig(pydantic.BaseModel):
    """Configuration parameters for the OpenAI fine-tuning job with WandB integration."""

    model_config = pydantic.ConfigDict(frozen=True, extra="allow")
    """Pydantic model configuration (freezes the dataclass)."""
    project: str
    """The name of the project that the new run will be created under."""
    entity: str | None
    """The entity to use for the run.

    This allows you to set the team or username of the WandB user that you would
    like associated with the run. If not set, the default entity for the registered
    WandB API key is used.
    """
    name: str | None
    """A display name to set for the run.

    If not set, we will use the Job ID as the name.
    """
    tags: list[str]
    """A list of tags to be attached to the newly created run.

    These tags are passed through directly to WandB. Some default tags are generated
    by OpenAI: "openai/finetune", "openai/{base-model}", "openai/{ftjob-abcdef}".
    """


class OpenAIFineTunerParamsConfig(pydantic.BaseModel):
    """Configuration parameters for the OpenAI fine-tuning API."""

    model_config = pydantic.ConfigDict(frozen=True, extra="allow")
    """Pydantic model configuration (freezes the dataclass)."""

    base_model: str
    """Name of the base model to fine-tune."""
    method: openai.types.fine_tuning.job_create_params.Method | None = None
    """The fine-tuning method to use. If `None`, defaults to the OpenAI 'not-given' default."""
    hyperparams: OpenAIFineTunerHyperparamsConfig = OpenAIFineTunerHyperparamsConfig()
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

    This can be useful for storing additional information about the object in a
    structured format, and querying for objects via API or the dashboard.

    Keys are strings with a maximum length of 64 characters. Values are strings with
    a maximum length of 512 characters. Up to 16 key-value pairs can be provided.
    """
    timeout_override: float | None = None
    """Override of the default client timeout for the fine-tuning job."""
    file_upload_params: dict[str, typing.Any] = pydantic.Field(default_factory=dict)
    """Additional parameters to pass to the Files API when uploading fine-tuning files."""
    files_purpose: str | None = "fine-tune"
    """Default 'purpose' to use with the Files API (set to None to not send a purpose)."""
    job_params: dict[str, typing.Any] = pydantic.Field(default_factory=dict)
    """Additional fine-tuning job parameters to pass through to jobs.create (method-specific)."""


class OpenAIFineTuner:
    """Thin wrapper over the OpenAI SDK for common fine-tuning operations."""

    def __init__(
        self,
        client: openai.OpenAI,
        config: OpenAIFineTunerParamsConfig,
    ) -> None:
        """
        Args:
            client: An initialized OpenAI client instance.
        """
        self.client = client
        self.config = config

    @backoff.on_exception(backoff.expo, Exception, max_time=120)
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

    @backoff.on_exception(backoff.expo, Exception, max_time=120)
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

    def list_remote_files(
        self,
        purpose: str | None = None,  # note: we do not override this one with internal config value
        pattern: str | None = None,  # optional regex pattern for matching
    ) -> list[openai.types.FileObject]:
        """Returns remote files that are available for a specific purpose (or for any)."""
        page = self.client.files.list(purpose=purpose) if purpose is not None else self.client.files.list()
        remote_files = list(page.data)
        if pattern is not None:
            regex: typing.Pattern[str] = re.compile(pattern)
            return [f for f in remote_files if regex.match(f.filename)]
        return remote_files

    def list_remote_models(
        self,
        pattern: str | None = None,  # optional regex pattern for matching
    ) -> list[openai.types.fine_tuning.fine_tuning_job.FineTuningJob]:
        """Returns remote models that have been created via fine-tuning."""
        page = self.client.fine_tuning.jobs.list()
        models = list(page.data)
        if pattern is not None:
            regex: typing.Pattern[str] = re.compile(pattern)
            return [m for m in models if m.fine_tuned_model and regex.match(m.fine_tuned_model)]
        return models

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

    @backoff.on_exception(backoff.expo, Exception, max_time=120)
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
            logger.debug(f"file already uploaded: {path} -> id={existing}")
            return existing
        return self.upload_file(str(path), purpose=purpose)

    @backoff.on_exception(backoff.expo, Exception, max_time=120)
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
        logger.debug(f"downloaded {file_size} file to: {dest}")
        return dest

    @backoff.on_exception(backoff.expo, Exception, max_time=120)
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
            extra_file_args: Additional file arguments to pass to the job creation call, e.g.,
                {'preference_file': 'file-abc', 'reward_file': 'file-def'} for RL methods.
            extra_job_params: Additional method-specific job parameters to pass through.

        Returns:
            The fine-tuning job id.
        """
        logger.debug("creating fine-tune job...")
        integrations = None
        if self.config.wandb_integration is not None:
            integrations = [self.config.wandb_integration.model_dump()]

        params: dict[str, typing.Any] = {
            "model": self.config.base_model,
            "hyperparameters": self.config.hyperparams.model_dump(),
            "integrations": integrations,
            "metadata": self.config.metadata,
            "method": self.config.method,
            "seed": self.config.seed,
            "suffix": self.config.suffix,
            "timeout": self.config.timeout_override,
        }
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

        job = self.client.fine_tuning.jobs.create(**params)
        logger.info(f"fine-tuning job created, id={job.id}, status={job.status}")
        return job.id

    def stream_job_events(
        self,
        job_id: str,
    ) -> None:
        """Stream fine-tuning job events for the given id until interrupted."""
        # pragma: no cover
        logger.info("streaming fine-tune events (Ctrl-C to stop streaming)...")
        for evt in self.client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id):
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
            job = self.client.fine_tuning.jobs.retrieve(job_id)
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
                        logger.error(f"timed out and failed to cancel job {job_id}: {e}")
                        raise e
                raise TimeoutError("timed out waiting for fine-tuning job")
            time.sleep(poll_seconds)

    @backoff.on_exception(backoff.expo, Exception, max_time=120)
    def cancel_job(self, job_id: str) -> None:
        """Cancels a fine-tuning job."""
        logger.info(f"cancelling fine-tune job: {job_id}")
        result = self.client.fine_tuning.jobs.cancel(fine_tuning_job_id=job_id)
        logger.info(f"cancel requested; new status: {getattr(result, 'status', 'unknown')}")

    def chat(
        self,
        model_id: str,
        system_prompt: str | None,
        user_prompt: str,
        **kwargs,
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
        messages = []
        if system_prompt:
            messages.append(SystemMessageType(content=system_prompt, role="system"))
        messages.append(UserMessageType(content=user_prompt, role="user"))
        resp = self.client.chat.completions.create(
            model=model_id,
            messages=messages,
            **kwargs,
        )
        return resp.choices[0].message.content or ""


class OpenAIFineTunerConfig(pyine.utils.pydantic.ClassImportSpec[OpenAIFineTuner]):
    """Wrapper over the OpenAI client SDK for easy instantiation."""

    class_path: str = pyine.utils.portability.get_fully_qualified_name(OpenAIFineTuner)
    base_class_path: str = pyine.utils.portability.get_fully_qualified_name(OpenAIFineTuner)
    params: OpenAIFineTunerParamsConfig  # note: params need to be specified explicitly
    """Configuration parameters for the OpenAI fine-tuning API helper."""
    params_key: str = "config"
    """Key to use when passing the params configuration to the class constructor."""


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
    models = client.models.list()
    matched_models = []
    for model in models.data:
        creation_dt = datetime.datetime.fromtimestamp(model.created)
        if regex.match(model.id) and creation_dt < cutoff_time:
            age_days = (datetime.datetime.now() - creation_dt).days
            matched_models.append(model)
            if dry_run:
                logger.info(f"[DRY RUN] Would delete {model.id} (age={age_days} days)")
            else:
                logger.info(f"Deleting {model.id} (age={age_days} days)")
                client.models.delete(model.id)
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
    file_page = client.files.list(purpose=purpose) if purpose is not None else client.files.list()
    matched_files = []
    for file in file_page.data:
        creation_dt = datetime.datetime.fromtimestamp(file.created_at)
        if regex.match(file.filename) and creation_dt < cutoff_time:
            age_days = (datetime.datetime.now() - creation_dt).days
            matched_files.append(file)
            if dry_run:
                logger.info(f"[DRY RUN] Would delete {file.id} (name={file.filename}; age={age_days} days)")
            else:
                logger.info(f"Deleting {file.id} (name={file.filename}; age={age_days} days)")
                client.files.delete(file.id)
    return matched_files
