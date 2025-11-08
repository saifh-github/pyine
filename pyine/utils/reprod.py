import functools
import hashlib
import importlib
import importlib.metadata
import json
import logging
import os
import pathlib
import platform
import re
import sys
import time
import typing
import warnings

import dotenv
import lightning.fabric.utilities.seed
import pydantic
import pydantic_core
import rich
import torch
import transformers
import wandb

import pyine.utils.distrib as distrib_utils
import pyine.utils.portability as portab_utils
import pyine.utils.pydantic as pydantic_utils

_ = portab_utils  # always used for yaml path dump fixer (triggered at import time)
_ = pydantic_utils  # always used for pydantic serialization hooks registration

if typing.TYPE_CHECKING:
    import pyine.configs.schemas  # noqa

logger = logging.getLogger(__name__)


class DryRunExit(SystemExit):
    """Exception raised when a dry run is requested."""

    pass


def get_python_version() -> str:
    """Return current Python version, e.g. '3.12.6'."""
    return platform.python_version()


def get_platform_name() -> str:
    """Returns a print-friendly platform name that can be used for logs / data tagging."""
    return str(platform.uname())


def get_timestamp(time_since_epoch: float | None = None) -> str:
    """Returns a print-friendly timestamp (year, month, day, hour, minute, second) for logs."""
    if time_since_epoch is None:
        time_since_epoch = time.time()
    local_time = time.localtime(time_since_epoch)
    return time.strftime("%Y%m%d-%H%M%S", local_time)


def get_framework_version() -> str:
    """Returns the version of the benchmark framework (MAJOR.MINOR.PATCH).

    Falls back to a safe sentinel when the package is not installed so that developer workflows
    (editable installs, direct source runs) do not break.
    """
    try:
        package_version = importlib.metadata.version("pyine")
    except importlib.metadata.PackageNotFoundError:
        # in dev environments or when running from source without an installed package,
        # return a safe sentinel instead of raising to avoid hard dependency during metadata collection
        return "0.0.0-unknown"
    return package_version


def get_git_revision_hash() -> str:
    """Returns a print-friendly hash (SHA1 signature) for the underlying git repository (if found).

    If no git repository is found, the function will return a static string.
    """
    try:
        import git  # noqa
    except (ImportError, AttributeError):
        return "git-import-error"
    try:
        repo = git.Repo(path=os.path.abspath(__file__), search_parent_directories=True)
        sha = repo.head.object.hexsha
        return str(sha)
    except (AttributeError, ValueError, git.InvalidGitRepositoryError):
        return "git-revision-unknown"


def is_git_repo_clean(include_untracked: bool = False) -> bool:
    """Returns whether the underlying git repository is clean (no uncommitted changes).

    By 'clean', we mean no staged changes, no unstaged changes, and optionally no untracked files.

    If no git repository is found (or if the git package is not found), the function will return False.
    """
    try:
        import git  # noqa
    except (ImportError, AttributeError):
        return False
    try:
        repo = git.Repo(path=os.path.abspath(__file__), search_parent_directories=True)
        return not repo.is_dirty(
            index=True,
            working_tree=True,
            untracked_files=include_untracked,
            submodules=True,
        )
    except (AttributeError, ValueError, git.InvalidGitRepositoryError):
        return False


def get_installed_packages() -> list[str]:
    """Returns a list of all packages installed in the current environment.

    If the required packages cannot be imported, the returned list will be empty. Note that some
    packages may not be properly detected by this approach, and it is pretty hacky, so use it with a
    grain of salt (i.e. just for logging is fine).
    """
    pkgs: list[str] = []
    try:
        distributions = importlib.metadata.distributions()
        pkgs = [f"{pkg.name}=={pkg.version}" for pkg in distributions]
    except (ImportError, AttributeError):
        try:
            pip_module = typing.cast("typing.Any", importlib.import_module("pip"))
            get_distributions = getattr(pip_module, "get_installed_distributions", None)
            if callable(get_distributions):
                pkgs = []
                for distribution in typing.cast("typing.Iterable[typing.Any]", get_distributions()):
                    key = getattr(distribution, "key", None) or getattr(distribution, "project_name", None)
                    version = getattr(distribution, "version", None)
                    if key is None or version is None:
                        continue
                    pkgs.append(f"{key}=={version}")
        except (ImportError, AttributeError):
            pkgs = []
    return sorted(pkgs, key=str.casefold)  # noqa


def get_params_hash(*args: typing.Any, **kwargs: typing.Any) -> str:
    """Computes and returns the hash (SHA1 checksum) of a given set of parameters.

    Args:
        *args: Positional arguments to include in the hash computation. Converted to strings
            using their repr() representation.
        **kwargs: Keyword arguments to include in the hash computation. Converted to strings
            using their repr() representation.

    Returns:
        The hashing result as a string of hexadecimal digits.
    """
    # by default, will use the repr of all params but remove the 'at 0x00000000' addresses
    clean_str = re.sub(r" at 0x[a-fA-F\d]+", "", str(args) + str(kwargs))
    return hashlib.sha1(clean_str.encode(), usedforsecurity=False).hexdigest()


def compute_hash(
    obj: pathlib.Path | str | bytes,
    algorithm: str = "sha256",
    chunk_size: int = 8192,
    raise_on_error: bool = True,
) -> str:
    """Compute the hash of a file, directory, or bytes array using a specified hashing algorithm.

    This function handles both individual files and directories. For files, it directly hashes
    the content. For directories, it recursively processes all contained files and combines
    their hashes deterministically in a way that depends only on content and relative paths, not
    on metadata like timestamps or permissions.

    If a bytes array is passed in, it will be hashed directly.

    Args:
        obj: path to the file or directory to hash, or the file contents to hash as a bytes array.
        algorithm: hash algorithm to use (e.g., "md5", "sha1", "sha256").
        chunk_size: size of chunks to read when hashing large files.
        raise_on_error: if True, raises an exception if any file cannot be read.

    Returns:
        Hexadecimal digest of the hash.
    """
    if isinstance(obj, bytes):
        # assume we gave the file contents directly, just hash that immediately
        return hashlib.new(algorithm, obj).hexdigest()
    # otherwise, assume it's the path to a file/dir we want to hash
    path_obj = pathlib.Path(obj).expanduser().resolve()
    if path_obj.is_file():
        # file case - direct hash of contents
        h = hashlib.new(algorithm)
        with path_obj.open("rb") as file_obj:
            while True:
                chunk = file_obj.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    if path_obj.is_dir():
        # directory case - combine hashes of all files
        dir_hash = hashlib.new(algorithm)
        all_files: list[tuple[str, pathlib.Path]] = []
        for file_path in sorted(path_obj.glob("**/*")):
            if file_path.is_file():
                rel_path = file_path.relative_to(path_obj)
                all_files.append((str(rel_path), file_path))
        all_files.sort()  # sort by relative path for deterministic ordering
        for rel_path, file_path in all_files:
            path_hash = hashlib.new(algorithm, rel_path.encode()).hexdigest()
            file_hash = hashlib.new(algorithm)
            try:
                with file_path.open("rb") as file_obj:
                    while True:
                        chunk = file_obj.read(chunk_size)
                        if not chunk:
                            break
                        file_hash.update(chunk)
            except (OSError, PermissionError) as e:
                if raise_on_error:
                    # default behavior: we probably don't expect a dataset to contain bad files
                    raise e
                # otherwise, include error information in the hash if we can't read a file
                file_hash.update(f"ERROR: {str(e)}".encode())
            combined = f"{path_hash}:{file_hash.hexdigest()}".encode()
            dir_hash.update(combined)
        return dir_hash.hexdigest()
    raise ValueError(f"path does not exist: {path_obj}")


def set_seed(
    seed: int | None = None,
    workers: bool = False,
    verbose: bool = False,
) -> None:
    """Sets the seed for all potentially relevant RNGs through lightning.

    For more information, refer to:
    https://lightning.ai/docs/fabric/stable/api/utilities.html#lightning.fabric.utilities.seed.seed_everything
    """
    logger.info(f"setting seed to: {seed} (workers: {workers})")
    lightning.fabric.utilities.seed.seed_everything(
        seed=seed,
        workers=workers,
        verbose=verbose,
    )
    if seed is not None:
        transformers.set_seed(seed)


def get_reprod_metadata(
    include_installed_packages: bool = True,
    with_gpu_info: bool = False,
    with_distrib_info: bool = False,
    with_hydra_info: bool = True,
) -> dict[str, str]:
    """Returns a dictionary of metadata that can be used to assess reproducibility."""
    import pyine.utils.filesystem

    curr_time_since_epoch = time.time()
    reprod_metadata = {
        "python_version": get_python_version(),
        "created_by": pyine.utils.filesystem.get_username(),
        "platform": get_platform_name(),
        "framework_version": get_framework_version(),
        "git_revision_hash": get_git_revision_hash(),
        "git_repo_clean": str(is_git_repo_clean()),
        "project_root": str(pyine.utils.filesystem.get_project_root_path()),
        "data_root": str(pyine.utils.filesystem.get_data_root_path()),
        "logs_root": str(pyine.utils.filesystem.get_logs_root_path()),
        "tmp_dir": str(pyine.utils.filesystem.get_tmp_dir()),
        "work_dir": str(os.getcwd()),
        "dotenv_path": str(pyine.utils.filesystem.find_dotenv_file()),
        "time_since_epoch": str(curr_time_since_epoch),
        "local_timestamp": get_timestamp(curr_time_since_epoch),
        "runtime_hash": hashlib.sha1(str(curr_time_since_epoch).encode(), usedforsecurity=False).hexdigest(),
        "sys_executable": str(sys.executable),
        "sys_argv": str(sys.argv),
    }
    if include_installed_packages:
        reprod_metadata["installed_packages"] = "\n".join(get_installed_packages())
    if with_gpu_info:
        dev_count = torch.cuda.device_count()
        reprod_metadata["cuda"] = json.dumps(
            {
                "is_available": torch.cuda.is_available(),
                "arch_list": torch.cuda.get_arch_list(),
                "device_count": dev_count,
                "device_names": [torch.cuda.get_device_name(i) for i in range(dev_count)],
                "device_capabilities": [torch.cuda.get_device_capability(i) for i in range(dev_count)],
            }
        )
    if with_distrib_info:
        is_available = torch.distributed.is_available()
        is_initialized = is_available and torch.distributed.is_initialized()
        reprod_metadata["distrib"] = json.dumps(
            {
                "is_available": is_available,
                "is_initialized": is_initialized,
                "backend": distrib_utils.get_backend(),
                "rank": distrib_utils.get_global_rank(default=0),
                "world_size": distrib_utils.get_world_size(default=-1),
            }
        )
    return reprod_metadata


def entrypoint_setup(
    *,
    runtime_config: "pyine.configs.schemas.RuntimeConfig | None" = None,
    disable_http_logging_info_msgs: bool = True,
    use_wandb_logging: bool = False,
    wandb_init_kwargs: dict[str, typing.Any] | None = None,
    persist_runtime_artifacts: bool = True,
    **extra_configs: typing.Any,
) -> None:
    """Sets up the framework (env vars, logging, rng) for reproducible experiments.

    The `runtime_config` argument corresponds to the runtime provided via hydra entrypoints. If it
    is not provided, we do not have a run output folder, and will be forced to only log high-level
    information to the framework's global logs folder.

    Extra configs that are forwarded to this function will be logged in the runtime output
    directory (if a config is provided). If no config is provided, does nothing.

    Args:
        runtime_config: runtime configuration provided via hydra entrypoints.
        disable_http_logging_info_msgs: whether to disable the HTTP request POST messages in info
            level logs when using llm providers.
        use_wandb_logging: whether to initialize wandb logging (if using runtime) and not dry run.
        wandb_init_kwargs: optional wandb initialization kwargs (to e.g. resume an existing run).
            If `use_wandb_logging` is False, does nothing.
        persist_runtime_artifacts: whether to write configs/metadata artifacts to the runtime output
            directory. Set to False for non-primary distributed ranks that should avoid disk writes.
        extra_configs: extra configs that are forwarded to this function (to be logged).
    """
    import pyine.utils.filesystem

    load_dotenv()
    # use a sentinel object to track first execution of things that should only be executed once
    setup_fn = typing.cast("typing.Any", entrypoint_setup)
    if not hasattr(setup_fn, "_executed"):
        import pyine.prompts
        import pyine.utils.concurrency
        import pyine.utils.logging

        if runtime_config is None:
            # setup logging with default settings if no config is provided (otherwise hydra handles it)
            pyine.utils.logging.setup_logging(
                level=os.environ.get("LOGLEVEL", logging.INFO),
                log_to_file=True,
                log_path=None,  # use the framework's shared default log path by default
            )
        pyine.utils.logging.ensure_distributed_rank_filter_attached()
        logging.captureWarnings(True)
        warnings.simplefilter("default")
        if disable_http_logging_info_msgs:
            for pkg_name in ("httpx", "httpcore"):
                # fix for the 'noisy' HTTP request POST messages in info level logs when using llm providers
                pkg_logger = logging.getLogger(pkg_name)
                pkg_logger.setLevel(logging.WARNING)
                pkg_logger.propagate = False
        # initialize the prompt-related utilities
        _ = pyine.prompts.get_framework_prompt_manager()
        _ = pyine.prompts.get_framework_db()
        # enforce the spawn start method for multiprocessing functions/pools
        pyine.utils.concurrency.ensure_spawn_start_method()
        setup_fn._executed = True
        # for distributed experiments, help async error handling via nccl
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    # if a runtime config is provided, log all configs to the output directory
    parent_app_name = runtime_config.app_name if runtime_config else "<missing runtime config>"
    if runtime_config is not None:
        assert runtime_config.output_dir_path.is_dir(), "invalid runtime config output dir"
        set_seed(seed=runtime_config.seed, workers=runtime_config.seed_workers)
        app_config_dict: dict[str, typing.Any] = {}
        for config_name, config in extra_configs.items():
            assert isinstance(config, pydantic.BaseModel), "extra configs must be pydantic models"
            try:
                curr_config = config.model_dump(mode="json")
            except (pydantic_core.PydanticSerializationError, TypeError, ValueError):
                curr_config = portab_utils.make_json_serializable(config.model_dump(mode="python"))  # type: ignore[reportUnknownVariableType]
            assert isinstance(curr_config, dict), "config must be serializable to a dict"
            curr_config = typing.cast("dict[str, typing.Any]", curr_config)
            if persist_runtime_artifacts:
                print(f"{config_name} config:")
                rich.print_json(data=curr_config, indent=2)
                sys.stdout.flush()
            app_config_dict[config_name] = curr_config
        if runtime_config.dry_run:
            if persist_runtime_artifacts:
                # dry run; log configs right away (even if potential wandb init not complete, no need for it)
                log_configs(runtime_config, app_config_dict)
            logger.info(f"entrypoint dry run complete for app: {parent_app_name}")
            raise DryRunExit()
        if use_wandb_logging:
            # initialize wandb if enabled and not dry run (and with the app config as metadata)
            wandb_init_kwargs = dict(wandb_init_kwargs or {})
            wandb_init_kwargs.setdefault("config", app_config_dict)
            runtime_config.init_wandb(**wandb_init_kwargs)
            hydra_metadata = runtime_config.metadata.get("hydra")
            if hydra_metadata:
                hydra_dict = typing.cast("dict[str, str]", json.loads(hydra_metadata))
                runtime_config.wandb_run.summary.update(  # type: ignore[reportUnknownMemberType]
                    {f"hydra/{key}": value for key, value in hydra_dict.items()}
                )
        logged_config_file_paths: list[pathlib.Path] = []
        if persist_runtime_artifacts:
            # logging configs after wandb init means that we also log wandb run id w/ runtime stuff
            logged_config_file_paths = log_configs(runtime_config, app_config_dict)
        if use_wandb_logging and persist_runtime_artifacts:
            assert runtime_config.wandb_run_id is not None
            artifact_name = pyine.utils.filesystem.slugify(
                text=f"{runtime_config.app_name}-{runtime_config.wandb_run.name}-configs",
            )
            configs_artifact = wandb.Artifact(
                name=artifact_name,
                type="configs",
                metadata={"run_id": runtime_config.wandb_run_id},
            )
            for config_file_path in logged_config_file_paths:
                config_file_prefix = config_file_path.name.split(".")[0]
                configs_artifact.add_file(
                    local_path=str(config_file_path),
                    name=f"{config_file_prefix}{config_file_path.suffix}",
                    skip_cache=True,
                    policy="immutable",
                )
            runtime_config.wandb_run.log_artifact(configs_artifact)  # noqa
    else:
        if use_wandb_logging:
            raise ValueError("cannot initialize wandb logging without a runtime config?")
    logger.info(f"entrypoint setup complete for app: {parent_app_name}")


@functools.wraps(dotenv.load_dotenv)
def load_dotenv(**kwargs: typing.Any) -> bool:
    """Parses the closest `.env` file and load all the variables found as environment variables."""
    from pyine.utils.filesystem import find_dotenv_file

    dotenv_path = find_dotenv_file()
    if dotenv_path is None:
        raise FileNotFoundError(
            "could not find .env file containing environment variable overrides; "
            "see the `.env.template` file for an example of how to set up your environment"
        )
    return dotenv.load_dotenv(dotenv_path, **kwargs)


def get_log_extension_slug(
    runtime_config: "pyine.configs.schemas.RuntimeConfig | None",
    extension_suffix: str = ".log",
) -> str:
    """Returns a log file extension that includes a sortable and timezone-independent timestamp."""
    if runtime_config is not None and "time_since_epoch" in runtime_config.metadata:
        seconds_since_epoch = int(float(runtime_config.metadata["time_since_epoch"]))
    else:
        time_since_epoch = time.time()
        seconds_since_epoch = int(time_since_epoch)
    rank_id = distrib_utils.get_global_rank(default=0) or 0
    return f".{seconds_since_epoch}.rank{rank_id:02d}{extension_suffix}"


def log_configs(
    runtime_config: "pyine.configs.schemas.RuntimeConfig",
    app_config_dict: dict[str, typing.Any],  # should be already 'dumped' into json-serializable format
) -> list[pathlib.Path]:
    """Saves runtime, metadata, and app config to the hydra runtime output directory.

    Returns the list of paths to the saved files.
    """
    assert runtime_config is not None, "missing runtime config"
    log_extension = get_log_extension_slug(runtime_config, extension_suffix=".json")
    reprod_metadata = get_reprod_metadata(  # get a new metadata dict will ALL fields
        include_installed_packages=True,
        with_gpu_info=True,
        with_distrib_info=True,
    )
    reprod_metadata.update(runtime_config.metadata)  # override with previously-defined fields

    print("reprod metadata:")
    rich.print_json(data=reprod_metadata, indent=2)
    sys.stdout.flush()
    output_metadata_path = runtime_config.output_dir_path / f"reprod_metadata{log_extension}"
    output_metadata_path.write_text(json.dumps(reprod_metadata, indent=2))
    logger.info(f"reprod metadata saved to: {output_metadata_path}")

    print("runtime info:")
    rich.print_json(data=runtime_config.model_dump(mode="json"), indent=2)
    sys.stdout.flush()
    output_runtime_path = runtime_config.output_dir_path / f"runtime{log_extension}"
    output_runtime_path.write_text(runtime_config.model_dump_json(indent=2))
    logger.info(f"runtime info saved to: {output_runtime_path}")

    output_app_config_path = runtime_config.output_dir_path / f"config{log_extension}"
    output_app_config_path.write_text(json.dumps(app_config_dict, indent=2))
    logger.info(f"app config saved to: {output_app_config_path}")

    return [output_metadata_path, output_runtime_path, output_app_config_path]


def load_logged_app_config(
    experiment_log_dir: pathlib.Path | str,
) -> dict[str, typing.Any]:
    """Loads an app config JSON that was previously logged in an experiment's output directory.

    If the directory does not exist, an exception will be raised. However, if the file does not
    exist, the function will return an empty dict. If more than one config files exist, the first
    one in the sorted list of files will be loaded and returned.
    """
    experiment_log_dir = pathlib.Path(experiment_log_dir).expanduser().resolve()
    assert experiment_log_dir.is_dir(), f"invalid experiment log dir: {experiment_log_dir}"
    potential_config_paths = sorted(experiment_log_dir.glob("config.*.rank*.json"))
    if not potential_config_paths:
        return {}
    config_path = potential_config_paths[0]
    return json.loads(config_path.read_text())
