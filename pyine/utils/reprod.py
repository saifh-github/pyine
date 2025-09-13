import functools
import hashlib
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

import dotenv
import lightning.fabric.utilities.seed
import torch
import yaml

if typing.TYPE_CHECKING:
    import pyine.configs.schemas


logger = logging.getLogger(__name__)


def get_python_version() -> str:
    """Return current Python version, e.g. '3.12.6'."""
    return platform.python_version()


def get_platform_name() -> str:
    """Returns a print-friendly platform name that can be used for logs / data tagging."""
    return str(platform.node())


def get_timestamp(time_since_epoch: float | None = None) -> str:
    """Returns a print-friendly timestamp (year, month, day, hour, minute, second) for logs."""
    if time_since_epoch is None:
        time_since_epoch = time.time()
    local_time = time.localtime(time_since_epoch)
    return time.strftime("%Y%m%d_%H%M%S", local_time)


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
    try:
        import importlib.metadata

        pkgs = [f"{pkg.name}=={pkg.version}" for pkg in importlib.metadata.distributions()]
    except (ImportError, AttributeError):
        try:
            import pip  # noqa

            # noinspection PyUnresolvedReferences
            pkgs = [f"{pkg.key}=={pkg.version}" for pkg in pip.get_installed_distributions()]
        except (ImportError, AttributeError):
            pkgs = []
    return sorted(pkgs, key=str.casefold)  # noqa


def get_params_hash(*args, **kwargs) -> str:
    """Computes and returns the hash (md5 checksum) of a given set of parameters.

    Args:
        Any combination of parameters that are hashable via their string representation.

    Returns:
        The hashing result as a string of hexadecimal digits.
    """
    # by default, will use the repr of all params but remove the 'at 0x00000000' addresses
    clean_str = re.sub(r" at 0x[a-fA-F\d]+", "", str(args) + str(kwargs))
    return hashlib.sha1(clean_str.encode(), usedforsecurity=False).hexdigest()


def compute_hash(
    obj: pathlib.Path | typing.AnyStr | bytes,
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
        with path_obj.open("rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()
    elif path_obj.is_dir():
        # directory case - combine hashes of all files
        dir_hash = hashlib.new(algorithm)
        all_files = []
        for file_path in sorted(path_obj.glob("**/*")):
            if file_path.is_file():
                rel_path = file_path.relative_to(path_obj)
                all_files.append((str(rel_path), file_path))
        all_files.sort()  # sort by relative path for deterministic ordering
        for rel_path, file_path in all_files:
            path_hash = hashlib.new(algorithm, rel_path.encode()).hexdigest()
            file_hash = hashlib.new(algorithm)
            try:
                with file_path.open("rb") as f:
                    for chunk in iter(lambda: f.read(chunk_size), b""):
                        file_hash.update(chunk)
            except (OSError, PermissionError) as e:
                if raise_on_error:
                    # default behavior: we probably don't expect a dataset to contain bad files
                    raise e
                else:
                    # otherwise, include error information in the hash if we can't read a file
                    file_hash.update(f"ERROR: {str(e)}".encode())
            combined = f"{path_hash}:{file_hash.hexdigest()}".encode()
            dir_hash.update(combined)
        return dir_hash.hexdigest()
    else:
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
    lightning.fabric.utilities.seed.seed_everything(
        seed=seed,
        workers=workers,
        verbose=verbose,
    )


def get_reprod_metadata(
    include_installed_packages: bool = True,
    with_gpu_info: bool = False,
    with_distrib_info: bool = False,
) -> dict[str, str]:
    """Returns a dictionary of metadata that can be used to assess reproducibility."""
    import pyine.utils.filesystem

    curr_time_since_epoch = time.time()
    reprod_metadata = dict(
        python_version=get_python_version(),
        created_by=pyine.utils.filesystem.get_username(),
        platform=get_platform_name(),
        framework_version=get_framework_version(),
        git_revision_hash=get_git_revision_hash(),
        git_repo_clean=str(is_git_repo_clean()),
        project_root=str(pyine.utils.filesystem.get_project_root_path()),
        data_root=str(pyine.utils.filesystem.get_data_root_path()),
        logs_root=str(pyine.utils.filesystem.get_logs_root_path()),
        tmp_dir=str(pyine.utils.filesystem.get_tmp_dir()),
        work_dir=str(os.getcwd()),
        dotenv_path=str(pyine.utils.filesystem.find_dotenv_file()),
        time_since_epoch=str(curr_time_since_epoch),
        local_timestamp=get_timestamp(curr_time_since_epoch),
        runtime_hash=hashlib.sha1(str(curr_time_since_epoch).encode(), usedforsecurity=False).hexdigest(),
        sys_argv=str(sys.argv),
    )
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
        reprod_metadata["distrib"] = json.dumps(
            {
                "is_available": torch.distributed.is_available(),
                "is_initialized": torch.distributed.is_initialized(),
                "backend": get_failsafe_backend(),
                "rank": get_failsafe_rank(),
                "world_size": get_failsafe_worldsize(),
            }
        )
    return reprod_metadata


def entrypoint_setup(
    config: "pyine.configs.schemas.RuntimeConfig | None" = None,  # None unless launched via hydra
    disable_http_logging_info_msgs: bool = True,
) -> None:
    """Sets up the framework (env vars, logging, rng) for reproducible experiments."""
    if config is not None:
        if config.seed is not None:
            set_seed(seed=config.seed, workers=config.seed_workers)
        log_reprod_metadata(config)
    # use a sentinel object to track first execution for the rest
    if not hasattr(entrypoint_setup, "_executed"):
        import pyine.prompts
        import pyine.utils.logging

        load_dotenv()
        if config is None:
            # setup logging with default settings if no config is provided (otherwise hydra handles it)
            pyine.utils.logging.setup_logging(
                level=os.environ.get("LOGLEVEL", logging.INFO).upper(),
                log_to_file=True,
                log_path=None,  # use the framework's shared default log path by default
            )
        if disable_http_logging_info_msgs:
            for pkg_name in ("httpx", "httpcore"):
                # fix for the 'noisy' HTTP request POST messages in info level logs when using llm providers
                pkg_logger = logging.getLogger(pkg_name)
                pkg_logger.setLevel(logging.WARNING)
                pkg_logger.propagate = False
        # initialize the prompt-related utilities
        _ = pyine.prompts.get_framework_prompt_manager()
        _ = pyine.prompts.get_framework_db()
        entrypoint_setup._executed = True
    parent_app_name = config.app_name if config else "<missing runtime config>"
    logger.info(f"entrypoint setup complete for app: {parent_app_name}")


@functools.wraps(dotenv.load_dotenv)
def load_dotenv(**kwargs) -> bool:
    """Parses the closest `.env` file and load all the variables found as environment variables."""
    from pyine.utils.filesystem import find_dotenv_file

    dotenv_path = find_dotenv_file()
    if dotenv_path is None:
        raise FileNotFoundError(
            "could not find .env file containing environment variable overrides; "
            "see the `.env.template` file for an example of how to set up your environment"
        )
    return dotenv.load_dotenv(dotenv_path, **kwargs)


def get_failsafe_rank(group: torch.distributed.ProcessGroup | None = None) -> int:
    """Returns the result of torch.distributed.get_rank, or zero if not in a process group."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank(group)
    return 0


def get_failsafe_worldsize(group: torch.distributed.ProcessGroup | None = None) -> int:
    """Returns the result of torch.distributed.get_world_size, or -1 if not in a process group."""
    if torch.distributed.is_initialized():
        torch.distributed.get_world_size(group)
    return -1


def get_failsafe_backend(group: torch.distributed.ProcessGroup | None = None) -> str:
    """Returns the result of torch.distributed.get_backend, or "n/a" if not in a process group."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_backend(group)
    return "n/a"


def get_log_extension_slug(
    config: "pyine.configs.schemas.RuntimeConfig | None",
    extension_suffix: str = ".log",
) -> str:
    """Returns a log file extension that includes a sortable and timezone-independent timestamp."""
    if config is not None and "time_since_epoch" in config.metadata:
        seconds_since_epoch = int(float(config.metadata["time_since_epoch"]))
    else:
        time_since_epoch = time.time()
        seconds_since_epoch = int(time_since_epoch)
    rank_id = get_failsafe_rank()
    return f".{seconds_since_epoch}.rank{rank_id:02d}{extension_suffix}"


def log_reprod_metadata(
    config: "pyine.configs.schemas.RuntimeConfig",
) -> pathlib.Path:
    """Saves a list of all runtime tags to a log file and returns the path to the saved file."""
    assert config is not None, "missing runtime config"
    output_dir = pathlib.Path(config.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_extension = get_log_extension_slug(config, extension_suffix=".yaml")
    output_log_path = output_dir / f"reprod_metadata{log_extension}"
    reprod_metadata = get_reprod_metadata(  # get a new metadata dict will ALL fields
        include_installed_packages=True,
        with_gpu_info=True,
        with_distrib_info=True,
    )
    reprod_metadata.update(config.metadata)  # override with previously-defined fields
    with open(output_log_path, "w") as fd:
        yaml.dump(reprod_metadata, fd, sort_keys=False)
    return output_log_path
