"""Shared helpers for trace annotation CLI applications."""

import asyncio
import functools
import glob
import json
import logging
import pathlib
import typing

import click
import yaml

import pyine.data.traces.dataset_reader
import pyine.data.traces.dataset_utils
import pyine.utils.llm_providers
import pyine.utils.portability

logger = logging.getLogger(__name__)


def load_yaml_or_json_file(
    path: pathlib.Path,
) -> typing.Any:
    """Loads YAML or JSON content from a file."""
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return yaml.safe_load(text)


def parse_yaml_or_json_value(
    value: str,
) -> typing.Any:
    """Parses a string as JSON, falling back to YAML, and then to raw string."""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return yaml.safe_load(value)
        except yaml.YAMLError:
            return value


def parse_kv_list_to_dict(
    items: tuple[str, ...],
) -> dict[str, typing.Any]:
    """Parses repeated KEY=VALUE strings to a dictionary, decoding JSON/YAML values when possible."""
    result: dict[str, typing.Any] = {}
    for item in items:
        if "=" not in item:
            raise click.BadParameter(f"expected KEY=VALUE, got '{item}'")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        result[key] = parse_yaml_or_json_value(raw_value)
    return result


def ensure_mapping_dict(
    value: typing.Any,
    error_message: str,
) -> dict[str, typing.Any]:
    """Parses and ensures that the provided value is a dictionary of kwargs."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise click.BadParameter(error_message)
    return typing.cast("dict[str, typing.Any]", value)


def build_dataset_reader(
    dataset_paths: list[pathlib.Path],
    dataset_loader_name: str | None,
) -> pyine.data.traces.dataset_reader.DatasetProtocol:
    """Instantiates a trace dataset reader for one or multiple dataset directories.

    If `dataset_loader_name` is provided, it must be a dotted path to either:
      - a class that can be instantiated with a dataset path (str/pathlib.Path),
      - a function that returns a reader when called as (dataset_path),
      - or a callable that accepts a list of dataset paths when multiple parts are provided.

    If not provided, attempt to instantiate:
      - `pyine.data.traces.dataset_reader.DatasetReader(path)` for single paths; or
      - `pyine.data.traces.dataset_reader.DatasetCollection([...])` when multiple paths are provided.
    """
    if not dataset_paths:
        raise click.BadParameter("no dataset paths were provided")
    if dataset_loader_name:
        try:
            loader = pyine.utils.portability.import_from_dotted_path(dataset_loader_name)
        except Exception as exc:
            raise click.ClickException(f"could not resolve dataset loader '{dataset_loader_name}': {exc}") from exc
        if not callable(loader):
            raise click.BadParameter(f"dotted path '{dataset_loader_name}' does not resolve to a callable")
        if len(dataset_paths) == 1:
            return typing.cast(
                "pyine.data.traces.dataset_reader.DatasetProtocol",
                loader(dataset_paths[0]),
            )
        try:
            return typing.cast(
                "pyine.data.traces.dataset_reader.DatasetProtocol",
                loader(dataset_paths),
            )
        except TypeError as exc:
            raise click.BadParameter(
                "dataset loader must accept a sequence of paths when --dataset is provided multiple times"
            ) from exc
    try:
        if len(dataset_paths) == 1:
            return typing.cast(
                "pyine.data.traces.dataset_reader.DatasetProtocol",
                pyine.data.traces.dataset_reader.DatasetReader(dataset_paths[0]),
            )
        return typing.cast(
            "pyine.data.traces.dataset_reader.DatasetProtocol",
            pyine.data.traces.dataset_reader.DatasetCollection(dataset_paths),
        )
    except Exception as exc:
        raise click.ClickException(f"Failed to instantiate dataset reader; original error: {exc}") from exc


def resolve_dataset_paths(
    dataset_paths: tuple[str, ...],
    dataset_latest_from: str | None,
) -> list[pathlib.Path]:
    """Resolves CLI dataset arguments into a list of validated directory paths.

    Exactly one of ``dataset_paths`` or ``dataset_latest_from`` must be provided.
    Glob patterns in ``dataset_paths`` are expanded automatically.
    """
    has_explicit_paths = len(dataset_paths) > 0
    has_latest = dataset_latest_from is not None
    if has_explicit_paths == has_latest:
        raise click.BadParameter("exactly one of --dataset or --dataset-latest-from must be provided")
    if dataset_latest_from is not None:
        try:
            resolved_path = pyine.data.traces.dataset_utils.get_latest_dataset_path(dataset_latest_from)
        except Exception as exc:
            raise click.ClickException(
                f"could not resolve latest trace dataset for source '{dataset_latest_from}': {exc}"
            ) from exc
        logger.info("resolved latest dataset for '%s' to: %s", dataset_latest_from, resolved_path)
        return [resolved_path]
    expanded_paths: list[pathlib.Path] = []
    for path_str in dataset_paths:
        if any(char in path_str for char in ["*", "?", "[", "]"]):
            matched_paths = glob.glob(path_str)
            if not matched_paths:
                raise click.BadParameter(f"glob pattern '{path_str}' did not match any paths")
            expanded_paths.extend(pathlib.Path(p) for p in sorted(matched_paths))
        else:
            path = pathlib.Path(path_str)
            if not path.exists():
                raise click.BadParameter(f"dataset path does not exist: {path}")
            if not path.is_dir():
                raise click.BadParameter(f"dataset path is not a directory: {path}")
            expanded_paths.append(path)
    return expanded_paths


def build_llm_provider_config(
    llm_kv: tuple[str, ...],
    llm_config_file: pathlib.Path | None,
) -> pyine.utils.llm_providers.LLMProviderConfig:
    """Builds an LLM provider config by merging a config file and CLI key-value overrides."""
    llm_kwargs: dict[str, typing.Any] = {}
    if llm_config_file is not None:
        file_cfg = ensure_mapping_dict(
            load_yaml_or_json_file(llm_config_file),
            "--llm-config-file must decode to a mapping/dict",
        )
        llm_kwargs.update(file_cfg)
    if llm_kv:
        llm_kwargs.update(parse_kv_list_to_dict(llm_kv))
        logger.debug(f"parsed LLM options: {llm_kwargs}")
    try:
        return pyine.utils.llm_providers.LLMProviderConfig.from_dict(llm_kwargs)
    except Exception as exc:
        raise click.BadParameter("llm options resulted in an invalid provider config") from exc


def async_main_wrapper(
    fn: typing.Callable[..., typing.Coroutine[typing.Any, typing.Any, None]],
) -> typing.Callable[..., None]:
    """Wraps an async function so that it can be used as a sync Click command entry point."""

    @functools.wraps(fn)
    def wrapper(
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> None:
        asyncio.run(fn(*args, **kwargs))

    return wrapper
