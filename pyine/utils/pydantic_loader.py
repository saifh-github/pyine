import importlib
import logging
import pathlib
import pkgutil
import typing

import pydantic
import yaml

logger = logging.getLogger(__name__)


class PydanticYAMLLoader(yaml.SafeLoader):
    """Custom YAML loader with automatic Pydantic model resolution.

    This loader extends `yaml.SafeLoader` to support custom YAML tags that automatically
    convert YAML structures to Pydantic model instances during loading.
    """

    # registry of available models for YAML tag resolution
    _model_registry: dict[str, type[pydantic.BaseModel]] = {}

    @classmethod
    def register_model(
        cls,
        tag_name: str,
        model_class: type[pydantic.BaseModel],
    ) -> None:
        """Register a Pydantic model for YAML loading.

        Args:
            tag_name: The tag name to use in YAML files (without the '!' prefix)
            model_class: The Pydantic model class to instantiate
        """
        if not issubclass(model_class, pydantic.BaseModel):
            raise ValueError(f"Model class must be a Pydantic BaseModel, got {model_class}")
        if tag_name in cls._model_registry:
            assert cls._model_registry[tag_name] is model_class, "registered model class mismatch"
            return  # nothing more to do, already registered
        cls._model_registry[tag_name] = model_class
        logger.debug(f"registered YAML tag '{tag_name}' for model {model_class.__name__}")

    @classmethod
    def register_models_from_module(
        cls,
        module_name: str,
        tag_prefix: str = "",
    ) -> None:
        """Auto-register all Pydantic models from a module.

        Args:
            module_name: Name of the module to scan for Pydantic models
            tag_prefix: Optional prefix to add to all tag names
        """
        try:
            module = importlib.import_module(module_name)
        except ImportError as error:
            logger.error(f"failed to import module '{module_name}': {error}")
            raise ValueError(f"failed to import module '{module_name}'") from error
        registered_count = 0
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, pydantic.BaseModel) and attr is not pydantic.BaseModel:
                # use fully qualified names (`module.Model`)
                qualified_name = f"{module_name}.{attr_name}"
                tag_name = f"{tag_prefix}{qualified_name}" if tag_prefix else qualified_name
                cls.register_model(tag_name, attr)
                registered_count += 1
        logger.debug(f"registered {registered_count} models from module '{module_name}'")

    @classmethod
    def register_models_from_package(
        cls,
        package_name: str,
        tag_prefix: str = "",
    ) -> None:
        """Auto-register all Pydantic models from all modules in a package.

        Args:
            package_name: Name of the package to scan recursively
            tag_prefix: Optional prefix to add to all tag names
        """
        try:
            package = importlib.import_module(package_name)
        except ImportError as error:
            raise ValueError(f"failed to import package '{package_name}'") from error
        total_registered = 0
        for _, module_name, _ in pkgutil.walk_packages(package.__path__, prefix=f"{package_name}."):
            try:
                module = importlib.import_module(module_name)
                registered_count = 0
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, pydantic.BaseModel)
                        and attr is not pydantic.BaseModel
                    ):
                        # use fully qualified names (`pkg.subpkg.module.Model`)
                        qualified_name = f"{module_name}.{attr_name}"
                        tag_name = f"{tag_prefix}{qualified_name}" if tag_prefix else qualified_name
                        cls.register_model(tag_name, attr)
                        registered_count += 1
                if registered_count > 0:
                    logger.debug(f"registered {registered_count} models from module '{module_name}'")
                total_registered += registered_count
            except ImportError:
                continue  # skip modules that can't be imported
        logger.info(f"registered {total_registered} models from package '{package_name}'")

    @classmethod
    def get_registered_models(cls) -> dict[str, type[pydantic.BaseModel]]:
        """Get a copy of all registered models."""
        return cls._model_registry.copy()

    @classmethod
    def clear_registry(cls) -> None:
        """Clear all registered models."""
        cls._model_registry.clear()
        logger.debug("cleared model registry")

    @classmethod
    def get_model_tag_by_class(cls, model_class: type[pydantic.BaseModel]) -> str | None:
        """Find the tag name for a registered model class.

        Args:
            model_class: The model class to find

        Returns:
            The tag name if found, otherwise None
        """
        for tag_name, registered_class in cls._model_registry.items():
            if registered_class is model_class:
                return tag_name
        return None


def _generic_pydantic_constructor(
    loader: PydanticYAMLLoader,
    tag_suffix: str,
    node: yaml.MappingNode,
) -> pydantic.BaseModel:
    """Generic constructor for any registered Pydantic model.

    Args:
        loader: The YAML loader instance
        tag_suffix: The tag suffix (model name) from the YAML tag
        node: The YAML mapping node to construct

    Returns:
        Validated Pydantic model instance

    Raises:
        ValueError: If the tag is not registered or validation fails
    """
    if tag_suffix not in loader._model_registry:
        available_tags = list(loader._model_registry.keys())
        raise ValueError(f"unknown Pydantic model tag: '{tag_suffix}'. " f"available tags: {available_tags}")
    model_class = loader._model_registry[tag_suffix]
    try:
        data = loader.construct_mapping(node, deep=True)  # will load data as a dictionary
    except Exception as error:
        raise ValueError(f"failed to construct YAML mapping: {error}") from error
    try:
        return model_class.model_validate(data)
    except pydantic.ValidationError as error:
        raise ValueError(f"failed to validate data for model '{model_class.__name__}': {error}") from error


# register the generic constructor for all tags starting with '!'
PydanticYAMLLoader.add_multi_constructor("!", _generic_pydantic_constructor)


def load_yaml_with_pydantic_support(
    file_path: pathlib.Path | str,
) -> typing.Any:
    """Load YAML file with direct Pydantic model parsing support.

    Args:
        file_path: Path to the YAML file to load.

    Returns:
        Parsed YAML data with Pydantic models instantiated.
    """
    file_path = pathlib.Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")
    try:
        with open(file_path, encoding="utf-8") as file:
            return yaml.load(file, Loader=PydanticYAMLLoader)
    except yaml.YAMLError as error:
        raise yaml.YAMLError(f"failed to parse YAML file '{file_path}': {error}") from error


def dump_yaml_with_pydantic_support(
    data: typing.Any,
    file_path: pathlib.Path | str | None = None,
) -> str:
    """Dump data to YAML with Pydantic model support.

    Args:
        data: Data to serialize to YAML.
        file_path: Optional file path to write to.

    Returns:
        After writing the YAML file, returns the YAML string representation that was written.
    """

    def _pydantic_representer(dumper: yaml.Dumper, data: pydantic.BaseModel) -> yaml.Node:
        # represent Pydantic models with fully qualified class name as tag
        model_class = data.__class__
        module_name = model_class.__module__
        class_name = model_class.__name__
        tag = f"!{module_name}.{class_name}"
        return dumper.represent_mapping(tag, data.model_dump())

    # register representer for all Pydantic models
    yaml.add_multi_representer(pydantic.BaseModel, _pydantic_representer)
    yaml_str = yaml.dump(data, default_flow_style=False, sort_keys=False)
    if file_path is not None:
        file_path = pathlib.Path(file_path)
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(yaml_str)
    return yaml_str
