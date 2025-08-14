import importlib
import inspect
import logging
import pathlib
import pkgutil
import typing

import pydantic
import yaml

import pyine.utils.portability

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


BaseT = typing.TypeVar("BaseT")
"""Base type for classes that can be resolved and instantiated from pydantic configs."""


class ClassImportSpec(
    pydantic.BaseModel,
    typing.Generic[BaseT],
):
    """Generic configuration to import a class by path and instantiate it."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
    """Pydantic model configuration (frozen, no extras)."""

    class_path: typing.Annotated[
        pydantic.StrictStr,
        pydantic.Field(
            min_length=1,
            description="Dotted import path to the target class, e.g. 'pkg.mod.MyImpl'.",
        ),
    ]
    base_class_path: typing.Annotated[
        pydantic.StrictStr,
        pydantic.Field(
            min_length=1,
            description=("Dotted import path to the required base class of the target class."),
        ),
    ]
    params: dict[str, typing.Any] = pydantic.Field(
        default_factory=dict, description="Keyword arguments passed to the target class constructor."
    )

    def instantiate(self) -> BaseT:
        """Instantiates the resolved class with the parameters held inside the config."""
        assert self._resolved_class is not None, "model must be validated before use"
        return self._resolved_class(**self.params)

    # ----------------- below is private stuff that does not affect serialization -----------------

    # cache resolved types so we don't re-resolve them in instantiate()
    _resolved_class: type | None = pydantic.PrivateAttr(default=None)
    _resolved_base: type | None = pydantic.PrivateAttr(default=None)

    @property
    def _generic_base_class_type(self) -> type | None:
        """Returns the generic base class type (if this is a generic class)."""
        meta = getattr(self.__class__, "__pydantic_generic_metadata__", None)
        if meta and meta.get("args"):
            return meta["args"][0]
        return None

    @pydantic.model_validator(mode="after")
    def _validate_and_resolve(self) -> "ClassImportSpec":
        """Validates and resolves the class and base class paths."""
        resolved_class = pyine.utils.portability.import_from_dotted_path(self.class_path)
        if not isinstance(resolved_class, type) or not callable(resolved_class):
            raise TypeError(f'"{self.class_path}" resolved to {resolved_class!r}, which is not a class')
        resolved_base = pyine.utils.portability.import_from_dotted_path(self.base_class_path)
        if not isinstance(resolved_base, type):
            raise TypeError(f'"{self.base_class_path}" resolved to {resolved_base!r}, which is not a class')
        expected_base = self._generic_base_class_type
        if expected_base:  # will work only when we have a concrete runtime type
            assert isinstance(expected_base, type), "expected base class must be a type"
            if resolved_base is not expected_base and not issubclass(resolved_base, expected_base):
                raise TypeError(
                    f"resolved base {resolved_base.__name__} is not compatible with expected {expected_base.__name__}"
                )
        if not issubclass(resolved_class, resolved_base):
            raise TypeError(f"{resolved_class.__name__} is not a subclass of {resolved_base.__name__}")
        self._validate_params_against_constructor(resolved_class, self.params)
        self._resolved_class = resolved_class
        self._resolved_base = resolved_base
        return self

    @staticmethod
    def _validate_params_against_constructor(cls: type, params: dict[str, typing.Any]) -> None:
        """Validates that the provided params match the constructor signature of the class."""
        sig = inspect.signature(cls)
        parameters = list(sig.parameters.values())
        # fail fast if there are required positional-only params (cannot pass via kwargs)
        pos_only_required = [
            p
            for p in parameters
            if p.kind is inspect.Parameter.POSITIONAL_ONLY and p.default is inspect.Parameter.empty
        ]
        if pos_only_required:
            names = ", ".join(p.name for p in pos_only_required)
            raise TypeError(
                f"{cls.__name__}.__init__ has required positional-only parameters ({names}); "
                f"cannot instantiate with keyword-only params"
            )
        accepts_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters)
        acceptable_names = {p.name for p in parameters if p.kind is not inspect.Parameter.POSITIONAL_ONLY}
        # unknown params (only an error if the constructor doesn't accept **kwargs)
        if not accepts_kwargs:
            unknown = set(params) - acceptable_names
            if unknown:
                raise TypeError(
                    f"unexpected parameter(s) for {cls.__name__}: {sorted(unknown)}; "
                    f"accepted: {sorted(acceptable_names)}"
                )
        # ensure required keywordable parameters are present
        required_missing = [
            p.name
            for p in parameters
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            and p.default is inspect.Parameter.empty
            and p.name not in params
        ]
        if required_missing:
            raise TypeError(f"missing required parameter(s) for {cls.__name__}: {sorted(required_missing)}")
        # optional: try binding (catches some edge-cases like duplicate/ambiguous)
        try:
            # binding with provided kwargs (ignores extra if **kwargs present)
            to_bind = {k: v for k, v in params.items() if k in acceptable_names or accepts_kwargs}
            sig.bind_partial(**to_bind)
        except TypeError as exc:
            raise TypeError(f"invalid parameters for {cls.__name__}: {exc}") from exc
