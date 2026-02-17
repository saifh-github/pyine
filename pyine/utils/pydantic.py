import collections.abc
import copy
import dataclasses
import importlib
import inspect
import logging
import pathlib
import pkgutil
import sys
import typing

import omegaconf
import pydantic
import pydantic.json
import pydantic.json_schema
import pydantic_core
import yaml

import pyine.utils.portability

logger = logging.getLogger(__name__)


def _dictconfig_plain_serializer(
    value: typing.Any,
) -> typing.Any:
    """Return a JSON-serializable representation for DictConfig instances.

    Args:
        value: Object given to Pydantic during serialization.

    Returns:
        A primitive container when the input is a DictConfig, otherwise the original value.
    """
    if isinstance(value, omegaconf.DictConfig):
        container = omegaconf.OmegaConf.to_container(value, resolve=True)
        return typing.cast("dict[str, typing.Any]", container)
    return value


def _dictconfig_core_schema(
    cls: type[omegaconf.DictConfig],
    source: typing.Any,
    handler: pydantic.GetCoreSchemaHandler,
) -> pydantic_core.core_schema.CoreSchema:
    """Expose DictConfig handling for Pydantic validation and serialization.

    Pydantic invokes this hook while building schemas for any models referencing DictConfig.

    Args:
        cls: DictConfig type provided by Pydantic.
        source: Original source annotation.
        handler: Callback to fetch the default schema for standard dicts.

    Returns:
        A CoreSchema accepting both DictConfig and plain dict inputs while serializing to primitives.
    """
    dict_schema = handler(dict[str, typing.Any])
    python_schema = pydantic_core.core_schema.union_schema(
        [
            pydantic_core.core_schema.is_instance_schema(omegaconf.DictConfig),
            dict_schema,
        ],
    )
    return pydantic_core.core_schema.json_or_python_schema(
        json_schema=dict_schema,
        python_schema=python_schema,
        serialization=pydantic_core.core_schema.plain_serializer_function_ser_schema(
            _dictconfig_plain_serializer,
            when_used="always",
        ),
    )


def _dictconfig_json_schema(
    cls: type[omegaconf.DictConfig],
    _core_schema: pydantic_core.core_schema.CoreSchema,
    handler: pydantic.GetJsonSchemaHandler,
) -> pydantic.json_schema.JsonSchemaValue:
    """Provide the JSON schema representation for DictConfig fields.

    Args:
        cls: DictConfig type provided by Pydantic.
        _core_schema: Core schema produced by `_dictconfig_core_schema`.
        handler: Callback used to derive the JSON schema.

    Returns:
        JSON schema describing a standard dictionary.
    """
    return handler(pydantic_core.core_schema.dict_schema())


def _install_dictconfig_serialization_support() -> None:
    """Register OmegaConf DictConfig support across all Pydantic models."""
    omegaconf.DictConfig.__get_pydantic_core_schema__ = classmethod(_dictconfig_core_schema)  # type: ignore[reportAttributeAccessIssue]
    omegaconf.DictConfig.__get_pydantic_json_schema__ = classmethod(_dictconfig_json_schema)  # type: ignore[reportAttributeAccessIssue]
    if hasattr(pydantic.json, "ENCODERS_BY_TYPE"):
        pydantic.json.ENCODERS_BY_TYPE[omegaconf.DictConfig] = _dictconfig_plain_serializer
        pydantic.json.ENCODERS_BY_TYPE[omegaconf.dictconfig.DictConfig] = _dictconfig_plain_serializer


_install_dictconfig_serialization_support()


class PydanticYAMLLoader(yaml.SafeLoader):
    """Custom YAML loader with automatic Pydantic model resolution.

    This loader extends `yaml.SafeLoader` to support custom YAML tags that automatically
    convert YAML structures to Pydantic model instances during loading.
    """

    # registry of available models for YAML tag resolution
    _model_registry: dict[str, type[pydantic.BaseModel]] = {}  # noqa: RUF012 (meant as mutable default)
    # cache of modules that were already scanned for pydantic models
    _scanned_modules: set[str] = set()  # noqa: RUF012 (meant as mutable default)

    @classmethod
    def register_model(
        cls,
        tag_name: str,
        model_class: type[pydantic.BaseModel] | type,
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
        force_rescan: bool = False,
    ) -> None:
        """Auto-register all Pydantic models from all modules in a package.

        Args:
            package_name: Name of the package to scan recursively
            tag_prefix: Optional prefix to add to all tag names
            force_rescan: When True, scan modules even if they were scanned before
        """
        try:
            package = importlib.import_module(package_name)
        except ImportError as error:
            raise ValueError(f"failed to import package '{package_name}'") from error
        total_new_registered = 0
        for _, module_name, _ in pkgutil.walk_packages(package.__path__, prefix=f"{package_name}."):
            # skip modules we've already scanned unless forced
            if not force_rescan and module_name in cls._scanned_modules:
                continue
            try:
                module = importlib.import_module(module_name)
                new_registered_in_module = 0
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
                        # only count if this is a newly registered model
                        if tag_name not in cls._model_registry:
                            cls.register_model(tag_name, attr)
                            new_registered_in_module += 1
                if new_registered_in_module > 0:
                    logger.debug(f"registered {new_registered_in_module} models from module '{module_name}'")
                total_new_registered += new_registered_in_module
                # mark module as scanned regardless of whether we found new models
                cls._scanned_modules.add(module_name)
            except ImportError:
                continue  # skip modules that can't be imported
        logger.info(f"registered {total_new_registered} new model(s) from package '{package_name}'")

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
    registry = type(loader).get_registered_models()
    if tag_suffix not in registry:
        available_tags = list(registry.keys())
        raise ValueError(f"unknown Pydantic model tag: '{tag_suffix}'. available tags: {available_tags}")
    model_class = registry[tag_suffix]
    try:
        data = loader.construct_mapping(node, deep=True)  # will load data as a dictionary
    except Exception as error:
        raise ValueError(f"failed to construct YAML mapping: {error}") from error
    try:
        return model_class.model_validate(data)
    except pydantic.ValidationError as error:
        raise ValueError(f"failed to validate data for model '{model_class.__name__}': {error}") from error


# register the generic constructor for all tags starting with '!'
typing.cast("typing.Any", PydanticYAMLLoader).add_multi_constructor("!", _generic_pydantic_constructor)


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
            return yaml.load(file, Loader=PydanticYAMLLoader)  # noqa: S506 (custom loader required)
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


def merge_configs(
    *configs: typing.Any,
) -> dict[str, typing.Any]:
    """Merges multiple config dictionaries hierarchically, ordered as base to override(s)."""

    def _to_primitives(obj: typing.Any) -> typing.Any:
        if isinstance(obj, pydantic.BaseModel):
            return _to_primitives(obj.model_dump())
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return _to_primitives(dataclasses.asdict(obj))
        if isinstance(obj, collections.abc.Mapping):
            return {k: _to_primitives(v) for k, v in obj.items()}  # type: ignore[reportUnknownVariableType]
        if isinstance(obj, (set, frozenset)):
            return [_to_primitives(v) for v in obj]  # type: ignore[reportUnknownVariableType]
        if isinstance(obj, collections.abc.Sequence) and not isinstance(obj, (str, bytes, bytearray)):  # type: ignore[reportUnknownVariableType]
            return [_to_primitives(v) for v in obj]  # type: ignore[reportUnknownVariableType]
        return obj

    assert len(configs) > 0, "at least one config must be provided"
    output_config = omegaconf.OmegaConf.create(_to_primitives(copy.deepcopy(configs[0])))
    for overrides in configs[1:]:
        overrides = omegaconf.OmegaConf.create(_to_primitives(overrides))
        output_config = omegaconf.OmegaConf.merge(output_config, overrides)
    container = omegaconf.OmegaConf.to_container(output_config, resolve=False)
    return typing.cast("dict[str, typing.Any]", container)


class ClassImportSpec(pydantic.BaseModel):
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
            description="Dotted import path to the required base class of the target class.",
        ),
    ]
    params: dict[str, typing.Any] | pydantic.SerializeAsAny[pydantic.BaseModel] = pydantic.Field(
        default_factory=lambda: typing.cast("dict[str, typing.Any]", {}),
        description="Keyword arguments passed to the target class constructor.",
    )
    params_key: typing.Annotated[
        str | None,
        pydantic.Field(
            description=(
                "Key to use for passing params as a single argument when instantiating objects. "
                "If not specified, all params will be unpacked and forwarded directly."
            ),
        ),
    ] = None

    def instantiate(
        self,
        *args: typing.Any,
        **extra_kwargs: typing.Any,
    ) -> typing.Any:
        """Instantiates the resolved class with the parameters held inside the config."""
        assert self._resolved_class is not None, "model must be validated before use"
        constr_params = self.get_params_dict()
        constr_params.update(extra_kwargs)
        if self.params_key is None:
            return self._resolved_class(*args, **constr_params)
        return self._resolved_class(*args, **{self.params_key: constr_params})

    def get_params_dict(self) -> dict[str, typing.Any]:
        """Returns the parameters held inside the config as a dictionary for obj instantiation."""
        return self.params.model_dump() if isinstance(self.params, pydantic.BaseModel) else self.params.copy()

    def get_non_params_dict(self) -> dict[str, typing.Any]:
        """Returns the other non-params fields inside the config as a dictionary."""
        return {k: v for k, v in self.model_dump().items() if k != "params"}

    def get_updated_spec(
        self,
        **extra_params: typing.Any,
    ) -> typing.Self:
        """Returns a new spec with the given extra params kwargs merged in.

        NOTE: the merge is done using OmegaConf to hierarchically merge the two dicts.
        """
        merged_params = merge_configs(self.get_params_dict(), extra_params)
        return type(self)(**self.get_non_params_dict(), params=merged_params)

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
        if resolved_class is not resolved_base and not issubclass(resolved_class, resolved_base):
            raise TypeError(f"{resolved_class.__name__} is not a subclass of {resolved_base.__name__}")
        self._validate_params_against_constructor(resolved_class)
        self._resolved_class = resolved_class
        self._resolved_base = resolved_base
        return self

    def _validate_params_against_constructor(self, cls: type) -> None:
        """Validates that the provided params match the constructor signature of the class."""
        params = self.get_params_dict() if self.params_key is None else {self.params_key: self.get_params_dict()}
        # check that all provided params exist in the target constructor
        sig = inspect.signature(cls)
        invalid_params = set(params) - set(sig.parameters)
        if invalid_params:
            raise ValueError(f"invalid parameter(s) for {cls.__name__}: {invalid_params}")
        # check if provided params can be bound to the constructor
        try:
            sig.bind_partial(**params)
        except TypeError as e:
            raise ValueError(f"cannot bind parameter(s) to {cls.__name__} constructor") from e


def is_jsonvalue(obj: typing.Any, *, strict: bool = False) -> bool:
    """Returns True if the given object is a pydantic json value."""
    try:
        pydantic.TypeAdapter(pydantic.JsonValue).validate_python(obj, strict=strict)
        return True
    except pydantic.ValidationError:
        return False


def model_from_callable(
    fn: typing.Any,
    name: str | None = None,
    *,
    include: set[str] | None = None,
    exclude: set[str] | None = None,
    include_kwargs: bool = True,
    base: type[pydantic.BaseModel] = pydantic.BaseModel,
    model_config: pydantic.ConfigDict | dict[str, typing.Any] | None = None,
    default_overrides: dict[str, typing.Any] | None = None,
    type_overrides: dict[str, typing.Any] | None = None,
    resolve_annotations: bool = True,
    module_for_created_model: str | None = None,
) -> type[pydantic.BaseModel]:
    """Creates a Pydantic model whose fields mirror the parameters of a specified function (`fn`).

    Supports default factories when parameter defaults (or overrides) are provided as
    `pydantic.Field` or `dataclasses.field` instances.

    Args:
        fn: The function whose parameters should be used as the model fields.
        name: Optional name for the model. If not specified, the name will be generated from `fn`.
        include: Optional set of parameter names to include. If not specified, all parameters will
            be included.
        exclude: Optional set of parameter names to exclude. If not specified, no parameters
            will be excluded.
        include_kwargs: Whether to include keyword arguments in the model. If False, keyword
            arguments will be excluded.
        base: Optional base class for the model. Defaults to `pydantic.BaseModel`.
        model_config: Optional model configuration dictionary or object.
        default_overrides: Optional dictionary of arguments with default values to override.
        type_overrides: Optional dictionary of argument annotations to override.
        resolve_annotations: Whether to resolve annotations for the model fields. If False,
            will use the raw annotations from the function signature directly.
        module_for_created_model: Optional module name to use for the model class. If not set,
            defaults to the caller’s module.
    Returns:
        The generated Pydantic model class.
    """
    sig = inspect.signature(fn)
    globalns = vars(sys.modules[getattr(fn, "__module__", "__name__")])
    hints: dict[str, typing.Any] = {}
    if resolve_annotations:
        try:
            hints = typing.get_type_hints(fn, globalns=globalns, include_extras=True)
        except NameError:
            # fallback to raw signature annotations, and typing.Any for strings/forward refs
            resolve_annotations = False
    if not resolve_annotations:
        anns = inspect.get_annotations(fn, eval_str=False) or {}
        hints = {k: (typing.Any if isinstance(v, str) else v) for k, v in anns.items()}
    field_info_type = getattr(pydantic.fields, "FieldInfo", None)
    default_overrides = default_overrides or {}
    type_overrides = type_overrides or {}
    dataclass_fields_by_name: dict[str, dataclasses.Field[typing.Any]] = {}
    if inspect.isclass(fn) and dataclasses.is_dataclass(fn):
        dataclass_fields_by_name = {field.name: field for field in dataclasses.fields(fn)}
    dataclasses_has_default_factory = getattr(dataclasses, "_HAS_DEFAULT_FACTORY_CLASS", None)

    def _normalize_default(
        param_name: str,
        raw_default: typing.Any,
    ) -> typing.Any:
        if field_info_type is not None and isinstance(raw_default, field_info_type):
            return raw_default
        if isinstance(raw_default, dataclasses.Field):
            dataclass_field_info = typing.cast("dataclasses.Field[typing.Any]", raw_default)
            default_factory_value = getattr(dataclass_field_info, "default_factory", dataclasses.MISSING)
            if default_factory_value is not dataclasses.MISSING:
                factory = typing.cast("typing.Callable[[], typing.Any]", default_factory_value)
                return pydantic.Field(default_factory=factory)
            default_value = getattr(dataclass_field_info, "default", dataclasses.MISSING)
            if default_value is not dataclasses.MISSING:
                return default_value
            return ...
        if (
            dataclasses_has_default_factory is not None
            and isinstance(raw_default, dataclasses_has_default_factory)
            and param_name in dataclass_fields_by_name
        ):
            dataclass_field = dataclass_fields_by_name[param_name]
            default_factory_value = getattr(dataclass_field, "default_factory", dataclasses.MISSING)
            if default_factory_value is not dataclasses.MISSING:
                factory = typing.cast("typing.Callable[[], typing.Any]", default_factory_value)
                return pydantic.Field(default_factory=factory)
            default_value = getattr(dataclass_field, "default", dataclasses.MISSING)
            if default_value is not dataclasses.MISSING:
                return default_value
            return ...
        return raw_default

    def _extra_behavior() -> typing.Any:
        if model_config is not None:
            assert isinstance(model_config, dict)
            extra_value = model_config.get("extra")
            if extra_value is not None:
                return extra_value
        base_config = getattr(base, "model_config", None)
        if isinstance(base_config, dict):
            return base_config.get("extra")  # type: ignore[reportUnknownMemberType]
        return getattr(base_config, "extra", None)

    field_definitions: dict[str, tuple[typing.Any, typing.Any]] = {}
    for p in sig.parameters.values():
        if include and p.name not in include:
            continue
        if exclude and p.name in exclude:
            continue
        if p.kind is inspect.Parameter.VAR_POSITIONAL:
            # *args -> tuple[Any, ...]
            anno = tuple[typing.Any, ...]
            default = ()
        elif p.kind is inspect.Parameter.VAR_KEYWORD:
            # **kwargs -> dict[str, Any]
            if not include_kwargs:
                continue
            anno = dict[str, typing.Any]
            default = pydantic.Field(default_factory=dict)
        else:
            anno = type_overrides.get(p.name, hints.get(p.name, typing.Any))
            raw_default = ... if p.default is inspect.Signature.empty else default_overrides.get(p.name, p.default)
            default = _normalize_default(p.name, raw_default)
        field_definitions[p.name] = (anno, default)
    override_order: list[str] = []
    for override_map in (default_overrides, type_overrides):
        for key in override_map:
            if key not in override_order:
                override_order.append(key)
    missing_override_fields = [key for key in override_order if key not in field_definitions]
    if missing_override_fields:
        extra_behavior = _extra_behavior()
        if extra_behavior == "allow":
            for key in missing_override_fields:
                anno = type_overrides.get(key, typing.Any)
                default = _normalize_default(key, default_overrides.get(key, ...))
                field_definitions[key] = (anno, default)
        else:
            missing_fields_csv = ", ".join(sorted(missing_override_fields))
            raise ValueError(f"Overrides provided for unknown parameters: {missing_fields_csv}")
    callable_name = getattr(fn, "__name__", type(fn).__name__)
    model_name = name or f"{callable_name}ParamsConfig"
    if module_for_created_model is None:
        # default to the caller’s module if not provided
        frame = inspect.currentframe()
        try:
            caller_globals = frame.f_back.f_globals if frame and frame.f_back else globals()
        finally:
            del frame
        module_for_created_model = str(caller_globals.get("__name__", __name__))
    module_name = module_for_created_model or __name__
    create_model_fn = typing.cast(
        "typing.Callable[..., type[pydantic.BaseModel]]",
        pydantic.create_model,
    )
    return create_model_fn(
        model_name,
        __config__=typing.cast("typing.Any", model_config),
        __base__=base,
        __module__=module_name,
        **typing.cast("dict[str, typing.Any]", field_definitions),
    )


def get_field_default(
    model_cls: type[pydantic.BaseModel],
    field_name: str,
    *,
    call_default_factory: bool = False,
) -> typing.Any:
    """Return the declared default for a field without instantiating the model.

    If the field has a default_factory, set call_default_factory=True to compute it.
    Raises KeyError if the field does not exist.
    """
    field = model_cls.model_fields[field_name]
    if getattr(field, "default_factory", None) is not None:
        if call_default_factory:
            factory: typing.Callable[[], typing.Any] = field.default_factory  # type: ignore[assignment]
            return factory()
        return field.default_factory  # return the factory itself
    return field.default
