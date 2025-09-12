import dataclasses
import typing

import hydra.conf
import pydantic
import pydantic.fields


def get_defaults(
    model: type[pydantic.BaseModel] | type[dataclasses.dataclass] | type[typing.TypedDict],
    **overrides,
) -> dict[str, typing.Any]:
    """Get the default values for all fields in a pydantic model, dataclass, or TypedDict.

    If a field does not have a default value, it will be assigned `hydra.conf.MISSING`.
    """
    if isinstance(model, type) and issubclass(model, pydantic.BaseModel):
        defaults = {
            name: field.default if field.default is not pydantic.fields.PydanticUndefined else hydra.conf.MISSING
            for name, field in model.__fields__.items()
        }
    elif isinstance(model, type) and hasattr(model, "__dataclass_fields__"):
        defaults = {
            name: field.default if field.default is not dataclasses.MISSING else hydra.conf.MISSING
            for name, field in model.__dataclass_fields__.items()
        }
    elif isinstance(model, type) and hasattr(model, "__annotations__"):  # TypedDict
        defaults = {name: hydra.conf.MISSING for name in model.__annotations__}
    else:
        raise ValueError(f"invalid model type: {type(model)}")
    defaults.update(overrides)
    return defaults
