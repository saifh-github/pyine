import typing

import hydra.core.global_hydra
import pytest

import pyine.utils.concurrency

# ensure spawn method is set before any tests run (usually set in entrypoint)
pyine.utils.concurrency.ensure_spawn_start_method()


@pytest.fixture(autouse=True)
def clear_hydra_config_store() -> typing.Iterator[None]:
    """Clear the Hydra config store before and after each test."""
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    yield
    hydra.core.global_hydra.GlobalHydra.instance().clear()
