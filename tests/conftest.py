import hydra.core.global_hydra
import pytest


@pytest.fixture(autouse=True)
def clear_hydra_config_store():
    """Clear the Hydra config store before and after each test."""
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    yield
    hydra.core.global_hydra.GlobalHydra.instance().clear()
