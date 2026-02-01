import os
import typing

import hydra.core.global_hydra
import hydra.core.plugins
import hydra.core.singleton
import pytest

import pyine.utils.concurrency
import pyine.utils.filesystem

# ensure spawn method is set before any tests run (usually set in entrypoint)
pyine.utils.concurrency.ensure_spawn_start_method()

# capture env var state at module import time (before any tests modify them)
# this serves as the "original" state that should be restored after each test
_ENV_VARS_TO_RESTORE = [
    pyine.utils.filesystem.DATA_ROOT_ENV_VAR,
    pyine.utils.filesystem.CACHE_ROOT_ENV_VAR,
    pyine.utils.filesystem.LOGS_ROOT_ENV_VAR,
]
_ORIGINAL_ENV_STATE: dict[str, str | None] = {var: os.environ.get(var) for var in _ENV_VARS_TO_RESTORE}


def _reset_hydra_singletons() -> None:
    """Reset Hydra singletons to prevent state leakage between tests.

    This clears both GlobalHydra and removes the Plugins singleton from the
    Singleton registry, ensuring that any plugins registered during a test
    (potentially with monkeypatched environment variables) don't affect
    subsequent tests.
    """
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    # remove singletons that cache plugin registrations and sources
    # so they get recreated fresh with correct environment variables next time
    from hydra._internal.sources_registry import SourcesRegistry

    for singleton_class in [hydra.core.plugins.Plugins, SourcesRegistry]:
        if singleton_class in hydra.core.singleton.Singleton._instances:
            del hydra.core.singleton.Singleton._instances[singleton_class]


def _restore_env_vars() -> None:
    """Restore environment variables to their original state from before tests ran.

    This is a safety net to ensure env vars set by monkeypatch.setenv are cleaned up
    even if monkeypatch's automatic cleanup fails (e.g., due to fixture ordering issues).
    """
    for var, original_value in _ORIGINAL_ENV_STATE.items():
        current_value = os.environ.get(var)
        if current_value != original_value:
            if original_value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = original_value


@pytest.fixture(autouse=True)
def clear_hydra_config_store() -> typing.Iterator[None]:
    """Clear Hydra config store and plugins before and after each test.

    This ensures that tests that register Hydra plugins or configs don't pollute
    other tests with cached paths or configurations that were resolved using
    monkeypatched environment variables.
    """
    _reset_hydra_singletons()
    yield
    _reset_hydra_singletons()
    _restore_env_vars()
