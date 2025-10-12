from __future__ import annotations

import contextlib
import dataclasses
import tempfile
import typing

import hydra
import hydra.core
import hydra.core.config_store
import hydra.core.global_hydra
import hydra_zen
import omegaconf

import pyine.configs.base
import pyine.configs.schemas
import pyine.configs.utils


def _get_default_entries(
    config: typing.Any,
) -> list[typing.Any] | None:
    """Extracts the default list embedded in a hydra-zen config, if any."""
    fields = getattr(config, "__dataclass_fields__", None)
    if not fields or "defaults" not in fields:
        return None
    defaults_field = fields["defaults"]
    default_factory = getattr(defaults_field, "default_factory", None)
    if default_factory in (None, dataclasses.MISSING):
        return None
    return list(default_factory())


def _resolve_embedded_defaults(
    *,
    obj_cfg: omegaconf.DictConfig,
    base_configs: typing.Sequence[pyine.configs.schemas.ConfigDescription],
    parent_group: str | None,
    fallback_defaults: list[typing.Any] | None,
) -> None:
    """Resolves nested Hydra defaults within a composed config node."""

    def _find_config_description(
        group_key: str,
        config_name: str,
    ) -> pyine.configs.schemas.ConfigDescription | None:
        candidate_groups = []
        if parent_group:
            candidate_groups.append(f"{parent_group}/{group_key}")
        candidate_groups.append(group_key)
        for config in base_configs:
            if config.name != config_name:
                continue
            if config.group in candidate_groups:
                return config
            if config.group and any(config.group.endswith(candidate) for candidate in candidate_groups):
                return config
        return None

    if "defaults" in obj_cfg:
        defaults_raw = obj_cfg.get("defaults")
        if isinstance(defaults_raw, list):
            defaults_list = list(defaults_raw)
        else:
            defaults_list = typing.cast(
                "list[typing.Any]",
                omegaconf.OmegaConf.to_container(defaults_raw, resolve=False),
            )
    elif fallback_defaults is not None:
        defaults_list = list(fallback_defaults)
    else:
        defaults_list = []
    if not defaults_list:
        return
    omegaconf.OmegaConf.set_readonly(obj_cfg, False)
    omegaconf.OmegaConf.set_struct(obj_cfg, False)
    for default_item in defaults_list:
        if not isinstance(default_item, dict) or len(default_item) != 1:
            continue
        default_key, default_value = next(iter(default_item.items()))
        if not isinstance(default_key, str) or not isinstance(default_value, str):
            continue
        matched_config = _find_config_description(default_key, default_value)
        if matched_config is None:
            continue
        obj_cfg[default_key] = matched_config.config
    if "defaults" in obj_cfg:
        del obj_cfg["defaults"]


@dataclasses.dataclass
class _LaunchHarnessProviderConfig:
    value: int = 0


@dataclasses.dataclass
class _LaunchHarnessContainerConfig:
    provider: _LaunchHarnessProviderConfig | None = None


@contextlib.contextmanager
def instantiate_from_defaults_with_launch(
    *,
    base_configs: typing.Iterable[pyine.configs.schemas.ConfigDescription],
    target_config: pyine.configs.schemas.ConfigDescription,
    overrides: typing.Sequence[str] | None = None,
    convert: str = "all",
) -> typing.Generator[typing.Any]:
    """Instantiates a Hydra-configured object by launching a minimal (dummy) Hydra application.

    This function is intended to be used as a context manager to ensure that the Hydra
    configuration of an object is properly registered and can be used to instantiate that
    object in testing environments; do NOT use this function in real apps!

    Args:
        base_configs: Iterable of config descriptions to register in the store. These should
            include the configs referenced by the `defaults` list of the target config.
        target_config: The target config for the object you want to instantiate.
        overrides: Optional Hydra CLI-style overrides applied at compose-time.
        convert: Passed to hydra-zen instantiate via `_convert_` for predictable types in tests.
            Default "all" converts OmegaConf containers to native Python types.

    Yields:
        The instantiated object.
    """
    materialized_base_configs = list(base_configs)
    if target_config.group is None:
        raise ValueError("target_config.group must be defined for launch-based instantiation")

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    store = hydra_zen.ZenStore()

    def _register_config(config: pyine.configs.schemas.ConfigDescription) -> None:
        store(config.config, name=config.name, group=config.group, package=config.package)

    for config in materialized_base_configs:
        _register_config(config)
    _register_config(target_config)
    store.add_to_hydra_store(overwrite_ok=True)
    target_defaults = _get_default_entries(target_config.config)
    launcher_config = hydra_zen.make_config(
        defaults=[
            {target_config.group: target_config.name},
            "_self_",
        ]
    )

    def _task_function(cfg: omegaconf.DictConfig) -> typing.Any:
        omegaconf.OmegaConf.set_readonly(cfg, False)
        omegaconf.OmegaConf.set_struct(cfg, False)
        obj_cfg = getattr(cfg, target_config.group)
        _resolve_embedded_defaults(
            obj_cfg=obj_cfg,
            base_configs=materialized_base_configs,
            parent_group=target_config.group,
            fallback_defaults=target_defaults,
        )
        return hydra_zen.instantiate(obj_cfg, _convert_=convert)

    with tempfile.TemporaryDirectory(prefix="hydra-launch-") as run_dir:
        launch_overrides = list(overrides or [])
        launch_overrides.extend(
            [
                f"hydra.run.dir={run_dir}",
                "hydra.output_subdir=null",
                "hydra/job_logging=disabled",
                "hydra/hydra_logging=disabled",
            ]
        )
        try:
            job_return = hydra_zen.launch(
                config=launcher_config,
                task_function=_task_function,
                overrides=launch_overrides,
                version_base=pyine.configs.base.target_hydra_version,
                config_name="tests_launch_harness",
                job_name="tests_launch_harness",
                with_log_configuration=False,
            )
            yield job_return.return_value
        finally:
            hydra.core.global_hydra.GlobalHydra.instance().clear()


def test_launch_harness_instantiates_defaults() -> None:
    provider_config_desc = pyine.configs.utils.make_config_description(
        _LaunchHarnessProviderConfig,
        name="provider_default",
        group="tests_launch/provider",
        description="dummy provider config for launch harness tests",
        config={
            "value": 7,
            "populate_full_signature": True,
            "hydra_convert": "object",
        },
    )
    container_config_desc = pyine.configs.utils.make_config_description(
        _LaunchHarnessContainerConfig,
        name="container_base",
        group="tests_launch",
        description="container config that pulls in provider via defaults",
        config={
            "hydra_defaults": [
                {"provider": "provider_default"},
                "_self_",
            ],
            "populate_full_signature": True,
            "hydra_convert": "object",
        },
    )

    with instantiate_from_defaults_with_launch(
        base_configs=[provider_config_desc],
        target_config=container_config_desc,
    ) as container_instance:
        assert isinstance(container_instance, _LaunchHarnessContainerConfig)
        assert container_instance.provider is not None
        assert container_instance.provider.value == 7
