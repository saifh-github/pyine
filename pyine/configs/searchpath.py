import importlib.util
import logging
import pathlib
import typing

import hydra.core.config_search_path
import hydra.plugins.search_path_plugin

import pyine.configs.schemas
import pyine.evals.common
import pyine.utils.filesystem

logger = logging.getLogger(__name__)


class SearchPathPlugin(hydra.plugins.search_path_plugin.SearchPathPlugin):
    """Hydra search path plugin that registers local user experiment configs.

    The default expected layout is that YAML config files defining settings for new experiments
    should be stored in:

        <repo_root>/pyine/configs/experiment/...

    For example, we could have a number of base experiment configs at the top level, and some
    more specific ones in subdirectories, e.g.:

        <repo_root>/pyine/configs/experiment/some_base_settings.yaml
        <repo_root>/pyine/configs/experiment/another_base.yaml
        <repo_root>/pyine/configs/experiment/big_category/derived_settings_A.yaml
        <repo_root>/pyine/configs/experiment/big_category/derived_settings_B.yaml
        ...

    Subdirectories under 'experiment' become nested groups automatically; the way to specify any
    config is simply:

        python -m pyine.apps.trainer.some_trainer +experiment=some_base_settings
        python -m pyine.apps.trainer.some_trainer +experiment=another_base
        python -m pyine.apps.trainer.some_trainer +experiment=big_category/derived_settings_A
        ...

    Note: we add the parent folder 'configs' to Hydra's search path, so the Hydra group name
    is 'experiment' (matching the folder name).
    """

    @typing.override
    def manipulate_search_path(
        self,
        search_path: hydra.core.config_search_path.ConfigSearchPath,
    ) -> None:
        """Adds the local user experiment configs to the search path."""
        for label, path in self._paths_to_try():
            if path.exists() and path.is_dir():
                # if we give a root config tree, subfolders (e.g., 'experiment') become groups
                search_path.append(label, f"file://{path}")

    @staticmethod
    def _paths_to_try() -> typing.Iterable[tuple[str, pathlib.Path]]:
        """Generates candidate search path locations in priority order (env, cwd, repo root).

        Yields:
            Tuples of (label, path) pointing to directories that contain the config tree.
        """
        # check for explicit env override (for e.g. CI/CD or forked/custom setups)
        env_path = pyine.utils.filesystem.get_configs_root_path()
        if env_path:
            yield "pyine_env", env_path
        # check cwd local tree (e.g. for repo in dev mode)
        yield "pyine_cwd", pathlib.Path.cwd() / "pyine" / "configs"
        # check using repo-root tree (installed in editable/dev, or running from a subdir)
        yield "pyine_repo", pyine.utils.filesystem.get_project_root_path() / "pyine" / "configs"

    @classmethod
    def get_external_configs(
        cls,
        app_name: str,
        eval_type: pyine.evals.common.EvalType,
        entrypoint_config: pyine.configs.schemas.ConfigDescription,
        app_configs: list[pyine.configs.schemas.ConfigDescription],
    ) -> list[pyine.configs.schemas.ConfigDescription]:
        """Generates and returns external configs for hydra zen storage (for all search paths).

        This function will look for python config files with a `_configs.py` suffix in each of
        the search paths, check if they possess a `register_hydra_configs` with the expected
        function signature, and use them to generate external configs for hydra zen storage.

        Args:
            app_name: The name of the app for which to generate external configs.
            eval_type: The type of evaluation for which to generate external configs.
            entrypoint_config: The entrypoint config for the app.
            app_configs: The list of all configs that have already been registered for the app.

        Returns:
            The list of newly generated app configs from external modules.
        """
        processed_files: set[pathlib.Path] = set()
        module_counter = 0
        output_configs: list[pyine.configs.schemas.ConfigDescription] = []
        # iterate all candidate roots (env -> cwd -> repo)
        for label, root in cls._paths_to_try():
            if not (root.exists() and root.is_dir()):
                continue
            # recursively find Python files named "*_configs.py"
            for py_file in root.rglob("*_configs.py"):
                py_path = py_file.resolve()
                if py_path in processed_files:
                    continue  # already seen (skip)
                processed_files.add(py_path)
                try:
                    module_counter += 1
                    mod_name = f"_pyine_ext_cfg_{label}_{module_counter}_{py_path.stem}"
                    spec = importlib.util.spec_from_file_location(mod_name, py_path)
                    if spec is None or spec.loader is None:
                        logger.warning(f"could not create import spec for external config file: {py_path}")
                        continue
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                except Exception as e:
                    logger.warning(f"failed to import external configs from '{py_path}': {e}")
                    continue
                # fetch the expected registration function
                register_fn = getattr(module, "register_hydra_configs", None)
                if register_fn is None:
                    continue  # silently skip files without the expected callable
                if not isinstance(register_fn, RegisterHydraConfigsFuncType):
                    raise TypeError(f"'register_hydra_configs' function in '{py_path}' is not callable")
                new_configs = register_fn(app_name, eval_type, entrypoint_config, app_configs)
                assert isinstance(new_configs, list), (
                    f"unexpected return from 'register_hydra_configs' function in '{py_path}': {type(new_configs)}"
                )
                valid_configs: list[pyine.configs.schemas.ConfigDescription] = []
                for cfg in new_configs:
                    assert isinstance(cfg, pyine.configs.schemas.ConfigDescription), (
                        f"invalid config from '{py_path}': expected ConfigDescription, got {type(cfg)}"
                    )
                    valid_configs.append(cfg)
                if valid_configs:
                    output_configs.extend(valid_configs)
                    logger.info(f"registered {len(valid_configs)} external configs from '{py_path}'")
        return output_configs


@typing.runtime_checkable
class RegisterHydraConfigsFuncType(typing.Protocol):
    """Protocol used to represent a callback used to register external configs for hydra."""

    def __call__(
        self,
        app_name: str,  # name of the app that we are looking to register configs for
        eval_type: pyine.evals.common.EvalType,  # eval type (task definition) for the configs to register
        entrypoint_config: pyine.configs.schemas.ConfigDescription,  # config for the app's entrypoint
        app_configs: list[pyine.configs.schemas.ConfigDescription],  # all registered configs for the app
    ) -> list[pyine.configs.schemas.ConfigDescription]:  # should return new app configs to register
        ...
