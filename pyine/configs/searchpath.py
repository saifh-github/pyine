import os
import pathlib
import typing

import hydra.core.config_search_path
import hydra.plugins.search_path_plugin

import pyine.utils.filesystem

CONFIGS_ROOT_ENV_VAR = "PYINE_CONFIGS_ROOT"
"""Name of the environment variable used to override the default configs search path."""


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
        env_root = os.environ.get(CONFIGS_ROOT_ENV_VAR)
        if env_root:
            env_path = pathlib.Path(env_root).expanduser().resolve()
            yield "pyine_env", env_path
        # check cwd local tree (e.g. for repo in dev mode)
        yield "pyine_cwd", pathlib.Path.cwd() / "pyine" / "configs"
        # check using repo-root tree (installed in editable/dev, or running from a subdir)
        yield "pyine_repo", pyine.utils.filesystem.get_project_root_path() / "pyine" / "configs"
