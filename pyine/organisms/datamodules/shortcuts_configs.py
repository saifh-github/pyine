"""Hydra-zen config builder for shortcuts data modules."""

import warnings

import hydra.conf
import hydra_zen
import hydra_zen.typing

import pyine.data.traces.dataset_utils
import pyine.data.utils.splits
import pyine.organisms.datamodules.shortcuts
import pyine.organisms.datamodules.utils.samples


def get_taco_configs(base_config: hydra_zen.typing.Builds) -> dict[str, hydra_zen.typing.Builds]:
    """Returns datamodule configs for the TACO dataset."""
    output_configs: dict[str, hydra_zen.typing.Builds] = dict()

    # first, get TACO dataset paths in a fail-safe manner
    taco_latest_path = None
    try:
        taco_latest_path = pyine.data.traces.dataset_utils.get_latest_dataset_path("TACO")
    except FileNotFoundError:
        pass
    taco_split_path = None
    try:
        taco_split_path = pyine.data.utils.splits.get_dataset_split_file_path("TACO")
    except FileNotFoundError:
        pass
    taco_10s10t_v1_paths = pyine.data.traces.dataset_utils.get_matching_dataset_paths(
        source_dataset_name="TACO",
        pattern="v1/10s10t.*of000026.*.lmdb",
    )

    # emit warnings for missing datasets (users should not be trying to launch experiments with these)
    if taco_latest_path is None:
        warnings.warn(
            "no TACO base dataset found, skipping TACO configs; "
            "if you intended to use TACO dataset demos/tests, please ensure that a dataset is present "
            "in the expected location (see the top-level README for more details)"
        )
    if taco_split_path is None:
        warnings.warn(
            "no TACO split file found, skipping TACO configs; "
            "if you intended to use TACO data modules, please ensure that the split file is present "
            "in the expected location (see the top-level README for more details)"
        )
    if not taco_10s10t_v1_paths:
        warnings.warn(
            "the TACO 10s10t v1 dataset is missing, skipping related configs; "
            "if you intended to conduct TACO-related experiments, please ensure that a dataset is present "
            "in the expected location (see the top-level README for more details)"
        )

    # ===========

    # @@@@@@ TODO: update bases w/ reasonable defaults for exps here?

    if taco_latest_path is not None and taco_split_path is not None:
        # if we have both of these paths, build the demo/testing configs
        output_configs["TACO_latest"] = (
            taco_latest_config := hydra_zen.builds(
                pyine.organisms.datamodules.shortcuts.ShortcutBiasDataModuleConfig,
                lmdb_paths=[taco_latest_path],
                split_file_path=taco_split_path,
                # -------------
                builds_bases=(base_config,),
            )
        )
        output_configs["TACO_latest_20s"] = hydra_zen.builds(
            pyine.organisms.datamodules.shortcuts.ShortcutBiasDataModuleConfig,
            max_solution_count=20,  # cap off the dataset size for quick experiments (across each subset)
            # -------------
            builds_bases=(taco_latest_config,),
        )

    # ===========

    if taco_10s10t_v1_paths and taco_split_path is not None:
        # if we have both of these paths, build TACO 10s10t configs for v1 experiments
        assert len(taco_10s10t_v1_paths) == 26, "unexpected number of v1 10s10t datasets"
        output_configs["TACO_10s10t_v1_full"] = (
            taco_10s10t_v1_config := hydra_zen.builds(
                pyine.organisms.datamodules.shortcuts.ShortcutBiasDataModuleConfig,
                lmdb_paths=taco_10s10t_v1_paths,
                split_file_path=taco_split_path,
                # -------------
                builds_bases=(base_config,),
            )
        )
        output_configs["TACO_10s10t_v1_part1"] = (
            taco_10s10t_v1_part1_config := hydra_zen.builds(
                pyine.organisms.datamodules.shortcuts.ShortcutBiasDataModuleConfig,
                lmdb_paths=[taco_10s10t_v1_paths[0]],  # smaller overall dataset for quick experiments
                # -------------
                builds_bases=(taco_10s10t_v1_config,),
            )
        )
        output_configs["TACO_10s10t_v1_part1_20s"] = hydra_zen.builds(
            pyine.organisms.datamodules.shortcuts.ShortcutBiasDataModuleConfig,
            lmdb_paths=[taco_10s10t_v1_paths[0]],
            split_file_path=taco_split_path,
            max_solution_count=20,  # cap off the dataset size for quick experiments (across each subset)
            # -------------
            builds_bases=(taco_10s10t_v1_part1_config,),
        )

    return output_configs


def store_hydra_configs(store: hydra_zen.ZenStore) -> list[str]:
    """Stores datamodule-specific configs in the provided store and returns stored config names."""

    base_config = hydra_zen.builds(
        pyine.organisms.datamodules.shortcuts.ShortcutBiasDataModuleConfig,
        lmdb_paths=hydra.conf.MISSING,  # must be specified by user
        split_file_path=hydra.conf.MISSING,  # must be specified by user
        split_seed="${runtime.seed}",
        # @@@@@@ TODO: put the configs below in their own store w/ named defaults/overrides?
        default_dataparser_config=dict(  # pyine.organisms.datamodules.utils.samples.SampleBuilderConfig
            params=dict(
                transform_config=dict(  # pyine.organisms.datamodules.utils.samples.SampleTransformConfig
                    seed="${runtime.seed}",
                    transform_strategy="never",
                ),
                selection_config=dict(  # pyine.organisms.datamodules.utils.samples.SampleSelectionConfig
                    seed="${runtime.seed}",
                    input_type_prob_map=dict(
                        original=1.0,
                    ),
                ),
            ),
        ),
        dataparser_config_overrides=dict(),
        dataloader_config_overrides=dict(
            train=dict(
                shuffle=True,
            ),
        ),
        # -------------
        populate_full_signature=True,
        hydra_convert="object",
    )
    store(base_config, name="base")

    dm_module_config_names: list[str] = []
    for config_name, config in get_taco_configs(base_config).items():
        store(config, name=config_name)
        dm_module_config_names.append(config_name)

    return dm_module_config_names
