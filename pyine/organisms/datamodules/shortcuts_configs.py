"""Hydra-zen config builder for shortcuts data modules."""

import warnings

import hydra.conf
import hydra_zen
import hydra_zen.typing

import pyine.configs.schemas
import pyine.data.traces.dataset_utils
import pyine.data.utils.splits
import pyine.organisms.datamodules.shortcuts
import pyine.organisms.datamodules.utils.samples


def _get_taco_configs(
    datamodule_base_config: pyine.configs.schemas.ConfigDescription,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Returns datamodule configs and their descriptions for the TACO dataset."""
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
        pattern="v1.3/10s10t.*of000026.*.lmdb",
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

    # build the actual config objects + descriptions
    outputs: list[pyine.configs.schemas.ConfigDescription] = []

    if taco_latest_path is not None and taco_split_path is not None:
        # if we have both of these paths, build the demo/testing configs
        outputs.append(
            pyine.configs.schemas.ConfigDescription(
                name="TACO_latest",
                group=datamodule_base_config.group,
                config=(
                    taco_latest_config := hydra_zen.builds(
                        pyine.organisms.datamodules.shortcuts.ShortcutBiasDataModuleConfig,
                        lmdb_paths=[taco_latest_path],
                        split_file_path=taco_split_path,
                        # -------------
                        builds_bases=(datamodule_base_config.config,),
                        zen_meta={
                            "__description__": (
                                "Specifies the single most recent instance of a TACO trace dataset found on disk. "
                                "May contain an arbitrary number of traces with any kind of augmentations."
                            ),
                        },
                    )
                ),
            )
        )
        outputs.append(
            pyine.configs.schemas.ConfigDescription(
                name="TACO_latest_20s",
                group=datamodule_base_config.group,
                config=hydra_zen.builds(
                    pyine.organisms.datamodules.shortcuts.ShortcutBiasDataModuleConfig,
                    max_solution_count=20,  # cap off the dataset size for quick experiments (across each subset)
                    # -------------
                    builds_bases=(taco_latest_config,),
                    zen_meta={
                        "__description__": (
                            "Specifies a subset of the most recent TACO trace dataset on disk, with a maximum of 20 "
                            "solutions per trace. Useful for quick experiments, testing, and demos."
                        ),
                    },
                ),
            )
        )

    # ===========

    if taco_10s10t_v1_paths and taco_split_path is not None:
        # if we have both of these paths, build TACO 10s10t configs for v1 experiments
        assert len(taco_10s10t_v1_paths) == 26, "unexpected number of v1 10s10t datasets"
        # @@@@@@ TODO: update bases w/ reasonable defaults for exps here?
        outputs.append(
            pyine.configs.schemas.ConfigDescription(
                name="TACO_10s10t_v1_full",
                group=datamodule_base_config.group,
                config=(
                    taco_10s10t_v1_config := hydra_zen.builds(
                        pyine.organisms.datamodules.shortcuts.ShortcutBiasDataModuleConfig,
                        lmdb_paths=taco_10s10t_v1_paths,
                        split_file_path=taco_split_path,
                        # -------------
                        builds_bases=(datamodule_base_config.config,),
                        zen_meta={
                            "__description__": (
                                "Specifies the full 26 instances of the PyINE-TACO 10s10t v1 trace dataset. "
                                "Used for full-sized experiments on the entire dataset proposed in our first "
                                "paper. See @@@@@@@ TODO URL for more information."
                            ),
                        },
                    )
                ),
            )
        )
        outputs.append(
            pyine.configs.schemas.ConfigDescription(
                name="TACO_10s10t_v1_part1",
                group=datamodule_base_config.group,
                config=(
                    taco_10s10t_v1_part1_config := hydra_zen.builds(
                        pyine.organisms.datamodules.shortcuts.ShortcutBiasDataModuleConfig,
                        lmdb_paths=[taco_10s10t_v1_paths[0]],
                        # -------------
                        builds_bases=(taco_10s10t_v1_config,),
                        zen_meta={
                            "__description__": (
                                "Specifies a subset consisting of the first part (of 26, so about 3.8%) of the "
                                "PyINE-TACO 10s10t v1 trace dataset. This is a much smaller subset than the full "
                                "dataset, but should be fairly representative of the full dataset's distribution, "
                                "and therefore useful for smaller-scale experiments."
                            ),
                        },
                    )
                ),
            )
        )
        outputs.append(
            pyine.configs.schemas.ConfigDescription(
                name="TACO_10s10t_v1_part1_20s",
                group=datamodule_base_config.group,
                config=hydra_zen.builds(
                    pyine.organisms.datamodules.shortcuts.ShortcutBiasDataModuleConfig,
                    lmdb_paths=[taco_10s10t_v1_paths[0]],
                    split_file_path=taco_split_path,
                    max_solution_count=20,  # cap off the dataset size for quick experiments (across each subset)
                    # -------------
                    builds_bases=(taco_10s10t_v1_part1_config,),
                    zen_meta={
                        "__description__": (
                            "Specifies a subset of the first part of the PyINE-TACO 10s10t v1 trace dataset, "
                            "with a maximum of 20 solutions per trace. Useful for quick experiments, testing, "
                            "and demos. Should not be used for anything serious."
                        ),
                    },
                ),
            )
        )

    return outputs


def get_configs(
    group: str,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Generates and returns shortcuts-datamodule-specific configs for hydra zen storage."""
    shortcuts_dm_base_config = pyine.configs.schemas.ConfigDescription(
        name="base",
        group=group,
        config=hydra_zen.builds(
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
            zen_meta={
                "__description__": "Base shortcuts datamodule settings; not specific to any actual source dataset.",
            },
        ),
    )
    return [
        shortcuts_dm_base_config,
        *_get_taco_configs(shortcuts_dm_base_config),
        # add config getters for more source datasets here, if needed
    ]
