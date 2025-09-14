"""Hydra-zen config builder for shortcuts data modules."""

import warnings

import hydra.conf
import hydra_zen

import pyine.data.traces.dataset_utils
import pyine.data.utils.splits
import pyine.organisms.datamodules.shortcuts
import pyine.organisms.datamodules.utils.samples


def store_hydra_configs(store: hydra_zen.ZenStore) -> None:
    """Stores datamodule-specific configs in the provided hydra zen store."""

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

    # ---------------- fail-safe TACO dataset path lookups  ----------------

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

    # ---------------- simple TACO-latest configs for demos and tests  ----------------

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
    if taco_latest_path is not None and taco_split_path is not None:
        taco_latest_config = hydra_zen.builds(
            pyine.organisms.datamodules.shortcuts.ShortcutBiasDataModuleConfig,
            lmdb_paths=[taco_latest_path],
            split_file_path=taco_split_path,
            # -------------
            builds_bases=(base_config,),
        )
        store(taco_latest_config, name="TACO_latest")
        taco_latest_20s_config = hydra_zen.builds(
            pyine.organisms.datamodules.shortcuts.ShortcutBiasDataModuleConfig,
            max_solution_count=20,  # cap off the dataset size for quick experiments (across each subset)
            # -------------
            builds_bases=(taco_latest_config,),
        )
        store(taco_latest_20s_config, name="TACO_latest_20s")

    # ---------------- TACO 10s10t configs for v1 experiments ----------------

    if not taco_10s10t_v1_paths:
        warnings.warn(
            "the TACO 10s10t v1 dataset is missing, skipping related configs; "
            "if you intended to conduct TACO-related experiments, please ensure that a dataset is present "
            "in the expected location (see the top-level README for more details)"
        )
    if taco_10s10t_v1_paths and taco_split_path is not None:
        assert len(taco_10s10t_v1_paths) == 26, "unexpected number of v1 10s10t datasets"
        # @@@@@@ TODO: update w/ better base (DRY) and reasonable defaults for exps here?
        taco_10s10t_v1_full = hydra_zen.builds(
            pyine.organisms.datamodules.shortcuts.ShortcutBiasDataModuleConfig,
            lmdb_paths=taco_10s10t_v1_paths,
            split_file_path=taco_split_path,
            # -------------
            builds_bases=(base_config,),
        )
        store(taco_10s10t_v1_full, name="TACO_10s10t_v1_full")
        taco_10s10t_v1_part1 = hydra_zen.builds(
            pyine.organisms.datamodules.shortcuts.ShortcutBiasDataModuleConfig,
            lmdb_paths=[taco_10s10t_v1_paths[0]],  # smaller overall dataset for quick experiments
            split_file_path=taco_split_path,
            # -------------
            builds_bases=(base_config,),
        )
        store(taco_10s10t_v1_part1, name="TACO_10s10t_v1_part1")
        taco_10s10t_v1_part1_20s = hydra_zen.builds(
            pyine.organisms.datamodules.shortcuts.ShortcutBiasDataModuleConfig,
            lmdb_paths=[taco_10s10t_v1_paths[0]],
            split_file_path=taco_split_path,
            max_solution_count=20,  # cap off the dataset size for quick experiments (across each subset)
            # -------------
            builds_bases=(base_config,),
        )
        store(taco_10s10t_v1_part1_20s, name="TACO_10s10t_v1_part1_20s")
