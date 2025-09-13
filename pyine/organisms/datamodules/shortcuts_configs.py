"""Hydra-zen config builder for shortcuts data modules."""

import typing

import hydra.conf
import hydra_zen
import hydra_zen.typing

import pyine.apps.trainers.openai_finetune
import pyine.configs.base
import pyine.data.traces.dataset_utils
import pyine.data.utils.splits
import pyine.organisms.datamodules.shortcuts
import pyine.organisms.models.utils.openai
import pyine.utils.llm_providers
import pyine.utils.reprod


def store_hydra_configs(store: hydra_zen.ZenStore) -> None:
    """Stores datamodule-specific configs in the provided hydra zen store."""
    base_config = hydra_zen.builds(
        pyine.organisms.datamodules.shortcuts.ShortcutBiasDataModuleConfig,
        lmdb_paths=hydra.conf.MISSING,  # must be specified by user
        split_file_path=hydra.conf.MISSING,  # must be specified by user
        # @@@@@@ TODO: update w/ reasonable defaults for exps here?
        # default_dataparser_config=dict(),
        # dataparser_config_overrides=dict(),
        # dataloader_config_overrides=dict(),
        # -------------
        populate_full_signature=True,
        hydra_convert="object",
    )
    store(
        hydra_zen.make_config(
            lmdb_paths=[
                pyine.data.traces.dataset_utils.get_latest_dataset_path("TACO"),
            ],
            max_trace_count=200,  # cap off the max dataset size (across each subset)
            split_file_path=pyine.data.utils.splits.get_dataset_split_file_path("TACO"),
            bases=(base_config,),
        ),
        name="TACO_latest_200t",
    )
    store(
        hydra_zen.make_config(
            lmdb_paths=[
                pyine.data.traces.dataset_utils.get_latest_dataset_path("TACO"),
            ],
            max_trace_count=None,  # no cap for subset sizes
            split_file_path=pyine.data.utils.splits.get_dataset_split_file_path("TACO"),
            bases=(base_config,),
        ),
        name="TACO_latest_full",
    )
