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


def register_hydra_configs(datamodule_config_store: hydra_zen.ZenStore) -> hydra_zen.ZenStore:
    """Registers datamodule-specific configs in the hydra store."""
    datamodule_config_store(
        pyine.organisms.datamodules.shortcuts.ShortcutBiasDataModuleConfig,
        lmdb_paths=hydra.conf.MISSING,  # must be specified by user
        default_dataparser_config=dict(),
        dataparser_config_overrides=dict(),
        dataloader_config_overrides=dict(),
        max_trace_count=None,
        split_file_path=hydra.conf.MISSING,  # must be specified by user
        base_filter_rule="",
        hydra_convert="object",
        name="default",
    )
    datamodule_config_store(
        pyine.organisms.datamodules.shortcuts.ShortcutBiasDataModuleConfig,
        lmdb_paths=[
            pyine.data.traces.dataset_utils.get_latest_dataset_path("TACO"),
        ],
        default_dataparser_config=dict(),
        dataparser_config_overrides=dict(),  # @@@@@@ TODO: use stored reasonable defaults for exps here
        dataloader_config_overrides=dict(),  # @@@@@@ TODO: use stored reasonable defaults for exps here
        max_trace_count=200,  # cap off the max dataset size (across each subset)
        split_file_path=pyine.data.utils.splits.get_dataset_split_file_path("TACO"),
        base_filter_rule="",
        hydra_convert="object",
        name="TACO_latest_200t",
    )
    datamodule_config_store(
        pyine.organisms.datamodules.shortcuts.ShortcutBiasDataModuleConfig,
        lmdb_paths=[
            pyine.data.traces.dataset_utils.get_latest_dataset_path("TACO"),
        ],
        default_dataparser_config=dict(),
        dataparser_config_overrides=dict(),  # @@@@@@ TODO: use stored reasonable defaults for exps here
        dataloader_config_overrides=dict(),  # @@@@@@ TODO: use stored reasonable defaults for exps here
        max_trace_count=None,  # no cap for subset sizes
        split_file_path=pyine.data.utils.splits.get_dataset_split_file_path("TACO"),
        base_filter_rule="",
        hydra_convert="object",
        name="TACO_latest_full",
    )
    return datamodule_config_store
