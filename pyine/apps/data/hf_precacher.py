"""CLI app to pre-generate datamodule caches (metadata, HF message datasets, tokenized examples).

This app should be run before training using the `hf_trainer.py` app to ensure that the datamodule
caches are available for faster, non-blocking startups.
"""

from __future__ import annotations

import asyncio
import logging
import typing

import hydra_zen
import pydantic
import transformers

import pyine.apps.data.utils
import pyine.apps.trainers.common
import pyine.apps.trainers.hf_trainer_configs
import pyine.configs.base
import pyine.configs.schemas
import pyine.configs.searchpath
import pyine.configs.utils
import pyine.data.datamodule
import pyine.evals.common
import pyine.utils.reprod
import pyine.utils.transformers

logger = logging.getLogger(__name__)


class PrecacherConfig(pydantic.BaseModel):
    """User-configurable options controlling cache pre-generation."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
    """Pydantic model configuration (freezes the dataclass)."""
    include_eval_subsets: bool = False
    """Whether to pre-build caches for evaluation subsets defined by the datamodule."""
    force_regenerate: bool = False
    """Whether to force regeneration of all caches."""
    max_seq_len_override: int | None = pydantic.Field(
        default=None,
        description=(
            "Optional override for max sequence length when building tokenized caches. "
            "If not provided, attempts to infer it from the model config without loading weights."
        ),
        ge=1,
    )
    epochs_override: int | None = pydantic.Field(
        default=None,
        description=(
            "Optional override for the number of training epochs to precache. "
            "If omitted, the epoch budget is inferred from training args and dataset size."
        ),
        ge=1,
    )


async def main(
    config: pyine.apps.trainers.hf_trainer_configs.HFTrainerAppMainConfig,
    runtime: pyine.configs.schemas.RuntimeConfig | None = None,
    precache_config: PrecacherConfig | None = None,
) -> None:
    """Computes metadata, prepares message datasets, and generates tokenized examples.

    By default, all data will be cached in some subdirectory of the path specified by the
    `pyine.utils.filesystem.get_data_cache_path` function.
    """
    precache_config = precache_config or PrecacherConfig()
    try:
        pyine.utils.reprod.entrypoint_setup(
            runtime_config=runtime,
            main_config=config,
            precache_config=precache_config,
        )
    except pyine.utils.reprod.DryRunExit:
        return
    logger.info("starting precaching run")
    datamodule = pyine.apps.trainers.common.prepare_datamodule(config, runtime)
    if not isinstance(datamodule, pyine.data.datamodule.ConversationDataModule):
        raise TypeError(
            f"precacher requires a ConversationDataModule; received {type(datamodule).__name__}",
        )
    try:
        train_epoch_budget = pyine.apps.data.utils.infer_precache_epoch_count(
            config=config,
            datamodule=datamodule,
            epochs_override=precache_config.epochs_override,
        )
        if precache_config.include_eval_subsets:
            subsets_to_precache = ("train", "valid", "eval")
        else:
            subsets_to_precache = ("train", "valid")
        logger.debug(f"will precache for the following subsets: {subsets_to_precache}")
        tokenizer = config.get_tokenizer()
        if precache_config.max_seq_len_override is not None:
            max_seq_len = precache_config.max_seq_len_override
            logger.info(f"specified max_seq_len={max_seq_len}")
        else:
            logger.info("determining max sequence length...")
            model_config = typing.cast(
                "transformers.PretrainedConfig",
                transformers.AutoConfig.from_pretrained(  # type: ignore[reportUnknownMemberType]
                    config.base_model,
                    **config.auto_model_config,
                ),
            )
            max_seq_len = pyine.utils.transformers.infer_effective_max_seq_len(model_config, tokenizer)
            logger.info(f"effective max_seq_len={max_seq_len}")
        for subset in subsets_to_precache:
            if subset == "train":
                for epoch_idx in range(train_epoch_budget):
                    logger.info(f"caching {subset} dataset for epoch {epoch_idx + 1}/{train_epoch_budget}")
                    pyine.apps.data.utils.prepare_subset_epoch(
                        datamodule=datamodule,
                        subset_names=getattr(config.datamodule_config, "train_subset_names", [subset]),
                        epoch=epoch_idx,
                    )
                    _ = datamodule.get_hf_tokenized_examples_dataset(
                        subset_name=subset,
                        tokenizer=tokenizer,
                        model_max_seq_len=max_seq_len,
                        force_regenerate=precache_config.force_regenerate,
                        epoch=epoch_idx,
                    )
            else:
                logger.info(f"caching {subset} dataset...")
                _ = datamodule.get_hf_tokenized_examples_dataset(
                    subset_name=subset,
                    tokenizer=tokenizer,
                    model_max_seq_len=max_seq_len,
                    force_regenerate=precache_config.force_regenerate,
                )
        logger.info("tokenized dataset caches built successfully")
    finally:
        datamodule.teardown()
        runtime.finalize()
        logger.info("precaching run completed")


def _async_main_wrapper(
    config: pyine.apps.trainers.hf_trainer_configs.HFTrainerAppMainConfig,
    runtime: pyine.configs.schemas.RuntimeConfig | None = None,
    precache_config: PrecacherConfig | None = None,
) -> None:
    asyncio.run(main(config=config, runtime=runtime, precache_config=precache_config))


def register_hydra_configs(
    eval_type: pyine.evals.common.EvalType,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Register Hydra configs required for the data precaching app, returning config descriptions.

    Note: strongly tied to (and inspired from) the HF trainer app config registration.
    """
    pyine.utils.reprod.load_dotenv()
    precache_config_builder = pyine.configs.utils.make_config_description(
        PrecacherConfig,
        name="default",
        group="precache_config",
        description="Default settings controlling dataset precaching behavior.",
        config={
            "populate_full_signature": True,
            "hydra_convert": "object",
        },
    )
    entrypoint_config = pyine.configs.utils.make_config_description(
        _async_main_wrapper,
        name="entrypoint",
        group=None,
        description="Entrypoint settings for the data precaching app.",
        config={
            "precache_config": precache_config_builder.config,
            # -------------
            "populate_full_signature": True,
            "hydra_defaults": [
                "_self_",
                {"config": "base"},  # from the hf trainer configs
                {"runtime": "default"},  # from pyine.configs.base
                *pyine.configs.base.get_base_hydra_default_overrides(),
            ],
        },
    )
    store, base_configs = pyine.configs.base.get_base_store_and_configs("hf_precacher")
    app_configs = pyine.apps.trainers.hf_trainer_configs._get_app_configs(  # type: ignore[reportPrivateUsage]
        eval_type=eval_type,
        group="config",
    )
    configs_to_register = [entrypoint_config, *app_configs, precache_config_builder]
    experiment_configs = pyine.apps.trainers.hf_trainer_configs._get_experiment_configs(  # type: ignore[reportPrivateUsage]
        eval_type=eval_type,
        entrypoint_config=entrypoint_config,
        app_configs=[*base_configs, *configs_to_register],
        group="experiment",
        package="_global_",
    )
    configs_to_register.extend(experiment_configs)
    external_configs = pyine.configs.searchpath.SearchPathPlugin.get_external_configs(
        app_name="hf_precacher",
        eval_type=eval_type,
        entrypoint_config=entrypoint_config,
        app_configs=[*base_configs, *configs_to_register],
    )
    configs_to_register.extend(external_configs)
    for config in configs_to_register:
        assert config.name is not None, "config names should have been set and validated by now"
        store(typing.cast("typing.Any", config.config), name=config.name, group=config.group, package=config.package)
    store.add_to_hydra_store(overwrite_ok=True)  # to avoid issues with name conflicts in tests
    return [*base_configs, *configs_to_register]


def hydra_main(eval_type: pyine.evals.common.EvalType) -> None:
    """Hydra main entrypoint for the HuggingFace data precaching app."""
    pyine.configs.base.register_searchpath_plugin()
    _ = register_hydra_configs(eval_type=eval_type)
    hydra_zen.zen(_async_main_wrapper).hydra_main(
        config_path=None,
        config_name="entrypoint",
        version_base=pyine.configs.base.target_hydra_version,
    )


if __name__ == "__main__":
    # TODO: if we ever have more than one eval type, add a selector based on launch args here
    hydra_main(pyine.evals.common.EvalType.CODE_EXEC)
