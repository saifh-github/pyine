"""Utility functions for data-related CLI apps (e.g. precaching)."""

from __future__ import annotations

import logging
import math
import typing

if typing.TYPE_CHECKING:
    import pyine.apps.trainers.common
    import pyine.apps.trainers.hf_sft_trainer_configs
    import pyine.data.datamodule

logger = logging.getLogger(__name__)


def infer_precache_epoch_count(
    *,
    config: pyine.apps.trainers.hf_sft_trainer_configs.SFTTrainerAppMainConfig,
    datamodule: pyine.data.datamodule.ConversationDataModule[typing.Any],
    epochs_override: int | None,
) -> int:
    """Return the number of train epochs that should be covered by precaching.

    Args:
        config: the main HF trainer app config containing training_args_config.
        datamodule: the conversation datamodule instance.
        epochs_override: optional override value; if provided, returns this directly.

    Returns:
        The number of epochs to precache (at least 1).
    """
    if epochs_override is not None:
        logger.info(f"using epochs_override={epochs_override} for precaching")
        return epochs_override
    training_args = getattr(config, "training_args_config", None)
    if training_args is None:
        logger.warning("training_args_config missing; defaulting to 1 epoch for precaching")
        return 1
    num_train_epochs = getattr(training_args, "num_train_epochs", None)
    if num_train_epochs is None or num_train_epochs <= 0:
        num_train_epochs = 1.0
    epoch_cap = max(1, math.ceil(num_train_epochs))
    grad_accum = max(1, getattr(training_args, "gradient_accumulation_steps", 1))
    per_device_batch = getattr(training_args, "per_device_train_batch_size", None)
    if per_device_batch is None or per_device_batch <= 0:
        per_device_batch = 1
    drop_last = bool(getattr(training_args, "dataloader_drop_last", False))
    train_example_count = get_total_train_example_count(config=config, datamodule=datamodule)
    if train_example_count is None or train_example_count <= 0:
        logger.warning(
            f"unable to determine train dataset size (iterable or zero-length); precaching {epoch_cap} epoch(s)",
        )
        return epoch_cap
    if drop_last:
        batches_per_epoch = train_example_count // per_device_batch
    else:
        batches_per_epoch = math.ceil(train_example_count / per_device_batch)
    batches_per_epoch = max(1, batches_per_epoch)
    updates_per_epoch = max(1, math.ceil(batches_per_epoch / grad_accum))
    max_steps = getattr(training_args, "max_steps", None)
    if max_steps is None or max_steps <= 0:
        logger.info(
            "precacher epoch budget: "
            f"examples={train_example_count} "
            f"batch={per_device_batch} "
            f"grad_accum={grad_accum} "
            f"batches/epoch={batches_per_epoch} "
            f"updates/epoch={updates_per_epoch} "
            f"epochs={epoch_cap}",
        )
        return epoch_cap
    epochs_via_steps = max(1, math.ceil(max_steps / updates_per_epoch))
    inferred_epochs = min(epoch_cap, epochs_via_steps)
    logger.info(
        "precacher epoch budget (max_steps clip): "
        f"examples={train_example_count} "
        f"batch={per_device_batch} "
        f"grad_accum={grad_accum} "
        f"batches/epoch={batches_per_epoch} "
        f"updates/epoch={updates_per_epoch} "
        f"max_steps={max_steps} "
        f"epochs={inferred_epochs}",
    )
    return inferred_epochs


def get_total_train_example_count(
    *,
    config: pyine.apps.trainers.common.AppMainConfig,
    datamodule: pyine.data.datamodule.ConversationDataModule[typing.Any],
) -> int | None:
    """Return the total number of train examples across all configured subsets.

    Args:
        config: the main HF trainer app config containing datamodule_config.
        datamodule: the conversation datamodule instance.

    Returns:
        The total example count, or None if the dataset is iterable or length cannot be determined.
    """
    subset_names = getattr(getattr(config, "datamodule_config", None), "train_subset_names", None)
    if not subset_names:
        subset_names = ("train",)
    total = 0
    for subset_name in subset_names:
        try:
            parser = datamodule.get_parser(subset_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"failed to fetch parser for subset {subset_name}: {exc}")
            return None
        try:
            subset_len = len(parser)  # type: ignore[arg-type]
        except TypeError:
            logger.warning("parser for subset %s does not expose __len__; treating dataset as iterable", subset_name)
            return None
        total += subset_len
    return total


def prepare_subset_epoch(
    *,
    datamodule: pyine.data.datamodule.ConversationDataModule[typing.Any],
    subset_names: typing.Iterable[str],
    epoch: int,
) -> None:
    """Set the epoch on any subset parser that exposes the setter.

    Args:
        datamodule: the conversation datamodule instance.
        subset_names: iterable of subset names whose parsers should be updated.
        epoch: the epoch index to set.
    """
    for subset_name in subset_names:
        try:
            parser = datamodule.get_parser(subset_name)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"skipping epoch prep for subset {subset_name}: {exc}")
            continue
        setter = getattr(parser, "set_epoch", None)
        if setter is None or not callable(setter):
            continue
        try:
            setter(epoch)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"failed to set epoch for subset {subset_name}: {exc}")
