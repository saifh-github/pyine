"""LLM classifier training app -- fine-tunes encoder LLMs as binary classifiers.

Fine-tunes an off-the-shelf encoder model (e.g., ModernBERT, DeBERTa-v3) as a
binary classifier on LMDB data from the probe training pipeline. Uses HuggingFace
Trainer for training with automatic DDP support.

See pyine/apps/trainers/LLM_CLASSIFIER_TRAINING_GUIDE.md for the full guide.
"""

from __future__ import annotations

import asyncio
import logging
import typing

import numpy as np
import scipy.special
import sklearn.metrics
import torch
import transformers

import pyine.apps.trainers.common
import pyine.configs.schemas
import pyine.evals.common
import pyine.probes.data.datamodule

if typing.TYPE_CHECKING:
    import datasets

    from pyine.apps.trainers.llm_classifier_trainer_configs import (
        LLMClassifierTrainerAppMainConfig,
    )

logger = logging.getLogger(__name__)

# Columns from the raw LMDB dataset that should be removed after tokenization.
# We explicitly list source columns to remove (rather than keeping a whitelist)
# so that tokenizer-generated columns like token_type_ids are preserved.
_RAW_COLUMNS_TO_REMOVE = ("text", "label", "sample_id", "code_type")


def _tokenize_for_classification(
    dataset: datasets.Dataset,
    tokenizer: transformers.PreTrainedTokenizerBase,
    max_seq_length: int,
    code_type_to_id: dict[str, int] | None = None,
) -> datasets.Dataset:
    """Tokenize a dataset split for sequence classification.

    Args:
        dataset: HF dataset with 'text' and 'label' columns.
        tokenizer: Encoder tokenizer (adds [CLS]/[SEP] automatically).
        max_seq_length: Maximum sequence length.
        code_type_to_id: Optional mapping for per-code-type metrics.

    Returns:
        Tokenized dataset with input_ids, attention_mask, labels columns.
        The dataset is left in default (arrow) format — Trainer handles
        tensor conversion internally.
    """

    def _tokenize(examples: dict[str, list[typing.Any]]) -> dict[str, typing.Any]:
        tokenized = tokenizer(
            examples["text"],
            max_length=max_seq_length,
            truncation=True,
            padding=False,  # Dynamic padding in collator
        )
        tokenized["labels"] = examples["label"]
        if code_type_to_id is not None:
            tokenized["code_type_id"] = [code_type_to_id[ct] for ct in examples["code_type"]]
        return tokenized  # pyright: ignore[reportReturnType]  # BatchEncoding is dict-like

    # Only remove known raw source columns; preserve any tokenizer-generated
    # columns (e.g., token_type_ids for DeBERTa/classic BERT).
    remove_cols = [c for c in _RAW_COLUMNS_TO_REMOVE if c in dataset.column_names]
    # NOTE: Do NOT call ds.set_format("torch") here. Trainer handles tensor
    # conversion internally, and set_format("torch") can interfere with
    # include_for_metrics (code_type_id may be dropped or mis-serialized
    # during eval prediction gathering).
    return dataset.map(_tokenize, batched=True, remove_columns=remove_cols)  # pyright: ignore[reportUnknownMemberType]  # datasets stubs


def _log_class_distribution(
    dataset_dict: datasets.DatasetDict,
    log: logging.Logger,
) -> None:
    """Log label counts and class balance for train/valid splits."""
    for split_name in ("train", "valid"):
        if split_name not in dataset_dict:
            continue
        labels = typing.cast("list[int]", dataset_dict[split_name]["label"])
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        ratio = n_pos / len(labels) if len(labels) > 0 else 0.0
        log.info(
            "%s split: %d samples (pos=%d, neg=%d, pos_ratio=%.3f)",
            split_name,
            len(labels),
            n_pos,
            n_neg,
            ratio,
        )


def _add_per_code_type_metrics(
    metrics: dict[str, float],
    labels: np.ndarray[typing.Any, typing.Any],
    probs: np.ndarray[typing.Any, typing.Any],
    predictions: np.ndarray[typing.Any, typing.Any],
    eval_pred: transformers.EvalPrediction,
    id_to_code_type: dict[int, str],
) -> None:
    """Compute per-code-type accuracy and AUROC."""
    # eval_pred.inputs is a dict when include_for_metrics is set.
    # Guard against missing/None inputs gracefully.
    code_type_ids = None
    if hasattr(eval_pred, "inputs") and isinstance(eval_pred.inputs, dict):  # pyright: ignore[reportUnknownMemberType]  # transformers stubs
        code_type_ids = eval_pred.inputs.get("code_type_id")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # transformers stubs
    if code_type_ids is None:
        logger.warning("code_type_id not found in eval_pred.inputs — skipping per-code-type metrics")
        return

    for ct_id, ct_name in id_to_code_type.items():
        mask = code_type_ids == ct_id  # pyright: ignore[reportUnknownVariableType]
        n_samples = int(mask.sum())  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        if n_samples < 2:
            continue

        ct_labels = labels[mask]  # pyright: ignore[reportUnknownVariableType]
        ct_preds = predictions[mask]  # pyright: ignore[reportUnknownVariableType]
        ct_probs = probs[mask]  # pyright: ignore[reportUnknownVariableType]

        metrics[f"accuracy/code_type/{ct_name}"] = float(
            sklearn.metrics.accuracy_score(ct_labels, ct_preds)  # pyright: ignore[reportUnknownMemberType]  # sklearn stubs
        )

        # AUROC requires both classes present AND at least 2 samples
        ct_unique = {int(v) for v in ct_labels}
        if len(ct_unique) >= 2 and n_samples >= 2:
            metrics[f"auroc/code_type/{ct_name}"] = float(
                sklearn.metrics.roc_auc_score(ct_labels, ct_probs)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # sklearn stubs
            )


def build_compute_metrics(
    id_to_code_type: dict[int, str] | None = None,
) -> typing.Callable[[transformers.EvalPrediction], dict[str, float]]:
    """Build a compute_metrics function for sequence classification.

    When id_to_code_type is provided, also computes per-code-type metrics.

    Returns:
        Function compatible with Trainer's compute_metrics parameter.
    """

    @typing.no_type_check
    def compute_metrics(eval_pred: transformers.EvalPrediction) -> dict[str, float]:
        # Handle tuple logits: some HF versions return (logits, hidden_states, ...)
        raw_preds = eval_pred.predictions
        logits = raw_preds[0] if isinstance(raw_preds, tuple) else raw_preds
        labels = eval_pred.label_ids
        predictions = np.argmax(logits, axis=1)
        probs = scipy.special.softmax(logits, axis=1)[:, 1]
        metrics: dict[str, float] = {
            "accuracy": float(sklearn.metrics.accuracy_score(labels, predictions)),
            "f1": float(sklearn.metrics.f1_score(labels, predictions, zero_division=0)),
            "precision": float(sklearn.metrics.precision_score(labels, predictions, zero_division=0)),
            "recall": float(sklearn.metrics.recall_score(labels, predictions, zero_division=0)),
        }
        # AUROC — guard against single-class eval AND degenerate probability
        # distributions (all identical probs).
        unique_labels = {int(v) for v in labels}
        if len(unique_labels) >= 2 and len(labels) >= 2:
            metrics["auroc"] = float(sklearn.metrics.roc_auc_score(labels, probs))
        else:
            metrics["auroc"] = float("nan")
        # Per-code-type metrics (if code_type_id available via include_for_metrics)
        if id_to_code_type is not None:
            _add_per_code_type_metrics(
                metrics,
                labels,
                probs,
                predictions,
                eval_pred,
                id_to_code_type,
            )
        return metrics

    return compute_metrics  # type: ignore[reportUnknownVariableType]


class WeightedLossTrainer(transformers.Trainer):
    """Trainer subclass that applies class weights to CrossEntropyLoss."""

    def __init__(self, *args: typing.Any, class_weights: torch.Tensor | None = None, **kwargs: typing.Any) -> None:
        super().__init__(*args, **kwargs)  # pyright: ignore[reportUnknownMemberType]  # transformers stubs
        self._class_weights = class_weights

    def compute_loss(  # type: ignore[override]
        self,
        model: typing.Any,
        inputs: dict[str, typing.Any],
        return_outputs: bool = False,
        **kwargs: typing.Any,
    ) -> typing.Any:
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self._class_weights is not None:
            weight = self._class_weights.to(logits.device)
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=weight)
        else:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss


def classifier_train(
    config: LLMClassifierTrainerAppMainConfig,
    runtime: pyine.configs.schemas.RuntimeConfig | None,
) -> transformers.Trainer:
    """Core LLM classifier training loop.

    Args:
        config: App configuration.
        runtime: Runtime configuration (wandb, output dir, etc.).

    Returns:
        The trained HuggingFace Trainer instance.
    """
    # --- 1. Load model + tokenizer ---
    tokenizer = config.get_tokenizer()
    model = config.get_model()

    # --- 2. Set truncation side (if configured) ---
    if config.truncation_side is not None:
        tokenizer.truncation_side = config.truncation_side

    # --- 3. Load LMDB data via DataModule (handles DDP coordination + caching) ---
    datamodule = pyine.apps.trainers.common.prepare_datamodule(config, runtime)
    assert isinstance(datamodule, pyine.probes.data.datamodule.ProbeDataModule)
    raw_ds = datamodule.get_probe_dataset()
    code_type_to_id = datamodule.code_type_to_id
    id_to_code_type = datamodule.id_to_code_type

    # --- 4. Log class distribution (always, regardless of class_weight_mode) ---
    _log_class_distribution(raw_ds, logger)

    # --- 5. Tokenize ---
    train_ds = _tokenize_for_classification(
        raw_ds["train"],
        tokenizer,
        config.max_seq_length,
        code_type_to_id if config.log_per_code_type_metrics else None,
    )
    valid_ds = _tokenize_for_classification(
        raw_ds["valid"],
        tokenizer,
        config.max_seq_length,
        code_type_to_id if config.log_per_code_type_metrics else None,
    )

    # --- 6. Setup training ---
    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    training_args_dict = config.training_args_config.model_dump()
    if runtime is not None:
        training_args_dict["output_dir"] = runtime.output_dir
    if runtime is not None and runtime.wandb_run is not None:
        training_args_dict["report_to"] = ["wandb"]
    pyine.apps.trainers.common.resolve_save_on_each_node(training_args_dict, runtime)

    # Wire include_for_metrics for per-code-type metrics
    if config.log_per_code_type_metrics:
        training_args_dict["include_for_metrics"] = ["code_type_id"]

    training_args = transformers.TrainingArguments(**training_args_dict)
    # Wire seed from runtime config for reproducibility
    if runtime is not None:
        training_args.seed = runtime.seed

    compute_metrics_fn = build_compute_metrics(
        id_to_code_type=id_to_code_type if config.log_per_code_type_metrics else None,
    )

    # --- 7. Optionally compute class weights for imbalanced data ---
    trainer_cls: type[transformers.Trainer] = transformers.Trainer
    trainer_kwargs: dict[str, typing.Any] = {}
    if config.class_weight_mode == "balanced":
        train_labels = np.array(raw_ds["train"]["label"])  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]  # datasets stubs
        class_counts = np.bincount(train_labels, minlength=config.num_labels)  # pyright: ignore[reportUnknownArgumentType]
        n_train = len(train_labels)  # pyright: ignore[reportUnknownArgumentType]
        class_weights = n_train / (config.num_labels * class_counts.clip(min=1))
        logger.info("Using balanced class weights: %s", class_weights)
        trainer_kwargs["class_weights"] = torch.tensor(class_weights, dtype=torch.float32)
        trainer_cls = WeightedLossTrainer

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        **trainer_kwargs,
    )

    # --- 8. Train ---
    trainer.train()  # pyright: ignore[reportUnknownMemberType]  # transformers stubs

    # --- 9. Save final model ---
    if config.save_model:
        trainer.save_model()

    return trainer


async def main(
    config: LLMClassifierTrainerAppMainConfig,
    runtime: pyine.configs.schemas.RuntimeConfig | None = None,
) -> None:
    """Main entrypoint for LLM classifier training."""
    import pyine.utils.reprod

    pyine.utils.reprod.entrypoint_setup(
        runtime_config=runtime,
        use_wandb_logging=config.use_wandb_logging,
        main_config=config,
    )

    if runtime is not None and runtime.dry_run:
        logger.info("dry run mode; skipping classifier training")
        return

    trainer = classifier_train(config=config, runtime=runtime)

    # benchmarking phase (if enabled)
    if config.evals_config is not None:
        await pyine.apps.trainers.common.evaluate_model(
            model=trainer.model,  # @@@@@ TODO: wrap this in GuardrailScorer-compat wrapper!
            tokenizer=None,
            datamodule=None,
            config=config,
            runtime=runtime,
        )

    if runtime is not None:
        runtime.finalize()


def async_classifier_trainer_main_wrapper(
    config: LLMClassifierTrainerAppMainConfig,
    runtime: pyine.configs.schemas.RuntimeConfig | None = None,
) -> None:
    """Synchronous wrapper around the async main."""
    asyncio.run(main(config=config, runtime=runtime))


if __name__ == "__main__":
    import pyine.apps.trainers.common
    import pyine.apps.trainers.llm_classifier_trainer_configs as llm_classifier_configs

    pyine.apps.trainers.common.hydra_main(
        eval_type=pyine.evals.common.EvalType.CORRECTNESS,
        hydra_config_registration_fn=llm_classifier_configs.register_hydra_configs,
        async_main_wrapper=async_classifier_trainer_main_wrapper,
    )
