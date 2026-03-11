"""LLM classifier training app -- fine-tunes encoder LLMs as binary classifiers.

Fine-tunes an off-the-shelf encoder model (e.g., ModernBERT, DeBERTa-v3) as a
binary classifier on LMDB data from the probe training pipeline. Uses HuggingFace
Trainer for training with automatic DDP support.

See pyine/apps/trainers/LLM_CLASSIFIER_TRAINING_GUIDE.md for the full guide.
"""

from __future__ import annotations

import asyncio
import logging
import pathlib
import shutil
import typing

import numpy as np
import scipy.special
import sklearn.metrics
import torch
import transformers

import pyine.apps.trainers.common
import pyine.configs.schemas
import pyine.evals.common
import pyine.evals.correctness.scorers as correctness_scorers
import pyine.utils.transformers.data

if typing.TYPE_CHECKING:
    import datasets

    from pyine.apps.trainers.llm_classifier_trainer_configs import (
        LLMClassifierTrainerAppMainConfig,
    )

logger = logging.getLogger(__name__)

_MODEL_ARTIFACT_PATTERNS = (
    "config.json",
    "adapter_config.json",
    "adapter_model*.safetensors",
    "adapter_model*.bin",
    "model*.safetensors",
    "model*.bin",
    "pytorch_model*.bin",
    "model*.index.json",
    "pytorch_model*.index.json",
)
"""Patterns for model/adaptor inference artifacts that should be exported for best checkpoints."""

_RAW_COLUMNS_TO_REMOVE = ("text", "label", "sample_id", "code_type", "messages")
"""Columns from the raw LMDB dataset that should be removed after tokenization.

We explicitly list source columns to remove (rather than keeping a whitelist) so that
tokenizer-generated columns like token_type_ids are preserved.
"""


def _sync_model_pad_token_id_with_tokenizer(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> None:
    """Sync the model's ``pad_token_id`` with the tokenizer used for batching.

    This is especially important for decoder-only sequence-classification models (e.g. Qwen),
    whose classifier head may need ``model.config.pad_token_id`` to locate the last non-padding
    token in batched inputs.
    """
    tokenizer_pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if tokenizer_pad_token_id is None:
        raise ValueError("tokenizer.pad_token_id must be set for batched sequence classification")
    model_pad_token_id = getattr(model.config, "pad_token_id", None)  # type: ignore[reportUnknownMemberType]
    if model_pad_token_id != tokenizer_pad_token_id:
        model.config.pad_token_id = tokenizer_pad_token_id  # type: ignore[reportUnknownMemberType]


def _export_best_model_artifacts(
    trainer: transformers.Trainer,
    tokenizer: transformers.PreTrainedTokenizerBase,
    output_dir: pathlib.Path,
    output_suffix: str,
) -> pathlib.Path:
    """Export the selected best checkpoint to a clearly named directory under ``output_dir``.

    Copies inference artifacts from ``output_dir`` when ``load_best_model_at_end=True`` because the
    caller must save the already-reloaded best weights there before invoking this helper.
    Otherwise copies only inference artifacts from ``trainer.state.best_model_checkpoint``. This
    preserves adapter-style checkpoints instead of attempting to merge them, while intentionally
    excluding optimizer/scheduler/trainer-state files.
    """
    best_model_checkpoint = trainer.state.best_model_checkpoint
    if best_model_checkpoint is None:
        raise ValueError("cannot export best model artifacts when trainer.state.best_model_checkpoint is unset")
    relative_output_suffix = output_suffix.lstrip("/\\")
    if not relative_output_suffix:
        raise ValueError("best-model export path must not be empty or root-only")
    export_path = pathlib.Path(relative_output_suffix)
    if any(part == ".." for part in export_path.parts):
        raise ValueError("best-model export path must stay within output_dir")
    export_dir = output_dir / export_path
    if export_dir.exists():
        raise FileExistsError(f"best-model export directory already exists: {export_dir}")
    source_dir = output_dir if trainer.args.load_best_model_at_end else pathlib.Path(best_model_checkpoint)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"best-model export source directory does not exist: {source_dir}")
    export_dir.mkdir(parents=True, exist_ok=False)
    copied_model_artifact = False
    for pattern in _MODEL_ARTIFACT_PATTERNS:
        for source_path in source_dir.glob(pattern):
            if not source_path.is_file():
                continue
            shutil.copy2(source_path, export_dir / source_path.name)
            copied_model_artifact = True
    if not copied_model_artifact:
        raise FileNotFoundError(f"no model inference artifacts found in best-model export source: {source_dir}")
    tokenizer.save_pretrained(str(export_dir))  # pyright: ignore[reportUnknownMemberType]
    return export_dir


def _tokenize_for_classification(
    dataset: datasets.Dataset,
    tokenizer: transformers.PreTrainedTokenizerBase,
    max_seq_length: int,
    code_type_to_id: dict[str, int] | None = None,
    add_special_tokens: bool = True,
) -> datasets.Dataset:
    """Tokenize a dataset split for sequence classification.

    Args:
        dataset: HF dataset with 'text' and 'label' columns.
        tokenizer: Encoder tokenizer (adds [CLS]/[SEP] automatically).
        max_seq_length: Maximum sequence length.
        code_type_to_id: Optional mapping for per-code-type metrics.
        add_special_tokens: Whether to add special tokens (e.g. [CLS]/[SEP]). Set to False when
            text already contains special tokens from a chat template to avoid double insertion.

    Returns:
        Tokenized dataset with input_ids, attention_mask, labels columns.
        The dataset is left in default (arrow) format - Trainer handles
        tensor conversion internally.
    """

    def _tokenize(examples: dict[str, list[typing.Any]]) -> dict[str, typing.Any]:
        tokenized = tokenizer(
            examples["text"],
            max_length=max_seq_length,
            truncation=True,
            padding=False,  # Dynamic padding in collator
            add_special_tokens=add_special_tokens,
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
        logger.warning("code_type_id not found in eval_pred.inputs - skipping per-code-type metrics")
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
        # AUROC - guard against single-class eval AND degenerate probability
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


class ClassifierTrainResult(typing.NamedTuple):
    """Return value of classifier_train() with extra context for downstream evaluation."""

    trainer: transformers.Trainer
    """The HuggingFace Trainer containing the fine-tuned classifier model."""
    tokenizer: transformers.PreTrainedTokenizerBase
    """The tokenizer associated with the classifier model."""


def classifier_train(
    config: LLMClassifierTrainerAppMainConfig,
    runtime: pyine.configs.schemas.RuntimeConfig | None,
) -> ClassifierTrainResult:
    """Core LLM classifier training loop.

    Args:
        config: App configuration.
        runtime: Runtime configuration (wandb, output dir, etc.).

    Returns:
        ClassifierTrainResult with the trained Trainer and tokenizer.
    """
    # --- 1. Load model + tokenizer ---
    tokenizer = config.get_tokenizer()
    model = config.get_model()
    _sync_model_pad_token_id_with_tokenizer(model, tokenizer)

    # --- 2. Set truncation side (if configured) ---
    if config.truncation_side is not None:
        tokenizer.truncation_side = config.truncation_side
    # --- 2b. Validate max_seq_length against tokenizer and model limits ---
    tokenizer_max_length: int | None = getattr(tokenizer, "model_max_length", None)
    model_max_positions: int | None = getattr(model.config, "max_position_embeddings", None)  # type: ignore[reportUnknownMemberType]
    effective_limit: int | None = None
    if tokenizer_max_length is not None and model_max_positions is not None:
        effective_limit = min(tokenizer_max_length, model_max_positions)
    elif tokenizer_max_length is not None:
        effective_limit = tokenizer_max_length
    elif model_max_positions is not None:
        effective_limit = model_max_positions
    if effective_limit is not None and config.max_seq_length > effective_limit:
        raise ValueError(
            f"max_seq_length ({config.max_seq_length}) exceeds effective model limit "
            f"(tokenizer.model_max_length={tokenizer_max_length}, "
            f"model.config.max_position_embeddings={model_max_positions}, "
            f"effective={effective_limit}); "
            "reduce max_seq_length or use a model/tokenizer with a larger context window"
        )

    # --- 3. Load LMDB data via DataModule (handles DDP coordination + caching) ---
    datamodule = pyine.apps.trainers.common.prepare_datamodule(config, runtime)
    # both ProbeDataModule and CorrectnessDataModule provide get_probe_dataset/code_type_to_id/id_to_code_type
    probe_dataset_owner = typing.cast("typing.Any", datamodule)
    raw_ds = typing.cast(
        "datasets.DatasetDict",
        probe_dataset_owner.get_probe_dataset(text_field=config.text_field),
    )
    code_type_to_id = typing.cast("dict[str, int]", datamodule.code_type_to_id)  # pyright: ignore[reportUnknownMemberType,reportAttributeAccessIssue]
    id_to_code_type = typing.cast("dict[int, str]", datamodule.id_to_code_type)  # pyright: ignore[reportUnknownMemberType,reportAttributeAccessIssue]

    # --- 4. Log class distribution (always, regardless of class_weight_mode) ---
    pyine.apps.trainers.common.log_class_distribution(raw_ds, logger)

    # --- 5. Format messages -> text (chat template if available, else plain concat), then tokenize ---
    has_chat_template = pyine.utils.transformers.data.tokenizer_has_chat_template(tokenizer)
    input_formatting_mode = "chat_template" if has_chat_template else "role_tagged_text"
    logger.info(
        f"classifier inputs: text_field={config.text_field}, truncation_side={tokenizer.truncation_side}, "
        f"input_formatting_mode={input_formatting_mode}, add_special_tokens={not has_chat_template}"
    )
    raw_ds = pyine.apps.trainers.common.apply_messages_formatting(raw_ds, tokenizer)
    # when a chat template produced the text, special tokens are already embedded;
    # encoder tokenizers (no chat template) need add_special_tokens=True for [CLS]/[SEP]
    add_special_tokens = not has_chat_template
    train_ds = _tokenize_for_classification(
        dataset=raw_ds["train"],
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        code_type_to_id=code_type_to_id if config.log_per_code_type_metrics else None,
        add_special_tokens=add_special_tokens,
    )
    valid_ds = _tokenize_for_classification(
        dataset=raw_ds["valid"],
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        code_type_to_id=code_type_to_id if config.log_per_code_type_metrics else None,
        add_special_tokens=add_special_tokens,
    )

    # --- 6. Setup training ---
    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    training_args_dict = config.training_args_config.model_dump()
    if runtime is not None:
        training_args_dict["output_dir"] = runtime.output_dir
    if runtime is not None and runtime.wandb_run is not None:
        training_args_dict["report_to"] = ["wandb"]
        runtime.wandb_run.summary["classifier/text_field"] = config.text_field  # type: ignore[reportUnknownMemberType]
        runtime.wandb_run.summary["classifier/truncation_side"] = tokenizer.truncation_side  # type: ignore[reportUnknownMemberType]
        runtime.wandb_run.summary["classifier/input_formatting_mode"] = input_formatting_mode  # type: ignore[reportUnknownMemberType]
        runtime.wandb_run.summary["classifier/add_special_tokens"] = not has_chat_template  # type: ignore[reportUnknownMemberType]
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
    assert config.num_labels == 2, "rebalancing code below only supports two classes"
    assert config.id2label[1] == "correct", "rebalancing code below expects label 1 = correct"
    train_labels = np.array(raw_ds["train"]["label"])  # type: ignore  # datasets stubs
    n_train, n_pos = len(train_labels), int(train_labels.sum())  # type: ignore
    n_neg = n_train - n_pos
    pos_ratio = n_pos / n_train
    trainer_cls: type[transformers.Trainer] = transformers.Trainer
    trainer_kwargs: dict[str, typing.Any] = {}
    if config.class_weight_mode == "balanced":
        if n_pos == 0 or n_neg == 0:
            raise ValueError(f"need both classes for balanced weighting (pos={n_pos}, neg={n_neg})")
        class_weights = torch.tensor([pos_ratio, 1 - pos_ratio], dtype=torch.float32)
        class_weights = class_weights / class_weights.mean()  # normalize so mean weight = 1
        logger.info(f"using balanced class weights (inverse-frequency): {class_weights.tolist()}")  # type: ignore
        trainer_kwargs["class_weights"] = class_weights
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

    metric_for_best_model = trainer.args.metric_for_best_model
    greater_is_better = trainer.args.greater_is_better
    best_metric_direction = "higher" if greater_is_better else "lower"
    if trainer.state.best_model_checkpoint is not None:
        logger.info(
            f"best model checkpoint: {trainer.state.best_model_checkpoint}; "
            f"criterion={metric_for_best_model} ({best_metric_direction} is better); "
            f"best_metric={trainer.state.best_metric}"
        )

    # --- 9. Save final model ---
    if config.save_model:
        assert trainer.args.output_dir is not None
        output_dir = pathlib.Path(trainer.args.output_dir)
        # when load_best_model_at_end=True, HuggingFace has already reloaded the winning weights
        # into memory by this point, so saving now materializes the best model at output_dir.
        trainer.save_model()
        tokenizer.save_pretrained(str(output_dir))  # pyright: ignore[reportUnknownMemberType]
        if config.save_best_model_export:
            if trainer.args.load_best_model_at_end:
                logger.debug(
                    f"output_dir already contains best-loaded weights because "
                    f"load_best_model_at_end={trainer.args.load_best_model_at_end}; "
                    f"exporting a suffixed best copy for clarity using output_dir={output_dir}"
                )
            best_export_dir = _export_best_model_artifacts(
                trainer=trainer,
                tokenizer=tokenizer,
                output_dir=output_dir,
                output_suffix=config.best_model_output_suffix,
            )
            logger.info(
                f"exported best model to: {best_export_dir}; "
                f"source_checkpoint={trainer.state.best_model_checkpoint}; "
                f"criterion={metric_for_best_model} ({best_metric_direction} is better); "
                f"best_metric={trainer.state.best_metric}"
            )

    return ClassifierTrainResult(trainer=trainer, tokenizer=tokenizer)


async def main(
    config: LLMClassifierTrainerAppMainConfig,
    runtime: pyine.configs.schemas.RuntimeConfig | None = None,
    skip_training: bool = False,
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

    if skip_training:
        if config.classifier_checkpoint_path is None:
            raise ValueError("skip_training=True requires config.classifier_checkpoint_path to be set")
        logger.info(f"skip_training mode; loading classifier from {config.classifier_checkpoint_path}")
        classifier_model = config.get_model(checkpoint_path=config.classifier_checkpoint_path)
        classifier_model.eval()
        classifier_model.requires_grad_(False)
        tokenizer = config.get_tokenizer(checkpoint_path=config.classifier_checkpoint_path)
        _sync_model_pad_token_id_with_tokenizer(classifier_model, tokenizer)
        if config.truncation_side is not None:
            tokenizer.truncation_side = config.truncation_side
    else:
        train_result = classifier_train(config=config, runtime=runtime)
        classifier_model = typing.cast("transformers.PreTrainedModel", train_result.trainer.model)  # pyright: ignore[reportUnknownMemberType]
        tokenizer = train_result.tokenizer

    # benchmarking phase (if enabled); goes through the standard evaluate_model pipeline
    # which handles datamodule setup, W&B metric definition, logging, etc.
    if config.evals_config is not None:
        scorer = correctness_scorers.LLMClassifierScorer(
            model=classifier_model,
            tokenizer=tokenizer,
            max_seq_length=config.max_seq_length,
            text_field=config.text_field,
        )
        await pyine.apps.trainers.common.evaluate_model(  # type: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            model=scorer,
            tokenizer=None,
            datamodule=None,  # correctness pipeline constructs its own datamodule
            config=config,
            runtime=runtime,
        )

    if runtime is not None:
        runtime.finalize()


def async_classifier_trainer_main_wrapper(
    config: LLMClassifierTrainerAppMainConfig,
    runtime: pyine.configs.schemas.RuntimeConfig | None = None,
    skip_training: bool = False,
) -> None:
    """Synchronous wrapper around the async main."""
    asyncio.run(main(config=config, runtime=runtime, skip_training=skip_training))


if __name__ == "__main__":
    import pyine.apps.trainers.common
    import pyine.apps.trainers.llm_classifier_trainer_configs as llm_classifier_configs

    pyine.apps.trainers.common.hydra_main(
        eval_type=pyine.evals.common.EvalType.CORRECTNESS,
        hydra_config_registration_fn=llm_classifier_configs.register_hydra_configs,
        async_main_wrapper=async_classifier_trainer_main_wrapper,
    )
