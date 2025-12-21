"""Shared fixtures and utilities for HF trainer integration tests.

This module provides common test infrastructure for both SFT and RL trainer integration tests,
including mock datamodules, sample generation, and environment isolation.
"""

import pathlib
import random
import typing

import datasets as hf_datasets
import pytest
import pytest_mock
import safetensors.torch
import torch
import transformers
import wandb

import pyine.configs.schemas
import pyine.data.datamodule
import pyine.data.traces.dataset_utils
import pyine.data.utils.splits
import pyine.evals.code_exec.evaluator
import pyine.organisms.datamodules.samples.common
import pyine.organisms.datamodules.shortcuts
import pyine.organisms.datamodules.shortcuts_configs
import pyine.utils.transformers
import tests.env_checks


def make_mock_sample(
    identifier: str,
    code: str,
    inputs: str,
    expected_output: str,
    description: str = "A simple test function.",
    code_type: str = "original",
    comma_separated_tags: str = "subset:mock,augment:something",
) -> pyine.organisms.datamodules.samples.common.SampleData:
    """Create a mock SampleData instance for testing."""
    return pyine.organisms.datamodules.samples.common.SampleData(
        identifier=identifier,
        code=code,
        description=description,
        entrypoint="",
        first_line=0,
        last_line=code.count("\n") + 1,
        inputs=inputs,
        expected_output=expected_output,
        predict_type=pyine.organisms.datamodules.samples.common.SamplePredictType.program_output,
        code_type=code_type,
        trace_step_count=5,
        comma_separated_tags=comma_separated_tags,
        has_code_override=False,
        complexity_metrics={},
    )


def generate_mock_samples(
    count: int,
    subset_name: str,
    seed: int = 42,
) -> list[pyine.organisms.datamodules.samples.common.SampleData]:
    """Generate a list of mock samples with simple arithmetic code snippets.

    Each sample is a simple Python program that performs basic arithmetic or string operations with
    predictable outputs.
    """
    rng = random.Random(seed)
    base = rng.randint(0, 10_000)
    samples = []
    for idx in range(count):
        sample_type = idx % 4
        sample_code_type = "original" if idx % 2 == 0 else "hinted"
        tags = f"subset:{subset_name},augment:something"
        if idx % 3 == 0:
            tags = f"{tags},bias_keyword:demo,has_bias_keyword:1"
        if sample_type == 0:
            code = f"x = {base + idx + 1}\ny = {base + idx + 2}\nprint(x + y)"
            inputs = ""
            expected_output = str((base + idx + 1) + (base + idx + 2))
        elif sample_type == 1:
            code = f"nums = [{base + idx}, {base + idx + 1}, {base + idx + 2}]\nprint(sum(nums))"
            inputs = ""
            expected_output = str((base + idx) + (base + idx + 1) + (base + idx + 2))
        elif sample_type == 2:
            code = f'word = "test{base + idx}"\nprint(word.upper())'
            inputs = ""
            expected_output = f"TEST{base + idx}"
        else:
            code = f"for i in range({idx % 3 + 1}):\n    print(i)"
            inputs = ""
            expected_output = "\n".join(str(i) for i in range(idx % 3 + 1))
        samples.append(
            make_mock_sample(
                identifier=f"mock/{subset_name}/p{idx:05d}/s0001/t0001",
                code=code,
                inputs=inputs,
                expected_output=expected_output,
                description=f"Mock sample #{idx} for testing.",
                code_type=sample_code_type,
                comma_separated_tags=tags,
            )
        )
    return samples


class MockShortcutsDataModule(pyine.data.datamodule.ConversationDataModule[typing.Any]):
    """A mock datamodule that provides fake trace samples for integration testing.

    This datamodule bypasses LMDB loading and instead uses in-memory mock samples.
    It implements the minimal interface required by the HF trainer app.
    """

    def __init__(
        self,
        train_samples: list[pyine.organisms.datamodules.samples.common.SampleData],
        valid_samples: list[pyine.organisms.datamodules.samples.common.SampleData],
        mocker: pytest_mock.MockerFixture,
        prompt_name: str = "code_execution",
    ) -> None:
        self._train_samples = train_samples
        self._valid_samples = valid_samples
        self._prompt_name = prompt_name
        self._hf_cache_dir: pathlib.Path | None = None
        self.config = mocker.MagicMock()
        self.config.prompt_config.prompt_name = prompt_name
        self.config.prompt_config.use_chat_template = True
        self.config.prompt_config.include_examples = False
        self.config.hf_messages_key = "messages"
        self.verbose = False

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str | None = None) -> None:
        pass

    def teardown(self, stage: str | None = None) -> None:
        pass

    def _samples_to_messages(
        self,
        samples: list[pyine.organisms.datamodules.samples.common.SampleData],
        append_answer: bool = True,
    ) -> list[dict[str, typing.Any]]:
        """Convert SampleData instances to message-format dicts with sample_data preserved.

        Creates simple user/assistant conversations for SFT training without using the full
        prompt template system to avoid complexity during testing.

        Args:
            samples: List of SampleData instances to convert.
            append_answer: Whether to include the assistant's answer in the conversation.
                For SFT, this should be True. For RL, this should be False.
        """
        results = []
        for sample in samples:
            user_content = f"What is the output of this Python code?\n\n```python\n{sample.code}\n```"
            if sample.inputs:
                user_content += f"\n\nInput: {sample.inputs}"
            messages: list[dict[str, str]] = [
                {"role": "system", "content": "You are a helpful assistant that predicts Python code output."},
                {"role": "user", "content": user_content},
            ]
            if append_answer:
                messages.append({"role": "assistant", "content": sample.expected_output})
            sample_data = {
                "identifier": sample.identifier,
                "code": sample.code,
                "description": sample.description,
                "entrypoint": sample.entrypoint,
                "first_line": sample.first_line,
                "last_line": sample.last_line,
                "inputs": sample.inputs,
                "expected_output": sample.expected_output,
                "predict_type": sample.predict_type.value,
                "code_type": sample.code_type,
                "trace_step_count": sample.trace_step_count,
                "comma_separated_tags": sample.comma_separated_tags,
                "has_code_override": sample.has_code_override,
                "complexity_metrics": sample.complexity_metrics,
            }
            results.append({"messages": messages, "sample_data": sample_data})
        return results

    def get_hf_messages_dataset(
        self,
        subset_name: str,
        *,
        keep_original_data: bool = False,
        force_regenerate: bool = False,
        append_answer: bool = True,
        merge_system_with_user: bool = False,
        epoch: int | None = None,
    ) -> hf_datasets.Dataset:
        """Build a HuggingFace dataset of messages from mock samples.

        For RL training (append_answer=False), also adds a 'prompt' column that TRL's GRPOTrainer
        expects. The 'prompt' column contains the same messages list as 'messages'.
        """
        samples = self._train_samples if "train" in subset_name else self._valid_samples
        message_dicts = self._samples_to_messages(samples, append_answer=append_answer)
        if merge_system_with_user:
            for item in message_dicts:
                messages = item["messages"]
                if len(messages) >= 2 and messages[0]["role"] == "system" and messages[1]["role"] == "user":
                    system_content = messages[0]["content"]
                    user_content = messages[1]["content"]
                    messages[1]["content"] = f"{system_content}\n\n{user_content}"
                    item["messages"] = messages[1:]
        # for RL training, TRL's GRPOTrainer expects a 'prompt' column
        if not append_answer:
            for item in message_dicts:
                item["prompt"] = item["messages"]
        return hf_datasets.Dataset.from_list(message_dicts)

    def get_hf_tokenized_examples_dataset(
        self,
        subset_name: str,
        tokenizer: transformers.PreTrainedTokenizer,
        model_max_seq_len: int,
        keep_extra_fields: list[str] | bool | None = None,
        num_proc: int = 1,
        force_regenerate: bool = False,
        epoch: int | None = None,
    ) -> hf_datasets.Dataset:
        """Build a tokenized dataset for SFT training."""
        messages_ds = self.get_hf_messages_dataset(subset_name)
        return pyine.utils.transformers.prepare_examples_from_conversations(
            convo_ds=messages_ds,
            tokenizer=tokenizer,
            max_seq_len=model_max_seq_len,
            num_proc=num_proc,
            keep_extra_fields=(
                ["sample_data"]
                if keep_extra_fields is None and subset_name != "train"
                else (False if keep_extra_fields is None else keep_extra_fields)
            ),
            keep_in_memory=True,
        )


@pytest.fixture
def isolate_caches_and_outputs(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Set minimal environment variables for isolated test execution.

    Only sets variables needed for wandb and tokenizer parallelism. The mock datamodule
    doesn't need real data paths, and output directories are already handled via tmp_path
    in the training config fixture.
    """
    monkeypatch.setenv("WANDB_PROJECT", "pyine-tests")
    wandb_dir = tmp_path / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("WANDB_DIR", str(wandb_dir))
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")


@pytest.fixture
def mock_datamodule(mocker: pytest_mock.MockerFixture) -> MockShortcutsDataModule:
    """Create a mock datamodule with fake trace samples."""
    train_samples = generate_mock_samples(count=20, subset_name="train", seed=42)
    valid_samples = generate_mock_samples(count=10, subset_name="valid", seed=123)
    return MockShortcutsDataModule(
        train_samples=train_samples,
        valid_samples=valid_samples,
        mocker=mocker,
    )


@pytest.fixture
def runtime_config(
    tmp_path: pathlib.Path,
) -> typing.Generator[pyine.configs.schemas.RuntimeConfig, None, None]:
    """Create a runtime config with wandb logging for integration testing."""
    output_dir = tmp_path / "runtime_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = pyine.configs.schemas.RuntimeConfig(
        exp_name="hf-trainer-test",
        run_name="integration-test",
        run_group="integration-tests",
        app_name="test_hf_trainer_integration",
        output_dir=str(output_dir),
        seed=42,
    )
    runtime.init_wandb()
    yield runtime
    runtime.finalize()


@pytest.fixture()
def real_datamodule_config() -> pyine.organisms.datamodules.shortcuts.ShortcutBiasDataModuleConfig:
    """Create a real (yet tiny) ShortcutBiasDataModuleConfig using the latest TACO traces dataset.

    This fixture requires:
    - TACO traces dataset to be available (skip via TACO_TRACES_DATASET_MISSING);
    - TACO dataset split file to be available (skip via TACO_TRACES_DATASET_SPLIT_MISSING).
    """
    if tests.env_checks.TACO_TRACES_DATASET_MISSING:
        pytest.skip("TACO traces dataset required for real datamodule")
    if tests.env_checks.TACO_TRACES_DATASET_SPLIT_MISSING:
        pytest.skip("TACO traces dataset split required for real datamodule")
    lmdb_path = pyine.data.traces.dataset_utils.get_latest_dataset_path("TACO")
    split_file_path = pyine.data.utils.splits.get_dataset_split_file_path("TACO", must_exist=True)
    return pyine.organisms.datamodules.shortcuts_configs.get_datamodule_config(
        lmdb_paths=[lmdb_path],
        split_file_path=split_file_path,
        seed=42,
        max_solution_count=20,
        as_pydantic=True,
        min_samples_with_hints=0,  # disable validation for integration tests
        min_samples_without_hints=0,  # disable validation for integration tests
    )


@pytest.fixture
def real_datamodule(
    real_datamodule_config: pyine.organisms.datamodules.shortcuts.ShortcutBiasDataModuleConfig,
) -> pyine.organisms.datamodules.shortcuts.ShortcutBiasDataModule:
    """Create a real (yet tiny) ShortcutBiasDataModule using the latest TACO traces dataset.

    This fixture requires:
    - TACO traces dataset to be available (skip via TACO_TRACES_DATASET_MISSING);
    - TACO dataset split file to be available (skip via TACO_TRACES_DATASET_SPLIT_MISSING).
    """
    datamodule = pyine.organisms.datamodules.shortcuts.ShortcutBiasDataModule(config=real_datamodule_config)
    datamodule.prepare_data()
    datamodule.setup()
    return datamodule


def get_expected_code_exec_eval_metrics() -> list[str]:
    """Returns expected code execution evaluation metric names.

    These are the core metrics from pyine.evals.code_exec that should be logged
    to wandb when evaluation runs.

    Returns:
        List of metric names that the code exec evaluator produces.
    """
    return pyine.evals.code_exec.evaluator.OutcomeEvaluator.get_supported_metric_names()


def verify_wandb_training_metrics(
    wandb_run: wandb.Run,
) -> dict[str, typing.Any]:
    """Verify that expected training metrics were logged to wandb and return the found metrics.

    Args:
        wandb_run: The wandb run object to check.

    Returns:
        Dictionary of found metrics (subset of wandb_run.summary).

    Raises:
        AssertionError: If expected metrics are not found.
    """
    expected_prefixes = [
        "train/loss",
        "train/global_step",
        "eval/loss",
    ]
    summary = dict(wandb_run.summary)  # type: ignore[reportUnknownArgumentType]
    found_metrics: dict[str, typing.Any] = {}
    missing_prefixes: list[str] = []
    for prefix in expected_prefixes:
        matching_keys = [key for key in summary if key.startswith(prefix)]
        if matching_keys:
            for key in matching_keys:
                found_metrics[key] = summary[key]
        else:
            missing_prefixes.append(prefix)
    assert not missing_prefixes, (
        f"missing expected wandb training metrics with prefixes: {missing_prefixes}; "
        f"available keys: {sorted(summary.keys())}"
    )
    return found_metrics


def verify_wandb_has_category_metrics(
    wandb_run: wandb.Run,
    expected_categories: list[str],
    metric_suffix: str = "loss",
) -> dict[str, typing.Any]:
    """Verify that category-wise metrics were logged to wandb.

    Args:
        wandb_run: The wandb run object to check.
        expected_categories: List of category names (e.g., ["code_type/original", "code_type/hinted"]).
        metric_suffix: The metric suffix to check for (e.g., "loss", "accuracy_hard").

    Returns:
        Dictionary of found category metrics.

    Raises:
        AssertionError: If expected category metrics are not found.
    """
    summary = dict(wandb_run.summary)  # type: ignore[reportUnknownArgumentType]
    found_metrics: dict[str, typing.Any] = {}
    missing_categories: list[str] = []
    for category in expected_categories:
        expected_key = f"eval/{category}/{metric_suffix}"
        if expected_key in summary:
            found_metrics[expected_key] = summary[expected_key]
        else:
            matching_keys = [key for key in summary if category in key and metric_suffix in key]
            if matching_keys:
                for key in matching_keys:
                    found_metrics[key] = summary[key]
            else:
                missing_categories.append(category)
    assert not missing_categories, (
        f"missing expected category metrics for: {missing_categories}; "
        f"available keys containing '{metric_suffix}': {[k for k in summary if metric_suffix in k]}"
    )
    return found_metrics


def verify_best_model_loaded(
    trainer: transformers.Trainer,
    model: transformers.PreTrainedModel,
) -> pathlib.Path:
    """Verify that the best model checkpoint was loaded and weights match.

    This function checks that:
    1. The trainer recorded a best_model_checkpoint;
    2. The checkpoint directory exists and contains model weights;
    3. The current model weights match those in the best checkpoint.

    Args:
        trainer: The trainer object after training completed.
        model: The model to verify (should be trainer.model).

    Returns:
        Path to the best model checkpoint directory.

    Raises:
        AssertionError: If verification fails.
    """
    best_checkpoint = trainer.state.best_model_checkpoint
    assert best_checkpoint is not None, "best_model_checkpoint should be set when load_best_model_at_end=True"
    best_checkpoint_path = pathlib.Path(best_checkpoint)
    assert best_checkpoint_path.exists(), f"best checkpoint path should exist: {best_checkpoint_path}"
    # check for model weights file (safetensors preferred, fallback to pytorch)
    safetensors_path = best_checkpoint_path / "model.safetensors"
    pytorch_path = best_checkpoint_path / "pytorch_model.bin"
    adapter_path = best_checkpoint_path / "adapter_model.safetensors"
    if safetensors_path.exists():
        checkpoint_weights = safetensors.torch.load_file(str(safetensors_path))
    elif adapter_path.exists():
        checkpoint_weights = safetensors.torch.load_file(str(adapter_path))
    elif pytorch_path.exists():
        checkpoint_weights = torch.load(str(pytorch_path), map_location="cpu", weights_only=True)
    else:
        raise AssertionError(
            f"no model weights found in best checkpoint: {best_checkpoint_path}; "
            f"contents: {list(best_checkpoint_path.iterdir())}"
        )
    # compare a subset of weights to verify they match
    model_state = model.state_dict()
    mismatched_keys: list[str] = []
    compared_keys: list[str] = []
    for key in list(checkpoint_weights.keys())[:5]:  # compare first 5 keys for efficiency
        if key in model_state:
            compared_keys.append(key)
            checkpoint_tensor = checkpoint_weights[key]
            model_tensor = model_state[key].cpu()
            if checkpoint_tensor.shape != model_tensor.shape:
                mismatched_keys.append(f"{key} (shape mismatch)")
            elif not checkpoint_tensor.allclose(model_tensor, atol=1e-6):
                mismatched_keys.append(f"{key} (value mismatch)")
    assert compared_keys, "should have found at least some matching keys to compare"
    assert not mismatched_keys, f"model weights should match best checkpoint; mismatched: {mismatched_keys}"
    return best_checkpoint_path
