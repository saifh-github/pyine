import typing

import datasets as hf_datasets
import pytest
import torch
import transformers

import pyine.data.traces.dataset_utils
import pyine.data.utils.splits
import pyine.organisms.datamodules.shortcuts_configs
import pyine.organisms.datamodules.utils.samples
import pyine.organisms.datamodules.utils.transforms
import pyine.utils.reprod
import pyine.utils.transformers
import tests.env_checks


def _assert_non_leaking_assignments(
    metadata: pyine.data.traces.dataset_utils.TraceDatasetMetadata,
) -> None:
    traces_to_subsets: dict[str, str] = {}
    problems_to_subsets: dict[str, str] = {}
    for subset_name, subset_traces in metadata.subset_traces.items():
        for trace in subset_traces:
            assert trace.identifier not in traces_to_subsets
            if trace.problem_id in problems_to_subsets:
                assert problems_to_subsets[trace.problem_id] == subset_name
            else:
                problems_to_subsets[trace.problem_id] = subset_name
            traces_to_subsets[trace.identifier] = subset_name


@pytest.fixture
def shortcuts_dm_config() -> pyine.organisms.datamodules.shortcuts_configs.ShortcutBiasDataModuleConfig:
    return pyine.organisms.datamodules.shortcuts_configs.ShortcutBiasDataModuleConfig(
        lmdb_paths=[
            pyine.data.traces.dataset_utils.get_latest_dataset_path("TACO"),
        ],
        dataparser_config_overrides={
            "train": {
                "transform_config": pyine.organisms.datamodules.utils.samples.SampleTransformConfig(
                    transform_strategy="hybrid",
                    output_type_prob_map={
                        "program_output": 0.5,
                        "frame_variables": 0.1,
                        "function_return": 0.4,
                    },
                ),
            },  # other subsets will default to never producing partial samples
        },
        dataloader_config_overrides={
            "train": {
                "batch_size": 16,
                "shuffle": True,
            },
        },
        max_solution_count=100,
        split_file_path=pyine.data.utils.splits.get_dataset_split_file_path("TACO"),
    )


@pytest.mark.slow
@pytest.mark.skipif(
    tests.env_checks.TACO_TRACES_DATASET_MISSING,
    reason="TACO traces dataset is missing, cannot check sample generation",
)
@pytest.mark.skipif(
    tests.env_checks.TACO_TRACES_DATASET_SPLIT_MISSING,
    reason="TACO traces dataset split is missing, cannot check sample generation",
)
@pytest.mark.skipif(
    tests.env_checks.HF_ACCESS_TOKEN_MISSING,
    reason="Hugging Face access token is missing, cannot check hf dataset loading",
)
def test_shortcuts_datamodule_integration(
    shortcuts_dm_config: pyine.organisms.datamodules.shortcuts_configs.ShortcutBiasDataModuleConfig,
) -> None:
    pyine.utils.reprod.load_dotenv()
    dm = shortcuts_dm_config.instantiate_datamodule(verbose=True)
    if dm._is_metadata_prepared():
        dm._clear_prepared_metadata()
    dm.prepare_data()
    dm.setup()
    assert dm._is_metadata_prepared()
    metadata = dm._load_prepared_metadata()
    assert isinstance(metadata, pyine.data.traces.dataset_utils.TraceDatasetMetadata)
    _assert_non_leaking_assignments(metadata)
    dataloader = dm.train_dataloader()
    assert dataloader.batch_size == 16
    batch = next(iter(dataloader))
    assert len(batch.code) == 16
    print("batch loading works")
    parser = dm.get_parser("train")
    sample = parser[0]
    sample_msgs_transf_fn = pyine.organisms.datamodules.utils.transforms.create_sample_transform(
        use_chat_template=True,
        append_answer=True,
        prompt_name="code_execution",
    )
    transformed_sample_msgs = sample_msgs_transf_fn(sample)
    assert isinstance(transformed_sample_msgs, list)
    assert all(hasattr(m, "type") and hasattr(m, "content") for m in transformed_sample_msgs)
    print("sample transform works")
    hf_msgs_ds = dm.get_hf_messages_dataset(subset_name="train")
    assert isinstance(hf_msgs_ds, hf_datasets.Dataset)
    assert len(hf_msgs_ds) == len(parser)  # noqa
    hf_msgs_sample = hf_msgs_ds[0]
    assert isinstance(hf_msgs_sample, dict)
    assert "messages" in hf_msgs_sample and isinstance(hf_msgs_sample["messages"], list)
    print("hf dataset works")
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, use_fast=True)
    hf_batch_dataset = pyine.utils.transformers.apply_model_template_to_messages(
        hf_messages_ds=hf_msgs_ds,
        tokenizer=tokenizer,
        apply_chat_template_kwargs={
            "tokenize": False,
            "add_generation_prompt": False,
        },
    )
    assert isinstance(hf_batch_dataset, hf_datasets.Dataset)
    assert len(hf_batch_dataset) == len(parser)  # noqa
    print("hf dataset batching and tokenization works")
    openai_dataset_path = dm.get_openai_messages_dataset("train")
    assert openai_dataset_path.exists() and openai_dataset_path.is_file()
    print("open dataset writing works")
    dm.teardown()


@pytest.mark.slow
@pytest.mark.skipif(
    tests.env_checks.TACO_TRACES_DATASET_MISSING,
    reason="TACO traces dataset is missing, cannot check sample generation",
)
@pytest.mark.skipif(
    tests.env_checks.TACO_TRACES_DATASET_SPLIT_MISSING,
    reason="TACO traces dataset split is missing, cannot check sample generation",
)
@pytest.mark.skipif(
    tests.env_checks.HF_ACCESS_TOKEN_MISSING,
    reason="Hugging Face access token is missing, cannot check hf dataset loading",
)
def test_shortcuts_datamodule_predefined_split(
    shortcuts_dm_config: pyine.organisms.datamodules.shortcuts_configs.ShortcutBiasDataModuleConfig,
) -> None:
    pyine.utils.reprod.load_dotenv()
    dm = shortcuts_dm_config.instantiate_datamodule(verbose=True)
    if dm._is_metadata_prepared():
        dm._clear_prepared_metadata()
    dm.prepare_data()
    dm.setup()
    assert dm._is_metadata_prepared()
    metadata = dm._load_prepared_metadata()
    assert isinstance(metadata, pyine.data.traces.dataset_utils.TraceDatasetMetadata)
    _assert_non_leaking_assignments(metadata)
    train_parser = dm.get_parser("train")
    train_sample, valid_sample = None, None
    if len(train_parser) > 0:  # noqa
        train_sample = train_parser[0]
    valid_parser = dm.get_parser("valid")
    if len(valid_parser) > 0:  # noqa
        valid_sample = valid_parser[0]
    if train_sample is not None or valid_sample is not None:
        assert train_sample != valid_sample
    dm.teardown()


@pytest.mark.slow
@pytest.mark.skipif(
    tests.env_checks.TACO_TRACES_DATASET_MISSING,
    reason="TACO traces dataset is missing, cannot check sample generation",
)
@pytest.mark.skipif(
    tests.env_checks.TACO_TRACES_DATASET_SPLIT_MISSING,
    reason="TACO traces dataset split is missing, cannot check sample generation",
)
@pytest.mark.skipif(
    tests.env_checks.HF_ACCESS_TOKEN_MISSING,
    reason="Hugging Face access token is missing, cannot check hf dataset loading",
)
def test_shortcuts_datamodule_examples_round_trip(
    shortcuts_dm_config: pyine.organisms.datamodules.shortcuts_configs.ShortcutBiasDataModuleConfig,
) -> None:
    pyine.utils.reprod.load_dotenv()
    dm = shortcuts_dm_config.instantiate_datamodule(verbose=True)
    if dm._is_metadata_prepared():
        dm._clear_prepared_metadata()
    dm.prepare_data()
    dm.setup()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_max_seq_len = 2048  # pretend the model is actually limited to this
    hf_msgs_ds = dm.get_hf_messages_dataset(
        subset_name="train",
        keep_original_data=True,
    )
    max_samples = min(len(hf_msgs_ds), 10)
    if max_samples == 0:
        pytest.skip("dataset contains no samples to verify")
    hf_msgs_ds = hf_msgs_ds.select(list(range(max_samples)))
    cached_samples: dict[str, dict[str, typing.Any]] = {}
    expected_sample_keys = ["messages", "identifier", "code", "inputs", "expected_output"]
    for sample in hf_msgs_ds:
        assert isinstance(sample, dict)
        assert all(k in sample for k in expected_sample_keys)
        identifier = sample["identifier"]
        assert isinstance(identifier, str) and identifier not in cached_samples
        cached_samples[identifier] = sample
    examples_ds = pyine.utils.transformers.prepare_examples_from_conversations(
        convo_ds=hf_msgs_ds,
        tokenizer=tokenizer,
        max_seq_len=model_max_seq_len,
        num_proc=2,
        keep_extra_fields=True,
    )
    assert len(examples_ds) >= max_samples, "fewer examples than samples??"
    collator = pyine.utils.transformers.PaddingCollatorWithPromptMask(
        tokenizer=tokenizer,
        max_length=model_max_seq_len,
        keep_extra_fields=True,
    )
    collator_batch_size = min(len(examples_ds), 4)
    data_loader = torch.utils.data.DataLoader(
        examples_ds,
        batch_size=collator_batch_size,
        collate_fn=collator,
        shuffle=False,
        drop_last=False,
    )
    assert len(data_loader) <= len(examples_ds), "more batches than examples??"
    expected_batch_keys = ["input_ids", "attention_mask", "labels"]
    expected_sample_keys = ["identifier", "code", "inputs", "expected_output"]  # dropped messages
    for example_batch in data_loader:
        assert "identifier" in example_batch and isinstance(example_batch["identifier"], list)
        batch_size = len(example_batch["identifier"])
        assert all(k in example_batch for k in expected_sample_keys)
        for k in expected_sample_keys:
            assert isinstance(example_batch[k], list) and len(example_batch[k]) == batch_size
        assert all(k in example_batch for k in expected_batch_keys)
        for k in expected_batch_keys:
            assert isinstance(example_batch[k], torch.Tensor)
            assert example_batch[k].dtype == torch.long and example_batch[k].ndim == 2
            assert example_batch[k].shape[0] == batch_size
            assert example_batch[k].shape[1] <= model_max_seq_len
        for iter_idx, (ids, attn, lbls) in enumerate(
            zip(
                example_batch["input_ids"],
                example_batch["attention_mask"],
                example_batch["labels"],
                strict=False,
            )
        ):
            identifier = example_batch["identifier"][iter_idx]
            assert identifier in cached_samples
            matched_sample = cached_samples[identifier]
            for sample_key in expected_sample_keys:
                assert matched_sample[sample_key] == example_batch[sample_key][iter_idx]
            non_ignore_labels_mask = lbls != -100
            output_ids = ids[non_ignore_labels_mask]
            assert len(output_ids) > 0
            output_txt = tokenizer.decode(output_ids, skip_special_tokens=True)
            assert output_txt == example_batch["expected_output"][iter_idx].strip()
            non_ignore_attn_mask = attn != 0
            padding_mask = ~non_ignore_attn_mask
            # Note: when always_pad_to_max_length=False, some examples may have no padding
            if padding_mask.any():
                assert torch.unique(ids[padding_mask]).tolist() == [tokenizer.pad_token_id], "unexpected padding tokens"
            inputs_mask = non_ignore_attn_mask & ~non_ignore_labels_mask
            assert inputs_mask.sum().item() > 0, "no input tokens in prepared tensors??"
            prompt_ids = ids[inputs_mask]
            prompt_txt = tokenizer.decode(prompt_ids, skip_special_tokens=True)
            assert "You are an expert at interpreting and executing Python 3 code" in prompt_txt, "prompt text missing"
            round_tripped_code = tokenizer.decode(  # to ensure compatibility with tokenizer quirks
                tokenizer.encode(example_batch["code"][iter_idx].strip()),
                skip_special_tokens=True,
            )
            assert round_tripped_code in prompt_txt, "code snippet missing from prompt text?"
            round_tripped_inputs = tokenizer.decode(  # to ensure compatibility with tokenizer quirks
                tokenizer.encode(example_batch["inputs"][iter_idx].strip()),
                skip_special_tokens=True,
            )
            assert round_tripped_inputs in prompt_txt, "inputs missing from prompt text?"
    dm.teardown()
