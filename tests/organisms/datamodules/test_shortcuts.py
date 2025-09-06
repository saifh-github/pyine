import datasets as hf_datasets
import pytest
import transformers

import pyine.data.traces.dataset_utils
import pyine.data.utils.splits
import pyine.organisms.datamodules.shortcuts
import pyine.organisms.datamodules.utils.samples
import pyine.organisms.datamodules.utils.transforms
import pyine.utils.reprod
import tests.data.utils.env_checks


def _assert_non_leaking_assignments(metadata):
    traces_to_subsets = dict()
    problems_to_subsets = dict()
    for subset_type, subset_traces in metadata.subset_traces.items():
        for trace in subset_traces:
            assert trace.identifier not in traces_to_subsets
            parent = trace.get_parent_problem_id()
            if parent in problems_to_subsets:
                assert problems_to_subsets[parent] == subset_type
            else:
                problems_to_subsets[parent] = subset_type
            traces_to_subsets[trace.identifier] = subset_type


@pytest.mark.slow
@pytest.mark.skipif(
    tests.data.utils.env_checks.TACO_TRACES_DATASET_MISSING,
    reason="TACO traces dataset is missing, cannot check sample generation",
)
@pytest.mark.skipif(
    tests.data.utils.env_checks.TACO_TRACES_DATASET_SPLIT_MISSING,
    reason="TACO traces dataset split is missing, cannot check sample generation",
)
@pytest.mark.skipif(
    tests.data.utils.env_checks.HF_ACCESS_TOKEN_MISSING,
    reason="Hugging Face access token is missing, cannot check hf dataset loading",
)
def test_shortcuts_datamodule_integration():
    pyine.utils.reprod.load_dotenv()
    config = pyine.organisms.datamodules.shortcuts.ShortcutBiasDataModuleConfig(
        lmdb_paths=[
            pyine.data.traces.dataset_utils.get_latest_dataset_path("TACO"),
        ],
        dataloader_config_overrides=dict(
            train=dict(
                batch_size=16,
                shuffle=True,
            ),
        ),
        max_trace_count=1000,
        split_file_path=pyine.data.utils.splits.get_dataset_split_result("TACO"),
    )
    dm = config.instantiate_datamodule(verbose=True)
    if dm._is_metadata_prepared():
        dm._clear_prepared_metadata()
    dm.prepare_data()
    dm.setup()
    assert dm._is_metadata_prepared()
    metadata = dm._load_prepared_metadata()
    assert isinstance(metadata, pyine.organisms.datamodules.utils.samples.TraceDatasetMetadata)
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
    assert all([hasattr(m, "type") and hasattr(m, "content") for m in transformed_sample_msgs])
    print("sample transform works")
    hf_msgs_dataset = dm.get_hf_messages_dataset(subset_type="train")
    assert isinstance(hf_msgs_dataset, hf_datasets.Dataset)
    assert len(hf_msgs_dataset) == len(parser)  # noqa
    hf_msgs_sample = hf_msgs_dataset[0]
    assert isinstance(hf_msgs_sample, dict)
    assert "messages" in hf_msgs_sample and isinstance(hf_msgs_sample["messages"], list)
    print("hf dataset works")
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, use_fast=True)
    hf_batch_dataset = pyine.organisms.datamodules.utils.transforms.apply_model_template_to_messages(
        hf_messages_dataset=hf_msgs_dataset,
        tokenizer=tokenizer,
        apply_chat_template_kwargs=dict(
            tokenize=False,
            add_generation_prompt=False,
        ),
        batching_map_kwargs=dict(
            keep_in_memory=True,
        ),
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
    tests.data.utils.env_checks.TACO_TRACES_DATASET_MISSING,
    reason="TACO traces dataset is missing, cannot check sample generation",
)
@pytest.mark.skipif(
    tests.data.utils.env_checks.TACO_TRACES_DATASET_SPLIT_MISSING,
    reason="TACO traces dataset split is missing, cannot check sample generation",
)
@pytest.mark.skipif(
    tests.data.utils.env_checks.HF_ACCESS_TOKEN_MISSING,
    reason="Hugging Face access token is missing, cannot check hf dataset loading",
)
def test_shortcuts_datamodule_predefined_split():
    pyine.utils.reprod.load_dotenv()
    config = pyine.organisms.datamodules.shortcuts.ShortcutBiasDataModuleConfig(
        lmdb_paths=[
            pyine.data.traces.dataset_utils.get_latest_dataset_path("TACO"),
        ],
        max_trace_count=1000,
        split_file_path=pyine.data.utils.splits.get_dataset_split_result("TACO"),
    )
    dm = config.instantiate_datamodule(verbose=True)
    if dm._is_metadata_prepared():
        dm._clear_prepared_metadata()
    dm.prepare_data()
    dm.setup()
    assert dm._is_metadata_prepared()
    metadata = dm._load_prepared_metadata()
    assert isinstance(metadata, pyine.organisms.datamodules.utils.samples.TraceDatasetMetadata)
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
