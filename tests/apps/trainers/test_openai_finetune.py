import json
import pathlib
import types

import pytest
import pytest_mock

import pyine.apps.trainers.openai_finetune
import pyine.apps.trainers.openai_finetune_configs
import pyine.data.datamodule
import pyine.data.traces.dataset_utils
import pyine.data.utils.splits
import pyine.evals.code_exec.configs
import pyine.organisms.datamodules.shortcuts_configs
import pyine.utils.openai
import tests.env_checks


def test_compute_estimated_train_token_count_handles_nested_messages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    train_path = tmp_path / "train.jsonl"
    train_path.write_text(
        "\n".join(
            [
                json.dumps([{"role": "user", "content": "hello world"}]),
                json.dumps(
                    [
                        {"role": "user", "content": "nested"},
                        {"role": "assistant", "content": "message"},
                    ],
                ),
            ],
        ),
        encoding="utf-8",
    )

    class _FakeTokenizer:
        def encode(self, content: str) -> list[int]:
            return list(range(len(content.split())))

    class _FakeConfig:
        def __init__(self) -> None:
            self.openai_finetuner_config = types.SimpleNamespace(params=types.SimpleNamespace(base_model="test"))

        def needs_answers_in_train_dataset(self) -> bool:
            return False

        def supports_system_prompt(self) -> bool:
            return True

    class _FakeDataModule:
        def get_openai_messages_dataset(
            self,
            subset_name: str,
            append_answer: bool = False,
            merge_system_with_user: bool = False,
        ) -> pathlib.Path:
            assert subset_name == "train"
            return train_path

    monkeypatch.setattr(
        pyine.apps.trainers.openai_finetune.pyine.utils.tokenizers,
        "get_openai_tokenizer",
        lambda **_kwargs: _FakeTokenizer(),
    )
    config = _FakeConfig()
    config.datamodule_config = types.SimpleNamespace(train_subset_names=["train"])
    dm = _FakeDataModule()
    count = pyine.apps.trainers.openai_finetune._compute_estimated_train_token_count(config, dm)
    # line 1: "hello world" = 2 words = 2 tokens
    # line 2: "nested" = 1 word = 1 token, "message" = 1 word = 1 token
    # total = 4 tokens
    assert count == 4


def test_train_streams_events_and_returns_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    train_calls: list[str] = []

    class _FakeFinetuner:
        def __init__(self) -> None:
            self.uploaded = []
            self.jobs = []

        def ensure_uploaded(self, path: str) -> str:
            self.uploaded.append(path)
            return f"file-{len(self.uploaded)}"

        def create_job(self, train_file: str, valid_file: str) -> str:
            self.jobs.append((train_file, valid_file))
            return "job-1"

        def stream_job_events(self, job_id: str) -> None:
            raise KeyboardInterrupt

        def wait_for_job(self, job_id: str) -> str:
            train_calls.append(job_id)
            return "ft-model-1"

    class _FakeDataModule:
        def get_openai_messages_dataset(
            self,
            subset_name: str,
            append_answer: bool,
            merge_system_with_user: bool,
        ) -> str:
            return str(tmp_path / f"{subset_name}.jsonl")

    fake_finetuner = _FakeFinetuner()
    config = types.SimpleNamespace(
        openai_finetuner_config=types.SimpleNamespace(
            params=types.SimpleNamespace(base_model="base"),
            instantiate=lambda client: fake_finetuner,
        ),
        datamodule_config=types.SimpleNamespace(
            train_subset_names=["train"],
            valid_subset_names=["valid"],
        ),
        needs_answers_in_train_dataset=lambda: True,
        supports_system_prompt=lambda: False,
        use_wandb_logging=False,
    )
    client = types.SimpleNamespace()
    monkeypatch.setattr(
        pyine.apps.trainers.openai_finetune,
        "_compute_estimated_train_token_count",
        lambda *_args, **_kwargs: 123,
    )
    model_name = pyine.apps.trainers.openai_finetune.train(
        client=client,
        datamodule=_FakeDataModule(),
        config=config,
        runtime=None,
    )
    assert model_name == "ft-model-1"
    assert fake_finetuner.uploaded == [
        str(tmp_path / "train.jsonl"),
        str(tmp_path / "valid.jsonl"),
    ]
    assert train_calls == ["job-1"]


@pytest.mark.asyncio
async def test_main_skip_fine_tuning_updates_wandb(
    monkeypatch: pytest.MonkeyPatch,
    mocker: pytest_mock.MockerFixture,
) -> None:
    evaluate_calls: list[dict[str, object]] = []
    provider_calls: list[dict[str, object]] = []
    wandb_updates: list[dict[str, object]] = []

    async def fake_evaluate_model(
        **kwargs: object,
    ) -> None:
        evaluate_calls.append(kwargs)

    def fake_prepare_datamodule(
        config: object,
        runtime: object,
    ) -> types.SimpleNamespace:
        dm_cfg = types.SimpleNamespace(get_prompt_chain=lambda model: model)
        dm_mock = mocker.Mock(spec=pyine.data.datamodule.ConversationDataModule)
        dm_mock.config = dm_cfg
        return dm_mock

    def fake_entrypoint_setup(
        **_kwargs: object,
    ) -> None:
        return None

    def fake_get_model_from_provider(
        **kwargs: object,
    ) -> str:
        provider_calls.append(kwargs)
        return "provider-model"

    class _FakeRun:
        def __init__(self) -> None:
            self.summary = types.SimpleNamespace(update=lambda data: wandb_updates.append(data))
            self._is_finished = True

        def finalize(self) -> None:
            pass

    class _FakeApi:
        def run(
            self,
            run_id: str,
        ) -> _FakeRun:
            assert run_id == "run-1"
            return _FakeRun()

    config = types.SimpleNamespace(
        openai_client_config=types.SimpleNamespace(
            instantiate=lambda: types.SimpleNamespace(chat=types.SimpleNamespace(completions="client"))
        ),
        openai_finetuner_config=types.SimpleNamespace(params=types.SimpleNamespace(base_model="base-model")),
        needs_answers_in_train_dataset=lambda: False,
        supports_system_prompt=lambda: True,
        use_wandb_logging=True,
    )
    runtime = types.SimpleNamespace(
        wandb_run=_FakeRun(),
        wandb_run_id="run-1",
        finalize=lambda: None,
    )
    monkeypatch.setattr(
        pyine.apps.trainers.openai_finetune.pyine.utils.reprod,
        "entrypoint_setup",
        fake_entrypoint_setup,
    )
    monkeypatch.setattr(
        pyine.apps.trainers.openai_finetune.pyine.apps.trainers.common,
        "prepare_datamodule",
        fake_prepare_datamodule,
    )
    monkeypatch.setattr(
        pyine.apps.trainers.openai_finetune.pyine.apps.trainers.common,
        "evaluate_model",
        fake_evaluate_model,
    )
    monkeypatch.setattr(
        pyine.apps.trainers.openai_finetune.pyine.utils.llm_providers,
        "get_model_from_provider",
        fake_get_model_from_provider,
    )
    monkeypatch.setattr(
        pyine.apps.trainers.openai_finetune.wandb,
        "Api",
        _FakeApi,
    )
    await pyine.apps.trainers.openai_finetune.main(
        config=config,
        runtime=runtime,
        skip_fine_tuning=True,
    )
    assert provider_calls[0]["model"] == "base-model"
    assert evaluate_calls and evaluate_calls[0]["model"] == "provider-model"
    assert wandb_updates[-1] == {"model_name": "base-model"}


@pytest.mark.slow
@pytest.mark.asyncio
@pytest.mark.skipif(
    tests.env_checks.TACO_TRACES_DATASET_MISSING,
    reason="TACO traces dataset is missing, cannot check sample generation",
)
@pytest.mark.skipif(
    tests.env_checks.TACO_TRACES_DATASET_SPLIT_MISSING,
    reason="TACO traces dataset split is missing, cannot check sample generation",
)
@pytest.mark.skipif(
    tests.env_checks.OPENAI_API_KEY_MISSING or tests.env_checks.NETWORK_UNAVAILABLE,
    reason="OpenAI API key or network not available, cannot run OpenAI-backed evaluation.",
)
async def test_code_exec_eval_base_model_with_skip_fine_tuning(
    tmp_path: pathlib.Path,
) -> None:
    """End-to-end exercise of main() using the real OpenAI API but skipping fine-tuning."""
    latest_dataset_path = pyine.data.traces.dataset_utils.get_latest_dataset_path("TACO")
    assert latest_dataset_path is not None and latest_dataset_path.exists()
    real_dm_config = pyine.organisms.datamodules.shortcuts_configs.ShortcutBiasDataModuleConfig(
        lmdb_paths=[latest_dataset_path],
        max_solution_count=2,
        split_file_path=pyine.data.utils.splits.get_dataset_split_file_path("TACO"),
    )
    openai_client_config = pyine.utils.openai.OpenAIClientConfig()
    finetuner_params_config = pyine.utils.openai.OpenAIFineTunerParamsConfig(
        base_model="gpt-4o-mini",
        method={"type": "supervised"},
        seed=0,
        suffix="pyine-pytest-openai-ft",
    )
    finetuner_cfg = pyine.utils.openai.OpenAIFineTunerConfig(params=finetuner_params_config)
    cfg = pyine.apps.trainers.openai_finetune_configs.OpenAIFineTuneAppMainConfig(
        datamodule_config=real_dm_config,
        openai_client_config=openai_client_config,
        openai_finetuner_config=finetuner_cfg,
        evals_config=pyine.evals.code_exec.configs.CodeExecEvalsConfig(),
    )
    await pyine.apps.trainers.openai_finetune.main(cfg, skip_fine_tuning=True)
