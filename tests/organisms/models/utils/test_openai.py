import pathlib
import time
import typing

import openai as openai_sdk
import pytest

import pyine.utils.reprod
from pyine.organisms.models.utils import openai as openai_utils
from tests.data.utils.env_checks import OPENAI_API_KEY_MISSING

fine_tune_base_model = "gpt-4.1-nano-2025-04-14"  # might need to be updated at some point...


class TestLocalDatasetIO:

    def test_get_local_file_directory(self) -> None:
        dir_path = openai_utils.get_local_file_directory()
        assert isinstance(dir_path, pathlib.Path)
        assert dir_path.exists() and dir_path.is_dir()

    def test_write_and_read_dataset_jsonl(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        dataset: list[dict[str, typing.Any] | list[dict[str, str]]] = [
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say hi"},
                    {"role": "assistant", "content": "Hi!"},
                ]
            },
            [
                {"role": "user", "content": "Echo: foo"},
                {"role": "assistant", "content": "foo"},
            ],
        ]
        out_path = tmp_path / "tiny_dataset.jsonl"
        openai_utils.write_dataset_to_jsonl(dataset, out_path)
        assert out_path.exists() and out_path.stat().st_size > 0

        messages_dataset = openai_utils.read_dataset_from_jsonl(out_path)
        assert isinstance(messages_dataset, list)
        assert len(messages_dataset) == 2
        assert isinstance(messages_dataset[0], list)
        assert messages_dataset[0][0]["role"] == "system"
        assert messages_dataset[0][1]["role"] == "user"
        assert messages_dataset[0][2]["role"] == "assistant"
        assert messages_dataset[1][0]["role"] == "user"
        assert messages_dataset[1][1]["role"] == "assistant"

    def test_estimate_token_count(self) -> None:
        count = openai_utils.estimate_token_count("Hello, world!", "gpt-4o-mini")
        assert isinstance(count, int)
        assert count > 0


@pytest.mark.skipif(OPENAI_API_KEY_MISSING, reason="OpenAI API key not available")
class TestOpenAIIntegration:

    @pytest.fixture(scope="class")
    def client(self) -> openai_sdk.OpenAI:
        return openai_sdk.OpenAI()

    @pytest.fixture(scope="class")
    def tuner(self, client: openai_sdk.OpenAI) -> openai_utils.OpenAIFineTuner:
        # use a distinctive suffix to enable easy cleanup and tracking
        suffix = f"pytest-openai-{int(time.time())}"
        config = openai_utils.OpenAIFineTunerParamsConfig(
            base_model=fine_tune_base_model,
            suffix=suffix,
            timeout_override=10,
            hyperparams=openai_utils.OpenAIFineTunerHyperparamsConfig(
                n_epochs=1,
            ),
        )
        return openai_utils.OpenAIFineTuner(client=client, config=config)

    @pytest.fixture
    def tiny_dataset_file(
        self,
        tmp_path: pathlib.Path,
    ):
        # keep this tiny and valid for fine-tune file upload shape
        dataset: list[dict[str, typing.Any]] = [
            {
                "messages": [
                    {"role": "system", "content": "You are a concise assistant."},
                    {"role": "user", "content": "Respond with OK."},
                    {"role": "assistant", "content": "OK."},
                ]
            }
        ]
        # ensure a unique filename to avoid collisions between runs
        filename = f"pytest-openai-tiny-finetune.{int(time.time() * 1000)}.jsonl"
        path = openai_utils.get_local_file_directory() / filename
        openai_utils.write_dataset_to_jsonl(dataset, path)
        yield path
        path.unlink()

    def test_chat_completion_simple(
        self,
        tuner: openai_utils.OpenAIFineTuner,
    ) -> None:
        reply = tuner.chat(
            model_id=fine_tune_base_model,
            system_prompt="You are a helpful assistant.",
            user_prompt="Please reply with the single word: ok",
            temperature=0.0,
            max_tokens=5,
        )
        assert isinstance(reply, str)

    def test_files_api_roundtrip(
        self,
        tuner: openai_utils.OpenAIFineTuner,
        tiny_dataset_file: pathlib.Path,
        tmp_path: pathlib.Path,
    ) -> None:
        remote_id: str | None = None
        try:
            # upload or reuse by name/size/hash
            remote_id = tuner.ensure_uploaded(tiny_dataset_file, confirm_by_hash=True)
            assert isinstance(remote_id, str) and len(remote_id) > 0
            # confirm detection of already uploaded file
            reused_id = tuner.is_file_already_uploaded(tiny_dataset_file, confirm_by_hash=True)
            assert reused_id == remote_id
            # quick list to ensure our file is visible among fine-tune files
            remotes = tuner.list_remote_files()
            assert any(getattr(r, "id", "") == remote_id for r in remotes)
            # verify remote size and hash match local
            remote_size = tuner.get_remote_file_size(remote_id)
            assert int(remote_size) == int(tiny_dataset_file.stat().st_size)
            local_hash = pyine.utils.reprod.compute_hash(tiny_dataset_file)
            remote_hash = tuner.get_remote_file_hash(remote_id)
            assert remote_hash == local_hash
            # download and verify integrity
            dest = tmp_path / f"downloaded-{tiny_dataset_file.name}"
            dl_path = tuner.download_file(remote_id, dest, overwrite=False)
            assert dl_path.exists()
            assert pyine.utils.reprod.compute_hash(dl_path) == local_hash
        finally:
            # cleanup remote file to avoid cluttering account storage
            if remote_id:
                try:
                    tuner.client.files.delete(remote_id)
                except Exception:
                    # best-effort cleanup; don't fail test teardown
                    pass

    def test_cleanup_finetuned_models_dry_run(
        self,
        tuner: openai_utils.OpenAIFineTuner,
    ) -> None:
        # match our test suffix pattern in case models exist from prior runs
        pattern = r".*pytest-openai-.*"
        openai_utils.cleanup_finetuned_models(
            client=tuner.client,
            pattern=pattern,
            max_age_days=0,
            dry_run=False,
        )
        remote_names = [r.filename for r in tuner.list_remote_files()]
        assert not any(name for name in remote_names if "pytest-openai-" in name)

    @pytest.mark.slow
    def test_finetune_nano_model_end_to_end(
        self,
        tuner: openai_utils.OpenAIFineTuner,
        tmp_path: pathlib.Path,
    ) -> None:
        ts = int(time.time() * 1000)
        train_path = openai_utils.get_local_file_directory() / f"pytest-openai-train-{ts}.jsonl"
        val_path = openai_utils.get_local_file_directory() / f"pytest-openai-val-{ts}.jsonl"
        train_dataset: list[dict[str, typing.Any]] = [
            {
                "messages": [
                    {"role": "system", "content": "You are brief."},
                    {"role": "user", "content": "Respond OK"},
                    {"role": "assistant", "content": "OK"},
                ]
            },
            {
                "messages": [
                    {"role": "system", "content": "You are brief."},
                    {"role": "user", "content": "Say YES"},
                    {"role": "assistant", "content": "YES"},
                ]
            },
            {
                "messages": [
                    {"role": "system", "content": "You are brief."},
                    {"role": "user", "content": "Say NO"},
                    {"role": "assistant", "content": "NO"},
                ]
            },
        ]
        val_dataset: list[dict[str, typing.Any]] = [
            {
                "messages": [
                    {"role": "system", "content": "You are brief."},
                    {"role": "user", "content": "Say OK"},
                    {"role": "assistant", "content": "OK"},
                ]
            },
        ]
        openai_utils.write_dataset_to_jsonl(train_dataset, train_path)
        openai_utils.write_dataset_to_jsonl(val_dataset, val_path)

        training_id: str | None = None
        validation_id: str | None = None
        job_id: str | None = None
        model_id: str = ""

        try:
            training_id = tuner.ensure_uploaded(train_path, confirm_by_hash=True)
            validation_id = tuner.ensure_uploaded(val_path, confirm_by_hash=True)
            job_id = tuner.create_job(
                training_file_id=training_id,
                validation_file_id=validation_id,
            )
            assert isinstance(job_id, str) and len(job_id) > 0
            model_id = tuner.wait_for_job(
                job_id=job_id,
                poll_seconds=10,
                timeout_seconds=15 * 60,
                cancel_on_timeout=True,
            )
            assert isinstance(model_id, str)
            if model_id:
                reply = tuner.chat(
                    model_id=model_id,
                    system_prompt="You are brief.",
                    user_prompt="Respond OK",
                    temperature=0.0,
                    max_tokens=4,
                )
                assert isinstance(reply, str) and len(reply) > 0
        finally:
            # cleanup: delete fine-tuned models with our suffix and remote files
            try:
                pattern = r".*pytest-openai-.*"
                openai_utils.cleanup_finetuned_models(
                    client=tuner.client,
                    pattern=pattern,
                    max_age_days=0,
                    dry_run=False,
                )
            except Exception:
                pass
            for fid in (training_id, validation_id):
                if fid:
                    try:
                        tuner.client.files.delete(fid)
                    except Exception:
                        pass
            for p in (train_path, val_path):
                try:
                    p.unlink()
                except Exception:
                    pass
