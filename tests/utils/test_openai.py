import contextlib
import functools
import itertools
import pathlib
import time
import typing

import langchain_core.messages
import openai as openai_sdk
import pytest

import pyine.utils.openai as openai_utils
import pyine.utils.reprod
import tests.env_checks

fine_tune_base_model = "gpt-4.1-nano-2025-04-14"  # might need to be updated at some point...
test_tag = "pyine-pytest-openai"  # will be useful to ignore if tests fail and start sending emails
fine_tune_timeout = 30  # wait 30 seconds, and skip test if still waiting after that
obj_expiration_days = 5  # if rerunning tests and encountering old test files, delete after this delay


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
        openai_utils.write_dataset_to_jsonl(dataset, out_path, enforce_openai_min_dataset_size=False)
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

    def test_write_dataset_enforces_minimum_size(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        out_path = tmp_path / "too_small.jsonl"
        with pytest.raises(ValueError):
            openai_utils.write_dataset_to_jsonl(
                dataset=[
                    {
                        "messages": [
                            {"role": "user", "content": "Hi"},
                            {"role": "assistant", "content": "Hello"},
                        ]
                    }
                ],
                path=out_path,
            )
        assert not out_path.exists()

    def test_write_and_read_objects_jsonl(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        objects = [
            {"id": 1, "name": "test1", "values": [1, 2, 3]},
            {"id": 2, "name": "test2", "nested": {"key": "value"}},
            [1, "two", 3.0],
        ]
        out_path = tmp_path / "objects.jsonl"
        openai_utils.write_objects_to_jsonl(objects, out_path)
        assert out_path.exists() and out_path.stat().st_size > 0
        loaded_objects = openai_utils.read_objects_from_jsonl(out_path)
        assert isinstance(loaded_objects, list)
        assert len(loaded_objects) == 3
        assert loaded_objects[0]["id"] == 1
        assert loaded_objects[0]["name"] == "test1"
        assert loaded_objects[0]["values"] == [1, 2, 3]
        assert loaded_objects[1]["nested"]["key"] == "value"
        assert loaded_objects[2] == [1, "two", 3.0]

    def test_estimate_token_count(self) -> None:
        count = openai_utils.estimate_token_count("Hello, world!", "gpt-4o-mini")
        assert isinstance(count, int)
        assert count > 0


class TestMessageConversion:
    def test_convert_dict_messages_and_tool_calls(self) -> None:
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "content": "Let me call a function.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "foo", "arguments": '{"x": 1}'},
                    }
                ],
            },
            {
                "role": "tool",
                "name": "foo",
                "tool_call_id": "call_1",
                "content": '{"result": 42}',
            },
        ]
        out = openai_utils.convert_messages_to_openai(messages)
        assert isinstance(out, list)
        assert [m["role"] for m in out] == ["system", "user", "assistant", "tool"]
        assert out[0]["content"] == "You are helpful."
        assert out[2]["content"] == "Let me call a function."
        assert isinstance(out[2].get("tool_calls"), list)
        assert out[2]["tool_calls"][0]["function"]["name"] == "foo"
        assert out[3]["tool_call_id"] == "call_1"
        assert out[3]["name"] == "foo"

    def test_convert_multimodal_and_generic_object(self) -> None:
        # multimodal: content list should be preserved as a list of dicts
        multimodal = [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
        ]
        out = openai_utils.convert_messages_to_openai([{"role": "user", "content": multimodal}])
        assert isinstance(out, list) and len(out) == 1
        assert isinstance(out[0]["content"], list)
        assert out[0]["content"][0]["type"] == "text"
        assert out[0]["content"][0]["text"] == "hello"

        # generic object with role/content attributes should be mapped correctly
        class DummyMsg:
            def __init__(self, role: str, content: str) -> None:
                self.role = role
                self.content = content

        out2 = openai_utils.convert_messages_to_openai([DummyMsg("assistant", "ok")])
        assert out2[0]["role"] == "assistant"
        assert out2[0]["content"] == "ok"

    def test_convert_langchain_message_types(self) -> None:
        system_msg = langchain_core.messages.SystemMessage(content="sys")
        human_msg = langchain_core.messages.HumanMessage(content="hi")
        tool_call = langchain_core.messages.tool.ToolCall(
            id="call-1",
            name="fn",
            args={"x": 1},
        )
        ai_msg = langchain_core.messages.AIMessage(
            content="ok",
            tool_calls=[tool_call],
        )
        tool_msg = langchain_core.messages.ToolMessage(
            content="result",
            tool_call_id="call-1",
            name="tool",
        )
        func_msg = langchain_core.messages.FunctionMessage(
            name="legacy",
            content="payload",
        )
        chat_msg = langchain_core.messages.ChatMessage(role="user", content="again")

        out = openai_utils.convert_messages_to_openai([system_msg, human_msg, ai_msg, tool_msg, func_msg, chat_msg])
        assert [entry["role"] for entry in out] == [
            "system",
            "user",
            "assistant",
            "tool",
            "function",
            "user",
        ]
        tool_calls = out[2]["tool_calls"]
        assert isinstance(tool_calls, list) and tool_calls[0]["function"]["name"] == "fn"
        assert tool_calls[0]["function"]["arguments"] == '{"x":1}'
        assert out[3]["tool_call_id"] == "call-1"
        assert out[4]["name"] == "legacy"
        assert out[5]["content"] == "again"


@pytest.fixture(scope="class")
def client() -> openai_sdk.OpenAI:
    return openai_sdk.OpenAI(max_retries=0)  # disable SDK retries; let backoff handle retries


@pytest.mark.integration
@pytest.mark.openai
@pytest.mark.skipif(
    tests.env_checks.OPENAI_API_KEY_MISSING or tests.env_checks.NETWORK_UNAVAILABLE,
    reason="OpenAI API key or network not available",
)
class TestOpenAIIntegration:
    @pytest.fixture(scope="class")
    def tuner(self, client: openai_sdk.OpenAI) -> openai_utils.OpenAIFineTuner:
        # use a distinctive suffix to enable easy cleanup and tracking
        suffix = f"{test_tag}-{int(time.time())}"
        config = openai_utils.OpenAIFineTunerParamsConfig(
            base_model=fine_tune_base_model,
            method={"type": "supervised"},
            suffix=suffix,
            timeout_override=10,
            hyperparams=openai_utils.OpenAIFineTunerHyperparamsConfig(
                n_epochs=1,
            ),
            file_upload_params={
                "expires_after": {
                    "seconds": 60 * 60 * 24 * 10,  # 10 days expiration for all uploaded files once created
                    "anchor": "created_at",
                },
            },
        )
        return openai_utils.OpenAIFineTuner(client=client, config=config)

    @pytest.fixture
    def tiny_dataset_file(
        self,
        tmp_path: pathlib.Path,
    ) -> typing.Iterator[pathlib.Path]:
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
        filename = f"{test_tag}-tiny-finetune.{int(time.time() * 1000)}.jsonl"
        path = openai_utils.get_local_file_directory() / filename
        openai_utils.write_dataset_to_jsonl(dataset, path, enforce_openai_min_dataset_size=False)
        yield path
        path.unlink()

    @pytest.mark.slow
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

    @pytest.mark.slow
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
            assert any(r.id == remote_id for r in remotes)
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
                with contextlib.suppress(Exception):
                    tuner.client.files.delete(remote_id)

    @pytest.mark.slow
    def test_finetune_nano_model_end_to_end(
        self,
        tuner: openai_utils.OpenAIFineTuner,
        tmp_path: pathlib.Path,
    ) -> None:
        train_path = openai_utils.get_local_file_directory() / f"{test_tag}-train.jsonl"
        sys_prompt = "You are brief."
        keywords = ["YES", "SURE", "OK", "NO", "WHAT"]
        requests = ["Respond", "Say", "Repeat", "Answer", "Output"]
        user_prompts = itertools.product(keywords, requests)
        dataset: list[dict[str, typing.Any]] = [
            {
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"{request} {keyword}"},
                    {"role": "assistant", "content": keyword},
                ]
            }
            for keyword, request in user_prompts
        ]
        openai_utils.write_dataset_to_jsonl(dataset, train_path)
        success, training_id = False, None
        try:
            training_id = tuner.ensure_uploaded(train_path, confirm_by_hash=True)
            job_id = tuner.create_job(
                training_file_id=training_id,
                validation_file_id=None,
            )
            assert isinstance(job_id, str) and len(job_id) > 0
            finetuning_jobs = tuner.list_finetuning_jobs()
            assert any(job.id == job_id for job in finetuning_jobs)
            model_id = tuner.wait_for_job(
                job_id=job_id,
                poll_seconds=10,
                timeout_seconds=fine_tune_timeout,
                cancel_on_timeout=False,
            )
            assert isinstance(model_id, str)
            if model_id:
                reply = tuner.chat(
                    model_id=model_id,
                    system_prompt=sys_prompt,
                    user_prompt="Respond OK",
                    temperature=0.0,
                    max_tokens=4,
                )
                assert isinstance(reply, str) and len(reply) > 0
                tuner.client.files.delete(training_id)
                success = True
        except TimeoutError:
            pytest.skip("timeout waiting for fine-tune job to complete")
        finally:
            train_path.unlink()
        if not success:
            pytest.fail("failed to fine-tune model")

    @pytest.mark.slow
    def test_finetuned_nano_model_inference(
        self,
        tuner: openai_utils.OpenAIFineTuner,
    ) -> None:
        matched_models = tuner.list_remote_models(pattern=f"ft:.*{test_tag}.*")
        if not matched_models:
            pytest.skip("No matching fine-tuned test models found")
        model_id = matched_models[0].id  # use first match
        reply = tuner.chat(
            model_id=model_id,
            system_prompt="You are brief.",
            user_prompt="Respond YES",
            temperature=0.0,
            max_tokens=4,
        )
        assert isinstance(reply, str)
        assert len(reply) > 0


@pytest.mark.integration
@pytest.mark.openai
@pytest.mark.skipif(
    tests.env_checks.OPENAI_API_KEY_MISSING or tests.env_checks.NETWORK_UNAVAILABLE,
    reason="OpenAI API key or network not available",
)
class TestCleanups:
    @pytest.mark.slow
    def test_cleanup_files(self, client: openai_sdk.OpenAI) -> None:
        cleanup = functools.partial(
            openai_utils.cleanup_files,
            client=client,
            pattern=f".*{test_tag}.*",
            max_age_days=obj_expiration_days,
        )
        cleanup_targets = cleanup(dry_run=True)
        if cleanup_targets:
            cleaned_up = cleanup(dry_run=False)
            assert cleaned_up == cleanup_targets
            new_targets = cleanup(dry_run=True)
            assert not new_targets

    @pytest.mark.slow
    def test_cleanup_finetuned_models(self, client: openai_sdk.OpenAI) -> None:
        cleanup = functools.partial(
            openai_utils.cleanup_finetuned_models,
            client=client,
            pattern=f"ft:.*{test_tag}.*",
            max_age_days=obj_expiration_days,
        )
        cleanup_targets = cleanup(dry_run=True)
        if cleanup_targets:
            try:
                cleaned_up = cleanup(dry_run=False)
                assert cleaned_up == cleanup_targets
                new_targets = cleanup(dry_run=True)
                assert not new_targets
            except openai_sdk.PermissionDeniedError as e:
                if "You have insufficient permissions for this operation" in str(e):
                    # current user does not have ownership rights required for cleanup
                    pass  # ...just skip the rest of the test, counts as success
                else:
                    raise e
