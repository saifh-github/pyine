import pathlib

import pytest

import pyine.prompts.configs.hints.docs as hints_docs
import pyine.prompts.configs.issues.docs as issues_docs
import pyine.prompts.manager as prompt_manager
import pyine.prompts.types as prompt_types
import pyine.prompts.utils as prompt_utils


class TestPromptManager:

    def test_fake_manager_init(self, mocker):
        manager = prompt_manager.PromptManager(package_name="custom.package", prompts_subdir="custom_dir")
        assert manager.package_name == "custom.package"
        assert manager.prompts_subdir == "custom_dir"
        path = manager._get_prompt_file_path("test_prompt")
        expected = pathlib.Path("custom_dir") / "test_prompt.yaml"
        assert path == expected
        mock_files = mocker.patch("importlib.resources.files")
        mock_package_files = mocker.MagicMock()
        mock_files.return_value = mock_package_files
        mock_prompt_file = mocker.MagicMock()
        mock_prompt_file.is_file.return_value = True
        mock_package_files.joinpath.return_value = mock_prompt_file
        mock_versioned_config = mocker.MagicMock()
        mock_versioned_config.versions = {"v1.0.0": "config1", "v2.0.0": "config2"}
        mocker.patch.object(prompt_utils.VersionedPromptConfig, "from_yaml", return_value=mock_versioned_config)
        result = manager.list_prompt_versions("test_prompt")
        assert result == ["v1.0.0", "v2.0.0"]

    def test_real_manager_init(self):
        manager = prompt_manager.get_framework_prompt_manager()
        assert manager.package_name == "pyine.prompts"
        assert manager.prompts_subdir == "templates"
        path = manager._get_prompt_file_path("test_prompt")
        expected = pathlib.Path("templates") / "test_prompt.yaml"
        assert path == expected
        results = manager.list_prompts()
        assert len(results) > 0
        for result in results:
            assert isinstance(result, str)
            assert not result.endswith(".yaml")
            prompt_config = manager.get_prompt_config(result)
            assert prompt_config.metadata.name == result

    def test_prompt_build_config_forwards_block_variables_to_template(self, monkeypatch):
        captured_kwargs: dict[str, dict[str, str] | None] = {}

        def fake_get_prompt_template(**kwargs):
            captured_kwargs.update(kwargs)
            return "template"

        monkeypatch.setattr(prompt_manager, "get_prompt_template", fake_get_prompt_template)
        build_config = prompt_types.PromptBuildConfig(
            prompt_name="dummy",
            role_variables={"role": "value"},
            context_variables={"context": "value"},
            examples_block_variables={"examples": "value"},
        )
        result = build_config.get_template()
        assert result == "template"
        assert captured_kwargs["role_variables"] == {"role": "value"}
        assert captured_kwargs["context_variables"] == {"context": "value"}
        assert captured_kwargs["examples_block_variables"] == {"examples": "value"}

    def test_prompt_build_config_forwards_block_variables_to_chain(self, monkeypatch):
        captured_kwargs: dict[str, dict[str, str] | None] = {}

        def fake_get_prompt_chain(**kwargs):
            captured_kwargs.update(kwargs)
            return "chain"

        monkeypatch.setattr(prompt_manager, "get_prompt_chain", fake_get_prompt_chain)
        build_config = prompt_types.PromptBuildConfig(
            prompt_name="dummy",
            role_variables={"role": "value"},
            context_variables={"context": "value"},
            examples_block_variables={"examples": "value"},
        )
        result = build_config.get_chain(model="model", runnable_name="run")
        assert result == "chain"
        assert captured_kwargs["role_variables"] == {"role": "value"}
        assert captured_kwargs["context_variables"] == {"context": "value"}
        assert captured_kwargs["examples_block_variables"] == {"examples": "value"}

    def test_prompt_aliases_reuse_hints_docs(self):
        manager = prompt_manager.get_framework_prompt_manager()
        prompts = manager.list_prompts()
        assert "issues/docs" in prompts
        hints_config = manager.get_prompt_config("hints/docs")
        issues_config = manager.get_prompt_config("issues/docs")
        assert issues_config.metadata.name == "issues/docs"
        assert hints_config.metadata.name == "hints/docs"
        assert issues_config.question.template == hints_config.question.template
        assert issues_config is not hints_config


def test_issues_docs_delegates_to_hints(monkeypatch: pytest.MonkeyPatch):
    captured_kwargs: dict[str, object] = {}
    marker = object()

    def fake_get_prompt_template(**kwargs):
        captured_kwargs.update(kwargs)
        return marker

    monkeypatch.setattr(hints_docs, "get_prompt_template", fake_get_prompt_template)
    result = issues_docs.get_prompt_template(
        version="v1",
        use_chat_template=True,
        include_examples=False,
        target_examples=[1, 2],
        partial_vars={"pv": 1},
        role_variables={"role": "analyst"},
        context_variables={"ctx": "value"},
        examples_block_variables={"examples": "value"},
    )
    assert result is marker
    assert captured_kwargs == {
        "version": "v1",
        "use_chat_template": True,
        "include_examples": False,
        "target_examples": [1, 2],
        "partial_vars": {"pv": 1},
        "role_variables": {"role": "analyst"},
        "context_variables": {"ctx": "value"},
        "examples_block_variables": {"examples": "value"},
    }
