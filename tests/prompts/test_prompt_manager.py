import pathlib

import pyine.prompts.manager as prompt_manager
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
        mock_package_files.__truediv__.return_value = mock_prompt_file
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
