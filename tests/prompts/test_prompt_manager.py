import pathlib
import tempfile

import pytest
import yaml

import pyine.prompts.manager as prompt_manager
import pyine.prompts.utils as prompt_utils


class TestPromptManager:

    @pytest.fixture
    def manager(self):
        mgr = prompt_manager.PromptManager(package_name="test.package", prompts_subdir="test_templates")
        path = mgr._get_prompt_path("test_prompt")
        expected = pathlib.Path("test_templates") / "test_prompt.yaml"
        assert path == expected
        return mgr

    def test_manager_init(self):
        manager = prompt_manager.PromptManager()
        assert manager.package_name == "pyine.prompts"
        assert manager.prompts_subdir == "templates"
        assert isinstance(manager._cache, dict)
        manager = prompt_manager.PromptManager(package_name="custom.package", prompts_subdir="custom_dir")
        assert manager.package_name == "custom.package"
        assert manager.prompts_subdir == "custom_dir"

    def test_load_invalid_prompt_config(self, mocker, manager: prompt_manager.PromptManager):
        mock_files = mocker.patch("importlib.resources.files")
        mock_package_files = mocker.MagicMock()
        mock_files.return_value = mock_package_files
        mock_prompt_file = mocker.MagicMock()
        mock_prompt_file.is_file.return_value = False
        mock_package_files.__truediv__.return_value = mock_prompt_file
        with pytest.raises(FileNotFoundError):
            manager._load_prompt_config("nonexistent")
        mock_prompt_file.is_file.return_value = True
        mocker.patch.object(prompt_utils.VersionedPromptConfig, "from_yaml", side_effect=yaml.YAMLError("Invalid YAML"))
        with pytest.raises(ValueError, match="Invalid YAML"):
            manager._load_prompt_config("invalid_yaml")

    def test_list_prompts_directory_exists(self, mocker, manager: prompt_manager.PromptManager):
        mock_files = mocker.patch("importlib.resources.files")
        mock_package_files = mocker.MagicMock()
        mock_files.return_value = mock_package_files
        mock_prompts_dir = mocker.MagicMock()
        mock_prompts_dir.is_dir.return_value = True
        mock_package_files.__truediv__.return_value = mock_prompts_dir

        mock_file1 = mocker.MagicMock()
        mock_file1.is_file.return_value = True
        mock_file1.name = "prompt1.yaml"
        mock_file2 = mocker.MagicMock()
        mock_file2.is_file.return_value = True
        mock_file2.name = "prompt2.YAML"  # test case insensitive
        mock_file3 = mocker.MagicMock()
        mock_file3.is_file.return_value = False  # directory, should be ignored
        mock_file3.name = "subdir"
        mock_file4 = mocker.MagicMock()
        mock_file4.is_file.return_value = True
        mock_file4.name = "readme.txt"  # non-yaml file, should be ignored

        mock_prompts_dir.iterdir.return_value = [mock_file1, mock_file2, mock_file3, mock_file4]
        result = manager.list_prompts()
        assert result == ["prompt1", "prompt2"]

    def test_list_prompt_versions(self, mocker, manager: prompt_manager.PromptManager):
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
