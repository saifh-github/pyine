import pathlib
import tempfile

import pytest
import yaml

import pyine.prompts.prompt_utils as prompt_utils


class TestPromptConfig:

    @pytest.fixture
    def sample_metadata(self):
        return prompt_utils.PromptMetadata(
            name="Sample Prompt",
            description="A sample prompt for testing",
            version="v1.dummy",
        )

    @pytest.fixture
    def question_template(self):
        return prompt_utils.PromptTemplate(
            template="Now, answer this question: {question}",
            format="f-string",
        )

    @pytest.fixture
    def role_template(self):
        return prompt_utils.PromptTemplate(template="You are a helpful assistant.")

    @pytest.fixture
    def context_template(self):
        return prompt_utils.PromptTemplate(template="You answer silly questions with true truth.")

    @pytest.fixture
    def sample_examples(self):
        return [
            prompt_utils.PromptExample(
                input_variables={"question": "What are potatoes?"},
                output="boil em, mash em, stick em in a stew",
            ),
            prompt_utils.PromptExample(
                input_variables={"question": "How does the universe work?"},
                output="based on necessity and sufficiency",
                description="The Faithfulness Secret Sauce",
            ),
        ]

    @pytest.fixture
    def example_template_fstring(self):
        return prompt_utils.PromptTemplate(
            template="""\
Example {example_idx}/{example_count}:
Example question: {question}
Expected output: {output}
""",
            format="f-string",
        )

    @pytest.fixture
    def example_template_jinja2(self):
        return prompt_utils.PromptTemplate(
            template="""\
Example question: {{question}}
Expected output: {{output}}
{%- if description %}
Note: {{description}}
{%- endif %}
""",
            format="jinja2",
        )

    def test_create_simple_config(
        self,
        sample_metadata: prompt_utils.PromptMetadata,
        question_template: prompt_utils.PromptTemplate,
    ):
        config = prompt_utils.PromptConfig(
            metadata=sample_metadata,
            question=question_template,
        )
        assert config.metadata == sample_metadata
        assert config.question == question_template
        assert config.examples == []
        assert config.get_examples_as_text() == ""
        rendered = config.render_prompt(question="What is the meaning of life?")
        assert rendered == "Now, answer this question: What is the meaning of life?"

    def test_create_example_config(
        self,
        sample_metadata: prompt_utils.PromptMetadata,
        sample_examples: list[prompt_utils.PromptExample],
        question_template: prompt_utils.PromptTemplate,
        example_template_fstring: prompt_utils.PromptTemplate,
    ):
        config = prompt_utils.PromptConfig(
            metadata=sample_metadata,
            question=question_template,
            example_template=example_template_fstring,
            examples=sample_examples,
        )
        assert config.metadata == sample_metadata
        assert config.question == question_template
        assert len(config.examples) == 2
        assert config.example_count == 2
        assert config.examples == sample_examples
        assert config.example_template == example_template_fstring
        examples_text = """\
Example 1/1:
Example question: How does the universe work?
Expected output: based on necessity and sufficiency
"""
        assert config.get_examples_as_text([1]) == examples_text

    def test_create_full_config(
        self,
        sample_metadata: prompt_utils.PromptMetadata,
        sample_examples: list[prompt_utils.PromptExample],
        role_template: prompt_utils.PromptTemplate,
        context_template: prompt_utils.PromptTemplate,
        question_template: prompt_utils.PromptTemplate,
        example_template_jinja2: prompt_utils.PromptTemplate,
    ):
        config = prompt_utils.PromptConfig(
            metadata=sample_metadata,
            role=role_template,
            context=context_template,
            question=question_template,
            example_template=example_template_jinja2,
            examples=sample_examples,
        )
        assert config.metadata == sample_metadata
        assert config.role == role_template
        assert config.context == context_template
        assert config.question == question_template
        assert len(config.examples) == 2
        assert config.example_count == 2
        assert config.examples == sample_examples
        assert config.example_template == example_template_jinja2
        examples_block = """\
Example question: What are potatoes?
Expected output: boil em, mash em, stick em in a stew

Example question: How does the universe work?
Expected output: based on necessity and sufficiency
Note: The Faithfulness Secret Sauce"""
        assert config.get_examples_as_text() == examples_block
        rendered = config.render_prompt(question="What is the meaning of life?")
        assert "Now, answer this question: What is the meaning of life?" in rendered
        assert role_template.template in rendered
        assert context_template.template in rendered
        assert examples_block in rendered


class TestVersionedPromptConfig:

    @pytest.fixture
    def sample_yaml_content_w_default(self):
        return """\
v1.0.0:
  metadata:
    name: "Test Prompt"
    description: "First version"
  question:
    template: "Hello {name}!"

v2.0.0:
  metadata:
    name: "Test Prompt"
    description: "Second version"
  question:
    template: "Greetings {name}, how are you?"
  examples:
    - input_variables:
        name: "Alice"
      output: "Greetings Alice, how are you?"
  example_template:
    template: |
      Example name: {name}
      Expected output: {output}

__default__: "v1.0.0"
"""

    @pytest.fixture
    def temp_yaml_file_w_default(self, sample_yaml_content_w_default: str):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(sample_yaml_content_w_default)
            temp_path = pathlib.Path(f.name)
        yield temp_path
        temp_path.unlink()

    @pytest.fixture
    def temp_yaml_file_wo_default(self, sample_yaml_content_w_default: str):
        sample_yaml_content_wo_default = "\n".join(sample_yaml_content_w_default.splitlines()[:-1])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(sample_yaml_content_wo_default)
            temp_path = pathlib.Path(f.name)
        yield temp_path
        temp_path.unlink()

    def test_from_yaml_with_default_key(self, temp_yaml_file_w_default: pathlib.Path):
        config = prompt_utils.VersionedPromptConfig.from_yaml(temp_yaml_file_w_default)
        assert config.version_count == 2
        assert "v1.0.0" in config.versions
        assert "v2.0.0" in config.versions
        assert config.default_version == "v1.0.0"
        default_config = config.get_default()
        assert default_config.question.template == "Hello {name}!"

    def test_from_yaml_without_default_key(self, temp_yaml_file_wo_default: pathlib.Path):
        config = prompt_utils.VersionedPromptConfig.from_yaml(temp_yaml_file_wo_default)
        assert config.version_count == 2
        assert "v1.0.0" in config.versions
        assert "v2.0.0" in config.versions
        assert config.default_version == "v2.0.0"
        default_config = config.get_default()
        assert default_config.question.template == "Greetings {name}, how are you?"

    def test_from_invalid_yaml(self):
        with pytest.raises(FileNotFoundError):
            prompt_utils.VersionedPromptConfig.from_yaml("nonexistent.yaml")

        yaml_content = """\
__default__: "v1.0.0"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = pathlib.Path(f.name)
        try:
            with pytest.raises(ValueError, match="no valid prompt versions found"):
                prompt_utils.VersionedPromptConfig.from_yaml(temp_path)
        finally:
            temp_path.unlink()

        yaml_content = """\
v1.0.0:
  metadata:
    name: "Test"
    description: "Test"
  question:
    template: "Hello!"

__default__: "v2.0.0"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = pathlib.Path(f.name)
        try:
            with pytest.raises(AssertionError, match="default version does not exist"):
                prompt_utils.VersionedPromptConfig.from_yaml(temp_path)
        finally:
            temp_path.unlink()


class TestPromptManager:

    @pytest.fixture
    def manager(self):
        mgr = prompt_utils.PromptManager(package_name="test.package", prompts_subdir="test_templates")
        path = mgr._get_prompt_path("test_prompt")
        expected = pathlib.Path("test_templates") / "test_prompt.yaml"
        assert path == expected
        return mgr

    def test_manager_init(self):
        manager = prompt_utils.PromptManager()
        assert manager.package_name == "pyine.prompts"
        assert manager.prompts_subdir == "templates"
        assert isinstance(manager._cache, dict)
        manager = prompt_utils.PromptManager(package_name="custom.package", prompts_subdir="custom_dir")
        assert manager.package_name == "custom.package"
        assert manager.prompts_subdir == "custom_dir"

    def test_load_invalid_prompt_config(self, mocker, manager: prompt_utils.PromptManager):
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

    def test_list_prompts_directory_exists(self, mocker, manager: prompt_utils.PromptManager):
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

    def test_list_prompt_versions(self, mocker, manager: prompt_utils.PromptManager):
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
