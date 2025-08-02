import pyine.utils.pydantic_loader as loader


def test_dump_and_load_yaml(tmp_path):
    loader.PydanticYAMLLoader.register_models_from_module("pyine.prompts.configs.callable_analysis")
    registered_models = loader.PydanticYAMLLoader.get_registered_models()
    expected_tag_name = "pyine.prompts.configs.callable_analysis.CallableAnalysisResponse"
    assert expected_tag_name in registered_models
    dummy_yaml_data = dict(
        model=registered_models[expected_tag_name](
            entrypoint_function_name="sum_two_numbers",
            entrypoint_function_arg_names=["a", "b"],
            parent_class_name="",
            parent_class_arg_names=[],
        )
    )

    dummy_yaml_path = tmp_path / "dummy.yaml"
    loader.dump_yaml_with_pydantic_support(dummy_yaml_data, dummy_yaml_path)
    loaded_data = loader.load_yaml_with_pydantic_support(dummy_yaml_path)
    model = loaded_data["model"]
    assert isinstance(model, registered_models[expected_tag_name])


def test_register_all_from_root():
    loader.PydanticYAMLLoader.register_models_from_package("pyine")
    registered_models = loader.PydanticYAMLLoader.get_registered_models()
    assert "pyine.prompts.configs.callable_analysis.CallableAnalysisResponse" in registered_models
    assert "pyine.prompts.utils.PromptTemplate" in registered_models
