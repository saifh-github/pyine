import langchain_core.prompts

import pyine.prompts.manager
import pyine.prompts.utils


def test_get_no_pressure_config_and_template() -> None:
    prompt_version = "no_pressure_demo"
    config = pyine.prompts.manager.get_prompt_config("code_execution", version=prompt_version)
    assert isinstance(config, pyine.prompts.utils.PromptConfig)
    first_example = config.examples[0]
    assert isinstance(first_example, pyine.prompts.utils.PromptExample)
    assert first_example.input_variables["code"].startswith('name = input("Enter your name: ")\n')
    assert first_example.input_variables["inputs"] == "Bob"
    assert first_example.output.rstrip("\n") == "Hello, Bob"
    template = pyine.prompts.manager.get_prompt_template("code_execution", version=prompt_version)
    assert isinstance(template, langchain_core.prompts.PromptTemplate)
    template_str = template.template
    assert template_str.startswith("You are an expert at interpreting and executing Python 3 code.")
    assert template_str.endswith("Now, provide your predicted output, and nothing else:\n")
    assert template.input_variables == ["code", "inputs"]
    rendered_str = template.format(
        code='name = input("Enter name: ")\nprint("Hello, " + name)',
        inputs="Bob",
    )
    assert "Enter name: " in rendered_str and "```\nBob\n```" in rendered_str


def test_get_unstructured_with_3_predict_types_config_and_template() -> None:
    prompt_version = "unstructured_with_3_predict_types"
    config = pyine.prompts.manager.get_prompt_config("code_execution", version=prompt_version)
    assert isinstance(config, pyine.prompts.utils.PromptConfig)
    assert config.metadata.name == "code_execution"
    assert config.example_count >= 3
    template = pyine.prompts.manager.get_prompt_template(
        "code_execution", version=prompt_version, include_examples=True
    )
    assert isinstance(template, langchain_core.prompts.PromptTemplate)
    core_vars = {"code", "predict_type", "inputs"}
    assert core_vars.issubset(set(template.input_variables))
    assert template.optional_variables == ["description", "entrypoint"]
    template_str = template.template
    assert template_str.startswith("You are an expert at interpreting and executing Python 3 code.")

    rendered_prog = template.format(
        code='print("Hello, " + input("Enter name: "))',
        description="Greets a user by name.",
        predict_type="program_output",
        inputs="Bobby",
    )
    assert "Type of execution output that should be predicted: program_output" in rendered_prog
    assert "Greets a user by name" in rendered_prog
    assert "Enter name: " in rendered_prog and "Bobby" in rendered_prog
    assert rendered_prog.endswith("Now, provide ONLY the execution output:")

    rendered_vars = template.format(
        code="x=2\ny=3\nz=x+y",
        description="Adds two numbers.",
        predict_type="frame_variables",
        inputs="",
        first_line=1,
        last_line=3,
        first_line_hit=1,
        last_line_hit=2,
    )
    assert "Type of execution output that should be predicted: frame_variables" in rendered_vars
    assert "First line" in rendered_vars and "Last line" in rendered_vars
    assert rendered_vars.endswith("dump that would be obtained after the specified hit of the last line:")

    rendered_ret = template.format(
        code="def add(a,b): return a+b\nresult = add(1,2)",
        description="Simple add function.",
        entrypoint="add",
        predict_type="function_return",
        inputs="a=1, b=2",
        first_line=1,
        last_line=1,
    )
    assert "Type of execution output that should be predicted: function_return" in rendered_ret
    assert "Consider only a call of the following function:" in rendered_ret
    assert "Use the following call input arguments:" in rendered_ret
    assert "a=1, b=2" in rendered_ret
    assert rendered_ret.endswith("Now, provide ONLY the function's returned value(s):")


# @@@@@ TODO: add optional tests w/ LLM invocations depending on cluster availability
