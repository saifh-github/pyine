import langchain_core.output_parsers
import langchain_core.prompts
import langchain_core.runnables
import pytest

import pyine.prompts.configs.pred_grader as pred_grader
import pyine.prompts.manager
import pyine.prompts.utils
import pyine.utils.code.output_compare
import pyine.utils.llm_providers
from tests.data.utils.env_checks import OPENAI_API_KEY_MISSING


def test_get_score_only_config_and_template():
    prompt_version = "score_only"
    config = pyine.prompts.manager.get_prompt_config("pred_grader", prompt_version)
    assert isinstance(config, pyine.prompts.utils.PromptConfig)
    assert config.metadata.name == "pred_grader"
    assert config.example_count >= 2
    ex = config.examples[0]
    assert isinstance(ex, pyine.prompts.utils.PromptExample)
    assert ex.input_variables["execution_type"] in {"program output", "frame variables", "function return"}
    assert isinstance(ex.output, pred_grader.GradingResult)
    assert 0.0 <= ex.output.score <= 1.0
    template = pyine.prompts.manager.get_prompt_template("pred_grader", prompt_version, include_examples=False)
    assert isinstance(template, langchain_core.prompts.PromptTemplate)
    tmpl = template.template
    assert tmpl.startswith("You are an expert evaluator of Python 3 code execution results.")
    assert tmpl.endswith("Now, provide ONLY the structured SCORE as per the required format above:\n")
    for var in ["execution_type", "expected_output", "predicted_output"]:
        assert var in template.input_variables
    rendered = template.format(
        execution_type="program output",
        expected_output="Hello, Bob",
        predicted_output="Hello Bob",
    )
    assert "Execution type: program output" in rendered
    assert "EXPECTED OUTPUT" in rendered and "PREDICTED OUTPUT" in rendered


def test_get_with_reasoning_config_and_template():
    prompt_version = "with_reasoning"
    config = pyine.prompts.manager.get_prompt_config("pred_grader", prompt_version)
    assert isinstance(config, pyine.prompts.utils.PromptConfig)
    assert config.metadata.name == "pred_grader"
    ex = config.examples[0]
    assert isinstance(ex.output, pred_grader.GradingResultWithReasoning)
    assert 0.0 <= ex.output.score <= 1.0
    assert ex.output.reasoning is None or isinstance(ex.output.reasoning, str)
    template = pyine.prompts.manager.get_prompt_template("pred_grader", prompt_version, include_examples=False)
    assert isinstance(template, langchain_core.prompts.PromptTemplate)
    tmpl = template.template
    assert tmpl.endswith(
        "Now, provide ONLY the structured output (SCORE, and optionally a brief REASONING) "
        "as per the required format above:\n"
    )
    rendered = template.format(
        execution_type="function return",
        expected_output="{'a': 1, 'b': 2}",
        predicted_output="{'b': 2, 'a': 1}",
    )
    assert "Execution type" in rendered


def test_get_chain_attaches_parser(mocker):
    model = mocker.MagicMock()
    # default chain should be score only
    score_only_chain = pyine.prompts.manager.get_prompt_chain(model, "pred_grader")
    assert isinstance(score_only_chain, langchain_core.runnables.RunnableSequence)
    assert isinstance(score_only_chain.last, langchain_core.output_parsers.PydanticOutputParser)
    assert score_only_chain.last.pydantic_object == pred_grader.GradingResult
    with_reasoning_chain = pyine.prompts.manager.get_prompt_chain(model, "pred_grader", "with_reasoning")
    assert isinstance(with_reasoning_chain, langchain_core.runnables.RunnableSequence)
    assert isinstance(with_reasoning_chain.last, langchain_core.output_parsers.PydanticOutputParser)
    assert with_reasoning_chain.last.pydantic_object == pred_grader.GradingResultWithReasoning


@pytest.mark.skipif(OPENAI_API_KEY_MISSING, reason="OpenAI API key not available")
def test_pred_grader_infer_score_only():
    model = pyine.utils.llm_providers.get_model_from_provider(
        provider="openai",
        model="gpt-4o-mini",
    )
    result = pyine.utils.code.output_compare.compare_exec_output_with_llm(
        predicted="Hello Bob",
        expected="Hello, Bob",
        execution_type="program output",
        llm=model,
        with_reasoning=False,
    )
    assert hasattr(result, "score")
    assert isinstance(result.score, float)
    assert 0.0 <= result.score <= 1.0


@pytest.mark.skipif(OPENAI_API_KEY_MISSING, reason="OpenAI API key not available")
def test_pred_grader_infer_with_reasoning():
    model = pyine.utils.llm_providers.get_model_from_provider(
        provider="openai",
        model="gpt-4o-mini",
    )
    result = pyine.utils.code.output_compare.compare_exec_output_with_llm(
        predicted="Hello Bob",
        expected="Hello, Bob",
        execution_type="program output",
        llm=model,
        with_reasoning=True,
    )
    assert hasattr(result, "score")
    assert 0.0 <= float(result.score) <= 1.0
    assert hasattr(result, "reasoning")
    if result.reasoning is not None:
        assert isinstance(result.reasoning, str)
        assert len(result.reasoning) > 0
