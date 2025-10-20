import pathlib
import types

import pytest_mock

import pyine.apps.traces.taco_trace_failure_analyzer
import pyine.prompts.configs.input_output_rewrite
import pyine.prompts.result_db
import pyine.prompts.types


class TestCollectProblemPaths:
    def test_returns_requested_relative_files(self, tmp_path: pathlib.Path) -> None:
        problem_dir = tmp_path
        target_file = problem_dir / "example.json"
        target_file.write_text("{}")
        other_file = problem_dir / "other.json"
        other_file.write_text("{}")

        collected = pyine.apps.traces.taco_trace_failure_analyzer._collect_problem_paths(
            problem_dir=problem_dir,
            filenames=("example.json",),
            override_log_path=None,
        )

        assert collected == [target_file]
        assert other_file not in collected

    def test_excludes_override_path_when_scanning(self, tmp_path: pathlib.Path) -> None:
        problem_dir = tmp_path
        first_file = problem_dir / "first.json"
        first_file.write_text("{}")
        nested_dir = problem_dir / "nested"
        nested_dir.mkdir()
        second_file = nested_dir / "second.json"
        second_file.write_text("{}")
        override_path = nested_dir / "problem_data_overrides.json"
        override_path.write_text("{}")

        collected = pyine.apps.traces.taco_trace_failure_analyzer._collect_problem_paths(
            problem_dir=problem_dir,
            filenames=(),
            override_log_path=override_path,
        )

        assert override_path.resolve() not in {path.resolve() for path in collected}
        assert collected == [first_file, second_file]

    def test_skips_directory_inputs(
        self,
        tmp_path: pathlib.Path,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        problem_dir = tmp_path
        target_dir = problem_dir / "nested"
        target_dir.mkdir()

        target_logger = pyine.apps.traces.taco_trace_failure_analyzer.logger
        mock_warning = mocker.patch.object(target_logger, "warning")
        collected = pyine.apps.traces.taco_trace_failure_analyzer._collect_problem_paths(
            problem_dir=problem_dir,
            filenames=(target_dir,),
            override_log_path=None,
        )

        assert collected == []
        mock_warning.assert_called_once()
        warning_args = mock_warning.call_args[0]
        assert warning_args and "path is not a file" in warning_args[0]


class TestGenerateCandidateInputOutput:
    def test_returns_latest_record_result(self, mocker: pytest_mock.MockerFixture) -> None:
        prompt_fetcher = mocker.Mock()
        expected = pyine.prompts.configs.input_output_rewrite.InputOutputRewriteResponse(
            inputs=['{"value": 1}'],
            outputs=[1],
            fn_name="solve",
        )
        prompt_fetcher.fetch_or_generate.return_value = [
            types.SimpleNamespace(result=expected),
        ]

        result = pyine.apps.traces.taco_trace_failure_analyzer._generate_candidate_input_output(
            model=object(),
            prompt_fetcher=prompt_fetcher,
            prompt_config=pyine.prompts.types.PromptBuildConfig(
                prompt_name="input_output_rewrite",
                version="v1.0",
            ),
            problem_identifier="problem-1",
            question="What is 1+0?",
            starter_code="def solve(): ...",
            first_solution="def solve(): return 1",
            current_input_output={"inputs": [], "outputs": []},
        )

        assert result == expected
        prompt_fetcher.fetch_or_generate.assert_called_once()
        kwargs = prompt_fetcher.fetch_or_generate.call_args.kwargs
        assert kwargs["force_generation"] is False
        assert kwargs["log_new_results"] is True

    def test_returns_none_on_validation_failure(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        prompt_fetcher = mocker.Mock()
        prompt_fetcher.fetch_or_generate.side_effect = pyine.prompts.result_db.ValidationFailedError("boom")

        result = pyine.apps.traces.taco_trace_failure_analyzer._generate_candidate_input_output(
            model=object(),
            prompt_fetcher=prompt_fetcher,
            prompt_config=pyine.prompts.types.PromptBuildConfig(
                prompt_name="input_output_rewrite",
                version="v1.0",
            ),
            problem_identifier="problem-2",
            question="placeholder",
            starter_code="def solve(): ...",
            first_solution="def solve(): return 1",
            current_input_output={},
        )

        assert result is None
        kwargs = prompt_fetcher.fetch_or_generate.call_args.kwargs
        assert kwargs["force_generation"] is False
        assert kwargs["log_new_results"] is True
