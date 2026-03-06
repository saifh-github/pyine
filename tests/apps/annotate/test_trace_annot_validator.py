"""Tests for the trace annotation validator app."""

import json
import typing

import langchain_core.prompts
import pytest
import pytest_mock

import pyine.data.traces.dataset_reader
import pyine.data.traces.dataset_utils
import pyine.organisms.datamodules.utils.validator as validator_utils
import pyine.prompts.manager
import pyine.prompts.result_db
import pyine.prompts.types
import pyine.prompts.utils
import pyine.utils.code.execution
import pyine.utils.llm_providers


class TestPromptTemplateLoading:
    """Tests that the validation/misleading prompt template loads correctly."""

    def test_prompt_config_loads(self) -> None:
        config = pyine.prompts.manager.get_prompt_config("validation/misleading", version="v1.0")
        assert isinstance(config, pyine.prompts.utils.PromptConfig)

    def test_prompt_template_renders(self) -> None:
        template = pyine.prompts.manager.get_prompt_template("validation/misleading", version="v1.0")
        assert isinstance(template, langchain_core.prompts.PromptTemplate)
        assert set(template.input_variables) == {"code", "inputs", "expected_output"}
        rendered = template.format(
            code="x = int(input())\nprint(x + 1)",
            inputs="5",
            expected_output="6",
        )
        assert "x = int(input())" in rendered
        assert "```\n5\n```" in rendered
        assert "```\n6\n```" in rendered

    def test_prompt_name_is_registered(self) -> None:
        registered = set(pyine.prompts.list_prompts())
        assert "validation/misleading" in registered


class TestVerdictParsing:
    """Tests for the make_validator closure's JSON parsing and verdict logic."""

    def test_valid_misleading_verdict(self) -> None:
        meta: dict[str, typing.Any] = {}
        tags: list[str] = []
        validate = validator_utils.make_validator(meta, tags)
        result_str = json.dumps({"verdict": "MISLEADING", "explanation": "hints mislead"})
        out = validate(result_str, None)
        assert out == result_str
        assert meta["verdict"] == "MISLEADING"
        assert meta["explanation"] == "hints mislead"
        assert "verdict:misleading" in tags

    def test_valid_not_misleading_verdict(self) -> None:
        meta: dict[str, typing.Any] = {}
        tags: list[str] = []
        validate = validator_utils.make_validator(meta, tags)
        result_str = json.dumps({"verdict": "NOT_MISLEADING", "explanation": "hints correct"})
        out = validate(result_str, None)
        assert out == result_str
        assert meta["verdict"] == "NOT_MISLEADING"
        assert "verdict:not_misleading" in tags

    def test_valid_uninformative_verdict(self) -> None:
        meta: dict[str, typing.Any] = {}
        tags: list[str] = []
        validate = validator_utils.make_validator(meta, tags)
        result_str = json.dumps({"verdict": "UNINFORMATIVE", "explanation": "no hints present"})
        out = validate(result_str, None)
        assert out == result_str
        assert meta["verdict"] == "UNINFORMATIVE"
        assert meta["explanation"] == "no hints present"
        assert "verdict:uninformative" in tags

    def test_malformed_json_returns_none(self) -> None:
        meta: dict[str, typing.Any] = {}
        tags: list[str] = []
        validate = validator_utils.make_validator(meta, tags)
        assert validate("not json at all", None) is None
        assert "verdict" not in meta
        assert len(tags) == 0

    def test_invalid_verdict_value_returns_none(self) -> None:
        meta: dict[str, typing.Any] = {}
        tags: list[str] = []
        validate = validator_utils.make_validator(meta, tags)
        result_str = json.dumps({"verdict": "MAYBE", "explanation": "unsure"})
        assert validate(result_str, None) is None
        assert "verdict" not in meta

    def test_missing_verdict_key_returns_none(self) -> None:
        meta: dict[str, typing.Any] = {}
        tags: list[str] = []
        validate = validator_utils.make_validator(meta, tags)
        result_str = json.dumps({"explanation": "no verdict here"})
        assert validate(result_str, None) is None

    def test_markdown_fenced_json_is_handled(self) -> None:
        meta: dict[str, typing.Any] = {}
        tags: list[str] = []
        validate = validator_utils.make_validator(meta, tags)
        inner = json.dumps({"verdict": "MISLEADING", "explanation": "fenced"})
        result_str = f"```json\n{inner}\n```"
        out = validate(result_str, None)
        assert out == result_str
        assert meta["verdict"] == "MISLEADING"

    def test_missing_explanation_defaults_to_empty(self) -> None:
        meta: dict[str, typing.Any] = {}
        tags: list[str] = []
        validate = validator_utils.make_validator(meta, tags)
        result_str = json.dumps({"verdict": "MISLEADING"})
        out = validate(result_str, None)
        assert out == result_str
        assert meta["verdict"] == "MISLEADING"
        assert meta["explanation"] == ""
        assert "verdict:misleading" in tags

    def test_non_dict_json_returns_none(self) -> None:
        meta: dict[str, typing.Any] = {}
        tags: list[str] = []
        validate = validator_utils.make_validator(meta, tags)
        assert validate(json.dumps(["not", "a", "dict"]), None) is None


class TestBuildValidationTags:
    """Tests for the build_validation_tags helper."""

    @pytest.fixture
    def mock_problem(self, mocker: pytest_mock.MockerFixture) -> pyine.data.traces.dataset_utils.CodingProblem:
        return mocker.MagicMock(
            spec=pyine.data.traces.dataset_utils.CodingProblem,
            problem_tags=["difficulty:easy", "topic:math"],
        )

    @pytest.fixture
    def mock_trace(self, mocker: pytest_mock.MockerFixture) -> pyine.utils.code.execution.TraceResult:
        return mocker.MagicMock(
            spec=pyine.utils.code.execution.TraceResult,
            tags=["trace_tag_1"],
        )

    @pytest.fixture
    def mock_source_record(self, mocker: pytest_mock.MockerFixture) -> pyine.prompts.result_db.PromptResultRecord:
        return mocker.MagicMock(
            spec=pyine.prompts.result_db.PromptResultRecord,
            tags=["augment:misleading", "augment:hinted", "other_tag"],
        )

    @pytest.fixture
    def mock_llm_config(self, mocker: pytest_mock.MockerFixture) -> pyine.utils.llm_providers.LLMProviderConfig:
        return mocker.MagicMock(
            spec=pyine.utils.llm_providers.LLMProviderConfig,
            provider="openai",
            model_kwargs={"model": "gpt-4o-mini"},
        )

    def test_contains_expected_tags(
        self,
        mock_problem: pyine.data.traces.dataset_utils.CodingProblem,
        mock_trace: pyine.utils.code.execution.TraceResult,
        mock_source_record: pyine.prompts.result_db.PromptResultRecord,
        mock_llm_config: pyine.utils.llm_providers.LLMProviderConfig,
    ) -> None:
        tags = validator_utils.build_validation_tags(
            mock_problem,
            mock_trace,
            mock_source_record,
            mock_llm_config,
            None,
        )
        assert "validation:misleading" in tags
        assert "difficulty:easy" in tags
        assert "topic:math" in tags
        assert "trace_tag_1" in tags
        assert "augment:misleading" in tags
        assert "augment:hinted" in tags
        assert "llm_provider:openai" in tags
        assert "llm_provider_model:gpt-4o-mini" in tags

    def test_excludes_non_augment_source_tags(
        self,
        mock_problem: pyine.data.traces.dataset_utils.CodingProblem,
        mock_trace: pyine.utils.code.execution.TraceResult,
        mock_source_record: pyine.prompts.result_db.PromptResultRecord,
        mock_llm_config: pyine.utils.llm_providers.LLMProviderConfig,
    ) -> None:
        tags = validator_utils.build_validation_tags(
            mock_problem,
            mock_trace,
            mock_source_record,
            mock_llm_config,
            None,
        )
        assert "other_tag" not in tags

    def test_no_model_tag_when_model_absent(
        self,
        mocker: pytest_mock.MockerFixture,
        mock_problem: pyine.data.traces.dataset_utils.CodingProblem,
        mock_trace: pyine.utils.code.execution.TraceResult,
        mock_source_record: pyine.prompts.result_db.PromptResultRecord,
    ) -> None:
        config = mocker.MagicMock(
            spec=pyine.utils.llm_providers.LLMProviderConfig,
            provider="openai",
            model_kwargs={},
        )
        tags = validator_utils.build_validation_tags(
            mock_problem,
            mock_trace,
            mock_source_record,
            config,
            None,
        )
        assert not any(t.startswith("llm_provider_model:") for t in tags)

    def test_shared_tags_included(
        self,
        mock_problem: pyine.data.traces.dataset_utils.CodingProblem,
        mock_trace: pyine.utils.code.execution.TraceResult,
        mock_source_record: pyine.prompts.result_db.PromptResultRecord,
        mock_llm_config: pyine.utils.llm_providers.LLMProviderConfig,
    ) -> None:
        tags = validator_utils.build_validation_tags(
            mock_problem,
            mock_trace,
            mock_source_record,
            mock_llm_config,
            ["custom_tag"],
        )
        assert "custom_tag" in tags

    def test_tags_are_deduplicated(
        self,
        mock_problem: pyine.data.traces.dataset_utils.CodingProblem,
        mock_trace: pyine.utils.code.execution.TraceResult,
        mock_source_record: pyine.prompts.result_db.PromptResultRecord,
        mock_llm_config: pyine.utils.llm_providers.LLMProviderConfig,
    ) -> None:
        tags = validator_utils.build_validation_tags(
            mock_problem,
            mock_trace,
            mock_source_record,
            mock_llm_config,
            ["trace_tag_1"],
        )
        assert tags.count("trace_tag_1") == 1


class TestFilterMisleadingRecords:
    """Tests for the filter_misleading_records helper."""

    def test_keeps_misleading_records(self, mocker: pytest_mock.MockerFixture) -> None:
        misleading_rec = mocker.MagicMock(
            spec=pyine.prompts.result_db.PromptResultRecord,
            tags=["augment:misleading"],
            record_uid="misleading_001",
        )
        non_misleading_rec = mocker.MagicMock(
            spec=pyine.prompts.result_db.PromptResultRecord,
            tags=["augment:hinted"],
            record_uid="hinted_001",
        )
        filtered, skipped = validator_utils.filter_misleading_records([misleading_rec, non_misleading_rec])
        assert len(filtered) == 1
        assert filtered[0].record_uid == "misleading_001"
        assert skipped == 1

    def test_issues_docs_recognized_as_misleading(self, mocker: pytest_mock.MockerFixture) -> None:
        """The 'augment:issues_docs' tag implies misleading via SampleCodeTypeSet."""
        rec = mocker.MagicMock(
            spec=pyine.prompts.result_db.PromptResultRecord,
            tags=["augment:issues_docs"],
            record_uid="issues_docs_001",
        )
        filtered, skipped = validator_utils.filter_misleading_records([rec])
        assert len(filtered) == 1
        assert skipped == 0
        # same test, but for the v2 prompt, that will be refactored out later
        rec = mocker.MagicMock(
            spec=pyine.prompts.result_db.PromptResultRecord,
            tags=["augment:issues_docs_v2"],
            record_uid="issues_docs_v2_001",
        )
        filtered, skipped = validator_utils.filter_misleading_records([rec])
        assert len(filtered) == 1
        assert skipped == 0

    def test_all_non_misleading(self, mocker: pytest_mock.MockerFixture) -> None:
        rec = mocker.MagicMock(
            spec=pyine.prompts.result_db.PromptResultRecord,
            tags=["augment:obfuscated"],
            record_uid="obf_001",
        )
        filtered, skipped = validator_utils.filter_misleading_records([rec])
        assert len(filtered) == 0
        assert skipped == 1

    def test_empty_input(self) -> None:
        filtered, skipped = validator_utils.filter_misleading_records([])
        assert len(filtered) == 0
        assert skipped == 0


class TestFilterBuggedRecords:
    """Tests for the filter_bugged_records helper."""

    def test_bugged_records_excluded(self, mocker: pytest_mock.MockerFixture) -> None:
        bugged_rec = mocker.MagicMock(
            spec=pyine.prompts.result_db.PromptResultRecord,
            tags=["augment:misleading", "augment:bugged"],
            record_uid="bugged_001",
            identifier="trace_001",
        )
        clean_rec = mocker.MagicMock(
            spec=pyine.prompts.result_db.PromptResultRecord,
            tags=["augment:misleading"],
            record_uid="clean_001",
            identifier="trace_002",
        )
        filtered, skipped = validator_utils.filter_bugged_records([bugged_rec, clean_rec])
        assert len(filtered) == 1
        assert filtered[0].record_uid == "clean_001"
        assert skipped == 1

    def test_no_bugged_records(self, mocker: pytest_mock.MockerFixture) -> None:
        rec = mocker.MagicMock(
            spec=pyine.prompts.result_db.PromptResultRecord,
            tags=["augment:misleading"],
            record_uid="clean_001",
            identifier="trace_001",
        )
        filtered, skipped = validator_utils.filter_bugged_records([rec])
        assert len(filtered) == 1
        assert skipped == 0

    def test_all_bugged(self, mocker: pytest_mock.MockerFixture) -> None:
        rec = mocker.MagicMock(
            spec=pyine.prompts.result_db.PromptResultRecord,
            tags=["augment:bugged"],
            record_uid="bugged_001",
            identifier="trace_001",
        )
        filtered, skipped = validator_utils.filter_bugged_records([rec])
        assert len(filtered) == 0
        assert skipped == 1

    def test_issues_variant_detected_as_bugged(self, mocker: pytest_mock.MockerFixture) -> None:
        """Tags like 'augment:issues_todos' imply bugged code via SampleCodeTypeSet parsing."""
        rec = mocker.MagicMock(
            spec=pyine.prompts.result_db.PromptResultRecord,
            tags=["augment:issues_todos"],
            record_uid="issues_001",
            identifier="trace_001",
        )
        filtered, skipped = validator_utils.filter_bugged_records([rec])
        assert len(filtered) == 0
        assert skipped == 1

    def test_empty_input(self) -> None:
        filtered, skipped = validator_utils.filter_bugged_records([])
        assert len(filtered) == 0
        assert skipped == 0


class TestResolveRecord:
    """Tests for resolve_record, which handles skipping and dataset reads on the main thread."""

    @pytest.fixture
    def mock_dataset(self, mocker: pytest_mock.MockerFixture) -> pyine.data.traces.dataset_reader.DatasetProtocol:
        dataset = mocker.MagicMock(spec=pyine.data.traces.dataset_reader.DatasetProtocol)
        dataset.__getitem__ = mocker.MagicMock(
            return_value=mocker.MagicMock(
                spec=pyine.utils.code.execution.TraceResult,
                tags=["trace_tag"],
                inputs="5",
                expected_output="6",
            )
        )
        dataset.get_problem_data = mocker.MagicMock(
            return_value=mocker.MagicMock(
                spec=pyine.data.traces.dataset_utils.CodingProblem,
                problem_tags=["topic:math"],
            )
        )
        return dataset

    @pytest.fixture
    def mock_record(self, mocker: pytest_mock.MockerFixture) -> pyine.prompts.result_db.PromptResultRecord:
        return mocker.MagicMock(
            spec=pyine.prompts.result_db.PromptResultRecord,
            record_uid="rec_001",
            identifier="trace_key_42",
            prompt_name="hints/docs",
            result="print(x + 1)",
            group="solution_1",
            tags=["augment:misleading"],
        )

    def test_skip_already_validated(
        self,
        mock_dataset: pyine.data.traces.dataset_reader.DatasetProtocol,
        mock_record: pyine.prompts.result_db.PromptResultRecord,
    ) -> None:
        result = validator_utils.resolve_record(
            mock_record, mock_dataset, target_keys=None, already_validated_uids={"rec_001"}, force_generation=False
        )
        assert result == "skipped_validated"
        mock_dataset.__getitem__.assert_not_called()

    def test_skip_target_indices_before_dataset_read(
        self,
        mock_dataset: pyine.data.traces.dataset_reader.DatasetProtocol,
        mock_record: pyine.prompts.result_db.PromptResultRecord,
    ) -> None:
        result = validator_utils.resolve_record(
            mock_record,
            mock_dataset,
            target_keys={"trace_key_0", "trace_key_1"},
            already_validated_uids=set(),
            force_generation=False,
        )
        assert result == "skipped_target_indices"
        mock_dataset.__getitem__.assert_not_called()

    def test_returns_trace_and_problem_when_not_skipped(
        self,
        mock_dataset: pyine.data.traces.dataset_reader.DatasetProtocol,
        mock_record: pyine.prompts.result_db.PromptResultRecord,
    ) -> None:
        result = validator_utils.resolve_record(
            mock_record, mock_dataset, target_keys=None, already_validated_uids=set(), force_generation=False
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        mock_dataset.__getitem__.assert_called_once_with("trace_key_42")
        mock_dataset.get_problem_data.assert_called_once_with("trace_key_42")

    def test_returns_error_on_missing_key(
        self,
        mocker: pytest_mock.MockerFixture,
        mock_record: pyine.prompts.result_db.PromptResultRecord,
    ) -> None:
        dataset = mocker.MagicMock(spec=pyine.data.traces.dataset_reader.DatasetProtocol)
        dataset.__getitem__ = mocker.MagicMock(side_effect=KeyError("not found"))
        result = validator_utils.resolve_record(
            mock_record, dataset, target_keys=None, already_validated_uids=set(), force_generation=False
        )
        assert result == "error"


class TestLineageMetaProtection:
    """Tests that --shared-meta cannot overwrite lineage keys."""

    def test_lineage_keys_not_overwritten_by_shared_meta(self) -> None:
        """The process_one_validation helper applies shared_meta first, then sets lineage keys."""
        shared_meta_dict = {"source_record_uid": "should_be_overwritten", "custom_key": "custom_val"}
        meta: dict[str, typing.Any] = {}
        meta.update(shared_meta_dict)
        meta["source_record_uid"] = "real_uid"
        meta["source_prompt_name"] = "hints/docs"
        meta["source_identifier"] = "trace_001"
        assert meta["source_record_uid"] == "real_uid"
        assert meta["custom_key"] == "custom_val"

    def test_reserved_keys_constant_coverage(self) -> None:
        assert "source_record_uid" in validator_utils.LINEAGE_META_KEYS
        assert "source_prompt_name" in validator_utils.LINEAGE_META_KEYS
        assert "source_identifier" in validator_utils.LINEAGE_META_KEYS


class TestClosureOrderingRegression:
    """Regression test: the validator closure must be able to mutate meta/tags that are
    then read by fetch_or_generate_prompt_results for storage.

    This pins the execution ordering at result_db.py:1133-1164 (validator runs, then
    tags_to_store = list(tags), then db.store with meta=meta).
    """

    def test_validator_mutations_visible_in_stored_record(
        self,
        mocker: pytest_mock.MockerFixture,
        tmp_path: typing.Any,
    ) -> None:
        db_path = tmp_path / "test_closure.db"
        db = pyine.prompts.result_db.PromptResultDB(db_path)
        mock_response = json.dumps({"verdict": "MISLEADING", "explanation": "test closure"})
        mocker.patch.object(pyine.prompts.manager, "list_prompts", return_value=["validation/misleading"])
        mock_template = langchain_core.prompts.PromptTemplate(
            template="test {code} {inputs} {expected_output}",
            input_variables=["code", "inputs", "expected_output"],
        )
        mock_chain = mocker.MagicMock()
        mock_chain.invoke.return_value = mock_response
        chain_config = mocker.MagicMock(spec=pyine.prompts.types.PromptChainBuildConfig)
        chain_config.prompt = mocker.MagicMock()
        chain_config.prompt.prompt_name = "validation/misleading"
        chain_config.prompt.version = "v1.0"
        chain_config.template = mock_template
        chain_config.chain = mock_chain
        meta: dict[str, typing.Any] = {"source_record_uid": "test_src"}
        tags: list[str] = ["validation:misleading"]
        closure_validator = validator_utils.make_validator(meta, tags)
        records = pyine.prompts.result_db.fetch_or_generate_prompt_results(
            identifier="test_closure_id",
            input_variables={"code": "print(1)", "inputs": "None", "expected_output": "1"},
            prompt_chain_config=chain_config,
            db=db,
            output_validator=closure_validator,
            generate_until_result_count=1,
            force_generation=True,
            log_new_results=True,
            tags=tags,
            meta=meta,
        )
        assert len(records) == 1
        stored_record = records[0]
        assert stored_record.meta.get("verdict") == "MISLEADING"
        assert stored_record.meta.get("explanation") == "test closure"
        assert "verdict:misleading" in stored_record.tags
        db_records = db.get_by_identifier("test_closure_id", prompt_name="validation/misleading")
        assert len(db_records) == 1
        assert db_records[0].meta.get("verdict") == "MISLEADING"
        assert "verdict:misleading" in db_records[0].tags
