import pytest

import pyine.organisms.models.rewards.core.configs
import pyine.organisms.models.rewards.core.logging
import pyine.organisms.models.rewards.core.manager
import pyine.organisms.models.rewards.core.registry
import pyine.organisms.models.rewards.core.types
import tests.organisms.models.rewards.conftest as rewards_conftest


class _CountingParser:
    def __init__(self) -> None:
        self.calls = 0

    def parse(
        self,
        prompt: str,
        model_output: str,
    ) -> pyine.organisms.models.rewards.core.types.ParsedOutput:
        self.calls += 1
        return pyine.organisms.models.rewards.core.types.ParsedOutput(raw=model_output, final_answer="x")


class _NeedsParsedTerm:
    def reset(
        self,
        run_init_ctx: pyine.organisms.models.rewards.core.types.RunInitContext,
    ) -> None:
        del run_init_ctx

    def __call__(
        self,
        sample_ctx: pyine.organisms.models.rewards.core.types.SampleContext,
    ) -> pyine.organisms.models.rewards.core.types.TermResult:
        assert sample_ctx.parsed is not None
        assert sample_ctx.parsed.final_answer is not None
        return pyine.organisms.models.rewards.core.types.TermResult(value=1.0)


class TestRewardManager:
    def test_weighting_and_breakdown(self) -> None:
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                    weight=2.0,
                    params={"reward_if_present": 1.0, "reward_if_missing": 0.0},
                )
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(fallback_policy="none"),
            output=pyine.organisms.models.rewards.core.configs.OutputConfig(return_breakdown_default=True),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config)
        sample_ctx = pyine.organisms.models.rewards.core.types.SampleContext(
            prompt="p",
            model_output="<final>ok</final>",
            sample_data=rewards_conftest.make_sample_data("s1"),
        )
        total, breakdown = manager.compute(sample_ctx)
        assert total == 2.0
        assert breakdown == {"parseable": 2.0}

    def test_parseable_answer_qol_clean_stop_and_single_block(self) -> None:
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                    params={
                        "final_tag": "final",
                        "reward_if_present": 1.0,
                        "reward_if_missing": 0.0,
                        "bonus_if_stops_after_final_tag": 0.5,
                        "bonus_if_single_final_block": 0.25,
                    },
                )
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(fallback_policy="none"),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config)
        sample_ctx = pyine.organisms.models.rewards.core.types.SampleContext(
            prompt="p",
            model_output="<final>ok</final>\n",
            sample_data=rewards_conftest.make_sample_data("s1"),
        )
        output = manager.compute_output(sample_ctx, log=False)
        assert output.total == 1.75
        assert output.metrics["parseable/has_final_answer"] is True
        assert output.metrics["parseable/has_single_final_block"] is True
        assert output.metrics["parseable/stops_after_final_tag"] is True

    def test_parseable_answer_qol_trailing_text_no_penalty(self) -> None:
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                    params={
                        "final_tag": "final",
                        "reward_if_present": 1.0,
                        "reward_if_missing": 0.0,
                        "bonus_if_stops_after_final_tag": 0.5,
                        "bonus_if_single_final_block": 0.25,
                    },
                )
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(fallback_policy="none"),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config)
        sample_ctx = pyine.organisms.models.rewards.core.types.SampleContext(
            prompt="p",
            model_output="<final>ok</final> trailing",
            sample_data=rewards_conftest.make_sample_data("s1"),
        )
        output = manager.compute_output(sample_ctx, log=False)
        assert output.total == 1.25
        assert output.metrics["parseable/stops_after_final_tag"] is False

    def test_parse_is_cached_per_sample(self) -> None:
        registry = pyine.organisms.models.rewards.core.registry.RewardRegistry()

        def factory(
            spec: pyine.organisms.models.rewards.core.configs.RewardTermSpec,
            *,
            parser: pyine.organisms.models.rewards.core.types.OutputParser | None,
        ) -> pyine.organisms.models.rewards.core.types.RewardTerm:
            del spec
            del parser
            return _NeedsParsedTerm()

        registry.register_term("test_needs_parsed", factory)
        parser = _CountingParser()
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(name="t1", type="test_needs_parsed"),
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(name="t2", type="test_needs_parsed"),
            ],
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config, parser=parser, registry=registry)
        sample_ctx = pyine.organisms.models.rewards.core.types.SampleContext(
            prompt="p",
            model_output="raw",
            sample_data=rewards_conftest.make_sample_data("s2"),
        )
        output = manager.compute_output(sample_ctx, log=False)
        assert output.total == 2.0
        assert parser.calls == 1

    def test_logging_frequency_and_scoping(self) -> None:
        logger_obj = pyine.organisms.models.rewards.core.logging.InMemoryRewardLogger()
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                )
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(fallback_policy="none"),
            logging=pyine.organisms.models.rewards.core.configs.LoggingConfig(
                enabled=True,
                log_every_n_examples=2,
                scope_prefix="reward",
            ),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config, logger=logger_obj)
        sample1 = pyine.organisms.models.rewards.core.types.SampleContext(
            prompt="p",
            model_output="<final>ok</final>",
            sample_data=rewards_conftest.make_sample_data("s1"),
        )
        sample2 = pyine.organisms.models.rewards.core.types.SampleContext(
            prompt="p",
            model_output="<final>ok</final>",
            sample_data=rewards_conftest.make_sample_data("s2"),
        )
        manager.compute_output(sample1)
        manager.compute_output(sample2)
        assert len(logger_obj.samples) == 1
        entry = logger_obj.samples[0]
        terms = entry["terms"]
        assert isinstance(terms, dict)
        assert "reward/terms/parseable" in terms
        manager.finalize_run()
        assert len(logger_obj.runs) == 1

    def test_step_is_propagated_to_logger(self) -> None:
        logger_obj = pyine.organisms.models.rewards.core.logging.InMemoryRewardLogger()
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                )
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(fallback_policy="none"),
            logging=pyine.organisms.models.rewards.core.configs.LoggingConfig(
                enabled=True,
                log_every_n_examples=1,
                scope_prefix="reward",
            ),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config, logger=logger_obj)
        manager.set_step(123)
        sample = pyine.organisms.models.rewards.core.types.SampleContext(
            prompt="p",
            model_output="<final>ok</final>",
            sample_data=rewards_conftest.make_sample_data("s1"),
        )
        manager.compute_output(sample)
        assert logger_obj.samples[0]["step"] == 123
        manager.compute_output(sample, step=7)
        assert logger_obj.samples[1]["step"] == 7

    def test_introspection_term_names_and_get_term(self) -> None:
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(name="t1", type="parseable_answer"),
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="t2",
                    type="parseable_answer",
                    enabled=False,
                ),
            ]
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config)
        assert manager.term_names == ("t1", "t2")
        assert manager.enabled_term_names == ("t1",)
        assert manager.get_term("t1") is not None
        with pytest.raises(KeyError, match="disabled"):
            manager.get_term("t2")

    def test_require_parsed_enforced(self) -> None:
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                    require_parsed=True,
                )
            ]
        )
        with pytest.raises(ValueError, match="require parsed"):
            pyine.organisms.models.rewards.core.manager.RewardManager(config)

    def test_registry_snapshot_supports_aliases(self) -> None:
        import pyine.organisms.models.rewards.terms

        pyine.organisms.models.rewards.terms.ensure_builtin_terms_registered()
        snapshot = pyine.organisms.models.rewards.core.registry.get_global_registry().snapshot()
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="format/parseable_answer",
                )
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(fallback_policy="none"),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config, registry=snapshot)
        sample_ctx = pyine.organisms.models.rewards.core.types.SampleContext(
            prompt="p",
            model_output="<final>ok</final>",
            sample_data=rewards_conftest.make_sample_data("s1"),
        )
        assert manager.compute(sample_ctx) == 1.0

    def test_no_enabled_terms_raises(self) -> None:
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="x",
                    type="parseable_answer",
                    enabled=False,
                )
            ]
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config)
        sample_ctx = pyine.organisms.models.rewards.core.types.SampleContext(
            prompt="p",
            model_output="<final>ok</final>",
            sample_data=rewards_conftest.make_sample_data("s1"),
        )
        with pytest.raises(ValueError, match="no enabled"):
            manager.compute_output(sample_ctx)

    def test_unknown_term_type_raises(self) -> None:
        with pytest.raises(KeyError, match="unknown key"):
            pyine.organisms.models.rewards.core.manager.RewardManager(
                pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
                    terms=[pyine.organisms.models.rewards.core.configs.RewardTermSpec(name="x", type="does_not_exist")]
                )
            )

    def test_logging_enabled_without_logger_raises(self) -> None:
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[pyine.organisms.models.rewards.core.configs.RewardTermSpec(name="x", type="parseable_answer")],
            logging=pyine.organisms.models.rewards.core.configs.LoggingConfig(enabled=True),
        )
        with pytest.raises(ValueError, match="logging is enabled.*but no logger"):
            pyine.organisms.models.rewards.core.manager.RewardManager(config)

    def test_tag_inconsistency_warning(self) -> None:
        import warnings

        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                    params={"final_tag": "answer"},  # different from parser's "final"
                )
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(final_tag="final"),
        )
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            pyine.organisms.models.rewards.core.manager.RewardManager(config)
        assert len(recorded) == 1
        assert "final_tag='answer'" in str(recorded[0].message)
        assert "parser uses final_tag='final'" in str(recorded[0].message)


class TestMakeSimpleManager:
    def test_creates_manager_with_single_term(self) -> None:
        manager = pyine.organisms.models.rewards.core.manager.make_simple_manager(
            [
                ("format", "parseable_answer", 0.5),
            ]
        )
        assert manager.term_names == ("format",)
        assert manager.enabled_term_names == ("format",)

    def test_computes_rewards(self) -> None:
        manager = pyine.organisms.models.rewards.core.manager.make_simple_manager(
            [
                ("format", "parseable_answer", 1.0),
            ]
        )
        ctx = pyine.organisms.models.rewards.core.types.SampleContext(
            prompt="p",
            model_output="<final>ok</final>",
            sample_data=rewards_conftest.make_sample_data("s1"),
        )
        total = manager.compute(ctx)
        assert total == 1.0

    def test_parsing_disabled_returns_zero_for_missing_final(self) -> None:
        # without parsing, parseable_answer can't find the final answer and returns reward_if_missing
        manager = pyine.organisms.models.rewards.core.manager.make_simple_manager(
            [("format", "parseable_answer", 1.0)],
            parsing=False,
        )
        ctx = pyine.organisms.models.rewards.core.types.SampleContext(
            prompt="p",
            model_output="<final>ok</final>",
            sample_data=rewards_conftest.make_sample_data("s1"),
        )
        output = manager.compute_output(ctx, log=False)
        assert output.total == 0.0  # parsed is None, so no final answer detected

    def test_custom_tags(self) -> None:
        manager = pyine.organisms.models.rewards.core.manager.make_simple_manager(
            [("format", "parseable_answer", 1.0)],
            final_tag="answer",
            reasoning_tag="think",
        )
        ctx = pyine.organisms.models.rewards.core.types.SampleContext(
            prompt="p",
            model_output="<answer>ok</answer>",
            sample_data=rewards_conftest.make_sample_data("s1"),
        )
        total = manager.compute(ctx)
        assert total == 1.0

    def test_empty_terms_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one term"):
            pyine.organisms.models.rewards.core.manager.make_simple_manager([])


class TestListAvailableTerms:
    def test_returns_term_info_list(self) -> None:
        terms = pyine.organisms.models.rewards.core.registry.list_available_terms()
        assert isinstance(terms, list)
        assert len(terms) >= 2  # at least parseable_answer and text_length

    def test_includes_parseable_answer(self) -> None:
        terms = pyine.organisms.models.rewards.core.registry.list_available_terms()
        canonical_types = [t.canonical_type for t in terms]
        assert "parseable_answer" in canonical_types

    def test_includes_text_length(self) -> None:
        terms = pyine.organisms.models.rewards.core.registry.list_available_terms()
        canonical_types = [t.canonical_type for t in terms]
        assert "text_length" in canonical_types

    def test_term_info_has_aliases(self) -> None:
        terms = pyine.organisms.models.rewards.core.registry.list_available_terms()
        parseable = next(t for t in terms if t.canonical_type == "parseable_answer")
        assert "format/parseable_answer" in parseable.aliases

    def test_term_info_has_docstring(self) -> None:
        terms = pyine.organisms.models.rewards.core.registry.list_available_terms()
        parseable = next(t for t in terms if t.canonical_type == "parseable_answer")
        assert parseable.docstring is not None
        assert "ParseableAnswerTerm" in parseable.docstring


class TestPackageLevelExports:
    def test_rewards_package_exports_rewardmanager(self) -> None:
        import pyine.organisms.models.rewards as rewards

        assert rewards.RewardManager is pyine.organisms.models.rewards.core.manager.RewardManager

    def test_rewards_package_exports_make_simple_manager(self) -> None:
        import pyine.organisms.models.rewards as rewards

        assert rewards.make_simple_manager is pyine.organisms.models.rewards.core.manager.make_simple_manager

    def test_rewards_package_exports_list_available_terms(self) -> None:
        import pyine.organisms.models.rewards as rewards

        assert rewards.list_available_terms is pyine.organisms.models.rewards.core.registry.list_available_terms

    def test_rewards_package_exports_samplecontext(self) -> None:
        import pyine.organisms.models.rewards as rewards

        assert rewards.SampleContext is pyine.organisms.models.rewards.core.types.SampleContext

    def test_rewards_package_exports_terminfo(self) -> None:
        import pyine.organisms.models.rewards as rewards

        assert rewards.TermInfo is pyine.organisms.models.rewards.core.registry.TermInfo

    def test_rewards_package_exports_runinitcontext(self) -> None:
        import pyine.organisms.models.rewards as rewards

        assert rewards.RunInitContext is pyine.organisms.models.rewards.core.types.RunInitContext

    def test_rewards_package_exports_parsedoutput(self) -> None:
        import pyine.organisms.models.rewards as rewards

        assert rewards.ParsedOutput is pyine.organisms.models.rewards.core.types.ParsedOutput

    def test_rewards_package_exports_codeexecevaldata(self) -> None:
        import pyine.organisms.models.rewards as rewards

        assert rewards.CodeExecEvalData is pyine.organisms.models.rewards.core.types.CodeExecEvalData

    def test_rewards_package_exports_rewardoutput(self) -> None:
        import pyine.organisms.models.rewards as rewards

        assert rewards.RewardOutput is pyine.organisms.models.rewards.core.types.RewardOutput
