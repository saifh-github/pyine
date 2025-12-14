import pytest

import pyine.organisms.datamodules.samples.common
import pyine.organisms.models.rewards.core.configs
import pyine.organisms.models.rewards.core.logging
import pyine.organisms.models.rewards.core.manager
import pyine.organisms.models.rewards.core.registry
import pyine.organisms.models.rewards.core.types


def _make_sample_data(
    identifier: str,
) -> pyine.organisms.datamodules.samples.common.SampleData:
    """Build a minimal `SampleData` instance for reward tests."""
    return pyine.organisms.datamodules.samples.common.SampleData(
        identifier=identifier,
        code="print('hi')",
        description="",
        entrypoint="",
        first_line=0,
        last_line=1,
        inputs="",
        expected_output="hi\n",
        predict_type=pyine.organisms.datamodules.samples.common.SamplePredictType.program_output,
        code_type="original",
        trace_step_count=1,
        comma_separated_tags="",
        has_code_override=False,
        complexity_metrics={},
    )


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


def _register_test_term_once() -> None:
    """Register a small in-test term factory once, to avoid global registry collisions."""
    term_type = "test_needs_parsed"
    try:
        pyine.organisms.models.rewards.core.registry.get_term_factory(term_type)
        return
    except KeyError:
        pass

    def factory(
        spec: pyine.organisms.models.rewards.core.configs.RewardTermSpec,
        *,
        parser: pyine.organisms.models.rewards.core.types.OutputParser | None,
    ) -> pyine.organisms.models.rewards.core.types.RewardTerm:
        del spec
        del parser
        return _NeedsParsedTerm()

    pyine.organisms.models.rewards.core.registry.register_term(term_type, factory)


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
            sample_data=_make_sample_data("s1"),
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
            sample_data=_make_sample_data("s1"),
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
            sample_data=_make_sample_data("s1"),
        )
        output = manager.compute_output(sample_ctx, log=False)
        assert output.total == 1.25
        assert output.metrics["parseable/stops_after_final_tag"] is False

    def test_parse_is_cached_per_sample(self) -> None:
        _register_test_term_once()
        parser = _CountingParser()
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(name="t1", type="test_needs_parsed"),
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(name="t2", type="test_needs_parsed"),
            ],
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config, parser=parser)
        sample_ctx = pyine.organisms.models.rewards.core.types.SampleContext(
            prompt="p",
            model_output="raw",
            sample_data=_make_sample_data("s2"),
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
            sample_data=_make_sample_data("s1"),
        )
        sample2 = pyine.organisms.models.rewards.core.types.SampleContext(
            prompt="p",
            model_output="<final>ok</final>",
            sample_data=_make_sample_data("s2"),
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
            sample_data=_make_sample_data("s1"),
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
            sample_data=_make_sample_data("s1"),
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
            sample_data=_make_sample_data("s1"),
        )
        with pytest.raises(ValueError, match="no enabled"):
            manager.compute_output(sample_ctx)

    def test_unknown_term_type_raises(self) -> None:
        with pytest.raises(KeyError, match="unknown term type"):
            pyine.organisms.models.rewards.core.manager.RewardManager(
                pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
                    terms=[pyine.organisms.models.rewards.core.configs.RewardTermSpec(name="x", type="does_not_exist")]
                )
            )
