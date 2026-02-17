import json
import pathlib
import types
import typing

import hydra
import hydra.core.utils
import hydra_zen
import omegaconf
import pydantic
import pytest

import pyine.apps.trainers.common as trainer_common
import pyine.configs.schemas
import pyine.data.datamodule
import pyine.evals.common
import pyine.organisms.models.rewards.core.configs as reward_configs
import pyine.organisms.models.rewards.core.logging as reward_logging
import pyine.utils.distrib
import tests.env_checks


class DummyDatamodule(pyine.data.datamodule.BaseDataModule):
    def __init__(self, stats: dict[str, int] | None = None) -> None:
        self.prepared = False
        self.setup_called = 0
        self.instantiate_verbose: list[bool] = []
        self._stats = stats or {"rows": 3}

    def prepare_data(self) -> None:
        self.prepared = True

    def setup(self) -> None:
        self.setup_called += 1

    def get_stats(
        self,
        target_subsets: typing.Any | None = None,
    ) -> dict[str, int | float | str]:
        return self._stats


class DummyDatamoduleConfig(pyine.data.datamodule.BaseDataModuleConfig):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, frozen=False)

    datamodule_class_path: str = "tests.apps.trainers.test_common.DummyDatamodule"
    default_dataparser_config: pyine.data.datamodule.BaseDataParserConfig = pydantic.Field(
        default_factory=lambda: pyine.data.datamodule.BaseDataParserConfig(class_path="torch.utils.data.Dataset")
    )
    datamodule: DummyDatamodule | None = pydantic.Field(exclude=True, default_factory=DummyDatamodule)
    calls: list[bool] = pydantic.Field(exclude=True, default_factory=list)
    train_subset_names: list[str] = pydantic.Field(default_factory=lambda: ["train"])
    valid_subset_names: list[str] = pydantic.Field(default_factory=lambda: ["valid"])
    eval_subset_names: list[str] = pydantic.Field(default_factory=lambda: ["valid"])

    def instantiate_datamodule(self, verbose: bool = False) -> DummyDatamodule:
        self.calls.append(verbose)
        self.datamodule.instantiate_verbose.append(verbose)
        return self.datamodule


class FakeEvaluationResult(pyine.evals.common.EvalResult):
    artifacts: list[str]


class DummyEvalsConfig(pyine.evals.common.BaseEvalsConfig):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, frozen=False)

    eval_type: str = "code_exec"
    eval_runnable_model_calls: list[dict] = pydantic.Field(default_factory=list)
    eval_hf_model_calls: list[dict] = pydantic.Field(default_factory=list)
    define_metrics_calls: list[dict] = pydantic.Field(default_factory=list)
    log_metrics_calls: list[dict] = pydantic.Field(default_factory=list)
    log_predictions_calls: list[dict] = pydantic.Field(default_factory=list)

    async def evaluate_runnable_model(self, **kwargs: typing.Any) -> pyine.evals.common.EvalResult:
        self.eval_runnable_model_calls.append(kwargs)
        return pyine.evals.common.EvalResult(metrics={})

    async def evaluate_hf_model(self, **kwargs: typing.Any) -> pyine.evals.common.EvalResult:
        self.eval_hf_model_calls.append(kwargs)
        return pyine.evals.common.EvalResult(metrics={})

    def define_metrics_for_wandb(self, **kwargs: typing.Any) -> None:
        self.define_metrics_calls.append(kwargs)

    def log_metrics(self, **kwargs: typing.Any) -> None:
        self.log_metrics_calls.append(kwargs)

    def log_predictions(self, **kwargs: typing.Any) -> None:
        self.log_predictions_calls.append(kwargs)


def _build_app_config(
    datamodule_config: DummyDatamoduleConfig,
    *,
    use_wandb_logging: bool = False,
    evals_config: DummyEvalsConfig | None = None,
) -> trainer_common.AppMainConfig:
    if evals_config is None:
        evals_config = DummyEvalsConfig()
    return trainer_common.AppMainConfig.model_construct(
        datamodule_config=datamodule_config,
        evals_config=evals_config,
        use_wandb_logging=use_wandb_logging,
    )


def test_prepare_datamodule_basic() -> None:
    datamodule = DummyDatamodule()
    config = _build_app_config(DummyDatamoduleConfig(datamodule=datamodule))
    result = trainer_common.prepare_datamodule(config, runtime=None)
    assert result is datamodule
    assert datamodule.prepared
    assert datamodule.setup_called == 1
    assert datamodule.instantiate_verbose == [True]


def test_prepare_datamodule_with_wandb_logging() -> None:
    datamodule = DummyDatamodule(stats={"rows": 42})
    evals_config = DummyEvalsConfig()
    config = _build_app_config(
        DummyDatamoduleConfig(datamodule=datamodule, eval_subset_names=["valid", "test"]),
        use_wandb_logging=True,
        evals_config=evals_config,
    )
    runtime = types.SimpleNamespace(
        wandb_run=types.SimpleNamespace(summary={}),
        wandb_run_id="run-123",
        finalize=lambda: None,
    )
    result = trainer_common.prepare_datamodule(config, runtime=runtime)
    assert result is datamodule
    assert runtime.wandb_run.summary["dataset_stats/rows"] == 42
    assert [call["prefix"] for call in evals_config.define_metrics_calls] == [
        "predict/valid",
        "predict/test",
    ]


@pytest.mark.asyncio
async def test_evaluate_model_sync_and_async(monkeypatch: pytest.MonkeyPatch) -> None:
    metrics_logged: list[tuple[dict[str, float], str]] = []

    def fake_print_metrics(metrics: dict[str, float], subset: str, printer: typing.Any) -> None:
        metrics_logged.append((metrics, subset))

    monkeypatch.setattr(trainer_common.pyine.evals.utils, "print_metrics", fake_print_metrics)
    synchronous_result = FakeEvaluationResult(metrics={"acc": 0.9}, artifacts=["art1"])
    asynchronous_result = FakeEvaluationResult(metrics={"acc": 0.95}, artifacts=["art2"])

    async def _async_wrapper() -> FakeEvaluationResult:
        return asynchronous_result

    async def fake_evaluate_runnable_model(
        chain: typing.Any,
        datamodule: DummyDatamodule,
        eval_subset_name: str,
        verbose: bool,
    ) -> FakeEvaluationResult:
        if eval_subset_name == "sync":
            return synchronous_result
        return await _async_wrapper()

    evals_config = DummyEvalsConfig()
    evals_config.evaluate_runnable_model = fake_evaluate_runnable_model
    datamodule = DummyDatamodule()
    config = _build_app_config(
        DummyDatamoduleConfig(
            datamodule=datamodule, subset_names=["train", "valid", "sync", "async"], eval_subset_names=["sync", "async"]
        ),
        evals_config=evals_config,
    )
    model = types.SimpleNamespace(invoke=lambda x: x)
    results = await trainer_common.evaluate_model(
        model=model,
        tokenizer=None,
        datamodule=datamodule,
        config=config,
        runtime=None,
    )
    assert results == {"sync": synchronous_result, "async": asynchronous_result}
    assert metrics_logged == [({"acc": 0.9}, "sync"), ({"acc": 0.95}, "async")]


@pytest.mark.asyncio
async def test_evaluate_model_requires_wandb_run_id() -> None:
    result = FakeEvaluationResult(metrics={"acc": 0.5}, artifacts=[])

    async def fake_evaluate_runnable_model(
        chain: typing.Any,
        datamodule: DummyDatamodule,
        eval_subset_name: str,
        verbose: bool,
    ) -> FakeEvaluationResult:
        return result

    evals_config = DummyEvalsConfig()
    evals_config.evaluate_runnable_model = fake_evaluate_runnable_model

    runtime = types.SimpleNamespace(
        wandb_run=None,
        wandb_run_id=None,
        finalize=lambda: None,
    )
    config = _build_app_config(
        DummyDatamoduleConfig(datamodule=DummyDatamodule()),
        use_wandb_logging=True,
        evals_config=evals_config,
    )
    model = types.SimpleNamespace(invoke=lambda x: x)
    with pytest.raises(RuntimeError):
        await trainer_common.evaluate_model(
            model=model,
            tokenizer=None,
            datamodule=DummyDatamodule(),
            config=config,
            runtime=runtime,
        )


def test_app_main_config_is_resuming() -> None:
    config_without_resume = trainer_common.AppMainConfig.model_construct(
        datamodule_config=DummyDatamoduleConfig(datamodule=DummyDatamodule()),
        evals_config=DummyEvalsConfig(),
    )
    assert not config_without_resume.is_resuming()
    config_with_resume = trainer_common.AppMainConfig.model_construct(
        datamodule_config=DummyDatamoduleConfig(datamodule=DummyDatamodule()),
        evals_config=DummyEvalsConfig(),
        resume_from_run_dir=pathlib.Path("/some/path"),
    )
    assert config_with_resume.is_resuming()


def test_app_main_config_normalize_for_resume_overlap_check() -> None:
    config = trainer_common.AppMainConfig.model_construct(
        datamodule_config=DummyDatamoduleConfig(datamodule=DummyDatamodule()),
        evals_config=DummyEvalsConfig(),
        use_wandb_logging=True,
        resume_from_run_dir=pathlib.Path("/some/path"),
        resume_checkpoint_name="checkpoint-100",
        resume_wandb_behavior="must",
    )
    normalized = config.normalize_for_resume_overlap_check()
    assert "use_wandb_logging" not in normalized
    assert "resume_from_run_dir" not in normalized
    assert "resume_checkpoint_name" not in normalized
    assert "resume_wandb_behavior" not in normalized
    assert "auto_resume_if_possible" not in normalized
    assert "datamodule_config" in normalized
    assert "evals_config" in normalized


def test_app_main_config_validate_bad_resume_settings(tmp_path: pathlib.Path) -> None:
    non_existent_dir = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError, match="invalid resume directory"):
        trainer_common.AppMainConfig(
            datamodule_config=DummyDatamoduleConfig(datamodule=DummyDatamodule()),
            evals_config=DummyEvalsConfig(),
            resume_from_run_dir=non_existent_dir,
        )
    run_dir = tmp_path / "run_dir"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="specified resume checkpoint not found"):
        trainer_common.AppMainConfig(
            datamodule_config=DummyDatamoduleConfig(datamodule=DummyDatamodule()),
            evals_config=DummyEvalsConfig(),
            resume_from_run_dir=run_dir,
            resume_checkpoint_name="checkpoint-100",
        )
    with pytest.raises(ValueError, match="resume_checkpoint_name requires resume_from_run_dir"):
        trainer_common.AppMainConfig(
            datamodule_config=DummyDatamoduleConfig(datamodule=DummyDatamodule()),
            evals_config=DummyEvalsConfig(),
            resume_checkpoint_name="checkpoint-100",
        )


def test_app_main_config_validate_resume_settings_valid(tmp_path: pathlib.Path) -> None:
    run_dir = tmp_path / "run_dir"
    run_dir.mkdir()
    checkpoint_dir = run_dir / "checkpoint-100"
    checkpoint_dir.mkdir()
    config = trainer_common.AppMainConfig(
        datamodule_config=DummyDatamoduleConfig(datamodule=DummyDatamodule()),
        evals_config=DummyEvalsConfig(),
        resume_from_run_dir=run_dir,
        resume_checkpoint_name="checkpoint-100",
    )
    assert config.resume_from_run_dir == run_dir
    assert config.resume_checkpoint_name == "checkpoint-100"


def test_resume_artifacts_find_first_matching_path(tmp_path: pathlib.Path) -> None:
    result = trainer_common.ResumeArtifacts._find_first_matching_path(tmp_path, "*.json")
    assert result is None
    (tmp_path / "config.b.rank0.json").write_text("{}")
    (tmp_path / "config.a.rank0.json").write_text("{}")
    result = trainer_common.ResumeArtifacts._find_first_matching_path(tmp_path, "config.*.rank*.json")
    assert result is not None
    assert result.name == "config.a.rank0.json"


def test_resume_artifacts_load_json_if_exists(tmp_path: pathlib.Path) -> None:
    result = trainer_common.ResumeArtifacts._load_json_if_exists(None)
    assert result == {}
    json_path = tmp_path / "test.json"
    json_path.write_text('{"key": "value"}')
    result = trainer_common.ResumeArtifacts._load_json_if_exists(json_path)
    assert result == {"key": "value"}


def test_resume_artifacts_ensure_resume_config_matches_identical() -> None:
    config1 = trainer_common.AppMainConfig.model_construct(
        datamodule_config=DummyDatamoduleConfig(datamodule=DummyDatamodule()),
        evals_config=DummyEvalsConfig(),
        use_wandb_logging=True,
    )
    config2 = trainer_common.AppMainConfig.model_construct(
        datamodule_config=DummyDatamoduleConfig(datamodule=DummyDatamodule()),
        evals_config=DummyEvalsConfig(),
        use_wandb_logging=False,  # different but should be ignored
    )
    trainer_common.ResumeArtifacts._ensure_resume_config_matches(config1, config2)


def test_resume_artifacts_ensure_resume_config_matches_different() -> None:
    config1 = trainer_common.AppMainConfig.model_construct(
        datamodule_config=DummyDatamoduleConfig(eval_subset_names=["valid"]),
        evals_config=DummyEvalsConfig(),
    )
    config2 = trainer_common.AppMainConfig.model_construct(
        datamodule_config=DummyDatamoduleConfig(eval_subset_names=["test"]),
        evals_config=DummyEvalsConfig(),
    )
    with pytest.raises(ValueError, match="resume configuration mismatch detected"):
        trainer_common.ResumeArtifacts._ensure_resume_config_matches(config1, config2)


def test_resume_artifacts_extract_wandb_resume_kwargs() -> None:
    config = trainer_common.AppMainConfig.model_construct(
        datamodule_config=DummyDatamoduleConfig(),
        evals_config=DummyEvalsConfig(),
        use_wandb_logging=False,
    )
    runtime_dict = {"wandb_run_id": "test-id"}
    result = trainer_common.ResumeArtifacts._extract_wandb_resume_kwargs(runtime_dict, config)
    assert result == {}

    config = trainer_common.AppMainConfig.model_construct(
        datamodule_config=DummyDatamoduleConfig(),
        evals_config=DummyEvalsConfig(),
        use_wandb_logging=True,
        resume_wandb_behavior="never",
    )
    runtime_dict = {"wandb_run_id": "test-id"}
    result = trainer_common.ResumeArtifacts._extract_wandb_resume_kwargs(runtime_dict, config)
    assert result == {}

    config = trainer_common.AppMainConfig.model_construct(
        datamodule_config=DummyDatamoduleConfig(),
        evals_config=DummyEvalsConfig(),
        use_wandb_logging=True,
        resume_wandb_behavior="must",
    )
    runtime_dict: dict[str, typing.Any] = {}
    with pytest.raises(RuntimeError, match="original run did NOT have wandb logging enabled"):
        trainer_common.ResumeArtifacts._extract_wandb_resume_kwargs(runtime_dict, config)

    config = trainer_common.AppMainConfig.model_construct(
        datamodule_config=DummyDatamoduleConfig(),
        evals_config=DummyEvalsConfig(),
        use_wandb_logging=True,
        resume_wandb_behavior="allow",
    )
    runtime_dict: dict[str, typing.Any] = {}
    result = trainer_common.ResumeArtifacts._extract_wandb_resume_kwargs(runtime_dict, config)
    assert result == {}

    config = trainer_common.AppMainConfig.model_construct(
        datamodule_config=DummyDatamoduleConfig(),
        evals_config=DummyEvalsConfig(),
        use_wandb_logging=True,
        resume_wandb_behavior="allow",
    )
    runtime_dict = {
        "wandb_run_id": "test-run-id",
        "wandb_run_project": "test-project",
        "wandb_run_entity": "test-entity",
    }
    result = trainer_common.ResumeArtifacts._extract_wandb_resume_kwargs(runtime_dict, config)
    assert result == {
        "project": "test-project",
        "entity": "test-entity",
        "id": "test-run-id",
        "resume": "allow",
    }


def test_resume_artifacts_create_bad_args(tmp_path: pathlib.Path) -> None:
    config = trainer_common.AppMainConfig.model_construct(
        datamodule_config=DummyDatamoduleConfig(),
        evals_config=DummyEvalsConfig(),
    )
    with pytest.raises(ValueError, match="resume_from_run_dir must be provided"):
        trainer_common.ResumeArtifacts.create(config)

    run_dir = tmp_path / "run_dir"
    run_dir.mkdir()
    config = trainer_common.AppMainConfig.model_construct(
        datamodule_config=DummyDatamoduleConfig(),
        evals_config=DummyEvalsConfig(),
        resume_from_run_dir=run_dir,
    )
    with pytest.raises(FileNotFoundError, match="could not locate logged config file"):
        trainer_common.ResumeArtifacts.create(config)

    config_file = run_dir / "config.test.rank0.json"
    config_file.write_text(json.dumps({"other_field": "value"}))
    config = trainer_common.AppMainConfig.model_construct(
        datamodule_config=DummyDatamoduleConfig(),
        evals_config=DummyEvalsConfig(),
        resume_from_run_dir=run_dir,
    )
    with pytest.raises(ValueError, match="resume directory is missing the logged main_config payload"):
        trainer_common.ResumeArtifacts.create(config)


def test_resume_artifacts_create_no_checkpoint_found(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "run_dir"
    run_dir.mkdir()
    # Create a config that matches what we're using to resume
    dm_config = DummyDatamoduleConfig()
    evals_config = DummyEvalsConfig()
    config_payload = {
        "main_config": {
            "datamodule_config": dm_config.model_dump(mode="json"),
            "evals_config": evals_config.model_dump(mode="json"),
        }
    }
    config_file = run_dir / "config.test.rank0.json"
    config_file.write_text(json.dumps(config_payload))
    config = trainer_common.AppMainConfig.model_construct(
        datamodule_config=dm_config,
        evals_config=evals_config,
        resume_from_run_dir=run_dir,
    )
    monkeypatch.setattr("transformers.trainer_utils.get_last_checkpoint", lambda x: None)
    with pytest.raises(FileNotFoundError, match="no checkpoint found under resume directory"):
        trainer_common.ResumeArtifacts.create(config)


def test_resume_artifacts_create_missing_trainer_state(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "run_dir"
    run_dir.mkdir()
    checkpoint_dir = run_dir / "checkpoint-100"
    checkpoint_dir.mkdir()
    dm_config = DummyDatamoduleConfig()
    evals_config = DummyEvalsConfig()
    config_payload = {
        "main_config": {
            "datamodule_config": dm_config.model_dump(mode="json"),
            "evals_config": evals_config.model_dump(mode="json"),
        }
    }
    config_file = run_dir / "config.test.rank0.json"
    config_file.write_text(json.dumps(config_payload))
    config = trainer_common.AppMainConfig.model_construct(
        datamodule_config=dm_config,
        evals_config=evals_config,
        resume_from_run_dir=run_dir,
    )
    monkeypatch.setattr("transformers.trainer_utils.get_last_checkpoint", lambda x: str(checkpoint_dir))
    with pytest.raises(FileNotFoundError, match="invalid checkpoint directory.*missing trainer_state.json"):
        trainer_common.ResumeArtifacts.create(config)


def test_resume_artifacts_create_success(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "run_dir"
    run_dir.mkdir()
    checkpoint_dir = run_dir / "checkpoint-100"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "trainer_state.json").write_text("{}")
    dm_config = DummyDatamoduleConfig()
    evals_config = DummyEvalsConfig()
    config_payload = {
        "main_config": {
            "datamodule_config": dm_config.model_dump(mode="json"),
            "evals_config": evals_config.model_dump(mode="json"),
        }
    }
    config_file = run_dir / "config.test.rank0.json"
    config_file.write_text(json.dumps(config_payload))
    runtime_file = run_dir / "runtime.test.rank0.json"
    runtime_file.write_text(
        json.dumps({"wandb_run_id": "test-id", "wandb_run_project": "proj", "wandb_run_entity": "ent"})
    )
    metadata_file = run_dir / "reprod_metadata.test.rank0.json"
    metadata_file.write_text(json.dumps({"seed": 42}))
    config = trainer_common.AppMainConfig.model_construct(
        datamodule_config=dm_config,
        evals_config=evals_config,
        resume_from_run_dir=run_dir,
        use_wandb_logging=True,
        resume_wandb_behavior="allow",
    )
    monkeypatch.setattr("transformers.trainer_utils.get_last_checkpoint", lambda x: str(checkpoint_dir))
    artifacts = trainer_common.ResumeArtifacts.create(config)
    assert artifacts.run_dir == run_dir
    assert artifacts.checkpoint_path == checkpoint_dir
    assert artifacts.previous_config_path == config_file
    assert artifacts.previous_runtime_path == runtime_file
    assert artifacts.previous_metadata_path == metadata_file
    assert artifacts.previous_runtime_dict == {
        "wandb_run_id": "test-id",
        "wandb_run_project": "proj",
        "wandb_run_entity": "ent",
    }
    assert artifacts.previous_metadata_dict == {"seed": 42}
    assert artifacts.wandb_resume_kwargs["id"] == "test-id"


def test_prepare_resume_artifacts_not_resuming() -> None:
    config = trainer_common.AppMainConfig.model_construct(
        datamodule_config=DummyDatamoduleConfig(),
        evals_config=DummyEvalsConfig(),
    )
    result = trainer_common.prepare_resume_artifacts(config, runtime=None)
    assert result is None


def test_prepare_resume_artifacts_success(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "run_dir"
    run_dir.mkdir()
    checkpoint_dir = run_dir / "checkpoint-100"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "trainer_state.json").write_text("{}")
    dm_config = DummyDatamoduleConfig()
    evals_config = DummyEvalsConfig()
    config_payload = {
        "main_config": {
            "datamodule_config": dm_config.model_dump(mode="json"),
            "evals_config": evals_config.model_dump(mode="json"),
        }
    }
    config_file = run_dir / "config.test.rank0.json"
    config_file.write_text(json.dumps(config_payload))
    runtime_file = run_dir / "runtime.test.rank0.json"
    runtime_file.write_text(json.dumps({"key": "value"}))
    metadata_file = run_dir / "reprod_metadata.test.rank0.json"
    metadata_file.write_text(json.dumps({"seed": 42}))
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    runtime = types.SimpleNamespace(
        output_dir_path=output_dir,
        metadata={},
    )
    config = trainer_common.AppMainConfig.model_construct(
        datamodule_config=dm_config,
        evals_config=evals_config,
        resume_from_run_dir=run_dir,
    )
    monkeypatch.setattr("transformers.trainer_utils.get_last_checkpoint", lambda x: str(checkpoint_dir))
    artifacts = trainer_common.prepare_resume_artifacts(config, runtime)
    assert artifacts is not None
    assert artifacts.checkpoint_path == checkpoint_dir
    assert runtime.metadata["resumed_from_run_dir"] == str(run_dir)
    assert runtime.metadata["resume_checkpoint_path"] == str(checkpoint_dir)
    assert (output_dir / "previous_config.json").exists()
    assert (output_dir / "previous_runtime.json").exists()
    assert (output_dir / "previous_reprod_metadata.json").exists()


def test_prepare_resume_artifacts_auto_resume_success(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run_dir"
    run_dir.mkdir()
    checkpoint_dir = run_dir / "checkpoint-100"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "trainer_state.json").write_text("{}")
    dm_config = DummyDatamoduleConfig()
    evals_config = DummyEvalsConfig()
    config_payload = {
        "main_config": {
            "datamodule_config": dm_config.model_dump(mode="json"),
            "evals_config": evals_config.model_dump(mode="json"),
        }
    }
    config_file = run_dir / "config.test.rank0.json"
    config_file.write_text(json.dumps(config_payload))
    runtime_file = run_dir / "runtime.test.rank0.json"
    runtime_file.write_text(json.dumps({"key": "value"}))
    metadata_file = run_dir / "reprod_metadata.test.rank0.json"
    metadata_file.write_text(json.dumps({"seed": 42}))
    runtime = types.SimpleNamespace(
        output_dir_path=run_dir,
        metadata={},
    )
    config = trainer_common.AppMainConfig.model_construct(
        datamodule_config=dm_config,
        evals_config=evals_config,
        auto_resume_if_possible=True,
    )
    monkeypatch.setattr("transformers.trainer_utils.get_last_checkpoint", lambda x: str(checkpoint_dir))
    artifacts = trainer_common.prepare_resume_artifacts(config, runtime)
    assert artifacts is not None
    assert config.resume_from_run_dir == run_dir
    assert artifacts.checkpoint_path == checkpoint_dir
    assert runtime.metadata["resumed_from_run_dir"] == str(run_dir)
    assert runtime.metadata["resume_checkpoint_path"] == str(checkpoint_dir)
    assert (run_dir / "previous_config.json").exists()
    assert (run_dir / "previous_runtime.json").exists()
    assert (run_dir / "previous_reprod_metadata.json").exists()


def test_prepare_resume_artifacts_auto_resume_no_checkpoint(tmp_path: pathlib.Path) -> None:
    run_dir = tmp_path / "run_dir"
    run_dir.mkdir()
    runtime = types.SimpleNamespace(
        output_dir_path=run_dir,
        metadata={},
    )
    config = trainer_common.AppMainConfig.model_construct(
        datamodule_config=DummyDatamoduleConfig(),
        evals_config=DummyEvalsConfig(),
        auto_resume_if_possible=True,
    )
    result = trainer_common.prepare_resume_artifacts(config, runtime)
    assert result is None
    assert config.resume_from_run_dir is None
    assert config.resume_checkpoint_name is None


def test_prepare_resume_artifacts_auto_resume_config_mismatch(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run_dir"
    run_dir.mkdir()
    checkpoint_dir = run_dir / "checkpoint-100"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "trainer_state.json").write_text("{}")
    previous_dm_config = DummyDatamoduleConfig(valid_subset_names=["test"], eval_subset_names=["test"])
    previous_evals_config = DummyEvalsConfig()
    config_payload = {
        "main_config": {
            "datamodule_config": previous_dm_config.model_dump(mode="json"),
            "evals_config": previous_evals_config.model_dump(mode="json"),
        }
    }
    config_file = run_dir / "config.test.rank0.json"
    config_file.write_text(json.dumps(config_payload))
    runtime_file = run_dir / "runtime.test.rank0.json"
    runtime_file.write_text(json.dumps({}))
    metadata_file = run_dir / "reprod_metadata.test.rank0.json"
    metadata_file.write_text(json.dumps({}))
    runtime = types.SimpleNamespace(
        output_dir_path=run_dir,
        metadata={},
    )
    config = trainer_common.AppMainConfig.model_construct(
        datamodule_config=DummyDatamoduleConfig(),
        evals_config=DummyEvalsConfig(),
        auto_resume_if_possible=True,
    )
    monkeypatch.setattr("transformers.trainer_utils.get_last_checkpoint", lambda x: str(checkpoint_dir))
    with pytest.raises(ValueError, match="resume configuration mismatch detected"):
        trainer_common.prepare_resume_artifacts(config, runtime)


@pytest.mark.integration
@pytest.mark.skipif(
    tests.env_checks.WANDB_API_KEY_MISSING or tests.env_checks.NETWORK_UNAVAILABLE,
    reason="WandB API key or network not available; cannot check sweeps integration",
)
def test_validate_wandb_sweeper_requirements(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # note: sweeps do NOT offer an offline mode, so stuff gets logged online: runs land in a test-specific project
    monkeypatch.setenv("WANDB_PROJECT", "pyine-tests")
    validation_called = []

    def test_task(cfg: omegaconf.DictConfig) -> dict[str, bool]:
        """Task function that tests the validation."""
        assert cfg.dummy_param in [1, 2, 3]
        app_config = _build_app_config(DummyDatamoduleConfig(), use_wandb_logging=cfg.use_wandb_logging)
        trainer_common.validate_wandb_sweeper_requirements(app_config)
        validation_called.append(True)
        return {"success": True}

    test_task_config = hydra_zen.make_config(
        use_wandb_logging=False,
        dummy_param=0,  # this default should always be overridden in the sweeps below
    )
    # with basic sweeper, wandb logging is not required even in multirun
    sweep_dir1 = tmp_path / "sweep1"
    job_returns = hydra_zen.launch(
        config=test_task_config,
        task_function=test_task,
        overrides=[
            "use_wandb_logging=false",
            "hydra.mode=MULTIRUN",
            f"hydra.sweep.dir={sweep_dir1}",
            "dummy_param=1,2,3",
        ],
        multirun=True,
        version_base="1.3",
    )
    # should complete successfully:
    assert len(job_returns) == 1  # list of list of job outputs (peculiarity of regular sweeps...)
    assert len(job_returns[0]) == 3  # three expected outputs for that grid sweep
    assert all(jr.return_value["success"] for jr in job_returns[0])
    assert len(validation_called) == 3
    validation_called.clear()

    # with wandb sweeper, if we don't setup wandb logging, we get an exception
    sweep_dir2 = tmp_path / "sweep2"
    with pytest.raises(Exception):  # noqa: B017
        _ = hydra_zen.launch(
            config=test_task_config,
            task_function=test_task,
            overrides=[
                "use_wandb_logging=false",
                "hydra.mode=MULTIRUN",
                "hydra/sweeper=wandb",  # use sweep config from plugin directly
                f"hydra.sweep.dir={sweep_dir2}",
                "hydra.sweeper.wandb_sweep_config.name=some_sweep",
                "hydra.sweeper.wandb_sweep_config.method=grid",
                "hydra.sweeper.wandb_sweep_config.budget=3",
                "+hydra.sweeper.params.dummy_param=[1,2,3]",
            ],
            multirun=True,
            version_base="1.3",
        )
    assert len(validation_called) == 0

    # however, if we do set up wandb logging, all should be OK
    sweep_dir3 = tmp_path / "sweep3"
    job_returns = hydra_zen.launch(
        config=test_task_config,
        task_function=test_task,
        overrides=[
            "use_wandb_logging=true",
            "hydra.mode=MULTIRUN",
            "hydra/sweeper=wandb",  # use sweep config from plugin directly
            f"hydra.sweep.dir={sweep_dir3}",
            "hydra.sweeper.wandb_sweep_config.name=some_sweep",
            "hydra.sweeper.wandb_sweep_config.method=grid",
            "hydra.sweeper.wandb_sweep_config.budget=3",
            "+hydra.sweeper.params.dummy_param=[1,2,3]",
        ],
        multirun=True,
        version_base="1.3",
    )
    assert len(job_returns) == 3  # list of job outputs
    assert all(jr.status == hydra.core.utils.JobStatus.COMPLETED for jr in job_returns)
    assert len(validation_called) == 3


def test_flash_attention_fallback_to_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that flash_attention_2 falls back to sdpa when unavailable."""
    monkeypatch.setattr(
        trainer_common.transformers.utils,
        "is_flash_attn_2_available",
        lambda: False,
    )
    auto_config = {"attn_implementation": "flash_attention_2", "use_cache": False}
    resolved = trainer_common._resolve_attn_implementation(auto_config)
    assert resolved["attn_implementation"] == "sdpa"
    assert resolved["use_cache"] is False


def test_flash_attention_kept_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that flash_attention_2 is kept when available."""
    monkeypatch.setattr(
        trainer_common.transformers.utils,
        "is_flash_attn_2_available",
        lambda: True,
    )
    auto_config = {"attn_implementation": "flash_attention_2"}
    resolved = trainer_common._resolve_attn_implementation(auto_config)
    assert resolved["attn_implementation"] == "flash_attention_2"


def test_resolve_attn_implementation_no_change_for_other_impl() -> None:
    """Test that non-flash_attention_2 implementations are not modified."""
    auto_config = {"attn_implementation": "sdpa", "use_cache": True}
    resolved = trainer_common._resolve_attn_implementation(auto_config)
    assert resolved == auto_config


def test_resolve_attn_implementation_empty_config() -> None:
    """Test that empty config returns empty config."""
    auto_config: dict[str, typing.Any] = {}
    resolved = trainer_common._resolve_attn_implementation(auto_config)
    assert resolved == {}


def _make_reward_manager_config(
    *,
    logging_enabled: bool = True,
    log_total: bool = True,
    main_process_only: bool = True,
) -> reward_configs.RewardManagerConfig:
    return reward_configs.RewardManagerConfig(
        terms=[reward_configs.RewardTermSpec(name="dummy", type="exact_match")],
        logging=reward_configs.LoggingConfig(
            enabled=logging_enabled,
            log_total=log_total,
            main_process_only=main_process_only,
        ),
    )


class TestCreateModelOrganismRewardComponents:
    def test_returns_components_without_export(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_manager = types.SimpleNamespace()
        mock_adapter = types.SimpleNamespace()
        monkeypatch.setattr(
            trainer_common.reward_manager_mod,
            "RewardManager",
            lambda *args, **kwargs: mock_manager,
        )
        monkeypatch.setattr(
            trainer_common.reward_trl,
            "TRLRewardAdapter",
            lambda *args, **kwargs: mock_adapter,
        )
        config = _make_reward_manager_config()
        result = trainer_common.create_model_organism_reward_components(
            reward_manager_config=config,
            tokenizer=None,
            generation_export_config=None,
            wandb_run=None,
        )
        assert result.manager is mock_manager
        assert result.adapter is mock_adapter
        assert result._disk_logger is None
        result.close()  # should be a safe no-op

    def test_returns_disk_logger_when_export_configured(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_manager = types.SimpleNamespace()
        mock_adapter = types.SimpleNamespace()
        monkeypatch.setattr(
            trainer_common.reward_manager_mod,
            "RewardManager",
            lambda *args, **kwargs: mock_manager,
        )
        monkeypatch.setattr(
            trainer_common.reward_trl,
            "TRLRewardAdapter",
            lambda *args, **kwargs: mock_adapter,
        )
        export_path = tmp_path / "export_lmdb"
        export_config = reward_configs.GenerationExportConfig(output_path=export_path)
        config = _make_reward_manager_config()
        result = trainer_common.create_model_organism_reward_components(
            reward_manager_config=config,
            tokenizer=None,
            generation_export_config=export_config,
            wandb_run=None,
        )
        assert result._disk_logger is not None
        result.close()

    def test_raises_when_export_with_logging_disabled(self) -> None:
        config = _make_reward_manager_config(logging_enabled=False)
        export_config = reward_configs.GenerationExportConfig(output_path=pathlib.Path("/nonexistent/dummy/export"))
        with pytest.raises(ValueError, match="logging.enabled=False"):
            trainer_common.create_model_organism_reward_components(
                reward_manager_config=config,
                tokenizer=None,
                generation_export_config=export_config,
            )

    def test_raises_when_export_with_log_total_false(self) -> None:
        config = _make_reward_manager_config(log_total=False)
        export_config = reward_configs.GenerationExportConfig(output_path=pathlib.Path("/nonexistent/dummy/export"))
        with pytest.raises(ValueError, match="log_total=False"):
            trainer_common.create_model_organism_reward_components(
                reward_manager_config=config,
                tokenizer=None,
                generation_export_config=export_config,
            )

    def test_raises_when_export_no_all_ranks_and_main_process_only_false(self) -> None:
        config = _make_reward_manager_config(main_process_only=False)
        export_config = reward_configs.GenerationExportConfig(output_path=pathlib.Path("/nonexistent/dummy/export"))
        with pytest.raises(ValueError, match="export_all_ranks=False"):
            trainer_common.create_model_organism_reward_components(
                reward_manager_config=config,
                tokenizer=None,
                generation_export_config=export_config,
            )

    def test_export_all_ranks_on_non_main_creates_disk_logger(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """export_all_ranks=True on non-main rank -> disk logger created with rank-specific path."""
        monkeypatch.setattr(pyine.utils.distrib, "is_main_process", lambda rank=None: False)
        monkeypatch.setattr(pyine.utils.distrib, "has_explicit_global_rank", lambda: True)
        monkeypatch.setattr(pyine.utils.distrib, "get_global_rank", lambda default=None: 2)
        mock_manager = types.SimpleNamespace()
        mock_adapter = types.SimpleNamespace()
        monkeypatch.setattr(
            trainer_common.reward_manager_mod,
            "RewardManager",
            lambda *args, **kwargs: mock_manager,
        )
        monkeypatch.setattr(
            trainer_common.reward_trl,
            "TRLRewardAdapter",
            lambda *args, **kwargs: mock_adapter,
        )
        export_path = tmp_path / "export_lmdb"
        export_config = reward_configs.GenerationExportConfig(
            output_path=export_path,
            export_all_ranks=True,
        )
        config = _make_reward_manager_config()
        result = trainer_common.create_model_organism_reward_components(
            reward_manager_config=config,
            tokenizer=None,
            generation_export_config=export_config,
            wandb_run=None,
        )
        assert result._disk_logger is not None
        result.close()
        assert (export_path / "rank_2").exists()

    def test_export_all_ranks_on_rank0_creates_composite(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """export_all_ranks=True on rank 0 with wandb -> composite wrapping WandB + Disk."""
        monkeypatch.setattr(pyine.utils.distrib, "is_main_process", lambda rank=None: True)
        monkeypatch.setattr(pyine.utils.distrib, "has_explicit_global_rank", lambda: True)
        monkeypatch.setattr(pyine.utils.distrib, "get_global_rank", lambda default=None: 0)
        mock_manager = types.SimpleNamespace()
        mock_adapter = types.SimpleNamespace()
        captured_kwargs: dict = {}

        def capture_manager(*args: object, **kwargs: object) -> types.SimpleNamespace:
            captured_kwargs.update(kwargs)
            return mock_manager

        monkeypatch.setattr(trainer_common.reward_manager_mod, "RewardManager", capture_manager)
        monkeypatch.setattr(
            trainer_common.reward_trl,
            "TRLRewardAdapter",
            lambda *args, **kwargs: mock_adapter,
        )
        export_path = tmp_path / "export_lmdb"
        export_config = reward_configs.GenerationExportConfig(
            output_path=export_path,
            export_all_ranks=True,
        )
        config = _make_reward_manager_config()
        mock_wandb_run = types.SimpleNamespace(log=lambda *a, **kw: None)
        result = trainer_common.create_model_organism_reward_components(
            reward_manager_config=config,
            tokenizer=None,
            generation_export_config=export_config,
            wandb_run=mock_wandb_run,
        )
        # logger passed to manager should be a composite
        logger_arg = captured_kwargs.get("logger")
        assert isinstance(logger_arg, reward_logging.CompositeRewardLogger)
        result.close()

    def test_export_all_ranks_ambiguous_distributed_raises(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """export_all_ranks=True + LOCAL_RANK=0 + no explicit global rank -> ValueError."""
        monkeypatch.setattr(pyine.utils.distrib, "is_main_process", lambda rank=None: False)
        monkeypatch.setattr(pyine.utils.distrib, "has_explicit_global_rank", lambda: False)
        monkeypatch.setattr(pyine.utils.distrib, "get_local_rank", lambda default=None: 0)
        mock_manager = types.SimpleNamespace()
        mock_adapter = types.SimpleNamespace()
        monkeypatch.setattr(
            trainer_common.reward_manager_mod,
            "RewardManager",
            lambda *args, **kwargs: mock_manager,
        )
        monkeypatch.setattr(
            trainer_common.reward_trl,
            "TRLRewardAdapter",
            lambda *args, **kwargs: mock_adapter,
        )
        export_path = tmp_path / "export_lmdb"
        export_config = reward_configs.GenerationExportConfig(
            output_path=export_path,
            export_all_ranks=True,
        )
        config = _make_reward_manager_config()
        with pytest.raises(ValueError, match="authoritative global rank"):
            trainer_common.create_model_organism_reward_components(
                reward_manager_config=config,
                tokenizer=None,
                generation_export_config=export_config,
            )
