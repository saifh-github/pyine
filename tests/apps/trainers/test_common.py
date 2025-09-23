import types

import pytest

import pyine.apps.trainers.common as trainer_common


class DummyDatamodule:

    def __init__(self, stats: dict[str, int] | None = None):
        self.prepared = False
        self.setup_called = 0
        self.instantiate_verbose: list[bool] = []
        self._stats = stats or {"rows": 3}

    def prepare_data(self) -> None:
        self.prepared = True

    def setup(self) -> None:
        self.setup_called += 1

    def get_stats(self) -> dict[str, int]:
        return self._stats


class DummyDatamoduleConfig:

    def __init__(self, datamodule: DummyDatamodule):
        self.datamodule = datamodule
        self.calls: list[bool] = []

    def instantiate_datamodule(self, verbose: bool = False) -> DummyDatamodule:
        self.calls.append(verbose)
        self.datamodule.instantiate_verbose.append(verbose)
        return self.datamodule


class FakeEvaluationResult:

    def __init__(self, metrics: dict[str, float], artifacts: list[str]):
        self.metrics = metrics
        self.artifacts = artifacts


def _build_app_config(
    datamodule_config: DummyDatamoduleConfig,
    *,
    use_wandb_logging: bool = False,
    eval_subset_names: list[str] | None = None,
) -> trainer_common.AppMainConfig:
    return trainer_common.AppMainConfig.model_construct(
        datamodule_config=datamodule_config,
        llm_grader_provider_config=None,
        eval_subset_names=eval_subset_names or ["valid"],
        use_wandb_logging=use_wandb_logging,
    )


def test_prepare_code_exec_datamodule_basic():
    datamodule = DummyDatamodule()
    config = _build_app_config(DummyDatamoduleConfig(datamodule))
    result = trainer_common.prepare_code_exec_datamodule(config, runtime=None)
    assert result is datamodule
    assert datamodule.prepared
    assert datamodule.setup_called == 1
    assert datamodule.instantiate_verbose == [True]


def test_prepare_code_exec_datamodule_with_wandb_logging(monkeypatch: pytest.MonkeyPatch):
    datamodule = DummyDatamodule(stats={"rows": 42})
    config = _build_app_config(
        DummyDatamoduleConfig(datamodule),
        use_wandb_logging=True,
        eval_subset_names=["valid", "test"],
    )
    define_calls: list[dict[str, str]] = []

    def fake_define_metrics_for_wandb(**kwargs):
        define_calls.append(kwargs)

    monkeypatch.setattr(
        trainer_common.pyine.evals.common,
        "define_metrics_for_wandb",
        fake_define_metrics_for_wandb,
    )
    runtime = types.SimpleNamespace(
        wandb_run=types.SimpleNamespace(summary={}),
        wandb_run_id="run-123",
    )
    result = trainer_common.prepare_code_exec_datamodule(config, runtime=runtime)
    assert result is datamodule
    assert runtime.wandb_run.summary["dataset_stats/rows"] == 42
    assert [call["prefix"] for call in define_calls] == ["evals/valid", "evals/test"]


@pytest.mark.asyncio
async def test_evaluate_code_execution_model_sync_and_async(monkeypatch: pytest.MonkeyPatch):
    metrics_logged: list[tuple[dict[str, float], str]] = []

    def fake_print_metrics(metrics, subset, printer):
        metrics_logged.append((metrics, subset))

    monkeypatch.setattr(trainer_common.pyine.evals.utils, "print_metrics", fake_print_metrics)
    synchronous_result = FakeEvaluationResult({"acc": 0.9}, ["art1"])
    asynchronous_result = FakeEvaluationResult({"acc": 0.95}, ["art2"])

    async def _async_wrapper():
        return asynchronous_result

    def eval_callback(eval_subset_name, datamodule, llm_grader_provider_config):
        if eval_subset_name == "sync":
            return synchronous_result
        return _async_wrapper()

    datamodule = DummyDatamodule()
    config = _build_app_config(DummyDatamoduleConfig(datamodule), eval_subset_names=["sync", "async"])
    results = await trainer_common.evaluate_code_execution_model(
        eval_callback=eval_callback,
        datamodule=datamodule,
        config=config,
        runtime=None,
    )
    assert results == {"sync": synchronous_result, "async": asynchronous_result}
    assert metrics_logged == [({"acc": 0.9}, "sync"), ({"acc": 0.95}, "async")]


@pytest.mark.asyncio
async def test_evaluate_code_execution_model_wandb_logging_reopens_run(monkeypatch: pytest.MonkeyPatch):
    logged_metrics: list[tuple] = []
    logged_samples: list[tuple] = []

    monkeypatch.setattr(trainer_common.pyine.evals.utils, "print_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        trainer_common.pyine.evals.common,
        "log_eval_metrics_table",
        lambda **kwargs: logged_metrics.append(kwargs),
    )
    monkeypatch.setattr(
        trainer_common.pyine.evals.common,
        "log_sample_predictions_table",
        lambda **kwargs: logged_samples.append(kwargs),
    )

    replacement_run = types.SimpleNamespace(summary={}, logged=[])  # new run returned by wandb.init

    def fake_wandb_init(**kwargs):
        replacement_run.log = lambda payload: replacement_run.logged.append(payload)
        return replacement_run

    monkeypatch.setattr(trainer_common.wandb, "init", fake_wandb_init)

    result = FakeEvaluationResult(metrics={"acc": 0.77}, artifacts=["sample"])

    def eval_callback(*_args, **_kwargs):
        return result

    runtime = types.SimpleNamespace(
        wandb_run=types.SimpleNamespace(summary={}),  # missing log attr triggers reopen
        wandb_run_id="run-42",
    )
    config = _build_app_config(
        DummyDatamoduleConfig(DummyDatamodule()),
        use_wandb_logging=True,
        eval_subset_names=["subset"],
    )

    results = await trainer_common.evaluate_code_execution_model(
        eval_callback=eval_callback,
        datamodule=DummyDatamodule(),
        config=config,
        runtime=runtime,
    )

    assert results["subset"] is result
    assert runtime.wandb_run is replacement_run
    assert replacement_run.summary["evals/subset/acc"] == 0.77
    assert logged_metrics and logged_samples


@pytest.mark.asyncio
async def test_evaluate_code_execution_model_requires_wandb_run_id():
    result = FakeEvaluationResult(metrics={"acc": 0.5}, artifacts=[])

    def eval_callback(*_args, **_kwargs):
        return result

    runtime = types.SimpleNamespace(wandb_run=types.SimpleNamespace(summary={}), wandb_run_id=None)
    config = _build_app_config(
        DummyDatamoduleConfig(DummyDatamodule()),
        use_wandb_logging=True,
    )

    with pytest.raises(RuntimeError):
        await trainer_common.evaluate_code_execution_model(
            eval_callback=eval_callback,
            datamodule=DummyDatamodule(),
            config=config,
            runtime=runtime,
        )
