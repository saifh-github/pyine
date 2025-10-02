import types
import typing

import pytest

import pyine.apps.trainers.common as trainer_common


class DummyDatamodule:
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


class DummyDatamoduleConfig:
    def __init__(
        self,
        datamodule: DummyDatamodule,
        eval_subset_names: list[str] | None = None,
    ) -> None:
        self.datamodule = datamodule
        self.calls: list[bool] = []
        self.train_subset_names = ["train"]
        self.valid_subset_names = ["valid"]
        self.eval_subset_names = eval_subset_names or ["valid"]

    def instantiate_datamodule(self, verbose: bool = False) -> DummyDatamodule:
        self.calls.append(verbose)
        self.datamodule.instantiate_verbose.append(verbose)
        return self.datamodule


class FakeEvaluationResult:
    def __init__(self, metrics: dict[str, float], artifacts: list[str]) -> None:
        self.metrics = metrics
        self.artifacts = artifacts


class DummyEvalsConfig:
    def __init__(self, eval_type: typing.Any) -> None:
        self.eval_type = eval_type
        self.eval_runnable_model_calls: list[dict] = []
        self.eval_hf_model_calls: list[dict] = []
        self.define_metrics_calls: list[dict] = []
        self.log_metrics_calls: list[dict] = []
        self.log_predictions_calls: list[dict] = []

    async def evaluate_runnable_model(self, **kwargs: typing.Any) -> dict[str, typing.Any]:
        self.eval_runnable_model_calls.append(kwargs)
        return {}

    async def evaluate_hf_model(self, **kwargs: typing.Any) -> dict[str, typing.Any]:
        self.eval_hf_model_calls.append(kwargs)
        return {}

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
        evals_config = DummyEvalsConfig("something")
    return trainer_common.AppMainConfig.model_construct(
        datamodule_config=datamodule_config,
        evals_config=evals_config,
        use_wandb_logging=use_wandb_logging,
    )


def test_prepare_datamodule_basic() -> None:
    datamodule = DummyDatamodule()
    config = _build_app_config(DummyDatamoduleConfig(datamodule))
    result = trainer_common.prepare_datamodule(config, runtime=None)
    assert result is datamodule
    assert datamodule.prepared
    assert datamodule.setup_called == 1
    assert datamodule.instantiate_verbose == [True]


def test_prepare_datamodule_with_wandb_logging() -> None:
    datamodule = DummyDatamodule(stats={"rows": 42})
    evals_config = DummyEvalsConfig("something")
    config = _build_app_config(
        DummyDatamoduleConfig(datamodule, ["valid", "test"]),
        use_wandb_logging=True,
        evals_config=evals_config,
    )
    runtime = types.SimpleNamespace(
        wandb_run=types.SimpleNamespace(summary={}),
        wandb_run_id="run-123",
    )
    result = trainer_common.prepare_datamodule(config, runtime=runtime)
    assert result is datamodule
    assert runtime.wandb_run.summary["dataset_stats/rows"] == 42
    assert [call["prefix"] for call in evals_config.define_metrics_calls] == [
        "evals/valid",
        "evals/test",
    ]


@pytest.mark.asyncio
async def test_evaluate_model_sync_and_async(monkeypatch: pytest.MonkeyPatch) -> None:
    metrics_logged: list[tuple[dict[str, float], str]] = []

    def fake_print_metrics(metrics: dict[str, float], subset: str, printer: typing.Any) -> None:
        metrics_logged.append((metrics, subset))

    monkeypatch.setattr(trainer_common.pyine.evals.utils, "print_metrics", fake_print_metrics)
    synchronous_result = FakeEvaluationResult({"acc": 0.9}, ["art1"])
    asynchronous_result = FakeEvaluationResult({"acc": 0.95}, ["art2"])

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

    evals_config = DummyEvalsConfig("something")
    evals_config.evaluate_runnable_model = fake_evaluate_runnable_model
    datamodule = DummyDatamodule()
    config = _build_app_config(
        DummyDatamoduleConfig(datamodule, eval_subset_names=["sync", "async"]),
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
async def test_evaluate_model_wandb_logging_reopens_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trainer_common.pyine.evals.utils, "print_metrics", lambda *args, **kwargs: None)
    replacement_run = types.SimpleNamespace(summary={}, logged=[])  # new run returned by wandb.init

    def fake_wandb_init(**kwargs: typing.Any) -> types.SimpleNamespace:
        replacement_run.log = lambda payload: replacement_run.logged.append(payload)
        return replacement_run

    monkeypatch.setattr(trainer_common.wandb, "init", fake_wandb_init)
    result = FakeEvaluationResult(metrics={"acc": 0.77}, artifacts=["sample"])

    async def fake_evaluate_runnable_model(
        chain: typing.Any,
        datamodule: DummyDatamodule,
        eval_subset_name: str,
        verbose: bool,
    ) -> FakeEvaluationResult:
        return result

    evals_config = DummyEvalsConfig("something")
    evals_config.evaluate_runnable_model = fake_evaluate_runnable_model
    runtime = types.SimpleNamespace(
        wandb_run=types.SimpleNamespace(summary={}),  # missing log attr triggers reopen
        wandb_run_id="run-42",
    )
    config = _build_app_config(
        DummyDatamoduleConfig(DummyDatamodule(), eval_subset_names=["subset"]),
        use_wandb_logging=True,
        evals_config=evals_config,
    )
    model = types.SimpleNamespace(invoke=lambda x: x)
    results = await trainer_common.evaluate_model(
        model=model,
        tokenizer=None,
        datamodule=DummyDatamodule(),
        config=config,
        runtime=runtime,
    )
    assert results["subset"] is result
    assert runtime.wandb_run is replacement_run
    assert len(evals_config.log_metrics_calls) == 1
    assert len(evals_config.log_predictions_calls) == 1


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

    evals_config = DummyEvalsConfig("something")
    evals_config.evaluate_runnable_model = fake_evaluate_runnable_model

    runtime = types.SimpleNamespace(wandb_run=types.SimpleNamespace(summary={}), wandb_run_id=None)
    config = _build_app_config(
        DummyDatamoduleConfig(DummyDatamodule()),
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
