import pathlib
import types

import pytest

import pyine.configs.base
import pyine.configs.schemas
import pyine.configs.searchpath
import pyine.configs.utils


def test_register_searchpath_plugin_registers_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered = []

    class _FakePlugins:
        def register(
            self,
            plugin: object,
        ) -> None:
            registered.append(plugin)

    fake_plugins = _FakePlugins()
    monkeypatch.setattr(
        pyine.configs.base.hydra.core.plugins.Plugins,
        "instance",
        classmethod(lambda cls: fake_plugins),
    )
    pyine.configs.base.register_searchpath_plugin()
    assert registered == [pyine.configs.searchpath.SearchPathPlugin]


def test_print_experiment_configs_lists_expected_sections(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = {"initialize": 0, "render": []}

    class _HydraContext:
        def __enter__(self) -> None:
            calls["initialize"] += 1
            return

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: types.TracebackType | None,
        ) -> bool:
            return False

    monkeypatch.setattr(
        pyine.configs.base.hydra,
        "initialize",
        lambda **_kwargs: _HydraContext(),
    )
    monkeypatch.setattr(
        pyine.configs.base.hydra,
        "compose",
        lambda **_kwargs: {"config": "value"},
    )
    monkeypatch.setattr(
        pyine.configs.base.pyine.utils.portability,
        "render_config",
        lambda cfg, composed: calls["render"].append((cfg, composed)),
    )
    experiment = types.SimpleNamespace(name="exp_a", group="experiment", config=types.SimpleNamespace())
    ignored = types.SimpleNamespace(name="other", group="misc", config=types.SimpleNamespace())
    pyine.configs.utils.print_experiment_configs([experiment, ignored], "app")
    captured = capsys.readouterr().out
    assert "+experiment=exp_a" in captured
    assert calls["initialize"] == 1
    assert calls["render"] == [(experiment.config, {"config": "value"})]


def test_runtime_config_wandb_flow(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_calls: list[dict[str, object]] = []

    class _FakeRun:
        def __init__(
            self,
            project: str | None = None,
            entity: str | None = None,
        ) -> None:
            self.id = "run-id"
            self.offline = False
            self.name = "run-name"
            self.url = "123"
            self.tags = ("tag",)
            self.notes = "note"
            self.project = project
            self.entity = entity

        def finalize(self) -> None:
            pass

    def fake_wandb_init(
        **kwargs: object,
    ) -> _FakeRun:
        run_calls.append(kwargs)
        return _FakeRun()

    monkeypatch.setattr(
        pyine.configs.schemas.wandb,
        "init",
        fake_wandb_init,
    )
    output_dir1 = tmp_path / "output1"
    output_dir1.mkdir(parents=True)
    runtime = pyine.configs.schemas.RuntimeConfig(
        exp_name="exp",
        run_name="run",
        output_dir=str(output_dir1),
    )
    assert runtime.wandb_run is None
    assert runtime.wandb_run_id is None
    assert runtime.wandb_run_entity is None
    assert runtime.wandb_run_project is None
    assert runtime.wandb_run_url is None
    run_id = runtime.init_wandb()
    assert run_id == "run-id"
    assert "extra" not in runtime.wandb_run.tags
    runtime.add_wandb_tag("extra")
    assert "extra" in runtime.wandb_run.tags
    assert runtime.wandb_run_id == "run-id"
    assert run_calls and run_calls[0]["tags"] is None
    assert runtime.wandb_run_entity is None
    assert runtime.wandb_run_project is None
    assert runtime.wandb_run_url == "123"
    run_calls.clear()
    output_dir2 = tmp_path / "output2"
    output_dir2.mkdir(parents=True)
    runtime = pyine.configs.schemas.RuntimeConfig(
        exp_name="exp",
        run_name="run2",
        output_dir=str(output_dir2),
    )
    _ = runtime.init_wandb(entity="ent", project="proj")
    assert run_calls and run_calls[0]["entity"] == "ent" and run_calls[0]["project"] == "proj"


def test_runtime_config_wandb_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = pyine.configs.schemas.RuntimeConfig(
        exp_name="exp",
        run_name="run",
        dry_run=True,
    )
    with pytest.raises(RuntimeError):
        runtime.init_wandb()
    runtime = pyine.configs.schemas.RuntimeConfig(
        exp_name="exp",
        run_name="run",
    )
    with pytest.raises(RuntimeError):
        runtime.add_wandb_tag("extra")
