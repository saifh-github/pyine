import pathlib
import types

import omegaconf
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


class TestPrintExperimentConfigs:
    @staticmethod
    def _make_configs() -> list[types.SimpleNamespace]:
        entrypoint = types.SimpleNamespace(
            name="entrypoint",
            group=None,
            description="Entrypoint desc",
            config=types.SimpleNamespace(),
        )
        experiment = types.SimpleNamespace(
            name="exp_a",
            group="experiment",
            description="Experiment A",
            config=types.SimpleNamespace(),
        )
        other = types.SimpleNamespace(
            name="other",
            group="misc",
            description="Other config",
            config=types.SimpleNamespace(),
        )
        return [entrypoint, experiment, other]

    def test_list_mode_shows_configs_without_hydra(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        calls: dict[str, list[object]] = {"initialize": [], "render": []}
        monkeypatch.setattr(
            pyine.configs.utils.hydra,
            "initialize",
            lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should not be called")),
        )
        monkeypatch.setattr(
            pyine.configs.utils.pyine.utils.portability,
            "render_config",
            lambda *_a, **_kw: calls["render"].append(True),
        )
        monkeypatch.setattr(
            pyine.configs.utils,
            "discover_yaml_experiment_configs",
            lambda: [],
        )
        configs = self._make_configs()
        pyine.configs.utils.print_experiment_configs(configs, "app")
        captured = capsys.readouterr().out
        assert "+experiment=exp_a" in captured
        assert "AVAILABLE CONFIGS" in captured
        assert not calls["render"]

    def test_show_mode_composes_and_renders(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        calls: dict[str, list[object]] = {"initialize": [], "render": []}

        class _HydraContext:
            def __enter__(self) -> None:
                calls["initialize"].append(True)
                return

            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                tb: types.TracebackType | None,
            ) -> bool:
                return False

        monkeypatch.setattr(
            pyine.configs.utils.hydra,
            "initialize",
            lambda **_kwargs: _HydraContext(),
        )
        monkeypatch.setattr(
            pyine.configs.utils.hydra,
            "compose",
            lambda **_kwargs: {"config": "value"},
        )
        monkeypatch.setattr(
            pyine.configs.utils.pyine.utils.portability,
            "render_config",
            lambda cfg, composed, **_kw: calls["render"].append((cfg, composed)),
        )
        configs = self._make_configs()
        pyine.configs.utils.print_experiment_configs(configs, "app", cli_args=["+experiment=exp_a"])
        assert len(calls["initialize"]) == 1
        assert calls["render"] == [(configs[0].config, {"config": "value"})]

    def test_error_on_invalid_args(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            pyine.configs.utils,
            "discover_yaml_experiment_configs",
            lambda: [],
        )
        configs = self._make_configs()
        pyine.configs.utils.print_experiment_configs(configs, "app", cli_args=["foobar"])
        captured = capsys.readouterr().out
        assert "Error" in captured
        assert "Usage" in captured


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


def test_experiment_configs_do_not_override_runtime_dry_run() -> None:
    experiment_root = pathlib.Path("pyine/configs/experiment")
    experiment_config_paths = sorted(experiment_root.rglob("*.yaml"))
    assert experiment_config_paths, "no experiment yaml configs found"
    paths_with_runtime_dry_run_override: list[pathlib.Path] = []
    for experiment_config_path in experiment_config_paths:
        config = omegaconf.OmegaConf.load(experiment_config_path)
        runtime_dry_run = omegaconf.OmegaConf.select(config, "runtime.dry_run")
        if runtime_dry_run is not None:
            paths_with_runtime_dry_run_override.append(experiment_config_path)
    assert not paths_with_runtime_dry_run_override, (
        f"experiment configs must not set runtime.dry_run; found overrides in: {paths_with_runtime_dry_run_override}"
    )
