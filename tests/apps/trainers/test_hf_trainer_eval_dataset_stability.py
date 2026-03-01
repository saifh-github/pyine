import hashlib
import pathlib
import typing

import hydra_zen
import pytest

import pyine.apps.trainers.hf_rl_trainer_configs
import pyine.configs.base
import pyine.data.traces.dataset_reader
import pyine.data.traces.dataset_utils
import pyine.data.utils.splits
import pyine.evals.common
import pyine.organisms.datamodules.base
import pyine.utils.filesystem
import tests.utils.fake_dataset_readers


def _extract_sample_fingerprints(
    dataset: typing.Iterable[dict[str, typing.Any]],
) -> list[tuple[str, typing.Any, typing.Any]]:
    fingerprints: list[tuple[str, typing.Any, typing.Any]] = []
    for row in dataset:
        sample_data = row.get("sample_data")
        if sample_data is None:
            raise AssertionError("expected sample_data in HF dataset row")
        identifier = sample_data.get("identifier")
        if identifier is None:
            raise AssertionError("expected identifier in sample_data")
        inputs = sample_data.get("inputs")
        expected_output = sample_data.get("expected_output")
        fingerprints.append((typing.cast("str", identifier), inputs, expected_output))
    return fingerprints


@pytest.mark.integration
def test_v0_rl_valid_traces_stable_across_epochs(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_data_root = tmp_path / "data"
    fake_data_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(pyine.utils.filesystem.DATA_ROOT_ENV_VAR, str(fake_data_root))
    monkeypatch.setenv(pyine.utils.filesystem.CACHE_ROOT_ENV_VAR, str(tmp_path / "cache"))
    monkeypatch.setenv(pyine.utils.filesystem.LOGS_ROOT_ENV_VAR, str(tmp_path / "logs"))
    hf_cache_dir = tmp_path / "hf_datasets_cache"
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HF_DATASETS_CACHE", str(hf_cache_dir))
    hf_home_dir = tmp_path / "hf_home"
    hf_home_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HF_HOME", str(hf_home_dir))
    monkeypatch.setenv("HF_HUB_CACHE", str(hf_home_dir / "hub"))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(hf_home_dir / "transformers"))
    xdg_cache_dir = tmp_path / "xdg_cache"
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg_cache_dir))
    xdg_config_dir = tmp_path / "xdg_config"
    xdg_config_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_config_dir))
    mpl_config_dir = tmp_path / "mpl_config"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MPLCONFIGDIR", str(mpl_config_dir))
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "wandb"))
    import datasets as hf_datasets  # local import to respect cache env vars

    # use monkeypatch.setattr to ensure the HF cache setting is restored after the test
    monkeypatch.setattr(hf_datasets.config, "HF_DATASETS_CACHE", str(hf_cache_dir))
    fake_lmdb_path = tmp_path / "fake.lmdb"
    fake_lmdb_path.mkdir(parents=True, exist_ok=True)
    fake_cfg = tests.utils.fake_dataset_readers.FakeTraceDataConfig(
        dataset_name="FAKE",
        subset_name="unit",
        num_problems=2,
        solutions_per_problem=2,
        tests_per_problem=3,
        seed=123,
    )
    split_config = pyine.data.utils.splits.SplitConfig(
        subset_names=["train", "valid", "test"],
        subset_assign_prob_map={"train": 0.5, "valid": 0.5, "test": 0.0},
    )
    fake_reader = tests.utils.fake_dataset_readers.FakeTraceDatasetReader(config=fake_cfg)
    problem_ids = sorted({str(meta.problem_id) for meta in fake_reader.trace_metadata})
    subset_assignments = {
        problem_id: ("train" if idx == 0 else "valid" if idx == 1 else "test")
        for idx, problem_id in enumerate(problem_ids)
    }
    split_result = pyine.data.utils.splits.SplitResult(
        source_dataset_name=fake_reader.parent_dataset_name,
        source_dataset_hash=fake_reader.hash,
        identifiers=problem_ids,
        tag_lists=[["toy"] for _ in problem_ids],
        source_data_hashes=[hashlib.sha256(pid.encode()).hexdigest()[:16] for pid in problem_ids],
        subset_assignments=subset_assignments,
        creation_metadata={"generator": "test"},
        config=split_config,
    )
    split_path = tmp_path / "fake_split.json"
    split_result.to_file(split_path)

    class _PatchedReader(tests.utils.fake_dataset_readers.FakeTraceDatasetReader):
        def __init__(self, lmdb_path: typing.Any, **kwargs: typing.Any) -> None:
            super().__init__(lmdb_path=lmdb_path, config=fake_cfg, **kwargs)

    monkeypatch.setattr(pyine.data.traces.dataset_reader, "DatasetReader", _PatchedReader)
    pyine.configs.base.register_searchpath_plugin()
    configs = pyine.apps.trainers.hf_rl_trainer_configs.register_hydra_configs(
        eval_type=pyine.evals.common.EvalType.CODE_EXEC,
    )
    entrypoint_config = next(cfg for cfg in configs if cfg.name == "entrypoint" and cfg.group is None)
    run_dir = tmp_path / "hydra"

    def _build_config(cfg: typing.Any) -> pyine.apps.trainers.hf_rl_trainer_configs.RLTrainerAppMainConfig:
        return hydra_zen.instantiate(cfg.config, _convert_="object")

    job = hydra_zen.launch(
        entrypoint_config.config,
        task_function=_build_config,
        overrides=[
            "+experiment=original/v0_rl",
            "runtime=dry_run",
            "runtime.exp_name=tests/v0_rl_validation_stability",
            "config.use_wandb_logging=false",
            "config/datamodule_config=shortcuts_base",
            f"config.datamodule_config.lmdb_paths=[{fake_lmdb_path}]",
            f"config.datamodule_config.split_file_path={split_path}",
            "config.datamodule_config.use_local_dataset_cache=false",
            "config.datamodule_config.keep_generated_datasets_in_memory=false",
            "config.datamodule_config.require_validated_misleading=false",
            "++config.datamodule_config.dataparser_config_overrides.valid.filtering_config.max_traces_per_solution=1",
            f"hydra.run.dir={run_dir}",
            "hydra.output_subdir=null",
            "hydra/job_logging=disabled",
            "hydra/hydra_logging=disabled",
        ],
        config_name="entrypoint",
        version_base=pyine.configs.base.target_hydra_version,
        with_log_configuration=False,
    )
    config = job.return_value
    datamodule = config.datamodule_config.instantiate_datamodule()
    datamodule.prepare_data()
    datamodule.setup()
    # use the base class get_parser/get_hf_messages_dataset to bypass the shortcuts-specific
    # derived-subset expansion (valid -> valid_hinted/misleading/hintless); the fake data
    # has no counterfactual groups, so derived subsets would be empty
    base_cls = pyine.organisms.datamodules.base.BiasDataModuleBase
    valid_parser = base_cls.get_parser(datamodule, "valid")
    assert valid_parser.filtering_config.max_traces_per_solution == 1
    assert valid_parser.current_epoch == 0
    valid_ds_epoch0 = base_cls.get_hf_messages_dataset(
        datamodule,
        subset_name="valid",
        append_answer=False,
        merge_system_with_user=True,
        keep_original_data=True,
        force_regenerate=True,
    )
    fingerprints_epoch0 = _extract_sample_fingerprints(valid_ds_epoch0)
    assert fingerprints_epoch0, "expected non-empty validation dataset"
    solution_counts: dict[pyine.data.traces.dataset_utils.SolutionIdentifier, int] = {}
    for identifier, _inputs, _expected in fingerprints_epoch0:
        trace_id = pyine.data.traces.dataset_utils.TraceIdentifier.from_string(identifier)
        solution_id = trace_id.get_parent_identifier()
        solution_counts[solution_id] = solution_counts.get(solution_id, 0) + 1
    assert all(count <= 1 for count in solution_counts.values())

    train_parser = datamodule.get_parser("train")
    train_parser.set_epoch(1)
    assert train_parser.current_epoch == 1
    assert valid_parser.current_epoch == 0
    valid_ds_epoch1 = base_cls.get_hf_messages_dataset(
        datamodule,
        subset_name="valid",
        append_answer=False,
        merge_system_with_user=True,
        keep_original_data=True,
        force_regenerate=True,
    )
    fingerprints_epoch1 = _extract_sample_fingerprints(valid_ds_epoch1)
    assert sorted(fingerprints_epoch1) == sorted(fingerprints_epoch0)
