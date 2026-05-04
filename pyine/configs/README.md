# PyINE experiment configs (Hydra / Hydra-Zen)

This guide covers how to write small YAML "overlays" on top of the framework's Hydra-based apps to
customize experiments without touching the structured configs in core code.

- Drop your YAML files under [`pyine/configs/experiment/`](./experiment/) (subfolders are fine,
  e.g. `guardrail/`, `username/`, `paper/`).
- Select them at launch time with `+experiment=<name>` (or `+experiment=<subdir>/<name>` for nested
  files).
- The framework wires up search paths and Hydra-Zen integration for you; you just write YAMLs (or,
  for richer cases, structured configs in `..._configs.py` files; see below).

## Discovering registered configs

Each Hydra-based app has a sibling `..._configs.py` file that registers structured configs
(experiments, datamodules, evaluators, models, ...) into the Hydra-Zen store. Running it directly
prints an exhaustive listing of every registered config that is runnable as-is, grouped by Hydra
group:

```bash
# trainers
python -m pyine.apps.trainers.openai_finetune_configs
python -m pyine.apps.trainers.hf_sft_trainer_configs
python -m pyine.apps.trainers.hf_rl_trainer_configs
python -m pyine.apps.trainers.probe_trainer_configs
python -m pyine.apps.trainers.llm_classifier_trainer_configs

# standalone guardrail evaluators
python -m pyine.apps.guardrail_eval.baseline_eval_configs
python -m pyine.apps.guardrail_eval.prompted_llm_eval_configs
python -m pyine.apps.guardrail_eval.debate_eval_configs
```

For a high-level view of *config groups* an app exposes (e.g. `datamodule_config`,
`training_args_config`, `evals_config`), pass `--help` to the corresponding launcher:

```bash
python -m pyine.apps.trainers.hf_trainer --help
python -m pyine.apps.trainers.openai_finetune --help
```

For datamodule-specific docs, see [`pyine/organisms/README.md`](../organisms/README.md). For an
overview of every app, see [`pyine/apps/README.md`](../apps/README.md).

## Quick start: writing an experiment overlay

1. Create a YAML file under `pyine/configs/experiment/`, e.g. `my_first_exp.yaml`. A working
   reference lives at [`experiment/example.yaml`](./experiment/example.yaml); a minimal version
   for the `openai_finetune` app:

   ```yaml
   # @package _global_
   defaults:
     - override /config: base                                  # framework default for the app
     - override /config/datamodule_config: shortcuts_TACO_latest
     - _self_                                                  # last so the values below win

   runtime:                                                    # pyine.configs.schemas.RuntimeConfig
     exp_name: my_first_exp                                    # required for every experiment
     seed: 123

   config:                                                     # app's main config object
     openai_finetuner_config:
       base_model: gpt-4.1-nano-2025-04-14
   ```

2. Launch:

   ```bash
   python -m pyine.apps.trainers.openai_finetune +experiment=my_first_exp

   # ad-hoc overrides
   python -m pyine.apps.trainers.openai_finetune +experiment=my_first_exp runtime.seed=999

   # dry-run (validates the resolved config without doing real work)
   python -m pyine.apps.trainers.openai_finetune +experiment=my_first_exp runtime.dry_run=True
   ```

Hydra writes the resolved run config to `<output>/.hydra/config.yaml` for reproducibility. Use
`--cfg job` to print it without launching, and `--info` to dump everything Hydra resolved.

For more on Hydra basics, see the [Hydra tutorial](https://hydra.cc/docs/tutorials/intro/).

## Directory layout

The `pyine/configs/` package is organized as:

```
pyine/configs/
  experiment/             # YAML overlays; what you'll edit most
    example.yaml          # reference template (openai_finetune)
    example_configs.py    # reference structured-config registration
    guardrail/            # baseline / prompted-LLM / debate / probe / classifier eval overlays
    keywords/             # keyword-trigger experiment overlays
    original/             # legacy / paper-baseline overlays
    shortcuts/            # shortcut-following RL overlays
  accelerate/             # accelerate launcher YAMLs (DeepSpeed ZeRO/ZeRO++, FSDP2)
  base.py                 # base config builders + Hydra setup
  callbacks.py            # Hydra callbacks (e.g. distributed-rank cleanup)
  schemas.py              # shared Pydantic schemas (RuntimeConfig, ConfigDescription)
  searchpath.py           # config search-path management
  utils.py                # config description / registration helpers
```

Hydra references nested YAMLs by relative path: `experiment/guardrail/baseline_eval.yaml` is
selected via `+experiment=guardrail/baseline_eval`.

## External configs root (optional)

To keep overlays outside the repo (e.g. while iterating on private experiments), point
`PYINE_CONFIGS_ROOT` at any directory that contains an `experiment/` subfolder:

```bash
export PYINE_CONFIGS_ROOT=/abs/path/to/my/configs
python -m pyine.apps.trainers.openai_finetune +experiment=my_first_exp
```

The search-path plugin tries three roots in order (the env override, `<cwd>/pyine/configs`, and
the repo's own `pyine/configs/`) and merges YAMLs and `..._configs.py` files from any that
exist, so multiple locations can contribute concurrently.

## Distributed runs

`pyine.configs.callbacks.NonPrimaryRankCleanupCallback` is registered as a Hydra callback for all
distributed apps. It reroutes non-primary ranks to a temporary output directory while keeping
file logging enabled (handy for debugging per-rank issues). On node-local filesystems (detected
via `pyine.utils.filesystem.is_path_on_shared_filesystem`), each node's local-rank-0 process
keeps the real output directory; every node has its own physical storage and the rank-suffixed
filenames avoid collisions.

The [`accelerate/`](./accelerate/) folder ships ready-made launcher YAMLs for `accelerate launch`
covering DeepSpeed ZeRO-3 / ZeRO++ and FSDP2 across single-node and multi-node topologies. Pass
one via `--config_file` when launching:

```bash
accelerate launch \
  --config_file pyine/configs/accelerate/deepspeed_zero3_1x8gpu.yaml \
  -m pyine.apps.trainers.hf_trainer +experiment=...
```

## Structured configs (advanced)

For experiments that go beyond YAML (i.e. programmatically derived from a base config, parameterized
across a sweep, or composed of multiple new app-level configs) you can register them via
Hydra-Zen by adding a `..._configs.py` file in any folder on the search path (typically next to
your YAMLs in `experiment/`). The framework auto-discovers any module whose name ends in
`_configs.py` and calls its `register_hydra_configs` function:

```python
import pyine.configs.schemas
import pyine.evals.common

def register_hydra_configs(
    app_name: str,                                                     # which app is being set up
    eval_type: pyine.evals.common.EvalType,                            # task type for these configs
    entrypoint_config: pyine.configs.schemas.ConfigDescription,        # the app's entrypoint config
    app_configs: list[pyine.configs.schemas.ConfigDescription],        # configs registered so far
) -> list[pyine.configs.schemas.ConfigDescription]:                    # new configs to register
    ...
```

A working reference (small-model and CPU-friendly experiments for the HF trainer) lives at
[`experiment/example_configs.py`](./experiment/example_configs.py). Returned
`ConfigDescription` objects are added to the Hydra-Zen store during app setup and become selectable
through their declared group (e.g. `+experiment=...`, `config/evals_config=...`).

## Troubleshooting

**`Could not override 'experiment'. No match in the defaults list.`**
You forgot the `+` in `+experiment=...` on the CLI.

**`Could not override 'config@experiment.config'.`**
Your YAML is missing the `# @package _global_` header on line 1.

**`Could not load config 'experiment=...'` / `Could not find 'experiment/...'`.**

- Check the file path: `pyine/configs/experiment/<name>.yaml` (subfolders allowed).
- Don't pass `runtime.exp_name` as the value; that's the run label, not the config name.
- Run from the repo root, or set `PYINE_CONFIGS_ROOT` to a directory that contains an
  `experiment/` subfolder.

**Overrides don't seem to apply.**

- Run with `--cfg job` to dump the resolved config and inspect the actual values.
- `_self_` should usually be **last** in `defaults:` so YAML values win over inherited ones.
- CLI overrides take precedence over YAML.

**How do I base my experiment on another?**
List it in `defaults`:

```yaml
defaults:
  - /experiment: base_experiment
  - _self_
```

**How do I see which fields are overridable?**
`--cfg job` prints the fully resolved config; `--help` prints the available config groups.
