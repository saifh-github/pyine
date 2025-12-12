# Tinkering with Experiments (Hydra / Hydra-Zen) — User Guide

This guide is for **you**, the user of the `pyine` framework. It shows how to create small YAML
"overlays" on top of Hydra-based apps to customize experiments without touching the core code or
structured configs inside the framework.

- For the simplest setup, you should put your YAML configuration files under:
  `<repo_root>/pyine/configs/experiment/**.yaml`
- You can organize these YAML files in subfolders (e.g., `big_models/`, `username/`, `paper/`).
- You can select these configurations at launch time via the Hydra group: `+experiment=<name or subdir/name>`.

> The framework already takes care of search paths and integration with Hydra/Hydra-Zen. You can
> focus on writing YAMLs and running commands.

To see a list of available, pre-registered experiment configurations, run the `..._configs.py` file
associated with the trainer you are interested in, for example:

```bash
python -m pyine.apps.trainers.openai_finetune_configs
# or
python -m pyine.apps.trainers.hf_trainer_configs
```

The above should provide an exhaustive description of all experiment configurations that can be
executed as-is without requiring you to specify any extra setting. To see a high-level list of
configuration groups beyond the `experiment` group itself (e.g. datamodules, evaluators, models,
etc.), use `--help`. For datamodule-specific documentation, see [`pyine/organisms/README.md`](../organisms/README.md).

```bash
python -m pyine.apps.trainers.openai_finetune --help
# or
python -m pyine.apps.trainers.hf_trainer --help
```

It is quite expected that you build your own experiment configurations by deriving from existing
configurations, whether by default-override (preferred) or by simply copy-pasting settings. For
a basic introduction to Hydra and how to structure configuration YAMLs, see
[this link](https://hydra.cc/docs/tutorials/intro/).

______________________________________________________________________

## Distributed Runs

Hydra integrates a callback (`pyine.configs.callbacks.NonPrimaryRankCleanupCallback`) that
automatically reroutes non-primary distributed ranks to a temporary output directory. By
default it keeps Hydra's file logging enabled and preserves the temporary directory, which is
often useful when debugging per-rank issues.

______________________________________________________________________

## Quick Start: Experiment Creation

1. Create a new YAML file for the experiment you would like to configure:

```
pyine/configs/experiment/my_first_exp.yaml
```

Example content (for the `pyine.apps.trainers.openai_finetune` app):

```yaml
# @package _global_
defaults:  # we inherit some settings from framework configs, and specify a few extra things manually
  - override /config: base  # part of the framework configs (basic settings for the openai_finetune app)
  - override /config/datamodule_config: shortcuts_TACO_latest  # also part of the framework configs
  - _self_  # by placing this last, the settings below override all inherited ones

runtime:  # builds the app's `pyine.configs.schemas.RuntimeConfig` object
  exp_name: my_first_exp  # this is a required setting to define for all experiments
  seed: 123  # override the default app seed

config:  # builds the app's `pyine.apps.trainers.openai_finetune_configs.OpenAIFineTuneAppMainConfig` object
  openai_finetuner_config:  # builds an expected attribute inside the above (which is another config)
    base_model: gpt-4.1-nano-2025-04-14  # override the default gpt-4.1-mini to an ever smaller model
```

2. Launch the targeted app using your YAML:

```bash
python -m pyine.apps.trainers.openai_finetune +experiment=my_first_exp

# or with additional ad-hoc overrides:
python -m pyine.apps.trainers.openai_finetune +experiment=my_first_exp runtime.seed=999

# if you'd like to do a 'dry-run' to check whether the config works and all required args are set:
python -m pyine.apps.trainers.openai_finetune +experiment=my_first_exp runtime.dry_run=True
```

By default, Hydra will save your exact run config in your experiment's output directory (under
`.hydra/config.yaml`) to help debugging and improve reproducibility.

To get more information on what Hydra is doing under the hood when resolving your experiment's
configuration, you can also call the same apps with `--info`, or `--help`.

______________________________________________________________________

## Directory Layout

You can add as many experiment 'overlays' as you want, and even nest them:

```
<repo_root>/
  pyine/
    configs/
      experiment/
        base.yaml
        eval_only.yaml
        my_first_exp.yaml
        llm/
          llama_8b.yaml
          llama_8b_nq.yaml
        big_paper_experiments/
          funky_data.yaml
```

In all cases, the Hydra group you will still need to override on the command line is `experiment`:

- `experiment=base`
- `experiment=eval_only`
- `experiment=my_first_exp`
- `experiment=llm/llama_8b`
- `experiment=big_paper_experiments/funky_data`

______________________________________________________________________

## Environment Override (Optional)

If you want to run with a different local configs root (that should still contain an `experiment/`
folder), you can set:

```bash
export PYINE_CONFIGS_ROOT=/abs/path/to/my/configs
python -m pyine.apps.trainers.openai_finetune +experiment=my_first_exp
```

This is handy if you keep your overlays outside the repo while developing.

## Using Structured Configs (Advanced)

If you would like to also use structured configurations based on
[Hydra-Zen](https://mit-ll-responsible-ai.github.io/hydra-zen/) to build your experiments,
you can create do so by defining a config registration function in any appropriately-named file
located in the same config search tree. Specifically, for any file whose name ends with
`..._configs.py` in the `<repo_root>/pyine/configs/` or in the custom-defined `PYINE_CONFIGS_ROOT`
folder (or subfolder), the framework will automatically look for a `register_hydra_configs`
function with the following signature:

```python
import pyine.configs.schemas
import pyine.evals.common

def register_hydra_configs(
    app_name: str,  # name of the app that we are looking to register configs for
    eval_type: pyine.evals.common.EvalType,  # eval type (task definition) for the configs to register
    entrypoint_config: pyine.configs.schemas.ConfigDescription,  # config for the app's entrypoint
    app_configs: list[pyine.configs.schemas.ConfigDescription],  # all registered configs for the app
) -> list[pyine.configs.schemas.ConfigDescription]:  # should return new app configs to register
    ...
```

The new config description objects you generate and return will be added to the Hydra-Zen store
during the setup of all apps in the framework.

______________________________________________________________________

## Troubleshooting / FAQ

**ERROR: "Could not override `experiment`. No match in the defaults list".**

You forgot to put `+` before `experiment=...` on the command line argument.

**ERROR: "Could not override `config@experiment.config`'\`"**

Make sure that your experiment configuration file starts with `# @package _global_`

**ERROR: "Could not load config `experiment=...`" OR "Could not find `experiment/...`"**

- Check the file path and name: `pyine/configs/experiment/<name>.yaml` (or in a subfolder);
- Ensure you are not using the `runtime.exp_name` define as the value passed to the command line;
- Ensure you are running from the repo root (or that your environment points to the right configs root);
- Verify that the group name matches the folder name (e.g. `experiment/` -> `experiment=`).

**ISSUE: Overrides do not seem to apply**

- Use `--cfg job` to confirm the final values;
- Remember that later defaults/overrides win, and that `_self_` should usually be last in YAMLs;
- Remember that CLI overrides take precedence over the YAML ones.

**Q: How can I base my experiment on an existing experiment?**

Include it in `defaults`:

```yaml
defaults:
  - /experiment: base_experiment
  - _self_
```

**Q: Can I mix YAML overlays with CLI overrides?**

Yes. YAML captures your baseline; CLI is perfect for quick one-off changes.

**Q: Do nested folders work?**

Yes. Configs in subfolders have nested names:
`experiment/llm/llama_8b.yaml` -> `experiment=llm/llama_8b`.

**Q: Where can I see what fields are available to override?**

Use `--cfg job` to print the final config, and/or `--help` if the app exposes a signature.

______________________________________________________________________

Have fun!
