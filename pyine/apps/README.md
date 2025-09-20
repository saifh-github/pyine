# PyINE Apps Overview

This folder contains Python applications that support the full PyINE workflow: preparing datasets,
generating execution traces and deltas, annotating traces with prompt chains, and
training/evaluating models.

Most apps are plain Python CLIs (using [click](https://click.palletsprojects.com/en/stable/)) and
a couple are [Hydra](https://hydra.cc/docs/intro/)-integrated launchers (for configuration-rich
training/evaluation jobs). Below is a high-level tour with quick-start examples; for more detailed
docs, refer to each app's docstring.

Dataset preparation:

- Splits: [`pyine/apps/splits/dataset_splitter.py`](./splits/dataset_splitter.py)
- Traces & deltas writer: [`pyine/apps/write/dataset_writer.py`](./write/dataset_writer.py)

Trace annotation:

- Prompt-chain annotator: [`pyine/apps/annotate/trace_annot_generator.py`](./annotate/trace_annot_generator.py)

Training/evaluation (Hydra-based apps):

- HuggingFace trainer: [`pyine/apps/trainers/hf_trainer.py`](./trainers/hf_trainer.py)
- OpenAI fine-tuner: [`pyine/apps/trainers/openai_finetune.py`](./trainers/openai_finetune.py)

For instruction on how to generate the PyINE 10s10t v1 dataset based on TACO, see
[this document](./README-10s10t-v1.md).

For instructions on how to create and manage new experiment configuration files for the apps that
rely on Hydra, see [this document](../configs/README.md).

______________________________________________________________________

### Split a source dataset into train/valid/test and partitions

**Script:** [`pyine/apps/splits/dataset_splitter.py`](./splits/dataset_splitter.py)

**Click CLI modes:**

- `split`: creates a reproducible split file (with metadata for experiments);
- `partition`: breaks the split into disjoint chunks to distribute processing.

**Examples:**

```bash
# Create an 80/10/10 split for the repackaged TACO dataset with useful grouping metadata
python -m pyine.apps.splits.dataset_splitter split \
    --dataset-name TACO \
    --train-fraction 0.8 \
    --valid-fraction 0.1 \
    --test-fraction 0.1 \
    --use-difficulty-group \
    --use-solution-counts-group \
    --progress

# Partition problem IDs into YAML chunks of 500
python -m pyine.apps.splits.dataset_splitter partition \
    --split-file data/splits/TACO-split.bin \
    --output-dir data/splits \
    --ids-per-chunk 500 \
    --format yaml
```

______________________________________________________________________

### Write execution traces and deltas datasets

**Script:** [`pyine/apps/write/dataset_writer.py`](./write/dataset_writer.py)

**Click CLI modes:**

- `traces`: generates code execution traces for a supported source dataset;
- `deltas`: converts an existing traces dataset into a deltas dataset.

**Examples:**

```bash
# Write a (capped) traces dataset from the latest repackaged TACO source
python -m pyine.apps.write.dataset_writer traces \
    --dataset-name TACO \
    --max-output-traces 1000 \
    --max-solutions-per-problem 10 \
    --max-tests-per-solution 10

# Derive a deltas dataset from a previously generated traces dataset
python -m pyine.apps.write.dataset_writer deltas \
    --traces-dataset /path/to/traces.lmdb \
    --output-path /path/to/deltas.lmdb
```

For instruction on how to generate the PyINE 10s10t v1 dataset based on TACO, see
[this document](./README-10s10t-v1.md).

______________________________________________________________________

### Trace annotation (prompt chains over traces)

**Script:** [`pyine/apps/annotate/trace_annot_generator.py`](./annotate/trace_annot_generator.py)

**Main use:** runs prompt chains over a traces dataset to generate annotations and store the results
in the framework’s prompt results SQLite DB.

**Examples:**

```bash
# minimal: run the `code_summary` prompt over every trace with a target word count
python -m pyine.apps.annotate.trace_annot_generator \
    --dataset /path/to/traces_dataset.lmdb \
    --prompt-name code_summary \
    --prompt-vars '{"target_word_count": "50"}' \
    --llm-option provider=openai \
    --llm-option model=gpt-4o-mini

# use the latest traces dataset for a given source (e.g. TACO)
python -m pyine.apps.annotate.trace_annot_generator \
    --dataset-latest-from TACO \
    --prompt-name hints/docs \
    --llm-option provider=openai \
    --llm-option model=gpt-5

# example utility flags:
#   --dry-run               do not write to the DB
#   --no-parallel           process sequentially
#   --no-progress           hide progress bar
#   --db-path               custom path for the prompt results DB
#   --max-workers           cap parallel workers
#   --target-indices        restrict to a subset (e.g., 0-99,150,200-205)
#   --shared-tags/meta      attach tags/metadata to all generated entries
#   --force-generation      always generate (even if prior results exist)
```

**Notes:**

- LLM provider options can be supplied inline via repeated `--llm-option key=value` pairs, or loaded
  from a YAML file with `--llm-config-file`.
- Results are written to `data/prompt_results.sqlite` by default (see
  [`pyine/prompts/result_db.py`](../../pyine/prompts/result_db.py) and
  [this README](../../pyine/prompts/README.md) for more information).

______________________________________________________________________

### Trainers

Training runs are launched via Hydra configs shipped alongside each app. You can either select
a pre-registered experiment configuration (via `+experiment=EXP_NAME`) or override individual
config fields as needed.

To see a list of available, pre-registered experiment configurations, run the `..._configs.py` file
associated with the trainer you are interested in, for example:

```bash
python -m pyine.apps.trainers.openai_finetune_configs
# or
python -m pyine.apps.trainers.hf_trainer_configs
```

The above should provide an exhaustive description of all experiment configurations that can be
executed as-is without requiring you to specify any extra setting; for example:

```bash
python -m pyine.apps.trainers.openai_finetune +experiment=example
# you can still use extra overrides if you want, for example:
python -m pyine.apps.trainers.openai_finetune +experiment=example some_group.some_setting=123
```

To see a high-level list of configuration groups beyond the `experiment` group itself (e.g.
datamodules, evaluators, models, etc.), use `--help`:

```bash
python -m pyine.apps.trainers.openai_finetune --help
# or
python -m pyine.apps.trainers.hf_trainer --help
```

For the development of new experiment configurations, refer to [this README](../configs/README.md).

### HuggingFace trainer

**Scripts:**

- Launcher: [`pyine/apps/trainers/hf_trainer.py`](./trainers/hf_trainer.py)
- Configs: [`pyine/apps/trainers/hf_trainer_configs.py`](./trainers/hf_trainer_configs.py)

**Listing available experiment configs:**

```bash
python -m pyine.apps.trainers.hf_trainer_configs
```

**Examples:**

```bash
# run a registered experiment
python -m pyine.apps.trainers.hf_trainer +experiment=exp_name
# use a registered experiment while overriding specific parameters inline
python -m pyine.apps.trainers.hf_trainer \
  +experiment=some_other_experiment \
  config.base_model=Qwen/Qwen2.5-7B-Instruct \
  config.training_args_config.eval_steps=50
```

TODO @@@@@ : document outputs and stuff.

### OpenAI fine-tuning and evaluation

**Scripts:**

- Launcher: [`pyine/apps/trainers/openai_finetune.py`](./trainers/openai_finetune.py)
- Configs: [`pyine/apps/trainers/openai_finetune_configs.py`](./trainers/openai_finetune_configs.py)

**Listing available experiment configs:**

```bash
python -m pyine.apps.trainers.openai_finetune_configs
```

**Examples:**

```bash
# evaluate the default base model directly (i.e. no fine-tuning; relies on )
python -m pyine.apps.trainers.openai_finetune \
  +experiment=TACO_latest \
  skip_fine_tuning=true
# supervised fine-tuning on a small dataset with default settings
python -m pyine.apps.trainers.openai_finetune \
  +experiment=TACO_latest \
  config.openai_finetuner_config=openai_gpt-4.1-mini_default_sft
```

**Notes:**

- Requires an OpenAI API key configured via your `.env` (see the
  [project root README](../../README.md) for details).
- Optionally logs to Weights & Biases; toggle via config or runtime flags. The app can also log
  evaluation tables with predictions per subset at the end of a run.

______________________________________________________________________

## Tips

- All apps accept `-h/--help`.
- For hydra-based apps, `+experiment=<name>` selects a registered experiment; inline overrides use
  the `config.<field>=<value>` syntax shown above.
- See the [notebooks](../../notebooks/README.md) for hands-on, visual walkthroughs of core framework features.
