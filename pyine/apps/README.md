# PyINE Apps Overview

This folder contains Python applications that support the full PyINE workflow: preparing datasets,
generating code execution traces and deltas, annotating traces with prompt chains, and
training/evaluating model organisms and monitors/reporters.

Most apps are plain Python CLIs (using [click](https://click.palletsprojects.com/en/stable/)) and
a couple are [Hydra](https://hydra.cc/docs/intro/)-integrated launchers (for configuration-rich
training/evaluation jobs). Below is a high-level tour with quick-start examples; for more detailed
docs, refer to each app's docstring.

@@@ TODO: start thinking about ddp-sweeps on short jobs

______________________________________________________________________

Dataset preparation:

- Splits: [`pyine/apps/splits/dataset_splitter.py`](./splits/dataset_splitter.py)
- Traces & deltas writer: [`pyine/apps/write/dataset_writer.py`](./write/dataset_writer.py)
- HuggingFace dataset precacher: [`pyine/apps/data/hf_precacher.py`](./data/hf_precacher.py)

Trace annotation:

- Prompt-chain annotator: [`pyine/apps/annotate/trace_annot_generator.py`](./annotate/trace_annot_generator.py)

Trace analysis and repair:

- Problem data (I/O) rewrite pipeline: [`pyine/apps/traces/taco_trace_failure_analyzer.py`](traces/taco_trace_failure_analyzer.py)

Training/evaluation (Hydra-based apps):

- HuggingFace trainer: [`pyine/apps/trainers/hf_trainer.py`](./trainers/hf_trainer.py)
- OpenAI fine-tuner: [`pyine/apps/trainers/openai_finetune.py`](./trainers/openai_finetune.py)

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

**Outputs and layout:**

- Split runs write a single binary file to `<PYINE_DATA_ROOT>/splits/<DATASET>-split.bin`. The file
  bundles subset assignments, hash lists, and grouping metadata used by downstream datamodules.
- Partition runs create chunk files named `<DATASET>-split.problem_ids.<rank>of<total>.{yaml,json}`
  in the directory supplied via `--output-dir`.
- Logs stream to stdout and to `PYINE_LOGS_ROOT/pyine.log` via the shared logging setup.

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

**Outputs and layout:**

- Trace runs default to `<PYINE_DATA_ROOT>/traces/<SOURCE>/<TAG>.<YYYY-MM-DD>.lmdb`. Each dataset is
  an LMDB directory (`data.mdb`, `lock.mdb`) with metadata entries capturing the writer config,
  source dataset hash, and reproducibility tags. Use
  `pyine.data.traces.dataset_reader.DatasetReader` to inspect contents.
- Code execution failures are summarized in
  `<PYINE_LOGS_ROOT>/traced-test-failures/<OUTPUT_NAME>.log`. Each append writes a multi-line block
  with the `TraceIdentifier`, failing inputs/outputs, and exception metadata for quick replay.
- Deltas runs default to `<PYINE_DATA_ROOT>/deltas/<SOURCE>/<TAG>.<YYYY-MM-DD>.lmdb` unless you pass
  `--output-path`. The resulting LMDB mirrors the traces structure but stores per-line delta records.
- If you supply `--output-path` or `--output-tag` the writer respects it verbatim, letting you keep
  large artifacts with specific names on external volumes.

For instruction on how to generate the PyINE 10s10t v1 dataset based on TACO, see
[this document](./README-10s10t-v1.md).

______________________________________________________________________

### HuggingFace dataset precacher

**Script:** [`pyine/apps/data/hf_precacher.py`](./data/hf_precacher.py)

**Main use:** pre-generates datamodule caches (metadata, HF message datasets, tokenized examples)
before training runs to ensure faster, non-blocking trainer startups. This is especially useful for
distributed training scenarios where cache generation on multiple ranks can cause conflicts.

**Listing available experiment configs:**

```bash
python -m pyine.apps.data.hf_precacher --help
```

**Examples:**

```bash
# precache datasets for a registered experiment (train and validation subsets only)
python -m pyine.apps.data.hf_precacher +experiment=exp_name

# precache with evaluation subsets included
python -m pyine.apps.data.hf_precacher \
  +experiment=exp_name \
  precache_config.include_eval_subsets=true

# force regeneration of all caches
python -m pyine.apps.data.hf_precacher \
  +experiment=exp_name \
  precache_config.force_regenerate=true

# override max sequence length for tokenization
python -m pyine.apps.data.hf_precacher \
  +experiment=exp_name \
  precache_config.max_seq_len_override=2048
```

**Outputs and layout:**

- Caches are stored in subdirectories of the path specified by `pyine.utils.filesystem.get_data_cache_path`.
- The precacher uses the same datamodule configuration as the HF trainer, ensuring consistency between
  precaching and training runs.
- All generated caches include metadata files for reproducibility and cache invalidation.

______________________________________________________________________

### Trace annotation (prompt chains over traces)

**Script:** [`pyine/apps/annotate/trace_annot_generator.py`](./annotate/trace_annot_generator.py)

**Main use:** runs prompt chains over a traces dataset to generate annotations and store the results
in the framework’s prompt results SQLite DB.

**Outputs and layout:**

- By default results append to `<PYINE_DATA_ROOT>/prompt_results.sqlite`. The database stores prompt
  name/version, tags, creation metadata, and the rendered prompt/response payloads per trace
  identifier. Use `pyine.prompts.result_db.PromptResultDB` or the notebooks to explore entries.
- Pass `--db-path` to isolate runs in their own SQLite file (handy for prototyping or exports).
- When you supply `--shared-tags` or `--shared-meta`, those attributes are persisted verbatim with
  each record so you can slice later via tag filters.
- Dry runs skip database writes entirely but still report which identifiers would be touched.

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
- For more information on the prompt result database, see
  [`pyine/prompts/result_db.py`](../../pyine/prompts/result_db.py) and
  [this README](../../pyine/prompts/README.md).

______________________________________________________________________

### TACO trace problem data (input/output) rewrite (LLM-assisted repair)

**Script:** [`pyine/apps/traces/taco_trace_failure_analyzer.py`](traces/taco_trace_failure_analyzer.py)

**Main use:** iterates over coding problems in the TACO dataset, invokes the `input_output_rewrite` prompt
where necessary, validates the generated samples against bundled solutions, and stores successful
repairs in a cache-backed JSON file (saved by default in `<PYINE_DATA_CACHE>`).

**Example:**

```bash
# rewrite every malformed problem under the repackaged dataset root
python -m pyine.apps.traces.taco_trace_failure_analyzer \
    --problem-dir data/TACO/repackaged/<version>

# target a handful of individual problems
python -m pyine.apps.traces.taco_trace_failure_analyzer \
    --problem-dir data/TACO/repackaged/<version> \
    --problem 014084.json \
    --problem 017304.json
```

Repairs are appended to any already-existing cache file (or the custom path supplied via
`--override-log`), making the run resilient to interruptions.

______________________________________________________________________

### Trainers

Training runs are launched via Hydra configs shipped alongside each app. You can either select
a pre-registered experiment configuration (via `+experiment=<EXP_NAME>`) or override individual
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

**Run outputs:**

Hydra snapshots every run under `<PYINE_LOGS_ROOT>/runs/<app>/<exp_name>/<run_name>/`, which should contain:

- A `.hydra` subfolder with the original, not-yet-resolved app config (`.hydra/config.yaml`),
  the hydra config itself (`.hydra/hydra.yaml`), and any command line overrides that may have
  been specified (`.hydra/overrides.yaml`);
- The resolved runtime (`runtime.<timestamp>.rank00.json`) and app configs (`config.<timestamp>.rank00.json`);
- The output stdout/stderr log of the app (`output.log`);
- Reproducibility metadata that includes platform and environment details (`reprod_metadata.<timestamp>.rank00.json`);
- The model checkpoints, tokenizer files, and/or trainer state (if relevant).

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
# evaluate the default base model directly (i.e. no fine-tuning; relies on skip_fine_tuning=true)
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

**Artifacts and logs:**

- Conversation datasets prepared for upload are cached in
  `${TMPDIR}/pyine-<user>/openai-data/shortcuts-data.<subset>.<hash>.jsonl`. The helper reuses
  existing files when the datamodule config matches, so you can inspect or upload them manually.
- After a fine-tune completes, the chosen model name is printed and (if W&B logging is enabled)
  recorded in the run summary alongside evaluation metrics.

______________________________________________________________________

## Tips

- All apps accept `-h/--help`.
- For hydra-based apps, `+experiment=<name>` selects a registered experiment; inline overrides use
  the `config.<field>=<value>` syntax shown above.
- See the [notebooks](../../notebooks/README.md) for hands-on, visual walkthroughs of core framework features.
