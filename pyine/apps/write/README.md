# PyINE Traces/Deltas Dataset Writer App

The `dataset_writer.py` script allows you to generate a dataset of code execution traces
(or code execution deltas) for a given source dataset of coding problems and solutions.

See the docstring inside [`dataset_writer.py`](./dataset_writer.py) for specific usage instructions.

## Generating the PyINE Traces Dataset from TACO (10s10t-v1, September 2025)

Internal documentation link: https://docs.google.com/document/d/13MQf50_cjLsFmMlNbiPLvTAgVsRY9c-uQyTTX7IlQQ0

The following instructions allow you to generate the PyINE traces dataset from the TACO dataset.
These instructions assume that 1) you have already downloaded the repackaged TACO dataset, and 2)
your working directory is the `pyine` root directory (i.e. the project root).

### Step 1: Generate a dataset split file (if not already done)

Run the following python script in order to generate a 80-10-10 split of the TACO source dataset:

```bash
python pyine/apps/splits/dataset_splitter.py split \
    --dataset-name=TACO \
    --train-fraction=0.8 \
    --valid-fraction=0.1 \
    --test-fraction=0.1 \
    --use-difficulty-group \
    --use-solution-counts-group \
    --progress
```

This should create a binary file at `<project_root>/data/splits/TACO-split.bin` used in subsequent
commands.

### Step 2: Generate a dataset partition (if not already done)

Since the dataset is big and it is probably not a good idea to try to generate it in one go (as you
could be interrupted before it is finalized), we partition it into chunks of 500 problems; given
that the original TACO dataset contains approximately 13,000 problems, this results in 26 parts
that should each take a few hours to process (depending on the number of solutions we decide
to trace for each problem).

To extract the partitions, run the following python script:

```bash
python pyine/apps/splits/dataset_splitter.py partition \
    --split-file=data/splits/TACO-split.bin \
    --output-dir=data/splits/ \
    --ids-per-chunk=500 \
    --format=yaml
```

This should create 26 partitions in `<project_root>/data/splits/` named as follows:

```
    TACO-split.problem_ids.000001of000026.yaml
    TACO-split.problem_ids.000002of000026.yaml
    TACO-split.problem_ids.000003of000026.yaml
    ...
```

### Step 3: Generate the PyINE traces dataset

For each of the above partitions, we will now generate a dataset of traced solutions; for this first
version, we will ask for 10 solutions to be traced per problem, each with 10 different tests
specifying different input arguments and expected output values (this is where the `10s10t` prefix
comes from). We also set some hard caps on the size of the tracing results, and ask for obfuscated
code to be traced as well (this will be useful for later evaluations).

For convenience (and copy-pasting pleasure), we specify all 26 script execution commands below. The
results should be saved under `<project_root>/data/traces/TACO/10s10t.<part_id>.<date>.lmdb` and
each file should be roughly 500MB (its content is already compressed).

```bash
python pyine/apps/write/dataset_writer.py traces \
    --dataset-name=TACO \
    --max-solutions-per-problem=10 \
    --max-tests-per-solution=10 \
    --max-trace-var-repr-length=10000 \
    --max-trace-valid-events=20000 \
    --generate-obfuscated-solutions \
    --target-problem-ids=data/splits/TACO-split.problem_ids.000001of000026.yaml \
    --output-tag="10s10t.000001of000026"

python pyine/apps/write/dataset_writer.py traces \
    --dataset-name=TACO \
    --max-solutions-per-problem=10 \
    --max-tests-per-solution=10 \
    --max-trace-var-repr-length=10000 \
    --max-trace-valid-events=20000 \
    --generate-obfuscated-solutions \
    --target-problem-ids=data/splits/TACO-split.problem_ids.000002of000026.yaml \
    --output-tag="10s10t.000002of000026"

python pyine/apps/write/dataset_writer.py traces \
    --dataset-name=TACO \
    --max-solutions-per-problem=10 \
    --max-tests-per-solution=10 \
    --max-trace-var-repr-length=10000 \
    --max-trace-valid-events=20000 \
    --generate-obfuscated-solutions \
    --target-problem-ids=data/splits/TACO-split.problem_ids.000003of000026.yaml \
    --output-tag="10s10t.000003of000026"

python pyine/apps/write/dataset_writer.py traces \
    --dataset-name=TACO \
    --max-solutions-per-problem=10 \
    --max-tests-per-solution=10 \
    --max-trace-var-repr-length=10000 \
    --max-trace-valid-events=20000 \
    --generate-obfuscated-solutions \
    --target-problem-ids=data/splits/TACO-split.problem_ids.000004of000026.yaml \
    --output-tag="10s10t.000004of000026"

python pyine/apps/write/dataset_writer.py traces \
    --dataset-name=TACO \
    --max-solutions-per-problem=10 \
    --max-tests-per-solution=10 \
    --max-trace-var-repr-length=10000 \
    --max-trace-valid-events=20000 \
    --generate-obfuscated-solutions \
    --target-problem-ids=data/splits/TACO-split.problem_ids.000005of000026.yaml \
    --output-tag="10s10t.000005of000026"

python pyine/apps/write/dataset_writer.py traces \
    --dataset-name=TACO \
    --max-solutions-per-problem=10 \
    --max-tests-per-solution=10 \
    --max-trace-var-repr-length=10000 \
    --max-trace-valid-events=20000 \
    --generate-obfuscated-solutions \
    --target-problem-ids=data/splits/TACO-split.problem_ids.000006of000026.yaml \
    --output-tag="10s10t.000006of000026"

python pyine/apps/write/dataset_writer.py traces \
    --dataset-name=TACO \
    --max-solutions-per-problem=10 \
    --max-tests-per-solution=10 \
    --max-trace-var-repr-length=10000 \
    --max-trace-valid-events=20000 \
    --generate-obfuscated-solutions \
    --target-problem-ids=data/splits/TACO-split.problem_ids.000007of000026.yaml \
    --output-tag="10s10t.000007of000026"

python pyine/apps/write/dataset_writer.py traces \
    --dataset-name=TACO \
    --max-solutions-per-problem=10 \
    --max-tests-per-solution=10 \
    --max-trace-var-repr-length=10000 \
    --max-trace-valid-events=20000 \
    --generate-obfuscated-solutions \
    --target-problem-ids=data/splits/TACO-split.problem_ids.000008of000026.yaml \
    --output-tag="10s10t.000008of000026"

python pyine/apps/write/dataset_writer.py traces \
    --dataset-name=TACO \
    --max-solutions-per-problem=10 \
    --max-tests-per-solution=10 \
    --max-trace-var-repr-length=10000 \
    --max-trace-valid-events=20000 \
    --generate-obfuscated-solutions \
    --target-problem-ids=data/splits/TACO-split.problem_ids.000009of000026.yaml \
    --output-tag="10s10t.000009of000026"

python pyine/apps/write/dataset_writer.py traces \
    --dataset-name=TACO \
    --max-solutions-per-problem=10 \
    --max-tests-per-solution=10 \
    --max-trace-var-repr-length=10000 \
    --max-trace-valid-events=20000 \
    --generate-obfuscated-solutions \
    --target-problem-ids=data/splits/TACO-split.problem_ids.000010of000026.yaml \
    --output-tag="10s10t.000010of000026"

python pyine/apps/write/dataset_writer.py traces \
    --dataset-name=TACO \
    --max-solutions-per-problem=10 \
    --max-tests-per-solution=10 \
    --max-trace-var-repr-length=10000 \
    --max-trace-valid-events=20000 \
    --generate-obfuscated-solutions \
    --target-problem-ids=data/splits/TACO-split.problem_ids.000011of000026.yaml \
    --output-tag="10s10t.000011of000026"

python pyine/apps/write/dataset_writer.py traces \
    --dataset-name=TACO \
    --max-solutions-per-problem=10 \
    --max-tests-per-solution=10 \
    --max-trace-var-repr-length=10000 \
    --max-trace-valid-events=20000 \
    --generate-obfuscated-solutions \
    --target-problem-ids=data/splits/TACO-split.problem_ids.000012of000026.yaml \
    --output-tag="10s10t.000012of000026"

python pyine/apps/write/dataset_writer.py traces \
    --dataset-name=TACO \
    --max-solutions-per-problem=10 \
    --max-tests-per-solution=10 \
    --max-trace-var-repr-length=10000 \
    --max-trace-valid-events=20000 \
    --generate-obfuscated-solutions \
    --target-problem-ids=data/splits/TACO-split.problem_ids.000013of000026.yaml \
    --output-tag="10s10t.000013of000026"

python pyine/apps/write/dataset_writer.py traces \
    --dataset-name=TACO \
    --max-solutions-per-problem=10 \
    --max-tests-per-solution=10 \
    --max-trace-var-repr-length=10000 \
    --max-trace-valid-events=20000 \
    --generate-obfuscated-solutions \
    --target-problem-ids=data/splits/TACO-split.problem_ids.000014of000026.yaml \
    --output-tag="10s10t.000014of000026"

python pyine/apps/write/dataset_writer.py traces \
    --dataset-name=TACO \
    --max-solutions-per-problem=10 \
    --max-tests-per-solution=10 \
    --max-trace-var-repr-length=10000 \
    --max-trace-valid-events=20000 \
    --generate-obfuscated-solutions \
    --target-problem-ids=data/splits/TACO-split.problem_ids.000015of000026.yaml \
    --output-tag="10s10t.000015of000026"

python pyine/apps/write/dataset_writer.py traces \
    --dataset-name=TACO \
    --max-solutions-per-problem=10 \
    --max-tests-per-solution=10 \
    --max-trace-var-repr-length=10000 \
    --max-trace-valid-events=20000 \
    --generate-obfuscated-solutions \
    --target-problem-ids=data/splits/TACO-split.problem_ids.000016of000026.yaml \
    --output-tag="10s10t.000016of000026"

python pyine/apps/write/dataset_writer.py traces \
    --dataset-name=TACO \
    --max-solutions-per-problem=10 \
    --max-tests-per-solution=10 \
    --max-trace-var-repr-length=10000 \
    --max-trace-valid-events=20000 \
    --generate-obfuscated-solutions \
    --target-problem-ids=data/splits/TACO-split.problem_ids.000017of000026.yaml \
    --output-tag="10s10t.000017of000026"

python pyine/apps/write/dataset_writer.py traces \
    --dataset-name=TACO \
    --max-solutions-per-problem=10 \
    --max-tests-per-solution=10 \
    --max-trace-var-repr-length=10000 \
    --max-trace-valid-events=20000 \
    --generate-obfuscated-solutions \
    --target-problem-ids=data/splits/TACO-split.problem_ids.000018of000026.yaml \
    --output-tag="10s10t.000018of000026"

python pyine/apps/write/dataset_writer.py traces \
    --dataset-name=TACO \
    --max-solutions-per-problem=10 \
    --max-tests-per-solution=10 \
    --max-trace-var-repr-length=10000 \
    --max-trace-valid-events=20000 \
    --generate-obfuscated-solutions \
    --target-problem-ids=data/splits/TACO-split.problem_ids.000019of000026.yaml \
    --output-tag="10s10t.000019of000026"

python pyine/apps/write/dataset_writer.py traces \
    --dataset-name=TACO \
    --max-solutions-per-problem=10 \
    --max-tests-per-solution=10 \
    --max-trace-var-repr-length=10000 \
    --max-trace-valid-events=20000 \
    --generate-obfuscated-solutions \
    --target-problem-ids=data/splits/TACO-split.problem_ids.000020of000026.yaml \
    --output-tag="10s10t.000020of000026"

python pyine/apps/write/dataset_writer.py traces \
    --dataset-name=TACO \
    --max-solutions-per-problem=10 \
    --max-tests-per-solution=10 \
    --max-trace-var-repr-length=10000 \
    --max-trace-valid-events=20000 \
    --generate-obfuscated-solutions \
    --target-problem-ids=data/splits/TACO-split.problem_ids.000021of000026.yaml \
    --output-tag="10s10t.000021of000026"

python pyine/apps/write/dataset_writer.py traces \
    --dataset-name=TACO \
    --max-solutions-per-problem=10 \
    --max-tests-per-solution=10 \
    --max-trace-var-repr-length=10000 \
    --max-trace-valid-events=20000 \
    --generate-obfuscated-solutions \
    --target-problem-ids=data/splits/TACO-split.problem_ids.000022of000026.yaml \
    --output-tag="10s10t.000022of000026"

python pyine/apps/write/dataset_writer.py traces \
    --dataset-name=TACO \
    --max-solutions-per-problem=10 \
    --max-tests-per-solution=10 \
    --max-trace-var-repr-length=10000 \
    --max-trace-valid-events=20000 \
    --generate-obfuscated-solutions \
    --target-problem-ids=data/splits/TACO-split.problem_ids.000023of000026.yaml \
    --output-tag="10s10t.000023of000026"

python pyine/apps/write/dataset_writer.py traces \
    --dataset-name=TACO \
    --max-solutions-per-problem=10 \
    --max-tests-per-solution=10 \
    --max-trace-var-repr-length=10000 \
    --max-trace-valid-events=20000 \
    --generate-obfuscated-solutions \
    --target-problem-ids=data/splits/TACO-split.problem_ids.000024of000026.yaml \
    --output-tag="10s10t.000024of000026"

python pyine/apps/write/dataset_writer.py traces \
    --dataset-name=TACO \
    --max-solutions-per-problem=10 \
    --max-tests-per-solution=10 \
    --max-trace-var-repr-length=10000 \
    --max-trace-valid-events=20000 \
    --generate-obfuscated-solutions \
    --target-problem-ids=data/splits/TACO-split.problem_ids.000025of000026.yaml \
    --output-tag="10s10t.000025of000026"

python pyine/apps/write/dataset_writer.py traces \
    --dataset-name=TACO \
    --max-solutions-per-problem=10 \
    --max-tests-per-solution=10 \
    --max-trace-var-repr-length=10000 \
    --max-trace-valid-events=20000 \
    --generate-obfuscated-solutions \
    --target-problem-ids=data/splits/TACO-split.problem_ids.000026of000026.yaml \
    --output-tag="10s10t.000026of000026"

```
