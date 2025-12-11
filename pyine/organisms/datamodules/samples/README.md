# Samples Package

(developer notes)

This package implements a three-stage data transformation pipeline that converts raw execution
traces from LMDB datasets into trace samples suitable for training/evaluating models. The samples
produced here are designed to be reusable across different data generation pipelines and data
loaders.

Currently, the primary consumers of this package are the `KeywordBiasDataModule` and the
`ShortcutBiasDataModule`, which use it to prepare code execution prediction samples for model
training and evaluation. Future data modules may also leverage this package to ensure consistency
in sample generation.

## Package Structure

```
samples/
├── __init__.py       # Public API exports
├── common.py         # Core data structures, enums, and utilities
├── configs.py        # Pydantic configuration classes for all stages
├── filtering.py      # Stage 1: Trace filtering logic
├── selection.py      # Stage 2: Sample selection from trace families
├── transform.py      # Stage 3: Sample generation and transformation
└── builder.py        # Main orchestrator (SampleBuilder class)
```

## Core Concepts

### Sample Code Types (`SampleCodeType`)

Defines the types of code augmentations that can be applied to execution traces:

| Type         | Description                                                         |
| ------------ | ------------------------------------------------------------------- |
| `original`   | Unmodified code from the source dataset                             |
| `obfuscated` | Code with obfuscated variables/functions (e.g., `a`, `b`, `c`, ...) |
| `stubbed`    | Code with hidden/masked sections (cannot be executed)               |
| `hinted`     | Code with execution output hints provided by an LLM                 |
| `misleading` | Code with false/misleading execution hints provided by an LLM       |
| `bugged`     | Code intentionally containing bugs that affect execution outcomes   |

**Important constraints on code type combinations:**

- `original` cannot be combined with any other type (otherwise it's no longer original...);
- `stubbed` cannot be combined with `hinted`, `misleading`, or `bugged` (it's hard to tell
  if stubbing kepts those intact);
- `misleading` and `hinted` are mutually exclusive (otherwise evaluations get ambiguous).

### Sample Prediction Types (`SamplePredictType`)

Defines what aspect of code execution the model should predict:

| Type              | Description                                                |
| ----------------- | ---------------------------------------------------------- |
| `program_output`  | Final output after executing the entire program            |
| `frame_variables` | In-memory variables at a specific code line                |
| `function_return` | Return value of a specific function call                   |
| `next_step_key`   | Trace key of the next execution step (not yet implemented) |

### Trace Families

A **trace family** is a set of traces that share the same original code and execution arguments,
but differ in how the code was augmented. For example, the same solution with the same test case
inputs might have:

- one trace with original code;
- one trace with obfuscated code;
- one trace with hinted and bugged code;
- etc.

This grouping enables controlled diversity in sample generation: we can sample from different
families to avoid redundancy while still covering various augmentation types.

### SampleData

The primary output data structure (`SampleData`) is a `NamedTuple` containing all information
needed for a training sample:

```python
SampleData(
    identifier: str,                  # unique trace identifier
    code: str,                        # full code snippet (always included for context)
    description: str,                 # high-level code description (may be empty)
    entrypoint: str,                  # function name for callable programs, or empty
    first_line: int,                  # first execution line (0 for full execs)
    last_line: int,                   # last potential execution line
    inputs: str,                      # execution inputs or intermediate program state
    expected_output: str,             # target prediction value
    predict_type: SamplePredictType,  # prediction (task) type
    code_type: str,                   # augmentation type(s) associated with the code
    trace_step_count: int,            # number of execution steps to completion
    comma_separated_tags: str,        # filterable metadata (for e.g. evals)
    has_code_override: bool,          # whether code was augmented using prompt result DB
    complexity_metrics: dict,         # code complexity measurements
    first_line_hit: int,              # which visit to first_line (for code segments)
    last_line_hit: int,               # which visit to last_line (for code segments)
    first_step_idx: int,              # absolute trace index of segment start
    last_step_idx: int,               # absolute trace index of segment end
)
```

## Pipeline Stages

### Stage 1: Filtering (`filtering.py`)

**Purpose:** Reduce the dataset to a manageable, high-quality subset of traces, possibly by
targeting traces that would be harder to work with.

**Configuration (`TraceFilteringConfig`):**

- `max_trace_families`: cap on total unique trace families allowed for sampling;
- `max_trace_steps`: skip traces exceeding this execution step count;
- `max_code_line_count`: skip traces exceeding this code line threshold;
- `max_code_line_length`: skip traces with overly long individual code lines;
- `max_code_length`: skip traces with massive code strings (e.g. that include whole libraries);
- `max_args_length`: skip traces with huge input/output strings (to avoid hard test cases).

**Output (`TraceFilteringResults`):**

- Groups remaining traces into families;
- Tracks why traces were filtered (step count, code length, args length, family cap);
- Uses round-robin sampling when capping families to ensure diversity across coding problems.

### Stage 2: Selection (`selection.py`)

**Purpose:** Choose which traces to use and what code versions to sample from each trace family. If
possible and allowed, will also consider pre-generated yet untraced code augmentations from the
[prompt result database](../../../prompts/README.md) in order to further diversity samples.

**Configuration (`SampleSelectionConfig`):**

- `code_type_prob_map`: probability distribution for selecting code augmentation types;
- `samples_per_family`: how many samples to generate per trace family;
- `draw_attempts`: retry count before giving up on a family;
- `allow_db_lookups`: whether to fetch augmented code from the [prompt result database](../../../prompts/README.md);
- `fallback_to_orig`: whether to fallback to original code if augmented code is unavailable.

**Algorithm:**

1. For each trace family, attempt to draw the desired code type(s);
2. If available in the trace dataset, use it directly;
3. If not available but `allow_db_lookups=True`, query the [prompt result database](../../../prompts/README.md);
4. If still not found and `fallback_to_orig=True`, use the original (unaugmented) code;
5. Track statistics on success/fallback/failure.

**Output (`SampleSelectionResults`):**

- List of `SelectedSample` objects specifying which trace to use and what code to apply;
- Counts of samples from direct traces vs. prompt DB lookups vs. fallbacks.

### Stage 3: Transformation (`transform.py`)

**Purpose:** Convert selected traces into training-ready samples with specific prediction targets.

**Configuration (`SampleTransformConfig`):**

- `transform_strategy`: when to create partial samples (`never`, `always`, `if_too_long`,
  `random`, `hybrid`);
- `predict_type_prob_map`: distribution for selecting prediction types (under `random` or `hybrid`
  transform strategies);
- `too_long_total_steps_threshold`: what constitutes "too long", in terms of step count;
- `max_partial_trace_steps`: cap on partial samples step count;
- `min_partial_trace_steps`: required minimum partial samples step count;
- `max_inputs_str_length` / `max_output_str_length`: caps on inputs/outputs string lengths;
- `combine_local_and_global_vars_for_partial_samples`: whether to merge input variable scopes;
- `fallback_to_orig`: whether to return full sample if partial sample transformation fails.

## Usage

### Basic Usage with SampleBuilder

```python
from pyine.organisms.datamodules.samples import SampleBuilder

builder = SampleBuilder(
    source_data=["path/to/traces.lmdb"],
    filtering_config={
        "max_trace_steps": 5000,
        "max_code_length": 8000,
    },
    selection_config={
        "code_type_prob_map": {"original": 0.7, "obfuscated": 0.3},
        "fallback_to_orig": True,
    },
    transform_config={
        "transform_strategy": "hybrid",
        "predict_type_prob_map": {
            "function_return": 0.4,
            "frame_variables": 0.4,
        },
    },
)

# PyTorch-compatible interface
sample_count = len(builder)
sample = builder[0]  # returns SampleData instance
stats = builder.get_stats()  # returns statistics dict
```

### Integration with DataModules

The `SampleBuilder` is typically instantiated by a data module (e.g., `ShortcutBiasDataModule`)
which handles:

- LMDB reader management;
- Train/valid/test split assignment;
- Configuration via Hydra-zen.

## Important Design Considerations

### Prompt Result Database Integration

Augmented code can come from two sources:

1. **LMDB trace datasets** written using the [dataset_writer.py](../../../apps/README.md) app: these
   contain full trace data for specific code snippets which support all sample prediction types;
2. **[Prompt result database](../../../prompts/README.md)** results: these only support samples
   where we try to predict full program outputs. This is because we don't have intermediate
   execution information and thus cannot access the required execution step data to generate
   partial samples. Supporting all prediction types for augmented code contained in this database
   can be done by creating a new LMDB trace dataset for these samples.

### Deterministic RNG

All random operations use seeded RNGs based on configuration seed, epoch, and sample index for
reproducibility. The seed sequence `[config.seed, epoch, sample_idx]` ensures that:

- Different epochs may produce different samples (epoch-level variation);
- The same sample index within an epoch is deterministic.

### Lazy Transformation

Samples are not fully generated until accessed via the builder's `__getitem__`. This enables:

- Dynamic randomization per epoch;
- Memory efficiency (don't store all transformed samples).

However, that also means we cannot predetermine all the statistics for the samples that will be
generated, as many transformation decisions require access to the full trace data.

### Strong Coupling Warning

There are strong links between:

- Sample code type definitions in this package;
- Prompt templates supported in the annotator app;
- How the trace dataset writer saves augmentation info; and
- How trace identifiers encode augmentation status.

If you change any of these, ensure all related areas are updated accordingly.
