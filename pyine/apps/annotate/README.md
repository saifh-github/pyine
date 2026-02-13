# Trace Annotation Apps

This package contains two CLIs that work together to annotate trace datasets with LLM-generated
documentation hints and then validate those annotations for quality.

## Apps

### Generator (`trace_annot_generator.py`)

Runs prompt chains over a traces dataset to generate annotations (code hints, stubs, issues, etc.)
and stores the results in the framework's prompt results SQLite DB. See the parent
[`README.md`](../README.md#trace-annotation-prompt-chains-over-traces) for usage examples.

### Validator (`trace_annot_validator.py`)

Queries the prompt result DB for misleading-tagged annotation records, loads corresponding traces
for ground truth, and uses an LLM prompt to assess whether each hint actually misleads a reader.
Validation verdicts are stored as new records in the same DB.

```bash
# minimal dry-run over a small subset
python -m pyine.apps.annotate.trace_annot_validator \
    --dataset /path/to/traces/dataset \
    --llm-option provider=openai \
    --llm-option model=gpt-4o-mini \
    --target-indices 0-5 \
    --dry-run

# full validation run
python -m pyine.apps.annotate.trace_annot_validator \
    --dataset-latest-from TACO \
    --llm-option provider=openai \
    --llm-option model=gpt-4o-mini
```

## Pipeline overview

```
traces dataset
      |
      v
[trace_annot_generator]  -- generates hints via LLM prompt chains
      |
      v
prompt result DB          -- stores annotation records (prompt_name = "hints/docs", "issues/docs", etc.)
      |
      v
[trace_annot_validator]   -- reads misleading annotations, asks LLM to judge hint quality
      |
      v
prompt result DB          -- stores validation records (prompt_name = "validation/misleading")
```

## Prompt result DB schema

### Generator records (source annotations)

Records produced by the generator use a prompt name like `"hints/docs"` or `"issues/docs"`.
The `identifier` field is a trace key from the dataset (e.g., `"TACO/train/p000001/s0001/t0001"`).
Tags include `augment:misleading`, `augment:hinted`, `augment:bugged`, etc., depending on the
augmentation type.

### Validator records

Each validation produces one `PromptResultRecord` with `prompt_name = "validation/misleading"`:

| Field                     | Value                                                                                                                                                                                  |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `identifier`              | Source record's `record_uid` (1:1 join key)                                                                                                                                            |
| `group`                   | Preserved from source (typically a solution ID)                                                                                                                                        |
| `result`                  | Raw LLM JSON, e.g. `{"verdict": "MISLEADING", "explanation": "..."}`                                                                                                                   |
| `meta.verdict`            | `"MISLEADING"`, `"NOT_MISLEADING"`, or `"UNINFORMATIVE"`                                                                                                                               |
| `meta.explanation`        | LLM's reasoning                                                                                                                                                                        |
| `meta.source_record_uid`  | UID of the annotation record that was validated                                                                                                                                        |
| `meta.source_prompt_name` | e.g. `"hints/docs"` or `"issues/docs"`                                                                                                                                                 |
| `meta.source_identifier`  | Original trace key from the annotation record                                                                                                                                          |
| `tags`                    | Includes `verdict:misleading` / `verdict:not_misleading` / `verdict:uninformative`, plus `validation:misleading`, problem tags, trace tags, `augment:*` lineage, and LLM provider tags |

**Verdict semantics:**

- **`MISLEADING`**: the hints actively point toward a wrong execution outcome (misleading attempt succeeded)
- **`NOT_MISLEADING`**: the hints correctly describe or are compatible with the true behavior (misleading attempt failed)
- **`UNINFORMATIVE`**: there are no meaningful hints about execution behavior (too generic, or no documentation present)

### Filtering pipeline

Before validation, source records pass through two filters (both use `SampleCodeTypeSet` for
robust tag classification):

1. **Misleading filter**: keeps only records whose tags imply the `misleading` code type
2. **Bugged filter**: removes records whose tags imply the `bugged` code type (since
   `trace.expected_output` is not authoritative for bugged code)

## Downstream usage

### Check if a specific annotation was validated

```python
validation_records = db.get_by_identifier(
    source_record.record_uid,
    prompt_name="validation/misleading",
)
if validation_records:
    verdict = validation_records[0].meta["verdict"]
```

### Bulk-query records by verdict

```python
# all records where the LLM confirmed the hints are misleading
misleading = db.get_by_prompt_name(
    "validation/misleading",
    tag_filter_rule="+verdict:misleading",
)

# all records where the hints were uninformative
uninformative = db.get_by_prompt_name(
    "validation/misleading",
    tag_filter_rule="+verdict:uninformative",
)
```

### Trace back to the original annotation

```python
source_uid = validation_record.meta["source_record_uid"]
source_prompt = validation_record.meta["source_prompt_name"]
source_identifier = validation_record.meta["source_identifier"]

# fetch the original annotation record
original = db.get_by_identifier(source_identifier, prompt_name=source_prompt)
```

## Shared utilities

- **`pyine/apps/annotate/_shared.py`**: CLI helpers shared between both apps (dataset path
  resolution, LLM provider config building, YAML/JSON parsing, async wrapper)
- **`pyine/organisms/datamodules/utils/validator.py`**: domain logic for the validator (tag
  building, record filtering, verdict parsing, per-record processing)
- **`pyine/prompts/configs/validation/misleading.py`**: `VerdictPayload` pydantic model and
  output parser for the validation prompt
