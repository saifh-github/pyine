# Datamodule Configuration Setup Guide for GRPO Training

This guide explains how to properly configure your datamodule for GRPO training with your existing TACO dataset.

## Quick Start

### Option 1: Automatic Configuration (Recommended)

Use the helper script to automatically detect your dataset paths and generate a properly configured datamodule file:

```bash
# For medium-sized experiments (15% of data, ~4 parts)
python -m pyine.apps.rl_trainers_proto.create_datamodule_config \
    --output pyine/apps/rl_trainers_proto/configs/taco_part1to4.yaml \
    --dataset-name TACO_10s10t_v1_part1to4

# For quick testing (limit to 100 solutions)
python -m pyine.apps.rl_trainers_proto.create_datamodule_config \
    --output pyine/apps/rl_trainers_proto/configs/taco_part1_test.yaml \
    --dataset-name TACO_10s10t_v1_part1 \
    --max-solutions 100

# For full experiments (100% of data, all 26 parts)
python -m pyine.apps.rl_trainers_proto.create_datamodule_config \
    --output pyine/apps/rl_trainers_proto/configs/taco_full.yaml \
    --dataset-name TACO_10s10t_v1_full
```

The script will:
1. Automatically detect your TACO dataset paths
2. Verify the split file exists
3. Generate a complete, ready-to-use configuration file with GRPO-optimized settings:
   - `grpo_minimal` prompt template (50% shorter, zero-shot by default)
   - Plain text format (not chat)
   - Structured output format for easy reward parsing
4. Show you the next steps

### Option 2: Manual Configuration

If you prefer to create the config manually, see `configs/datamodule_grpo_example.yaml` for a template. You'll need to:

1. Find your LMDB paths:
   ```python
   import pyine.data.traces.dataset_utils
   paths = pyine.data.traces.dataset_utils.get_matching_dataset_paths(
       "TACO", "v1.5/10s10t.*of000026.*.lmdb"
   )
   print(paths[0:4])  # For part1to4
   ```

2. Find your split file:
   ```python
   import pyine.data.utils.splits
   split_path = pyine.data.utils.splits.get_dataset_split_file_path("TACO")
   print(split_path)
   ```

3. Update the paths in the YAML file

## Dataset Size Options

Choose based on your computational resources and experiment goals:

| Dataset Variant | Size | Parts | Use Case |
|----------------|------|-------|----------|
| `shortcuts_TACO_10s10t_v1_part1` | ~3.8% | 1/26 | Quick prototyping, debugging |
| `shortcuts_TACO_10s10t_v1_part1to4` | ~15% | 4/26 | Medium experiments, proof of concept |
| `shortcuts_TACO_10s10t_v1_part1to13` | ~50% | 13/26 | Large experiments, good representation |
| `shortcuts_TACO_10s10t_v1_full` | 100% | 26/26 | Full experiments, final results |

**Recommendation for GRPO testing:** Start with `part1` or `part1to4` with `--max-solutions 100` to quickly validate everything works.

## Validation Workflow

After generating your config, **always validate it before training**:

### Step 1: Check the Config File

```bash
# View the generated config
cat pyine/apps/rl_trainers_proto/configs/taco_part1to4.yaml
```

Verify:
- ✓ All LMDB paths exist and are correct
- ✓ Split file path exists
- ✓ `prompt_config` is set correctly
- ✓ `max_solution_count` is appropriate (if set)

### Step 2: Inspect Generated Prompts

```bash
# Inspect prompts from the datamodule
python -m pyine.apps.rl_trainers_proto.inspect_prompts \
    --datamodule-config pyine/apps/rl_trainers_proto/configs/taco_part1to4.yaml \
    --subset train \
    --num-samples 5 \
    --max-samples 50 \
    --max-prompt-length 2000
```

**Review the output carefully:**

✓ **Prompt Structure:**
- Should include system role instructions
- Should include few-shot examples
- Should have the actual code snippet and inputs
- Should specify the output type clearly

✓ **Expected Outputs:**
- Should show the correct expected output for each sample
- Should match the predict_type (program_output, frame_variables, or function_return)

✓ **Statistics:**
- Check the distribution of output types
- Verify the code types match your expectations

### Step 3: Save Inspection for Review (Optional)

```bash
# Save to file for detailed review
python -m pyine.apps.rl_trainers_proto.inspect_prompts \
    --datamodule-config pyine/apps/rl_trainers_proto/configs/taco_part1to4.yaml \
    --subset train \
    --num-samples 10 \
    --output-file prompt_inspection_report.txt

# Review the report
cat prompt_inspection_report.txt
```

## Using the Config in GRPO Training

Once validated, update your `train_grpo.py` configuration:

```python
from pyine.apps.rl_trainers_proto.config import DataConfig, ExperimentConfig

config = ExperimentConfig(
    experiment_name="GRPO_CodeExec_Test",
    seed=42,
    use_wandb=True,
    wandb_project="your-project-name",
    data=DataConfig(
        use_datamodule=True,
        datamodule_config_path="pyine/apps/rl_trainers_proto/configs/taco_part1to4.yaml",
        train_subset_name="train",
        max_samples=1000,  # Start small for testing
    ),
    # ... rest of config
)
```

## Configuration Fields Explained

### Key Fields You Might Want to Modify

#### `max_solution_count`
```yaml
max_solution_count: 100  # Limit dataset size for testing
```
- Controls how many coding problems to include
- Good for quick experiments
- Set to `null` or remove for full dataset

#### `prompt_config.version`
```yaml
prompt_config:
  version: "grpo_minimal"  # Default for GRPO (optimized, zero-shot, structured output)
  # OR
  version: "unstructured_with_3_output_types"  # More verbose with examples
  # OR
  version: "no_pressure_demo"  # Simpler prompts without output type complexity
```

**For GRPO training:** Use `grpo_minimal` (default in generated configs). It's:
- ~50% shorter than evaluation templates
- Zero-shot by default (saves tokens with N samples)
- Structured output format (` ```output...``` `) for easy reward parsing

#### `prompt_config.use_chat_template`
```yaml
prompt_config:
  use_chat_template: false  # Required for GRPO (plain text, not chat messages)
```

#### `prompt_config.include_examples`
```yaml
prompt_config:
  include_examples: false  # Default for GRPO (zero-shot)
  # Set to true if you want few-shot examples (costs more tokens)
```

#### `input_type_prob_map`
```yaml
input_type_prob_map:
  original: 1.0  # Only use original code (no augmentations)
  hinted: 0.0    # Set to > 0.0 to include hint-augmented code
  obfuscated: 0.0  # Set to > 0.0 to include obfuscated code
```

### What Each File Contains

**Your experiment config** (`configs/experiment/original/v0.yaml`):
- High-level experiment settings
- Model configuration (LoRA, quantization)
- Training arguments (batch size, learning rate)
- References the datamodule config via `override /config/datamodule_config`

**Standalone datamodule config** (`configs/taco_part1to4.yaml`):
- Dataset paths and split file
- Prompt template configuration
- Sample selection strategy
- Data filtering rules
- **This is what you use for GRPO training**

## Troubleshooting

### "FileNotFoundError: No TACO dataset files found"

Your TACO dataset is not in the expected location. Check:

1. Environment variable: `echo $PYINE_TRACE_DATASETS_ROOT`
2. Default location: `~/.cache/pyine/trace_datasets/`
3. Dataset structure should be: `TACO/v1.5/10s10t_part01of000026_*.lmdb`

See the main project README for dataset setup instructions.

### "Split file not found"

The dataset split file is missing. Check:

1. Environment variable: `echo $PYINE_DATASET_SPLITS_ROOT`
2. Default location: `~/.cache/pyine/dataset_splits/`
3. File pattern: `TACO_split_*.json`

### "Prompts look weird or incomplete"

Common causes:
- Wrong prompt version: Try changing `prompt_config.version` to `"no_pressure_demo"`
- Missing examples: Check `prompt_config.include_examples` is `true`
- Template issue: Verify the `code_execution.yaml` template file exists

### "Datamodule fails to load"

Check:
1. Config file is valid YAML (no syntax errors)
2. All paths in the config exist
3. `datamodule_class_path` is correct
4. You have required Python packages installed

## Best Practices

1. **Start Small:** Always test with a small dataset first (`part1` + `max_solutions: 100`)

2. **Inspect Before Training:** Use `inspect_prompts.py` to validate prompts look correct

3. **Version Control:** Keep your datamodule configs in version control (they're small YAML files)

4. **Naming Convention:** Use descriptive names like `taco_part1to4_test100.yaml` so you know what's in them

5. **Document Changes:** Add comments to your config files explaining non-standard settings

6. **Reproducibility:** Always set the same `seed` in both the datamodule config and training config

## Next Steps

After setting up your datamodule config:

1. ✅ **Validate prompts** with `inspect_prompts.py`
2. ✅ **Test with small dataset** (part1, 100 solutions)
3. ✅ **Run short training** (1 epoch, verify no errors)
4. ✅ **Implement Approach 2** (code execution rewards)
5. ✅ **Scale up** to full dataset once everything works

## Summary

You now have two ways to create datamodule configs:

- **Automatic:** Use `create_datamodule_config.py` to auto-detect paths ⭐ Recommended
- **Manual:** Edit `datamodule_grpo_example.yaml` template

Both produce standalone YAML files you can use with `inspect_prompts.py` and `train_grpo.py`.

Always validate your prompts before training!
