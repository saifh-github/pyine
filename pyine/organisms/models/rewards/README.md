# Rewards Package

Composable reward term management for RL experiments.

## Architecture

```
RewardManager
    │
    ├── OutputParser (optional)       # Parses model output once per sample
    │
    ├── RewardTerm[]                  # Enabled terms from config
    │   ├── parseable_answer
    │   ├── text_length
    │   └── (custom terms)
    │
    ├── WeightedSumAggregator         # Combines term values
    │
    └── RewardLogger (optional)       # Logs metrics to W&B/memory
```

## Quick Start

```python
import pyine.organisms.models.rewards.core.configs as reward_configs
import pyine.organisms.models.rewards.core.manager as reward_manager
import pyine.organisms.models.rewards.core.types as reward_types

# Configure the manager
config = reward_configs.RewardManagerConfig(
    terms=[
        reward_configs.RewardTermSpec(
            name="parseable",
            type="parseable_answer",
            weight=1.0,
            params={"reward_if_present": 1.0, "reward_if_missing": 0.0},
        ),
    ],
    parsing=reward_configs.ParsingConfig(
        final_tag="answer",
        fallback_policy="none",
    ),
)

# Create manager
manager = reward_manager.RewardManager(config)

# Compute reward for a sample
ctx = reward_types.SampleContext(
    prompt="What is 2+2?",
    model_output="<answer>4</answer>",
    sample_data=sample_data,  # from the datamodule
)
reward = manager.compute(ctx)  # returns float
```

## Built-in Terms

### `parseable_answer`

Rewards samples whose output contains a parseable final answer tag.

```python
params = {
    "final_tag": "final",              # tag name to look for
    "reward_if_present": 1.0,          # reward when tag found
    "reward_if_missing": 0.0,          # reward when missing
    "bonus_if_stops_after_final_tag": 0.0,
    "bonus_if_single_final_block": 0.0,
}
```

### `text_length`

Maps text lengths to rewards for budget shaping.

```python
params = {
    "components": [
        {
            "name": "prompt",
            "source": "prompt",         # prompt | model_output | parsed_reasoning | ...
            "unit": "chars",            # chars | lines | openai_tokens
            "flat_until_length": 100,
            "end_length": 500,
            "reward_at_start": 1.0,
            "reward_at_end": 0.0,
        }
    ]
}
```

## Multi-Term Reward Shaping

Combine multiple terms with different weights for complex reward signals:

```python
config = reward_configs.RewardManagerConfig(
    terms=[
        # Primary: correctness reward (high weight)
        reward_configs.RewardTermSpec(
            name="format_compliance",
            type="parseable_answer",
            weight=1.0,
            params={
                "reward_if_present": 1.0,
                "reward_if_missing": 0.0,
                "bonus_if_single_final_block": 0.25,
                "bonus_if_stops_after_final_tag": 0.25,
            },
        ),
        # Secondary: brevity bonus (lower weight)
        reward_configs.RewardTermSpec(
            name="brevity",
            type="text_length",
            weight=0.1,
            params={
                "components": [
                    {
                        "name": "output_length",
                        "source": "model_output",
                        "unit": "chars",
                        "start_length": 0,
                        "end_length": 2000,
                        "reward_at_start": 1.0,
                        "reward_at_end": 0.0,
                    }
                ]
            },
        ),
    ],
    parsing=reward_configs.ParsingConfig(final_tag="answer"),
    aggregation=reward_configs.AggregationConfig(
        clip_total_min=0.0,
        clip_total_max=2.0,
    ),
)
```

## Creating Custom Terms

### 1. Define a config class

```python
import pyine.organisms.models.rewards.core.types as reward_types

class MyTermConfig(reward_types.BaseConfig):
    threshold: float = 0.5
    penalty: float = -0.1
```

### 2. Implement the term

```python
import pyine.organisms.models.rewards.core.term as reward_term

class MyTerm(reward_term.BaseRewardTerm):
    def __init__(self, config: MyTermConfig) -> None:
        self._config = config

    def __call__(self, sample_ctx: reward_types.SampleContext) -> reward_types.TermResult:
        # Access parsed output if available
        has_answer = (
            sample_ctx.parsed is not None
            and sample_ctx.parsed.final_answer is not None
        )

        # Compute reward based on your logic
        if has_answer:
            value = 1.0
        else:
            value = self._config.penalty

        # Return result with optional metrics
        return reward_types.TermResult(
            value=value,
            metrics={"has_answer": has_answer},
        )
```

### 3. Register the factory

```python
import pyine.organisms.models.rewards.core.registry as reward_registry
import pyine.organisms.models.rewards.core.configs as reward_configs

def _factory(
    spec: reward_configs.RewardTermSpec,
    *,
    parser: reward_types.OutputParser | None,
) -> reward_types.RewardTerm:
    config = MyTermConfig.model_validate(spec.params)
    return MyTerm(config)

reward_registry.register_term("my_term", _factory)
reward_registry.register_term_aliases("my_term", ["custom/my_term"])
```

## Creating Custom Parsers

Implement the `OutputParser` protocol for custom parsing logic:

```python
import pyine.organisms.models.rewards.core.types as reward_types

class JsonOutputParser:
    """Parser that extracts JSON from model output."""

    def parse(
        self,
        prompt: str,
        model_output: str,
    ) -> reward_types.ParsedOutput:
        import json
        import re

        # Try to find JSON in the output
        json_match = re.search(r'\{[^{}]*\}', model_output)
        final_answer = None
        if json_match:
            try:
                parsed_json = json.loads(json_match.group())
                final_answer = parsed_json.get("answer")
            except json.JSONDecodeError:
                pass

        return reward_types.ParsedOutput(
            raw=model_output,
            final_answer=final_answer,
            reasoning=None,
            fields={},
        )

# Use custom parser with manager
manager = reward_manager.RewardManager(config, parser=JsonOutputParser())
```

## Integration with Training Loops

### Basic Integration

```python
import pyine.organisms.models.rewards.core.types as reward_types

# Initialize manager once
manager = reward_manager.RewardManager(config)

# Reset at start of each training run/epoch
manager.reset(reward_types.RunInitContext(datamodule=datamodule))

# In training loop
for batch in dataloader:
    prompts, outputs, sample_data_list = generate_outputs(batch)

    rewards = []
    for prompt, output, sample_data in zip(prompts, outputs, sample_data_list):
        ctx = reward_types.SampleContext(
            prompt=prompt,
            model_output=output,
            sample_data=sample_data,
        )
        reward = manager.compute(ctx)
        rewards.append(reward)

    # Use rewards for policy gradient update
    loss = compute_policy_loss(rewards, ...)

# At end of run
manager.finalize_run()
```

### Batch Processing

```python
# Process multiple samples at once
sample_ctxs = [
    reward_types.SampleContext(prompt=p, model_output=o, sample_data=sd)
    for p, o, sd in zip(prompts, outputs, sample_data_list)
]
outputs = manager.compute_batch(sample_ctxs)
rewards = [out.total for out in outputs]
```

### With Logging

```python
import pyine.organisms.models.rewards.core.logging as reward_logging

# Create W&B logger
logger = reward_logging.make_wandb_reward_logger(
    wandb_run,
    config.logging,
    step=global_step,
)

# Create manager with logger
manager = reward_manager.RewardManager(config, logger=logger)

# Set step before computing
manager.set_step(global_step)

# Rewards are automatically logged according to LoggingConfig
reward = manager.compute(ctx)
```

## Distributed Training

The rewards package supports distributed training with proper synchronization:

```python
config = reward_configs.RewardManagerConfig(
    terms=[...],
    logging=reward_configs.LoggingConfig(
        enabled=True,
        main_process_only=True,           # Only log on rank 0
        gather_distributed_summaries=True, # Gather stats from all ranks
        barrier_before_finalize=True,      # Sync before finalize
    ),
)

# Manager automatically:
# - Skips logging on non-main ranks (when main_process_only=True)
# - Gathers statistics from all ranks (when gather_distributed_summaries=True)
# - Synchronizes before finalize (when barrier_before_finalize=True)
```

## Configuration Reference

### RewardManagerConfig

- `terms`: List of `RewardTermSpec` (required, min 1)
- `aggregation`: `AggregationConfig` (optional)
- `logging`: `LoggingConfig` (optional)
- `output`: `OutputConfig` (optional)
- `parsing`: `ParsingConfig` (optional)

### RewardTermSpec

- `name`: Stable term name used as breakdown/logging key
- `type`: Registry key for the term implementation
- `weight`: Scalar weight applied during aggregation (default: 1.0)
- `enabled`: Whether the term is active (default: True)
- `require_parsed`: Whether this term requires parsed output (default: False)
- `params`: Term-specific configuration payload

### AggregationConfig

- `strategy`: `"weighted_sum"` (default, only supported strategy)
- `clip_term_min/max`: Optional per-term clipping (applied before weighting)
- `clip_total_min/max`: Optional total clipping (applied after summation)

### LoggingConfig

- `enabled`: bool (default False)
- `wandb_key_prefix`: Extra key prefix for W&B logging
- `log_total`: Include total reward in logs (default True)
- `log_terms`: Include per-term values in logs (default True)
- `log_metrics`: Include term-emitted metrics in logs (default True)
- `log_every_n_examples`: int (default 1)
- `scope_prefix`: str (default "reward/")
- `main_process_only`: bool (default True)
- `gather_distributed_summaries`: Gather stats across ranks (default False)
- `barrier_before_finalize`: Sync before finalize (default False)
- `log_tables`: Log per-sample breakdowns to W&B table (default False)
- `table_key`: W&B key for the rewards table
- `table_flush_every_n_logs`: Flush table every N logs (default 100)
- `table_max_rows`: Max buffered rows before flush (default 1000)

### ParsingConfig

- `mode`: `"tags"` (default, only supported mode)
- `enabled_fields`: `"both"` | `"final_only"` | `"reasoning_only"`
- `final_tag`: Tag name for final answer (default "final")
- `reasoning_tag`: Tag name for reasoning (default "reasoning")
- `reasoning_from_final_prefix`: Use text before final tag as reasoning (default False)
- `fallback_policy`: `"none"` | `"last_line"` | `"entire_output"`
- `multi_tag_policy`: `"last"` | `"first"` | `"error"`
- `strict`: Raise on malformed tag structure (default False)
- `capture_diagnostics`: Include tag diagnostics in fields (default True)

### OutputConfig

- `return_breakdown_default`: Whether `compute()` returns breakdown by default
- `return_unweighted_breakdown`: Include unweighted values in breakdown

## Module Structure

```
rewards/
├── core/
│   ├── types.py        # Protocols, dataclasses
│   ├── configs.py      # Pydantic configurations
│   ├── manager.py      # RewardManager orchestration
│   ├── registry.py     # Term factory registry
│   ├── aggregator.py   # Weighted sum aggregation
│   ├── parser.py       # Tag-based output parsing
│   ├── logging.py      # W&B and in-memory loggers
│   └── term.py         # BaseRewardTerm helper
└── terms/
    └── format/
        ├── parseable_answer.py
        └── text_length.py
```

## Testing

Use registry snapshots for test isolation:

```python
import pyine.organisms.models.rewards.core.registry as reward_registry
import pyine.organisms.models.rewards.terms

# Ensure builtins are registered
pyine.organisms.models.rewards.terms.ensure_builtin_terms_registered()

# Take a snapshot for isolated testing
snapshot = reward_registry.get_global_registry().snapshot()

# Use snapshot with manager
manager = reward_manager.RewardManager(config, registry=snapshot)
```

Use `InMemoryRewardLogger` for testing logging:

```python
import pyine.organisms.models.rewards.core.logging as reward_logging

logger = reward_logging.InMemoryRewardLogger()
manager = reward_manager.RewardManager(config, logger=logger)

# After computing rewards...
assert len(logger.samples) == expected_count
assert logger.samples[0]["total"] == expected_reward
```
