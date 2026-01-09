# `rewards` Package

Composable reward term management for RL experiments.

## Architecture

```
RewardManager
    │
    ├── OutputParser (optional)       # parses model output once per sample
    │
    ├── RewardTerm[]                  # enabled terms from config
    │   ├── parseable_answer          # format term
    │   ├── text_length               # format term
    │   ├── hard_match                # code_exec term
    │   ├── soft_match                # code_exec term
    │   ├── llm_grader                # code_exec term
    │   └── (custom terms)
    │
    ├── WeightedSumAggregator         # combines term values
    │
    └── RewardLogger (optional)       # logs metrics to W&B/memory
```

## Quick Start

```python
import pyine.organisms.models.rewards.core.configs as reward_configs
import pyine.organisms.models.rewards.core.manager as reward_manager
import pyine.organisms.models.rewards.core.types as reward_types

# configure the manager
config = reward_configs.RewardManagerConfig(
    terms=[
        reward_configs.RewardTermSpec(
            name="parseable",  # name used for breakdown/logging key
            type="parseable_answer",  # registry type used to identify the implemented class
            weight=1.0,  # scalar weight applied to the term value during aggregation
            params={"reward_if_present": 1.0, "reward_if_missing": 0.0},
        ),
    ],
    parsing=reward_configs.ParsingConfig(
        final_tag="answer",  # parseable_answer term will inherit this tag
        fallback_policy="none",
    ),
)

# create manager
manager = reward_manager.RewardManager(config)

# compute reward for a sample; note: sample_data required (should come from datamodule)
ctx = reward_types.SampleContext(
    prompt="What is 2+2?",
    model_output="<answer>4</answer>",
    sample_data=sample_data,
)
output = manager.compute(ctx)
reward = output.total  # structured RewardOutput also has weighted_terms, raw_terms, metrics
```

## Built-in Terms

### `parseable_answer`

Rewards samples whose output has a final answer (requires parsing to be enabled).

Note: This term checks `SampleContext.parsed.final_answer`, not raw output tags. If parsing is
disabled or fails to extract a final answer, the sample receives `reward_if_missing`.

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
            # available sources: prompt, model_output, parsed_reasoning,
            #                    parsed_final_answer, sample_code
            "source": "prompt",
            "unit": "chars",            # chars | lines | openai_tokens
            # note: when unit="openai_tokens", you must also set openai_model_id
            # e.g., "openai_model_id": "gpt-4"
            # length mapping: reward is flat at reward_at_start until flat_until_length (or start_length
            #   if flat_until_length is unset), then interpolates linearly to reward_at_end at end_length
            "flat_until_length": 100,
            "end_length": 500,
            "reward_at_start": 1.0,
            "reward_at_end": 0.0,
        }
    ]
}
```

### Code Execution Terms

These terms evaluate model outputs based on code execution results. They require
`SampleContext.code_exec_eval` to be populated with a `CodeExecEvalData` instance.

#### Flipped rewards (bias mitigation / corrupted samples)

For some sample types (e.g., buggy code or keyword-injected samples), it can be useful to invert
the correctness signal so that "matching the expected output" is treated as a failure for reward
purposes.

Code execution terms support a per-sample "flip" decision:

- Automatic flip: `SampleData.has_bugged_code()` OR `SampleData.has_bias_keyword()` is True.
- Explicit flip: set `CodeExecEvalData.should_flip_reward` (overrides the automatic logic).

Semantics:

- For `hard_match` / `soft_match` / `llm_grader` (binary): when flipped, `reward_if_match` and
  `reward_if_no_match` are swapped.
- For `llm_grader` (continuous): when flipped, the effective score becomes `1.0 - llm_score`.

Diagnostics:

- All code execution terms emit a boolean `reward_flipped` metric.

#### `hard_match`

Exact string comparison between expected and predicted execution outputs.

```python
params = {
    "reward_if_match": 1.0,       # reward when outputs match exactly
    "reward_if_no_match": 0.0,    # reward when they differ
    "strip_whitespace": True,     # strip before comparing
}
```

#### `soft_match`

Heuristic-based semantic comparison with numeric tolerances and structured data support.

```python
params = {
    "reward_if_match": 1.0,
    "reward_if_no_match": 0.0,
    "compare_options": {          # optional, uses defaults if not set
        ...
    },
}
```

#### `llm_grader`

LLM-as-a-judge evaluation using a pre-computed grader score.

```python
params = {
    "reward_if_match": 1.0,
    "reward_if_no_match": 0.0,
    "score_threshold": 0.5,        # threshold for binary decision
    "use_continuous_reward": False, # use raw score instead of binary
    "fallback_to_soft_match": True, # fallback when LLM score unavailable
    "fallback_to_hard_match": False,
}
```

## Multi-Term Reward Shaping

Combine multiple terms with different weights for complex reward signals:

```python
import pyine.organisms.models.rewards.core.configs as reward_configs

config = reward_configs.RewardManagerConfig(
    terms=[
        # primary: correctness rewards (high weights)
        reward_configs.RewardTermSpec(
            name="format_compliance",
            type="parseable_answer",
            weight=0.5,
            params={
                "reward_if_present": 1.0,
                "reward_if_missing": 0.0,
                "bonus_if_single_final_block": 0.25,
                "bonus_if_stops_after_final_tag": 0.25,
            },
        ),
        reward_configs.RewardTermSpec(
            name="accuracy",
            type="hard_match",
            weight=1.0,
            params = {
                "reward_if_match": 1.0,
                "reward_if_no_match": 0.0,
                "strip_whitespace": True,
            },
        ),
        # secondary: brevity bonus (lower weight)
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
    parsing=reward_configs.ParsingConfig(final_tag="final"),
    aggregation=reward_configs.AggregationConfig(
        clip_total_min=0.0,
        clip_total_max=2.0,
    ),
)
```

## Verbosity Scaling

The reward manager supports optional verbosity-based scaling that multiplies rewards by a factor
based on output length. This encourages concise responses by penalizing verbose outputs.

### Setup Requirements

Verbosity scaling requires token counting. You have two options:

1. **Provide a HuggingFace tokenizer** to `RewardManager(tokenizer=tokenizer)` - this is the
   recommended approach when you already have a tokenizer loaded for your model.

2. **Configure a tiktoken tokenizer** via `parsing.openai_tokenizer_model` - note that this
   requires a `ParsingConfig` to be defined as well.

### Modes

**Absolute mode**: Factor decays from `max_factor` toward `min_factor` based on absolute token
thresholds. Use `threshold_tokens` (no penalty below) and `end_tokens` (full penalty at).

```python
import pyine.organisms.models.rewards.core.configs as reward_configs

config = reward_configs.RewardManagerConfig(
    terms=[...],
    verbosity_scaling=reward_configs.VerbosityScalingConfig(
        enabled=True,
        mode="absolute",
        length_source="model_output",
        decay_type="linear",    # or "exponential"
        threshold_tokens=500,   # no penalty below this
        end_tokens=2000,        # full penalty at this (for linear)
        min_factor=0.1,         # minimum scaling factor
        max_factor=1.0,         # factor starts here, decays toward min_factor
    ),
)
```

**Relative mode**: Factor computed relative to other samples in the same "group" (prompt). Samples
at or below mean length get factor ~1.0; above mean get factor < 1.0.

```python
config = reward_configs.RewardManagerConfig(
    terms=[...],
    verbosity_scaling=reward_configs.VerbosityScalingConfig(
        enabled=True,
        mode="relative",
        length_source="model_output",
        temperature=1.0,  # higher = gentler slope
        min_factor=0.1,
        max_factor=1.0,
    ),
)
```

### Group Definition (GRPO)

In GRPO training, "group" = all generations for the same prompt. The reward manager groups
completions by `sample_data.identifier` before computing relative scaling, so samples sharing a
prompt are normalized together.

For relative mode with `compute()` (single sample), a warning is emitted and scaling is
skipped. Use `compute_batch(sample_ctxs)` instead (samples are automatically grouped by
`sample_data.identifier`).

### How It Works

The verbosity factor is applied as a post-aggregation multiplier:

```
final_reward = aggregated_reward * verbosity_factor
```

Where `verbosity_factor` is in `[min_factor, max_factor]` (typically 0.0 to 1.0).

### Metrics

When `emit_metrics=True` (default), verbosity scaling emits:

- `verbosity/factor`: the computed scaling factor;
- `verbosity/token_count`: token count considered for the sample;
- `verbosity/pre_scaling_reward`: reward before scaling;
- `verbosity/post_scaling_reward`: reward after scaling;
- In relative mode: `verbosity/group_mean`, `verbosity/group_std`, `verbosity/z_score`.

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
import pyine.organisms.models.rewards.core.types as reward_types

class MyTerm(reward_term.BaseRewardTerm):
    def __init__(self, config: MyTermConfig) -> None:
        self._config = config

    def __call__(self, sample_ctx: reward_types.SampleContext) -> reward_types.TermResult:
        # you can access the model's parsed output if available directly:
        has_answer = (
            sample_ctx.parsed is not None
            and sample_ctx.parsed.final_answer is not None
        )
        # compute reward based on your logic
        if has_answer:
            # ...
            value = 1.0
        else:
            value = self._config.penalty
        # return result with optional metrics
        return reward_types.TermResult(
            value=value,
            metrics={"has_answer": has_answer},
        )
```

### 3. Register the term with the factory

```python
import pyine.organisms.models.rewards.core.configs as reward_configs
import pyine.organisms.models.rewards.core.registry as reward_registry
import pyine.organisms.models.rewards.core.types as reward_types

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
import json
import re

import pyine.organisms.models.rewards.core.manager as reward_manager
import pyine.organisms.models.rewards.core.types as reward_types

class JsonOutputParser:
    """Parser that extracts a JSON object with an 'answer' field from model output."""

    def parse(
        self,
        prompt: str,
        model_output: str,
    ) -> reward_types.ParsedOutput:
        del prompt  # unused in this simple parser
        final_answer = None
        # look for a JSON object anywhere in the output (simple heuristic)
        json_match = re.search(r'\{[^{}]*"answer"[^{}]*\}', model_output)
        if json_match:
            try:
                parsed_json = json.loads(json_match.group())
                final_answer = parsed_json.get("answer")
            except json.JSONDecodeError:
                pass  # fall through to return None as final_answer
        return reward_types.ParsedOutput(
            raw=model_output,
            final_answer=final_answer,
            reasoning=None,
            fields={},
        )

# use custom parser with manager
manager = reward_manager.RewardManager(config, parser=JsonOutputParser())
```

## Integration with Training Loops

### Basic Integration

```python
import pyine.organisms.models.rewards.core.manager as reward_manager
import pyine.organisms.models.rewards.core.types as reward_types

# initialize manager once
manager = reward_manager.RewardManager(config)
# reset at start of each training run/epoch
manager.reset(reward_types.RunInitContext(datamodule=datamodule))

# in training loop:
for batch in dataloader:
    prompts, outputs, sample_data_list = generate_outputs(batch)
    rewards = []
    for prompt, output, sample_data in zip(prompts, outputs, sample_data_list):
        ctx = reward_types.SampleContext(
            prompt=prompt,
            model_output=output,
            sample_data=sample_data,
        )
        output = manager.compute(ctx)
        rewards.append(output.total)
    # use rewards for policy gradient update
    loss = compute_policy_loss(rewards, ...)

# at end of run:
manager.finalize_run()  # to make sure run-level stats are logged properly
```

### With Logging

```python
import pyine.organisms.models.rewards.core.logging as reward_logging
import pyine.organisms.models.rewards.core.manager as reward_manager

# create W&B logger
logger = reward_logging.make_wandb_reward_logger(
    wandb_run,
    config.logging,
    step=global_step,  # optional fixed W&B step to use for all logs (unless manually updated)
)

# create manager with logger
manager = reward_manager.RewardManager(config, logger=logger)
# ...
manager.set_step(global_step)  # if needed, for subsequent steps
output = manager.compute(ctx)  # rewards are automatically logged according to LoggingConfig
```

## Distributed Training

The rewards package supports distributed training with proper synchronization:

```python
import pyine.organisms.models.rewards.core.configs as reward_configs

config = reward_configs.RewardManagerConfig(
    terms=[...],
    logging=reward_configs.LoggingConfig(
        enabled=True,
        main_process_only=True,            # only log on rank 0
        gather_distributed_summaries=True, # gather stats from all ranks
        barrier_before_finalize=True,      # sync before finalize
    ),
)

# manager will automatically (all activated by default):
# - skip logging on non-main ranks (when main_process_only=True);
# - gather statistics from all ranks (when gather_distributed_summaries=True);
# - synchronize before finalize (when barrier_before_finalize=True).
```

## Testing

Use registry snapshots for test isolation:

```python
import pyine.organisms.models.rewards.core.registry as reward_registry
import pyine.organisms.models.rewards.terms

# ensure builtins are registered
pyine.organisms.models.rewards.terms.ensure_builtin_terms_registered()
# take a snapshot for isolated testing
snapshot = reward_registry.get_global_registry().snapshot()
# use snapshot with manager
manager = reward_manager.RewardManager(config, registry=snapshot)
```

Use `InMemoryRewardLogger` for testing logging:

```python
import pyine.organisms.models.rewards.core.logging as reward_logging

logger = reward_logging.InMemoryRewardLogger()
manager = reward_manager.RewardManager(config, logger=logger)
# after computing rewards...
assert len(logger.samples) == expected_count
assert logger.samples[0]["reward_total"] == expected_reward
```
