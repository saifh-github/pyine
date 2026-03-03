# `rewards` Package

Composable reward term management for RL experiments.

## Architecture

```
RewardManager
    │
    ├── OutputParser (optional)              # parses model output once per sample
    │
    ├── RewardTerm[]                         # enabled terms from config
    │   ├── parseable_answer                 # format term
    │   ├── text_length                      # format term
    │   ├── traced_reasoning                 # format term
    │   ├── hard_match                       # code_exec term
    │   ├── soft_match                       # code_exec term
    │   ├── llm_grader                       # code_exec term
    │   └── (custom terms)
    │
    ├── WeightedSumAggregator                # combines term values
    ├── CorrectnessClassifierScaler (opt.)   # classifier-based reward scaling
    ├── VerbosityScaler (optional)           # length-based reward scaling
    │
    └── RewardLogger (optional)              # logs metrics to W&B/memory
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
            require_parsed=True,  # fail fast if SampleContext.parsed is missing
            params={"reward_if_present": 1.0, "reward_if_missing": 0.0},
        ),
    ],
    logging=reward_configs.LoggingConfig(enabled=False),  # optional; disable by default for quick start
    parsing=reward_configs.ParsingConfig(
        final_tag="answer",  # parseable_answer term will inherit this tag
        fallback_policy="none",
    ),
)

# create manager
manager = reward_manager.RewardManager(config)

# compute reward for a sample; note: sample_data required (should come from datamodule)
ctx = manager.build_sample_context(
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

### `traced_reasoning`

Rewards structured reasoning traces referencing source code line numbers. The model produces
a JSONL `<steps>` block where each line is `{"step": N, "line": L, "text": "..."}`, and this
term computes three sub-rewards:

- **A) Format presence** (binary): is a valid `<steps>` block with enough parsed steps present?
- **B) Structural validity** (proportional): how many steps reference valid, executable code
  lines? Scales with valid step count using a configurable reward curve (diminishing by default).
  Sequence-level penalties apply for non-monotonic or non-contiguous step numbering.
- **C) Code grounding** (proportional): how many valid steps share non-numeric identifier tokens
  with the actual source line they reference? Controlled by `grounding_curve` (linear by default).

An optional correctness gating multiplier scales down the reward when the final answer is
objectively incorrect. The penalty is continuous: `incorrectness_penalty=1.0` zeros out the
reward entirely, `0.5` halves it, `0.0` disables gating.

**Requirements:** `add_line_numbers: true` in the datamodule config, and prompt version
`rl_stepped_reasoning` (or equivalent) that instructs the model to produce `<steps>` + `<final>`.

**Executable line detection:** uses `ast.parse` to identify executable statements, excluding
blank lines, comments, decorators, and docstrings (including those with inline code examples).
Raises `SyntaxError` on malformed code.

**Gating semantics:** gating always uses default compare options (canonical soft-match).
This is independent of any custom `compare_options` on a sibling `soft_match` term. Flip-aware:
for bugged/keyword samples, `is_objectively_correct = is_semantic_match XOR should_flip`.

**Verbosity scaling interaction:** when `reasoning_from_outside_final: true` is set, the
`<steps>` content contributes to the length penalty via `parsed_reasoning`. This is intentional
and encourages concise reasoning traces.

```yaml
- name: traced_reasoning
  type: traced_reasoning
  weight: 1.0
  enabled: true
  require_parsed: false
  params:
    # tag and block selection
    steps_tag: steps              # default
    multi_block_policy: last      # first | last | error

    # A) format presence (binary)
    format_presence_reward: 0.05
    require_min_parsed_steps: 1

    # B) structural validity (proportional, capped)
    structural_validity_weight: 0.05
    max_rewarded_steps: 20
    validity_curve: diminishing    # linear | diminishing
    diminishing_decay: 0.75       # per-step decay (used by diminishing curves)
    check_line_in_range: true
    check_executable_line: true
    check_monotonicity: true
    monotonicity_penalty_factor: 0.5  # 0.0 = full penalty, 1.0 = no penalty
    check_contiguity: true
    contiguity_penalty_factor: 0.5

    # C) code grounding (proportional, linear by default)
    grounding_weight: 0.05
    grounding_curve: linear       # linear | diminishing
    min_token_overlap: 1

    # gating (1.0 = zero out on incorrect, 0.5 = halve, 0.0 = no gating)
    incorrectness_penalty: 0.5
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
            require_parsed=True,
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
            params={
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
    logging=reward_configs.LoggingConfig(enabled=False),
    parsing=reward_configs.ParsingConfig(final_tag="final"),
    aggregation=reward_configs.AggregationConfig(
        clip_total_min=0.0,
        clip_total_max=2.0,
    ),
)
```

## Post-Aggregation Reward Scaling

After term aggregation, the reward manager applies up to two optional multiplicative scalers
in a fixed order:

```
aggregated_reward  -->  classifier scaling  -->  verbosity scaling  -->  final reward
```

Per-term breakdowns (`weighted_terms`) are **not** scaled; they are preserved as-is so you can
compare `reward/total` (post-scaling) against `reward/terms/*` (pre-scaling) in logged metrics.

### Correctness Classifier Scaling

Multiplies the aggregated reward by a pretrained classifier's predicted probability that the
model output is correct. The classifier is loaded lazily on first use (or eagerly via `reset()`).
It constructs a two-message conversation `[user, assistant]` from the prompt and model output,
formats it using the same pipeline as classifier training, and runs inference to get a correctness
probability. The scaling factor is `clamp(classifier_prob, min_factor, max_factor)`.

`correctness_classifier/pre_scaling_reward` always contains the raw aggregated reward from terms.

```python
import pyine.organisms.models.rewards.core.configs as reward_configs

config = reward_configs.RewardManagerConfig(
    terms=[...],
    logging=reward_configs.LoggingConfig(enabled=False),
    correctness_classifier_scaling=reward_configs.CorrectnessClassifierScalingConfig(
        enabled=True,
        checkpoint_path="/path/to/classifier/checkpoint",  # model + tokenizer directory
        temperature=1.0,          # logit temperature (>1 softer, <1 sharper)
        min_factor=0.0,           # clamp scaling factor from below
        max_factor=1.0,           # clamp scaling factor from above
        positive_label="correct", # label name resolved from model.config.label2id
        skip_negative_rewards=True,  # pass negative rewards through unscaled
        only_for_keyword_samples=False,  # restrict to keyword samples only
        neutral_factor=1.0,       # factor used when sample is skipped
    ),
)
```

When `emit_metrics=True` (default), classifier scaling emits:

- `correctness_classifier/factor`: the computed scaling factor;
- `correctness_classifier/pre_scaling_reward`: reward before scaling;
- `correctness_classifier/classifier_score`: raw classifier probability;
- `correctness_classifier/was_truncated`: 0/1 flag for truncation to `max_seq_length`;
- `correctness_classifier/temperature`: effective logit temperature used before softmax;
- `correctness_classifier/skipped_negative`: 1 if the sample was skipped due to negative reward.

**Important:** with `strict_single_turn=True` (default), this scaler raises if `sample_ctx.prompt`
looks like serialized JSON messages (e.g., `[{\"role\": ...}]`) instead of a flat prompt string.
Set `strict_single_turn=False` to warn once and continue.

See `CorrectnessClassifierScalingConfig` in [`core/configs.py`](core/configs.py) for full parameter
documentation. For training classifier checkpoints used by this scaler, see
[`llm_classifier_trainer.py`](../../../apps/trainers/llm_classifier_trainer.py) and
[`LLM_CLASSIFIER_TRAINING_GUIDE.md`](../../../apps/trainers/LLM_CLASSIFIER_TRAINING_GUIDE.md).

### Verbosity Scaling

Multiplies rewards by a factor based on output length, encouraging concise responses by
penalizing verbose outputs.

**Setup:** verbosity scaling requires token counting. Either provide a HuggingFace tokenizer
to `RewardManager(tokenizer=tokenizer)`, or configure a tiktoken tokenizer via
`parsing.openai_tokenizer_model`.

**Absolute mode**: Factor decays from `max_factor` toward `min_factor` based on absolute token
thresholds. Use `threshold_tokens` (no penalty below) and `end_tokens` (full penalty at).

```python
import pyine.organisms.models.rewards.core.configs as reward_configs

config = reward_configs.RewardManagerConfig(
    terms=[...],
    logging=reward_configs.LoggingConfig(enabled=False),
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
    logging=reward_configs.LoggingConfig(enabled=False),
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

**Group definition (GRPO):** "group" = all generations for the same prompt. The reward manager
groups completions by `sample_data.identifier` before computing relative scaling, so samples
sharing a prompt are normalized together. For relative mode with `compute()` (single sample), a
warning is emitted and scaling is skipped; use `compute_batch(sample_ctxs)` instead.

`verbosity/pre_scaling_reward` contains the post-classifier-scaled value (if classifier scaling
is enabled), or the raw aggregated reward otherwise.

When `emit_metrics=True` (default), verbosity scaling emits:

- `verbosity/factor`: the computed scaling factor;
- `verbosity/token_count`: token count considered for the sample;
- `verbosity/pre_scaling_reward`: reward before verbosity scaling;
- `verbosity/skipped_negative`: 1 if scaling was skipped for a negative reward;
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
import pyine.utils.parsing

def _factory(
    spec: reward_configs.RewardTermSpec,
    *,
    parser: pyine.utils.parsing.OutputParser | None,
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
import pyine.utils.parsing

class JsonOutputParser:
    """Parser that extracts a JSON object with an 'answer' field from model output."""

    def parse(
        self,
        prompt: str,
        model_output: str,
    ) -> pyine.utils.parsing.ParsedOutput:
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
        return pyine.utils.parsing.ParsedOutput(
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

### Metric Indexing in WandB

When using WandBRewardLogger, metrics are indexed to different x-axes depending on their type:

**Per-generation metrics** (indexed to `{prefix}/generation_count`): These metrics are logged for
individual generations (completions) and use generation_count as the x-axis. This ensures each
logged generation has a unique x-coordinate, avoiding WandB aggregation issues when multiple
generations are logged within the same trainer step (e.g. with gradient accumulation or multiple
generations per prompt in GRPO).

- `{prefix}/reward/total`: total reward for the generation;
- `{prefix}/reward/terms/*`: per-term weighted reward values;
- `{prefix}/reward/metrics/*`: term-emitted metrics (containing other useful information);
- `{prefix}/reward/raw_terms/*`: pre-clipping, pre-weighting reward term values;
- `{prefix}/categories/*`: category-wise metrics (if a category extractor is configured).

**Batch-level metrics** (indexed to `{prefix}/batch_count`): These metrics are logged once per
compute_batch call (when enabled) and use batch_count as the x-axis. This ensures each logged batch
has a unique x-coordinate, avoiding WandB aggregation issues when multiple batches are processed
within the same trainer step (e.g. gradient accumulation).

Note: Batch-level metrics are only emitted when `LoggingConfig.log_batch_stats=True`.

- `{prefix}/reward/batch/mean`: mean reward across the batch;
- `{prefix}/reward/batch/std`: standard deviation of rewards in the batch.

**Run-level summaries** (indexed to `step_metric_key`, default `train/global_step`): These metrics
are logged when flush_stats() is called (e.g. at phase transitions).

- `{prefix}/reward/run/total/{mean,std,min,max,count}`: accumulated total reward statistics;
- `{prefix}/reward/run/terms/{term}/{mean,std,min,max,count}`: per-term reward statistics;
- `{prefix}/reward/run/categories/{category}/{mean,std,min,max,count}`: per-category reward statistics;
- `{prefix}/parsing/*`: aggregated parsing statistics (lengths, missing ratios);
- `{prefix}/failures/failure_ratio`: ratio of failed generations in the phase;
- `{prefix}/failures/failure_count`: count of failed generations in the phase.
- `{prefix}/difficulty/run/*`: aggregated difficulty diagnostics (when enabled).

Where `{prefix}` is typically "train" or "eval" depending on the training phase.

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

### All-Rank Disk Export

To capture generated completions from **all** distributed ranks (for later SFT re-import),
use `GenerationExportConfig.export_all_ranks=True`:

```python
import pyine.organisms.models.rewards.core.configs as reward_configs

export_config = reward_configs.GenerationExportConfig(
    output_path=pathlib.Path("/path/to/export"),
    export_all_ranks=True,
)
```

Each rank writes to a rank-specific subdirectory: `output_path/rank_0/`, `output_path/rank_1/`,
etc. This avoids write contention across ranks.

Note: WandB logging remains rank-0-only; only generation export fans out to all ranks.

## Testing

Use registry snapshots for test isolation:

```python
import pyine.organisms.models.rewards.core.manager as reward_manager
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
