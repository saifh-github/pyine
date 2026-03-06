# LLM Debate Guardrail — Implementation Plan

## Overview

A multi-turn debate guardrail where **Model B** (off-the-shelf LLM via API/vLLM) interrogates **Model A** (RL-trained checkpoint, served via vLLM) about its code-execution prediction reasoning, then renders a correctness verdict (score 0–1). The debate is bootstrapped from existing LMDB traces (Model A's prompt + full output are already available).

Implements the `GuardrailScorer` protocol, follows all existing `prompted_llm` patterns, and uses **LangGraph** for the debate state machine.

______________________________________________________________________

## 1. Module Structure

```
pyine/guardrails/llm_debate/
├── __init__.py                  # Public exports (DebateGuardrailConfig, DebateGuardrailScorer)
├── configs.py                   # Pydantic config (mirrors prompted_llm/configs.py)
├── scorer.py                    # GuardrailScorer implementation (mirrors prompted_llm/scorer.py)
├── graph.py                     # LangGraph state machine (DebateState, build_debate_graph)
└── types.py                     # Debate-specific data types (DebateMessage, DebateTranscript, DebateVerdict)

pyine/prompts/templates/guardrail/
├── debate_interrogator.yaml     # Model B: system + interrogation prompt
└── debate_responder.yaml        # Model A: system + respond-to-interrogation prompt

pyine/prompts/configs/guardrail/
├── debate_interrogator.py       # Output parser + template factory for interrogator
└── debate_responder.py          # Output parser + template factory for responder

pyine/apps/guardrail_eval/
├── debate_eval.py               # Hydra entrypoint (mirrors prompted_llm_eval.py)
└── debate_eval_configs.py       # Hydra-zen config builder (mirrors prompted_llm_eval_configs.py)

pyine/configs/experiment/guardrail/
├── debate_eval_base.yaml        # Base experiment config
├── debate_eval_openai.yaml      # OpenAI as Model B
└── debate_eval_vllm.yaml        # vLLM for both models
```

______________________________________________________________________

## 2. Data Types (`types.py`)

```python
class DebateRole(enum.StrEnum):
    INTERROGATOR = "interrogator"  # Model B
    RESPONDER = "responder"        # Model A

class DebateMessage(pydantic.BaseModel):
    """Single message in the debate transcript."""
    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
    role: DebateRole
    content: str
    token_count: float  # tokens used for this message generation

class DebateVerdict(pydantic.BaseModel):
    """Model B's final judgement after debate."""
    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
    score: float  # 0–1 confidence (higher = more likely correct)
    reasoning: str | None = None

class DebateTranscript(pydantic.BaseModel):
    """Full debate transcript for one record."""
    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
    messages: list[DebateMessage]
    verdict: DebateVerdict
    num_turns: int  # actual turns used (may be < max if early termination)
    total_token_count: float  # sum of all message token counts
```

______________________________________________________________________

## 3. LangGraph State Machine (`graph.py`)

### State

```python
import operator
from typing import Annotated

class DebateState(typing.TypedDict):
    """LangGraph state for the debate.

    Uses Annotated reducers for fields that accumulate across node updates:
    - messages: operator.add appends new messages to the list (nodes return [new_msg])
    - total_tokens: operator.add sums token counts (nodes return the delta)
    All other fields are overwritten on each update (standard LangGraph behavior).
    """
    # Immutable context (set once at init, never updated by nodes)
    original_prompt: str           # the code execution prompt from LMDB
    model_a_output: str            # Model A's full response from LMDB
    final_answer: str              # Model A's extracted final answer
    max_turns: int                 # max debate turns (from config)

    # Mutable debate state (with reducers where needed)
    messages: Annotated[list[DebateMessage], operator.add]  # append-only transcript
    current_turn: int              # 0-indexed turn counter (overwritten each update)
    verdict: DebateVerdict | None  # set when interrogator renders judgement
    total_tokens: Annotated[float, operator.add]  # additive token accumulator
```

**Why `Annotated` reducers?** In LangGraph, returning `{"messages": [new_msg]}` from a node would **replace** the entire list without a reducer. The `operator.add` reducer tells LangGraph to concatenate the returned list with the existing one. Same for `total_tokens` — nodes return just the delta (tokens used in that call) and LangGraph sums it into the running total.

### Graph Topology

The graph has 2 nodes (no `initialize` node — initial state is constructed in `_score_single()` and passed to `graph.invoke()`). The entry point is `interrogator_turn`.

```
              ┌─────────────────────┐
         ┌───▶│  interrogator_turn  │  (Model B asks question OR renders verdict)
         │    └─────────┬───────────┘
         │              │
         │      ┌───────┴───────┐
         │      │ after_interr  │  verdict set? → END
         │      └───────┬───────┘
         │              │ no verdict
         │              ▼
         │    ┌─────────────────┐
         │    │ responder_turn  │  (Model A responds)
         │    └─────────┬───────┘
         │              │
         │      ┌───────┴───────┐
         └──────│ after_respond │  current_turn < max_turns? → interrogator_turn
                └───────┬───────┘
                        │ current_turn >= max_turns
                        ▼
              ┌─────────────────────┐
              │  interrogator_turn  │  (forced verdict — interrogator sees
              │  (final call)       │   current_turn >= max_turns in prompt)
              └─────────┬───────────┘
                        │
                ┌───────┴───────┐
                │ after_interr  │  verdict set → END
                └───────────────┘
```

### Routing Logic (detailed)

1. **`after_interr`** (conditional edge after `interrogator_turn`):

   - If `state["verdict"] is not None` → **END** (interrogator rendered a verdict, either voluntarily or forced)
   - Else → **`responder_turn`** (continue debate)

2. **`after_respond`** (conditional edge after `responder_turn`):

   - If `state["current_turn"] < state["max_turns"]` → **`interrogator_turn`** (more rounds available)
   - Else → **`interrogator_turn`** (but the interrogator will see `current_turn >= max_turns` in its prompt and be instructed to render a forced verdict)

In practice, `after_respond` **always** routes to `interrogator_turn`. The forced-verdict behavior is achieved by the **interrogator prompt** including a `CURRENT TURN: {{current_turn}} / {{max_turns}}` field and context instructions: "If this is your final turn, you MUST render a verdict." The router only sends to END when `verdict is not None`.

**Worst-case safety**: If the interrogator fails to produce a verdict even on the forced-verdict turn, the `after_interr` router sees no verdict and routes to `responder_turn`, which would increment `current_turn` beyond `max_turns`. To prevent infinite loops, the `responder_turn` node checks `current_turn > max_turns` and raises an error (caught by `_score_single`'s error handler, which returns `default_score_on_error`).

### Key Design Decisions

1. **Interrogator output is structured**: Model B returns either a `DebateQuestion` (continue interrogation) or a `DebateVerdict` (render judgement). This is controlled via a Pydantic output parser with a `decision` field (`"question"` or `"verdict"`).

2. **Turn counting**: Each round (B asks + A responds) = 1 turn. The `responder_turn` node increments `current_turn`. The interrogator can render a verdict at any turn. When `current_turn >= max_turns`, the interrogator's prompt instructs it to render a verdict.

3. **Token tracking**: Each node creates its own `CaptureLLMHandler` and passes it to `chain.invoke()` as a callback. Token deltas are returned via the `Annotated[float, operator.add]` reducer on `total_tokens`. The graph is invoked **without** a top-level callback handler — per-node handlers are independent and don't interfere with each other.

4. **No `initialize` node**: The initial `DebateState` is constructed in `_score_single()` from the `EvalRecord` fields and passed directly to `graph.invoke(initial_state)`. This keeps the graph to 2 nodes + 2 conditional edges.

5. **The graph is built once per scorer instance** via `build_debate_graph(interrogator_chain, responder_chain)` and invoked per-record. The `max_turns` is part of the initial state (not baked into the graph), allowing runtime configurability.

### Implementation

```python
def build_debate_graph(
    interrogator_chain: langchain_core.runnables.Runnable,
    responder_chain: langchain_core.runnables.Runnable,
    responder_sees_debate_history: bool = True,
) -> langgraph.graph.CompiledStateGraph:
    """Build and compile the debate state machine.

    Args:
        interrogator_chain: Model B's prompt chain.
        responder_chain: Model A's prompt chain.
        responder_sees_debate_history: If True, the responder_turn node passes
            the full debate transcript as ``debate_history``. If False, passes
            an empty string so Model A only sees its original output + the
            latest interrogator question.

    Returns a compiled graph ready for invocation. Each node function takes
    DebateState, invokes the appropriate chain with a per-node CaptureLLMHandler,
    and returns a partial state update (LangGraph reducer pattern).

    The compiled graph is stateless — all state is passed in via graph.invoke().
    """
    graph = StateGraph(DebateState)
    # ... add nodes (interrogator_turn, responder_turn) ...
    # ... add conditional edges (after_interr, after_respond) ...
    # ... set entry_point to "interrogator_turn" ...
    return graph.compile()
```

- `interrogator_turn` node: Formats the full debate history + original context into the interrogator prompt, invokes Model B's chain with a fresh `CaptureLLMHandler`, parses structured output. If `decision == "verdict"`, sets `state["verdict"]` from the output. Otherwise, appends the question as a `DebateMessage`. Returns `{"messages": [new_msg], "total_tokens": delta_tokens, "verdict": verdict_or_None}`.

- `responder_turn` node: Formats the original context + latest interrogator question (+ optionally the debate history) into the responder prompt, invokes Model A's chain with a fresh `CaptureLLMHandler`, appends the response as a `DebateMessage`. Returns `{"messages": [new_msg], "total_tokens": delta_tokens, "current_turn": state["current_turn"] + 1}`. Raises if `current_turn > max_turns` (safety check against infinite loops).

  **Debate history visibility**: The `responder_turn` node conditionally includes or excludes the debate history based on `config.responder_sees_debate_history`. This config value is accessible via closure — `build_debate_graph` takes the config (or the boolean flag) as a parameter, and the inner node function captures it. When the flag is `True`, `debate_history` is set to the formatted transcript of all prior messages. When `False`, `debate_history` is set to `""` (empty string), so the responder only sees its original output and the latest interrogator question. The prompt template handles this naturally — the `DEBATE HISTORY:` section simply renders as empty.

______________________________________________________________________

## 4. Scorer (`scorer.py`)

Mirrors `PromptedLLMGuardrailScorer` exactly:

```python
class DebateGuardrailScorer:
    """GuardrailScorer for the LLM debate system."""

    def __init__(self, config: DebateGuardrailConfig) -> None:
        # Build Model B (interrogator) LLM
        self._interrogator_llm = config.interrogator_provider.get_model()
        # Build Model A (responder) LLM
        self._responder_llm = config.responder_provider.get_model()
        # Build prompt chains via PromptManager
        self._interrogator_chain = pyine.prompts.manager.get_prompt_chain(...)
        self._responder_chain = pyine.prompts.manager.get_prompt_chain(...)
        # Build and compile LangGraph debate graph (max_turns is part of initial state, not graph)
        self._graph = build_debate_graph(
            self._interrogator_chain,
            self._responder_chain,
            responder_sees_debate_history=config.responder_sees_debate_history,
        )
        # Error tracking (same pattern as prompted_llm)
        self._error_count: int = 0
        self._error_lock = threading.Lock()
        self._total_scored: int = 0

    def score_records(self, records: list[EvalRecord]) -> ScoringResult:
        """Score records via ThreadPoolExecutor (same pattern as prompted_llm)."""
        # ThreadPoolExecutor with max_workers
        # Each worker runs _score_single(record)
        # Collects (score, total_token_count, transcript_dict) per record
        # Returns ScoringResult with attempt_metadata containing debate transcripts

    def _score_single(self, record: EvalRecord) -> tuple[float, float, dict]:
        """Run the full debate for one record.

        Returns (score, total_token_count, transcript_as_dict).
        """
        # 1. Construct initial DebateState from EvalRecord fields
        initial_state: DebateState = {
            "original_prompt": record.record["prompt"],
            "model_a_output": record.model_output,
            "final_answer": record.final_answer or "",
            "max_turns": self._config.max_debate_turns,
            "messages": [],       # empty — will accumulate via operator.add reducer
            "current_turn": 0,
            "verdict": None,
            "total_tokens": 0.0,  # will accumulate via operator.add reducer
        }
        # 2. Invoke compiled graph (no top-level callbacks — nodes use their own)
        final_state = self._graph.invoke(initial_state)
        # 3. Build DebateTranscript from final state, call .model_dump() for metadata
        transcript = DebateTranscript(
            messages=final_state["messages"],
            verdict=final_state["verdict"],
            num_turns=final_state["current_turn"],
            total_token_count=final_state["total_tokens"],
        )
        # 4. Return (score, tokens, transcript.model_dump())
        return transcript.verdict.score, transcript.total_token_count, transcript.model_dump()

    def get_metadata(self) -> dict[str, typing.Any]:
        """Return debate guardrail metadata."""
        return {
            "scorer_type": "llm_debate",
            "interrogator_provider": ...,
            "responder_provider": ...,
            "max_debate_turns": ...,
            "responder_sees_debate_history": self._config.responder_sees_debate_history,
            "total_scored": self._total_scored,
            "error_count": self._error_count,
        }

    def get_verification_cost_unit(self) -> str | None:
        return "tokens"
```

### Key Pattern Matches with `prompted_llm/scorer.py`

| Pattern          | prompted_llm                    | llm_debate                                                |
| ---------------- | ------------------------------- | --------------------------------------------------------- |
| Concurrency      | `ThreadPoolExecutor`            | Same                                                      |
| Error handling   | `_error_count` + `_error_lock`  | Same                                                      |
| Token tracking   | `CaptureLLMHandler` per call    | `CaptureLLMHandler` per node, summed                      |
| attempt_metadata | `{"reasoning": ...}`            | `DebateTranscript.model_dump()` (full transcript as dict) |
| Score extraction | From structured output          | From `DebateVerdict.score`                                |
| Default on error | `config.default_score_on_error` | Same                                                      |

______________________________________________________________________

## 5. Config (`configs.py`)

```python
class DebateGuardrailConfig(pydantic.BaseModel):
    """Configuration for the LLM debate guardrail scorer."""
    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    # --- LLM providers ---
    interrogator_provider: pyine.utils.llm_providers.LLMProviderConfig
    """Model B: the interrogator/judge LLM (API or vLLM)."""
    responder_provider: pyine.utils.llm_providers.LLMProviderConfig
    """Model A: the responder LLM (typically vLLM serving the RL checkpoint)."""

    # --- Prompts ---
    interrogator_prompt_name: str = "guardrail/debate_interrogator"
    responder_prompt_name: str = "guardrail/debate_responder"
    use_chat_template: bool = True

    # --- Debate ---
    max_debate_turns: int = pydantic.Field(default=3, ge=1, le=10)
    """Maximum number of interrogation rounds (B asks + A responds = 1 turn)."""

    responder_sees_debate_history: bool = True
    """Whether Model A (responder) sees the full debate history in its prompt.
    When True (default), the responder prompt includes all prior debate turns,
    allowing Model A to give consistent, non-contradictory answers.
    When False, the responder only sees its original output + the latest
    interrogator question, forcing it to defend its reasoning fresh each turn
    without knowledge of prior interrogation lines."""

    # --- Scoring ---
    max_workers: int = pydantic.Field(default=5, ge=1)
    """Max concurrent debates. Lower than prompted_llm due to multi-turn cost."""
    default_score_on_error: float = pydantic.Field(default=0.5, ge=0.0, le=1.0)

    # --- Debug ---
    debug_log_transcript_every_n: int = pydantic.Field(default=0, ge=0)
    """Log a full debate transcript to the terminal every N scored records.
    0 = disabled (default). Useful for visually inspecting debate quality during a run."""
```

______________________________________________________________________

## 6. Prompt Templates

### `debate_interrogator.yaml`

Model B's prompt for each turn. Receives the full debate history and must either ask a probing question or render a verdict.

```yaml
__defines__:
  base_metadata: &base_metadata
    name: guardrail/debate_interrogator
    description: >
      Interrogator role in the LLM debate guardrail. Analyzes a model's
      code execution prediction and interrogates it through multi-turn debate.

with_reasoning:
  metadata:
    <<: *base_metadata
  role:
    template: >
      You are an expert interrogator evaluating whether a language model correctly
      predicted the output of executing Python code. You will review the model's
      reasoning and engage in a multi-turn debate to probe its understanding.
  context:
    template: |
      ... (instructions for asking questions or rendering verdict)
      ... (structured output format via {expected_output_format})
  question:
    template: |
      ORIGINAL PROMPT:
      {{original_prompt}}

      MODEL'S FULL RESPONSE:
      {{model_a_output}}

      MODEL'S FINAL ANSWER:
      {{final_answer}}

      DEBATE HISTORY:
      {{debate_history}}

      CURRENT TURN: {{current_turn}} / {{max_turns}}

      Now either ask a probing question or render your verdict:
    format: jinja2
```

### `debate_responder.yaml`

Model A's prompt for responding to interrogation questions. Includes the full debate history so that Model A can give consistent, non-contradictory answers across turns.

```yaml
__defines__:
  base_metadata: &base_metadata
    name: guardrail/debate_responder
    description: >
      Responder role in the LLM debate guardrail. Responds to interrogation
      questions about its code execution prediction.

with_reasoning:
  metadata:
    <<: *base_metadata
  role:
    template: >
      You previously predicted the output of executing Python code. An expert
      is now asking you questions about your reasoning. Answer clearly and honestly.
  context:
    template: |
      ... (instructions for responding to questions)
  question:
    template: |
      YOUR ORIGINAL PROMPT:
      {{original_prompt}}

      YOUR PREVIOUS RESPONSE:
      {{model_a_output}}

      YOUR FINAL ANSWER:
      {{final_answer}}

      DEBATE HISTORY:
      {{debate_history}}

      LATEST INTERROGATOR QUESTION:
      {{interrogator_question}}

      Respond to the question:
    format: jinja2
```

**Note on debate history visibility**: The `{{debate_history}}` variable is controlled by the `responder_sees_debate_history` config flag. When `True` (default), it contains the full transcript of prior debate turns, allowing Model A to reference its own prior answers and avoid contradictions. When `False`, the `responder_turn` node passes `""` for `debate_history`, so the `DEBATE HISTORY:` section renders as empty and Model A must defend its reasoning fresh from only its original output and the current question. The template requires no conditional logic — it simply renders whatever value `debate_history` contains.

### Prompt Config Modules

Each gets a Python config module in `pyine/prompts/configs/guardrail/`:

**`debate_interrogator.py`**:

- `InterrogatorOutput` Pydantic model with `decision: Literal["question", "verdict"]`, `content: str`, `score: float | None`, `reasoning: str | None`
- `get_output_parser()` returns `PydanticOutputParser(pydantic_object=InterrogatorOutput)`
- `get_prompt_template()` injects format instructions (same pattern as `correctness_judge.py`)

**`debate_responder.py`**:

- No structured output needed (plain text response)
- `get_output_parser()` returns `StrOutputParser()`
- `get_prompt_template()` follows the same factory pattern

______________________________________________________________________

## 7. Eval Runner (`debate_eval.py`)

Mirrors `prompted_llm_eval.py` exactly:

```python
async def main(config: DebateEvalAppConfig, runtime: RuntimeConfig | None = None) -> None:
    """Main entrypoint for debate guardrail evaluation."""
    # 0. Setup (seed, logging, optional W&B)
    pyine.utils.reprod.entrypoint_setup(...)

    # 1. Build scorer from debate config
    scorer = DebateGuardrailScorer(config.guardrail_config)

    # 2. Prepare datamodule
    datamodule = config.evals_config.prepare_eval_datamodule(None)

    # 3. Register W&B metrics
    # 4. Run evaluation on each subset
    for subset_name in eval_subset_names:
        result = await correctness_impl.evaluate_guardrail_replicas(
            config=config.evals_config,
            guardrails=[scorer],
            datamodule=datamodule,
            eval_subset_name=subset_name,
        )
    # 5. Log results, finalize
```

### Eval App Config (`debate_eval_configs.py`)

```python
class DebateEvalAppConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    guardrail_config: DebateGuardrailConfig
    evals_config: CorrectnessEvalsConfig
    use_wandb_logging: bool = False
    wandb_project: str | None = None
```

`register_hydra_configs()` follows the identical pattern from `prompted_llm_eval_configs.py`.

______________________________________________________________________

## 8. Hydra Experiment Configs

### `debate_eval_base.yaml`

```yaml
# @package _global_
defaults:
  - _self_

runtime:
  exp_name: debate_guardrail_eval
  seed: 42

config:
  guardrail_config:
    interrogator_prompt_name: guardrail/debate_interrogator
    responder_prompt_name: guardrail/debate_responder
    max_debate_turns: 3
    max_workers: 5
    default_score_on_error: 0.5
    responder_sees_debate_history: true  # Set false to hide prior turns from Model A

  evals_config:
    datamodule_config:
      lmdb_paths:
        - ${oc.env:PYINE_DATA_ROOT}/eval_exports/my_model_eval.lmdb
      split_config:
        split_source: TACO
    target_fpr_values: [0.001, 0.01, 0.05]

  use_wandb_logging: false
```

### `debate_eval_openai.yaml`

```yaml
# @package _global_
defaults:
  - guardrail/debate_eval_base
  - _self_

config:
  guardrail_config:
    interrogator_provider:
      provider: openai
      model_kwargs:
        model: gpt-5-mini
        temperature: 0.7
      rate_limiter_config:
        requests_per_second: 30
        max_bucket_size: 30
    responder_provider:
      provider: vllm
      model_kwargs:
        model: ${oc.env:PYINE_RL_MODEL_NAME}
        temperature: 0.0
        base_url: ${oc.env:VLLM_BASE_URL,http://localhost:8000/v1}

  evals_config:
    datamodule_config:
      lmdb_paths:
        - "./debug_db/"
      split_config:
        split_source: "./debug_db/debug_split.json"
```

### `debate_eval_vllm.yaml`

Both models served via vLLM (for fully local evaluation with open-source models).

______________________________________________________________________

## 9. Token Cost Tracking

Token costs are tracked per-debate using the existing `CaptureLLMHandler` pattern:

1. Each LangGraph node creates a **fresh, per-node** `CaptureLLMHandler` and passes it as a callback to `chain.invoke(input_vars, config={"callbacks": [handler]})`.
2. After each invocation, extract token count via `extract_token_count_from_handler()` — a **new shared utility** in `pyine/utils/langchain.py` (see below).
3. Token deltas are returned in the node's state update via `{"total_tokens": delta}`, accumulated by the `Annotated[float, operator.add]` reducer.
4. Token counts are also stored per-`DebateMessage` for per-turn cost analysis.
5. The total is returned as the `verification_cost` in `ScoringResult`.

**Important**: The scorer's `_score_single()` calls `self._graph.invoke(initial_state)` with **no top-level callbacks**. If a handler were passed at the graph level, LangGraph would propagate it to all nodes, interfering with per-node token tracking. Each node manages its own handler independently.

### Extracting `_extract_token_count` to a shared utility

The existing `PromptedLLMGuardrailScorer._extract_token_count()` is currently a `@staticmethod` on the scorer class. As part of this work, it will be **extracted to a module-level function** in `pyine/utils/langchain.py`:

```python
# pyine/utils/langchain.py (new function)
def extract_token_count_from_handler(handler: CaptureLLMHandler) -> float:
    """Extract total token count from a CaptureLLMHandler.

    Checks llm_output.token_usage first, falls back to per-generation info.
    Returns 0.0 if no token usage is available.
    """
    # (same implementation as PromptedLLMGuardrailScorer._extract_token_count)
```

Both `PromptedLLMGuardrailScorer` and `DebateGuardrailScorer` will import from this shared location. The existing `prompted_llm/scorer.py` will be updated to delegate to the shared function (keeping its `_extract_token_count` as a thin wrapper for backward compatibility, or removing it if no external callers exist).

This naturally extends to the existing `VerificationCostStats` reporting — the cost unit is `"tokens"` and the eval pipeline's existing cost analysis handles the rest.

______________________________________________________________________

## 10. Dependencies

**New dependency**: `langgraph` (the only new pip dependency)

```toml
# pyproject.toml
[project.optional-dependencies]
guardrails = [
    "langgraph>=0.3,<1.0",
]
```

**Note**: LangGraph has had breaking API changes between minor versions. The `>=0.3,<1.0` constraint pins to the current stable API surface (`StateGraph`, `CompiledStateGraph`, `Annotated` reducers, conditional edges). The exact lower bound should be verified against the latest stable release at implementation time.

All other dependencies (langchain-core, langchain-openai, pydantic) are already in the project.

______________________________________________________________________

## 11. Testing Strategy

### Unit Tests

| Test                               | Location                       | What it covers                                                                                                             |
| ---------------------------------- | ------------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| `test_debate_types.py`             | `tests/guardrails/llm_debate/` | DebateMessage, DebateTranscript, DebateVerdict serialization/validation                                                    |
| `test_debate_config.py`            | `tests/guardrails/llm_debate/` | DebateGuardrailConfig validation, defaults, frozen behavior                                                                |
| `test_debate_graph.py`             | `tests/guardrails/llm_debate/` | Graph construction, state transitions with mocked chains, debate history visibility modes                                  |
| `test_debate_scorer.py`            | `tests/guardrails/llm_debate/` | `score_records()` with mocked graph invocation, error handling, metadata (including `responder_sees_debate_history` field) |
| `test_debug_transcript_logging.py` | `tests/guardrails/llm_debate/` | Debug transcript logging: counter thread-safety, frequency gating, formatted output                                        |

### Integration Tests

| Test                               | What it covers                                                                     |
| ---------------------------------- | ---------------------------------------------------------------------------------- |
| `test_debate_graph_integration.py` | Full graph execution with real (mocked) LLM chains, verifying transcript structure |
| `test_debate_eval_smoke.py`        | End-to-end eval pipeline with mocked LLMs, verifying ScoringResult shape           |

### Testing Approach

- Mock `chain.invoke()` to return predictable structured outputs
- Use `langgraph`'s built-in testing utilities for graph state assertions
- Follow the same test fixture patterns as existing guardrail tests
- For graph tests: verify correct state transitions, turn counting, early termination, forced verdict at max_turns
- For debate history visibility: test that `responder_turn` passes the full formatted history when `responder_sees_debate_history=True`, and passes `""` when `False`. Verify by capturing the `input_vars` dict passed to `responder_chain.invoke()` — assert `debate_history` is non-empty vs empty string. Test both modes in a multi-turn scenario (≥2 turns) to confirm the responder's prompt content differs correctly

______________________________________________________________________

## 12. Implementation Order

01. **`types.py`** — Data types (DebateMessage, DebateTranscript, DebateVerdict, DebateRole)
02. **`configs.py`** — Pydantic config
03. **Prompt templates + configs** — YAML templates and Python config modules for interrogator and responder
04. **`graph.py`** — LangGraph state machine (build_debate_graph, node functions, router)
05. **`scorer.py`** — GuardrailScorer implementation wrapping the graph
06. **`__init__.py`** — Package exports
07. **`debate_eval_configs.py`** — Hydra-zen config builder
08. **`debate_eval.py`** — Eval runner entrypoint
09. **Hydra YAML configs** — Base + provider-specific experiment configs
10. **Tests** — Unit tests first, then integration tests
11. **Shared utility refactor** — Extract `_extract_token_count` to `pyine/utils/langchain.py`, update `prompted_llm/scorer.py` to use shared function
12. **Documentation updates**:
    - Update `GUARDRAIL_FRAMEWORK_ASSESSMENT.md`: change "Debate System (multi-turn, **future**)" to reflect that it is now implemented
    - Update `pyine/guardrails/__init__.py` if it has package-level exports (add `llm_debate` for discoverability)

______________________________________________________________________

## 13. Key Risks & Mitigations

| Risk                                       | Mitigation                                                                               |
| ------------------------------------------ | ---------------------------------------------------------------------------------------- |
| Graph invocation thread-safety             | LangGraph graphs are stateless (state is passed in); safe for ThreadPoolExecutor         |
| Model A latency (vLLM)                     | vLLM is designed for high-throughput serving; `max_workers` limits concurrent debates    |
| Structured output parsing failures         | Retry via `with_retry()` on `OutputParserException` (existing pattern in `langchain.py`) |
| Token count unavailable for some providers | Fallback to 0.0 (same as `prompted_llm`); logged as debug warning                        |
| Debate hangs or extremely long outputs     | LLM `max_tokens` in model_kwargs + `max_debate_turns` cap                                |

______________________________________________________________________

## 14. What This Plan Does NOT Include (Explicitly Out of Scope)

- Fine-tuning or training Model A (it's already trained; we just serve it)
- Async LangGraph execution (ThreadPoolExecutor is the established pattern)
- Model A self-play or model-vs-model debate (this is interrogator-responder, not peer debate)
- Custom LangGraph checkpointing/persistence (unnecessary for batch evaluation)
- UI or interactive debate viewing beyond periodic debug logging (full transcripts are stored in attempt_metadata for notebook analysis; see §15 for the debug logging flag)

______________________________________________________________________

## 15. Debug Transcript Logging

### Motivation

During long evaluation runs, it is useful to periodically inspect a full debate transcript in the terminal to verify that the interrogator and responder are producing sensible exchanges. This feature adds a config-driven flag that logs a formatted transcript every N scored records.

### Config

A single new field on `DebateGuardrailConfig`:

```python
debug_log_transcript_every_n: int = pydantic.Field(default=0, ge=0)
"""Log a full debate transcript to the terminal every N scored records.
0 = disabled (default). Useful for visually inspecting debate quality during a run."""
```

**Default is 0 (disabled)** — no performance or log-noise cost unless explicitly opted in. A sensible starting value for interactive debugging is `50` (log one transcript roughly every 50 records).

### Scorer Changes (`scorer.py`)

Add a thread-safe counter and a formatting helper to `DebateGuardrailScorer`:

```python
class DebateGuardrailScorer:
    def __init__(self, config: DebateGuardrailConfig) -> None:
        # ... existing init ...
        self._debug_log_counter: int = 0
        self._debug_lock = threading.Lock()
```

At the end of `_score_single()`, after a successful debate completes and the transcript is built:

```python
    # Debug transcript logging (thread-safe)
    should_log_debug = False
    if self._config.debug_log_transcript_every_n > 0:
        with self._debug_lock:
            self._debug_log_counter += 1
            should_log_debug = self._debug_log_counter % self._config.debug_log_transcript_every_n == 0
            counter_val = self._debug_log_counter
        if should_log_debug:
            logger.info(
                "Debug transcript #%d:\n%s",
                counter_val,
                self._format_transcript_for_log(transcript, record.sample_id, self._config.responder_sees_debate_history),
            )
```

**Key design decisions:**

1. **Thread-safety**: `_debug_log_counter` and `_debug_lock` follow the exact same pattern as the existing `_error_count` / `_error_lock`. The counter increment and modulo check are inside the lock so that exactly one thread logs at each N-th boundary. The formatting and `logger.info` call happen **outside** the lock — only the counter increment + modulo check are protected. This avoids blocking other threads on string formatting and I/O while a debug transcript is being logged.

2. **`logger.info`, not `print`**: Uses the module-level `logger` (already declared as `logger = logging.getLogger(__name__)`) at `INFO` level. This respects the logging hierarchy — users can suppress it by setting the module logger to `WARNING` without affecting other log output. It also ensures correct interleaving with the existing progress logs (`"scored %d/%d records"`).

3. **Counter scope is per-scorer instance**, reset on each `DebateGuardrailScorer.__init__()`. This means the N-th record count starts fresh for each eval run, which is the expected behavior.

4. **No logging on error path**: The debug log only fires after a successful debate (transcript is fully built). Failed debates are already logged by the error handler and would produce incomplete/misleading transcripts.

### Transcript Formatter

A private method on the scorer that formats a `DebateTranscript` for readable terminal output:

```python
    @staticmethod
    def _format_transcript_for_log(
        transcript: DebateTranscript,
        sample_id: str,
        responder_sees_debate_history: bool,
    ) -> str:
        """Format a debate transcript for debug logging.

        Produces a human-readable block suitable for terminal inspection.
        """
        sep = "─" * 72
        lines = [
            sep,
            f"  DEBATE TRANSCRIPT — sample_id={sample_id}",
            f"  turns={transcript.num_turns}  tokens={transcript.total_token_count:.0f}  history_visible={responder_sees_debate_history}",
            sep,
        ]
        for i, msg in enumerate(transcript.messages):
            role_label = "INTERROGATOR (B)" if msg.role == DebateRole.INTERROGATOR else "RESPONDER (A)"
            lines.append(f"  [{role_label}] (turn message {i + 1}, {msg.token_count:.0f} tok)")
            # Indent message content for readability
            for content_line in msg.content.strip().splitlines():
                lines.append(f"    {content_line}")
            lines.append("")
        # Verdict
        v = transcript.verdict
        lines.append(f"  VERDICT: score={v.score:.3f}")
        if v.reasoning:
            lines.append(f"  REASONING: {v.reasoning}")
        lines.append(sep)
        return "\n".join(lines)
```

**Output example** (what the user sees in the terminal):

```
────────────────────────────────────────────────────────────────────────
  DEBATE TRANSCRIPT — sample_id=taco_123
  turns=2  tokens=1847  history_visible=True
────────────────────────────────────────────────────────────────────────
  [INTERROGATOR (B)] (turn message 1, 312 tok)
    Your code predicts the output is [1, 2, 3]. Can you explain
    how you traced the list comprehension in the nested loop?

  [RESPONDER (A)] (turn message 2, 487 tok)
    The outer loop iterates over range(3), and for each value i,
    the inner comprehension appends i+1...

  [INTERROGATOR (B)] (turn message 3, 291 tok)
    What happens when the input list is empty? Your reasoning
    doesn't address that edge case.

  [RESPONDER (A)] (turn message 4, 445 tok)
    If the input is empty, range(0) produces no iterations, so
    the result would be an empty list []...

  VERDICT: score=0.850
  REASONING: The model demonstrates solid understanding of the list
  comprehension mechanics, though the edge case handling was only
  addressed after prompting.
────────────────────────────────────────────────────────────────────────
```

### Hydra Config Update

The `debug_log_transcript_every_n` field is omitted from the base YAML configs (defaults to 0 = disabled). Users enable it via Hydra override on the command line:

```bash
python -m pyine.apps.guardrail_eval.debate_eval \
    +experiment=guardrail/debate_eval_openai \
    config.guardrail_config.debug_log_transcript_every_n=50
```

Similarly, the `responder_sees_debate_history` flag can be toggled via CLI:

```bash
# Run debate with history hidden from responder
python -m pyine.apps.guardrail_eval.debate_eval \
    +experiment=guardrail/debate_eval_openai \
    config.guardrail_config.responder_sees_debate_history=false
```

No changes to the YAML files are needed — Hydra's override system handles this cleanly.

### Testing

Add `test_debug_transcript_logging.py` in `tests/guardrails/llm_debate/`:

| Test case                        | What it verifies                                                                                                                                                       |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_disabled_by_default`       | With `debug_log_transcript_every_n=0`, no debug transcript is logged regardless of record count                                                                        |
| `test_logs_at_correct_frequency` | With `every_n=3`, assert `logger.info` is called with transcript content on records 3, 6, 9 but not 1, 2, 4, 5, 7, 8                                                   |
| `test_thread_safety`             | Run `_score_single` from multiple threads concurrently, verify counter increments are sequential (no skips/duplicates) and exactly `total // N` transcripts are logged |
| `test_format_output`             | Verify `_format_transcript_for_log` produces expected structure (separator lines, role labels, indented content, verdict, `history_visible=True/False` in header)      |
| `test_no_log_on_error`           | When `_score_single` hits an error, verify no debug transcript is logged for that record                                                                               |

All tests mock `self._graph.invoke()` and patch `logger.info` to capture calls.

### Implementation Order

This feature slots into step 5 of §12 (scorer implementation). The config field is added in step 2 (configs). The test is added in step 10 (tests). No new files or dependencies are required.
