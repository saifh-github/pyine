"""Code execution-related reward terms.

These terms evaluate model outputs based on code execution results. They support three
levels of matching strictness:

- **hard_match**: Exact string comparison (strictest).
- **soft_match**: Heuristic-based semantic comparison with numeric tolerances.
- **llm_grader**: LLM-as-a-judge evaluation (most flexible).

All terms expect `SampleContext.code_exec_eval` to be populated with a `CodeExecEvalData`
instance containing the expected and predicted outputs.

Example usage:
    ```python
    from pyine.organisms.models.rewards.core.types import CodeExecEvalData, SampleContext

    # prepare sample context with code execution data
    sample_ctx = SampleContext(
        prompt="...",
        model_output="...",
        sample_data=sample_data,
        code_exec_eval=CodeExecEvalData(
            expected="42",
            predicted="42",
            predict_type="output",
            llm_grader_score=0.95,  # optional
            should_flip_reward=None,  # optional; leave None for SampleData-based flip logic
        ),
    )
    ```

Term registration:
    Terms are registered with a primary key and a namespaced alias:

    - Primary: ``hard_match``, Alias: ``code_exec/hard_match``
    - Primary: ``soft_match``, Alias: ``code_exec/soft_match``
    - Primary: ``llm_grader``, Alias: ``code_exec/llm_grader``
"""

import pyine.organisms.models.rewards.terms.code_exec.hard_match as hard_match
import pyine.organisms.models.rewards.terms.code_exec.llm_grader as llm_grader
import pyine.organisms.models.rewards.terms.code_exec.soft_match as soft_match
import pyine.organisms.models.rewards.terms.code_exec.utils as utils

__all__ = [
    "hard_match",
    "soft_match",
    "llm_grader",
    "utils",
]
