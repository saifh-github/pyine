"""TRL (HuggingFace Transformers Reinforcement Learning) integration for reward computation.

This module provides adapters to use our reward system with TRL-based training pipelines
(e.g., GRPOTrainer, PPOTrainer). The key challenge is mapping TRL's batch-based reward function
signature to our SampleContext-based reward computation.

TRL reward function signature (as used by GRPOTrainer):
    ```python
    def reward_fn(
        prompts: list[str] | list[list[dict[str, str]]],
        completions: list[list[dict[str, str]]],
        **kwargs,
    ) -> list[float | None]: ...
    ```

Where:
- `prompts` is a list of prompts (strings for standard format, message lists for conversational);
- `completions[i]` is a list of messages (dicts with "role" and "content" keys) for sample `i`;
- `kwargs` contains other dataset columns (e.g., sample_data, ground_truth);
- the returned output is a list of rewards (or None to skip a sample).

Usage:
    ```python
    import pyine.organisms.models.rewards as rewards
    import pyine.organisms.models.rewards.trl as rewards_trl
    from pyine.organisms.models.rewards.core import configs as reward_configs

    # create a reward manager with config
    config = reward_configs.RewardManagerConfig(
        terms=[reward_configs.RewardTermSpec(name="match", type="hard_match", weight=1.0)],
        parsing=reward_configs.ParsingConfig(final_tag="final"),
        logging=reward_configs.LoggingConfig(enabled=False),  # or pass logger= to RewardManager
    )
    manager = rewards.RewardManager(config)

    # create a TRL-compatible reward adapter
    adapter = rewards_trl.TRLRewardAdapter(
        manager=manager,
        sample_data_key="sample_data",
    )

    # use with GRPOTrainer
    trainer = GRPOTrainer(..., reward_funcs=[adapter])
    ```
"""

import collections.abc
import dataclasses
import enum
import logging
import typing

import pyine.organisms.datamodules.samples.common as samples_common
import pyine.organisms.models.rewards.core.types as reward_types

if typing.TYPE_CHECKING:
    import pyine.organisms.models.rewards.core.manager as reward_manager

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, slots=True)
class TRLRewardResult:
    """Result from TRL reward computation with optional diagnostics."""

    rewards: list[float | None]
    """List of reward values (None indicates a skipped sample)."""
    outputs: list[reward_types.RewardOutput | None] | None = None
    """Full RewardOutput objects for each sample (if requested)."""
    errors: dict[int, str] = dataclasses.field(default_factory=lambda: {})
    """Error messages for failed samples (indexed by sample position)."""


type TRLCompletions = list[list[dict[str, str]]]
"""TRL completions format: list of samples, where each sample is a list of message dicts."""
type SampleContextBuilder = collections.abc.Callable[
    [str, str, int, collections.abc.Mapping[str, typing.Any]],
    reward_types.SampleContext | None,
]
"""Callable that builds a SampleContext from (prompt, model_output, sample_idx, kwargs).

Returns None to skip the sample (reward will be None).
"""


class MessageSelectionPolicy(enum.Enum):
    """Policy for selecting which message to extract from a completion with multiple messages."""

    first = "first"
    """Select the first message with the specified role."""
    last = "last"
    """Select the last message with the specified role."""


def _extract_model_output(
    completion: list[dict[str, str]],
    role: str = "assistant",
    policy: MessageSelectionPolicy = MessageSelectionPolicy.last,
) -> str:
    """Extract the model output text from a TRL completion.

    Args:
        completion: List of message dicts (each with "role" and "content" keys).
        role: Role to extract content from (default "assistant").
        policy: Which message to select if multiple match (default "last").

    Returns:
        The content string from the selected message with the specified role.

    Raises:
        ValueError: If no message with the specified role is found.
    """
    matches = [str(msg["content"]) for msg in completion if msg.get("role") == role and msg.get("content") is not None]
    if not matches:
        raise ValueError(f"no message with role '{role}' found in completion: {completion}")
    if policy == MessageSelectionPolicy.first:
        return matches[0]
    return matches[-1]


class TRLRewardAdapter:
    """Adapter that wraps a RewardManager for use with TRL training pipelines.

    This adapter provides a callable with TRL's expected reward function signature,
    handling the conversion between TRL's batch-based format and our SampleContext objects.

    The adapter can be customized via:
    - `prompt_key`: the kwarg key containing prompts (default "prompts", as passed by TRL);
    - `sample_data_key`: the kwarg key containing SampleData objects;
    - `context_builder`: a custom callable to build SampleContext objects.

    Example:
        ```python
        adapter = TRLRewardAdapter(
            manager=reward_manager,
            sample_data_key="sample_data",
        )

        # use as TRL reward function (TRL passes prompts automatically)
        rewards = adapter(completions, prompts=prompts, sample_data=sample_data_list)
        ```
    """

    def __init__(
        self,
        manager: "reward_manager.RewardManager",
        *,
        prompt_key: str = "prompts",
        sample_data_key: str = "sample_data",
        context_builder: SampleContextBuilder | None = None,
        assistant_role: str = "assistant",
        message_selection_policy: MessageSelectionPolicy = MessageSelectionPolicy.last,
        skip_on_error: bool = False,
        return_none_on_skip: bool = True,
    ) -> None:
        """Initialize the TRL reward adapter.

        Args:
            manager: The RewardManager instance to use for reward computation.
            prompt_key: Kwarg key for the list of prompts (default "prompts", as passed by TRL).
            sample_data_key: Kwarg key for SampleData objects (default "sample_data").
                Required unless a custom context_builder is provided.
            context_builder: Optional custom callable to build SampleContext objects.
                Signature: (prompt, model_output, sample_idx, kwargs) -> SampleContext | None.
                When provided, overrides prompt_key and sample_data_key logic.
            assistant_role: Message role to extract model output from (default "assistant").
            message_selection_policy: Which message to select if multiple match (default "last").
            skip_on_error: If True, return None for samples that fail; if False, raise.
            return_none_on_skip: If True, return None for skipped samples; if False, return 0.0.
        """
        self._manager = manager
        self._prompt_key = prompt_key
        self._sample_data_key = sample_data_key
        self._context_builder = context_builder
        self._assistant_role = assistant_role
        self._message_selection_policy = message_selection_policy
        self._skip_on_error = skip_on_error
        self._return_none_on_skip = return_none_on_skip
        self._total_count = 0
        self._skip_count = 0
        self._error_count = 0
        self.__name__ = "TRLRewardAdapter"

    def __call__(
        self,
        completions: TRLCompletions,
        **kwargs: typing.Any,
    ) -> list[float | None]:
        """Compute rewards for a batch of TRL completions.

        Args:
            completions: List of completions, each is a list of message dicts.
            **kwargs: Additional data passed by TRL (prompts, completions, etc.).

        Returns:
            List of reward values. None indicates a skipped sample.
        """
        result = self.compute(completions, **kwargs)
        return result.rewards

    def compute(
        self,
        completions: TRLCompletions,
        *,
        return_outputs: bool = False,
        **kwargs: typing.Any,
    ) -> TRLRewardResult:
        """Compute rewards with full result information.

        For relative verbosity scaling, groups completions by trace_id (prompt) before computing,
        so samples for the same prompt are normalized together. Per-sample errors are caught
        individually during context building.

        Args:
            completions: List of completions, each is a list of message dicts.
            return_outputs: If True, include full RewardOutput objects in result.
            **kwargs: Additional data passed by TRL.

        Returns:
            TRLRewardResult with rewards, optional outputs, and error info.
        """
        batch_size = len(completions)
        self._total_count += batch_size
        # build contexts with per-sample error handling
        contexts: list[reward_types.SampleContext | None] = []
        errors: dict[int, str] = {}
        prompts = self._get_prompts(batch_size, kwargs)
        sample_data_list = None if self._context_builder else self._get_sample_data_list(batch_size, kwargs)
        for sample_idx in range(batch_size):
            try:
                ctx = self._build_context(
                    completion=completions[sample_idx],
                    prompt=prompts[sample_idx] if prompts else "",
                    sample_data=sample_data_list[sample_idx] if sample_data_list else None,
                    sample_idx=sample_idx,
                    kwargs=kwargs,
                )
                if ctx is None:
                    self._skip_count += 1
                contexts.append(ctx)
            except Exception as exc:
                if not self._skip_on_error:
                    raise
                self._error_count += 1
                error_msg = f"{type(exc).__name__}: {exc}"
                errors[sample_idx] = error_msg
                logger.warning(f"TRL context build failed for sample {sample_idx}: {error_msg}")
                contexts.append(None)
        # separate valid contexts for batch processing
        valid_indices = [idx for idx, ctx in enumerate(contexts) if ctx is not None]
        valid_contexts = [contexts[idx] for idx in valid_indices]
        if not valid_contexts:
            # all samples failed/skipped
            rewards: list[float | None] = [None if self._return_none_on_skip else 0.0] * batch_size
            outputs_when_empty: list[reward_types.RewardOutput | None] | None = (
                typing.cast("list[reward_types.RewardOutput | None]", [None] * batch_size) if return_outputs else None
            )
            return TRLRewardResult(
                rewards=rewards,
                outputs=outputs_when_empty,
                errors=errors,
            )
        # compute rewards (compute_batch auto-groups by trace_id for relative verbosity scaling)
        # note: we intentionally don't catch errors here; if compute_batch fails on valid contexts,
        # that's a bug in the reward manager that should be fixed, not silently degraded
        reward_outputs = self._manager.compute_batch(valid_contexts)  # type: ignore[arg-type]
        # map results back to original indices
        rewards: list[float | None] = [None if self._return_none_on_skip else 0.0] * batch_size
        full_outputs: list[reward_types.RewardOutput | None] | None = (
            typing.cast("list[reward_types.RewardOutput | None]", [None] * batch_size) if return_outputs else None
        )
        for local_idx, global_idx in enumerate(valid_indices):
            output = reward_outputs[local_idx]
            rewards[global_idx] = output.total
            if full_outputs is not None:
                full_outputs[global_idx] = output
        return TRLRewardResult(rewards=rewards, outputs=full_outputs, errors=errors)

    def _get_prompts(
        self,
        batch_size: int,
        kwargs: collections.abc.Mapping[str, typing.Any],
    ) -> list[str] | None:
        """Extract prompts from kwargs.

        TRL passes prompts in different formats depending on the dataset:
        - Standard format: list of strings;
        - Conversational format: list of message lists (each message is a dict with "role" and "content").

        This method normalizes both formats to a list of strings for logging purposes.
        """
        prompts: typing.Any = kwargs.get(self._prompt_key)
        if prompts is None:
            return None
        if not isinstance(prompts, (list, tuple)):
            raise TypeError(f"expected list for '{self._prompt_key}', got {type(prompts)}")
        prompts = typing.cast("collections.abc.Sequence[typing.Any]", prompts)
        if len(prompts) != batch_size:
            raise ValueError(f"prompts length ({len(prompts)}) != batch size ({batch_size})")
        result: list[str] = []
        for prompt_idx, prompt in enumerate(prompts):
            if isinstance(prompt, str):
                result.append(prompt)
            elif isinstance(prompt, (list, tuple)):
                # conversational format: list of message dicts with "role" and "content"
                prompt = typing.cast("collections.abc.Sequence[typing.Any]", prompt)
                assert all(isinstance(msg, dict) and isinstance(msg["content"], str) for msg in prompt)
                result.append("\n\n".join([msg["content"] for msg in prompt]))
            else:
                raise TypeError(f"expected str or list for prompt at index {prompt_idx}, got {type(prompt)}")
        return result

    @staticmethod
    def _fix_sample_data_dict(data: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """Fix enum fields in SampleData dict that were serialized to strings by HF datasets.

        HuggingFace datasets serialize enums to strings, so we need to convert them back.
        """
        fixed = dict(data)
        # Convert predict_type from string back to enum if needed
        if "predict_type" in fixed and isinstance(fixed["predict_type"], str):
            fixed["predict_type"] = samples_common.SamplePredictType(fixed["predict_type"])
        return fixed

    def _get_sample_data_list(
        self,
        batch_size: int,
        kwargs: collections.abc.Mapping[str, typing.Any],
    ) -> list[samples_common.SampleData]:
        """Extract SampleData objects from kwargs.

        Supports two common formats:
        - `SampleData` instances (already reconstructed), and
        - dict-like mappings (HuggingFace datasets can serialize NamedTuples to dicts).

        For mapping inputs, reconstructs `SampleData` instances and fixes enum fields that were
        serialized to strings by HF datasets.
        """
        sample_data: typing.Any = kwargs.get(self._sample_data_key)

        if not isinstance(sample_data, (list, tuple)):
            raise TypeError(f"expected list for '{self._sample_data_key}', got {type(sample_data)}")
        sample_data = typing.cast("collections.abc.Sequence[typing.Any]", sample_data)
        if len(sample_data) != batch_size:
            raise ValueError(f"sample_data length ({len(sample_data)}) != batch size ({batch_size})")

        result: list[samples_common.SampleData] = []
        for idx, sd in enumerate(sample_data):
            if isinstance(sd, samples_common.SampleData):
                result.append(sd)
                continue
            if isinstance(sd, collections.abc.Mapping):
                try:
                    sd_fixed = self._fix_sample_data_dict(dict(sd))  # type: ignore[arg-type]
                    result.append(samples_common.SampleData(**sd_fixed))
                except Exception as exc:
                    raise ValueError(f"failed to reconstruct SampleData from mapping at index {idx}: {exc}") from exc
                continue
            raise TypeError(
                f"expected SampleData or mapping for '{self._sample_data_key}' at index {idx}, got {type(sd)}"
            )
        return result

    def get_failure_stats(
        self,
    ) -> dict[str, int]:
        """Return accumulated failure statistics since last reset.

        Returns:
            Dictionary with keys: total_count, skip_count, error_count.
        """
        return {
            "total_count": self._total_count,
            "skip_count": self._skip_count,
            "error_count": self._error_count,
        }

    def reset_failure_stats(
        self,
    ) -> None:
        """Reset all failure statistics counters to zero."""
        self._total_count = 0
        self._skip_count = 0
        self._error_count = 0

    def get_state(
        self,
    ) -> dict[str, int]:
        """Return serializable state for checkpoint persistence.

        Returns:
            Dictionary with keys: total_count, skip_count, error_count.
        """
        return self.get_failure_stats()

    def load_state(
        self,
        state: dict[str, int],
    ) -> None:
        """Restore state from checkpoint.

        Args:
            state: Dictionary with keys: total_count, skip_count, error_count.

        Raises:
            KeyError: If required keys are missing from state.
        """
        self._total_count = state["total_count"]
        self._skip_count = state["skip_count"]
        self._error_count = state["error_count"]

    def _build_context(
        self,
        completion: list[dict[str, str]],
        prompt: str,
        sample_data: samples_common.SampleData | None,
        sample_idx: int,
        kwargs: collections.abc.Mapping[str, typing.Any],
    ) -> reward_types.SampleContext | None:
        """Build a SampleContext from TRL inputs."""
        model_output = _extract_model_output(
            completion,
            role=self._assistant_role,
            policy=self._message_selection_policy,
        )
        if self._context_builder is not None:
            return self._context_builder(prompt, model_output, sample_idx, kwargs)
        assert sample_data is not None  # guaranteed by _get_sample_data_list when no custom builder
        parsed_output = self._manager.maybe_parse(prompt, model_output)
        code_exec_eval = reward_types.CodeExecEvalData(
            expected=sample_data.expected_output,
            predicted=(
                parsed_output.final_answer if parsed_output and parsed_output.final_answer is not None else model_output
            ),
            predict_type=str(sample_data.predict_type.value),
            should_flip_reward=sample_data.should_flip_reward(),
        )
        return reward_types.SampleContext(
            prompt=prompt,
            model_output=model_output,
            parsed=parsed_output,
            sample_data=sample_data,
            code_exec_eval=code_exec_eval,
        )
