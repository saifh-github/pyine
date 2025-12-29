"""TRL (HuggingFace Transformers Reinforcement Learning) integration for reward computation.

This module provides adapters to use our reward system with TRL-based training pipelines
(e.g., GRPOTrainer, PPOTrainer). The key challenge is mapping TRL's batch-based reward function
signature to our SampleContext-based reward computation.

TRL reward function signature (as used by GRPOTrainer):
    ```python
    def reward_fn(
        completions: list[list[dict[str, str]]],
        **kwargs,
    ) -> list[float | None]: ...
    ```

Where:
- `completions[i]` is a list of messages (dicts with "role" and "content" keys) for sample `i`;
- `kwargs` contains batch-level data (e.g., prompts, solutions, any extra columns from the dataset);
- the returned output is a list of rewards (or None to skip a sample).

Usage:
    ```python
    import pyine.organisms.models.rewards as rewards
    import pyine.organisms.models.rewards.trl as rewards_trl

    # create a reward manager with desired terms
    manager = rewards.make_simple_manager([("match", "hard_match", 1.0)])

    # create a TRL-compatible reward function
    reward_fn = rewards_trl.make_trl_reward_fn(
        manager=manager,
        prompt_key="prompt",  # kwarg key for prompts
        sample_data_key="sample_data",  # kwarg key for SampleData objects
    )

    # use with GRPOTrainer
    trainer = GRPOTrainer(..., reward_funcs=[reward_fn])
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
    errors: dict[int, str] = dataclasses.field(default_factory=lambda: dict[int, str]())
    """List of error messages for failed samples (indexed by sample position)."""


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
    - `prompt_key`: the kwarg key containing prompts (list of strings);
    - `sample_data_key`: the kwarg key containing SampleData objects;
    - `context_builder`: a custom callable to build SampleContext objects.

    Example:
        ```python
        adapter = TRLRewardAdapter(
            manager=reward_manager,
            prompt_key="prompt",
            sample_data_key="sample_data",
        )

        # use as TRL reward function
        rewards = adapter(completions, prompt=prompts, sample_data=sample_data_list)
        ```
    """

    def __init__(
        self,
        manager: "reward_manager.RewardManager",
        *,
        prompt_key: str = "prompt",
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
            prompt_key: Kwarg key for the list of prompts (default "prompt").
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

        Args:
            completions: List of completions, each is a list of message dicts.
            return_outputs: If True, include full RewardOutput objects in result.
            **kwargs: Additional data passed by TRL.

        Returns:
            TRLRewardResult with rewards, optional outputs, and error info.
        """
        batch_size = len(completions)
        self._total_count += batch_size
        rewards: list[float | None] = []
        outputs: list[reward_types.RewardOutput | None] | None = [] if return_outputs else None
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
                    rewards.append(None if self._return_none_on_skip else 0.0)
                    if outputs is not None:
                        outputs.append(None)
                    continue
                output = self._manager.compute_output(ctx)
                rewards.append(output.total)
                if outputs is not None:
                    outputs.append(output)
            except Exception as exc:
                if not self._skip_on_error:
                    raise
                self._error_count += 1
                error_msg = f"{type(exc).__name__}: {exc}"
                errors[sample_idx] = error_msg
                logger.warning("TRL reward computation failed for sample %d: %s", sample_idx, error_msg)
                rewards.append(None if self._return_none_on_skip else 0.0)
                if outputs is not None:
                    outputs.append(None)
        return TRLRewardResult(rewards=rewards, outputs=outputs, errors=errors)

    def _get_prompts(
        self,
        batch_size: int,
        kwargs: collections.abc.Mapping[str, typing.Any],
    ) -> list[str] | None:
        """Extract prompts from kwargs."""
        prompts: typing.Any = kwargs.get(self._prompt_key)
        if prompts is None:
            return None
        if not isinstance(prompts, (list, tuple)):
            raise TypeError(f"expected list for '{self._prompt_key}', got {type(prompts)}")
        prompts = typing.cast("collections.abc.Sequence[typing.Any]", prompts)
        if len(prompts) != batch_size:
            raise ValueError(f"prompts length ({len(prompts)}) != batch size ({batch_size})")
        for idx, p in enumerate(prompts):
            if not isinstance(p, str):
                raise TypeError(f"expected str for prompt at index {idx}, got {type(p)}")
        return list(prompts)

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

        Handles three cases:
        1. "sample_data" key with SampleData objects
        2. "sample_data" key with dicts (HF datasets serialize NamedTuples to dicts)
        """
        sample_data: typing.Any = kwargs.get(self._sample_data_key)

        if not isinstance(sample_data, (list, tuple)):
            raise TypeError(f"expected list for '{self._sample_data_key}', got {type(sample_data)}")
        sample_data = typing.cast("collections.abc.Sequence[typing.Any]", sample_data)
        if len(sample_data) != batch_size:
            raise ValueError(f"sample_data length ({len(sample_data)}) != batch size ({batch_size})")

        result: list[samples_common.SampleData] = []
        for idx, sd in enumerate(sample_data):
            # Dictionary from HuggingFace dataset - reconstruct SampleData (NamedTuple)
            # Need to convert enum fields back from strings
            try:
                sd_dict = typing.cast("dict[str, typing.Any]", sd)
                sd_fixed = self._fix_sample_data_dict(sd_dict)
                result.append(samples_common.SampleData(**sd_fixed))
            except Exception as exc:
                raise ValueError(f"failed to reconstruct SampleData from dict at index {idx}: {exc}") from exc
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


def make_trl_reward_fn(
    manager: "reward_manager.RewardManager",
    *,
    prompt_key: str = "prompt",
    sample_data_key: str = "sample_data",
    context_builder: SampleContextBuilder | None = None,
    assistant_role: str = "assistant",
    message_selection_policy: MessageSelectionPolicy = MessageSelectionPolicy.last,
    skip_on_error: bool = True,
) -> collections.abc.Callable[..., list[float | None]]:
    """Create a TRL-compatible reward function from a RewardManager.

    This is the recommended way to integrate our rewards with TRL trainers.

    Args:
        manager: The RewardManager instance to use for reward computation.
        prompt_key: Kwarg key for the list of prompts (default "prompt").
        sample_data_key: Kwarg key for SampleData objects (default "sample_data").
            Required unless a custom context_builder is provided.
        context_builder: Optional custom callable to build SampleContext objects.
        assistant_role: Message role to extract model output from (default "assistant").
        message_selection_policy: Which message to select if multiple match (default "last").
        skip_on_error: If True, return None for samples that fail; if False, raise.

    Returns:
        A callable with signature `(completions, **kwargs) -> list[float | None]`.
        The returned function has a `__name__` attribute for TRL compatibility.

    Example:
        ```python
        import pyine.organisms.models.rewards as rewards
        import pyine.organisms.models.rewards.trl as rewards_trl

        manager = rewards.make_simple_manager([("format", "parseable_answer", 1.0)])
        reward_fn = rewards_trl.make_trl_reward_fn(manager, prompt_key="prompt")

        # use with TRL trainer
        from trl import GRPOTrainer

        trainer = GRPOTrainer(..., reward_funcs=[reward_fn])
        ```
    """
    # Create the adapter instance
    adapter = TRLRewardAdapter(
        manager=manager,
        prompt_key=prompt_key,
        sample_data_key=sample_data_key,
        context_builder=context_builder,
        assistant_role=assistant_role,
        message_selection_policy=message_selection_policy,
        skip_on_error=skip_on_error,
    )

    # Return a proper function (not a class instance) for cleaner TRL integration
    # Functions naturally have __name__ attribute, so TRL's reward tracking works seamlessly
    def trl_reward_function(
        completions: TRLCompletions,
        **kwargs: typing.Any,
    ) -> list[float | None]:
        """TRL-compatible reward function that wraps a RewardManager."""
        return adapter(completions, **kwargs)

    return trl_reward_function
