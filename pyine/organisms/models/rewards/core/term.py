"""Base classes and helpers for reward terms."""

import pyine.organisms.models.rewards.core.types


class BaseRewardTerm:
    """Base class for reward terms with a no-op reset.

    Most reward terms are stateless; implementing `reset()` as a no-op avoids boilerplate.
    """

    def reset(
        self,
        run_init_ctx: pyine.organisms.models.rewards.core.types.RunInitContext,
    ) -> None:
        """Reset the term state for a new run."""
        del run_init_ctx
