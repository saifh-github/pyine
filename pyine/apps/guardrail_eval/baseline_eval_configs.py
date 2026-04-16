"""Hydra-zen config builder for the baseline guardrail evaluation app."""

from __future__ import annotations

import logging
import typing

import pydantic

import pyine.configs.base
import pyine.configs.schemas
import pyine.configs.searchpath
import pyine.configs.utils
import pyine.evals.common
import pyine.evals.correctness.baseline_scorers as baseline_scorers
import pyine.evals.correctness.configs
import pyine.evals.correctness.types as correctness_types
import pyine.utils.reprod

logger = logging.getLogger(__name__)


class ConstantBaselineConfig(pydantic.BaseModel):
    """Configuration for a constant-value baseline scorer."""

    model_config = pydantic.ConfigDict(extra="forbid")
    """Pydantic model configuration."""

    scorer_type: typing.Literal["constant"] = "constant"
    """Discriminator tag for the baseline config union."""
    name: str
    """Unique name for this baseline (used as the guardrail type key)."""
    score_value: float = 0.5
    """Fixed score returned for every record."""
    num_replicas: int = 10
    """Number of scorer replicas to evaluate (all identical for constant baselines)."""

    def build_scorers(self) -> list[correctness_types.GuardrailScorer]:
        """Build the requested number of constant scorer replicas."""
        scorer = baseline_scorers.ConstantScorer(score_value=self.score_value)
        return [scorer] * self.num_replicas


class UniformRandomBaselineConfig(pydantic.BaseModel):
    """Configuration for a uniform-random baseline scorer."""

    model_config = pydantic.ConfigDict(extra="forbid")
    """Pydantic model configuration."""

    scorer_type: typing.Literal["uniform_random"] = "uniform_random"
    """Discriminator tag for the baseline config union."""
    name: str
    """Unique name for this baseline (used as the guardrail type key)."""
    seed: int = 42
    """Base random seed. Each replica gets ``seed + replica_index``."""
    num_replicas: int = 10
    """Number of scorer replicas to evaluate (each with an independent seed)."""

    def build_scorers(self) -> list[correctness_types.GuardrailScorer]:
        """Build the requested number of uniform-random scorer replicas."""
        return [
            baseline_scorers.UniformRandomScorer(seed=self.seed + replica_idx)
            for replica_idx in range(self.num_replicas)
        ]


BaselineConfig = typing.Annotated[
    ConstantBaselineConfig | UniformRandomBaselineConfig,
    pydantic.Discriminator("scorer_type"),
]
"""Discriminated union of baseline scorer configurations."""


class BaselineEvalAppConfig(pydantic.BaseModel):
    """Standalone evaluation app for baseline sanity-check guardrails.

    Runs the correctness evaluation pipeline with trivial scorers (constant, uniform random) to
    establish reference metrics.
    """

    model_config = pydantic.ConfigDict(extra="forbid")
    """Pydantic model configuration."""

    baseline_configs: list[BaselineConfig]
    """One or more baseline scorer configurations to evaluate."""
    evals_config: pyine.evals.correctness.configs.CorrectnessEvalsConfig
    """Correctness eval pipeline configuration."""
    use_wandb_logging: bool = False
    """Whether to log results to Weights & Biases."""
    wandb_project: str | None = None
    """W&B project name (only used when use_wandb_logging=True)."""

    @pydantic.model_validator(mode="after")
    def _validate_unique_names(self) -> BaselineEvalAppConfig:
        """Ensure all baseline names are unique."""
        names = [config.name for config in self.baseline_configs]
        duplicates = [name for name in names if names.count(name) > 1]
        if duplicates:
            raise ValueError(f"duplicate baseline names: {sorted(set(duplicates))}")
        return self


def _get_app_configs(
    group: str,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Generates and returns baseline eval application configs for hydra zen storage."""
    evals_configs = pyine.evals.correctness.configs.get_evals_configs(
        group=f"{group}/evals_config",
    )
    app_main_config = pyine.configs.utils.make_config_description(
        BaselineEvalAppConfig,
        name="base",
        group=group,
        description="Base settings for the baseline guardrail eval app.",
        config={
            "populate_full_signature": True,
            "hydra_convert": "object",
        },
    )
    return [app_main_config, *evals_configs]


def register_hydra_configs(
    eval_type: pyine.evals.common.EvalType,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Registers baseline-eval-specific configs in hydra and returns config descriptions."""
    from pyine.apps.guardrail_eval.baseline_eval import baseline_eval_main_wrapper

    pyine.utils.reprod.load_dotenv()

    entrypoint_config = pyine.configs.utils.make_config_description(
        baseline_eval_main_wrapper,
        name="entrypoint",
        group=None,
        description="Entrypoint settings for the baseline guardrail eval app.",
        config={
            "populate_full_signature": True,
            "hydra_defaults": [
                "_self_",
                {"config": "base"},
                {"runtime": "default"},
                *pyine.configs.base.get_base_hydra_default_overrides(),
            ],
        },
    )

    store, base_configs = pyine.configs.base.get_base_store_and_configs("baseline_eval")
    app_configs = _get_app_configs(group="config")
    configs_to_register = [entrypoint_config, *app_configs]

    external_configs = pyine.configs.searchpath.SearchPathPlugin.get_external_configs(
        app_name="baseline_eval",
        eval_type=eval_type,
        entrypoint_config=entrypoint_config,
        app_configs=[*base_configs, *configs_to_register],
    )
    configs_to_register.extend(external_configs)

    for config in configs_to_register:
        assert config.name is not None
        store(
            typing.cast("typing.Any", config.config),
            name=config.name,
            group=config.group,
            package=config.package,
        )
    store.add_to_hydra_store(overwrite_ok=True)
    return [*base_configs, *configs_to_register]


if __name__ == "__main__":
    import sys

    pyine.configs.base.register_searchpath_plugin()
    pyine.configs.utils.print_experiment_configs(
        config_descriptions=register_hydra_configs(eval_type=pyine.evals.common.EvalType.CORRECTNESS),
        app_name="baseline_eval",
        cli_args=sys.argv[1:],
    )
