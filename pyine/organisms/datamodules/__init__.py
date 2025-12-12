from pyine.configs.schemas import ConfigDescription
from pyine.evals.common import EvalType


def get_configs(
    eval_type: EvalType,
    group: str,
) -> list[ConfigDescription]:
    """Generates and returns shortcuts-datamodule-specific configs for hydra zen storage."""
    from .keywords_configs import get_configs as get_keywords_configs
    from .shortcuts_configs import get_configs as get_shortcuts_configs

    return [
        *get_keywords_configs(eval_type=eval_type, group=group),
        *get_shortcuts_configs(eval_type=eval_type, group=group),
    ]
