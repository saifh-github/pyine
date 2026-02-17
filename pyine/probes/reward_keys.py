"""Canonical reward metric/term key constants for probe training.

These match the prefixed format produced by
``RewardManager._scope_reward_sample_fields()`` (which applies
``reward/metrics/`` and ``reward/terms/`` prefixes) and stored in LMDB
by ``DiskRewardLogger``.
"""

# Canonical key prefixes
REWARD_METRICS_PREFIX = "reward/metrics/"
REWARD_TERMS_PREFIX = "reward/terms/"

# Canonical metric keys used by probe training
SOFT_MATCH_KEY = f"{REWARD_METRICS_PREFIX}soft_match/is_match"
HARD_MATCH_KEY = f"{REWARD_METRICS_PREFIX}hard_match/is_match"
