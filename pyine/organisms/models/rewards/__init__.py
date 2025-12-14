"""Reward definition framework for RL-style training loops.

This package provides:
- `RewardManager`: orchestrates term evaluation, aggregation, and optional logging.
- `RewardTerm`: composable per-sample reward components ("terms").
- `OutputParser`: optional output parsing to extract structured fields once per sample.

Key assumption (by design): rewards are computed once per training example (one prompt + full
model output + optional metadata). This package does not compute per-token rewards.
"""
