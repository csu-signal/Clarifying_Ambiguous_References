"""Training pools.

- `real`: every real episode of the split that some legal question sequence
  can resolve. Evaluation keeps the unsolvable ones; training drops them.
- `real+synth`: `real` plus every anchored synthetic episode, uniform over
  ask-depths 0-3. Used only to train SFT.
"""

from __future__ import annotations

from collections import Counter

from ..env.oracle import is_solvable
from .episodes import Episode, load_episodes
from .synthetic import load_synthetic_episodes

POOLS = ("real", "real+synth")


def load_pool(split: str, pool: str) -> list[Episode]:
    if pool not in POOLS:
        raise ValueError(f"unknown pool {pool!r}; expected one of {POOLS}")
    episodes = [e for e in load_episodes(split) if is_solvable(e)]
    if pool == "real+synth":
        episodes += load_synthetic_episodes(split)
    return episodes


def describe_pool(split: str, pool: str, episodes: list[Episode]) -> str:
    conditions = Counter(e.condition for e in episodes)
    synthetic = sum(e.synthetic for e in episodes)
    return f"pool={pool!r} split={split!r}: {len(episodes)} episodes ({dict(sorted(conditions.items()))}, synthetic={synthetic})"
