"""The oracle that defines an episode's gold questions and its gold depth.

It knows the referent and repeatedly asks the attribute that best splits the
current survivors, recomputing information gain after every reply, until
only the referent is left. Each attribute is asked at most once; attributes
the history already pins down or the turn itself requests are skipped after
the first question. The number of questions it asks is the episode's gold
depth (0 for episodes that need none).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..config import BUDGET
from ..data.episodes import Episode
from ..data.splitters import SplitterEntry, intrinsic_entries, spatial_bin_entries, spatial_graph_entries
from .attributes import question_for_attribute
from .simulator import apply_attribute_reveal

# Always splits two or more survivors, so the first question never stalls.
DEFAULT_ASK_ATTRIBUTE = "left_right"
# Deep enough that a chain only stops short when every attribute is used up.
SOLVABILITY_BUDGET = 10


def ranked_attributes(episode: Episode, survivor_ids: list[int], exclude: set[str]) -> list[SplitterEntry]:
    """Every attribute not in `exclude`, best information gain on `survivor_ids` first."""
    survivor_ids = list(dict.fromkeys(survivor_ids))
    s_size = len(survivor_ids)
    h_s = math.log2(s_size) if s_size > 1 else 0.0
    raw_attrs = {cid: episode.candidates[cid].raw_attrs for cid in survivor_ids}
    entries = intrinsic_entries(h_s, s_size, survivor_ids, raw_attrs)
    entries += spatial_bin_entries(h_s, s_size, survivor_ids, episode.scene)
    entries += spatial_graph_entries(h_s, s_size, survivor_ids, episode.scene)
    kept = [e for e in entries if e.attribute not in exclude]
    kept.sort(key=lambda e: (-e.ig, e.kind, e.attribute))
    return kept


def first_question_attribute(episode: Episode) -> str:
    """The best-ranked splitter label not already pinned by the history."""
    for attribute in episode.gold_splitting_attributes:
        if attribute not in episode.established_attrs:
            return attribute
    return DEFAULT_ASK_ATTRIBUTE


@dataclass(frozen=True)
class GoldStep:
    attribute: str
    question: str
    survivors_before: list[int]
    survivors_after: list[int]
    revealed_value: Any


@dataclass(frozen=True)
class GoldChain:
    steps: list[GoldStep]
    resolved: bool  # survivors narrowed to the gold referent

    @property
    def final_survivors(self) -> list[int]:
        return self.steps[-1].survivors_after if self.steps else []


def build_gold_chain(episode: Episode, budget: int) -> GoldChain:
    survivors = list(episode.survivor_ids)
    asked: set[str] = set()
    steps: list[GoldStep] = []
    resolved = survivors == [episode.gold_referent]
    while not resolved and len(steps) < budget:
        if not steps:
            attribute = first_question_attribute(episode)
        else:
            if len(survivors) <= 1:
                attribute = DEFAULT_ASK_ATTRIBUTE
            else:
                ranked = ranked_attributes(episode, survivors, asked | set(episode.established_attrs) | set(episode.request_slots))
                attribute = ranked[0].attribute if ranked else None
            if attribute is None or attribute in asked:
                break
        new_survivors, value = apply_attribute_reveal(episode, attribute, survivors)
        steps.append(GoldStep(attribute, question_for_attribute(attribute), survivors, new_survivors, value))
        asked.add(attribute)
        survivors = new_survivors
        resolved = survivors == [episode.gold_referent]
    return GoldChain(steps=steps, resolved=resolved)


_DEPTHS: dict[tuple[int, int], tuple[Episode, int]] = {}


def gold_depth(episode: Episode, budget: int = BUDGET) -> int:
    """Questions the oracle asks within `budget`. Episodes that need more, or
    can't be resolved at all, count as `budget`. Memoised per episode object."""
    cached = _DEPTHS.get((id(episode), budget))
    if cached is None or cached[0] is not episode:
        cached = _DEPTHS[(id(episode), budget)] = (episode, len(build_gold_chain(episode, budget).steps))
    return cached[1]


def is_solvable(episode: Episode) -> bool:
    """Whether some legal question sequence isolates the referent."""
    return episode.condition != "irreducible" or build_gold_chain(episode, SOLVABILITY_BUDGET).resolved
