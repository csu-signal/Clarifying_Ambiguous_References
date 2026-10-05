"""BACE's belief: the candidates the policy can't yet rule out, computed from its own observation.

The policy never sees the environment's survivor set, so BACE keeps its own
belief B: the candidate list, narrowed by each reply using only what the
observation shows (permitted attributes, positions, relations). One rule per
kind of question:

- permitted attribute: keep the candidates whose listed value matches the
  reply. For availableSizes, "exact" keeps a candidate when its set of sizes
  equals the reply's (the environment's own grouping); "overlap" keeps any
  candidate sharing a size, which is looser and clears the exact flag.
- visual attribute: nothing to check against; clears the exact flag.
- spatial relation, "yes": keep the candidates with a neighbour in B in that
  direction. "No": the complement, only while B is exact.
- scene half: repeat the environment's median split on B, only while B is exact.
- unparsed, repeated or refused questions: no change.

The exact flag records whether B still equals the survivor set. Replayed on
every real episode, B contains the survivors at every state and, with
"exact" sizes, equals them.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from ...data.episodes import Episode
from ...data.simmc import NON_VISUAL_PERMITTED, VISUAL_FORBIDDEN
from ...data.splitters import rank_split
from ...env.attributes import SPATIAL_BIN_ATTRS, SPATIAL_GRAPH_ATTRS, UNRESOLVED_ASK_ATTRIBUTE
from ...env.environment import REPEATED_ATTRIBUTE_REPLY
from ...env.simulator import MISSING_REPLY, UNRESOLVED_REPLY, is_missing_value

SIZES_MODES = ("exact", "overlap")
_NO_OP_REPLIES = frozenset({UNRESOLVED_REPLY, REPEATED_ATTRIBUTE_REPLY})

# Words that state each side of a scene-half reply.
_BIN_WORDS = {
    "left_right": (("left",), ("right",)),
    "up_down": (("top", "above", "higher", "upper"), ("bottom", "below", "lower")),
}


def _bin_side(reply: str, attribute: str) -> str | None:
    lowered = reply.lower()
    low, high = _BIN_WORDS[attribute]
    if any(word in lowered for word in low):
        return "low"
    if any(word in lowered for word in high):
        return "high"
    return None


def _relation_leg(reply: str) -> bool | None:
    lowered = reply.lower()
    negative = any(p in lowered for p in ("no,", "no.", "not ", "n't", "isn't", "doesn't"))
    if any(p in lowered for p in ("yes", "it is", "it's")) and not negative:
        return True
    return False if negative else None


@dataclass(frozen=True)
class BeliefState:
    ids: tuple[int, ...]
    exact: bool
    asked: frozenset[str]
    asks_used: int


class BeliefTracker:
    def __init__(self, episode: Episode, candidate_ids: list[int], sizes: str = "exact") -> None:
        if sizes not in SIZES_MODES:
            raise ValueError(f"sizes must be one of {SIZES_MODES}, got {sizes!r}")
        self.episode = episode
        self.sizes = sizes
        self.ids = list(candidate_ids)
        self.exact = True
        self.asked: set[str] = set()
        self.asks_used = 0

    @property
    def state(self) -> BeliefState:
        return BeliefState(tuple(self.ids), self.exact, frozenset(self.asked), self.asks_used)

    def copy(self) -> "BeliefTracker":
        clone = BeliefTracker(self.episode, list(self.ids), self.sizes)
        clone.exact, clone.asked, clone.asks_used = self.exact, set(self.asked), self.asks_used
        return clone

    def note_refusal(self) -> None:
        """A repeated question: the turn is spent, nothing is learned."""
        self.asks_used += 1

    def observe(self, attribute: str, reply: str) -> None:
        self.asks_used += 1
        if attribute == UNRESOLVED_ASK_ATTRIBUTE:
            return
        self.asked.add(attribute)
        if reply in _NO_OP_REPLIES:
            return
        if attribute in VISUAL_FORBIDDEN:
            self.exact = False
        elif attribute in NON_VISUAL_PERMITTED:
            self._narrow_permitted(attribute, reply)
        elif attribute in SPATIAL_GRAPH_ATTRS:
            self._narrow_relation(attribute, reply)
        elif attribute in SPATIAL_BIN_ATTRS:
            self._narrow_half(attribute, reply)
        else:
            raise ValueError(f"unknown ask attribute: {attribute!r}")

    def _values(self, cid: int, attribute: str):
        return self.episode.candidates[cid].permitted_attrs.get(attribute)

    def _narrow_permitted(self, attribute: str, reply: str) -> None:
        if reply == MISSING_REPLY:
            kept = [cid for cid in self.ids if is_missing_value(self._values(cid, attribute))]
        elif not (reply.startswith("It's ") and reply.endswith(".")):
            self.exact = False  # not the scripted template, so nothing reliable to match
            return
        else:
            tokens = {token.strip().lower() for token in reply[len("It's "):-1].split(",")}
            listed = attribute == "availableSizes"
            kept = []
            for cid in self.ids:
                value = self._values(cid, attribute)
                if value is None:
                    continue
                values = value if isinstance(value, list) else [value]
                if listed and self.sizes == "exact":
                    if frozenset(str(v).strip().lower() for v in values if not is_missing_value(v)) == tokens:
                        kept.append(cid)
                elif {str(v).strip().lower() for v in values} & tokens:
                    kept.append(cid)
            if kept and listed and self.sizes == "overlap":
                self.ids = kept
                self.exact = False
                return
        if kept:
            self.ids = kept
        else:
            self.exact = False

    def _with_neighbour(self, attribute: str, ids: list[int]) -> set[int]:
        neighbours = self.episode.scene.relationships.get(attribute, {})
        id_set = set(ids)
        return {cid for cid in ids if any(n in id_set for n in neighbours.get(cid, []))}

    def _narrow_relation(self, attribute: str, reply: str) -> None:
        leg = _relation_leg(reply)
        if leg is None:
            return
        has = self._with_neighbour(attribute, self.ids)
        if leg:
            if has:
                self.ids = [cid for cid in self.ids if cid in has]
        elif self.exact:
            without = [cid for cid in self.ids if cid not in has]
            if without:
                self.ids = without

    def _narrow_half(self, attribute: str, reply: str) -> None:
        if not self.exact:
            return
        side = _bin_side(reply, attribute)
        if side is None:
            return
        axis = 0 if attribute == "left_right" else 1
        low, high = rank_split([(cid, self.episode.scene.bbox_center(cid)[axis]) for cid in self.ids])
        chosen = low if side == "low" else high
        if chosen:
            self.ids = chosen


def _log2(n: int) -> float:
    return math.log2(n) if n > 0 else 0.0


def listed_information_gain(episode: Episode, ids: list[int], attribute: str) -> float:
    """Expected gain of asking `attribute` with B uniform, from the listed
    (permitted) values only. Spatial attributes have no listed value, so they score zero."""
    n = len(ids)
    if n <= 1:
        return 0.0
    groups: Counter = Counter()
    for cid in ids:
        value = episode.candidates[cid].permitted_attrs.get(attribute)
        groups[tuple(value) if isinstance(value, list) else value] += 1
    return max(_log2(n) - sum((c / n) * _log2(c) for c in groups.values()), 0.0)


def belief_information_gain(episode: Episode, belief: BeliefState, attribute: str) -> float:
    """Expected gain under the tracker's own narrowing rules, spatial questions included."""
    ids = list(belief.ids)
    b = len(ids)
    if b <= 1:
        return 0.0
    if attribute in SPATIAL_BIN_ATTRS:
        if not belief.exact:
            return 0.0
        axis = 0 if attribute == "left_right" else 1
        low, high = rank_split([(cid, episode.scene.bbox_center(cid)[axis]) for cid in ids])
        sizes = [len(low)] * len(low) + [len(high)] * len(high)
    elif attribute in SPATIAL_GRAPH_ATTRS:
        neighbours = episode.scene.relationships.get(attribute, {})
        id_set = set(ids)
        has = {cid for cid in ids if any(n in id_set for n in neighbours.get(cid, []))}
        without = b - len(has)
        sizes = [len(has)] * len(has) + [without if belief.exact else b] * without
    else:
        return listed_information_gain(episode, ids, attribute)
    return max(_log2(b) - sum(_log2(size) for size in sizes) / b, 0.0)
