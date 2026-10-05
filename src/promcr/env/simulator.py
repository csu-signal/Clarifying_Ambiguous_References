"""The scripted user: answers a clarifying question truthfully about the gold referent."""

from __future__ import annotations

from typing import Any

from ..data.episodes import Episode
from ..data.splitters import canonical_value, rank_split
from .attributes import INTRINSIC_ATTRS, SPATIAL_BIN_ATTRS, SPATIAL_GRAPH_ATTRS, UNRESOLVED_ASK_ATTRIBUTE

UNRESOLVED_REPLY = "Sorry, I didn't understand the question. Could you rephrase it?"
MISSING_REPLY = "I don't have that information."

# y grows downward in SIMMC bboxes, so the "low" half is the top one.
SPATIAL_BIN_PHRASES = {
    "left_right": ("It's the one on the left.", "It's the one on the right."),
    "up_down": ("It's the one at the top.", "It's the one at the bottom."),
}


def apply_attribute_reveal(episode: Episode, attribute: str, survivor_ids: list[int]) -> tuple[list[int], Any]:
    """(survivors that agree with the gold referent on `attribute`, the gold referent's value).

    Uses the same grouping rules as the splitter labels: canonical value
    equality for intrinsic attributes, a median split for spatial bins, and
    "has a neighbour among the survivors" for spatial relations. An
    unresolved ask narrows nothing.
    """
    if attribute == UNRESOLVED_ASK_ATTRIBUTE:
        return list(survivor_ids), None
    gold = episode.gold_referent
    if gold not in survivor_ids:
        raise ValueError(f"gold referent {gold} is not among the survivors {survivor_ids}")

    if attribute in INTRINSIC_ATTRS:
        raw = {cid: episode.candidates[cid].raw_attrs for cid in survivor_ids}
        gold_value = raw[gold].get(attribute)
        gold_key = canonical_value(gold_value) if attribute in raw[gold] else None
        kept = [cid for cid in survivor_ids
                if (canonical_value(raw[cid][attribute]) if attribute in raw[cid] else None) == gold_key]
        return kept, gold_value

    if attribute in SPATIAL_BIN_ATTRS:
        axis = 0 if attribute == "left_right" else 1
        low, high = rank_split([(cid, episode.scene.bbox_center(cid)[axis]) for cid in survivor_ids])
        return (low, "low") if gold in low else (high, "high")

    if attribute in SPATIAL_GRAPH_ATTRS:
        ids = set(survivor_ids)
        neighbours = episode.scene.relationships.get(attribute, {})
        has = {cid for cid in survivor_ids if any(n in ids for n in neighbours.get(cid, []))}
        in_has = gold in has
        return list(has if in_has else ids - has), in_has

    raise ValueError(f"unknown ask attribute: {attribute!r}")


def is_missing_value(value: object) -> bool:
    """None, a blank string, or a list of only blanks: nothing to tell."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, list):
        return all(is_missing_value(v) for v in value)
    return False


def phrase_answer(attribute: str, value: object) -> str:
    if attribute == UNRESOLVED_ASK_ATTRIBUTE:
        return UNRESOLVED_REPLY
    if attribute in SPATIAL_BIN_ATTRS:
        low, high = SPATIAL_BIN_PHRASES[attribute]
        return low if value == "low" else high
    if attribute in SPATIAL_GRAPH_ATTRS:
        return f"Yes, it's {attribute} of another one." if value else f"No, it's not {attribute} of another one."
    if is_missing_value(value):
        return MISSING_REPLY
    if isinstance(value, list):
        return f"It's {', '.join(str(v).strip() for v in value if not is_missing_value(v))}."
    return f"It's {value}."


class ScriptedUserSimulator:
    """Template answers. Deterministic, and it only ever reveals the attribute asked about."""

    def answer(self, episode: Episode, attribute: str, survivor_ids: list[int]) -> tuple[str, list[int]]:
        kept, value = apply_attribute_reveal(episode, attribute, survivor_ids)
        return phrase_answer(attribute, value), kept
