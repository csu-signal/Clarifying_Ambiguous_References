"""Condition labels: which turns are episodes, their candidates, survivors and gold referent.

Two streams of turns become episodes:

- unambiguous: `disambiguation_label == 0` turns that annotate exactly one
  referent. That object is the only candidate.
- ambiguous: `disambiguation_label == 1` turns with their annotated
  candidates. The dialogue history can rule some of them out; the ones still
  consistent with it are the survivors. Two or more survivors make the turn
  `irreducible` (the policy has to ask), one makes it `recoverable`, none
  makes it `over_constrained` (dropped later), and fewer than two candidates
  make it `excluded_trivial`.

History constraints are the permitted slot values the user stated, carried
forward only while the dialogue stays on the same reference: the accumulator
is cleared whenever the set of objects a turn is about changes.

SIMMC doesn't say which candidate the user meant, so the gold referent comes
from the first later turn in the same scene whose objects form a strict
subset of the candidates: its single object, or the first of two. When no
later turn narrows the set, one candidate is drawn with a seed derived from
the turn.

    python -m promcr.data.build conditions
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import LABELS_DIR, SIMMC_DIR
from .simmc import (
    NON_VISUAL_PERMITTED,
    Dialogue,
    Scene,
    Turn,
    load_dialogues,
    load_scenes,
    object_metadata,
    permitted,
    scene_for_turn,
)


@dataclass
class LabeledTurn:
    dialogue_idx: int
    turn_idx: int
    scene_id: str
    condition: str
    candidate_ids: list[int]
    survivor_ids: list[int]
    n_constraints_applied: int
    gold_referent: int | None
    referent_source: str | None

    def to_json(self) -> dict[str, Any]:
        return {
            "dialogue_idx": self.dialogue_idx,
            "turn_idx": self.turn_idx,
            "active_scene_id": self.scene_id,
            "condition": self.condition,
            "C": self.candidate_ids,
            "S": self.survivor_ids,
            "n_permitted_constraints_applied": self.n_constraints_applied,
            "gold_referent": self.gold_referent,
            "referent_source": self.referent_source,
            "splitting_attribute": None,
        }


def _reference_signature(scene_id: str, turn: Turn) -> tuple[str, frozenset[int]] | None:
    """What a turn is about: its candidate set if it is an ambiguous mention,
    else the objects it resolves to. None for turns about nothing."""
    if turn.disambig_label == 1 and turn.disambig_candidates:
        return (scene_id, frozenset(turn.disambig_candidates))
    if turn.object_local:
        return (scene_id, frozenset(turn.object_local))
    if turn.system_object_local:
        return (scene_id, frozenset(turn.system_object_local))
    return None


def history_constraints(dialogue: Dialogue, turn_position: int) -> dict[str, Any]:
    """Permitted slot values stated up to and including `turn_position`,
    cleared whenever the reference signature changes."""
    constraints: dict[str, Any] = {}
    last_signature = None
    for turn in dialogue.turns[: turn_position + 1]:
        signature = _reference_signature(scene_for_turn(dialogue, turn.turn_idx), turn)
        if signature is not None:
            if last_signature is not None and signature != last_signature:
                constraints = {}
            last_signature = signature
        for key, value in turn.slot_values.items():
            if key in NON_VISUAL_PERMITTED:
                constraints[key] = value
    return constraints


def _values_match(candidate_value: Any, constraint_value: Any) -> bool:
    if isinstance(candidate_value, list) or isinstance(constraint_value, list):
        a = candidate_value if isinstance(candidate_value, list) else [candidate_value]
        b = constraint_value if isinstance(constraint_value, list) else [constraint_value]
        return set(a) == set(b)
    return candidate_value == constraint_value


def _comparable(candidate_value: Any, constraint_value: Any) -> bool:
    """Whether a stated value can be checked against catalog metadata. A
    qualitative slot value ("affordable") against a numeric price can't, so
    that constraint is skipped rather than failing every candidate."""
    if isinstance(candidate_value, list) or isinstance(constraint_value, list):
        return True
    if isinstance(candidate_value, bool) or isinstance(constraint_value, bool):
        return isinstance(candidate_value, bool) and isinstance(constraint_value, bool)
    if isinstance(candidate_value, (int, float)) and isinstance(constraint_value, (int, float)):
        return True
    return isinstance(candidate_value, str) and isinstance(constraint_value, str)


def survivors(
    dialogue: Dialogue, turn_position: int, scene: Scene, candidate_ids: list[int]
) -> tuple[list[int], int]:
    """(candidates consistent with the history constraints, constraints applied)."""
    attrs = {cid: permitted(object_metadata(scene, cid, dialogue.domain)) for cid in candidate_ids}
    constraints = history_constraints(dialogue, turn_position)
    kept = list(candidate_ids)
    n_applied = 0
    for key in sorted(constraints):
        value = constraints[key]
        checkable = [cid for cid in candidate_ids if key in attrs[cid] and _comparable(attrs[cid][key], value)]
        if not checkable:
            continue
        n_applied += 1
        kept = [cid for cid in kept if cid in checkable and _values_match(attrs[cid][key], value)]
    return kept, n_applied


def _mine_referent(dialogue: Dialogue, turn_position: int, scene_id: str, candidate_ids: list[int]):
    """(referent, source) from the first later same-scene turn whose objects
    are a strict subset of the candidates; (None, None) if that subset has
    more than two objects or no such turn exists."""
    candidates = set(candidate_ids)
    for later in dialogue.turns[turn_position + 1 :]:
        if scene_for_turn(dialogue, later.turn_idx) != scene_id:
            continue
        objects = later.object_local or later.system_object_local
        if not objects or set(objects) == candidates or not set(objects) <= candidates:
            continue
        if len(objects) == 1:
            return objects[0], "followup_singleton"
        if len(objects) == 2:
            return objects[0], "narrowed_pair"
        return None, None
    return None, None


def _simulated_referent(dialogue_idx: int, turn_idx: int, candidate_ids: list[int]) -> int:
    return random.Random(dialogue_idx * 1_000_003 + turn_idx).choice(sorted(candidate_ids))


def _classify(candidate_ids: list[int], survivor_ids: list[int], gold: int, source: str) -> str:
    survivor_set = set(survivor_ids)
    if not survivor_set:
        return "over_constrained"
    if len(survivor_set) > 1:
        return "irreducible"
    if source == "narrowed_pair" or survivor_set == {gold}:
        return "recoverable"
    # One survivor that isn't the referent: the context misleads, so the
    # policy still has to ask. load_episodes drops these (gold not in S).
    return "irreducible"


def label_split(split: str, data_dir: Path = SIMMC_DIR) -> list[LabeledTurn]:
    dialogues = load_dialogues(split, data_dir)
    scenes = load_scenes(dialogues, data_dir)
    unambiguous: list[LabeledTurn] = []
    ambiguous: list[LabeledTurn] = []
    for dialogue in dialogues:
        for position, turn in enumerate(dialogue.turns):
            scene_id = scene_for_turn(dialogue, turn.turn_idx)
            if turn.disambig_label != 1:
                if len(turn.object_local) == 1:
                    gold = turn.object_local[0]
                    unambiguous.append(
                        LabeledTurn(dialogue.idx, turn.turn_idx, scene_id, "unambiguous", [gold], [gold], 0, gold, None)
                    )
                continue

            scene = scenes.get(scene_id)
            if scene is None:
                candidate_ids, survivor_ids, n_applied = [], [], 0
            else:
                candidate_ids = [c for c in turn.disambig_candidates if c in scene.index_set]
                survivor_ids, n_applied = survivors(dialogue, position, scene, candidate_ids)
            if len(candidate_ids) <= 1:
                ambiguous.append(LabeledTurn(dialogue.idx, turn.turn_idx, scene_id, "excluded_trivial",
                                             candidate_ids, survivor_ids, n_applied, None, None))
                continue

            gold, source = _mine_referent(dialogue, position, scene_id, candidate_ids)
            if gold is None:
                gold, source = _simulated_referent(dialogue.idx, turn.turn_idx, candidate_ids), "simulated"
            condition = _classify(candidate_ids, survivor_ids, gold, source)
            ambiguous.append(LabeledTurn(dialogue.idx, turn.turn_idx, scene_id, condition,
                                         candidate_ids, survivor_ids, n_applied, gold, source))
    return unambiguous + ambiguous


def write_conditions(split: str, data_dir: Path = SIMMC_DIR, labels_dir: Path = LABELS_DIR) -> Path:
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    path = labels_dir / f"conditions_{split}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in label_split(split, data_dir):
            handle.write(json.dumps(row.to_json()) + "\n")
    return path
