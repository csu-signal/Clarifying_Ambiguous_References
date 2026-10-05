"""Episodes: a labeled turn joined with its dialogue history, scene and candidate metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import LABELS_DIR, SIMMC_DIR
from .conditions import history_constraints
from .simmc import RELATIONS, Scene, load_dialogues, load_scenes, object_metadata, permitted

# Conditions that become episodes. over_constrained and excluded_trivial don't.
CONDITIONS_USED = ("unambiguous", "recoverable", "irreducible")


@dataclass(frozen=True)
class Candidate:
    candidate_id: int
    permitted_attrs: dict[str, Any]  # what the policy's observation shows
    raw_attrs: dict[str, Any]  # full metadata, visual attributes included; environment only


@dataclass
class Episode:
    dialogue_idx: int
    turn_idx: int
    scene_id: str
    domain: str
    condition: str
    candidate_ids: list[int]
    survivor_ids: list[int]
    candidates: dict[int, Candidate]
    relations: dict[str, dict[int, list[int]]]  # scene relations restricted to the candidates
    history: list[tuple[str, str]]  # (speaker, text) for the turns before this one
    user_turn: str
    gold_referent: int | None
    referent_source: str | None
    gold_splitting_attributes: list[str]  # best first question(s); empty unless irreducible
    scene: Scene = field(repr=False)
    established_attrs: dict[str, Any] = field(default_factory=dict)  # permitted values the history already pins
    synthetic: bool = False
    request_slots: frozenset[str] = field(default_factory=frozenset)  # attributes this turn itself asks about

    @property
    def key(self) -> str:
        return f"{self.dialogue_idx}:{self.turn_idx}"

    @property
    def gold_action(self) -> str:
        return "ask" if self.condition == "irreducible" else "answer"

    def render_input(self) -> str:
        """The observation before any clarification: history, utterance, and
        every candidate with its permitted attributes and scene position."""
        lines: list[str] = []
        if self.history:
            lines.append("Dialogue history:")
            lines += [f"{speaker}: {text}" for speaker, text in self.history]
            lines.append("")
        lines += [f"User: {self.user_turn}", "", f"Candidate objects ({len(self.candidate_ids)}):"]
        for cid in self.candidate_ids:
            attrs = dict(self.candidates[cid].permitted_attrs)
            x, y = self.scene.bbox_center(cid)
            attrs["position"] = f"({round(x)}, {round(y)})"
            attr_text = ", ".join(f"{k}={v}" for k, v in sorted(attrs.items())) or "(no permitted attributes)"
            lines.append(f"- object {cid}: {attr_text}")
        for relation in RELATIONS:
            if mapping := self.relations.get(relation, {}):
                lines.append(f"Spatial relation ({relation}): {mapping}")
        return "\n".join(lines)


def restrict_relations(scene: Scene, candidate_ids: list[int]) -> dict[str, dict[int, list[int]]]:
    ids = set(candidate_ids)
    return {
        relation: {
            source: [target for target in targets if target in ids]
            for source, targets in scene.relationships.get(relation, {}).items()
            if source in ids
        }
        for relation in RELATIONS
    }


def make_candidate(scene: Scene, cid: int, domain: str, data_dir: Path = SIMMC_DIR) -> Candidate:
    raw = object_metadata(scene, cid, domain, data_dir)
    return Candidate(candidate_id=cid, permitted_attrs=permitted(raw), raw_attrs=dict(raw))


def history_before(dialogue, turn_idx: int) -> tuple[list[tuple[str, str]], str]:
    """(history, utterance): every user and system line before `turn_idx`, and that turn's own utterance."""
    history: list[tuple[str, str]] = []
    for turn in dialogue.turns:
        if turn.turn_idx < turn_idx:
            history.append(("User", turn.transcript))
            if turn.system_transcript:
                history.append(("Assistant", turn.system_transcript))
        elif turn.turn_idx == turn_idx:
            return history, turn.transcript
    return history, ""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_episodes(split: str, data_dir: Path = SIMMC_DIR, labels_dir: Path = LABELS_DIR) -> list[Episode]:
    """Every usable labeled turn of `split`.

    Two kinds of inconsistent label are skipped: a mined gold referent outside
    its own survivor set (no question could ever isolate it), and an
    unambiguous turn whose annotated object isn't in the active scene.
    """
    rows = _read_jsonl(Path(labels_dir) / f"conditions_{split}.jsonl")
    splitters = {(r["dialogue_idx"], r["turn_idx"]): r for r in _read_jsonl(Path(labels_dir) / f"splitters_{split}.jsonl")}
    dialogues = load_dialogues(split, data_dir)
    by_idx = {d.idx: d for d in dialogues}
    scenes = load_scenes(dialogues, data_dir)

    episodes = []
    for row in rows:
        condition = row["condition"]
        gold = row.get("gold_referent")
        if condition not in CONDITIONS_USED or (gold is not None and gold not in row["S"]):
            continue
        scene = scenes.get(row["active_scene_id"])
        if scene is None or any(cid not in scene.index_set for cid in row["C"]):
            continue
        dialogue = by_idx[row["dialogue_idx"]]
        position = next(i for i, t in enumerate(dialogue.turns) if t.turn_idx == row["turn_idx"])
        splitter = splitters.get((row["dialogue_idx"], row["turn_idx"]))
        history, user_turn = history_before(dialogue, row["turn_idx"])
        episodes.append(
            Episode(
                dialogue_idx=row["dialogue_idx"],
                turn_idx=row["turn_idx"],
                scene_id=row["active_scene_id"],
                domain=dialogue.domain,
                condition=condition,
                candidate_ids=list(row["C"]),
                survivor_ids=list(row["S"]),
                candidates={cid: make_candidate(scene, cid, dialogue.domain, data_dir) for cid in row["C"]},
                relations=restrict_relations(scene, row["C"]),
                history=history,
                user_turn=user_turn,
                gold_referent=gold,
                referent_source=row.get("referent_source"),
                gold_splitting_attributes=(
                    [entry["attribute"] for entry in splitter["best_attributes"]]
                    if condition == "irreducible" and splitter is not None else []
                ),
                scene=scene,
                established_attrs=history_constraints(dialogue, position - 1) if position > 0 else {},
                request_slots=frozenset(dialogue.turns[position].request_slots),
            )
        )
    return episodes
