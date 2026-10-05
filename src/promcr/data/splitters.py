"""Splitting-attribute labels: which question best separates an irreducible turn's survivors.

The belief over the survivors S is uniform, so H(S) = log2 |S| and an
attribute's information gain is H(S) minus the size-weighted entropy of the
groups it splits S into. Three kinds of attribute are scored:

- intrinsic: the permitted metadata keys. List values (availableSizes) are
  grouped by set equality; a missing value forms its own group.
- spatial bin (left_right, up_down): a rank-based median split of the bbox
  centres along one axis.
- spatial graph (left, right, up, down): survivors that have a neighbour in S
  in that direction against those that don't. Skipped when one side is empty.

Attributes the turn itself requests ("what size is that?") are excluded from
the best attributes, unless they are the only informative ones.

    python -m promcr.data.build splitters
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import LABELS_DIR, SIMMC_DIR
from .simmc import Scene, load_dialogues, load_scenes

# Fixed order; equal to simmc.NON_VISUAL_PERMITTED.
INTRINSIC_ATTRS = ("brand", "price", "customerReview", "availableSizes", "size")
RELATIONS = ("left", "right", "up", "down")
UNKNOWN = "__unknown__"
EPS = 1e-9


@dataclass
class SplitterEntry:
    attribute: str
    kind: str  # "intrinsic" | "spatial_bin" | "spatial_graph"
    ig: float
    partition_sizes: list[int]
    missing_count: int = 0

    def to_json(self) -> dict[str, Any]:
        record = {"attribute": self.attribute, "kind": self.kind, "IG": round(self.ig, 6),
                  "resulting_partition_sizes": self.partition_sizes}
        if self.kind == "intrinsic":
            record["missing_count"] = self.missing_count
        return record


def _entropy(n: int) -> float:
    return math.log2(n) if n > 0 else 0.0


def _information_gain(h_s: float, s_size: int, group_sizes: list[int]) -> float:
    assert sum(group_sizes) == s_size
    ig = h_s - sum((size / s_size) * _entropy(size) for size in group_sizes)
    return 0.0 if abs(ig) < EPS else ig


def canonical_value(value: Any) -> Any:
    """Grouping key for a metadata value; lists compare as sets."""
    return frozenset(value) if isinstance(value, list) else value


def rank_split(values: list[tuple[int, float]]) -> tuple[list[int], list[int]]:
    """Median split of (id, coordinate) pairs, ties broken by id: (low half, high half)."""
    ordered = sorted(values, key=lambda pair: (pair[1], pair[0]))
    mid = len(ordered) // 2
    return [cid for cid, _ in ordered[:mid]], [cid for cid, _ in ordered[mid:]]


def intrinsic_entries(h_s: float, s_size: int, survivor_ids: list[int],
                      raw_attrs: dict[int, dict[str, Any]]) -> list[SplitterEntry]:
    entries = []
    for attr in INTRINSIC_ATTRS:
        groups: dict[Any, int] = {}
        missing = 0
        for cid in survivor_ids:
            if attr in raw_attrs[cid]:
                key = canonical_value(raw_attrs[cid][attr])
            else:
                key, missing = UNKNOWN, missing + 1
            groups[key] = groups.get(key, 0) + 1
        sizes = sorted(groups.values(), reverse=True)
        entries.append(SplitterEntry(attr, "intrinsic", _information_gain(h_s, s_size, sizes), sizes, missing))
    return entries


def spatial_bin_entries(h_s: float, s_size: int, survivor_ids: list[int], scene: Scene) -> list[SplitterEntry]:
    centers = {cid: scene.bbox_center(cid) for cid in survivor_ids}
    entries = []
    for axis, attribute in ((0, "left_right"), (1, "up_down")):
        low, high = rank_split([(cid, centers[cid][axis]) for cid in survivor_ids])
        sizes = sorted([len(low), len(high)], reverse=True)
        entries.append(SplitterEntry(attribute, "spatial_bin", _information_gain(h_s, s_size, sizes), sizes))
    return entries


def spatial_graph_entries(h_s: float, s_size: int, survivor_ids: list[int], scene: Scene) -> list[SplitterEntry]:
    s_set = set(survivor_ids)
    entries = []
    for relation in RELATIONS:
        neighbours = scene.relationships.get(relation, {})
        has = {cid for cid in survivor_ids if any(n in s_set for n in neighbours.get(cid, []))}
        if not has or not s_set - has:
            continue
        sizes = sorted([len(has), len(s_set - has)], reverse=True)
        entries.append(SplitterEntry(relation, "spatial_graph", _information_gain(h_s, s_size, sizes), sizes))
    return entries


@dataclass
class TurnSplitters:
    dialogue_idx: int
    turn_idx: int
    scene_id: str
    s_size: int
    h_s: float
    referent_source: str | None
    ranked: list[SplitterEntry]
    had_duplicate_ids: bool
    request_slots: frozenset[str] = field(default_factory=frozenset)

    @property
    def request_slots_fallback(self) -> bool:
        """True when only the requested attributes are informative."""
        if not self.request_slots:
            return False
        rest = [e for e in self.ranked if e.attribute not in self.request_slots]
        return not rest or max(e.ig for e in rest) <= EPS

    def _rankable(self) -> list[SplitterEntry]:
        if not self.request_slots or self.request_slots_fallback:
            return self.ranked
        return [e for e in self.ranked if e.attribute not in self.request_slots]

    @property
    def top_ig(self) -> float:
        return max((e.ig for e in self._rankable()), default=0.0)

    @property
    def best_entries(self) -> list[SplitterEntry]:
        return [e for e in self._rankable() if abs(e.ig - self.top_ig) < EPS]

    def to_json(self) -> dict[str, Any]:
        return {
            "dialogue_idx": self.dialogue_idx,
            "turn_idx": self.turn_idx,
            "scene_id": self.scene_id,
            "S_size": self.s_size,
            "H_S": round(self.h_s, 6),
            "ranked_splitters": [e.to_json() for e in self.ranked],
            "top_IG": round(self.top_ig, 6),
            "best_attributes": [{"attribute": e.attribute, "kind": e.kind} for e in self.best_entries],
            "splittable": self.top_ig > EPS,
            "referent_source": self.referent_source,
            "upstream_inconsistency_S_lt_2": self.s_size < 2,
            "had_duplicate_ids_in_committed_S": self.had_duplicate_ids,
            "request_slots": sorted(self.request_slots),
            "request_slots_fallback": self.request_slots_fallback,
        }


def turn_splitters(row: dict[str, Any], scene: Scene, metadata: dict[str, dict[str, Any]],
                   request_slots: frozenset[str]) -> TurnSplitters:
    """Rank every attribute for one irreducible label row. Duplicate ids in
    the annotated S (an artifact of SIMMC's candidate lists) are dropped."""
    survivor_ids = list(dict.fromkeys(row["S"]))
    s_size = len(survivor_ids)
    raw_attrs = {cid: metadata.get(scene.objects[cid].prefab_path) or {} for cid in survivor_ids}
    if s_size < 2:
        ranked = intrinsic_entries(0.0, max(s_size, 1), survivor_ids, raw_attrs) if survivor_ids else []
        h_s = 0.0
    else:
        h_s = math.log2(s_size)
        ranked = (intrinsic_entries(h_s, s_size, survivor_ids, raw_attrs)
                  + spatial_bin_entries(h_s, s_size, survivor_ids, scene)
                  + spatial_graph_entries(h_s, s_size, survivor_ids, scene))
        ranked.sort(key=lambda e: (-e.ig, e.kind, e.attribute))
    return TurnSplitters(row["dialogue_idx"], row["turn_idx"], row["active_scene_id"], s_size, h_s,
                         row["referent_source"], ranked, len(row["S"]) != s_size, request_slots)


def write_splitters(split: str, data_dir: Path = SIMMC_DIR, labels_dir: Path = LABELS_DIR) -> Path:
    from .simmc import load_metadata

    labels_dir = Path(labels_dir)
    dialogues = load_dialogues(split, data_dir)
    domains = {d.idx: d.domain for d in dialogues}
    request_slots = {(d.idx, t.turn_idx): frozenset(t.request_slots) for d in dialogues for t in d.turns}
    scenes = load_scenes(dialogues, data_dir)
    rows = [json.loads(line) for line in (labels_dir / f"conditions_{split}.jsonl").read_text().splitlines()]
    path = labels_dir / f"splitters_{split}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if row["condition"] != "irreducible" or row["active_scene_id"] not in scenes:
                continue
            key = (row["dialogue_idx"], row["turn_idx"])
            metadata = load_metadata(domains.get(row["dialogue_idx"], "fashion"), data_dir)
            result = turn_splitters(row, scenes[row["active_scene_id"]], metadata, request_slots.get(key, frozenset()))
            handle.write(json.dumps(result.to_json()) + "\n")
    return path
