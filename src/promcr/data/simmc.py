"""Readers for the raw SIMMC 2.1 files: dialogues, scenes and the prefab metadata."""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..config import SIMMC_DIR

# Object metadata a text-only policy may see. The visual keys are withheld,
# following SIMMC's rule against visual metadata at inference time.
NON_VISUAL_PERMITTED = frozenset({"brand", "price", "customerReview", "availableSizes", "size"})
VISUAL_FORBIDDEN = frozenset({"color", "pattern", "type", "assetType", "sleeveLength"})

RELATIONS = ("left", "right", "up", "down")


@dataclass
class SceneObject:
    index: int
    unique_id: int
    prefab_path: str
    bbox: list[int]


@dataclass
class Scene:
    objects: dict[int, SceneObject]  # keyed by scene-local index
    relationships: dict[str, dict[int, list[int]]]

    @property
    def index_set(self) -> set[int]:
        return set(self.objects)

    def bbox_center(self, index: int) -> tuple[float, float]:
        x, y, h, w = self.objects[index].bbox
        return (x + w / 2.0, y + h / 2.0)


@dataclass
class Turn:
    turn_idx: int
    transcript: str
    system_transcript: str
    slot_values: dict[str, Any]
    request_slots: list[str]
    object_local: list[int]
    system_object_local: list[int]
    disambig_label: int
    disambig_candidates: list[int]


@dataclass
class Dialogue:
    idx: int
    domain: str
    scene_ids: dict[int, str]  # first turn index -> scene id
    turns: list[Turn] = field(default_factory=list)


def dialogues_path(split: str, data_dir: Path = SIMMC_DIR) -> Path:
    return Path(data_dir) / f"simmc2.1_dials_dstc11_{split}.json"


def load_dialogues(split: str, data_dir: Path = SIMMC_DIR) -> list[Dialogue]:
    with dialogues_path(split, data_dir).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    dialogues = []
    for raw in payload["dialogue_data"]:
        turns = []
        for turn in raw["dialogue"]:
            user = turn["transcript_annotated"]
            user_attrs = user.get("act_attributes", {})
            system_attrs = turn.get("system_transcript_annotated", {}).get("act_attributes", {})
            turns.append(
                Turn(
                    turn_idx=int(turn["turn_idx"]),
                    transcript=str(turn["transcript"]),
                    system_transcript=str(turn.get("system_transcript", "")),
                    slot_values=dict(user_attrs.get("slot_values", {})),
                    request_slots=list(user_attrs.get("request_slots", [])),
                    object_local=[int(v) for v in user_attrs.get("objects", [])],
                    system_object_local=[int(v) for v in system_attrs.get("objects", [])],
                    disambig_label=int(user.get("disambiguation_label", 0)),
                    disambig_candidates=[int(v) for v in user.get("disambiguation_candidates", [])],
                )
            )
        dialogues.append(
            Dialogue(
                idx=int(raw["dialogue_idx"]),
                domain=str(raw.get("domain", raw.get("domains", ""))),
                scene_ids={int(k): str(v) for k, v in raw["scene_ids"].items()},
                turns=turns,
            )
        )
    return dialogues


def scene_for_turn(dialogue: Dialogue, turn_idx: int) -> str:
    """The scene active at `turn_idx`: the last one bound at or before it."""
    eligible = [scene_id for start, scene_id in sorted(dialogue.scene_ids.items()) if start <= turn_idx]
    assert eligible, f"no scene bound for dialogue {dialogue.idx} turn {turn_idx}"
    return eligible[-1]


@lru_cache(maxsize=None)
def load_metadata(domain: str, data_dir: Path = SIMMC_DIR) -> dict[str, dict[str, Any]]:
    """Prefab path -> metadata for one domain ("fashion" or "furniture")."""
    with (Path(data_dir) / f"{domain}_prefab_metadata_all.json").open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        prefab: dict(value["metadata"]) if isinstance(value, dict) and isinstance(value.get("metadata"), dict) else dict(value)
        for prefab, value in payload.items()
    }


def object_metadata(scene: Scene, index: int, domain: str, data_dir: Path = SIMMC_DIR) -> dict[str, Any]:
    return load_metadata(domain, data_dir).get(scene.objects[index].prefab_path) or {}


def permitted(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metadata.items() if key in NON_VISUAL_PERMITTED}


def _parse_scene(payload: dict[str, Any]) -> Scene:
    data = payload["scenes"][0]
    objects: dict[int, SceneObject] = {}
    for raw in data["objects"]:
        obj = SceneObject(int(raw["index"]), int(raw["unique_id"]), str(raw["prefab_path"]), list(raw["bbox"]))
        assert obj.index not in objects, f"duplicate local index {obj.index}"
        objects[obj.index] = obj
    relationships: dict[str, dict[int, list[int]]] = {relation: {} for relation in RELATIONS}
    for relation, mapping in data.get("relationships", {}).items():
        assert relation in RELATIONS, f"unexpected relation {relation}"
        relationships[relation] = {int(k): [int(n) for n in v] for k, v in mapping.items()}
    return Scene(objects=objects, relationships=relationships)


@lru_cache(maxsize=8)
def _open_zip(path: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(path)


def load_scene(scene_id: str, data_dir: Path = SIMMC_DIR) -> Scene | None:
    """A scene file, read from `data_dir` or from inside the scene-json zips.

    `m_`-prefixed scene ids are second-camera views; the file may be stored
    with or without the prefix.
    """
    data_dir = Path(data_dir)
    stripped = scene_id[2:] if scene_id.startswith("m_") else scene_id
    for path in (data_dir / f"{scene_id}_scene.json", data_dir / f"{stripped}_scene.json"):
        if path.exists():
            return _parse_scene(json.loads(path.read_text(encoding="utf-8")))
    members = [f"{folder}/{name}_scene.json" for folder in ("public", "simmc2_scene_jsons_dstc10_teststd")
               for name in (scene_id, stripped)]
    for zip_path in sorted(data_dir.glob("*.zip")):
        archive = _open_zip(str(zip_path))
        names = set(archive.namelist())
        for member in members:
            if member in names:
                return _parse_scene(json.loads(archive.read(member).decode("utf-8")))
    return None


def load_scenes(dialogues: list[Dialogue], data_dir: Path = SIMMC_DIR) -> dict[str, Scene]:
    scenes: dict[str, Scene] = {}
    for dialogue in dialogues:
        for scene_id in set(dialogue.scene_ids.values()):
            if scene_id not in scenes and (scene := load_scene(scene_id, data_dir)) is not None:
                scenes[scene_id] = scene
    return scenes
