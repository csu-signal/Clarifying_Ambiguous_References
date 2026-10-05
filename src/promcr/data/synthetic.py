"""Anchored synthetic episodes, which fill the ask-depth buckets the real data barely covers.

Each synthetic episode starts from a real irreducible turn with at least
three annotated candidates (its template). It keeps the template's scene,
history and utterance, replaces the candidate list with a strict subset of at
least two of the annotated candidates, and takes any member of that subset as
the gold referent; annotators marked every candidate as something the
utterance could mean. Nothing else is invented. The oracle then gives the
episode its depth.

A draw is rejected when the utterance no longer fits the subset (it names a
type, colour or pattern a kept candidate lacks, or states a count), when the
same subset of the same template was already kept, or when the template has
already produced `max_per_template` episodes at that depth. Depths 1-3 are
filled until each, real and synthetic together, is as large as the real
depth-0 bucket.

    python -m promcr.data.build synthetic
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..config import LABELS_DIR, SIMMC_DIR
from ..env.oracle import build_gold_chain, is_solvable, ranked_attributes
from .episodes import Episode, load_episodes, make_candidate, restrict_relations
from .simmc import load_scene

SUPPORTED_DEPTHS = (0, 1, 2, 3)
EPS = 1e-9

# Utterance descriptors, mapped to the metadata `type` values they can name.
_TYPE_WORDS: dict[str, frozenset[str]] = {
    "jacket": frozenset({"jacket"}),
    "coat": frozenset({"coat"}),
    "blouse": frozenset({"blouse"}),
    "dress": frozenset({"dress"}),
    "sweater": frozenset({"sweater"}),
    "jumper": frozenset({"sweater"}),
    "hoodie": frozenset({"hoodie"}),
    "jeans": frozenset({"jeans"}),
    "trousers": frozenset({"trousers"}),
    "pants": frozenset({"trousers", "jeans", "joggers"}),
    "tshirt": frozenset({"tshirt"}),
    "tee": frozenset({"tshirt"}),
    "shirt": frozenset({"shirt", "shirt, vest"}),
    "tank top": frozenset({"tank top"}),
    "hat": frozenset({"hat"}),
    "shoe": frozenset({"shoes"}),
    "shoes": frozenset({"shoes"}),
    "sneakers": frozenset({"shoes"}),
    "joggers": frozenset({"joggers"}),
    "vest": frozenset({"vest", "shirt, vest"}),
    "skirt": frozenset({"skirt"}),
    "suit": frozenset({"suit"}),
    "sofa": frozenset({"Sofa"}),
    "couch": frozenset({"Sofa", "CouchChair"}),
    "armchair": frozenset({"CouchChair"}),
    "chair": frozenset({"Chair", "CouchChair"}),
    "rug": frozenset({"AreaRug"}),
    "lamp": frozenset({"Lamp"}),
    "bed": frozenset({"Bed"}),
    "shelf": frozenset({"Shelves"}),
    "shelves": frozenset({"Shelves"}),
    "coffee table": frozenset({"CoffeeTable"}),
    "end table": frozenset({"EndTable"}),
    "side table": frozenset({"EndTable"}),
    "table": frozenset({"Table", "EndTable", "CoffeeTable"}),
}
# Matched as substrings of the metadata `color` and `pattern` values.
_COLOR_WORDS = ("black", "white", "grey", "brown", "blue", "green", "purple", "yellow", "red", "pink",
                "orange", "beige", "olive", "maroon", "violet", "wooden")
_COLOR_ALIASES = {"gray": "grey", "wood": "wooden"}
_PATTERN_WORDS = {"striped": "stripe", "stripes": "stripe", "stripe": "stripe", "spotted": "spots",
                  "spots": "spots", "dots": "spots", "denim": "denim", "plaid": "check", "checkered": "check"}
_SPATIAL_RE = re.compile(
    r"\b(left|right|front|back|top|bottom|middle|center|centre|corner|wall|rack|behind|next to|beside|"
    r"above|below|under|closest|nearest|furthest|farthest|far)\b"
)
# A count word pins the number of candidates, which a subset would contradict.
_COUNT_RE = re.compile(r"\b(both|two|three|four|five|six|pair of|either|neither)\b")


@dataclass(frozen=True)
class UtteranceConstraints:
    """What an utterance says about its referent; an empty field constrains nothing."""

    types: frozenset[str]
    colors: frozenset[str]
    patterns: frozenset[str]


def utterance_constraints(utterance: str) -> UtteranceConstraints | None:
    """Type, colour and pattern words in `utterance`; None when it has a spatial phrase."""
    text = utterance.lower()
    text = re.sub(r"\ball right\b|\bright now\b|\bthat's right\b", " ", text)
    if _SPATIAL_RE.search(text):
        return None
    text = re.sub(r"\bt-?\s?shirts?\b", "tshirt", text)
    types: set[str] = set()
    for word in sorted(_TYPE_WORDS, key=len, reverse=True):  # "coffee table" before "table"
        pattern = r"\b" + re.escape(word) + r"s?\b"
        if re.search(pattern, text):
            types |= _TYPE_WORDS[word]
            text = re.sub(pattern, " ", text)
    tokens = re.findall(r"[a-z]+", text)
    colors = {_COLOR_ALIASES.get(t, t) for t in tokens if _COLOR_ALIASES.get(t, t) in _COLOR_WORDS}
    patterns = {_PATTERN_WORDS[t] for t in tokens if t in _PATTERN_WORDS}
    return UtteranceConstraints(frozenset(types), frozenset(colors), frozenset(patterns))


def object_fits(raw_attrs: dict[str, Any], constraints: UtteranceConstraints) -> bool:
    """Whether an object's metadata satisfies every descriptor class the utterance uses."""
    if constraints.types and raw_attrs.get("type") not in constraints.types:
        return False
    color = str(raw_attrs.get("color") or "").lower()
    if constraints.colors and not any(c in color for c in constraints.colors):
        return False
    pattern = str(raw_attrs.get("pattern") or "").lower()
    if constraints.patterns and not any(p in pattern for p in constraints.patterns):
        return False
    return True


def depth_targets(split: str, data_dir: Path = SIMMC_DIR, labels_dir: Path = LABELS_DIR) -> dict[int, int]:
    """Synthetic episodes needed per depth so depths 0-3 match the largest real bucket."""
    real = Counter(
        len(build_gold_chain(e, 10).steps) if e.condition == "irreducible" else 0
        for e in load_episodes(split, data_dir, labels_dir)
        if is_solvable(e)  # the training pool drops the rest
    )
    ceiling = max(real.get(depth, 0) for depth in SUPPORTED_DEPTHS)
    return {depth: max(0, ceiling - real.get(depth, 0)) for depth in SUPPORTED_DEPTHS}


def generate_anchored_episodes(
    split: str,
    targets: dict[int, int],
    seed: int = 0,
    max_attempts: int = 2_000_000,
    max_per_template: int = 20,
    data_dir: Path = SIMMC_DIR,
    labels_dir: Path = LABELS_DIR,
) -> tuple[list[Episode], list[tuple[int, int]], dict[str, int]]:
    """(episodes, the template key of each, rejection counts)."""
    templates = [
        e for e in load_episodes(split, data_dir, labels_dir)
        if e.condition == "irreducible" and len(set(e.candidate_ids) & set(e.survivor_ids)) >= 3
    ]
    rng = random.Random(seed)
    remaining = {depth: count for depth, count in targets.items() if count > 0 and depth > 0}
    walk_budget = max(remaining) if remaining else 0
    seen: set[tuple[int, int, tuple[int, ...]]] = set()
    per_template: Counter = Counter()
    rejects: Counter = Counter()
    episodes: list[Episode] = []
    sources: list[tuple[int, int]] = []
    attempts = 0

    while remaining and attempts < max_attempts:
        attempts += 1
        template = rng.choice(templates)
        key = (template.dialogue_idx, template.turn_idx)
        if _COUNT_RE.search(template.user_turn.lower()):
            rejects["count_word"] += 1
            continue
        pool = sorted(set(template.candidate_ids) & set(template.survivor_ids))
        subset = sorted(rng.sample(pool, rng.randint(2, len(pool) - 1)))
        gold = rng.choice(subset)
        if (*key, tuple(subset)) in seen:
            rejects["duplicate"] += 1
            continue
        constraints = utterance_constraints(template.user_turn)
        if constraints is not None and not all(object_fits(template.candidates[c].raw_attrs, constraints) for c in subset):
            rejects["descriptor_misfit"] += 1
            continue

        episode = replace(
            template,
            dialogue_idx=-(len(episodes) + 1),  # negative, so never equal to a real key
            turn_idx=0,
            candidate_ids=subset,
            survivor_ids=list(subset),
            candidates={c: template.candidates[c] for c in subset},
            relations=restrict_relations(template.scene, subset),
            gold_referent=gold,
            referent_source="anchored_subset",
            synthetic=True,
        )
        ranked = ranked_attributes(episode, subset, exclude=set(episode.request_slots))
        if not ranked:
            rejects["no_splitter"] += 1
            continue
        episode.gold_splitting_attributes = [e.attribute for e in ranked if abs(e.ig - ranked[0].ig) < EPS]
        chain = build_gold_chain(episode, walk_budget)
        depth = len(chain.steps)
        if not chain.resolved or remaining.get(depth, 0) == 0:
            rejects["depth_full_or_unresolved"] += 1
            continue
        if per_template[(*key, depth)] >= max_per_template:
            rejects["template_cap"] += 1
            continue

        seen.add((*key, tuple(subset)))
        episodes.append(episode)
        sources.append(key)
        per_template[(*key, depth)] += 1
        remaining[depth] -= 1
        if remaining[depth] == 0:
            del remaining[depth]

    rejects["attempts"] = attempts
    rejects["templates"] = len(templates)
    rejects["templates_used"] = len({k[:2] for k in per_template})
    return episodes, sources, dict(rejects)


def synthetic_path(split: str, labels_dir: Path = LABELS_DIR) -> Path:
    return Path(labels_dir) / f"synthetic_anchored_{split}.jsonl"


def write_synthetic(split: str, episodes: list[Episode], sources: list[tuple[int, int]],
                    labels_dir: Path = LABELS_DIR) -> Path:
    path = synthetic_path(split, labels_dir)
    with path.open("w", encoding="utf-8") as handle:
        for episode, source in zip(episodes, sources):
            record = {
                "dialogue_idx": episode.dialogue_idx,
                "turn_idx": episode.turn_idx,
                "scene_id": episode.scene_id,
                "domain": episode.domain,
                "condition": episode.condition,
                "C": episode.candidate_ids,
                "gold_referent": episode.gold_referent,
                "referent_source": episode.referent_source,
                "user_turn": episode.user_turn,
                "gold_splitting_attributes": episode.gold_splitting_attributes,
                "request_slots": sorted(episode.request_slots),
                "S": episode.survivor_ids,
                "history": [list(pair) for pair in episode.history],
                "established_attrs": episode.established_attrs,
                "source": list(source),
            }
            handle.write(json.dumps(record) + "\n")
    return path


def load_synthetic_episodes(split: str, data_dir: Path = SIMMC_DIR, labels_dir: Path = LABELS_DIR) -> list[Episode]:
    path = synthetic_path(split, labels_dir)
    if not path.exists():
        return []
    scenes: dict = {}
    episodes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["scene_id"] not in scenes:
            scenes[row["scene_id"]] = load_scene(row["scene_id"], data_dir)
        scene = scenes[row["scene_id"]]
        if scene is None:
            continue
        candidate_ids = list(row["C"])
        episodes.append(
            Episode(
                dialogue_idx=row["dialogue_idx"],
                turn_idx=row["turn_idx"],
                scene_id=row["scene_id"],
                domain=row["domain"],
                condition=row["condition"],
                candidate_ids=candidate_ids,
                survivor_ids=list(row["S"]),
                candidates={cid: make_candidate(scene, cid, row["domain"], data_dir) for cid in candidate_ids},
                relations=restrict_relations(scene, candidate_ids),
                history=[(speaker, text) for speaker, text in row["history"]],
                user_turn=row["user_turn"],
                gold_referent=row["gold_referent"],
                referent_source=row["referent_source"],
                gold_splitting_attributes=list(row["gold_splitting_attributes"]),
                scene=scene,
                established_attrs=dict(row["established_attrs"]),
                synthetic=True,
                request_slots=frozenset(row["request_slots"]),
            )
        )
    return episodes
