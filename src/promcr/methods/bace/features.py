"""The scalar features BACE's chooser reads next to the critic's hidden state.

phi(B), shared by both branches, all in [0, 1], with b = |B|, n candidates,
k questions used of budget K and A the attributes asked:
    [b/n, log b / log n, 1/b, k/K, 1 - k/K, 1[b = 1], |A| / 11, exact flag]
psi(ask), for the question's attribute a:
    [entropy a is expected to leave in B (normalised), 1[a visual], 1[a already asked]]
psi(answer), for the named object c:
    [1[c in B], 1/b, 0]

The expected entropy is computed under the belief's own narrowing rules
(psi0 "belief") or from the listed values only ("permitted", which gives
spatial questions no gain). A checkpoint records its settings in
`bace_features.json`, and evaluation computes the features the same way.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from ...data.episodes import Episode
from ...data.simmc import NON_VISUAL_PERMITTED, VISUAL_FORBIDDEN
from ...env.attributes import SPATIAL_BIN_ATTRS, SPATIAL_GRAPH_ATTRS
from .belief import BeliefState, _log2, belief_information_gain, listed_information_gain

N_BELIEF_FEATURES = 8
N_BRANCH_FEATURES = 3
ASKABLE_ATTRIBUTES = len(NON_VISUAL_PERMITTED) + len(SPATIAL_BIN_ATTRS) + len(SPATIAL_GRAPH_ATTRS)
FEATURE_FILE = "bace_features.json"


@dataclass(frozen=True)
class FeatureSpec:
    psi0: str = "belief"  # "belief" | "permitted"
    sizes: str = "exact"  # how availableSizes narrows the belief: "exact" | "overlap"
    exact_feature: bool = False  # whether phi's last slot carries the exact flag (else pinned to 0)
    asked_norm: float = float(ASKABLE_ATTRIBUTES)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        record = {"psi0_mode": self.psi0, "sizes_narrow": self.sizes, "exact_feature": self.exact_feature,
                  "asked_norm": self.asked_norm}
        (directory / FEATURE_FILE).write_text(json.dumps(record, indent=2) + "\n")

    @classmethod
    def load(cls, directory: str | Path) -> "FeatureSpec":
        """Missing keys take the settings of the checkpoints that predate them."""
        path = Path(directory) / FEATURE_FILE
        record = json.loads(path.read_text()) if path.exists() else {}
        return cls(psi0=record.get("psi0_mode", "permitted"), sizes=record.get("sizes_narrow", "overlap"),
                   exact_feature=bool(record.get("exact_feature", True)),
                   asked_norm=float(record.get("asked_norm", 16.0)))


def belief_features(belief: BeliefState, n_candidates: int, budget: int, spec: FeatureSpec) -> list[float]:
    b = len(belief.ids)
    log_n = _log2(max(n_candidates, 1))
    return [
        b / max(n_candidates, 1),
        (_log2(b) / log_n) if log_n > 0 else 0.0,
        1.0 / max(b, 1),
        belief.asks_used / max(budget, 1),
        max(budget - belief.asks_used, 0) / max(budget, 1),
        1.0 if b == 1 else 0.0,
        len(belief.asked) / spec.asked_norm,
        1.0 if belief.exact and spec.exact_feature else 0.0,
    ]


def ask_features(episode: Episode, belief: BeliefState, attribute: str, n_candidates: int, spec: FeatureSpec) -> list[float]:
    ids = list(belief.ids)
    normaliser = _log2(max(n_candidates, 2))
    if spec.psi0 == "belief":
        gain = belief_information_gain(episode, belief, attribute)
    else:
        gain = listed_information_gain(episode, ids, attribute)
    remaining = max(_log2(len(ids)) - gain, 0.0)
    return [min(remaining / normaliser, 1.0) if normaliser > 0 else 0.0,
            1.0 if attribute in VISUAL_FORBIDDEN else 0.0,
            1.0 if attribute in belief.asked else 0.0]


def answer_features(belief: BeliefState, answer_id: int) -> list[float]:
    return [1.0 if answer_id in belief.ids else 0.0, 1.0 / max(len(belief.ids), 1), 0.0]


def annotate(observation: str, belief: BeliefState) -> str:
    """The observation with the belief appended, as the chooser's prompt shows it."""
    return f"{observation}\n\nStill possible ({len(belief.ids)}): {list(belief.ids)}"


def q_input(hidden: torch.Tensor, shared: list[float], branch: list[float]) -> torch.Tensor:
    return torch.cat([hidden, torch.tensor(list(shared) + list(branch), dtype=hidden.dtype, device=hidden.device)])


def v_input(hidden: torch.Tensor, shared: list[float]) -> torch.Tensor:
    return torch.cat([hidden, torch.tensor(list(shared), dtype=hidden.dtype, device=hidden.device)])
