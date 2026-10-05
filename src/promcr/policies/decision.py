"""The policy for checkpoints that write the decision line themselves (SFT and GRPO)."""

from __future__ import annotations

from pathlib import Path

from ..data.episodes import Episode
from ..env.actions import Action
from ..models.backbone import Backbone, load_merged_adapter
from .contract import DECISION_INSTRUCTION, parse_decision


class DecisionPolicy:
    def __init__(self, backbone: Backbone) -> None:
        self.backbone = backbone

    def act(self, episode: Episode, observation: str, candidate_ids: list[int], asks_used: int) -> Action:
        text = self.backbone.chat(f"{DECISION_INSTRUCTION}\n\n{observation}", max_new_tokens=64)
        return parse_decision(text, candidate_ids)


def load_decision_policy(checkpoint: str | Path, device: str = "cuda:0") -> DecisionPolicy:
    return DecisionPolicy(load_merged_adapter(checkpoint, device))
