"""ArCHer at evaluation: greedy branches, and the branch with the higher min(Q_a, Q_b)."""

from __future__ import annotations

from pathlib import Path

import torch

from ...data.episodes import Episode
from ...env.actions import Action, Answer, Ask
from ...models.backbone import Backbone, load_two_adapters
from ...models.generation import sample_completions
from ...policies.contract import parse_referent, resolve_ask
from ..chooser import ANSWER_PREFIX, branch_prompt, decision_probs
from .heads import load_head
from .rollout import branch_q, sample_constrained_ask


class ArcherPolicy:
    def __init__(self, model, tokenizer, device: str, q_a, q_b, budget: int, max_new_tokens: int = 64) -> None:
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.q_a, self.q_b = q_a, q_b
        self.budget = budget
        self.max_new_tokens = max_new_tokens
        self.backbone = Backbone(model, tokenizer, device)  # the actor, for the evaluation probe
        self.last_decision_q: dict[str, list[float]] | None = None

    def act(self, episode: Episode, observation: str, candidate_ids: list[int], asks_used: int) -> Action:
        model, tokenizer, device = self.model, self.tokenizer, self.device
        state_prompt = branch_prompt(tokenizer, observation)
        model.set_adapter("actor")
        ask_text, ask_ids = sample_constrained_ask(model, tokenizer, device, state_prompt, 0.0, self.max_new_tokens)
        answer_text, answer_ids = sample_completions(model, tokenizer, device, state_prompt + ANSWER_PREFIX, 1,
                                                     self.max_new_tokens, 0.0)[0]
        q_a, q_b = branch_q(model, tokenizer, device, self.q_a, self.q_b, state_prompt, ask_ids, answer_ids)
        self.last_decision_q = {"ask": [q_a[0].item(), q_b[0].item()], "answer": [q_a[1].item(), q_b[1].item()]}
        probs = decision_probs(torch.min(q_a, q_b), 0.0, (asks_used < self.budget, True))
        if int(torch.argmax(probs).item()) == 0:
            return Ask(*resolve_ask(ask_text))
        return Answer(parse_referent(answer_text, candidate_ids))


def load_archer_policy(checkpoint: str | Path, budget: int, device: str = "cuda:0") -> ArcherPolicy:
    model, tokenizer = load_two_adapters(checkpoint, device)
    checkpoint = Path(checkpoint)
    return ArcherPolicy(model, tokenizer, device, load_head(checkpoint / "q_a_head", device),
                        load_head(checkpoint / "q_b_head", device), budget)
