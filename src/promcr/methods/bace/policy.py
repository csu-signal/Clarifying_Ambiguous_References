"""BACE at evaluation: sampled candidates, log-probability finalists, greedy choice.

The belief is rebuilt from what the policy sees. `act` gets no episode-start
signal, so a new episode is detected by `asks_used == 0`, and each reply is
read back from the "A: " lines of the rendered clarification log.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ...data.episodes import Episode
from ...env.actions import Action, Answer, Ask
from ...env.environment import MUST_ANSWER_NOTICE, REPEATED_ATTRIBUTE_REPLY
from ...models.backbone import Backbone, load_two_adapters
from ..chooser import decision_probs
from .belief import BeliefTracker
from .features import FeatureSpec
from .heads import load_head
from .rollout import chooser_q, prepare_state


class BacePolicy:
    def __init__(self, model, tokenizer, device: str, q_a, q_b, budget: int, spec: FeatureSpec, k: int = 2,
                 temperature: float = 0.8, max_new_tokens: int = 64, max_prompt_tokens: int = 10**9) -> None:
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.q_a, self.q_b = q_a, q_b
        self.budget, self.spec, self.k = budget, spec, k
        self.temperature, self.max_new_tokens, self.max_prompt_tokens = temperature, max_new_tokens, max_prompt_tokens
        self.backbone = Backbone(model, tokenizer, device)  # the generator, for the evaluation probe
        self.last_decision_q: dict[str, list[float]] | None = None
        self._tracker: BeliefTracker | None = None
        self._pending: str | None = None
        self._replies_seen = 0

    def _sync_belief(self, episode: Episode, observation: str, candidate_ids: list[int], asks_used: int) -> None:
        if asks_used == 0 or self._tracker is None:
            self._tracker = BeliefTracker(episode, list(candidate_ids), self.spec.sizes)
            self._pending, self._replies_seen = None, 0
            return
        replies = [line[len("A: "):] for line in observation.splitlines() if line.startswith("A: ")]
        if len(replies) > self._replies_seen and self._pending is not None:
            if replies[-1] == REPEATED_ATTRIBUTE_REPLY:
                self._tracker.note_refusal()
            else:
                self._tracker.observe(self._pending, replies[-1])
            self._pending = None
        self._replies_seen = len(replies)

    def act(self, episode: Episode, observation: str, candidate_ids: list[int], asks_used: int) -> Action:
        self._sync_belief(episode, observation, candidate_ids, asks_used)
        state = prepare_state(self.model, self.tokenizer, self.device, episode, observation, self._tracker.state,
                              self.budget, self.k, self.temperature, self.max_new_tokens, self.max_prompt_tokens,
                              "logprob", None, self.spec)
        q_a, q_b = chooser_q(self.model, self.tokenizer, self.device, self.q_a, self.q_b, state)
        self.last_decision_q = {"ask": [q_a[0].item(), q_b[0].item()], "answer": [q_a[1].item(), q_b[1].item()]}
        available = (asks_used < self.budget and MUST_ANSWER_NOTICE not in observation, True)
        if int(torch.argmax(decision_probs(torch.min(q_a, q_b), 0.0, available)).item()) == 0:
            self._pending = state.attribute
            return Ask(state.attribute, state.question)
        return Answer(state.answer_id)


def load_bace_policy(checkpoint: str | Path, budget: int, device: str = "cuda:0") -> BacePolicy:
    model, tokenizer = load_two_adapters(checkpoint, device)
    checkpoint = Path(checkpoint)
    return BacePolicy(model, tokenizer, device, load_head(checkpoint / "q_a_head", device),
                      load_head(checkpoint / "q_b_head", device), budget, FeatureSpec.load(checkpoint))
