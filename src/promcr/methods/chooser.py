"""Pieces shared by the two critic-chooser methods, ArCHer and BACE.

At every decision the actor writes one ASK and one ANSWER branch, each
continuing a forced prefix, and a critic scores the two. Both methods keep
the critic as a separate LoRA adapter on the same backbone and read its last
hidden state at the end of the branch.
"""

from __future__ import annotations

import random

import torch

from ..env.environment import MCREnv
from ..models.generation import chat_prompt, last_hidden_state
from ..policies.contract import ASK_CONTRACT

ASK_PREFIX = "ASK: "
ANSWER_PREFIX = "ANSWER: "

# The actor continues a prefix the critic already chose, so the instruction
# describes both continuations and no decision.
BRANCH_INSTRUCTION = (
    "You are a shopping assistant helping a customer identify exactly one "
    "object among the candidates listed below.\n\n"
    'Your reply has already been started for you, with either "ASK: " or '
    '"ANSWER: ". Continue that line and stop. Do not repeat the prefix, do '
    "not write the other one, and do not add any explanation.\n\n"
    f'Continuing "ASK: " -- {ASK_CONTRACT}\n'
    "Never resolve the customer's request in place of asking.\n\n"
    'Continuing "ANSWER: " -- write only the id number of the single '
    "candidate the customer means, and nothing else.\n"
    "Example: 12"
)


def branch_prompt(tokenizer, observation: str) -> str:
    return chat_prompt(tokenizer, f"{BRANCH_INSTRUCTION}\n\n{observation}")


def prompt_length(tokenizer, prompt: str) -> int:
    return len(tokenizer(prompt, add_special_tokens=False).input_ids)


def available_actions(env: MCREnv) -> tuple[bool, bool]:
    """(ask allowed, answer allowed): asking stops at the budget."""
    return (env.asks_used < env.budget, True)


def critic_features(model, tokenizer, device: str, prompt: str, completion_ids: list[int] | None, adapter: str) -> torch.Tensor:
    """The critic encoder's last hidden state at the end of `prompt` (+ a branch),
    read through `adapter` ("critic" or "target_critic"). Leaves that adapter active."""
    model.set_adapter(adapter)
    return last_hidden_state(model, tokenizer, device, prompt, completion_ids)


def decision_probs(q: torch.Tensor, temperature: float, available: tuple[bool, bool] = (True, True),
                   epsilon: float = 0.0) -> torch.Tensor:
    """Softmax over the available branches of [Q(ask), Q(answer)] (argmax at
    temperature 0), mixed with a uniform floor of weight `epsilon`."""
    masked = q.clone()
    for index, allowed in enumerate(available):
        if not allowed:
            masked[index] = float("-inf")
    if temperature <= 0.0:
        probs = torch.zeros_like(masked)
        probs[torch.argmax(masked)] = 1.0
    else:
        probs = torch.softmax(masked / temperature, dim=-1)
    if epsilon > 0.0:
        n = sum(available)
        uniform = torch.tensor([(1.0 / n) if allowed else 0.0 for allowed in available], dtype=probs.dtype,
                               device=probs.device)
        probs = (1.0 - epsilon) * probs + epsilon * uniform
    return probs


def sample_action_kind(probs: torch.Tensor) -> str:
    return "ask" if int(torch.multinomial(probs, num_samples=1).item()) == 0 else "answer"


class NormedHead(torch.nn.Module):
    """Scalar value head over a backbone hidden state: LayerNorm, then a ReLU MLP whose
    output layer starts near zero."""

    def __init__(self, hidden_size: int, mlp_hidden: int = 512) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.mlp_hidden = mlp_hidden
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.net = torch.nn.Sequential(torch.nn.Linear(hidden_size, mlp_hidden), torch.nn.ReLU(),
                                       torch.nn.Linear(mlp_hidden, 1))
        torch.nn.init.normal_(self.net[-1].weight, std=1e-3)  # not zero, which would tie Q(ask) and Q(answer)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(features)).squeeze(-1)


class ReplayBuffer:
    """FIFO buffer with uniform sampling from its own seeded RNG."""

    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = capacity
        self._data: list = []
        self._rng = random.Random(seed)

    def add(self, transition) -> None:
        self._data.append(transition)
        if len(self._data) > self.capacity:
            self._data.pop(0)

    def sample(self):
        return self._data[0] if len(self._data) == 1 else self._rng.sample(self._data, 1)[0]

    def __len__(self) -> int:
        return len(self._data)

    def state(self) -> dict:
        return {"data": list(self._data), "rng": self._rng.getstate()}

    def load(self, state: dict) -> None:
        self._data = list(state["data"])
        self._rng.setstate(state["rng"])
