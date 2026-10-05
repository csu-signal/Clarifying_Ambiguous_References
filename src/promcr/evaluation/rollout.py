"""Run a policy through the environment and keep what happened at every step."""

from __future__ import annotations

import random
import zlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import torch

from ..config import BUDGET, LAMBDA
from ..data.episodes import Episode
from ..env.actions import Action, Ask
from ..env.environment import MCREnv
from ..env.simulator import ScriptedUserSimulator
from ..policies.contract import unparsed_ask_count

EVAL_SEED = 12345


class Policy(Protocol):
    def act(self, episode: Episode, observation: str, candidate_ids: list[int], asks_used: int) -> Action: ...


@dataclass
class StepLog:
    action: Action
    reward: float
    survivors_before: list[int]  # environment bookkeeping, for metrics only
    survivors_after: list[int]
    unparsed_ask: bool = False  # the ask's attribute could not be read from its text
    confidence: Any = None  # probe.StepConfidence, when a probe ran
    decision_q: dict[str, list[float]] | None = None  # each branch's Q from every critic head


@dataclass
class EpisodeTrace:
    episode: Episode
    steps: list[StepLog] = field(default_factory=list)
    total_reward: float = 0.0
    asks_used: int = 0
    correct: bool = False
    forced_answer: bool = False  # the environment answered after a refused second over-budget ask

    @property
    def asked_attributes(self) -> list[str]:
        return [s.action.attribute for s in self.steps if isinstance(s.action, Ask)]


@contextmanager
def episode_rng(key: str, base_seed: int = EVAL_SEED):
    """Seed python's and torch's RNGs from the episode key for the block, then
    restore them. Any sampling a policy does at evaluation (BACE's candidates)
    is then the same on every run and for every checkpoint."""
    seed = zlib.crc32(f"{base_seed}:{key}".encode())
    python_state = random.getstate()
    devices = [torch.cuda.current_device()] if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        random.seed(seed)
        torch.default_generator.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed(seed)
        try:
            yield
        finally:
            random.setstate(python_state)


def run_episode(policy: Policy, episode: Episode, lambda_penalty: float = LAMBDA, budget: int = BUDGET,
                probe: Callable | None = None) -> EpisodeTrace:
    with episode_rng(episode.key):
        env = MCREnv(episode, ScriptedUserSimulator(), lambda_penalty, budget)
        observation = env.reset()
        trace = EpisodeTrace(episode)
        while True:
            survivors = env.survivors
            confidence = probe(policy, episode, observation, episode.candidate_ids) if probe else None
            if hasattr(policy, "last_decision_q"):
                policy.last_decision_q = None
            unparsed_before = unparsed_ask_count()
            action = policy.act(episode, observation, episode.candidate_ids, env.asks_used)
            unparsed = unparsed_ask_count() > unparsed_before
            result = env.step(action)
            trace.steps.append(StepLog(action, result.reward, survivors, env.survivors, unparsed, confidence,
                                       getattr(policy, "last_decision_q", None)))
            trace.total_reward += result.reward
            observation = result.observation
            if result.done:
                trace.asks_used = env.asks_used
                trace.correct = result.info["outcome"] == "correct"
                trace.forced_answer = result.info.get("forced_answer", False)
                return trace


def run_split(policy: Policy, episodes: list[Episode], lambda_penalty: float = LAMBDA, budget: int = BUDGET,
              probe: Callable | None = None) -> list[EpisodeTrace]:
    return [run_episode(policy, e, lambda_penalty, budget, probe) for e in episodes]
