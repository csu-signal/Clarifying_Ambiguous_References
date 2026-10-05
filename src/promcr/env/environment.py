"""The episode MDP: Ask or Answer, +1 / -1 for the answer, -lambda per question, a question budget.

Rules the environment enforces:

- Each attribute can be asked once. A repeat spends the turn, reveals
  nothing and narrows nothing.
- Visual attributes (color, pattern, ...) are refused like an unparsable
  question, since the text-only observation never shows them.
- Once the budget is spent, the next Ask is refused and the policy is told
  to answer; it doesn't cost anything. If it asks again, the environment
  ends the episode with a random survivor, drawn with a seed derived from
  the episode so reruns agree.

The survivor set is environment bookkeeping and never part of the observation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from ..data.episodes import Episode
from ..data.simmc import VISUAL_FORBIDDEN
from .actions import Action, Answer, Ask
from .attributes import UNRESOLVED_ASK_ATTRIBUTE
from .simulator import ScriptedUserSimulator

MUST_ANSWER_NOTICE = (
    "You have used all of your clarification questions. You must answer now "
    "with the object id you believe is correct."
)
REPEATED_ATTRIBUTE_REPLY = "I already answered that one."


@dataclass
class StepResult:
    observation: str
    reward: float
    done: bool
    info: dict[str, Any]


def render_observation(episode: Episode, qa_log: list[tuple[str, str]]) -> str:
    """What the policy sees: the episode input plus the clarification exchanges so far."""
    base = episode.render_input()
    if not qa_log:
        return base
    exchanges = "\n".join(f"Q: {q}\nA: {a}" for q, a in qa_log)
    return f"{base}\n\nClarification so far:\n{exchanges}"


class MCREnv:
    def __init__(self, episode: Episode, simulator: ScriptedUserSimulator, lambda_penalty: float, budget: int) -> None:
        if budget < 1:
            raise ValueError("budget must be >= 1")
        self.episode = episode
        self.simulator = simulator
        self.lambda_penalty = lambda_penalty
        self.budget = budget
        self.reset()

    def reset(self) -> str:
        self._survivors = list(self.episode.survivor_ids)
        self._asks_used = 0
        self._qa_log: list[tuple[str, str]] = []
        self._asked: set[str] = set()
        self._done = False
        self._must_answer = False
        self._forced_rng = random.Random(self.episode.key)
        return self.observation

    def clone(self) -> "MCREnv":
        """An independent copy at the same point of the episode."""
        clone = MCREnv(self.episode, self.simulator, self.lambda_penalty, self.budget)
        clone._survivors = list(self._survivors)
        clone._asks_used = self._asks_used
        clone._qa_log = list(self._qa_log)
        clone._asked = set(self._asked)
        clone._done = self._done
        clone._must_answer = self._must_answer
        return clone

    @property
    def asks_used(self) -> int:
        return self._asks_used

    @property
    def survivors(self) -> list[int]:
        return list(self._survivors)

    @property
    def qa_log(self) -> list[tuple[str, str]]:
        return list(self._qa_log)

    @property
    def done(self) -> bool:
        return self._done

    @property
    def observation(self) -> str:
        base = render_observation(self.episode, self._qa_log)
        return f"{base}\n\n{MUST_ANSWER_NOTICE}" if self._must_answer else base

    def step(self, action: Action) -> StepResult:
        if self._done:
            raise RuntimeError("step() called on a finished episode")
        if isinstance(action, Ask):
            return self._ask(action)
        if isinstance(action, Answer):
            return self._answer(action.referent_id, forced=False)
        raise TypeError(f"unknown action type: {type(action)!r}")

    def _ask(self, action: Ask) -> StepResult:
        if self._asks_used >= self.budget:
            if self._must_answer:
                return self._answer(self._forced_rng.choice(self._survivors), forced=True)
            self._must_answer = True
            return StepResult(self.observation, 0.0, False, {"outcome": "must_answer", "asks_used": self._asks_used})

        attribute = UNRESOLVED_ASK_ATTRIBUTE if action.attribute in VISUAL_FORBIDDEN else action.attribute
        self._asks_used += 1
        info: dict[str, Any] = {"outcome": "asked", "attribute": attribute}
        if attribute in self._asked:
            self._qa_log.append((action.question, REPEATED_ATTRIBUTE_REPLY))
            info["repeated_attribute"] = True
        else:
            self._asked.add(attribute)
            reply, self._survivors = self.simulator.answer(self.episode, attribute, self._survivors)
            self._qa_log.append((action.question, reply))
        info["survivors"] = list(self._survivors)
        return StepResult(self.observation, -self.lambda_penalty, False, info)

    def _answer(self, referent_id: int, forced: bool) -> StepResult:
        self._done = True
        correct = referent_id == self.episode.gold_referent
        info: dict[str, Any] = {"outcome": "correct" if correct else "incorrect", "asks_used": self._asks_used}
        if forced:
            info |= {"referent_id": referent_id, "forced_answer": True}
        return StepResult(self.observation, 1.0 if correct else -1.0, True, info)
