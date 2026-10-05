"""Untrained policies on the instruct model: Natural Behavior and four prompted variants.

Every policy implements `act(episode, observation, candidate_ids, asks_used)`.
`candidate_ids` is always the episode's full candidate list, never the
environment's survivor set.
"""

from __future__ import annotations

import math
import re

from ..config import BUDGET
from ..data.episodes import Episode
from ..env.actions import Action, Answer, Ask
from ..models.backbone import Backbone
from .contract import ASK_CONTRACT, parse_decision, parse_referent, resolve_ask

_ASK_GENERATION_INSTRUCTION = (
    "You are resolving which object a user is referring to among several "
    "candidates. Ask ONE clarifying question that would best help you "
    "figure out which candidate the user means. Respond with only one "
    f"line.\n\n{ASK_CONTRACT}"
)

CONFIDENCE_INSTRUCTION = (
    "You are resolving which object a user is referring to. Read the "
    "candidate objects below and judge which one the user means."
)

_ANSWER_AFTER_ASKING = (
    "Given the candidate objects and the clarification exchange above, answer "
    'with the single object id you believe is correct: "object_id: <id>".'
)


def generate_ask_question(backbone: Backbone, observation: str) -> str:
    """The model's own "<attribute> | <question>" line for the current observation."""
    return backbone.chat(f"{_ASK_GENERATION_INSTRUCTION}\n\n{observation}", max_new_tokens=40).strip()


def _softmax(values: list[float]) -> list[float]:
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    return [e / sum(exps) for e in exps]


def referent_confidence(backbone: Backbone, observation: str, candidate_ids: list[int]) -> dict[int, float]:
    """Softmax over candidates of the length-normalised log-likelihood of " object_id: <id>"."""
    options = [f" object_id: {cid}" for cid in candidate_ids]
    scores = backbone.option_logprobs(f"{CONFIDENCE_INSTRUCTION}\n\n{observation}", options)
    lengths = [len(backbone.tokenizer(option, add_special_tokens=False).input_ids) for option in options]
    return dict(zip(candidate_ids, _softmax([s / max(1, n) for s, n in zip(scores, lengths)])))


class NaturalBehaviorPolicy:
    """No task description and no mention of asking: the reply is classified
    afterwards. A reply that ends with "?" or opens with a question word is a
    question, and its attribute comes from keywords."""

    _INSTRUCTION = (
        "You are a shopping assistant helping a customer. Here is the "
        "conversation so far and the candidate objects you can see. Respond to "
        "the user."
    )
    _QUESTION = re.compile(r"\?\s*$|^\s*(is|are|do|does|did|can|could|would|which|what|who|where|when|why|how)\b",
                           re.IGNORECASE)

    def __init__(self, backbone: Backbone) -> None:
        self.backbone = backbone

    def act(self, episode: Episode, observation: str, candidate_ids: list[int], asks_used: int) -> Action:
        text = self.backbone.chat(f"{self._INSTRUCTION}\n\n{observation}", max_new_tokens=64)
        if self._QUESTION.search(text.strip()):
            attribute, question = resolve_ask(text)
            return Ask(attribute, question.strip())
        return Answer(parse_referent(text, candidate_ids))


class DirectAnswerPolicy:
    """Told to answer immediately."""

    _INSTRUCTION = (
        "You are resolving which object a user is referring to. Given the "
        "candidate objects below, answer immediately with the single object id "
        "you believe is correct. Do not ask any clarifying questions. "
        'Respond with exactly: "object_id: <id>".'
    )

    def __init__(self, backbone: Backbone) -> None:
        self.backbone = backbone

    def act(self, episode: Episode, observation: str, candidate_ids: list[int], asks_used: int) -> Action:
        text = self.backbone.chat(f"{self._INSTRUCTION}\n\n{observation}", max_new_tokens=16)
        return Answer(parse_referent(text, episode.candidate_ids))


class AskOncePolicy:
    """Asks one question of its own choosing, then answers."""

    def __init__(self, backbone: Backbone) -> None:
        self.backbone = backbone

    def act(self, episode: Episode, observation: str, candidate_ids: list[int], asks_used: int) -> Action:
        if asks_used == 0:
            return Ask(*resolve_ask(generate_ask_question(self.backbone, observation)))
        text = self.backbone.chat(f"{_ANSWER_AFTER_ASKING}\n\n{observation}", max_new_tokens=16)
        return Answer(parse_referent(text, candidate_ids))


class UncertaintyGatedPolicy:
    """Asks while its probability for the top candidate is below `threshold`."""

    def __init__(self, backbone: Backbone, threshold: float = 0.6, budget: int = BUDGET) -> None:
        self.backbone = backbone
        self.threshold = threshold
        self.budget = budget

    def act(self, episode: Episode, observation: str, candidate_ids: list[int], asks_used: int) -> Action:
        best_id, best_prob = max(referent_confidence(self.backbone, observation, candidate_ids).items(),
                                 key=lambda item: item[1])
        if asks_used >= self.budget or best_prob >= self.threshold:
            return Answer(best_id)
        return Ask(*resolve_ask(generate_ask_question(self.backbone, observation)))


class ProCoTPolicy:
    """Reasons about the ambiguity, then commits to ASK or ANSWER on its last line (0-shot)."""

    _INSTRUCTION = f"""\
You are resolving which object a user is referring to among several candidates. \
Think step by step about whether the user's request is ambiguous among the \
candidate objects below, and whether that ambiguity can be resolved from the \
dialogue history and candidate attributes alone. Then commit to exactly one of \
the following as your final line:
- "ANSWER: <object id>" if you can resolve it (or must just guess) -- the id number alone.
- "ASK: <attribute> | <clarification question>" if it is genuinely ambiguous and asking would help.

{ASK_CONTRACT}"""

    def __init__(self, backbone: Backbone) -> None:
        self.backbone = backbone

    def act(self, episode: Episode, observation: str, candidate_ids: list[int], asks_used: int) -> Action:
        text = self.backbone.chat(f"{self._INSTRUCTION}\n\n{observation}", max_new_tokens=200)
        return parse_decision(text, candidate_ids)


PROMPTED_POLICIES = {
    "natural-behavior": NaturalBehaviorPolicy,
    "direct-answer": DirectAnswerPolicy,
    "ask-once": AskOncePolicy,
    "uncertainty-gated": UncertaintyGatedPolicy,
    "procot": ProCoTPolicy,
}
