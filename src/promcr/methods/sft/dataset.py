"""Supervised examples: one per decision along the oracle's chain.

An episode that needs no question gives one ANSWER example. An irreducible
episode gives one ASK example per oracle question, each rendered with the
exchanges before it, then a final ANSWER example. If the chain can't
isolate the referent within the budget, that last target is its first
remaining survivor. Questions use the template wording: SFT learns the
decision and the attribute, not the phrasing.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...config import BUDGET
from ...data.episodes import Episode
from ...env.environment import render_observation
from ...env.oracle import build_gold_chain
from ...env.simulator import phrase_answer
from ...policies.contract import DECISION_INSTRUCTION


@dataclass(frozen=True)
class Example:
    prompt: str
    target: str


def episode_examples(episode: Episode, budget: int = BUDGET) -> list[Example]:
    if episode.gold_action != "ask":
        return [Example(f"{DECISION_INSTRUCTION}\n\n{episode.render_input()}", f"ANSWER: {episode.gold_referent}")]
    chain = build_gold_chain(episode, budget)
    examples, qa_log = [], []
    for step in chain.steps:
        examples.append(Example(f"{DECISION_INSTRUCTION}\n\n{render_observation(episode, qa_log)}",
                                f"ASK: {step.attribute} | {step.question}"))
        qa_log.append((step.question, phrase_answer(step.attribute, step.revealed_value)))
    final = episode.gold_referent if chain.resolved else chain.final_survivors[0]
    examples.append(Example(f"{DECISION_INSTRUCTION}\n\n{render_observation(episode, qa_log)}", f"ANSWER: {final}"))
    return examples


def build_examples(episodes: list[Episode], budget: int = BUDGET) -> list[Example]:
    return [example for episode in episodes for example in episode_examples(episode, budget)]
