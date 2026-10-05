"""BACE rollouts: the frozen generator proposes, the chooser picks.

At each state the generator (the SFT model) samples K completions after
"ASK: " and K after "ANSWER: ", and one finalist per branch is kept: at
random in training, the one with the highest mean token log-probability at
evaluation. The chooser scores the two finalists from the critic's hidden
state and the belief features, and samples or takes the argmax.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch

from ...data.episodes import Episode
from ...env.actions import Action, Answer, Ask
from ...env.environment import REPEATED_ATTRIBUTE_REPLY, MCREnv
from ...models.generation import completion_token_logprobs, sample_completions
from ...policies.contract import parse_referent, resolve_ask
from ..chooser import (
    ANSWER_PREFIX,
    ASK_PREFIX,
    available_actions,
    branch_prompt,
    critic_features,
    decision_probs,
    prompt_length,
    sample_action_kind,
)
from .belief import BeliefState, BeliefTracker
from .features import FeatureSpec, annotate, answer_features, ask_features, belief_features, q_input


@dataclass
class Transition:
    state_prompt: str  # includes the belief annotation
    action_kind: str
    ask_ids: list[int]
    answer_ids: list[int]
    available: tuple[bool, bool]
    reward: float
    next_state_prompt: str | None
    next_available: tuple[bool, bool]
    done: bool
    next_ask_ids: list[int] | None
    next_answer_ids: list[int] | None
    # Features are frozen at rollout time: the belief can't be rebuilt from a replayed prompt.
    belief_features: list[float] = field(default_factory=list)
    ask_features: list[float] = field(default_factory=list)
    answer_features: list[float] = field(default_factory=list)
    next_belief_features: list[float] | None = None
    next_ask_features: list[float] | None = None
    next_answer_features: list[float] | None = None
    mc_return: float | None = None  # return-to-go of the behaviour policy


@dataclass
class State:
    """One decision point: the annotated prompt, both finalists and their features."""

    prompt: str
    belief: BeliefState
    shared: list[float]
    ask_text: str
    ask_ids: list[int]
    answer_ids: list[int]
    attribute: str
    question: str
    answer_id: int
    ask: list[float]
    answer: list[float]


def mean_logprob(model, tokenizer, device: str, prompt: str, ids: list[int]) -> float:
    if not ids:
        return float("-inf")
    with torch.no_grad():
        return float(completion_token_logprobs(model, tokenizer, device, prompt, ids).mean().item())


def pick_finalist(completions, mode: str, rng: random.Random | None, model, tokenizer, device: str, prompt: str) -> int:
    if mode == "random":
        return (rng or random.Random(0)).randrange(len(completions))
    scores = [mean_logprob(model, tokenizer, device, prompt, ids) for _, ids in completions]
    return max(range(len(scores)), key=scores.__getitem__)


def prepare_state(model, tokenizer, device: str, episode: Episode, observation: str, belief: BeliefState, budget: int,
                  k: int, temperature: float, max_new_tokens: int, max_prompt_tokens: int, finalist: str,
                  rng: random.Random | None, spec: FeatureSpec) -> State | None:
    """Sample both branches at this state and pick their finalists; None if the prompt is too long."""
    n = len(episode.candidate_ids)
    prompt = branch_prompt(tokenizer, annotate(observation, belief))
    if prompt_length(tokenizer, prompt) > max_prompt_tokens:
        return None
    model.set_adapter("actor")
    asks = sample_completions(model, tokenizer, device, prompt + ASK_PREFIX, k, max_new_tokens, temperature)
    answers = sample_completions(model, tokenizer, device, prompt + ANSWER_PREFIX, k, max_new_tokens, temperature)
    ask_text, ask_ids = asks[pick_finalist(asks, finalist, rng, model, tokenizer, device, prompt + ASK_PREFIX)]
    answer_text, answer_ids = answers[pick_finalist(answers, finalist, rng, model, tokenizer, device, prompt + ANSWER_PREFIX)]
    attribute, question = resolve_ask(ask_text)
    answer_id = parse_referent(answer_text, episode.candidate_ids)
    return State(prompt, belief, belief_features(belief, n, budget, spec), ask_text, ask_ids, answer_ids,
                 attribute, question, answer_id, ask_features(episode, belief, attribute, n, spec),
                 answer_features(belief, answer_id))


def chooser_q(model, tokenizer, device: str, q_a, q_b, state: State) -> tuple[torch.Tensor, torch.Tensor]:
    """(q_a stack, q_b stack), each [Q(ask), Q(answer)]. Leaves the actor adapter active."""
    with torch.no_grad():
        ask = critic_features(model, tokenizer, device, state.prompt + ASK_PREFIX, state.ask_ids, "critic")
        answer = critic_features(model, tokenizer, device, state.prompt + ANSWER_PREFIX, state.answer_ids, "critic")
        ask_in, answer_in = q_input(ask, state.shared, state.ask), q_input(answer, state.shared, state.answer)
        stacks = torch.stack([q_a(ask_in), q_a(answer_in)]), torch.stack([q_b(ask_in), q_b(answer_in)])
    model.set_adapter("actor")
    return stacks


def observe_step(tracker: BeliefTracker, env: MCREnv, attribute: str, result) -> None:
    """Fold the reply to an executed Ask into the belief."""
    if result.done or result.info.get("outcome") == "must_answer" or not env.qa_log:
        return
    reply = env.qa_log[-1][1]
    if reply == REPEATED_ATTRIBUTE_REPLY:
        tracker.note_refusal()
    else:
        tracker.observe(attribute, reply)


def rollout_episode(model, tokenizer, device: str, q_a, q_b, episode: Episode, simulator, cfg, rng: random.Random,
                    spec: FeatureSpec) -> list[Transition]:
    env = MCREnv(episode, simulator, cfg.lambda_penalty, cfg.budget)
    env.reset()
    tracker = BeliefTracker(episode, list(episode.candidate_ids), spec.sizes)
    transitions: list[Transition] = []

    def prepare() -> State | None:
        return prepare_state(model, tokenizer, device, episode, env.observation, tracker.state, cfg.budget, cfg.k,
                             cfg.temperature_low, cfg.max_new_tokens, cfg.max_prompt_tokens, "random", rng, spec)

    current = prepare()
    while current is not None:
        q = torch.min(*chooser_q(model, tokenizer, device, q_a, q_b, current))
        available = available_actions(env)
        kind = sample_action_kind(decision_probs(q, cfg.temperature_high, available, cfg.epsilon))
        if kind == "ask" and not available[0]:
            kind = "answer"
        action: Action = Ask(current.attribute, current.question) if kind == "ask" else Answer(current.answer_id)
        result = env.step(action)
        if kind == "ask":
            observe_step(tracker, env, current.attribute, result)
        nxt = None if result.done else prepare()
        transitions.append(Transition(
            state_prompt=current.prompt, action_kind=kind, ask_ids=current.ask_ids, answer_ids=current.answer_ids,
            available=available, reward=result.reward,
            next_state_prompt=nxt.prompt if nxt else None,
            next_available=available_actions(env) if nxt else (False, False), done=nxt is None,
            next_ask_ids=nxt.ask_ids if nxt else None, next_answer_ids=nxt.answer_ids if nxt else None,
            belief_features=current.shared, ask_features=current.ask, answer_features=current.answer,
            next_belief_features=nxt.shared if nxt else None, next_ask_features=nxt.ask if nxt else None,
            next_answer_features=nxt.answer if nxt else None,
        ))
        current = nxt

    running = 0.0
    for transition in reversed(transitions):
        running = transition.reward + cfg.gamma * running
        transition.mc_return = running
    return transitions
