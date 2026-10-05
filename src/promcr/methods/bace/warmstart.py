"""BACE's warm start: fit the chooser to returns measured at each state, before RL.

States come from walking training episodes with a coin-flip chooser. At each
state both branches are played out `n_mc` times on clones of the
environment: answering, which ends the episode, and asking, followed by one
greedy answer. Paired replays share their random seeds (common random
numbers), so their difference comes from the action. The states are then
resampled so that half have more than one candidate in B, and the heads
regress onto both returns, onto their difference and (V) onto their mean.

Before RL starts, `diagnose` logs the fitted chooser's greedy ask rate on
these states and how often the sign of its Q gap agrees with the measured
advantage. It decides nothing: whether a run passes such a check didn't
predict where its RL ends.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from ...data.episodes import Episode
from ...env.actions import Answer, Ask
from ...env.environment import MCREnv
from ..chooser import ANSWER_PREFIX, ASK_PREFIX, available_actions, critic_features, decision_probs
from .belief import BeliefTracker
from .features import FeatureSpec, q_input, v_input
from .rollout import observe_step, prepare_state

SIGN_DEADBAND = 0.05  # measured advantages this close to zero don't count towards sign agreement


@dataclass
class WarmStartExample:
    state_prompt: str
    ask_ids: list[int]
    answer_ids: list[int]
    belief_features: list[float]
    ask_features: list[float]
    answer_features: list[float]
    available: tuple[bool, bool]
    g_ask: float
    g_answer: float


def _answer_now(env: MCREnv, tracker: BeliefTracker, model, tokenizer, device: str, episode: Episode, cfg,
                rng: random.Random, spec: FeatureSpec) -> float:
    """The continuation after a forced question: one greedy answer."""
    if env.done:
        return 0.0
    state = prepare_state(model, tokenizer, device, episode, env.observation, tracker.state, cfg.budget, cfg.k,
                          cfg.temperature_low, cfg.max_new_tokens, cfg.max_prompt_tokens, "logprob", rng, spec)
    if state is None:
        return 0.0
    return cfg.gamma * env.step(Answer(state.answer_id)).reward


def collect_examples(model, tokenizer, device: str, episodes: list[Episode], simulator, cfg, rng: random.Random,
                     spec: FeatureSpec) -> list[WarmStartExample]:
    examples = []
    for episode in episodes:
        env = MCREnv(episode, simulator, cfg.lambda_penalty, cfg.budget)
        env.reset()
        tracker = BeliefTracker(episode, list(episode.candidate_ids), spec.sizes)
        while not env.done:
            state = prepare_state(model, tokenizer, device, episode, env.observation, tracker.state, cfg.budget, cfg.k,
                                  cfg.temperature_low, cfg.max_new_tokens, cfg.max_prompt_tokens, "random", rng, spec)
            if state is None:
                break
            available = available_actions(env)
            seeds = [rng.randrange(2**31) for _ in range(cfg.warm_start_mc)]
            returns = {"ask": [], "answer": []}
            for kind in ("ask", "answer"):
                action = Ask(state.attribute, state.question) if kind == "ask" else Answer(state.answer_id)
                for seed in seeds:
                    leg_rng = random.Random(seed)
                    torch.manual_seed(seed)  # the generator samples from torch's RNG
                    leg_env, leg_tracker = env.clone(), tracker.copy()
                    result = leg_env.step(action)
                    if kind == "ask":
                        observe_step(leg_tracker, leg_env, state.attribute, result)
                    total = result.reward
                    if not leg_env.done:
                        total += _answer_now(leg_env, leg_tracker, model, tokenizer, device, episode, cfg, leg_rng, spec)
                    returns[kind].append(total)
            examples.append(WarmStartExample(state.prompt, state.ask_ids, state.answer_ids, state.shared, state.ask,
                                             state.answer, available, sum(returns["ask"]) / len(returns["ask"]),
                                             sum(returns["answer"]) / len(returns["answer"])))
            # The walk itself continues under a fair coin.
            kind = "answer" if not available[0] else ("ask" if rng.random() < 0.5 else "answer")
            result = env.step(Ask(state.attribute, state.question) if kind == "ask" else Answer(state.answer_id))
            if kind == "ask":
                observe_step(tracker, env, state.attribute, result)
    return examples


def rebalance(examples: list[WarmStartExample], rng: random.Random) -> list[WarmStartExample]:
    """Downsample the larger side so that half the states have |B| > 1."""
    ambiguous = [e for e in examples if e.belief_features[5] == 0.0]
    resolved = [e for e in examples if e.belief_features[5] != 0.0]
    if not ambiguous or not resolved:
        return list(examples)
    if len(ambiguous) > len(resolved):
        ambiguous = rng.sample(ambiguous, min(len(resolved), len(ambiguous)))
    else:
        resolved = rng.sample(resolved, min(len(ambiguous), len(resolved)))
    mixed = ambiguous + resolved
    rng.shuffle(mixed)
    return mixed


def _q_inputs(model, tokenizer, device: str, example: WarmStartExample):
    ask = critic_features(model, tokenizer, device, example.state_prompt + ASK_PREFIX, example.ask_ids, "critic")
    answer = critic_features(model, tokenizer, device, example.state_prompt + ANSWER_PREFIX, example.answer_ids, "critic")
    return q_input(ask, example.belief_features, example.ask_features), q_input(answer, example.belief_features, example.answer_features)


def fit(model, tokenizer, device: str, heads, optimizer, examples: list[WarmStartExample], cfg,
        rng: random.Random) -> list[float]:
    """One optimizer step per example per epoch. Returns the losses. Leaves the critic adapter active."""
    q_a, q_b, v_a, v_b = heads
    params = [p for group in optimizer.param_groups for p in group["params"]]
    losses = []
    for _ in range(cfg.warm_start_epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        for index in order:
            example = examples[index]
            ask, answer = _q_inputs(model, tokenizer, device, example)
            state = v_input(critic_features(model, tokenizer, device, example.state_prompt, None, "critic"),
                            example.belief_features)
            model.set_adapter("critic")
            g_ask = torch.tensor(example.g_ask, device=device, dtype=torch.float32)
            g_answer = torch.tensor(example.g_answer, device=device, dtype=torch.float32)
            g_mean = torch.tensor(0.5 * (example.g_ask + example.g_answer), device=device, dtype=torch.float32)
            loss = torch.zeros((), device=device, dtype=torch.float32)
            for head in (q_a, q_b):
                loss = loss + (head(ask) - g_ask) ** 2 + (head(answer) - g_answer) ** 2
            for head in (v_a, v_b):
                loss = loss + (head(state) - g_mean) ** 2
            for head in (q_a, q_b):  # the gap is what the chooser acts on
                loss = loss + cfg.warm_start_gap_weight * ((head(ask) - head(answer)) - (g_ask - g_answer)) ** 2
            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.critic_grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            losses.append(float(loss.item()))
    return losses


@dataclass
class WarmStartReport:
    greedy_ask_rate: float
    sign_agreement: float
    n_states: int

    def __str__(self) -> str:
        return (f"warm-start chooser: greedy ask rate {self.greedy_ask_rate:.3f}, sign agreement "
                f"{self.sign_agreement:.3f} over {self.n_states} states")


def diagnose(model, tokenizer, device: str, q_a, q_b, examples: list[WarmStartExample]) -> WarmStartReport:
    asks, decided, agree = 0, 0, 0
    with torch.no_grad():
        for example in examples:
            ask, answer = _q_inputs(model, tokenizer, device, example)
            q = torch.min(torch.stack([q_a(ask), q_a(answer)]), torch.stack([q_b(ask), q_b(answer)]))
            asks += int(decision_probs(q, 0.0, example.available)[0].item() > 0.5)
            measured = example.g_ask - example.g_answer
            if abs(measured) > SIGN_DEADBAND:
                decided += 1
                agree += (float((q[0] - q[1]).item()) > 0) == (measured > 0)
    n = len(examples)
    return WarmStartReport(asks / n if n else float("nan"), agree / decided if decided else float("nan"), n)
