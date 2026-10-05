"""ArCHer rollouts: both branches sampled by the actor, the ask/answer choice made by the critic.

The ASK branch is decoded in two steps: the attribute is drawn from the
actor's own probabilities over the askable attributes, then the actor writes
the question freely. Without this, the SFT-initialised actor often wrote an
object id after "ASK: ", and more than half of its questions revealed nothing.

During the critic-only warm-up, SFT makes the choice instead (with the same
epsilon floor), so the critic first learns to score the policy it starts from.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...data.episodes import Episode
from ...env.actions import Action, Answer, Ask
from ...env.attributes import TEXT_ASKABLE_ATTRS
from ...env.environment import MCREnv
from ...models.generation import option_logprobs_cached, sample_completions
from ...policies.contract import DECISION_INSTRUCTION, parse_referent, resolve_ask
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


@dataclass
class Transition:
    state_prompt: str
    action_kind: str  # "ask" | "answer"
    ask_ids: list[int]
    answer_ids: list[int]
    available: tuple[bool, bool]
    reward: float
    next_state_prompt: str | None  # None when done
    next_available: tuple[bool, bool]
    done: bool
    next_ask_ids: list[int] | None
    next_answer_ids: list[int] | None
    decision_probs: tuple[float, float] | None = None  # set when SFT made the choice


def attribute_logprobs(model, tokenizer, device: str, prompt: str) -> torch.Tensor:
    """Log-probability of each "<attribute> |" continuing `prompt`, for every
    askable attribute, from one prompt forward and a shared KV cache."""
    candidates = [tokenizer(attr + " |", add_special_tokens=False).input_ids for attr in TEXT_ASKABLE_ATTRS]
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_out = model(prompt_ids, use_cache=True)
    first = torch.log_softmax(prompt_out.logits[0, -1].float(), dim=-1)

    n, width = len(candidates), max(len(c) for c in candidates)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    batch = torch.full((n, width), pad_id, dtype=torch.long, device=device)
    mask = torch.ones((n, prompt_ids.shape[1] + width), dtype=torch.long, device=device)
    for i, ids in enumerate(candidates):
        batch[i, : len(ids)] = torch.tensor(ids, device=device)
        mask[i, prompt_ids.shape[1] + len(ids) :] = 0
    cache = prompt_out.past_key_values
    cache.batch_repeat_interleave(n)
    rest = torch.log_softmax(model(batch, past_key_values=cache, attention_mask=mask).logits.float(), dim=-1)

    scores = torch.empty(n, device=device)
    for i, ids in enumerate(candidates):
        total = first[ids[0]]
        for j in range(1, len(ids)):
            total = total + rest[i, j - 1, ids[j]]
        scores[i] = total
    return scores


@torch.no_grad()
def sample_constrained_ask(model, tokenizer, device: str, state_prompt: str, temperature: float,
                           max_new_tokens: int) -> tuple[str, list[int]]:
    """(text, ids) of an ASK branch whose attribute comes from the closed menu.
    The attribute is sampled at `temperature` (argmax at 0); the question
    continues from its token ids."""
    prompt = state_prompt + ASK_PREFIX
    scores = attribute_logprobs(model, tokenizer, device, prompt)
    if temperature <= 0.0:
        index = int(torch.argmax(scores).item())
    else:
        index = int(torch.multinomial(torch.softmax(scores / temperature, dim=-1), 1).item())
    attribute_ids = tokenizer(TEXT_ASKABLE_ATTRS[index] + " |", add_special_tokens=False).input_ids
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    context = torch.cat([prompt_ids, torch.tensor([attribute_ids], device=device)], dim=1)
    output = model.generate(input_ids=context, attention_mask=torch.ones_like(context), max_new_tokens=max_new_tokens,
                            do_sample=temperature > 0.0, temperature=temperature if temperature > 0.0 else None,
                            pad_token_id=tokenizer.pad_token_id)
    question_ids = output[0, context.shape[1]:].tolist()
    if tokenizer.pad_token_id in question_ids:
        question_ids = question_ids[: question_ids.index(tokenizer.pad_token_id)]
    ids = list(attribute_ids) + question_ids
    return tokenizer.decode(ids, skip_special_tokens=True).strip(), ids


def sample_branches(model, tokenizer, device: str, state_prompt: str, temperature: float, max_new_tokens: int):
    """(ask text, ask ids, answer text, answer ids) from the actor."""
    model.set_adapter("actor")
    ask_text, ask_ids = sample_constrained_ask(model, tokenizer, device, state_prompt, temperature, max_new_tokens)
    answer_text, answer_ids = sample_completions(model, tokenizer, device, state_prompt + ANSWER_PREFIX, 1,
                                                 max_new_tokens, temperature)[0]
    return ask_text, ask_ids, answer_text, answer_ids


def branch_q(model, tokenizer, device: str, q_a, q_b, state_prompt: str, ask_ids: list[int],
             answer_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Both Q heads on both branches: (q_a stack, q_b stack), each [ask, answer]."""
    with torch.no_grad():
        ask = critic_features(model, tokenizer, device, state_prompt + ASK_PREFIX, ask_ids, "critic")
        answer = critic_features(model, tokenizer, device, state_prompt + ANSWER_PREFIX, answer_ids, "critic")
        stacks = torch.stack([q_a(ask), q_a(answer)]), torch.stack([q_b(ask), q_b(answer)])
    model.set_adapter("actor")
    return stacks


def sft_decision_logodds(model, tokenizer, device: str, observation: str) -> float:
    """log P("ASK:") - log P("ANSWER:") under the SFT decision prompt, every adapter disabled."""
    with model.disable_adapter():
        ask, answer = option_logprobs_cached(model, tokenizer, device, f"{DECISION_INSTRUCTION}\n\n{observation}",
                                             ["ASK:", "ANSWER:"])
    return ask - answer


def sft_decision_probs(model, tokenizer, device: str, observation: str, available: tuple[bool, bool],
                       epsilon: float) -> torch.Tensor:
    p_ask = torch.sigmoid(torch.tensor(sft_decision_logodds(model, tokenizer, device, observation)))
    probs = torch.stack([p_ask, 1.0 - p_ask]) if available[0] else torch.tensor([0.0, 1.0])
    if epsilon > 0.0:
        n = sum(available)
        probs = (1.0 - epsilon) * probs + epsilon * torch.tensor([(1.0 / n) if a else 0.0 for a in available])
    return probs


def rollout_episode(model, tokenizer, device: str, q_a, q_b, episode: Episode, simulator, cfg,
                    sft_decides: bool = False) -> list[Transition]:
    """One episode. Each state's branches are sampled once and reused as the
    previous transition's next-state branches."""
    env = MCREnv(episode, simulator, cfg.lambda_penalty, cfg.budget)
    env.reset()
    transitions: list[Transition] = []
    state_prompt = branch_prompt(tokenizer, env.observation)
    if prompt_length(tokenizer, state_prompt) > cfg.max_prompt_tokens:
        return transitions
    ask_text, ask_ids, answer_text, answer_ids = sample_branches(model, tokenizer, device, state_prompt,
                                                                 cfg.temperature_low, cfg.max_new_tokens)
    while True:
        available = available_actions(env)
        if sft_decides:
            probs = sft_decision_probs(model, tokenizer, device, env.observation, available, cfg.epsilon)
        else:
            q = torch.min(*branch_q(model, tokenizer, device, q_a, q_b, state_prompt, ask_ids, answer_ids))
            probs = decision_probs(q, cfg.temperature_high, available, cfg.epsilon)
        kind = sample_action_kind(probs)
        action: Action = Ask(*resolve_ask(ask_text)) if kind == "ask" else Answer(parse_referent(answer_text, episode.candidate_ids))
        result = env.step(action)

        nxt = None
        if not result.done:
            next_prompt = branch_prompt(tokenizer, env.observation)
            if prompt_length(tokenizer, next_prompt) <= cfg.max_prompt_tokens:
                nxt = (next_prompt, available_actions(env),
                       *sample_branches(model, tokenizer, device, next_prompt, cfg.temperature_low, cfg.max_new_tokens))
        transitions.append(Transition(
            state_prompt=state_prompt, action_kind=kind, ask_ids=ask_ids, answer_ids=answer_ids,
            available=available, reward=result.reward,
            next_state_prompt=nxt[0] if nxt else None, next_available=nxt[1] if nxt else (False, False),
            done=nxt is None, next_ask_ids=nxt[3] if nxt else None, next_answer_ids=nxt[5] if nxt else None,
            decision_probs=(float(probs[0]), float(probs[1])) if sft_decides else None,
        ))
        if nxt is None:
            return transitions
        state_prompt, _, ask_text, ask_ids, answer_text, answer_ids = nxt
