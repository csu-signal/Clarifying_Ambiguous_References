"""ArCHer's critic and actor updates.

Critic (Sec. 3.4 and 3.6 of the ArCHer paper): two independent {Q, V} pairs
on one encoder. Q_i regresses onto r + target V_i(s'), V_i onto the expected
target Q_i under the behaviour policy's choice distribution. While SFT makes
the choice, that expectation uses SFT's stored probabilities.

Actor (Eq. 5): REINFORCE on the branch it wrote, weighted by
A = min(Q_a, Q_b) - min(V_a, V_b) minus a per-token baseline, plus a KL
penalty to the SFT policy, which the paper doesn't use with its smaller actor.

PEFT routes gradients to whichever adapter is active when `.backward()` is
called, so each update leaves the adapter it trains active.
"""

from __future__ import annotations

import torch

from ...models.generation import completion_token_logprobs, kl_k3, reference_token_logprobs
from ..chooser import ANSWER_PREFIX, ASK_PREFIX, critic_features, decision_probs
from ..common import frozen
from .heads import token_features
from .rollout import Transition


def critic_loss(model, tokenizer, device: str, heads, targets, transition: Transition, cfg,
                behaviour_v_target: bool) -> torch.Tensor:
    """Sum of the four squared TD errors for one transition. `heads` and
    `targets` are (q_a, q_b, v_a, v_b). Leaves the "critic" adapter active."""
    q_a, q_b, v_a, v_b = heads
    tq_a, tq_b, tv_a, tv_b = targets
    chosen_prefix, chosen_ids = ((ASK_PREFIX, transition.ask_ids) if transition.action_kind == "ask"
                                 else (ANSWER_PREFIX, transition.answer_ids))
    action_features = critic_features(model, tokenizer, device, transition.state_prompt + chosen_prefix, chosen_ids, "critic")
    current_q_a, current_q_b = q_a(action_features), q_b(action_features)
    state_features = critic_features(model, tokenizer, device, transition.state_prompt, None, "critic")
    current_v_a, current_v_b = v_a(state_features), v_b(state_features)

    with torch.no_grad(), frozen(model, *targets):
        ask = critic_features(model, tokenizer, device, transition.state_prompt + ASK_PREFIX, transition.ask_ids, "target_critic")
        answer = critic_features(model, tokenizer, device, transition.state_prompt + ANSWER_PREFIX,
                                 transition.answer_ids, "target_critic")
        stack_a, stack_b = torch.stack([tq_a(ask), tq_a(answer)]), torch.stack([tq_b(ask), tq_b(answer)])
        if behaviour_v_target and transition.decision_probs is not None:
            weights = torch.tensor(transition.decision_probs, device=stack_a.device, dtype=stack_a.dtype)
            v_a_target, v_b_target = (weights * stack_a).sum(), (weights * stack_b).sum()
        else:
            v_a_target = (decision_probs(stack_a, cfg.temperature_high, transition.available, cfg.epsilon) * stack_a).sum(-1)
            v_b_target = (decision_probs(stack_b, cfg.temperature_high, transition.available, cfg.epsilon) * stack_b).sum(-1)
        if transition.done:
            q_a_target = torch.tensor(float(transition.reward), device=device, dtype=current_q_a.dtype)
            q_b_target = torch.tensor(float(transition.reward), device=device, dtype=current_q_b.dtype)
        else:
            next_features = critic_features(model, tokenizer, device, transition.next_state_prompt, None, "target_critic")
            q_a_target = transition.reward + cfg.gamma * tv_a(next_features)
            q_b_target = transition.reward + cfg.gamma * tv_b(next_features)

    model.set_adapter("critic")
    return ((current_q_a - q_a_target) ** 2 + (current_q_b - q_b_target) ** 2
            + (current_v_a - v_a_target) ** 2 + (current_v_b - v_b_target) ** 2)


def accumulate_actor_gradient(model, tokenizer, device: str, heads, baseline, baseline_optimizer,
                              transition: Transition, cfg, advantage_eps: float = 1e-4, advantage_clip: float = 10.0,
                              logprob_floor: float = 5.0) -> tuple[float | None, float]:
    """Backpropagate the actor loss of one transition and take one step on the
    token baseline. Returns (actor loss or None if skipped, KL)."""
    q_a, q_b, v_a, v_b = heads
    chosen_prefix, chosen_ids = ((ASK_PREFIX, transition.ask_ids) if transition.action_kind == "ask"
                                 else (ANSWER_PREFIX, transition.answer_ids))
    prompt = transition.state_prompt + chosen_prefix
    with torch.no_grad():
        features = critic_features(model, tokenizer, device, prompt, chosen_ids, "critic")
        q = torch.min(q_a(features), q_b(features))
        state = critic_features(model, tokenizer, device, transition.state_prompt, None, "critic")
        v = torch.min(v_a(state), v_b(state))
        advantage = max(-advantage_clip, min(advantage_clip, (q - v).item()))
    model.set_adapter("actor")
    if not chosen_ids or abs(advantage) < advantage_eps:
        return None, 0.0

    predicted = baseline(token_features(model, tokenizer, device, prompt, chosen_ids))
    logprobs = completion_token_logprobs(model, tokenizer, device, prompt, chosen_ids)
    # The floor bounds the policy-gradient term only; the KL reads the unclipped values.
    loss = -((advantage - predicted.detach()) * torch.clamp(logprobs, min=-logprob_floor)).sum()
    kl_value = 0.0
    if cfg.actor_kl_coef > 0.0:
        reference = reference_token_logprobs(model, tokenizer, device, prompt, chosen_ids)
        model.set_adapter("actor")
        kl = kl_k3(logprobs, reference, reduction="sum")
        kl_value = kl.item()
        loss = loss + cfg.actor_kl_coef * kl
    if not torch.isfinite(loss):
        return None, kl_value
    (loss / cfg.actor_grad_accum).backward()

    baseline_loss = ((predicted - torch.full_like(predicted, float(advantage))) ** 2).mean()
    baseline_optimizer.zero_grad()
    baseline_loss.backward()
    baseline_optimizer.step()
    return loss.item(), kl_value
