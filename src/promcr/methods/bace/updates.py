"""The chooser's TD update.

Each Q head regresses onto the average of the one-step target and the
observed return-to-go G, y = (1 - w)(r + V_target(s')) + w G (y = r at the
last step, blended the same way). Each V head regresses onto the target
critic's Q of the branch the chooser would pick greedily (clipped double Q:
selected on min(Q_a, Q_b), evaluated by each head), so V estimates the
policy that is deployed.
"""

from __future__ import annotations

import torch

from ..chooser import ANSWER_PREFIX, ASK_PREFIX, critic_features
from ..common import frozen
from .features import q_input, v_input
from .rollout import Transition


def greedy_branch(stack_a: torch.Tensor, stack_b: torch.Tensor, available: tuple[bool, bool]) -> int:
    masked = torch.min(stack_a, stack_b).clone()
    for index, allowed in enumerate(available):
        if not allowed:
            masked[index] = float("-inf")
    return int(torch.argmax(masked).item())


def critic_loss(model, tokenizer, device: str, heads, targets, transition: Transition, cfg) -> torch.Tensor:
    """Sum of the four squared errors for one transition. Leaves the critic adapter active."""
    q_a, q_b, v_a, v_b = heads
    tq_a, tq_b, tv_a, tv_b = targets
    if transition.action_kind == "ask":
        prefix, ids, branch = ASK_PREFIX, transition.ask_ids, transition.ask_features
    else:
        prefix, ids, branch = ANSWER_PREFIX, transition.answer_ids, transition.answer_features
    action = q_input(critic_features(model, tokenizer, device, transition.state_prompt + prefix, ids, "critic"),
                     transition.belief_features, branch)
    current_q_a, current_q_b = q_a(action), q_b(action)
    state = v_input(critic_features(model, tokenizer, device, transition.state_prompt, None, "critic"),
                    transition.belief_features)
    current_v_a, current_v_b = v_a(state), v_b(state)

    with torch.no_grad(), frozen(model, *targets):
        ask = critic_features(model, tokenizer, device, transition.state_prompt + ASK_PREFIX, transition.ask_ids, "target_critic")
        answer = critic_features(model, tokenizer, device, transition.state_prompt + ANSWER_PREFIX,
                                 transition.answer_ids, "target_critic")
        ask = q_input(ask, transition.belief_features, transition.ask_features)
        answer = q_input(answer, transition.belief_features, transition.answer_features)
        stack_a, stack_b = torch.stack([tq_a(ask), tq_a(answer)]), torch.stack([tq_b(ask), tq_b(answer)])
        best = greedy_branch(stack_a, stack_b, transition.available)
        v_a_target, v_b_target = stack_a[best], stack_b[best]

        if transition.done:
            q_a_target = torch.tensor(float(transition.reward), device=device, dtype=current_q_a.dtype)
            q_b_target = torch.tensor(float(transition.reward), device=device, dtype=current_q_b.dtype)
        else:
            following = v_input(critic_features(model, tokenizer, device, transition.next_state_prompt, None, "target_critic"),
                                transition.next_belief_features)
            q_a_target = transition.reward + cfg.gamma * tv_a(following)
            q_b_target = transition.reward + cfg.gamma * tv_b(following)
        if cfg.mc_weight > 0.0 and transition.mc_return is not None:
            mc = torch.tensor(float(transition.mc_return), device=device, dtype=current_q_a.dtype)
            q_a_target = (1.0 - cfg.mc_weight) * q_a_target + cfg.mc_weight * mc
            q_b_target = (1.0 - cfg.mc_weight) * q_b_target + cfg.mc_weight * mc

    model.set_adapter("critic")
    return ((current_q_a - q_a_target) ** 2 + (current_q_b - q_b_target) ** 2
            + (current_v_a - v_a_target) ** 2 + (current_v_b - v_b_target) ** 2)
