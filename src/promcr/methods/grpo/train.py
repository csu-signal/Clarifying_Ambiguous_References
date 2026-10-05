"""Trajectory-level GRPO, warm-started from SFT.

For each training episode, K full rollouts are sampled from the current
policy (a fresh sample at every turn). A rollout's advantage is its episode
return normalised by the group's mean and standard deviation, and that one
advantage applies to every token the policy wrote at every turn. The loss is
PPO's clipped surrogate (two passes over each batch) plus a k3 KL penalty to
the SFT policy, averaged over all of a rollout's tokens.

    python -m promcr.methods.grpo.train --config configs/grpo.yaml --seed 0 --output checkpoints/grpo-seed0

Snapshots go to `output/snapshots/episodes_NNNNNN`, named after the first
episode of the batch they were saved after. A resumable state is saved after
every batch; `--resume` continues from it.
"""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from peft import get_peft_model

from ...config import BUDGET, LAMBDA, MODEL_NAME
from ...data.episodes import Episode
from ...data.pools import describe_pool, load_pool
from ...env import MCREnv, ScriptedUserSimulator
from ...models.backbone import load_base_model, load_tokenizer, lora_config, record_warm_start
from ...models.generation import chat_prompt, completion_token_logprobs, reference_token_logprobs, sample_completions
from ...policies.contract import DECISION_INSTRUCTION, parse_decision
from ..common import (
    MetricsLog,
    Progress,
    frozen,
    gradients,
    linear_decay,
    load_config,
    load_gradients,
    load_lora_state,
    load_resume,
    lora_state,
    restore_global_rng,
    save_config,
    save_resume,
    seed_everything,
    write_progress,
)


@dataclass
class GRPOConfig:
    output: str = "checkpoints/grpo"
    sft_checkpoint: str = "checkpoints/sft-real-synth"
    pool: str = "real"
    seed: int = 0
    device: str = "cuda:0"
    lambda_penalty: float = LAMBDA
    budget: int = BUDGET
    k: int = 4  # rollouts per episode
    temperature: float = 0.8
    max_new_tokens: int = 64
    max_prompt_tokens: int = 1280
    kl_coef: float = 0.04
    clip_eps: float = 0.2
    ppo_epochs: int = 2
    episodes_per_batch: int = 8
    grad_accum: int = 4  # episode groups per optimizer step
    lr: float = 5e-5
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_episodes: int | None = None
    stop_after_episodes: int | None = None  # end early, keeping the full run's episode order and LR schedule
    snapshot_every_batches: int = 125
    log_every: int = 10
    resume: bool = False


@dataclass
class Turn:
    prompt: str
    token_ids: list[int]


@dataclass
class Trajectory:
    turns: list[Turn] = field(default_factory=list)
    total_return: float = 0.0


def group_advantages(returns: list[float]) -> list[float]:
    """Returns normalised by the group mean and (population) std; zeros for a zero-variance group."""
    n = len(returns)
    if n <= 1:
        return [0.0] * n
    mean = sum(returns) / n
    std = (sum((r - mean) ** 2 for r in returns) / n) ** 0.5
    return [0.0] * n if std < 1e-6 else [(r - mean) / std for r in returns]


def sample_trajectory(model, tokenizer, cfg: GRPOConfig, episode: Episode, simulator) -> Trajectory | None:
    """One rollout of the current policy, or None if a prompt runs over `max_prompt_tokens`."""
    env = MCREnv(episode, simulator, cfg.lambda_penalty, cfg.budget)
    observation = env.reset()
    trajectory = Trajectory()
    while not env.done:
        prompt = chat_prompt(tokenizer, f"{DECISION_INSTRUCTION}\n\n{observation}")
        if len(tokenizer(prompt, add_special_tokens=False).input_ids) > cfg.max_prompt_tokens:
            return None
        text, ids = sample_completions(model, tokenizer, cfg.device, prompt, 1, cfg.max_new_tokens, cfg.temperature)[0]
        result = env.step(parse_decision(text, episode.candidate_ids))
        trajectory.total_return += result.reward
        trajectory.turns.append(Turn(prompt, ids))
        observation = result.observation
    return trajectory


def old_logprobs(model, tokenizer, device: str, trajectory: Trajectory) -> list[torch.Tensor]:
    """The sampling policy's per-token log-probabilities, the denominator of PPO's ratio."""
    with torch.no_grad():
        return [completion_token_logprobs(model, tokenizer, device, t.prompt, t.token_ids).detach()
                for t in trajectory.turns]


def accumulate_trajectory(model, tokenizer, cfg: GRPOConfig, trajectory: Trajectory, advantage: float,
                          old: list[torch.Tensor], n_in_group: int, clip: float = 5.0) -> tuple[float, float]:
    """Backpropagate one rollout's share of the loss, one turn at a time.
    Returns (loss, KL) summed over its turns."""
    total_tokens = sum(len(lp) for lp in old)
    loss_total = kl_total = 0.0
    for turn, old_lp in zip(trajectory.turns, old):
        new_lp = completion_token_logprobs(model, tokenizer, cfg.device, turn.prompt, turn.token_ids)
        ref_lp = reference_token_logprobs(model, tokenizer, cfg.device, turn.prompt, turn.token_ids)
        kl_ratio = torch.clamp(ref_lp - new_lp, min=-clip, max=clip)
        kl = torch.exp(kl_ratio) - kl_ratio - 1
        ratio = torch.exp(torch.clamp(new_lp - old_lp, min=-clip, max=clip))
        surrogate = torch.minimum(ratio * advantage, torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * advantage)
        contribution = (-surrogate.sum() + cfg.kl_coef * kl.sum()) / total_tokens
        if not torch.isfinite(contribution):
            continue
        (contribution / (n_in_group * cfg.grad_accum)).backward()
        loss_total += contribution.item()
        kl_total += (kl.sum() / total_tokens).item()
    return loss_total, kl_total


def train(cfg: GRPOConfig) -> Path:
    rng = random.Random(cfg.seed)
    seed_everything(cfg.seed)
    output = Path(cfg.output)
    save_config(cfg, output)

    tokenizer = load_tokenizer(MODEL_NAME)
    base = load_base_model(cfg.device, cfg.sft_checkpoint)
    record_warm_start(output, cfg.sft_checkpoint)
    model = get_peft_model(base, lora_config(cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout))
    model.print_trainable_parameters()

    episodes = load_pool("train", cfg.pool)
    print(f"[grpo] {describe_pool('train', cfg.pool, episodes)}")
    rng.shuffle(episodes)
    if cfg.max_episodes is not None:
        episodes = episodes[: cfg.max_episodes]
    simulator = ScriptedUserSimulator()

    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    scheduler = linear_decay(optimizer, max(1, (len(episodes) * cfg.ppo_epochs) // cfg.grad_accum))
    log = MetricsLog(output)
    progress = Progress()
    step = micro_step = first_batch = 0
    losses, kls = [], []
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    state = load_resume(output, "grpo") if cfg.resume else None
    if state is not None:
        load_lora_state(model, state["lora"])
        load_gradients(trainable, state["gradients"])  # a half-accumulated step
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        rng.setstate(state["rng"])
        restore_global_rng(state["global_rng"])
        progress = Progress(**state["progress"])
        step, micro_step, first_batch = state["step"], state["micro_step"], state["next_batch"]
        losses, kls = state["losses"], state["kls"]
    t0 = time.time()

    for batch_start in range(first_batch, len(episodes), cfg.episodes_per_batch):
        # Roll out every group under the current weights; they are reused for all PPO epochs.
        groups, batch_returns = [], []
        with frozen(model):
            for episode in episodes[batch_start : batch_start + cfg.episodes_per_batch]:
                trajectories = [sample_trajectory(model, tokenizer, cfg, episode, simulator) for _ in range(cfg.k)]
                progress.episodes += 1
                progress.transitions += sum(len(t.turns) for t in trajectories if t is not None)
                returns = [t.total_return for t in trajectories if t is not None]
                if not returns:
                    continue
                advantage_iter = iter(group_advantages(returns))
                advantages = [0.0 if t is None else next(advantage_iter) for t in trajectories]
                batch_returns.append(sum(returns) / len(returns))
                if all(a == 0.0 for a in advantages):
                    continue  # nothing to learn from a group whose rollouts all scored the same
                olds = [old_logprobs(model, tokenizer, cfg.device, t) if t is not None else None for t in trajectories]
                groups.append((trajectories, advantages, olds))
        if batch_returns:
            print(f"[grpo] episodes={batch_start + cfg.episodes_per_batch} groups={len(groups)} "
                  f"mean_return={sum(batch_returns) / len(batch_returns):.3f} elapsed={time.time() - t0:.0f}s")

        for ppo_epoch in range(cfg.ppo_epochs):
            for trajectories, advantages, olds in groups:
                micro_step += 1
                members = [(t, a, o) for t, a, o in zip(trajectories, advantages, olds) if t is not None and t.turns and a != 0.0]
                if members:
                    results = [accumulate_trajectory(model, tokenizer, cfg, t, a, o, len(members)) for t, a, o in members]
                    losses.append(sum(r[0] for r in results) / len(members))
                    kls.append(sum(r[1] for r in results) / len(members))
                if micro_step % cfg.grad_accum:
                    continue
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if cfg.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                step += 1
                progress.actor_steps += 1
                if step % cfg.log_every == 0:
                    log.write(step, progress, ppo_epoch=ppo_epoch, groups_processed=micro_step,
                              loss=sum(losses) / len(losses) if losses else None,
                              kl=sum(kls) / len(kls) if kls else None, lr=scheduler.get_last_lr()[0])
                    losses, kls = [], []

        if cfg.snapshot_every_batches and (batch_start // cfg.episodes_per_batch) % cfg.snapshot_every_batches == 0:
            directory = output / "snapshots" / f"episodes_{batch_start:06d}"
            model.save_pretrained(directory)
            tokenizer.save_pretrained(directory)
            record_warm_start(directory, cfg.sft_checkpoint)
            write_progress(directory, progress, phase="rl", batches_completed=batch_start // cfg.episodes_per_batch + 1)
        save_resume(output, lora=lora_state(model), gradients=gradients(trainable), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), rng=rng.getstate(), progress=asdict(progress), step=step,
                    micro_step=micro_step, next_batch=batch_start + cfg.episodes_per_batch, losses=losses, kls=kls)
        if cfg.stop_after_episodes is not None and batch_start >= cfg.stop_after_episodes:
            break

    if micro_step % cfg.grad_accum:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        progress.actor_steps += 1
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    write_progress(output, progress, phase="final")
    print(f"[grpo] saved {output} ({time.time() - t0:.0f}s)")
    return output


def main(argv: list[str] | None = None) -> None:
    train(load_config(GRPOConfig, argv, __doc__))


if __name__ == "__main__":
    main()
