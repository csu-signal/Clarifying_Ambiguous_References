"""ArCHer: an utterance-level critic trained off-policy by TD, a token-level actor trained on its advantage.

The actor, the critic encoder and its Polyak target are three LoRA adapters
on one backbone with the SFT adapter merged in. Two {Q, V} head pairs read
the critic's hidden state, and the critic also makes the ask/answer choice:
it samples from an epsilon-floored softmax over Q(ask), Q(answer) in
training and takes the argmax at evaluation.

Each iteration rolls out `episodes_per_iteration` episodes into a replay
buffer, makes `critic_updates_per_iteration` critic updates from it (in steps
of `critic_batch_size` transitions), then `actor_updates_per_iteration`
actor steps on this iteration's own transitions. The first
`actor_warmup_iters` iterations train the critic only, with SFT choosing.
A resumable state, replay buffer included, is saved after every iteration;
`--resume` continues from it.

    python -m promcr.methods.archer.train --config configs/archer.yaml --seed 0 --output checkpoints/archer-seed0
"""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from peft import get_peft_model

from ...config import BUDGET, LAMBDA
from ...data.pools import describe_pool, load_pool
from ...env import ScriptedUserSimulator
from ...models.backbone import load_base_model, load_tokenizer, lora_config, record_warm_start
from ..common import (
    MetricsLog,
    Progress,
    frozen,
    linear_decay,
    load_config,
    load_lora_state,
    load_resume,
    lora_state,
    polyak_update,
    polyak_update_adapter,
    restore_global_rng,
    save_config,
    save_resume,
    seed_everything,
    write_progress,
)
from ..chooser import NormedHead, ReplayBuffer
from .heads import TokenBaseline, save_head
from .rollout import rollout_episode
from .updates import accumulate_actor_gradient, critic_loss


@dataclass
class ArcherConfig:
    output: str = "checkpoints/archer"
    sft_checkpoint: str = "checkpoints/sft-real-synth"
    pool: str = "real"
    seed: int = 0
    device: str = "cuda:0"
    lambda_penalty: float = LAMBDA
    budget: int = BUDGET
    iterations: int = 28
    episodes_per_iteration: int = 394
    critic_updates_per_iteration: int = 800  # replayed transitions per iteration
    critic_batch_size: int = 4  # transitions per critic optimizer step
    actor_warmup_iters: int = 10
    actor_updates_per_iteration: int = 3  # actor optimizer steps per iteration after the warm-up
    actor_grad_accum: int = 4
    actor_kl_coef: float = 0.04
    sft_decides_during_warmup: bool = True
    temperature_high: float = 0.5  # the critic's choice
    temperature_low: float = 0.8  # the actor's branches
    epsilon: float = 0.15
    gamma: float = 1.0
    max_new_tokens: int = 64
    max_prompt_tokens: int = 1280
    actor_lr: float = 5e-5
    critic_lr: float = 1e-3  # value heads
    critic_lora_lr: float = 1e-4  # critic encoder
    baseline_lr: float = 1e-3
    baseline_mlp_hidden: int = 256
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    critic_lora_r: int = 8
    critic_lora_alpha: int = 16
    critic_mlp_hidden: int = 512
    buffer_capacity: int = 5000
    target_tau: float = 0.005
    critic_grad_clip: float = 10.0
    actor_grad_clip: float = 1.0
    snapshot_every_iterations: int = 4
    resume: bool = False


def save_policy(directory: Path, model, tokenizer, q_a, q_b, sft_checkpoint: str, progress: Progress, **extra) -> None:
    """What evaluation loads: the actor and critic adapters and the two Q heads."""
    for adapter in ("actor", "critic"):
        model.set_adapter(adapter)
        model.save_pretrained(directory, selected_adapters=[adapter])
    model.set_adapter("actor")
    tokenizer.save_pretrained(directory)
    save_head(q_a, directory / "q_a_head")
    save_head(q_b, directory / "q_b_head")
    record_warm_start(directory, sft_checkpoint)
    write_progress(directory, progress, **extra)


def train(cfg: ArcherConfig) -> Path:
    if cfg.actor_warmup_iters >= cfg.iterations:
        raise ValueError("actor_warmup_iters leaves no iteration for the actor")
    rng = random.Random(cfg.seed)
    seed_everything(cfg.seed)
    output = Path(cfg.output)
    save_config(cfg, output)
    device = cfg.device

    tokenizer = load_tokenizer(cfg.sft_checkpoint)
    base = load_base_model(device, cfg.sft_checkpoint)
    hidden_size = base.config.hidden_size
    record_warm_start(output, cfg.sft_checkpoint)

    model = get_peft_model(base, lora_config(cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout), adapter_name="actor")
    critic_lora = lora_config(cfg.critic_lora_r, cfg.critic_lora_alpha, cfg.lora_dropout)
    model.add_adapter("critic", critic_lora)
    model.add_adapter("target_critic", critic_lora)
    polyak_update_adapter(model, "target_critic", "critic", tau=1.0)
    for name, parameter in model.named_parameters():
        if ".target_critic." in name:
            parameter.requires_grad = False
    model.set_adapter("actor")
    model.print_trainable_parameters()
    model.train()

    heads = [NormedHead(hidden_size, cfg.critic_mlp_hidden).to(device) for _ in range(4)]  # q_a, q_b, v_a, v_b
    targets = [NormedHead(hidden_size, cfg.critic_mlp_hidden).to(device) for _ in range(4)]
    for target, head in zip(targets, heads):
        target.load_state_dict(head.state_dict())
        target.eval()
        for parameter in target.parameters():
            parameter.requires_grad = False
    baseline = TokenBaseline(hidden_size, cfg.baseline_mlp_hidden).to(device)
    baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=cfg.baseline_lr)

    named = dict(model.named_parameters())
    actor_params = [p for n, p in named.items() if ".actor." in n]
    critic_adapter_params = [p for n, p in named.items() if ".critic." in n and ".target_critic." not in n]
    head_params = [p for head in heads for p in head.parameters()]
    actor_optimizer = torch.optim.AdamW(actor_params, lr=cfg.actor_lr)
    critic_optimizer = torch.optim.AdamW([{"params": critic_adapter_params, "lr": cfg.critic_lora_lr},
                                          {"params": head_params, "lr": cfg.critic_lr}])
    critic_scheduler = linear_decay(critic_optimizer, cfg.iterations * -(-cfg.critic_updates_per_iteration // cfg.critic_batch_size))
    actor_scheduler = linear_decay(actor_optimizer, max(1, (cfg.iterations - cfg.actor_warmup_iters) * cfg.actor_updates_per_iteration))
    baseline_scheduler = linear_decay(baseline_optimizer, cfg.iterations)

    episodes = load_pool("train", cfg.pool)
    print(f"[archer] {describe_pool('train', cfg.pool, episodes)}")
    order = list(range(len(episodes)))  # shuffled in place, as the episode list itself was
    rng.shuffle(order)
    cursor = first_iteration = 0
    simulator = ScriptedUserSimulator()
    buffer = ReplayBuffer(cfg.buffer_capacity, cfg.seed)
    log = MetricsLog(output)
    progress = Progress()
    optimizers = (actor_optimizer, critic_optimizer, baseline_optimizer)
    schedulers = (actor_scheduler, critic_scheduler, baseline_scheduler)
    state = load_resume(output, "archer") if cfg.resume else None
    if state is not None:
        load_lora_state(model, state["lora"])
        for module, saved in zip([*heads, *targets, baseline], state["modules"]):
            module.load_state_dict(saved)
        for optimizer, saved in zip(optimizers, state["optimizers"]):
            optimizer.load_state_dict(saved)
        for scheduler, saved in zip(schedulers, state["schedulers"]):
            scheduler.load_state_dict(saved)
        buffer.load(state["buffer"])
        rng.setstate(state["rng"])
        restore_global_rng(state["global_rng"])
        progress = Progress(**state["progress"])
        order, cursor, first_iteration = state["order"], state["cursor"], state["next_iteration"]
    t0 = time.time()

    for iteration in range(first_iteration, cfg.iterations):
        warmup = iteration < cfg.actor_warmup_iters
        sft_decides = cfg.sft_decides_during_warmup and warmup
        if cursor + cfg.episodes_per_iteration > len(order):
            rng.shuffle(order)
            cursor = 0
        batch = [episodes[i] for i in order[cursor : cursor + cfg.episodes_per_iteration]]
        cursor += cfg.episodes_per_iteration

        fresh, returns = [], []
        with frozen(model, heads[0], heads[1]):
            for episode in batch:
                transitions = rollout_episode(model, tokenizer, device, heads[0], heads[1], episode, simulator, cfg,
                                              sft_decides=sft_decides)
                for transition in transitions:
                    buffer.add(transition)
                fresh += transitions
                progress.episodes += 1
                progress.transitions += len(transitions)
                returns.append(sum(t.reward for t in transitions))

        # Critic: TD updates on replayed transitions, the targets tracking by Polyak averaging.
        critic_losses, window = [], []
        for index in range(cfg.critic_updates_per_iteration):
            if not len(buffer):
                break
            loss = critic_loss(model, tokenizer, device, heads, targets, buffer.sample(), cfg, behaviour_v_target=sft_decides)
            if torch.isfinite(loss):
                (loss / cfg.critic_batch_size).backward()
                window.append(loss.item())
            if (index + 1) % cfg.critic_batch_size and index + 1 != cfg.critic_updates_per_iteration:
                continue
            if not window:
                critic_optimizer.zero_grad()
                continue
            torch.nn.utils.clip_grad_norm_(critic_adapter_params + head_params, cfg.critic_grad_clip)
            critic_optimizer.step()
            critic_scheduler.step()
            critic_optimizer.zero_grad()
            progress.critic_steps += 1
            polyak_update_adapter(model, "target_critic", "critic", cfg.target_tau)
            for target, head in zip(targets, heads):
                polyak_update(target, head, cfg.target_tau)
            critic_losses.append(sum(window) / len(window))
            window = []

        # Actor: a few steps on transitions sampled from this iteration's rollouts.
        actor_losses = []
        chosen = []
        if not warmup:
            wanted = cfg.actor_updates_per_iteration * cfg.actor_grad_accum
            chosen = rng.sample(fresh, wanted) if len(fresh) > wanted else list(fresh)
        for count, transition in enumerate(chosen, start=1):
            loss, _ = accumulate_actor_gradient(model, tokenizer, device, heads, baseline, baseline_optimizer, transition, cfg)
            if loss is not None:
                actor_losses.append(loss)
            if count % cfg.actor_grad_accum == 0 or count == len(chosen):
                torch.nn.utils.clip_grad_norm_(actor_params, cfg.actor_grad_clip)
                actor_optimizer.step()
                actor_scheduler.step()
                actor_optimizer.zero_grad()
                progress.actor_steps += 1
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
        if not warmup:
            baseline_scheduler.step()

        mean = lambda values: sum(values) / len(values) if values else float("nan")
        log.write(iteration, progress, buffer=len(buffer), mean_episode_reward=mean(returns),
                  critic_loss=mean(critic_losses), actor_loss=None if warmup else mean(actor_losses), actor_warmup=warmup)
        print(f"[archer] iter={iteration} episodes={len(returns)} transitions={len(fresh)} buffer={len(buffer)} "
              f"reward={mean(returns):.3f} critic_loss={mean(critic_losses):.4f} "
              f"actor_loss={'warm-up' if warmup else f'{mean(actor_losses):.4f}'} elapsed={time.time() - t0:.0f}s")
        if (iteration + 1) % cfg.snapshot_every_iterations == 0 and iteration + 1 < cfg.iterations:
            save_policy(output / "snapshots" / f"step_{iteration:03d}", model, tokenizer, heads[0], heads[1],
                        cfg.sft_checkpoint, progress, phase="rl", iterations_completed=iteration + 1)
        save_resume(output, lora=lora_state(model), modules=[m.state_dict() for m in [*heads, *targets, baseline]],
                    optimizers=[o.state_dict() for o in optimizers], schedulers=[s.state_dict() for s in schedulers],
                    buffer=buffer.state(), rng=rng.getstate(), progress=asdict(progress), order=order, cursor=cursor,
                    next_iteration=iteration + 1)

    save_policy(output, model, tokenizer, heads[0], heads[1], cfg.sft_checkpoint, progress, phase="final",
                iterations_completed=cfg.iterations)
    print(f"[archer] saved {output} ({time.time() - t0:.0f}s)")
    return output


def main(argv: list[str] | None = None) -> None:
    train(load_config(ArcherConfig, argv, __doc__))


if __name__ == "__main__":
    main()
