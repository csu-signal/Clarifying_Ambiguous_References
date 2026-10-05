"""BACE: a value-based chooser between the frozen SFT model's own question and answer.

The SFT adapter is merged into the backbone and stays frozen; it writes every
question and answer (the `actor` adapter on top of it is never trained). RL
trains only the chooser: a `critic` LoRA adapter and four belief-aware heads
(two {Q, V} pairs), with a `target_critic` copy for the bootstrap targets.

1. Warm start (`warmstart.py`): fit the heads to measured returns on
   `warm_start_episodes` training episodes and log how the fitted chooser
   decides on them.
2. RL, `iterations` times: roll out `episodes_per_iteration` episodes into a
   replay buffer, then `critic_updates_per_iteration` replayed transitions
   in steps of `critic_batch_size`; the target is copied every
   `target_sync_every` steps (once per iteration at the defaults).

A resumable state is saved after the warm start and after every iteration;
`--resume` continues from it.

    python -m promcr.methods.bace.train --config configs/bace.yaml --seed 0 --output checkpoints/bace-seed0
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
from ..chooser import ReplayBuffer
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
from . import warmstart
from .features import N_BELIEF_FEATURES, N_BRANCH_FEATURES, FeatureSpec
from .heads import BeliefHead, save_head
from .rollout import rollout_episode
from .updates import critic_loss


@dataclass
class BaceConfig:
    output: str = "checkpoints/bace"
    sft_checkpoint: str = "checkpoints/sft-real-synth"
    pool: str = "real"
    seed: int = 0
    device: str = "cuda:0"
    lambda_penalty: float = LAMBDA
    budget: int = BUDGET
    # Generator
    k: int = 2  # completions per branch
    temperature_low: float = 0.8
    max_new_tokens: int = 64
    max_prompt_tokens: int = 1280
    # Chooser features (see features.py)
    psi0: str = "belief"
    belief_sizes: str = "exact"
    exact_feature: bool = False
    # Warm start
    warm_start_episodes: int = 150
    warm_start_mc: int = 2  # playouts per branch per state
    warm_start_epochs: int = 6
    warm_start_gap_weight: float = 1.0
    # RL
    iterations: int = 8
    episodes_per_iteration: int = 20
    critic_updates_per_iteration: int = 200  # replayed transitions per iteration
    critic_batch_size: int = 4
    temperature_high: float = 0.5
    epsilon: float = 0.15
    gamma: float = 1.0
    mc_weight: float = 0.5
    target_sync_every: int = 50  # optimizer steps
    target_tau: float = 1.0  # 1.0 copies the target
    buffer_capacity: int = 5000
    critic_lr: float = 1e-3  # heads
    critic_lora_lr: float = 1e-4  # critic encoder
    critic_grad_clip: float = 10.0
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    critic_lora_r: int = 8
    critic_lora_alpha: int = 16
    critic_mlp_hidden: int = 512
    scalar_width: int = 64
    snapshot_every_iterations: int = 1
    resume: bool = False

    @property
    def features(self) -> FeatureSpec:
        return FeatureSpec(psi0=self.psi0, sizes=self.belief_sizes, exact_feature=self.exact_feature)


def save_policy(directory: Path, model, tokenizer, q_a, q_b, cfg: BaceConfig, progress: Progress, **extra) -> None:
    """What evaluation loads: both adapters, the two Q heads and the feature settings."""
    for adapter in ("actor", "critic"):
        model.set_adapter(adapter)
        model.save_pretrained(directory, selected_adapters=[adapter])
    model.set_adapter("actor")
    tokenizer.save_pretrained(directory)
    save_head(q_a, directory / "q_a_head")
    save_head(q_b, directory / "q_b_head")
    record_warm_start(directory, cfg.sft_checkpoint)
    cfg.features.save(directory)
    write_progress(directory, progress, **extra)


def train(cfg: BaceConfig) -> Path:
    rng = random.Random(cfg.seed)
    seed_everything(cfg.seed)
    output = Path(cfg.output)
    save_config(cfg, output)
    device, spec = cfg.device, cfg.features

    tokenizer = load_tokenizer(cfg.sft_checkpoint)
    base = load_base_model(device, cfg.sft_checkpoint)
    hidden_size = base.config.hidden_size
    record_warm_start(output, cfg.sft_checkpoint)
    spec.save(output)

    model = get_peft_model(base, lora_config(cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout), adapter_name="actor")
    critic_lora = lora_config(cfg.critic_lora_r, cfg.critic_lora_alpha, cfg.lora_dropout)
    model.add_adapter("critic", critic_lora)
    model.add_adapter("target_critic", critic_lora)
    polyak_update_adapter(model, "target_critic", "critic", tau=1.0)
    model.set_adapter("actor")
    for name, parameter in model.named_parameters():
        if ".actor." in name or ".target_critic." in name:
            parameter.requires_grad = False  # the generator stays the SFT model
    model.print_trainable_parameters()
    model.train()

    def head(n_scalar: int) -> BeliefHead:
        return BeliefHead(hidden_size, n_scalar, cfg.critic_mlp_hidden, cfg.scalar_width).to(device)

    n_q, n_v = N_BELIEF_FEATURES + N_BRANCH_FEATURES, N_BELIEF_FEATURES
    heads = [head(n_q), head(n_q), head(n_v), head(n_v)]  # q_a, q_b, v_a, v_b
    targets = [head(h.n_scalar) for h in heads]
    for target, live in zip(targets, heads):
        target.load_state_dict(live.state_dict())
        target.eval()
        for parameter in target.parameters():
            parameter.requires_grad = False

    named = dict(model.named_parameters())
    adapter_params = [p for n, p in named.items() if ".critic." in n and ".target_critic." not in n]
    head_params = [p for h in heads for p in h.parameters()]
    optimizer = torch.optim.AdamW([{"params": adapter_params, "lr": cfg.critic_lora_lr},
                                   {"params": head_params, "lr": cfg.critic_lr}])
    # The schedule counts replayed transitions while stepping once per batch of
    # them, so the LR only decays to about 78% of its start over a run.
    scheduler = linear_decay(optimizer, cfg.iterations * cfg.critic_updates_per_iteration)

    episodes = load_pool("train", cfg.pool)
    print(f"[bace] {describe_pool('train', cfg.pool, episodes)}")
    order = list(range(len(episodes)))  # shuffled in place, as the episode list itself was
    rng.shuffle(order)
    cursor = 0

    def next_batch(n: int):
        nonlocal cursor
        if cursor + n > len(order):
            rng.shuffle(order)
            cursor = 0
        cursor += n
        return [episodes[i] for i in order[cursor - n : cursor]]

    simulator = ScriptedUserSimulator()
    buffer = ReplayBuffer(cfg.buffer_capacity, cfg.seed)
    log = MetricsLog(output)
    progress = Progress()

    def checkpoint(next_iteration: int) -> None:
        save_resume(output, lora=lora_state(model), modules=[m.state_dict() for m in [*heads, *targets]],
                    optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), buffer=buffer.state(),
                    rng=rng.getstate(), progress=asdict(progress), order=order, cursor=cursor,
                    next_iteration=next_iteration)

    state = load_resume(output, "bace") if cfg.resume else None
    if state is not None:
        load_lora_state(model, state["lora"])
        for module, saved in zip([*heads, *targets], state["modules"]):
            module.load_state_dict(saved)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        buffer.load(state["buffer"])
        rng.setstate(state["rng"])
        restore_global_rng(state["global_rng"])
        progress = Progress(**state["progress"])
        order, cursor, first_iteration = state["order"], state["cursor"], state["next_iteration"]
    else:
        # --- warm start ---
        with frozen(model, *heads):
            examples = warmstart.collect_examples(model, tokenizer, device, next_batch(cfg.warm_start_episodes),
                                                  simulator, cfg, rng, spec)
        progress.episodes += cfg.warm_start_episodes
        progress.transitions += len(examples)
        examples = warmstart.rebalance(examples, rng)
        model.set_adapter("critic")
        optimizer.zero_grad()
        losses = warmstart.fit(model, tokenizer, device, heads, optimizer, examples, cfg, rng)
        progress.critic_steps += len(losses)
        log.write(-1, progress, phase="warm_start", warm_start_loss=sum(losses) / len(losses) if losses else None)
        # Its forward passes run with LoRA dropout on and so draw from torch's RNG, as in every reported
        # run; removing them would change every later draw.
        report = warmstart.diagnose(model, tokenizer, device, heads[0], heads[1], examples)
        print(f"[bace] {len(examples)} warm-start states, {len(losses)} steps; {report}")
        for target, live in zip(targets, heads):
            target.load_state_dict(live.state_dict())
        save_policy(output / "snapshots" / "warmstart", model, tokenizer, heads[0], heads[1], cfg, progress,
                    phase="warm_start", iterations_completed=0)
        first_iteration = 0
        checkpoint(0)

    # --- RL ---
    t0 = time.time()
    for iteration in range(first_iteration, cfg.iterations):
        model.set_adapter("actor")
        fresh, returns = [], []
        with frozen(model, *heads):
            for episode in next_batch(cfg.episodes_per_iteration):
                transitions = rollout_episode(model, tokenizer, device, heads[0], heads[1], episode, simulator, cfg, rng, spec)
                for transition in transitions:
                    buffer.add(transition)
                fresh += transitions
                returns.append(sum(t.reward for t in transitions))
                progress.episodes += 1
                progress.transitions += len(transitions)

        model.set_adapter("critic")
        optimizer.zero_grad()
        window, critic_losses, steps_since_sync = 0, [], 0
        draws = (buffer.sample() for _ in range(cfg.critic_updates_per_iteration) if len(buffer))
        for transition in draws:
            loss = critic_loss(model, tokenizer, device, heads, targets, transition, cfg)
            if not torch.isfinite(loss):
                continue
            (loss / cfg.critic_batch_size).backward()
            critic_losses.append(loss.item())
            window += 1
            if window % cfg.critic_batch_size == 0:
                steps_since_sync = _critic_step(model, heads, targets, optimizer, scheduler, adapter_params + head_params,
                                                cfg, steps_since_sync, progress)
        if window % cfg.critic_batch_size:
            _critic_step(model, heads, targets, optimizer, scheduler, adapter_params + head_params, cfg,
                         steps_since_sync, progress)

        mean = lambda values: sum(values) / len(values) if values else float("nan")
        asks = sum(t.action_kind == "ask" for t in fresh)
        log.write(iteration, progress, buffer=len(buffer), ask_rate=asks / max(1, len(fresh)),
                  mean_episode_reward=mean(returns), critic_loss=mean(critic_losses), critic_lr=scheduler.get_last_lr()[0])
        print(f"[bace] iter={iteration} transitions={len(fresh)} buffer={len(buffer)} ask_rate={asks / max(1, len(fresh)):.3f} "
              f"reward={mean(returns):.3f} critic_loss={mean(critic_losses):.4f} elapsed={time.time() - t0:.0f}s")
        model.set_adapter("actor")
        if (iteration + 1) % cfg.snapshot_every_iterations == 0 and iteration + 1 < cfg.iterations:
            save_policy(output / "snapshots" / f"step_{iteration:03d}", model, tokenizer, heads[0], heads[1], cfg,
                        progress, phase="rl", iterations_completed=iteration + 1)
        checkpoint(iteration + 1)

    save_policy(output, model, tokenizer, heads[0], heads[1], cfg, progress, phase="final",
                iterations_completed=cfg.iterations)
    print(f"[bace] saved {output} ({time.time() - t0:.0f}s)")
    return output


def _critic_step(model, heads, targets, optimizer, scheduler, params, cfg: BaceConfig, steps_since_sync: int,
                 progress: Progress) -> int:
    """One optimizer step on the accumulated gradient, then the periodic target copy."""
    torch.nn.utils.clip_grad_norm_(params, cfg.critic_grad_clip)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    progress.critic_steps += 1
    steps_since_sync += 1
    if steps_since_sync >= cfg.target_sync_every:
        polyak_update_adapter(model, "target_critic", "critic", cfg.target_tau)
        for target, live in zip(targets, heads):
            polyak_update(target, live, cfg.target_tau)
        steps_since_sync = 0
    return steps_since_sync


def main(argv: list[str] | None = None) -> None:
    train(load_config(BaceConfig, argv, __doc__))


if __name__ == "__main__":
    main()
