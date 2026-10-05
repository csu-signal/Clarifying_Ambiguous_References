"""Supervised fine-tuning: a LoRA adapter that imitates the oracle's decisions.

    python -m promcr.methods.sft.train --config configs/sft.yaml --output checkpoints/sft-real-synth

Writes the adapter to `output`. With `snapshot_every_steps`, an evaluable
adapter also goes to `output/snapshots/step_NNNNNN` before the first step
and every N optimizer steps. A resumable state is saved every
`checkpoint_every_steps` steps; `--resume` continues from it.
"""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from peft import get_peft_model

from ...config import BUDGET, MODEL_NAME
from ...data.pools import describe_pool, load_pool
from ...models.backbone import load_base_model, load_tokenizer, lora_config, record_warm_start
from ..common import (
    MetricsLog,
    Progress,
    linear_decay,
    load_config,
    load_lora_state,
    load_resume,
    lora_state,
    restore_global_rng,
    save_config,
    save_resume,
    seed_everything,
    write_progress,
)
from .dataset import Example, build_examples


@dataclass
class SFTConfig:
    output: str = "checkpoints/sft-real-synth"
    pool: str = "real+synth"
    seed: int = 0
    device: str = "cuda:0"
    budget: int = BUDGET
    max_len: int = 1280  # tokens; longer examples are dropped
    batch_size: int = 4
    grad_accum: int = 4
    epochs: int = 1
    lr: float = 1e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_examples: int | None = None
    snapshot_every_steps: int = 0
    checkpoint_every_steps: int = 100
    log_every: int = 20
    resume: bool = False


def tokenize(tokenizer, example: Example) -> tuple[list[int], list[int]]:
    """(input ids, labels) with the prompt masked out of the loss."""
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": example.prompt}], tokenize=False,
                                           add_generation_prompt=True)
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    target_ids = tokenizer(example.target + tokenizer.eos_token, add_special_tokens=False).input_ids
    return prompt_ids + target_ids, [-100] * len(prompt_ids) + target_ids


def collate(batch: list[tuple[list[int], list[int]]], pad_id: int, device: str):
    width = max(len(ids) for ids, _ in batch)
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for row, (ids, target) in enumerate(batch):
        input_ids[row, : len(ids)] = torch.tensor(ids)
        labels[row, : len(ids)] = torch.tensor(target)
        mask[row, : len(ids)] = 1
    return input_ids.to(device), mask.to(device), labels.to(device)


def train(cfg: SFTConfig) -> Path:
    rng = random.Random(cfg.seed)
    seed_everything(cfg.seed)
    output = Path(cfg.output)
    save_config(cfg, output)

    tokenizer = load_tokenizer(MODEL_NAME)
    model = get_peft_model(load_base_model(cfg.device), lora_config(cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout))
    model.print_trainable_parameters()

    episodes = load_pool("train", cfg.pool)
    print(f"[sft] {describe_pool('train', cfg.pool, episodes)}")
    examples = build_examples(episodes, cfg.budget)
    n_examples = len(examples)
    rng.shuffle(examples)
    if cfg.max_examples is not None:
        examples = examples[: cfg.max_examples]
    kept = [item for item in (tokenize(tokenizer, e) for e in examples) if len(item[0]) <= cfg.max_len]
    print(f"[sft] {len(kept)} of {len(examples)} examples fit in {cfg.max_len} tokens")
    # Examples seen convert to episodes for the learning curve: one pass over `kept` covers the pool.
    episodes_per_pass = len(episodes) * len(kept) / max(1, n_examples)

    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    scheduler = linear_decay(optimizer, max(1, (len(kept) * cfg.epochs) // (cfg.batch_size * cfg.grad_accum)))
    log = MetricsLog(output)
    progress = Progress(episodes=None)
    step = micro_step = start_epoch = start_batch = 0
    order = list(range(len(kept)))  # shuffled in place each epoch, as `kept` itself was
    state = load_resume(output, "sft") if cfg.resume else None
    if state is not None:
        load_lora_state(model, state["lora"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        rng.setstate(state["rng"])
        restore_global_rng(state["global_rng"])
        progress = Progress(**state["progress"])
        step, micro_step, order = state["step"], state["micro_step"], state["order"]
        start_epoch, start_batch = state["epoch"], state["next_batch"]

    def snapshot(epoch: int) -> None:
        directory = output / "snapshots" / f"step_{step:06d}"
        model.save_pretrained(directory)
        tokenizer.save_pretrained(directory)
        record_warm_start(directory, None)
        write_progress(directory, progress, phase="sft", epoch=epoch,
                       episodes_seen=round(progress.transitions / max(1, len(kept)) * episodes_per_pass))

    if cfg.snapshot_every_steps > 0 and state is None:
        snapshot(0)  # the untrained adapter
    t0 = time.time()
    for epoch in range(start_epoch, cfg.epochs):
        if state is None or epoch > start_epoch:
            rng.shuffle(order)
        first = start_batch if state is not None and epoch == start_epoch else 0
        for start in range(first, len(kept), cfg.batch_size):
            batch = [kept[i] for i in order[start : start + cfg.batch_size]]
            input_ids, mask, labels = collate(batch, tokenizer.pad_token_id, cfg.device)
            loss = model(input_ids=input_ids, attention_mask=mask, labels=labels).loss
            (loss / cfg.grad_accum).backward()
            micro_step += 1
            progress.transitions += len(batch)
            if micro_step % cfg.grad_accum:
                continue
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            progress.actor_steps += 1
            if step % cfg.log_every == 0:
                log.write(step, progress, epoch=epoch, loss=loss.item(), lr=scheduler.get_last_lr()[0])
                print(f"[sft] epoch={epoch} step={step} loss={loss.item():.4f} elapsed={time.time() - t0:.0f}s")
            if cfg.snapshot_every_steps > 0 and step % cfg.snapshot_every_steps == 0:
                snapshot(epoch)
            if step % cfg.checkpoint_every_steps == 0:
                save_resume(output, lora=lora_state(model), optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(), rng=rng.getstate(), progress=asdict(progress),
                            step=step, micro_step=micro_step, order=order, epoch=epoch,
                            next_batch=start + cfg.batch_size)
        if micro_step % cfg.grad_accum:  # leftover accumulation window at the end of the epoch
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.actor_steps += 1

    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    record_warm_start(output, None)
    write_progress(output, progress, phase="final", epoch=cfg.epochs,
                   episodes_seen=round(progress.transitions / max(1, len(kept)) * episodes_per_pass))
    print(f"[sft] saved {output} ({time.time() - t0:.0f}s)")
    return output


def main(argv: list[str] | None = None) -> None:
    train(load_config(SFTConfig, argv, __doc__))


if __name__ == "__main__":
    main()
