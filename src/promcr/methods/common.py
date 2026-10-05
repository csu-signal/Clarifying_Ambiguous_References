"""What every trainer shares: config loading, seeding, the LR schedule, progress counters and logs."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import time
import types
import typing
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml


# --- configuration -------------------------------------------------------------------------------

def _parse_value(kind, text: str):
    if kind is bool:
        if text.lower() in ("1", "true", "yes"):
            return True
        if text.lower() in ("0", "false", "no"):
            return False
        raise argparse.ArgumentTypeError(f"expected a boolean, got {text!r}")
    return kind(text)


def _field_type(cls, field):
    kind = typing.get_type_hints(cls)[field.name]
    if typing.get_origin(kind) in (typing.Union, types.UnionType):  # Optional[X]
        args = [a for a in typing.get_args(kind) if a is not type(None)]
        kind = args[0]
    return kind


def load_config(cls, argv: list[str] | None = None, description: str | None = None):
    """An instance of the dataclass `cls`: its defaults, then the YAML file
    given with `--config`, then any `--<field> value` on the command line."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, help="YAML file of field values")
    for field in dataclasses.fields(cls):
        kind = _field_type(cls, field)
        extra = {"nargs": "?", "const": True} if kind is bool else {}  # `--resume` alone means true
        parser.add_argument(f"--{field.name.replace('_', '-')}", dest=field.name, default=None,
                            type=lambda text, kind=kind: _parse_value(kind, text), metavar=kind.__name__.upper(),
                            **extra)
    args = parser.parse_args(argv)
    values = yaml.safe_load(args.config.read_text()) if args.config else {}
    unknown = set(values) - {f.name for f in dataclasses.fields(cls)}
    if unknown:
        parser.error(f"unknown keys in {args.config}: {sorted(unknown)}")
    values |= {k: v for k, v in vars(args).items() if k != "config" and v is not None}
    return cls(**values)


def save_config(config, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    record = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()}
    (output_dir / "config.json").write_text(json.dumps(record, indent=2) + "\n")


# --- randomness ------------------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def global_rng_state() -> dict:
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_global_rng(state: dict) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_initialized():
        torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def frozen(*modules):
    """eval() for the block (no LoRA dropout), restoring each module's mode afterwards.
    Rollouts, old-policy log-probabilities and target-network reads run under it."""
    previous = [(m, m.training) for m in modules if m is not None]
    for module, _ in previous:
        module.eval()
    try:
        yield
    finally:
        for module, was_training in previous:
            module.train(was_training)


# --- optimisation ------------------------------------------------------------------------------------

def linear_decay(optimizer: torch.optim.Optimizer, total_steps: int, min_factor: float = 0.1):
    """Linear decay to `min_factor` of the initial LR over `total_steps`, then flat."""
    def factor(step: int) -> float:
        return 1.0 - (1.0 - min_factor) * min(step / total_steps, 1.0) if total_steps > 0 else 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def polyak_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)


def polyak_update_adapter(model, target: str, source: str, tau: float) -> None:
    """The same update between two LoRA adapters of one PEFT model, matched by parameter name."""
    named = dict(model.named_parameters())
    with torch.no_grad():
        for name, param in named.items():
            if f".{target}." in name:
                param.data.mul_(1.0 - tau).add_(named[name.replace(f".{target}.", f".{source}.")].data, alpha=tau)


# --- resuming ---------------------------------------------------------------------------------------
#
# A trainer saves `_resume_state.pt` into its output directory at points where
# no gradient is half accumulated (or saves that gradient too). The file holds
# everything the rest of the run depends on: trainable weights, optimizer and
# scheduler states, every RNG, loop counters and buffers. A resumed run rebuilds
# the model as a fresh one would, then overwrites it from the file, so it
# continues the same random stream as an uninterrupted run.

RESUME_FILE = "_resume_state.pt"


def lora_state(model) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in model.named_parameters() if "lora_" in name}


def load_lora_state(model, state: dict[str, torch.Tensor]) -> None:
    named = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            named[name].copy_(value)


def gradients(params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: p.grad.detach().clone() for name, p in params.items() if p.grad is not None}


def load_gradients(params: dict[str, torch.Tensor], grads: dict[str, torch.Tensor]) -> None:
    for name, grad in grads.items():
        params[name].grad = grad.to(params[name].device)


def save_resume(output_dir: str | Path, **state) -> None:
    path = Path(output_dir) / RESUME_FILE
    tmp = path.with_name(path.name + ".tmp")
    torch.save({**state, "global_rng": global_rng_state()}, tmp)
    os.replace(tmp, path)  # a crash mid-write leaves the previous file intact


def load_resume(output_dir: str | Path, tag: str) -> dict | None:
    path = Path(output_dir) / RESUME_FILE
    if not path.exists():
        print(f"[{tag}] nothing to resume in {output_dir}; starting fresh")
        return None
    print(f"[{tag}] resuming from {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


# --- progress and logs --------------------------------------------------------------------------------

@dataclass
class Progress:
    """Cumulative training work, written next to every snapshot.

    episodes: training episodes rolled out (each counted once).
    transitions: decision points (RL) or supervised examples (SFT).
    actor_steps / critic_steps: optimizer steps on the policy's own
    parameters / on the critic.
    """

    episodes: int | None = 0
    transitions: int = 0
    actor_steps: int = 0
    critic_steps: int = 0


def write_progress(directory: str | Path, progress: Progress, **extra) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    record = {"episodes_seen": progress.episodes, "transitions_seen": progress.transitions,
              "actor_steps": progress.actor_steps, "critic_steps": progress.critic_steps, **extra}
    (directory / "snapshot_progress.json").write_text(json.dumps(record, indent=2) + "\n")


class MetricsLog:
    """Append-only JSONL of per-update training metrics, stamped with the progress counters."""

    def __init__(self, output_dir: str | Path, filename: str = "train_metrics.jsonl") -> None:
        self.path = Path(output_dir) / filename
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._t0 = time.time()

    def write(self, step: int, progress: Progress | None = None, **metrics) -> None:
        record = {"step": step, "elapsed_s": round(time.time() - self._t0, 1)}
        if progress is not None:
            record |= asdict(progress)
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            if isinstance(value, float) and not math.isfinite(value):
                value = None
            record[key] = value
        with self.path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
