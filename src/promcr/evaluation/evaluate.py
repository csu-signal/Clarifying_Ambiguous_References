"""Evaluate a policy on a full split.

    python -m promcr.evaluation.evaluate --checkpoint checkpoints/bace-seed0 --split dev
    python -m promcr.evaluation.evaluate --checkpoint checkpoints/bace-seed0 --split dev --snapshots
    python -m promcr.evaluation.evaluate --policy natural-behavior --split devtest

Writes `<results>/<run>/<point>/<split>.json` (every metric, with 95%
bootstrap intervals) and `<split>.episodes.jsonl` (one record per episode).
`run` is the training run's directory name and `point` is `final` or the
snapshot's name (pass a snapshot directory to `--checkpoint` to score one
snapshot, or `--snapshots` to score them all). The method is read from the checkpoint
layout: Q heads and `bace_features.json` for BACE, Q heads alone for ArCHer,
a single adapter for SFT and GRPO.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from ..config import BUDGET, LAMBDA
from ..data.episodes import load_episodes
from .calibration import critic_diagnostics, decision_calibration
from .metrics import summarize
from .probe import confidence_probe
from .records import episode_record, records_path, write_records
from .rollout import run_split
from .stats import bootstrap_cis

METHODS = ("decision", "archer", "bace")


def detect_method(checkpoint: Path) -> str:
    if (checkpoint / "q_a_head").exists():
        return "bace" if (checkpoint / "bace_features.json").exists() else "archer"
    return "decision"


def load_policy(checkpoint: Path, method: str, budget: int, device: str):
    if method == "bace":
        from ..methods.bace.policy import load_bace_policy
        return load_bace_policy(checkpoint, budget, device)
    if method == "archer":
        from ..methods.archer.policy import load_archer_policy
        return load_archer_policy(checkpoint, budget, device)
    from ..policies.decision import load_decision_policy
    return load_decision_policy(checkpoint, device)


def load_prompted(name: str, budget: int, device: str):
    from ..models.backbone import load_backbone
    from ..policies.prompted import PROMPTED_POLICIES, UncertaintyGatedPolicy
    backbone = load_backbone(device)
    cls = PROMPTED_POLICIES[name]
    return cls(backbone, budget=budget) if cls is UncertaintyGatedPolicy else cls(backbone)


def evaluate(policy, split: str, out: Path, budget: int = BUDGET, lambda_penalty: float = LAMBDA,
             probe: bool = True, source: str | None = None) -> dict:
    t0 = time.time()
    traces = run_split(policy, load_episodes(split), lambda_penalty, budget, confidence_probe if probe else None)
    records = [episode_record(t, budget) for t in traces]
    summary = summarize(traces, budget)
    metrics = summary["metrics"]
    metrics |= decision_calibration(traces) or {}
    metrics |= critic_diagnostics(traces, lambda_penalty, budget) or {}
    result = {"source": source, "split": split, "budget": budget, "lambda_penalty": lambda_penalty, **summary,
              "ci95": {k: asdict(v) for k, v in bootstrap_cis(records).items()},
              "elapsed_s": round(time.time() - t0, 1)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    write_records(records_path(out), records)
    print(f"{source} on {split}: reward {metrics['reward']:.3f}, accuracy {metrics['accuracy']:.3f}, "
          f"questions {metrics['questions']:.2f}, Ask F1 {metrics['ask_f1']:.3f}, "
          f"hit {metrics['gold_turn_hit']:.3f} -> {out}")
    return result


def result_dir(results: Path, checkpoint: Path) -> Path:
    """results/<run>/final for a run directory, results/<run>/<snapshot> for one of its snapshots."""
    if checkpoint.parent.name == "snapshots":
        return results / checkpoint.parent.parent.name / checkpoint.name
    return results / checkpoint.name / "final"


def _release(device: str) -> None:
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--checkpoint", type=Path)
    what.add_argument("--policy", choices=("natural-behavior", "direct-answer", "ask-once", "uncertainty-gated", "procot"))
    parser.add_argument("--method", choices=METHODS, help="override the method read from the checkpoint")
    parser.add_argument("--snapshots", action="store_true", help="also evaluate every checkpoint under snapshots/")
    parser.add_argument("--split", default="dev", choices=("dev", "devtest"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--no-probe", action="store_true", help="skip the confidence probe (no calibration metrics)")
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--lambda-penalty", type=float, default=LAMBDA)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    if args.policy:
        policy = load_prompted(args.policy, args.budget, args.device)
        evaluate(policy, args.split, args.results / args.policy / "final" / f"{args.split}.json", args.budget,
                 args.lambda_penalty, not args.no_probe, args.policy)
        return
    paths = [args.checkpoint]
    if args.snapshots:
        paths += [p for p in sorted((args.checkpoint / "snapshots").iterdir()) if p.is_dir()]
    for path in paths:
        policy = load_policy(path, args.method or detect_method(path), args.budget, args.device)
        out = result_dir(args.results, path) / f"{args.split}.json"
        evaluate(policy, args.split, out, args.budget, args.lambda_penalty, not args.no_probe, str(path))
        del policy
        _release(args.device)


if __name__ == "__main__":
    main()
