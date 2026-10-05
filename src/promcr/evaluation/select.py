"""Choose the checkpoint that represents a method, from its runs' dev results.

    python -m promcr.evaluation.select checkpoints/bace-seed{0,1,7,17,123} --reference results/sft-real-synth/final

Each run is one seed; its points are the final checkpoint and its snapshots,
scored with `evaluate --snapshots`. Only points with RL behind them are
candidates (at least one optimizer step, not BACE's warm start). The rule:

1. Health gates (`GATES`). A point that fails one is out of the single-point
   ranking. A gate on a metric the method doesn't have is skipped.
2. Six metric families (`FAMILIES`). A family's score is the mean percentile
   of its metrics among the candidates (0 worst, 1 best), and points are
   ranked by their weakest family, then by the mean over families.

Two choices are printed. The single point ranks every (seed, point) that
passes the gates. The across-seed point is one point for all seeds: points
where any seed has collapsed (Ask F1 < 0.80 or over-asking > 0.05) are left
out, and the family scores are averaged over seeds. Choose on dev only, and
report the chosen point on devtest.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .records import load_records, records_path
from .stats import holm, paired_comparison

GATES = {  # metric: (comparison, threshold)
    "ask_f1": (">=", 0.95),
    "over_asking": ("<=", 0.02),
    "unparsed_asks": ("<=", 0.03),
    "out_of_range": ("<=", 0.10),
    "step0_gap_std": (">=", 0.10),
}
FAMILIES = {  # family: {metric: +1 if higher is better, -1 if lower is better}
    "decision": {"ask_f1": 1, "over_asking": -1, "under_asking": -1, "hallucinated_certainty": -1,
                 "gold_turn_hit": 1, "early": -1, "late": -1},
    "outcome": {"reward": 1, "accuracy": 1},
    "hard": {"hard_acc": 1, "depth2_acc": 1, "depth3_acc": 1, "depth2_hit": 1, "depth3_hit": 1},
    "questions": {"ig_ratio": 1, "ig_best": 1, "gold_path": 1, "entropy_reduction": 1, "redundancy": -1,
                  "unparsed_asks": -1, "guess_rate": -1},
    "calibration": {"answer_ece": -1, "auroc_wrong": 1, "auroc_ambiguous": 1, "ask_score_auroc": 1,
                    "step0_act_gap": -1, "survivor_mass": 1},
    "critic": {"gap_auroc": 1, "explained_variance": 1, "value_calibration_error": -1, "out_of_range": -1,
               "twin_gap": -1},
}
COLLAPSE_F1, COLLAPSE_OVER = 0.80, 0.05


@dataclass
class Candidate:
    seed: int
    point: str
    result: Path
    m: dict[str, float]
    failed: list[str]


def load_metrics(result: Path) -> dict[str, float] | None:
    if not result.exists():
        return None
    m = {k: v for k, v in json.loads(result.read_text())["metrics"].items() if isinstance(v, (int, float))}
    if "step0_policy_risk" in m and "step0_threshold_risk" in m:
        m["step0_act_gap"] = m["step0_policy_risk"] - m["step0_threshold_risk"]
    return m


def has_rl(point_dir: Path) -> bool:
    path = point_dir / "snapshot_progress.json"
    progress = json.loads(path.read_text()) if path.exists() else {}
    return progress.get("phase") != "warm_start" and (progress.get("actor_steps") or 0) + (progress.get("critic_steps") or 0) > 0


def candidates(runs: list[Path], results: Path, split: str) -> list[Candidate]:
    out = []
    for seed, run in enumerate(runs):
        points = [("final", run)] + [(p.name, p) for p in sorted((run / "snapshots").glob("*")) if p.is_dir()]
        for point, directory in points:
            result = results / run.name / point / f"{split}.json"
            if has_rl(directory) and (m := load_metrics(result)) is not None:
                failed = [f"{k} {m[k]:.3f}" for k, (op, t) in GATES.items()
                          if k in m and not (m[k] >= t if op == ">=" else m[k] <= t)]
                out.append(Candidate(seed, point, result, m, failed))
    return out


def family_scores(pool: list[Candidate]) -> dict[str, np.ndarray]:
    """Mean percentile of each family's metrics within `pool`, in pool order."""
    out = {}
    for family, directions in FAMILIES.items():
        ranks = []
        for metric, sign in directions.items():
            if not all(metric in c.m for c in pool):
                continue
            v = np.array([c.m[metric] * sign for c in pool])
            ranks.append(np.array([((v < x).sum() + 0.5 * ((v == x).sum() - 1)) / max(len(v) - 1, 1) for x in v]))
        if ranks:
            out[family] = np.mean(ranks, axis=0)
    return out


def rank_single(pool: list[Candidate]) -> list[tuple[Candidate, float, float]]:
    fam = family_scores(pool)
    stack = np.stack(list(fam.values()))
    weakest, mean = stack.min(axis=0), stack.mean(axis=0)
    order = sorted(range(len(pool)), key=lambda i: (-weakest[i], -mean[i]))
    return [(pool[i], float(weakest[i]), float(mean[i])) for i in order]


def rank_across(pool: list[Candidate]) -> list[tuple[str, float, float, list[Candidate]]]:
    collapsed = {c.point for c in pool if c.m["ask_f1"] < COLLAPSE_F1 or c.m["over_asking"] > COLLAPSE_OVER}
    pool = [c for c in pool if c.point not in collapsed]
    if not pool:
        return []
    fam = family_scores(pool)
    rows = []
    for point in dict.fromkeys(c.point for c in pool):
        idx = [i for i, c in enumerate(pool) if c.point == point]
        scores = [float(np.mean(fam[k][idx])) for k in fam]
        rows.append((point, min(scores), statistics.mean(scores), [pool[i] for i in idx]))
    return sorted(rows, key=lambda r: (-r[1], -r[2]))


def report(runs: list[Path], results: Path, split: str = "dev", reference: Path | None = None, top: int = 10) -> str:
    pool = candidates(runs, results, split)
    if not pool:
        return "No scored point with RL behind it."
    lines = [f"{sum(not c.failed for c in pool)} of {len(pool)} points pass the gates.", ""]
    lines += [f"- out: seed {c.seed} ({runs[c.seed].name}), {c.point}: {', '.join(c.failed)}" for c in pool if c.failed]

    lines += ["", "Across seeds:", "", "| Point | Seeds | Reward, mean ± SD | Min Ask F1 | Weakest family | Mean |",
              "| --- | --- | --- | --- | --- | --- |"]
    across = rank_across(pool)
    for rank, (point, weakest, mean, members) in enumerate(across):
        rewards = [c.m["reward"] for c in members]
        sd = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
        lines.append(f"| {point}{' (chosen)' if rank == 0 else ''} | {len(members)} of {len(runs)} | "
                     f"{statistics.mean(rewards):.3f} ± {sd:.3f} | {min(c.m['ask_f1'] for c in members):.3f} | "
                     f"{weakest:.2f} | {mean:.2f} |")
    if not across:
        lines.append("Every point has a collapsed seed.")

    kept = [c for c in pool if not c.failed]
    if not kept:
        return "\n".join(lines + ["", "No point passes the gates, so there is no single-point choice."])
    ranked = rank_single(kept)
    lines += ["", "Single point:", "", "| Rank | Seed | Point | Weakest family | Mean | Reward | Ask F1 | ≥2-q acc |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for rank, (c, weakest, mean) in enumerate(ranked[:top], 1):
        lines.append(f"| {rank} | {c.seed} | {c.point} | {weakest:.2f} | {mean:.2f} | {c.m['reward']:.3f} | "
                     f"{c.m['ask_f1']:.3f} | {c.m.get('hard_acc', float('nan')):.3f} |")

    winner = ranked[0][0]
    controls = [c.result for c, _, _ in ranked[1:2]]
    final = results / runs[winner.seed].name / "final" / f"{split}.json"
    if winner.point != "final" and final.exists():
        controls.append(final)
    if reference is not None:
        controls.append(reference / f"{split}.json")
    controls = list(dict.fromkeys(controls))
    if controls:
        lines += ["", f"Paired tests, control -> chosen ({winner.result.parent}):", ""]
        lines += paired_table([(c, winner.result) for c in controls])
    return "\n".join(lines)


def paired_table(pairs: list[tuple[Path, Path]]) -> list[str]:
    """Reward, accuracy and Ask F1 differences (treatment - control), Holm-adjusted over the table."""
    comparisons = [paired_comparison(load_records(records_path(c)), load_records(records_path(t))) for c, t in pairs]
    keys = ("mean_reward", "accuracy", "ask_f1")
    adjusted = iter(holm([cmp.deltas[k].p_value for cmp in comparisons for k in keys]))
    lines = ["| Control | Treatment | Reward | Accuracy | Ask F1 |", "| --- | --- | --- | --- | --- |"]
    for (control, treatment), cmp in zip(pairs, comparisons):
        cells = []
        for k in keys:
            d = cmp.deltas[k]
            cells.append(f"{d.delta:+.3f} [{d.low:+.3f}, {d.high:+.3f}] p{'<' if d.p_is_bound else '='}{next(adjusted):.2g}")
        lines.append(f"| {control.parent} | {treatment.parent} | " + " | ".join(cells) + " |")
    return lines


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="checkpoint directories, one per seed")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--split", default="dev")
    parser.add_argument("--reference", type=Path, help="a result point to pair the winner against, e.g. results/sft-real-synth/final")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args(argv)
    print(report(args.runs, args.results, args.split, args.reference, args.top))


if __name__ == "__main__":
    main()
