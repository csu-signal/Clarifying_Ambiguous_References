"""Comparison tables from per-episode records, and paired tests.

    python -m promcr.evaluation.report table \\
        --row "Natural Behavior=results/natural-behavior/final" \\
        --row "SFT=results/sft-real-synth/final" \\
        --row "BACE=results/bace-seed0/step_004,results/bace-seed1/step_004,..."
    python -m promcr.evaluation.report groups --row ...
    python -m promcr.evaluation.report compare results/sft-real-synth/final results/bace-seed0/final --split devtest
    python -m promcr.evaluation.report curves --run checkpoints/bace-seed0 --run ... --metric ask_f1 --out curves.png

A row lists one result point per seed; a cell is the mean over them with the
SD in parentheses. Dev and devtest are printed one under the other.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .metrics import GROUPS
from .records import load_records, records_path
from .stats import holm, paired_comparison

COLUMNS = (("precision", "Prec."), ("recall", "Rec."), ("f1", "Ask F1"), ("over", "Over"), ("under", "Under"),
           ("hallucinated", "Halluc."), ("hit", "Hit"), ("early", "Early"), ("late", "Late"),
           ("hit_correct", "Hit+corr."), ("accuracy", "Acc."), ("questions", "Qs"), ("reward", "Reward"))
SPLITS = ("dev", "devtest")


def episode_metrics(records: list[dict]) -> dict[str, float]:
    """The paper's metrics from episode records, plus hit and hit-and-correct per ask-k group."""
    n = len(records)
    ambiguous = [r for r in records if r["gold_depth"] >= 1]
    unambiguous = [r for r in records if r["gold_depth"] == 0]
    asked = [r for r in records if r["asks_used"] > 0]
    precision = sum(r["gold_depth"] >= 1 for r in asked) / len(asked) if asked else 0.0
    recall = sum(r["asks_used"] > 0 for r in ambiguous) / len(ambiguous)
    out = {
        "reward": sum(r["total_reward"] for r in records) / n,
        "accuracy": sum(r["correct"] for r in records) / n,
        "questions": sum(r["asks_used"] for r in records) / n,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "over": sum(r["asks_used"] > 0 for r in unambiguous) / len(unambiguous),
        "under": 1 - recall,
        "hallucinated": sum(r["asks_used"] == 0 and not r["correct"] for r in ambiguous) / len(ambiguous),
        "hit": sum(r["asks_used"] == r["gold_depth"] for r in records) / n,
        "early": sum(r["asks_used"] < r["gold_depth"] for r in records) / n,
        "late": sum(r["asks_used"] > r["gold_depth"] for r in records) / n,
        "hit_correct": sum(r["asks_used"] == r["gold_depth"] and r["correct"] for r in records) / n,
    }
    for name, test in GROUPS.items():
        group = [r for r in records if test(r["gold_depth"])]
        if group:
            out[f"{name}_n"] = len(group)
            out[f"{name}_hit"] = sum(r["asks_used"] == r["gold_depth"] for r in group) / len(group)
            out[f"{name}_hit_correct"] = sum(r["asks_used"] == r["gold_depth"] and r["correct"] for r in group) / len(group)
    return out


def _records(point: Path, split: str) -> list[dict] | None:
    return load_records(records_path(point / f"{split}.json"))


def _cell(values: list[float], fmt: str = ".3f") -> str:
    if not values:
        return "--"
    if len(values) == 1:
        return format(values[0], fmt)
    return f"{statistics.mean(values):{fmt}} ({statistics.stdev(values):{fmt}})"


def _parse_rows(rows: list[str]) -> list[tuple[str, list[Path]]]:
    out = []
    for row in rows:
        label, _, points = row.partition("=")
        out.append((label, [Path(p) for p in points.split(",") if p]))
    return out


def table(rows: list[tuple[str, list[Path]]], columns) -> str:
    lines = ["| Method | Seeds | " + " | ".join(name for _, name in columns) + " |",
             "| --- | --- | " + " | ".join("---" for _ in columns) + " |"]
    for split in SPLITS:
        lines.append(f"| *{split}* |" + " |" * (len(columns) + 1))
        for label, points in rows:
            scored = [episode_metrics(records) for records in (_records(p, split) for p in points) if records]
            cells = [_cell([m[key] for m in scored if key in m], ".2f" if key == "questions" else ".3f") for key, _ in columns]
            lines.append(f"| {label} | {len(scored)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def compare(control: Path, treatment: Path, split: str) -> str:
    result = paired_comparison(_records(control, split), _records(treatment, split))
    adjusted = holm([d.p_value for d in result.deltas.values()])
    lines = [f"{treatment} - {control} on {split}, {result.n_paired} paired episodes "
             f"(only control right: {result.only_control_correct}, only treatment right: {result.only_treatment_correct})", ""]
    lines += [f"  {name}: {d.delta:+.3f} [{d.low:+.3f}, {d.high:+.3f}] p{'<' if d.p_is_bound else '='}{p:.2g} (Holm)"
              for (name, d), p in zip(result.deltas.items(), adjusted)]
    return "\n".join(lines)


def curves(runs: list[Path], metric: str, out: Path, results: Path = Path("results"), split: str = "dev") -> None:
    """`metric` against training episodes, one line per run, from the snapshots' results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for run in runs:
        points = []
        for directory in [*sorted((run / "snapshots").glob("*")), run]:
            progress, result = directory / "snapshot_progress.json", results / run.name / (
                "final" if directory == run else directory.name) / f"{split}.json"
            if progress.exists() and result.exists():
                episodes = json.loads(progress.read_text()).get("episodes_seen")
                value = json.loads(result.read_text())["metrics"].get(metric)
                if episodes is not None and value is not None:
                    points.append((episodes, value))
        points.sort()
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=run.name)
    ax.set_xscale("symlog")
    ax.set_xlabel("training episodes")
    ax.set_ylabel(f"{metric} ({split})")
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("table", "groups"):
        p = sub.add_parser(name)
        p.add_argument("--row", action="append", required=True, help="LABEL=point[,point...]")
    p = sub.add_parser("compare")
    p.add_argument("control", type=Path)
    p.add_argument("treatment", type=Path)
    p.add_argument("--split", default="dev")
    p = sub.add_parser("curves")
    p.add_argument("--run", action="append", type=Path, required=True)
    p.add_argument("--metric", default="reward")
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--out", type=Path, default=Path("curves.png"))
    args = parser.parse_args(argv)

    if args.command == "table":
        print(table(_parse_rows(args.row), COLUMNS))
    elif args.command == "groups":
        columns = [(f"{g}_{k}", f"{g} {'Hit' if k == 'hit' else 'H+C'}") for g in GROUPS for k in ("hit", "hit_correct")]
        print(table(_parse_rows(args.row), columns))
    elif args.command == "compare":
        print(compare(args.control, args.treatment, args.split))
    else:
        curves(args.run, args.metric, args.out, args.results)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
