"""One JSON line per evaluated episode, next to each result file.

The result file holds aggregates; the records hold what they came from, so
confidence intervals, paired tests and the paper's tables run offline.
Records are keyed by `dialogue_idx:turn_idx`, unique within a split.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..env.actions import Answer, Ask
from ..env.oracle import gold_depth
from .rollout import EpisodeTrace


def episode_record(trace: EpisodeTrace, budget: int) -> dict:
    last = len(trace.steps) - 1
    steps = []
    for index, step in enumerate(trace.steps):
        record = {"action": "ask" if isinstance(step.action, Ask) else "answer",
                  "n_survivors_before": len(step.survivors_before), "n_survivors_after": len(step.survivors_after),
                  "reward": step.reward}
        if isinstance(step.action, Ask):
            record["attribute"] = step.action.attribute
        if isinstance(step.action, Answer):
            record["referent_id"] = step.action.referent_id
        if trace.forced_answer and index == last:
            record["forced"] = True
        if step.confidence is not None:
            record["referent_probs"] = {str(k): round(v, 5) for k, v in step.confidence.referent_probs.items()}
            if step.confidence.ask_score is not None:
                record["ask_score"] = round(step.confidence.ask_score, 5)
        if step.decision_q is not None:
            record["decision_q"] = {k: [round(q, 5) for q in v] for k, v in step.decision_q.items()}
        steps.append(record)
    return {
        "key": trace.episode.key,
        "condition": trace.episode.condition,
        "gold_action": trace.episode.gold_action,
        "gold_depth": gold_depth(trace.episode, budget),
        "gold_referent": trace.episode.gold_referent,
        "n_candidates": len(trace.episode.candidate_ids),
        "correct": trace.correct,
        "asks_used": trace.asks_used,
        "asked": trace.asks_used > 0,
        "forced_answer": trace.forced_answer,
        "total_reward": trace.total_reward,
        "steps": steps,
    }


def records_path(result_path: str | Path) -> Path:
    return Path(result_path).with_suffix(".episodes.jsonl")


def write_records(path: str | Path, records: list[dict]) -> None:
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in records))


def load_records(path: str | Path) -> list[dict] | None:
    path = Path(path)
    if not path.exists():
        return None
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
