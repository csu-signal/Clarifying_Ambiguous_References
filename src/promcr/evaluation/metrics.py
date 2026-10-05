"""Metrics computed from episode traces.

When to ask: precision, recall and F1 of the ask decision against the
episodes that need a question, over-asking (asked on an episode that needed
none), under-asking (1 - recall) and hallucinated certainty (answered an
ambiguous episode without asking, and wrongly).

When to stop, against the episode's gold depth (questions the oracle asks
within the budget): hit (as many questions as the oracle), early, late,
hit-and-correct, and gold path (a hit whose every informative question was
tied for the best information gain on the policy's own survivors).

Question quality: entropy reduction per question, redundancy (an attribute
asked before in the episode or already pinned by the history), information-
gain ratio and IG-best rate against the best askable attribute, and the
share of questions whose attribute couldn't be parsed.
"""

from __future__ import annotations

import math
from collections import defaultdict

from ..env.actions import Ask
from ..env.attributes import TEXT_ASKABLE_ATTRS
from ..env.oracle import gold_depth, ranked_attributes
from .rollout import EpisodeTrace

# Ask-k groups: ask-3 also holds the episodes that need more than the budget or can't be resolved.
GROUPS = {"ask0": lambda d: d == 0, "ask1": lambda d: d == 1, "ask2": lambda d: d == 2,
          "ask3": lambda d: d >= 3, "multi": lambda d: d >= 2}


def _rate(count: float, total: float) -> float:
    return count / total if total else 0.0


def _log2(n: int) -> float:
    return math.log2(n) if n > 0 else 0.0


def detection(traces: list[EpisodeTrace]) -> dict[str, float]:
    tp = sum(t.episode.gold_action == "ask" and t.asks_used > 0 for t in traces)
    fn = sum(t.episode.gold_action == "ask" and t.asks_used == 0 for t in traces)
    fp = sum(t.episode.gold_action != "ask" and t.asks_used > 0 for t in traces)
    tn = len(traces) - tp - fn - fp
    precision, recall = _rate(tp, tp + fp), _rate(tp, tp + fn)
    hallucinated = sum(t.episode.gold_action == "ask" and t.asks_used == 0 and not t.correct for t in traces)
    return {
        "precision": precision,
        "recall": recall,
        "ask_f1": _rate(2 * precision * recall, precision + recall),
        "over_asking": _rate(fp, fp + tn),
        "under_asking": _rate(fn, fn + tp),
        "hallucinated_certainty": _rate(hallucinated, tp + fn),
    }


def _ask_scores(trace: EpisodeTrace) -> list[tuple[float, list]]:
    """(IG of the attribute asked, legal attributes ranked by IG) for every ask,
    on the survivors at that step. The ranking is empty when nothing could split them."""
    scores, seen = [], set()
    exclude = set(trace.episode.established_attrs) | trace.episode.request_slots
    for step in trace.steps:
        if not isinstance(step.action, Ask):
            continue
        ranked = [e for e in ranked_attributes(trace.episode, step.survivors_before, exclude | seen)
                  if e.attribute in TEXT_ASKABLE_ATTRS]
        if not ranked or ranked[0].ig <= 0:
            ranked = []
        match = next((e for e in ranked if e.attribute == step.action.attribute), None)
        scores.append((match.ig if match else 0.0, ranked))
        seen.add(step.action.attribute)
    return scores


def questions(traces: list[EpisodeTrace]) -> dict[str, float]:
    reductions, within, history, unparsed = [], 0, 0, 0
    ratios, best = [], 0
    for trace in traces:
        seen: set[str] = set()
        for step in trace.steps:
            if not isinstance(step.action, Ask):
                continue
            reductions.append(_log2(len(step.survivors_before)) - _log2(len(step.survivors_after)))
            unparsed += step.unparsed_ask
            if step.action.attribute in seen:
                within += 1
            elif step.action.attribute in trace.episode.established_attrs:
                history += 1
            seen.add(step.action.attribute)
        for chosen, ranked in _ask_scores(trace):
            if ranked:
                ratios.append(chosen / ranked[0].ig)
                best += chosen >= ranked[0].ig
    n = len(reductions)
    return {
        "n_asks": n,
        "entropy_reduction": _rate(sum(reductions), n),
        "redundancy": _rate(within + history, n),
        "ig_ratio": _rate(sum(ratios), len(ratios)),
        "ig_best": _rate(best, len(ratios)),
        "unparsed_asks": _rate(unparsed, n),
    }


def gold_turn(traces: list[EpisodeTrace], budget: int) -> dict:
    out = defaultdict(float)
    by_depth: dict[int, list[tuple[bool, bool, bool]]] = defaultdict(list)
    for trace in traces:
        depth = gold_depth(trace.episode, budget)
        hit = trace.asks_used == depth
        path = hit and all(not ranked or chosen >= ranked[0].ig for chosen, ranked in _ask_scores(trace))
        out["gold_turn_hit"] += hit
        out["early"] += trace.asks_used < depth
        out["late"] += trace.asks_used > depth
        out["hit_correct"] += hit and trace.correct
        out["gold_path"] += path
        by_depth[depth].append((hit, path, hit and trace.correct))
    result = {k: _rate(v, len(traces)) for k, v in out.items()}
    result["by_depth"] = {d: {"n": len(rows), "hit": sum(r[0] for r in rows) / len(rows),
                              "gold_path": sum(r[1] for r in rows) / len(rows),
                              "hit_correct": sum(r[2] for r in rows) / len(rows)}
                          for d, rows in sorted(by_depth.items())}
    return result


def summarize(traces: list[EpisodeTrace], budget: int) -> dict:
    """Every metric for one population, flat where the selection rule reads it."""
    n = len(traces)
    depths = [gold_depth(t.episode, budget) for t in traces]
    turn = gold_turn(traces, budget)
    metrics = {
        "n_episodes": n,
        "reward": _rate(sum(t.total_reward for t in traces), n),
        "accuracy": _rate(sum(t.correct for t in traces), n),
        "questions": _rate(sum(t.asks_used for t in traces), n),
        **detection(traces),
        **{k: v for k, v in turn.items() if k != "by_depth"},
        **questions(traces),
        "guess_rate": _rate(sum(len(t.steps[-1].survivors_before) > 1 for t in traces if t.steps), n),
        "hard_acc": _rate(sum(t.correct for t, d in zip(traces, depths) if d >= 2), sum(d >= 2 for d in depths)),
    }
    by_depth: dict[int, list[EpisodeTrace]] = defaultdict(list)
    for trace, depth in zip(traces, depths):
        by_depth[depth].append(trace)
    for depth in (2, 3):
        if depth in by_depth:
            metrics[f"depth{depth}_acc"] = _rate(sum(t.correct for t in by_depth[depth]), len(by_depth[depth]))
            metrics[f"depth{depth}_hit"] = turn["by_depth"][depth]["hit"]
    groups = {}
    for name, test in GROUPS.items():
        members = [t for t, d in zip(traces, depths) if test(d)]
        if members:
            groups[name] = {"n": len(members),
                            "hit": sum(t.asks_used == d for t, d in zip(traces, depths) if test(d)) / len(members),
                            "hit_correct": sum(t.asks_used == d and t.correct for t, d in zip(traces, depths) if test(d)) / len(members),
                            "accuracy": sum(t.correct for t in members) / len(members)}
    matrix = {d: {} for d in sorted(by_depth)}
    for depth, members in sorted(by_depth.items()):
        for asks in sorted({t.asks_used for t in members}):
            cell = [t for t in members if t.asks_used == asks]
            matrix[depth][asks] = {"n": len(cell), "accuracy": sum(t.correct for t in cell) / len(cell)}
    return {"metrics": metrics, "groups": groups, "gold_turn_by_depth": turn["by_depth"], "depth_by_asks": matrix}
