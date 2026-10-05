"""Does the policy know when it doesn't know, and does its critic track anything?

Both read per-step readings stored in the traces: the probe's referent
belief (`StepLog.confidence`) and the critic's Q values (`StepLog.decision_q`).
Two labels come from the environment: a decision is "would-be wrong" when the
probe's top referent isn't the gold one, and "still ambiguous" when more than
one candidate survives the replies so far.
"""

from __future__ import annotations

import math

from ..env.actions import Answer, Ask
from .rollout import EpisodeTrace

N_BINS = 10


def auroc(scores: list[float], labels: list[bool]) -> float | None:
    """P(score of a positive > score of a negative), ties counting half."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    rank_sum = sum(r for r, y in zip(ranks, labels) if y)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _bins(pairs: list[tuple[float, float]]) -> list[tuple[float, float, int]]:
    """(mean x, mean y, n) over equal-size bins of `pairs` sorted by x."""
    if not pairs:
        return []
    ordered = sorted(pairs)
    n_bins = min(N_BINS, len(ordered))
    size = len(ordered) // n_bins
    out = []
    for b in range(n_bins):
        chunk = ordered[b * size : (b + 1) * size if b < n_bins - 1 else len(ordered)]
        out.append((sum(x for x, _ in chunk) / len(chunk), sum(y for _, y in chunk) / len(chunk), len(chunk)))
    return out


def _binned_error(bins: list[tuple[float, float, int]]) -> float:
    total = sum(n for _, _, n in bins)
    return sum(n * abs(x - y) for x, y, n in bins) / total if total else 0.0


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def _forced(trace: EpisodeTrace, index: int) -> bool:
    return trace.forced_answer and index == len(trace.steps) - 1


def decision_calibration(traces: list[EpisodeTrace]) -> dict | None:
    """answer_ece: calibration of the confidence in the referent answered.
    auroc_wrong / auroc_ambiguous: how well 1 - top probability ranks the
    would-be-wrong / still-ambiguous decisions. ask_score_auroc: the same for
    the policy's own ask score. survivor_mass: probability the belief puts on
    still-consistent candidates. step0_*: at the first decision, the error of
    the policy's own answers against a threshold on its confidence at the same coverage."""
    probed = [(t, i, s) for t in traces for i, s in enumerate(t.steps) if s.confidence is not None]
    if not probed:
        return None
    answers, wrong_scores, wrong, ambiguous, ask_scores, ask_labels, mass = [], [], [], [], [], [], []
    for trace, index, step in probed:
        probs = step.confidence.referent_probs
        top_id, top_p = max(probs.items(), key=lambda kv: kv[1])
        is_ambiguous = len(step.survivors_before) > 1
        wrong_scores.append(1.0 - top_p)
        wrong.append(top_id != trace.episode.gold_referent)
        ambiguous.append(is_ambiguous)
        mass.append(sum(probs.get(c, 0.0) for c in step.survivors_before))
        if step.confidence.ask_score is not None:
            ask_scores.append(step.confidence.ask_score)
            ask_labels.append(is_ambiguous)
        elif step.decision_q is not None:
            ask_scores.append(min(step.decision_q["ask"]) - min(step.decision_q["answer"]))
            ask_labels.append(is_ambiguous)
        if isinstance(step.action, Answer) and not _forced(trace, index):
            answers.append((probs.get(step.action.referent_id, 0.0), float(step.action.referent_id == trace.episode.gold_referent)))

    rows = []  # step 0: (top probability, top wrong, answered, answer wrong)
    for trace in traces:
        if trace.steps and trace.steps[0].confidence is not None:
            step = trace.steps[0]
            top_id, top_p = max(step.confidence.referent_probs.items(), key=lambda kv: kv[1])
            answered = isinstance(step.action, Answer) and not _forced(trace, 0)
            rows.append((top_p, top_id != trace.episode.gold_referent, answered,
                         answered and step.action.referent_id != trace.episode.gold_referent))
    n_answered = sum(r[2] for r in rows)
    risks, errors = [], 0
    for k, row in enumerate(sorted(rows, key=lambda r: -r[0]), start=1):
        errors += row[1]
        risks.append(errors / k)
    bins = _bins(answers)
    return {
        "answer_ece": _binned_error(bins) if bins else None,
        "auroc_wrong": auroc(wrong_scores, wrong),
        "auroc_ambiguous": auroc(wrong_scores, ambiguous),
        "ask_score_auroc": auroc(ask_scores, ask_labels) if ask_scores else None,
        "survivor_mass": _mean(mass),
        "step0_policy_risk": sum(r[3] for r in rows) / n_answered if n_answered else None,
        "step0_threshold_risk": risks[n_answered - 1] if n_answered else None,
    }


def critic_diagnostics(traces: list[EpisodeTrace], lambda_penalty: float, budget: int) -> dict | None:
    """step0_gap_std: spread of Q(ask) - Q(answer) over first decisions (about 0
    for a critic that ignores the state). gap_auroc: how well that gap ranks
    still-ambiguous states. explained_variance / value_calibration_error: Q of
    the chosen branch against the return that followed. twin_gap: mean |Q_a - Q_b|.
    out_of_range: share of Q values outside the achievable [-1 - lambda*budget, 1]."""
    rows = [(t, i, s) for t in traces for i, s in enumerate(t.steps) if s.decision_q is not None]
    if not rows:
        return None
    low, high = -1.0 - lambda_penalty * budget, 1.0
    all_q, twin, gaps, labels, step0, pairs = [], [], [], [], [], []
    for trace, index, step in rows:
        ask, answer = step.decision_q["ask"], step.decision_q["answer"]
        all_q += ask + answer
        if len(ask) > 1:
            twin += [abs(ask[0] - ask[1]), abs(answer[0] - answer[1])]
        gap = min(ask) - min(answer)
        gaps.append(gap)
        labels.append(len(step.survivors_before) > 1)
        if index == 0:
            step0.append(gap)
        if not _forced(trace, index):
            chosen = ask if isinstance(step.action, Ask) else answer
            pairs.append((min(chosen), sum(s.reward for s in trace.steps[index:])))
    var_g = _variance([g for _, g in pairs])
    bins = _bins(pairs)
    return {
        "step0_gap_std": math.sqrt(_variance(step0)),
        "gap_auroc": auroc(gaps, labels),
        "explained_variance": (1.0 - _variance([g - q for q, g in pairs]) / var_g) if var_g > 0 else None,
        "value_calibration_error": _binned_error(bins) if bins else None,
        "twin_gap": _mean(twin),
        "out_of_range": sum(q < low - 1e-6 or q > high + 1e-6 for q in all_q) / len(all_q),
    }
