"""Uncertainty from per-episode records (`records.py`).

- `bootstrap_cis`: percentile bootstrap over episodes for accuracy, average
  questions, mean reward, Ask F1 and over-/under-asking. Decoding is greedy
  and BACE's sampling is seeded per episode, so the episodes evaluated are the
  only randomness at evaluation time.
- `paired_comparison`: two policies on the same episodes. Exact McNemar test
  for accuracy, paired bootstrap for the other metrics.
- `holm`: Holm-Bonferroni adjustment over a set of p-values.

Seed-to-seed spread is not covered here; it needs retrained seeds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

DEFAULT_BOOTSTRAP = 10_000
DEFAULT_SEED = 0
LEVEL = 0.95
_CHUNK = 1_000  # resamples per vectorised batch; bounds memory at ~CHUNK x n int64

METRICS = ("accuracy", "avg_questions", "mean_reward", "ask_f1", "over_asking_rate", "under_asking_rate")


def _arrays(records: list[dict]) -> dict[str, np.ndarray]:
    return {
        "correct": np.array([r["correct"] for r in records], dtype=float),
        "asks": np.array([r["asks_used"] for r in records], dtype=float),
        "reward": np.array([r["total_reward"] for r in records], dtype=float),
        "gold_ask": np.array([r["gold_action"] == "ask" for r in records], dtype=float),
        "asked": np.array([r["asked"] for r in records], dtype=float),
    }


def _metric_matrix(a: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    """Every metric for each resample (row of `idx`), vectorised."""
    correct, asks, reward = a["correct"][idx], a["asks"][idx], a["reward"][idx]
    gold, asked = a["gold_ask"][idx], a["asked"][idx]
    tp = (gold * asked).sum(1)
    fp = ((1 - gold) * asked).sum(1)
    fn = (gold * (1 - asked)).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
        n_gold_answer = (1 - gold).sum(1)
        over = np.where(n_gold_answer > 0, fp / n_gold_answer, 0.0)
        under = np.where(tp + fn > 0, fn / (tp + fn), 0.0)
    return {
        "accuracy": correct.mean(1),
        "avg_questions": asks.mean(1),
        "mean_reward": reward.mean(1),
        "ask_f1": f1,
        "over_asking_rate": over,
        "under_asking_rate": under,
    }


def _resamples(n: int, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    done = 0
    while done < n_boot:
        size = min(_CHUNK, n_boot - done)
        yield rng.integers(0, n, size=(size, n))
        done += size


@dataclass
class Interval:
    point: float
    low: float
    high: float

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.low:.3f}, {self.high:.3f}]"


def bootstrap_cis(records: list[dict], n_boot: int = DEFAULT_BOOTSTRAP, seed: int = DEFAULT_SEED,
                  level: float = LEVEL) -> dict[str, Interval]:
    """95% percentile-bootstrap interval for every metric in `METRICS`."""
    if not records:
        return {}
    a = _arrays(records)
    point = {k: float(v[0]) for k, v in _metric_matrix(a, np.arange(len(records))[None, :]).items()}
    draws = {k: [] for k in METRICS}
    for idx in _resamples(len(records), n_boot, seed):
        for k, v in _metric_matrix(a, idx).items():
            draws[k].append(v)
    tail = (1 - level) / 2 * 100
    out = {}
    for k in METRICS:
        values = np.concatenate(draws[k])
        out[k] = Interval(point[k], float(np.percentile(values, tail)), float(np.percentile(values, 100 - tail)))
    return out


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts: b = only
    the first policy right, c = only the second right."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


@dataclass
class PairedDelta:
    delta: float  # treatment - control
    low: float
    high: float
    p_value: float  # McNemar for accuracy, paired bootstrap otherwise
    p_is_bound: bool = False  # bootstrap p below its 1/n_boot resolution: p_value is that bound

    def p_text(self) -> str:
        return f"p<{self.p_value:.0e}" if self.p_is_bound else f"p={self.p_value:.2g}"

    def __str__(self) -> str:
        return f"{self.delta:+.3f} [{self.low:+.3f}, {self.high:+.3f}] {self.p_text()}"


@dataclass
class PairedComparison:
    n_paired: int
    n_control_only: int  # episodes the other policy was not scored on (left out)
    n_treatment_only: int
    only_control_correct: int
    only_treatment_correct: int
    deltas: dict[str, PairedDelta]


def paired_comparison(control: list[dict], treatment: list[dict], n_boot: int = DEFAULT_BOOTSTRAP,
                      seed: int = DEFAULT_SEED, level: float = LEVEL) -> PairedComparison:
    """Treatment minus control on the episodes both were scored on."""
    c_by_key = {r["key"]: r for r in control}
    t_by_key = {r["key"]: r for r in treatment}
    keys = [k for k in c_by_key if k in t_by_key]
    if not keys:
        raise ValueError("no episodes in common")
    c = _arrays([c_by_key[k] for k in keys])
    t = _arrays([t_by_key[k] for k in keys])
    everyone = np.arange(len(keys))[None, :]
    point_c, point_t = _metric_matrix(c, everyone), _metric_matrix(t, everyone)
    diffs = {k: [] for k in METRICS}
    for idx in _resamples(len(keys), n_boot, seed):
        mc, mt = _metric_matrix(c, idx), _metric_matrix(t, idx)
        for k in METRICS:
            diffs[k].append(mt[k] - mc[k])
    only_c = int(((c["correct"] == 1) & (t["correct"] == 0)).sum())
    only_t = int(((c["correct"] == 0) & (t["correct"] == 1)).sum())
    tail = (1 - level) / 2 * 100
    deltas = {}
    for k in METRICS:
        values = np.concatenate(diffs[k])
        bound = False
        if k == "accuracy":
            p = mcnemar_exact(only_c, only_t)
        else:  # two-sided: how often the resampled difference crosses zero
            p = min(1.0, 2 * min(float((values <= 0).mean()), float((values >= 0).mean())))
            if p == 0.0:
                p, bound = 1.0 / len(values), True
        deltas[k] = PairedDelta(float(point_t[k][0] - point_c[k][0]), float(np.percentile(values, tail)),
                                float(np.percentile(values, 100 - tail)), p, bound)
    return PairedComparison(len(keys), len(c_by_key) - len(keys), len(t_by_key) - len(keys), only_c, only_t, deltas)


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted
