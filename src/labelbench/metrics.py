"""Metrics for single-label classification, computed per hierarchy level.

The functions here are pure: they take gold and predicted labels and return
plain dictionaries that go straight into metrics.json and the report.
"""

from __future__ import annotations

from collections import Counter
from math import comb
from typing import Any

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, precision_recall_fscore_support, roc_auc_score

from labelbench.task import INVALID

N_BOOTSTRAP = 1000
# Bins for the reliability table and thresholds for the coverage table (probability of the prediction)
PROBABILITY_BINS = (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0)
PROBABILITY_THRESHOLDS = (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99)
MATRIX_MAX_CLASSES = 20
TOP_K = (1, 3, 5)


def evaluate(
    gold: list[str],
    pred: list[str],
    *,
    level_maps: dict[str, dict[str, str]],
    level_gold: dict[str, list[str]] | None = None,
    candidates: list[list[str]] | None = None,
    weights: list[float] | None = None,
    probabilities: list[float | None] | None = None,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """Metrics for the label itself (level "label") and every coarser level.

    `pred` must already be normalised: anything outside the label schema is
    INVALID. On coarser levels INVALID stays INVALID. If `level_gold` holds
    per-item gold values for a level, they replace the values derived from
    the gold label.
    """
    level_gold = level_gold or {}
    results = {"label": level_metrics(gold, pred, candidates, weights, seed)}
    if probabilities is not None:
        correct = [g == p for g, p in zip(gold, pred, strict=True)]
        results["label"]["probability"] = probability_metrics(correct, probabilities)
    for level, mapping in level_maps.items():
        g = level_gold.get(level) or [mapping[x] for x in gold]
        p = [mapping.get(x, INVALID) for x in pred]
        c = None
        if candidates is not None:
            c = [_dedupe(mapping.get(x, INVALID) for x in cands) for cands in candidates]
        results[level] = level_metrics(g, p, c, weights, seed)
    return results


def level_metrics(
    gold: list[str],
    pred: list[str],
    candidates: list[list[str]] | None,
    weights: list[float] | None,
    seed: int,
) -> dict[str, Any]:
    n = len(gold)
    g = np.array(gold, dtype=object)
    p = np.array(pred, dtype=object)
    correct = g == p
    gold_classes = sorted(set(gold))
    all_classes = sorted(set(gold) | set(pred))

    out: dict[str, Any] = {
        "n": n,
        "n_correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "macro_f1": _macro_f1(gold, pred),
        "macro_f1_classes": len(gold_classes),
        "kappa": _kappa(gold, pred),
        "invalid_share": float(np.mean(p == INVALID)),
        "n_classes_gold": len(gold_classes),
    }
    if weights is not None:
        w = np.array(weights, dtype=float)
        out["weighted_accuracy"] = float((w * correct).sum() / w.sum())

    ci = _bootstrap(gold, pred, correct, seed)
    out["accuracy_ci"] = ci["accuracy"]
    out["macro_f1_ci"] = ci["macro_f1"]
    out["kappa_ci"] = ci["kappa"]

    if candidates is not None:
        out["top_k"] = {}
        for k in TOP_K:
            hits = [gl in _dedupe([pl, *cs])[:k] for gl, pl, cs in zip(gold, pred, candidates, strict=True)]
            out["top_k"][str(k)] = float(np.mean(hits))

    out["per_class"] = _per_class(gold, pred, all_classes)
    out["confusions"] = [
        {"gold": gl, "pred": pl, "count": cnt}
        for (gl, pl), cnt in Counter((a, b) for a, b in zip(gold, pred, strict=True) if a != b).most_common()
    ]
    if len(all_classes) <= MATRIX_MAX_CLASSES:
        index = {c: i for i, c in enumerate(all_classes)}
        matrix = [[0] * len(all_classes) for _ in all_classes]
        for a, b in zip(gold, pred, strict=True):
            matrix[index[a]][index[b]] += 1
        out["matrix"] = {"labels": all_classes, "counts": matrix}
    return out


def probability_metrics(correct: list[bool], probs: list[float | None]) -> dict[str, Any]:
    """How well the classifier's probabilities match its actual accuracy.

    Only items with a probability count. `bins` is a reliability table: within
    each probability range, the mean stated probability against the share
    that is actually correct; for a well calibrated classifier both agree.
    `coverage` answers the practical question which accuracy one gets when
    only predictions above a threshold are accepted and the rest is checked
    by hand. ECE is the item-weighted mean gap between the two columns of
    `bins`, the Brier score the mean squared gap per item (0 is perfect).
    AUC measures discrimination instead: how well the probability separates
    correct from wrong predictions (0.5 is chance, 1 perfect). A cautious
    classifier can have a poor ECE and still a high AUC, which is what matters
    when low-probability predictions are to be checked by hand.
    """
    pairs = [(float(p), bool(c)) for p, c in zip(probs, correct, strict=True) if p is not None]
    out: dict[str, Any] = {"n": len(pairs), "n_missing": len(probs) - len(pairs)}
    if not pairs:
        return out
    p = np.array([x[0] for x in pairs])
    c = np.array([x[1] for x in pairs], dtype=float)
    out["mean_probability"] = float(p.mean())
    out["accuracy"] = float(c.mean())
    out["brier"] = float(np.mean((p - c) ** 2))
    out["auc"] = float(roc_auc_score(c, p)) if 0 < c.sum() < len(c) else None
    bins, ece = [], 0.0
    edges = PROBABILITY_BINS
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        last = i == len(edges) - 2
        mask = (p >= lo) & ((p <= hi) if last else (p < hi))
        k = int(mask.sum())
        row = {"from": lo, "to": hi, "n": k}
        if k:
            row["mean_probability"] = float(p[mask].mean())
            row["accuracy"] = float(c[mask].mean())
            ece += k / len(p) * abs(row["accuracy"] - row["mean_probability"])
        bins.append(row)
    out["bins"] = bins
    out["ece"] = float(ece)
    out["coverage"] = []
    for t in PROBABILITY_THRESHOLDS:
        mask = p >= t
        k = int(mask.sum())
        out["coverage"].append({"threshold": t, "n": k, "share": k / len(p),
                                "accuracy": float(c[mask].mean()) if k else None})
    return out


def mcnemar_exact(fixed: int, broken: int) -> float:
    """Two-sided exact McNemar test on the discordant pairs of two paired runs."""
    n = fixed + broken
    if n == 0:
        return 1.0
    k = min(fixed, broken)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, p)


def _macro_f1(gold: list[str], pred: list[str]) -> float:
    # Averaged over the classes that occur in the gold standard only; classes
    # that were merely predicted lower precision elsewhere but get no F1 of their own.
    return float(f1_score(gold, pred, labels=sorted(set(gold)), average="macro", zero_division=0))


def _kappa(gold: list[str], pred: list[str]) -> float | None:
    if len(set(gold) | set(pred)) < 2:
        return None
    value = cohen_kappa_score(gold, pred)
    return None if np.isnan(value) else float(value)


def _per_class(gold: list[str], pred: list[str], classes: list[str]) -> list[dict[str, Any]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        gold, pred, labels=classes, zero_division=np.nan
    )
    predicted = Counter(pred)
    hits = Counter(a for a, b in zip(gold, pred, strict=True) if a == b)
    rows = []
    for i, c in enumerate(classes):
        rows.append({
            "label": c,
            "support": int(support[i]),
            "predicted": predicted[c],
            "correct": hits[c],
            "precision": _num(precision[i]) if predicted[c] else None,
            "recall": _num(recall[i]) if support[i] else None,
            "f1": _num(f1[i]) if predicted[c] and support[i] else None,
        })
    return rows


def _bootstrap(gold: list[str], pred: list[str], correct: np.ndarray, seed: int) -> dict[str, list]:
    """95 % percentile intervals from resampling the items with replacement."""
    n = len(gold)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(N_BOOTSTRAP, n))
    acc = correct[idx].mean(axis=1)
    g = np.array(gold, dtype=object)
    p = np.array(pred, dtype=object)
    f1s, kappas = [], []
    for row in idx:
        gs, ps = list(g[row]), list(p[row])
        f1s.append(_macro_f1(gs, ps))
        k = _kappa(gs, ps)
        if k is not None:
            kappas.append(k)
    return {
        "accuracy": _interval(acc),
        "macro_f1": _interval(np.array(f1s)),
        "kappa": _interval(np.array(kappas)) if kappas else None,
    }


def _interval(values: np.ndarray) -> list[float]:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [float(lo), float(hi)]


def _num(x: float) -> float | None:
    return None if np.isnan(x) else float(x)


def _dedupe(values) -> list[str]:
    seen: dict[str, None] = {}
    for v in values:
        if v not in seen:
            seen[v] = None
    return list(seen)
