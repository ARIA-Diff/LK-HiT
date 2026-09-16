"""Discrimination metrics, frequency tiers, subgroup scores, seed aggregation and paired bootstrap.

Multi-label tasks are scored with micro- and macro-F1 at a fixed 0.5 threshold
(LexGLUE convention); CAIL2018 with accuracy, macro-precision, macro-recall and
macro-F1 (LADAN convention). Nothing here tunes thresholds, so the same
probabilities feed the calibration metrics in :mod:`lkhit.calibration`.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
from scipy import stats

from lkhit.data.tasks import TaskSpec


# --------------------------------------------------------------------------- decisions


def decisions_from_probs(probs: np.ndarray, spec: TaskSpec, threshold: float = 0.5) -> np.ndarray:
    """Binary decision matrix ``[n_docs, n_labels]`` (multi-label) or one-hot arg-max (single-label)."""
    probs = np.asarray(probs)
    if spec.multi_label:
        return (probs >= threshold).astype(np.int64)
    out = np.zeros_like(probs, dtype=np.int64)
    out[np.arange(len(probs)), probs.argmax(axis=1)] = 1
    return out


def one_hot(labels: np.ndarray, n_labels: int) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim == 2:
        return labels.astype(np.int64)
    out = np.zeros((len(labels), n_labels), dtype=np.int64)
    out[np.arange(len(labels)), labels.astype(np.int64)] = 1
    return out


# --------------------------------------------------------------------------- F1 family


def _prf(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
        recall = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-12), 0.0)
    return precision, recall, f1


def confusion_counts(pred: np.ndarray, gold: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = np.asarray(pred).astype(bool)
    gold = np.asarray(gold).astype(bool)
    tp = (pred & gold).sum(axis=0).astype(np.float64)
    fp = (pred & ~gold).sum(axis=0).astype(np.float64)
    fn = (~pred & gold).sum(axis=0).astype(np.float64)
    return tp, fp, fn


def per_label_f1(pred: np.ndarray, gold: np.ndarray) -> np.ndarray:
    tp, fp, fn = confusion_counts(pred, gold)
    return _prf(tp, fp, fn)[2]


def micro_f1(pred: np.ndarray, gold: np.ndarray) -> float:
    tp, fp, fn = confusion_counts(pred, gold)
    return float(_prf(tp.sum(), fp.sum(), fn.sum())[2])


def macro_f1(pred: np.ndarray, gold: np.ndarray, label_subset: Sequence[int] | None = None) -> float:
    tp, fp, fn = confusion_counts(pred, gold)
    if label_subset is not None:
        idx = np.asarray(list(label_subset), dtype=np.int64)
        tp, fp, fn = tp[idx], fp[idx], fn[idx]
    return float(_prf(tp, fp, fn)[2].mean()) if tp.size else 0.0


def macro_precision_recall(pred: np.ndarray, gold: np.ndarray) -> tuple[float, float]:
    tp, fp, fn = confusion_counts(pred, gold)
    precision, recall, _ = _prf(tp, fp, fn)
    return float(precision.mean()), float(recall.mean())


def accuracy(pred: np.ndarray, gold: np.ndarray) -> float:
    pred = np.asarray(pred)
    gold = np.asarray(gold)
    if pred.ndim == 2:
        pred = pred.argmax(axis=1)
    if gold.ndim == 2:
        gold = gold.argmax(axis=1)
    return float((pred == gold).mean()) if len(pred) else 0.0


def jaccard_per_document(pred: np.ndarray, gold: np.ndarray) -> np.ndarray:
    """Jaccard index between predicted and gold label sets; 1 when both sets are empty."""
    pred = np.asarray(pred).astype(bool)
    gold = np.asarray(gold).astype(bool)
    inter = (pred & gold).sum(axis=1).astype(np.float64)
    union = (pred | gold).sum(axis=1).astype(np.float64)
    return np.where(union > 0, inter / np.maximum(union, 1), 1.0)


def document_errors(pred: np.ndarray, gold: np.ndarray, spec: TaskSpec) -> np.ndarray:
    """Per-document error used as the selective-prediction risk: ``1 - Jaccard`` or ``1[y_hat != y]``."""
    if spec.multi_label:
        return 1.0 - jaccard_per_document(pred, gold)
    return (np.asarray(pred).argmax(axis=1) != np.asarray(gold).argmax(axis=1)).astype(np.float64)


def discrimination_metrics(pred: np.ndarray, gold: np.ndarray, spec: TaskSpec) -> dict[str, float]:
    """The headline scores reported in the paper, in percentage points."""
    out = {"micro_f1": 100 * micro_f1(pred, gold), "macro_f1": 100 * macro_f1(pred, gold)}
    if spec.multi_label:
        out["mean_jaccard"] = 100 * float(jaccard_per_document(pred, gold).mean())
        out["n_predicted_per_doc"] = float(np.asarray(pred).sum(axis=1).mean())
    else:
        mp, mr = macro_precision_recall(pred, gold)
        out["accuracy"] = 100 * accuracy(pred, gold)
        out["macro_precision"] = 100 * mp
        out["macro_recall"] = 100 * mr
    return out


# --------------------------------------------------------------------------- frequency tiers


TIER_NAMES = ("frequent", "medium", "rare")


def label_tiers(train_counts: np.ndarray, scheme: dict, exclude: Iterable[int] = ()) -> dict[str, list[int]]:
    """Group labels into frequent / medium / rare tiers.

    ``scheme = {"scheme": "rank", "boundaries": [33, 66]}`` uses frequency
    ranks (EUR-LEX: ranks 1-33 / 34-66 / 67-100);
    ``scheme = {"scheme": "count", "boundaries": [500, 2000]}`` uses training
    counts (CAIL2018: >= 2000 / 500-1999 / 100-499). Labels in ``exclude``
    (the explicit ``none`` label) are left out of every tier.
    """
    train_counts = np.asarray(train_counts)
    kind = (scheme or {}).get("scheme", "none")
    excluded = set(int(i) for i in exclude)
    labels = [i for i in range(len(train_counts)) if i not in excluded]
    tiers: dict[str, list[int]] = {name: [] for name in TIER_NAMES}
    if kind == "none" or not labels:
        return tiers
    boundaries = list(scheme["boundaries"])
    if kind == "rank":
        order = sorted(labels, key=lambda i: (-train_counts[i], i))
        for rank, label in enumerate(order, start=1):
            if rank <= boundaries[0]:
                tiers["frequent"].append(label)
            elif rank <= boundaries[1]:
                tiers["medium"].append(label)
            else:
                tiers["rare"].append(label)
        return tiers
    if kind == "count":
        low, high = sorted(boundaries)
        for label in labels:
            n = train_counts[label]
            if n >= high:
                tiers["frequent"].append(label)
            elif n >= low:
                tiers["medium"].append(label)
            else:
                tiers["rare"].append(label)
        return tiers
    raise ValueError(f"unknown tier scheme '{kind}'")


def tier_metrics(pred: np.ndarray, gold: np.ndarray, tiers: dict[str, list[int]]) -> dict[str, dict[str, float]]:
    out = {}
    for name, members in tiers.items():
        if members:
            out[name] = {"macro_f1": 100 * macro_f1(pred, gold, members), "n_labels": len(members)}
    return out


def gain_frequency_correlation(gain: np.ndarray, train_counts: np.ndarray, labels: Sequence[int]) -> dict[str, float]:
    """Spearman correlation between per-label F1 gain and training frequency."""
    idx = np.asarray(list(labels), dtype=np.int64)
    if len(idx) < 3:
        return {"spearman_rho": float("nan"), "p_value": float("nan"), "n_labels": int(len(idx))}
    rho, p = stats.spearmanr(np.asarray(gain)[idx], np.asarray(train_counts)[idx])
    return {"spearman_rho": float(rho), "p_value": float(p), "n_labels": int(len(idx))}


# --------------------------------------------------------------------------- subgroups


def subgroup_metrics(
    pred: np.ndarray,
    gold: np.ndarray,
    spec: TaskSpec,
    group_of_doc: Sequence[str | None],
) -> dict[str, dict[str, float]]:
    """Headline metric per subgroup (decision year, length tier ...), skipping unlabelled documents."""
    groups = np.asarray([g if g is not None else "" for g in group_of_doc], dtype=object)
    out: dict[str, dict[str, float]] = {}
    for name in sorted(set(groups.tolist())):
        if not name:
            continue
        mask = groups == name
        sub = discrimination_metrics(np.asarray(pred)[mask], np.asarray(gold)[mask], spec)
        sub["n"] = int(mask.sum())
        out[str(name)] = sub
    return out


def length_tier_name(value: int, boundaries: Sequence[int], unit: str) -> str:
    bounds = list(boundaries)
    if value <= bounds[0]:
        return f"<={bounds[0]} {unit}"
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if value <= hi:
            return f"{lo + 1}-{hi} {unit}"
    return f">{bounds[-1]} {unit}"


# --------------------------------------------------------------------------- seeds and bootstrap


def aggregate_seeds(records: Sequence[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Mean and standard deviation of every numeric entry across seed-level metric dictionaries."""
    keys = sorted({k for r in records for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
    out = {}
    for key in keys:
        values = np.asarray([r[key] for r in records if key in r], dtype=np.float64)
        out[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0, "n": int(len(values))}
    return out


def _per_document_counts(pred: np.ndarray, gold: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred).astype(bool)
    gold = np.asarray(gold).astype(bool)
    tp = (pred & gold).astype(np.float64)
    fp = (pred & ~gold).astype(np.float64)
    fn = (~pred & gold).astype(np.float64)
    return np.stack([tp, fp, fn], axis=-1)  # [n_docs, n_labels, 3]


def _macro_f1_from_counts(counts: np.ndarray) -> np.ndarray:
    """``counts``: [..., n_labels, 3] -> macro-F1 over the label axis."""
    tp, fp, fn = counts[..., 0], counts[..., 1], counts[..., 2]
    denom = 2 * tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0)
    return f1.mean(axis=-1)


def paired_bootstrap(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    gold: np.ndarray,
    n_resamples: int = 10_000,
    seed: int = 0,
    chunk: int = 250,
) -> dict[str, float]:
    """Paired bootstrap over test documents for the macro-F1 difference ``A - B``.

    Documents are resampled with replacement, macro-F1 is recomputed for both
    systems on each resample, and the two-sided p-value is twice the proportion
    of resamples whose difference has the opposite sign to the observed one
    (Koehn, 2004). Returns the observed difference, the 95% percentile interval
    and the p-value, all in percentage points.
    """
    counts_a = _per_document_counts(pred_a, gold)
    counts_b = _per_document_counts(pred_b, gold)
    n_docs = counts_a.shape[0]
    observed = 100 * float(_macro_f1_from_counts(counts_a.sum(0)) - _macro_f1_from_counts(counts_b.sum(0)))
    rng = np.random.RandomState(seed)
    flat_a = counts_a.reshape(n_docs, -1)
    flat_b = counts_b.reshape(n_docs, -1)
    diffs = np.empty(n_resamples, dtype=np.float64)
    done = 0
    while done < n_resamples:
        m = min(chunk, n_resamples - done)
        weights = rng.multinomial(n_docs, np.full(n_docs, 1.0 / n_docs), size=m).astype(np.float64)
        agg_a = (weights @ flat_a).reshape(m, -1, 3)
        agg_b = (weights @ flat_b).reshape(m, -1, 3)
        diffs[done : done + m] = 100 * (_macro_f1_from_counts(agg_a) - _macro_f1_from_counts(agg_b))
        done += m
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    if observed >= 0:
        opposite = float((diffs < 0).mean())
    else:
        opposite = float((diffs > 0).mean())
    p_value = min(1.0, 2 * opposite)
    return {
        "delta_macro_f1": observed,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": p_value,
        "n_resamples": int(n_resamples),
        "n_documents": int(n_docs),
    }


def seed_averaged_probabilities(prob_files: Sequence[np.ndarray]) -> np.ndarray:
    """Average the probability matrices of several seeds (used before the bootstrap test)."""
    stack = np.stack([np.asarray(p, dtype=np.float64) for p in prob_files])
    return stack.mean(axis=0)


def format_pp(value: float, digits: int = 1) -> str:
    return "nan" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.{digits}f}"
