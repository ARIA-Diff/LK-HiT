"""Discrimination, calibration, selective prediction, and paired bootstrap."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def probabilities(logits: np.ndarray, temperature: float, single_label: bool) -> np.ndarray:
    scaled = logits / temperature
    if single_label:
        scaled = scaled - scaled.max(axis=-1, keepdims=True)
        exp = np.exp(scaled)
        return exp / exp.sum(axis=-1, keepdims=True)
    return 1.0 / (1.0 + np.exp(-scaled))


def predict_labels(probs: np.ndarray, single_label: bool, threshold: float = 0.5) -> np.ndarray:
    if single_label:
        return probs.argmax(axis=-1)
    return (probs >= threshold).astype(np.int64)


def discrimination(probs: np.ndarray, targets: np.ndarray, single_label: bool, threshold: float = 0.5) -> dict:
    pred = predict_labels(probs, single_label, threshold)
    if single_label:
        return {
            "accuracy": float(accuracy_score(targets, pred)),
            "macro_precision": float(precision_score(targets, pred, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(targets, pred, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(targets, pred, average="macro", zero_division=0)),
        }
    return {
        "micro_f1": float(f1_score(targets, pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(targets, pred, average="macro", zero_division=0)),
    }


def _binary_ece(confidence: np.ndarray, correct: np.ndarray, n_bins: int) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(confidence)
    if total == 0:
        return 0.0
    error = 0.0
    for index in range(n_bins):
        if index == 0:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence > edges[index]) & (confidence <= edges[index + 1])
        count = int(mask.sum())
        if count == 0:
            continue
        error += (count / total) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(error)


def expected_calibration_error(probs: np.ndarray, targets: np.ndarray, single_label: bool, n_bins: int = 15) -> float:
    """Fifteen equal-width bins.

    Multi-label ECE is computed on every label-wise decision. CAIL2018 ECE uses
    the arg-max probability.
    """
    if single_label:
        confidence = probs.max(axis=-1)
        correct = (probs.argmax(axis=-1) == targets).astype(np.float64)
        return _binary_ece(confidence, correct, n_bins)
    pred = (probs >= 0.5).astype(np.float64)
    confidence = np.maximum(probs, 1.0 - probs).reshape(-1)
    correct = (pred == targets).astype(np.float64).reshape(-1)
    return _binary_ece(confidence, correct, n_bins)


def brier_score(probs: np.ndarray, targets: np.ndarray, single_label: bool) -> float:
    if single_label:
        one_hot = np.eye(probs.shape[1], dtype=np.float64)[targets]
        return float(np.mean(np.sum((probs - one_hot) ** 2, axis=-1)))
    return float(np.mean((probs - targets) ** 2))


def negative_log_likelihood(probs: np.ndarray, targets: np.ndarray, single_label: bool) -> float:
    clipped = np.clip(probs, 1e-8, 1.0 - 1e-8)
    if single_label:
        rows = np.arange(len(targets))
        return float(-np.mean(np.log(clipped[rows, targets])))
    log_lik = targets * np.log(clipped) + (1.0 - targets) * np.log(1.0 - clipped)
    return float(-np.mean(log_lik))


def document_uncertainty(probs: np.ndarray) -> np.ndarray:
    """One minus the confidence of the least confident label."""
    confidence = np.maximum(probs, 1.0 - probs)
    return 1.0 - confidence.min(axis=-1)


def jaccard_error(pred: np.ndarray, gold: np.ndarray) -> np.ndarray:
    errors = np.empty(len(pred), dtype=np.float64)
    for index in range(len(pred)):
        union = np.logical_or(pred[index] > 0, gold[index] > 0).sum()
        if union == 0:
            errors[index] = 0.0
            continue
        intersection = np.logical_and(pred[index] > 0, gold[index] > 0).sum()
        errors[index] = 1.0 - (intersection / union)
    return errors


def selective_prediction(
    probs: np.ndarray,
    targets: np.ndarray,
    single_label: bool,
    threshold: float = 0.5,
    risk_level: float = 0.10,
) -> dict:
    """Risk is 1 - Jaccard for multi-label tasks and 0/1 error for CAIL2018."""
    pred = predict_labels(probs, single_label, threshold)
    if single_label:
        errors = (pred != targets).astype(np.float64)
    else:
        errors = jaccard_error(pred, targets)
    uncertainty = document_uncertainty(probs)
    order = np.argsort(uncertainty, kind="mergesort")
    ordered = errors[order]
    n_docs = len(ordered)
    coverage = np.arange(1, n_docs + 1) / n_docs
    risk = np.cumsum(ordered) / np.arange(1, n_docs + 1)
    aurc = float(np.trapz(risk, coverage))
    acceptable = np.where(risk <= risk_level)[0]
    coverage_at_risk = float(coverage[acceptable[-1]]) if len(acceptable) else 0.0
    return {
        "aurc": aurc,
        "coverage_at_10_risk": coverage_at_risk,
        "uncertainty": uncertainty,
    }


def micro_f1_at_coverage(
    probs: np.ndarray,
    targets: np.ndarray,
    coverage: float,
    threshold: float = 0.5,
) -> float:
    """Micro-F1 on the most confident documents at a fixed coverage. Multi-label only."""
    uncertainty = document_uncertainty(probs)
    n_keep = max(1, int(round(coverage * len(probs))))
    chosen = np.argsort(uncertainty, kind="mergesort")[:n_keep]
    pred = (probs[chosen] >= threshold).astype(np.int64)
    return float(f1_score(targets[chosen], pred, average="micro", zero_division=0))


def frequency_tiers(train_counts: np.ndarray, task: str) -> dict[str, np.ndarray]:
    """EUR-LEX ranks 1–33 / 34–66 / 67–100. CAIL2018: ≥2000 / 500–1999 / 100–499."""
    if task == "eurlex":
        order = np.argsort(-train_counts, kind="mergesort")
        tiers = {"frequent": order[:33], "medium": order[33:66], "rare": order[66:]}
        return tiers
    if task == "cail2018":
        return {
            "frequent": np.where(train_counts >= 2000)[0],
            "medium": np.where((train_counts >= 500) & (train_counts <= 1999))[0],
            "rare": np.where((train_counts >= 100) & (train_counts <= 499))[0],
        }
    return {}


def tier_macro_f1(probs: np.ndarray, targets: np.ndarray, indices: np.ndarray, single_label: bool) -> float:
    if len(indices) == 0:
        return float("nan")
    if single_label:
        pred = probs.argmax(axis=-1)
        gold = targets
        scores = []
        for label in indices:
            binary_gold = (gold == label).astype(np.int64)
            binary_pred = (pred == label).astype(np.int64)
            scores.append(f1_score(binary_gold, binary_pred, zero_division=0))
        return float(np.mean(scores))
    pred = (probs[:, indices] >= 0.5).astype(np.int64)
    return float(f1_score(targets[:, indices], pred, average="macro", zero_division=0))


def per_label_f1(probs: np.ndarray, targets: np.ndarray, single_label: bool) -> np.ndarray:
    n_labels = probs.shape[1]
    scores = np.zeros(n_labels, dtype=np.float64)
    if single_label:
        pred = probs.argmax(axis=-1)
        for label in range(n_labels):
            scores[label] = f1_score(targets == label, pred == label, zero_division=0)
        return scores
    pred = (probs >= 0.5).astype(np.int64)
    for label in range(n_labels):
        scores[label] = f1_score(targets[:, label], pred[:, label], zero_division=0)
    return scores


def gain_frequency_correlation(gain: np.ndarray, train_counts: np.ndarray) -> dict:
    rho, p_value = spearmanr(gain, train_counts)
    return {"spearman_rho": float(rho), "p_value": float(p_value)}


def paired_bootstrap_macro_f1(
    targets: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    single_label: bool,
    n_resamples: int = 10000,
    seed: int = 0,
    threshold: float = 0.5,
) -> dict:
    """Two-sided paired bootstrap of the macro-F1 difference, system A minus system B."""

    def macro_f1(sample_targets: np.ndarray, sample_probs: np.ndarray) -> float:
        return discrimination(sample_probs, sample_targets, single_label, threshold)["macro_f1"]

    observed = macro_f1(targets, probs_a) - macro_f1(targets, probs_b)
    rng = np.random.default_rng(seed)
    n_docs = len(targets)
    diffs = np.empty(n_resamples, dtype=np.float64)
    for draw in range(n_resamples):
        index = rng.integers(0, n_docs, n_docs)
        diffs[draw] = macro_f1(targets[index], probs_a[index]) - macro_f1(targets[index], probs_b[index])
    if observed == 0:
        p_value = 1.0
    else:
        opposite = np.sign(diffs) == -np.sign(observed)
        p_value = float(min(1.0, 2.0 * opposite.mean()))
    low, high = np.quantile(diffs, [0.025, 0.975])
    return {
        "delta_macro_f1": float(observed),
        "ci95": [float(low), float(high)],
        "p_value": p_value,
    }
