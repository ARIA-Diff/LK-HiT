"""Selective prediction: document uncertainty, risk-coverage curves, AURC and coverage at a target risk.

The uncertainty of a document is the confidence of its least confident label,

    u(x) = 1 - min_l max(p_l, 1 - p_l)          (multi-label)
    u(x) = 1 - max_l p_l                        (single-label),

and the system abstains when ``u(x) > tau_u``. Sweeping ``tau_u`` traces the
risk-coverage curve (Geifman & El-Yaniv, 2017), where coverage is the fraction
of accepted documents and risk the mean per-document error among them
(``1 - Jaccard`` for label sets, ``1[y_hat != y]`` for CAIL2018).
"""

from __future__ import annotations

import numpy as np

from lkhit.data.tasks import TaskSpec
from lkhit.metrics import decisions_from_probs, document_errors, micro_f1, one_hot


def uncertainty_score(probs: np.ndarray, multi_label: bool) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    if multi_label:
        confidence = np.maximum(probs, 1.0 - probs).min(axis=1)
    else:
        confidence = probs.max(axis=1)
    return 1.0 - confidence


def risk_coverage_curve(uncertainty: np.ndarray, errors: np.ndarray) -> dict[str, np.ndarray]:
    """Coverage and selective risk when accepting the ``k`` most confident documents, ``k = 1..n``."""
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    order = np.argsort(uncertainty, kind="stable")
    cumulative = np.cumsum(errors[order])
    k = np.arange(1, len(order) + 1, dtype=np.float64)
    coverage = k / len(order)
    risk = cumulative / k
    return {"coverage": coverage, "risk": risk, "threshold": uncertainty[order], "order": order}


def aurc(uncertainty: np.ndarray, errors: np.ndarray) -> float:
    """Area under the risk-coverage curve (mean selective risk over all coverage levels)."""
    curve = risk_coverage_curve(uncertainty, errors)
    return float(curve["risk"].mean())


def excess_aurc(uncertainty: np.ndarray, errors: np.ndarray) -> float:
    """AURC minus the AURC of an oracle that orders documents by their true error."""
    curve = risk_coverage_curve(uncertainty, errors)
    oracle = risk_coverage_curve(np.asarray(errors, dtype=np.float64), errors)
    return float(curve["risk"].mean() - oracle["risk"].mean())


def coverage_at_risk(uncertainty: np.ndarray, errors: np.ndarray, target_risk: float = 0.10) -> float:
    """Largest coverage whose selective risk does not exceed ``target_risk``."""
    curve = risk_coverage_curve(uncertainty, errors)
    ok = np.where(curve["risk"] <= target_risk)[0]
    return float(curve["coverage"][ok.max()]) if ok.size else 0.0


def risk_at_coverage(uncertainty: np.ndarray, errors: np.ndarray, coverage: float) -> float:
    curve = risk_coverage_curve(uncertainty, errors)
    k = max(1, int(round(coverage * len(curve["risk"]))))
    return float(curve["risk"][k - 1])


def f1_at_coverage(
    probs: np.ndarray, gold: np.ndarray, spec: TaskSpec, coverage_levels: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
) -> dict[str, float]:
    """Micro-F1 (accuracy for single-label) on the accepted documents at fixed coverage levels."""
    probs = np.asarray(probs, dtype=np.float64)
    gold_matrix = one_hot(gold, probs.shape[1])
    pred = decisions_from_probs(probs, spec)
    unc = uncertainty_score(probs, spec.multi_label)
    order = np.argsort(unc, kind="stable")
    out = {}
    for level in coverage_levels:
        k = max(1, int(round(level * len(order))))
        keep = order[:k]
        if spec.multi_label:
            score = 100 * micro_f1(pred[keep], gold_matrix[keep])
        else:
            score = 100 * float((pred[keep].argmax(1) == gold_matrix[keep].argmax(1)).mean())
        out[f"{int(round(level * 100))}"] = score
    return out


def selective_metrics(probs: np.ndarray, gold: np.ndarray, spec: TaskSpec, target_risk: float = 0.10) -> dict[str, float]:
    probs = np.asarray(probs, dtype=np.float64)
    gold_matrix = one_hot(gold, probs.shape[1])
    pred = decisions_from_probs(probs, spec)
    errors = document_errors(pred, gold_matrix, spec)
    unc = uncertainty_score(probs, spec.multi_label)
    return {
        "aurc": 100 * aurc(unc, errors),
        "excess_aurc": 100 * excess_aurc(unc, errors),
        f"coverage_at_{int(round(target_risk * 100))}pct_risk": 100 * coverage_at_risk(unc, errors, target_risk),
        "risk_at_full_coverage": 100 * float(errors.mean()),
        "risk_at_80pct_coverage": 100 * risk_at_coverage(unc, errors, 0.8),
        "risk_at_60pct_coverage": 100 * risk_at_coverage(unc, errors, 0.6),
    }


def risk_coverage_table(probs: np.ndarray, gold: np.ndarray, spec: TaskSpec, n_points: int = 101) -> dict[str, list[float]]:
    """Risk-coverage curve resampled on an even coverage grid (for storage alongside predictions)."""
    probs = np.asarray(probs, dtype=np.float64)
    gold_matrix = one_hot(gold, probs.shape[1])
    pred = decisions_from_probs(probs, spec)
    errors = document_errors(pred, gold_matrix, spec)
    unc = uncertainty_score(probs, spec.multi_label)
    curve = risk_coverage_curve(unc, errors)
    grid = np.linspace(0.0, 1.0, n_points)[1:]
    risk = np.interp(grid, curve["coverage"], curve["risk"])
    return {"coverage": grid.tolist(), "risk": risk.tolist()}
