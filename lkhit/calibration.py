"""Post-hoc calibration: temperature scaling fitted on the development split, ECE, Brier score and NLL.

A single temperature per task is fitted by minimising the development-set
negative log-likelihood with L-BFGS (Guo et al., 2017); decision thresholds
stay at 0.5, so calibration never changes the F1 scores. ECE uses 15
equal-width bins (Naeini et al., 2015): over all label-wise probabilities for
the multi-label tasks, over the arg-max probability for CAIL2018.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from lkhit.data.tasks import TaskSpec
from lkhit.metrics import one_hot


# --------------------------------------------------------------------------- probabilities


def logits_to_probs(logits: np.ndarray, multi_label: bool, temperature: float = 1.0) -> np.ndarray:
    z = torch.as_tensor(np.asarray(logits), dtype=torch.float64) / float(temperature)
    if multi_label:
        return torch.sigmoid(z).numpy()
    return torch.softmax(z, dim=-1).numpy()


# --------------------------------------------------------------------------- temperature scaling


@dataclass
class TemperatureScaler:
    multi_label: bool
    temperature: float = 1.0
    max_iter: int = 50
    lr: float = 0.01

    def nll(self, logits: torch.Tensor, targets: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        z = logits / temperature
        if self.multi_label:
            return F.binary_cross_entropy_with_logits(z, targets.float())
        return F.cross_entropy(z, targets.long())

    def fit(self, logits: np.ndarray, targets: np.ndarray) -> "TemperatureScaler":
        logits_t = torch.as_tensor(np.asarray(logits), dtype=torch.float64)
        targets_t = torch.as_tensor(np.asarray(targets))
        if not self.multi_label and targets_t.ndim == 2:
            targets_t = targets_t.argmax(dim=1)
        # optimise log T so the temperature stays positive
        log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        optimiser = torch.optim.LBFGS([log_t], lr=self.lr, max_iter=self.max_iter, line_search_fn="strong_wolfe")

        def closure():
            optimiser.zero_grad()
            loss = self.nll(logits_t, targets_t, torch.exp(log_t))
            loss.backward()
            return loss

        optimiser.step(closure)
        self.temperature = float(torch.exp(log_t).item())
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return logits_to_probs(logits, self.multi_label, self.temperature)

    def to_dict(self) -> dict:
        return {"method": "temperature_scaling", "temperature": self.temperature, "multi_label": self.multi_label}

    @classmethod
    def from_dict(cls, payload: dict) -> "TemperatureScaler":
        return cls(multi_label=bool(payload["multi_label"]), temperature=float(payload["temperature"]))


# --------------------------------------------------------------------------- calibration metrics


def _bin_edges(n_bins: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_bins + 1)


def reliability_bins(confidence: np.ndarray, correctness: np.ndarray, n_bins: int = 15) -> dict[str, np.ndarray]:
    """Per-bin mean confidence, empirical accuracy and count (for reliability diagrams)."""
    confidence = np.asarray(confidence, dtype=np.float64).ravel()
    correctness = np.asarray(correctness, dtype=np.float64).ravel()
    edges = _bin_edges(n_bins)
    idx = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins).astype(np.float64)
    conf_sum = np.bincount(idx, weights=confidence, minlength=n_bins)
    acc_sum = np.bincount(idx, weights=correctness, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_conf = np.where(counts > 0, conf_sum / np.maximum(counts, 1), np.nan)
        mean_acc = np.where(counts > 0, acc_sum / np.maximum(counts, 1), np.nan)
    return {"edges": edges, "confidence": mean_conf, "accuracy": mean_acc, "count": counts}


def expected_calibration_error(confidence: np.ndarray, correctness: np.ndarray, n_bins: int = 15) -> float:
    bins = reliability_bins(confidence, correctness, n_bins)
    counts = bins["count"]
    total = counts.sum()
    if total == 0:
        return float("nan")
    gap = np.abs(np.nan_to_num(bins["accuracy"]) - np.nan_to_num(bins["confidence"]))
    return float((counts / total * gap).sum())


def confidence_and_correctness(probs: np.ndarray, gold: np.ndarray, spec: TaskSpec) -> tuple[np.ndarray, np.ndarray]:
    """Multi-label: every label-wise probability against its binary target.
    Single-label: the arg-max probability against arg-max correctness."""
    probs = np.asarray(probs, dtype=np.float64)
    gold_matrix = one_hot(gold, probs.shape[1])
    if spec.multi_label:
        return probs.ravel(), gold_matrix.astype(np.float64).ravel()
    pred = probs.argmax(axis=1)
    return probs.max(axis=1), (pred == gold_matrix.argmax(axis=1)).astype(np.float64)


def brier_score(probs: np.ndarray, gold: np.ndarray, spec: TaskSpec) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    target = one_hot(gold, probs.shape[1]).astype(np.float64)
    sq = (probs - target) ** 2
    if spec.multi_label:
        return float(sq.mean())
    return float(sq.sum(axis=1).mean())


def negative_log_likelihood(probs: np.ndarray, gold: np.ndarray, spec: TaskSpec, eps: float = 1e-12) -> float:
    probs = np.clip(np.asarray(probs, dtype=np.float64), eps, 1 - eps)
    target = one_hot(gold, probs.shape[1]).astype(np.float64)
    if spec.multi_label:
        return float(-(target * np.log(probs) + (1 - target) * np.log(1 - probs)).mean())
    return float(-np.log(probs[np.arange(len(probs)), target.argmax(axis=1)]).mean())


def calibration_metrics(probs: np.ndarray, gold: np.ndarray, spec: TaskSpec, n_bins: int = 15) -> dict[str, float]:
    confidence, correctness = confidence_and_correctness(probs, gold, spec)
    return {
        "ece": 100 * expected_calibration_error(confidence, correctness, n_bins),
        "brier": brier_score(probs, gold, spec),
        "nll": negative_log_likelihood(probs, gold, spec),
    }


def reliability_table(probs: np.ndarray, gold: np.ndarray, spec: TaskSpec, n_bins: int = 15) -> dict[str, list[float]]:
    confidence, correctness = confidence_and_correctness(probs, gold, spec)
    bins = reliability_bins(confidence, correctness, n_bins)
    return {
        "bin_lower": bins["edges"][:-1].tolist(),
        "bin_upper": bins["edges"][1:].tolist(),
        "confidence": [None if np.isnan(v) else float(v) for v in bins["confidence"]],
        "accuracy": [None if np.isnan(v) else float(v) for v in bins["accuracy"]],
        "count": [int(c) for c in bins["count"]],
    }
