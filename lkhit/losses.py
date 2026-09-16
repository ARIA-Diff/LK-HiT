"""Training objectives: asymmetric loss for long-tail multi-label targets and class-balanced cross-entropy.

The multi-label tasks are trained with the asymmetric loss of Ridnik et al. (2021),

    L_ASL = 1/L * sum_l [ -y_l (1 - p_l)^{gamma+} log p_l
                         -(1 - y_l) p~_l^{gamma-} log(1 - p~_l) ],   p~_l = max(p_l - m, 0),

with gamma+ = 0, gamma- = 2 and margin m = 0.05, so that the many confident
negatives of a 100-label problem stop dominating the gradient. CAIL2018 is a
single-label task and uses cross-entropy with class-balanced weights
w_l = (1 - beta) / (1 - beta^{n_l}), beta = 0.999 (Cui et al., 2019).
Baselines use plain BCE / CE, as in their original papers.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 2.0, margin: float = 0.05, eps: float = 1e-8) -> None:
        super().__init__()
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.margin = float(margin)
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        targets = targets.float()
        p = torch.sigmoid(logits)
        p_neg = (p - self.margin).clamp(min=0.0) if self.margin > 0 else p

        log_p = torch.log(p.clamp(min=self.eps))
        log_one_minus = torch.log((1.0 - p_neg).clamp(min=self.eps))

        pos_term = targets * log_p
        neg_term = (1.0 - targets) * log_one_minus
        if self.gamma_pos > 0:
            pos_term = pos_term * torch.pow(1.0 - p, self.gamma_pos)
        if self.gamma_neg > 0:
            neg_term = neg_term * torch.pow(p_neg, self.gamma_neg)
        # mean over labels (1/L) and over the documents of the batch
        return -(pos_term + neg_term).mean()

    def extra_repr(self) -> str:
        return f"gamma_pos={self.gamma_pos}, gamma_neg={self.gamma_neg}, margin={self.margin}"


class BinaryCrossEntropy(nn.Module):
    """Plain BCE over label-wise logits, used by every multi-label baseline."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits.float(), targets.float())


def class_balanced_weights(counts: np.ndarray | torch.Tensor, beta: float = 0.999) -> torch.Tensor:
    """``(1 - beta) / (1 - beta^n_l)`` rescaled so that the weights sum to the number of classes."""
    counts = torch.as_tensor(np.asarray(counts), dtype=torch.float64).clamp(min=1.0)
    effective = 1.0 - torch.pow(torch.tensor(beta, dtype=torch.float64), counts)
    weights = (1.0 - beta) / effective
    weights = weights / weights.sum() * counts.numel()
    return weights.float()


class ClassBalancedCrossEntropy(nn.Module):
    def __init__(self, counts: np.ndarray | torch.Tensor, beta: float = 0.999) -> None:
        super().__init__()
        self.beta = float(beta)
        self.register_buffer("weight", class_balanced_weights(counts, beta))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits.float(), targets.long(), weight=self.weight)

    def extra_repr(self) -> str:
        return f"beta={self.beta}, n_classes={self.weight.numel()}"


class CrossEntropy(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits.float(), targets.long())


def build_loss(cfg: dict, multi_label: bool, train_counts: np.ndarray | None = None) -> nn.Module:
    """Instantiate the objective declared under ``loss`` in the configuration.

    ``loss.multi_label`` is ``asymmetric`` or ``bce``; ``loss.single_label`` is
    ``class_balanced_ce`` or ``ce``. Models that declare no ``loss`` section
    fall back to BCE / CE.
    """
    loss_cfg = cfg.get("loss", {}) or {}
    if multi_label:
        kind = loss_cfg.get("multi_label", "bce")
        if kind == "asymmetric":
            asl = loss_cfg.get("asymmetric", {}) or {}
            return AsymmetricLoss(
                gamma_pos=asl.get("gamma_pos", 0.0),
                gamma_neg=asl.get("gamma_neg", 2.0),
                margin=asl.get("margin", 0.05),
            )
        if kind == "bce":
            return BinaryCrossEntropy()
        raise ValueError(f"unknown multi-label loss '{kind}'")

    kind = loss_cfg.get("single_label", "ce")
    if kind == "class_balanced_ce":
        if train_counts is None:
            raise ValueError("class-balanced cross-entropy needs the training label counts")
        beta = (loss_cfg.get("class_balanced", {}) or {}).get("beta", 0.999)
        return ClassBalancedCrossEntropy(train_counts, beta=beta)
    if kind == "ce":
        return CrossEntropy()
    raise ValueError(f"unknown single-label loss '{kind}'")


def total_loss(main: torch.Tensor, aux: torch.Tensor | None, aux_weight: float) -> torch.Tensor:
    """Add the auxiliary objective some baselines return (e.g. the LADAN community loss)."""
    if aux is None or aux_weight == 0:
        return main
    return main + aux_weight * aux
