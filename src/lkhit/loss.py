"""Asymmetric loss and class-balanced cross-entropy."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def asymmetric_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma_pos: float = 0.0,
    gamma_neg: float = 2.0,
    margin: float = 0.05,
) -> torch.Tensor:
    """Focal-style multi-label loss with a probability margin on negatives.

    The mean is taken over labels, matching the 1/L factor in the paper.
    Temperature is 1 during training; logits are unscaled.
    """
    probs = torch.sigmoid(logits).clamp(1e-8, 1.0 - 1e-8)
    targets = targets.to(probs.dtype)
    positive = -targets * (1.0 - probs).pow(gamma_pos) * torch.log(probs)
    shifted = (probs - margin).clamp(min=0.0, max=1.0 - 1e-8)
    negative = -(1.0 - targets) * shifted.pow(gamma_neg) * torch.log(1.0 - shifted)
    return (positive + negative).mean()


def class_balanced_weights(frequencies: torch.Tensor, beta: float = 0.999) -> torch.Tensor:
    """w_l proportional to (1 - beta) / (1 - beta^{n_l}), scaled to mean 1."""
    freq = frequencies.to(dtype=torch.float64).clamp(min=1)
    weights = (1.0 - beta) / (1.0 - torch.pow(torch.tensor(beta, dtype=torch.float64), freq))
    weights = weights * (weights.numel() / weights.sum())
    return weights.to(dtype=torch.float32)


def classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    cfg: dict,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    kind = cfg["loss"]
    if kind == "asl":
        return asymmetric_loss(
            logits,
            targets,
            gamma_pos=float(cfg["asl_gamma_pos"]),
            gamma_neg=float(cfg["asl_gamma_neg"]),
            margin=float(cfg["asl_margin"]),
        )
    if kind == "bce":
        return F.binary_cross_entropy_with_logits(logits, targets.to(logits.dtype))
    if kind == "ce":
        return F.cross_entropy(logits, targets.long())
    if kind == "cbce":
        return F.cross_entropy(logits, targets.long(), weight=class_weights)
    raise ValueError(f"Unknown loss {kind!r}")
