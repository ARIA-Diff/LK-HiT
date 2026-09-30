"""Post-hoc temperature scaling fitted on the development set."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _nll(logits: torch.Tensor, targets: torch.Tensor, temperature: torch.Tensor, single_label: bool) -> torch.Tensor:
    scaled = logits / temperature
    if single_label:
        return F.cross_entropy(scaled, targets.long())
    probs = torch.sigmoid(scaled).clamp(1e-8, 1.0 - 1e-8)
    return F.binary_cross_entropy(probs, targets.to(probs.dtype))


def fit_temperature(
    logits: torch.Tensor,
    targets: torch.Tensor,
    single_label: bool,
    max_iter: int = 50,
) -> float:
    """Minimise development negative log-likelihood in a positive scalar temperature.

    L-BFGS is run on log temperature so the temperature stays positive.
    Decision thresholds are not changed.
    """
    logits = logits.detach().float().cpu()
    targets = targets.detach().cpu()
    log_temperature = torch.nn.Parameter(torch.zeros(1))
    optimiser = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimiser.zero_grad()
        temperature = log_temperature.exp().clamp(min=1e-3)
        loss = _nll(logits, targets, temperature, single_label)
        loss.backward()
        return loss

    optimiser.step(closure)
    return float(log_temperature.detach().exp().clamp(min=1e-3).item())
