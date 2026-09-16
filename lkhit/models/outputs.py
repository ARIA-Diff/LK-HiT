"""Container returned by every model so that the engine can treat them uniformly."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ModelOutput:
    logits: torch.Tensor
    label_attention: torch.Tensor | None = None  # [B, L, N] label-aware attention over segments
    document_attention: torch.Tensor | None = None  # [B, N, N] second-level self-attention (last layer, head-averaged)
    label_embeddings: torch.Tensor | None = None  # [L, d] refined label embeddings
    aux_loss: torch.Tensor | None = None  # auxiliary objectives of some baselines (e.g. LADAN community loss)
    aux_logits: torch.Tensor | None = None
