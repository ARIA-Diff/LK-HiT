"""Label-aware attention over paragraph states and the bilinear label scorer (Eqs. 3-4 of the paper).

    alpha_{l i} = softmax_i( e_l^T W_a h_i / sqrt(d) ),   d_l = sum_i alpha_{l i} h_i
    s_l = d_l^T W_o e_l + b_l

Single head, ``1/sqrt(d)`` scaling. The attention weights are kept in the model
output because they double as paragraph rationales.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LabelAwareAttention(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.query_transform = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(hidden_size)
        nn.init.xavier_uniform_(self.query_transform.weight)

    def forward(
        self, label_embeddings: torch.Tensor, segment_states: torch.Tensor, segment_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return label-specific summaries ``[B, L, d]`` and attention weights ``[B, L, N]``."""
        queries = self.query_transform(label_embeddings)  # [L, d] = E W_a
        scores = torch.einsum("ld,bnd->bln", queries.to(segment_states.dtype), segment_states) * self.scale
        mask = segment_mask.bool().unsqueeze(1)  # [B, 1, N]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=-1).to(segment_states.dtype)
        weights = weights * mask.to(weights.dtype)  # padded documents without any segment stay at zero
        summaries = torch.einsum("bln,bnd->bld", self.dropout(weights), segment_states)
        return summaries, weights


class BilinearScorer(nn.Module):
    """``s_l = d_l^T W_o e_l + b_l`` for label-specific summaries, or ``h^T W_o e_l + b_l`` for a pooled document."""

    def __init__(self, hidden_size: int, n_labels: int) -> None:
        super().__init__()
        self.output_transform = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(n_labels))
        nn.init.xavier_uniform_(self.output_transform.weight)

    def forward(self, summaries: torch.Tensor, label_embeddings: torch.Tensor) -> torch.Tensor:
        projected = self.output_transform(label_embeddings).to(summaries.dtype)  # [L, d] = E W_o^T
        if summaries.dim() == 3:  # [B, L, d]
            logits = (summaries * projected.unsqueeze(0)).sum(dim=-1)
        else:  # [B, d] pooled document
            logits = summaries @ projected.T
        return logits + self.bias.to(logits.dtype)


class LabelSpecificOutput(nn.Module):
    """Per-label output layer of LSAN: ``s_l = w_l^T tanh(W d_l) + b_l`` with a label-shared hidden layer."""

    def __init__(self, hidden_size: int, n_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.label_weights = nn.Parameter(torch.empty(n_labels, hidden_size))
        self.bias = nn.Parameter(torch.zeros(n_labels))
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.label_weights)

    def forward(self, summaries: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.hidden(self.dropout(summaries)))  # [B, L, d]
        return (hidden * self.label_weights.to(hidden.dtype).unsqueeze(0)).sum(dim=-1) + self.bias.to(hidden.dtype)
