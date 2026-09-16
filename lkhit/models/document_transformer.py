"""Second-level document transformer over paragraph states (Eq. 1 of the paper).

    H = g_phi([h_1 + u_1, ..., h_n + u_n])

with learned paragraph-position embeddings ``u_i`` and a two-layer,
eight-head post-LN transformer encoder (width 768, dropout 0.1). The layers
are written out rather than taken from ``nn.TransformerEncoder`` so that the
head-averaged self-attention of the last layer can be returned; the
hierarchical baseline uses it for rationale extraction.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DocumentTransformerLayer(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, ffn_size: int, dropout: float) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(hidden_size, n_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, ffn_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_size, hidden_size),
        )
        self.norm_attention = nn.LayerNorm(hidden_size, eps=1e-12)
        self.norm_ffn = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor, need_weights: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attended, weights = self.self_attention(
            x, x, x, key_padding_mask=key_padding_mask, need_weights=need_weights, average_attn_weights=True
        )
        x = self.norm_attention(x + self.dropout(attended))
        x = self.norm_ffn(x + self.dropout(self.feed_forward(x)))
        return x, weights


class DocumentTransformer(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        n_layers: int = 2,
        n_heads: int = 8,
        ffn_size: int | None = None,
        dropout: float = 0.1,
        max_segments: int = 64,
    ) -> None:
        super().__init__()
        ffn_size = ffn_size or 4 * hidden_size
        self.position_embeddings = nn.Embedding(max_segments, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [DocumentTransformerLayer(hidden_size, n_heads, ffn_size, dropout) for _ in range(n_layers)]
        )
        self.max_segments = max_segments
        nn.init.normal_(self.position_embeddings.weight, std=0.02)

    def forward(
        self, segment_states: torch.Tensor, segment_mask: torch.Tensor, return_attention: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``segment_states``: [B, N, d]; ``segment_mask``: [B, N] (True for real paragraphs)."""
        bsz, n_seg, _ = segment_states.shape
        if n_seg > self.max_segments:
            raise ValueError(f"{n_seg} segments exceed the position table of size {self.max_segments}")
        positions = torch.arange(n_seg, device=segment_states.device).unsqueeze(0).expand(bsz, n_seg)
        x = segment_states + self.position_embeddings(positions)
        x = self.input_dropout(self.input_norm(x))
        key_padding_mask = ~segment_mask.bool()
        # a document with no real segment would make every key masked; keep its first slot attendable
        empty = key_padding_mask.all(dim=1)
        if empty.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[empty, 0] = False
        attention = None
        for i, layer in enumerate(self.layers):
            last = i == len(self.layers) - 1
            x, weights = layer(x, key_padding_mask, need_weights=return_attention and last)
            if weights is not None:
                attention = weights
        x = x * segment_mask.unsqueeze(-1).to(x.dtype)
        return x, attention


def masked_mean(states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over the unmasked positions of ``[B, N, d]`` states."""
    mask = mask.to(states.dtype).unsqueeze(-1)
    total = (states * mask).sum(dim=1)
    count = mask.sum(dim=1).clamp(min=1.0)
    return total / count
