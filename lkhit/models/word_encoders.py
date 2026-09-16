"""Word-level building blocks shared by the re-implemented Chinese LJP baselines (TopJudge, LADAN, NeurJudge)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class WordEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int, pad_index: int, pretrained: torch.Tensor | None = None, freeze: bool = False, dropout: float = 0.2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=pad_index)
        if pretrained is not None:
            if pretrained.shape != (vocab_size, dim):
                raise ValueError(f"pretrained vectors {tuple(pretrained.shape)} do not match ({vocab_size}, {dim})")
            self.embedding.weight.data.copy_(pretrained)
        else:
            nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
            self.embedding.weight.data[pad_index].zero_()
        self.embedding.weight.requires_grad_(not freeze)
        self.dropout = nn.Dropout(dropout)
        self.dim = dim

    def forward(self, word_ids: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.embedding(word_ids))


def load_word_vectors(vocab, model_cfg: dict, dim: int) -> torch.Tensor | None:
    path = model_cfg.get("pretrained_vectors")
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"pretrained word vectors {path} not found")
    return vocab.load_pretrained_embeddings(path, dim)


class CNNEncoder(nn.Module):
    """Multi-width convolutional encoder with max-over-time pooling (Kim, 2014), as used by TopJudge."""

    def __init__(self, input_dim: int, n_filters: int = 64, widths: tuple[int, ...] = (2, 3, 4, 5), dropout: float = 0.5) -> None:
        super().__init__()
        self.convolutions = nn.ModuleList([nn.Conv1d(input_dim, n_filters, kernel_size=w, padding=w // 2) for w in widths])
        self.dropout = nn.Dropout(dropout)
        self.output_dim = n_filters * len(widths)

    def forward(self, embedded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = embedded.transpose(1, 2)  # [B, dim, T]
        pooled = []
        for conv in self.convolutions:
            feature = F.relu(conv(x))[:, :, : mask.size(1)]
            feature = feature.masked_fill(~mask.unsqueeze(1), float("-inf"))
            pooled.append(feature.max(dim=2).values)
        out = torch.cat(pooled, dim=1)
        out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
        return self.dropout(out)


class BiGRUEncoder(nn.Module):
    """Bidirectional GRU returning contextual token states ``[B, T, 2h]``."""

    def __init__(self, input_dim: int, hidden_size: int = 128, n_layers: int = 1, dropout: float = 0.2) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.output_dim = 2 * hidden_size

    def forward(self, embedded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=embedded.size(1))
        return out


class AttentivePooling(nn.Module):
    """Additive attention pooling ``alpha_i = softmax(u^T tanh(W h_i [+ V q]))``, with an optional query."""

    def __init__(self, input_dim: int, attention_dim: int | None = None, query_dim: int | None = None) -> None:
        super().__init__()
        attention_dim = attention_dim or input_dim
        self.project = nn.Linear(input_dim, attention_dim)
        self.query_project = nn.Linear(query_dim, attention_dim, bias=False) if query_dim else None
        self.context = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, states: torch.Tensor, mask: torch.Tensor, query: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.project(states)
        if query is not None and self.query_project is not None:
            hidden = hidden + self.query_project(query).unsqueeze(1)
        scores = self.context(torch.tanh(hidden)).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.bmm(weights.unsqueeze(1), states).squeeze(1)
        return pooled, weights


def masked_max(states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    filled = states.masked_fill(~mask.unsqueeze(-1), torch.finfo(states.dtype).min)
    out = filled.max(dim=1).values
    return torch.where(torch.isfinite(out), out, torch.zeros_like(out))


def encode_label_texts(embedding: WordEmbedding, encoder: nn.Module, pooling: AttentivePooling, word_ids: torch.Tensor, word_mask: torch.Tensor, chunk: int = 64) -> torch.Tensor:
    """Vector per statute text; processed in chunks because CAIL has 103 articles of up to 200 words."""
    outputs = []
    for start in range(0, word_ids.size(0), chunk):
        ids, mask = word_ids[start : start + chunk], word_mask[start : start + chunk]
        states = encoder(embedding(ids), mask)
        pooled, _ = pooling(states, mask)
        outputs.append(pooled)
    return torch.cat(outputs, dim=0)
