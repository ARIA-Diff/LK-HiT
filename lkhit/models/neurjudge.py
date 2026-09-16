"""NeurJudge (Yue et al., 2021): label-semantic circumstance separation for law-article prediction.

The fact and the statute texts are encoded with Bi-GRUs. Article
representations are made distinguishable by removing what each article shares
with the articles it resembles (a similarity graph over statute texts). The
fact tokens are then decomposed, per token, into the component aligned with
the article-related context (the description of the crime) and the orthogonal
residual (the circumstances); both parts are pooled and concatenated before
the article classifier. The charge and prison-term branches of the original
model are not part of the law-article protocol used in the paper.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lkhit.data.tasks import TaskSpec
from lkhit.models.ladan import article_similarity
from lkhit.models.outputs import ModelOutput
from lkhit.models.word_encoders import AttentivePooling, BiGRUEncoder, WordEmbedding, encode_label_texts


def similarity_neighbours(descriptions, threshold: float = 0.3, language: str = "zh") -> torch.Tensor:
    """Row-normalised neighbour matrix from the TF-IDF similarity of statute texts."""
    sim = article_similarity(descriptions, language)
    adj = (sim >= threshold).astype(np.float32)
    degree = adj.sum(axis=1, keepdims=True)
    adj = np.divide(adj, degree, out=np.zeros_like(adj), where=degree > 0)
    return torch.as_tensor(adj)


class LabelDistinguisher(nn.Module):
    """l_j <- ReLU( W_1 l_j - W_2 mean_{k in N_j} l_k )."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.self_transform = nn.Linear(dim, dim)
        self.neighbour_transform = nn.Linear(dim, dim, bias=False)

    def forward(self, labels: torch.Tensor, neighbours: torch.Tensor) -> torch.Tensor:
        shared = neighbours @ labels
        return F.relu(self.self_transform(labels) - self.neighbour_transform(shared))


class NeurJudge(nn.Module):
    encoder_prefixes: tuple[str, ...] = ()

    def __init__(
        self,
        model_cfg: dict,
        spec: TaskSpec,
        vocab,
        article_word_ids: torch.Tensor,
        article_word_mask: torch.Tensor,
        neighbours: torch.Tensor,
        pretrained: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.spec = spec
        dim = int(model_cfg.get("embedding_dim", 200))
        hidden = int(model_cfg.get("hidden_size", 128))
        dropout = float(model_cfg.get("dropout", 0.3))
        self.embedding = WordEmbedding(len(vocab), dim, vocab.pad_index, pretrained, dropout=float(model_cfg.get("embedding_dropout", 0.2)))
        self.fact_encoder = BiGRUEncoder(dim, hidden)
        self.article_encoder = BiGRUEncoder(dim, hidden)
        self.article_pooling = AttentivePooling(2 * hidden)
        self.distinguisher = LabelDistinguisher(2 * hidden)
        self.crime_pooling = AttentivePooling(2 * hidden)
        self.circumstance_pooling = AttentivePooling(2 * hidden)
        self.register_buffer("article_word_ids", article_word_ids.long())
        self.register_buffer("article_word_mask", article_word_mask.bool())
        self.register_buffer("neighbours", neighbours.float())
        self.classifier = nn.Linear(4 * hidden, spec.n_labels)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / np.sqrt(2 * hidden)

    def article_vectors(self) -> torch.Tensor:
        articles = encode_label_texts(self.embedding, self.article_encoder, self.article_pooling, self.article_word_ids, self.article_word_mask)
        return self.distinguisher(articles, self.neighbours)  # [L, 2h]

    def forward(self, batch: dict) -> ModelOutput:
        mask = batch["word_mask"].bool()
        states = self.fact_encoder(self.embedding(batch["word_ids"]), mask)  # [B, T, 2h]
        articles = self.article_vectors()

        # article-related context of every token: attention of the token over the distinguishable articles
        scores = torch.einsum("btd,ld->btl", states, articles) * self.scale
        context = torch.softmax(scores, dim=-1) @ articles  # [B, T, 2h]

        # per-token orthogonal decomposition: crime component along the context, circumstances as the residual
        unit = F.normalize(context, dim=-1)
        projection = (states * unit).sum(dim=-1, keepdim=True) * unit
        residual = states - projection

        crime, _ = self.crime_pooling(projection, mask)
        circumstances, _ = self.circumstance_pooling(residual, mask)
        logits = self.classifier(self.dropout(torch.cat([crime, circumstances], dim=-1)))
        return ModelOutput(logits=logits)

    def parameter_groups(self, lr_encoder: float, lr_upper: float, weight_decay: float) -> list[dict]:
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if name.endswith("bias") or "embedding" in name else decay).append(p)
        return [
            {"params": decay, "lr": lr_upper, "weight_decay": weight_decay, "name": "upper"},
            {"params": no_decay, "lr": lr_upper, "weight_decay": 0.0, "name": "upper_no_decay"},
        ]

    def parameter_summary(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad), "encoder": 0, "upper_layers": total}
