"""LADAN (Xu et al., 2020): law-article communities, graph distillation and community-guided re-encoding.

1. Articles whose statute texts are similar (TF-IDF cosine above a threshold)
   are linked; communities are found with the Louvain method.
2. A graph distillation operator (GDO) removes the features that articles in
   the same community share, leaving distinguishable article representations;
   pooling them gives one distinguishing vector per community.
3. A basic Bi-GRU encoder reads the fact and predicts the community; the fact
   is then re-encoded with attention guided by the distinguishing vector of the
   (gold at training time, predicted at test time) community.
4. Basic and distinguishing fact representations are concatenated to predict
   the law article. The community prediction is returned as an auxiliary loss.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lkhit.data.tasks import TaskSpec
from lkhit.models.outputs import ModelOutput
from lkhit.models.word_encoders import AttentivePooling, BiGRUEncoder, WordEmbedding, encode_label_texts

logger = logging.getLogger("lkhit.models.ladan")


# --------------------------------------------------------------------------- article communities


def article_similarity(descriptions: Sequence[str], language: str = "zh") -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer

    if language == "zh":
        from lkhit.data.word_level import tokenize_zh

        vectoriser = TfidfVectorizer(tokenizer=tokenize_zh, token_pattern=None, sublinear_tf=True)
    else:
        vectoriser = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
    matrix = vectoriser.fit_transform(list(descriptions))
    sim = (matrix @ matrix.T).toarray()
    np.fill_diagonal(sim, 0.0)
    return sim


def detect_article_communities(descriptions: Sequence[str], threshold: float = 0.3, language: str = "zh", seed: int = 0) -> np.ndarray:
    """Community id per article from the thresholded similarity graph (Louvain; connected components as fallback)."""
    sim = article_similarity(descriptions, language)
    n = sim.shape[0]
    edges = [(i, j, float(sim[i, j])) for i in range(n) for j in range(i + 1, n) if sim[i, j] >= threshold]
    communities: list[set[int]]
    try:
        import networkx as nx

        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        graph.add_weighted_edges_from(edges)
        communities = [set(c) for c in nx.community.louvain_communities(graph, weight="weight", seed=seed)]
    except ImportError:  # connected components of the thresholded graph
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, j, _ in edges:
            parent[find(i)] = find(j)
        roots = {}
        for i in range(n):
            roots.setdefault(find(i), set()).add(i)
        communities = list(roots.values())
    assignment = np.zeros(n, dtype=np.int64)
    for cid, members in enumerate(sorted(communities, key=lambda s: min(s))):
        for m in members:
            assignment[m] = cid
    sizes = np.bincount(assignment)
    logger.info("%d article communities (largest %d, singletons %d) at similarity >= %.2f", len(sizes), sizes.max(), int((sizes == 1).sum()), threshold)
    return assignment


def community_adjacency(assignment: np.ndarray) -> torch.Tensor:
    """Row-normalised neighbour matrix: article ``i`` is linked to the other members of its community."""
    a = torch.as_tensor(assignment)
    same = (a.unsqueeze(0) == a.unsqueeze(1)).float()
    same.fill_diagonal_(0.0)
    degree = same.sum(dim=1, keepdim=True).clamp(min=1.0)
    return same / degree


# --------------------------------------------------------------------------- model


class GraphDistillationOperator(nn.Module):
    """beta_i <- ReLU( W_self beta_i - sum_{j in N_i} W_nb (beta_i - beta_j) / |N_i| + b ), stacked ``n_layers`` times."""

    def __init__(self, dim: int, n_layers: int = 2) -> None:
        super().__init__()
        self.self_transform = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])
        self.neighbour_transform = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(n_layers)])

    def forward(self, features: torch.Tensor, neighbours: torch.Tensor) -> torch.Tensor:
        x = features
        for self_lin, nb_lin in zip(self.self_transform, self.neighbour_transform):
            neighbour_mean = neighbours @ x
            has_neighbour = (neighbours.sum(dim=1, keepdim=True) > 0).to(x.dtype)
            contrast = nb_lin(x - neighbour_mean) * has_neighbour
            x = F.relu(self_lin(x) - contrast)
        return x


class LADAN(nn.Module):
    encoder_prefixes: tuple[str, ...] = ()

    def __init__(
        self,
        model_cfg: dict,
        spec: TaskSpec,
        vocab,
        article_word_ids: torch.Tensor,
        article_word_mask: torch.Tensor,
        communities: np.ndarray,
        pretrained: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.spec = spec
        dim = int(model_cfg.get("embedding_dim", 200))
        hidden = int(model_cfg.get("hidden_size", 128))
        dropout = float(model_cfg.get("dropout", 0.3))
        self.embedding = WordEmbedding(len(vocab), dim, vocab.pad_index, pretrained, dropout=float(model_cfg.get("embedding_dropout", 0.2)))

        self.basic_encoder = BiGRUEncoder(dim, hidden)
        self.basic_pooling = AttentivePooling(2 * hidden)
        self.article_encoder = BiGRUEncoder(dim, hidden)
        self.article_pooling = AttentivePooling(2 * hidden)
        self.distillation = GraphDistillationOperator(2 * hidden, n_layers=int(model_cfg.get("gdo_layers", 2)))
        self.community_pooling = nn.Linear(2 * hidden, 2 * hidden)
        self.reencoder = BiGRUEncoder(dim, hidden)
        self.guided_pooling = AttentivePooling(2 * hidden, query_dim=2 * hidden)

        self.register_buffer("article_word_ids", article_word_ids.long())
        self.register_buffer("article_word_mask", article_word_mask.bool())
        self.register_buffer("community_of_article", torch.as_tensor(communities, dtype=torch.long))
        self.register_buffer("neighbours", community_adjacency(communities))
        self.n_communities = int(communities.max()) + 1
        members = torch.zeros(self.n_communities, spec.n_labels)
        members[self.community_of_article, torch.arange(spec.n_labels)] = 1.0
        self.register_buffer("community_members", members / members.sum(dim=1, keepdim=True))

        self.community_classifier = nn.Linear(2 * hidden, self.n_communities)
        self.article_classifier = nn.Linear(4 * hidden, spec.n_labels)
        self.dropout = nn.Dropout(dropout)
        self.aux_weight = float(model_cfg.get("community_loss_weight", 0.1))

    def distinguishing_vectors(self) -> torch.Tensor:
        """One distinguishing vector per community, from GDO-refined article representations."""
        articles = encode_label_texts(self.embedding, self.article_encoder, self.article_pooling, self.article_word_ids, self.article_word_mask)
        refined = self.distillation(articles, self.neighbours)
        return torch.tanh(self.community_pooling(self.community_members @ refined))  # [C, 2h]

    def forward(self, batch: dict) -> ModelOutput:
        mask = batch["word_mask"].bool()
        embedded = self.embedding(batch["word_ids"])
        basic_states = self.basic_encoder(embedded, mask)
        basic, _ = self.basic_pooling(basic_states, mask)
        community_logits = self.community_classifier(self.dropout(basic))

        if self.training and "labels" in batch:
            community = self.community_of_article[batch["labels"].long()]
        else:
            community = community_logits.argmax(dim=-1)
        community_vectors = self.distinguishing_vectors()  # [C, 2h]
        guide = community_vectors[community]  # [B, 2h]

        reencoded = self.reencoder(embedded, mask)
        distinguishing, _ = self.guided_pooling(reencoded, mask, query=guide)
        fused = torch.cat([basic, distinguishing], dim=-1)
        logits = self.article_classifier(self.dropout(fused))

        aux_loss = None
        if "labels" in batch:
            gold_community = self.community_of_article[batch["labels"].long()]
            aux_loss = self.aux_weight * F.cross_entropy(community_logits.float(), gold_community)
        return ModelOutput(logits=logits, aux_loss=aux_loss, aux_logits=community_logits)

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
