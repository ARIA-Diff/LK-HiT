"""LK-HiT document branch, knowledge branch, and label-aware scoring."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from transformers import AutoModel


class LKHiT(nn.Module):
    """Hierarchical encoder, statute label graph, and label-aware attention.

    Without statute embeddings and with mean pooling, the document branch is the
    hierarchical legal encoder. Turning hierarchy off reads the first 512 tokens
    of the joined document.
    """

    def __init__(self, cfg: dict, n_labels: int):
        super().__init__()
        encoder_name = cfg["general_encoder"] if cfg.get("use_general_encoder") else cfg["encoder"]
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.hidden = hidden
        self.n_labels = n_labels
        self.use_hierarchy = bool(cfg["use_hierarchy"])
        self.use_statute_knowledge = bool(cfg["use_statute_knowledge"])
        self.use_label_graph = bool(cfg["use_label_graph"])
        self.use_label_attention = bool(cfg["use_label_attention"])
        self.freeze_label_encoder = bool(cfg["freeze_label_encoder"])
        self.single_label = bool(cfg["single_label"])
        dropout = float(cfg["dropout"])

        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=int(cfg["doc_heads"]),
            dim_feedforward=hidden,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.doc_encoder = nn.TransformerEncoder(layer, num_layers=int(cfg["doc_layers"]))
        self.position = nn.Embedding(int(cfg["max_paragraphs"]), hidden)
        self.gcn_w1 = nn.Linear(hidden, hidden, bias=False)
        self.gcn_w2 = nn.Linear(hidden, hidden, bias=False)
        self.attn_w = nn.Linear(hidden, hidden, bias=False)
        self.score_w = nn.Linear(hidden, hidden, bias=False)
        self.score_bias = nn.Parameter(torch.zeros(n_labels))
        self.label_embed = nn.Embedding(n_labels, hidden)
        self.pool_classifier = nn.Linear(hidden, n_labels)
        self.truncated_classifier = nn.Linear(hidden, n_labels)
        nn.init.normal_(self.label_embed.weight, std=0.02)
        nn.init.normal_(self.position.weight, std=0.02)
        self._freeze_unused()
        self.register_buffer("adjacency", torch.eye(n_labels), persistent=True)
        self.register_buffer("label_input_ids", torch.zeros(n_labels, int(cfg["label_max_tokens"]), dtype=torch.long), persistent=False)
        self.register_buffer("label_attention_mask", torch.ones(n_labels, int(cfg["label_max_tokens"]), dtype=torch.long), persistent=False)

    def _freeze_unused(self) -> None:
        def off(*tensors: torch.nn.Parameter) -> None:
            for tensor in tensors:
                tensor.requires_grad = False

        if not self.use_hierarchy:
            off(*self.doc_encoder.parameters(), *self.position.parameters())
            off(*self.gcn_w1.parameters(), *self.gcn_w2.parameters(), *self.attn_w.parameters(), *self.score_w.parameters())
            off(self.score_bias, self.label_embed.weight, *self.pool_classifier.parameters())
            return
        off(*self.truncated_classifier.parameters())
        if not self.use_label_attention:
            off(*self.gcn_w1.parameters(), *self.gcn_w2.parameters(), *self.attn_w.parameters(), *self.score_w.parameters())
            off(self.score_bias, self.label_embed.weight)
            return
        off(*self.pool_classifier.parameters())
        if self.use_statute_knowledge:
            off(self.label_embed.weight)
        if not self.use_label_graph:
            off(*self.gcn_w1.parameters(), *self.gcn_w2.parameters())

    def set_graph(self, adjacency: torch.Tensor) -> None:
        self.adjacency.copy_(adjacency.to(device=self.adjacency.device, dtype=self.adjacency.dtype))

    def set_label_inputs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
        self.label_input_ids.copy_(input_ids.to(device=self.label_input_ids.device))
        self.label_attention_mask.copy_(attention_mask.to(device=self.label_attention_mask.device))

    def encoder_parameters(self):
        return self.encoder.parameters()

    def upper_parameters(self):
        encoder_ids = {id(parameter) for parameter in self.encoder.parameters()}
        for parameter in self.parameters():
            if parameter.requires_grad and id(parameter) not in encoder_ids:
                yield parameter

    def _cls(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state[:, 0]

    def _label_embeddings(self) -> torch.Tensor:
        if not self.use_statute_knowledge:
            base = self.label_embed.weight
        elif self.freeze_label_encoder:
            with torch.no_grad():
                base = self._cls(self.label_input_ids, self.label_attention_mask)
        else:
            base = self._cls(self.label_input_ids, self.label_attention_mask)
        if not self.use_label_graph:
            return base
        hidden = torch.relu(self.adjacency @ self.gcn_w1(base))
        return self.adjacency @ self.gcn_w2(hidden) + base

    def _document_states(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, para_mask: torch.Tensor) -> torch.Tensor:
        batch, n_para, n_tok = input_ids.shape
        flat_ids = input_ids.reshape(batch * n_para, n_tok)
        flat_mask = attention_mask.reshape(batch * n_para, n_tok)
        # Empty padded paragraphs still have a [CLS] token; their states are masked later.
        cls = self._cls(flat_ids, flat_mask).reshape(batch, n_para, self.hidden)
        positions = torch.arange(n_para, device=cls.device)
        states = cls + self.position(positions)[None, :, :]
        encoded = self.doc_encoder(states, src_key_padding_mask=~para_mask)
        return encoded

    def _label_attention(self, states: torch.Tensor, labels: torch.Tensor, para_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.attn_w(states)
        scores = torch.einsum("ld,bnd->bln", labels, projected) / math.sqrt(self.hidden)
        neg = -1e4 if scores.dtype == torch.float16 else torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~para_mask[:, None, :], neg)
        alpha = torch.softmax(scores, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        summary = torch.einsum("bln,bnd->bld", alpha, states)
        projected_labels = self.score_w(labels)
        logits = (summary * projected_labels[None, :, :]).sum(dim=-1) + self.score_bias
        return logits, alpha

    def _mean_pool(self, states: torch.Tensor, para_mask: torch.Tensor) -> torch.Tensor:
        mask = para_mask.to(states.dtype).unsqueeze(-1)
        pooled = (states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.pool_classifier(pooled)

    def forward(self, batch: dict) -> dict:
        if not self.use_hierarchy:
            cls = self._cls(batch["input_ids"], batch["attention_mask"])
            return {"logits": self.truncated_classifier(cls), "attention": None}
        states = self._document_states(batch["input_ids"], batch["attention_mask"], batch["para_mask"])
        if not self.use_label_attention:
            return {"logits": self._mean_pool(states, batch["para_mask"]), "attention": None}
        label_embeddings = self._label_embeddings()
        logits, attention = self._label_attention(states, label_embeddings, batch["para_mask"])
        return {"logits": logits, "attention": attention}


class SequenceClassifier(nn.Module):
    """Truncated or sparse-attention encoder with a linear head.

    Multi-label baselines trained through this class use binary cross-entropy.
    """

    def __init__(self, model_name: str, n_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden, n_labels)
        self.model_type = self.encoder.config.model_type

    def encoder_parameters(self):
        return self.encoder.parameters()

    def upper_parameters(self):
        return self.classifier.parameters()

    def forward(self, batch: dict) -> dict:
        kwargs = {}
        if self.model_type in {"longformer", "big_bird"}:
            global_mask = torch.zeros_like(batch["attention_mask"])
            global_mask[:, 0] = 1
            kwargs["global_attention_mask"] = global_mask
        output = self.encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            **kwargs,
        )
        logits = self.classifier(output.last_hidden_state[:, 0])
        return {"logits": logits, "attention": None}
