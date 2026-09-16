"""Hierarchical baselines: HIER-BERT / HIER-Legal-BERT / HIER-RoBERTa-wwm and HIER-Legal-BERT+LSAN.

``pooling: mean`` is the LexGLUE hierarchical configuration (Chalkidis et al.,
2019, 2022): shared encoder over paragraphs, two-layer document transformer,
mean-pooled paragraph states and a linear classifier. LK-HiT without its
knowledge branch and with a mean-pooled classifier is exactly this model.

``pooling: lsan`` adds label-specific attention (Xiao et al., 2019) whose
label embeddings are randomly initialised and learned, i.e. label-aware
attention without statutory knowledge; it isolates the effect of the statute
text in LK-HiT.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lkhit.data.tasks import TaskSpec
from lkhit.models.document_transformer import DocumentTransformer, masked_mean
from lkhit.models.encoders import TextEncoder, encode_segments, split_parameter_groups
from lkhit.models.label_attention import LabelAwareAttention, LabelSpecificOutput
from lkhit.models.outputs import ModelOutput


class HierarchicalClassifier(nn.Module):
    encoder_prefixes = ("encoder.",)

    def __init__(self, model_cfg: dict, spec: TaskSpec, encoder_name: str) -> None:
        super().__init__()
        self.spec = spec
        self.pooling = model_cfg.get("pooling", "mean")
        dropout = float(model_cfg.get("dropout", 0.1))
        self.encoder = TextEncoder(encoder_name, gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", False)))
        d = self.encoder.hidden_size
        self.segment_chunk_size = model_cfg.get("segment_chunk_size")
        dt = model_cfg.get("document_transformer", {}) or {}
        self.document_transformer = DocumentTransformer(
            hidden_size=d,
            n_layers=int(dt.get("layers", 2)),
            n_heads=int(dt.get("heads", 8)),
            ffn_size=dt.get("ffn_size"),
            dropout=float(dt.get("dropout", 0.1)),
            max_segments=int(spec.max_segments),
        )
        self.dropout = nn.Dropout(dropout)
        if self.pooling == "mean":
            self.classifier = nn.Linear(d, spec.n_labels)
            nn.init.normal_(self.classifier.weight, std=0.02)
            nn.init.zeros_(self.classifier.bias)
        elif self.pooling == "lsan":
            self.label_embeddings = nn.Embedding(spec.n_labels, d)
            nn.init.normal_(self.label_embeddings.weight, std=0.02)
            self.label_attention = LabelAwareAttention(d, dropout=float(model_cfg.get("attention_dropout", 0.1)))
            self.output = LabelSpecificOutput(d, spec.n_labels, dropout=dropout)
        else:
            raise ValueError(f"unknown pooling '{self.pooling}' for the hierarchical classifier")

    def encode_document(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        states = encode_segments(
            self.encoder, batch["input_ids"], batch["attention_mask"], batch["segment_mask"], self.segment_chunk_size
        )
        mask = batch["segment_mask"].bool()
        states, attention = self.document_transformer(states, mask, return_attention=not self.training)
        return states, mask, attention

    def forward(self, batch: dict) -> ModelOutput:
        states, mask, document_attention = self.encode_document(batch)
        states = self.dropout(states)
        if self.pooling == "mean":
            pooled = masked_mean(states, mask)
            logits = self.classifier(pooled)
            return ModelOutput(logits=logits, document_attention=document_attention)
        label_embeddings = self.label_embeddings.weight
        summaries, attention = self.label_attention(label_embeddings, states, mask)
        logits = self.output(summaries)
        return ModelOutput(
            logits=logits, label_attention=attention, document_attention=document_attention, label_embeddings=label_embeddings
        )

    def parameter_groups(self, lr_encoder: float, lr_upper: float, weight_decay: float) -> list[dict]:
        return split_parameter_groups(self.named_parameters(), self.encoder_prefixes, lr_encoder, lr_upper, weight_decay)

    def parameter_summary(self) -> dict[str, int]:
        encoder = sum(p.numel() for p in self.encoder.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad), "encoder": encoder, "upper_layers": total - encoder}
