"""LK-HiT: label-knowledge hierarchical transformer for statute-aware legal judgment prediction.

Document branch
    paragraphs -> shared pre-trained encoder ([CLS]) -> + position embeddings
    -> two-layer document transformer -> paragraph states H            (Eq. 1)
Knowledge branch
    statute texts -> the same encoder ([CLS], 256 tokens) -> E0
    -> two-layer graph convolution over the label graph -> E           (Eq. 2)
Fusion
    label-aware attention of E over H -> label-specific summaries d_l  (Eq. 3)
    bilinear score s_l = d_l^T W_o e_l + b_l                          (Eq. 4)

Every component can be switched off from the configuration, which is how the
ablation variants of the paper are produced:

    label_knowledge: statute | random      (random = learned label embeddings, no statute text)
    label_graph.enabled: true | false      (skip the graph convolution)
    pooling: label_attention | mean        (mean = one pooled document vector scored against every label)
    hierarchy: true | false                (false = first 512 tokens, attention over token states)
    label_encoder.freeze: true | false     (frozen copy of the encoder for the statute texts)
"""

from __future__ import annotations

import copy
import logging

import numpy as np
import torch
import torch.nn as nn

from lkhit.data.tasks import TaskSpec
from lkhit.models.document_transformer import DocumentTransformer, masked_mean
from lkhit.models.encoders import TextEncoder, encode_segments, split_parameter_groups
from lkhit.models.label_attention import BilinearScorer, LabelAwareAttention
from lkhit.models.label_graph import LabelGraphConvolution
from lkhit.models.outputs import ModelOutput

logger = logging.getLogger("lkhit.models.lkhit")


class LKHiT(nn.Module):
    encoder_prefixes = ("encoder.", "separate_label_encoder.")

    def __init__(
        self,
        model_cfg: dict,
        spec: TaskSpec,
        encoder_name: str,
        label_inputs: dict[str, torch.Tensor] | None = None,
        adjacency: np.ndarray | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.n_labels = spec.n_labels
        self.hierarchy = bool(model_cfg.get("hierarchy", True))
        self.label_knowledge = model_cfg.get("label_knowledge", "statute")
        self.pooling = model_cfg.get("pooling", "label_attention")
        dropout = float(model_cfg.get("dropout", 0.1))

        self.encoder = TextEncoder(encoder_name, gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", False)))
        d = self.encoder.hidden_size
        self.hidden_size = d
        self.segment_chunk_size = model_cfg.get("segment_chunk_size")

        # ---- document branch
        if self.hierarchy:
            dt = model_cfg.get("document_transformer", {}) or {}
            self.document_transformer = DocumentTransformer(
                hidden_size=d,
                n_layers=int(dt.get("layers", 2)),
                n_heads=int(dt.get("heads", 8)),
                ffn_size=dt.get("ffn_size"),
                dropout=float(dt.get("dropout", 0.1)),
                max_segments=int(spec.max_segments),
            )
        else:
            self.document_transformer = None
        self.state_dropout = nn.Dropout(dropout)

        # ---- knowledge branch
        label_cfg = model_cfg.get("label_encoder", {}) or {}
        self.label_chunk_size = int(label_cfg.get("chunk_size", 32))
        self.freeze_label_encoder = bool(label_cfg.get("freeze", False))
        self.shared_label_encoder = bool(label_cfg.get("shared", True)) and not self.freeze_label_encoder
        self.separate_label_encoder: TextEncoder | None = None
        self._frozen_label_cache: torch.Tensor | None = None
        if self.label_knowledge == "statute":
            if label_inputs is None:
                raise ValueError("statute label knowledge requires tokenised label descriptions")
            self.register_buffer("label_input_ids", label_inputs["input_ids"].long())
            self.register_buffer("label_attention_mask", label_inputs["attention_mask"].long())
            if not self.shared_label_encoder:
                self.separate_label_encoder = copy.deepcopy(self.encoder)
                if self.freeze_label_encoder:
                    self.separate_label_encoder.freeze()
            self.label_embeddings = None
        elif self.label_knowledge == "random":
            self.label_embeddings = nn.Embedding(self.n_labels, d)
            nn.init.normal_(self.label_embeddings.weight, std=0.02)
        else:
            raise ValueError(f"unknown label_knowledge '{self.label_knowledge}'")

        graph_cfg = model_cfg.get("label_graph", {}) or {}
        self.use_label_graph = bool(graph_cfg.get("enabled", True))
        if self.use_label_graph:
            if adjacency is None:
                raise ValueError("the label graph is enabled but no adjacency matrix was provided")
            adjacency_t = torch.as_tensor(np.asarray(adjacency), dtype=torch.float32)
            if adjacency_t.shape != (self.n_labels, self.n_labels):
                raise ValueError(f"adjacency of shape {tuple(adjacency_t.shape)} does not match {self.n_labels} labels")
            self.register_buffer("adjacency", adjacency_t)
            self.graph_convolution = LabelGraphConvolution(d, n_layers=int(graph_cfg.get("layers", 2)))
        else:
            self.graph_convolution = None

        # ---- fusion
        if self.pooling == "label_attention":
            self.label_attention = LabelAwareAttention(d, dropout=float(model_cfg.get("attention_dropout", 0.1)))
        elif self.pooling == "mean":
            self.label_attention = None
        else:
            raise ValueError(f"unknown pooling '{self.pooling}'")
        self.scorer = BilinearScorer(d, self.n_labels)

        logger.info(
            "LK-HiT: hierarchy=%s, label_knowledge=%s, graph=%s, pooling=%s, shared_label_encoder=%s, frozen=%s",
            self.hierarchy,
            self.label_knowledge,
            self.use_label_graph,
            self.pooling,
            self.shared_label_encoder,
            self.freeze_label_encoder,
        )

    # ------------------------------------------------------------------ knowledge branch

    @property
    def label_encoder(self) -> TextEncoder:
        return self.encoder if self.shared_label_encoder else self.separate_label_encoder  # type: ignore[return-value]

    def initial_label_embeddings(self) -> torch.Tensor:
        """``E0``: [CLS] of every statute text through the (shared or frozen) encoder, or learned vectors."""
        if self.label_knowledge == "random":
            return self.label_embeddings.weight
        if self.freeze_label_encoder:
            device = self.label_input_ids.device
            if self._frozen_label_cache is None or self._frozen_label_cache.device != device:
                with torch.no_grad():
                    self._frozen_label_cache = self.label_encoder.encode_in_chunks(
                        self.label_input_ids, self.label_attention_mask, self.label_chunk_size
                    ).detach()
            return self._frozen_label_cache
        return self.label_encoder.encode_in_chunks(self.label_input_ids, self.label_attention_mask, self.label_chunk_size)

    def refined_label_embeddings(self) -> torch.Tensor:
        e0 = self.initial_label_embeddings()
        if self.graph_convolution is not None:
            return self.graph_convolution(e0, self.adjacency)
        return e0

    # ------------------------------------------------------------------ document branch

    def encode_document(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Return paragraph (or token) states ``[B, N, d]``, their mask ``[B, N]`` and document attention."""
        if self.hierarchy:
            states = encode_segments(
                self.encoder, batch["input_ids"], batch["attention_mask"], batch["segment_mask"], self.segment_chunk_size
            )
            mask = batch["segment_mask"].bool()
            states, attention = self.document_transformer(states, mask, return_attention=not self.training)
            return states, mask, attention
        token_states = self.encoder(batch["input_ids"], batch["attention_mask"])
        return token_states, batch["attention_mask"].bool(), None

    # ------------------------------------------------------------------ forward

    def forward(self, batch: dict) -> ModelOutput:
        states, mask, document_attention = self.encode_document(batch)
        states = self.state_dropout(states)
        label_embeddings = self.refined_label_embeddings()
        if self.label_attention is not None:
            summaries, attention = self.label_attention(label_embeddings, states, mask)
            logits = self.scorer(summaries, label_embeddings)
        else:
            pooled = masked_mean(states, mask)
            logits = self.scorer(pooled, label_embeddings)
            attention = None
        return ModelOutput(
            logits=logits,
            label_attention=attention,
            document_attention=document_attention,
            label_embeddings=label_embeddings,
        )

    # ------------------------------------------------------------------ optimisation helpers

    def parameter_groups(self, lr_encoder: float, lr_upper: float, weight_decay: float) -> list[dict]:
        return split_parameter_groups(self.named_parameters(), self.encoder_prefixes, lr_encoder, lr_upper, weight_decay)

    def parameter_summary(self) -> dict[str, int]:
        encoder = sum(p.numel() for p in self.encoder.parameters())
        separate = sum(p.numel() for p in self.separate_label_encoder.parameters()) if self.separate_label_encoder else 0
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "encoder": encoder,
            "separate_label_encoder": separate,
            "upper_layers": total - encoder - separate,
        }
