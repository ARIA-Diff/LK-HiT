"""Model registry: LK-HiT and the baselines, built from the ``model`` section of a configuration."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from lkhit.config import resolve_encoder_name
from lkhit.data.tasks import TaskSpec
from lkhit.models.outputs import ModelOutput

logger = logging.getLogger("lkhit.models")

TORCH_ARCHITECTURES = ("lkhit", "hierarchical", "truncated", "sparse", "topjudge", "ladan", "neurjudge")
SKLEARN_ARCHITECTURES = ("tfidf_svm",)
EXTERNAL_ARCHITECTURES = ("zero_shot_llm",)


def tokenize_label_descriptions(tokenizer, descriptions: Sequence[str], max_tokens: int = 256) -> dict[str, torch.Tensor]:
    enc = tokenizer(list(descriptions), padding="max_length", truncation=True, max_length=max_tokens, return_tensors="pt")
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


def needs_label_descriptions(model_cfg: dict) -> bool:
    arch = model_cfg["arch"]
    if arch == "lkhit":
        return model_cfg.get("label_knowledge", "statute") == "statute"
    return arch in ("ladan", "neurjudge")


def needs_label_graph(model_cfg: dict) -> bool:
    return model_cfg["arch"] == "lkhit" and bool((model_cfg.get("label_graph", {}) or {}).get("enabled", True))


def build_model(
    cfg: dict,
    spec: TaskSpec,
    tokenizer=None,
    label_descriptions: Sequence[str] | None = None,
    adjacency: np.ndarray | None = None,
    vocab=None,
) -> nn.Module:
    model_cfg = cfg["model"]
    arch = model_cfg["arch"]

    if arch == "lkhit":
        from lkhit.models.lkhit import LKHiT

        label_inputs = None
        if model_cfg.get("label_knowledge", "statute") == "statute":
            if tokenizer is None or label_descriptions is None:
                raise ValueError("LK-HiT with statute knowledge needs a tokenizer and the label descriptions")
            max_tokens = int((model_cfg.get("label_encoder", {}) or {}).get("max_tokens", 256))
            label_inputs = tokenize_label_descriptions(tokenizer, label_descriptions, max_tokens)
        return LKHiT(model_cfg, spec, resolve_encoder_name(cfg), label_inputs=label_inputs, adjacency=adjacency)

    if arch == "hierarchical":
        from lkhit.models.hierarchical import HierarchicalClassifier

        return HierarchicalClassifier(model_cfg, spec, resolve_encoder_name(cfg))

    if arch == "truncated":
        from lkhit.models.truncated import TruncatedEncoderClassifier

        return TruncatedEncoderClassifier(model_cfg, spec, resolve_encoder_name(cfg))

    if arch == "sparse":
        from lkhit.models.sparse import SparseAttentionClassifier

        return SparseAttentionClassifier(model_cfg, spec, resolve_encoder_name(cfg))

    if arch in ("topjudge", "ladan", "neurjudge"):
        if vocab is None:
            raise ValueError(f"{arch} needs a word vocabulary")
        from lkhit.models.word_encoders import load_word_vectors

        pretrained = load_word_vectors(vocab, model_cfg, int(model_cfg.get("embedding_dim", 200)))
        if arch == "topjudge":
            from lkhit.models.topjudge import TopJudge

            return TopJudge(model_cfg, spec, vocab, pretrained=pretrained)

        if label_descriptions is None:
            raise ValueError(f"{arch} needs the statute texts of the law articles")
        from lkhit.data.word_level import encode_article_texts

        ids, mask = encode_article_texts(label_descriptions, vocab, max_words=int(model_cfg.get("article_max_words", 200)))
        threshold = float(model_cfg.get("similarity_threshold", 0.3))
        if arch == "ladan":
            from lkhit.models.ladan import LADAN, detect_article_communities

            communities = detect_article_communities(label_descriptions, threshold=threshold, language=spec.language)
            return LADAN(model_cfg, spec, vocab, ids, mask, communities, pretrained=pretrained)

        from lkhit.models.neurjudge import NeurJudge, similarity_neighbours

        neighbours = similarity_neighbours(label_descriptions, threshold=threshold, language=spec.language)
        return NeurJudge(model_cfg, spec, vocab, ids, mask, neighbours, pretrained=pretrained)

    if arch in SKLEARN_ARCHITECTURES:
        raise ValueError(f"'{arch}' is not a torch model; use lkhit.models.tfidf_svm.TfidfSvmClassifier")
    if arch in EXTERNAL_ARCHITECTURES:
        raise ValueError(f"'{arch}' is served by `python -m lkhit.cli.zero_shot_llm`")
    raise ValueError(f"unknown model architecture '{arch}'")


def describe(model: nn.Module) -> str:
    summary = model.parameter_summary() if hasattr(model, "parameter_summary") else {}
    parts = [f"{k}={v / 1e6:.2f}M" for k, v in summary.items()]
    return f"{model.__class__.__name__}(" + ", ".join(parts) + ")"


__all__ = [
    "ModelOutput",
    "build_model",
    "describe",
    "needs_label_descriptions",
    "needs_label_graph",
    "tokenize_label_descriptions",
    "TORCH_ARCHITECTURES",
    "SKLEARN_ARCHITECTURES",
    "EXTERNAL_ARCHITECTURES",
]
