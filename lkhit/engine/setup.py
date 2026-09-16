"""Assemble everything a run needs: task, documents, tokenizer or vocabulary, label knowledge, datasets and loaders."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from lkhit.data.dataset import Document, label_counts
from lkhit.data.lexglue import attach_rationales, load_silver_rationales
from lkhit.data.loading import (
    build_collate,
    build_dataloader,
    build_dataset,
    build_word_vocab,
    input_format,
    load_documents,
    load_task,
    load_tokenizer,
)
from lkhit.data.tasks import TaskSpec
from lkhit.engine.knowledge import copy_knowledge_into_run, ensure_label_descriptions, ensure_label_graph, load_run_knowledge
from lkhit.models import build_model, needs_label_descriptions, needs_label_graph

logger = logging.getLogger("lkhit.engine.setup")


@dataclass
class RunInputs:
    spec: TaskSpec
    documents: dict[str, list[Document]]
    tokenizer: object | None = None
    vocab: object | None = None
    descriptions: list[str] | None = None
    adjacency: np.ndarray | None = None
    train_counts: np.ndarray | None = None
    datasets: dict[str, object] = field(default_factory=dict)
    loaders: dict[str, object] = field(default_factory=dict)
    format: str = ""


def load_split_documents(cfg: dict, splits: Sequence[str]) -> dict[str, list[Document]]:
    documents = {split: load_documents(cfg, split) for split in splits}
    if cfg["data"]["task"] == "ecthr_a" and "test" in documents and cfg["data"].get("rationales"):
        try:
            hits = attach_rationales(documents["test"], load_silver_rationales(cfg, "test"))
            logger.info("silver rationales attached to %d test documents", hits)
        except Exception as exc:  # the rationale corpus is optional for training
            logger.warning("silver rationales not loaded: %s", exc)
    return documents


def prepare_inputs(
    cfg: dict,
    splits: Sequence[str],
    run_dir: Path,
    seed: int,
    for_training: bool,
    eval_batch_size: int | None = None,
) -> RunInputs:
    spec = load_task(cfg)
    documents = load_split_documents(cfg, splits)
    fmt = input_format(cfg["model"])
    inputs = RunInputs(spec=spec, documents=documents, format=fmt)
    model_cfg = cfg["model"]

    if "train" in documents:
        inputs.train_counts = label_counts(documents["train"], spec)

    # ---- label knowledge
    if for_training:
        if needs_label_descriptions(model_cfg):
            inputs.descriptions, groups = ensure_label_descriptions(cfg, spec)
        else:
            groups = None
        if needs_label_graph(model_cfg):
            inputs.adjacency = ensure_label_graph(cfg, spec, documents.get("train"), groups)
        copy_knowledge_into_run(cfg, run_dir, with_graph=needs_label_graph(model_cfg))
    else:
        inputs.descriptions, inputs.adjacency = load_run_knowledge(run_dir, spec, with_graph=needs_label_graph(model_cfg))
        if needs_label_descriptions(model_cfg) and inputs.descriptions is None:
            inputs.descriptions, groups = ensure_label_descriptions(cfg, spec)
        if needs_label_graph(model_cfg) and inputs.adjacency is None:
            inputs.adjacency = ensure_label_graph(cfg, spec, documents.get("train"), None)

    # ---- tokeniser / vocabulary
    if fmt in ("hierarchical", "flat"):
        inputs.tokenizer = load_tokenizer(cfg)
    elif fmt == "words":
        train_docs = documents.get("train")
        if train_docs is None:
            train_docs = load_documents(cfg, "train")
        inputs.vocab = build_word_vocab(cfg, train_docs, run_dir)

    # ---- datasets and loaders
    if fmt in ("hierarchical", "flat", "words"):
        collate = build_collate(cfg, tokenizer=inputs.tokenizer, vocab=inputs.vocab)
        train_bs = int(cfg["train"].get("batch_size", 4))
        eval_bs = int(eval_batch_size or cfg["train"].get("eval_batch_size", 2 * train_bs))
        for split, docs in documents.items():
            dataset = build_dataset(cfg, spec, docs, split, tokenizer=inputs.tokenizer, vocab=inputs.vocab)
            inputs.datasets[split] = dataset
            shuffle = split == "train" and for_training
            inputs.loaders[split] = build_dataloader(dataset, cfg, collate, shuffle=shuffle, batch_size=train_bs if shuffle else eval_bs, seed=seed)
            logger.info("%s: %d documents, %d batches (batch %d)", split, len(dataset), len(inputs.loaders[split]), train_bs if shuffle else eval_bs)
    return inputs


def instantiate_model(cfg: dict, inputs: RunInputs, device: torch.device) -> torch.nn.Module:
    model = build_model(
        cfg,
        inputs.spec,
        tokenizer=inputs.tokenizer,
        label_descriptions=inputs.descriptions,
        adjacency=inputs.adjacency,
        vocab=inputs.vocab,
    )
    model.to(device)
    summary = model.parameter_summary()
    logger.info(
        "model %s: %.1fM parameters (%.1fM trainable, %.1fM in the pre-trained encoder, %.2fM upper layers)",
        model.__class__.__name__,
        summary["total"] / 1e6,
        summary["trainable"] / 1e6,
        summary.get("encoder", 0) / 1e6,
        summary.get("upper_layers", 0) / 1e6,
    )
    return model
