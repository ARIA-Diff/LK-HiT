"""Task-agnostic entry points: documents per split, tokenizer, datasets and data loaders."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader

from lkhit.config import resolve_encoder_name
from lkhit.data.dataset import Document, FlatDataset, HierarchicalDataset, collate_hierarchical, make_flat_collate
from lkhit.data.tasks import TaskSpec, build_task_spec, load_label_names
from lkhit.data.word_level import WordLevelDataset, WordVocab, make_word_collate

logger = logging.getLogger("lkhit.data")

SPLITS = ("train", "dev", "test")


def load_documents(cfg: dict, split: str) -> list[Document]:
    task = cfg["data"]["task"]
    if task in ("ecthr_a", "ecthr_b", "eurlex"):
        from lkhit.data.lexglue import load_lexglue_documents

        return load_lexglue_documents(cfg, split)
    if task == "cail2018":
        from lkhit.data.cail import load_cail_documents

        return load_cail_documents(cfg, split)
    raise ValueError(f"unknown task '{task}'")


def load_task(cfg: dict) -> TaskSpec:
    return build_task_spec(cfg, load_label_names(cfg))


def load_tokenizer(cfg: dict, role: str | None = None):
    from transformers import AutoTokenizer

    name = resolve_encoder_name(cfg, role)
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
    return tokenizer


def input_format(model_cfg: dict) -> str:
    """``hierarchical`` (N x T segments), ``flat`` (single token sequence) or ``words``."""
    arch = model_cfg["arch"]
    if arch in ("lkhit", "hierarchical"):
        return "hierarchical" if model_cfg.get("hierarchy", True) else "flat"
    if arch in ("truncated", "sparse"):
        return "flat"
    if arch in ("topjudge", "ladan", "neurjudge"):
        return "words"
    if arch == "tfidf_svm":
        return "text"
    raise ValueError(f"unknown model architecture '{arch}'")


def flat_max_tokens(cfg: dict, spec: TaskSpec) -> int:
    arch = cfg["model"]["arch"]
    if arch == "sparse":
        return int(cfg["model"].get("max_tokens", spec.sparse_max_tokens))
    return int(cfg["model"].get("max_tokens", spec.flat_max_tokens))


def build_dataset(cfg: dict, spec: TaskSpec, documents: Sequence[Document], split: str, tokenizer=None, vocab: WordVocab | None = None):
    fmt = input_format(cfg["model"])
    data_cfg = cfg["data"]
    if fmt == "hierarchical":
        cache_dir = data_cfg.get("cache_dir")
        encoder_tag = resolve_encoder_name(cfg).replace("/", "__")
        cache_path = Path(cache_dir) / f"{split}.{encoder_tag}.{spec.max_segments}x{spec.max_segment_tokens}.pt" if cache_dir else None
        return HierarchicalDataset(documents, tokenizer, spec, cache_path=cache_path, precompute=bool(data_cfg.get("precompute", False)))
    if fmt == "flat":
        return FlatDataset(documents, tokenizer, spec, max_tokens=flat_max_tokens(cfg, spec))
    if fmt == "words":
        if vocab is None:
            raise ValueError("word-level models need a vocabulary")
        return WordLevelDataset(documents, vocab, spec, max_words=int(cfg["model"].get("max_words", 512)))
    raise ValueError(f"no torch dataset for input format '{fmt}'")


def build_collate(cfg: dict, tokenizer=None, vocab: WordVocab | None = None):
    fmt = input_format(cfg["model"])
    if fmt == "hierarchical":
        return collate_hierarchical
    if fmt == "flat":
        pad_multiple = cfg["model"].get("pad_to_multiple_of")
        return make_flat_collate(tokenizer.pad_token_id or 0, pad_multiple)
    if fmt == "words":
        return make_word_collate(vocab.pad_index)
    raise ValueError(f"no collate function for input format '{fmt}'")


def build_dataloader(dataset, cfg: dict, collate, shuffle: bool, batch_size: int, seed: int | None = None) -> DataLoader:
    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=int(cfg["data"].get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator,
        persistent_workers=bool(cfg["data"].get("num_workers", 0)) and shuffle,
    )


def build_word_vocab(cfg: dict, train_documents: Sequence[Document], run_dir: Path) -> WordVocab:
    vocab_cfg = cfg["model"].get("vocab", {}) or {}
    shared = Path(cfg["data"].get("processed_dir", run_dir)) / "word_vocab.json"
    if shared.exists():
        vocab = WordVocab.load(shared)
        logger.info("loaded word vocabulary from %s (%d types)", shared, len(vocab))
        return vocab
    vocab = WordVocab.build(
        (doc.full_text("") for doc in train_documents),
        min_freq=int(vocab_cfg.get("min_freq", 5)),
        max_size=int(vocab_cfg.get("max_size", 100_000)),
    )
    vocab.save(shared)
    return vocab
