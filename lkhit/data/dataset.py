"""Torch datasets for segmented (hierarchical) and flat encodings of legal documents."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from lkhit.data.segmentation import chunk_token_ids, clean_text, split_paragraphs, split_sentences_zh
from lkhit.data.tasks import TaskSpec

logger = logging.getLogger("lkhit.data")


@dataclass
class Document:
    """One case or legislative act with its gold labels and analysis metadata."""

    doc_id: str
    labels: list[int]
    segments: list[str] | None = None
    text: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def full_text(self, joiner: str = "\n") -> str:
        if self.text is not None:
            return self.text
        return joiner.join(self.segments or [])

    def segment_list(self, spec: TaskSpec) -> list[str]:
        if spec.segmentation == "paragraph":
            return split_paragraphs(self.segments or [self.text or ""])
        if spec.segmentation == "sentence":
            return split_sentences_zh(self.full_text(""))
        raise ValueError(f"segment_list() is not defined for '{spec.segmentation}' segmentation")


def make_target(labels: Sequence[int], spec: TaskSpec) -> torch.Tensor:
    if spec.multi_label:
        target = torch.zeros(spec.n_labels, dtype=torch.float32)
        if len(labels) == 0 and spec.none_index is not None:
            target[spec.none_index] = 1.0
        for idx in labels:
            target[idx] = 1.0
        return target
    if len(labels) != 1:
        raise ValueError(f"single-label task expects exactly one label, got {list(labels)}")
    return torch.tensor(labels[0], dtype=torch.long)


def label_sets(documents: Sequence[Document], spec: TaskSpec) -> list[list[int]]:
    """Gold label index sets including the explicit ``none`` label where applicable."""
    out = []
    for doc in documents:
        labels = list(doc.labels)
        if spec.multi_label and not labels and spec.none_index is not None:
            labels = [spec.none_index]
        out.append(labels)
    return out


def label_counts(documents: Sequence[Document], spec: TaskSpec) -> np.ndarray:
    counts = np.zeros(spec.n_labels, dtype=np.int64)
    for labels in label_sets(documents, spec):
        for idx in labels:
            counts[idx] += 1
    return counts


class SegmentEncoder:
    """Turns a document into an ``N x T`` block of token ids according to the task geometry."""

    def __init__(self, tokenizer, spec: TaskSpec):
        self.tokenizer = tokenizer
        self.spec = spec
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def __call__(self, doc: Document) -> tuple[torch.Tensor, torch.Tensor, int]:
        N, T = self.spec.max_segments, self.spec.max_segment_tokens
        if self.spec.segmentation == "chunk":
            ids = self.tokenizer(clean_text(doc.full_text(" ")), add_special_tokens=False, truncation=False)["input_ids"]
            chunks = chunk_token_ids(ids, T, max_chunks=N, cls_id=self.cls_id, sep_id=self.sep_id)
            n_total = max(1, -(-len(ids) // max(T - 2, 1)))
        else:
            segments = doc.segment_list(self.spec)
            n_total = len(segments)
            segments = segments[:N] or [""]
            enc = self.tokenizer(segments, add_special_tokens=True, truncation=True, max_length=T)
            chunks = enc["input_ids"]
        input_ids = torch.full((N, T), self.pad_id, dtype=torch.long)
        attention = torch.zeros((N, T), dtype=torch.long)
        for i, chunk in enumerate(chunks[:N]):
            chunk = chunk[:T]
            input_ids[i, : len(chunk)] = torch.tensor(chunk, dtype=torch.long)
            attention[i, : len(chunk)] = 1
        return input_ids, attention, n_total


class HierarchicalDataset(Dataset):
    """Paragraph-segmented documents for the hierarchical encoders.

    Tokenisation is performed on access; ``precompute`` materialises every
    document once, which is worthwhile for the smaller ECtHR splits, and an
    optional on-disk cache avoids repeating the work between seeds.
    """

    def __init__(
        self,
        documents: Sequence[Document],
        tokenizer,
        spec: TaskSpec,
        cache_path: str | Path | None = None,
        precompute: bool = False,
    ) -> None:
        self.documents = list(documents)
        self.spec = spec
        self.encoder = SegmentEncoder(tokenizer, spec)
        self.targets = [make_target(doc.labels, spec) for doc in self.documents]
        self._cache: list[tuple[torch.Tensor, torch.Tensor, int]] | None = None
        if cache_path is not None and Path(cache_path).exists():
            payload = torch.load(cache_path, map_location="cpu")
            if payload.get("n_docs") == len(self.documents) and payload.get("geometry") == (spec.max_segments, spec.max_segment_tokens):
                self._cache = payload["items"]
                logger.info("loaded tokenised cache %s (%d documents)", cache_path, len(self._cache))
        if self._cache is None and precompute:
            self._cache = [self.encoder(doc) for doc in self.documents]
            if cache_path is not None:
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"n_docs": len(self.documents), "geometry": (spec.max_segments, spec.max_segment_tokens), "items": self._cache},
                    cache_path,
                )
                logger.info("wrote tokenised cache %s", cache_path)

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._cache is not None:
            input_ids, attention, n_total = self._cache[index]
        else:
            input_ids, attention, n_total = self.encoder(self.documents[index])
        segment_mask = attention.sum(dim=1) > 0
        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "segment_mask": segment_mask,
            "labels": self.targets[index],
            "n_segments": int(segment_mask.sum().item()),
            "n_segments_total": n_total,
            "index": index,
        }


class FlatDataset(Dataset):
    """First ``max_tokens`` tokens of the document for truncated and sparse-attention encoders."""

    def __init__(self, documents: Sequence[Document], tokenizer, spec: TaskSpec, max_tokens: int, joiner: str = "\n") -> None:
        self.documents = list(documents)
        self.tokenizer = tokenizer
        self.spec = spec
        self.max_tokens = max_tokens
        self.joiner = joiner
        self.targets = [make_target(doc.labels, spec) for doc in self.documents]

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, index: int) -> dict[str, Any]:
        doc = self.documents[index]
        text = doc.full_text(self.joiner)
        enc = self.tokenizer(text, add_special_tokens=True, truncation=True, max_length=self.max_tokens)
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": self.targets[index],
            "n_tokens": len(enc["input_ids"]),
            "index": index,
        }


def _stack_labels(items: list[dict[str, Any]]) -> torch.Tensor:
    return torch.stack([item["labels"] for item in items])


def collate_hierarchical(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad to the longest document in the batch (in segments) instead of the task maximum."""
    n_max = max(max(item["n_segments"], 1) for item in items)
    input_ids = torch.stack([item["input_ids"][:n_max] for item in items])
    attention = torch.stack([item["attention_mask"][:n_max] for item in items])
    segment_mask = torch.stack([item["segment_mask"][:n_max] for item in items])
    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "segment_mask": segment_mask,
        "labels": _stack_labels(items),
        "n_segments": torch.tensor([item["n_segments"] for item in items]),
        "n_segments_total": torch.tensor([item["n_segments_total"] for item in items]),
        "index": torch.tensor([item["index"] for item in items]),
    }


def collate_flat(items: list[dict[str, Any]], pad_id: int = 0, pad_to_multiple_of: int | None = None) -> dict[str, Any]:
    length = max(item["input_ids"].numel() for item in items)
    if pad_to_multiple_of:
        length = -(-length // pad_to_multiple_of) * pad_to_multiple_of
    input_ids = torch.full((len(items), length), pad_id, dtype=torch.long)
    attention = torch.zeros((len(items), length), dtype=torch.long)
    for i, item in enumerate(items):
        n = item["input_ids"].numel()
        input_ids[i, :n] = item["input_ids"]
        attention[i, :n] = item["attention_mask"]
    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "labels": _stack_labels(items),
        "n_tokens": torch.tensor([item["n_tokens"] for item in items]),
        "index": torch.tensor([item["index"] for item in items]),
    }


def make_flat_collate(pad_id: int, pad_to_multiple_of: int | None = None):
    def _collate(items):
        return collate_flat(items, pad_id=pad_id, pad_to_multiple_of=pad_to_multiple_of)

    return _collate
