"""JSON-line datasets and batching for paragraph, chunk, and sentence inputs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

_SENTENCE_BREAK = re.compile(r"[^。！？；\n]+[。！？；]?")


def split_sentences(text: str) -> list[str]:
    """Split a CAIL fact at sentence-final punctuation."""
    parts = [piece.strip() for piece in _SENTENCE_BREAK.findall(text) if piece.strip()]
    return parts or [text.strip()]


def chunk_token_ids(input_ids: list[int], max_tokens: int, max_paragraphs: int) -> list[list[int]]:
    """Consecutive token windows. Two positions are reserved for [CLS] and [SEP]."""
    window = max(1, max_tokens - 2)
    chunks = []
    for start in range(0, len(input_ids), window):
        chunks.append(input_ids[start : start + window])
        if len(chunks) >= max_paragraphs:
            break
    return chunks or [[]]


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_labels(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def multi_hot(records: list[dict], n_labels: int) -> np.ndarray:
    matrix = np.zeros((len(records), n_labels), dtype=np.float32)
    for row, record in enumerate(records):
        for label in record["labels"]:
            matrix[row, int(label)] = 1.0
    return matrix


def class_counts(records: list[dict], n_labels: int, single_label: bool) -> np.ndarray:
    counts = np.zeros(n_labels, dtype=np.int64)
    for record in records:
        if single_label:
            counts[int(record["labels"][0])] += 1
        else:
            for label in record["labels"]:
                counts[int(label)] += 1
    return counts


class LJPDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        return {
            "index": index,
            "paragraphs": record["paragraphs"],
            "text": record.get("text", ""),
            "segment": record.get("segment", "paragraph"),
            "labels": record["labels"],
            "year": record.get("year"),
            "rationales": record.get("rationales") or [],
            "n_chars": record.get("n_chars", 0),
        }


class Collator:
    """Tokenise a batch of documents according to the task segmentation."""

    def __init__(self, tokenizer, cfg: dict, n_labels: int, keep_text: bool = False):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.n_labels = n_labels
        self.keep_text = keep_text
        self.hierarchical = cfg["arch"] == "lkhit" and bool(cfg["use_hierarchy"])
        self.single_label = bool(cfg["single_label"])
        self.max_paragraphs = int(cfg["max_paragraphs"])
        self.max_tokens = int(cfg["max_tokens"])
        self.truncated_max_tokens = int(cfg["truncated_max_tokens"])
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        self.pad_id = tokenizer.pad_token_id
        tokenizer.model_max_length = int(1e9)

    def _content_windows(self, item: dict) -> tuple[list[list[int]], int]:
        """Return per-segment content ids and the full document token count."""
        if item["segment"] == "token_chunk":
            ids = self.tokenizer.encode(item["text"], add_special_tokens=False, truncation=False)
            return chunk_token_ids(ids, self.max_tokens, self.max_paragraphs), len(ids)
        windows = []
        n_tokens = 0
        limit = max(1, self.max_tokens - 2)
        for paragraph in item["paragraphs"][: self.max_paragraphs]:
            piece = self.tokenizer.encode(paragraph, add_special_tokens=False, truncation=True, max_length=limit)
            windows.append(piece)
            n_tokens += len(piece)
        return windows or [[]], n_tokens

    def _with_special_tokens(self, content: list[int]) -> list[int]:
        room = max(0, self.max_tokens - 2)
        return [self.cls_id] + content[:room] + [self.sep_id]

    def _flat_text(self, item: dict) -> str:
        if item["segment"] == "token_chunk":
            return item["text"]
        return "\n".join(item["paragraphs"])

    def _label_tensor(self, labels: list[int]) -> torch.Tensor:
        if self.single_label:
            return torch.tensor(int(labels[0]), dtype=torch.long)
        target = torch.zeros(self.n_labels, dtype=torch.float32)
        for label in labels:
            target[int(label)] = 1.0
        return target

    def _lengths(self, item: dict, n_tokens: int) -> tuple[int, int]:
        if item["segment"] == "token_chunk":
            return 0, n_tokens
        return len(item["paragraphs"]), n_tokens

    def __call__(self, items: list[dict]) -> dict:
        labels = torch.stack([self._label_tensor(item["labels"]) for item in items])
        years = [item["year"] if item["year"] is not None else -1 for item in items]
        batch = {
            "labels": labels,
            "year": torch.tensor(years, dtype=torch.long),
            "index": torch.tensor([item["index"] for item in items], dtype=torch.long),
            "n_chars": torch.tensor([item["n_chars"] for item in items], dtype=torch.long),
            "rationales": [item["rationales"] for item in items],
        }
        if not self.hierarchical:
            encoded = self.tokenizer(
                [self._flat_text(item) for item in items],
                truncation=True,
                max_length=self.truncated_max_tokens,
                padding=True,
                return_tensors="pt",
            )
            n_paragraphs, n_tokens = [], []
            for item in items:
                if item["segment"] == "token_chunk":
                    ids = self.tokenizer.encode(item["text"], add_special_tokens=False, truncation=False)
                    paragraphs, tokens = self._lengths(item, len(ids))
                else:
                    paragraphs, tokens = self._lengths(item, 0)
                n_paragraphs.append(paragraphs)
                n_tokens.append(tokens)
            batch["input_ids"] = encoded["input_ids"]
            batch["attention_mask"] = encoded["attention_mask"]
            batch["n_paragraphs"] = torch.tensor(n_paragraphs, dtype=torch.long)
            batch["n_tokens"] = torch.tensor(n_tokens, dtype=torch.long)
            if self.keep_text:
                batch["paragraphs"] = [list(item["paragraphs"][: self.max_paragraphs]) for item in items]
            return batch

        packed_docs = []
        texts = []
        n_paragraphs, n_tokens = [], []
        for item in items:
            windows, tokens = self._content_windows(item)
            packed = [self._with_special_tokens(window) for window in windows]
            packed_docs.append(packed)
            paragraphs, tokens = self._lengths(item, tokens)
            n_paragraphs.append(paragraphs)
            n_tokens.append(tokens)
            if self.keep_text:
                if item["segment"] == "paragraph":
                    texts.append(list(item["paragraphs"][: len(packed)]))
                else:
                    texts.append([self.tokenizer.decode(window) for window in windows])
        n_para = self.max_paragraphs
        input_ids = torch.full((len(items), n_para, self.max_tokens), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        para_mask = torch.zeros(len(items), n_para, dtype=torch.bool)
        for row, packed in enumerate(packed_docs):
            for col, ids in enumerate(packed[:n_para]):
                ids = ids[: self.max_tokens]
                width = len(ids)
                input_ids[row, col, :width] = torch.tensor(ids, dtype=torch.long)
                attention_mask[row, col, :width] = 1
                para_mask[row, col] = True
        batch["input_ids"] = input_ids
        batch["attention_mask"] = attention_mask
        batch["para_mask"] = para_mask
        batch["n_paragraphs"] = torch.tensor(n_paragraphs, dtype=torch.long)
        batch["n_tokens"] = torch.tensor(n_tokens, dtype=torch.long)
        if self.keep_text:
            batch["paragraphs"] = texts
        return batch


def encode_label_texts(tokenizer, texts: list[str], max_tokens: int) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=max_tokens,
        padding="max_length",
        return_tensors="pt",
    )
    return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}
