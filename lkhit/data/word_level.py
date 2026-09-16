"""Word-level vocabulary and dataset for the re-implemented Chinese LJP baselines."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from lkhit.data.dataset import Document, make_target
from lkhit.data.tasks import TaskSpec
from lkhit.utils import load_json, save_json

logger = logging.getLogger("lkhit.data.word_level")

PAD, UNK = "<pad>", "<unk>"


def tokenize_zh(text: str) -> list[str]:
    import jieba

    jieba.setLogLevel(logging.WARNING)
    return [tok for tok in jieba.lcut(text) if tok.strip()]


class WordVocab:
    def __init__(self, tokens: Sequence[str]) -> None:
        self.itos = list(tokens)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad_index(self) -> int:
        return self.stoi[PAD]

    @property
    def unk_index(self) -> int:
        return self.stoi[UNK]

    def encode(self, tokens: Iterable[str], max_len: int | None = None) -> list[int]:
        ids = [self.stoi.get(tok, self.unk_index) for tok in tokens]
        if max_len is not None:
            ids = ids[:max_len]
        return ids

    @classmethod
    def build(cls, texts: Iterable[str], min_freq: int = 5, max_size: int = 100_000) -> "WordVocab":
        counter: Counter = Counter()
        for text in texts:
            counter.update(tokenize_zh(text))
        tokens = [PAD, UNK] + [tok for tok, c in counter.most_common(max_size) if c >= min_freq]
        logger.info("word vocabulary: %d types (min_freq=%d)", len(tokens), min_freq)
        return cls(tokens)

    def save(self, path: Path) -> None:
        save_json(self.itos, path)

    @classmethod
    def load(cls, path: Path) -> "WordVocab":
        return cls(load_json(path))

    def load_pretrained_embeddings(self, path: Path, dim: int) -> torch.Tensor:
        """Read a text word2vec file (``word v1 v2 ...``); missing words get small random vectors."""
        rng = np.random.RandomState(0)
        weights = rng.normal(0.0, 0.1, size=(len(self), dim)).astype(np.float32)
        weights[self.pad_index] = 0.0
        found = 0
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                parts = line.rstrip().split(" ")
                if len(parts) != dim + 1:
                    continue
                idx = self.stoi.get(parts[0])
                if idx is not None:
                    weights[idx] = np.asarray(parts[1:], dtype=np.float32)
                    found += 1
        logger.info("pretrained vectors found for %d/%d vocabulary entries", found, len(self))
        return torch.from_numpy(weights)


class WordLevelDataset(Dataset):
    def __init__(self, documents: Sequence[Document], vocab: WordVocab, spec: TaskSpec, max_words: int = 512) -> None:
        self.documents = list(documents)
        self.vocab = vocab
        self.spec = spec
        self.max_words = max_words
        self.targets = [make_target(doc.labels, spec) for doc in self.documents]
        self._ids = [vocab.encode(tokenize_zh(doc.full_text("")), max_words) for doc in self.documents]

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, index: int) -> dict[str, Any]:
        ids = self._ids[index] or [self.vocab.unk_index]
        return {
            "word_ids": torch.tensor(ids, dtype=torch.long),
            "labels": self.targets[index],
            "n_tokens": len(ids),
            "index": index,
        }


def make_word_collate(pad_index: int):
    def _collate(items: list[dict[str, Any]]) -> dict[str, Any]:
        length = max(item["word_ids"].numel() for item in items)
        word_ids = torch.full((len(items), length), pad_index, dtype=torch.long)
        mask = torch.zeros((len(items), length), dtype=torch.bool)
        for i, item in enumerate(items):
            n = item["word_ids"].numel()
            word_ids[i, :n] = item["word_ids"]
            mask[i, :n] = True
        return {
            "word_ids": word_ids,
            "word_mask": mask,
            "labels": torch.stack([item["labels"] for item in items]),
            "n_tokens": torch.tensor([item["n_tokens"] for item in items]),
            "index": torch.tensor([item["index"] for item in items]),
        }

    return _collate


def encode_article_texts(descriptions: Sequence[str], vocab: WordVocab, max_words: int = 200) -> tuple[torch.Tensor, torch.Tensor]:
    """Word ids of the statute texts, used by LADAN and NeurJudge as article representations."""
    encoded = [vocab.encode(tokenize_zh(text), max_words) or [vocab.unk_index] for text in descriptions]
    length = max(len(e) for e in encoded)
    ids = torch.full((len(encoded), length), vocab.pad_index, dtype=torch.long)
    mask = torch.zeros((len(encoded), length), dtype=torch.bool)
    for i, e in enumerate(encoded):
        ids[i, : len(e)] = torch.tensor(e)
        mask[i, : len(e)] = True
    return ids, mask
