"""Shared paths, batching, and label-graph setup for training and evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from lkhit.data import Collator, LJPDataset, encode_label_texts, load_labels, multi_hot, read_jsonl
from lkhit.graph import build_normalised_adjacency
from lkhit.model import LKHiT

ROOT = Path(__file__).resolve().parents[2]


def load_splits(cfg: dict):
    data_dir = ROOT / cfg["data_dir"]
    splits = {name: read_jsonl(data_dir / f"{name}.jsonl") for name in ("train", "dev", "test")}
    labels = load_labels(data_dir / "labels.json")
    return splits, labels


def make_loader(records, tokenizer, cfg, n_labels: int, shuffle: bool, keep_text: bool = False) -> DataLoader:
    return DataLoader(
        LJPDataset(records),
        batch_size=int(cfg["batch_size"]),
        shuffle=shuffle,
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=Collator(tokenizer, cfg, n_labels, keep_text=keep_text),
        pin_memory=torch.cuda.is_available(),
    )


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def prepare_lkhit(model: LKHiT, tokenizer, label_spec: dict, train_records: list[dict], device: torch.device) -> None:
    n_labels = len(label_spec["names"])
    if label_spec["single_label"]:
        hot = np.zeros((len(train_records), n_labels), dtype=np.float32)
        for row, record in enumerate(train_records):
            hot[row, int(record["labels"][0])] = 1.0
    else:
        hot = multi_hot(train_records, n_labels)
    adjacency = build_normalised_adjacency(hot, label_spec["groups"])
    model.set_graph(torch.from_numpy(adjacency))
    if model.use_statute_knowledge:
        encoded = encode_label_texts(tokenizer, label_spec["texts"], int(model.label_input_ids.shape[1]))
        model.set_label_inputs(encoded["input_ids"], encoded["attention_mask"])
    model.to(device)


def tokenizer_name(cfg: dict) -> str:
    if cfg["arch"] == "encoder":
        return cfg.get("model_name_or_path") or cfg["encoder"]
    if cfg.get("use_general_encoder"):
        return cfg["general_encoder"]
    return cfg["encoder"]


def load_tokenizer(cfg: dict):
    return AutoTokenizer.from_pretrained(tokenizer_name(cfg))
