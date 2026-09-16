"""CAIL2018 preparation under the LADAN protocol and loading of the processed splits."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np

from lkhit.data.dataset import Document
from lkhit.utils import load_json, read_jsonl, save_json, write_jsonl

logger = logging.getLogger("lkhit.data.cail")

RAW_FILES = {"train": "data_train.json", "test": "data_test.json"}


def _iter_raw(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            yield row


def _single_article_cases(path: Path) -> list[dict]:
    """Keep cases whose ``relevant_articles`` list has exactly one entry."""
    kept = []
    total = 0
    for row in _iter_raw(path):
        total += 1
        articles = row.get("meta", {}).get("relevant_articles", [])
        fact = (row.get("fact") or "").strip()
        if len(articles) != 1 or not fact:
            continue
        kept.append({"fact": fact, "article": int(articles[0]), "accusation": row["meta"].get("accusation", [])})
    logger.info("%s: %d/%d cases with a single applicable article", path.name, len(kept), total)
    return kept


def stratified_holdout(labels: list[int], fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-class stratified index split; every class keeps at least one training case."""
    rng = np.random.RandomState(seed)
    labels_arr = np.asarray(labels)
    dev_idx = []
    for label in np.unique(labels_arr):
        members = np.where(labels_arr == label)[0]
        rng.shuffle(members)
        n_dev = int(round(len(members) * fraction))
        n_dev = min(n_dev, len(members) - 1)
        dev_idx.extend(members[:n_dev].tolist())
    dev_mask = np.zeros(len(labels_arr), dtype=bool)
    dev_mask[dev_idx] = True
    return np.where(~dev_mask)[0], np.where(dev_mask)[0]


def prepare_cail(raw_dir: Path, out_dir: Path, min_train_cases: int = 100, dev_fraction: float = 0.10, seed: int = 13) -> dict:
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = _single_article_cases(raw_dir / RAW_FILES["train"])
    test_rows = _single_article_cases(raw_dir / RAW_FILES["test"])

    counts = Counter(r["article"] for r in train_rows)
    kept_articles = sorted(a for a, c in counts.items() if c >= min_train_cases)
    keep = set(kept_articles)
    train_rows = [r for r in train_rows if r["article"] in keep]
    test_rows = [r for r in test_rows if r["article"] in keep]
    logger.info("%d articles with >= %d training cases", len(kept_articles), min_train_cases)

    article_to_index = {a: i for i, a in enumerate(kept_articles)}
    train_idx, dev_idx = stratified_holdout([article_to_index[r["article"]] for r in train_rows], dev_fraction, seed)

    def _rows(subset, split):
        for j, i in enumerate(subset):
            r = train_rows[i] if split != "test" else test_rows[i]
            yield {"id": f"cail2018-{split}-{j:06d}", "fact": r["fact"], "article": r["article"], "accusation": r["accusation"]}

    write_jsonl(out_dir / "train.jsonl", _rows(train_idx, "train"))
    write_jsonl(out_dir / "dev.jsonl", _rows(dev_idx, "dev"))
    write_jsonl(out_dir / "test.jsonl", _rows(range(len(test_rows)), "test"))
    save_json([str(a) for a in kept_articles], out_dir / "label_list.json")

    stats = {
        "n_articles": len(kept_articles),
        "n_train": int(len(train_idx)),
        "n_dev": int(len(dev_idx)),
        "n_test": len(test_rows),
        "min_train_cases": min_train_cases,
        "dev_fraction": dev_fraction,
        "split_seed": seed,
        "train_counts": {str(a): int(counts[a]) for a in kept_articles},
        "fact_chars_mean": float(np.mean([len(r["fact"]) for r in train_rows])),
        "fact_chars_median": float(np.median([len(r["fact"]) for r in train_rows])),
    }
    save_json(stats, out_dir / "stats.json")
    logger.info("wrote %s (train %d / dev %d / test %d)", out_dir, stats["n_train"], stats["n_dev"], stats["n_test"])
    return stats


def load_cail_documents(cfg: dict, split: str) -> list[Document]:
    processed = Path(cfg["data"]["processed_dir"])
    label_list = [str(a) for a in load_json(processed / "label_list.json")]
    index = {a: i for i, a in enumerate(label_list)}
    rows = read_jsonl(processed / f"{split}.jsonl")
    documents = []
    for row in rows:
        article = str(row["article"])
        documents.append(
            Document(
                doc_id=row["id"],
                labels=[index[article]],
                text=row["fact"],
                meta={"n_chars": len(row["fact"]), "article": article, "accusation": row.get("accusation", [])},
            )
        )
    logger.info("cail2018/%s: %d documents", split, len(documents))
    return documents


def load_criminal_law_articles(path: Path) -> dict[str, str]:
    payload = load_json(path)
    return {str(k): str(v) for k, v in payload.items()}
