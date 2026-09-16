"""LexGLUE loaders (ECtHR-A, ECtHR-B, EUR-LEX) and the ECtHR silver rationales."""

from __future__ import annotations

import csv
import hashlib
import logging
from pathlib import Path
from typing import Iterable

from lkhit.data.dataset import Document

logger = logging.getLogger("lkhit.data.lexglue")

SPLIT_ALIASES = {"train": "train", "dev": "validation", "validation": "validation", "test": "test"}


def _load_dataset(hf_dataset: str, hf_config: str, split: str, cache_dir: str | None = None):
    from datasets import load_dataset

    return load_dataset(hf_dataset, hf_config, split=SPLIT_ALIASES[split], cache_dir=cache_dir)


def lexglue_label_names(hf_dataset: str, hf_config: str, cache_dir: str | None = None) -> list[str]:
    ds = _load_dataset(hf_dataset, hf_config, "validation", cache_dir=cache_dir)
    feature = ds.features["labels"]
    names = getattr(getattr(feature, "feature", feature), "names", None)
    if names is None:
        raise ValueError(f"{hf_dataset}/{hf_config} does not expose label names")
    return [str(n) for n in names]


def facts_fingerprint(paragraphs: Iterable[str]) -> str:
    joined = "\n".join(p.strip() for p in paragraphs)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def load_lexglue_documents(cfg: dict, split: str) -> list[Document]:
    """Read one split of a LexGLUE task into :class:`Document` objects.

    ECtHR documents keep the fact paragraphs provided by the corpus; EUR-LEX acts
    are single strings that are chunked at tokenisation time. Label ids follow the
    LexGLUE class order; the explicit ``none`` label of the ECtHR tasks is added
    by the task specification when a case has no gold article.
    """
    data = cfg["data"]
    task = data["task"]
    ds = _load_dataset(data["hf_dataset"], data["hf_config"], split, cache_dir=data.get("hf_cache_dir"))
    documents: list[Document] = []
    for i, row in enumerate(ds):
        labels = [int(l) for l in row["labels"]]
        doc_id = f"{task}-{split}-{i:05d}"
        if task.startswith("ecthr"):
            paragraphs = [p for p in row["text"] if p and p.strip()]
            meta = {"n_paragraphs": len(paragraphs), "fingerprint": facts_fingerprint(paragraphs)}
            documents.append(Document(doc_id=doc_id, labels=labels, segments=paragraphs, meta=meta))
        else:
            text = row["text"]
            documents.append(Document(doc_id=doc_id, labels=labels, text=text, meta={"n_chars": len(text)}))
    attach_decision_years(documents, data, split)
    logger.info("%s/%s: %d documents", task, split, len(documents))
    return documents


def attach_decision_years(documents: list[Document], data_cfg: dict, split: str) -> None:
    """Attach HUDOC decision years to ECtHR documents when a metadata table is available.

    The table has one row per test document (``doc_index,itemid,judgment_date``);
    the year is used only for the temporal subgroup analysis.
    """
    path = (data_cfg.get("metadata") or {}).get("decision_years")
    if not path or split != "test":
        return
    path = Path(path)
    if not path.exists():
        logger.warning("decision-year table %s not found; temporal subgroups will be skipped", path)
        return
    years: dict[int, int] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            years[int(row["doc_index"])] = int(str(row["judgment_date"])[:4])
    for i, doc in enumerate(documents):
        if i in years:
            doc.meta["year"] = years[i]


def load_silver_rationales(cfg: dict, split: str = "test") -> dict[str, list[int]]:
    """Silver rationales of the ECtHR corpus keyed by the fingerprint of the fact paragraphs.

    ``ecthr_cases`` and LexGLUE share the same case texts, so matching on the
    paragraphs is robust to any difference in ordering between the two releases.
    """
    rat_cfg = cfg["data"].get("rationales")
    if not rat_cfg:
        return {}
    from datasets import load_dataset

    ds = load_dataset(rat_cfg["dataset"], rat_cfg["config"], split=SPLIT_ALIASES[split], cache_dir=cfg["data"].get("hf_cache_dir"))
    field = rat_cfg.get("field", "silver_rationales")
    out: dict[str, list[int]] = {}
    for row in ds:
        paragraphs = [p for p in row["facts"] if p and p.strip()]
        rationales = [int(r) for r in row.get(field, []) or []]
        if rationales:
            out[facts_fingerprint(paragraphs)] = rationales
    logger.info("loaded %d silver-rationale annotations from %s", len(out), rat_cfg["dataset"])
    return out


def attach_rationales(documents: list[Document], rationales: dict[str, list[int]]) -> int:
    hits = 0
    for doc in documents:
        key = doc.meta.get("fingerprint")
        if key in rationales:
            doc.meta["silver_rationales"] = rationales[key]
            hits += 1
    return hits
