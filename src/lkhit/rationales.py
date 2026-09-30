"""Paragraph rationales from label-aware attention, and the lexical baselines."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def topk_from_scores(scores: np.ndarray, k: int, valid: np.ndarray | None = None) -> list[int]:
    if valid is not None:
        scores = scores.copy()
        scores[~valid] = -np.inf
    n_take = min(k, int(np.isfinite(scores).sum()))
    if n_take <= 0:
        return []
    chosen = np.argpartition(-scores, n_take - 1)[:n_take]
    return [int(index) for index in chosen[np.argsort(-scores[chosen])]]


def label_aware_rationale(
    attention: np.ndarray,
    predicted: np.ndarray,
    k: int = 5,
    fallback_label: int | None = None,
) -> list[int]:
    """Salience of paragraph i is the sum of attention from the predicted labels.

    If the predicted set is empty, the top-scoring label is used.
    `attention` has shape (labels, paragraphs).
    """
    chosen = np.where(predicted > 0)[0]
    if len(chosen) == 0:
        if fallback_label is None:
            raise ValueError("An empty prediction needs fallback_label, the top-scoring label.")
        chosen = np.array([int(fallback_label)])
    salience = attention[chosen].sum(axis=0)
    return topk_from_scores(salience, k)


def lead_rationale(n_paragraphs: int, k: int = 5) -> list[int]:
    return list(range(min(k, n_paragraphs)))


def random_rationale(n_paragraphs: int, k: int = 5, rng: np.random.Generator | None = None) -> list[int]:
    generator = rng or np.random.default_rng(0)
    n_take = min(k, n_paragraphs)
    return [int(index) for index in generator.choice(n_paragraphs, size=n_take, replace=False)]


def tfidf_rationale(paragraphs: list[str], article_text: str, k: int = 5) -> list[int]:
    if not paragraphs:
        return []
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(paragraphs + [article_text])
    similarity = cosine_similarity(matrix[:-1], matrix[-1])[:, 0]
    return topk_from_scores(similarity, k)


def agreement_at_k(selected: list[int], gold: list[int], k: int = 5) -> dict:
    gold_set = set(int(index) for index in gold)
    selected_set = set(selected)
    if not gold_set:
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
    overlap = len(selected_set & gold_set)
    precision = overlap / k
    recall = overlap / len(gold_set)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def mean_agreement(rows: list[dict]) -> dict:
    usable = [row for row in rows if row["f1"] == row["f1"]]
    if not usable:
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan"), "n": 0}
    return {
        "precision": float(np.mean([row["precision"] for row in usable])),
        "recall": float(np.mean([row["recall"] for row in usable])),
        "f1": float(np.mean([row["f1"] for row in usable])),
        "n": len(usable),
    }
