"""Paragraph rationales from label-aware attention and their agreement with the ECtHR silver rationales.

For a document with predicted label set ``Y_hat`` (or the top-scoring label
when ``Y_hat`` is empty) the salience of paragraph ``i`` is
``r_i = sum_{l in Y_hat} alpha_{l i}`` and the ``k = 5`` most salient
paragraphs are returned. Baselines: random paragraphs, the first ``k``
paragraphs (Lead-k), the ``k`` paragraphs with the highest TF-IDF similarity
to the text of the predicted articles, and the second-level self-attention of
the hierarchical encoder pooled over heads.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from lkhit.data.tasks import TaskSpec


# --------------------------------------------------------------------------- salience


def predicted_label_set(probs: np.ndarray, spec: TaskSpec, threshold: float = 0.5, exclude_none: bool = True) -> list[int]:
    probs = np.asarray(probs, dtype=np.float64)
    if spec.multi_label:
        labels = [int(i) for i in np.where(probs >= threshold)[0]]
        if exclude_none and spec.none_index is not None:
            labels = [i for i in labels if i != spec.none_index]
        if not labels:
            order = np.argsort(-probs)
            for i in order:
                if not (exclude_none and spec.none_index is not None and int(i) == spec.none_index):
                    labels = [int(i)]
                    break
        return labels
    return [int(probs.argmax())]


def label_attention_salience(label_attention: np.ndarray, labels: Sequence[int], n_segments: int) -> np.ndarray:
    """``label_attention``: [n_labels, n_segments_padded] -> salience over the first ``n_segments``."""
    att = np.asarray(label_attention, dtype=np.float64)[:, :n_segments]
    if len(labels) == 0:
        return att.mean(axis=0)
    return att[list(labels)].sum(axis=0)


def document_attention_salience(document_attention: np.ndarray, n_segments: int) -> np.ndarray:
    """Second-level self-attention [n, n] (head-averaged): how much every paragraph is attended to."""
    att = np.asarray(document_attention, dtype=np.float64)[:n_segments, :n_segments]
    return att.mean(axis=0)


def top_k_indices(salience: np.ndarray, k: int = 5) -> list[int]:
    salience = np.asarray(salience, dtype=np.float64)
    k = min(k, len(salience))
    if k == 0:
        return []
    order = np.argsort(-salience, kind="stable")
    return [int(i) for i in order[:k]]


# --------------------------------------------------------------------------- baselines


def lead_k(n_segments: int, k: int = 5) -> list[int]:
    return list(range(min(k, n_segments)))


def random_k(n_segments: int, k: int = 5, rng: np.random.RandomState | None = None) -> list[int]:
    rng = rng or np.random.RandomState(0)
    k = min(k, n_segments)
    return sorted(int(i) for i in rng.choice(n_segments, size=k, replace=False)) if k else []


class TfidfArticleSimilarity:
    """TF-IDF cosine similarity between each paragraph and the text of the predicted articles."""

    def __init__(self, descriptions: Sequence[str], corpus_paragraphs: Sequence[str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectoriser = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english")
        self.vectoriser.fit(list(corpus_paragraphs) + list(descriptions))
        self.description_matrix = self.vectoriser.transform(list(descriptions))

    def salience(self, paragraphs: Sequence[str], labels: Sequence[int]) -> np.ndarray:
        if not paragraphs:
            return np.zeros(0)
        para = self.vectoriser.transform(list(paragraphs))
        if len(labels) == 0:
            query = self.description_matrix.mean(axis=0)
            query = np.asarray(query)
        else:
            query = np.asarray(self.description_matrix[list(labels)].sum(axis=0))
        scores = para @ query.T
        return np.asarray(scores).ravel()


# --------------------------------------------------------------------------- agreement


def agreement_at_k(selected: Sequence[int], silver: Sequence[int]) -> dict[str, float]:
    sel, gold = set(int(i) for i in selected), set(int(i) for i in silver)
    if not sel or not gold:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    hit = len(sel & gold)
    precision = hit / len(sel)
    recall = hit / len(gold)
    f1 = 0.0 if hit == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def mean_agreement(per_document: Sequence[dict[str, float]]) -> dict[str, float]:
    if not per_document:
        return {"precision_at_k": float("nan"), "recall_at_k": float("nan"), "f1_at_k": float("nan"), "n_documents": 0}
    return {
        "precision_at_k": 100 * float(np.mean([d["precision"] for d in per_document])),
        "recall_at_k": 100 * float(np.mean([d["recall"] for d in per_document])),
        "f1_at_k": 100 * float(np.mean([d["f1"] for d in per_document])),
        "n_documents": len(per_document),
    }


def evaluate_rationales(
    probs: np.ndarray,
    spec: TaskSpec,
    documents: Sequence,
    label_attention: Sequence[np.ndarray] | None,
    document_attention: Sequence[np.ndarray] | None,
    descriptions: Sequence[str] | None,
    k: int = 5,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Agreement of every rationale method with the silver rationales attached to ``documents``."""
    rng = np.random.RandomState(seed)
    methods: dict[str, list[dict[str, float]]] = {"random": [], "lead": []}
    tfidf = None
    if descriptions is not None:
        corpus = [p for doc in documents for p in (doc.segments or [])]
        tfidf = TfidfArticleSimilarity(descriptions, corpus)
        methods["tfidf_article_text"] = []
    if document_attention is not None:
        methods["document_attention"] = []
    if label_attention is not None:
        methods["label_aware_attention"] = []

    for i, doc in enumerate(documents):
        silver = doc.meta.get("silver_rationales")
        if not silver:
            continue
        paragraphs = doc.segments or []
        n = min(len(paragraphs), spec.max_segments)
        if n == 0:
            continue
        silver = [s for s in silver if s < n]
        if not silver:
            continue
        labels = predicted_label_set(probs[i], spec)
        methods["random"].append(agreement_at_k(random_k(n, k, rng), silver))
        methods["lead"].append(agreement_at_k(lead_k(n, k), silver))
        if tfidf is not None:
            methods["tfidf_article_text"].append(agreement_at_k(top_k_indices(tfidf.salience(paragraphs[:n], labels), k), silver))
        if document_attention is not None:
            sal = document_attention_salience(document_attention[i], n)
            methods["document_attention"].append(agreement_at_k(top_k_indices(sal, k), silver))
        if label_attention is not None:
            sal = label_attention_salience(label_attention[i], labels, n)
            methods["label_aware_attention"].append(agreement_at_k(top_k_indices(sal, k), silver))

    return {name: mean_agreement(rows) for name, rows in methods.items()}


def extract_rationale(
    probs: np.ndarray, label_attention: np.ndarray, spec: TaskSpec, n_segments: int, k: int = 5
) -> dict[str, object]:
    labels = predicted_label_set(probs, spec)
    salience = label_attention_salience(label_attention, labels, n_segments)
    top = top_k_indices(salience, k)
    return {
        "labels": [spec.label_names[l] for l in labels],
        "paragraphs": top,
        "salience": [float(salience[i]) for i in top],
    }
