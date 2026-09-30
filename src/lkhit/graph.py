"""Label graph: positive normalised PMI, statutory structure, symmetric normalisation."""

from __future__ import annotations

import numpy as np


def npmi_adjacency(multi_hot: np.ndarray) -> np.ndarray:
    """Undirected edges whose normalised PMI on the training set is positive.

    NPMI(i, j) = PMI(i, j) / -log p(i, j), with p estimated by document frequency.
    """
    labels = np.asarray(multi_hot, dtype=np.float64)
    n_docs, n_labels = labels.shape
    adjacency = np.zeros((n_labels, n_labels), dtype=np.float64)
    if n_docs == 0:
        return adjacency
    freq = labels.sum(axis=0)
    cooccurrence = labels.T @ labels
    for i in range(n_labels):
        for j in range(i + 1, n_labels):
            both = cooccurrence[i, j]
            if both <= 0 or freq[i] <= 0 or freq[j] <= 0:
                continue
            p_ij = both / n_docs
            p_i = freq[i] / n_docs
            p_j = freq[j] / n_docs
            pmi = np.log(p_ij / (p_i * p_j))
            if pmi <= 0 or p_ij >= 1.0:
                continue
            npmi = pmi / (-np.log(p_ij))
            if npmi > 0:
                adjacency[i, j] = adjacency[j, i] = npmi
    return adjacency


def structural_adjacency(groups: list[str]) -> np.ndarray:
    """Weight-1 edges between labels that share a statutory section, micro-thesaurus, or chapter."""
    n_labels = len(groups)
    adjacency = np.zeros((n_labels, n_labels), dtype=np.float64)
    for i in range(n_labels):
        group_i = groups[i]
        if not group_i:
            continue
        for j in range(i + 1, n_labels):
            if groups[j] == group_i:
                adjacency[i, j] = adjacency[j, i] = 1.0
    return adjacency


def symmetric_normalise(adjacency: np.ndarray) -> np.ndarray:
    """Add self-loops and apply D^{-1/2} (A + I) D^{-1/2}."""
    augmented = adjacency + np.eye(adjacency.shape[0], dtype=np.float64)
    degree = augmented.sum(axis=1)
    inv_sqrt = np.zeros_like(degree)
    positive = degree > 0
    inv_sqrt[positive] = degree[positive] ** -0.5
    return inv_sqrt[:, None] * augmented * inv_sqrt[None, :]


def build_normalised_adjacency(multi_hot: np.ndarray, groups: list[str]) -> np.ndarray:
    """Sum co-occurrence and structural edges, then symmetrically normalise.

    On a single-label task the co-occurrence matrix is empty, so only structural
    edges remain.
    """
    combined = npmi_adjacency(multi_hot) + structural_adjacency(groups)
    return symmetric_normalise(combined).astype(np.float32)
