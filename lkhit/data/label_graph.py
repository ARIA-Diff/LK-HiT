"""Label graph: positive-PMI co-occurrence edges plus statutory-structure edges."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from lkhit.data.descriptions import RESOURCES_DIR
from lkhit.data.tasks import TaskSpec
from lkhit.utils import load_json

logger = logging.getLogger("lkhit.data.label_graph")


def cooccurrence_pmi(label_sets: Sequence[Sequence[int]], n_labels: int, eps: float = 1e-12) -> np.ndarray:
    """Normalised PMI between labels, kept only where PMI > 0, zero diagonal.

    ``npmi(i, j) = pmi(i, j) / -log p(i, j)`` lies in (0, 1] for positively
    associated pairs, which makes co-occurrence weights comparable with the unit
    weights of structural edges.
    """
    n_docs = len(label_sets)
    if n_docs == 0:
        return np.zeros((n_labels, n_labels), dtype=np.float32)
    single = np.zeros(n_labels, dtype=np.float64)
    joint = np.zeros((n_labels, n_labels), dtype=np.float64)
    for labels in label_sets:
        labels = sorted(set(labels))
        for a in labels:
            single[a] += 1
            for b in labels:
                if a != b:
                    joint[a, b] += 1
    p_single = single / n_docs
    p_joint = joint / n_docs
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((p_joint + eps) / (np.outer(p_single, p_single) + eps))
        npmi = pmi / (-np.log(p_joint + eps))
    weights = np.where((p_joint > 0) & (pmi > 0), npmi, 0.0)
    np.fill_diagonal(weights, 0.0)
    return weights.astype(np.float32)


def structural_adjacency(label_names: Sequence[str], groups: dict[str, str] | None, weight: float = 1.0) -> np.ndarray:
    """Unit-weight edges between labels that share a chapter / section / micro-thesaurus."""
    n = len(label_names)
    adj = np.zeros((n, n), dtype=np.float32)
    if not groups:
        return adj
    for i, a in enumerate(label_names):
        for j, b in enumerate(label_names):
            if i != j and a in groups and b in groups and groups[a] == groups[b]:
                adj[i, j] = weight
    return adj


def normalise_adjacency(adjacency: np.ndarray) -> np.ndarray:
    """Symmetric normalisation ``D^{-1/2} (A + I) D^{-1/2}`` used by the graph convolution."""
    a_hat = adjacency.astype(np.float64) + np.eye(adjacency.shape[0])
    degree = a_hat.sum(axis=1)
    d_inv_sqrt = np.power(degree, -0.5, where=degree > 0, out=np.zeros_like(degree))
    return (d_inv_sqrt[:, None] * a_hat * d_inv_sqrt[None, :]).astype(np.float32)


def echr_groups(label_names: Sequence[str], resources_dir: Path | None = None) -> dict[str, str]:
    """Convention sections: Articles 2-3, 5-6 and 8-11 form groups; Article 14 and P1-1 are singletons."""
    sections = load_json((resources_dir or RESOURCES_DIR) / "echr_sections.json")
    groups: dict[str, str] = {}
    for section, members in sections.items():
        for article in members:
            if article in label_names:
                groups[article] = section
    return groups


def _article_number(name: str) -> int | None:
    """``264``, ``133-1`` and ``第234条`` all resolve to the leading article number."""
    text = str(name).strip().lstrip("第").rstrip("条")
    head = text.split("-")[0].split("之")[0]
    try:
        return int(head)
    except ValueError:
        return None


def criminal_law_groups(label_names: Sequence[str], resources_dir: Path | None = None) -> dict[str, str]:
    """Chapter of the Special Provisions of the Criminal Law that each article belongs to."""
    chapters = load_json((resources_dir or RESOURCES_DIR) / "criminal_law_chapters.json")
    groups: dict[str, str] = {}
    for name in label_names:
        article = _article_number(name)
        if article is None:
            continue
        for chapter in chapters:
            if chapter["first_article"] <= article <= chapter["last_article"]:
                groups[name] = chapter["id"]
                break
    return groups


def structural_groups(spec: TaskSpec, cfg: dict, description_groups: dict[str, str] | None = None) -> dict[str, str]:
    resources_dir = Path((cfg["data"].get("knowledge") or {}).get("resources_dir", RESOURCES_DIR))
    if spec.name in ("ecthr_a", "ecthr_b"):
        return echr_groups(spec.label_names, resources_dir)
    if spec.name == "cail2018":
        return criminal_law_groups(spec.label_names, resources_dir)
    if spec.name == "eurlex":
        return dict(description_groups or {})
    return {}


def build_label_graph(
    spec: TaskSpec,
    train_label_sets: Sequence[Sequence[int]],
    groups: dict[str, str],
    use_cooccurrence: bool = True,
    use_structure: bool = True,
) -> dict[str, np.ndarray]:
    n = spec.n_labels
    if spec.multi_label and use_cooccurrence:
        cooc = cooccurrence_pmi(train_label_sets, n)
    else:
        cooc = np.zeros((n, n), dtype=np.float32)
    struct = structural_adjacency(spec.label_names, groups if use_structure else None)
    combined = cooc + struct
    adjacency = normalise_adjacency(combined)
    logger.info(
        "label graph for %s: %d co-occurrence edges, %d structural edges, %d isolated labels",
        spec.name,
        int((cooc > 0).sum() // 2),
        int((struct > 0).sum() // 2),
        int(((combined > 0).sum(axis=1) == 0).sum()),
    )
    return {"cooccurrence": cooc, "structure": struct, "adjacency": adjacency}


def save_label_graph(graph: dict[str, np.ndarray], spec: TaskSpec, groups: dict[str, str], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        cooccurrence=graph["cooccurrence"],
        structure=graph["structure"],
        adjacency=graph["adjacency"],
        label_names=np.array(spec.label_names, dtype=object),
        groups=np.array([groups.get(n, "") for n in spec.label_names], dtype=object),
    )


def load_label_graph(path: Path, spec: TaskSpec) -> np.ndarray:
    payload = np.load(path, allow_pickle=True)
    names = [str(n) for n in payload["label_names"].tolist()]
    if names != list(spec.label_names):
        raise ValueError(f"label order in {path} does not match the task specification")
    return payload["adjacency"].astype(np.float32)


def graph_statistics(graph: dict[str, np.ndarray]) -> dict[str, float]:
    cooc, struct = graph["cooccurrence"], graph["structure"]
    n = cooc.shape[0]
    degree = ((cooc + struct) > 0).sum(axis=1)
    return {
        "n_labels": int(n),
        "cooccurrence_edges": int((cooc > 0).sum() // 2),
        "structure_edges": int((struct > 0).sum() // 2),
        "mean_degree": float(degree.mean()),
        "isolated_labels": int((degree == 0).sum()),
        "mean_cooccurrence_weight": float(cooc[cooc > 0].mean()) if (cooc > 0).any() else 0.0,
    }
