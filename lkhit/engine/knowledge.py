"""Label knowledge for a run: statute descriptions and the label graph, built once per task and cached."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np

from lkhit.data.dataset import Document, label_sets
from lkhit.data.descriptions import (
    build_label_descriptions,
    description_lengths,
    load_label_descriptions,
    load_label_groups,
    save_label_descriptions,
)
from lkhit.data.label_graph import build_label_graph, graph_statistics, load_label_graph, save_label_graph, structural_groups
from lkhit.data.tasks import TaskSpec
from lkhit.utils import save_json

logger = logging.getLogger("lkhit.engine.knowledge")


def processed_dir(cfg: dict) -> Path:
    data = cfg["data"]
    return Path(data.get("processed_dir") or Path("data/processed") / data["task"])


def descriptions_path(cfg: dict) -> Path:
    knowledge = cfg["data"].get("knowledge", {}) or {}
    return Path(knowledge.get("descriptions_file") or processed_dir(cfg) / "label_descriptions.json")


def graph_path(cfg: dict) -> Path:
    knowledge = cfg["data"].get("knowledge", {}) or {}
    return Path(knowledge.get("graph_file") or processed_dir(cfg) / "label_graph.npz")


def ensure_label_descriptions(cfg: dict, spec: TaskSpec, rebuild: bool = False) -> tuple[list[str], dict[str, str] | None]:
    """Return the ordered descriptions and (for EUR-LEX) the micro-thesaurus groups, building the file if needed."""
    path = descriptions_path(cfg)
    if path.exists() and not rebuild:
        descriptions = load_label_descriptions(path, spec)
        groups = load_label_groups(path)
        logger.info("loaded %d label descriptions from %s", len(descriptions), path)
        return descriptions, groups
    logger.info("building label descriptions for %s", spec.name)
    mapping, groups = build_label_descriptions(cfg, spec)
    save_label_descriptions(mapping, groups, path)
    stats = description_lengths(mapping, spec.language)
    logger.info("label descriptions: mean %.1f %s (min %d, max %d) -> %s", stats["mean"], stats["unit"], stats["min"], stats["max"], path)
    return [mapping[n] for n in spec.label_names], groups


def ensure_label_graph(
    cfg: dict,
    spec: TaskSpec,
    train_documents: Sequence[Document] | None,
    description_groups: dict[str, str] | None,
    rebuild: bool = False,
) -> np.ndarray:
    """Normalised adjacency of the label graph, built from the training split and the structural groups."""
    path = graph_path(cfg)
    if path.exists() and not rebuild:
        adjacency = load_label_graph(path, spec)
        logger.info("loaded label graph from %s", path)
        return adjacency
    if train_documents is None:
        raise FileNotFoundError(f"{path} not found and no training documents were given to build it")
    graph_cfg = cfg["model"].get("label_graph", {}) or {}
    groups = structural_groups(spec, cfg, description_groups)
    graph = build_label_graph(
        spec,
        label_sets(train_documents, spec),
        groups,
        use_cooccurrence=bool(graph_cfg.get("cooccurrence", True)),
        use_structure=bool(graph_cfg.get("structure", True)),
    )
    save_label_graph(graph, spec, groups, path)
    save_json(graph_statistics(graph), path.with_suffix(".stats.json"))
    logger.info("label graph written to %s: %s", path, graph_statistics(graph))
    return graph["adjacency"]


def copy_knowledge_into_run(cfg: dict, run_dir: Path, with_graph: bool) -> None:
    """Keep the exact descriptions and graph a run was trained with next to its checkpoints."""
    run_dir.mkdir(parents=True, exist_ok=True)
    src = descriptions_path(cfg)
    if src.exists():
        shutil.copyfile(src, run_dir / "label_descriptions.json")
    if with_graph:
        src = graph_path(cfg)
        if src.exists():
            shutil.copyfile(src, run_dir / "label_graph.npz")


def load_run_knowledge(run_dir: Path, spec: TaskSpec, with_graph: bool) -> tuple[list[str] | None, np.ndarray | None]:
    descriptions = None
    adjacency = None
    desc_file = run_dir / "label_descriptions.json"
    if desc_file.exists():
        descriptions = load_label_descriptions(desc_file, spec)
    graph_file = run_dir / "label_graph.npz"
    if with_graph and graph_file.exists():
        adjacency = load_label_graph(graph_file, spec)
    return descriptions, adjacency
