"""Corpus loading, segmentation, label knowledge and label-graph construction."""

from lkhit.data.dataset import Document, FlatDataset, HierarchicalDataset, collate_flat, collate_hierarchical
from lkhit.data.tasks import TaskSpec, build_task_spec

__all__ = [
    "Document",
    "FlatDataset",
    "HierarchicalDataset",
    "TaskSpec",
    "build_task_spec",
    "collate_flat",
    "collate_hierarchical",
]
