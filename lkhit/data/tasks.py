"""Task specifications: label spaces, segmentation geometry and evaluation conventions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from lkhit.utils import load_json

LEXGLUE_TASKS = ("ecthr_a", "ecthr_b", "eurlex")
CAIL_TASKS = ("cail2018",)

ECTHR_ARTICLES = ["2", "3", "5", "6", "8", "9", "10", "11", "14", "P1-1"]


@dataclass
class TaskSpec:
    name: str
    multi_label: bool
    language: str
    label_names: list[str]
    segmentation: str
    max_segments: int
    max_segment_tokens: int
    none_label: str | None = None
    flat_max_tokens: int = 512
    sparse_max_tokens: int = 4096
    tier_scheme: dict = field(default_factory=dict)

    @property
    def n_labels(self) -> int:
        return len(self.label_names)

    @property
    def none_index(self) -> int | None:
        if self.none_label is None:
            return None
        return self.label_names.index(self.none_label)

    def index(self, name: str) -> int:
        return self.label_names.index(name)

    @property
    def is_lexglue(self) -> bool:
        return self.name in LEXGLUE_TASKS

    @property
    def is_cail(self) -> bool:
        return self.name in CAIL_TASKS

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "multi_label": self.multi_label,
            "language": self.language,
            "label_names": list(self.label_names),
            "none_label": self.none_label,
            "segmentation": self.segmentation,
            "max_segments": self.max_segments,
            "max_segment_tokens": self.max_segment_tokens,
            "flat_max_tokens": self.flat_max_tokens,
            "sparse_max_tokens": self.sparse_max_tokens,
            "tier_scheme": dict(self.tier_scheme),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TaskSpec":
        return cls(**payload)


def build_task_spec(cfg: dict, label_names: list[str]) -> TaskSpec:
    data = cfg["data"]
    none_label = data.get("none_label")
    names = list(label_names)
    if none_label and none_label not in names:
        names.append(none_label)
    return TaskSpec(
        name=data["task"],
        multi_label=bool(data["multi_label"]),
        language=data.get("language", "en"),
        label_names=names,
        none_label=none_label,
        segmentation=data["segmentation"],
        max_segments=int(data["max_segments"]),
        max_segment_tokens=int(data["max_segment_tokens"]),
        flat_max_tokens=int(data.get("flat_max_tokens", 512)),
        sparse_max_tokens=int(data.get("sparse_max_tokens", 4096)),
        tier_scheme=dict(data.get("tiers", {}) or {}),
    )


def load_label_names(cfg: dict) -> list[str]:
    """Return the base label names of a task (without the explicit ``none`` label)."""
    data = cfg["data"]
    task = data["task"]
    if task in ("ecthr_a", "ecthr_b"):
        return list(ECTHR_ARTICLES)
    if task == "eurlex":
        from lkhit.data.lexglue import lexglue_label_names

        return lexglue_label_names(data["hf_dataset"], data["hf_config"], cache_dir=data.get("hf_cache_dir"))
    if task == "cail2018":
        label_file = Path(data["processed_dir"]) / "label_list.json"
        if not label_file.exists():
            raise FileNotFoundError(
                f"{label_file} not found; run `python -m lkhit.cli.prepare_cail` before training on CAIL2018"
            )
        return [str(a) for a in load_json(label_file)]
    raise ValueError(f"unknown task '{task}'")
