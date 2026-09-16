"""YAML configuration loading with ``defaults`` composition and dotted overrides.

An experiment file lists the fragments it is built from::

    defaults:
      - data/ecthr_a
      - models/lkhit
      - train/default
    run:
      name: lkhit

Fragments are resolved relative to the nearest ``configs/`` ancestor of the file,
merged in order, and the file's own keys are applied last. Command-line
overrides use ``section.key=value`` with YAML-parsed values.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml

CONFIG_DIR_NAME = "configs"


def find_configs_root(path: Path) -> Path:
    for parent in [path] + list(path.parents):
        if parent.name == CONFIG_DIR_NAME:
            return parent
    raise FileNotFoundError(f"{path} is not located inside a '{CONFIG_DIR_NAME}' directory")


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _resolve_defaults(path: Path, root: Path, seen: set[Path]) -> dict:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"circular 'defaults' reference through {path}")
    seen = seen | {path}
    raw = load_yaml(path)
    merged: dict = {}
    for item in raw.pop("defaults", None) or []:
        fragment = root / f"{item}.yaml"
        if not fragment.exists():
            raise FileNotFoundError(f"config fragment '{item}' referenced by {path} not found at {fragment}")
        merged = deep_merge(merged, _resolve_defaults(fragment, root, seen))
    return deep_merge(merged, raw)


def parse_override(item: str) -> tuple[str, Any]:
    if "=" not in item:
        raise ValueError(f"override '{item}' must have the form section.key=value")
    key, raw_value = item.split("=", 1)
    return key.strip(), yaml.safe_load(raw_value)


def set_by_path(cfg: dict, dotted: str, value: Any) -> None:
    node = cfg
    parts = dotted.split(".")
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def get_by_path(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> dict:
    path = Path(path).resolve()
    root = find_configs_root(path)
    cfg = _resolve_defaults(path, root, set())
    for item in overrides or []:
        key, value = parse_override(item)
        set_by_path(cfg, key, value)
    cfg.setdefault("_meta", {})["config_path"] = str(path)
    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)


def resolve_encoder_name(cfg: dict, role: str | None = None) -> str:
    """Map an encoder role (``legal``, ``general``, ``sparse`` ...) to a checkpoint name.

    Roles are declared per task under ``data.encoders`` so that the same model
    fragment can be composed with English and Chinese corpora. A value that is
    not a declared role is treated as a checkpoint name or local path.
    """
    role = role or cfg["model"]["encoder"]
    table = cfg.get("data", {}).get("encoders", {}) or {}
    return table.get(role, role)


def run_directory(cfg: dict, seed: int) -> Path:
    root = Path(cfg.get("run", {}).get("root", "runs"))
    task = cfg["data"]["task"]
    name = cfg["run"]["name"]
    return root / task / name / f"seed{seed}"
