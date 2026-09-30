"""Configuration loading."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    with open(path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    for item in overrides or []:
        if "=" not in item:
            raise SystemExit(f"Override must be key=value, got {item!r}")
        key, raw = item.split("=", 1)
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw
        cfg[key] = value
    return cfg
