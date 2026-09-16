"""Checkpoint I/O: ``best.pt`` (weights selected on development macro-F1) and ``last.pt`` (resumable state)."""

from __future__ import annotations

import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger("lkhit.engine.checkpoint")


def checkpoint_dir(run_dir: Path) -> Path:
    path = Path(run_dir) / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _rng_state() -> dict[str, Any]:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_best(run_dir: Path, model: torch.nn.Module, epoch: int, dev_metrics: dict, cfg: dict, spec_dict: dict, seed: int) -> Path:
    path = checkpoint_dir(run_dir) / "best.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": epoch,
            "dev_metrics": dev_metrics,
            "config": cfg,
            "task_spec": spec_dict,
            "seed": seed,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "torch_version": torch.__version__,
        },
        path,
    )
    return path


def save_last(
    run_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    global_step: int,
    history: list[dict],
    best: dict,
    cfg: dict,
    spec_dict: dict,
    seed: int,
) -> Path:
    path = checkpoint_dir(run_dir) / "last.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "global_step": global_step,
            "history": history,
            "best": best,
            "config": cfg,
            "task_spec": spec_dict,
            "seed": seed,
            "rng": _rng_state(),
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        path,
    )
    return path


def load_weights(model: torch.nn.Module, path: Path, device: torch.device, strict: bool = True) -> dict:
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload["model"] if "model" in payload else payload
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing or unexpected:
        logger.warning("loaded %s with %d missing and %d unexpected keys", path, len(missing), len(unexpected))
    else:
        logger.info("loaded weights from %s (epoch %s)", path, payload.get("epoch"))
    return payload


def resume_state(path: Path, model, optimizer, scheduler, scaler, device: torch.device) -> dict:
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if "rng" in payload:
        _restore_rng_state(payload["rng"])
    logger.info("resumed from %s at epoch %d (global step %d)", path, payload["epoch"], payload["global_step"])
    return payload
