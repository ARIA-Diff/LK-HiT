"""Argument-parsing helpers shared by the command-line entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lkhit.config import load_config, run_directory
from lkhit.data.tasks import TaskSpec
from lkhit.utils import get_device, load_json, seed_everything, setup_logging


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="experiment YAML under configs/experiments/")
    parser.add_argument("--seed", type=int, default=1, help="random seed (the paper uses 1-5)")
    parser.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE", help="dotted config overrides, e.g. train.batch_size=2")
    parser.add_argument("--device", default="auto", help="cuda | cpu | auto")
    parser.add_argument("--run-root", default=None, help="override run.root (default: runs/)")


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", required=True, help="run directory written by lkhit.cli.train (runs/<task>/<model>/seed<k>)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None, help="evaluation batch size (default: train.eval_batch_size)")


def resolve_config(args: argparse.Namespace) -> tuple[dict, Path]:
    overrides = list(args.override or [])
    if args.run_root:
        overrides.append(f"run.root={args.run_root}")
    cfg = load_config(args.config, overrides)
    cfg.setdefault("train", {})["seed"] = args.seed
    return cfg, run_directory(cfg, args.seed)


def load_run(run_path: str | Path) -> tuple[dict, Path]:
    run_dir = Path(run_path)
    config_file = run_dir / "config.yaml"
    if not config_file.exists():
        raise FileNotFoundError(f"{config_file} not found; is {run_dir} a run directory?")
    import yaml

    with open(config_file, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg, run_dir


def load_run_spec(run_dir: Path) -> TaskSpec:
    return TaskSpec.from_dict(load_json(run_dir / "task_spec.json"))


def begin(cfg: dict, run_dir: Path, seed: int, device_name: str, log_name: str = "lkhit"):
    seed_everything(seed, deterministic=bool(cfg.get("train", {}).get("deterministic", False)))
    logger = setup_logging(run_dir, name=log_name)
    device = get_device(device_name)
    logger.info("run directory %s | device %s | seed %d", run_dir, device, seed)
    if device.type == "cuda":
        logger.info("GPU %s (%.1f GB)", torch.cuda.get_device_name(device), torch.cuda.get_device_properties(device).total_memory / 1024**3)
    return logger, device


def dump_args(args: argparse.Namespace, run_dir: Path, name: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / name, "w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, indent=2, default=str)
