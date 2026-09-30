"""Train LK-HiT or an encoder baseline and evaluate the selected checkpoint once."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from lkhit.baselines import fit_tfidf_svm, predict_tfidf_svm
from lkhit.calibrate import fit_temperature
from lkhit.config import load_config
from lkhit.data import class_counts, multi_hot
from lkhit.evaluate import run_evaluation, summarise_runs
from lkhit.loss import class_balanced_weights, classification_loss
from lkhit.metrics import discrimination, probabilities
from lkhit.model import LKHiT, SequenceClassifier
from lkhit.runtime import ROOT, load_splits, load_tokenizer, make_loader, move_batch, prepare_lkhit


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def output_dir(cfg: dict, seed: int) -> Path:
    return ROOT / cfg.get("output_dir", f"outputs/{cfg['task']}") / f"seed_{seed}"


def build_model(cfg: dict, n_labels: int) -> torch.nn.Module:
    if cfg["arch"] == "lkhit":
        return LKHiT(cfg, n_labels)
    if cfg["arch"] == "encoder":
        return SequenceClassifier(cfg.get("model_name_or_path") or cfg["encoder"], n_labels)
    raise SystemExit(f"Unknown arch {cfg['arch']!r}")


def optimiser_and_schedule(model, cfg, steps_per_epoch: int):
    optimiser = torch.optim.AdamW(
        [
            {"params": [p for p in model.encoder_parameters() if p.requires_grad], "lr": float(cfg["lr_encoder"])},
            {"params": [p for p in model.upper_parameters() if p.requires_grad], "lr": float(cfg["lr_upper"])},
        ],
        weight_decay=float(cfg["weight_decay"]),
    )
    total_steps = steps_per_epoch * int(cfg["max_epochs"])
    warmup = int(total_steps * float(cfg["warmup_ratio"]))
    scheduler = get_linear_schedule_with_warmup(optimiser, warmup, max(total_steps, 1))
    return optimiser, scheduler


def train_epochs(model, loader, dev_loader, cfg, class_weights, device) -> tuple[int, float]:
    accum = max(1, int(cfg.get("grad_accum_steps", 1)))
    steps_per_epoch = math.ceil(len(loader) / accum)
    optimiser, scheduler = optimiser_and_schedule(model, cfg, steps_per_epoch)
    use_amp = bool(cfg["mixed_precision"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None
    best_f1 = -1.0
    best_epoch = 0
    patience = 0
    best_state = None
    weights = class_weights.to(device) if class_weights is not None else None

    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        model.train()
        optimiser.zero_grad(set_to_none=True)
        running = 0.0
        seen = 0
        for step, batch in enumerate(tqdm(loader, desc=f"epoch {epoch}", leave=False), start=1):
            batch = move_batch(batch, device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(batch)["logits"]
                loss = classification_loss(logits, batch["labels"], cfg, weights) / accum
            if scaler is None:
                loss.backward()
            else:
                scaler.scale(loss).backward()
            running += float(loss.detach()) * accum * batch["labels"].shape[0]
            seen += batch["labels"].shape[0]
            if step % accum == 0 or step == len(loader):
                if scaler is None:
                    optimiser.step()
                else:
                    scaler.step(optimiser)
                    scaler.update()
                scheduler.step()
                optimiser.zero_grad(set_to_none=True)
        dev_f1 = dev_macro_f1(model, dev_loader, cfg, device, use_amp)
        print(f"epoch {epoch} train_loss {running / max(seen, 1):.4f} dev_macro_f1 {dev_f1:.4f}")
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_epoch = epoch
            patience = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            patience += 1
            if patience >= int(cfg["patience"]):
                print(f"early stopping at epoch {epoch}")
                break
    if best_state is None:
        raise RuntimeError("Training produced no checkpoint.")
    model.load_state_dict(best_state)
    return best_epoch, best_f1


@torch.no_grad()
def dev_macro_f1(model, loader, cfg, device, use_amp: bool) -> float:
    model.eval()
    logits, targets = [], []
    single = bool(cfg["single_label"])
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits.append(model(batch)["logits"].float().cpu())
        targets.append(batch["labels"].cpu())
    scores = probabilities(torch.cat(logits).numpy(), temperature=1.0, single_label=single)
    gold = torch.cat(targets).numpy()
    return discrimination(scores, gold, single, float(cfg["threshold"]))["macro_f1"]


def save_checkpoint(path: Path, model, epoch: int, dev_f1: float, temperature: float, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "epoch": epoch,
            "dev_macro_f1": dev_f1,
            "temperature": temperature,
            "config": cfg,
        },
        path,
    )


def baseline_loss(cfg: dict) -> dict:
    """Neural baselines keep the shared schedule and use their original losses."""
    if cfg["arch"] != "encoder":
        return cfg
    cfg = dict(cfg)
    cfg["loss"] = "ce" if cfg["single_label"] else "bce"
    print(f"encoder baseline loss: {cfg['loss']}")
    return cfg


def train_neural(cfg: dict, seed: int) -> dict:
    cfg = baseline_loss(cfg)
    set_seed(seed)
    splits, label_spec = load_splits(cfg)
    n_labels = len(label_spec["names"])
    tokenizer = load_tokenizer(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, n_labels)
    if isinstance(model, LKHiT):
        prepare_lkhit(model, tokenizer, label_spec, splits["train"], device)
    else:
        model.to(device)
    train_loader = make_loader(splits["train"], tokenizer, cfg, n_labels, shuffle=True)
    dev_loader = make_loader(splits["dev"], tokenizer, cfg, n_labels, shuffle=False)
    counts = class_counts(splits["train"], n_labels, bool(cfg["single_label"]))
    weights = None
    if cfg["loss"] == "cbce":
        weights = class_balanced_weights(torch.from_numpy(counts), beta=float(cfg["cb_beta"]))
    epoch, dev_f1 = train_epochs(model, train_loader, dev_loader, cfg, weights, device)
    dev_logits = collect_logits(model, dev_loader, device, bool(cfg["mixed_precision"]) and device.type == "cuda")
    if cfg["single_label"]:
        dev_targets = np.array([int(record["labels"][0]) for record in splits["dev"]])
    else:
        dev_targets = multi_hot(splits["dev"], n_labels)
    temperature = fit_temperature(torch.from_numpy(dev_logits), torch.from_numpy(dev_targets), bool(cfg["single_label"]))
    print(f"temperature {temperature:.4f}")
    directory = output_dir(cfg, seed)
    save_checkpoint(directory / "best.pt", model, epoch, dev_f1, temperature, cfg)
    test_loader = make_loader(splits["test"], tokenizer, cfg, n_labels, shuffle=False, keep_text=True)
    report = run_evaluation(
        model,
        test_loader,
        cfg,
        temperature,
        label_spec,
        counts,
        device,
        seed=seed,
    )
    report["epoch"] = epoch
    report["dev_macro_f1"] = dev_f1
    report["temperature"] = temperature
    with open(directory / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({key: report[key] for key in report if key in {"micro_f1", "macro_f1", "accuracy"}}, indent=2))
    return report


def collect_logits(model, loader, device, use_amp: bool) -> np.ndarray:
    model.eval()
    pieces = []
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
            pieces.append(model(batch)["logits"].float().cpu())
    return torch.cat(pieces).numpy()


def train_svm(cfg: dict, seed: int) -> dict:
    set_seed(seed)
    splits, label_spec = load_splits(cfg)
    n_labels = len(label_spec["names"])
    single = bool(cfg["single_label"])
    if single:
        train_y = np.array([int(record["labels"][0]) for record in splits["train"]])
        test_y = np.array([int(record["labels"][0]) for record in splits["test"]])
    else:
        train_y = multi_hot(splits["train"], n_labels)
        test_y = multi_hot(splits["test"], n_labels)
    vectorizer, classifier = fit_tfidf_svm([record["paragraphs"] or [record.get("text", "")] for record in splits["train"]], train_y, single, seed)
    pred = predict_tfidf_svm(vectorizer, classifier, [record["paragraphs"] or [record.get("text", "")] for record in splits["test"]], single)
    if single:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        report = {
            "accuracy": float(accuracy_score(test_y, pred)),
            "macro_precision": float(precision_score(test_y, pred, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(test_y, pred, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(test_y, pred, average="macro", zero_division=0)),
        }
    else:
        from sklearn.metrics import f1_score

        report = {
            "micro_f1": float(f1_score(test_y, pred, average="micro", zero_division=0)),
            "macro_f1": float(f1_score(test_y, pred, average="macro", zero_division=0)),
        }
    directory = output_dir(cfg, seed)
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    return report


def run_seed(cfg: dict, seed: int) -> dict:
    cfg = dict(cfg)
    cfg["seed"] = seed
    if cfg["arch"] == "tfidf_svm":
        return train_svm(cfg, seed)
    return train_neural(cfg, seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LK-HiT.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    if args.all_seeds:
        seeds = [int(seed) for seed in cfg.get("seeds", [1, 2, 3, 4, 5])]
    else:
        seeds = [int(args.seed if args.seed is not None else cfg.get("seed", 1))]
    reports = [run_seed(cfg, seed) for seed in seeds]
    if len(reports) > 1:
        summary = summarise_runs(reports)
        path = ROOT / cfg.get("output_dir", f"outputs/{cfg['task']}") / "summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
