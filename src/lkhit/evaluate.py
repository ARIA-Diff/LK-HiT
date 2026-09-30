"""Test-set evaluation: discrimination, calibration, abstention, subgroups, rationales."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from lkhit.config import load_config
from lkhit.data import class_counts
from lkhit.metrics import (
    brier_score,
    discrimination,
    expected_calibration_error,
    frequency_tiers,
    gain_frequency_correlation,
    micro_f1_at_coverage,
    negative_log_likelihood,
    paired_bootstrap_macro_f1,
    per_label_f1,
    probabilities,
    selective_prediction,
    tier_macro_f1,
)
from lkhit.model import LKHiT, SequenceClassifier
from lkhit.runtime import load_splits, load_tokenizer, make_loader, move_batch, prepare_lkhit
from lkhit.rationales import (
    agreement_at_k,
    label_aware_rationale,
    lead_rationale,
    mean_agreement,
    random_rationale,
    tfidf_rationale,
)
def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    return value


def flatten(value: dict, prefix: str = "") -> dict[str, float]:
    flat = {}
    for key, item in value.items():
        name = f"{prefix}{key}"
        if isinstance(item, dict):
            flat.update(flatten(item, name + "."))
        elif isinstance(item, (int, float)) and not isinstance(item, bool) and np.isfinite(item):
            flat[name] = float(item)
    return flat


def summarise_runs(reports: list[dict]) -> dict:
    rows = [flatten(report) for report in reports]
    keys = sorted(set().union(*(row.keys() for row in rows)))
    summary = {}
    for key in keys:
        values = np.array([row[key] for row in rows if key in row], dtype=np.float64)
        summary[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=1) if len(values) > 1 else 0.0)}
    return summary


@torch.no_grad()
def collect_outputs(model, loader, device, use_amp: bool) -> dict:
    model.eval()
    logits, targets = [], []
    years, n_paragraphs, n_tokens, n_chars = [], [], [], []
    attentions = []
    rationales, paragraphs = [], []
    for batch in loader:
        moved = move_batch(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            output = model(moved)
        logits.append(output["logits"].float().cpu())
        targets.append(batch["labels"])
        years.append(batch["year"])
        n_paragraphs.append(batch["n_paragraphs"])
        n_tokens.append(batch["n_tokens"])
        n_chars.append(batch["n_chars"])
        if output["attention"] is not None:
            attentions.append(output["attention"].float().cpu())
        rationales.extend(batch["rationales"])
        if "paragraphs" in batch:
            paragraphs.extend(batch["paragraphs"])
    return {
        "logits": torch.cat(logits).numpy(),
        "targets": torch.cat(targets).numpy(),
        "year": torch.cat(years).numpy(),
        "n_paragraphs": torch.cat(n_paragraphs).numpy(),
        "n_tokens": torch.cat(n_tokens).numpy(),
        "n_chars": torch.cat(n_chars).numpy(),
        "attention": torch.cat(attentions).numpy() if attentions else None,
        "rationales": rationales,
        "paragraphs": paragraphs or None,
    }


def _subset_metric(probs, targets, mask, single_label: bool, threshold: float) -> dict | None:
    if int(mask.sum()) == 0:
        return None
    scores = discrimination(probs[mask], targets[mask], single_label, threshold)
    key = "accuracy" if single_label else "micro_f1"
    return {"n": int(mask.sum()), key: scores[key]}


def length_groups(task: str, outputs: dict) -> dict[str, np.ndarray]:
    if task in {"ecthr_a", "ecthr_b"}:
        count = outputs["n_paragraphs"]
        return {"<=20": count <= 20, "21-50": (count >= 21) & (count <= 50), ">50": count > 50}
    if task == "eurlex":
        count = outputs["n_tokens"]
        return {"<=512": count <= 512, "513-2048": (count >= 513) & (count <= 2048), ">2048": count > 2048}
    count = outputs["n_chars"]
    return {"<=200": count <= 200, "201-500": (count >= 201) & (count <= 500), ">500": count > 500}


def score_rationales(outputs: dict, probs: np.ndarray, label_texts: list[str], k: int, seed: int, single_label: bool, threshold: float) -> dict:
    attention = outputs["attention"]
    paragraphs = outputs["paragraphs"]
    if paragraphs is None:
        return {}
    pred = (probs >= threshold).astype(np.int64) if not single_label else probs.argmax(axis=-1)
    rng = np.random.default_rng(seed)
    buckets = {"label_aware_attention": [], "random": [], "lead5": [], "tfidf": []}
    for index, gold in enumerate(outputs["rationales"]):
        if not gold:
            continue
        text = paragraphs[index]
        width = attention.shape[-1] if attention is not None else len(text)
        n_para = min(len(text), width)
        text = text[:n_para]
        if n_para == 0:
            continue
        if single_label:
            predicted = np.zeros(probs.shape[1], dtype=np.int64)
            predicted[int(pred[index])] = 1
            fallback = int(pred[index])
        else:
            predicted = pred[index]
            fallback = int(probs[index].argmax())
        if attention is not None:
            selected = label_aware_rationale(attention[index][:, :n_para], predicted, k, fallback)
            buckets["label_aware_attention"].append(agreement_at_k(selected, gold, k))
        buckets["random"].append(agreement_at_k(random_rationale(n_para, k, rng), gold, k))
        buckets["lead5"].append(agreement_at_k(lead_rationale(n_para, k), gold, k))
        active = np.where(predicted > 0)[0]
        article = " ".join(label_texts[int(label)] for label in (active if len(active) else [fallback]))
        buckets["tfidf"].append(agreement_at_k(tfidf_rationale(text, article, k), gold, k))
    return {name: mean_agreement(rows) for name, rows in buckets.items() if rows}


def run_evaluation(model, loader, cfg, temperature: float, label_spec: dict, train_counts: np.ndarray, device, seed: int = 1, baseline_probs: np.ndarray | None = None) -> dict:
    use_amp = bool(cfg.get("mixed_precision", False)) and device.type == "cuda"
    outputs = collect_outputs(model, loader, device, use_amp)
    single = bool(cfg["single_label"])
    threshold = float(cfg["threshold"])
    targets = outputs["targets"]
    raw = probabilities(outputs["logits"], 1.0, single)
    calibrated = probabilities(outputs["logits"], temperature, single)
    report = discrimination(calibrated, targets, single, threshold)
    report["ece_raw"] = expected_calibration_error(raw, targets, single, int(cfg.get("ece_bins", 15)))
    report["ece_ts"] = expected_calibration_error(calibrated, targets, single, int(cfg.get("ece_bins", 15)))
    report["brier_ts"] = brier_score(calibrated, targets, single)
    report["nll_ts"] = negative_log_likelihood(calibrated, targets, single)
    selective = selective_prediction(calibrated, targets, single, threshold)
    report["aurc"] = selective["aurc"]
    report["coverage_at_10_risk"] = selective["coverage_at_10_risk"]
    if not single:
        report["micro_f1_at_coverage"] = {
            "1.0": micro_f1_at_coverage(calibrated, targets, 1.0, threshold),
            "0.8": micro_f1_at_coverage(calibrated, targets, 0.8, threshold),
            "0.6": micro_f1_at_coverage(calibrated, targets, 0.6, threshold),
        }
    tiers = frequency_tiers(train_counts, cfg["task"])
    if tiers:
        report["tiers"] = {name: tier_macro_f1(calibrated, targets, index, single) for name, index in tiers.items()}
    if baseline_probs is not None:
        gain = per_label_f1(calibrated, targets, single) - per_label_f1(baseline_probs, targets, single)
        report["gain_vs_frequency"] = gain_frequency_correlation(gain, train_counts)
        report["bootstrap_macro_f1"] = paired_bootstrap_macro_f1(
            targets,
            calibrated,
            baseline_probs,
            single,
            n_resamples=int(cfg.get("bootstrap_samples", 10000)),
            seed=seed,
            threshold=threshold,
        )
    year = outputs["year"]
    if np.any(year > 0):
        report["by_year"] = {}
        for value in (2017, 2018, 2019):
            grouped = _subset_metric(calibrated, targets, year == value, single, threshold)
            if grouped:
                report["by_year"][str(value)] = grouped
    report["by_length"] = {}
    for name, mask in length_groups(cfg["task"], outputs).items():
        grouped = _subset_metric(calibrated, targets, mask, single, threshold)
        if grouped:
            report["by_length"][name] = grouped
    rationales = score_rationales(outputs, calibrated, label_spec["texts"], int(cfg.get("rationale_k", 5)), seed, single, threshold)
    if rationales:
        report["rationales"] = rationales
    return json_ready(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained LK-HiT checkpoint on the test split.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--baseline-probs", default=None, help="NumPy .npy of another model's test probabilities, for bootstrap and long-tail gains.")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    try:
        checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.ckpt, map_location="cpu")
    cfg = {**checkpoint.get("config", {}), **cfg}
    splits, label_spec = load_splits(cfg)
    n_labels = len(label_spec["names"])
    tokenizer = load_tokenizer(cfg)
    tokenizer_name = tokenizer.name_or_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg["arch"] == "lkhit":
        model = LKHiT(cfg, n_labels)
        prepare_lkhit(model, tokenizer, label_spec, splits["train"], device)
    else:
        model = SequenceClassifier(tokenizer_name, n_labels).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    ignored = {"label_input_ids", "label_attention_mask"}
    missing = [key for key in missing if key not in ignored]
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch. Missing: {missing}. Unexpected: {unexpected}.")
    model.to(device)
    loader = make_loader(splits["test"], tokenizer, cfg, n_labels, shuffle=False, keep_text=True)
    counts = class_counts(splits["train"], n_labels, bool(cfg["single_label"]))
    baseline = np.load(args.baseline_probs) if args.baseline_probs else None
    report = run_evaluation(
        model,
        loader,
        cfg,
        float(checkpoint.get("temperature", 1.0)),
        label_spec,
        counts,
        device,
        seed=int(cfg.get("seed", 1)),
        baseline_probs=baseline,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
