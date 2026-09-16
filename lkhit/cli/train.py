"""Train one configuration with one seed, then calibrate on the development split and evaluate on the test split.

    python -m lkhit.cli.train --config configs/experiments/ecthr_a/lkhit.yaml --seed 1

Writes ``runs/<task>/<run.name>/seed<k>/`` with ``config.yaml``, ``task_spec.json``,
``train.log``, ``metrics.jsonl``, ``checkpoints/{best,last}.pt``,
``temperature.json``, ``predictions/{dev,test}.npz`` and ``metrics/{dev,test}.json``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from lkhit.cli.common import add_config_arguments, begin, dump_args, resolve_config
from lkhit.config import save_config
from lkhit.engine.evaluator import Evaluator, Predictions, fit_temperature, metric_report, save_predictions, save_report
from lkhit.engine.setup import instantiate_model, prepare_inputs
from lkhit.engine.trainer import Trainer, amp_dtype_from_config
from lkhit.engine.checkpoint import checkpoint_dir, load_weights
from lkhit.losses import build_loss
from lkhit.models import EXTERNAL_ARCHITECTURES, SKLEARN_ARCHITECTURES
from lkhit.rationales import evaluate_rationales
from lkhit.utils import Timer, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arguments(parser)
    parser.add_argument("--no-test", action="store_true", help="skip the final test-set evaluation")
    parser.add_argument("--rationales", action="store_true", help="also score paragraph rationales against the ECtHR-A silver rationales")
    parser.add_argument("--overwrite", action="store_true", help="train even if the run directory already holds a best.pt")
    return parser.parse_args()


def evaluate_split(cfg, model, inputs, split, run_dir, device, temperature, logger, collect_attention):
    train_cfg = cfg["train"]
    evaluator = Evaluator(model, inputs.spec, device, amp_dtype=amp_dtype_from_config(train_cfg, device))
    preds = evaluator.predict(inputs.loaders[split], collect_attention=collect_attention)
    report = metric_report(
        preds,
        inputs.spec,
        temperature=temperature,
        train_counts=inputs.train_counts,
        documents=inputs.documents[split],
        subgroup_cfg=cfg["data"].get("subgroups"),
        target_risk=float(train_cfg.get("target_risk", 0.10)),
    )
    save_predictions(preds, inputs.spec, run_dir / "predictions" / f"{split}.npz", temperature=temperature, with_attention=collect_attention)
    save_report(report, run_dir / "metrics" / f"{split}.json")
    return preds, report


def score_rationales(cfg, inputs, preds, run_dir, logger) -> None:
    documents = [inputs.documents["test"][i] for i in preds.index]
    if not any("silver_rationales" in doc.meta for doc in documents):
        logger.warning("no silver rationales attached to the test documents; skipping the rationale analysis")
        return
    from lkhit.calibration import logits_to_probs

    probs = logits_to_probs(preds.logits, inputs.spec.multi_label)
    result = evaluate_rationales(
        probs,
        inputs.spec,
        documents,
        label_attention=preds.label_attention,
        document_attention=preds.document_attention,
        descriptions=inputs.descriptions,
        k=int(cfg["train"].get("rationale_k", 5)),
    )
    save_json(result, run_dir / "metrics" / "rationales_test.json")
    for name, scores in result.items():
        logger.info("rationales %-24s P@5 %.1f R@5 %.1f F1@5 %.1f (n=%d)", name, scores["precision_at_k"], scores["recall_at_k"], scores["f1_at_k"], scores["n_documents"])


def train_sklearn(cfg, run_dir, seed, logger) -> None:
    from lkhit.data.loading import load_task
    from lkhit.engine.setup import load_split_documents
    from lkhit.models.tfidf_svm import TfidfSvmClassifier
    from lkhit.data.dataset import label_counts

    spec = load_task(cfg)
    save_json(spec.to_dict(), run_dir / "task_spec.json")
    documents = load_split_documents(cfg, ("train", "dev", "test"))
    train_counts = label_counts(documents["train"], spec)
    clf = TfidfSvmClassifier(spec, cfg["model"])
    timer = Timer()
    fit_info = clf.fit(documents["train"], documents["dev"])
    fit_info["fit_time_s"] = timer.elapsed()
    save_json(fit_info, run_dir / "fit.json")
    clf.save(checkpoint_dir(run_dir) / "model.joblib")
    for split in ("dev", "test"):
        timer.reset()
        logits = clf.predict_logits(documents[split])
        labels = clf.targets(documents[split])
        preds = Predictions(logits=logits, labels=labels, index=np.arange(len(labels)), wall_time=timer.elapsed())
        report = metric_report(preds, spec, train_counts=train_counts, documents=documents[split], subgroup_cfg=cfg["data"].get("subgroups"))
        save_predictions(preds, spec, run_dir / "predictions" / f"{split}.npz")
        save_report(report, run_dir / "metrics" / f"{split}.json")


def main() -> None:
    args = parse_args()
    cfg, run_dir = resolve_config(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger, device = begin(cfg, run_dir, args.seed, args.device)
    save_config(cfg, run_dir / "config.yaml")
    dump_args(args, run_dir, "train_args.json")
    arch = cfg["model"]["arch"]

    if arch in EXTERNAL_ARCHITECTURES:
        raise SystemExit("the zero-shot LLM baseline is run with `python -m lkhit.cli.zero_shot_llm`")
    if arch in SKLEARN_ARCHITECTURES:
        train_sklearn(cfg, run_dir, args.seed, logger)
        return

    best_file = checkpoint_dir(run_dir) / "best.pt"
    if best_file.exists() and not args.overwrite and not (checkpoint_dir(run_dir) / "last.pt").exists():
        raise SystemExit(f"{best_file} exists; pass --overwrite to retrain or use lkhit.cli.evaluate")

    splits = ("train", "dev") if args.no_test else ("train", "dev", "test")
    inputs = prepare_inputs(cfg, splits, run_dir, args.seed, for_training=True)
    save_json(inputs.spec.to_dict(), run_dir / "task_spec.json")
    if inputs.train_counts is not None:
        save_json({n: int(c) for n, c in zip(inputs.spec.label_names, inputs.train_counts)}, run_dir / "train_label_counts.json")

    model = instantiate_model(cfg, inputs, device)
    loss_fn = build_loss(cfg, inputs.spec.multi_label, inputs.train_counts).to(device)
    logger.info("loss: %s", loss_fn)

    trainer = Trainer(cfg, inputs.spec, model, loss_fn, inputs.loaders["train"], inputs.loaders["dev"], run_dir, device, args.seed, inputs.train_counts)
    started = time.perf_counter()
    summary = trainer.fit()
    summary["train_time_s"] = time.perf_counter() - started
    summary.pop("history", None)
    logger.info("training finished: best dev %s %.2f at epoch %d (%.1f min)", trainer.selection_metric, summary["best_score"] or 0.0, summary["best_epoch"], summary["train_time_s"] / 60)

    # ---- calibration on the development split, evaluation of the selected checkpoint
    load_weights(model, best_file, device)
    evaluator = Evaluator(model, inputs.spec, device, amp_dtype=amp_dtype_from_config(cfg["train"], device))
    dev_preds = evaluator.predict(inputs.loaders["dev"])
    scaler = fit_temperature(dev_preds, inputs.spec, max_iter=int((cfg["train"].get("calibration") or {}).get("max_iter", 50)))
    save_json(scaler.to_dict(), run_dir / "temperature.json")
    dev_report = metric_report(dev_preds, inputs.spec, temperature=scaler.temperature, train_counts=inputs.train_counts, documents=inputs.documents["dev"], subgroup_cfg=cfg["data"].get("subgroups"))
    save_predictions(dev_preds, inputs.spec, run_dir / "predictions" / "dev.npz", temperature=scaler.temperature)
    save_report(dev_report, run_dir / "metrics" / "dev.json")

    if not args.no_test:
        collect = bool(args.rationales) and inputs.spec.name == "ecthr_a"
        test_preds, _ = evaluate_split(cfg, model, inputs, "test", run_dir, device, scaler.temperature, logger, collect_attention=collect)
        if collect:
            score_rationales(cfg, inputs, test_preds, run_dir, logger)

    summary["temperature"] = scaler.temperature
    summary["parameters"] = model.parameter_summary()
    save_json(summary, run_dir / "summary.json")
    logger.info("done: %s", run_dir)


if __name__ == "__main__":
    main()
