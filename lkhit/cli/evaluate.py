"""Evaluate a finished run on one split, optionally with the paragraph-rationale analysis.

    python -m lkhit.cli.evaluate --run runs/ecthr_a/lkhit/seed1 --split test
    python -m lkhit.cli.evaluate --run runs/ecthr_a/lkhit/seed1 --split test --rationales

Re-uses the temperature fitted at training time (``temperature.json``); pass
``--refit-temperature`` to fit it again on the development split.
"""

from __future__ import annotations

import argparse

from lkhit.cli.common import add_run_arguments, begin, load_run
from lkhit.engine.checkpoint import checkpoint_dir, load_weights
from lkhit.engine.evaluator import Evaluator, fit_temperature, metric_report, save_predictions, save_report
from lkhit.engine.setup import instantiate_model, prepare_inputs
from lkhit.engine.trainer import amp_dtype_from_config
from lkhit.models import SKLEARN_ARCHITECTURES
from lkhit.rationales import evaluate_rationales
from lkhit.utils import load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_arguments(parser)
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--rationales", action="store_true", help="score top-5 paragraph rationales against the ECtHR-A silver rationales")
    parser.add_argument("--refit-temperature", action="store_true")
    parser.add_argument("--checkpoint", default="best.pt", help="file under checkpoints/ to evaluate")
    parser.add_argument("--save-attention", action="store_true", help="store label-aware / document attention next to the predictions")
    return parser.parse_args()


def evaluate_sklearn(cfg, run_dir, split, logger) -> None:
    import numpy as np

    from lkhit.data.dataset import label_counts
    from lkhit.data.loading import load_task
    from lkhit.engine.evaluator import Predictions
    from lkhit.engine.setup import load_split_documents
    from lkhit.models.tfidf_svm import TfidfSvmClassifier

    spec = load_task(cfg)
    clf = TfidfSvmClassifier.load(checkpoint_dir(run_dir) / "model.joblib", cfg["model"])
    documents = load_split_documents(cfg, ("train", split) if split != "train" else ("train",))
    logits = clf.predict_logits(documents[split])
    preds = Predictions(logits=logits, labels=clf.targets(documents[split]), index=np.arange(len(documents[split])))
    report = metric_report(preds, spec, train_counts=label_counts(documents["train"], spec), documents=documents[split], subgroup_cfg=cfg["data"].get("subgroups"))
    save_predictions(preds, spec, run_dir / "predictions" / f"{split}.npz")
    save_report(report, run_dir / "metrics" / f"{split}.json")


def main() -> None:
    args = parse_args()
    cfg, run_dir = load_run(args.run)
    seed = int(cfg.get("train", {}).get("seed", 1))
    logger, device = begin(cfg, run_dir, seed, args.device, log_name="lkhit.evaluate")

    if cfg["model"]["arch"] in SKLEARN_ARCHITECTURES:
        evaluate_sklearn(cfg, run_dir, args.split, logger)
        return

    splits = ("train", args.split) if args.split != "train" else ("train",)
    if args.refit_temperature and "dev" not in splits:
        splits = splits + ("dev",)
    inputs = prepare_inputs(cfg, splits, run_dir, seed, for_training=False, eval_batch_size=args.batch_size)
    model = instantiate_model(cfg, inputs, device)
    load_weights(model, checkpoint_dir(run_dir) / args.checkpoint, device)
    evaluator = Evaluator(model, inputs.spec, device, amp_dtype=amp_dtype_from_config(cfg["train"], device))

    temperature_file = run_dir / "temperature.json"
    if args.refit_temperature or not temperature_file.exists():
        dev_preds = evaluator.predict(inputs.loaders["dev"])
        scaler = fit_temperature(dev_preds, inputs.spec)
        save_json(scaler.to_dict(), temperature_file)
        temperature = scaler.temperature
    else:
        temperature = float(load_json(temperature_file)["temperature"])
        logger.info("using the stored temperature T = %.4f", temperature)

    collect = bool(args.rationales or args.save_attention)
    preds = evaluator.predict(inputs.loaders[args.split], collect_attention=collect)
    report = metric_report(
        preds,
        inputs.spec,
        temperature=temperature,
        train_counts=inputs.train_counts,
        documents=inputs.documents[args.split],
        subgroup_cfg=cfg["data"].get("subgroups"),
        target_risk=float(cfg["train"].get("target_risk", 0.10)),
    )
    save_predictions(preds, inputs.spec, run_dir / "predictions" / f"{args.split}.npz", temperature=temperature, with_attention=args.save_attention)
    save_report(report, run_dir / "metrics" / f"{args.split}.json")

    if args.rationales:
        from lkhit.calibration import logits_to_probs

        documents = [inputs.documents[args.split][i] for i in preds.index]
        if not any("silver_rationales" in d.meta for d in documents):
            logger.warning("no silver rationales on split '%s'; nothing to score", args.split)
            return
        result = evaluate_rationales(
            logits_to_probs(preds.logits, inputs.spec.multi_label),
            inputs.spec,
            documents,
            label_attention=preds.label_attention,
            document_attention=preds.document_attention,
            descriptions=inputs.descriptions,
            k=int(cfg["train"].get("rationale_k", 5)),
        )
        save_json(result, run_dir / "metrics" / f"rationales_{args.split}.json")
        for name, scores in result.items():
            logger.info("rationales %-24s P@5 %.1f R@5 %.1f F1@5 %.1f (n=%d)", name, scores["precision_at_k"], scores["recall_at_k"], scores["f1_at_k"], scores["n_documents"])


if __name__ == "__main__":
    main()
