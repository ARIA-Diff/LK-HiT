"""Aggregate seed-level metric files and, optionally, run a paired bootstrap test.

    python -m lkhit.cli.aggregate --runs runs/ecthr_a/lkhit --split test
    python -m lkhit.cli.aggregate --compare runs/ecthr_a/lkhit runs/ecthr_a/hier_legal_bert_lsan --split test

``--runs`` writes mean ± s.d. over the seed directories that contain
``metrics/<split>.json``. ``--compare`` additionally averages the stored
test probabilities and reports the paired-bootstrap Δmacro-F1 of the first
run against the second (10,000 document resamples).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lkhit.data.tasks import TaskSpec
from lkhit.metrics import aggregate_seeds, decisions_from_probs, paired_bootstrap, seed_averaged_probabilities
from lkhit.utils import load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", help="run family directories (runs/<task>/<model>)")
    parser.add_argument("--compare", nargs=2, metavar="RUNS", help="two run families; first minus second")
    parser.add_argument("--split", default="test")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def seed_dirs(family: Path) -> list[Path]:
    family = Path(family)
    if family.name.startswith("seed"):
        return [family]
    return sorted(p for p in family.glob("seed*") if p.is_dir())


def load_seed_reports(family: Path, split: str) -> list[dict]:
    reports = []
    for seed_dir in seed_dirs(family):
        path = seed_dir / "metrics" / f"{split}.json"
        if path.exists():
            reports.append(load_json(path))
    if not reports:
        raise FileNotFoundError(f"no metrics/{split}.json under {family}")
    return reports


def flatten_discrimination(report: dict) -> dict[str, float]:
    out = dict(report.get("discrimination") or {})
    if "selective" in report:
        for key, value in report["selective"].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out[key] = value
    cal = report.get("calibration") or {}
    for tag, block in cal.items():
        if isinstance(block, dict) and "ece" in block:
            out[f"ece_{tag}"] = block["ece"]
    return {k: float(v) for k, v in out.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def load_probabilities(family: Path, split: str) -> tuple[np.ndarray, np.ndarray, TaskSpec]:
    mats, labels, spec = [], None, None
    for seed_dir in seed_dirs(family):
        pred_path = seed_dir / "predictions" / f"{split}.npz"
        if not pred_path.exists():
            continue
        payload = np.load(pred_path, allow_pickle=False)
        key = "probs_ts" if "probs_ts" in payload.files else "probs_raw"
        mats.append(payload[key])
        if labels is None:
            labels = payload["labels"]
        if spec is None:
            spec = TaskSpec.from_dict(load_json(seed_dir / "task_spec.json"))
    if not mats or spec is None or labels is None:
        raise FileNotFoundError(f"no predictions/{split}.npz under {family}")
    return seed_averaged_probabilities(mats), labels, spec


def summarise_family(family: Path, split: str) -> dict:
    reports = load_seed_reports(family, split)
    flat = [flatten_discrimination(r) for r in reports]
    return {
        "family": str(family),
        "split": split,
        "n_seeds": len(reports),
        "seeds": [p.name for p in seed_dirs(family) if (p / "metrics" / f"{split}.json").exists()],
        "aggregate": aggregate_seeds(flat),
    }


def main() -> None:
    args = parse_args()
    families = list(args.runs or [])
    if args.compare:
        families.extend(args.compare)
    if not families:
        raise SystemExit("give --runs and/or --compare")

    payload: dict = {"split": args.split, "families": {}}
    for family in families:
        family = Path(family)
        summary = summarise_family(family, args.split)
        payload["families"][str(family)] = summary
        print(f"{family}  ({summary['n_seeds']} seeds)")
        for key, stats in summary["aggregate"].items():
            print(f"  {key:24s} {stats['mean']:.2f} ± {stats['std']:.2f}")

    if args.compare:
        a, b = Path(args.compare[0]), Path(args.compare[1])
        probs_a, labels, spec = load_probabilities(a, args.split)
        probs_b, labels_b, _ = load_probabilities(b, args.split)
        if labels.shape != labels_b.shape or not np.array_equal(labels, labels_b):
            raise ValueError("the two run families do not share the same gold labels")
        pred_a = decisions_from_probs(probs_a, spec)
        pred_b = decisions_from_probs(probs_b, spec)
        test = paired_bootstrap(pred_a, pred_b, labels, n_resamples=args.resamples, seed=args.seed)
        payload["bootstrap"] = {"a": str(a), "b": str(b), **test}
        print(
            f"bootstrap {a.name} - {b.name}: Δmacro-F1 {test['delta_macro_f1']:.2f} "
            f"95% CI [{test['ci_low']:.2f}, {test['ci_high']:.2f}] p={test['p_value']:.3f}"
        )

    out = Path(args.output) if args.output else Path(families[0]) / f"aggregate_{args.split}.json"
    save_json(payload, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
