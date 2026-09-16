"""Prepare CAIL2018-small under the LADAN protocol.

    python -m lkhit.cli.prepare_cail --raw-dir data/raw/cail2018 --out-dir data/processed/cail2018

Keeps cases with a single applicable law article, drops articles with fewer
than 100 training cases (103 articles remain), carves a stratified 10 %
development split from the training cases for early stopping and calibration,
and writes ``train/dev/test.jsonl``, ``label_list.json`` and ``stats.json``.
The Criminal Law article texts (``criminal_law_articles.json``) are copied
next to the splits when present so that the knowledge branch can find them.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from lkhit.data.cail import prepare_cail
from lkhit.utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw/cail2018", help="directory with data_train.json and data_test.json (exercise_contest)")
    parser.add_argument("--out-dir", default="data/processed/cail2018")
    parser.add_argument("--min-train-cases", type=int, default=100)
    parser.add_argument("--dev-fraction", type=float, default=0.10)
    parser.add_argument("--split-seed", type=int, default=13, help="seed of the stratified development hold-out (fixed across model seeds)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging(None, name="lkhit.prepare_cail")
    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)
    for name in ("data_train.json", "data_test.json"):
        if not (raw_dir / name).exists():
            raise SystemExit(f"{raw_dir / name} not found; unpack CAIL2018_ALL_DATA.zip and copy exercise_contest/{name} there")
    stats = prepare_cail(raw_dir, out_dir, min_train_cases=args.min_train_cases, dev_fraction=args.dev_fraction, seed=args.split_seed)
    articles = raw_dir / "criminal_law_articles.json"
    if articles.exists():
        shutil.copyfile(articles, out_dir / "criminal_law_articles.json")
        logger.info("copied Criminal Law article texts to %s", out_dir)
    else:
        logger.warning("%s not found; the knowledge branch will fall back to article numbers only", articles)
    logger.info(
        "%d articles | train %d | dev %d | test %d | mean fact length %.0f characters (median %.0f)",
        stats["n_articles"],
        stats["n_train"],
        stats["n_dev"],
        stats["n_test"],
        stats["fact_chars_mean"],
        stats["fact_chars_median"],
    )


if __name__ == "__main__":
    main()
