"""Write the statute-derived label descriptions of a task to ``data/processed/<task>/label_descriptions.json``.

    python -m lkhit.cli.build_label_descriptions --task ecthr_a
    python -m lkhit.cli.build_label_descriptions --task eurlex --eurovoc-skos data/raw/eurovoc
    python -m lkhit.cli.build_label_descriptions --task cail2018

ECtHR: full English text of each Convention article (resources/echr_articles.json)
plus the authors' one-sentence description of the ``none`` label. EUR-LEX:
EuroVoc preferred term, scope note and broader-term path from the SKOS export.
CAIL2018: article text from the Criminal Law JSON shipped with the release.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lkhit.config import load_config
from lkhit.data.descriptions import build_label_descriptions, description_lengths, first_sentence, save_label_descriptions
from lkhit.data.loading import load_task
from lkhit.engine.knowledge import descriptions_path
from lkhit.utils import setup_logging

DEFAULT_CONFIGS = {
    "ecthr_a": "configs/experiments/ecthr_a/lkhit.yaml",
    "ecthr_b": "configs/experiments/ecthr_b/lkhit.yaml",
    "eurlex": "configs/experiments/eurlex/lkhit.yaml",
    "cail2018": "configs/experiments/cail2018/lkhit.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=sorted(DEFAULT_CONFIGS), help="task name (uses the LK-HiT experiment config of the task)")
    parser.add_argument("--config", default=None, help="explicit experiment config instead of --task")
    parser.add_argument("--eurovoc-skos", default=None, help="EuroVoc SKOS export (file or directory) for EUR-LEX")
    parser.add_argument("--criminal-law-articles", default=None, help="JSON {article: text} for CAIL2018")
    parser.add_argument("--output", default=None, help="destination JSON (default: data/processed/<task>/label_descriptions.json)")
    parser.add_argument("--show", type=int, default=3, help="print the first N descriptions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.config and not args.task:
        raise SystemExit("give --task or --config")
    overrides = []
    if args.eurovoc_skos:
        overrides.append(f"data.knowledge.eurovoc_skos={args.eurovoc_skos}")
    if args.criminal_law_articles:
        overrides.append(f"data.knowledge.criminal_law_articles={args.criminal_law_articles}")
    cfg = load_config(args.config or DEFAULT_CONFIGS[args.task], overrides)
    logger = setup_logging(None, name="lkhit.descriptions")
    spec = load_task(cfg)
    descriptions, groups = build_label_descriptions(cfg, spec)
    out = Path(args.output) if args.output else descriptions_path(cfg)
    save_label_descriptions(descriptions, groups, out)
    stats = description_lengths(descriptions, spec.language)
    logger.info("%d descriptions for %s -> %s", len(descriptions), spec.name, out)
    logger.info("mean length %.1f %s (min %d, max %d)", stats["mean"], stats["unit"], stats["min"], stats["max"])
    if groups:
        logger.info("%d labels with structural groups (%d distinct)", len(groups), len(set(groups.values())))
    for name in spec.label_names[: args.show]:
        logger.info("  [%s] %s", name, first_sentence(descriptions[name]))


if __name__ == "__main__":
    main()
