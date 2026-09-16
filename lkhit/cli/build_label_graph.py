"""Build the label graph of a task from public statutory sources and the training split.

    python -m lkhit.cli.build_label_graph --config configs/experiments/ecthr_a/lkhit.yaml

Co-occurrence edges carry the normalised positive PMI of label pairs in the
training split; structural edges (weight 1) link labels from the same
Convention section, EuroVoc micro-thesaurus or Criminal Law chapter. The
summed adjacency is symmetrically normalised with self-loops and written to
``data/processed/<task>/label_graph.npz`` together with the label
descriptions the knowledge branch encodes.
"""

from __future__ import annotations

import argparse

from lkhit.cli.common import add_config_arguments, begin, resolve_config
from lkhit.data.dataset import label_sets
from lkhit.data.label_graph import build_label_graph, graph_statistics, save_label_graph, structural_groups
from lkhit.data.loading import load_documents, load_task
from lkhit.engine.knowledge import ensure_label_descriptions, graph_path
from lkhit.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arguments(parser)
    parser.add_argument("--no-cooccurrence", action="store_true", help="structural edges only")
    parser.add_argument("--no-structure", action="store_true", help="co-occurrence edges only")
    parser.add_argument("--rebuild-descriptions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg, run_dir = resolve_config(args)
    logger, _ = begin(cfg, run_dir.parent / "_label_graph", args.seed, "cpu", log_name="lkhit.label_graph")
    spec = load_task(cfg)
    descriptions, groups_from_descriptions = ensure_label_descriptions(cfg, spec, rebuild=args.rebuild_descriptions)
    train = load_documents(cfg, "train")
    groups = structural_groups(spec, cfg, groups_from_descriptions)
    graph = build_label_graph(
        spec,
        label_sets(train, spec),
        groups,
        use_cooccurrence=not args.no_cooccurrence,
        use_structure=not args.no_structure,
    )
    path = graph_path(cfg)
    save_label_graph(graph, spec, groups, path)
    stats = graph_statistics(graph)
    save_json({**stats, "groups": groups, "label_names": spec.label_names}, path.with_suffix(".stats.json"))
    logger.info("label graph for %s written to %s", spec.name, path)
    for key, value in stats.items():
        logger.info("  %-26s %s", key, value)
    strongest = sorted(
        ((graph["cooccurrence"][i, j], spec.label_names[i], spec.label_names[j]) for i in range(spec.n_labels) for j in range(i + 1, spec.n_labels) if graph["cooccurrence"][i, j] > 0),
        reverse=True,
    )[:10]
    for weight, a, b in strongest:
        logger.info("  co-occurrence %-12s -- %-12s npmi %.3f", a, b, weight)


if __name__ == "__main__":
    main()
