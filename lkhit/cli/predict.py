"""Score new cases with a trained run: calibrated probabilities, decisions, uncertainty and paragraph rationales.

    python -m lkhit.cli.predict --run runs/ecthr_a/lkhit/seed1 --input my_cases.jsonl --output scored.jsonl

Input rows are JSON objects with an ``id`` and either ``paragraphs`` (list of
strings, ECtHR-style) or ``text`` (a single string). Output rows add
``probabilities`` (after temperature scaling), ``labels`` (decisions at 0.5 /
arg-max), ``uncertainty`` (Eq. 5 of the paper), ``abstain`` when an
uncertainty threshold is given, and the top-k paragraph rationale.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from lkhit.calibration import logits_to_probs
from lkhit.cli.common import add_run_arguments, begin, load_run, load_run_spec
from lkhit.data.dataset import Document
from lkhit.data.loading import build_collate, build_dataloader, build_dataset, build_word_vocab, input_format, load_tokenizer
from lkhit.engine.checkpoint import checkpoint_dir, load_weights
from lkhit.engine.evaluator import Evaluator
from lkhit.engine.knowledge import load_run_knowledge
from lkhit.engine.trainer import amp_dtype_from_config
from lkhit.models import build_model, needs_label_graph
from lkhit.rationales import extract_rationale
from lkhit.selective import uncertainty_score
from lkhit.utils import load_json, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_arguments(parser)
    parser.add_argument("--input", required=True, help="JSONL with id + paragraphs | text")
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=5, help="number of rationale paragraphs")
    parser.add_argument("--abstain-above", type=float, default=None, help="uncertainty threshold tau_u for abstention")
    return parser.parse_args()


def rows_to_documents(rows: list[dict], spec) -> list[Document]:
    documents = []
    placeholder = [] if spec.multi_label else [0]
    for i, row in enumerate(rows):
        doc_id = str(row.get("id", i))
        if "paragraphs" in row:
            documents.append(Document(doc_id=doc_id, labels=placeholder, segments=[str(p) for p in row["paragraphs"]]))
        elif "text" in row:
            documents.append(Document(doc_id=doc_id, labels=placeholder, text=str(row["text"])))
        else:
            raise ValueError(f"row {doc_id} has neither 'paragraphs' nor 'text'")
    return documents


def main() -> None:
    args = parse_args()
    cfg, run_dir = load_run(args.run)
    logger, device = begin(cfg, run_dir, int(cfg.get("train", {}).get("seed", 1)), args.device, log_name="lkhit.predict")
    spec = load_run_spec(run_dir)
    rows = read_jsonl(Path(args.input))
    documents = rows_to_documents(rows, spec)
    logger.info("%d documents to score", len(documents))

    fmt = input_format(cfg["model"])
    tokenizer = load_tokenizer(cfg) if fmt in ("hierarchical", "flat") else None
    vocab = build_word_vocab(cfg, [], run_dir) if fmt == "words" else None
    descriptions, adjacency = load_run_knowledge(run_dir, spec, with_graph=needs_label_graph(cfg["model"]))
    model = build_model(cfg, spec, tokenizer=tokenizer, label_descriptions=descriptions, adjacency=adjacency, vocab=vocab).to(device)
    load_weights(model, checkpoint_dir(run_dir) / "best.pt", device)

    temperature_file = run_dir / "temperature.json"
    temperature = float(load_json(temperature_file)["temperature"]) if temperature_file.exists() else 1.0

    cfg["data"]["cache_dir"] = None  # never read or write the tokenised training caches for ad-hoc input
    cfg["data"]["precompute"] = False
    dataset = build_dataset(cfg, spec, documents, "predict", tokenizer=tokenizer, vocab=vocab)
    loader = build_dataloader(dataset, cfg, build_collate(cfg, tokenizer=tokenizer, vocab=vocab), shuffle=False, batch_size=int(args.batch_size or cfg["train"].get("eval_batch_size", 8)))
    evaluator = Evaluator(model, spec, device, amp_dtype=amp_dtype_from_config(cfg["train"], device))
    preds = evaluator.predict(loader, collect_attention=True, log_every=0)

    probs = logits_to_probs(preds.logits, spec.multi_label, temperature)
    uncertainty = uncertainty_score(probs, spec.multi_label)
    out_rows = []
    for i, doc in enumerate(documents):
        p = probs[i]
        if spec.multi_label:
            chosen = [spec.label_names[j] for j in np.where(p >= 0.5)[0]]
        else:
            chosen = [spec.label_names[int(p.argmax())]]
        row = {
            "id": doc.doc_id,
            "probabilities": {name: float(v) for name, v in zip(spec.label_names, p)},
            "labels": chosen,
            "uncertainty": float(uncertainty[i]),
        }
        if args.abstain_above is not None:
            row["abstain"] = bool(uncertainty[i] > args.abstain_above)
        if preds.label_attention is not None and fmt == "hierarchical":
            n_seg = int(preds.n_segments[i]) if preds.n_segments is not None else preds.label_attention[i].shape[-1]
            row["rationale"] = extract_rationale(p, preds.label_attention[i], spec, n_seg, k=args.top_k)
        out_rows.append(row)
    write_jsonl(Path(args.output), out_rows)
    logger.info("wrote %s (temperature %.3f)", args.output, temperature)


if __name__ == "__main__":
    main()
