"""Zero-shot LLM baseline (Qwen2.5-7B-Instruct) served with vLLM.

    python -m lkhit.cli.zero_shot_llm --task ecthr_a --model Qwen/Qwen2.5-7B-Instruct --split test

The prompt states the task, lists every candidate label with the first
sentence of its description and gives the document truncated to 8,000 tokens;
the model answers with a JSON array of labels. Labels outside the candidate
list are discarded. Decoding is greedy (temperature 0). Predictions and
metrics are written in the same layout as the fine-tuned runs so that the
same aggregation code applies; there is no probability output, so calibration
metrics are not meaningful for this baseline.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from lkhit.config import load_config, run_directory, save_config
from lkhit.data.dataset import label_counts
from lkhit.data.descriptions import RESOURCES_DIR, first_sentence
from lkhit.data.loading import load_task
from lkhit.engine.evaluator import Predictions, metric_report, save_predictions, save_report
from lkhit.engine.knowledge import ensure_label_descriptions
from lkhit.engine.setup import load_split_documents
from lkhit.utils import save_json, setup_logging, write_jsonl

DEFAULT_CONFIGS = {task: f"configs/experiments/{task}/qwen_zero_shot.yaml" for task in ("ecthr_a", "ecthr_b", "eurlex", "cail2018")}

TASK_DESCRIPTIONS = {
    "ecthr_a": "Given the facts of a case before the European Court of Human Rights, list the articles of the Convention that the Court found to have been VIOLATED.",
    "ecthr_b": "Given the facts of a case before the European Court of Human Rights, list the articles of the Convention that the applicant ALLEGED to have been violated.",
    "eurlex": "Given the text of a European Union legislative act, list the EuroVoc concepts (subject descriptors) that apply to it.",
    "cail2018": "根据下面的刑事案件事实描述，判断被告人所触犯的《中华人民共和国刑法》条文（只有一条适用条文）。",
}

_JSON_ARRAY = re.compile(r"\[[^\[\]]*\]", re.S)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=sorted(DEFAULT_CONFIGS))
    parser.add_argument("--config", default=None, help="experiment config (default: configs/experiments/<task>/qwen_zero_shot.yaml)")
    parser.add_argument("--model", default=None, help="Hugging Face id of the instruction-tuned model (overrides model.name)")
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--max-doc-tokens", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="score only the first N documents (debugging)")
    parser.add_argument("--dry-run", action="store_true", help="write the prompts without loading the model")
    return parser.parse_args()


def load_prompt_template(language: str, resources_dir: Path) -> str:
    name = "zero_shot_zh.txt" if language == "zh" else "zero_shot_en.txt"
    return (resources_dir / "prompts" / name).read_text(encoding="utf-8")


def build_prompt(template: str, task: str, spec, descriptions: list[str], document: str) -> str:
    candidates = "\n".join(f"- {name}: {first_sentence(desc)}" for name, desc in zip(spec.label_names, descriptions))
    return template.format(
        task_description=TASK_DESCRIPTIONS[task],
        candidates=candidates,
        document=document,
        max_labels="one label" if not spec.multi_label else "all applicable labels",
    )


def truncate_to_tokens(tokenizer, text: str, max_tokens: int) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens])


def parse_labels(answer: str, spec) -> list[int]:
    names = {n.lower(): i for i, n in enumerate(spec.label_names)}
    aliases = {}
    for name, idx in names.items():
        aliases[name] = idx
        aliases[f"article {name}"] = idx
        aliases[f"art. {name}"] = idx
        aliases[f"第{name}条"] = idx
    match = _JSON_ARRAY.search(answer)
    chosen: list[int] = []
    if match:
        try:
            items = json.loads(match.group(0))
        except json.JSONDecodeError:
            items = re.findall(r'"([^"]+)"', match.group(0))
        for item in items:
            key = str(item).strip().lower()
            key = re.sub(r"^protocol\s*1\s*article\s*1$", "p1-1", key)
            if key in aliases:
                if aliases[key] not in chosen:
                    chosen.append(aliases[key])
            elif key.replace("article", "").strip() in aliases and aliases[key.replace("article", "").strip()] not in chosen:
                chosen.append(aliases[key.replace("article", "").strip()])
    if not spec.multi_label:
        chosen = chosen[:1]
    return chosen


def main() -> None:
    args = parse_args()
    if not args.config and not args.task:
        raise SystemExit("give --task or --config")
    overrides = []
    if args.model:
        overrides.append(f"model.name={args.model}")
    cfg = load_config(args.config or DEFAULT_CONFIGS[args.task], overrides)
    task = cfg["data"]["task"]
    run_dir = run_directory(cfg, 1)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(run_dir, name="lkhit.zero_shot")
    save_config(cfg, run_dir / "config.yaml")

    spec = load_task(cfg)
    save_json(spec.to_dict(), run_dir / "task_spec.json")
    descriptions, _ = ensure_label_descriptions(cfg, spec)
    documents = load_split_documents(cfg, ("train", args.split))
    train_counts = label_counts(documents["train"], spec)
    docs = documents[args.split][: args.limit] if args.limit else documents[args.split]

    model_cfg = cfg["model"]
    model_name = model_cfg["name"]
    max_doc_tokens = int(args.max_doc_tokens or model_cfg.get("max_doc_tokens", 8000))
    max_new_tokens = int(args.max_new_tokens or model_cfg.get("max_new_tokens", 128))
    resources_dir = Path((cfg["data"].get("knowledge") or {}).get("resources_dir", RESOURCES_DIR))
    template = load_prompt_template(spec.language, resources_dir)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompts = []
    for doc in docs:
        body = truncate_to_tokens(tokenizer, doc.full_text("\n"), max_doc_tokens)
        user = build_prompt(template, task, spec, descriptions, body)
        messages = [{"role": "system", "content": model_cfg.get("system_prompt", "You are a careful legal assistant. Answer only with the requested JSON.")}, {"role": "user", "content": user}]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    write_jsonl(run_dir / f"prompts_{args.split}.jsonl", ({"id": d.doc_id, "prompt": p} for d, p in zip(docs, prompts)))
    logger.info("%d prompts written (%s, <= %d document tokens)", len(prompts), model_name, max_doc_tokens)
    if args.dry_run:
        return

    from vllm import LLM, SamplingParams

    llm = LLM(model=model_name, tensor_parallel_size=args.tensor_parallel, max_model_len=int(model_cfg.get("max_model_len", 12288)), dtype=model_cfg.get("dtype", "bfloat16"))
    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    outputs = llm.generate(prompts, sampling)
    answers = [o.outputs[0].text for o in outputs]

    n = len(docs)
    logits = np.full((n, spec.n_labels), -4.0, dtype=np.float32)  # ~0.018 after the logistic
    labels = np.zeros((n, spec.n_labels), dtype=np.int64) if spec.multi_label else np.zeros(n, dtype=np.int64)
    n_unparsed = 0
    rows = []
    for i, (doc, answer) in enumerate(zip(docs, answers)):
        chosen = parse_labels(answer, spec)
        if not chosen:
            n_unparsed += 1
            if spec.multi_label and spec.none_index is not None:
                chosen = [spec.none_index]
            elif not spec.multi_label:
                chosen = [int(np.argmax(train_counts))]
        logits[i, chosen] = 4.0
        if spec.multi_label:
            gold = list(doc.labels) or ([spec.none_index] if spec.none_index is not None else [])
            labels[i, gold] = 1
        else:
            labels[i] = doc.labels[0]
        rows.append({"id": doc.doc_id, "answer": answer, "labels": [spec.label_names[c] for c in chosen]})
    write_jsonl(run_dir / f"answers_{args.split}.jsonl", rows)
    logger.info("%d/%d answers without a parsable label", n_unparsed, n)

    preds = Predictions(logits=logits, labels=labels, index=np.arange(n))
    report = metric_report(preds, spec, train_counts=train_counts, documents=docs, subgroup_cfg=cfg["data"].get("subgroups"))
    report["n_unparsed_answers"] = n_unparsed
    report["calibration_note"] = "hard decisions only; probability-based metrics are not meaningful for this baseline"
    save_predictions(preds, spec, run_dir / "predictions" / f"{args.split}.npz")
    save_report(report, run_dir / "metrics" / f"{args.split}.json")


if __name__ == "__main__":
    main()
