"""Zero-shot Qwen2.5-7B-Instruct baseline.

The prompt lists each candidate label with the first sentence of its description
and the document truncated to 8,000 tokens. Decoding uses temperature 0.
Labels outside the candidate list are discarded.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request

from tqdm import tqdm
from transformers import AutoTokenizer

from lkhit.config import load_config
from lkhit.runtime import load_splits

TASK_TEXT = {
    "ecthr_a": "List the Convention articles that the Court found to have been violated.",
    "ecthr_b": "List the Convention articles that the applicant alleged to have been violated.",
    "eurlex": "List the EuroVoc concepts that apply to this EU legislative act.",
    "cail2018": "Give the single applicable article of the Criminal Law of the People's Republic of China.",
}


def first_sentence(text: str) -> str:
    piece = text.strip().split(". ")[0].strip()
    return piece[:-1] if piece.endswith(".") else piece


def build_prompt(task: str, names: list[str], texts: list[str], document: str, tokenizer, max_doc_tokens: int) -> str:
    ids = tokenizer.encode(document, add_special_tokens=False, truncation=True, max_length=max_doc_tokens)
    clipped = tokenizer.decode(ids)
    lines = [TASK_TEXT[task], "", "Candidate labels:"]
    for name, description in zip(names, texts):
        lines.append(f"- {name}: {first_sentence(description)}")
    lines.extend(["", "Document:", clipped, "", "Return a JSON array of labels."])
    return "\n".join(lines)


def parse_labels(response: str, names: list[str]) -> list[int]:
    match = re.search(r"\[.*\]", response, flags=re.DOTALL)
    if match is None:
        return []
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    allowed = {str(name): index for index, name in enumerate(names)}
    chosen = []
    for item in payload:
        index = allowed.get(str(item).strip())
        if index is not None and index not in chosen:
            chosen.append(index)
    return chosen


def complete(base_url: str, model: str, prompt: str) -> str:
    body = json.dumps(
        {"model": model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}
    ).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"]


def document_text(record: dict) -> str:
    if record.get("text"):
        return record["text"]
    return "\n".join(record["paragraphs"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot label prediction with a vLLM OpenAI-compatible server.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--max-doc-tokens", type=int, default=8000)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    splits, label_spec = load_splits(cfg)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    predictions = []
    for record in tqdm(splits[args.split]):
        prompt = build_prompt(cfg["task"], label_spec["names"], label_spec["texts"], document_text(record), tokenizer, args.max_doc_tokens)
        response = complete(args.base_url, args.model, prompt)
        predictions.append(parse_labels(response, label_spec["names"]))
    print(json.dumps({"n": len(predictions), "labels": predictions}, ensure_ascii=False))


if __name__ == "__main__":
    main()
