"""Build paragraph-segmented documents and statute label descriptions.

ECtHR-A, ECtHR-B, and EUR-LEX are read from the LexGLUE release. CAIL2018 is
read from local CAIL-small JSON and filtered with the LADAN protocol. Raw
corpora are not stored in this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from lkhit.data import split_sentences, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
ECTHR_STATUTES = ROOT / "data" / "statutes" / "ecthr.json"
LEXGLUE = "coastalcph/lex_glue"

NONE_TEXT = {
    "ecthr_a": "The Court finds that none of the Convention articles under examination has been violated",
    "ecthr_b": "The applicant does not allege a violation of any of the Convention articles under examination",
}


def criminal_law_chapter(article: int) -> str:
    """Chapter of the Special Provisions of the Criminal Law of the PRC."""
    spans = (
        (102, 113, "ch1"),
        (114, 139, "ch2"),
        (140, 231, "ch3"),
        (232, 262, "ch4"),
        (263, 276, "ch5"),
        (277, 367, "ch6"),
        (368, 381, "ch7"),
        (382, 396, "ch8"),
        (397, 419, "ch9"),
        (420, 451, "ch10"),
    )
    for start, end, name in spans:
        if start <= article <= end:
            return name
    return f"art-{article}"


def _canon_ecthr(name: str) -> str:
    text = str(name).strip().lower().replace("article", "").strip()
    aliases = {
        "p1-1": "P1-1",
        "p1": "P1-1",
        "1 of protocol 1": "P1-1",
        "1 of protocol no. 1": "P1-1",
        "protocol 1 article 1": "P1-1",
    }
    return aliases.get(text, text)


def _fact_key(paragraphs: list[str]) -> str:
    payload = "\n".join(paragraph.strip() for paragraph in paragraphs)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _load_lexglue(config: str):
    from datasets import load_dataset

    return load_dataset(LEXGLUE, config)


def _label_names(features) -> list[str]:
    label_feature = features["labels"] if "labels" in features else features["label"]
    feature = getattr(label_feature, "feature", label_feature)
    names = getattr(feature, "names", None)
    if not names:
        raise RuntimeError("LexGLUE label names are missing from the dataset features.")
    return list(names)


def _row_labels(row: dict) -> list[int]:
    raw = row.get("labels", row.get("label"))
    if raw is None:
        return []
    if isinstance(raw, int):
        return [raw]
    return [int(label) for label in raw]


def _row_text(row: dict):
    return row.get("text", row.get("facts"))


def _year_from(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value if 1900 < value < 2100 else None
    text = str(value)
    for token in text.replace("/", "-").split("-"):
        if len(token) == 4 and token.isdigit():
            return int(token)
    return None


def _load_ecthr_side_info() -> dict[str, dict]:
    """Map a fact-list hash to silver rationales and decision year, when the release is available."""
    from datasets import load_dataset

    side = {}
    dataset = None
    for name in ("ecthr_cases", "coastalcph/ecthr_cases"):
        try:
            dataset = load_dataset(name)
            break
        except Exception:
            dataset = None
    if dataset is None:
        print("ecthr_cases is not available; year and silver rationales will be empty.", file=sys.stderr)
        return side
    splits = dataset.values() if hasattr(dataset, "values") else [dataset]
    for split in splits:
        for row in split:
            paragraphs = row.get("facts") or row.get("text") or []
            if isinstance(paragraphs, str):
                paragraphs = [paragraphs]
            rationales = None
            for key in ("silver_rationales", "silver_rationale", "rationales"):
                if key in row and row[key] is not None:
                    rationales = row[key]
                    break
            indices: list[int] = []
            if rationales:
                first = rationales[0]
                if isinstance(first, (int, np.integer)):
                    indices = [int(item) for item in rationales]
                elif isinstance(first, (float, np.floating, bool, np.bool_)):
                    indices = [i for i, flag in enumerate(rationales) if float(flag) > 0]
                elif isinstance(first, (list, tuple)):
                    indices = [int(item) for item in first]
            year = None
            for key in ("judgment_date", "decision_date", "date", "year", "judgment_year"):
                if key in row:
                    year = _year_from(row[key])
                    if year is not None:
                        break
            side[_fact_key(list(paragraphs))] = {"rationales": indices, "year": year}
    return side


def _ecthr_records(config: str, statutes: dict) -> tuple[dict[str, list[dict]], dict]:
    dataset = _load_lexglue(config)
    names = _label_names(dataset["train"].features)
    by_id = {_canon_ecthr(item["id"]): item for item in statutes["articles"]}
    label_spec = []
    for name in names:
        article = by_id[_canon_ecthr(name)]
        label_spec.append({"name": name, "text": article["text"], "group": article["group"]})
    label_spec.append({"name": "no violation" if config == "ecthr_a" else "no allegation", "text": NONE_TEXT[config], "group": f"none-{config}"})
    side = _load_ecthr_side_info()
    none_index = len(label_spec) - 1
    splits = {}
    for split_name, output_name in (("train", "train"), ("validation", "dev"), ("test", "test")):
        records = []
        for row in dataset[split_name]:
            paragraphs = list(_row_text(row))
            gold = _row_labels(row)
            labels = [none_index] if not gold else gold
            info = side.get(_fact_key(paragraphs), {})
            records.append(
                {
                    "paragraphs": paragraphs,
                    "labels": labels,
                    "segment": "paragraph",
                    "year": info.get("year"),
                    "rationales": info.get("rationales") or [],
                    "n_chars": sum(len(paragraph) for paragraph in paragraphs),
                }
            )
        splits[output_name] = records
    spec = {
        "names": [item["name"] for item in label_spec],
        "texts": [item["text"] for item in label_spec],
        "groups": [item["group"] for item in label_spec],
        "single_label": False,
        "none_index": none_index,
    }
    return splits, spec


def _sparql(query: str) -> list[dict]:
    url = "https://publications.europa.eu/webapi/rdf/sparql?query=" + urllib.parse.quote(query)
    request = urllib.request.Request(url, headers={"Accept": "application/sparql-results+json", "User-Agent": "LK-HiT"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload.get("results", {}).get("bindings", [])


def _eurovoc_query(eurovoc_id: str) -> dict:
    """Preferred term, scope note, broader-term chain, and micro-thesaurus."""
    concept = f"http://eurovoc.europa.eu/{eurovoc_id}"
    rows = _sparql(
        f"""
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?pref ?scope ?scheme WHERE {{
  <{concept}> skos:prefLabel ?pref .
  FILTER(lang(?pref) = "en")
  OPTIONAL {{ <{concept}> skos:scopeNote ?scope . FILTER(lang(?scope) = "en") }}
  OPTIONAL {{
    <{concept}> skos:inScheme ?s .
    ?s skos:prefLabel ?scheme .
    FILTER(lang(?scheme) = "en")
  }}
}}
"""
    )
    pref, scope, schemes = "", "", []
    for binding in rows:
        pref = pref or binding.get("pref", {}).get("value", "")
        scope = scope or binding.get("scope", {}).get("value", "")
        scheme = binding.get("scheme", {}).get("value", "")
        if scheme and scheme not in schemes:
            schemes.append(scheme)
    if not pref:
        raise RuntimeError(f"EuroVoc concept {eurovoc_id} returned no English preferred term.")
    micro = next((name for name in schemes if name.lower() != "eurovoc"), schemes[0] if schemes else eurovoc_id)
    broader = []
    current = concept
    seen = {current}
    for _ in range(12):
        parents = _sparql(
            f"""
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?parent ?label WHERE {{
  <{current}> skos:broader ?parent .
  ?parent skos:prefLabel ?label .
  FILTER(lang(?label) = "en")
}} LIMIT 1
"""
        )
        if not parents:
            break
        parent = parents[0]["parent"]["value"]
        label = parents[0]["label"]["value"]
        if parent in seen:
            break
        seen.add(parent)
        broader.append(label)
        current = parent
    return {"pref": pref, "scope": scope, "broader": broader, "microthesaurus": micro}


def download_eurovoc(ids: list[str], destination: Path) -> dict:
    concepts = {}
    for eurovoc_id in ids:
        concepts[str(eurovoc_id)] = _eurovoc_query(str(eurovoc_id))
        print(f"EuroVoc {eurovoc_id}: {concepts[str(eurovoc_id)]['pref']}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(concepts, handle, ensure_ascii=False, indent=2)
    return concepts


def _eurovoc_text(entry: dict) -> str:
    parts = [entry["pref"]]
    if entry.get("scope"):
        parts.append(entry["scope"])
    broader = entry.get("broader") or []
    if broader:
        parts.append("Broader terms: " + "; ".join(broader))
    return ". ".join(parts)


def _eurlex_records(eurovoc_path: Path) -> tuple[dict[str, list[dict]], dict]:
    dataset = _load_lexglue("eurlex")
    names = _label_names(dataset["train"].features)
    with open(eurovoc_path, encoding="utf-8") as handle:
        eurovoc = json.load(handle)
    missing = [name for name in names if str(name) not in eurovoc]
    if missing:
        raise SystemExit(f"EuroVoc file is missing {len(missing)} LexGLUE concepts, for example {missing[:5]}.")
    spec = {
        "names": names,
        "texts": [_eurovoc_text(eurovoc[str(name)]) for name in names],
        "groups": [eurovoc[str(name)].get("microthesaurus") or f"id-{name}" for name in names],
        "single_label": False,
        "none_index": None,
    }
    splits = {}
    for split_name, output_name in (("train", "train"), ("validation", "dev"), ("test", "test")):
        records = []
        for row in dataset[split_name]:
            text = _row_text(row)
            records.append(
                {
                    "paragraphs": [],
                    "text": text,
                    "labels": _row_labels(row),
                    "segment": "token_chunk",
                    "year": None,
                    "rationales": [],
                    "n_chars": len(text),
                }
            )
        splits[output_name] = records
    return splits, spec


def _read_cail(path: Path) -> list[dict]:
    raw = path.read_bytes()
    text = None
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise SystemExit(f"Could not decode {path}")
    text = text.strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _cail_fact(row: dict) -> str:
    return row.get("fact") or row.get("content") or ""


def _cail_articles(row: dict) -> list[int]:
    meta = row.get("meta") or {}
    raw = meta.get("relevant_articles", row.get("relevant_articles", []))
    return [int(article) for article in raw]


def _load_article_texts(path: Path) -> dict[int, str]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    texts = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, dict):
                texts[int(value.get("id", key))] = value.get("text") or value.get("content") or ""
            else:
                texts[int(key)] = str(value)
    else:
        for item in payload:
            texts[int(item["id"])] = item.get("text") or item.get("content") or ""
    return texts


def _cail_records(train_path: Path, test_path: Path, articles_path: Path, dev_size: int, split_seed: int):
    train_rows = [row for row in _read_cail(train_path) if len(_cail_articles(row)) == 1]
    test_rows = [row for row in _read_cail(test_path) if len(_cail_articles(row)) == 1]
    counts = Counter(_cail_articles(row)[0] for row in train_rows)
    kept = {article for article, count in counts.items() if count >= 100}
    train_rows = [row for row in train_rows if _cail_articles(row)[0] in kept]
    test_rows = [row for row in test_rows if _cail_articles(row)[0] in kept]
    article_ids = sorted(kept)
    index = {article: i for i, article in enumerate(article_ids)}
    texts = _load_article_texts(articles_path)
    missing = [article for article in article_ids if article not in texts]
    if missing:
        raise SystemExit(f"Article text file is missing {len(missing)} articles, for example {missing[:5]}.")
    spec = {
        "names": [str(article) for article in article_ids],
        "texts": [texts[article] for article in article_ids],
        "groups": [criminal_law_chapter(article) for article in article_ids],
        "article_ids": article_ids,
        "single_label": True,
        "none_index": None,
    }

    def convert(row: dict) -> dict:
        fact = _cail_fact(row)
        return {
            "paragraphs": split_sentences(fact),
            "text": fact,
            "labels": [index[_cail_articles(row)[0]]],
            "segment": "sentence",
            "year": None,
            "rationales": [],
            "n_chars": len(fact),
        }

    labels = [index[_cail_articles(row)[0]] for row in train_rows]
    try:
        train_part, dev_part = train_test_split(
            train_rows,
            test_size=dev_size,
            train_size=len(train_rows) - dev_size,
            stratify=labels,
            random_state=split_seed,
        )
    except ValueError as exc:
        raise SystemExit(f"Stratified development split failed: {exc}") from exc
    return {"train": [convert(row) for row in train_part], "dev": [convert(row) for row in dev_part], "test": [convert(row) for row in test_rows]}, spec


def _write_task(task: str, splits: dict[str, list[dict]], spec: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name, records in splits.items():
        write_jsonl(output / f"{name}.jsonl", records)
        print(f"{task} {name}: {len(records)} documents")
    with open(output / "labels.json", "w", encoding="utf-8") as handle:
        json.dump(spec, handle, ensure_ascii=False, indent=2)
    print(f"labels: {len(spec['names'])} -> {output / 'labels.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess ECtHR, EUR-LEX, or CAIL2018 for LK-HiT.")
    parser.add_argument("--task", required=True, choices=["ecthr_a", "ecthr_b", "eurlex", "cail2018"])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--eurovoc", type=Path, default=ROOT / "data" / "statutes" / "eurovoc.json")
    parser.add_argument("--download-eurovoc", action="store_true")
    parser.add_argument("--cail-train", type=Path, default=ROOT / "data" / "raw" / "cail2018" / "data_train.json")
    parser.add_argument("--cail-test", type=Path, default=ROOT / "data" / "raw" / "cail2018" / "data_test.json")
    parser.add_argument("--cail-articles", type=Path, default=ROOT / "data" / "raw" / "cail2018" / "articles.json")
    parser.add_argument("--dev-size", type=int, default=10162)
    parser.add_argument("--split-seed", type=int, default=1)
    args = parser.parse_args()
    output = args.output or (ROOT / "data" / "processed" / args.task)

    if args.task in {"ecthr_a", "ecthr_b"}:
        with open(ECTHR_STATUTES, encoding="utf-8") as handle:
            statutes = json.load(handle)
        splits, spec = _ecthr_records(args.task, statutes)
    elif args.task == "eurlex":
        if args.download_eurovoc:
            dataset = _load_lexglue("eurlex")
            download_eurovoc(_label_names(dataset["train"].features), args.eurovoc)
        if not args.eurovoc.exists():
            raise SystemExit(f"Missing {args.eurovoc}. Build it with --download-eurovoc or supply pref/scope/broader/microthesaurus entries.")
        splits, spec = _eurlex_records(args.eurovoc)
    else:
        splits, spec = _cail_records(args.cail_train, args.cail_test, args.cail_articles, args.dev_size, args.split_seed)
    _write_task(args.task, splits, spec, output)


if __name__ == "__main__":
    main()
