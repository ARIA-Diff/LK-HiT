"""Statute-derived label descriptions: Convention articles, EuroVoc concepts, Criminal Law articles."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from lkhit.data.tasks import TaskSpec
from lkhit.utils import load_json, save_json

logger = logging.getLogger("lkhit.data.descriptions")

RESOURCES_DIR = Path(__file__).resolve().parents[2] / "resources"

ECTHR_NONE_DESCRIPTIONS = {
    "ecthr_a": "The Court finds that none of the Convention articles under examination has been violated.",
    "ecthr_b": "The applicant does not allege a violation of any of the Convention articles under examination.",
}

EUROVOC_NS = "http://eurovoc.europa.eu/"
EUROVOC_THESAURUS_SCHEME = EUROVOC_NS + "100141"

_SENTENCE_END = re.compile(r"(?<=[.!?。！？])\s+")


def first_sentence(text: str) -> str:
    parts = _SENTENCE_END.split(text.strip(), maxsplit=1)
    return parts[0].strip() if parts else text.strip()


def echr_descriptions(spec: TaskSpec, resources_dir: Path | None = None) -> dict[str, str]:
    """Full English text of each Convention article plus the authors' sentence for the ``none`` label."""
    resources_dir = resources_dir or RESOURCES_DIR
    articles = load_json(resources_dir / "echr_articles.json")
    out: dict[str, str] = {}
    for name in spec.label_names:
        if name == spec.none_label:
            out[name] = ECTHR_NONE_DESCRIPTIONS[spec.name]
        else:
            entry = articles[name]
            out[name] = f"{entry['title']}. {entry['text']}"
    return out


def _eurovoc_graph(skos_path: Path):
    import rdflib

    graph = rdflib.Graph()
    path = Path(skos_path)
    files = [path] if path.is_file() else sorted(path.glob("*.rdf")) + sorted(path.glob("*.ttl")) + sorted(path.glob("*.xml"))
    if not files:
        raise FileNotFoundError(f"no SKOS files found under {skos_path}")
    for f in files:
        fmt = "turtle" if f.suffix == ".ttl" else "xml"
        graph.parse(str(f), format=fmt)
    logger.info("parsed EuroVoc SKOS export (%d triples)", len(graph))
    return graph


def _literal(graph, subject, predicate, lang: str = "en") -> str | None:
    fallback = None
    for obj in graph.objects(subject, predicate):
        if getattr(obj, "language", None) == lang:
            return str(obj)
        if fallback is None:
            fallback = str(obj)
    return fallback


def _pref_label(graph, subject, lang: str = "en") -> str | None:
    import rdflib
    from rdflib.namespace import SKOS

    label = _literal(graph, subject, SKOS.prefLabel, lang)
    if label:
        return label
    SKOSXL = rdflib.Namespace("http://www.w3.org/2008/05/skos-xl#")
    for xl in graph.objects(subject, SKOSXL.prefLabel):
        label = _literal(graph, xl, SKOSXL.literalForm, lang)
        if label:
            return label
    return None


def eurovoc_descriptions(
    spec: TaskSpec, skos_path: Path, lang: str = "en", max_broader: int = 6
) -> tuple[dict[str, str], dict[str, str]]:
    """EuroVoc descriptions (preferred term, scope note, broader-term path) and micro-thesaurus membership.

    Label names in LexGLUE EUR-LEX are EuroVoc concept identifiers; the concept
    URI is ``http://eurovoc.europa.eu/<id>``.
    """
    import rdflib
    from rdflib.namespace import SKOS

    graph = _eurovoc_graph(skos_path)
    descriptions: dict[str, str] = {}
    groups: dict[str, str] = {}
    for name in spec.label_names:
        concept = rdflib.URIRef(EUROVOC_NS + str(name))
        pref = _pref_label(graph, concept, lang) or f"EuroVoc concept {name}"
        scope = _literal(graph, concept, SKOS.scopeNote, lang) or _literal(graph, concept, SKOS.definition, lang)
        broader_terms: list[str] = []
        node = concept
        seen = {concept}
        while len(broader_terms) < max_broader:
            parents = [p for p in graph.objects(node, SKOS.broader) if p not in seen]
            if not parents:
                break
            node = parents[0]
            seen.add(node)
            label = _pref_label(graph, node, lang)
            if label:
                broader_terms.append(label)
        schemes = [s for s in graph.objects(concept, SKOS.inScheme) if str(s) != EUROVOC_THESAURUS_SCHEME]
        if schemes:
            groups[name] = _pref_label(graph, schemes[0], lang) or str(schemes[0])
        parts = [pref.strip().rstrip(".") + "."]
        if scope:
            parts.append(scope.strip().rstrip(".") + ".")
        if broader_terms:
            parts.append("Broader terms: " + " > ".join(reversed(broader_terms)) + ".")
        descriptions[name] = " ".join(parts)
    missing = [n for n in spec.label_names if n not in groups]
    if missing:
        logger.warning("%d EuroVoc concepts without micro-thesaurus membership", len(missing))
    return descriptions, groups


def criminal_law_descriptions(spec: TaskSpec, articles_path: Path) -> dict[str, str]:
    """Text of each article of the Criminal Law of the PRC as distributed with the CAIL2018 release."""
    articles = {str(k): str(v) for k, v in load_json(articles_path).items()}
    out: dict[str, str] = {}
    missing = []
    for name in spec.label_names:
        text = articles.get(name)
        if text is None:
            missing.append(name)
            text = f"《中华人民共和国刑法》第{name}条"
        out[name] = text.strip()
    if missing:
        logger.warning("no statute text for %d articles (%s ...)", len(missing), ", ".join(missing[:5]))
    return out


def build_label_descriptions(cfg: dict, spec: TaskSpec) -> tuple[dict[str, str], dict[str, str] | None]:
    """Return ``(descriptions, structural_groups)`` for the task; groups are ``None`` when they
    come from a static resource file rather than from the description source."""
    knowledge = cfg["data"].get("knowledge", {}) or {}
    if spec.name in ("ecthr_a", "ecthr_b"):
        return echr_descriptions(spec, Path(knowledge.get("resources_dir", RESOURCES_DIR))), None
    if spec.name == "eurlex":
        descriptions, groups = eurovoc_descriptions(spec, Path(knowledge["eurovoc_skos"]), lang=knowledge.get("lang", "en"))
        return descriptions, groups
    if spec.name == "cail2018":
        return criminal_law_descriptions(spec, Path(knowledge["criminal_law_articles"])), None
    raise ValueError(f"no label-description source for task '{spec.name}'")


def save_label_descriptions(descriptions: dict[str, str], groups: dict[str, str] | None, path: Path) -> None:
    payload = {"descriptions": descriptions}
    if groups:
        payload["groups"] = groups
    save_json(payload, path)


def load_label_descriptions(path: Path, spec: TaskSpec) -> list[str]:
    payload = load_json(path)
    descriptions = payload["descriptions"] if "descriptions" in payload else payload
    missing = [n for n in spec.label_names if n not in descriptions]
    if missing:
        raise KeyError(f"{path} lacks descriptions for labels {missing[:5]}...")
    return [descriptions[n] for n in spec.label_names]


def load_label_groups(path: Path) -> dict[str, str] | None:
    payload = load_json(path)
    return payload.get("groups") if isinstance(payload, dict) else None


def description_lengths(descriptions: dict[str, str], language: str) -> dict[str, float]:
    if language == "zh":
        lengths = [len(t) for t in descriptions.values()]
        unit = "characters"
    else:
        lengths = [len(t.split()) for t in descriptions.values()]
        unit = "words"
    return {"unit": unit, "mean": sum(lengths) / max(len(lengths), 1), "min": min(lengths), "max": max(lengths)}
