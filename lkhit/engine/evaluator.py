"""Inference over a split and the full metric report: discrimination, tiers, calibration, abstention, subgroups.

Decisions are taken at 0.5 (multi-label) or by arg-max (single-label) on the
raw probabilities; temperature scaling changes neither, so discrimination
metrics are identical before and after calibration. Attention weights are
collected only when requested, for the rationale analysis.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from lkhit.calibration import TemperatureScaler, calibration_metrics, logits_to_probs, reliability_table
from lkhit.data.dataset import Document
from lkhit.data.tasks import TaskSpec
from lkhit.metrics import (
    decisions_from_probs,
    discrimination_metrics,
    label_tiers,
    length_tier_name,
    one_hot,
    per_label_f1,
    subgroup_metrics,
    tier_metrics,
)
from lkhit.selective import risk_coverage_table, selective_metrics, uncertainty_score, f1_at_coverage
from lkhit.utils import move_to_device, save_json

logger = logging.getLogger("lkhit.engine.evaluator")


@dataclass
class Predictions:
    logits: np.ndarray
    labels: np.ndarray  # [n, L] binary matrix (multi-label) or [n] indices (single-label)
    index: np.ndarray
    n_segments: np.ndarray | None = None
    label_attention: list[np.ndarray] | None = None
    document_attention: list[np.ndarray] | None = None
    label_embeddings: np.ndarray | None = None
    aux_logits: np.ndarray | None = None
    wall_time: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.logits.shape[0])

    def label_matrix(self, n_labels: int) -> np.ndarray:
        return one_hot(self.labels, n_labels)


class Evaluator:
    def __init__(self, model: torch.nn.Module, spec: TaskSpec, device: torch.device, amp_dtype: torch.dtype | None = None) -> None:
        self.model = model
        self.spec = spec
        self.device = device
        self.amp_dtype = amp_dtype

    @torch.no_grad()
    def predict(self, loader, collect_attention: bool = False, log_every: int = 50) -> Predictions:
        self.model.eval()
        logits, labels, index, n_segments = [], [], [], []
        label_att: list[np.ndarray] = []
        doc_att: list[np.ndarray] = []
        aux = []
        label_embeddings = None
        start = time.perf_counter()
        use_amp = self.amp_dtype is not None and self.device.type == "cuda"
        for step, batch in enumerate(loader, start=1):
            batch = move_to_device(batch, self.device)
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype or torch.float16, enabled=use_amp):
                out = self.model(batch)
            logits.append(out.logits.float().cpu().numpy())
            labels.append(batch["labels"].cpu().numpy())
            index.append(batch["index"].cpu().numpy())
            if "n_segments" in batch:
                n_segments.append(batch["n_segments"].cpu().numpy())
            if out.aux_logits is not None:
                aux.append(out.aux_logits.float().cpu().numpy())
            if collect_attention:
                if out.label_attention is not None:
                    att = out.label_attention.float().cpu().numpy()
                    label_att.extend(list(att))
                if out.document_attention is not None:
                    att = out.document_attention.float().cpu().numpy()
                    doc_att.extend(list(att))
            if out.label_embeddings is not None and label_embeddings is None:
                label_embeddings = out.label_embeddings.detach().float().cpu().numpy()
            if log_every and step % log_every == 0:
                logger.info("eval batch %d/%d (%.1fs)", step, len(loader), time.perf_counter() - start)
        wall = time.perf_counter() - start
        preds = Predictions(
            logits=np.concatenate(logits),
            labels=np.concatenate(labels),
            index=np.concatenate(index),
            n_segments=np.concatenate(n_segments) if n_segments else None,
            label_attention=label_att if label_att else None,
            document_attention=doc_att if doc_att else None,
            label_embeddings=label_embeddings,
            aux_logits=np.concatenate(aux) if aux else None,
            wall_time=wall,
        )
        order = np.argsort(preds.index, kind="stable")
        preds = _reorder(preds, order)
        logger.info("predicted %d documents in %.1fs (%.1f docs/s)", preds.n, wall, preds.n / max(wall, 1e-9))
        return preds


def _reorder(preds: Predictions, order: np.ndarray) -> Predictions:
    preds.logits = preds.logits[order]
    preds.labels = preds.labels[order]
    preds.index = preds.index[order]
    if preds.n_segments is not None:
        preds.n_segments = preds.n_segments[order]
    if preds.aux_logits is not None:
        preds.aux_logits = preds.aux_logits[order]
    if preds.label_attention is not None:
        preds.label_attention = [preds.label_attention[i] for i in order]
    if preds.document_attention is not None:
        preds.document_attention = [preds.document_attention[i] for i in order]
    return preds


# --------------------------------------------------------------------------- metric report


def fit_temperature(preds: Predictions, spec: TaskSpec, max_iter: int = 50) -> TemperatureScaler:
    scaler = TemperatureScaler(multi_label=spec.multi_label, max_iter=max_iter)
    scaler.fit(preds.logits, preds.label_matrix(spec.n_labels) if spec.multi_label else preds.labels)
    logger.info("temperature fitted on the development split: T = %.4f", scaler.temperature)
    return scaler


def document_groups(documents: Sequence[Document], spec: TaskSpec, subgroup_cfg: dict | None) -> dict[str, list[str | None]]:
    """Subgroup labels per document: decision year (ECtHR test) and document-length tier."""
    groups: dict[str, list[str | None]] = {}
    if any("year" in doc.meta for doc in documents):
        groups["year"] = [str(doc.meta["year"]) if "year" in doc.meta else None for doc in documents]
    subgroup_cfg = subgroup_cfg or {}
    boundaries = subgroup_cfg.get("length_boundaries")
    if boundaries:
        unit = subgroup_cfg.get("length_unit", "paragraphs" if spec.segmentation == "paragraph" else "tokens")
        values = []
        for doc in documents:
            if unit == "paragraphs":
                values.append(doc.meta.get("n_paragraphs", len(doc.segments or [])))
            elif unit == "characters":
                values.append(doc.meta.get("n_chars", len(doc.full_text(""))))
            else:
                values.append(doc.meta.get("n_tokens", doc.meta.get("n_chars", len(doc.full_text(" ")))))
        groups["length"] = [length_tier_name(int(v), boundaries, unit) for v in values]
    return groups


def metric_report(
    preds: Predictions,
    spec: TaskSpec,
    temperature: float | None = None,
    train_counts: np.ndarray | None = None,
    documents: Sequence[Document] | None = None,
    subgroup_cfg: dict | None = None,
    target_risk: float = 0.10,
) -> dict:
    gold = preds.label_matrix(spec.n_labels)
    probs_raw = logits_to_probs(preds.logits, spec.multi_label, 1.0)
    pred = decisions_from_probs(probs_raw, spec)
    report: dict = {
        "n_documents": preds.n,
        "wall_time_s": preds.wall_time,
        "docs_per_second": preds.n / max(preds.wall_time, 1e-9),
        "discrimination": discrimination_metrics(pred, gold, spec),
        "per_label_f1": (100 * per_label_f1(pred, gold)).round(2).tolist(),
        "label_names": list(spec.label_names),
        "calibration": {"raw": calibration_metrics(probs_raw, gold, spec)},
        "reliability": {"raw": reliability_table(probs_raw, gold, spec)},
    }
    probs_for_selection = probs_raw
    if temperature is not None:
        probs_ts = logits_to_probs(preds.logits, spec.multi_label, temperature)
        report["temperature"] = float(temperature)
        report["calibration"]["temperature_scaled"] = calibration_metrics(probs_ts, gold, spec)
        report["reliability"]["temperature_scaled"] = reliability_table(probs_ts, gold, spec)
        probs_for_selection = probs_ts
    report["selective"] = selective_metrics(probs_for_selection, gold, spec, target_risk=target_risk)
    report["selective"]["f1_at_coverage"] = f1_at_coverage(probs_for_selection, gold, spec)
    report["risk_coverage"] = risk_coverage_table(probs_for_selection, gold, spec)

    if train_counts is not None and spec.tier_scheme:
        exclude = [spec.none_index] if spec.none_index is not None else []
        tiers = label_tiers(train_counts, spec.tier_scheme, exclude=exclude)
        report["tiers"] = tier_metrics(pred, gold, tiers)
        report["tier_members"] = {k: [spec.label_names[i] for i in v] for k, v in tiers.items()}
    if train_counts is not None:
        report["train_counts"] = [int(c) for c in train_counts]

    if documents is not None:
        ordered = [documents[i] for i in preds.index]
        for name, labels_of_doc in document_groups(ordered, spec, subgroup_cfg).items():
            report.setdefault("subgroups", {})[name] = subgroup_metrics(pred, gold, spec, labels_of_doc)
    return report


def save_predictions(preds: Predictions, spec: TaskSpec, path: Path, temperature: float | None = None, with_attention: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    probs_raw = logits_to_probs(preds.logits, spec.multi_label, 1.0)
    payload = {
        "logits": preds.logits.astype(np.float32),
        "probs_raw": probs_raw.astype(np.float32),
        "labels": preds.label_matrix(spec.n_labels).astype(np.int8),
        "index": preds.index.astype(np.int64),
        "uncertainty_raw": uncertainty_score(probs_raw, spec.multi_label).astype(np.float32),
    }
    if temperature is not None:
        probs_ts = logits_to_probs(preds.logits, spec.multi_label, temperature)
        payload["probs_ts"] = probs_ts.astype(np.float32)
        payload["uncertainty"] = uncertainty_score(probs_ts, spec.multi_label).astype(np.float32)
        payload["temperature"] = np.asarray(temperature, dtype=np.float32)
    if preds.n_segments is not None:
        payload["n_segments"] = preds.n_segments.astype(np.int32)
    if preds.aux_logits is not None:
        payload["aux_logits"] = preds.aux_logits.astype(np.float32)
    if preds.label_embeddings is not None:
        payload["label_embeddings"] = preds.label_embeddings.astype(np.float32)
    np.savez_compressed(path, **payload)
    if with_attention and preds.label_attention is not None:
        att_path = path.with_name(path.stem + "_label_attention.npz")
        np.savez_compressed(att_path, **{f"doc_{i:06d}": a.astype(np.float16) for i, a in enumerate(preds.label_attention)})
    if with_attention and preds.document_attention is not None:
        att_path = path.with_name(path.stem + "_document_attention.npz")
        np.savez_compressed(att_path, **{f"doc_{i:06d}": a.astype(np.float16) for i, a in enumerate(preds.document_attention)})
    logger.info("predictions written to %s", path)


def save_report(report: dict, path: Path) -> None:
    save_json(report, path)
    d = report["discrimination"]
    headline = ", ".join(f"{k}={v:.2f}" for k, v in d.items() if isinstance(v, float))
    cal = report["calibration"]
    ece = f"ECE raw {cal['raw']['ece']:.2f}"
    if "temperature_scaled" in cal:
        ece += f" / TS {cal['temperature_scaled']['ece']:.2f}"
    logger.info("%s | %s | AURC %.2f -> %s", headline, ece, report["selective"]["aurc"], path)
