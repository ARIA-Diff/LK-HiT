"""Truncated encoders (BERT-512, Legal-BERT-512, RoBERTa-wwm-512): [CLS] of the first 512 tokens + linear layer."""

from __future__ import annotations

import torch.nn as nn

from lkhit.data.tasks import TaskSpec
from lkhit.models.encoders import TextEncoder, split_parameter_groups
from lkhit.models.outputs import ModelOutput


class TruncatedEncoderClassifier(nn.Module):
    encoder_prefixes = ("encoder.",)

    def __init__(self, model_cfg: dict, spec: TaskSpec, encoder_name: str, attention_type: str | None = None) -> None:
        super().__init__()
        self.spec = spec
        self.encoder = TextEncoder(
            encoder_name,
            gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", False)),
            attention_type=attention_type,
        )
        d = self.encoder.hidden_size
        self.dropout = nn.Dropout(float(model_cfg.get("dropout", 0.1)))
        self.classifier = nn.Linear(d, spec.n_labels)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, batch: dict) -> ModelOutput:
        cls = self.encoder.cls(batch["input_ids"], batch["attention_mask"])
        logits = self.classifier(self.dropout(cls))
        return ModelOutput(logits=logits)

    def parameter_groups(self, lr_encoder: float, lr_upper: float, weight_decay: float) -> list[dict]:
        return split_parameter_groups(self.named_parameters(), self.encoder_prefixes, lr_encoder, lr_upper, weight_decay)

    def parameter_summary(self) -> dict[str, int]:
        encoder = sum(p.numel() for p in self.encoder.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad), "encoder": encoder, "upper_layers": total - encoder}
