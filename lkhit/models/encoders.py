"""Pre-trained text encoders shared by the document and knowledge branches, and segment-wise encoding."""

from __future__ import annotations

import logging
from typing import Iterable

import torch
import torch.nn as nn

logger = logging.getLogger("lkhit.models.encoders")

LONGFORMER_TYPES = ("longformer",)
BIGBIRD_TYPES = ("big_bird", "bigbird")


class TextEncoder(nn.Module):
    """Thin wrapper around a Hugging Face encoder returning token states and the [CLS] vector.

    LEGAL-BERT, BERT, Chinese RoBERTa-wwm-ext, Longformer, BigBird and Lawformer
    all go through this class; the only architecture-specific behaviour is the
    global-attention mask that Longformer-style models expect on [CLS].
    """

    def __init__(self, name_or_path: str, gradient_checkpointing: bool = False, attention_type: str | None = None) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained(name_or_path)
        if attention_type and hasattr(config, "attention_type"):
            config.attention_type = attention_type
        try:
            self.model = AutoModel.from_pretrained(name_or_path, config=config, add_pooling_layer=False)
        except TypeError:
            self.model = AutoModel.from_pretrained(name_or_path, config=config)
        if gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        self.config = self.model.config
        self.name_or_path = name_or_path
        self.hidden_size = int(self.config.hidden_size)
        logger.info("encoder %s: %s, hidden size %d", name_or_path, self.config.model_type, self.hidden_size)

    @property
    def is_longformer(self) -> bool:
        return self.config.model_type in LONGFORMER_TYPES

    @property
    def is_bigbird(self) -> bool:
        return self.config.model_type in BIGBIRD_TYPES

    def _global_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(attention_mask)
        mask[:, 0] = 1
        return mask

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, "return_dict": True}
        if self.is_longformer:
            kwargs["global_attention_mask"] = self._global_attention_mask(attention_mask)
        out = self.model(**kwargs)
        return out.last_hidden_state

    def cls(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.forward(input_ids, attention_mask)[:, 0]

    def encode_in_chunks(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, chunk_size: int) -> torch.Tensor:
        """[CLS] vectors of many sequences, processed ``chunk_size`` at a time to bound peak memory."""
        outputs = []
        for start in range(0, input_ids.size(0), chunk_size):
            outputs.append(self.cls(input_ids[start : start + chunk_size], attention_mask[start : start + chunk_size]))
        return torch.cat(outputs, dim=0)

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        frozen = not any(p.requires_grad for p in self.parameters())
        return super().train(mode and not frozen)


def encode_segments(
    encoder: TextEncoder,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    segment_mask: torch.Tensor,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Encode an ``[B, N, T]`` block of segments into ``[B, N, d]`` [CLS] states.

    Padded segments (``segment_mask == 0``) are skipped rather than encoded, which
    is where most of the saving of batch-level padding comes from on ECtHR,
    whose documents range from a handful to more than a hundred paragraphs.
    """
    bsz, n_seg, seq_len = input_ids.shape
    flat_ids = input_ids.reshape(bsz * n_seg, seq_len)
    flat_mask = attention_mask.reshape(bsz * n_seg, seq_len)
    keep = segment_mask.reshape(-1).bool()
    if keep.sum() == 0:
        keep = keep.clone()
        keep[0] = True
    kept_ids, kept_mask = flat_ids[keep], flat_mask[keep]
    if chunk_size:
        kept_states = encoder.encode_in_chunks(kept_ids, kept_mask, chunk_size)
    else:
        kept_states = encoder.cls(kept_ids, kept_mask)
    states = kept_states.new_zeros((bsz * n_seg, kept_states.size(-1)))
    states[keep] = kept_states
    return states.reshape(bsz, n_seg, -1)


def no_decay_parameter(name: str) -> bool:
    """Bias and normalisation weights are excluded from weight decay, as is standard for BERT fine-tuning."""
    lowered = name.lower()
    return lowered.endswith(".bias") or "layernorm" in lowered or "layer_norm" in lowered or ".norm" in lowered


def split_parameter_groups(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    encoder_prefixes: tuple[str, ...],
    lr_encoder: float,
    lr_upper: float,
    weight_decay: float,
) -> list[dict]:
    """AdamW groups: pre-trained encoder at ``lr_encoder``, everything else at ``lr_upper``; no decay on bias / norm."""
    groups: dict[tuple[str, bool], list[nn.Parameter]] = {}
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        is_encoder = any(name.startswith(prefix) for prefix in encoder_prefixes)
        key = ("encoder" if is_encoder else "upper", no_decay_parameter(name))
        groups.setdefault(key, []).append(param)
    out = []
    for (role, no_decay), params in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        out.append(
            {
                "params": params,
                "lr": lr_encoder if role == "encoder" else lr_upper,
                "weight_decay": 0.0 if no_decay else weight_decay,
                "name": f"{role}{'_no_decay' if no_decay else ''}",
            }
        )
    return out
