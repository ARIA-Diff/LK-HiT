"""Sparse-attention long-document encoders: Longformer and BigBird (4,096 tokens), Lawformer (1,024 tokens).

All three read one long token sequence and classify from [CLS]. Longformer and
Lawformer (a Longformer pre-trained on Chinese legal documents) receive global
attention on [CLS]; BigBird runs in block-sparse mode and falls back to full
attention on sequences too short for its block pattern, as in the reference
implementation.
"""

from __future__ import annotations

import logging

from lkhit.data.tasks import TaskSpec
from lkhit.models.truncated import TruncatedEncoderClassifier

logger = logging.getLogger("lkhit.models.sparse")


class SparseAttentionClassifier(TruncatedEncoderClassifier):
    def __init__(self, model_cfg: dict, spec: TaskSpec, encoder_name: str) -> None:
        attention_type = model_cfg.get("attention_type")  # BigBird: block_sparse | original_full
        super().__init__(model_cfg, spec, encoder_name, attention_type=attention_type)
        self.max_tokens = int(model_cfg.get("max_tokens", spec.sparse_max_tokens))
        max_positions = getattr(self.encoder.config, "max_position_embeddings", None)
        if max_positions is not None and self.max_tokens > max_positions:
            logger.warning(
                "%s supports %d positions but max_tokens=%d; inputs are truncated by the tokenizer",
                encoder_name,
                max_positions,
                self.max_tokens,
            )
        logger.info(
            "sparse encoder %s (%s): window %d tokens, global attention on [CLS]=%s",
            encoder_name,
            self.encoder.config.model_type,
            self.max_tokens,
            self.encoder.is_longformer,
        )
