"""Document segmentation: court paragraphs, fixed-length token chunks and Chinese sentences."""

from __future__ import annotations

import re
from typing import Sequence

_ZH_SENTENCE_END = re.compile(r"(?<=[。！？；!?;])")
_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def split_paragraphs(paragraphs: Sequence[str], max_segments: int | None = None) -> list[str]:
    """Keep the paragraph boundaries provided by the corpus, dropping empty ones."""
    out = [clean_text(p) for p in paragraphs]
    out = [p for p in out if p]
    if max_segments is not None:
        out = out[:max_segments]
    return out


def split_sentences_zh(text: str, max_segments: int | None = None, min_chars: int = 2) -> list[str]:
    """Split a Chinese fact description at sentence-final punctuation.

    Very short fragments (a trailing punctuation mark, a number) are glued to the
    preceding sentence so that every segment carries some content.
    """
    pieces = [p.strip() for p in _ZH_SENTENCE_END.split(clean_text(text)) if p.strip()]
    sentences: list[str] = []
    for piece in pieces:
        if sentences and len(piece) < min_chars:
            sentences[-1] += piece
        else:
            sentences.append(piece)
    if max_segments is not None:
        sentences = sentences[:max_segments]
    return sentences


def chunk_token_ids(
    token_ids: Sequence[int],
    chunk_tokens: int,
    max_chunks: int | None = None,
    cls_id: int | None = None,
    sep_id: int | None = None,
) -> list[list[int]]:
    """Cut a token-id sequence into consecutive windows of ``chunk_tokens`` tokens.

    Special tokens are reserved inside every window so that each chunk can be
    fed to the encoder as an independent sequence.
    """
    reserved = int(cls_id is not None) + int(sep_id is not None)
    body = max(chunk_tokens - reserved, 1)
    chunks = []
    for start in range(0, len(token_ids), body):
        window = list(token_ids[start : start + body])
        if cls_id is not None:
            window = [cls_id] + window
        if sep_id is not None:
            window = window + [sep_id]
        chunks.append(window)
        if max_chunks is not None and len(chunks) >= max_chunks:
            break
    if not chunks:
        window = []
        if cls_id is not None:
            window.append(cls_id)
        if sep_id is not None:
            window.append(sep_id)
        chunks.append(window)
    return chunks


def length_tier(value: int, boundaries: Sequence[int]) -> int:
    """Index of the tier a length falls in given increasing upper boundaries."""
    for i, upper in enumerate(boundaries):
        if value <= upper:
            return i
    return len(boundaries)
