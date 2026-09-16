import torch

from lkhit.models.document_transformer import DocumentTransformer, masked_mean
from lkhit.models.label_attention import BilinearScorer, LabelAwareAttention
from lkhit.models.label_graph import LabelGraphConvolution


def test_label_aware_attention_shapes() -> None:
    d, b, n, l = 16, 3, 5, 4
    attn = LabelAwareAttention(d, dropout=0.0)
    labels = torch.randn(l, d)
    states = torch.randn(b, n, d)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]], dtype=torch.bool)
    summaries, weights = attn(labels, states, mask)
    assert summaries.shape == (b, l, d)
    assert weights.shape == (b, l, n)
    assert torch.allclose(weights[0, :, 3:].sum(), torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(weights.sum(dim=-1)[mask.any(dim=1)], torch.ones(mask.any(dim=1).sum(), l), atol=1e-5)


def test_bilinear_scorer_label_specific_and_pooled() -> None:
    scorer = BilinearScorer(8, 3)
    labels = torch.randn(3, 8)
    summaries = torch.randn(2, 3, 8)
    logits = scorer(summaries, labels)
    assert logits.shape == (2, 3)
    pooled = torch.randn(2, 8)
    assert scorer(pooled, labels).shape == (2, 3)


def test_graph_convolution_residual() -> None:
    gcn = LabelGraphConvolution(8, n_layers=2)
    e0 = torch.randn(6, 8)
    adj = torch.eye(6)
    out = gcn(e0, adj)
    assert out.shape == e0.shape
    assert not torch.allclose(out, e0)


def test_document_transformer_respects_mask() -> None:
    layer = DocumentTransformer(hidden_size=16, n_layers=2, n_heads=4, max_segments=8, dropout=0.0)
    states = torch.randn(2, 6, 16)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    hidden, attention = layer(states, mask, return_attention=True)
    assert hidden.shape == states.shape
    assert attention is not None
    assert torch.allclose(hidden[0, 3:], torch.zeros(3, 16), atol=1e-6)
    pooled = masked_mean(hidden, mask)
    assert pooled.shape == (2, 16)
