"""Graph convolution over the legal label graph (Eq. 2 of the paper).

    E = A~ ReLU(A~ E0 W1) W2 + E0

where ``A~`` is the symmetrically normalised adjacency (co-occurrence PMI edges
plus statutory-structure edges, self-loops added; see
:mod:`lkhit.data.label_graph`) and the residual term keeps the statute text
dominant for labels with few neighbours.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LabelGraphConvolution(nn.Module):
    def __init__(self, hidden_size: int, n_layers: int = 2, residual: bool = True) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("the graph convolution needs at least one layer")
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(n_layers)])
        self.residual = residual
        for lin in self.transforms:
            nn.init.xavier_uniform_(lin.weight)

    def forward(self, label_embeddings: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """``label_embeddings``: [L, d]; ``adjacency``: [L, L] normalised."""
        adjacency = adjacency.to(label_embeddings.dtype)
        x = label_embeddings
        n = len(self.transforms)
        for i, lin in enumerate(self.transforms):
            x = adjacency @ lin(x)
            if i < n - 1:
                x = torch.relu(x)
        if self.residual:
            x = x + label_embeddings
        return x


def label_similarity(label_embeddings: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between refined label embeddings (reported as a similarity matrix in the paper)."""
    normed = torch.nn.functional.normalize(label_embeddings.float(), dim=-1)
    return normed @ normed.T
