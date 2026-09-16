"""TopJudge (Zhong et al., 2018): CNN fact encoder and a topological (DAG) LSTM decoder over sub-tasks.

The original model predicts law articles, charges and prison terms with one
LSTM cell per sub-task; the hidden and cell states of a task are initialised
from linear combinations of the states of its predecessors in the dependency
graph. Under the LADAN protocol only the law-article sub-task is scored, so the
default configuration declares a single task without predecessors; the decoder
keeps the general DAG form so that the full three-task chain can be enabled
from the configuration when charges and terms are available.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lkhit.data.tasks import TaskSpec
from lkhit.models.outputs import ModelOutput
from lkhit.models.word_encoders import CNNEncoder, WordEmbedding


class TopologicalLSTMDecoder(nn.Module):
    def __init__(self, tasks: list[str], dependencies: dict[str, list[str]], input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.tasks = list(tasks)
        self.dependencies = {t: list(dependencies.get(t, [])) for t in self.tasks}
        self.hidden_size = hidden_size
        self.cells = nn.ModuleDict({t: nn.LSTMCell(input_size, hidden_size) for t in self.tasks})
        self.hidden_transfer = nn.ModuleDict()
        self.cell_transfer = nn.ModuleDict()
        for task in self.tasks:
            for parent in self.dependencies[task]:
                if parent not in self.tasks:
                    raise ValueError(f"task '{task}' depends on unknown task '{parent}'")
                if self.tasks.index(parent) >= self.tasks.index(task):
                    raise ValueError("tasks must be listed in topological order")
                self.hidden_transfer[f"{parent}->{task}"] = nn.Linear(hidden_size, hidden_size, bias=False)
                self.cell_transfer[f"{parent}->{task}"] = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, fact_vector: torch.Tensor) -> dict[str, torch.Tensor]:
        bsz = fact_vector.size(0)
        hidden: dict[str, torch.Tensor] = {}
        cell: dict[str, torch.Tensor] = {}
        for task in self.tasks:
            h0 = fact_vector.new_zeros(bsz, self.hidden_size)
            c0 = fact_vector.new_zeros(bsz, self.hidden_size)
            for parent in self.dependencies[task]:
                h0 = h0 + self.hidden_transfer[f"{parent}->{task}"](hidden[parent])
                c0 = c0 + self.cell_transfer[f"{parent}->{task}"](cell[parent])
            h, c = self.cells[task](fact_vector, (h0, c0))
            hidden[task], cell[task] = h, c
        return hidden


class TopJudge(nn.Module):
    encoder_prefixes: tuple[str, ...] = ()

    def __init__(self, model_cfg: dict, spec: TaskSpec, vocab, pretrained: torch.Tensor | None = None) -> None:
        super().__init__()
        self.spec = spec
        dim = int(model_cfg.get("embedding_dim", 200))
        self.embedding = WordEmbedding(len(vocab), dim, vocab.pad_index, pretrained, dropout=float(model_cfg.get("embedding_dropout", 0.2)))
        self.fact_encoder = CNNEncoder(
            dim,
            n_filters=int(model_cfg.get("n_filters", 64)),
            widths=tuple(model_cfg.get("filter_widths", [2, 3, 4, 5])),
            dropout=float(model_cfg.get("dropout", 0.5)),
        )
        hidden = int(model_cfg.get("hidden_size", 256))
        self.fact_projection = nn.Linear(self.fact_encoder.output_dim, hidden)
        tasks = list(model_cfg.get("tasks", ["article"]))
        dependencies = dict(model_cfg.get("dependencies", {}) or {})
        self.decoder = TopologicalLSTMDecoder(tasks, dependencies, hidden, hidden)
        self.output_task = model_cfg.get("output_task", "article")
        if self.output_task not in tasks:
            raise ValueError(f"output task '{self.output_task}' is not among {tasks}")
        self.classifiers = nn.ModuleDict({t: nn.Linear(hidden, spec.n_labels if t == self.output_task else int(model_cfg.get("aux_task_sizes", {}).get(t, spec.n_labels))) for t in tasks})
        self.dropout = nn.Dropout(float(model_cfg.get("dropout", 0.5)))

    def forward(self, batch: dict) -> ModelOutput:
        mask = batch["word_mask"].bool()
        embedded = self.embedding(batch["word_ids"])
        fact = torch.relu(self.fact_projection(self.fact_encoder(embedded, mask)))
        hidden = self.decoder(self.dropout(fact))
        logits = self.classifiers[self.output_task](self.dropout(hidden[self.output_task]))
        return ModelOutput(logits=logits)

    def parameter_groups(self, lr_encoder: float, lr_upper: float, weight_decay: float) -> list[dict]:
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if name.endswith("bias") or "embedding" in name else decay).append(p)
        return [
            {"params": decay, "lr": lr_upper, "weight_decay": weight_decay, "name": "upper"},
            {"params": no_decay, "lr": lr_upper, "weight_decay": 0.0, "name": "upper_no_decay"},
        ]

    def parameter_summary(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad), "encoder": 0, "upper_layers": total}
