"""Training loop shared by LK-HiT and every neural baseline.

AdamW with two learning rates (pre-trained encoder 3e-5, upper layers 1e-4),
linear warm-up over the first 10 % of optimisation steps followed by linear
decay, weight decay 0.01, gradient accumulation to an effective batch of 16
documents, mixed precision, at most 20 epochs and early stopping with patience
3 on development macro-F1. ``best.pt`` holds the weights of the selected epoch,
``last.pt`` the resumable state, and ``metrics.jsonl`` one record per epoch.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR

from lkhit.data.tasks import TaskSpec
from lkhit.engine.checkpoint import checkpoint_dir, resume_state, save_best, save_last
from lkhit.engine.evaluator import Evaluator, metric_report
from lkhit.losses import total_loss
from lkhit.utils import JsonlWriter, move_to_device, peak_memory_gb

logger = logging.getLogger("lkhit.engine.trainer")


def linear_warmup_decay(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    def factor(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        remaining = total_steps - step
        return max(0.0, remaining / max(1, total_steps - warmup_steps))

    return LambdaLR(optimizer, factor)


def amp_dtype_from_config(train_cfg: dict, device: torch.device) -> torch.dtype | None:
    if not train_cfg.get("amp", True) or device.type != "cuda":
        return None
    name = str(train_cfg.get("amp_dtype", "float16")).lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float16


class EarlyStopping:
    def __init__(self, patience: int, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.best: float | None = None
        self.best_epoch = 0
        self.bad_epochs = 0

    def step(self, value: float, epoch: int) -> bool:
        """Return True when ``value`` improves on the best score so far."""
        improved = self.best is None or (value > self.best if self.mode == "max" else value < self.best)
        if improved:
            self.best, self.best_epoch, self.bad_epochs = value, epoch, 0
        else:
            self.bad_epochs += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


class Trainer:
    def __init__(
        self,
        cfg: dict,
        spec: TaskSpec,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        train_loader,
        dev_loader,
        run_dir: Path,
        device: torch.device,
        seed: int,
        train_counts=None,
    ) -> None:
        self.cfg = cfg
        self.train_cfg = cfg["train"]
        self.spec = spec
        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.run_dir = Path(run_dir)
        self.device = device
        self.seed = seed
        self.train_counts = train_counts

        self.accumulation = int(self.train_cfg.get("grad_accumulation", 1))
        self.epochs = int(self.train_cfg.get("epochs", 20))
        self.max_grad_norm = float(self.train_cfg.get("max_grad_norm", 1.0))
        self.log_every = int(self.train_cfg.get("log_every", 50))
        self.selection_metric = self.train_cfg.get("selection_metric", "macro_f1")
        self.aux_weight = float(self.train_cfg.get("aux_loss_weight", 1.0))

        self.optimizer = torch.optim.AdamW(
            model.parameter_groups(
                lr_encoder=float(self.train_cfg.get("lr_encoder", 3e-5)),
                lr_upper=float(self.train_cfg.get("lr_upper", 1e-4)),
                weight_decay=float(self.train_cfg.get("weight_decay", 0.01)),
            ),
            betas=tuple(self.train_cfg.get("betas", (0.9, 0.999))),
            eps=float(self.train_cfg.get("adam_eps", 1e-8)),
        )
        steps_per_epoch = math.ceil(len(train_loader) / self.accumulation)
        self.total_steps = steps_per_epoch * self.epochs
        self.warmup_steps = int(round(float(self.train_cfg.get("warmup_ratio", 0.10)) * self.total_steps))
        self.scheduler = linear_warmup_decay(self.optimizer, self.warmup_steps, self.total_steps)

        self.amp_dtype = amp_dtype_from_config(self.train_cfg, device)
        self.scaler = torch.amp.GradScaler("cuda") if self.amp_dtype == torch.float16 else None
        self.evaluator = Evaluator(model, spec, device, amp_dtype=self.amp_dtype)
        self.early_stopping = EarlyStopping(int(self.train_cfg.get("patience", 3)))
        self.metrics_writer = JsonlWriter(self.run_dir / "metrics.jsonl")
        self.history: list[dict] = []
        self.global_step = 0
        self.start_epoch = 1

        for group in self.optimizer.param_groups:
            logger.info("optimiser group %-16s lr=%.2e wd=%.3f params=%d", group.get("name", "?"), group["lr"], group["weight_decay"], sum(p.numel() for p in group["params"]))
        logger.info(
            "%d optimisation steps (%d per epoch, %d warm-up), accumulation %d, amp=%s",
            self.total_steps,
            steps_per_epoch,
            self.warmup_steps,
            self.accumulation,
            self.amp_dtype,
        )

    # ------------------------------------------------------------------ resume

    def maybe_resume(self) -> None:
        last = checkpoint_dir(self.run_dir) / "last.pt"
        if not last.exists():
            return
        payload = resume_state(last, self.model, self.optimizer, self.scheduler, self.scaler, self.device)
        self.history = list(payload.get("history", []))
        self.global_step = int(payload.get("global_step", 0))
        self.start_epoch = int(payload["epoch"]) + 1
        best = payload.get("best") or {}
        if best:
            self.early_stopping.best = best.get("score")
            self.early_stopping.best_epoch = best.get("epoch", 0)
            self.early_stopping.bad_epochs = int(payload["epoch"]) - self.early_stopping.best_epoch

    # ------------------------------------------------------------------ one epoch

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        running, running_aux, n_batches = 0.0, 0.0, 0
        start = time.perf_counter()
        use_amp = self.amp_dtype is not None
        n_loader = len(self.train_loader)
        for i, batch in enumerate(self.train_loader, start=1):
            batch = move_to_device(batch, self.device)
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype or torch.float16, enabled=use_amp):
                out = self.model(batch)
                main = self.loss_fn(out.logits, batch["labels"])
                loss = total_loss(main, out.aux_loss, self.aux_weight) / self.accumulation
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            running += float(main.detach())
            if out.aux_loss is not None:
                running_aux += float(out.aux_loss.detach())
            n_batches += 1

            boundary = i % self.accumulation == 0 or i == n_loader
            if not boundary:
                continue
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1
            if self.log_every and self.global_step % self.log_every == 0:
                elapsed = time.perf_counter() - start
                lrs = "/".join(f"{g['lr']:.2e}" for g in self.optimizer.param_groups[:2])
                logger.info(
                    "epoch %d step %d (%d/%d batches) loss %.4f%s grad %.2f lr %s %.1fs mem %.1fGB",
                    epoch,
                    self.global_step,
                    i,
                    n_loader,
                    running / n_batches,
                    f" aux {running_aux / n_batches:.4f}" if running_aux else "",
                    float(grad_norm),
                    lrs,
                    elapsed,
                    peak_memory_gb(self.device),
                )
        return {
            "train_loss": running / max(n_batches, 1),
            "train_aux_loss": running_aux / max(n_batches, 1) if running_aux else None,
            "epoch_time_s": time.perf_counter() - start,
        }

    # ------------------------------------------------------------------ fit

    def fit(self) -> dict:
        self.maybe_resume()
        spec_dict = self.spec.to_dict()
        for epoch in range(self.start_epoch, self.epochs + 1):
            train_stats = self.train_epoch(epoch)
            dev_preds = self.evaluator.predict(self.dev_loader, log_every=0)
            dev_report = metric_report(dev_preds, self.spec, train_counts=self.train_counts)
            score = float(dev_report["discrimination"][self.selection_metric])
            improved = self.early_stopping.step(score, epoch)
            record = {
                "epoch": epoch,
                "global_step": self.global_step,
                "lr_encoder": self.optimizer.param_groups[0]["lr"],
                **train_stats,
                "dev": dev_report["discrimination"],
                "dev_ece_raw": dev_report["calibration"]["raw"]["ece"],
                "dev_eval_time_s": dev_preds.wall_time,
                "peak_memory_gb": peak_memory_gb(self.device),
                "improved": improved,
                "best_epoch": self.early_stopping.best_epoch,
            }
            self.history.append(record)
            self.metrics_writer.write(record)
            logger.info(
                "epoch %d | train loss %.4f | dev %s %.2f (best %.2f @ %d) | %.0fs",
                epoch,
                train_stats["train_loss"],
                self.selection_metric,
                score,
                self.early_stopping.best or 0.0,
                self.early_stopping.best_epoch,
                train_stats["epoch_time_s"],
            )
            if improved:
                save_best(self.run_dir, self.model, epoch, dev_report["discrimination"], self.cfg, spec_dict, self.seed)
            save_last(
                self.run_dir,
                self.model,
                self.optimizer,
                self.scheduler,
                self.scaler,
                epoch,
                self.global_step,
                self.history,
                {"score": self.early_stopping.best, "epoch": self.early_stopping.best_epoch},
                self.cfg,
                spec_dict,
                self.seed,
            )
            if self.early_stopping.should_stop:
                logger.info("early stopping: no improvement for %d epochs", self.early_stopping.patience)
                break
        return {
            "best_epoch": self.early_stopping.best_epoch,
            "best_score": self.early_stopping.best,
            "epochs_run": len(self.history),
            "global_steps": self.global_step,
            "history": self.history,
        }
